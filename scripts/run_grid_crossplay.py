#!/usr/bin/env python3
"""Play an existing scenario set with a DIFFERENT learner — the off-diagonal of the matrix.

The matrix separates two claims that gen-90 had tangled together:

    DIAGONAL      set calibrated to M, played by M   -> the curriculum claim (did we hit M's frontier)
    OFF-DIAGONAL  set calibrated to M, played by N   -> the BENCHMARK claim (do these items order
                                                        models correctly)

Only the off-diagonal can tell you whether the items measure anything transferable. If a set
calibrated to a weak learner comes out `too_easy` for a strong one, and a set calibrated to a strong
learner comes out `beyond_frontier` for a weak one, difficulty is ordered by model strength and the
items are measuring something real. That is the figure.

No generation happens here — the scenarios already exist. Just the K<=4 loop, with the partner and
judge held to the same frozen models used at generation time (they must be, or a stronger learner
would face a stronger partner and the ordering would cancel).

    # one cell
    uv run scripts/run_grid_crossplay.py --set results/matrix_v1/sets/gpt5mini \\
        --learner-tag gpt4omini --learner-model gpt-4o-mini --out results/matrix_v1/crossplay

    # every off-diagonal pair, plus the diagonal extraction
    uv run scripts/run_grid_crossplay.py --all --matrix-root results/matrix_v1

Resume is by episode-file existence, so an interrupted run picks up where it stopped.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.adversarial_agent import AdversarialAgent               # noqa: E402
from social_omni_epic.coherence_check import CoherenceChecker                 # noqa: E402
from social_omni_epic.curriculum import run_episode_two_loop                  # noqa: E402
from social_omni_epic.data_models import SocialScenario                       # noqa: E402
from social_omni_epic.fm import make_fm                                            # noqa: E402
from social_omni_epic.meta_reflection import MetaReflectionModule             # noqa: E402
from social_omni_epic.reflection_module import ReflectionModule               # noqa: E402
from social_omni_epic.scenario_title import ScenarioTitleGenerator            # noqa: E402
from social_omni_epic.task_generator import TaskGenerator                     # noqa: E402
from social_omni_epic.tracing_fm import print_info, print_step, print_warn    # noqa: E402

MAX_CONCURRENCY = 8


def load_set(set_dir: Path) -> list[SocialScenario]:
    """Load the completed scenarios of a set. Skips in-progress stubs, as every reader must."""
    out = []
    for p in sorted((set_dir / "bank" / "generated").glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if rec.get("status") == "in_progress":
            continue
        try:
            out.append(SocialScenario(**rec))
        except Exception as e:
            print_warn(f"  {p.name}: {type(e).__name__}: {str(e)[:100]}")
    return out


def band_of(solved: bool, first_attempt_solved: bool, lp_value: float) -> str:
    """Same classification as the generation path, so cells are directly comparable.

    cold pass -> too_easy; failed cold but recovered or improved -> frontier; neither -> beyond.
    """
    if first_attempt_solved:
        return "too_easy"
    if solved or (lp_value or 0.0) > 0:
        return "frontier"
    return "beyond_frontier"


async def play_one(scn: SocialScenario, svc_bits, config, out_dir: Path, tag: str) -> dict:
    ep_path = out_dir / "episodes" / f"{scn.id}.json"
    if ep_path.exists():
        try:
            return json.loads(ep_path.read_text())
        except Exception:
            pass   # corrupt cache -> re-run

    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    try:
        scn2, terminal_state, _outcome, final_scores, loop_info = await run_episode_two_loop(
            scenario=scn, anchor=None, **svc_bits, config=config,
        )
    except Exception as e:
        rec = {"scenario_id": scn.id, "terminal_state": "discarded", "error": str(e)}
        ep_path.write_text(json.dumps(rec, indent=2, default=str))
        return rec

    attempts = loop_info.get("skill_attempts", [])
    first_solved = bool(attempts and attempts[0].get("solved"))
    rec = {
        "scenario_id": scn.id,
        "root_seed_env_pk": scn.root_seed_env_pk,
        "terminal_state": terminal_state,
        "band": band_of(scn2.terminal_success, first_solved, loop_info.get("lp_value")),
        "first_attempt_solved": first_solved,          # the cold-pass readout
        "solved": bool(scn2.terminal_success),
        "n_attempts": len(attempts),
        "lp_value": loop_info.get("lp_value"),
        "lp_inference": loop_info.get("lp_inference"),
        "goal_trajectory": [(a.get("diagnostics_scores") or {}).get("goal")
                            for a in attempts],
        "final_scores": final_scores,
        # Kept so the staged-vs-state disagreement rate is measurable off-diagonal too.
        "key_check_verdicts": [a.get("key_check_result") for a in attempts],
        "transcripts": [a.get("transcript_clean") for a in attempts],
    }
    ep_path.write_text(json.dumps(rec, indent=2, default=str))
    print_info(f"{tag} {scn.id[:20]} → {rec['band']} (cold={'✓' if first_solved else '✗'}, "
               f"goals={rec['goal_trajectory']})")
    return rec


async def run_cell(set_dir: Path, set_tag: str, args) -> dict:
    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    out_dir = Path(args.out) / f"{set_tag}__{args.learner_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_set(set_dir)
    if not scenarios:
        print_warn(f"no completed scenarios in {set_dir}")
        return {}

    # Frozen everywhere except the learner. fm_reflection is pinned to the learner for the same
    # reason as at generation time: otherwise recoverability measures the teacher.
    fm_learner = make_fm(model=args.learner_model, temperature=1.0)
    fm_judge = make_fm(model=args.judge_model, temperature=0.3)
    fm_aux = make_fm(model=args.aux_model, temperature=0.2)

    svc_bits = dict(
        task_gen=TaskGenerator(fm_aux, num_examples=1, num_failed_examples=0, max_retries=1),
        reflection_mod=ReflectionModule(fm_learner),
        meta_mod=MetaReflectionModule(fm_learner),
        adversarial=AdversarialAgent(fm_aux),
        title_gen=ScenarioTitleGenerator(fm_aux),
        coherence_checker=CoherenceChecker(fm_aux),
        run_single_episode=run_single_episode,
        scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
        fm=fm_learner,
        fm_judge=fm_judge,
        fm_reflection=fm_learner,
    )
    config = {
        "learner_model": args.learner_model, "partner_model": args.partner_model,
        "max_attempts": args.max_attempts, "max_turns": args.max_turns,
        "use_expel_memory": True, "chronicle_max_entries": 8,
        "adversarial": {"re_reflect_on_rejection": False},
        "skip_lp_when_solved": True,
    }

    print_step(f"crossplay {set_tag} x {args.learner_tag}: {len(scenarios)} scenario(s) → {out_dir}")
    sem = asyncio.Semaphore(min(args.concurrency, MAX_CONCURRENCY))
    t0 = time.time()

    async def _slot(s):
        async with sem:
            return await play_one(s, svc_bits, config, out_dir,
                                  f"[{set_tag}x{args.learner_tag}]")

    recs = await asyncio.gather(*[_slot(s) for s in scenarios], return_exceptions=True)
    rows = [r for r in recs if isinstance(r, dict) and r.get("band")]

    bands: dict = {}
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    cold = sum(1 for r in rows if r.get("first_attempt_solved"))
    summary = {
        "set_tag": set_tag, "learner_tag": args.learner_tag,
        "diagonal": set_tag == args.learner_tag,
        "n": len(rows),
        "band_counts": bands,
        "cold_pass": cold,
        "cold_pass_rate": round(cold / len(rows), 4) if rows else None,
        "models": {"learner": args.learner_model, "partner": args.partner_model,
                   "judge": args.judge_model},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print_info(f"  → {bands}  cold_pass={cold}/{len(rows)}")
    return summary


def extract_diagonal(matrix_root: Path, set_tag: str) -> dict:
    """Copy a set's own generation results into crossplay/<X>__<X>/ so analysis sees one shape.

    The diagonal is produced BY generation (calibrating to M means playing against M), so re-running
    it would be paying twice for the same measurement. This just reshapes it.
    """
    set_dir = matrix_root / "sets" / set_tag
    out_dir = matrix_root / "crossplay" / f"{set_tag}__{set_tag}"
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted((set_dir / "bank" / "generated").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("status") == "in_progress":
            continue
        attempts = rec.get("attempts") or []
        goals = [(a.get("scores") or {}).get("goal") for a in attempts]
        first_solved = bool(attempts and attempts[0].get("solved"))
        row = {
            "scenario_id": rec.get("id"),
            "root_seed_env_pk": rec.get("root_seed_env_pk"),
            "terminal_state": rec.get("classification"),
            "band": rec.get("classification"),
            "first_attempt_solved": first_solved,
            "solved": bool(rec.get("terminal_success")),
            "n_attempts": rec.get("n_attempts"),
            "lp_value": rec.get("lp_value"),
            "goal_trajectory": goals,
            "from_generation": True,
        }
        rows.append(row)
        (out_dir / "episodes" / f"{row['scenario_id']}.json").write_text(
            json.dumps(row, indent=2, default=str))

    bands: dict = {}
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    cold = sum(1 for r in rows if r["first_attempt_solved"])
    summary = {
        "set_tag": set_tag, "learner_tag": set_tag, "diagonal": True,
        "n": len(rows), "band_counts": bands, "cold_pass": cold,
        "cold_pass_rate": round(cold / len(rows), 4) if rows else None,
        "source": "extracted from generation (not re-run)",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print_info(f"  diagonal {set_tag}: {bands} cold_pass={cold}/{len(rows)}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_dir", default=None, help="One set directory to play")
    ap.add_argument("--set-tag", default=None, help="Defaults to the set directory name")
    ap.add_argument("--learner-tag", default=None)
    ap.add_argument("--learner-model", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Every off-diagonal pair under --matrix-root, plus diagonal extraction.")
    ap.add_argument("--matrix-root", default="results/matrix_v1")
    ap.add_argument("--learners", default=None,
                    help="With --all: comma list of tag:model pairs. Defaults to the sets present, "
                         "each with its own model read from its grid_manifest.json.")
    ap.add_argument("--partner-model", default="gpt-5.4-mini")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--aux-model", default="gpt-4.1-mini",
                    help="Non-learner scaffolding (titles, coherence). Never scores anything.")
    ap.add_argument("--out", default=None, help="Defaults to <matrix-root>/crossplay")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.matrix_root)
    args.out = args.out or str(root / "crossplay")

    if args.all:
        sets = sorted(p for p in (root / "sets").glob("*") if p.is_dir())
        if not sets:
            print(f"no sets under {root / 'sets'}", file=sys.stderr)
            sys.exit(2)

        # Learner model per tag: from the CLI if given, else each set's own manifest.
        learners: dict = {}
        if args.learners:
            for pair in args.learners.split(","):
                tag, _, model = pair.partition(":")
                learners[tag.strip()] = model.strip()
        else:
            for s in sets:
                mf = s / "grid_manifest.json"
                if mf.exists():
                    m = json.loads(mf.read_text())
                    learners[m.get("learner_tag", s.name)] = m["models"]["learner"]
                else:
                    print_warn(f"{s.name}: no grid_manifest.json — pass --learners explicitly")

        print_step(f"matrix: {len(sets)} set(s) x {len(learners)} learner(s)")
        for s in sets:
            extract_diagonal(root, s.name)
        for s in sets:
            for tag, model in learners.items():
                if tag == s.name:
                    continue   # diagonal already extracted
                args.learner_tag, args.learner_model = tag, model
                asyncio.run(run_cell(s, s.name, args))
        return

    if not (args.set_dir and args.learner_tag and args.learner_model):
        print("need --set, --learner-tag and --learner-model (or --all)", file=sys.stderr)
        sys.exit(2)
    set_dir = Path(args.set_dir)
    asyncio.run(run_cell(set_dir, args.set_tag or set_dir.name, args))


if __name__ == "__main__":
    main()
