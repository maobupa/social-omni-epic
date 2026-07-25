"""ExpeL experience gathering on the 90 SOTOPIA seeds — for bank construction.

Faithfully implements Algorithm 1 (ExpeL - Experience Gathering) from Zhao et al. 2024:

  Gathering phase (this script):
    - Attempt 1: no memory (reuse existing baseline transcript)
    - Fail → generate Reflexion string for THIS scenario only
    - Attempt 2: memory = only the Reflexion strings from this scenario's previous failures
    - Continue up to K attempts
    - The per-seed chronicle = accumulated scenario-specific Reflexion strings
    - Global insights do NOT exist yet and are NOT injected here

  The global insights (cross-task rules) are extracted AFTER gathering via the
  ExpeL extract stage (run_expel_baseline.py extract), by reading across all
  gathered trajectories. They are used at eval time, not here.

Output mirrors results/phase0_Base90_Skills_Chronicle/:
  seeds/seed_NNN_<source>.json    — per-seed record (same schema as phase0)
  chronicles/seed_NNN_<source>.md — scenario-specific Reflexion strings (the
                                    ExpeL per-seed chronicle; empty for too_easy)
  summary.json

LP computation and 3-way classification (too_easy / frontier / beyond_frontier)
are added on top of ExpeL's original gather loop so the output is directly
comparable to the phase0 Skills Chronicle folder.

Usage:
    python scripts/run_expel_chronicle.py --run-name jun10

    # test on a few seeds first
    python scripts/run_expel_chronicle.py --run-name test --seed-indices 0,3,7

    # resume a partial run
    python scripts/run_expel_chronicle.py --run-name jun10 --resume
"""
import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.episode_runner import clean_transcript, run_single_episode, score_transcript
from social_omni_epic.expel_baseline import (
    _format_reflections,
    _format_transcript,
    _reflect,
    ExpelTrajectory,
    trajectories_to_dict,
)
from social_omni_epic.fm import FM
from social_omni_epic.lp_judge import compute_lp
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles
from social_omni_epic.data_models import SocialScenario

GOAL_THRESHOLD = 7.0
REL_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_terminal_success(scores: dict) -> bool:
    return (
        float(scores.get("goal", 0.0)) >= GOAL_THRESHOLD
        and float(scores.get("relationship", 0.0)) >= REL_THRESHOLD
    )


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s).lower()[:40]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    std = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    return round(m, 3), round(std, 3)


def _load_baseline_episodes(baseline_dir: Path) -> list[dict]:
    episodes_dir = baseline_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No episodes/ subdir in {baseline_dir}")
    records = [json.loads(p.read_text()) for p in sorted(episodes_dir.glob("seed_*.json"))]
    records.sort(key=lambda r: int(r.get("seed_idx", 0)))
    return records


def _load_rescored_scores(baseline_dir: Path) -> dict[int, dict]:
    """Load pre-computed Gemini-judged attempt-1 scores from the rescored dir."""
    rescored_dir = baseline_dir.parent / (baseline_dir.name + "_rescored") / "episodes"
    scores: dict[int, dict] = {}
    if rescored_dir.exists():
        for p in rescored_dir.glob("seed_*.json"):
            try:
                d = json.loads(p.read_text())
                scores[int(d["seed_idx"])] = d["scores"]
            except Exception:
                pass
    return scores, rescored_dir


def _build_scenarios_by_pk(seeds_path: str) -> dict[str, SocialScenario]:
    seeds = load_sotopia_seeds(seeds_path=seeds_path, both_perspectives=False)
    return {s.source_env_id: s for s in seeds}


# ---------------------------------------------------------------------------
# Per-seed async loop (Algorithm 1 — gather, with LP added for comparison)
# ---------------------------------------------------------------------------

async def _process_seed(
    seed_idx: int,
    ep_data: dict,
    scenario: SocialScenario,
    fm_learner: FM,
    fm_judge: FM,
    K: int,
    max_turns: int,
    partner_model: str | None = None,
    fm_reflect: FM | None = None,
    precomputed_scores1: dict | None = None,
) -> tuple[dict, str, list[ExpelTrajectory], list[ExpelTrajectory]]:
    """Run ExpeL gather loop for one seed.

    Returns (record, chronicle_md, succeeded_trajs, failed_trajs).
    chronicle_md = scenario-specific Reflexion strings (the per-seed ExpeL chronicle).
    """
    source = ep_data.get("source", "unknown")
    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
    learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ep_data.get("learner_goal", "")
    partner_goal = env_profile.agent_goals[1] if len(env_profile.agent_goals) > 1 else ep_data.get("partner_goal", "")
    relational_stakes = scenario.relationship or "general social interaction"

    header = {
        "scenario_title": scenario.scenario_title,
        "social_dynamic": scenario.social_dynamic,
        "target_perspective": scenario.target_perspective,
        "scenario": scenario.scenario,
        "learner_goal": learner_goal,
        "partner_goal": partner_goal,
        "seed_idx": seed_idx,
        "env_pk": ep_data.get("env_pk", ""),
        "id": scenario.id,
        "source": source,
        "is_sotopia_hard": ep_data.get("is_sotopia_hard", False),
    }

    succeeded_trajs: list[ExpelTrajectory] = []
    failed_trajs: list[ExpelTrajectory] = []

    # ---- Attempt 1: reuse pre-computed baseline scores (no memory) ----
    transcript1 = ep_data.get("transcript", [])
    if precomputed_scores1 is not None:
        scores1 = precomputed_scores1
        reasoning1 = ""
    else:
        scores1, _, reasoning1 = score_transcript(transcript1, fm_judge, learner_goal)
    attempt1_success = _is_terminal_success(scores1)

    transcript1_text = _format_transcript(transcript1)
    traj1 = ExpelTrajectory(
        scenario_id=scenario.id, task_idx=seed_idx, task=scenario.scenario,
        learner_goal=learner_goal, transcript_text=transcript1_text,
        success=attempt1_success, goal_score=float(scores1.get("goal", 0.0)), trial=0,
    )
    (succeeded_trajs if attempt1_success else failed_trajs).append(traj1)

    attempt_traces: list[dict] = [{
        "attempt": 1, "source": "baseline",
        "transcript": transcript1, "scores": scores1,
        "solved": attempt1_success, "scoring_reasoning": reasoning1,
    }]

    # ---- too_easy fast-path ----
    if attempt1_success:
        # No Reflexion needed — chronicle is empty (nothing to reflect on)
        chronicle_md = ""
        return {
            **header,
            "classification": "too_easy",
            "n_attempts": 1,
            "terminal_success": True,
            "scores": scores1,
            "lp_value": None,
            "lp_votes": 0,
            "lp_improved_votes": 0,
            "lp_details": [],
            "final_chronicle_md": chronicle_md,
            "final_check_flag": None,
            "attempts": attempt_traces,
        }, chronicle_md, succeeded_trajs, failed_trajs

    # ---- ExpeL gather loop: attempts 2..K ----
    # Memory per attempt = ONLY this scenario's accumulated Reflexion strings.
    # Global insights do not exist yet — they are extracted after gathering.
    all_transcripts: list[list[dict]] = [transcript1]
    all_scores: list[dict] = [{"attempt": 1, "scores": scores1, "solved": attempt1_success}]
    reflexion_strings: list[str] = []
    solved = False

    for attempt in range(2, K + 1):
        # Inject ONLY scenario-specific Reflexion strings (Algorithm 1 line: ν_{n,z})
        memory_prompt = _format_reflections(reflexion_strings)

        try:
            result = await run_single_episode(
                env_profile=env_profile, agent_profiles=agent_profiles, fm=fm_learner,
                learner_model=fm_learner.model, partner_model=partner_model or fm_learner.model,
                memory_prompt=memory_prompt, max_turns=max_turns,
                learner_goal=learner_goal, rubric=None, partner_profile=None,
                judge_self_consistency_k=1,
                fm_judge=fm_judge,   # cross-lab judge; never self-score with the learner
            )
        except Exception as e:
            print(f"    [seed {seed_idx} attempt {attempt}] episode error: {e}")
            attempt_traces.append({"attempt": attempt, "error": str(e)})
            break

        t_clean = clean_transcript(result.transcript)
        scores_j, _, reasoning_j = score_transcript(t_clean, fm_judge, learner_goal)
        attempt_solved = _is_terminal_success(scores_j)

        all_transcripts.append(t_clean)
        all_scores.append({"attempt": attempt, "scores": scores_j, "solved": attempt_solved})

        traj = ExpelTrajectory(
            scenario_id=scenario.id, task_idx=seed_idx, task=scenario.scenario,
            learner_goal=learner_goal, transcript_text=_format_transcript(t_clean),
            success=attempt_solved, goal_score=float(scores_j.get("goal", 0.0)),
            trial=attempt - 1, reflections=list(reflexion_strings),
        )
        (succeeded_trajs if attempt_solved else failed_trajs).append(traj)

        trace: dict = {
            "attempt": attempt, "source": "reflexion_loop",
            "transcript": t_clean, "scores": scores_j,
            "solved": attempt_solved, "scoring_reasoning": reasoning_j,
            "n_reflexion_strings": len(reflexion_strings),
        }

        if attempt_solved:
            solved = True
            attempt_traces.append(trace)
            break

        # Generate Reflexion string (injected into next attempt's memory)
        if attempt < K:
            reflexion = _reflect(
                fm_reflect or fm_learner, scenario.scenario, learner_goal,
                _format_transcript(t_clean), float(scores_j.get("goal", 0.0))
            )
            reflexion_strings.append(reflexion)
            trace["reflexion_generated"] = reflexion

        attempt_traces.append(trace)

    # ---- LP judge (added for phase0 comparison — not in original ExpeL) ----
    lp_value = lp_votes = lp_improved_votes = 0
    lp_details = []
    try:
        lp_result = await compute_lp(
            fm_judge=fm_judge, scenario=scenario, transcripts=all_transcripts,
            learner_goal=learner_goal, relational_stakes=relational_stakes,
        )
        lp_value = lp_result.lp_value
        lp_votes = lp_result.total_votes
        lp_improved_votes = lp_result.improved_votes
        lp_details = [
            {"pair": v.pair, "order": v.order, "verdict": v.verdict, "rationale": v.rationale}
            for v in lp_result.votes
        ]
    except Exception as e:
        print(f"    [seed {seed_idx}] LP error: {e}")
        lp_value = None

    # ---- Classify ----
    if solved or (lp_value is not None and lp_value > 0.0):
        classification = "frontier"
    else:
        classification = "beyond_frontier"

    # ---- Per-seed chronicle = scenario-specific Reflexion strings only ----
    # This is what the agent "learned" about this scenario type.
    # At eval time, this trajectory is retrieved as a few-shot example.
    # Global insights are extracted separately across all seeds after gathering.
    chronicle_md = _format_reflections(reflexion_strings)

    scores_final = all_scores[-1]["scores"] if all_scores else {}

    record = {
        **header,
        "classification": classification,
        "n_attempts": len(all_transcripts),
        "terminal_success": solved,
        "scores": scores_final,
        "lp_value": lp_value,
        "lp_votes": lp_votes,
        "lp_improved_votes": lp_improved_votes,
        "lp_details": lp_details,
        "final_chronicle_md": chronicle_md,
        "final_check_flag": None,
        "attempts": attempt_traces,
    }
    return record, chronicle_md, succeeded_trajs, failed_trajs


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(records, baseline_dir, run_name, judge_model, learner_model, K):
    counts: dict[str, int] = defaultdict(int)
    by_source: dict[str, list[str]] = defaultdict(list)
    lp_values: list[float] = []

    for r in records:
        if "error" in r:
            counts["error"] += 1
            continue
        c = r.get("classification", "unknown")
        counts[c] += 1
        by_source[r.get("source", "unknown")].append(c)
        if r.get("lp_value") is not None:
            lp_values.append(float(r["lp_value"]))

    lp_mean, lp_std = _mean_std(lp_values)
    return {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "baseline_dir": baseline_dir,
        "judge_model": judge_model,
        "learner_model": learner_model,
        "K_attempts": K,
        "n_seeds": len(records),
        "n_errors": counts.get("error", 0),
        "classification_counts": {
            "too_easy": counts.get("too_easy", 0),
            "frontier": counts.get("frontier", 0),
            "beyond_frontier": counts.get("beyond_frontier", 0),
        },
        "lp_stats": {"mean": lp_mean, "std": lp_std, "n": len(lp_values)},
        "by_source": {src: {c: v.count(c) for c in set(v)} for src, v in by_source.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="ExpeL gather phase — build per-seed Reflexion chronicle bank"
    )
    ap.add_argument("--baseline", type=str,
                    default="results/baseline_eval_20260604_222545",
                    help="Baseline dir with episodes/ subfolder (attempt-1 transcripts)")
    ap.add_argument("--seeds", type=str, default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Suffix for output dir (default: timestamp)")
    ap.add_argument("--out", type=str, default=None,
                    help="Override output dir entirely")
    ap.add_argument("--model", type=str, default="openai/gpt-5-mini",
                    help="Model for learner episodes and Reflexion generation")
    ap.add_argument("--learner-model", type=str, default=None)
    ap.add_argument("--reflection-model", type=str, default="openai/gpt-5-mini",
                    help="Model that writes the Reflexion strings (the 'teacher'). Kept on a "
                         "Lightning-served model since it runs through FM; decoupled from the "
                         "episode learner so the learner can be a model FM can't serve (gpt-4.1-mini).")
    ap.add_argument("--partner-model", type=str, default=None,
                    help="Partner (interlocutor) model. Default: same as learner (original behavior). "
                         "Set to hold the environment fixed while varying the learner.")
    ap.add_argument("--judge-model", type=str, default="google/gemini-3-flash-preview")
    ap.add_argument("--K", type=int, default=4, help="Max Reflexion attempts per seed")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=None)
    ap.add_argument("--seed-indices", type=str, default=None,
                    help="Comma-separated indices to run (e.g. 0,3,7)")
    ap.add_argument("--resume", action="store_true", default=False)
    args = ap.parse_args()

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("ERROR: No API key found.", file=sys.stderr)
        sys.exit(1)

    args.learner_model = args.learner_model or args.model
    args.partner_model = args.partner_model or args.learner_model

    if args.out:
        out_dir = Path(args.out)
    else:
        suffix = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"results/expel_{suffix}")
    seeds_dir = out_dir / "seeds"
    chronicles_dir = out_dir / "chronicles"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    chronicles_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir   : {out_dir}")
    print(f"Baseline     : {args.baseline}")
    print(f"Judge model  : {args.judge_model}")
    print(f"Learner model: {args.learner_model}")
    print(f"Reflection model: {args.reflection_model}")
    print(f"Partner model: {args.partner_model}")
    print(f"K            : {args.K}")
    print()

    using_lightning = bool(os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("LIGHTNING_AI_BASE_URL"))
    fm_model = args.learner_model
    if not using_lightning and fm_model.startswith("openai/"):
        fm_model = fm_model.split("/", 1)[1]
    fm_learner = FM(model=fm_model)
    # Reflexion-writer ("teacher"): runs through FM, so it MUST be a model the FM endpoint serves
    # (Lightning has gpt-5-mini but NOT gpt-4.1-mini). Decoupled from the episode learner agent.
    refl_model = args.reflection_model
    if not using_lightning and refl_model.startswith("openai/"):
        refl_model = refl_model.split("/", 1)[1]
    fm_reflect = FM(model=refl_model)
    fm_judge = FM(model=args.judge_model)  # no explicit temp — matches run_phase0_calibration.py

    baseline_dir = Path(args.baseline)
    baseline_eps = _load_baseline_episodes(baseline_dir)
    rescored_scores, rescored_dir = _load_rescored_scores(baseline_dir)
    if rescored_scores:
        print(f"Pre-computed scores : {len(rescored_scores)} seeds from {rescored_dir}")
    else:
        print(f"NOTE: no rescored dir found — will call judge for attempt-1 scores")
    scenarios_by_pk = _build_scenarios_by_pk(args.seeds)
    print(f"Baseline episodes : {len(baseline_eps)}")
    print(f"Seed scenarios    : {len(scenarios_by_pk)}")
    print()

    if args.seed_indices:
        run_indices = set(int(x.strip()) for x in args.seed_indices.split(","))
    elif args.n_seeds:
        run_indices = set(range(args.n_seeds))
    else:
        run_indices = set(range(len(baseline_eps)))

    records: list[dict] = []
    # Experience pool — saved for use by the ExpeL extract stage
    all_succeeded: dict[int, list[ExpelTrajectory]] = {}
    all_failed: dict[int, list[ExpelTrajectory]] = {}
    idx2task: dict[int, str] = {}

    for ep_data in baseline_eps:
        seed_idx = int(ep_data.get("seed_idx", 0))
        if seed_idx not in run_indices:
            continue

        source = ep_data.get("source", "unknown")
        safe = _safe_name(source)
        seed_file = seeds_dir / f"seed_{seed_idx:03d}_{safe}.json"
        chron_file = chronicles_dir / f"seed_{seed_idx:03d}_{safe}.md"

        if args.resume and seed_file.exists():
            try:
                rec = json.loads(seed_file.read_text())
                records.append(rec)
                print(f"[{seed_idx:3d}] {source:<22} — RESUMED")
                continue
            except Exception:
                pass

        env_pk = ep_data.get("env_pk", "")
        scenario = scenarios_by_pk.get(env_pk)
        if scenario is None:
            print(f"[{seed_idx:3d}] {source:<22} — SKIP (env_pk {env_pk!r} not found)")
            continue

        idx2task[seed_idx] = scenario.scenario
        print(f"[{seed_idx:3d}] {source:<22}  K={args.K}...", flush=True)

        try:
            record, chronicle_md, succ, fail = asyncio.run(_process_seed(
                seed_idx=seed_idx, ep_data=ep_data, scenario=scenario,
                fm_learner=fm_learner, fm_judge=fm_judge,
                K=args.K, max_turns=args.max_turns,
                partner_model=args.partner_model,
                fm_reflect=fm_reflect,
                precomputed_scores1=rescored_scores.get(seed_idx),
            ))
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}\n{traceback.format_exc()}")
            record = {**ep_data, "seed_idx": seed_idx, "error": str(e)}
            chronicle_md, succ, fail = "", [], []

        all_succeeded[seed_idx] = succ
        all_failed[seed_idx] = fail

        goal = float(record.get("scores", {}).get("goal", 0.0))
        cl = record.get("classification", "?")
        n_att = record.get("n_attempts", "?")
        lp = record.get("lp_value")
        lp_str = f"{lp:.2f}" if lp is not None else "n/a"
        n_reflex = len([a for a in record.get("attempts", []) if a.get("reflexion_generated")])
        print(f"        → {cl:20s}  n_att={n_att}  goal={goal:.1f}  lp={lp_str}  "
              f"reflexions={n_reflex}")

        seed_file.write_text(json.dumps(record, indent=2, default=str))
        chron_file.write_text(chronicle_md)
        records.append(record)

        # Save experience pool checkpoint (enables ExpeL extract stage later)
        traj_data = trajectories_to_dict(all_succeeded, all_failed, idx2task,
                                          set(all_succeeded.keys()))
        (out_dir / "trajectories.json").write_text(
            json.dumps(traj_data, indent=2, default=str)
        )

        summary = _build_summary(records, str(baseline_dir),
                                  args.run_name or out_dir.name,
                                  args.judge_model, args.learner_model, args.K)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Final summary
    final = _build_summary(records, str(baseline_dir), args.run_name or out_dir.name,
                           args.judge_model, args.learner_model, args.K)
    cc = final["classification_counts"]
    lp = final["lp_stats"]
    print(f"\n=== EXPEL GATHER COMPLETE ===")
    print(f"  too_easy={cc['too_easy']}  frontier={cc['frontier']}  "
          f"beyond_frontier={cc['beyond_frontier']}  errors={final['n_errors']}")
    print(f"  LP mean={lp['mean']:.3f} ± {lp['std']:.3f}  (n={lp['n']})")
    print(f"  Results      : {out_dir}")
    print(f"  Trajectories : {out_dir}/trajectories.json")
    print(f"  (Run ExpeL extract next to get global insights:)")
    print(f"  python scripts/run_expel_baseline.py extract --out {out_dir}")


if __name__ == "__main__":
    main()
