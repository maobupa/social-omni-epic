"""Phase 0 calibration: resume from baseline run, run reflection loop on failures,
classify each seed as too_easy / frontier / beyond_frontier using the LP judge.

Usage:
  python scripts/run_phase0_calibration.py \
      --resume-from-baseline results/baseline_eval_20260604_222545 \
      --run-name jun10 \
      [--judge-model google/gemini-3-flash-preview] \
      [--learner-model openai/gpt-5-mini] \
      [--K 4] \
      [--n-seeds 5] \
      [--seed-indices 0,3,7]

Output goes to results/phase0_<run-name>/
  seeds/seed_000_craigslist_bargains.json  — one per seed
  summary.json
"""
import argparse
import asyncio
import json
import os
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

from social_omni_epic.adversarial_agent import AdversarialAgent
from social_omni_epic.data_models import SocialScenario
from social_omni_epic.episode_runner import clean_transcript, run_single_episode, score_transcript
from social_omni_epic.fm import FM
from social_omni_epic.lp_judge import compute_lp
from social_omni_epic.meta_reflection import MetaReflectionModule
from social_omni_epic.reflection_module import ReflectionModule
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.skills_chronicle import SkillsChronicle
from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

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


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    std = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    return round(m, 3), round(std, 3)


def _load_baseline_episodes(baseline_dir: Path) -> list[dict]:
    """Load all seed episode JSON files from baseline dir, sorted by seed_idx."""
    episodes_dir = baseline_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No episodes/ subdir found in {baseline_dir}")
    records = []
    for p in sorted(episodes_dir.glob("seed_*.json")):
        with open(p) as f:
            records.append(json.load(f))
    records.sort(key=lambda r: int(r.get("seed_idx", 0)))
    return records


def _build_scenarios_by_pk(seeds_path: str) -> dict[str, SocialScenario]:
    """Map env_pk → SocialScenario (learner perspective, both_perspectives=False)."""
    seeds = load_sotopia_seeds(seeds_path=seeds_path, both_perspectives=False)
    return {s.source_env_id: s for s in seeds}


def _output_filename(seed_idx: int, source: str) -> str:
    safe = source.replace("/", "_").replace(" ", "_")
    return f"seed_{seed_idx:03d}_{safe}.json"


# ---------------------------------------------------------------------------
# Per-seed async loop
# ---------------------------------------------------------------------------

async def _process_seed(
    seed_idx: int,
    ep_data: dict,
    scenario: SocialScenario,
    fm_learner: FM,
    fm_judge: FM,
    reflection_mod: ReflectionModule,
    meta_mod: MetaReflectionModule,
    adversarial: AdversarialAgent,
    K: int,
    max_turns: int,
) -> dict:
    """Run calibration for one seed. Returns the record dict to be saved."""
    source = ep_data.get("source", "unknown")
    relational_stakes = scenario.relationship or "general social interaction"
    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
    learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ep_data.get("learner_goal", "")
    partner_goal = env_profile.agent_goals[1] if len(env_profile.agent_goals) > 1 else ep_data.get("partner_goal", "")

    # shared header fields — ordered so retrieval-facing fields appear first
    header = {
        # --- scenario identity (retrieval keys) ---
        "scenario_title": scenario.scenario_title,
        "social_dynamic": scenario.social_dynamic,
        "target_perspective": scenario.target_perspective,
        "scenario": scenario.scenario,
        "learner_goal": learner_goal,
        "partner_goal": partner_goal,
        # --- bookkeeping ---
        "seed_idx": seed_idx,
        "env_pk": ep_data.get("env_pk", ""),
        "id": scenario.id,
        "source": source,
        "is_sotopia_hard": ep_data.get("is_sotopia_hard", False),
    }

    # ---- Step 1: Rescore attempt-1 transcript with the judge model ----
    transcript1 = ep_data.get("transcript", [])
    scores1, _, reasoning1 = score_transcript(transcript1, fm_judge, learner_goal)
    attempt1_success = _is_terminal_success(scores1)

    attempt_traces: list[dict] = [{
        "attempt": 1,
        "source": "baseline",
        "transcript": transcript1,
        "scores": scores1,
        "solved": attempt1_success,
        "scoring_reasoning": reasoning1,
    }]

    # ---- Step 2: too_easy fast-path ----
    if attempt1_success:
        return {
            **header,
            # outcome
            "classification": "too_easy",
            "n_attempts": 1,
            "terminal_success": True,
            # scores (top-level for quick scanning)
            "scores": scores1,
            # LP
            "lp_value": None,
            "lp_votes": 0,
            "lp_improved_votes": 0,
            "lp_details": [],
            # meta-reflection (none needed — solved on attempt 1)
            "final_chronicle_md": None,
            "final_check_flag": None,
            # per-attempt traces
            "attempts": attempt_traces,
        }

    # ---- Step 3: Reflection loop (attempts 2..K) ----
    chronicle = SkillsChronicle()
    all_transcripts: list[list[dict]] = [transcript1]
    all_scores: list[dict] = [{"attempt": 1, "scores": scores1, "solved": attempt1_success}]
    all_edit_reasons: dict[str, str] = {}
    all_chronicle_versions: list[SkillsChronicle] = [deepcopy(chronicle)]

    solved = False

    for attempt in range(2, K + 1):
        mem = chronicle.format_for_prompt(max_entries=8)
        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                fm=fm_learner,
                learner_model=fm_learner.model,
                partner_model=fm_learner.model,
                memory_prompt=mem,
                max_turns=max_turns,
                learner_goal=learner_goal,
                rubric=None,
                partner_profile=None,
                judge_self_consistency_k=1,
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

        trace: dict = {
            "attempt": attempt,
            "source": "reflection_loop",
            "transcript": t_clean,
            "scores": scores_j,
            "solved": attempt_solved,
            "scoring_reasoning": reasoning_j,
        }

        if attempt_solved:
            solved = True
            all_chronicle_versions.append(deepcopy(chronicle))
            attempt_traces.append(trace)
            break

        # Reflect if budget remains
        if attempt < K:
            ref_out = reflection_mod.reflect(
                chronicle=chronicle,
                scenario=scenario,
                transcripts=all_transcripts,
                prior_edit_reasons=all_edit_reasons,
                attempt_num=attempt,
                anchor_task=None,
                key_check_verdicts=None,
                attempt_scores=all_scores,
            )
            adv_ref = adversarial.check_reflection(
                ref_out, t_clean, anchor_task=None, scenario=scenario
            )
            if not adv_ref.approved:
                ref_out = reflection_mod.synthesize_with_critique(
                    reflection_output=ref_out,
                    adversarial_critique=adv_ref.critique,
                    chronicle=chronicle,
                    scenario=scenario,
                    transcripts=all_transcripts,
                    prior_edit_reasons=all_edit_reasons,
                    attempt_num=attempt,
                    anchor_task=None,
                )
            all_edit_reasons.update(ref_out.edit_reasons)
            chronicle = ref_out.updated_chronicle
            all_chronicle_versions.append(deepcopy(chronicle))
            trace["reflection_diagnosis"] = ref_out.diagnosis
            trace["reflection_edit_reasons"] = ref_out.edit_reasons
            trace["adversarial_reflection_approved"] = adv_ref.approved
            trace["chronicle_after_reflection"] = chronicle.to_markdown()

        attempt_traces.append(trace)

    outcome = 2 if solved else 3

    # ---- Step 4: Meta-reflection ----
    final_chronicle = meta_mod.synthesize(
        chronicle_versions=all_chronicle_versions,
        transcripts=all_transcripts,
        edit_reasons=all_edit_reasons,
        outcome=outcome,
        scenario=scenario,
        anchor_task=None,
        attempt_scores=all_scores,
    )

    # ---- Step 5: check_final with one retry ----
    final_check_flag = None
    adv_final = adversarial.check_final(final_chronicle, "", outcome=outcome)
    if not adv_final.approved:
        final_chronicle = meta_mod.synthesize(
            chronicle_versions=all_chronicle_versions,
            transcripts=all_transcripts,
            edit_reasons=all_edit_reasons,
            outcome=outcome,
            scenario=scenario,
            anchor_task=None,
            attempt_scores=all_scores,
            adversarial_critique=adv_final.critique,
        )
        adv_final2 = adversarial.check_final(final_chronicle, "", outcome=outcome)
        if not adv_final2.approved:
            final_check_flag = adv_final2.issues

    # ---- Step 6: LP judge ----
    lp_result = await compute_lp(
        fm_judge=fm_judge,
        scenario=scenario,
        transcripts=all_transcripts,
        learner_goal=learner_goal,
        relational_stakes=relational_stakes,
    )

    # ---- Step 7: Classify ----
    if solved or lp_result.lp_value > 0.0:
        classification = "frontier"
    else:
        classification = "beyond_frontier"

    # top-level scores = final attempt (best signal of where the agent ended up)
    scores_final = all_scores[-1]["scores"] if all_scores else {}

    return {
        **header,
        # outcome
        "classification": classification,
        "n_attempts": len(all_transcripts),
        "terminal_success": solved,
        # scores (top-level for quick scanning — from final attempt)
        "scores": scores_final,
        # LP signal
        "lp_value": lp_result.lp_value,
        "lp_votes": lp_result.total_votes,
        "lp_improved_votes": lp_result.improved_votes,
        "lp_details": [
            {"pair": v.pair, "order": v.order, "verdict": v.verdict, "rationale": v.rationale}
            for v in lp_result.votes
        ],
        # meta-reflection (the distilled skills chronicle — primary ICL payload)
        "final_chronicle_md": final_chronicle.to_markdown(),
        "final_check_flag": final_check_flag,
        # per-attempt traces (intermediary detail — human audit, not ICL)
        "attempts": attempt_traces,
    }


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(
    records: list[dict],
    baseline_dir: str,
    run_name: str,
    judge_model: str,
    learner_model: str,
    K: int,
) -> dict:
    counts: dict[str, int] = defaultdict(int)
    by_source: dict[str, list[str]] = defaultdict(list)
    lp_values = []

    for r in records:
        if "error" in r:
            counts["error"] += 1
            continue
        c = r.get("classification", "unknown")
        counts[c] += 1
        by_source[r.get("source", "unknown")].append(c)
        if r.get("lp_value") is not None:
            lp_values.append(r["lp_value"])

    lp_mean, lp_std = _mean_std(lp_values)
    n_total = len(records)

    by_source_summary = {}
    for src, classes in by_source.items():
        by_source_summary[src] = {c: classes.count(c) for c in set(classes)}

    return {
        "timestamp": datetime.now().isoformat(),
        "run_name": run_name,
        "baseline_dir": baseline_dir,
        "judge_model": judge_model,
        "learner_model": learner_model,
        "K_attempts": K,
        "n_seeds": n_total,
        "n_errors": counts.get("error", 0),
        "classification_counts": {
            "too_easy": counts.get("too_easy", 0),
            "frontier": counts.get("frontier", 0),
            "beyond_frontier": counts.get("beyond_frontier", 0),
        },
        "lp_stats": {"mean": lp_mean, "std": lp_std, "n": len(lp_values)},
        "by_source": by_source_summary,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main_async(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.resume_from_baseline)
    if not baseline_dir.exists():
        print(f"ERROR: baseline dir not found: {baseline_dir}", file=sys.stderr)
        sys.exit(1)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/phase0_{run_name}")
    seeds_dir = out_dir / "seeds"
    chronicles_dir = out_dir / "chronicles"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    chronicles_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Models
    judge_model = args.judge_model
    learner_model = args.learner_model
    fm_learner = FM(model=learner_model)
    fm_judge = FM(model=judge_model)

    # Modules
    reflection_mod = ReflectionModule(fm_learner)
    meta_mod = MetaReflectionModule(fm_learner)
    adversarial = AdversarialAgent(fm_learner)

    # Load baseline episodes
    print(f"Loading baseline from {baseline_dir} ...")
    ep_records = _load_baseline_episodes(baseline_dir)
    print(f"Found {len(ep_records)} baseline episodes")

    # Build scenario map from seeds file
    scenarios_by_pk = _build_scenarios_by_pk(args.seeds_path)
    print(f"Loaded {len(scenarios_by_pk)} scenarios from {args.seeds_path}")

    # Filter seeds
    if args.seed_indices:
        allowed = {int(i.strip()) for i in args.seed_indices.split(",")}
        ep_records = [r for r in ep_records if int(r.get("seed_idx", -1)) in allowed]
        print(f"Filtered to {len(ep_records)} seeds by --seed-indices")
    elif args.n_seeds is not None:
        ep_records = ep_records[:args.n_seeds]
        print(f"Limited to first {len(ep_records)} seeds")

    results: list[dict] = []

    for ep_data in ep_records:
        seed_idx = int(ep_data.get("seed_idx", 0))
        source = ep_data.get("source", "unknown")
        out_path = seeds_dir / _output_filename(seed_idx, source)

        # Resumable: skip already done
        if out_path.exists():
            print(f"[seed {seed_idx:3d}] SKIP (already done)")
            with open(out_path) as f:
                results.append(json.load(f))
            continue

        env_pk = ep_data.get("env_pk", "")
        scenario = scenarios_by_pk.get(env_pk)
        if scenario is None:
            print(f"[seed {seed_idx:3d}] SKIP — no matching scenario for env_pk={env_pk}")
            continue

        print(f"[seed {seed_idx:3d}] {source:<22} | running K={args.K} ...")
        try:
            record = await _process_seed(
                seed_idx=seed_idx,
                ep_data=ep_data,
                scenario=scenario,
                fm_learner=fm_learner,
                fm_judge=fm_judge,
                reflection_mod=reflection_mod,
                meta_mod=meta_mod,
                adversarial=adversarial,
                K=args.K,
                max_turns=args.max_turns,
            )
        except Exception as e:
            import traceback
            print(f"  [seed {seed_idx}] ERROR: {e}\n{traceback.format_exc()}")
            record = {
                "seed_idx": seed_idx,
                "env_pk": env_pk,
                "source": source,
                "error": str(e),
            }

        results.append(record)
        out_path.write_text(json.dumps(record, indent=2, default=str))

        # Write human-readable chronicle alongside the JSON
        chronicle_md = record.get("final_chronicle_md") or ""
        if chronicle_md:
            stem = _output_filename(seed_idx, source).replace(".json", "")
            chronicle_path = chronicles_dir / f"{stem}.md"
            header_lines = [
                f"# {record.get('scenario_title', stem)}",
                f"",
                f"**Source:** {source}  **Classification:** {record.get('classification', '?')}  "
                f"**LP:** {record.get('lp_value')}  **Attempts:** {record.get('n_attempts', '?')}",
                f"",
                f"**Scenario:** {record.get('scenario', '')[:200]}",
                f"",
                f"**Learner goal:** {record.get('learner_goal', '')[:200]}",
                f"",
                f"---",
                f"",
            ]
            chronicle_path.write_text("\n".join(header_lines) + chronicle_md + "\n")

        c = record.get("classification", record.get("error", "error"))
        lp = record.get("lp_value")
        lp_str = f"LP={lp:.3f}" if lp is not None else "LP=n/a"
        n_att = record.get("n_attempts", "?")
        print(f"           → {c}  {lp_str}  n_attempts={n_att}")

        # Flush summary incrementally
        summary = _build_summary(
            results, str(baseline_dir), run_name,
            judge_model, learner_model, args.K,
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Final summary
    summary = _build_summary(
        results, str(baseline_dir), run_name,
        judge_model, learner_model, args.K,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    cc = summary["classification_counts"]
    print(
        f"\n=== PHASE 0 CALIBRATION DONE ===\n"
        f"  too_easy:       {cc['too_easy']:3d}\n"
        f"  frontier:       {cc['frontier']:3d}\n"
        f"  beyond_frontier:{cc['beyond_frontier']:3d}\n"
        f"  errors:         {summary['n_errors']:3d}\n"
        f"  LP mean:        {summary['lp_stats']['mean']:.3f} ± {summary['lp_stats']['std']:.3f}\n"
        f"  Output:         {out_dir}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 calibration: resume + LP classification")
    ap.add_argument(
        "--resume-from-baseline", required=True,
        help="Path to baseline eval dir (e.g. results/baseline_eval_20260604_222545)",
    )
    ap.add_argument("--run-name", type=str, default=None,
                    help="Name suffix for output dir (default: timestamp)")
    ap.add_argument("--seeds-path", type=str, default="data/sotopia_90_seeds.jsonl",
                    help="Path to the .jsonl seeds file used in the baseline run")
    ap.add_argument("--judge-model", type=str, default="google/gemini-3-flash-preview")
    ap.add_argument("--learner-model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--K", type=int, default=4, help="Max attempts including attempt 1")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="Limit to first N seeds (for smoke-testing)")
    ap.add_argument("--seed-indices", type=str, default=None,
                    help="Comma-separated seed indices to run")
    args = ap.parse_args()

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY or OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
