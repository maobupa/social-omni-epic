"""Step 1 baseline evaluation: run gpt-5-mini on the original 90 SOTOPIA seeds
with NO chronicle, NO structured goals, NO rubric — just flat agent_goals and
SOTOPIA-EVAL 7-dimension scoring.

Output layout:
  output/baseline_eval_<timestamp>/
    episodes/
      seed_000_craigslist_bargains.json   ← one per scenario
      seed_001_social_chemistry.json
      ...
    summary.json                          ← aggregate stats, by_source, hard vs easy
    analysis.md                           ← human-readable breakdown

Run from project root:
  python scripts/run_baseline_eval.py --n-seeds 5          # quick smoke test
  python scripts/run_baseline_eval.py                       # full 90 seeds
  python scripts/run_baseline_eval.py --seed-indices 0,4,9,13,15,18,23,40,44,59,67,68,72,89
"""
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
from social_omni_epic.seeds import load_sotopia_seeds

SOTOPIA_HARD_PKS = {
    "01H7VFHPQQQY6H4DNC6NBQ8XTG",
    "01H7VFHNNYH3W0VRWVY178K2TK",
    "01H7VFHNV13MHN97GAH73E3KM8",
    "01H7VFHPSWGDGEYRP63H2DJKV0",
    "01H7VFHN7A1ZX5KSMT2YN9RXC4",
    "01H7VFHQ11NAMZS4A2RDGDB01V",
    "01H7VFHNN7XTR99319DS8KZCQM",
    "01H7VFHN5WVC5HKKVBHZBA553R",
    "01H7VFHNF4G18PC9JHGRC8A1R6",
    "01H7VFHPS5WJW2694R1MNC8JFY",
    "01H7VFHN7WJK7VWVRZZTQ6DX9T",
    "01H7VFHP8AN5643B0NR0NP00VE",
    "01H7VFHN9W0WAFZCBT09PKJJNK",
    "01H7VFHPDZVVCDZR3AARA547CY",
}

GOAL_SUCCESS_THRESHOLD = 7.0  # SOTOPIA-PI uses this as their success threshold


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    std = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    return round(m, 3), round(std, 3)


def _build_summary(results: list[dict], learner_model: str, partner_model: str, seeds_path: str) -> dict:
    dim_accum: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    hard_goals, easy_goals = [], []
    failed, succeeded = [], []  # by GOAL >= threshold

    for r in results:
        if "error" in r:
            continue
        scores = r["scores"]
        goal = float(scores.get("goal", 0.0))
        source = r["source"]

        for dim, val in scores.items():
            dim_accum[dim].append(float(val))
        by_source[source].append(goal)

        if r["is_sotopia_hard"]:
            hard_goals.append(goal)
        else:
            easy_goals.append(goal)

        if goal >= GOAL_SUCCESS_THRESHOLD:
            succeeded.append(r["seed_idx"])
        else:
            failed.append(r["seed_idx"])

    overall = {}
    for dim, vals in dim_accum.items():
        m, s = _mean_std(vals)
        overall[dim] = {"mean": m, "std": s, "n": len(vals)}

    by_source_summary = {}
    for src, vals in by_source.items():
        m, s = _mean_std(vals)
        by_source_summary[src] = {"goal_mean": m, "goal_std": s, "n": len(vals)}

    hard_m, hard_s = _mean_std(hard_goals)
    easy_m, easy_s = _mean_std(easy_goals)

    return {
        "timestamp": datetime.now().isoformat(),
        "learner_model": learner_model,
        "partner_model": partner_model,
        "seeds_path": seeds_path,
        "n_completed": len([r for r in results if "error" not in r]),
        "n_errors": len([r for r in results if "error" in r]),
        "goal_success_threshold": GOAL_SUCCESS_THRESHOLD,
        "overall": overall,
        "by_source": by_source_summary,
        "sotopia_hard": {
            "goal_mean": hard_m, "goal_std": hard_s, "n": len(hard_goals),
            "success_rate": round(sum(1 for g in hard_goals if g >= GOAL_SUCCESS_THRESHOLD) / max(len(hard_goals), 1), 3),
        },
        "sotopia_easy": {
            "goal_mean": easy_m, "goal_std": easy_s, "n": len(easy_goals),
            "success_rate": round(sum(1 for g in easy_goals if g >= GOAL_SUCCESS_THRESHOLD) / max(len(easy_goals), 1), 3),
        },
        "failed_seed_indices": sorted(failed),
        "succeeded_seed_indices": sorted(succeeded),
        "hard_failures": sorted([r["seed_idx"] for r in results
                                  if r.get("is_sotopia_hard") and "error" not in r
                                  and float(r["scores"].get("goal", 0)) < GOAL_SUCCESS_THRESHOLD]),
    }


def _build_analysis_md(summary: dict, results: list[dict]) -> str:
    lines = []
    ts = summary["timestamp"]
    lm = summary["learner_model"]
    pm = summary["partner_model"]
    n = summary["n_completed"]

    lines += [
        f"# Baseline Eval Analysis",
        f"",
        f"**Timestamp:** {ts}  ",
        f"**Learner:** `{lm}`  **Partner:** `{pm}`  ",
        f"**N completed:** {n}  **N errors:** {summary['n_errors']}",
        f"**Goal success threshold:** {summary['goal_success_threshold']} (SOTOPIA-PI convention)",
        f"",
    ]

    # Overall scores
    lines += ["## Overall SOTOPIA-EVAL Scores", ""]
    lines += ["| Dimension | Mean | Std | Range |"]
    lines += ["|-----------|------|-----|-------|"]
    ranges = {"believability": "[0,10]", "relationship": "[-5,5]", "knowledge": "[0,10]",
               "secret": "[-10,0]", "social_rules": "[-10,0]",
               "financial_and_material_benefits": "[-5,5]", "goal": "[0,10]", "overall_score": "—"}
    for dim, stats in summary["overall"].items():
        r = ranges.get(dim, "—")
        lines.append(f"| {dim} | {stats['mean']:.3f} | {stats['std']:.3f} | {r} |")
    lines.append("")

    # SOTOPIA-HARD vs rest
    h = summary["sotopia_hard"]
    e = summary["sotopia_easy"]
    lines += [
        "## SOTOPIA-HARD vs. Rest",
        "",
        f"| Split | N | GOAL mean ± std | Success rate (≥{summary['goal_success_threshold']}) |",
        "|-------|---|-----------------|-------------|",
        f"| SOTOPIA-HARD | {h['n']} | {h['goal_mean']:.3f} ± {h['goal_std']:.3f} | {h['success_rate']:.1%} |",
        f"| Rest | {e['n']} | {e['goal_mean']:.3f} ± {e['goal_std']:.3f} | {e['success_rate']:.1%} |",
        "",
    ]

    # By source
    lines += ["## By Source (GOAL)", ""]
    lines += ["| Source | N | GOAL mean ± std |"]
    lines += ["|--------|---|-----------------|"]
    for src, stats in sorted(summary["by_source"].items(), key=lambda x: -x[1]["goal_mean"]):
        lines.append(f"| {src} | {stats['n']} | {stats['goal_mean']:.3f} ± {stats['goal_std']:.3f} |")
    lines.append("")

    # Hard failures (SOTOPIA-HARD scenarios the model failed)
    hard_fail_indices = summary["hard_failures"]
    if hard_fail_indices:
        lines += [f"## SOTOPIA-HARD Failures ({len(hard_fail_indices)} scenarios)", ""]
        lines += ["These are scenarios in the SOTOPIA-HARD set where GOAL < threshold — primary targets for improvement.", ""]
        for idx in hard_fail_indices:
            r = next((x for x in results if x["seed_idx"] == idx), None)
            if r:
                goal = r["scores"].get("goal", "?")
                lines.append(f"- **seed {idx}** ({r['source']}) GOAL={goal:.1f} — {r['scenario'][:80]}")
        lines.append("")

    # All failures by source
    lines += ["## All Failures by Source", ""]
    fail_by_source: dict[str, list] = defaultdict(list)
    for r in results:
        if "error" not in r and float(r["scores"].get("goal", 0)) < GOAL_SUCCESS_THRESHOLD:
            fail_by_source[r["source"]].append(r)
    for src in sorted(fail_by_source):
        items = fail_by_source[src]
        lines.append(f"**{src}** ({len(items)} failures):")
        for r in sorted(items, key=lambda x: x["scores"].get("goal", 0)):
            hard_tag = " 🔴 HARD" if r["is_sotopia_hard"] else ""
            lines.append(f"  - seed {r['seed_idx']}{hard_tag} GOAL={r['scores'].get('goal','?'):.1f} — {r['scenario'][:70]}")
        lines.append("")

    # Interpretation
    goal_mean = summary["overall"].get("goal", {}).get("mean", 0)
    lines += ["## Interpretation", ""]
    if goal_mean >= 7.5:
        lines.append("⚠️ **Model is near ceiling on SOTOPIA overall** (GOAL ≥ 7.5). Consider evaluating on SOTOPIA-HARD only, or switching to a weaker learner model.")
    elif goal_mean >= 5.0:
        lines.append(f"✅ **Meaningful headroom exists** (GOAL = {goal_mean:.2f}). Proceed with curriculum generation and minimum viable experiment.")
    else:
        lines.append(f"⚠️ **Low overall GOAL ({goal_mean:.2f}).** Check transcripts — partner model may be too adversarial, or episodes may be collapsing.")

    hard_goal = h["goal_mean"]
    if h["n"] > 0:
        lines.append(f"\nSOTOPIA-HARD GOAL = {hard_goal:.2f} (GPT-4 baseline: ~4.85 with human partner, ~7.62 overall). " +
                     ("Hard set shows headroom — good primary eval target." if hard_goal < 7.0 else "Hard set may be near ceiling too."))

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline SOTOPIA-EVAL on original 90 seeds")
    ap.add_argument("--seeds-path", type=str, default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="Limit to first N seeds (default: all 90)")
    ap.add_argument("--seed-indices", type=str, default=None,
                    help="Comma-separated seed indices to run (e.g. 0,4,9,13)")
    ap.add_argument("--model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--learner-model", type=str, default=None)
    ap.add_argument("--partner-model", type=str, default=None)
    ap.add_argument("--judge-model", type=str, default="google/gemini-3-flash-preview",
                    help="Cross-lab scoring judge (must be a Lightning-served, non-learner-lab "
                         "model). Required so the learner model is never used to self-score — "
                         "critical when the learner (e.g. gpt-4.1-mini) isn't on the FM endpoint.")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: output/baseline_eval_<timestamp>)")
    args = ap.parse_args()

    learner_model = args.learner_model or args.model
    partner_model = args.partner_model or args.model

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY or OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    fm = FM(model=args.model)
    # Cross-lab judge: scoring must NOT go through the learner model (it may not exist on the
    # FM/Lightning endpoint, e.g. gpt-4.1-mini). fm_judge overrides the self-score fallback.
    fm_judge = FM(model=args.judge_model)
    if str(args.judge_model).split("/")[0] == str(learner_model).split("/")[0]:
        print(f"WARNING: judge provider matches learner ({learner_model}) — not cross-lab.",
              file=sys.stderr)
    print(f"Judge model: {args.judge_model}")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"results/baseline_eval_{ts}")
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load raw seed rows to get env_pk for SOTOPIA-HARD lookup
    raw_seeds: dict[str, bool] = {}
    with open(args.seeds_path) as f:
        for line in f:
            row = json.loads(line)
            raw_seeds[row["env_pk"]] = bool(row.get("is_sotopia_hard", False))

    seeds = load_sotopia_seeds(
        seeds_path=args.seeds_path,
        limit=args.n_seeds,
        both_perspectives=False,
    )

    if args.seed_indices:
        indices = {int(i.strip()) for i in args.seed_indices.split(",")}
        seeds = [s for i, s in enumerate(seeds) if i in indices]
        print(f"Running {len(seeds)} seeds (filtered by --seed-indices)")
    else:
        print(f"Running {len(seeds)} seeds")

    results: list[dict] = []

    for i, scenario in enumerate(seeds):
        env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
        learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ""
        source = scenario.interaction_type or "unknown"
        is_hard = raw_seeds.get(scenario.source_env_id, False)

        error = None
        scores = {}
        transcript = []
        try:
            result = asyncio.run(
                run_single_episode(
                    env_profile=env_profile,
                    agent_profiles=agent_profiles,
                    fm=fm,
                    learner_model=learner_model,
                    partner_model=partner_model,
                    memory_prompt="",
                    max_turns=args.max_turns,
                    learner_goal=learner_goal,
                    rubric=None,
                    partner_profile=None,
                    judge_self_consistency_k=1,
                    fm_judge=fm_judge,   # cross-lab judge; never self-score with the learner
                )
            )
            scores = result.learner_scores
            from social_omni_epic.episode_runner import clean_transcript
            transcript = clean_transcript(result.transcript)
        except Exception as e:
            error = str(e)
            print(f"  [seed {i}] ERROR: {e}")

        goal = float(scores.get("goal", 0.0))
        rel = float(scores.get("relationship", 0.0))

        # Compute running GOAL mean from completed (non-error) results
        completed_goals = [float(r["scores"].get("goal", 0)) for r in results if "error" not in r]
        if scores:
            completed_goals.append(goal)
        running_mean = sum(completed_goals) / len(completed_goals) if completed_goals else 0.0

        hard_tag = " [HARD]" if is_hard else ""
        status = f"goal={goal:.1f} rel={rel:.1f}" if not error else f"ERROR"
        print(
            f"[{i+1:3d}/{len(seeds)}]{hard_tag:<7} {source:<22} | "
            f"{status} | running_avg_goal={running_mean:.2f}"
        )

        record: dict = {
            "seed_idx": i,
            "env_pk": scenario.source_env_id,
            "id": scenario.id,
            "source": source,
            "is_sotopia_hard": is_hard,
            "scenario": scenario.scenario,
            "learner_goal": learner_goal,
            "partner_goal": env_profile.agent_goals[1] if len(env_profile.agent_goals) > 1 else "",
            "scores": scores,
            "transcript": transcript,
        }
        if error:
            record["error"] = error

        results.append(record)

        # Save individual episode file
        safe_source = source.replace("/", "_").replace(" ", "_")
        ep_path = episodes_dir / f"seed_{i:03d}_{safe_source}.json"
        ep_path.write_text(json.dumps(record, indent=2, default=str))

        # Update summary + analysis incrementally
        _flush_summary(out_dir, results, learner_model, partner_model, args.seeds_path)

    # Final console summary
    summary = _build_summary(results, learner_model, partner_model, args.seeds_path)
    goal_stats = summary["overall"].get("goal", {})
    print(f"\n=== BASELINE EVAL COMPLETE ===")
    print(f"  GOAL     : {goal_stats.get('mean', 0):.3f} ± {goal_stats.get('std', 0):.3f}")
    print(f"  HARD GOAL: {summary['sotopia_hard']['goal_mean']:.3f} ± {summary['sotopia_hard']['goal_std']:.3f}  (n={summary['sotopia_hard']['n']})")
    print(f"  Results  : {out_dir}")


def _flush_summary(out_dir: Path, results: list[dict], learner_model: str, partner_model: str, seeds_path: str) -> None:
    summary = _build_summary(results, learner_model, partner_model, seeds_path)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "analysis.md").write_text(_build_analysis_md(summary, results))


if __name__ == "__main__":
    main()
