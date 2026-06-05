"""Step 1 baseline evaluation: run gpt-5-mini on the original 90 SOTOPIA seeds
with NO chronicle, NO structured goals, NO rubric — just flat agent_goals and
SOTOPIA-EVAL 7-dimension scoring.

This tells us whether gpt-5-mini has headroom to improve on the evaluation set.

Run from project root:
  python scripts/run_baseline_eval.py --n-seeds 5          # quick smoke test
  python scripts/run_baseline_eval.py                       # full 90 seeds
  python scripts/run_baseline_eval.py --seed-indices 6,11,34,82  # specific seeds
"""
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
from social_omni_epic.seeds import load_sotopia_seeds


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    std = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    return round(m, 3), round(std, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline SOTOPIA-EVAL on original 90 seeds")
    ap.add_argument("--seeds-path", type=str, default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="Limit to first N seeds (default: all 90)")
    ap.add_argument("--seed-indices", type=str, default=None,
                    help="Comma-separated seed indices to run (e.g. 6,11,34,82)")
    ap.add_argument("--model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--learner-model", type=str, default=None,
                    help="Override learner model (default: same as --model)")
    ap.add_argument("--partner-model", type=str, default=None,
                    help="Override partner model (default: same as --model)")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--output", type=str, default="output/baseline_eval.json")
    args = ap.parse_args()

    learner_model = args.learner_model or args.model
    partner_model = args.partner_model or args.model

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY or OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    fm = FM(model=args.model)

    # Load seeds — one perspective per seed (learner = agent 0)
    seeds = load_sotopia_seeds(
        seeds_path=args.seeds_path,
        limit=args.n_seeds,
        both_perspectives=False,
    )

    # Apply --seed-indices filter if given
    if args.seed_indices:
        indices = [int(i.strip()) for i in args.seed_indices.split(",")]
        seeds = [s for i, s in enumerate(seeds) if i in indices]
        print(f"Running {len(seeds)} seeds (filtered by --seed-indices)")
    else:
        print(f"Running {len(seeds)} seeds")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    dim_accum: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)  # source → goal scores

    for i, scenario in enumerate(seeds):
        env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
        learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ""

        try:
            result = asyncio.run(
                run_single_episode(
                    env_profile=env_profile,
                    agent_profiles=agent_profiles,
                    fm=fm,
                    learner_model=learner_model,
                    partner_model=partner_model,
                    memory_prompt="",      # no chronicle
                    max_turns=args.max_turns,
                    learner_goal=learner_goal,
                    rubric=None,           # no rubric gate
                    partner_profile=None,
                    judge_self_consistency_k=1,  # single-sample eval for speed
                )
            )
            scores = result.learner_scores
        except Exception as e:
            print(f"  [seed {i}] ERROR: {e}")
            scores = {}

        goal = float(scores.get("goal", 0.0))
        rel = float(scores.get("relationship", 0.0))
        source = scenario.interaction_type or "unknown"

        for dim, val in scores.items():
            dim_accum[dim].append(float(val))
        by_source[source].append(goal)

        running_goal_mean = sum(dim_accum["goal"]) / len(dim_accum["goal"]) if dim_accum["goal"] else 0.0

        print(
            f"[{i+1:3d}/{len(seeds)}] {source:<22} | "
            f"goal={goal:.1f} rel={rel:.1f} | "
            f"running_avg_goal={running_goal_mean:.2f}"
        )

        results.append({
            "seed_idx": i,
            "id": scenario.id,
            "source": source,
            "scenario": scenario.scenario[:120],
            "learner_goal": learner_goal[:120],
            "scores": scores,
        })

        # Save incrementally so you can read partial results live
        _write_output(args, results, dim_accum, by_source, learner_model, partner_model)

    # Final print
    print("\n=== BASELINE EVAL SUMMARY ===")
    goal_m, goal_s = _mean_std(dim_accum.get("goal", []))
    rel_m, rel_s = _mean_std(dim_accum.get("relationship", []))
    bel_m, _ = _mean_std(dim_accum.get("believability", []))
    print(f"  GOAL     : {goal_m:.3f} ± {goal_s:.3f}  (n={len(dim_accum.get('goal',[]))})")
    print(f"  REL      : {rel_m:.3f} ± {rel_s:.3f}")
    print(f"  BEL      : {bel_m:.3f}")
    print(f"\nBy source (mean GOAL):")
    for src, vals in sorted(by_source.items()):
        m, s = _mean_std(vals)
        print(f"  {src:<25} {m:.2f} ± {s:.2f}  (n={len(vals)})")
    print(f"\nFull results written to {args.output}")


def _write_output(args, results, dim_accum, by_source, learner_model, partner_model):
    summary = {}
    for dim, vals in dim_accum.items():
        m, s = _mean_std(vals)
        summary[f"{dim}_mean"] = m
        summary[f"{dim}_std"] = s
    summary["n"] = len(results)

    by_source_summary = {}
    for src, vals in by_source.items():
        m, s = _mean_std(vals)
        by_source_summary[src] = {"goal_mean": m, "goal_std": s, "n": len(vals)}

    out = {
        "timestamp": datetime.now().isoformat(),
        "args": {
            "seeds_path": args.seeds_path,
            "learner_model": learner_model,
            "partner_model": partner_model,
            "max_turns": args.max_turns,
        },
        "summary": summary,
        "by_source": by_source_summary,
        "per_scenario": results,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
