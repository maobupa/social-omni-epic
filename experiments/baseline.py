"""Experiment 1: vanilla LLMAgent on all canonical Sotopia seeds, NO memory.

Usage:
  python -m experiments.baseline \
      --learner-model gpt-4o-mini --partner-model gpt-4o-mini \
      --evaluator-model gpt-4o --output-dir output/baseline
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from social_omni_epic.episode_runner import episode_record, run_single_episode
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles
from social_omni_epic.success_detector import SuccessDetector


async def run_baseline(
    learner_model: str,
    partner_model: str,
    evaluator_model: str,
    output_dir: str,
    seed_limit: int | None = None,
    goal_threshold: float = 7.0,
):
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    detector = SuccessDetector(goal_threshold=goal_threshold)
    seeds = load_sotopia_seeds(restrict_to_episodes_v1=True, limit=seed_limit)
    print(f"Running baseline on {len(seeds)} seeds.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "baseline_scores.json"
    transcript_dir = Path(output_dir) / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip ids already in the output file.
    results: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            done_ids = {r["scenario_id"] for r in results}
            print(f"Resuming. {len(done_ids)} already done.")
        except Exception:
            results = []

    for i, scn in enumerate(seeds):
        if scn.id in done_ids:
            continue
        env_profile, agent_profiles = scenario_to_sotopia_profiles(scn)
        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                learner_model=learner_model,
                partner_model=partner_model,
                evaluator_model=evaluator_model,
                memory_prompt="",
            )
        except Exception as e:
            print(f"  [{i + 1}] FAILED: {e}")
            results.append(
                {
                    "scenario_id": scn.id,
                    "scenario": scn.scenario[:200],
                    "error": str(e),
                }
            )
            out_path.write_text(json.dumps(results, indent=2))
            continue

        progress = detector.compute_progress_score(result.learner_scores)
        solved = detector.is_solved(result.learner_scores)
        results.append(
            {
                "scenario_id": scn.id,
                "scenario": scn.scenario[:200],
                "learner_scores": result.learner_scores,
                "partner_scores": result.partner_scores,
                "progress_score": progress,
                "solved": solved,
                "num_turns": result.num_turns,
            }
        )
        out_path.write_text(json.dumps(results, indent=2))
        (transcript_dir / f"{scn.id}.json").write_text(
            json.dumps(
                episode_record(
                    result,
                    scenario_id=scn.id,
                    scenario=scn.scenario,
                    progress_score=progress,
                    solved=solved,
                ),
                indent=2,
            )
        )
        ls = result.learner_scores
        print(
            f"[{i + 1}/{len(seeds)}] goal={ls.get('goal', 0):.1f} "
            f"rel={ls.get('relationship', 0):.1f} "
            f"progress={progress:.3f} solved={solved}"
        )

    goals = [
        r["learner_scores"]["goal"]
        for r in results
        if "learner_scores" in r and "goal" in r["learner_scores"]
    ]
    progresses = [r["progress_score"] for r in results if "progress_score" in r]
    if goals:
        print(f"\nBaseline mean goal: {sum(goals) / len(goals):.2f} "
              f"(n={len(goals)})")
    if progresses:
        print(f"Baseline mean progress_score: "
              f"{sum(progresses) / len(progresses):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner-model", default="gpt-4.1-mini")
    ap.add_argument("--partner-model", default="gpt-4.1-mini")
    ap.add_argument("--evaluator-model", default="gpt-5.2")
    ap.add_argument("--output-dir", default="output/baseline")
    ap.add_argument("--seed-limit", type=int, default=None)
    ap.add_argument("--goal-threshold", type=float, default=7.0)
    args = ap.parse_args()
    asyncio.run(
        run_baseline(
            learner_model=args.learner_model,
            partner_model=args.partner_model,
            evaluator_model=args.evaluator_model,
            output_dir=args.output_dir,
            seed_limit=args.seed_limit,
            goal_threshold=args.goal_threshold,
        )
    )


if __name__ == "__main__":
    main()
