"""Baseline 2 — Reflexion (§5.1).

Faithfully implements the Reflexion paradigm:
  - Run each SOTOPIA seed up to max_attempts tries
  - After each failure: verbal reflection (what went wrong, what to try next)
  - Accumulated verbal reflections passed as memory_prompt for next attempt
  - At episode end: ALL verbal memory is discarded (no cross-episode persistence)

Usage:
  python -m experiments.reflexion_baseline
  python -m experiments.reflexion_baseline --seed-limit 3
  python -m experiments.reflexion_baseline --max-attempts 3 --output output/reflexion
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.success_detector import SuccessDetector


# ---------------------------------------------------------------------------
# Verbal reflection prompt
# ---------------------------------------------------------------------------

_REFLECT_SYSTEM = """You are a social skills coach analyzing a failed social interaction episode.

Given the transcript of a failed attempt and any prior verbal reflections, produce a concise reflection note (3-5 bullet points) that:
1. Identifies the specific moment(s) the interaction went wrong
2. Identifies the root cause (misread of social cues, wrong strategy, timing, etc.)
3. Proposes a concrete alternative approach for the next attempt
4. Notes any assumptions that proved wrong

Keep it actionable and concise. The output will be prepended to the next attempt's context."""


def _build_reflection_prompt(
    transcript: list[dict],
    prior_reflections: list[str],
    scenario: str,
    goal: str,
    attempt_num: int,
) -> str:
    parts = [
        f"SCENARIO: {scenario}",
        f"YOUR GOAL: {goal}",
        f"ATTEMPT: {attempt_num}",
    ]
    if prior_reflections:
        parts.append("\nPRIOR REFLECTIONS (from earlier failed attempts):")
        for i, r in enumerate(prior_reflections, 1):
            parts.append(f"--- Reflection {i} ---\n{r}")
    parts.append(f"\nTRANSCRIPT OF FAILED ATTEMPT {attempt_num}:")
    for t in transcript:
        parts.append(f"[T{t['turn']}] {t['sender']}→{t['receiver']}: {t['content']}")
    parts.append("\nWrite your reflection on what went wrong and what to try next.")
    return "\n\n".join(parts)


def _format_reflections_as_prompt(reflections: list[str]) -> str:
    if not reflections:
        return ""
    header = "=== Verbal Reflections (from prior failed attempts this episode) ===\n"
    body = "\n\n".join(
        f"Reflection {i+1}:\n{r}" for i, r in enumerate(reflections)
    )
    return header + body + "\n=== End of Reflections ==="


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_one_seed(
    scenario,
    fm: FM,
    run_single_episode,
    scenario_to_sotopia_profiles,
    success_detector: SuccessDetector,
    learner_model: str,
    partner_model: str,
    evaluator_model: str,
    max_attempts: int = 5,
    max_turns: int = 20,
) -> dict:
    """Run one seed with Reflexion. Returns a result dict."""
    verbal_reflections: list[str] = []
    target_idx = 0
    goal = scenario.agent_goals[target_idx] if scenario.agent_goals else ""

    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)

    outcome = "failed"
    final_scores = {}
    attempts_used = 0

    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        memory_prompt = _format_reflections_as_prompt(verbal_reflections)

        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                learner_model=learner_model,
                partner_model=partner_model,
                evaluator_model=evaluator_model,
                memory_prompt=memory_prompt,
                max_turns=max_turns,
            )
        except Exception as e:
            print(f"    [seed {scenario.id}, attempt {attempt}] Episode error: {e}")
            break

        final_scores = result.learner_scores
        if success_detector.is_solved(final_scores):
            outcome = "solved"
            break

        # Generate verbal reflection for next attempt
        if attempt < max_attempts:
            try:
                reflection_prompt = _build_reflection_prompt(
                    transcript=result.transcript,
                    prior_reflections=verbal_reflections,
                    scenario=scenario.scenario,
                    goal=goal,
                    attempt_num=attempt,
                )
                reflection_text = fm.query(
                    _REFLECT_SYSTEM, reflection_prompt, temperature=0.4
                )
                verbal_reflections.append(reflection_text)
            except Exception:
                pass  # missing reflection is non-fatal

    # Discard all verbal memory — no cross-episode persistence
    return {
        "scenario_id": scenario.id,
        "scenario": scenario.scenario[:150],
        "interaction_type": scenario.interaction_type,
        "outcome": outcome,
        "attempts_used": attempts_used,
        "final_scores": final_scores,
        "goal_score": float(final_scores.get("goal", 0.0)),
        "relationship_score": float(final_scores.get("relationship", 0.0)),
        "knowledge_score": float(final_scores.get("knowledge", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflexion baseline")
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--goal-threshold", type=float, default=7.0)
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--learner-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--partner-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--evaluator-model", type=str, default="gpt-5.2")
    parser.add_argument("--seed-data-dir", type=str, default="data/sotopia_seeds")
    parser.add_argument("--episodes-path", type=str, default="data/sotopia_episodes_v1.jsonl")
    parser.add_argument("--output", type=str, default="output/reflexion_baseline")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    from social_omni_epic.episode_runner import run_single_episode  # noqa
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles  # noqa

    fm = FM(model=args.model)
    success_detector = SuccessDetector(goal_threshold=args.goal_threshold)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        seeds = load_sotopia_seeds(
            data_dir=args.seed_data_dir,
            episodes_path=args.episodes_path,
            limit=args.seed_limit,
        )
    except FileNotFoundError as e:
        print(f"ERROR loading seeds: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Running Reflexion baseline on {len(seeds)} seeds, max {args.max_attempts} attempts each.")
    results = []

    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] {seed.scenario[:60]}...")
        res = asyncio.run(
            run_one_seed(
                scenario=seed,
                fm=fm,
                run_single_episode=run_single_episode,
                scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
                success_detector=success_detector,
                learner_model=args.learner_model,
                partner_model=args.partner_model,
                evaluator_model=args.evaluator_model,
                max_attempts=args.max_attempts,
                max_turns=args.max_turns,
            )
        )
        results.append(res)
        print(
            f"  → outcome={res['outcome']} attempts={res['attempts_used']} "
            f"goal={res['goal_score']:.1f}"
        )

    # Summary statistics
    solved = [r for r in results if r["outcome"] == "solved"]
    mean_goal = sum(r["goal_score"] for r in results) / max(len(results), 1)
    mean_attempts = sum(r["attempts_used"] for r in results) / max(len(results), 1)

    summary = {
        "total_seeds": len(results),
        "solved": len(solved),
        "solve_rate": len(solved) / max(len(results), 1),
        "mean_goal_score": mean_goal,
        "mean_attempts_used": mean_attempts,
        "max_attempts": args.max_attempts,
        "per_seed": results,
    }

    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nReflexion baseline complete.\n"
        f"  Solved:     {len(solved)}/{len(results)} ({100*summary['solve_rate']:.1f}%)\n"
        f"  Mean goal:  {mean_goal:.2f}\n"
        f"  Mean tries: {mean_attempts:.2f}\n"
        f"  Results:    {out_path}"
    )


if __name__ == "__main__":
    main()
