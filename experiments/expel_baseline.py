"""Baseline 3 — ExpeL (§5.1).

Implements Experience-driven Policy Evolution (ExpeL) on fixed SOTOPIA seeds.

Two phases:

  TRAIN phase (gather trajectories + extract insights):
    1. Run all seeds through run_single_episode with no memory (vanilla)
    2. Group into (success, failure) pairs by scenario type
    3. LLM insight extractor applies ADD / UPVOTE / DOWNVOTE / EDIT operations
       to a running insight list, presented with success+failure pairs
    4. Final insight list saved as insights.json with importance scores

  EVAL phase (run seeds WITH top-K insights):
    1. Load trained insight list
    2. Inject top-K insights (sorted by importance DESC) as memory_prompt
    3. Run all seeds again, record scores

Usage:
  python -m experiments.expel_baseline train
  python -m experiments.expel_baseline eval
  python -m experiments.expel_baseline train --seed-limit 5  # smoke test
  python -m experiments.expel_baseline eval  --top-k 5
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
# Insight extraction prompts
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """You are an experience-driven policy optimizer extracting reusable social insights.

You are given:
  - A CURRENT INSIGHT LIST (may be empty on first call)
  - A SUCCESS EXAMPLE and a FAILURE EXAMPLE from similar social scenarios

You may apply the following operations to improve the insight list:

  ADD <insight_text>
    Add a new insight not covered by existing ones. New insights start with importance=2.

  UPVOTE <index>
    Increase importance of insight at 1-based index by 1. Use when a success confirms it.

  DOWNVOTE <index>
    Decrease importance by 1. If importance reaches 0, the insight is implicitly removed.

  EDIT <index> <new_text>
    Replace the text of an existing insight (importance unchanged).

RULES:
  - Insights must be abstract (no proper nouns, no scenario-specific details)
  - Insights describe SOCIAL SKILLS applicable across multiple scenarios
  - Keep the list concise (≤15 insights)
  - Insights should be actionable (start with a verb: "Probe...", "Avoid...", "When X, do Y...")

Output format (one operation per line, then DONE):
  ADD ...
  UPVOTE 3
  DOWNVOTE 1
  EDIT 2 ...
  DONE"""


def _build_extraction_prompt(
    current_insights: list[dict],
    success: dict | None,
    failure: dict | None,
) -> str:
    parts = []

    if current_insights:
        parts.append("CURRENT INSIGHT LIST:")
        for i, ins in enumerate(current_insights, 1):
            parts.append(f"  {i}. [importance={ins['importance']}] {ins['text']}")
    else:
        parts.append("CURRENT INSIGHT LIST: (empty)")

    if success:
        parts.append(f"\nSUCCESS EXAMPLE (scenario: {success['scenario'][:100]}):")
        parts.append(f"  Goal: {success.get('goal', '')[:100]}")
        parts.append(f"  Goal score: {success.get('goal_score', 0):.1f}")
        if success.get("transcript"):
            for t in success["transcript"][:8]:
                parts.append(f"  [T{t['turn']}] {t['sender']}: {t['content'][:80]}")

    if failure:
        parts.append(f"\nFAILURE EXAMPLE (scenario: {failure['scenario'][:100]}):")
        parts.append(f"  Goal: {failure.get('goal', '')[:100]}")
        parts.append(f"  Goal score: {failure.get('goal_score', 0):.1f}")
        if failure.get("transcript"):
            for t in failure["transcript"][:8]:
                parts.append(f"  [T{t['turn']}] {t['sender']}: {t['content'][:80]}")

    parts.append("\nApply operations to improve the insight list. End with DONE.")
    return "\n".join(parts)


def _apply_operations(current: list[dict], operations_text: str) -> list[dict]:
    """Parse and apply ADD/UPVOTE/DOWNVOTE/EDIT operations to the insight list."""
    insights = [dict(ins) for ins in current]  # shallow copy

    for line in operations_text.splitlines():
        line = line.strip()
        if not line or line == "DONE":
            continue

        if line.startswith("ADD "):
            text = line[4:].strip()
            if text:
                insights.append({"text": text, "importance": 2})

        elif line.startswith("UPVOTE "):
            try:
                idx = int(line.split()[1]) - 1
                if 0 <= idx < len(insights):
                    insights[idx]["importance"] += 1
            except (ValueError, IndexError):
                pass

        elif line.startswith("DOWNVOTE "):
            try:
                idx = int(line.split()[1]) - 1
                if 0 <= idx < len(insights):
                    insights[idx]["importance"] -= 1
            except (ValueError, IndexError):
                pass

        elif line.startswith("EDIT "):
            parts = line.split(None, 2)
            if len(parts) == 3:
                try:
                    idx = int(parts[1]) - 1
                    new_text = parts[2].strip()
                    if 0 <= idx < len(insights) and new_text:
                        insights[idx]["text"] = new_text
                except (ValueError, IndexError):
                    pass

    # Remove insights with importance ≤ 0
    insights = [ins for ins in insights if ins["importance"] > 0]
    return insights


def _format_insights_as_prompt(insights: list[dict], top_k: int | None = None) -> str:
    if not insights:
        return ""
    sorted_ins = sorted(insights, key=lambda x: x["importance"], reverse=True)
    if top_k is not None:
        sorted_ins = sorted_ins[:top_k]
    header = "=== Social Skills Insights ===\n"
    body = "\n".join(f"- {ins['text']}" for ins in sorted_ins)
    return header + body + "\n=== End of Insights ==="


# ---------------------------------------------------------------------------
# Train phase
# ---------------------------------------------------------------------------

async def _gather_trajectories(
    seeds,
    run_single_episode,
    scenario_to_sotopia_profiles,
    learner_model: str,
    partner_model: str,
    evaluator_model: str,
    success_detector: SuccessDetector,
    max_turns: int = 20,
) -> list[dict]:
    """Run all seeds with no memory; return trajectory dicts."""
    trajectories = []
    for i, seed in enumerate(seeds):
        print(f"  [gather {i+1}/{len(seeds)}] {seed.scenario[:60]}...")
        env_profile, agent_profiles = scenario_to_sotopia_profiles(seed)
        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                learner_model=learner_model,
                partner_model=partner_model,
                evaluator_model=evaluator_model,
                memory_prompt="",
                max_turns=max_turns,
            )
            scores = result.learner_scores
            trajectories.append({
                "scenario_id": seed.id,
                "scenario": seed.scenario,
                "interaction_type": seed.interaction_type,
                "goal": seed.agent_goals[0] if seed.agent_goals else "",
                "goal_score": float(scores.get("goal", 0.0)),
                "relationship_score": float(scores.get("relationship", 0.0)),
                "solved": success_detector.is_solved(scores),
                "transcript": result.transcript,
            })
        except Exception as e:
            print(f"    Episode failed: {e}")
            trajectories.append({
                "scenario_id": seed.id,
                "scenario": seed.scenario,
                "interaction_type": seed.interaction_type,
                "goal": seed.agent_goals[0] if seed.agent_goals else "",
                "goal_score": 0.0,
                "relationship_score": 0.0,
                "solved": False,
                "transcript": [],
                "error": str(e),
            })
    return trajectories


def _extract_insights(
    trajectories: list[dict],
    fm: FM,
) -> list[dict]:
    """Run the insight extraction loop over trajectory pairs."""
    insights: list[dict] = []

    # Group by interaction_type for pairing
    by_type: dict[str, list[dict]] = {}
    for t in trajectories:
        it = t.get("interaction_type", "unknown")
        by_type.setdefault(it, []).append(t)

    pairs: list[tuple[dict | None, dict | None]] = []
    for it, traj_list in by_type.items():
        successes = [t for t in traj_list if t["solved"]]
        failures = [t for t in traj_list if not t["solved"]]
        # Pair success with failure; leftover presented solo
        for i in range(max(len(successes), len(failures))):
            s = successes[i] if i < len(successes) else None
            f = failures[i] if i < len(failures) else None
            pairs.append((s, f))

    print(f"  Insight extraction: {len(pairs)} pairs/singles")
    for i, (success, failure) in enumerate(pairs):
        prompt = _build_extraction_prompt(insights, success, failure)
        try:
            ops_text = fm.query(_EXTRACT_SYSTEM, prompt, temperature=0.5)
            insights = _apply_operations(insights, ops_text)
        except Exception as e:
            print(f"    Insight extraction pair {i+1} failed: {e}")

    return insights


def run_train(args) -> None:
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

    print(f"ExpeL TRAIN: {len(seeds)} seeds")

    # Phase 1: gather trajectories
    print("Phase 1: gathering trajectories (vanilla, no memory)...")
    trajectories = asyncio.run(
        _gather_trajectories(
            seeds,
            run_single_episode,
            scenario_to_sotopia_profiles,
            args.learner_model,
            args.partner_model,
            args.evaluator_model,
            success_detector,
            max_turns=args.max_turns,
        )
    )

    traj_path = output_dir / "trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(trajectories, f, indent=2)
    print(f"  Trajectories saved: {traj_path}")

    # Phase 2: insight extraction
    print("Phase 2: extracting insights...")
    insights = _extract_insights(trajectories, fm)

    insights_path = output_dir / "insights.json"
    with open(insights_path, "w") as f:
        json.dump(insights, f, indent=2)

    n_solved = sum(1 for t in trajectories if t["solved"])
    mean_goal = sum(t["goal_score"] for t in trajectories) / max(len(trajectories), 1)
    print(
        f"\nExpeL TRAIN complete.\n"
        f"  Vanilla solve rate: {n_solved}/{len(trajectories)} "
        f"({100*n_solved/max(len(trajectories),1):.1f}%)\n"
        f"  Vanilla mean goal:  {mean_goal:.2f}\n"
        f"  Insights extracted: {len(insights)}\n"
        f"  Insights saved:     {insights_path}"
    )


def run_eval(args) -> None:
    from social_omni_epic.episode_runner import run_single_episode  # noqa
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles  # noqa

    output_dir = Path(args.output)
    insights_path = output_dir / "insights.json"
    if not insights_path.exists():
        print(f"ERROR: insights.json not found at {insights_path}. Run train first.", file=sys.stderr)
        sys.exit(1)

    with open(insights_path) as f:
        insights = json.load(f)
    print(f"Loaded {len(insights)} insights from {insights_path}")

    fm = FM(model=args.model)
    success_detector = SuccessDetector(goal_threshold=args.goal_threshold)

    try:
        seeds = load_sotopia_seeds(
            data_dir=args.seed_data_dir,
            episodes_path=args.episodes_path,
            limit=args.seed_limit,
        )
    except FileNotFoundError as e:
        print(f"ERROR loading seeds: {e}", file=sys.stderr)
        sys.exit(1)

    memory_prompt = _format_insights_as_prompt(insights, top_k=args.top_k)
    print(f"ExpeL EVAL: {len(seeds)} seeds, top-{args.top_k} insights injected")

    results = []
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] {seed.scenario[:60]}...")
        env_profile, agent_profiles = scenario_to_sotopia_profiles(seed)
        try:
            result = asyncio.run(
                run_single_episode(
                    env_profile=env_profile,
                    agent_profiles=agent_profiles,
                    learner_model=args.learner_model,
                    partner_model=args.partner_model,
                    evaluator_model=args.evaluator_model,
                    memory_prompt=memory_prompt,
                    max_turns=args.max_turns,
                )
            )
            scores = result.learner_scores
            results.append({
                "scenario_id": seed.id,
                "scenario": seed.scenario[:150],
                "interaction_type": seed.interaction_type,
                "goal_score": float(scores.get("goal", 0.0)),
                "relationship_score": float(scores.get("relationship", 0.0)),
                "knowledge_score": float(scores.get("knowledge", 0.0)),
                "solved": success_detector.is_solved(scores),
            })
            print(f"  → goal={scores.get('goal',0):.1f} solved={results[-1]['solved']}")
        except Exception as e:
            print(f"  Episode failed: {e}")
            results.append({
                "scenario_id": seed.id,
                "scenario": seed.scenario[:150],
                "interaction_type": seed.interaction_type,
                "goal_score": 0.0,
                "relationship_score": 0.0,
                "knowledge_score": 0.0,
                "solved": False,
                "error": str(e),
            })

    solved = [r for r in results if r["solved"]]
    mean_goal = sum(r["goal_score"] for r in results) / max(len(results), 1)

    summary = {
        "total_seeds": len(results),
        "solved": len(solved),
        "solve_rate": len(solved) / max(len(results), 1),
        "mean_goal_score": mean_goal,
        "top_k_insights": args.top_k,
        "per_seed": results,
    }
    out_path = output_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\nExpeL EVAL complete.\n"
        f"  Solved:    {len(solved)}/{len(results)} ({100*summary['solve_rate']:.1f}%)\n"
        f"  Mean goal: {mean_goal:.2f}\n"
        f"  Results:   {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ExpeL baseline")
    sub = parser.add_subparsers(dest="phase", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--seed-limit", type=int, default=None)
    shared.add_argument("--max-turns", type=int, default=20)
    shared.add_argument("--goal-threshold", type=float, default=7.0)
    shared.add_argument("--model", type=str, default="gpt-4.1")
    shared.add_argument("--learner-model", type=str, default="gpt-4.1-mini")
    shared.add_argument("--partner-model", type=str, default="gpt-4.1-mini")
    shared.add_argument("--evaluator-model", type=str, default="gpt-5.2")
    shared.add_argument("--seed-data-dir", type=str, default="data/sotopia_seeds")
    shared.add_argument("--episodes-path", type=str, default="data/sotopia_episodes_v1.jsonl")
    shared.add_argument("--output", type=str, default="output/expel_baseline")

    train_p = sub.add_parser("train", parents=[shared])  # noqa
    eval_p = sub.add_parser("eval", parents=[shared])
    eval_p.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    if not hasattr(args, "top_k"):
        args.top_k = 5

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if args.phase == "train":
        run_train(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
