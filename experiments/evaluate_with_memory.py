"""Experiment 3: post-training evaluation. Run the 90 canonical seeds with
the retrieval bank + skill profile from a completed Phase 1 run.

Usage:
  python -m experiments.evaluate_with_memory \
      --bank output/phase1_full/retrieval_bank.json \
      --skills output/phase1_full/social_skills.md \
      --output-dir output/post_training_eval
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
from social_omni_epic.fm import FM
from social_omni_epic.memory import RetrievalBank, SkillProfile
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles
from social_omni_epic.success_detector import SuccessDetector


async def run_eval(
    bank_path: str,
    skills_path: str,
    learner_model: str,
    partner_model: str,
    evaluator_model: str,
    output_dir: str,
    retrieval_top_k: int = 3,
    seed_limit: int | None = None,
    goal_threshold: float = 7.0,
):
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    detector = SuccessDetector(goal_threshold=goal_threshold)
    fm = FM()
    bank = RetrievalBank(fm, bank_path=bank_path)
    skills = SkillProfile(fm, profile_path=skills_path)
    print(f"Loaded memory: bank={bank.size}, skill_profile_chars={len(skills.text)}")

    seeds = load_sotopia_seeds(restrict_to_episodes_v1=True, limit=seed_limit)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "post_training_scores.json"
    transcript_dir = Path(output_dir) / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text())
            done_ids = {r["scenario_id"] for r in results}
        except Exception:
            results = []

    # Embed scenarios in batch (skip ones already done).
    pending = [s for s in seeds if s.id not in done_ids]
    if pending:
        texts = [s.to_text_for_embedding() for s in pending]
        embs: list[list[float]] = []
        for i in range(0, len(texts), 100):
            embs.extend(fm.get_embeddings(texts[i:i + 100]))
        for s, e in zip(pending, embs):
            s.embedding = e

    for i, scn in enumerate(seeds):
        if scn.id in done_ids:
            continue
        retrieved = bank.retrieve(scn.embedding or [], top_k=retrieval_top_k)
        memory_prompt = ""
        sp_text = skills.format_for_prompt()
        if sp_text:
            memory_prompt += sp_text + "\n\n"
        rb_text = bank.format_for_prompt(retrieved)
        if rb_text:
            memory_prompt += rb_text

        env_profile, agent_profiles = scenario_to_sotopia_profiles(scn)
        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                learner_model=learner_model,
                partner_model=partner_model,
                evaluator_model=evaluator_model,
                memory_prompt=memory_prompt,
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
                "retrieved_ids": [e.scenario_id for e in retrieved],
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
                    retrieved_ids=[e.scenario_id for e in retrieved],
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
        print(f"\nWith-memory mean goal: {sum(goals) / len(goals):.2f} "
              f"(n={len(goals)})")
    if progresses:
        print(f"With-memory mean progress_score: "
              f"{sum(progresses) / len(progresses):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--skills", required=True)
    ap.add_argument("--learner-model", default="gpt-4.1-mini")
    ap.add_argument("--partner-model", default="gpt-4.1-mini")
    ap.add_argument("--evaluator-model", default="gpt-5.2")
    ap.add_argument("--output-dir", default="output/post_training_eval")
    ap.add_argument("--retrieval-top-k", type=int, default=3)
    ap.add_argument("--seed-limit", type=int, default=None)
    ap.add_argument("--goal-threshold", type=float, default=7.0)
    args = ap.parse_args()
    asyncio.run(
        run_eval(
            bank_path=args.bank,
            skills_path=args.skills,
            learner_model=args.learner_model,
            partner_model=args.partner_model,
            evaluator_model=args.evaluator_model,
            output_dir=args.output_dir,
            retrieval_top_k=args.retrieval_top_k,
            seed_limit=args.seed_limit,
            goal_threshold=args.goal_threshold,
        )
    )


if __name__ == "__main__":
    main()
