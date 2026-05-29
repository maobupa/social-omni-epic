"""Unified runner for Social OMNI-EPIC.

Three modes (set via config.run_mode):

  - "phase0"            : Phase 0 behavior (no episodes). Generate -> embed
                          -> MoI -> archive. Equivalent to main.py.
  - "phase1_no_memory"  : Phase 1 with real Sotopia episodes but no memory
                          injection. Useful as the "episodes without ICL"
                          ablation.
  - "phase1"            : Full Phase 1. Episodes + retrieval bank + skill
                          profile.

Run from project root:
  python scripts/run_phase1.py run_mode=phase0
  python scripts/run_phase1.py run_mode=phase1_no_memory
  python scripts/run_phase1.py run_mode=phase1
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

from social_omni_epic.archive import Archive
from social_omni_epic.embedding_utils import (
    compute_cell_coverage,
    get_similar_scenarios,
)
from social_omni_epic.fm import FM
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.seeds import build_fallback_seeds, load_sotopia_seeds
from social_omni_epic.task_generator import TaskGenerator

# Phase 1 modules — imported lazily inside main() only when run_mode != "phase0"
# so Phase 0 can run without sotopia installed (e.g., for quick scenario
# generation experiments).


def _generate_one(
    iteration: int,
    archive: Archive,
    task_gen: TaskGenerator,
    moi: ModelOfInterestingness,
    fm: FM,
    config: DictConfig,
    choose_probs: np.ndarray,
):
    """Generate -> validate -> embed -> MoI. Returns (scenario, examples) or
    (None, None) if rejected. Mirrors the Phase 0 main loop logic."""
    examples: list = []
    example_indices: list[int] = []
    if config.use_archive and archive.size > 0:
        examples, example_indices = task_gen.select_examples(
            archive,
            choose_probs,
            config.task_generator.num_examples,
            strategy=config.task_generator.get("example_strategy", "knn"),
        )
        failed = (
            archive.state.failed_interestingness[
                -config.task_generator.num_failed_examples:
            ]
            if config.task_generator.num_failed_examples > 0
            else []
        )
        existing_types = (
            list(
                {
                    s.interaction_type
                    for s in archive.state.successful
                    if s.interaction_type
                }
            )
            if config.task_generator.get("show_existing_types", True)
            else None
        )
        scenario = task_gen.generate_from_archive(
            examples, failed, existing_types=existing_types
        )
    else:
        scenario = task_gen.generate_unconditioned()

    if scenario is None:
        archive.add_failed_generation(
            {"iteration": iteration, "reason": "validation_failed_after_retries"}
        )
        return None, example_indices

    scenario.iteration = iteration
    scenario.parent_example_ids = [ex.id for ex in examples]

    try:
        scenario.embedding = fm.get_embeddings([scenario.to_text_for_embedding()])[0]
    except Exception as e:
        archive.add_failed_generation(
            {"iteration": iteration, "reason": f"embedding_error: {e}"}
        )
        return None, example_indices

    if config.enable_moi and archive.size >= config.moi.min_archive_size:
        sim_indices = get_similar_scenarios(
            scenario.embedding,
            archive.get_successful_embeddings(),
            num_returns=config.moi.num_examples,
        )
        similar = [archive.state.successful[i] for i in sim_indices]
        is_interesting, moi_reason = moi.evaluate(scenario, similar)
        scenario.moi_reasoning = moi_reason
        if not is_interesting:
            archive.add_failed_interestingness(scenario)
            return None, example_indices

    return scenario, example_indices


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="social_omni_epic_phase1",
)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    np.random.seed(config.random_seed)

    run_mode = str(config.get("run_mode", "phase1"))
    assert run_mode in ("phase0", "phase1_no_memory", "phase1"), (
        f"run_mode must be one of phase0|phase1_no_memory|phase1, got {run_mode}"
    )
    run_episodes = run_mode in ("phase1_no_memory", "phase1")
    use_memory = run_mode == "phase1"
    print(f"[run_mode={run_mode}] run_episodes={run_episodes} use_memory={use_memory}")

    fm = FM(model=config.model, temperature=config.temperature)
    archive = Archive(checkpoint_dir=config.checkpoint_dir)
    task_gen = TaskGenerator(
        fm,
        num_examples=config.task_generator.num_examples,
        num_failed_examples=config.task_generator.num_failed_examples,
        max_retries=config.task_generator.max_retries,
    )
    moi = ModelOfInterestingness(
        fm,
        num_examples=config.moi.num_examples,
        min_archive_size=config.moi.min_archive_size,
    )

    # Optional: resume from a Phase 0 archive checkpoint
    if config.get("archive_from_ckpt"):
        try:
            archive.load_checkpoint(config.archive_from_ckpt)
            print(f"Loaded archive from {config.archive_from_ckpt}: "
                  f"size={archive.size}")
        except FileNotFoundError:
            print(f"WARN: archive_from_ckpt not found, ignoring.")

    # Seed the archive (only when fresh)
    if archive.size == 0:
        try:
            seed_scenarios = load_sotopia_seeds(
                data_dir=config.seed_data_dir,
                episodes_path=config.get("episodes_path"),
                restrict_to_episodes_v1=config.get("restrict_to_episodes_v1", True),
                limit=config.get("seed_limit"),
            )
            print(f"Loaded {len(seed_scenarios)} Sotopia seed scenarios")
        except FileNotFoundError:
            if config.get("use_fallback_seeds_if_missing", False):
                seed_scenarios = build_fallback_seeds(fm)
            else:
                print("ERROR: seeds missing and fallback disabled.", file=sys.stderr)
                sys.exit(1)
        if seed_scenarios:
            texts = [s.to_text_for_embedding() for s in seed_scenarios]
            embs: list[list[float]] = []
            for i in range(0, len(texts), 100):
                embs.extend(fm.get_embeddings(texts[i:i + 100]))
            for scn, e in zip(seed_scenarios, embs):
                scn.embedding = e
                archive.add_successful(scn)
        print(f"Archive seeded: size={archive.size}")

    # Phase 1 components (lazy import so phase0 doesn't require sotopia)
    retrieval_bank = None
    skill_profile = None
    success_detector = None
    run_single_episode = None
    scenario_to_sotopia_profiles = None
    episode_record = None
    transcript_dir = None
    if run_episodes:
        from social_omni_epic.episode_runner import (  # noqa
            episode_record,
            run_single_episode,
        )
        from social_omni_epic.sotopia_bridge import (
            scenario_to_sotopia_profiles,
        )  # noqa
        from social_omni_epic.success_detector import SuccessDetector

        success_detector = SuccessDetector(
            goal_threshold=config.get("goal_threshold", 7.0)
        )
        transcript_dir = Path(config.checkpoint_dir) / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
    if use_memory:
        from social_omni_epic.memory import RetrievalBank, SkillProfile

        retrieval_bank = RetrievalBank(
            fm,
            bank_path=f"{config.checkpoint_dir}/retrieval_bank.json",
        )
        skill_profile = SkillProfile(
            fm,
            profile_path=f"{config.checkpoint_dir}/social_skills.md",
            update_every=config.memory.get("skill_profile_update_every", 10),
        )

    choose_probs = np.ones(archive.size) if archive.size > 0 else np.array([])
    metrics_log: list[dict] = []

    for iteration in tqdm(range(config.iterations), desc=run_mode):
        scenario, example_indices = _generate_one(
            iteration, archive, task_gen, moi, fm, config, choose_probs
        )
        # Bookkeeping for choose_probs (matches Phase 0 main.py)
        if archive.size > 0:
            choose_probs = choose_probs + 1
            for idx in example_indices:
                if idx < len(choose_probs):
                    choose_probs[idx] = 0
        if scenario is None:
            continue

        # --- Phase 0 path: simulated success (always accept) ---
        if not run_episodes:
            archive.add_successful(scenario)
            choose_probs = np.append(choose_probs, 1.0)
            if archive.size > 1:
                embs = np.array(archive.get_successful_embeddings())
                coverage = compute_cell_coverage(embs)
                metrics_log.append(
                    {
                        "iteration": iteration,
                        "archive_size": archive.size,
                        "cell_coverage": coverage,
                        "total_failed_gen": len(archive.state.failed_generation),
                        "total_failed_interest": len(
                            archive.state.failed_interestingness
                        ),
                    }
                )
            if iteration % config.checkpoint_every == 0 and iteration > 0:
                archive.save_checkpoint(iteration)
                with open(Path(config.checkpoint_dir) / "metrics.json", "w") as f:
                    json.dump(metrics_log, f, indent=2)
            continue

        # --- Phase 1 path: real episode ---
        memory_prompt = ""
        if use_memory:
            retrieved = retrieval_bank.retrieve(
                scenario.embedding,
                top_k=config.memory.get("retrieval_top_k", 3),
            )
            if config.memory.get("skill_profile_enabled", True):
                sp_text = skill_profile.format_for_prompt()
                if sp_text:
                    memory_prompt += sp_text + "\n\n"
            if config.memory.get("retrieval_enabled", True):
                rb_text = retrieval_bank.format_for_prompt(retrieved)
                if rb_text:
                    memory_prompt += rb_text

        env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)

        try:
            episode_result = asyncio.run(
                run_single_episode(
                    env_profile=env_profile,
                    agent_profiles=agent_profiles,
                    learner_model=config.learner_model,
                    partner_model=config.partner_model,
                    evaluator_model=config.evaluator_model,
                    memory_prompt=memory_prompt,
                    max_turns=config.get("max_turns", 20),
                )
            )
        except Exception as e:
            print(f"[iter {iteration}] Episode failed: {e}")
            scenario.moi_reasoning = (scenario.moi_reasoning or "") + f"\nepisode_error: {e}"
            archive.add_failed_task(scenario)
            continue

        # Persist the full transcript + scores for this episode.
        (transcript_dir / f"iter_{iteration:04d}.json").write_text(
            json.dumps(
                episode_record(
                    episode_result,
                    iteration=iteration,
                    scenario_id=scenario.id,
                    scenario=scenario.scenario,
                    interaction_type=scenario.interaction_type,
                    memory_used=bool(memory_prompt),
                ),
                indent=2,
            )
        )

        scores = episode_result.learner_scores
        progress = success_detector.compute_progress_score(scores)
        is_solved = success_detector.is_solved(scores)
        lp = success_detector.record_and_compute_learning_progress(
            scenario.interaction_type, progress
        )
        scenario.goal_score = float(scores.get("goal", 0.0))
        scenario.progress_score = float(progress)

        if use_memory:
            retrieval_bank.add_episode(episode_result, scenario)
            skill_profile.maybe_update(retrieval_bank)

        if is_solved:
            archive.add_successful(scenario)
            choose_probs = np.append(choose_probs, 1.0)
        else:
            archive.add_failed_task(scenario)

        metrics_log.append(
            {
                "iteration": iteration,
                "type": scenario.interaction_type,
                "goal": scores.get("goal", 0.0),
                "relationship": scores.get("relationship", 0.0),
                "knowledge": scores.get("knowledge", 0.0),
                "overall": scores.get("overall_score", 0.0),
                "progress": progress,
                "learning_progress": lp,
                "solved": is_solved,
                "archive": archive.size,
                "memory": retrieval_bank.size if retrieval_bank else 0,
                "turns": episode_result.num_turns,
            }
        )

        print(
            f"[iter {iteration}] {scenario.scenario[:50]}... | "
            f"goal={scores.get('goal', 0):.1f} rel={scores.get('relationship', 0):.1f} | "
            f"solved={is_solved} | archive={archive.size} | "
            f"mem={retrieval_bank.size if retrieval_bank else 0}"
        )

        if iteration % config.checkpoint_every == 0 and iteration > 0:
            archive.save_checkpoint(iteration)
            with open(Path(config.checkpoint_dir) / "metrics_phase1.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    archive.save_checkpoint(config.iterations)
    metrics_name = "metrics.json" if not run_episodes else "metrics_phase1.json"
    with open(Path(config.checkpoint_dir) / metrics_name, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Done. Final archive size: {archive.size}")


if __name__ == "__main__":
    main()
