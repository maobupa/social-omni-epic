"""Phase 2 runner for Social OMNI-EPIC.

Two modes (set via config.run_mode):
  - "phase0"  : Phase 0 behavior (no episodes). Generate → embed → MoI → archive.
                UCB1 selection + verbalized sampling active, but no episodes run.
  - "phase2"  : Full Phase 2. UCB1 anchor selection, verbalized sampling, multi-
                attempt episodes with Skills Chronicle + Reflection loop.

Run from project root:
  python scripts/run_phase2.py
  python scripts/run_phase2.py run_mode=phase0
  python scripts/run_phase2.py iterations=50 checkpoint_dir=output/test_run
"""
import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.adversarial_agent import AdversarialAgent
from social_omni_epic.archive import Archive
from social_omni_epic.coherence_check import CoherenceChecker
from social_omni_epic.curriculum import run_coherence_gate, run_episode_two_loop
from social_omni_epic.data_models import SocialScenario
from social_omni_epic.embedding_utils import (
    compute_cell_coverage,
    get_similar_scenarios,
)
from social_omni_epic.fm import FM
from social_omni_epic.meta_reflection import MetaReflectionModule
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.reflection_module import ReflectionModule
from social_omni_epic.scenario_title import (
    ScenarioTitleGenerator,
    designate_target_agent,
)
from social_omni_epic.seeds import load_sotopia_seeds_with_embeddings
from social_omni_epic.skills_chronicle import SkillsChronicle
from social_omni_epic.task_generator import TaskGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_archive(archive: Archive, fm: FM, config: DictConfig) -> None:
    try:
        seed_scenarios = load_sotopia_seeds_with_embeddings(
            fm=fm,
            seeds_path=config.get("seeds_path", "data/sotopia_90_seeds.jsonl"),
            limit=config.get("seed_limit"),
            both_perspectives=config.get("seed_both_perspectives", True),
        )
        print(f"Loaded {len(seed_scenarios)} Sotopia seed scenarios (embeddings from cache or freshly computed)")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not seed_scenarios:
        return

    for scn in seed_scenarios:
        archive.add_successful(scn)
    print(f"Archive seeded: size={archive.size}")


def _select_anchor_and_examples(
    archive: Archive,
    task_gen: TaskGenerator,
    fm: FM,
    config: DictConfig,
) -> tuple[SocialScenario, int, list[SocialScenario]]:
    """UCB1 select anchor; get KNN examples around it for proposer context."""
    C = float(config.get("ucb1_C", 1.0))
    D = float(config.get("ucb1_D", 0.1))
    anchor_idx = archive.ucb1_select(C=C, D=D)
    anchor = archive.state.successful[anchor_idx]

    # KNN examples around anchor for the task generator context
    n_examples = config.task_generator.num_examples
    all_embs = archive.get_successful_embeddings()
    if anchor.embedding and all_embs and len(all_embs) >= n_examples:
        source_ids = [s.source_scenario_id for s in archive.state.successful]
        agent_idxs = [s.target_agent_idx for s in archive.state.successful]
        example_indices = get_similar_scenarios(
            anchor.embedding, all_embs, num_returns=n_examples,
            source_ids=source_ids, agent_idxs=agent_idxs,
            preferred_agent_idx=anchor.target_agent_idx,
        )
        if anchor_idx not in example_indices:
            example_indices = [anchor_idx] + example_indices[: n_examples - 1]
    else:
        example_indices = [anchor_idx]

    examples = [archive.state.successful[i] for i in example_indices]
    return anchor, anchor_idx, examples


def _sample_episode_failed(
    archive: Archive, n: int, anchor_embedding: list | None = None
) -> list[SocialScenario]:
    """Return up to n episode-failed scenarios that have a skills chronicle.

    If anchor_embedding is provided, returns the n closest by cosine similarity
    (most relevant negative examples for the current generation region).
    Falls back to recency if no embeddings are available.
    """
    candidates = [s for s in archive.state.failed_tasks if s.skills_final_md]
    if not candidates or n <= 0:
        return []
    if anchor_embedding and any(s.embedding for s in candidates):
        embs = [s.embedding for s in candidates]
        idxs = get_similar_scenarios(
            anchor_embedding, embs, num_returns=n,
            source_ids=[s.source_scenario_id for s in candidates],
            agent_idxs=[s.target_agent_idx for s in candidates],
        )
        return [candidates[i] for i in idxs]
    return candidates[-n:] if len(candidates) > n else candidates


def _generate_scenario(
    examples: list[SocialScenario],
    task_gen: TaskGenerator,
    archive: Archive,
    config: DictConfig,
    existing_types: list[str],
    anchor_embedding: list | None = None,
) -> SocialScenario | None:
    n_ep_failed = int(config.get("task_generator", {}).get("num_episode_failed_examples", 2))
    episode_failed = _sample_episode_failed(archive, n_ep_failed, anchor_embedding=anchor_embedding)
    use_vs = bool(config.get("use_verbalized_sampling", False))
    n_cands = int(config.get("vs_num_candidates", 5))
    if use_vs:
        return task_gen.generate_with_verbalized_sampling(
            examples, episode_failed_examples=episode_failed,
            existing_types=existing_types, n_candidates=n_cands,
        )
    return task_gen.generate_from_archive(
        examples, episode_failed_examples=episode_failed, existing_types=existing_types
    )




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="social_omni_epic_phase2",
)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY (or OPENAI_API_KEY) not set.", file=sys.stderr)
        sys.exit(1)
    np.random.seed(config.random_seed)

    run_mode = str(config.get("run_mode", "phase2"))
    assert run_mode in ("phase0", "phase2"), (
        f"run_mode must be phase0 or phase2, got {run_mode}"
    )
    run_episodes = run_mode == "phase2"
    print(f"[run_mode={run_mode}] run_episodes={run_episodes}")

    fm = FM(model=config.model, temperature=config.temperature)
    archive = Archive(checkpoint_dir=config.checkpoint_dir)
    task_gen = TaskGenerator(
        fm,
        num_examples=config.task_generator.num_examples,
        num_failed_examples=0,  # VS doesn't use recently-rejected
        max_retries=config.task_generator.max_retries,
    )
    moi = ModelOfInterestingness(
        fm,
        num_examples=config.moi.num_examples,
        min_archive_size=config.moi.min_archive_size,
    )
    coherence_checker = CoherenceChecker(fm)
    title_gen = ScenarioTitleGenerator(fm)

    # Phase 2 episode modules (lazy for phase0 compatibility)
    reflection_mod = adversarial = meta_mod = None
    run_single_episode = scenario_to_sotopia_profiles = episode_record = None
    transcript_dir = None

    if run_episodes:
        from social_omni_epic.episode_runner import clean_transcript, episode_record, run_single_episode  # noqa
        from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles  # noqa

        reflection_mod = ReflectionModule(fm)
        meta_mod = MetaReflectionModule(fm)
        adversarial = AdversarialAgent(fm)
        transcript_dir = Path(config.checkpoint_dir) / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if config.get("archive_from_ckpt"):
        try:
            archive.load_checkpoint(config.archive_from_ckpt)
            print(f"Loaded archive from {config.archive_from_ckpt}: size={archive.size}")
        except FileNotFoundError:
            print("WARN: archive_from_ckpt not found, ignoring.")

    # Seed archive if fresh
    if archive.size == 0:
        _seed_archive(archive, fm, config)

    metrics_log: list[dict] = []
    solved_count = 0  # solved-after-biting count (the ANNECS-style stopping signal)
    stopping_N = config.get("stopping", {}).get("N", None)

    for iteration in tqdm(range(config.iterations), desc=run_mode):
        # 1. Select anchor via UCB1
        if archive.size == 0:
            print("Archive empty — skipping iteration.")
            continue

        anchor, anchor_idx, examples = _select_anchor_and_examples(archive, task_gen, fm, config)
        archive.record_selection(anchor_idx, iteration)

        existing_types = (
            list({s.interaction_type for s in archive.state.successful if s.interaction_type})
            if config.task_generator.get("show_existing_types", True)
            else None
        )

        # 2. Generate scenario
        scenario = _generate_scenario(examples, task_gen, archive, config, existing_types or [], anchor_embedding=anchor.embedding)
        if scenario is None:
            archive.add_failed_generation({"iteration": iteration, "reason": "generation_failed"})
            continue

        scenario.iteration = iteration
        scenario.parent_example_ids = [anchor.id]

        # 3. Embed
        try:
            scenario.embedding = fm.get_embeddings([scenario.to_text_for_embedding()])[0]
        except Exception as e:
            archive.add_failed_generation({"iteration": iteration, "reason": f"embed_error: {e}"})
            continue

        # 4. MoI auditor+editor: audit social tension/novelty/learnability; edit-up rather than discard
        if config.enable_moi and archive.size >= config.moi.min_archive_size:
            _src_ids = [s.source_scenario_id for s in archive.state.successful]
            _agt_idxs = [s.target_agent_idx for s in archive.state.successful]
            sim_indices = get_similar_scenarios(
                scenario.embedding,
                archive.get_successful_embeddings(),
                num_returns=config.moi.num_examples,
                source_ids=_src_ids,
                agent_idxs=_agt_idxs,
                preferred_agent_idx=scenario.target_agent_idx,
            )
            similar = [archive.state.successful[i] for i in sim_indices]
            max_moi_edits = int(config.moi.get("max_edits", 2))
            moi_ok = False
            for _m in range(max_moi_edits + 1):
                passed, moi_reason, moi_edits = moi.evaluate(scenario, similar)
                scenario.moi_reasoning = moi_reason
                if passed:
                    moi_ok = True
                    break
                if _m >= max_moi_edits or not moi_edits:
                    break
                edited = task_gen.edit_scenario(scenario, moi_edits, intent="improve_interestingness")
                if edited is None:
                    break
                edited.iteration = iteration
                edited.parent_example_ids = [anchor.id]
                try:
                    edited.embedding = fm.get_embeddings([edited.to_text_for_embedding()])[0]
                except Exception:
                    edited = None
                    break
                scenario = edited
            if scenario is None or not moi_ok:
                if scenario is not None:
                    archive.add_failed_interestingness(scenario)
                continue

        # 5. Coherence gate (structural validity + rubric/shortcut validity; retry with feedback)
        scenario, passed_coherence = run_coherence_gate(
            scenario, coherence_checker, task_gen, fm, config, anchor, iteration
        )
        if not passed_coherence or scenario is None:
            archive.add_failed_generation({"iteration": iteration, "reason": "coherence_failed"})
            continue

        # 6. Diversity gate (programmatic cosine similarity; no LLM call)
        if config.get("enable_diversity_gate", True) and archive.size > 0:
            threshold = float(config.get("diversity_similarity_threshold", 0.92))
            all_embs = archive.get_successful_embeddings()
            if all_embs and scenario.embedding:
                emb_arr = np.array(all_embs)
                s_emb = np.array(scenario.embedding)
                sims = emb_arr @ s_emb / (
                    np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(s_emb) + 1e-9
                )
                if float(sims.max()) > threshold:
                    archive.add_failed_interestingness(scenario)
                    continue

        # 7. Designate target agent
        scenario.target_agent_idx, scenario.target_agent_goal_abstract = designate_target_agent(
            scenario, anchor, fm
        )

        # 8. Phase 0 path: no episodes
        if not run_episodes:
            title_data = title_gen.generate(scenario, scenario.target_agent_idx)
            scenario.scenario_title = title_data["scenario_title"]
            scenario.social_dynamic = title_data["social_dynamic"]
            scenario.target_perspective = title_data["target_perspective"]

            archive.add_successful(scenario)
            archive.record_child(anchor_idx)

            if archive.size > 1:
                embs = np.array(archive.get_successful_embeddings())
                coverage = compute_cell_coverage(embs)
                metrics_log.append({
                    "iteration": iteration,
                    "archive_size": archive.size,
                    "cell_coverage": coverage,
                    "total_failed_gen": len(archive.state.failed_generation),
                    "total_failed_interest": len(archive.state.failed_interestingness),
                })
            if iteration % config.checkpoint_every == 0 and iteration > 0:
                archive.save_checkpoint(iteration)
                with open(Path(config.checkpoint_dir) / "metrics.json", "w") as f:
                    json.dump(metrics_log, f, indent=2)
            continue

        # 9. Full Phase 2 episode (difficulty loop → skill loop)
        try:
            scenario, terminal_state, outcome, final_scores, loop_info = asyncio.run(
                _run_phase2_episode(
                    scenario=scenario,
                    anchor=anchor,
                    task_gen=task_gen,
                    reflection_mod=reflection_mod,
                    meta_mod=meta_mod,
                    adversarial=adversarial,
                    title_gen=title_gen,
                    coherence_checker=coherence_checker,
                    run_single_episode=run_single_episode,
                    scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
                    fm=fm,
                    config=config,
                )
            )
        except Exception as e:
            print(f"[iter {iteration}] Phase 2 episode loop failed: {e}")
            archive.add_failed_task(scenario)
            archive.record_child(anchor_idx)
            continue

        # 10. Persist transcript summary
        (transcript_dir / f"iter_{iteration:04d}.json").write_text(
            json.dumps({
                "iteration": iteration,
                "scenario_id": scenario.id,
                "scenario": scenario.scenario[:200],
                "terminal_state": terminal_state,
                "outcome": outcome,
                "n_difficulty_edits": loop_info.get("n_difficulty_edits", 0),
                "final_scores": final_scores,
                "scenario_title": scenario.scenario_title,
                "chronicle_entries": len(
                    SkillsChronicle.from_markdown(scenario.skills_final_md or "").entries
                ),
            }, indent=2, default=str)
        )

        # 11. Archive policy by terminal state
        #   discarded         → not archived (too easy / unbiteable); metrics only
        #   solved_after_biting → add_successful + counts toward stopping
        #   failed            → add_failed_task (conditioning), not counted
        if terminal_state == "discarded":
            archive.add_failed_generation({"iteration": iteration, "reason": "discarded_too_easy"})
        elif terminal_state == "solved_after_biting":
            archive.add_successful(scenario)
            archive.record_child(anchor_idx)
            solved_count += 1
        else:  # failed
            archive.add_failed_task(scenario)
            archive.record_child(anchor_idx)

        metrics_log.append({
            "iteration": iteration,
            "terminal_state": terminal_state,
            "outcome": outcome,
            "n_difficulty_edits": loop_info.get("n_difficulty_edits", 0),
            "goal": final_scores.get("goal", 0.0),
            "relationship": final_scores.get("relationship", 0.0),
            "knowledge": final_scores.get("knowledge", 0.0),
            "overall": final_scores.get("overall_score", 0.0),
            "solved_after_biting": terminal_state == "solved_after_biting",
            "solved_count": solved_count,
            "archive_size": archive.size,
            "interaction_type": scenario.interaction_type,
        })

        print(
            f"[iter {iteration}] {terminal_state} | "
            f"edits={loop_info.get('n_difficulty_edits', 0)} | "
            f"goal={final_scores.get('goal', 0):.1f} | "
            f"archive={archive.size} solved={solved_count} | "
            f"{scenario.scenario[:50]}..."
        )

        if iteration % config.checkpoint_every == 0 and iteration > 0:
            archive.save_checkpoint(iteration)
            with open(Path(config.checkpoint_dir) / "metrics_phase2.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

        # Stop budget (ANNECS-style): stop once we have N solved-after-biting scenarios.
        if stopping_N and solved_count >= int(stopping_N):
            print(f"Stopping: reached {solved_count} solved-after-biting scenarios (N={stopping_N}).")
            break

    archive.save_checkpoint(config.iterations)
    metrics_name = "metrics.json" if not run_episodes else "metrics_phase2.json"
    with open(Path(config.checkpoint_dir) / metrics_name, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Done. Final archive size: {archive.size}")


if __name__ == "__main__":
    main()
