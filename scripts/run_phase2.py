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
    classify_scenario,
    designate_target_agent,
)
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.skills_chronicle import SkillsChronicle
from social_omni_epic.task_generator import TaskGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_archive(archive: Archive, fm: FM, config: DictConfig) -> None:
    try:
        seed_scenarios = load_sotopia_seeds(
            seeds_path=config.get("seeds_path", "data/sotopia_90_seeds.jsonl"),
            limit=config.get("seed_limit"),
            both_perspectives=config.get("seed_both_perspectives", True),
        )
        print(f"Loaded {len(seed_scenarios)} Sotopia seed scenarios")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not seed_scenarios:
        return

    texts = [s.to_text_for_embedding() for s in seed_scenarios]
    embs: list[list[float]] = []
    for i in range(0, len(texts), 100):
        embs.extend(fm.get_embeddings(texts[i : i + 100]))
    for scn, e in zip(seed_scenarios, embs):
        scn.embedding = e
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


def _generate_scenario(
    examples: list[SocialScenario],
    task_gen: TaskGenerator,
    config: DictConfig,
    existing_types: list[str],
) -> SocialScenario | None:
    use_vs = bool(config.get("use_verbalized_sampling", True))
    n_cands = int(config.get("vs_num_candidates", 5))
    if use_vs:
        return task_gen.generate_with_verbalized_sampling(
            examples, existing_types=existing_types, n_candidates=n_cands
        )
    return task_gen.generate_from_archive(examples, existing_types=existing_types)


# ---------------------------------------------------------------------------
# Phase 2 episode loop
# ---------------------------------------------------------------------------

async def _run_phase2_episode(
    scenario: SocialScenario,
    anchor: SocialScenario,
    reflection_mod: ReflectionModule,
    meta_mod: MetaReflectionModule,
    adversarial: AdversarialAgent,
    title_gen: ScenarioTitleGenerator,
    run_single_episode,
    scenario_to_sotopia_profiles,
    config: DictConfig,
) -> tuple[SocialScenario, int, dict]:
    """Run multi-attempt episode with the reflection loop.

    Returns (updated_scenario, outcome, final_scores).
    outcome: 1=first-attempt success, 2=multi-attempt success, 3=all failed.
    """
    max_attempts = int(config.get("max_attempts", 5))
    max_entries = int(config.get("chronicle_max_entries", 8))
    re_reflect = bool(config.adversarial.get("re_reflect_on_rejection", True))

    # Inherit skills chronicle from anchor
    current_chronicle = SkillsChronicle.from_markdown(anchor.skills_final_md or "")

    all_transcripts: list[list[dict]] = []
    all_versions: list[SkillsChronicle] = [deepcopy(current_chronicle)]
    all_edit_reasons: dict[str, str] = {}
    outcome = 3
    final_scores: dict = {}

    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)

    for attempt in range(1, max_attempts + 1):
        memory_prompt = current_chronicle.format_for_prompt(max_entries=max_entries)

        try:
            result = await run_single_episode(
                env_profile=env_profile,
                agent_profiles=agent_profiles,
                learner_model=config.learner_model,
                partner_model=config.partner_model,
                evaluator_model=config.evaluator_model,
                memory_prompt=memory_prompt,
                max_turns=config.get("max_turns", 20),
            )
        except Exception as e:
            print(f"    [attempt {attempt}] Episode error: {e}")
            break

        all_transcripts.append(result.transcript)
        final_scores = result.learner_scores

        from social_omni_epic.success_detector import SuccessDetector
        success_detector = SuccessDetector(goal_threshold=config.get("goal_threshold", 7.0))

        if success_detector.is_solved(final_scores):
            outcome = 1 if attempt == 1 else 2
            break

        if attempt < max_attempts:
            ref_out = reflection_mod.reflect(
                chronicle=current_chronicle,
                scenario=scenario,
                transcripts=all_transcripts,
                prior_edit_reasons=all_edit_reasons,
                attempt_num=attempt,
                anchor_task=anchor,
            )

            # Adversarial check on reflection
            adv_result = adversarial.check_reflection(
                ref_out, result.transcript, anchor_task=anchor
            )
            if not adv_result.approved and re_reflect:
                ref_out = reflection_mod.reflect_with_critique(
                    original_output=ref_out,
                    critique=adv_result.critique,
                    chronicle=current_chronicle,
                    scenario=scenario,
                    transcripts=all_transcripts,
                    prior_edit_reasons=all_edit_reasons,
                    attempt_num=attempt,
                    anchor_task=anchor,
                )

            current_chronicle = ref_out.updated_chronicle
            all_versions.append(deepcopy(current_chronicle))
            all_edit_reasons.update(ref_out.edit_reasons)

    # Meta-reflection (outcome 2 or 3)
    if outcome == 1:
        final_chronicle = current_chronicle
    else:
        final_chronicle = meta_mod.synthesize(
            chronicle_versions=all_versions,
            transcripts=all_transcripts,
            edit_reasons=all_edit_reasons,
            outcome=outcome,
            scenario=scenario,
            anchor_task=anchor,
        )

    # Adversarial check on final chronicle
    adv_final = adversarial.check_final(
        final_chronicle,
        anchor.skills_final_md or "",
        outcome=outcome,
    )

    # Generate SCENARIO_TITLE
    title_data = title_gen.generate(scenario, scenario.target_agent_idx)
    scenario.scenario_title = title_data["scenario_title"]
    scenario.social_dynamic = title_data["social_dynamic"]
    scenario.target_perspective = title_data["target_perspective"]

    scenario.skills_final_md = final_chronicle.to_markdown()
    scenario.goal_score = float(final_scores.get("goal", 0.0))

    return scenario, outcome, final_scores


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
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
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
        from social_omni_epic.episode_runner import episode_record, run_single_episode  # noqa
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
        scenario = _generate_scenario(examples, task_gen, config, existing_types or [])
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

        # 4. MoI + validity check
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
            is_interesting, moi_reason = moi.evaluate(scenario, similar)
            scenario.moi_reasoning = moi_reason
            if not is_interesting:
                archive.add_failed_interestingness(scenario)
                continue

        # 5. Coherence gate (structural validity; retry with issue feedback)
        if config.get("enable_coherence_check", True):
            coherence_feedback = None
            max_coherence = int(config.get("coherence_max_retries", 2))
            passed_coherence = False
            for _c in range(max_coherence + 1):
                c_result = coherence_checker.check(scenario)
                if c_result.passed:
                    passed_coherence = True
                    break
                coherence_feedback = c_result.issues
                scenario = task_gen.generate_from_archive(
                    examples,
                    existing_types=existing_types,
                    coherence_feedback=coherence_feedback,
                )
                if scenario is None:
                    break
                scenario.iteration = iteration
                scenario.parent_example_ids = [anchor.id]
                try:
                    scenario.embedding = fm.get_embeddings(
                        [scenario.to_text_for_embedding()]
                    )[0]
                except Exception:
                    scenario = None
                    break
            if not passed_coherence or scenario is None:
                archive.add_failed_generation({
                    "iteration": iteration,
                    "reason": "coherence_failed",
                    "issues": coherence_feedback or [],
                })
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

        # 7. Classify + designate target agent
        scenario.goal_structure, scenario.info_position = classify_scenario(scenario, fm)
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

        # 9. Full Phase 2 episode
        try:
            scenario, outcome, final_scores = asyncio.run(
                _run_phase2_episode(
                    scenario=scenario,
                    anchor=anchor,
                    reflection_mod=reflection_mod,
                    meta_mod=meta_mod,
                    adversarial=adversarial,
                    title_gen=title_gen,
                    run_single_episode=run_single_episode,
                    scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
                    config=config,
                )
            )
        except Exception as e:
            print(f"[iter {iteration}] Phase 2 episode loop failed: {e}")
            archive.add_failed_task(scenario)
            continue

        # 10. Persist transcript summary
        (transcript_dir / f"iter_{iteration:04d}.json").write_text(
            json.dumps({
                "iteration": iteration,
                "scenario_id": scenario.id,
                "scenario": scenario.scenario[:200],
                "outcome": outcome,
                "final_scores": final_scores,
                "scenario_title": scenario.scenario_title,
                "goal_structure": scenario.goal_structure,
                "info_position": scenario.info_position,
                "chronicle_entries": len(
                    SkillsChronicle.from_markdown(scenario.skills_final_md or "").entries
                ),
            }, indent=2)
        )

        # 11. Archive and bookkeeping
        if outcome in (1, 2):
            archive.add_successful(scenario)
        else:
            archive.add_failed_task(scenario)
        archive.record_child(anchor_idx)

        metrics_log.append({
            "iteration": iteration,
            "outcome": outcome,
            "goal": final_scores.get("goal", 0.0),
            "relationship": final_scores.get("relationship", 0.0),
            "knowledge": final_scores.get("knowledge", 0.0),
            "overall": final_scores.get("overall_score", 0.0),
            "solved": outcome in (1, 2),
            "archive_size": archive.size,
            "interaction_type": scenario.interaction_type,
            "goal_structure": scenario.goal_structure,
            "info_position": scenario.info_position,
        })

        print(
            f"[iter {iteration}] outcome={outcome} | "
            f"goal={final_scores.get('goal', 0):.1f} | "
            f"archive={archive.size} | "
            f"{scenario.scenario[:50]}..."
        )

        if iteration % config.checkpoint_every == 0 and iteration > 0:
            archive.save_checkpoint(iteration)
            with open(Path(config.checkpoint_dir) / "metrics_phase2.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    archive.save_checkpoint(config.iterations)
    metrics_name = "metrics.json" if not run_episodes else "metrics_phase2.json"
    with open(Path(config.checkpoint_dir) / metrics_name, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Done. Final archive size: {archive.size}")


if __name__ == "__main__":
    main()
