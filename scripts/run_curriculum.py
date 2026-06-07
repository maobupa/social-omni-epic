"""Production curriculum runner for Social OMNI-EPIC.

Generates solved-after-biting scenarios from 90 SOTOPIA seeds using hierarchical
Thompson Sampling for anchor selection. Supports resume — run in manual batches of
any size without losing progress.

Output layout (all under results/{run_name}/):
    success/{scenario_id}.json   — solved_after_biting; used by eval retrieval + task gen
    failed/{scenario_id}.json    — bit but never solved; used by task gen as negative examples
    discarded/{iter}.json        — trivially easy; analysis only, not used for retrieval
    archive_latest.json          — full archive state (Thompson priors included); resume point
    archive_iter_{N}.json        — periodic checkpoint snapshots
    metrics.json                 — per-iteration metrics

Resume: just re-run with the same run_name. archive_latest.json is auto-detected.

    # First batch — see how many succeed/fail in 10 runs
    python scripts/run_curriculum.py run_name=my_run iterations=10

    # Continue without restarting
    python scripts/run_curriculum.py run_name=my_run iterations=50

    # Run until 90 solved-after-biting scenarios total
    python scripts/run_curriculum.py run_name=my_run stopping.N=90

Parallel execution: batch_size concurrent episodes per round using asyncio.gather().
Thompson anchor selection is done sequentially within each batch (so n_i is updated
before the next selection — prevents duplicate anchor selection).
"""
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from social_omni_epic.embedding_utils import get_similar_scenarios
from social_omni_epic.fm import FM
from social_omni_epic.meta_reflection import MetaReflectionModule
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.reflection_module import ReflectionModule
from social_omni_epic.scenario_title import ScenarioTitleGenerator, designate_target_agent
from social_omni_epic.seeds import load_sotopia_seeds_with_embeddings
from social_omni_epic.task_generator import TaskGenerator


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_scenario_file(scenario: SocialScenario, folder: Path) -> None:
    """Write scenario as JSON. Uses scenario.id as filename — idempotent across resumes."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{scenario.id}.json"
    if path.exists():
        return
    path.write_text(json.dumps(scenario.model_dump(), indent=2, default=str))


def _save_discarded(iteration: int, reason: str, anchor_id: str, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"iter_{iteration:06d}.json").write_text(
        json.dumps({"iteration": iteration, "reason": reason, "anchor_id": anchor_id}, indent=2)
    )


def _count_solved(run_dir: Path) -> int:
    """Count solved-after-biting scenarios already saved. Source of truth for resume."""
    d = run_dir / "success"
    return len(list(d.glob("*.json"))) if d.exists() else 0


# ---------------------------------------------------------------------------
# Shared stateless services (safe to reference across concurrent async tasks)
# ---------------------------------------------------------------------------

@dataclass
class _Services:
    fm: FM
    task_gen: TaskGenerator
    moi: ModelOfInterestingness
    coherence_checker: CoherenceChecker
    title_gen: ScenarioTitleGenerator
    reflection_mod: ReflectionModule
    meta_mod: MetaReflectionModule
    adversarial: AdversarialAgent
    run_single_episode: object
    scenario_to_sotopia_profiles: object


# ---------------------------------------------------------------------------
# Per-scenario pipeline  (one concurrent task per batch slot)
# ---------------------------------------------------------------------------

async def _run_one_scenario(
    anchor_idx: int,
    archive: Archive,
    svc: _Services,
    config: DictConfig,
    global_iter: int,
) -> tuple[str, Optional[SocialScenario], dict, int]:
    """
    Full pipeline for one scenario: generate → filter gates → two-loop episode.

    Returns (terminal_state, scenario_or_None, info_dict, anchor_idx).
    terminal_state ∈ {"solved_after_biting", "failed", "discarded", "generation_failed"}

    Safe for concurrent execution: reads archive (read-only during gather); archive
    writes happen only in the sequential update step after gather() returns.
    """
    fm = svc.fm
    anchor = archive.state.successful[anchor_idx]

    # --- KNN examples around anchor for generation context ---
    n_examples = int(config.task_generator.num_examples)
    all_embs = archive.get_successful_embeddings()
    if anchor.embedding and all_embs and len(all_embs) >= n_examples:
        src_ids = [s.source_scenario_id for s in archive.state.successful]
        agt_idxs = [s.target_agent_idx for s in archive.state.successful]
        ex_idxs = get_similar_scenarios(
            anchor.embedding, all_embs, num_returns=n_examples,
            source_ids=src_ids, agent_idxs=agt_idxs,
            preferred_agent_idx=anchor.target_agent_idx,
        )
        if anchor_idx not in ex_idxs:
            ex_idxs = [anchor_idx] + ex_idxs[:n_examples - 1]
    else:
        ex_idxs = [anchor_idx]
    examples = [archive.state.successful[i] for i in ex_idxs]

    # --- Failed-task negative examples for generator ---
    n_ep_failed = int(config.task_generator.get("num_episode_failed_examples", 2))
    ep_failed_candidates = [s for s in archive.state.failed_tasks if s.skills_final_md]
    episode_failed: list[SocialScenario] = []
    if ep_failed_candidates and n_ep_failed > 0:
        if anchor.embedding and any(s.embedding for s in ep_failed_candidates):
            neg_idxs = get_similar_scenarios(
                anchor.embedding,
                [s.embedding for s in ep_failed_candidates],
                num_returns=n_ep_failed,
                source_ids=[s.source_scenario_id for s in ep_failed_candidates],
                agent_idxs=[s.target_agent_idx for s in ep_failed_candidates],
            )
            episode_failed = [ep_failed_candidates[i] for i in neg_idxs]
        else:
            episode_failed = ep_failed_candidates[-n_ep_failed:]

    existing_types = (
        list({s.interaction_type for s in archive.state.successful if s.interaction_type})
        if config.task_generator.get("show_existing_types", True) else None
    )

    # --- Generate ---
    use_vs = bool(config.get("use_verbalized_sampling", False))
    scenario = (
        svc.task_gen.generate_with_verbalized_sampling(
            examples, episode_failed_examples=episode_failed,
            existing_types=existing_types or [],
            n_candidates=int(config.get("vs_num_candidates", 5)),
        ) if use_vs else
        svc.task_gen.generate_from_archive(
            examples, episode_failed_examples=episode_failed,
            existing_types=existing_types or [],
        )
    )
    if scenario is None:
        return "generation_failed", None, {"reason": "generation_returned_none"}, anchor_idx

    scenario.iteration = global_iter
    scenario.parent_example_ids = [anchor.id]

    # --- Embed ---
    try:
        scenario.embedding = fm.get_embeddings([scenario.to_text_for_embedding()])[0]
    except Exception as e:
        return "generation_failed", None, {"reason": f"embed_error: {e}"}, anchor_idx

    # --- MoI gate ---
    if config.enable_moi and archive.size >= config.moi.min_archive_size:
        src_ids = [s.source_scenario_id for s in archive.state.successful]
        agt_idxs = [s.target_agent_idx for s in archive.state.successful]
        sim_idxs = get_similar_scenarios(
            scenario.embedding, all_embs, num_returns=config.moi.num_examples,
            source_ids=src_ids, agent_idxs=agt_idxs,
            preferred_agent_idx=scenario.target_agent_idx,
        )
        similar = [archive.state.successful[i] for i in sim_idxs]
        moi_ok = False
        for _m in range(int(config.moi.get("max_edits", 2)) + 1):
            passed, moi_reason, moi_edits = svc.moi.evaluate(scenario, similar)
            scenario.moi_reasoning = moi_reason
            if passed:
                moi_ok = True
                break
            if _m >= int(config.moi.get("max_edits", 2)) or not moi_edits:
                break
            edited = svc.task_gen.edit_scenario(scenario, moi_edits, intent="improve_interestingness")
            if edited is None:
                break
            edited.iteration = global_iter
            edited.parent_example_ids = [anchor.id]
            try:
                edited.embedding = fm.get_embeddings([edited.to_text_for_embedding()])[0]
            except Exception:
                edited = None
                break
            scenario = edited
        if scenario is None or not moi_ok:
            return "generation_failed", scenario, {"reason": "moi_failed"}, anchor_idx

    # --- Coherence gate ---
    scenario, passed_coherence = run_coherence_gate(
        scenario, svc.coherence_checker, svc.task_gen, fm, config, anchor, global_iter
    )
    if not passed_coherence or scenario is None:
        return "generation_failed", None, {"reason": "coherence_failed"}, anchor_idx

    # --- Diversity gate ---
    if config.get("enable_diversity_gate", True) and all_embs and scenario.embedding:
        threshold = float(config.get("diversity_similarity_threshold", 0.92))
        emb_arr = np.array(all_embs)
        s_emb = np.array(scenario.embedding)
        sims = emb_arr @ s_emb / (np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(s_emb) + 1e-9)
        if float(sims.max()) > threshold:
            return "generation_failed", scenario, {"reason": "diversity_failed"}, anchor_idx

    # --- Designate target agent ---
    scenario.target_agent_idx, scenario.target_agent_goal_abstract = designate_target_agent(
        scenario, anchor, fm
    )

    # --- Two-loop curriculum episode ---
    try:
        scenario, terminal_state, _outcome, final_scores, loop_info = await run_episode_two_loop(
            scenario=scenario,
            anchor=anchor,
            task_gen=svc.task_gen,
            reflection_mod=svc.reflection_mod,
            meta_mod=svc.meta_mod,
            adversarial=svc.adversarial,
            title_gen=svc.title_gen,
            coherence_checker=svc.coherence_checker,
            run_single_episode=svc.run_single_episode,
            scenario_to_sotopia_profiles=svc.scenario_to_sotopia_profiles,
            fm=fm,
            config=config,
        )
    except Exception as e:
        import traceback
        print(f"[iter {global_iter}] Episode exception: {e}\n{traceback.format_exc()}")
        return "failed", scenario, {"reason": f"episode_exception: {e}"}, anchor_idx

    loop_info["final_scores"] = final_scores
    return terminal_state, scenario, loop_info, anchor_idx


# ---------------------------------------------------------------------------
# Archive seeding
# ---------------------------------------------------------------------------

def _seed_archive(archive: Archive, fm: FM, config: DictConfig) -> None:
    try:
        seeds = load_sotopia_seeds_with_embeddings(
            fm=fm,
            seeds_path=config.get("seeds_path", "data/sotopia_90_seeds.jsonl"),
            limit=config.get("seed_limit"),
            both_perspectives=config.get("seed_both_perspectives", True),
        )
        print(f"Loaded {len(seeds)} seed scenarios")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    for scn in seeds:
        archive.add_successful(scn)
    print(f"Archive seeded: {archive.size} entries")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="social_omni_epic_curriculum",
)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: API key not set.", file=sys.stderr)
        sys.exit(1)
    np.random.seed(config.random_seed)

    from hydra.utils import get_original_cwd
    run_dir = Path(get_original_cwd()) / "results" / config.run_name
    success_dir = run_dir / "success"
    failed_dir = run_dir / "failed"
    discarded_dir = run_dir / "discarded"
    for d in (success_dir, failed_dir, discarded_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Build services
    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    fm = FM(model=config.model, temperature=config.temperature)
    svc = _Services(
        fm=fm,
        task_gen=TaskGenerator(
            fm,
            num_examples=config.task_generator.num_examples,
            num_failed_examples=0,
            max_retries=config.task_generator.max_retries,
        ),
        moi=ModelOfInterestingness(
            fm,
            num_examples=config.moi.num_examples,
            min_archive_size=config.moi.min_archive_size,
        ),
        coherence_checker=CoherenceChecker(fm),
        title_gen=ScenarioTitleGenerator(fm),
        reflection_mod=ReflectionModule(fm),
        meta_mod=MetaReflectionModule(fm),
        adversarial=AdversarialAgent(fm),
        run_single_episode=run_single_episode,
        scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
    )

    # Resume or fresh start — archive checkpoint lives in run_dir
    archive = Archive(checkpoint_dir=str(run_dir))
    ckpt_file = run_dir / "archive_latest.json"
    if ckpt_file.exists():
        archive.load_checkpoint(str(ckpt_file))
        print(f"Resumed from checkpoint: {archive.size} archive entries")
    else:
        _seed_archive(archive, fm, config)

    # solved_count from success/ folder is the source of truth across resumes
    solved_count = _count_solved(run_dir)
    stopping_N = config.get("stopping", {}).get("N", None)
    batch_size = int(config.get("batch_size", 4))

    metrics_path = run_dir / "metrics.json"
    metrics_log: list[dict] = []
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics_log = json.load(f)

    print(
        f"Run: iterations={config.iterations} | batch_size={batch_size} | "
        f"solved_so_far={solved_count} | stopping_N={stopping_N} | archive={archive.size}"
    )

    async def _run_all() -> None:
        nonlocal solved_count
        global_iter = 0
        iterations_done = 0
        pbar = tqdm(total=config.iterations, desc="curriculum")

        while iterations_done < config.iterations:
            if stopping_N and solved_count >= int(stopping_N):
                print(f"Target reached: {solved_count} solved-after-biting scenarios.")
                break
            if archive.size == 0:
                print("Archive empty — cannot continue.")
                break

            # Sequential Thompson selection within batch: each pick updates n_i immediately
            # so subsequent picks in the same batch see the updated distribution.
            current_batch = min(batch_size, config.iterations - iterations_done)
            batch_anchor_indices: list[int] = []
            for b in range(current_batch):
                idx = archive.thompson_select()
                archive.record_selection(idx, global_iter + b)
                batch_anchor_indices.append(idx)

            # Run batch concurrently — archive is read-only during gather()
            raw_results = await asyncio.gather(
                *[
                    _run_one_scenario(
                        anchor_idx=anchor_idx,
                        archive=archive,
                        svc=svc,
                        config=config,
                        global_iter=global_iter + i,
                    )
                    for i, anchor_idx in enumerate(batch_anchor_indices)
                ],
                return_exceptions=True,
            )

            # Update archive sequentially — all writes happen here, never inside gather()
            for result in raw_results:
                global_iter += 1
                iterations_done += 1
                pbar.update(1)

                if isinstance(result, Exception):
                    print(f"Task raised uncaught exception: {result}")
                    archive.add_failed_generation({"reason": f"uncaught: {result}"})
                    metrics_log.append({"iteration": global_iter, "terminal_state": "exception"})
                    continue

                terminal_state, scenario, info, anchor_idx = result
                anchor_id = archive.state.successful[anchor_idx].id
                final_scores = info.get("final_scores", {})

                if terminal_state == "generation_failed":
                    archive.add_failed_generation({"iteration": global_iter, **info})

                elif terminal_state == "discarded":
                    archive.add_failed_generation({
                        "iteration": global_iter, "reason": "discarded_too_easy"
                    })
                    _save_discarded(global_iter, "discarded_too_easy", anchor_id, discarded_dir)

                elif terminal_state == "solved_after_biting":
                    # Record success on parent first; child inherits parent's updated posterior
                    archive.record_solved_child(anchor_idx)
                    scenario.prior_alpha, scenario.prior_beta = archive.child_prior_from_parent(anchor_idx)
                    archive.add_successful(scenario)
                    archive.record_child(anchor_idx)
                    _save_scenario_file(scenario, success_dir)
                    solved_count += 1
                    print(
                        f"  ✓ solved | total={solved_count} | archive={archive.size} | "
                        f"{scenario.scenario_title or scenario.scenario[:50]}"
                    )

                else:  # failed — bit but never solved across K attempts
                    archive.add_failed_task(scenario)
                    archive.record_child(anchor_idx)
                    _save_scenario_file(scenario, failed_dir)

                metrics_log.append({
                    "iteration": global_iter,
                    "terminal_state": terminal_state,
                    "n_difficulty_edits": info.get("n_difficulty_edits", 0),
                    "goal": final_scores.get("goal", 0.0),
                    "relationship": final_scores.get("relationship", 0.0),
                    "solved_count": solved_count,
                    "archive_size": archive.size,
                    "anchor_id": anchor_id,
                })

                if stopping_N and solved_count >= int(stopping_N):
                    break

            # Checkpoint after every batch
            archive.save_checkpoint(global_iter)
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

        pbar.close()

    asyncio.run(_run_all())

    archive.save_checkpoint(0)
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    print(f"Done. solved={solved_count} | archive={archive.size} | dir={run_dir}")


if __name__ == "__main__":
    main()
