"""Production curriculum runner for Social OMNI-EPIC (Gen-90 launch).

Generates an expanding bank of social scenarios from the 90 SOTOPIA seeds using
hierarchical Thompson Sampling for anchor selection and ExpeL within-episode memory.
Supports resume — run in manual batches without losing progress.

Output layout (results/{run_name}/):
    bank/seeds/<env_pk>.json      phase0-annotated seeds (written once at seeding; view/export)
    bank/generated/<id>.json      EVERY completed generated scenario (all classifications) = Generated-N
    quarantine/iter_<N>.json      generation failures / gate quarantines / episode-error discards
    trajectories.json             ExpeL-extract-ready pool (rebuilt from bank/generated each checkpoint)
    chronicles/<id>.md            skills_final_md (reflexion strings) per generated scenario
    lineage.json                  {id: {parent_id, root_seed_env_pk, depth, operator, classification, source}}
    summary.json                  classification/operator counts, lp stats, models, n
    archive_latest.json           CANONICAL resume state (Thompson posteriors + votes)
    archive_iter_<N>.json         periodic snapshots
    metrics.json                  per-iteration log
    compute_report.json           per-FM-instance call/token meter

Resume: re-run with the same run_name. archive_latest.json is the source of Thompson state;
folder contents are exports and are never read to reconstruct posteriors. The stopping driver
is generated_count = number of completed generated scenarios (too_easy ∪ frontier ∪ beyond).

    python scripts/run_curriculum.py run_name=gen90_expel \
        seed_from_phase0_dir=results/expel_phase0_Base90_ExpeL \
        use_expel_memory=true stopping.N=90 iterations=250
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from social_omni_epic.tracing_fm import print_info, print_step, print_warn

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.adversarial_agent import AdversarialAgent
from social_omni_epic.archive import Archive
from social_omni_epic.coherence_check import CoherenceChecker
from social_omni_epic.data_models import SocialScenario, K_VOTES_EQUIV
from social_omni_epic.expel_export import flush_aggregates
from social_omni_epic.fm import FM
from social_omni_epic.meta_reflection import MetaReflectionModule
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.reflection_module import ReflectionModule
from social_omni_epic.scenario_title import ScenarioTitleGenerator
from social_omni_epic.seeds import load_sotopia_seeds_with_embeddings
from social_omni_epic.task_generator import TaskGenerator


# ---------------------------------------------------------------------------
# Shared machinery now lives in social_omni_epic.generation_cell so the grid driver
# (scripts/run_grid_generate.py) can reuse it. Aliased to the old private names so every
# call site in the main loop below is untouched — the Stage-B acceptance gate is that this
# script still behaves identically.
# ---------------------------------------------------------------------------

from social_omni_epic.generation_cell import (  # noqa: E402
    Services as _Services,
    context_from_archive,
    cosine as _cosine,
    count_generated as _count_generated,
    direction_sanity as _direction_sanity,
    run_generation_cell,
    write_compute_report as _write_compute_report,
    write_lineage as _write_lineage_scenarios,
    write_quarantine as _write_quarantine,
)

def _write_lineage(run_dir: Path, archive: Archive) -> None:
    """Adapter: the shared writer takes a scenario iterable, not an Archive."""
    _write_lineage_scenarios(run_dir, archive.state.tasks)


# ---------------------------------------------------------------------------
# Per-scenario pipeline (one concurrent task per batch slot)
# ---------------------------------------------------------------------------

async def _run_one_scenario(
    anchor_idx: int,
    archive: Archive,
    svc: _Services,
    config: DictConfig,
    global_iter: int,
    run_dir: Path,
) -> tuple[str, Optional[SocialScenario], dict, int]:
    """Generate -> gates -> K-loop for one scenario, with archive-derived context.

    Thin wrapper: context_from_archive() is the original header (archive-wide exemplar KNN,
    dead-end negatives, existing_types, diversity embeddings, operator-from-band) and
    run_generation_cell() is the original body, both moved verbatim.
    """
    ctx = context_from_archive(archive, anchor_idx, config)
    terminal_state, scenario, info = await run_generation_cell(
        ctx, svc, config, iteration=global_iter, run_dir=run_dir,
    )
    return terminal_state, scenario, info, anchor_idx


# ---------------------------------------------------------------------------
# Archive seeding
# ---------------------------------------------------------------------------

def _seed_archive(archive: Archive, fm: FM, config: DictConfig) -> None:
    """Flat seeding (no phase0 calibration): 90 seeds, agent 0 learner, flat Beta(1,1)."""
    seeds = load_sotopia_seeds_with_embeddings(
        fm=fm, seeds_path=config.get("seeds_path", "data/sotopia_90_seeds.jsonl"),
        limit=config.get("seed_limit"), both_perspectives=False,
    )
    for scn in seeds:
        scn.root_seed_env_pk = scn.source_env_id
        scn.lineage_depth = 0
        archive.add_successful(scn)
    print(f"Archive seeded (flat): {archive.size} entries")


def _seed_archive_from_phase0(archive: Archive, fm: FM, config: DictConfig, run_dir: Path) -> None:
    """LP-initialized seeding from a phase0 ExpeL results dir.

    Carries per-seed classification + LP as metadata; classification drives the first
    operator; LP enters Thompson ONLY as a soft asymmetric prior (frontier → Beta(2,1),
    else Beta(1,1)); alpha_votes = beta_votes = 0. Writes bank/seeds/ export.
    """
    phase0_dir = Path(config.seed_from_phase0_dir)
    seeds_dir = phase0_dir / "seeds"
    if not seeds_dir.exists():
        print(f"ERROR: phase0 seeds dir not found: {seeds_dir}", file=sys.stderr)
        sys.exit(1)

    seeds = load_sotopia_seeds_with_embeddings(
        fm=fm, seeds_path=config.get("seeds_path", "data/sotopia_90_seeds.jsonl"),
        limit=config.get("seed_limit"), both_perspectives=False,
    )
    by_env = {s.source_env_id: s for s in seeds}

    sp = config.get("seed_prior", {}) or {}
    f_a = float(sp.get("frontier_alpha", 2.0)); f_b = float(sp.get("frontier_beta", 1.0))
    d_a = float(sp.get("default_alpha", 1.0)); d_b = float(sp.get("default_beta", 1.0))
    allow_uncal = bool(config.get("allow_uncalibrated", False))

    hist: dict[str, int] = {}
    matched = 0
    missing_class: list[str] = []
    for p in sorted(seeds_dir.glob("seed_*.json")):
        rec = json.loads(p.read_text())
        env_pk = rec.get("env_pk")
        scn = by_env.get(env_pk)
        if scn is None:
            print_warn(f"  phase0 seed {p.name} (env_pk={env_pk}) has no raw-seed match — skipped")
            continue
        cls = rec.get("classification")
        scn.classification = cls
        scn.lp_value = rec.get("lp_value")
        scn.lp_votes = int(rec.get("lp_votes") or 0)
        scn.terminal_success = bool(rec.get("terminal_success"))
        scn.n_attempts = int(rec.get("n_attempts") or 0)
        scn.scenario_title = rec.get("scenario_title") or scn.scenario_title
        scn.social_dynamic = rec.get("social_dynamic") or scn.social_dynamic
        scn.target_perspective = rec.get("target_perspective") or scn.target_perspective
        # Soft asymmetric prior; votes stay zero (real child LP fills them in).
        if cls == "frontier":
            scn.prior_alpha, scn.prior_beta = f_a, f_b
        else:
            scn.prior_alpha, scn.prior_beta = d_a, d_b
        scn.alpha_votes = scn.beta_votes = 0.0
        scn.root_seed_env_pk = scn.source_env_id
        scn.lineage_depth = 0
        scn.parent_id = None
        archive.add_successful(scn)
        matched += 1
        hist[cls or "uncalibrated"] = hist.get(cls or "uncalibrated", 0) + 1
        if not cls:
            missing_class.append(env_pk)

    if missing_class and not allow_uncal:
        print(f"ERROR: {len(missing_class)} seed(s) lack classification "
              f"(set allow_uncalibrated=true to proceed): {missing_class[:5]}...", file=sys.stderr)
        sys.exit(1)

    # bank/seeds export.
    seeds_export = run_dir / "bank" / "seeds"
    seeds_export.mkdir(parents=True, exist_ok=True)
    for scn in archive.state.tasks:
        (seeds_export / f"{scn.source_env_id or scn.id}.json").write_text(
            json.dumps(scn.model_dump(exclude={"embedding"}), indent=2, default=str)
        )

    print_step("Seed classification / prior histogram")
    for cls, n in sorted(hist.items()):
        prior = f"Beta({f_a},{f_b})" if cls == "frontier" else f"Beta({d_a},{d_b})"
        print_info(f"  {cls:16s} n={n:3d}  prior={prior}")
    print(f"Archive seeded from phase0: {matched} seeds (all target_agent_idx=0)")


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

    # --- Startup asserts (§9.7) ---
    if config.get("random_seed") is None:
        print("ERROR: random_seed must be set.", file=sys.stderr)
        sys.exit(1)
    learner_provider = str(config.learner_model).split("/")[0]
    judge_provider = str(config.judge.model).split("/")[0]
    if learner_provider == judge_provider:
        print(f"ERROR: judge provider ({judge_provider}) must differ from learner provider "
              f"({learner_provider}) — cross-lab judge required.", file=sys.stderr)
        sys.exit(1)
    np.random.seed(config.random_seed)

    from hydra.utils import get_original_cwd
    run_dir = Path(get_original_cwd()) / "results" / config.run_name
    for d in (run_dir / "bank" / "seeds", run_dir / "bank" / "generated", run_dir / "quarantine"):
        d.mkdir(parents=True, exist_ok=True)
    (run_dir / "README.txt").write_text(
        "bank/generated/ = every completed generated scenario (all classifications) = Generated-N.\n"
        "bank/seeds/ = the phase0-annotated SOTOPIA-90 seeds. quarantine/ = gen/episode/LP failures.\n"
        "archive_latest.json is the canonical Thompson resume state; folders are exports.\n"
        "Run `run_expel_baseline.py extract --out <this dir>` to produce insights.json.\n"
    )

    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    # Three generation-side FMs so the grid can point them at different models. Here they are all
    # built from config.model, which is what gen-90 did (one shared FM), so behaviour is unchanged.
    fm = FM(model=config.model, temperature=config.temperature)
    fm_judge = FM(model=config.judge.model, temperature=float(config.judge.get("lp_temperature", 0.3)))
    svc = _Services(
        fm_generator=fm, fm_judge=fm_judge, fm_reflection=fm, fm_gates=fm,
        task_gen=TaskGenerator(fm, num_examples=config.task_generator.num_examples,
                               num_failed_examples=0, max_retries=config.task_generator.max_retries),
        moi=ModelOfInterestingness(fm, num_examples=config.moi.num_examples),
        coherence_checker=CoherenceChecker(fm),
        title_gen=ScenarioTitleGenerator(fm),
        reflection_mod=ReflectionModule(fm), meta_mod=MetaReflectionModule(fm),
        adversarial=AdversarialAgent(fm),
        run_single_episode=run_single_episode,
        scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
    )
    fms = {"learner_and_generator": fm, "judge": fm_judge}
    child_prior_mass = float(config.get("child_prior_mass", 4.0))

    # Resume or fresh start.
    archive = Archive(checkpoint_dir=str(run_dir))
    ckpt_file = run_dir / "archive_latest.json"
    if ckpt_file.exists():
        archive.load_checkpoint(str(ckpt_file))
        print(f"Resumed from checkpoint: {archive.size} archive entries")
    elif config.get("seed_from_phase0_dir"):
        _seed_archive_from_phase0(archive, fm, config, run_dir)
    else:
        _seed_archive(archive, fm, config)

    generated_count = _count_generated(run_dir)
    solved_count = sum(1 for s in archive.state.tasks
                       if s.source != "seed_sotopia" and s.terminal_success)
    stopping_N = config.get("stopping", {}).get("N", None)
    batch_size = int(config.get("batch_size", 4))

    metrics_path = run_dir / "metrics.json"
    metrics_log: list[dict] = json.loads(metrics_path.read_text()) if metrics_path.exists() else []

    # Anchor selection mode: "thompson" (default) or "random" (the DOF-1 ablation baseline).
    # Override on the CLI with anchor_selection=random. See docs/post_run_experiments.md §3.
    anchor_selection_mode = str(config.get("anchor_selection", "thompson")).lower()
    if anchor_selection_mode not in ("thompson", "random"):
        anchor_selection_mode = "thompson"

    print(f"Run: iterations={config.iterations} | batch_size={batch_size} | "
          f"anchor_selection={anchor_selection_mode} | "
          f"generated_so_far={generated_count} | stopping_N={stopping_N} | archive={archive.size}")

    async def _run_all() -> None:
        nonlocal generated_count, solved_count
        global_iter = len(metrics_log)
        iterations_done = 0
        pbar = tqdm(total=config.iterations, desc="curriculum")

        while iterations_done < config.iterations:
            if stopping_N and generated_count >= int(stopping_N):
                print(f"Target reached: {generated_count} completed generated scenarios.")
                break
            if archive.size == 0:
                print("Archive empty — cannot continue.")
                break

            current_batch = min(batch_size, config.iterations - iterations_done)
            batch_anchor_indices: list[int] = []
            for b in range(current_batch):
                idx = archive.select_anchor(anchor_selection_mode)
                archive.record_selection(idx, global_iter + b)
                batch_anchor_indices.append(idx)

            print_step(f"Batch | generated={generated_count} | archive={archive.size} | "
                       f"selecting {current_batch} anchor(s)")

            raw_results = await asyncio.gather(
                *[_run_one_scenario(anchor_idx=ai, archive=archive, svc=svc, config=config,
                                    global_iter=global_iter + i, run_dir=run_dir)
                  for i, ai in enumerate(batch_anchor_indices)],
                return_exceptions=True,
            )

            batch_added: list[tuple[str, list]] = []  # (id, embedding) for within-batch dupe log
            for result in raw_results:
                global_iter += 1
                iterations_done += 1
                pbar.update(1)

                if isinstance(result, Exception):
                    print_warn(f"Task raised uncaught exception: {result}")
                    _write_quarantine(run_dir, global_iter, "uncaught_exception", {"error": str(result)})
                    archive.add_failed_generation({"reason": f"uncaught: {result}"})
                    metrics_log.append({"iteration": global_iter, "terminal_state": "exception"})
                    continue

                terminal_state, scenario, info, anchor_idx = result
                anchor_id = archive.state.tasks[anchor_idx].id
                anchor_emb = archive.state.tasks[anchor_idx].embedding
                final_scores = info.get("final_scores", {})
                lp_improved = info.get("lp_improved_votes", 0)
                lp_total = info.get("lp_votes", 0)

                if terminal_state in ("generation_failed", "discarded"):
                    # Soft posterior penalty so Thompson deprioritises repeatedly-failing anchors.
                    # Charge 1 beta vote (0 improved out of 1) — much lighter than K_VOTES_EQUIV
                    # but enough to dilute a flat Beta(1,1) after a few consecutive failures.
                    archive.record_child_outcome(anchor_idx, improved_votes=0, total_votes=1)
                    _write_quarantine(run_dir, global_iter, terminal_state,
                                      {"anchor_id": anchor_id, **{k: v for k, v in info.items()
                                                                  if k not in ("skill_attempts",)}})
                    metrics_log.append({
                        "iteration": global_iter,
                        "terminal_state": terminal_state,
                        "anchor_id": anchor_id,
                        "reason": info.get("reason"),
                        "gate_fail_log": info.get("gate_fail_log", []),
                    })
                    continue

                # Completed scenario → add to archive with capped warm-start prior + LP votes.
                scenario.prior_alpha, scenario.prior_beta = archive.child_prior_from_parent(
                    anchor_idx, child_prior_mass=child_prior_mass)
                archive.add_task(scenario)
                archive.record_child(anchor_idx)
                if terminal_state == "too_easy":
                    archive.record_child_outcome(anchor_idx, 0, K_VOTES_EQUIV)
                elif terminal_state == "frontier":
                    archive.record_child_outcome(anchor_idx, lp_improved, max(lp_total, 1))
                    if scenario.terminal_success:
                        archive.record_solved_child(anchor_idx)
                        solved_count += 1
                else:  # beyond_frontier
                    archive.record_child_outcome(anchor_idx, 0, lp_total if lp_total > 0 else K_VOTES_EQUIV)
                generated_count += 1

                # Within-batch near-dupe (log-only — the snapshot gate can't see same-batch peers).
                dupe = 0.0
                if scenario.embedding:
                    for _id, emb in batch_added:
                        dupe = max(dupe, _cosine(scenario.embedding, emb))
                    batch_added.append((scenario.id, scenario.embedding))
                if dupe > float(config.get("diversity_similarity_threshold", 0.92)):
                    print_warn(f"within_batch_dupe: {scenario.id} max_sim={dupe:.3f}")

                metrics_log.append({
                    "iteration": global_iter,
                    "terminal_state": terminal_state,
                    "classification": getattr(scenario, "classification", terminal_state),
                    "mutation_operator": scenario.mutation_operator,
                    "mutated_slots": scenario.mutated_slots,
                    "lp_value": info.get("lp_value", 0.0),
                    "n_error_votes": info.get("n_error_votes", 0),
                    "n_attempts": scenario.n_attempts,
                    "admitted_moi_rank": info.get("admitted_moi_rank", -1),
                    "parent_child_cosine": round(_cosine(scenario.embedding, anchor_emb), 4)
                                           if (scenario.embedding and anchor_emb) else None,
                    "within_batch_dupe": round(dupe, 4),
                    "goal": final_scores.get("goal", 0.0),
                    "relationship": final_scores.get("relationship", 0.0),
                    "generated_count": generated_count,
                    "solved_count": solved_count,
                    "archive_size": archive.size,
                    "anchor_id": anchor_id,
                })

                if stopping_N and generated_count >= int(stopping_N):
                    break

            # Checkpoint after every batch (saved BEFORE the direction-sanity check below, so a
            # halt's "losslessly resumable" promise holds at print time).
            archive.save_checkpoint(global_iter)
            metrics_path.write_text(json.dumps(metrics_log, indent=2))
            flush_aggregates(run_dir, learner_model=str(config.learner_model),
                             judge_model=str(config.judge.model))
            _write_lineage(run_dir, archive)
            _write_compute_report(run_dir, fms)

            # Direction-sanity monitor (reads the summary.json just flushed above).
            # WARN-ONLY (Patch 10 §C downgraded from hard-halt per run-owner preference): the
            # check fired on razor-thin margins (e.g. 0.286 vs 0.25 ≈ one child), so it now logs
            # the trend and flags the gross-failure signature but NEVER stops the run. The
            # per-operator classification table in summary.json remains the post-hoc direction
            # evidence; halting was only a compute-saving safety, not required for the claim.
            ds = _direction_sanity(run_dir)
            print_info(
                f"direction-sanity: escalate too_easy={ds['escalate_too_easy']}/{ds['escalate_total']} "
                f"(r={ds['escalate_rate']}) · lateral too_easy={ds['lateral_too_easy']}/{ds['lateral_total']} "
                f"(r={ds['lateral_rate']}) · armed={ds['armed']}"
            )
            if ds["halt"]:
                print_warn(
                    f"direction-sanity FLAG (not halting): escalate too_easy-rate ({ds['escalate_rate']}) > "
                    f"lateral ({ds['lateral_rate']}) with {ds['escalate_too_easy']} too_easy escalate "
                    "children — escalate may be under-escalating. Continuing; review the per-operator "
                    "split in summary.json."
                )

        pbar.close()

    asyncio.run(_run_all())

    archive.save_checkpoint(0)
    metrics_path.write_text(json.dumps(metrics_log, indent=2))
    n = flush_aggregates(run_dir, learner_model=str(config.learner_model),
                         judge_model=str(config.judge.model))
    _write_lineage(run_dir, archive)
    _write_compute_report(run_dir, fms)
    print(f"Done. generated={generated_count} (records={n}) | solved={solved_count} | "
          f"archive={archive.size} | dir={run_dir}")


if __name__ == "__main__":
    main()
