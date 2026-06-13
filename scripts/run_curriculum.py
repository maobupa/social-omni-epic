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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from social_omni_epic.tracing_fm import print_info, print_section, print_step, print_warn

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.adversarial_agent import AdversarialAgent
from social_omni_epic.archive import Archive
from social_omni_epic.coherence_check import CoherenceChecker, fuzzy_key_leak_check
from social_omni_epic.validation import surface_novelty_check
from social_omni_epic.curriculum import run_coherence_gate, run_episode_two_loop
from social_omni_epic.data_models import SocialScenario, K_VOTES_EQUIV
from social_omni_epic.embedding_utils import get_similar_scenarios
from social_omni_epic.expel_export import write_scenario_record, write_chronicle, flush_aggregates, write_live_record
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

def _count_generated(run_dir: Path) -> int:
    """Count completed generated scenarios — excludes in-progress stubs and .tmp files."""
    import json as _json
    d = run_dir / "bank" / "generated"
    if not d.exists():
        return 0
    count = 0
    for p in d.glob("*.json"):
        try:
            rec = _json.loads(p.read_text())
            if rec.get("status") != "in_progress":
                count += 1
        except Exception:
            pass
    return count


def _write_quarantine(run_dir: Path, iteration: int, reason: str, info: dict) -> None:
    d = run_dir / "quarantine"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"iter_{iteration:06d}.json").write_text(
        json.dumps({"iteration": iteration, "reason": reason, **info}, indent=2, default=str)
    )


def _write_lineage(run_dir: Path, archive: Archive) -> None:
    lineage = {
        s.id: {
            "parent_id": s.parent_id,
            "root_seed_env_pk": s.root_seed_env_pk,
            "lineage_depth": s.lineage_depth,
            "mutation_operator": s.mutation_operator,
            "classification": s.classification,
            "source": s.source,
        }
        for s in archive.state.tasks
    }
    (run_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))


def _write_compute_report(run_dir: Path, fms: dict) -> None:
    (run_dir / "compute_report.json").write_text(
        json.dumps({name: fm.usage_report() for name, fm in fms.items()}, indent=2)
    )


# Direction-sanity registered stopping rule (Patch 10, §III.C). Pre-declared so it is a legitimate
# stopping rule under the freeze, not a post-hoc edit. Arms only with enough children per operator;
# halts only on the gross-failure signature (escalate strictly more too_easy than lateral, with a
# floor of too_easy escalate children) — strict ">" so a tie/zero (the best case) never halts.
_DIR_SANITY_MIN_CHILDREN = 8   # per operator before the rule arms
_DIR_SANITY_MIN_TOO_EASY = 3   # escalate too_easy children before a halt can fire


def _direction_sanity(run_dir: Path) -> dict:
    """Read the just-flushed summary.json and compare escalate vs lateral too_easy-rates.

    Returns {escalate_*, lateral_*, armed, halt} for logging + the stopping decision.
    """
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
        op_counts = summary.get("per_operator_classification_counts", {}) or {}
    except Exception:
        op_counts = {}

    def _stats(op: str) -> tuple[int, int, float]:
        counts = op_counts.get(op, {}) or {}
        total = sum(int(v) for v in counts.values())
        too_easy = int(counts.get("too_easy", 0))
        return too_easy, total, (too_easy / total if total else 0.0)

    e_te, e_tot, e_rate = _stats("escalate")
    l_te, l_tot, l_rate = _stats("lateral")
    armed = (e_tot >= _DIR_SANITY_MIN_CHILDREN and l_tot >= _DIR_SANITY_MIN_CHILDREN)
    halt = bool(armed and e_rate > l_rate and e_te >= _DIR_SANITY_MIN_TOO_EASY)
    return {
        "escalate_too_easy": e_te, "escalate_total": e_tot, "escalate_rate": round(e_rate, 3),
        "lateral_too_easy": l_te, "lateral_total": l_tot, "lateral_rate": round(l_rate, 3),
        "armed": armed, "halt": halt,
    }


def _cosine(a, b) -> float:
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Shared stateless services
# ---------------------------------------------------------------------------

@dataclass
class _Services:
    fm: FM
    fm_judge: FM
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
    """Generate → gates → K-loop for one scenario.

    Returns (terminal_state, scenario_or_None, info, anchor_idx).
    terminal_state ∈ {too_easy, frontier, beyond_frontier, discarded, generation_failed}.
    Completed scenarios are written to bank/generated/ here (crash-safe, unique filenames);
    archive/posterior updates happen only in the sequential main-loop step.
    """
    fm = svc.fm
    anchor = archive.state.tasks[anchor_idx]
    tag = f"[iter {global_iter:04d}]"
    print_step(f"{tag} Generating from anchor: {anchor.scenario[:70]}...")

    # --- KNN examples around anchor (context for generation) ---
    # Exclude same-root lineage members: surface-identical siblings/ancestors are
    # useless prompt context (same premise, different hidden keys). They remain
    # selectable anchors — just not shown as structural exemplars.
    n_examples = int(config.task_generator.num_examples)
    anchor_root = anchor.root_seed_env_pk or anchor.source_env_id or anchor.id
    all_tasks = archive.state.tasks
    knn_pool_idxs = [
        i for i, s in enumerate(all_tasks)
        if (s.root_seed_env_pk or s.source_env_id or s.id) != anchor_root
        or i == anchor_idx  # always keep the anchor itself
    ]
    pool_embs = [all_tasks[i].embedding for i in knn_pool_idxs]
    all_embs = archive.get_successful_embeddings()
    if anchor.embedding and pool_embs and len(knn_pool_idxs) >= n_examples:
        src_ids = [all_tasks[i].source_scenario_id for i in knn_pool_idxs]
        agt_idxs = [all_tasks[i].target_agent_idx for i in knn_pool_idxs]
        rel_idxs = get_similar_scenarios(
            anchor.embedding, pool_embs, num_returns=n_examples,
            source_ids=src_ids, agent_idxs=agt_idxs,
            preferred_agent_idx=anchor.target_agent_idx,
        )
        ex_idxs = [knn_pool_idxs[r] for r in rel_idxs]
        if anchor_idx not in ex_idxs:
            ex_idxs = [anchor_idx] + ex_idxs[:n_examples - 1]
    else:
        ex_idxs = [anchor_idx]
    examples = [all_tasks[i] for i in ex_idxs]

    # --- Dead-end negatives: beyond_frontier scenarios from the archive (KNN-nearest) ---
    # (Sourced from tasks filtered by classification — the runner never populates failed_tasks.)
    n_ep_failed = int(config.task_generator.get("num_episode_failed_examples", 2))
    beyond = [s for s in archive.state.tasks if s.classification == "beyond_frontier"]
    episode_failed: list[SocialScenario] = []
    if beyond and n_ep_failed > 0:
        if anchor.embedding and any(s.embedding for s in beyond):
            neg_idxs = get_similar_scenarios(
                anchor.embedding, [s.embedding for s in beyond], num_returns=n_ep_failed,
                source_ids=[s.source_scenario_id for s in beyond],
                agent_idxs=[s.target_agent_idx for s in beyond],
            )
            episode_failed = [beyond[i] for i in neg_idxs]
        else:
            episode_failed = beyond[-n_ep_failed:]

    existing_types = (
        list({s.interaction_type for s in archive.state.tasks if s.interaction_type})
        if config.task_generator.get("show_existing_types", True) else None
    )

    # --- Mutation operator from anchor classification ---
    classification = getattr(anchor, "classification", None)
    if classification == "too_easy":
        mutation_op = "escalate"
    elif classification == "beyond_frontier":
        mutation_op = "relax"
    else:
        mutation_op = "lateral"

    # --- Generate batch ---
    candidates = svc.task_gen.generate_batch_from_archive(
        examples, anchor=anchor, mutation_operator=mutation_op,
        episode_failed_examples=episode_failed, existing_types=existing_types or [],
        batch_size=int(config.get("gen_batch_size", 3)),
    )
    if not candidates:
        return "generation_failed", None, {"reason": "generation_returned_none"}, anchor_idx

    # --- Free key-leak filter, then MOI ranking ---
    pre = len(candidates)
    candidates = [c for c in candidates if not fuzzy_key_leak_check(c)]
    if len(candidates) < pre:
        print_warn(f"{tag} dropped {pre - len(candidates)} key-leaking candidate(s)")
    if not candidates:
        return "generation_failed", None, {"reason": "all_candidates_leaked_key"}, anchor_idx
    if config.enable_moi and len(candidates) > 1:
        candidates = svc.moi.rank_batch(candidates)

    # --- Show the generated candidate batch (MOI-ranked, best-first) ---
    print_step(f"{tag} {len(candidates)} candidate(s) generated (op={mutation_op}), MOI-ranked best-first:")
    for r, c in enumerate(candidates):
        names = " & ".join(p.first_name for p in (c.agent_profiles or []) if p.first_name)
        km = c.partner_key.key_mechanism if c.partner_key else "—"
        print_info(f"{tag}   [{r}] {names or '?'} | mech={km} | slots={c.mutated_slots or []}")
        print_info(f"{tag}       {c.scenario[:160]}")

    # --- Walk the MOI-ranked list through embed + UNIVERSAL admission gates (Patch 10) ---
    # Single-axis ownership: the embedding diversity gate owns surface novelty for EVERY child,
    # every operator. Because completed children are archived, it covers siblings/parents/all.
    # surface_novelty_check is a free deterministic pre-check (no anchor name reuse, no clone).
    # There is no operator-conditional branch: operators set difficulty DIRECTION; whether a child
    # landed where directed is LP's verdict, not a generation-time gate.
    emb_threshold = float(config.get("diversity_similarity_threshold", 0.92))
    scenario = None
    admitted_rank = -1
    gate_fail_log: list[dict] = []
    for rank, cand in enumerate(candidates):
        cand.iteration = global_iter
        cand.parent_example_ids = [anchor.id]
        try:
            cand.embedding = fm.get_embeddings([cand.to_text_for_embedding()])[0]
        except Exception as e:
            print_warn(f"{tag} embed failed (rank {rank}): {e}")
            gate_fail_log.append({"rank": rank, "gate": "embed", "issues": [str(e)]})
            continue
        cand, ok, coherence_issues = run_coherence_gate(cand, svc.coherence_checker, svc.task_gen, fm, config, anchor, global_iter)
        if not ok or cand is None:
            gate_fail_log.append({"rank": rank, "gate": "coherence", "issues": coherence_issues})
            break  # best candidate failed coherence — lower-ranked candidates share the same anchor/structural issues
        # Deterministic surface-novelty pre-check (free): reused names / clone text.
        novelty_issues = surface_novelty_check(cand, anchor)
        if novelty_issues:
            print_warn(f"{tag} surface-novelty FAIL (rank {rank}): {novelty_issues[0]}")
            gate_fail_log.append({"rank": rank, "gate": "surface_novelty", "issues": novelty_issues})
            continue
        # Embedding diversity gate vs the whole archive (universal — every operator).
        if config.get("enable_diversity_gate", True) and all_embs and cand.embedding:
            emb_arr = np.array(all_embs); s_emb = np.array(cand.embedding)
            sims = emb_arr @ s_emb / (np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(s_emb) + 1e-9)
            max_sim = float(sims.max())
            if max_sim > emb_threshold:
                print_warn(f"{tag} diversity gate FAIL (rank {rank}, max_sim={max_sim:.3f})")
                gate_fail_log.append({"rank": rank, "gate": "diversity", "max_sim": max_sim})
                continue
        # mutated_slots is descriptive self-report (not a contract) — warn only, never reject.
        if not (cand.mutated_slots or []):
            print_warn(f"{tag} note: candidate (rank {rank}) reported empty mutated_slots")
        scenario = cand; admitted_rank = rank
        break
    if scenario is None:
        print_warn(f"{tag} ALL candidates failed gates — no_candidate_passed_gates. Gate log: {gate_fail_log}")
        return "generation_failed", None, {
            "reason": "no_candidate_passed_gates",
            "gate_fail_log": gate_fail_log,
        }, anchor_idx
    # Re-stamp the operator: the coherence-fix path (edit_scenario) rebuilds the scenario
    # object and drops mutation_operator, which would break the per-operator diagnostics.
    scenario.mutation_operator = mutation_op
    _sel_names = " & ".join(p.first_name for p in (scenario.agent_profiles or []) if p.first_name)
    print_info(f"{tag} ✓ SELECTED candidate [{admitted_rank}] ({mutation_op}) — {_sel_names or '?'}: {scenario.scenario[:120]}")

    # --- Lineage (pointers; full ancestors live in the archive by id) ---
    scenario.parent_id = anchor.id
    scenario.parent_is_sotopia_seed = (anchor.source == "seed_sotopia")
    scenario.parent_classification = anchor.classification   # what band the anchor was when selected
    scenario.parent_scenario = anchor.scenario               # parent text, for quick lineage eyeballing
    scenario.root_seed_env_pk = anchor.root_seed_env_pk or anchor.source_env_id or None
    scenario.lineage_depth = (anchor.lineage_depth or 0) + 1
    scenario.ancestor_ids = list(anchor.ancestor_ids or []) + [anchor.id]

    # --- Target agent designation ---
    scenario.target_agent_idx, scenario.target_agent_goal_abstract = designate_target_agent(
        scenario, anchor, fm
    )
    # Resolve the _pX placeholder now that we know the real perspective index.
    if scenario.id.endswith("_pX"):
        scenario.id = scenario.id[:-2] + f"p{scenario.target_agent_idx}"

    # --- Incremental live-write setup ---
    # Write a stub record immediately so bank/generated/<id>.json exists as soon as
    # the scenario is admitted. on_turn rewrites it turn-by-turn with the live transcript;
    # on_attempt_done flushes each completed attempt. Final write_scenario_record below
    # marks the record completed.
    generated_dir = run_dir / "bank" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    live_path = generated_dir / f"{scenario.id}.json"
    _live_loop_info: dict = {"skill_attempts": []}
    write_live_record(live_path, scenario, _live_loop_info, live_transcript=[])

    def _on_turn(partial: list[dict]) -> None:
        # partial is raw Sotopia format (Environment msgs, 'did nothing', '[private to]' prefixes).
        # Clean it the same way completed-attempt transcripts are cleaned before persisting.
        from social_omni_epic.episode_runner import clean_transcript
        write_live_record(live_path, scenario, _live_loop_info, live_transcript=clean_transcript(partial))

    def _on_attempt_done(loop_info: dict) -> None:
        nonlocal _live_loop_info
        _live_loop_info = loop_info
        skill = loop_info.get("skill_attempts", [])
        latest = skill[-1] if skill else None
        if latest:
            goal = (latest.get("diagnostics_scores") or {}).get("goal", "?")
            status = "SOLVED ✓" if latest.get("solved") else "FAILED ✗"
            print_info(f"{tag}   attempt {latest['attempt']}: {status}  GOAL={goal}")
        # Flush completed attempt to disk; clear live_transcript (attempt boundary).
        write_live_record(live_path, scenario, loop_info, live_transcript=None)

    try:
        scenario, terminal_state, _outcome, final_scores, loop_info = await run_episode_two_loop(
            scenario=scenario, anchor=anchor, task_gen=svc.task_gen,
            reflection_mod=svc.reflection_mod, meta_mod=svc.meta_mod, adversarial=svc.adversarial,
            title_gen=svc.title_gen, coherence_checker=svc.coherence_checker,
            run_single_episode=svc.run_single_episode,
            scenario_to_sotopia_profiles=svc.scenario_to_sotopia_profiles,
            fm=fm, config=config, on_attempt_done=_on_attempt_done, on_turn=_on_turn,
            fm_judge=svc.fm_judge,
        )
    except Exception as e:
        import traceback
        print_warn(f"{tag} Episode exception: {e}\n{traceback.format_exc()}")
        return "discarded", scenario, {"reason": f"episode_exception: {e}"}, anchor_idx

    loop_info["final_scores"] = final_scores
    loop_info["admitted_moi_rank"] = admitted_rank

    # Episode/LP quarantine routed up from the K-loop.
    if terminal_state == "discarded":
        print_warn(f"{tag} ⚠ QUARANTINE (episode/LP error) → quarantine/")
        return "discarded", scenario, loop_info, anchor_idx

    # Completed (too_easy / frontier / beyond_frontier): overwrite live stub with final record.
    write_scenario_record(scenario, loop_info, run_dir / "bank" / "generated")
    write_chronicle(scenario, run_dir / "chronicles")
    lp_str = f"LP={loop_info.get('lp_value', 0.0):.2f}"
    g = final_scores.get("goal", 0.0)
    print_info(f"{tag} → {terminal_state}  GOAL={g:.1f}  {lp_str}  "
               f"title={scenario.scenario_title or scenario.scenario[:40]}")
    return terminal_state, scenario, loop_info, anchor_idx


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

    fm = FM(model=config.model, temperature=config.temperature)
    fm_judge = FM(model=config.judge.model, temperature=float(config.judge.get("lp_temperature", 0.3)))
    svc = _Services(
        fm=fm, fm_judge=fm_judge,
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
