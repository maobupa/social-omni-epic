"""One scenario-generation cell: generate → gates → K-loop, with its context injected.

Extracted verbatim from `scripts/run_curriculum.py::_run_one_scenario` so two drivers can share it:

  * `run_curriculum.py` — the open-ended evolutionary loop. Builds its context from the growing
    archive via `context_from_archive()`, which is the original header code unchanged.
  * `run_grid_generate.py` — the paired grid. Builds the context from the 90 phase-0-banded seeds,
    one cell per seed, so no cell's inputs depend on another cell's outputs.

The split point matters: `_run_one_scenario` read the archive in five places, all in its header
(exemplar KNN, dead-end negatives, existing_types, the diversity-comparison embeddings, and the
anchor itself) and wrote to it in none. `GenerationContext` makes those five reads explicit
parameters, which is what lets the grid guarantee cell independence — every prompt input and gate
comparison is auditable by reading one frozen object.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .coherence_check import CoherenceChecker, fuzzy_key_leak_check
from .curriculum import run_coherence_gate, run_episode_two_loop
from .data_models import SocialScenario
from .embedding_utils import get_similar_scenarios
from .expel_export import write_chronicle, write_live_record, write_scenario_record
from .fm import FM
from .scenario_title import designate_target_agent
from .tracing_fm import print_info, print_step, print_warn
from .validation import surface_novelty_check


# ---------------------------------------------------------------------------
# Shared stateless services
# ---------------------------------------------------------------------------

@dataclass
class Services:
    """The long-lived collaborators one driver process shares across cells.

    Three separate generation-side FMs, because they must be able to differ (see the matrix plan):
      fm_generator  — scenario generation, embeddings, titles, coherence patching
      fm_reflection — writes the Reflexion string the learner reads on attempts 2..K.
                      MUST be the learner's own model: recoverability means "did the learner
                      recover after being told what went wrong", so a stronger writer measures the
                      teacher instead, and does so unequally across learners.
      fm_gates      — coherence checker + MOI ranker, i.e. whatever grades the generator's output.
      fm_judge      — cross-lab Sotopia-Eval / key-check / LP judge.
    """
    fm_generator: FM
    fm_judge: FM
    task_gen: object
    moi: object
    coherence_checker: CoherenceChecker
    title_gen: object
    reflection_mod: object
    meta_mod: object
    adversarial: object
    run_single_episode: object
    scenario_to_sotopia_profiles: object
    fm_reflection: Optional[FM] = None   # None → falls back to fm_generator (legacy behaviour)
    fm_gates: Optional[FM] = None        # None → the checker/MOI keep whatever FM they were built with

    @property
    def fm(self) -> FM:
        """Back-compat alias. The generation-side FM was called `fm` when there was only one."""
        return self.fm_generator


# ---------------------------------------------------------------------------
# Injected per-cell context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationContext:
    """Everything the generation prompt and the admission gates are allowed to see.

    Frozen and explicit so that cell independence is verifiable by inspection rather than by
    tracing archive state. In the grid, every field is a function of (seed, learner, phase-0 bands,
    fixed corpora) only.
    """
    anchor: SocialScenario                            # the parent; a seed in the grid
    mutation_operator: str                            # escalate | lateral | relax
    exemplars: list                                   # structural examples for the prompt
    episode_failed_examples: list = field(default_factory=list)   # dead-end negatives
    existing_types: Optional[list] = None             # interaction types already present
    diversity_embeddings: list = field(default_factory=list)      # what the diversity gate compares against
    child_id: Optional[str] = None                    # deterministic id stem (grid); None → uuid
    rng: Optional[object] = None                      # per-cell np.random.Generator (grid)


def context_from_archive(archive, anchor_idx: int, config) -> GenerationContext:
    """Build a context the way the evolutionary loop always has.

    This is `run_curriculum.py:201-258` unchanged — archive-wide exemplar KNN, archive-wide
    dead-end negatives, archive-wide existing_types, archive-wide diversity embeddings, and the
    mutation operator inferred from the anchor's band.
    """
    anchor = archive.state.tasks[anchor_idx]

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

    return GenerationContext(
        anchor=anchor,
        mutation_operator=operator_for_band(getattr(anchor, "classification", None)),
        exemplars=examples,
        episode_failed_examples=episode_failed,
        existing_types=existing_types,
        diversity_embeddings=all_embs,
    )


def operator_for_band(classification: Optional[str]) -> str:
    """Band → difficulty direction. In the grid this mapping IS the calibration mechanism:
    the same seed yields a harder child for a strong learner and an easier one for a weak one."""
    if classification == "too_easy":
        return "escalate"
    if classification == "beyond_frontier":
        return "relax"
    return "lateral"


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------

async def run_generation_cell(
    ctx: GenerationContext,
    svc: Services,
    config,
    iteration: int,
    run_dir: Path,
    tag: Optional[str] = None,
) -> tuple[str, Optional[SocialScenario], dict]:
    """Generate → gates → K-loop for one cell.

    Returns (terminal_state, scenario_or_None, info).
    terminal_state ∈ {too_easy, frontier, beyond_frontier, discarded, generation_failed}.
    Completed scenarios are written to <run_dir>/bank/generated/ here (crash-safe, unique
    filenames); any archive/posterior bookkeeping is the caller's job.
    """
    fm = svc.fm_generator
    anchor = ctx.anchor
    mutation_op = ctx.mutation_operator
    tag = tag if tag is not None else f"[iter {iteration:04d}]"
    print_step(f"{tag} Generating from anchor: {anchor.scenario[:70]}...")

    # --- Generate batch ---
    candidates = svc.task_gen.generate_batch_from_archive(
        ctx.exemplars, anchor=anchor, mutation_operator=mutation_op,
        episode_failed_examples=ctx.episode_failed_examples,
        existing_types=ctx.existing_types or [],
        batch_size=int(config.get("gen_batch_size", 3)),
    )
    if not candidates:
        return "generation_failed", None, {"reason": "generation_returned_none"}

    # --- Free key-leak filter, then MOI ranking ---
    pre = len(candidates)
    candidates = [c for c in candidates if not fuzzy_key_leak_check(c)]
    if len(candidates) < pre:
        print_warn(f"{tag} dropped {pre - len(candidates)} key-leaking candidate(s)")
    if not candidates:
        return "generation_failed", None, {"reason": "all_candidates_leaked_key"}
    if config.get("enable_moi", True) and len(candidates) > 1:
        candidates = svc.moi.rank_batch(candidates)

    # --- Show the generated candidate batch (MOI-ranked, best-first) ---
    print_step(f"{tag} {len(candidates)} candidate(s) generated (op={mutation_op}), MOI-ranked best-first:")
    for r, c in enumerate(candidates):
        names = " & ".join(p.first_name for p in (c.agent_profiles or []) if p.first_name)
        km = c.partner_key.key_mechanism if c.partner_key else "—"
        print_info(f"{tag}   [{r}] {names or '?'} | mech={km}")
        print_info(f"{tag}       {c.scenario[:160]}")

    # --- Walk the MOI-ranked list through embed + UNIVERSAL admission gates (Patch 10) ---
    # Single-axis ownership: the embedding diversity gate owns surface novelty for EVERY child,
    # every operator. surface_novelty_check is a free deterministic pre-check (no anchor name
    # reuse, no clone). There is no operator-conditional branch: operators set difficulty
    # DIRECTION; whether a child landed where directed is LP's verdict, not a generation-time gate.
    emb_threshold = float(config.get("diversity_similarity_threshold", 0.92))
    all_embs = ctx.diversity_embeddings
    scenario = None
    admitted_rank = -1
    gate_fail_log: list[dict] = []
    for rank, cand in enumerate(candidates):
        cand.iteration = iteration
        cand.parent_example_ids = [anchor.id]
        # Deterministic ids (grid): makes resume-by-existence exact and set-to-set pairing a
        # filename join. The _pX suffix is resolved below, exactly as for uuid-based ids.
        if ctx.child_id:
            cand.source_scenario_id = ctx.child_id
            cand.id = f"{ctx.child_id}_pX"
        try:
            cand.embedding = fm.get_embeddings([cand.to_text_for_embedding()])[0]
        except Exception as e:
            print_warn(f"{tag} embed failed (rank {rank}): {e}")
            gate_fail_log.append({"rank": rank, "gate": "embed", "issues": [str(e)]})
            continue
        cand, ok, coherence_issues = run_coherence_gate(
            cand, svc.coherence_checker, svc.task_gen, fm, config, anchor, iteration
        )
        if not ok or cand is None:
            gate_fail_log.append({"rank": rank, "gate": "coherence", "issues": coherence_issues})
            break  # best candidate failed coherence — lower-ranked candidates share the same anchor/structural issues
        # Deterministic surface-novelty pre-check (free): reused names / clone text.
        novelty_issues = surface_novelty_check(cand, anchor)
        if novelty_issues:
            print_warn(f"{tag} surface-novelty FAIL (rank {rank}): {novelty_issues[0]}")
            gate_fail_log.append({"rank": rank, "gate": "surface_novelty", "issues": novelty_issues})
            continue
        # Embedding diversity gate vs the injected comparison set (universal — every operator).
        if config.get("enable_diversity_gate", True) and all_embs and cand.embedding:
            emb_arr = np.array(all_embs); s_emb = np.array(cand.embedding)
            sims = emb_arr @ s_emb / (np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(s_emb) + 1e-9)
            max_sim = float(sims.max())
            if max_sim > emb_threshold:
                print_warn(f"{tag} diversity gate FAIL (rank {rank}, max_sim={max_sim:.3f})")
                gate_fail_log.append({"rank": rank, "gate": "diversity", "max_sim": max_sim})
                continue
        scenario = cand; admitted_rank = rank
        break
    if scenario is None:
        print_warn(f"{tag} ALL candidates failed gates — no_candidate_passed_gates. Gate log: {gate_fail_log}")
        return "generation_failed", None, {
            "reason": "no_candidate_passed_gates",
            "gate_fail_log": gate_fail_log,
        }
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
        from .episode_runner import clean_transcript
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
            fm_judge=svc.fm_judge, fm_reflection=svc.fm_reflection,
        )
    except Exception as e:
        import traceback
        print_warn(f"{tag} Episode exception: {e}\n{traceback.format_exc()}")
        return "discarded", scenario, {"reason": f"episode_exception: {e}"}

    loop_info["final_scores"] = final_scores
    loop_info["admitted_moi_rank"] = admitted_rank

    # Episode/LP quarantine routed up from the K-loop.
    if terminal_state == "discarded":
        print_warn(f"{tag} ⚠ QUARANTINE (episode/LP error) → quarantine/")
        return "discarded", scenario, loop_info

    # Completed (too_easy / frontier / beyond_frontier): overwrite live stub with final record.
    write_scenario_record(scenario, loop_info, run_dir / "bank" / "generated")
    write_chronicle(scenario, run_dir / "chronicles")
    lp_str = f"LP={loop_info.get('lp_value', 0.0):.2f}"
    g = final_scores.get("goal", 0.0)
    print_info(f"{tag} → {terminal_state}  GOAL={g:.1f}  {lp_str}  "
               f"title={scenario.scenario_title or scenario.scenario[:40]}")
    return terminal_state, scenario, loop_info


# ---------------------------------------------------------------------------
# Output helpers (driver-agnostic)
# ---------------------------------------------------------------------------

def count_generated(run_dir: Path) -> int:
    """Count completed generated scenarios — excludes in-progress stubs and .tmp files."""
    d = run_dir / "bank" / "generated"
    if not d.exists():
        return 0
    count = 0
    for p in d.glob("*.json"):
        try:
            rec = json.loads(p.read_text())
            if rec.get("status") != "in_progress":
                count += 1
        except Exception:
            pass
    return count


def write_quarantine(run_dir: Path, key, reason: str, info: dict) -> None:
    """Key is an iteration index (evolutionary loop) or a cell key string (grid)."""
    d = run_dir / "quarantine"
    d.mkdir(parents=True, exist_ok=True)
    name = f"iter_{key:06d}" if isinstance(key, int) else f"cell_{key}"
    (d / f"{name}.json").write_text(
        json.dumps({"iteration": key, "reason": reason, **info}, indent=2, default=str)
    )


def write_lineage(run_dir: Path, scenarios) -> None:
    lineage = {
        s.id: {
            "parent_id": s.parent_id,
            "root_seed_env_pk": s.root_seed_env_pk,
            "lineage_depth": s.lineage_depth,
            "mutation_operator": s.mutation_operator,
            "classification": s.classification,
            "source": s.source,
        }
        for s in scenarios
    }
    (run_dir / "lineage.json").write_text(json.dumps(lineage, indent=2))


def write_compute_report(run_dir: Path, fms: dict) -> None:
    (run_dir / "compute_report.json").write_text(
        json.dumps({name: fm.usage_report() for name, fm in fms.items()}, indent=2)
    )


def cosine(a, b) -> float:
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# Direction-sanity registered stopping rule (Patch 10, §III.C). Pre-declared so it is a legitimate
# stopping rule under the freeze, not a post-hoc edit. Arms only with enough children per operator;
# halts only on the gross-failure signature (escalate strictly more too_easy than lateral, with a
# floor of too_easy escalate children) — strict ">" so a tie/zero (the best case) never halts.
DIR_SANITY_MIN_CHILDREN = 8   # per operator before the rule arms
DIR_SANITY_MIN_TOO_EASY = 3   # escalate too_easy children before a halt can fire


def direction_sanity(run_dir: Path) -> dict:
    """Read the just-flushed summary.json and compare escalate vs lateral too_easy-rates.

    Returns {escalate_*, lateral_*, armed, halt} for logging + the stopping decision.
    In the grid the operator mix is fixed up front by phase-0, so call this once post-hoc
    rather than inside a loop.
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
    armed = (e_tot >= DIR_SANITY_MIN_CHILDREN and l_tot >= DIR_SANITY_MIN_CHILDREN)
    halt = bool(armed and e_rate > l_rate and e_te >= DIR_SANITY_MIN_TOO_EASY)
    return {
        "escalate_too_easy": e_te, "escalate_total": e_tot, "escalate_rate": round(e_rate, 3),
        "lateral_too_easy": l_te, "lateral_total": l_tot, "lateral_rate": round(l_rate, 3),
        "armed": armed, "halt": halt,
    }


# ---------------------------------------------------------------------------
# Per-learner seed bands (the calibration input)
# ---------------------------------------------------------------------------

def load_phase0_annotated_seeds(
    phase0_dir: Path,
    seeds_path: str,
    fm: FM,
    limit: Optional[int] = None,
) -> tuple[list, dict]:
    """Load the SOTOPIA seeds carrying their phase-0 band FOR ONE LEARNER.

    This is the calibration input the grid runs on: the band decides the mutation operator
    (too_easy -> escalate, frontier -> lateral, beyond_frontier -> relax), so the *same* seed
    yields a harder child for a strong learner and an easier one for a weak one. That mapping is
    the entire mechanism by which a scenario set becomes learner-relative.

    Extracted from run_curriculum.py::_seed_archive_from_phase0 minus the archive mutation and the
    Beta-prior assignment — the grid has no posterior, so priors are meaningless there.

    Returns (seeds, {env_pk: band}). Seeds missing a band are returned with classification=None;
    the caller decides whether that is fatal (for the grid it is: no band means no operator).
    """
    from .seeds import load_sotopia_seeds_with_embeddings

    seeds_subdir = phase0_dir / "seeds"
    if not seeds_subdir.exists():
        raise FileNotFoundError(f"phase-0 seeds dir not found: {seeds_subdir}")

    seeds = load_sotopia_seeds_with_embeddings(
        fm=fm, seeds_path=seeds_path, limit=limit, both_perspectives=False,
    )
    by_env = {s.source_env_id: s for s in seeds}

    bands: dict = {}
    for p in sorted(seeds_subdir.glob("seed_*.json")):
        rec = json.loads(p.read_text())
        env_pk = rec.get("env_pk")
        scn = by_env.get(env_pk)
        if scn is None:
            print_warn(f"  phase-0 record {p.name} (env_pk={env_pk}) has no raw-seed match — skipped")
            continue
        scn.classification = rec.get("classification")
        scn.lp_value = rec.get("lp_value")
        scn.lp_votes = int(rec.get("lp_votes") or 0)
        scn.terminal_success = bool(rec.get("terminal_success"))
        scn.n_attempts = int(rec.get("n_attempts") or 0)
        scn.scenario_title = rec.get("scenario_title") or scn.scenario_title
        scn.social_dynamic = rec.get("social_dynamic") or scn.social_dynamic
        scn.target_perspective = rec.get("target_perspective") or scn.target_perspective
        scn.root_seed_env_pk = scn.source_env_id
        scn.lineage_depth = 0
        scn.parent_id = None
        bands[env_pk] = scn.classification

    return seeds, bands


def build_grid_context(
    seed,
    all_seeds: list,
    bands: dict,
    n_examples: int = 3,
    n_failed_examples: int = 2,
    show_existing_types: bool = True,
    child_id: Optional[str] = None,
    rng=None,
) -> GenerationContext:
    """Context for one grid cell. Every field is a function of (this seed, this learner's bands,
    the fixed 90-seed corpus) — never of another cell's child. That is the invariant the paired
    design rests on: set A and set B must be comparable, so no cell may see another's output.

    Two deliberate differences from the evolutionary context:

      * EXEMPLARS ARE A LABELLED MIXTURE, not one band. The nearest too_easy + nearest frontier +
        nearest beyond_frontier seed for this learner. The steering signal is the CONTRAST ("this
        was too easy / just right / too hard"); filtering to a single band throws it away. Thin
        bands degrade gracefully — gpt-5-mini has only 5 beyond_frontier seeds, and a strong
        learner may have zero too_easy.
      * DIVERSITY COMPARES AGAINST THE 90 SEEDS ONLY (tier 1), never against sibling children.
        Gating on siblings would make admission order-dependent AND asymmetric across sets: a
        stronger learner draws more relax/lateral children, so it would hit a peer gate at a
        different rate, silently giving the two arms of a paired design different admission
        pressure. Within-set concentration is instead REPORTED post-hoc (grid_diversity.json).
    """
    seed_root = seed.source_env_id or seed.id

    # Exemplars: nearest seed per band, excluding this seed's own lineage.
    by_band: dict = {"too_easy": [], "frontier": [], "beyond_frontier": []}
    for s in all_seeds:
        if (s.source_env_id or s.id) == seed_root:
            continue
        b = bands.get(s.source_env_id)
        if b in by_band:
            by_band[b].append(s)

    exemplars = []
    for band in ("too_easy", "frontier", "beyond_frontier"):
        pool = by_band[band]
        if not pool:
            continue
        if seed.embedding and any(s.embedding for s in pool):
            idx = get_similar_scenarios(
                seed.embedding, [s.embedding for s in pool], num_returns=1,
                source_ids=[s.source_scenario_id for s in pool],
                agent_idxs=[s.target_agent_idx for s in pool],
            )
            exemplars.append(pool[idx[0]])
        else:
            exemplars.append(pool[0])
    exemplars = exemplars[:max(1, n_examples)] or [seed]

    # Dead-end negatives = seeds that are beyond_frontier FOR THIS LEARNER. Per-learner and
    # seed-derived, so still cell-independent. Note the semantic shift from the evolutionary
    # loop: these are "impossible for M", not "generation dead-ends".
    beyond = [s for s in all_seeds
              if bands.get(s.source_env_id) == "beyond_frontier"
              and (s.source_env_id or s.id) != seed_root]
    episode_failed = beyond[:n_failed_examples] if n_failed_examples > 0 else []

    existing_types = (
        sorted({s.interaction_type for s in all_seeds if s.interaction_type})
        if show_existing_types else None
    )

    return GenerationContext(
        anchor=seed,
        mutation_operator=operator_for_band(bands.get(seed.source_env_id)),
        exemplars=exemplars,
        episode_failed_examples=episode_failed,
        existing_types=existing_types,
        diversity_embeddings=[s.embedding for s in all_seeds if s.embedding],
        child_id=child_id,
        rng=rng,
    )
