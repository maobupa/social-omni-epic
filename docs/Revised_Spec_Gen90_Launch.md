# Social OMNI-EPIC — Final Revision Spec before the Gen-90 Run

> **Audience:** Claude Code. This supersedes and extends the approved plan ("ExpeL-driven Gen-90 curriculum on top of the Base90 ExpeL seed bank"). All four plan changes are RATIFIED with modifications noted inline. Items are tagged **P0** (launch blocker), **P1** (correctness, do before launch), **P2** (logging/diagnostics, do before launch if cheap, else first thing after).
>
> **Prime directive unchanged:** redesign only what is specified; preserve everything else. DELETE means delete the code path. Out of scope: the Random90 baseline (separate workstream, do not implement), niching in the run loop (post-hoc only, §10), any chronicle-path changes beyond bug fixes (the chronicle stays importable behind `use_expel_memory: false`).

---

## 0. Decision log (context for every change below)

1. **ExpeL memory in the K-loop** is final for the workshop run (`use_expel_memory: true`). No parent→child memory inheritance: attempt 1 is always cold. LP is therefore a stationary property of (scenario × base learner × reflexion mechanism).
2. **The bank is the contribution.** Headline comparison: ExpeL-Generated90 vs ExpeL-Base90 vs Vanilla (Random90 arrives from the other workstream). Extraction input for Generated90 = trajectories of **all** completed generated scenarios regardless of classification (protocol parity with Base90, which ran all 90 seeds).
3. **Stopping condition** = 90 *completed generated scenarios total* (too_easy + frontier + beyond_frontier). NOT 90 solved.
4. **Seeding** = the 90 SOTOPIA seeds annotated with Base90/phase0 ExpeL-run results: classification + LP carried as metadata; classification drives the first operator; LP enters Thompson **only** as a soft asymmetric prior (frontier → Beta(2,1); too_easy / beyond_frontier → Beta(1,1)); `alpha_votes = beta_votes = 0` for all seeds. Seed self-LP never becomes votes.
5. **Chronicle text out of the generation prompt** (all three call sites). Mutation targeting is carried by structured diagnoses instead: `too_easy_diagnosis` (exists) for escalate, **new `beyond_frontier_diagnosis`** (§5) for relax.
6. **Canonical run state** = `archive_latest.json`. Folders are exports/views. Resume = same `run_name`.
7. **Eval-time memory** = global extracted insights only, injected identically across ExpeL conditions. No trajectory retrieval, no per-entry retrieval. (Future-work ablation, not now.)

---

## 1. [P0] Stopping condition & counters — `scripts/run_curriculum.py`

Current behavior stops on `solved_count` (count of `success/*.json` = frontier-solved only). **Wrong.**

- Replace `solved_count` as the stopping driver with `generated_count` = number of completed generated scenarios (terminal_state ∈ {too_easy, frontier, beyond_frontier}). `generation_failed` and quarantined scenarios do NOT count.
- Resume source of truth: `generated_count = len(list((run_dir / "bank" / "generated").glob("*.json")))` (§3 layout).
- The loop condition becomes: run until `generated_count >= stopping.N`, with `config.iterations` retained as a hard safety cap (document: launch with `iterations=250 stopping.N=90`).
- Keep `solved_count` as a logged metric only (rename internal references accordingly; it must no longer gate anything).

## 2. [P0] Persist too_easy scenarios fully — `scripts/run_curriculum.py`

Currently `_save_discarded` writes only `{iteration, reason, anchor_id}` for too_easy results; the scenario object and its attempt-1 trajectory are lost as files. This breaks both the bank-of-90 definition and ExpeL extraction parity.

- On `terminal_state == "too_easy"`: call `_save_scenario_file(scenario, generated_dir)` exactly like frontier/beyond_frontier, AND include its `loop_info` in the trajectory export (§4). The too_easy trajectory is a single *successful* attempt — ExpeL gather consumes it as a success example.
- Reserve `_save_discarded`-style stub records for genuine discards only (generation failures, quarantines), written to `quarantine/` (§3).

## 3. [P0] Output layout & expanding-bank semantics — `scripts/run_curriculum.py`

New layout (replaces `success/ failed/ discarded/`; keep a one-line README in the run dir documenting it):

```
results/<run_name>/
├── bank/
│   ├── seeds/<env_pk>.json        # the 90 phase0-annotated seeds (written once at seeding; export/view)
│   └── generated/<id>.json        # EVERY completed generated scenario (all classifications) — this IS Generated-N
├── quarantine/iter_<N>.json       # generation failures, gate quarantines, episode-error discards (stub + error info)
├── trajectories.json              # ExpeL-extract-ready (§4); flushed at every checkpoint
├── chronicles/<id>.md             # skills_final_md per generated scenario (reflexion strings)
├── lineage.json                   # {id: {parent_id, root_seed_env_pk, depth, operator, classification, source}}
├── summary.json                   # classification_counts, per-operator counts, lp_stats, models, n
├── archive_latest.json            # CANONICAL resume state (Thompson posteriors + votes live here)
├── archive_iter_<N>.json          # periodic snapshots
├── metrics.json                   # per-iteration log (§9)
└── compute_report.json            # FM call/token meter (§9)
```

- Classification views (which scenarios are frontier vs beyond etc.) are derived from the `classification` field inside each JSON — do not maintain parallel classification folders.
- **Resume contract:** re-running with the same `run_name` loads `archive_latest.json` (seeds + all prior generated children with their posteriors) and continues the same expanding bank and lineages. Folder contents are never read to reconstruct Thompson state.
- **Frozen-set rule (document in the README):** the workshop artifact "ExpeL-Generated90" is the first 90 completed generated scenarios under the logged config and seed. Extending the bank later must write new extraction artifacts under a new name, never mutate `insights.json` from the 90-run.

## 4. [P0] ExpeL export bridge — new `social_omni_epic/expel_export.py` (plan change 4, ratified with one modification)

Implement as planned (`build_expel_trajectories`, `write_scenario_record`, chronicles/, summary.json), with this modification:

- **Include too_easy scenarios** in `trajectories.json` (single successful trajectory, `reflections=[]`, `success=True`, `trial=0`). The plan's record set "success/failed/discarded except generation_failed" maps onto the new layout as: every file in `bank/generated/`.
- Success labeling for trajectories: use the per-attempt `solved` flag from `loop_info["skill_attempts"]` (this is terminal_success = GOAL≥7 ∧ REL≥0 ∧ key_check). Do not re-derive from `goal_score >= 7` alone — the key check is part of what success means on keyed scenarios. Document this asymmetry vs Base90 (seeds have no key) in summary.json as `success_label: "goal_rel_key"`.
- Flush `trajectories.json` (and summary/lineage) at every checkpoint, not only at run end, so resume and mid-run extraction are safe.
- Round-trip requirement: `trajectories_from_dict(json.load(open(...)))` must succeed (plan verification item 2 stands).

## 5. [P0] Mutation targeting without chronicle prose — `task_generator.py`, `curriculum.py`

### 5.1 Chronicle-off in generation (plan change 1, ratified)
- Flip `include_chronicle=True → False` at all three call sites (parent block, frontier exemplars, dead ends). Keep the parameter for revivability.
- Reword the three section headers in `_build_user_prompt` that promise a chronicle. New text must describe what is actually shown: scenario structure, classification, and structured diagnosis. E.g. frontier header: "FRONTIER EXEMPLARS — scenarios at the current difficulty boundary: the learner failed the first attempt, then improved. Target this difficulty level:". Dead-end header: "STRUCTURAL DEAD ENDS — never solved across all attempts. The stuck_knob diagnosis names the slot that made each unwinnable. Do NOT reproduce the same structural failure:".
- `skills_final_md` continues to be written by the K-loop (raw material for extraction); it simply never enters generation prompts.

### 5.2 NEW: `analyze_beyond_frontier()` — structured relax targeting
The relax operator currently depends on chronicle WARNINGs; with the chronicle gone it must not go blind. You already store everything needed.

- Add `TaskGenerator.analyze_beyond_frontier(scenario, key_checks: list[dict], attempt_scores: list[dict]) -> dict` returning `{"stuck_knob": ..., "rationale": ...}`.
- Implement as a **code heuristic first** (no LLM call), over the per-attempt key-check verdicts and GOAL/REL trajectories:
  - Triggers tripped (and never repaired) in ≥ half the attempts → `stuck_knob: "hardening_triggers_too_congruent"` → relax (c).
  - No movement condition ever in `conditions_met` AND few/no triggers tripped → `stuck_knob: "surface_misdirection_undiscoverable"` → relax (b).
  - ≥1 condition met on some attempt but never `terminal_success` (cost not paid / GOAL stayed low) → `stuck_knob: "cost_coupling_too_high"` → relax (d).
  - No key (shouldn't happen for generated) or verdicts missing → `stuck_knob: "unknown"`, fall back to the blind ordering.
- Optional single LLM fallback call (mirror `_ANALYZE_TOO_EASY_SYSTEM`) only when the heuristic returns "unknown"; never sees chronicle text or the learner's transcripts beyond the last attempt.
- Store on the scenario as `beyond_frontier_diagnosis: Optional[dict]` (add field to `SocialScenario`), set inside `run_episode_k_loop` when classification == beyond_frontier (call site mirrors the too_easy fast path's `analyze_too_easy`).
- Show it in the parent block the same way `too_easy_diagnosis` is shown.

### 5.3 Rewrite `_EDIT_INTENTS["relax"]`
Replace the chronicle-WARNING sentence with:
"The parent was NEVER SOLVED across all attempts (beyond_frontier). The beyond_frontier_diagnosis above names the stuck knob — loosen exactly that slot. If the diagnosis is missing or 'unknown', loosen (c) hardening_triggers first, then (d) cost_coupling. Preserve all other slots. The goal is a scenario that is hard but genuinely winnable by a skilled, non-capitulating actor."

## 6. [P0] Phase0 seeding & priors — `scripts/run_curriculum.py`, config (plan change 2, ratified with constraints)

Implement `_seed_archive_from_phase0` as planned, with these hard constraints:

- `both_perspectives=False` is **forced** in this path (and delete the `seed_both_perspectives` config flag plus its branch everywhere — decision is permanently false). 90 entries, `target_agent_idx=0`.
- Carry per-seed: `classification`, `lp_value`, `lp_votes` (metadata only), `terminal_success`, `n_attempts`, `scenario_title` / `social_dynamic` / `target_perspective`.
- Priors: `classification == "frontier"` → `(prior_alpha, prior_beta) = (2.0, 1.0)`; anything else (too_easy, beyond_frontier, missing) → `(1.0, 1.0)`. `alpha_votes = beta_votes = 0.0` for all seeds. Expose as `seed_prior:` config block exactly as the plan specifies.
- Lineage on seeds: `root_seed_env_pk = source_env_id`, `lineage_depth = 0`, `parent_id = None`.
- Write the `bank/seeds/` export immediately after seeding (one JSON per seed with the phase0 metadata attached).
- Log a classification/prior histogram at startup (plan verification item 1 stands).
- The runner refuses to start the phase0-seeded path if any seed lacks `classification` (unless `--allow-uncalibrated`).

## 7. [P0] Episode-error quarantine — `curriculum.py`, `run_curriculum.py`

Two failure paths currently corrupt classification and the posterior:

1. **Mid-loop episode exception.** In `run_episode_k_loop`, an exception on attempt k>1 does `break`, then falls through to LP with truncated transcripts. If only attempt 1 completed, `compute_lp` returns lp=0 → classification = beyond_frontier → the anchor is charged (0, K_VOTES_EQUIV) **for an infrastructure failure**.
   - Fix: track `episode_error: str | None` in `loop_info`. If the loop ended via exception AND `not solved` AND completed attempts < K, return terminal state `"discarded"` (quarantine), not a classification. The runner writes a stub to `quarantine/`, records **no** posterior votes, does **not** add the scenario to the archive, and does **not** count it toward `stopping.N`.
   - If the exception happened after solving (can't currently happen — solved breaks first) or on the final attempt with ≥2 clean transcripts, proceed normally (LP is computable).
2. **LP-error vote deflation.** `_cast_vote` converts judge exceptions into `no_difference` votes counted in the denominator. Since the frontier/beyond boundary is `lp == 0` *exactly*, errored votes can flip the classification.
   - Fix in `lp_judge.py`: tag error votes (`rationale` already carries `[judge error: ...]`; add `is_error: bool` to `VoteRecord`). In `compute_lp`, count `n_error_votes`. If `n_error_votes > 0` and the resulting `lp_value == 0.0`, re-run the errored votes once. If errors persist for **all** votes of **all** pairs → raise, and the caller (`run_episode_k_loop`) routes to the same quarantine path as (1). Store `n_error_votes` in `loop_info` and `metrics.json`.

## 8. [P0] Dead-end negatives bug — `scripts/run_curriculum.py`

`episode_failed` candidates are read from `archive.state.failed_tasks`, but nothing in this runner ever calls `add_failed_task` — the dead-end prompt section has been silently empty.

- Fix: source negatives from `archive.state.tasks` filtered by `classification == "beyond_frontier"` (keep the KNN-nearest selection logic; drop the `skills_final_md` filter — it was a chronicle-era proxy for "completed a loop", which `classification` now guarantees). Delete the `failed_tasks` dependence from this path (the list itself can stay for back-compat reads).

## 9. [P1] Correctness & hygiene fixes

1. **Embedding consistency** (`data_models.py`, `seeds.py`): remove the `scenario_title` branch from `to_text_for_embedding()` — full-text embeddings are title-free everywhere (generated scenarios are embedded pre-title anyway; seeds may or may not carry titles in the jsonl, making the space asymmetric). Delete the seed embedding cache files (`data/sotopia_90_seeds.embeddings.npy` + `.meta.json`) so seeds re-embed under the new text (90 embeddings, trivial cost). The abstract `social_dynamic | target_perspective` text is reserved for post-hoc niching/coverage (§10) and is never mixed into the full-text space.
2. **Child prior damping** (`archive.py`): in `child_prior_from_parent`, rescale the parent's posterior to a capped total mass before returning: `m = alpha + beta; if m > child_prior_mass: scale = child_prior_mass / m; return (alpha*scale, beta*scale)`. Config `child_prior_mass: 4.0`. Rationale: deep-lineage children otherwise inherit posterior mass that drowns their own first ~6 votes; capping preserves the structural-similarity mean while keeping children responsive to their own evidence.
3. **MOI candidate walk-through** (`run_curriculum.py`): (a) before MOI ranking, run the free `_fuzzy_key_leak_check` on **all** candidates and drop leakers (import it or expose a public wrapper in `coherence_check.py`); (b) instead of taking only `candidates[0]` into the coherence gate, walk the MOI-ranked list: if candidate i fails coherence after patch retries, try candidate i+1; `generation_failed` only when the list is exhausted. Log which rank was admitted.
4. **`_re` NameError** (`skills_chronicle.py::validate_synthesis`): `_re.IGNORECASE` → `re.IGNORECASE` (three occurrences). Dead under ExpeL but a latent crash on the chronicle path.
5. **Lineage fields** (plan change 3, ratified verbatim): add `parent_id`, `parent_is_sotopia_seed`, `root_seed_env_pk`, `lineage_depth`, `ancestor_ids` to `SocialScenario`; populate in `_run_one_scenario`; write `lineage.json` at every checkpoint.
6. **Within-batch near-dupes** (`run_curriculum.py`): the diversity gate snapshots `all_embs` before the concurrent batch, so two same-batch children are never checked against each other. Do NOT discard post-episode (compute already spent); at the sequential archive-update step, compute cosine of each newly added scenario against scenarios added earlier in the same batch and log a `within_batch_dupe` warning + flag in metrics when > 0.92. (Log-only; expected to be rare at batch_size ≤ 4.)
7. **Config deletions/assertions**: `random_seed` must be set (runner refuses null); assert `judge.model` provider ≠ learner model provider at startup (string-prefix check on model names is fine); `use_expel_memory: true`, `enable_moi: true`, `gen_batch_size: 3`, `max_attempts: 4`, `stopping.N: 90` in the launch config.

## 10. [P2] Logging, compute, post-hoc niching

1. **metrics.json per-iteration additions**: `mutation_operator`, `mutated_slots`, `parent_child_cosine` (embedding of child vs anchor — both exist at update time), `n_error_votes`, `n_attempts`, `admitted_moi_rank`. **summary.json**: classification counts, per-operator classification counts (the escalate→frontier conversion rate is a key diagnostic), `mean_lp`, `lp_by_lineage_depth`.
2. **FM compute meter** (`fm.py`): add `self.n_calls`, `self.prompt_tokens`, `self.completion_tokens` incremented in `_chat` from `r.usage` when present, plus a `component` tag settable via an optional kwarg threaded from call sites where cheap (at minimum: learner episodes vs judge vs generator, distinguishable because they are separate FM instances — per-instance totals suffice). Dump `compute_report.json` at every checkpoint. Needed for the paper's compute-accounting table (the curriculum condition's budget is a superset of Base90's and must be reported).
3. **Niching is post-hoc only.** Do not wire `NicheManager` into the runner. After the run, a standalone analysis script clusters the abstract embeddings (`social_dynamic | target_perspective`, generated fresh from archived titles) of seeds + bank for niche counts, coverage curves, and the UMAP figure. `niches.py` stays as-is as the offline tool; `cosine_dedup` in it is unused by the runner (the inline gate stands).
4. **MOI worth logging**: `rank_batch` currently discards worth scores; store the admitted candidate's worth + rationale into `scenario.moi_reasoning` and log the full ranking per iteration.
5. **K_VOTES_EQUIV import**: move the constant to `data_models.py` (or a new `constants.py`) and update the `archive.py` / `run_curriculum.py` imports — removes the archive→curriculum import edge. Mechanical.

## 11. Explicitly OUT OF SCOPE — do not implement
1. Random90 baseline (separate workstream; the eval harness must merely not preclude a fourth condition).
2. Trajectory/insight retrieval at eval time (insights-only, identical across ExpeL conditions).
3. Two-level / niche-aware Thompson sampling; in-run niching of any kind.
4. Chronicle inheritance of any flavor (including "abstracted ExpeL chronicle" inheritance — rejected: lineage transfer would want *concrete* reflexions, and either way it's an untested confound for this run).
5. Mechanism-library extension; multi-seed runs (single seeded run, limitation sentence in the draft, cite Henderson et al.).

## 12. Acceptance checklist (run before the 90-launch)
- [ ] Pilot (`iterations=4`, batch_size 2, phase0 seeding on): startup prints the seed classification/prior histogram; 90 seeds, all `target_agent_idx=0`, frontier seeds at Beta(2,1).
- [ ] First mutations match parent classification (too_easy→escalate, frontier→lateral, beyond→relax); generation user prompt contains **no** chronicle text (temporarily log it) and shows `too_easy_diagnosis` / `beyond_frontier_diagnosis` where applicable.
- [ ] A forced too_easy result produces: full JSON in `bank/generated/`, a trajectory entry in `trajectories.json` with `success=true`, anchor charged (0, K_VOTES_EQUIV), counted toward `stopping.N`.
- [ ] A forced mid-loop episode exception (mock) produces: `quarantine/` stub, no archive entry, no posterior change, not counted toward `stopping.N`.
- [ ] A forced LP-judge error with lp==0 triggers the one-shot revote; `n_error_votes` appears in metrics.
- [ ] `trajectories_from_dict` round-trips; `run_expel_baseline.py extract` writes a non-empty `insights.json` on the pilot.
- [ ] Lineage: child JSON carries `parent_id` / `root_seed_env_pk` / `lineage_depth ≥ 1`; `analysis/lineage_stats.py` reconstructs the tree; seeds are roots.
- [ ] Resume: kill the pilot mid-run, re-run same `run_name`, confirm `generated_count` resumes from `bank/generated/` and posteriors from the checkpoint.
- [ ] `grep -rn "seed_both_perspectives"` returns only this spec; dead-end section of a relax/lateral prompt is non-empty once a beyond_frontier scenario exists in `tasks`.
- [ ] Seed embedding cache deleted and regenerated; `to_text_for_embedding` has no title branch.
- [ ] `compute_report.json` non-empty after the pilot; per-instance totals distinguish learner/judge/generator.

## 13. Launch commands (after checklist passes)

```bash
# 1) Generate the bank: 90 completed scenarios, phase0-warm-started, ExpeL memory.
python scripts/run_curriculum.py \
    run_name=gen90_expel \
    seed_from_phase0_dir=results/expel_phase0_Base90_ExpeL \
    use_expel_memory=true \
    stopping.N=90 iterations=250

# 2) Extract ExpeL insights from ALL 90 generated trajectories.
python scripts/run_expel_baseline.py extract --out results/gen90_expel
#    → results/gen90_expel/insights.json == "ExpeL-Generated90"

# 3) Contamination check before eval (seeds + bank vs eval-150).
python scripts/check_contamination.py \
    --seeds data/sotopia_90_seeds.jsonl \
    --archive results/gen90_expel/archive_latest.json \
    --eval data/eval_150.jsonl --cosine-threshold 0.85 \
    --output results/gen90_expel/contamination_report.json
```