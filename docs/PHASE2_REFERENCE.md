# Social OMNI-EPIC — Phase 2 Reference

*What changed, why, and the intuition. Companion to `IMPLEMENTATION_SPEC_phase2.md` (which has the nitty-gritty).*

---

## The diagnosis that drove everything

Phase 1 runs were **bimodal**: scenarios were either solved on the first try (discarded) or never solved at all (archived as failures). The fail-then-succeed band — the only place learning happens, and the system's entire scientific claim — was nearly empty. Three causes:

1. **Difficulty is a property of (scenario, partner policy, evaluation), but we only calibrated the scenario.** The partner prompt ("concede nothing") was a cliff, not a dial. Scenarios with a zone of agreement on paper had none in execution.
2. **The AND-gated binary rubric destroyed the gradient.** Partial wins scored as total failures; the continuous SOTOPIA scores sat unused as diagnostics.
3. **Calibration was asymmetric and ran on single rollouts.** Too-easy → edit harder; too-hard → walk away. No path back from beyond the frontier, and one Bernoulli sample misclassifies a true 50%-scenario half the time.

A quieter problem: the tempting shortcut almost never fired (preference-tuned learners won't blackmail), so failures were "partner refused," not "constraint violated" — the theory and the data were about different phenomena.

## The redesign in one paragraph

Difficulty now lives in a **hidden partner key** — theory-grounded conditions under which the partner genuinely moves — so every generated scenario is *solvable by construction* and hard by *discoverability, prior-incongruence, and cost*. The curriculum signal is **learning progress (LP)**, measured by a cross-lab pairwise judge comparing attempt 1 against later attempts; LP feeds Thompson sampling as pseudo-votes and doubles as the unsolvability detector (flat LP = beyond frontier). The per-scenario difficulty-editing loop is gone; calibration happens through **selection pressure** (Thompson over anchors) and **directed mutation** (escalate / relax / lateral operators conditioned on the anchor's measured outcome). Reported success is the standard, comparable label **GOAL ≥ 7 ∧ REL ≥ 0**, plus an objective key-aware check on generated scenarios.

## Component by component

**Goal triple (outcome–constraint–shortcut).** Kept as *generation scaffolding* — it's what forces structural tension into scenarios — but removed from evaluation. Scope claim: the system covers *dyadic influence interactions* (Dillard's primary/secondary goals; Brown & Levinson's face threat). We construct difficulty from theory rather than discovering it post-hoc; we claim sufficiency, never universality.

**Partner key (new).** Five fields: a `key_mechanism` from a small theory library (reactance, face needs, validation-before-change, procedural voice, reciprocity/disclosure), `movement_conditions` (what actually shifts the partner), `hardening_triggers` (what locks them — reactance in action), `surface_misdirection` (stated vs. real objection — the discoverability dial), `cost_coupling` (what satisfying the key costs the learner's own goal). Solvability is guaranteed because a key exists; difficulty is tuned by how hidden, how counter-to-RLHF-instinct, and how costly the key is. Realism rests on mechanism-tagging: every key instantiates a documented phenomenon, which is auditable — unlike free-form "realistic" personas. The library is explicitly non-exhaustive and swappable; no completeness claim. One structural consequence: generated scenarios are role-asymmetric by construction (learner = agent 0 with the triple; partner = agent 1 with the key), with roles committed at generation time to continue the anchor's perspective — replacing the old post-hoc embedding-based target designation, which was symmetric in schema but asymmetric in use (only the learner's triple was ever evaluated).

**LP judge (new).** Key-blind. Compares attempt 1 vs. each later attempt, both presentation orders, "A / B / no difference." LP = fraction of improved-votes ∈ [0,1]. Order-flip = no-difference (kills position bias). Coarseness is fine: Thompson aggregates votes across an anchor's whole lineage, so selection operates on smooth lineage-level posteriors built from coarse per-scenario atoms.

**Classification & minimal criterion.** Every completed scenario is archived (the archive is a map, not a trophy case): `too_easy` (first-try success — archived, no chronicle, full down-weight), `frontier` (failed first, LP > 0 — the viable curriculum artifact; chronicle enters retrieval), `beyond_frontier` (flat LP — archived as a negative example *and* re-selectable for the relax operator). The two-sided bar — failed attempt 1 AND LP > 0 — is a Minimal Criterion in the POET/MCC sense.

**Mutation operators (replaces the edit loop).** The generator mutates 1–2 named structural slots of its anchor, direction chosen by the anchor's outcome: too easy → **escalate** (tighten misdirection/triggers/cost), beyond frontier → **relax** (loosen the implicated slot), at frontier → **lateral** (hold difficulty, explore a different mechanism/asymmetry — coverage). This is bidirectional calibration relocated from per-scenario surgery to generation, where it's free. Chronicle inheritance means escalation is met by a more capable learner; LP flatlining is the automatic brake against difficulty inflation.

**Gate pipeline.** Each axis owned by exactly one mechanism:
- *Solvability* → coherence gate, now verifying the key (exists, consistent, never leaks, cost is real) instead of guessing ZOPA from narrative.
- *Worth* → MOI, slimmed to one axis and converted from gate to **ranker** over a 3-candidate batch (a ranker can't saturate to "yes").
- *Novelty* → cosine dedup (full-text embedding, admission) + niche bookkeeping (abstract `social_dynamic | target_perspective` embedding, k-means) — structure for coverage claims, surface for de-duplication.
- *Learnability* → measured by LP, not predicted by a judge.
- All gates now **fail closed** (retry → quarantine), never default-pass.

**Chronicle.** Architecture unchanged (reflect → adversarial check → synthesize → meta-reflect). Fixes: diagnosis vocabulary rewritten around the key world (pressure failure / discovery failure / cost avoidance / capitulation); reflection sees key-check *verdicts* but never the key text itself (entries must stay key-blind to transfer); injection truncation is now relevance-ranked top-8, not first-8; the dormant final adversarial check is wired in.

**Phase 0 (new).** One calibration pass over the SOTOPIA-90 seeds, run *native* (no triple retrofit, no key — the schema belongs to generated scenarios only). Triple duty: informed Thompson priors, chronicles on frontier seeds for the generator, and it *is* the "chronicle-without-generation" ablation. Defines t=0 for saturation/coverage curves. Headline runs start from clean seeds — dev-run chronicles are never imported (they were produced by the deprecated pipeline; attribution dies otherwise).

**Evaluation protocol.** Held-out 150 from SOTOPIA-π. One **frozen partner engine** (model + prompt + temperature) across all conditions; personas vary per scenario but identically for everyone; eval partners are vanilla (no key, no chronicle — the claim is transfer to *standard* tasks). Report raw scores and deltas vs. the vanilla learner (cancels judge leniency and partner agreeableness). Judge model chosen by validation — human-agreement on annotated SOTOPIA episodes + swap-consistency — and must come from a different lab than the learner (breaks the self-evaluation monoculture). Contamination check (ID + embedding overlap between eval set and banks) and compute accounting per condition; ExpeL gets ~3 trials/task for episode parity.

## The one untested assumption

Everything above leans on the partner *holding its key* across 20 turns: never leaking it, never softening before a condition is met, hardening on triggers. **Five probe scenarios run first**; acceptance bar is near-zero leak/premature-softening. If the partner can't hold the key, the partner prompt iterates before anything else is built.

## Deferred (documented, not implemented)

Two-level (niche-first) Thompson sampling → future work in the paper; niche counts are logged so the need is observable. Multi-seed variance runs → deferred for the workshop; seed fixed and logged now so the workshop run becomes run 1 of 3; the draft carries the single-run limitation sentence (cite Henderson et al., *Deep RL That Matters*). Also deferred: MOI positive controls, cross-lineage chronicle consolidation, mechanism-library extension.

## Paper positioning (one breath)

Social OMNI-EPIC is a quality-diversity coevolutionary system: the environment is a social scenario, the minimal criterion is positive learning progress (POET/MCC lineage), mutation is FM-driven structural editing over a theory-grounded grammar (Dillard; Brown & Levinson), parent selection is Thompson sampling rewarded by descendants' learning (Darwin-Gödel-style), transfer is chronicle inheritance along lineages — and the contribution is constructing social difficulty from specified mechanisms rather than discovering it post-hoc, with solvability guaranteed by construction and difficulty verified by measurement.
