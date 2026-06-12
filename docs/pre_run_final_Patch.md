# Social OMNI-EPIC — System State & Final Pre-Run Redesign (Patch 10)

*2026-06-11. Consolidates the full audit conversation: current-state map from `run_curriculum.py`
down, the first-principles rationale, the unified-operator redesign, validation criteria, and the
freeze protocol. This supersedes the operator-related parts of Patches 8/9 and absorbs them.*

---

## Part I — The system as it exists (entry point: `run_curriculum.py`)

### I.1 One iteration, end to end

```
thompson_select (archive.py)
  → operator := f(anchor.classification)        too_easy→escalate · beyond→relax · else→lateral
  → context assembly                            KNN examples (lineage-EXCLUDED) +
                                                beyond-frontier negatives + existing types
  → generate_batch_from_archive (×3 candidates) anchor-first prompt + operator block + key schema
  → fuzzy_key_leak_check                        free pre-filter (leaks, off-channel artifacts)
  → MOI rank_batch                              worth-only ranker, best-first
  → per candidate, until one passes:
       embed → coherence gate (LLM, patch-retry ×2, fail-closed)
       → IF lateral:        embedding diversity gate (max cosine vs archive > 0.92 → reject)
         IF escalate/relax: key_delta_check vs anchor (skipped entirely if anchor has no key)
  → lineage stamp, target designation, live record stub
  → run_episode_two_loop (curriculum.py)        K attempts; attempt-1 success → too_easy fast path;
                                                Reflexion/chronicle memory between attempts;
                                                LP judge over (1,j) pairs; LPAllErrors → quarantine
  → classification: too_easy | frontier (LP>0) | beyond_frontier (flat LP)
  → posterior update on the ANCHOR:             too_easy → (0, 6) · frontier → (improved, total)
                                                beyond → (0, total|6) · gen-fail/discard → (0, 1)
  → child enters archive with capped warm-start prior; checkpoints + aggregates flushed per batch
```

### I.2 What is healthy (do not touch)

- **Fail-closed discipline everywhere it matters**: coherence quarantines on FM error; LP raises
  `LPAllErrorsError` into the quarantine path instead of mislabeling beyond_frontier; the key
  check fails closed at temperature 0.
- **Crash-safety**: atomic per-scenario records, live-write on every turn, aggregates rebuilt from
  disk, resume from `archive_latest.json` as the single source of Thompson state.
- **Honest-experiment guards**: cross-lab judge asserted at startup; `random_seed` mandatory;
  per-FM compute metering.
- **The soft failure penalty** (`(0, 1)` per generation failure/discard): quietly load-bearing —
  see III.3, it becomes the anti-mill mechanism for free.
- **Lineage-excluded KNN examples**: someone already noticed that same-root siblings are useless
  prompt context. Correct instinct; Patch 10 extends the same insight to admission gating.
- **Instrumentation that Patch 10 needs already exists**: `parent_child_cosine` and
  `within_batch_dupe` are logged per iteration in `metrics.json`;
  `per_operator_classification_counts` is in `summary.json`. Zero new instrumentation required
  to validate this redesign.
- The seven contract patches (judge strictness, partner_goal leak scoping, off-channel bans,
  sensor-form conditions, artifact regex, cost survivability, resistance-phrase variation) and
  the keyed partner prompt's recency-placement + in-character-sensor design.

### I.3 The three structural faults (what this round fixes)

**Fault 1 — The operator wiring creates same-surface mills.**
`anchor.classification` is written once and never updated, and the operator is a pure function of
it. A phase0 `too_easy` seed whose escalate child lands at frontier gets *credited* in its
posterior (Darwin-Gödel reward, by design) → re-selected → `too_easy` → escalate again →
escalate's contract *forces* surface preservation. A productive too-easy anchor is therefore a
machine for stamping out same-surface children, and the reward signal feeds the machine.

**Fault 2 — Gate asymmetry leaves sibling duplication completely unchecked.**
Escalate/relax children bypass the embedding diversity gate by design and are checked only
against their *anchor* via `key_delta_check`. Siblings are never compared to each other. Worse:
when the anchor is a keyless seed, `key_delta_check` returns `[]` (explicit skip) — so an
escalate-on-seed child is admitted with **no novelty or fidelity gating whatsoever**. Under
phase0 seeding, seeds carry classifications, so this wide-open channel is hit constantly in the
early run, exactly when most generation happens off seeds.

**Fault 3 — Operators are incoherent on keyless anchors.**
Escalate says "tighten (b)/(c)/(d)"; relax says "loosen the implicated slot"; lateral says
"preserve (b)/(c)/(d)". A seed has no (b)/(c)/(d). All three instructions are vacuous or
misleading on depth-1 generation — which is most of the run.

(Minor cleanup, noted in passing: `TaskGenerator.select_examples` is unused by the production
path — `run_curriculum.py` does its own inline KNN — and should be deleted or marked legacy.
`ModelOfInterestingness.rank_batch`'s `archive_scenarios` parameter is dead; comment or remove.)

---

## Part II — First principles: what this system is actually for

The thesis: **for capable LLMs, the frontier of competence survives in the social domain because
social difficulty is continuous — and that frontier can be located, tracked, and harvested.**
The system embodies a strict division of labor:

| Question | Owner | Mode |
|---|---|---|
| Is this scenario at the frontier? | LP judge + classification | **measured** (never predicted) |
| Is it worth being at the frontier of? | MOI worth ranker | **judged** (the one FM-taste axis) |
| Does the bank span the space? | embedding gate + niches | **enforced** (population-level) |
| Is the signal identifiable? | partner key + coherence gates | **constructed** (the instrument) |

Phase 1 failed because it asked the generator to *predict* difficulty ("be interesting and
learnable") — a prediction no generator can make about a learner it has never observed. Phase 2's
core move was to stop predicting and start measuring: LP reveals the frontier post-hoc, Thompson
converts the measurements into selection pressure. **That is the entire reason the measurement
apparatus exists: so the generator does not have to be right about difficulty.**

The slot-surgery operators are a relapse into Phase 1 thinking. "Tighten slot (d) while
preserving slots (a)(e)(g)" is generation-time difficulty micro-management — the exact job LP was
built to take over. The operator's only legitimate responsibility is to set a **direction** on
the difficulty dial (harder / easier / same); whether the child actually landed where directed is
LP's verdict, accumulated into the anchor's posterior, self-correcting by construction. An
"escalate" that fails to escalate produces too_easy children, charges its anchor (0, 6), and
selection moves on. The loop already polices direction; the slot rules only constrain creativity.

There is a second first-principles commitment hiding here: **the paper's central claim is that
difficulty lives in the key (genotype), not the surface (phenotype).** An escalate operator that
clamps the surface "to keep the comparison controlled" is hedging against that claim. A system
that believes difficulty-by-construction should expect difficulty to survive re-skinning — and
should *test* that expectation by re-skinning everything. Universal surface variation is not a
patch; it is the thesis applied to the system's own design.

Finally, the project's own §6.3 principle — *each axis owned by exactly one mechanism* — is
currently violated by the gate routing: surface novelty is owned by the embedding gate for
laterals and by nobody for escalate/relax. Patch 10 restores single ownership: **the embedding
gate owns surface novelty for every child, period.**

---

## Part III — The change set (Patch 10)

### III.1 Unify the operators as direction-setters with universal fresh surface

**File: `task_generator.py`.** Replace the three operator entries in `_EDIT_INTENTS`
(`escalate`, `relax`, `lateral`; keep `fix_coherence` unchanged) with a shared preamble plus
three direction clauses.

**Shared preamble** (prepended to all three via `_MUTATION_OPERATOR_TEXT` or a new
`_OPERATOR_PREAMBLE` constant):

> Generate a NEW scenario in the parent's structural family — the kind of tension, asymmetry,
> and learner vantage point named in its scenario_title — with a completely fresh surface: new
> character names, new setting, new occupational world, new specific stakes. NEVER reuse the
> parent's character names, venue, or figures. The parent's value to you is its structural
> family, not its text.
>
> If the parent has a partner_key, treat it as the reference point for difficulty. If it has
> none (a seed), you are inventing the key from scratch: sensor-form movement conditions,
> survivable cost coupling, spoken-turn satisfiable throughout.

**`escalate`** →
> Target difficulty ABOVE the parent's: a better-hidden path (deeper surface misdirection),
> a higher cost of satisfying the movement conditions, or conditions that cut more strongly
> against an agreeable model's trained instincts. SURVIVABILITY (unchanged, critical):
> harder or partial, never strictly unreachable — after satisfying the conditions, a skilled
> actor must still have a path to a meaningful version of their stated outcome.

**`relax`** →
> Target difficulty BELOW the parent's: a more discoverable path, a lower cost of meeting the
> conditions, or conditions less opposed to an agreeable model's instincts. The
> beyond_frontier_diagnosis above names what was stuck — make sure THAT dimension is the one
> that eases. Hard but genuinely winnable.

**`lateral`** →
> Target the SAME difficulty as the parent, expressed through a different mechanism, a
> different asymmetry, or a different relationship structure.

`mutated_slots` survives as a **descriptive log** — the generator self-reports what it varied —
but it is no longer a contract that any guard enforces. `mutation_rationale` unchanged.

*Rationale:* direction is the only difficulty decision a generator can responsibly make;
everything finer is LP's job. Creative freedom returns to where Phase 1's verbalized-sampling
instincts wanted it — but now with a measurement loop that makes the freedom safe.

### III.2 Universal admission gating

**File: `run_curriculum.py`, the candidate-walk (~lines 274–300).** Delete the
`if mutation_op == "lateral": ... else: key_delta_check ...` branch. Every candidate, every
operator, runs:

1. **Embedding diversity gate** vs the whole archive (threshold 0.92, unchanged). Because
   completed children are archived, this automatically covers siblings, parents, and everything
   else — sibling deduplication requires no new mechanism.
2. **`surface_novelty_check(cand, anchor)`** — new, deterministic, ~15 lines in
   `validation.py`: flag if (i) any anchor first name appears among child first names,
   (ii) `fuzz.partial_ratio(child.scenario, anchor.scenario) > 90` (cheap clone catch before
   the embedding call), (iii) `mutated_slots` is empty (log-quality floor only). Violations
   route to the same continue-to-next-candidate path as the diversity gate.

Remove the `key_delta_check` import and call. Leave the function in `validation.py` with a
deprecation comment (it documents the old design and may be useful for post-hoc analysis).
Delete the `orig_slots` capture/restore dance — it existed only to protect `key_delta_check`
from the coherence-patch path.

*Rationale:* single ownership of the novelty axis; closes Fault 2 in one move; net code
deletion.

### III.3 The anti-mill mechanism is emergent — verify it, don't build it

With universal gating, a mill anchor self-limits: once its neighborhood is populated, new
candidates off it fail the diversity gate → `generation_failed` → the existing **(0, 1) soft
posterior penalty** fires → repeated failures erode exactly the anchors whose neighborhoods are
saturated → Thompson redistributes. The mechanism is already in the code
(`record_child_outcome(anchor_idx, 0, 1)` on failure); Patch 10 merely connects it to the
duplication problem by making duplication *visible* to the gate.

*Action:* none in code. During the run, watch `metrics.json` for anchors with repeated
`no_candidate_passed_gates` — that is the mechanism working, not a bug. If a single anchor
racks up many such failures while remaining frequently selected, only then consider a stronger
per-selection decay (post-workshop).

### III.4 Carried forward unchanged from this audit cycle

- **Patch 9**: port the MOI worth language into the generator's INTERESTING definition
  (replaces the one-line version in `SYSTEM_PROMPT`):
  > INTERESTING: the tension is one a thoughtful person would recognize as a real, meaningful
  > social situation — power imbalances that aren't just positional, face costs that aren't
  > just ego, dynamics a perceptive person would recognize from life. NOT a logic puzzle
  > wearing a social costume, NOT a management-training vignette, NOT a generic archetype.
- All seven contract patches (judge, leaks, artifacts, sensor form, survivability, variation).
- LP judge, key check, Thompson math, capped child priors, phase0 seeding, coherence pipeline.
- Mechanism library stays a **closed set** (auditability + `VALID_MECHANISMS` validation);
  opening it to novel grounded mechanisms is explicitly post-workshop future work.
- Cleanup (optional, zero-risk): delete unused `select_examples`; remove MOI's dead
  `archive_scenarios` parameter.

### III.5 What this costs, honestly

- **Ceteris-paribus attribution is gone.** A surface-varied escalate child that lands too_easy is
  ambiguous between "key not tight enough" and "new surface accidentally forgiving." Nothing in
  the loop consumes that attribution (diagnoses run on the child itself; Thompson eats votes),
  and the aggregate claim survives (III.6 criterion 4) — but per-pair causal statements about
  single mutations are no longer available. One limitation sentence in the paper.
- **The old separation criterion dies** (lateral-far / escalate-near). It is replaced, not lost:
  the operators' identities now live in the *difficulty direction* of their children, which is
  the property that ever mattered.

---

## Part IV — Validation: success criteria for the mini-batch, tripwires for the run

**Mini-batch** (generation-side only, no episodes): ~15 laterals + ~5 escalates + ~3 relaxes
off seed and (where available) keyed anchors. Precondition: billing fixed, embeddings live.

1. **Hard compliance, 100%**: no child reuses an anchor first name; every child carries a
   non-empty `mutated_slots`; zero `key_delta` entries in the gate-fail log (the gate no longer
   exists).
2. **Prompt does the work**: ≥80% of candidates pass the diversity gate + surface check on the
   first generation attempt.
3. **Unimodal novelty**: the `parent_child_cosine` distribution (already in `metrics.json`) is
   below ~0.90 for *all* operators — no bimodal escalate-hugging lobe.
4. **No quality regression**: coherence pass rate, leak/artifact hits, and MOI worth comparable
   to the pre-patch batch.
5. **Eyeball, last**: five children read as "same structural family, genuinely new situation."

**Run-time tripwires** (all from existing logs):

- `per_operator_classification_counts` (`summary.json`): escalate children skew harder-classified
  than their anchors, relax easier. This is the measured form of the direction claim — and a
  paper table.
- `niche_counts` concentration: if >50% of generations land in <20% of niches, lineage collapse
  is occurring → documented evidence for niche-first Thompson next iteration.
- Repeated `no_candidate_passed_gates` on one anchor: anti-mill working (III.3).

---

## Part V — Freeze protocol and what the paper gets

Document this file as **Patch 10** alongside Patches 1–9, with the Fault 1–3 evidence (the
Gourmet Delights record, the Sasha/Emily lateral, the anchor-reselection trace) as the observed
justification. Then the order of operations is fixed:

1. Implement III.1–III.4 (≈ one focused day; net-negative lines).
2. Mini-batch → Part IV criteria.
3. Green → **freeze for real**: commit hash recorded, no prompt or gate edits after this point
   for any reason short of a crash bug, and never in response to evaluation results.
4. Probe run (5 keyed probes × episodes; partner key-holding + judge honesty counters).
5. Full Gen-90 run → ExpeL extraction → held-out evaluation.

What the paper gains from this redesign, beyond a cleaner run: the gate-ownership table becomes
genuinely clean (one novelty gate, one coherence gate, one worth ranker, one LP signal); the
operator story sharpens from "we surgically edit difficulty slots" (invites the hard-coding
critique) to "operators set direction; learning progress verifies it" (which is the thesis,
restated as engineering); and the per-operator classification table converts the operators from
assumed mechanisms into measured ones. The system stops hedging against its own central claim —
that difficulty lives in a specified counterpart model, not in the costume the scenario wears.