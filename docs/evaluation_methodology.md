# Evaluation Methodology: Internal Signal vs. External Claim

> There are two evaluation jobs in this project, and they want **opposite** properties.
> Conflating them is the fastest way to produce an indefensible result. This document
> pins down which instrument serves which job, and why.

---

## 0. The research question (the thing every eval choice must serve)

> *Does our method (scenario-generation curriculum + skills chronicle + reflection loop)
> produce a learner agent that is more socially competent on **unseen** scenarios than
> baselines (vanilla LLM, CoT, fine-tuned model, other methods)?*

Every evaluation decision below is justified by reference to this sentence. If a choice
does not help answer it more truthfully or more defensibly, it does not belong.

### 0.1 The claim is *self-improvement*, and that constrains the design

The claim is that **in-context, self-built social skills make a vanilla LLM more socially
competent at inference — competitive with fine-tuning, but cheaper and extensible**. This is
a **self-improvement** claim, not a distillation one:

- **Self-improvement (our claim):** the *same* model plays episodes, reflects on its own
  failures, and writes its own chronicle. At test time it is that same vanilla model + its
  self-built, retrieved chronicle, injected in-context. No weight updates, no stronger
  teacher model in the loop.
- **Distillation (not our claim):** a *stronger* model builds the chronicle that a *weaker*
  model consumes. The learning happens in the teacher's head.

**Why the claim forces self-improvement.** The selling point is "cheaper than fine-tuning,
just in-context." If a stronger model secretly built the chronicle, (a) the cost argument
collapses, and (b) a reviewer rightly reframes it as "you distilled a strong model into
prompts — compare against simply giving the weak model that strong model's generic advice."
So: **learner = reflector = the same model, throughout.** The partner and evaluator may be a
fixed/strong model, because they are *training/eval-time* costs, not part of the deployed
artifact (which is just "vanilla LLM + chronicle").

**The honest practical claims** (stronger than raw FLOPs): (a) no weight access needed (works
on closed API models); (b) extensible without retraining (add scenarios → better, no gradient
step); (c) interpretable (the chronicle is readable). Fine-tuning is opaque, needs weights,
and is frozen.

### 0.2 The existential pre-condition: a reflect–act gap must exist

Self-improvement is only possible if the same model is **a good-enough reflector** (can
diagnose what went wrong and write usable guidance) **and** has **behavioral headroom** (does
not already act that well live). That requires a **gap between the model's analytical
capability and its in-the-moment behavioral capability** — it can name the right move in
hindsight but does not reliably execute it under multi-turn pressure. Whether this gap exists
for a given model is **the go/no-go question for the whole project** and must be measured
first (§4).

---

## 1. The split

| | **Internal eval** | **External eval** |
|---|---|---|
| Question it answers | "Did this attempt succeed, so the loop can learn?" | "How much better is method A than method B?" |
| Consumer | the reflection / meta-reflection loop | the researcher, the reviewer, the paper |
| Form | **binary** (a success gate) | **graded** (continuous comparison) |
| Must be | **valid for this specific scenario** | **comparable across methods, method-agnostic** |
| Primary instrument | the **goal-derived rubric** (+ partner-perspective judge) | **SOTOPIA-EVAL** continuous dimensions |
| Arbitrariness risk | avoided by deriving criteria from the goal | avoided by using a standardized external metric |

The internal eval should be maximally faithful to *this scenario's contract*. The external
eval should be maximally *detached from our method*. These are different goals, so they use
different instruments. Using one instrument for both is the core mistake to avoid.

---

## 2. Internal evaluation — drives the learning loop

### 2.1 The success gate is the goal-derived rubric, not a SOTOPIA threshold

The success bit that decides solved/unsolved (and therefore whether reflection fires) is the
**rubric**, derived from the goal:

```
success = outcome_achieved  AND  constraint_preserved
```

- **Outcome** (e.g. "Maya agrees to take ≥ a week off and see a doctor") — checked by the
  **neutral transcript judge**; it is observable in the conversation.
- **Constraint** (e.g. "without Maya believing Anna went behind her back") — checked by the
  **partner-perspective judge**, given the partner's private profile and secret, because the
  cost the constraint protects lives *inside the partner* and a neutral reader cannot
  authentically assess it.

This is **non-arbitrary**: we are not picking a threshold, we are checking the contract the
goal already specified. The criteria are **derived from each goal**; we own the *method*
(decompose goal → route each condition to the right instrument), the goal owns the *content*.

> The decisive failure case this catches: **outcome achieved but constraint violated**
> = *hollow extraction*. In the cofounder run, the learner extracted a verbal commitment
> (`goal=9`) but the underlying conflict was unresolved and a betrayal-style shortcut was
> available. A `goal > 7` threshold scored that as success. The rubric's AND gate scores it
> as **not solved** — which is correct, and is what lets the loop learn.

### 2.2 Why NOT `goal > 7` (or any fixed SOTOPIA threshold) as the gate

- It is **arbitrary** — no principled basis for 7 vs 6 vs 8.
- It is **redundant** now that the rubric exists.
- It **cannot distinguish genuine buy-in from hollow extraction** — the exact failure mode
  we are trying to detect.

### 2.3 The legitimate internal roles of SOTOPIA-EVAL

1. **Diagnostics for reflection.** The 7-dimension vector tells reflection *which facet* was
   weak (low `relationship`, breached `social_rules`, leaked `secret`), so it can write a
   targeted chronicle entry. This is a primary internal use — keep it.
2. **Degeneracy floor (validity check, not success definition).** Reject episodes where the
   agent failed to play a coherent human at all — e.g. `believability` collapsed or
   `social_rules` was severely breached. This guards against degenerate transcripts; it does
   **not** define social success.

### 2.4 The partner-perspective judge (internal only)

- It is **not introspection** — there is no stored feeling to read back. It is a second LLM
  judgment, but **conditioned on the partner's private profile + secret + a first-person
  stance**, which lets it weigh things a neutral judge structurally cannot.
- It only ever answers the **constraint conditions the goal put there** — it does not invent
  criteria. (No goal constraint → nothing for it to check.)
- **Anchor its questions to behavioral follow-through and private-interest reconciliation,
  not vibes** ("would you actually do X?", "is your real interest met or did you just avoid
  conflict?"), because LLM partners are sycophantic and "did you feel respected?" inflates.
- Fixed evaluator, **always runs**, parameterized by the goal — not a router-gated, spun-up-
  on-the-fly agent. Within-scenario comparability (attempt 2 vs attempt 1 judged identically)
  comes from freezing the rubric across attempts.

### 2.5 Anti-gaming

Because the generator emits both the goal and (later) the rubric, the rubric must be
**validated by the coherence / adversarial gate**: is it faithful to the goal, genuinely
checkable, and not trivially satisfiable by the naive move? This extends the existing gate
role; it is not a new subsystem.

---

## 3. External evaluation — proves the research claim

The claim "method A beats method B" is defensible only when four conditions hold. Each maps
to a concrete design choice.

### 3.1 The measure is method-agnostic and standardized → use SOTOPIA-EVAL

- Use **SOTOPIA-EVAL's continuous dimensions** as the comparative metric — **not** our own
  rubric. The rubric is produced by our pipeline; scoring baselines against an artifact our
  method generated is a contamination a reviewer will (correctly) attack.
- SOTOPIA-EVAL is **external, published, human-validated, field-standard** → defensible and
  directly comparable to other work.
- **Report all 7 dimensions**, with `goal` as the headline. Our method may trade dimensions
  (higher `goal`, lower `relationship`); reporting the full vector is more honest and more
  informative, not less.

### 3.2 The test set is uncontaminated and not authored by the method under test

- **Do not** evaluate on scenarios our own generator produced — that is training-on-the-test
  by another name.
- Use the **canonical held-out SOTOPIA scenarios** (or a frozen neutral set), identical for
  every method. This also makes us directly comparable to the SOTOPIA paper and its
  descendants.
- **Seeding-leakage wrinkle (we already have this).** Our archive is *seeded* with SOTOPIA
  scenarios, so SOTOPIA scenarios are already inside the training pipeline. "SOTOPIA as
  held-out external eval" therefore requires a clean split: seed/train on SOTOPIA subset A,
  final-eval on a **disjoint** subset B, **split by social dynamic** (not just surface
  scenario) so the held-out dynamics are genuinely unseen. **Reserve subset B now** and never
  touch it during development. If all 90 seeds are currently used, carve out B before
  building.

### 3.3 The only thing that varies is the method

Fix across all conditions:
- same held-out scenarios,
- same **partner model**,
- same **evaluator model**,
- same turn budget.

The treatment: our agent carries its **learned chronicle** (retrieved per test scenario);
baselines do not. Everything else is identical. This isolates the measured effect to the
method.

### 3.4 Generalization, not memorization

- Split train / test **by social dynamic**, not just by surface scenario.
- The premise of abstract chronicle conditions is that skills **transfer**, so the strongest
  result is performance on social dynamics **not seen during training**.
- If the held-out set is structurally similar to training, a reviewer will say we memorized
  — and they would be right.

### 3.5 Statistics

- Multiple held-out scenarios × multiple episodes each (episodes are stochastic).
- Report **means with confidence intervals**.
- Use **paired** significance tests (same scenarios across methods).
- Validate the automated evaluator against **human judgment on a sampled subset** — SOTOPIA
  publishes human-correlation figures for exactly this. Gold standard for the paper:
  automated SOTOPIA-EVAL on the full set **+** human eval on a subset.

### 3.6 On "success rate" thresholds externally

Do **not** lead with a thresholded success rate — continuous scores have more resolution and
methods cluster near any cutoff. If a success rate is requested, derive the threshold from a
principled reference (SOTOPIA's own convention, or human-judged success on a calibration
subset) and present it as **one view alongside** the continuous results, never as the
headline.

---

## 3b. The upfront go/no-go experiment (run this FIRST)

Before building any machinery, run one cheap experiment that de-risks the whole project and
selects the learner model.

**What it measures (two things):**
1. **Headroom** — does the model fail a meaningful fraction of well-designed scenarios?
2. **Reflect–act gap** — does the model's *own* reflection, injected back, measurably improve
   the *same* model's behavior on a retry? (This is the existential pre-condition, §0.2.)

**Test set: our own generated sweet-spot scenarios (a held-out *dev* split) — NOT the
SOTOPIA held-out set.** Three reasons:
- **Keep SOTOPIA sealed.** Subset B (§3.2) is the final external benchmark; using it for a
  build decision contaminates the external claim. Non-negotiable.
- **Relevance.** SOTOPIA's distribution is mostly easier/different from the socially-rich
  scenarios our method targets. Its pass rate would mis-estimate headroom on *our* operating
  distribution.
- **The question is internal.** "Does the mechanism work at all" is about our scenarios; it
  makes no external generalization claim, so using our own scenarios is appropriate here — and
  not circular *for this purpose*. External validity is the separate, later job on sealed
  SOTOPIA.

**Transparency / anti-circularity:** state plainly that this check validates the *mechanism*
on scenarios built to our design principles; *external* generalization is tested separately on
held-out SOTOPIA. (Optional: calibrate the difficulty notion against a few SOTOPIA *seed/train*
scenarios — never subset B.)

**Decision rule:**
- Model aces most dev scenarios naively → no headroom → use a weaker learner *or* harden
  scenarios.
- Model fails a healthy fraction **and** reflection-injected-back improves its retry → green
  light, and this is the learner model.
- Model fails but reflection does *not* improve retries → the reflect–act gap is absent for
  this model → the method cannot work on it as-is.

## 3c. Scenario-quality filtering during archive expansion

The behavioral signal is the `outcome` the loop already computes — no separate naive rollout.
But rather than *cull* too-easy scenarios, we **edit them to bite** (the OMNI-EPIC-style
difficulty editor, run upward). The full algorithm — the difficulty-calibration loop, the
skill-learning loop, the archive policy, and why this dissolves the per-scenario counterfactual
— lives in **[`curriculum_loop.md`](curriculum_loop.md)**. In brief:

- **Solved on attempt 1** → not archived as success; routed to the **difficulty editor**
  (ratchet up via social knobs) until it bites, or discarded after `D` edits.
- **Solved-after-biting** (`outcome=2`) → the gold case; archived and **counts** toward the
  stopping condition.
- **Never solved** → archived as **failed** (frontier/too-hard); kept for generator
  conditioning but does **not** count.

Because the editor ratchets up until the *chronicle-equipped* agent fails attempt-1, every
archived success is one the chronicle could not already handle — so bite, a skilled path, and
diagnosability are all verified by construction (`scenario_design_sweet_spot.md` §4).

## 3d. Validate the core assumption: archive-size scaling curve

"A wider repertoire of past experiences → better performance on new tasks" (the OMNI /
open-ended-learning premise) is a **hypothesis, not a given**. Validate it directly: plot
**held-out SOTOPIA-EVAL score vs. archive size**. A rising curve is one of the strongest
possible results; a flat curve is itself an informative finding. Do not assert it — measure it.

---

## 4. One-line summary

> **The rubric is for learning truthfully; SOTOPIA-EVAL is for comparing fairly.**
> Internal success = goal-derived rubric (outcome AND constraint), with SOTOPIA dims as
> diagnostics and a degeneracy floor. External comparison = SOTOPIA-EVAL continuous
> dimensions on canonical held-out scenarios, fixed partner + evaluator, train/test split by
> social dynamic, paired statistics, human-validated subset.

See [`scenario_design_sweet_spot.md`](scenario_design_sweet_spot.md) for how scenarios are
designed so that this internal success signal is meaningful in the first place, and
[`curriculum_loop.md`](curriculum_loop.md) for the expansion algorithm (difficulty editor,
two-counter loops) and the ANNECS-style stopping condition (count solved-after-biting only).
