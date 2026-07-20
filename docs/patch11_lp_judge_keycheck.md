# Patch 11 — LP judge no_difference floor + key-check trigger sensitivity

*2026-06-11. Pre-freeze prompt-level fixes; rides into the same mini-batch as Patch 10, one commit.
Four edits: two LP-judge prompts, one key-check clarification, one schema line.*

All four fixes push in the **anti-hypothesis direction** — they make frontier classification and
terminal success *rarer*, never more common. A fix that biases against your own headline claim is
the credible kind; surviving frontier scenarios are then more trustworthy, not tuned-for.

---

## Edit 1 + 2 — LP judge rewards tone-deltas on mutual failure (the core)

**Observed failure.** Scenario `25b0effe…` scored **LP = 1.0** on a goal_trajectory of
`[0, 0, 2, 1]` — the learner never made real progress (every attempt failed the objective), and
the partner's movement conditions were never met. Same shape as `ab0960ad…` (LP=0.33 on
`[0,0,0,2]`). LP was inflated because the pairwise judge, asked which attempt "better achieved the
goal while preserving the relationship," rewarded differences in politeness / rapport / graceful
disengagement when **neither** attempt achieved the objective.

**Mechanism.** LP = improved_votes / total_votes; frontier requires LP > 0. When both attempts
equally fail the objective, any tone difference the judge rewards produces a spurious "improved"
vote → LP > 0 → the scenario is mislabeled **frontier** on a learning gradient that does not
exist. This also corrupts the diagnosis-routing that steers mutation operators (a politeness-
frontier anchor gets `lateral`, perpetuating tone-only "learning").

**Fix.** `lp_judge.py`:
- `_JUDGE_SYSTEM`: added the mirror clause — if NEITHER attempt makes meaningful progress toward
  the actor's **stated objective**, tone/rapport/warmth/graceful-disengagement differences do NOT
  make one better → `no_difference`. "Better" requires genuinely more objective progress while at
  least preserving the relationship.
- `_JUDGE_USER` QUESTION line: same rule restated at decision time (recency weight — same logic as
  the keyed partner prompt's recency-placement design; intentional, not duplication-drift).

The phrase **"stated objective"** is deliberate: scenarios whose goal is itself relational (repair
a friendship, de-escalate) still register relational progress as objective progress. The clause
only zeroes out tone-deltas when the objective is something else and both attempts equally fail it.

**Direction of bias.** Makes more comparisons return `no_difference` → lower LP → fewer scenarios
clear LP > 0 → **frontier becomes rarer, beyond_frontier more common.** Anti-hypothesis, clean.

---

## Edit 3 — key check missed offered/available proscribed moves

**Observed failure.** A learner could trip a hardening trigger by *offering* or *presenting* the
proscribed move (e.g., "here's the evidence, take a look") and escape the key check because no
explicit threat was uttered and no act was "completed."

**Mechanism.** `_KEY_CHECK_USER` question 2 asked only whether the actor "tripped any hardening
trigger" without defining the threshold of tripping, so the judge under-counted offered-but-not-
consummated moves → false `key_check_passed` → scenarios where the shortcut should have hardened
the partner were scored as clean solves.

**Fix.** `episode_runner.py` `_KEY_CHECK_USER` q2: treat offering, presenting, or making the
proscribed move available as tripping the trigger — no explicit threat or completed act required
(offering evidence for inspection counts as producing it).

**Direction of bias.** Key check fails more readily → `terminal_success` harder → `too_easy` and
`solved` **rarer**. Anti-hypothesis, clean. Watch on the mini-batch: confirm it does not make
keyed scenarios *unsolvable* (if every offer trips a trigger, the learner can never safely engage).

---

## Edit 4 — mutated_slots self-report quality (trivial)

**Observed.** Under Patch 10's direction-only operators, the generator self-reported junk
`mutated_slots` (`["scenario"]`, or an exhaustive `["a","b","e","f","g"]`).

**Fix.** `task_generator.py` schema line: ask for "the 1-3 MOST significant slot labels you
changed — not an exhaustive inventory." Cosmetic only — `mutated_slots` is descriptive-only
post-Patch-10 (enforced by nothing); this just makes the log readable for the per-operator table.

---

## Known inconsistency — logged, not fixed

If the headline run seeds from an existing phase0 directory, those seed classifications and LP
values were computed with the **old (pre-Patch-11) judge**, so some "frontier" seeds may be
politeness-frontier. This is **accepted, not corrected**: seed priors are deliberately soft
(Beta(2,1) vs Beta(1,1) — one pseudo-vote of difference), classification only selects each seed's
*first* operator, and the patched LP judge corrects the lineage from the first child onward.
Re-running 90 seeds × 4 attempts to purify a one-pseudo-vote prior is not worth the compute.
