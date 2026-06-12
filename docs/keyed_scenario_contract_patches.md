# Keyed-Scenario Contract Patches (2026-06-11)

Seven prompt/regex-level fixes for failure modes observed in the first `gen90_expel`
runs. None are architectural — the keyed partner prompt, key check, and curriculum loop
are sound. The failures lived in the **contracts between parts**: a strict rule attached
to a field nobody reads, a hidden key leaked through the partner's own goal, a ban that
covered the learner but not the partner_key.

**Systemic stakes:** lenient GOAL scoring does not merely mislabel one scenario. A
vetoable/partial outcome scored `GOAL=10` is labelled *solved*, which feeds the wrong
signal into `analyze_beyond_frontier`'s diagnosis-routing and selects the wrong mutation
operator. Judge strictness (Patch 1) is therefore the highest-leverage fix.

---

## Patch 1 — Judge strictness on the *gating* score

**File:** `social_omni_epic/episode_runner.py`

**Problem:** `_EVAL_PROMPT` carried strict language — but attached to `goal_achieved`,
which was parsed and then ignored. `terminal_success` gated on the `goal` *score*, whose
only guidance was an example reading "7 = Achieved most of their stated goal." A Micah-vetoed
contingent pilot scored 10/10.

**Before** (`_EVAL_PROMPT` tail):
```
Score ranges: believability 0–10, relationship −5 to 5, ... goal 0–10.
For goal_achieved: true only if the agent substantially completed the specific, verifiable
objective stated in their goal — not just partial progress.
```
```python
# terminal success
base_success = (goal_score >= 7.0) and (rel_score >= 0.0)
```

**After** (added GOAL-dimension rubric; conjoined `goal_achieved`):
```
Scoring the GOAL dimension (critical — this is the dimension that decides success):
- Score against the LITERAL end-state reached when the conversation closes...
- An outcome that remains under the OTHER party's unilateral future discretion
  (they can still veto, delay, or walk it back) is NOT achieved — cap at 6.
- An outcome that is PARTIAL relative to the agent's stated target ... — cap at 6.
- A firm, MUTUAL commitment to the agent's actual stated objective ... CAN score 7–10.
- Treat claimed off-channel artifacts ... as NON-evidence.
```
```python
judge_goal_achieved = bool(learner_scores.get("goal_achieved", True))
base_success = (goal_score >= 7.0) and (rel_score >= 0.0) and judge_goal_achieved
```

**Deliberate deviation from the review:** did **not** blanket-cap all promises at 6. A
firm mutual verbal commitment to the literal objective is often the legitimate terminal
state in negotiation; capping all of them would make every negotiation unsolvable and
break `too_easy` detection. The cap targets *unilateral-discretion* and *partial* outcomes
only.

Also threaded the discarded `goal_achieved` bool out of `_evaluate_diagnostics`:
```python
# before:  learner_scores, _ = _unpack_dimensions(a1)
# after:
learner_scores, learner_goal_achieved = _unpack_dimensions(a1)
learner_scores["goal_achieved"] = learner_goal_achieved
```

---

## Patch 2 — `partner_goal` leak (the smoking gun)

**Files:** `social_omni_epic/coherence_check.py` (check #6), `social_omni_epic/task_generator.py`

**Problem:** the partner enumerated its conditions in turn 1 *despite* a well-built keyed
prompt forbidding it — because the generator wrote the movement_conditions verbatim into
`partner_goal`. The partner was faithfully pursuing its stated goal. The fuzzy leak check
is lexical (paraphrase < 85) and never covered `partner_goal`.

Observed record:
```
partner_goal: "...you will only accept a tightly scoped, blind trial on a non-flagship
               product with your direct oversight and public credit."
movement_conditions: ["Micah insists on leading and being visibly credited...",
                      "Micah requires the pilot be limited to a small, non-flagship SKU..."]
```

**Before** (generator rule, `task_generator.py`):
```
- The `partner_goal` must be written as "Your goal is to ..." and encode the partner's
  position, stake, and what they are willing to concede. It must NOT reveal any partner_key field.
```

**After** (added WRONG/RIGHT contrast):
```
- ... It must NOT reveal any partner_key field, and CRITICALLY it must NOT state the
  conditions under which the partner would move or concede.
  WRONG (leaks the key): "...you will only accept a tightly scoped blind trial ... with your
  direct oversight and public credit" — hands the learner the movement_conditions verbatim.
  RIGHT: "...protect the brand's premium positioning and your standing as its quality steward;
  you are deeply skeptical of cost-driven supplier changes."
```

**Before** (coherence check #6 — no partner_goal clause).
**After** (added semantic clause, since lexical can't catch paraphrase):
```
- PARTNER_GOAL LEAK (judge semantically, not lexically): Does the partner_goal state the
  CONDITIONS under which the partner would move, concede, or accept? ... If the partner_goal
  names the movement_conditions ... flag it — that hands the learner the hidden key.
```

---

## Patch 3 — Off-channel artifact ban in turn prompts

**File:** `social_omni_epic/episode_runner.py`

**Problem:** no prompt constrained agents to the conversation, so "I've just forwarded the
confirmation to your email" was a legal move (DocuSign theater).

**Before:** turn prompts ended with "Keep your responses conversational..." with no spoken-only
constraint.

**After** (added to learner `_TURN_PROMPT` and **both** partner prompts):
```
Everything happens within this spoken conversation. You cannot send, sign, forward, show,
email, or have already sent any document, contract, or artifact — and claiming you did is
not a real move. Every commitment is made and tested in words, here.
```

---

## Patch 4 — Condition phrasing contract (sensor form, not demand form)

**File:** `social_omni_epic/task_generator.py` (`_PARTNER_KEY_RULES`, wired into `SYSTEM_PROMPT`)

**Problem:** the generator authored conditions as *demands* ("Micah insists on independent
lab verification"). A model role-playing a character described as *insisting on X* will
insist on X, out loud, immediately — defeating the keyed prompt's sensor framing.

**Before:** the `partner_key` schema described `movement_conditions` only as
`"condition 1 (concrete, behaviorally checkable)"`, with no phrasing contract.

**After** (new `_PARTNER_KEY_RULES` block, abridged):
```
1. SENSOR FORM, NOT DEMAND FORM. Write each movement_condition as a thing the LEARNER does
   that the partner silently responds to — "the learner, unprompted, offers them visible
   leadership" — NEVER as a partner demand — "the partner insists on leading." If you could
   rewrite the condition as "I want X", rephrase it as "something shifts when the other person
   actually does X."
2. NOT VOLUNTEERED. movement_conditions must be conditions the partner would NOT announce
   unprompted...
3. SPOKEN TURNS ONLY. ... NO external artifacts ... rewrite "a written guarantee that his role
   remains intact" as "he hears the learner publicly commit, in front of him, that his role stays intact."
4. The cost_coupling must leave a survivable path ...
```

---

## Patch 5 — Artifact regex over partner_key fields (soft → patch)

**File:** `social_omni_epic/coherence_check.py` (`_KEY_ARTIFACT_PATTERNS`)

**Problem:** `_ARTIFACT_PATTERNS` already existed in `skills_chronicle.py` but was pointed at
chronicle synthesis, never at `partner_key`. The key itself mandated paperwork; the agents
fabricating documents was downstream of the scenario demanding documents.

**Design choice:** made it a **soft** check that returns a coherence issue (routes to
patch/regen), **not** a hard drop — tokens like "signature" risk false positives, and a
false positive on a soft check just triggers a rewrite. Patterns tightened to require
artifact-*requirement* phrasing:

**After:**
```python
_KEY_ARTIFACT_PATTERNS = [
    r"\bin writing\b",
    r"\bwritten (guarantee|agreement|commitment|confirmation|...|sign[- ]?off)\b",
    r"\bsign(ed|s|ature)\b", r"\bsign[- ]?off\b", r"\bcontract\b", r"\bescrow\b",
    r"\b(send|...|forwarded) (a |an |the )?(email|document|confirmation|note|letter|memo)\b",
    r"\b(documented|documentation)\b", r"\breceipt\b", r"\bnotari", r"\bon paper\b",
]
```
Wired into `_fuzzy_key_leak_check`, scanning `movement_conditions` + `hardening_triggers`.

Validation (8 cases): catches "a written guarantee...", "he signs off...", "she sends a
confirmation email..."; correctly ignores "the design of the pilot", "his significant
contribution", "publicly commits in front of him". One known acceptable FP: "a signature
achievement in his career" (rare in conditions, and soft → patch).

---

## Patch 6 — Cost survivability

**Files:** `social_omni_epic/task_generator.py` (escalate block), `social_omni_epic/coherence_check.py` (check #7)

**Problem:** the escalate operator guarded condition-*satisfiability* but not outcome-
*reachability*. The Gourmet Delights scenario satisfied the rule to the letter — conditions
*were* met in 3 of 4 attempts — while `cost_coupling` made the *outcome* unreachable ("makes
hitting a 60% shift impossible," in its own words). The prompt protected the wrong invariant.

**Before** (escalate operator):
```
Do NOT make the scenario impossible: the movement_conditions must remain genuinely
satisfiable by a skilled, non-capitulating actor. Keep a zone of possible agreement.
```

**After** (added survivability clause):
```
CRITICAL — SURVIVABILITY: tightening cost_coupling makes the outcome HARDER or PARTIAL,
never STRICTLY UNREACHABLE. After satisfying the movement_conditions, a skilled actor must
still have a path to a meaningful version of their stated outcome. If satisfying the
conditions makes the learner's stated target literally impossible to reach (e.g. the goal
requires a 60% shift but the conditions cap any change at a tiny pilot), the scenario is
cost-stuck, not hard — that is a failure, not escalation.
```

Mirrored in coherence check #7 (COST COUPLING):
```
- SURVIVABILITY: also flag the OPPOSITE failure — if satisfying the movement_conditions
  makes the learner's stated outcome STRICTLY UNREACHABLE (not merely harder or partial),
  the scenario is cost-stuck, not hard.
```

---

## Patch 7 — Resistance-phrase variation

**File:** `social_omni_epic/episode_runner.py` (`_PARTNER_TURN_PROMPT_KEYED`, rule 2)

**Problem:** rule 2 supplied a verbatim example resistance line; the partner model latched
onto it as a script and repeated it ~8× identically, producing degenerate transcripts and
starving reflection of signal.

**Before:**
```
Express that resistance by restating your objection more firmly or shutting the topic down:
"I've already said what the issue is — I don't want to keep going in circles on this."
```

**After:**
```
Express that resistance by restating your objection more firmly or shutting the topic down —
but phrase it in your OWN words, varying your wording each time so you never repeat the same
sentence twice (e.g. as illustration only, something like "I've already explained where I stand
on this"). ... Just resist, naturally and in character.
```

---

## Verified clean — no change needed

Per the same review: the key check (temperature 0, fail-closed on error, runs every attempt,
head+tail truncation), the LP judge, MOI (fail-open is acceptable for a ranker), the
lateral/relax operator blocks, and the recency-placement + in-character-reframe design choices
in the keyed partner prompt.

## Files touched

| File | Patches |
|---|---|
| `social_omni_epic/episode_runner.py` | 1, 3, 7 |
| `social_omni_epic/task_generator.py` | 2, 4, 6 |
| `social_omni_epic/coherence_check.py` | 2, 5, 6 |

## Next step

A **probe run** with three counters before the full generation run:
1. condition-surfacing turn index (partner must not enumerate conditions in turn 1),
2. promise-scored-as-success rate,
3. artifact-mediated-win rate.

Clean probes license the full run, and the before/after gate-rejection statistics make a
credible methods-section appendix.
