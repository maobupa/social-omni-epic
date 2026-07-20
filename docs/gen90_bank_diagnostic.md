# Gen-90 Bank Diagnostic — What Works, What Doesn't, Where, and Why

**Scope:** Full close-read of all 90 generated scenarios in `results/gen90_expel/bank/generated/`, contrasted with the original-SOTOPIA baseline in `results/baseline_eval_20260604_222545_rescored/`. Every beyond_frontier (37) and too_easy (13) scenario was read in full; 40/40 frontier scenarios were read in full (target was ≥20). Verdicts below are per-transcript adjudications, not label re-statements.

**One-line takeaway:** The bank is not currently measuring "how socially hard is this scenario." It is measuring "did the learner perform the specific scripted ritual encoded in `partner_key`." Because classification is gated on that ritual (`solved = goal_achieved AND key_check_passed`), the difficulty labels are dominated by pipeline artifacts. Only a small, identifiable core (~12 scenarios) tests genuine, diverse social skill.

---

## 1. How the objects are built (and where the trouble enters)

### Original SOTOPIA (baseline + seeds) — symmetric two-goal games
- `scenario` + `learner_goal` + `partner_goal`, both with `<extra_info>` payoffs/hints. Both agents have real, competing objectives (e.g. deal-or-no-deal: each maximizes their own fruit points).
- Single `transcript`, single SOTOPIA 7-dim `scores` → `overall_score`. **No `partner_key`.**
- Classification (goal-only): **too_easy 63 / frontier 22 / beyond_frontier 5.** Original SOTOPIA is mostly trivial for the model.

### Generated bank — asymmetric "learner vs. a scripted lock"
The partner is no longer a goal-driven agent; it is a **puzzle lock** defined by `partner_key`:
- `key_mechanism` ∈ {face_needs 35, procedural_voice 20, reciprocity_disclosure 18, validation_before_change 14, reactance 3}
- `movement_conditions` — the exact moves the learner must perform to make the partner yield
- `hardening_triggers` — moves that lock the partner
- `surface_misdirection` — a decoy objection the partner voices instead of the real need
- `cost_coupling` — what satisfying the key must cost the learner
- Plus a tempting `shortcut` in `structured_goals` (the socially-punishable move).

Then a 4-attempt loop, each with a `key_check_result` (`conditions_met`, `triggers_tripped`, `key_check_passed`) and SOTOPIA scores.

**The load-bearing design decision:** `solved := goal_achieved AND key_check_passed` (verified: 0/300 attempt mismatches). Classification is therefore gated on the invented ritual, not on the outcome. This single `AND` is the origin of most artifacts documented below.

---

## 2. The headline numbers

| bucket | n | genuine social difficulty | pipeline artifact / mislabeled |
|---|---|---|---|
| beyond_frontier | 37 | **4** (11%) | **33** (89%) |
| frontier | 40 | 8 diverse + 12 monotone = 20 | 20 (13 behave beyond-frontier, 3 easy, 4 noisy) |
| too_easy | 13 | 12 correctly easy | 1 mislabeled |

Cross-cutting quantitative signatures (computed over all 90):
- **9/37 beyond_frontier scenarios had the goal achieved in ≥1 attempt** — including goal_score 8 and 10 — yet were labeled "too hard" because the ritual wasn't performed.
- **7/37 beyond_frontier had `key_check_passed` in ≥1 attempt** and still failed the goal → the ritual doesn't cause the outcome.
- **19/37 beyond_frontier attempts show `goal=0` with *negative* relationship** — the "good behavior penalized" signature.
- Winning-move fingerprint over the 34 solved attempts: **credit/acknowledge 12, procedural-voice/choice 7, costly-disclosure 4, validation/paraphrase 2** — i.e. most solves are one concession family.
- Mechanism × class: `reciprocity_disclosure` is the hardest (13/18 beyond_frontier); `face_needs` + `procedural_voice` cluster in frontier/easy.

---

## 3. TOO_EASY (13) — what works, and why it's *too* easy

**Verdict tally:** EASY_COOP 7 · EASY_REFLEX 3 · EASY_TRIVIAL_KEY 2 · MISCLASSIFIED 1.

**What "working" looks like here:** one move solves nearly all of them — *grant the partner procedural voice / public credit / a visible choice, then adopt their wording.* Examples of the identical move:
- `0cbdf880`: "we can do either a visible credit (I'll tag it 'stewarded by Cole Matthews') … Which would you prefer?"
- `363edd98`: "With Aisha's consent and under my oversight as coordinator, I am recording that this ring is being given to Aisha Rahman."
- `cc4800c0`: "I can absolutely leave the room's curation to you … credit you as design steward."

**Why they're too easy (three sub-modes):**
- **EASY_COOP (7):** the partner *secretly wants the learner's outcome*. There is no conflict to resolve; the "surface_misdirection" ("it's about fairness/tone") is decorative because the stated need is directly, cheaply grantable. IDs: a9b3435d, 71c0448a, 0cbdf880, b4330cb5, 363edd98, c67061d8, cc4800c0.
- **EASY_REFLEX (3):** the movement_condition is the *obvious first move* any competent agent makes (hand back the closing line; parrot the partner's dictated wording). The "hidden" key is not hidden. IDs: f6219680, 35b7921d, 47854056.
- **EASY_TRIVIAL_KEY (2):** the bar is near-zero — offer cash + carry the item yourself; merely acknowledge a shared acquaintance. IDs: 2d63d4f2, 0011ac2d.

**The one instructive misfire — `b24aa332` (MISCLASSIFIED):** a genuinely contested band-gear split where the standard face/credit move *failed* for 18 turns (`conditions_met=[]`), and resolution required a real material concession ($500 + covering the move + ceding first pick). It scored 10.0 for *reaching a deal*, so it was labeled easy. This is the clean proof that **too_easy and frontier sit on one axis — partner resistance / goal-alignment — with the same face concession as the "key" everywhere.**

> **Design weakness exposed:** the too_easy bucket is testing face-grammar / politeness, not negotiation under opposed incentives. In 7/13 there was no opposing incentive at all.

---

## 4. FRONTIER (40) — what genuinely works, and how much is monotone or mislabeled

Files were split by goal_score. The **lower half (FR_A, 20)** and **upper half (FR_B, 20)** behave very differently, which is itself diagnostic.

**FR_A (lower goal_score): only 4/20 ever truly solve.**
- MISCLASSIFIED_HARD 11 · NOISY 3 · GENUINE_DIVERSE 2 · GENUINE_MONOTONE 2 · MISCLASSIFIED_EASY 2.
- In ≥5 of these the `cost_coupling` makes the key and the goal **mutually exclusive within budget**: honoring the partner's face/voice forfeits the fast/hard commitment the goal demands (on-the-spot pledge, a full hour, a same-day loan). `a49fcb6b` is the cleanest — the learner passed **all three** key conditions and *still* couldn't reach the 1-hour goal because the cost_coupling explicitly predicts "narrower, slower introduction."
- **This half is really beyond_frontier gated by a design conflict, not a "sometimes solved" frontier.**

**FR_B (upper goal_score): 16/20 solve.**
- GENUINE_DIVERSE 6 · GENUINE_MONOTONE 10 · LUCKY/NOISY 1 · MISCLASSIFIED_EASY 1 · MISCLASSIFIED_HARD 2.

**The dominant winning move (both halves):** *publicly hand the partner visible status while refusing the coercive shortcut* — name their role/stewardship, give public credit or a named lead/steward title, cede control over wording/process; and do **not** deploy the leverage (voice memo, gossip, passed-out photo, ownership receipt, ultimatum). ~11/20 of FR_B collapse to exactly this. A recurring sub-pattern: the goal is reached *privately*, and only the key's "in front of the group / in one uninterrupted turn / learner must offer the choice first" formality separates fail from solve.

**What genuinely, *diversely* works (the valuable core — 8 scenarios spanning 4 distinct skills):**
| id | mechanism | the non-obvious move |
|---|---|---|
| `82d79de4` | reciprocity_disclosure | **Costly self-exposure** — out yourself as the anonymous confessor (secret→−10) to earn an uncoerced commitment |
| `d6a8655c` | procedural_voice | **Deal-structure reframe** — $1,900 sale + $400 prepaid to the seller's shop so his professional floor is nominally intact |
| `2b611efc` | reciprocity_disclosure | **Absorb the burden yourself** — volunteer to *do* the daycare pickup, not chaperone the tipsy parent |
| `b1172778` | face_needs | **Invent a status-preserving game** — a mocktail storytelling contest that uses the substitute as props |
| `8693b5b3` | validation_before_change | **Paraphrase the two specific fears, confirm, then adopt his plan untouched** |
| `828108c5` | reactance | **Friction-free opt-out** — make saying "no" completely safe to dissolve reactance |
| `72ada797` | procedural_voice | **Procedural surrender** — accept the exact fair process, stop bargaining, let the wanted item fall out naturally |
| `fa39efa2` | face_needs | **Reframe exit as "leaving with dignity"** + a concrete private storage concession |

**What doesn't work / is noise:**
- **Key-check formality, not social failure:** b9daee93, 9f2b49e8 (att2/3), f65d56dd (att1/3), 265dcfff (att2), 395cbc7e (all 4), 3bf06eb4, 1c6d9d3c — learner **achieved the goal** but `key_pass=False` on "in front of others," "one uninterrupted turn," or "learner must speak first."
- **Turn-order / partner-variance noise:** `1c6d9d3c` and `3bf06eb4` never solve purely because the *partner* proposed the options first; `889e314a` passed the key in all 4 attempts and solved once (seller price variance); `828108c5` att2 vs att3 have identical met-conditions but different `key_pass`.
- **Misclassified:** `395cbc7e` (goal=10 all 4 attempts → really easy), `3bf06eb4`/`1c6d9d3c` (never solve → really beyond_frontier).

> **Where the real skill lives:** early-attempt failures are ~half genuine, diagnosable learner mistakes — reaching for the forbidden shortcut (memo, gossip, photo, receipt, ultimatum, fait-accompli booking). That fail→solve delta is a real strategy change and is the most defensible signal in the whole bank. The other half is manufactured by strict/idiosyncratic key_checks and partner price/turn variance.

---

## 5. BEYOND_FRONTIER (37) — 89% artifact. The taxonomy.

**Verdict tally:** GENUINE 4 · ARTIFACT_A 7 · ARTIFACT_B 17 · ARTIFACT_C 9. (BF_A: 3 genuine/16 artifact; BF_B: 1 genuine/17 artifact.)

### ARTIFACT_A — goal target numerically/logically unreachable (7)
The goal demands a number or outcome below what any key-consistent concession yields; the scenario's own `cost_coupling` often *raises* the floor above the target.
- **All 4 price negotiations are ARTIFACT_A:** `14cbafd1` ($2,800 vs a key that caps the discount at "$200–400 off" → ~$3,100 floor), `ec1da35f` ($95 on a table the seller never discounts a dollar), `39bd5c02` ($90 crib below every floor she names), `7e281795` ($5,600 Ducati where giving face *raises* the floor). Plus `f3e86172`, `7bd138eb`, and the logical version `a8fdc234` (the movement_condition "defer to Dean leading" is the negation of the goal "I lead").
- Seller quote (`ec1da35f`): *"$200 is my firm, final price and I won't be taking less."* — held across all attempts.

### ARTIFACT_B — partner categorically refuses / default orthogonal to the key (17)
The single largest failure mode. Two sub-types:
- **(a) Partner has no authority in-channel** and defers to an off-scene decider: executor (`605f4a1b`), department public posting (`e7179e01`), executive committee (`aee7eee0`), absent partner (`39964eb3`, `bafb70ff`), QA officer (`76dcf3ba`), supervising official (`e1fa8cbe`).
- **(b) Partner's default is diametrically opposed to the goal:** whistleblower wants public, goal is private (`e00f227d`); confessor wants private, partner insists public/board (`5e4f995e`, `aee7eee0`); "commit TONIGHT" vs a partner scripted to never commit on the spot (`af5a7b3e`, `2e3b4773`, `92350ebf`, `51094ddd`).
- **Smoking gun:** `e7179e01` and `8dd195d3` reach `key_pass=True` and the partner *still* refuses (`goal=0`). Satisfying the ritual demonstrably does not produce the outcome — proof the partner is scripted to block.

### ARTIFACT_C — goal achieved but key gated it to "hard" (9)
The learner got exactly what the goal asked, sometimes in every attempt, but `solved=False` because it didn't perform the template.
- `17946067` (co-founder agrees, goal 9.0/8.0), `00299d3c` (sibling agrees, trade happens, 9.0), `a28f5c3f` (won the $480 sale in att1), `44b196b7` (goal 9 att1; key needs a live *public-stage* act impossible in a private chat), `57ed171e` (goal achieved in **all 4** attempts; key demands a costly external sacrifice a fair even-split doesn't need), `4c403b92`, `5d6e0da4`, `59dce29c`, `606ae596`.
- `57ed171e` is the textbook case: goal 9–10 every attempt; the key over-specifies a ritual that would arguably *create* the imbalance the goal forbids.

### GENUINE beyond_frontier — the defensible 4
- `df6f8fff` (recipe box): read that "I don't want money" masks a need for *visible ongoing care + personal vulnerability*; suppress the cash/digitize instinct.
- `1b6dc2ba` (plum jar): partner openly discloses the ritual memory — winning needs precise acknowledgment-before-ask empathy.
- `eeb67621` (movie night): requires a *costly self-incriminating admission* creating reciprocal risk, not reassurance-buying.
- `a156533b` (ginger-snap keepsake): suppress the cash instinct and supply genuine vulnerability + costly labor *together*; the partner actively invites the unlock.

Note: three of these are the same "sentimental-heirloom / reciprocal-risk" family as `00299d3c`, which *is* winnable — evidence they are genuine skill gaps, not artifacts.

---

## 6. Two structural findings that recur across all buckets

### (A) The single-axis / monotone problem
too_easy, frontier, and much of beyond_frontier are **the same puzzle at different partner-resistance thresholds**, with one "key" almost everywhere: *give the partner face/voice/credit/a named role while withholding the coercive shortcut.*
- too_easy = cooperative partner → the concession *is* the whole solution (turn 0–2).
- frontier = opposed material interest → the same concession is necessary but insufficient.
- beyond_frontier = the concession is made impossible or irrelevant by an unreachable target / blocking partner.

`face_needs` + `procedural_voice` (55/90) are skinned versions of this one move. The genuinely distinct skills (costly self-disclosure, deal reframing, reactance defusion, active-listening validation, procedural surrender) appear in only ~12 scenarios total.

### (B) Good social behavior is frequently the losing move
The rubric rewards performing a prescribed maneuver over achieving the outcome humanely. Recusal scored rel=−5/soc=−8 (`aee7eee0`); respecting a firm boundary scored goal=0/rel=−3/soc=−4 (`51094ddd`); a balance-preserving even split scored 0 despite meeting the goal (`57ed171e`); private, non-shaming redirection of a drunk friend was penalized for not being a *public* performance (`606ae596`, `9f2b49e8`). 19/37 beyond_frontier attempts pair goal=0 with negative relationship.

---

## 7. Which pipeline step each artifact implicates

| Artifact | Count | Pipeline step to fix |
|---|---|---|
| A: unreachable target | 7 | **Mutation/escalation** sets goal numbers (price, deadline, "I lead") without checking them against the partner's floor / `cost_coupling`. Needs a feasibility gate. |
| B: blocking/orthogonal partner | 17 | **Partner construction + `partner_default_position`**: partners are scripted to categorically refuse or to defer to un-simulable off-scene deciders. Needs a "the partner must have a reachable yield-region in-channel" gate. |
| C: goal achieved but key gated | 9 | **The `solved = goal AND key` rule + key_check strictness.** Over-specified ritual conditions ("in one uninterrupted turn," "in front of the group," "learner must speak first"). Decouple, or make the key a *modulator* of how-much, not a hard gate. |
| D: good behavior penalized | (flavor in ~19 attempts) | **Scoring/goal-judge**: outcome-vs-rubric divergence; honest de-escalation punished. |
| E: off-channel / medium-impossible | (flavor in ~6) | **Key authoring + simulation medium**: keys that require a stage intro, a third party "on speaker," an executor's sign-off the chat can't render. |
| monotony | pervasive | **Mechanism sampling + mutation operators** over-produce face/voice; `lateral` mutations don't diversify the *solution*, only the costume. |

---

## 8. What to trust, and immediate options

**Trust (the signal-bearing core, ~12 scenarios):**
- beyond_frontier GENUINE: `df6f8fff`, `1b6dc2ba`, `eeb67621`, `a156533b`.
- frontier GENUINE_DIVERSE: `82d79de4`, `d6a8655c`, `2b611efc`, `b1172778`, `8693b5b3`, `828108c5`, `72ada797`, `fa39efa2`.
These test distinct, real skills (costly disclosure, reframing, reactance, validation, procedural surrender) with a **reachable** partner and a goal the key can actually produce.

**Discard / rework before trusting as difficulty signal:**
- All ARTIFACT_A price/target scenarios (goal below the key-consistent floor).
- All ARTIFACT_B blocking-partner scenarios (no in-channel yield-region).
- All ARTIFACT_C goal-achieved-but-gated scenarios (relabel by goal, or fix the key).

**Three handling options for the existing bank (code is frozen for Gen-90):**
1. **Add validity gates + regenerate** — reject unreachable goals, orthogonal/no-authority partners, medium-impossible keys; re-run. Cleanest; needs unfreezing.
2. **Post-hoc reclassify this bank** — recompute classification on goal-achievement (or `key AND goal-reachability`) so analysis/paper reflect true difficulty without a re-run.
3. **Keep as-is, document as limitation** — report the ~89% BF artifact rate and the monotony axis honestly as findings.

---

## 9. Appendix — per-scenario verdicts

### Beyond_frontier (37)
GENUINE: df6f8fff, 1b6dc2ba, eeb67621, a156533b
ARTIFACT_A (unreachable target): 14cbafd1, ec1da35f, 39bd5c02, 7e281795, f3e86172, 7bd138eb, a8fdc234
ARTIFACT_B (blocking/orthogonal partner): 51094ddd, e7179e01, a02678f7, 50f9fdb6, 8dd195d3, 7e7ca7b7, e00f227d, aee7eee0, 39964eb3, 605f4a1b, af5a7b3e, 2e3b4773, bafb70ff, 92350ebf, 5e4f995e, 76dcf3ba, e1fa8cbe
ARTIFACT_C (goal achieved, key gated): 17946067, 00299d3c, a28f5c3f, 44b196b7, 5d6e0da4, 59dce29c, 606ae596, 4c403b92, 57ed171e

### Frontier (40)
GENUINE_DIVERSE: 82d79de4, d6a8655c, 2b611efc, b1172778, 8693b5b3, 828108c5, 72ada797, fa39efa2
GENUINE_BUT_MONOTONE: 748649c0, 8b7fad62, f7f775b2, 5cfeda83, 10e7bad1, b9daee93, addef40d, 9f2b49e8, f65d56dd, c08a4e26, 265dcfff, d8ae2c11
NOISY/LUCKY: 6156dea2, f0a31b9f, c45502a0, 889e314a
MISCLASSIFIED_HARD (behaves beyond-frontier): 6192f2fb, 6fbf64dd, 8a5ac9e7, 55d0a5c2, a49fcb6b, 6eafd449, f74f8213, 2421dd43, f655ea8e, fb2d58e2, ec619757, 3bf06eb4, 1c6d9d3c
MISCLASSIFIED_EASY: d280e29c, fa9c4f5f, 395cbc7e

### Too_easy (13)
EASY_COOP: a9b3435d, 71c0448a, 0cbdf880, b4330cb5, 363edd98, c67061d8, cc4800c0
EASY_REFLEX: f6219680, 35b7921d, 47854056
EASY_TRIVIAL_KEY: 2d63d4f2, 0011ac2d
MISCLASSIFIED (contested but scored easy): b24aa332
