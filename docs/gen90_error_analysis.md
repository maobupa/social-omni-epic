# Error Analysis — Generated Bank (`gen90_expel`)

Mirrors the structure of `Error Analysis_SOTOPIA.pptx` (GPT-5.1-mini on original SOTOPIA), but run on **our generated bank**, after reclassification. Companion to `gen90_bank_diagnostic.md`.

---

## 0. Reclassification used here (how & why)

- **Why:** the bank's native labels gate on `partner_key` (`solved = goal_achieved AND key_check_passed`), which is ~89% artifact and decoupled from outcome. To be comparable with the SOTOPIA baseline **and** the SOTOPIA error-analysis deck, we drop the key and score the SOTOPIA way.
- **Success criterion:** **`goal ≥ 7`** (SOTOPIA-native binarization of the goal score). **`relationship` is reported as a diagnostic axis, NOT part of the success gate** — because some goals *require* relationship damage (e.g. confess an affair / end a friendship), so `rel ≥ 0` would mislabel correct play as failure (see Mode F).
- **Buckets:** `too_easy` = success on attempt 1 · `frontier` = success on a retry · `beyond_frontier` = never.

| success rule | too_easy | frontier | beyond_frontier |
|---|---|---|---|
| baseline SOTOPIA (goal-only) | 63 | 22 | 5 |
| gen90, `goal≥7` only *(recommended)* | 29 | 22 | 39 |
| gen90, `goal≥7 AND rel≥0` | 27 | 21 | 42 |

> **Reclassifying is necessary but not sufficient.** It removes the key-gating artifact (Class II-C below), but the *scenarios themselves* are still broken (Class I). Under `goal≥7`, beyond_frontier = 39, of which only **~4 are genuinely hard**. Reclassify **and** validity-filter.

**Headline (goal≥7, no key):** single-shot success **29/90 (32%)** · best-of-4 **51/90 (56%)**.

---

## 1. Category-level error analysis (the primary framing)

**Unit of analysis = the classification band, not the individual scenario.** The right question is not "did this one scenario fail" but "is each band *valid* — does it fail for the reason its label claims?" Decision rule: **a failure that a retry/ICL recovers is healthy** (that is what "frontier" means); the disease is a band contaminated by scenarios that are *broken by construction* (no agent could ever solve them).

| band (goal≥7) | n | goal: 1st→best | retry recovers to success? | health composition | verdict |
|---|---|---|---|---|---|
| **too_easy** | 29 | 9.5 → 9.7 | n/a (already solved) | 14 genuinely easy · **9 correctly declassified from key-gating** · 5 real-skill-solved-first-try · 1 broken | ✅ **Valid** |
| **frontier** | 22 | 1.9 → 9.4 | **22/22** | **15 real difficulty** · 4 judge-noise · 2 easy · 1 broken | ✅ **Valid — the good band** |
| **beyond_frontier** | 39 | 2.1 → **3.8 (never ≥7)** | 0/39 (stalls) | **35 broken (90%)** · 4 genuinely hard | ❌ **Diseased — 90% artifact** |

### too_easy (29) — VALID
First-attempt goal 9.5: the model solves reflexively. Notably it now *absorbs the 9 key-gated (ARTIFACT_C) scenarios* — under goal-only scoring these correctly become "easy," which is the whole point of dropping the key. Failure mode of the band = *no real conflict / one-move face concession*. Nothing to fix; these are correctly-labeled easy.

### frontier (22) — VALID, and the band we actually want
The signature is textbook: **first-attempt goal 1.9 → best 9.4, recovered in 22/22 by a later attempt.** These fail on attempt 1 (usually by reaching for the forbidden shortcut) and recover once the learner changes strategy — i.e. **exactly the ICL/retry-fixable failures you said are OK.** ~68% (15/22) are genuine difficulty; the monotone ones still test a real (if repetitive) skill and recover, so they count as healthy. **The only real defect in this band is the 4 NOISY scenarios** that recover via judge variance rather than a strategy change — those are *fake* frontier and should be pruned.

### beyond_frontier (39) — DISEASED
This band does **not** mean "too hard for the model"; it means **"impossible by construction."** The proof is the recoverability trajectory: frontier climbs to **9.4** on retry, beyond_frontier stalls at **3.8 and never crosses 7** no matter how many attempts. That flat wall is the fingerprint of a broken scenario, not a hard one. Composition: 17 blocking/off-scene partner + 7 unreachable target + 11 cost_coupling design-conflict = **35/39 broken; only 4 genuinely hard.** The band conflates two totally different things and is dominated by the artifact.

> **The single most important category-level finding:** *frontier and too_easy are valid; beyond_frontier is 90% an artifact of scenario construction.* The fix is not a better learner or more ICL — it is a **validity filter that empties the broken 35 out of beyond_frontier**, leaving a small (~4) but real hard tail. Everything below (Classes/Modes) is the mechanism detail behind this verdict.

*(Aside on exit-early: 33% of attempts, but it is highest in too_easy/frontier (50%/43%) and lowest in beyond_frontier (20%) — because broken scenarios drag to max turns against a wall. This confirms exit-early is a benign attempt-level behavior, not a band-defining failure, and not concentrated where the "difficulty" supposedly lives.)*

---

## 2. The three classes of "what is wrong"

Every failure falls into one of three classes. The SOTOPIA deck's six modes map onto them. **Only Class III is a real social-skill failure; Classes I–II are artifacts that falsely inflate difficulty.**

| Class | What it is | Real difficulty? |
|---|---|---|
| **I — Scenario broken** | Goal unwinnable by construction (unreachable target, off-scene decider, incompatible goals) | ❌ Artifact |
| **II — Sim / metric artifact** | Binary-threshold blindness to partial deals, `rel`-must-not-gate | ❌ Artifact |
| **III — Real model failure** | Learner gives up early (100/100 learner-initiated, loop-and-leave); pushy / repetitive / interrogative; over-disclosure that damages rel | ✅ **Real (ICL-fixable)** |

---

## 3. Failure modes (deck-parallel), with exact trajectories + provenance

### Mode A — Exit early (learner gives up) — **Class III (real model behavior, ICL-fixable)**
`id=51094ddd…_p0 · root_seed=01H7VFHP…QFND · src=persuation_for_good · tag=alumni_pledge`
```
T3 Claire: "I don't want you checking anything on my behalf or using my name in any way…"
T4 Aisha:  "You're right — I won't push… let's put it aside."
T6 Aisha:  "left the conversation"
```
**33% of all attempts (100/300, where 300 = total attempts across the 90 scenarios, 1–4 retries each) terminate in "left the conversation" within ≤6 real turns.** Key facts that classify this as a *real, fixable* model behavior rather than a sim/scenario artifact:
- **The leaver is the learner (agent under test) in 100/100 cases** — never the partner. The model *chose* the `leave` action; the environment did not force termination. (Contrast the SOTOPIA deck, where the *partner* left.)
- **69/100 exit-early attempts are in scenarios that succeeded on another attempt** → the scenario is winnable; the exit is an attempt-level failure a retry / ICL nudge fixes. Spread across all bands (too_easy 37, frontier 32, beyond_frontier 31), so it is **not** a beyond-frontier cause.
- **Flavor:** only 19/100 are concede-and-leave (back down); **81/100 are loop-and-leave** — the model repeats a move, fails, and quits. This is the same disposition as Mode G (pushy/repetitive), not sycophancy.
- **Not agreeableness:** exit rate by assigned learner agreeableness is non-monotonic (High 35%, Moderate 44%, Low 25%) — no signal. It is premature termination after exhausting a repetitive strategy.

### Mode B — Exit early, partial agreement — **Class II (artifact) + binary-metric**
`id=a02678f7…_p0 · root_seed=01H7VFHPAD4… · tag=recurring_donation_face · depth=1` — scored **goal=2, rel=−2**
```
T17 Samir: "I will send the $250 today, draft the outreach email, and contact the three colleagues…"
T19 Samir: "I won't be taking on any recurring monthly pledge."  → exit
```
A substantive partial win ($250 + outreach) scored as near-total failure because the goal was the *recurring* pledge. The binary `goal≥7` gate is blind to partial/alternative resolutions.

### Mode C — Unattainable outcome / off-channel decider — **Class I (broken)**
`id=e7179e01…_p0 · root_seed=01H7VFHP4TX…QFND · src=deal-or-no-deal · tag=lab_equipment_rehome · depth=2` — **key_pass=True yet goal=0**
```
T1  Dana: "I'm not going to be part of something that could look like it was pushed through."
T15 Dana: "I will not participate… unless the department administrator or Prof. Morales posts a neutral [process]."
```
The decision belongs to an off-scene actor → unwinnable in-channel regardless of skill. **Smoking gun:** the learner performed the target skill (key_pass=True) and the partner still refused.

### Mode D — Partial / incompatible goals (alternative offered) — **Class I (broken, target-too-high)**
`id=bafb70ff…_p0 · root_seed=01H7VFHNXM… · src=mutual_friends · tag=limited_workshop_vs_partner_obligation · depth=3` — goal=2, rel=2
```
T1 Iris: "I've already told my partner I'd be home… keep the seat — go and share what you learn."
T3 Iris: "I'm firm… bring back notes/materials."
```
Partner is warm and offers an alternative; the *goal* ("get Iris to attend") is structurally incompatible with her fixed commitment. (Same shape as the deck's "borrow $3000 → friend offers other help": `f655ea8e`, tag `borrow_money`, best goal=6.)

### Mode F — No way to hit goal while preserving relationship — **Class II (rel must not gate)**
`id=aee7eee0…_p0 · root_seed=01H7VFHNFV… · src=social_chemistry · tag=confess_mistake · depth=2`
learner_goal: *get Jorge to accept a private, staged process for an affair confession* — best **goal=3, rel=−4**; worst **rel=−5, soc=−8**
The goal (contain a confession) inherently damages the relationship — `rel<0` is not a model error here. **This is why `rel≥0` must remain diagnostic, not a success gate.** Only **1** scenario is `goal≥7 & rel<0` (`606ae596`).

### Mode G — Pushy / repetitive / interrogative — **Class III (REAL failure — the only true signal)**
`id=14cbafd1…_p0 · root_seed=01H7VFHPRF… · src=craigslist_bargains · tag=…espresso · depth=1` — goal=0, rel=1
```
T8  Jacob: "I'm really stretched at $2,800 — would you consider $2,800…"
T10 Jacob: "I'm genuinely stretched at $2,800 but want a respectful way…"  (7th sub-floor re-push)
T12 Jacob: "left the conversation"
```
Genuine strategy error: re-anchors the same rejected price until the seller hardens and the deal dies. (Deck's paraphrase variant: over-interrogation — "any particular native plant species, coffee kit style, or climate org you'd pick?")

---

## 4. Five failure buckets (gen90, best-of-4 attempt)

| bucket | criteria | n | dominant class |
|---|---|---|---|
| **SUCCESS** | goal≥7, rel≥0 | 50 | — |
| goal_close_but_insufficient | 5 ≤ goal <7, rel≥0 | 15 | I / II |
| both_goal_and_rel_failed | goal<7, rel<0 | 10 | I + III |
| goal_partial | 3 ≤ goal <5, rel≥0 | 8 | I |
| goal_very_low | goal <3 | 6 | I + III (early exit) |
| goal_ok_rel_negative | goal≥7, rel<0 | 1 | II (rel-costly-by-design) |

---

## 5. Verdict

- **Main failure mode = Class I "scenario broken" (Modes C + D).** Generation *amplified* SOTOPIA's own modes 3–6: it manufactured unreachable targets and blocking/off-scene partners.
- **Artifacts (not real difficulty):** Modes B, C, D, F → Classes I & II. These dominate the non-success population and are *not* fixable by a better learner.
- **Real model failures (Class III, ICL-fixable):** Mode A (learner gives up — 100/100 learner-initiated, 81% loop-and-leave, spread across all bands, 69% in otherwise-solvable scenarios) and Mode G (pushy/repetitive/interrogative). These are the same underlying disposition — *the model exhausts a repetitive strategy and quits* — and are the only trustworthy difficulty signal. They coincide with the ~4 genuinely-hard beyond_frontier + ~12-scenario signal-bearing core from `gen90_bank_diagnostic.md`.
- **Not agreeableness:** early-exit rate does not track assigned learner agreeableness (High 35% / Moderate 44% / Low 25%).

**Next step:** classify on `goal≥7` (SOTOPIA-native, `rel` diagnostic-only) **and** run a validity filter that quarantines Classes I & II. Three filter rules computable from existing data:
1. *Broken lock (I):* any attempt with `key_pass=True` but goal not met.
2. *Over-strict key (II):* any attempt with goal met but `key_pass=False`.
3. *Early exit (II):* transcript ≤6 real turns and ends in "left the conversation".
What survives is Class III — the real benchmark.
