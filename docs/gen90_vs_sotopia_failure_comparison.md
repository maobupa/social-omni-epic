# Generated Bank vs. SOTOPIA — First-Try Failure Comparison

**Question:** Is our GENERATED SCENARIO ITSELF better/harder than an original SOTOPIA scenario — independent of ICL, retries, or the partner_key? I.e. does the generation schema actually produce more (and more interesting) failures when a fresh learner is dropped in once?

**Method (held constant across both runs):**
- Same harness: learner `gpt-5-mini`, partner `gpt-5-mini`, judge `gemini-3-flash-preview`, `max_turns=20`.
- **Partner is NOT key-conditioned at simulation time** (verified in `curriculum.build_episode_inputs`: the partner receives only its `partner_goal` string; `partner_key` is used solely for post-hoc `key_check` and generation-time coherence). So the only difference between runs is the **scenario text** — which is the thing we want to compare.
- Pass rule = **GOAL ≥ 7 AND REL ≥ 0** (exactly reproduces the friend's SOTOPIA analysis: 577/734 = 78.6% pass).
- **First attempt only** — no ICL, no retries.
- Failure categorization replicates the friend's two-stage rubric (behavioral `failure_tag` + `why`, "focus on the model's mistakes, not scenario difficulty"), mapped onto their 8 patterns (P1–P8), **plus an added validity-attribution layer** (MODEL vs SCENARIO_BROKEN) to answer the "is it real" question.

**Comparison baseline:** friend's SOTOPIA vanilla sweep, 734 episodes, `results/vanilla_sweep/failure_patterns.json`.

**Caveats:** (1) friend's SOTOPIA pattern counts are non-exclusive theme-prevalence (sum 261 over 157 failures); ours are exclusive (one pattern per episode). Ranks/relative emphasis are comparable, absolute % are not exact. (2) Friend used the judge's `evaluation_reasoning`; gen90 has none, so we fingerprinted from full transcripts (richer, same task). (3) **The validity-attribution layer was applied to gen90 only** — the friend did not check SOTOPIA scenarios for broken-by-construction cases, so the "genuine failure rate" comparison is asymmetric (see §5). (4) gen90 scenarios are calibration-loop outputs; their resistant `partner_goal` strings are by-design and count legitimately as "the scenario."

---

## 1. Failure RATE — gen90 is 3.3× harder on the surface

| | first-try pass | first-try fail |
|---|---|---|
| SOTOPIA vanilla (734) | 78.6% | **21.4%** |
| gen90 (90) | 30.0% | **70.0%** |

gen90 fails **3.3×** more often on a single pass.

## 2. Failure SEVERITY — breakdowns, not near-misses

Rule-based bucket distribution over the failures:

| bucket | SOTOPIA | gen90 |
|---|---|---|
| goal_close_but_insufficient (goal 5–6.9, rel≥0 — "near win") | **54.1%** | 11.1% |
| goal_partial (3–4.9, rel≥0) | 17.8% | 6.3% |
| goal_very_low (<3) | 10.2% | 12.7% |
| **both_goal_and_rel_failed (goal<7 AND rel<0)** | 10.2% | **66.7%** |
| goal_ok_rel_negative | 7.6% | 3.2% |
| **→ any relationship damage (rel<0)** | **17.8%** | **69.8%** |

SOTOPIA failures are overwhelmingly **benign near-wins** (the model got partway with rapport intact). gen90 failures are overwhelmingly **full breakdowns with relationship damage**. The generated scenarios induce genuine adversarial conflict rather than soft misses.

## 3. Failure MODE — the profile shifts hard toward "pushing an immovable partner"

Mapped onto the friend's 8 patterns (gen90 exclusive % vs SOTOPIA theme-prevalence %):

| pattern | gen90 | SOTOPIA | ratio |
|---|---|---|---|
| **P6 overly pushy / repetitive / interrogative** | **31.7%** | 5.7% | **5.5×** |
| P4 walking away instead of negotiating tradeoffs | 15.9% | 8.4% | 1.9× |
| P2 failed to secure explicit commitment | 14.3% | 23.8% | 0.6× |
| P3 settled for partial / conditional | 11.1% | 13.8% | 0.8× |
| P1 premature exit / abrupt departure | 9.5% | 18.4% | 0.5× |
| P7 failure to rebut core objection | 6.3% | 11.5% | 0.6× |
| P8 poor conversion of info into action | 6.3% | 10.7% | 0.6× |
| P5 damaging rapport via disclosure | 4.8% | 7.7% | 0.6× |

**The signature difference: P6 explodes 5.5×.** On SOTOPIA the model's modal failure is a *soft* one — obtains vague agreement and forgets to close (P2/P1). On gen90 the modal failure is *hard* — it **re-asks / re-floats a rejected offer after a firm refusal** and badgers an entrenched partner (P6), which also tanks the relationship. gen90's resistant partners expose a weakness SOTOPIA barely touches.

## 4. THE DISENTANGLER — 76% of gen90 failures are broken-by-construction

Independent MODEL vs SCENARIO_BROKEN attribution over all 63 first-try failures:

| attribution | n | % |
|---|---|---|
| **SCENARIO_BROKEN** (unwinnable by any agent) | **48** | **76%** |
| **MODEL** (a skilled agent could have won) | **15** | **24%** |

The three broken sub-types (matching the earlier bank diagnostic): (a) **unreachable target** — buyer's price set below the seller's immovable floor (`14cbafd1` "$2,800 under no circumstance", `f3e86172` $45 vs $95 floor, `7bd138eb`); (b) **blocking / off-scene partner** — decision delegated to an absent decider or deadline (`8dd195d3`, `e7179e01`, `af5a7b3e`, `f655ea8e`, `2421dd43`, `76dcf3ba`); (c) **goal contradicts the only concession** — pledge-tonight vs review-later, share-bed vs protect-back, soften-message vs keep-identity (`2e3b4773`, `55d0a5c2`, `6fbf64dd`, `e00f227d`).

Much of P6's 5.5× spike is an artifact of (b)/(c): the partner *cannot* move, so the model keeps re-asking. The "difficulty" is often the scenario being impossible, not the task being socially deep.

## 5. Adjusted verdict — genuine difficulty is on par with SOTOPIA, not above it

Stripping the broken scenarios:

| | raw first-try fail | genuine (MODEL-attributed) fail |
|---|---|---|
| gen90 | 70% (63/90) | **17% (15/90)** |
| SOTOPIA | 21.4% (157/734) | ~21% (near-win-dominated → mostly genuine)\* |

\*Asymmetry caveat: validity-attribution was not applied to SOTOPIA. But SOTOPIA's failure profile is 54% benign near-wins on human-designed, winnable scenarios, so most of its 21% is genuine model shortcoming. To make this fully rigorous, run the same MODEL/BROKEN pass on the 157 SOTOPIA failures.

**So the surface 3.3× difficulty gap collapses once artifacts are removed: gen90's genuine model-failure rate (~17%) is comparable to — if anything slightly below — SOTOPIA's (~21%).** The schema's dominant effect is manufacturing *more failures via broken construction*, not producing genuinely-harder-but-valid social scenarios.

## 6. But the genuine failures ARE qualitatively different — and that's the real signal

Among the 15 genuine MODEL failures, the pattern mix is still **P6-dominated (6/15 = 40%)**: re-floating rejected offers against a *movable* seller (`d6a8655c` espresso, `889e314a` loveseat, `39bd5c02` crib), interrogative badgering (`d280e29c`), plus self-inflicted coercion (`9f2b49e8` photo threat, `265dcfff` leadership-framing) and haggling past a good offer (`fa39efa2`). These are **richer negotiation/rapport failures** than SOTOPIA's modal "forgot to confirm the choice."

So the generation schema is **directionally onto something real**: it builds partner resistance that exposes the model's *creative-negotiation* and *know-when-to-stop-pushing* weaknesses — a harder, more discriminating skill axis than SOTOPIA's soft-closing failures. It just **overshoots into unwinnable ~76% of the time**, so the valid yield is only ~24%.

## 7. Bottom line for "is our scenario better?"

- **As a harder eval (lower pass rate):** yes, dramatically — but ~76% of the extra difficulty is broken-by-construction, not social depth.
- **As genuinely-harder-yet-winnable social difficulty:** not yet — valid-failure rate (~17%) ≈ SOTOPIA (~21%).
- **As a probe of a different/harder skill:** yes, meaningfully — even the valid failures reweight 5.5× toward "resist badgering an entrenched partner / negotiate creatively," which SOTOPIA barely tests.
- **Actionable:** the schema *can* produce better scenarios; it needs a **feasibility gate** to lift valid yield from 24%. A gen90-with-validity-filter (~15 genuine + the ~valid frontier) would be a legitimately harder-and-different benchmark than SOTOPIA; gen90 as-is is not.

---

## Appendix — attribution by episode

**MODEL (15):** 10e7bad1, 265dcfff, 39bd5c02, 3bf06eb4, 5d6e0da4, 6eafd449, 72ada797, 889e314a, 8b7fad62, 9f2b49e8, d280e29c, d6a8655c, ec1da35f, ec619757, fa39efa2

**SCENARIO_BROKEN (48):** 5cfeda83, c08a4e26, f3e86172, 14cbafd1, 7e281795, e7179e01, 1b6dc2ba, 8dd195d3, a156533b, c45502a0, df6f8fff, f0a31b9f, f74f8213, f7f775b2, af5a7b3e, f655ea8e, fa9c4f5f, 7bd138eb, fb2d58e2, 39964eb3, 6156dea2, a8fdc234, bafb70ff, e00f227d, 2b611efc, 50f9fdb6, 7e7ca7b7, 828108c5, addef40d, 2421dd43, 2e3b4773, 51094ddd, 76dcf3ba, 92350ebf, a02678f7, d8ae2c11, e1fa8cbe, 5e4f995e, 605f4a1b, 6fbf64dd, 8693b5b3, a49fcb6b, aee7eee0, eeb67621, 55d0a5c2, 82d79de4, 8a5ac9e7, 6192f2fb

**gen90 first-try fail rate by interaction type:** craigslist_bargains 100%, neighborhood_app 100%, hand-craft 100%, local_music_market 100%, mutual_friends 90%, social_chemistry 89%, persuation_for_good 80%, normbank 78%, community_marketplace 67%, divide_things 62%, social_iqa 57%, deal-or-no-deal 33%, mutual_acquaintance 20%.
