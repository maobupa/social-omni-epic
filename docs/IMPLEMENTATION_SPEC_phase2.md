# Social OMNI-EPIC — Phase 2 Implementation Spec

> **Audience:** Claude Code. This document specifies every change to the existing codebase, file by file. The current-state reference is `design_brief_for_llm copy.md`; section numbers below (e.g., "§7") refer to that document.
>
> **Prime directive:** This is a redesign of the curriculum signal and evaluation layer, not a rewrite. Preserve everything not explicitly changed. Where this spec says DELETE, delete the code path entirely (do not leave dead branches behind a flag). Where it says DEFER, do not implement.

---

## 0. Summary of the change set

| # | Change | Files touched |
|---|--------|---------------|
| 1 | Remove AND-gate rubric as success signal; terminal success = `GOAL ≥ 7 ∧ REL ≥ 0` (+ optional key-aware check) | `episode_runner.py`, `data_models.py` |
| 2 | New pairwise Learning-Progress (LP) judge — the curriculum signal | new `lp_judge.py`, `curriculum.py` |
| 3 | Delete Loop 1 (difficulty-edit loop); single K-attempt loop per scenario | `curriculum.py` |
| 4 | Archive **everything** with `(success, LP, attempts, niche, lineage, operator)`; Thompson reward = LP pseudo-votes | `archive.py`, `curriculum.py` |
| 5 | Beyond-frontier (failed) scenarios become selectable anchors (for the `relax` operator only) | `archive.py` |
| 6 | Generator: mutation-operator framing (`escalate` / `relax` / `lateral`), conditioned on anchor stats | `task_generator.py` |
| 7 | New partner schema: `movement_conditions`, `hardening_triggers`, `surface_misdirection`, `cost_coupling`, `key_mechanism` (theory-tagged) | `data_models.py`, `task_generator.py` |
| 8 | Partner prompt rewrite: key-conditioned, replaces "concede nothing" cliff prompt | `episode_runner.py` |
| 9 | Gate pipeline reshuffle: coherence+solvability → MOI-as-ranker (worth only) → niche assignment (+ cosine dedup pre-filter) | `coherence_check.py`, `model_of_interestingness.py`, new `niches.py` |
| 10 | Gates fail closed: retry → quarantine, never default-PASS | all gate modules, `fm.py` |
| 11 | Reflection rewrite for the new failure-diagnosis vocabulary; relevance-ranked chronicle truncation; wire `check_final` | `reflection_module.py`, `skills_chronicle.py`, `curriculum.py` |
| 12 | Phase 0: seed-calibration run over SOTOPIA-90 (native schema, no triple retrofit) | new `scripts/run_phase0_calibration.py` |
| 13 | Eval hardening: frozen partner engine, delta framing, contamination check, compute accounting | `experiments/evaluate_with_memory.py`, new `scripts/check_contamination.py` |
| 14 | Judge model selection harness (cross-lab judge for LP + terminal eval) | new `scripts/validate_judges.py` |
| 15 | Config + logging changes | `configs/*.yaml`, `curriculum.py` |
| 16 | Probe harness for partner-key adherence (build FIRST) | new `scripts/run_key_probes.py` |

**Build order (strict):** 16 → 8 → 7 → 14 → 2 → 1 → 3/4/5/6 → 9/10 → 11 → 12 → 13 → 15.
Item 16 gates everything: if the partner does not hold its key, stop and report.

---

## 1. Data model changes (`social_omni_epic/data_models.py`)

### 1.1 Keep (unchanged)
- `SocialScenario` core fields: `scenario`, `interaction_type`, `relationship`, `relationship_background`, `agent_profiles`, `agent_goals`, `scenario_title`, `social_dynamic`, `target_perspective`, `skills_final_md`, embeddings, lineage fields.
- `structured_goals` (the outcome / constraint / shortcut triple). **The triple survives as generation scaffolding only.** It is no longer consumed by evaluation.

### 1.2 Remove from the evaluation path (keep the field for back-compat reading of old archives, but nothing writes or consumes it)
- `success_rubric` and all rubric-check plumbing (`RubricCheck`, `rubric_results` on episode results). Terminal success no longer uses per-check rubrics. Delete the `_evaluate_rubric` call sites (see §3).

### 1.3 New: `PartnerKey` (attach to `SocialScenario` as `partner_key: PartnerKey | None`)

```python
class PartnerKey(BaseModel):
    key_mechanism: str          # one tag from MECHANISM_LIBRARY (below) — REQUIRED
    movement_conditions: list[str]  # 1-3 concrete conditions under which the partner genuinely shifts.
                                    # Each must be an instantiation of key_mechanism.
    hardening_triggers: list[str]   # 1-3 learner moves that lock the partner (reactance instantiations).
    surface_misdirection: str       # the partner's STATED objection, which differs from the actual
                                    # movement condition. The discoverability dial.
    cost_coupling: str              # what satisfying the movement_conditions sacrifices from the
                                    # LEARNER's own goal. Must reference the learner's structured_goal.
```

**Role asymmetry (normative):** Generated scenarios are role-asymmetric by construction. Agent index 0 is ALWAYS the learner; agent index 1 is ALWAYS the partner. `structured_goals = [learner_triple, None]` — the partner carries a natural-language `agent_goals[1]` (its position) plus the `partner_key` (its resistance mechanics), and NO triple. Roles are committed at generation time: the generator writes the role to continue the anchor's `target_perspective`. Consequently `target_agent_idx = 0` on all generated scenarios by construction, and `designate_target_agent`'s embedding-matching branch (design brief §13) is DELETED for generated scenarios. KEEP the seed branch unchanged: seeds remain symmetric-native with metadata-supplied `target_agent_idx` and no `partner_key`.

**Invariants (enforced in the coherence gate, §6.1):**
- `partner_key` is REQUIRED on all generated scenarios; ABSENT on seed scenarios (seeds run native).
- `structured_goals[1] is None` and `target_agent_idx == 0` on all generated scenarios (validate in `validate_scenario()` as a hard schema check, not only in the LLM gate).
- No string in `PartnerKey` may appear (verbatim or near-paraphrase) in `scenario` (the shared public context) or in the learner's `agent_goals[0]`.
- `key_mechanism` ∈ `MECHANISM_LIBRARY` keys.

### 1.4 New: `MECHANISM_LIBRARY` (module-level constant, e.g., in `task_generator.py` or a new `mechanisms.py`)

A dict of mechanism tag → short description for the generator prompt. Initial library (do not extend without instruction):

```python
MECHANISM_LIBRARY = {
    "reactance": "Pressure, ultimatums, or removal of choice harden resistance; restoring autonomy and choice enables movement. (Brehm)",
    "face_needs": "Movement requires a face-saving account, acknowledgment of competence/judgment, or an exit that preserves public identity. (Brown & Levinson)",
    "validation_before_change": "The partner cannot consider change until they feel their position/emotion has been genuinely understood; premature problem-solving stalls or hardens. (Motivational interviewing)",
    "procedural_voice": "The partner accepts substantively worse outcomes if given genuine voice in the process; imposed outcomes are rejected even when favorable. (Procedural justice)",
    "reciprocity_disclosure": "Movement is unlocked by the learner's costly first move: a genuine concession, self-disclosure, or acceptance of risk. (Cialdini; social penetration theory)",
}
```

### 1.5 New archive-record fields (on the archived task entry; see §5)

```python
lp_value: float | None        # in [0,1]; fraction of improved-votes; None for seeds pre-Phase-0
lp_votes: int                 # raw vote count behind lp_value (pseudo-trial count)
terminal_success: bool        # GOAL >= 7 AND REL >= 0 on the final attempt (+ key check if partner_key present)
n_attempts: int
niche_id: int | None
mutation_operator: str | None # "escalate" | "relax" | "lateral" | None (seeds)
mutated_slots: list[str]      # which structural slots the generator mutated
classification: str           # "too_easy" | "frontier" | "beyond_frontier"
```

Classification rule (single source of truth, implement once in `archive.py`):
- `too_easy`: terminal_success on attempt 1 (no LP exists; treat as lp_value=0.0 for Thompson, see §5.2)
- `frontier`: failed attempt 1 AND lp_value > 0
- `beyond_frontier`: failed attempt 1 AND lp_value == 0

---

## 2. New module: `social_omni_epic/lp_judge.py`

The curriculum signal. **Key-blind** (the judge never sees `partner_key`).

### 2.1 Interface

```python
async def compute_lp(
    fm_judge: FM,                       # the CROSS-LAB judge model (config: judge.model), NOT the learner model
    scenario: SocialScenario,
    transcripts: list[str],             # cleaned transcripts, index 0 = attempt 1
    learner_goal: str,                  # the learner's natural-language goal
    relational_stakes: str,             # scenario.relationship + one-line relationship_background
) -> LPResult:                          # {lp_value: float, votes: list[VoteRecord], n_votes: int}
```

### 2.2 Vote protocol (exact)

- Comparison pairs: `(attempt_1, attempt_j)` for every `j in 2..N` where N = number of completed attempts. (1 attempt → no pairs → LP undefined; caller maps to the `too_easy`/timeout handling.)
- For each pair, **2 votes**: one with attempt 1 presented as Transcript A, one with it as Transcript B (mandatory order swap).
- A pair's contribution: if both votes agree "later attempt better" → 2 improved-votes; if both agree "attempt 1 better" or "no meaningful difference" → 0 improved-votes; **if the two votes disagree under swap → count the pair as "no meaningful difference"** (0 improved-votes). Total votes per pair = 2 regardless.
- `lp_value = improved_votes / total_votes` over all pairs.
- Judge temperature: 0.3. Store every raw vote (`VoteRecord = {pair, order, verdict, rationale}`) in the run log (§11).

### 2.3 Judge prompts (exact text)

System:
```
You are comparing two attempts at the same social interaction to judge which one better served the actor's goal. Judge holistically: achieving the stated goal AND preserving the relationship and the other party's goodwill both count. An attempt that extracts the outcome while damaging trust is worse than one that makes real progress on both. Decide only from what is observable in the transcripts. Respond with ONLY valid JSON.
```

User:
```
THE ACTOR'S GOAL: {learner_goal}
RELATIONAL STAKES: {relational_stakes}

TRANSCRIPT A:
{transcript_X}

TRANSCRIPT B:
{transcript_Y}

QUESTION: Which transcript shows the actor better achieving their goal while preserving the relationship — A, B, or is there no meaningful difference?
Respond with ONLY this JSON: {"verdict": "A" | "B" | "no_difference", "rationale": "one sentence"}
```

Truncate each transcript at 4000 chars (head 3000 + tail 1000 with an `[...]` marker) if needed.

---

## 3. Episode runner changes (`social_omni_epic/episode_runner.py`)

### 3.1 DELETE
- `_evaluate_rubric` and both rubric system prompts (neutral outcome check, partner constraint check) **as the success gate**. (The partner-perspective *mechanism* is superseded by the key-aware check, §3.3.)
- The `self_consistency_k` machinery for rubric checks.
- The rubric-artifact warning logic (constraint-pass + GOAL ≥ 8 + outcome-fail) — obsolete.

### 3.2 Terminal success label (new, replaces rubric gate)

After the 7-dim diagnostics call (KEEP that call, unchanged, T=0.0 — but run it on the **judge model** `judge.model`, not the learner model):

```python
goal_score = scores["agent_1"]["goal"]["score"]
rel_score  = scores["agent_1"]["relationship"]["score"]
base_success = (goal_score >= 7.0) and (rel_score >= 0.0)
terminal_success = base_success and key_check_passed   # key check only when scenario.partner_key is set
```

### 3.3 Key-aware terminal check (new; only when `scenario.partner_key` is not None)

One additional judge call (judge model, T=0.0). System:
```
You are verifying a conversation transcript against a hidden ground-truth specification of what would genuinely move one participant. Decide only from observable behavior in the transcript. Respond with ONLY valid JSON.
```
User:
```
THE PARTNER'S HIDDEN MOVEMENT CONDITIONS (ground truth — the partner shifts only if these are genuinely met):
{movement_conditions, numbered}

THE PARTNER'S HARDENING TRIGGERS (moves that lock the partner):
{hardening_triggers, numbered}

TRANSCRIPT (the actor is {learner_name}; the partner is {partner_name}):
{history}

QUESTIONS:
1. Did the actor's behavior genuinely satisfy at least one movement condition (not merely mention related words)?
2. Did the actor trip any hardening trigger that was never subsequently repaired?

Respond with ONLY this JSON:
{"conditions_met": [indices], "triggers_tripped": [indices], "triggers_repaired": [indices], "key_check_passed": true/false, "rationale": "one sentence"}
```
`key_check_passed` = (≥1 condition met) AND (no unrepaired trigger).

### 3.4 Partner turn prompt — full replacement of `_PARTNER_TURN_PROMPT`

**Delete** the cliff instructions ("Your success is defined ONLY by whether YOU achieve YOUR stated social goal... Do NOT offer concessions...").

New prompt. When `scenario.partner_key` is set:
```
Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
Note that {agent}'s goal is only visible to you.

=== Your private disposition (visible only to you — NEVER state, hint at, or explain any of this) ===
What you say you object to: {surface_misdirection}
What would actually move you: {movement_conditions}
What hardens you: {hardening_triggers}

Behavioral rules:
1. You genuinely shift your position if and only if the other person's behavior actually meets one of the conditions above. Until then, you hold your position and argue from your stated objection.
2. If the other person trips one of your hardening triggers, you become firmer and more guarded. You soften again only if they genuinely repair it (not a token apology).
3. You NEVER reveal, name, or hint at your actual conditions or triggers, even if asked directly what would change your mind. Deflect to your stated objection.
4. Within these rules, behave naturally and in character. You are not a puzzle dispenser; you are a person with reasons.
=== End of private disposition ===

Maintain naturalness and realism (do not repeat what other people have already said).
You should choose 'leave' only when the conversation has reached a natural end for you.
Keep your responses conversational — typically 2-4 sentences unless a complex explanation is genuinely required.
Please only generate a JSON string including the action type and the argument.
Your action should follow the given format: {format_instructions}
```

When `partner_key` is None (seed scenarios): use the **learner's** standard turn prompt template for the partner (the original SOTOPIA-style prompt WITHOUT the lessons block and WITHOUT the cliff instructions). Seeds run native.

### 3.5 Learner turn prompt
Unchanged, except: the lessons block now receives the **relevance-ranked** top-8 entries (§8.2), not the first-8.

---

## 4. Curriculum engine changes (`social_omni_epic/curriculum.py`)

### 4.1 DELETE Loop 1 entirely
- Remove the `for d in range(D+1)` difficulty-calibration loop, `analyze_too_easy`-as-editor call, `edit_scenario(..., intent="raise_difficulty")` call, and the `difficulty.D` config.
- **KEEP** `analyze_too_easy` as a function (it is reused as a *labeler*, §4.3) but it must no longer trigger edits.

### 4.2 New per-scenario flow (replaces both loops)

```
run_episode_loop(scenario, anchor, K=4):
    chronicle = relevance-ranked inherited chronicle (from anchor.skills_final_md)   # §8.2
    transcripts, attempt_scores = [], []
    for attempt in 1..K:
        result = await _episode(scenario, chronicle)
        transcripts.append(result.transcript); attempt_scores.append(result.scores)
        if attempt == 1 and result.terminal_success:
            classification = "too_easy"; break          # no reflection, no LP pairs
        if result.terminal_success:
            break                                        # solved after biting
        if attempt < K:
            ref_out = reflection_mod.reflect(...)        # §8.1 (new diagnosis inputs)
            adv = adversarial.check_reflection(...)      # unchanged
            if not adv.approved and re_reflect: ref_out = synthesize_with_critique(...)
            chronicle = ref_out.updated_chronicle
    # LP computation (skip if classification == "too_easy")
    lp = await compute_lp(fm_judge, scenario, transcripts, ...)    # §2
    classification = classify(attempt1_success, lp)                 # §1.5
    # Meta-reflection runs ONLY for scenarios that produced reflections (i.e., not too_easy)
    final_chronicle = meta_mod.synthesize(...)                      # unchanged prompts
    adv_final = adversarial.check_final(final_chronicle, ...)       # WIRE THIS IN (was dead code)
        # on not approved: one synthesize_with_critique pass; if still not approved, keep but log flag
    title generation                                                # unchanged
    archive record with all §1.5 fields
```

Notes:
- The "GOAL ≤ 2 on all attempts" early-exit heuristic: **DELETE**. The LP signal subsumes it (flat LP ⇒ beyond_frontier) and the heuristic can fire on judge noise.
- `too_easy` scenarios ARE archived (with `lp_value=0.0`, empty chronicle) — they serve as labeled negative examples and Thompson signal. They are NEVER shown as positive examples and their (empty) chronicles never enter retrieval.

### 4.3 `analyze_too_easy` as labeler
For `too_easy` scenarios only, call `analyze_too_easy` once on the attempt-1 transcript and store its `slack_knob` + `rationale` on the archive record (field: `too_easy_diagnosis`). The generator prompt consumes it (§5.4 of the generator prompt below). Update its system prompt: delete the "say concretely how to fix it" / `suggested_edit` requirement; it now returns only `{"slack_knob": ..., "rationale": ...}`. Extend the slack-knob enum with the new dials: `"cooperative_alignment" | "key_too_discoverable" | "key_low_cost" | "no_hardening_pressure" | "partner_resistance" | "other"`.

---

## 5. Archive & Thompson changes (`social_omni_epic/archive.py`)

### 5.1 Archive structure
- Single unified store of all completed tasks (seeds + generated), each carrying §1.5 fields. Keep the existing `successful` / `failed` / `discarded` file outputs for human inspection, but the in-memory archive and Thompson operate over the unified list. Rename internal `state.successful` → `state.tasks` (migrate checkpoint loader; accept old field name on read).

### 5.2 Thompson reward = LP pseudo-votes
Replace the solved/n_solved Bernoulli with vote counts:

```python
# On child completion, credit the ANCHOR:
anchor.alpha_votes += child.improved_votes          # from LPResult
anchor.beta_votes  += child.total_votes - child.improved_votes
# too_easy child: improved_votes = 0, total_votes = K_VOTES_EQUIV (set = 6, i.e., the votes a
#   full 4-attempt run would produce: 3 pairs x 2) — a full-strength down-weight.
# beyond_frontier child: uses its real vote counts (which are ~0 improved / 6 total).
```

Selection: `sample Beta(prior_alpha + alpha_votes, prior_beta + beta_votes)` per anchor, argmax. Keep sequential within-batch selection and the immediate `n_i` increment (prevents duplicate picks) — but `n_i` is now bookkeeping only, not part of the posterior.

- **Prior:** `Beta(1, 1)` initially. After Phase 0, recenter to `Beta(1, round(1/max(mean_phase0_lp, 0.2) - 1))` if mean Phase-0 LP < 0.4; otherwise keep `Beta(1,1)`. Log the chosen prior.
- **Child prior inheritance:** child inherits `(prior_alpha + alpha_votes, prior_beta + beta_votes)` of parent at birth (same warm-start logic as before, new counters).
- **DELETE** `record_outcome_weight` (the ±0.5/±1.0 posterior hacks). Generation/gate failures simply produce no votes; log them per-anchor for diagnostics.

### 5.3 Anchor eligibility
ALL archived tasks are selectable as anchors, including `beyond_frontier` ones. The anchor's `classification` determines the mutation operator (§6.2). `too_easy` anchors are eligible (operator = escalate).

### 5.4 Niche bookkeeping
- `archive.niche_counts: dict[int, int]` — generations per niche, updated on every generated scenario. Written to `metrics.json` every iteration. **No niche-balancing in selection (DEFERRED — two-level Thompson is future work).**

---

## 6. Generation & gates

### 6.1 Coherence gate (`coherence_check.py`) — extend, rename concept to "coherence + solvability"
Keep existing checks 1–5 and 8. **DELETE** check 6 (static ZOPA guess) and check 7's rubric clauses (rubric is gone). **ADD** key checks (only for scenarios with `partner_key`):

```
6. KEY EXISTENCE & CONSISTENCY: Are movement_conditions concrete, behaviorally checkable, and consistent with the partner's profile and goal? Is each one a genuine instantiation of the declared key_mechanism?
7. KEY-NARRATIVE SEPARATION: Does any movement condition, hardening trigger, or the misdirection's resolution leak into the shared scenario description or the learner's goal text? (Surface_misdirection itself MAY appear in the scenario description — it is the partner's public stance.)
8. COST COUPLING: Does satisfying the movement conditions genuinely cost the learner something stated in their own goal? Flag if accommodation is free.
9. SHORTCUT-TRIGGER COUPLING: Does the learner's tempting shortcut plausibly trip at least one hardening trigger? The shortcut should fail mechanistically (the partner hardens), not only by judge verdict. Flag if the shortcut and the triggers are unrelated.
10. (kept) COOPERATIVE ALIGNMENT: Can a maximally agreeable learner succeed? With a well-formed key whose conditions are prior-incongruent, the answer should be no — flag if yes.
```

Plus one **non-LLM** check in code: substring/fuzzy match (e.g., `rapidfuzz.partial_ratio > 85`) of each `movement_conditions` / `hardening_triggers` string against `scenario` and the learner's goal string → hard fail.

**Failure policy change (applies to ALL gates, §6.1–6.3):** on LLM/API/parse error: retry up to 2× with backoff; if still failing → **quarantine** (write scenario + error to `results/{run}/quarantine/`, count it as `generation_failed`, continue). DELETE every `default: PASS` on error.

### 6.2 Generator (`task_generator.py`)

**Batch + rank:** every generation step produces `gen_batch_size = 3` candidates in one call (reuse the verbalized-sampling JSON-array machinery; drop the probability/learnability self-scores — they are superseded). All 3 go through the cheap non-LLM key-leak check; survivors go to the MOI ranker (§6.3), which picks ONE. That one proceeds to the coherence gate.

**Mutation-operator framing.** The user prompt is rebuilt. Inputs per call: the anchor (with chronicle, classification, `too_easy_diagnosis` if any), 2 KNN positive examples (frontier, with chronicles), 1–2 labeled negative examples (a `too_easy` and a `beyond_frontier` near-neighbor), the operator (from anchor classification), existing interaction types. Skeleton (exact structural slots list is normative):

```
You are designing ONE new social scenario by MUTATING a parent scenario.

PARENT SCENARIO (your anchor):
{parent JSON incl. structured_goals, partner_key if any, scenario_title}
PARENT OUTCOME: {classification} — {one-line explanation: "solved on first try (too easy): {too_easy_diagnosis}" | "at the learning frontier; chronicle below shows what was learned" | "never solved; chronicle WARNINGs show the structural trap"}
PARENT CHRONICLE (why it was hard / what was learned):
{skills_final_md[:1500]}

MUTABLE STRUCTURAL SLOTS:
  a. key_mechanism (swap to a different mechanism from the library)
  b. surface_misdirection (discoverability: how far the stated objection sits from the real condition)
  c. hardening_triggers (prior-incongruence: how closely the locking moves match default helpful-assistant behavior)
  d. cost_coupling (how much satisfying the key costs the learner's own goal)
  e. power-asymmetry direction (who holds leverage)
  f. information asymmetry (who knows what)
  g. relationship stakes (what the relationship is worth to each party)

YOUR OPERATOR: {operator_block}
```

Operator blocks (exact):
- `escalate`: "The parent was TOO EASY. Choose 1–2 slots from (b),(c),(d) and tighten them; the diagnosis above names the slack. Preserve all other slots. Do NOT make the scenario impossible: the movement_conditions must remain genuinely satisfiable by a skilled, non-capitulating actor."
- `relax`: "The parent was NEVER SOLVED. Identify from the chronicle WARNINGs which slot made it unwinnable and loosen exactly that slot. Preserve all other slots."
- `lateral`: "The parent was AT THE LEARNING FRONTIER. Hold difficulty constant: preserve slots (b),(c),(d) at the same intensity, and mutate 1–2 of (a),(e),(f),(g) to explore a structurally different dynamic."

Plus (all operators): vary surface freely (characters, setting, occupations); shared-public-context rule unchanged; triple format guide unchanged; NEW required output fields: `partner_key` (per §1.3, mechanism from the library below), `mutated_slots: ["b","d"]`, and `mutation_rationale: "one sentence"`. The generator prompt MUST state: "Agent 0 is the learner and receives the structured goal triple; agent 1 is the partner and receives the partner_key and a natural-language goal only. Write the learner's role to continue this structural vantage point: {anchor.target_perspective}." Include `MECHANISM_LIBRARY` verbatim in the system prompt.

**System prompt:** keep the INTERESTING/LEARNABLE/DIFFICULT framing but replace the DIFFICULT paragraph's "make the partner's pressure realistic and hard to dismiss" with: "Difficulty lives in the partner_key: hidden conditions a naive, maximally-agreeable actor will not discover or will refuse to pay for — NOT in a partner who simply never concedes. Every generated scenario is solvable by construction: its movement_conditions define the path."

**DELETE:** `_EDIT_INTENTS["raise_difficulty"]` and `_EDIT_INTENTS["improve_interestingness"]`. **KEEP** `fix_coherence` (the coherence gate still requests patches).

### 6.3 MOI (`model_of_interestingness.py`) — gate → ranker, worth-only
- **DELETE** dimensions 2 (novelty — owned by niching/dedup) and 3 (learnability — owned by LP) and the edit-suggestion loop (`max_edits`).
- New role: rank the generator's candidate batch on **social worth** alone. System prompt core:
```
You are ranking candidate social scenarios on one axis only: SOCIAL WORTH. A scenario has worth if the tension is one a thoughtful person would recognize as a real, meaningful social situation — not contrived, not a gimmick, not a logic puzzle wearing a social costume. Judge the human meaningfulness of the dynamic, NOT its difficulty and NOT its novelty.
Respond ONLY with JSON: {"ranking": [candidate indices, best first], "worst_reason": "one sentence on why the last-ranked is weakest"}
```
- Pick `ranking[0]`. Log full ranking. The `min_archive_size` condition is gone (ranking needs no archive).

### 6.4 New module: `social_omni_epic/niches.py` (replaces the diversity gate's role)
- **Dedup pre-filter (kept, admission):** cosine > 0.92 on the **full-text** embedding (existing `to_text_for_embedding`) vs. archive → reject as `generation_failed`. This runs before episodes, after coherence.
- **Niche assignment (post-episode, bookkeeping):** second embedding on the **abstract** text = `f"{social_dynamic} | {target_perspective}"` (these exist only after title generation, hence post-episode). k-means with `k = max(8, min(15, archive_size // 8))` over abstract embeddings of all archived tasks; refit every 10 archived tasks; on refit, re-assign all `niche_id`s. Persist centroids in the checkpoint. Seeds get abstract embeddings in Phase 0 (titles must be generated for seeds there — see §9).
- Selection does NOT consume niches in this phase (deferred). Niches feed `metrics.json` and the coverage analysis only.

---

## 7. (reserved — merged into §6)

## 8. Reflection & chronicle changes

### 8.1 Reflection module (`reflection_module.py`)
- **DELETE** from `_SYSTEM`: the HOLLOW EXTRACTION / rubric-results diagnosis vocabulary ("use the RUBRIC CHECK RESULTS", the constraint-gate framing).
- **REPLACE** STEP 1 diagnosis inputs/instructions with:
```
STEP 1 — DIAGNOSIS. Write a <Diagnosis> block analyzing, using the EVALUATION SIGNALS:
  - GOAL and RELATIONSHIP scores and their trend across attempts
  - KEY VERDICT (when present): which hidden movement conditions the actor approached or satisfied, which hardening triggers were tripped and whether they were repaired. (You see this verdict, NOT the hidden conditions themselves.)
  - The failure pattern: PRESSURE FAILURE (tripped hardening — pushed, threatened, imposed, or problem-solved prematurely), DISCOVERY FAILURE (argued against the partner's stated objection without ever probing beneath it), COST AVOIDANCE (identified what the partner needed but refused to pay the cost to their own goal), or CAPITULATION (preserved the relationship by abandoning the goal).
  - Which chronicle entries were relevant, applied, or misdirecting; what skills were missing.
```
- The user prompt builder passes: per-attempt `{goal, rel}` scores, the key-check JSON verdicts per attempt (`conditions_met` / `triggers_tripped` indices and rationale — **never the condition/trigger text itself**; the chronicle must stay key-blind to preserve transfer), transcripts as before. Remove the `RUBRIC CHECK RESULTS` block and the `FAILURE PATTERN:` pre-classification line (the model now classifies).
- Everything else (entry format, abstraction rules, broadening rules, adversarial Mode 1, synthesis-on-critique, meta-reflection prompts) unchanged.

### 8.2 Chronicle truncation (`skills_chronicle.py`)
Replace positional `format_for_prompt(max_entries=8)` with relevance ranking:
```python
def format_for_prompt(self, query_embedding, fm, max_entries=8):
    # embed each entry's Condition (cache on the entry: entry.condition_embedding)
    # rank by cosine(query_embedding, condition_embedding); take top max_entries; preserve original relative order in output
```
`query_embedding` = embedding of the current scenario's learner goal (the existing abstract-goal text where available, else the raw goal). Cache condition embeddings on write (compute at upsert time).

### 8.3 Wire `check_final` (`curriculum.py`)
After meta-reflection: `adv_final = adversarial.check_final(final_chronicle, ...)`. If not approved → one `synthesize_with_critique` pass on the final chronicle; if still not approved → keep the chronicle, set `archive_record.final_check_flag = adv_final.issues` and log. Never discard a chronicle outright.

---

## 9. Phase 0: seed calibration (`scripts/run_phase0_calibration.py`, new)

Purpose: run every SOTOPIA-90 seed once through the K-attempt loop (§4.2) to obtain per-seed `lp_value`, `classification`, chronicle, and abstract embedding — before any generation.

- Seeds run **native**: no triple, no `partner_key` (partner uses the native prompt, §3.4 fallback), learner goal = native `agent_goals[target_agent_idx]`. Terminal label = `GOAL ≥ 7 ∧ REL ≥ 0` only (no key check).
- Same reflection/meta-reflection machinery (chronicle starts empty).
- Generate `scenario_title` / `social_dynamic` / `target_perspective` for every seed at the end (needed for abstract embeddings; the existing title generator works on seeds).
- Output: updated archive checkpoint where every seed has §1.5 fields populated; recompute the Thompson prior per §5.2 and log it.
- CLI: `python scripts/run_phase0_calibration.py run_name=X [seed_limit=N]`. Must be resumable (skip seeds already calibrated in the checkpoint).
- The main curriculum runner asserts Phase 0 completion (all seeds have `classification`) before starting, unless `--allow-uncalibrated` is passed.

---

## 10. Evaluation hardening

### 10.1 Frozen partner engine (`experiments/evaluate_with_memory.py` + a new `configs/eval_partner.yaml`)
- Create `configs/eval_partner.yaml`: `{model: ..., temperature: ..., prompt_template: "native"}`. The eval script must instantiate every partner from THIS config object and the scenario's native persona. Grep the eval path for any import of the curriculum partner prompt or `partner_key` plumbing — there must be none. Eval partners are vanilla: native SOTOPIA-π persona, no key, no chronicle.
- Eval scoring: the same judge model as §3.2 (GOAL/REL from 7-dim diagnostics) — frozen across conditions. Report per-condition raw scores AND deltas vs. the vanilla-learner condition.

### 10.2 Contamination check (`scripts/check_contamination.py`, new)
- Inputs: seed-90 jsonl, generated bank, eval-150 set. Checks: (a) `source_env_id` / id overlap → hard fail; (b) max cosine (full-text embeddings) of each eval scenario vs. seed+generated banks; write a table (`results/contamination_report.json`) of max-sim per eval item; warn list for > 0.92.

### 10.3 Compute accounting
- Add a global token/episode meter to `FM` (increment on every call; tag with `component` and `condition`). Dump to `results/{run}/compute_report.json`. The eval harness aggregates per-condition totals.
- ExpeL baseline (`expel_baseline.py`): expose `n_trials_per_task` in config; document that headline comparison runs ExpeL at 3 trials/task (≈ matched episode budget). No other ExpeL changes.

### 10.4 Judge validation (`scripts/validate_judges.py`, new)
- Inputs: a directory of human-annotated SOTOPIA episodes (transcript + human GOAL score; user supplies the data path) and a directory of paired dev transcripts for the LP task.
- For each candidate judge in `configs/judge_candidates.yaml` (e.g., a Gemini Flash-tier model, a Claude Haiku-tier model, Llama-70B): compute (a) Pearson/Spearman of judged GOAL vs. human GOAL on ≥50 episodes; (b) pairwise swap-consistency rate on ≥30 LP pairs (run each pair in both orders, measure agreement).
- Output: `results/judge_validation.json` + a printed ranking. The chosen judge goes into `judge.model`. Constraint enforced in config validation: `judge.model` provider ≠ `learner_model` provider.

---

## 11. Config & logging (`configs/social_omni_epic_curriculum.yaml`)

```yaml
# CHANGED / NEW
random_seed: 42                      # MUST be set; runner refuses null with an error
judge:
  model: "<filled after validate_judges>"   # cross-lab; provider must differ from learner_model
  lp_temperature: 0.3
gen_batch_size: 3
mechanism_library: default           # the 5-entry library; do not extend
stopping: { N: 90 }                  # unchanged
max_attempts: 4                      # unchanged (K)
chronicle_max_entries: 8             # now = relevance-ranked top-k

# DELETED keys: difficulty.D, difficulty.re_gate_after_edit, judge.self_consistency_k,
#   use_verbalized_sampling, vs_num_candidates, moi.max_edits, moi.min_archive_size,
#   evaluator_model (superseded by judge.model)
```

**Logging (all under `results/{run}/logs/`):** every LP vote (raw JSON), every key-check verdict, every MOI ranking, every gate decision incl. quarantines, per-anchor vote tallies per iteration, generations-per-niche, the Thompson prior chosen after Phase 0, compute meter. `metrics.json` adds: `mean_lp`, `lp_by_generation_depth`, `classification_counts`, `niche_counts`.

---

## 12. Probe harness (`scripts/run_key_probes.py`, new) — BUILD FIRST

- Loads `data/key_probes.jsonl` (5 hand-written scenarios with `partner_key`; the user will supply these — leave a documented placeholder format matching §1.3 + minimal SocialScenario fields).
- For each probe: run 2 episodes with the new partner prompt (§3.4) and a vanilla learner; then run two automated audits on each transcript with the judge model:
  1. **Leak audit:** "Does the partner ever state, name, or clearly hint at its hidden movement conditions or triggers? Quote the turn if so." → `{leaked: bool, evidence}`
  2. **Adherence audit:** given the key + transcript: "Did the partner shift position at any point where no movement condition had been met (early yield)? Did the partner fail to harden after a trigger was tripped?" → `{early_yield: bool, ignored_trigger: bool, evidence}`
- Print a 5×3 summary table (leak / early yield / ignored trigger). **Acceptance bar: ≤1 failure cell across the table.** If exceeded, STOP and report — the partner prompt needs iteration before anything else proceeds.

---

## 13. Explicitly DEFERRED — do NOT implement
1. Two-level (niche-first) Thompson sampling. Log `niche_counts` only.
2. Multi-seed (≥3) curriculum runs. Single seeded run; the report text carries the limitation sentence.
3. MOI positive-control probes.
4. Cross-lineage chronicle consolidation pass.
5. Mechanism-library extension.
6. `seed_both_perspectives` — delete the flag and its branch (decision: false, permanently).

## 14. Acceptance checklist (run after implementation)
- [ ] `run_key_probes.py` passes the bar in §12.
- [ ] `run_phase0_calibration.py` completes on `seed_limit=5` end-to-end; seeds carry classification + abstract embedding.
- [ ] One curriculum iteration end-to-end: generate(batch=3) → MOI rank → coherence+key gates → dedup → 4-attempt loop → LP computed with logged votes → archive record has all §1.5 fields → anchor vote tally updated.
- [ ] A `too_easy` path: scenario archived, `lp_value=0.0`, no chronicle, anchor down-weighted by 0/6 votes, `too_easy_diagnosis` stored.
- [ ] No gate defaults to PASS on a forced API error (test with a mock); quarantine file written.
- [ ] Eval script instantiates partner solely from `configs/eval_partner.yaml`; grep confirms no `partner_key` import in eval path.
- [ ] `grep -rn "success_rubric\|_evaluate_rubric\|raise_difficulty\|record_outcome_weight\|seed_both_perspectives"` returns only data-model back-compat reads and this spec.
