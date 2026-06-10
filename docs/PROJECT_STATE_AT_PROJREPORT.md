# Social OMNI-EPIC: Complete Technical Architecture (Current State)

> **Purpose:** This document is a complete, ground-truth description of the Social OMNI-EPIC codebase as it currently stands — intended as context for brainstorming next steps with an LLM. It includes the full pipeline, all major data structures, every system/user prompt used in the system, and the reasoning behind key design decisions.

---

## 0. Project Overview

Social OMNI-EPIC is a **curriculum learning framework** for training social skill in LLMs. Starting from 90 SOTOPIA seed scenarios, the system autonomously generates progressively harder social interaction scenarios, runs simulated episodes where a "learner" agent tries to achieve a social goal, and distills lessons into a **Skills Chronicle** — a structured memory that the agent carries into future attempts.

The key hypothesis: by iterating on scenarios at the frontier of the agent's current ability (scenarios it fails on first try but can learn to solve), and by reflecting on failures to build transferable skills, the system produces both better scenarios and a richer skills corpus.

**Entry point:** `scripts/run_curriculum.py`

**Output:** `results/{run_name}/`
- `success/{scenario_id}.json` — scenarios where the agent failed first, then learned to solve
- `failed/{scenario_id}.json` — scenarios the agent never solved across K attempts
- `discarded/{iter}.json` — scenarios the agent solved trivially on the first try
- `archive_latest.json` — full archive state with Thompson priors (resume point)
- `metrics.json` — per-iteration metrics

---

## 1. Source File Map

```
scripts/
  run_curriculum.py          ← main entry point (this document traces from here)
  run_debug.py               ← single-scenario debug runner
  run_phase0.py, run_phase1.py, run_phase2.py  ← legacy phased runners
  generate_seed_titles.py    ← one-off script to title seeds
  build_eval_candidates.py   ← eval dataset construction

social_omni_epic/
  fm.py                      ← Foundation Model wrapper (chat + embeddings)
  data_models.py             ← Pydantic data models (SocialScenario, AgentProfile, etc.)
  archive.py                 ← Archive state + Hierarchical Thompson Sampling
  curriculum.py              ← Two-loop engine (shared by run_curriculum + run_debug)
  task_generator.py          ← Scenario generation, editing, difficulty diagnosis
  episode_runner.py          ← Sotopia episode execution + rubric evaluation
  reflection_module.py       ← Per-attempt chronicle editing after failure
  meta_reflection.py         ← Cross-attempt synthesis (final skills_final_md)
  adversarial_agent.py       ← Quality gate on chronicle edits
  coherence_check.py         ← Structural validity gate on generated scenarios
  model_of_interestingness.py ← Interestingness/novelty/learnability gate
  scenario_title.py          ← Scenario title generation + target agent designation
  skills_chronicle.py        ← Skills Chronicle data structure + serialization
  sotopia_bridge.py          ← SocialScenario → Sotopia env/agent profiles
  seeds.py                   ← Load and embed 90 SOTOPIA seed scenarios
  validation.py              ← JSON schema validation for generated scenarios
  embedding_utils.py         ← KNN retrieval + diversity metrics
  tracing_fm.py              ← FM subclass with per-call logging
  memory.py                  ← Legacy memory module (used by baseline experiments)

configs/
  social_omni_epic_curriculum.yaml   ← primary config

data/
  sotopia_90_seeds.jsonl     ← 90 seed scenarios

analysis/                    ← post-hoc analysis scripts (not part of pipeline)
experiments/                 ← baseline comparison scripts (ExpeL, Reflexion)
```

---

## 2. Core Data Models (`social_omni_epic/data_models.py`)

### `AgentProfile`
Mirrors Sotopia's AgentProfile exactly (field names must match for bridge compatibility).

```python
class AgentProfile(BaseModel):
    id: str                          # UUID
    first_name: str
    last_name: str
    age: int
    gender_identity: str
    occupation: str
    big_five: str                    # personality descriptor
    moral_values: str
    schwartz_portrait_value: str
    decision_making_style: str
    secret: str                      # hidden leverage for THIS scenario's shortcut
    mbti: str
    public_info: str                 # 2-3 sentence bio shown to the agent
```

### `StructuredGoal`
The three-part social goal that makes scenarios genuinely hard:

```python
class StructuredGoal(BaseModel):
    outcome: str    # "get X to agree to Y" — the core state-change
    constraint: str # "X feeling manipulated" — the relational cost to avoid
    shortcut: str   # "You hold Z and could..." — tempting move that breaks constraint
```

**Design rationale:** The shortcut is a *generative* field — it drives the naive failure mode and the difficulty editor. It is NOT a rubric check. Taking the shortcut is evaluated indirectly: shortcut → constraint check fails.

### `RubricCheck`
```python
class RubricCheck(BaseModel):
    kind: str        # "outcome" | "constraint"
    question: str    # yes/no question; TRUE = learner succeeded
    perspective: str # "neutral" (transcript-observable) | "partner" (partner internal state)
```

Outcome checks: single pass, neutral perspective, transcript-observable.
Constraint checks: k-sample majority vote, partner perspective, conditioned on partner's private background + secret.

### `SocialScenario`
The central data object. Key fields:

```python
class SocialScenario(BaseModel):
    id: str                              # UUID
    scenario: str                        # shared public context (NO secrets/shortcuts)
    agent_profiles: list[AgentProfile]   # [0] = learner, [1] = partner (after designation)
    agent_goals: list[str]               # RENDERED from structured_goals (Sotopia-facing)
    structured_goals: list[StructuredGoal]  # symmetric — one per agent
    success_rubric: SuccessRubric        # learner's rubric checks (AND gate for success)
    relationship: str                    # stranger/acquaintance/friend/romantic/family
    relationship_background: str
    interaction_type: str
    difficulty_tags: list[str]
    source: str                          # "generated" | "seed_sotopia"

    # Cooperative alignment guards
    competing_interest: Optional[str]    # family (a): what learner forfeits by accommodating
    partner_default_position: Optional[str]  # what partner naturally offers (must fall short)

    # Phase 2: skills chronicle
    skills_final_md: Optional[str]       # final chronicle markdown

    # Phase 2: title / retrieval
    scenario_title: Optional[str]        # "social dynamic | target perspective"
    social_dynamic: Optional[str]        # left half
    target_perspective: Optional[str]    # right half

    # Phase 2: target agent
    target_agent_idx: int                # 0 or 1
    target_agent_goal_abstract: Optional[str]  # scenario-agnostic goal description

    # Curriculum results
    goal_trajectory: list[float]         # GOAL scores per attempt
    goal_score: Optional[float]

    # Thompson Sampling bookkeeping
    n_i: float        # effective selection count (float for weighted outcomes)
    n_solved: int     # children that reached solved_after_biting
    n_children: int   # total children generated from this anchor
    last_chosen: int  # iteration when last selected
    prior_alpha: float  # Beta prior alpha (1.0 for seeds; inherited for children)
    prior_beta: float   # Beta prior beta
    embedding: Optional[list[float]]
```

### `ArchiveState`
```python
class ArchiveState(BaseModel):
    successful: list[SocialScenario]      # seeds + solved children (Thompson arms)
    failed_generation: list[dict]         # generation/gate failures (metadata only)
    failed_interestingness: list[SocialScenario]  # MoI-rejected scenarios
    failed_tasks: list[SocialScenario]    # bit but never solved across K attempts
```

---

## 3. Foundation Model Wrapper (`social_omni_epic/fm.py`)

```python
class FM:
    def __init__(self, model, temperature, embedding_model="text-embedding-3-small"):
        # Chat: Lightning.ai (if LIGHTNING_AI_API_KEY set) → OpenAI fallback
        # Embeddings: always direct OpenAI (Lightning.ai doesn't support /v1/embeddings)

    def query(system, user, temperature=None) -> str
    def query_json(system, user, temperature=None) -> dict   # response_format: json_object
    def get_embeddings(texts) -> list[list[float]]
    def _retry(fn, max_retries=5)  # exponential backoff: 2s, 4s, 8s, 16s, 32s
```

**Temperature auto-fallback:** If a model rejects the `temperature` parameter (e.g., `gpt-5-mini` only supports default), sets `_temperature_supported = False` and retries without the parameter.

---

## 4. Archive & Hierarchical Thompson Sampling (`social_omni_epic/archive.py`)

### Algorithm

Each archive entry is modeled as a Bernoulli arm. Posterior:
```
posterior_i ~ Beta(prior_alpha_i + n_solved_i, prior_beta_i + n_i - n_solved_i)
```

- **Seeds:** flat prior `Beta(1, 1)` — no evidence
- **Children:** inherit parent's posterior at birth — warm start reflecting structural similarity

At each selection step, sample once from every posterior and pick `argmax`. No tunable constants; exploration/exploitation balance emerges naturally.

```python
def thompson_select(self) -> int:
    samples = []
    for task in self.state.successful:
        alpha = task.prior_alpha + task.n_solved
        beta_param = task.prior_beta + (task.n_i - task.n_solved)
        samples.append(np.random.beta(alpha, beta_param))
    return int(np.argmax(samples))
```

### Sequential within-batch updates
Thompson selection is done **sequentially within a batch** (not in parallel) so each pick updates `n_i` before the next pick, preventing duplicate anchor selection:
```python
for b in range(current_batch):
    idx = archive.thompson_select()
    archive.record_selection(idx, global_iter + b)  # increments n_i immediately
    batch_anchor_indices.append(idx)
```

### Posterior adjustments post-outcome
```python
archive.record_outcome_weight(anchor_idx, extra_n_i=-0.5)  # generation/gate failure: halve penalty
archive.record_outcome_weight(anchor_idx, extra_n_i=+1.0)  # structural failure: double penalty
```

### Child prior inheritance
```python
def child_prior_from_parent(self, parent_idx) -> tuple[float, float]:
    p = self.state.successful[parent_idx]
    alpha = p.prior_alpha + p.n_solved
    beta_param = p.prior_beta + (p.n_i - p.n_solved)
    return float(alpha), float(beta_param)
```

---

## 5. Scenario Generation (`social_omni_epic/task_generator.py`)

### Generation System Prompt (`SYSTEM_PROMPT`)

```
You are a creative social scenario designer. Generate social scenarios that are INTERESTING, LEARNABLE, and GENUINELY DIFFICULT.

INTERESTING: explores a novel social dynamic, power structure, or relational tension — not a generic archetype. Creative, specific, worth engaging with.

LEARNABLE: the learner agent's outcome must be meaningfully responsive to HOW they engage. Avoid scenarios where any polite response already succeeds.

DIFFICULT: the scenario must be hard by default. A naive language model playing the learner — one that is overly agreeable, cooperative, or socially deferential — should fail to achieve the learner's goal. Success requires resisting social pressure, holding a specific position, using strategic information, or navigating genuine opposition. Make the partner agent's pressure realistic and hard to dismiss.

[_SHARED_RULES]
Scenarios must involve realistic human social dynamics. No fantasy or sci-fi.
The scenario description must set up the conflict clearly: who wants what, what is at stake for each party, and why a quick agreement is NOT the natural outcome.
CRITICAL — THE SCENARIO DESCRIPTION IS SHARED PUBLIC CONTEXT: both agents read it before the conversation begins. It must NEVER describe or allude to either agent's shortcut, private leverage, or secret.

[_GOAL_FORMAT_GUIDE — key constraints]
  - "outcome": verb phrase completing "Your goal is to ___"; CORE ONLY (no delivery mechanisms); use qualifiers like "at least"
  - "constraint": phrase as the thing to avoid, reads naturally after "without"
  - "shortcut": MUST begin with "You"; names available leverage; does NOT label it as a trap
  - COOPERATIVE ALIGNMENT GUARD:
    Family (a): learner has competing_interest they forfeit by accommodating
    Family (b): partner won't volunteer what learner needs (pure face-threat)
  - SUCCESS RUBRIC: 1 outcome check (neutral) + 1-2 constraint checks (partner); 2-3 total
  - DELIVERY-MECHANISM TRAP: outcome checks must test STATE-CHANGE not delivery format

[_PROFILE_GUIDE]
  - Internally consistent: occupation, personality, public_info must cohere
  - secret: one specific hidden fact creating leverage relevant to THIS scenario

Respond with valid JSON matching exactly this schema: [_SCENARIO_SCHEMA]
```

### Generation User Prompt (`_build_user_prompt`)

```
EXAMPLE SCENARIOS FROM THE ARCHIVE — each was genuinely difficult: the agent failed on the first attempt, then learned. The skills chronicle shows WHY it was hard and what the naive agent got wrong. Build on these dynamics:

--- Example 1 ---
{scenario JSON with scenario_title, agent_profiles, relationship, interaction_type, difficulty_tags, agent_structured_goals, success_rubric}
[Skills chronicle — what made this scenario hard / what the agent learned:]
{skills_final_md[:1500]}

--- Example 2 --- [etc.]

SCENARIOS REJECTED AS UNINTERESTING BEFORE ANY EPISODE (avoid these patterns):
--- Rejected 1 ---
{scenario JSON without chronicle}

SCENARIOS BEYOND THE CURRENT FRONTIER — ran full episodes but the agent never solved them.
WARNING entries show what made them unlearnable. Do NOT generate scenarios with the same structural failure:
--- Beyond-frontier 1 ---
{scenario JSON with chronicle}

INTERACTION TYPES already present in the archive: [type_str].
You may set `interaction_type` to one of these if it genuinely fits, OR coin a new descriptive type.

Generate ONE NEW social scenario.
  TRANSFER the latent social structure. Each example's `scenario_title` is the authoritative structural description — its left half names the social dynamic, its right half names the learner's structural vantage point. Use these as your primary guide for what to preserve: the TYPE of constraint that bites, the FORM of the shortcut, the NATURE of the power asymmetry.
  VARY the surface freely: characters, setting, occupations, relationship, specific stakes.
  AIM FOR THE FRONTIER: target at least the same difficulty as the examples.
  Do NOT re-skin (same dynamic, different names). Do NOT jump to a completely different type of social challenge.
Return ONLY a JSON object matching the required schema.
```

### Verbalized Sampling System Prompt (`VS_SYSTEM_PROMPT`)

Same as `SYSTEM_PROMPT` plus:
```
VERBALIZED SAMPLING: You will generate {n_candidates} distinct candidates and score each on two axes:
- "probability": typicality (0.01–0.50) — how likely would a standard AI spontaneously propose this exact social dynamic? Low = more interesting.
- "learnability_score": skill-responsiveness (0.0–1.0). High = more learnable.

The ideal candidate has LOW probability AND learnability_score >= 0.6.

Each candidate: {"probability": ..., "learnability_score": ..., "scenario_json": {...}}
Return JSON: {"candidates": [...]}
```

**Selection:** inverse-probability sampling (`weight = 1/p`) among learnable candidates (≥0.6). Falls back to `generate_from_archive` if VS fails.

### Edit Intents (3 modes in `_EDIT_INTENTS`)

**`fix_coherence`:**
```
The scenario has coherence issues that must be fixed. Fix ONLY the identified issues; preserve the premise, characters, structured goals, success_rubric, and interaction type except where an issue requires a change.
```

**`improve_interestingness`:**
```
The scenario is not interestingly difficult enough. Revise the STRUCTURED GOALS so the constraint genuinely bites and the shortcut is genuinely tempting...
MOST COMMON FAILURE — cooperative alignment: if the two agents' goals are cooperatively aligned at their core (both want the same outcome; the only obstacle is face or framing that an agreeable agent provides freely), the fix is NOT to make the partner more resistant. The fix is to give the LEARNER a genuine competing interest — a constraint or obligation that directly conflicts with what the partner needs, so that being maximally accommodating costs the learner something real.
```

**`raise_difficulty`:**
```
A naive agent solved this on the first try, so it is TOO EASY. Make it HARDER along the named social knob only. Do NOT add facts, parties, or numeric complexity, and do NOT make it impossible...
SPECIAL CASE — if the slack_knob is cooperative_alignment: the scenario's goals are cooperatively aligned at the core. Increasing partner resistance will NOT fix this; it only delays the cooperative solution. Instead, add a genuine competing interest to the LEARNER...
```

### `analyze_too_easy` — Difficulty Diagnosis System Prompt

```
A learner agent solved a social scenario on the FIRST try, which means it is not interestingly difficult. Identify the SINGLE root cause and say concretely how to fix it — WITHOUT adding facts, parties, or numeric complexity.

IMPORTANT: your suggested_edit must be a SCENARIO DESIGN change — a change to the structured goals, partner profile, or scenario description. It must NOT be a transcript edit, a suggested dialogue line, or instructions for what a character should say.

SPECIAL CASE — cooperative_alignment: This is the most common failure mode. It occurs when the two agents' goals are cooperatively aligned at their core — both want the same outcome, and the only obstacle is face/framing that an agreeable agent provides for free. The fix is NOT to make the partner more resistant — that only delays the cooperative solution. The fix is to add a genuine competing interest to the LEARNER that makes being maximally accommodating costly.

Respond with ONLY valid JSON.
```

**User prompt:**
```
SCENARIO: {JSON}

TRANSCRIPT (the learner solved this on the first try):
[T0] Agent A: ...

Respond JSON: {"slack_knob": "cooperative_alignment|shortcut_salience|constraint_bite|partner_resistance|partner_stake", "rationale": "...", "suggested_edit": "..."}
```

---

## 6. Quality Gates

### 6a. Coherence Checker (`social_omni_epic/coherence_check.py`)

**System prompt (8 checks):**
```
You are a structural validator for social scenarios. Your job is NOT to judge quality or creativity — only to flag logical inconsistencies that would make the scenario internally broken.

1. RELATIONSHIP CONSISTENCY: Does the relationship label match the relationship_background?
2. CONSTRAINT PHRASING: Flag ONLY if the constraint field literally begins with "without" (produces broken double-"without" rendering).
3. GOAL DISTINCTNESS: Are the two agent goals genuinely different?
4. PROFILE-GOAL PLAUSIBILITY: Does occupation/personality/decision_making_style make their goal believable?
5. SCENARIO-INTERACTION MATCH: Does the scenario description match the stated interaction_type?
6. GOAL ACHIEVABILITY (ZOPA): Does a zone of possible agreement exist where both agents could partially win through skilled play? Flag only structural impossibility, not difficulty.
7. RUBRIC & SHORTCUT VALIDITY: Is the shortcut load-bearing? Is the secret non-decorative? ≤3 rubric checks total? No procedural checks? No delivery-mechanism outcome checks? No extractable-utterance outcomes?
8. COOPERATIVE ALIGNMENT: Can the learner win by simply accepting whatever the partner naturally offers, or by freely accommodating without sacrificing anything real?

Return JSON: {"passed": true/false, "issues": ["specific issue 1", ...]}
```

**Temperature:** 0.2. **Default on error:** PASS.

### 6b. Model of Interestingness (`social_omni_epic/model_of_interestingness.py`)

**System prompt (3 dimensions):**
```
You are an expert auditor of social scenarios used to train social skill.

DIMENSION 1 — SOCIAL TENSION (most important):
A scenario PASSES tension only if:
  - the constraint genuinely BITES — pursuing the outcome bluntly really would damage the relationship/face;
  - the shortcut is genuinely TEMPTING — a naive agent would take it and WIN the outcome but BREAK the constraint;
  - AND a maximally COOPERATIVE, ACCOMMODATING agent also FAILS — cannot succeed by being agreeable.
It FAILS tension if: positional bargaining over a number; extract-a-sentence fetch quest; goals cooperatively aligned at core; polite accommodating move already succeeds.

DIMENSION 2 — NOVELTY (vs archive): Is it interestingly new?
FAILS if it re-skins an existing dynamic (same power structure / tension / strategic challenge) with new names.

DIMENSION 3 — LEARNABILITY / ZOPA: Is there at least one discoverable path to success a skilled agent could find?

Respond with ONLY valid JSON:
{"social_tension": true/false, "novel": true/false, "learnable": true/false, "reason": "...", "suggested_edits": [...]}

When the failure is cooperative alignment: the suggested_edit MUST give the learner a competing interest — NOT increase partner resistance.
```

**Passing condition:** `social_tension AND learnable AND (novel OR no archive shown)`. Up to `max_edits=2` edit iterations before rejection.

### 6c. Diversity Gate

Not an LLM call. Cosine similarity check:
```python
sims = emb_arr @ s_emb / (np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(s_emb) + 1e-9)
if sims.max() > diversity_similarity_threshold:  # default: 0.92
    return "generation_failed"
```

---

## 7. Episode Runner (`social_omni_epic/episode_runner.py`)

### Architecture

Uses Sotopia's `ParallelSotopiaEnv` with `RuleBasedTerminatedEvaluator` only. Intentionally bypasses `terminal_evaluators` due to two known bugs in the pinned Sotopia commit:
1. `complete_rating: 0` hardcoded in `ParallelSotopiaEnv.astep` — real scores discarded
2. `EpisodeLLMEvaluator.__acall__` KeyError swallowed silently — terminal evaluation yields empty ratings

All evaluation is done via `FM.query_json()` directly.

### Learner Turn Prompt (`_TURN_PROMPT`)

```
Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
Note that {agent}'s goal is only visible to you.
You should try your best to achieve {agent}'s goal in a way that align with their character traits.
Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).

=== Lessons from prior similar interactions (visible only to you) ===
{memory_text}
=== End of lessons ===

{history}.
You are at Turn #{turn_number}. Your available action types are {action_list}.
Note: If you have substantially achieved your social goal or reached a clear agreement, you SHOULD choose 'leave'...
Important: If your previous 2-3 attempts at the same approach have not moved the other person, do NOT repeat the same offer or request again — try a genuinely different strategy (ask a question, make a different concession, shift your framing, or acknowledge that you may not reach your goal today).
Keep your responses conversational — typically 2-4 sentences unless a complex explanation is genuinely required.

Please only generate a JSON string including the action type and the argument.
Your action should follow the given format: {format_instructions}
```

The `=== Lessons ===` block is only injected when the chronicle is non-empty.

### Partner Turn Prompt (`_PARTNER_TURN_PROMPT`)

```
Imagine you are {agent}...
Your success is defined ONLY by whether YOU achieve YOUR stated social goal. A mutual agreement or resolution that does not achieve your goal is NOT success for you. Do NOT offer concessions, compromises, or alternatives unless they directly advance your own goal.
You should choose 'leave' only if you have substantially achieved your own social goal — NOT simply because a clear agreement or resolution has been reached.
Keep your responses conversational — typically 2-4 sentences unless a complex explanation is genuinely required.
```

**Key difference from learner:** Partner is explicitly instructed to be self-interested. Prevents partner from cooperating out of social kindness.

### SOTOPIA 7-Dimension Diagnostics

Diagnostics only — NOT the success gate.

**System:**
```
Evaluate both agents' performance in this social interaction. Respond with ONLY valid JSON — no markdown fences, no other text.
```

**User prompt:**
```
{conversation history}

There are exactly 2 agents. Use "agent_1" for the first agent to appear in the conversation and "agent_2" for the second. agent_1 is the learner agent.
LEARNER GOAL (agent_1): {learner_goal}

Respond with ONLY this JSON (replace the example values with your actual scores and reasoning):
{
  "agent_1": {
    "believability":                   {"score": 8,  "reasoning": "Spoke consistently..."},
    "relationship":                    {"score": 0,  "reasoning": "..."},
    "knowledge":                       {"score": 5,  "reasoning": "..."},
    "secret":                          {"score": 0,  "reasoning": "..."},
    "social_rules":                    {"score": 0,  "reasoning": "..."},
    "financial_and_material_benefits": {"score": 2,  "reasoning": "..."},
    "goal":                            {"score": 7,  "reasoning": "..."},
    "goal_achieved":                   true
  },
  "agent_2": { ... same fields ... }
}
Score ranges: believability 0–10, relationship -5 to 5, knowledge 0–10, secret -10 to 0, social_rules -10 to 0, financial_and_material_benefits -5 to 5, goal 0–10.
For goal_achieved: true only if the agent substantially completed the specific, verifiable objective stated in their goal.
```

**Temperature:** 0.0 (deterministic).

### Rubric Evaluation (the actual success gate)

**Neutral outcome check system:**
```
You are a neutral observer judging whether a specific thing happened in a conversation. Decide ONLY from what is observable in the transcript; do not speculate about private feelings. Respond with ONLY valid JSON.
```

**Partner constraint check system:**
```
You ARE the second person in this conversation. Using your PRIVATE background and how the conversation actually landed for you, answer honestly from your own point of view — be candid, not polite or agreeable, and think about whether you would actually follow through. Respond with ONLY valid JSON.
```

**Shared user template:**
```
{context}
  ← neutral: "TRANSCRIPT:\n{history}"
  ← partner: "YOUR PRIVATE BACKGROUND:\n{partner_bg}\nYour secret: {secret}\n\nTHE CONVERSATION (you are {partner_name}):\n{history}"

QUESTION (a YES/true answer means the OTHER person succeeded): {question}
Respond with ONLY this JSON: {"verdict": true, "confidence": 0.8, "rationale": "one sentence"}
```

**Self-consistency for constraint checks:** k=3 samples, majority vote. Temperature 0.7 when k>1, 0.0 when k=1. Ties resolve to `True`.

**Success gate:** `goal_achieved = all(r["verdict"] for r in rubric_results)` — AND of ALL checks.

---

## 8. Two-Loop Curriculum Engine (`social_omni_epic/curriculum.py`)

### Loop 1: Difficulty Calibration

```python
for d in range(D + 1):  # initial run + up to D=2 edits
    result = await _episode(scenario, current_chronicle)
    if not result.goal_achieved:
        bit = True
        break  # agent failed → "biting failure" → proceed to Loop 2
    if d >= D:
        break  # too easy even after D edits → "discarded"
    # Agent solved on first try → too easy → diagnose and edit
    feedback = task_gen.analyze_too_easy(scenario, clean_transcript(result.transcript))
    edited = task_gen.edit_scenario(scenario, [feedback["suggested_edit"]], intent="raise_difficulty")
    edited, ok = run_coherence_gate(edited, ...)
    if not ok: break
    scenario = edited
```

**Rubric-artifact warning:** When constraint passes + GOAL ≥ 8.0 + outcome check fails → system logs warning that outcome check may be overconstrained. Still treated as `bit=True`.

### Loop 2: Skill Learning

```python
# Reuse attempt 1 transcript (the "bite")
for attempt in range(1, K + 1):  # K=4: attempt 1 = bite, attempts 2-4 = reflection retries
    if attempt > 1:
        result = await _episode(scenario, current_chronicle)  # with updated chronicle

    if result.goal_achieved:
        terminal_state = "solved_after_biting"
        break

    # Early structural failure: GOAL ≤ 2 on all attempts so far with ≥ 2 data points
    if len(all_scores) >= 2 and all(s["scores"].get("goal", 0.0) <= 2.0 for s in all_scores):
        loop_info["structural_failure"] = True
        break

    if attempt < K:
        ref_out = reflection_mod.reflect(chronicle, scenario, transcripts, ...)
        adv_result = adversarial.check_reflection(ref_out, transcript, ...)
        if not adv_result.approved and re_reflect:
            ref_out = reflection_mod.synthesize_with_critique(ref_out, adv_result.critique, ...)
        current_chronicle = ref_out.updated_chronicle
```

### After Both Loops

```python
# Meta-reflection runs for BOTH success (consolidate) and failure (document the trap)
final_chronicle = meta_mod.synthesize(
    chronicle_versions=all_versions, transcripts=all_transcripts,
    edit_reasons=all_edit_reasons, outcome=outcome,  # 2=solved, 3=failed
    scenario=scenario, anchor_task=anchor, attempt_scores=all_scores,
)
scenario.skills_final_md = final_chronicle.to_markdown()

title_data = title_gen.generate(scenario, target_agent_idx)
scenario.scenario_title = title_data["scenario_title"]
scenario.social_dynamic = title_data["social_dynamic"]
scenario.target_perspective = title_data["target_perspective"]
```

---

## 9. Reflection Module (`social_omni_epic/reflection_module.py`)

### Per-Attempt Reflection System Prompt (`_SYSTEM`)

```
You are a reflective coach analyzing a failed social interaction episode to improve a Skills Chronicle.

A Skills Chronicle is a document of structured entries that guide an AI agent's social behavior. Each entry has a Condition (when to apply it) and Guidance (what to do). Guidance must be PRESCRIPTIVE — specific enough that an agent reading it before a conversation would behave observably differently. "Be more strategic" is not guidance. "Lead with a question about X before making any ask" is guidance.

Your task after a FAILED episode:

STEP 1 — DIAGNOSIS:
Write a <Diagnosis> block analyzing:
  - The FAILURE PATTERN (use the RUBRIC CHECK RESULTS): HOLLOW EXTRACTION (got the outcome but broke the constraint — took the tempting shortcut), an ATTUNEMENT failure (failed a relational/internal constraint — model DEFAULTED TO PROBLEM-SOLVING when validation-first was needed; do NOT diagnose it as "discomfort"), or simply not achieving the outcome.
  - Which chronicle entries were relevant, applied, or misdirecting
  - What skills were missing

STEP 2 — EDITS:
For each entry you modify:
  <EditReason id="ENTRY_ID">Justification with SPECIFIC transcript evidence (quote directly)</EditReason>
  <Entry id="ENTRY_ID">
  <Condition>abstract structural pattern — no proper nouns or scenario-specific details</Condition>
  <Guidance>
  1. Primary guidance: [specific enough that an agent reading this before a conversation would behave observably differently]
  2. Warning (optional): [only if a specific tempting behavior contrasts with the primary guidance and backfires in a non-obvious way]
  3. Exception: when [a specific circumstance makes the primary guidance inappropriate], do [alternative] instead
  Note: Later clauses take precedence over earlier ones when their conditions apply.
  </Guidance>
  <Type>HEURISTIC | WARNING</Type>
  <Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
  <Provenance>[existing provenance, add ", attempt K"]</Provenance>
  </Entry>

For NEW entries:
  <EditReason id="NEW_ID">Why no existing entry covered this</EditReason>
  <Entry id="NEW_ID">... complete entry ...</Entry>

For entries that ACTIVELY MISDIRECTED the agent:
  <MisdirectionFlag id="ENTRY_ID"/>

CONDITION FIELD RULES (enforced strictly):
  - NO proper nouns, specific occupations, or scenario-unique details
  - Narrowing a Condition: low-risk, do freely
  - BROADENING a Condition: high-risk — EditReason MUST cite specific transcript evidence. If the broadened Condition would also apply to the parent/anchor scenario's social dynamic, it is too broad.
  - INHERITED ENTRIES: if an inherited entry's Condition contains domain-specific vocabulary from a previous scenario (e.g., "manuscript", "authorship", "lease"), re-abstract to remove prior domain's vocabulary while preserving the structural pattern.

GUIDANCE FIELD RULES (enforced strictly):
  - NO scenario-specific nouns, numbers, or domain terms
  - NO verbatim script patterns tied to this scenario's specifics. Describe the MOVE not the literal words.
  - Guidance MUST be abstract enough to transfer to a different scenario with the same structural pattern but completely different surface details.

Output ONLY the <Diagnosis>, <EditReason>, <Entry>, and <MisdirectionFlag> blocks. No other text.
```

**Temperature:** 0.4.

### Reflection User Prompt (built by `_build_prompt()`)

```
SCENARIO: {scenario.scenario}
LEARNER GOAL (structured):
  outcome:    {sg.outcome}
  constraint: {sg.constraint}
  shortcut (the tempting move that wins the outcome but breaks the constraint): {sg.shortcut}
RUBRIC CHECK RESULTS (this attempt):
  [PASS] (outcome) Did X agree to Y?  — rationale...
  [FAIL] (constraint) Did X come away feeling Z?  — rationale...
FAILURE PATTERN: HOLLOW EXTRACTION — the learner got the outcome but broke the constraint.
INTERACTION TYPE: {scenario.interaction_type}
PARENT SCENARIO SOCIAL DYNAMIC (for broadening check): {anchor_task.social_dynamic}

CURRENT SKILLS CHRONICLE:
<Entry id="...">...</Entry>
---
<Entry id="...">...</Entry>

PRIOR FAILED ATTEMPTS (N-1 attempts before this one):
=== Attempt 1 Transcript ===
[T0] AgentA: ...
...[truncated at 1500 chars]...

PRIOR EDIT REASONS (changes already made in earlier reflections):
  [ENTRY_ID]: reason cited specific evidence...

MOST RECENT FAILED ATTEMPT (attempt N — primary focus):
=== Attempt N Transcript ===
[T0] AgentA: ...
...[up to 3500 chars]...

SCORE PROGRESSION ACROSS ATTEMPTS (use this to identify what is and isn't improving):
  attempt |   goal |    rel |    bel | ...
        1 |    2.0 |    0.0 |    8.0 |
        2 |    4.0 |    1.0 |    8.5 |  ✓ [if solved]
  ---------+-...
           |   +2.0 |   +1.0 |   +0.5 |  (Δ first→last)

This is attempt N. The overall score trend is the primary learning signal: if it rose, something in the approach is working — preserve or strengthen it before adding new edits. The rubric check results above remain your primary guide for what specifically failed. Diagnose the failure and produce targeted chronicle edits.
```

### Synthesis With Critique System Prompt (`_SYNTHESIS_SYSTEM`)

```
You are the final arbiter synthesizing chronicle edits after a reflection-adversarial review cycle.

You have:
  1. The scenario and transcript showing what failed
  2. The reflection agent's diagnosis and proposed chronicle edits
  3. The adversarial agent's critique of those proposed edits

Your task: produce the DEFINITIVE chronicle edits.
  - Preserve the reflection's valid insights about what failed and why
  - Address legitimate adversarial concerns: evidence gaps, abstraction violations, over-broad Conditions that would also describe the parent scenario
  - Discard adversarial objections that are overly strict or miss the point of the failure
  - You are NOT re-doing the reflection from scratch — you are refining the proposed edits using the critique as a quality filter

Output format: <EditReason>, <Entry>, <MisdirectionFlag> blocks only. No <Diagnosis>, no commentary.
```

**Temperature:** 0.3.

---

## 10. Meta-Reflection Module (`social_omni_epic/meta_reflection.py`)

### Success Path System Prompt (`_SYSTEM_SUCCESS`)

```
You are a reflective coach synthesizing a Skills Chronicle after a SUCCESSFUL episode (solved in ≥2 attempts).

Your job is to produce a FINAL, coherent Skills Chronicle that:
1. CONSOLIDATE: Merge redundant entries covering the same condition into one.
2. RECONCILE: When entries contradict, produce one entry with an exception clause ("Exception: when X, do Y instead").
3. WEIGHT TOWARD HEURISTICS: Final chronicle should be predominantly HEURISTIC entries. Retain WARNINGs only for essential contrast.
4. CAPTURE WHAT WORKED: Final Guidance should reflect what the successful attempt did differently from the failed ones.
5. RETAIN ABSTRACTION: No proper nouns, specific occupations, scenario-unique details.

Output format — ONLY the <Entry> blocks, no other text.
[Full Entry format: Condition, Guidance with numbered clauses, Type, Dimension, Provenance]
Provenance: [carry forward and append "meta-reflection"]
```

### Failure Path System Prompt (`_SYSTEM_FAILURE`)

```
You are a reflective coach synthesizing a Skills Chronicle after a FAILED episode (never solved within the attempt budget).

Your job is to produce a FINAL, coherent Skills Chronicle that:
1. DOCUMENT STRUCTURAL RESISTANCE: Identify what made this scenario type persistently difficult. Add or strengthen WARNING entries that name the structural traps.
2. PROPOSE ALTERNATIVES: Where a strategy was tried and failed repeatedly, document what a DIFFERENT approach might look like (even untested), marked as a WARNING to flag uncertainty.
3. RECONCILE CONTRADICTIONS: Synthesize into exception clauses.
4. WEIGHT TOWARD WARNINGS: Predominantly WARNING entries. HEURISTIC entries limited to what reliably worked (if anything did).
5. RETAIN ABSTRACTION.

Output format — ONLY the <Entry> blocks.
Provenance: [carry forward and append "meta-reflection (failed)"]
```

**Temperature:** 0.3 for both paths.

**Success prompt:** passes only the final chronicle + edit reasons (cleanup pass — no transcripts needed).

**Failure prompt:** passes final chronicle + first attempt transcript + last attempt transcript (structural resistance pattern diagnosis).

---

## 11. Adversarial Agent (`social_omni_epic/adversarial_agent.py`)

### Mode 1: Post-Reflection Check (`_REFLECTION_CHECK_SYSTEM`)

```
You are an adversarial quality-control agent reviewing chronicle edits produced after a failed social interaction episode.

CHECK 1 — EVIDENCE: Does the EditReason cite SPECIFIC evidence from the transcript (direct quotes or specific turn references)? Generic reasoning is NOT sufficient.
  EXCEPTION: If the entry concerns a resource, capability, or information the agent possessed but did NOT use (a missed opportunity), scenario context is acceptable evidence.

CHECK 2 — CONDITION ABSTRACTION: Does the revised Condition remain abstract? Flag if it contains: proper nouns (person names, place names), specific occupations (e.g., "the nurse", "the landlord"), scenario-unique surface details.

CHECK 3 — GUIDANCE ABSTRACTION: Does the Guidance contain scenario-specific leakage? Flag if it contains: specific percentages, named parties, domain artifacts like "the manuscript" or "the loan covenant", verbatim script patterns tied to this scenario's surface, guidance that would not transfer to a structurally similar scenario with different surface context.

CHECK 4 — BROADENING: If a Condition was broadened, does the broadened Condition ALSO describe the parent scenario's social dynamic? If yes, the broadening is too aggressive — reject.

CHECK 4 [sic] — MISDIRECTION: Did any entry actively guide the agent toward worse behavior?

Respond with JSON only:
{
  "approved": true/false,
  "issues": ["specific issue 1", "specific issue 2"],
  "flagged_entry_ids": ["id1", "id2"],
  "active_misdirection_ids": ["id3"],
  "critique": "brief overall critique for the reflection module to address (empty string if approved)"
}
```

### Mode 2: Final Chronicle Check (`_FINAL_CHECK_SYSTEM`)

```
You are an adversarial quality-control agent reviewing a final synthesized Skills Chronicle.

CHECK 1 — INTERNAL CONSISTENCY: Overlapping Conditions with contradicting Guidance (not exception-clause-related)?
CHECK 2 — SYNTHESIS DRIFT: Does the final chronicle contradict an inherited entry without explicit justification?
CHECK 3 — COHERENCE: Redundant entries that could be merged?
CHECK 4 — OUTCOME BALANCE:
  - Failure outcome: Chronicle should be predominantly WARNING entries. Heavy HEURISTIC dominance in a failure case suggests over-confidence.
  - Success outcome: Chronicle may mix freely; WARNINGs from failed attempts are valuable contrast.
CHECK 5 — ACTIVE MISDIRECTION: Entries that would actively mislead an agent?

Respond with JSON only: {"approved": true/false, "issues": [...], "flagged_entry_ids": [...], "active_misdirection_ids": [...], "critique": "..."}
```

**Temperature:** 0.2. **Default on error:** PASS (`approved=True`).

**Note:** Mode 2 (`check_final`) is implemented but not currently wired into the main `run_episode_two_loop`. Mode 1 (`check_reflection`) is active after each per-attempt reflection.

---

## 12. Skills Chronicle (`social_omni_epic/skills_chronicle.py`)

### Storage Format

```xml
<Entry id="SCENARIO_ID_N">

<Condition>
Abstract social dynamic pattern — no proper nouns, specific occupations, or
scenario-unique details. Phrased as a recognizable structural pattern.
</Condition>

<Guidance>
1. Primary guidance: [specific enough to change behavior observably]
2. Warning (optional): [specific tempting behavior that backfires non-obviously — omit if primary guidance already covers it]
3. Exception: when [circumstance makes primary guidance inappropriate], do [alternative] instead
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>

<Type>HEURISTIC | WARNING</Type>

<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>

<Provenance>scenario_ids and iteration numbers</Provenance>

</Entry>
```

**Dimensions:**
- GOAL = achieving the social outcome
- FIN = financial/material benefits
- REL = relationship quality
- BEL = believability / in-character behavior
- KNO = knowledge gained
- SOC = social norms
- SEC = secrets kept

### Injection into Agent Context Window

```
=== Skills Chronicle (prior experience — visible only to you) ===
These lessons come from past interactions that may differ in setting or characters.
Apply the underlying principle and adapt it to your current situation — do not treat them as literal scripts.

<Entry id="...">...</Entry>

---

<Entry id="...">...</Entry>

=== End of Skills Chronicle ===
```

Truncated at `chronicle_max_entries=8` from the front (hard count cutoff for context-window budgeting).

### Key Operations

```python
chronicle.upsert_entry(entry)           # replace if exists, append if new
chronicle.format_for_prompt(max_entries=8)  # inject into agent context
SkillsChronicle.from_markdown(text)     # parse <Entry> tag blocks
chronicle.to_markdown()                 # serialize to tag-block string
```

---

## 13. Scenario Title & Target Agent (`social_omni_epic/scenario_title.py`)

### Title System Prompt (`_TITLE_SYSTEM`)

```
You generate concise retrieval keys for social interaction scenarios.

A SCENARIO_TITLE has exactly two parts separated by a pipe character (|):
  [Social dynamic description] | [Target perspective]

LEFT HALF — describe the abstract social dynamic:
- The structural type of interaction (negotiation, secret-keeping, persuasion under pressure, etc.)
- Key asymmetries (time pressure, information gap, power imbalance)
- What makes this scenario strategically interesting
- Relationship structure only when strategically relevant
- NO proper nouns, specific occupations, or scenario-specific surface details
- Aim for 8–20 words

RIGHT HALF — describe what skills were built from the TARGET agent's perspective:
- The structural vantage point (e.g., the uninformed buyer, the secret-holder, the mediator)
- What the target agent was trying to navigate
- NO proper nouns or scenario-specific details
- Aim for 8–15 words starting with "skills from the..."
- Format: "skills from the [structural role] [brief characterization of what they were navigating]"

EXAMPLES:
  "Zero-sum resource negotiation with asymmetric time pressure and anchoring behavior | skills from the uninformed, patient party managing concession timing against an anchoring opponent"
  "Cooperative secret-keeping under social pressure from a trusted third party | skills from the secret-holder navigating loyalty conflict without damaging the relationship"

Return JSON: {"scenario_title": "LEFT | RIGHT", "social_dynamic": "LEFT", "target_perspective": "RIGHT"}
The pipe separator is mandatory. Both halves are mandatory.
```

**Temperature:** 0.4.

### Target Agent Designation (`designate_target_agent`)

- For seed anchors (no abstract goal): return `anchor.target_agent_idx`, generate abstract goal via LLM
- For generated scenarios: embed `anchor.target_agent_goal_abstract` + both agent goals → pick argmax cosine similarity
- Ties → Agent 0 by convention

### Abstract Goal System Prompt (`_ABSTRACT_SYSTEM`)

```
Rewrite the following agent goal as a one-sentence abstract description. Remove all scenario-specific details (names, prices, locations, occupations). Describe only the abstract structural goal (e.g., 'secure a favorable deal under time pressure', 'protect a secret while maintaining the relationship'). Return only the one-sentence abstract goal, nothing else.
```

---

## 14. Configuration (`configs/social_omni_epic_curriculum.yaml`)

```yaml
run_name: "run_001"
model: "openai/gpt-5-mini"          # scenario generation, reflection, gates
temperature: 1.0
learner_model: "openai/gpt-5-mini"  # Sotopia agent
partner_model: "openai/gpt-5-mini"  # Sotopia agent
evaluator_model: "openai/gpt-5-mini"  # kept for compat; unused in main pipeline

random_seed: 42
iterations: 10                       # per invocation
batch_size: 4                        # concurrent episodes per round (asyncio.gather)
stopping:
  N: 90                              # stop when this many solved-after-biting exist across resumes

seeds_path: "data/sotopia_90_seeds.jsonl"
seed_both_perspectives: false        # if true: 180 entries (2 per seed)
seed_limit: null

max_turns: 20                        # per episode
max_attempts: 4                      # K: attempt 1 = bite, then up to K-1 reflection retries
chronicle_max_entries: 8             # context window truncation (hard count from front)

difficulty:
  D: 2                               # up to 2 difficulty edits before biting required
  re_gate_after_edit: true           # re-run coherence gate after each difficulty edit

judge:
  self_consistency_k: 3              # k-sample majority vote for constraint rubric checks

use_verbalized_sampling: false       # use VS_SYSTEM_PROMPT instead of SYSTEM_PROMPT
vs_num_candidates: 5

task_generator:
  num_examples: 3                    # archive examples shown in generation prompt
  max_retries: 3                     # retries on schema validation failure
  num_episode_failed_examples: 2     # beyond-frontier (never-solved) negative examples
  show_existing_types: true

enable_moi: true
moi:
  num_examples: 5                    # similar scenarios shown for novelty comparison
  min_archive_size: 10               # don't apply MoI gate if archive < 10
  max_edits: 2                       # edit iterations before rejection

enable_coherence_check: true
coherence_max_retries: 2

enable_diversity_gate: true
diversity_similarity_threshold: 0.92

adversarial:
  re_reflect_on_rejection: true      # if adversarial check fails: re-reflect with critique
```

---

## 15. Complete Pipeline Flow (annotated)

```
ENTRY: python scripts/run_curriculum.py run_name=X iterations=N
│
├── Load config via Hydra (configs/social_omni_epic_curriculum.yaml)
│
├── Initialize _Services:
│   FM, TaskGenerator, ModelOfInterestingness, CoherenceChecker,
│   ScenarioTitleGenerator, ReflectionModule, MetaReflectionModule, AdversarialAgent
│
├── Initialize Archive
│   ├── IF results/{run_name}/archive_latest.json exists → RESUME
│   │   └── archive.load_checkpoint() — restores state + Thompson priors
│   └── ELSE → FRESH START
│       └── _seed_archive(): load_sotopia_seeds_with_embeddings()
│           ← 90 seeds; embeddings file-cached at data/*.embeddings.npy
│
└── asyncio.run(_run_all())

    WHILE iterations_done < iterations AND solved_count < stopping.N:
    │
    ├── SEQUENTIAL THOMPSON SELECTION (batch_size picks):
    │   for b in range(current_batch):
    │     idx = archive.thompson_select()
    │       # Sample Beta(alpha_i + n_solved_i, beta_i + n_i - n_solved_i) for each entry
    │       # Pick argmax — no tunable constants
    │     archive.record_selection(idx)  # increment n_i IMMEDIATELY (prevents duplicate picks)
    │
    ├── CONCURRENT EXECUTION — asyncio.gather(...):
    │   (archive is read-only during gather; writes only in sequential update step below)
    │
    │   Per anchor: _run_one_scenario(anchor_idx, ...)
    │   │
    │   ├── KNN EXAMPLE RETRIEVAL:
    │   │   get_similar_scenarios(anchor.embedding, all_archive_embs, n=3)
    │   │     dedup: one per source_scenario_id (prefer anchor's target_agent_idx)
    │   │   Also: get_similar_scenarios(anchor.embedding, failed_task_embs, n=2)
    │   │     for negative (beyond-frontier) examples
    │   │
    │   ├── GENERATE SCENARIO:
    │   │   TaskGenerator.generate_from_archive(examples, neg_examples, existing_types)
    │   │   └── FM.query_json(SYSTEM_PROMPT, user_prompt)
    │   │       user_prompt: 3 archive examples w/ chronicles + 2 rejected + 2 beyond-frontier + type list
    │   │   └── validate_scenario() — checks required fields, 2 agents, 2 goals, rubric
    │   │   └── dict_to_scenario() → SocialScenario
    │   │   retry up to max_retries=3 with error feedback appended
    │   │
    │   ├── EMBED SCENARIO:
    │   │   FM.get_embeddings([scenario.to_text_for_embedding()])
    │   │   embedding text: scenario_title + scenario + interaction_type + learner/partner goals
    │   │
    │   ├── MOI GATE (if enable_moi AND archive.size >= 10):
    │   │   get_similar_scenarios(scenario.embedding, archive, n=5)
    │   │   moi.evaluate(scenario, similar)
    │   │   └── FM.query_json(MOI_SYSTEM_PROMPT, new_scenario + most_similar_existing, T=0.3)
    │   │       checks: social_tension AND novel AND learnable
    │   │   On fail: task_gen.edit_scenario(improve_interestingness) → re-embed → retry ≤2
    │   │
    │   ├── COHERENCE GATE (if enable_coherence_check):
    │   │   coherence_checker.check(scenario)
    │   │   └── FM.query_json(COHERENCE_SYSTEM_PROMPT, scenario_JSON, T=0.2)
    │   │       8 checks; issues are specific and actionable
    │   │   On fail: task_gen.patch_scenario(issues) → re-embed → retry ≤ coherence_max_retries=2
    │   │
    │   ├── DIVERSITY GATE (if enable_diversity_gate):
    │   │   cosine(scenario.embedding, all_archive_embs).max() > 0.92 → reject
    │   │
    │   ├── TARGET AGENT DESIGNATION:
    │   │   designate_target_agent(scenario, anchor, fm)
    │   │   ├── seed anchor: inherit anchor.target_agent_idx
    │   │   ├── generated: embed [anchor.target_agent_goal_abstract, goal_0, goal_1]
    │   │   │   → pick argmax cosine similarity
    │   │   └── _abstract_goal(goal, fm) → remove scenario-specific details
    │   │
    │   └── TWO-LOOP EPISODE (run_episode_two_loop in curriculum.py)
    │       │
    │       │ initial_chronicle = SkillsChronicle.from_markdown(anchor.skills_final_md)
    │       │
    │       ├── LOOP 1: DIFFICULTY CALIBRATION
    │       │   for d in range(D+1=3):
    │       │     result = await _episode(scenario, chronicle)
    │       │       └── run_single_episode(env_profile, agent_profiles, fm, ...)
    │       │           ├── ParallelSotopiaEnv (RuleBasedTerminatedEvaluator only)
    │       │           ├── learner LLMAgent (custom_template=_TURN_PROMPT + chronicle)
    │       │           ├── partner LLMAgent (custom_template=_PARTNER_TURN_PROMPT)
    │       │           ├── Async turn loop until done (max_turns=20, max_stale_turn=2)
    │       │           ├── _evaluate_diagnostics(env.inbox, fm) → 7-dim scores, T=0.0
    │       │           └── _evaluate_rubric(env.inbox, rubric, partner_profile, fm, k=3)
    │       │               ├── outcome checks: single pass, neutral system, T=0.0
    │       │               └── constraint checks: k=3 samples, partner system, T=0.7, majority vote
    │       │     if NOT result.goal_achieved → bit=True → break
    │       │     if d >= D → discarded (too easy) → return
    │       │     feedback = task_gen.analyze_too_easy(scenario, transcript)
    │       │       └── FM.query_json(ANALYZE_TOO_EASY_SYSTEM, scenario+transcript, T=0.3)
    │       │           → {slack_knob, rationale, suggested_edit}
    │       │     edited = task_gen.edit_scenario(suggested_edit, raise_difficulty)
    │       │       └── FM.query_json(SYSTEM_PROMPT, RAISE_DIFFICULTY_INTENT + current_scenario)
    │       │     edited, ok = run_coherence_gate(edited, ...)
    │       │     if ok: scenario = edited
    │       │
    │       ├── LOOP 2: SKILL LEARNING (K=4 attempts)
    │       │   all_transcripts = [attempt_1_transcript]
    │       │   for attempt in range(1, K+1):
    │       │     if attempt > 1: result = await _episode(scenario, current_chronicle)
    │       │     if result.goal_achieved → solved_after_biting → break
    │       │     if GOAL ≤ 2 on all attempts with ≥ 2 data points → structural_failure → break
    │       │     if attempt < K:
    │       │       ref_out = reflection_mod.reflect(chronicle, scenario, transcripts,
    │       │                    rubric_results, attempt_scores, ...)
    │       │         └── FM.query(_SYSTEM, reflection_user_prompt, T=0.4)
    │       │             → <Diagnosis> + <EditReason> + <Entry> + <MisdirectionFlag> blocks
    │       │         └── _parse_reflection_output() → ReflectionOutput
    │       │       adv_result = adversarial.check_reflection(ref_out, transcript, ...)
    │       │         └── FM.query_json(_REFLECTION_CHECK_SYSTEM, ..., T=0.2)
    │       │             → {approved, issues, flagged_entry_ids, active_misdirection_ids, critique}
    │       │       if NOT approved AND re_reflect:
    │       │         ref_out = reflection_mod.synthesize_with_critique(ref_out, critique, ...)
    │       │           └── FM.query(_SYNTHESIS_SYSTEM, ..., T=0.3)
    │       │               → revised <Entry> blocks
    │       │       current_chronicle = ref_out.updated_chronicle
    │       │
    │       ├── META-REFLECTION (runs for both success and failure):
    │       │   meta_mod.synthesize(all_versions, transcripts, edit_reasons, outcome, ...)
    │       │   ├── outcome=2 (success): FM.query(_SYSTEM_SUCCESS, final_chronicle+edit_reasons, T=0.3)
    │       │   │   → cleanup pass only (no transcripts passed)
    │       │   └── outcome=3 (failure): FM.query(_SYSTEM_FAILURE, chronicles+first+last_transcript, T=0.3)
    │       │       → structural resistance diagnosis
    │       │   → SkillsChronicle.from_markdown(llm_output) → <Entry> blocks
    │       │
    │       └── TITLE GENERATION:
    │           title_gen.generate(scenario, target_agent_idx)
    │           └── FM.query_json(_TITLE_SYSTEM, scenario_context, T=0.4)
    │               → {"scenario_title": "X | Y", "social_dynamic": "X", "target_perspective": "Y"}
    │
    ├── SAVE FILES (inside _run_one_scenario, before returning):
    │   solved_after_biting → results/{run_name}/success/{scenario_id}.json
    │   failed              → results/{run_name}/failed/{scenario_id}.json
    │   discarded           → results/{run_name}/discarded/iter_{N}.json
    │
    ├── SEQUENTIAL ARCHIVE UPDATE (after all concurrent tasks complete):
    │   "generation_failed" → archive.add_failed_generation()
    │                       → archive.record_outcome_weight(anchor_idx, extra_n_i=-0.5)
    │   "discarded"         → archive.add_failed_generation(reason=discarded)
    │   "solved_after_biting" → archive.record_solved_child(anchor_idx)
    │                         → scenario.prior_alpha/beta = child_prior_from_parent(anchor)
    │                         → archive.add_successful(scenario)
    │                         → archive.record_child(anchor_idx)
    │                         → solved_count += 1
    │   "failed"            → archive.add_failed_task(scenario)
    │                       → archive.record_child(anchor_idx)
    │
    ├── archive.save_checkpoint(global_iter)
    │   → archive_iter_{N}.json + archive_latest.json (symlink)
    └── metrics.json updated
```

---

## 16. Embedding & Retrieval (`social_omni_epic/embedding_utils.py`)

```python
def get_similar_scenarios(
    query_embedding, archive_embeddings, num_returns=5,
    source_ids=None, agent_idxs=None, preferred_agent_idx=None
) -> list[int]:
```

**Deduplication:** keeps one entry per `source_scenario_id` (prefers matching `preferred_agent_idx`). Prevents returning both agent-0 and agent-1 views of the same seed as "different" examples.

**Metric:** cosine similarity.

**Embedding text** (`SocialScenario.to_text_for_embedding()`):
```
Scenario type: {scenario_title}   ← primary structural signal (abstract; absent for seeds without titles)
Scenario: {scenario}
Interaction type: {interaction_type}
Relationship: {relationship}
Learner goal: {agent_goals[target_agent_idx]}
Partner goal: {agent_goals[1 - target_agent_idx]}
```

---

## 17. Key Design Principles & Rationale

1. **Hierarchical Thompson Sampling** — balances exploration/exploitation; children inherit parent posteriors (warm start). Seeds start `Beta(1,1)`; productive seeds develop peaked posteriors near 1; unproductive ones stay near `Beta(1,1)` or drift lower.

2. **Two-Loop Curriculum** — difficulty calibration (Loop 1) fully decoupled from skill learning (Loop 2). A scenario must "bite" (agent fails first try) before entering the skill-learning loop. This ensures the curriculum only learns from genuinely hard scenarios.

3. **Rubric over SOTOPIA Dimensions** — the 7-dim SOTOPIA scores are diagnostics only. The actual success gate is the per-check rubric. This avoids two known Sotopia evaluation bugs and enables perspective-aware judging (partner's private state for constraint checks).

4. **Cooperative Alignment Guards** — the system explicitly checks (at both generation and coherence gate) that scenarios cannot be solved by a maximally cooperative/agreeable agent. This is the most common failure mode in social scenario generation.

5. **Adversarial Quality Gates** — chronicle edits must survive adversarial review. If rejected, a synthesis pass reconciles reflection insights with adversarial concerns rather than either accepting or discarding wholesale.

6. **Abstraction Enforcement** — both reflection and adversarial agent enforce that chronicle entries contain no proper nouns, specific occupations, or scenario-unique details. This is what makes skills transfer across surface contexts. Inherited entries with prior-domain vocabulary must be re-abstracted.

7. **Graceful Degradation** — all LLM gates default to PASS on API/parse error. Archive is always recoverable from checkpoint. Pipeline does not block on non-critical failures.

8. **Perspective-Aware Deduplication** — embedding retrieval tracks `source_scenario_id + target_agent_idx` to prevent showing both perspectives of the same seed as "different" examples.

9. **Chronicle Inheritance** — when a scenario is generated from an anchor, the new episode starts with the anchor's `skills_final_md` as its initial chronicle. Skills propagate forward through the lineage — each generation of scenarios inherits the accumulated learning from its ancestors.

10. **Shared Public Context Rule** — the scenario description shown to both agents MUST NOT hint at either agent's shortcuts, leverage, or secrets. Asymmetric information structure collapses if the other agent learns what they should only discover through conversation.

---

## 18. Known Limitations / Design Debt

1. **Sotopia bugs vendored around:** `arun_one_episode` → `complete_rating: 0` hardcoded; `EpisodeLLMEvaluator.__acall__` → KeyError swallowed. Both bypass the Sotopia eval path entirely.

2. **Rubric-artifact warning not auto-resolved:** When constraint passes + GOAL ≥ 8.0 + outcome check fails, system logs a warning but still treats as `bit=True`. If this fires on >25-30% of biting failures, consider auto-discarding.

3. **`check_final` not wired into main loop:** `AdversarialAgent.check_final` (Mode 2, post-meta-reflection) is implemented but not called in `run_episode_two_loop`.

4. **`seed_both_perspectives: false`** — seeds are loaded with only agent 0 as learner. Setting `true` doubles seed corpus to 180 but hasn't been extensively tested in the curriculum loop.

5. **Verbalized Sampling disabled by default** — `use_verbalized_sampling: false`. The VS path is implemented and tested but the standard generation path is the primary mode.

6. **Memory module (`memory.py`)** — legacy module used by baseline experiment scripts. Not part of the main curriculum pipeline.

7. **`select_examples` in `TaskGenerator`** — implements `knn`, `diverse`, and `farthest` selection strategies, but the curriculum runner bypasses this method entirely (it does its own KNN selection via `get_similar_scenarios` before calling `generate_from_archive`). The `select_examples` method is used only by legacy scripts.

---

## 19. Baseline Comparison Modules (not part of main pipeline)

**`social_omni_epic/expel_baseline.py`** — implements ExpeL (experience-driven learning) as a comparison condition, using `memory.py` instead of `skills_chronicle.py`.

**`experiments/reflexion_baseline.py`** — implements Reflexion (Shinn et al.) as a comparison condition.

**`experiments/evaluate_with_memory.py`** — evaluates a trained memory/chronicle on held-out scenarios.

**`ExpeL-main/`** — original ExpeL codebase (vendored, used as reference implementation).
