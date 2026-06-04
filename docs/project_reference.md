# Social OMNI-EPIC: Project Reference

> **Audience:** a collaborator reading this cold. This document is the single source of truth for the research question, system design, implementation decisions, and codebase structure. Cross-references to the design rationale docs are given where the *why* goes deeper than this document does.

---

## 1. Research Question

**Can we borrow the open-ended environment-generation playbook from physical world models (OMNI-EPIC), and apply it to social worlds — to continuously generate socially interesting yet learnable scenarios, and use them to improve the social intelligence of LLM agents through in-context learning?**

The motivating observation: LLM agents fail in characteristic ways (sycophantic, overly cooperative, socially naive) even though they have absorbed social patterns from training data. Fine-tuning is expensive, opaque, and requires weight access. This project asks: can a growing archive of curated social "past experiences," retrieved and injected at inference time, make an LLM agent meaningfully more socially competent on unseen scenarios — without any weight updates?

### Core Claims

1. **An open-ended method for generating socially interesting yet learnable scenarios** — especially useful in data-scarce settings where social intelligence matters (e.g., novice skills training, mental health counselling agents).
2. **An in-context learning method that improves social intelligence without fine-tuning** — each learned scenario contributes a distilled "skills" entry; retrieve similar scenarios' skills at inference time.
3. **Improvement should theoretically scale with archive size** — more diverse, difficult past experiences → better performance on new social scenarios.

---

## 2. The OMNI-EPIC Analogy

OMNI-EPIC (Faldor et al., ICLR 2025) is a framework for physical task generation for RL agents:
- Select a seed task → generate an interesting + learnable task → train an RL agent on it → if solved, add to archive; if not, iterate. Failed tasks inform future generation. RL agents inherit previous policies.

**Social OMNI-EPIC inverts the difficulty direction:**

| OMNI-EPIC | Social OMNI-EPIC |
|---|---|
| Generate physical environments | Generate social scenarios |
| Tasks learnable for RL agent (start incompetent) | Scenarios hard enough to *not* be solved trivially (agent starts competent) |
| On failure: edit environment to make it *easier* | On immediate success (too easy): edit scenario to make it *harder* |
| RL policy inherits from prior task | Social agent inherits "skills" chronicle |

The key intuition: **scenarios are like past social experiences**. An agent with access to more, more varied, and harder past experiences should be more socially apt on new scenarios.

---

## 3. Why "Social Intelligence" is Different from Logic Puzzles

This is the central implementational insight that shapes every design decision.

A **logic puzzle** has explicit positional constraints. An LLM solves it in ~2 turns (if a satisfying assignment exists) or deadlocks forever (if none does). This is not social intelligence — it's constraint satisfaction.

A **socially difficult scenario** is hard on a different axis: success depends on changing another person's **internal state** through means that aren't logically derivable. The difficulty is in the *how*, not the *what*.

Three criteria for a scenario that teaches social skill (from Clark & Delia 1979, Dillard et al. 1989, Brown & Levinson 1987):
1. **Multiple goals in tension** — instrumental goal + relational/face goal pulling against each other
2. **The efficient move is the wrong move** — the blunt/direct path wins the surface outcome but violates the relational constraint. There is a tempting shortcut that *looks like* the solution.
3. **Success lives in the other person's head** — genuine buy-in, felt respect, willingness to follow through — only the partner can authentically judge this, not a transcript reader.

**Example:** *"Get your burnt-out colleague Maya to see a doctor, without her feeling like you're angling for the promotion."*
- Naive move: push hard, invoke the CEO → Maya complies but feels managed → not a social success
- Skilled move: peer-to-peer care, let her arrive at it herself, withhold the CEO card → genuine buy-in

This example motivates the entire goal structure (see §5).

---

## 4. Evaluation Strategy

### Internal evaluation (drives the learning loop)
Used during curriculum generation to determine: *did the agent succeed?*

- **Gate = AND of all rubric checks** (not a SOTOPIA score threshold)
- Two types of checks per scenario:
  - **Outcome check** (perspective: `neutral`) — observable in the transcript, judged by a neutral judge
  - **Constraint check** (perspective: `partner`) — about the partner's internal state, judged by the partner-perspective judge (sees partner's private profile + secret, asked from first-person stance, sampled 3× for self-consistency majority vote)
- **`goal_achieved` = AND of all checks** — hollow extraction (outcome✓, constraint✗) is scored **not solved**
- SOTOPIA 7-dim scores are **diagnostics only** (feed reflection, not the gate)

### External evaluation (paper claim)
Used to compare methods on held-out scenarios:
- **SOTOPIA-EVAL 7 dimensions** on canonical held-out scenarios (disjoint from training)
- Conditions: Vanilla LLM | CoT | ExpeL (insights from fixed SOTOPIA trajectories) | **Ours** (skills from generated archive) | Fine-tuning baseline (SOTOPIA-PI)
- Fixed partner + evaluator model across conditions; only the agent's prompt treatment varies

**Status:** External evaluation is deferred. The current build focuses on the methodology (curriculum + evaluation gate + debug inspection).

---

## 5. The Three-Part Goal Structure

Every generated scenario has a **structured goal** per agent (symmetric — both agents get one):

```
outcome:       the instrumental ask — a genuine state-change requiring authentic buy-in,
               NOT an extractable utterance ("get them to say X" is banned)
constraint:    the "without Y" relational/face cost — what blunt pursuit would damage
shortcut:      the tempting move that wins the outcome but breaks the constraint
               (an asset = private leverage, or a manner = blunt/coercive style)
shortcut_form: "asset" | "manner"  (descriptive only — hints the difficulty editor)
```

**Polarity rule:** every rubric check is phrased so **YES = the learner succeeded**.
- Outcome check: "Did [partner] agree to [outcome]?"
- Constraint check: "Did the learner secure this WITHOUT [partner] feeling [cost] — such that [partner] would actually follow through?"

The rendered `agent_goals` (what Sotopia feeds the agents) is derived from these fields by `render_agent_goal()` — the shortcut is surfaced as *available and tempting* in `<extra_info>` but not labeled as a trap.

**Theoretical grounding:**
- Outcome → Dillard's primary (influence) goal
- Constraint → Dillard's secondary goals (relational, identity) = Clark & Delia's three goal types
- Shortcut → Brown & Levinson's Face-Threatening Act (FTA)

The `goal_type` field is an **open descriptive label only** (e.g., "persuade-resistant-peer") — it never drives control flow. There are no hardcoded "regions."

---

## 6. The Curriculum Loop

Two nested loops per generated scenario. See `social_omni_epic/curriculum.py` for the implementation.

### Loop 1 — Difficulty Calibration (D=2 edits, up to 3 scenario versions)

```
for d in range(D + 1):
    run episode with chronicle-equipped agent (attempt 1)
    if agent FAILS → scenario has BITE → break, go to Loop 2
    if d < D:
        analyze_too_easy(transcript) → which social knob is slack?
        edit_scenario(intent="raise_difficulty") → tighten that knob
        re-gate coherence/ZOPA
if never bit → DISCARD (not archived)
```

The difficulty editor raises **social knobs only**: shortcut salience, constraint bite, partner resistance, partner stake. It never adds facts, parties, or numeric complexity (which would re-create the logic-puzzle problem).

### Loop 2 — Skill Learning (K=4 attempts, 3 chronicle edits)

```
attempt 1 = the biting failure from Loop 1 (reused, not wasted)
for attempt in 1..K:
    if succeeded → terminal_state = "solved_after_biting" → break
    if attempt < K:
        reflect → adversarial_check → (re-reflect if rejected) → update chronicle
meta_reflection (runs always: HEURISTIC-dominant on success, WARNING-dominant on failure)
```

### Archive Policy

| Terminal state | Archived? | Counts toward stopping? | Use |
|---|---|---|---|
| **Discarded** | No | No | Too easy; D edits couldn't manufacture bite |
| **Solved-after-biting** | Yes (`add_successful`) | **Yes** | Genuine learning event; stepping stone |
| **Failed** | Yes (`add_failed_task`) | No | Generator conditioning ("beyond frontier") |

**Stopping condition (ANNECS-style):** count solved-after-biting scenarios only. Primary stop = dev held-out plateau (deferred); current stop = budget ceiling N or iteration count.

### Why this matters (the counterfactual problem)
Loop 1 ratchets up until the *chronicle-equipped* agent fails attempt-1. So every archived success is by construction one the chronicle could not already handle → the chronicle was load-bearing → no per-scenario counterfactual needed during curriculum.

---

## 7. The Skills Chronicle

Each archived scenario accumulates a **Skills Chronicle** — a set of structured entries:

```xml
<Entry id="SCENARIO_ID_N">
  <Condition>Abstract social dynamic pattern — no proper nouns, generalisable</Condition>
  <Guidance>
    1. Primary guidance: [what to do or not — observable behavioral change]
    2. Warning (optional): [specific tempting behavior that backfires non-obviously]
    3. Exception: when [circumstance within Condition makes primary guidance wrong], do [alternative]
    (add more exceptions as needed)
    Note: Later clauses take precedence.
  </Guidance>
  <Type>HEURISTIC | WARNING</Type>
  <Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
  <Provenance>scenario IDs and iteration numbers</Provenance>
</Entry>
```

**Guidance is prescriptive** — specific enough that an agent reading it *before a conversation* would behave observably differently. "Be more strategic" is not guidance.

At inference time: retrieve the K most similar archived scenarios (by scenario embedding), inject their chronicles as a memory block into the agent's prompt. This is pure in-context learning — no weight updates.

**Reflection diagnosis** uses the rubric check results to name the actual failure pattern:
- Outcome✗ → didn't achieve the ask
- Outcome✓, Constraint✗ → **hollow extraction** (took the shortcut, paid the relational price)
- Both✗ where constraint is relational → **attunement failure** — the LLM defaulted to problem-solving mode when validation-first was needed (NOT "discomfort" — that's a human anthropomorphization)

---

## 8. Codebase Map

```
social_omni_epic/
  data_models.py          SocialScenario, StructuredGoal, RubricCheck, SuccessRubric, AgentProfile
  validation.py           render_agent_goal(), validate_scenario(), dict_to_scenario()
  task_generator.py       TaskGenerator: generate_*, edit_scenario(intent), analyze_too_easy
  curriculum.py           run_episode_two_loop(), run_coherence_gate(), build_episode_inputs()
                          ← SHARED ENGINE: imported by run_phase2.py AND run_debug.py
  episode_runner.py       run_single_episode(), _evaluate_rubric(), _evaluate_diagnostics()
                          EpisodeResult (rubric_results, outcome_achieved, constraint_preserved, goal_achieved)
  success_detector.py     SuccessDetector.is_solved() — prefers goal_achieved (AND of rubric checks)
  skills_chronicle.py     SkillsChronicle, ChronicleEntry, format_for_prompt()
  reflection_module.py    ReflectionModule.reflect() — diagnoses hollow-extraction / attunement / outcome failure
  adversarial_agent.py    AdversarialAgent.check_reflection(), check_final()
  meta_reflection.py      MetaReflectionModule.synthesize() — HEURISTIC vs WARNING dominant
  model_of_interestingness.py  MoI auditor+editor: social tension + novelty + learnability
  coherence_check.py      CoherenceChecker — structural + rubric validity + ZOPA check
  scenario_title.py       ScenarioTitleGenerator, designate_target_agent()
  archive.py              Archive (UCB1 selection, add_successful/failed/discarded, checkpointing)
  seeds.py                load_sotopia_seeds_with_embeddings()
  embedding_utils.py      get_similar_scenarios(), compute_cell_coverage()
  fm.py                   FM (foundation model wrapper, query_json, get_embeddings)
  sotopia_bridge.py       scenario_to_sotopia_profiles()

scripts/
  run_phase2.py           Curriculum expansion loop (uses curriculum.py engine)
  run_debug.py            Single-scenario debug (uses curriculum.py engine)
  run_go_nogo.py          Headroom + reflect-act gap experiment (go/no-go checkpoint)
  run_phase0.py           Phase 0: generation only, no episodes

configs/
  social_omni_epic_phase2.yaml   Main config (D=2, K=4, judge.self_consistency_k=3, etc.)

docs/
  scenario_design_sweet_spot.md  Rationale for the three-part goal structure (some sections outdated)
  evaluation_methodology.md      Internal vs external eval, SOTOPIA-EVAL as external metric
  curriculum_loop.md             Algorithm spec (counter values now outdated: D=2, K=4 in code)
  project_reference.md           ← THIS DOCUMENT (authoritative)
```

---

## 9. Key Design Decisions (What Changed from Earlier Designs and Why)

### No `gate` enum or region taxonomy
**Old design:** `gate ∈ {AND, FUSED}`, branching on `goal_type`. **Actual:** universal AND gate. Fused/relational scenarios (support, repair) are just ones where the two checks co-move — AND gives the right answer anyway (the hollow-extraction cell is empty). `goal_type` is a descriptive label only, never control flow. This removed a per-scenario classification that reviewers could attack.

### No degeneracy floor
**Old design:** believability/social_rules floor as a third gate condition. **Actual:** dropped. The partner-perspective constraint judge already catches abusive/coerced "wins." Gibberish fails the rubric naturally. Adding a threshold-based floor reintroduced the arbitrary `goal>7` style problem we'd rejected.

### `edit_scenario(intent)` not a separate `task_editor.py`
**Old design:** a new `task_editor.py` module. **Actual:** the existing `patch_scenario` generalized into `TaskGenerator.edit_scenario(scenario, feedback, intent)` with three intents: `fix_coherence`, `improve_interestingness`, `raise_difficulty`. One method, one module, less surface area.

### Shared `curriculum.py` engine
`run_phase2.py` and `run_debug.py` both import from `social_omni_epic/curriculum.py`. This eliminates logic drift (the two-loop, archive policy, reflection loop exist in exactly one place). `run_debug` now shows the full difficulty loop (Loop 1 records) in its JSON output.

### MoI upgraded to auditor+editor
MoI was a pure binary gate that defaulted to pass on error and never checked social tension. Now it:
1. Checks **social tension first** (does the constraint bite? is the shortcut tempting? does the naive move fail?) in addition to novelty + learnability
2. Returns `{verdict, reason, suggested_edits}` — if below bar, routes to `edit_scenario(intent="improve_interestingness")` up to `moi.max_edits=2` times before discarding
3. Fixes the error-default-to-pass bug

### Rubric questions authored at generation, frozen per scenario
The LLM writes the check questions once at generation time. The same questions are used for all attempts 1..K and across all difficulty edits. This is what makes per-attempt scores comparable.

### Partner judge self-consistency (not confidence-threshold escalation)
For `perspective="partner"` checks: always sample K=3 times at temperature 0.7, take majority vote. The agree-fraction (`n_agree/k`) is the uncertainty signal, visible in the debug output. The `confidence` field is for auditing only — it does not trigger conditional re-sampling.

---

## 10. Config Reference

`configs/social_omni_epic_phase2.yaml` — key knobs:

```yaml
max_attempts: 4          # K: biting failure is attempt 1; 3 reflection-driven retries
difficulty:
  D: 2                   # max scenario edits before discarding (≤3 scenario versions)
  re_gate_after_edit: true
judge:
  self_consistency_k: 3  # partner-perspective check: 3 samples, majority vote
moi:
  max_edits: 2           # MoI auditor: edit-up attempts before discarding
stopping:
  N: null                # stop at N solved-after-biting scenarios (null = run all iterations)
goal_threshold: 7.0      # legacy/mock fallback only; real gate is the rubric AND
```

---

## 11. Running the System

```bash
# Single-scenario debug (inspect the full loop, structured goal, rubric verdicts)
python scripts/run_debug.py
python scripts/run_debug.py --skip-episode   # generation + gates only, no API episode call

# Full curriculum expansion
python scripts/run_phase2.py
python scripts/run_phase2.py run_mode=phase0  # generation only, no episodes

# Go/no-go checkpoint (does the model have headroom + a reflect-act gap?)
python scripts/run_go_nogo.py --n-scenarios 20
```

The debug JSON (`debug_log/debug_*.json`) is structured as:
```
generated_scenario       ← structured goals, rubric, goal_type, rendered agent_goals
moi_audit               ← verdict, reason, suggested_edits
difficulty_loop[]       ← per-edit: slack_knob, suggested_edit, before/after goals, rubric_results
skill_attempts[]        ← per-attempt: transcript_clean, rubric_results, diagnostics_scores, reflection
meta_reflection         ← final chronicle entries
terminal_state          ← "discarded" | "solved_after_biting" | "failed"
archive_disposition
```

---

## 12. What Is Not Yet Built

- **External evaluation script** (`run_external_eval.py`) — compare our method vs baselines on held-out scenarios using standalone SOTOPIA-EVAL. Baselines: Vanilla LLM, CoT, ExpeL (insights from fixed SOTOPIA trajectories), Fine-tuning (SOTOPIA-PI). Metric: all 7 SOTOPIA dims continuous, `goal` as headline, paired stats.
- **Held-out split** (`split_seeds_by_dynamic.py`) — carve sealed subset B from the 800 SOTOPIA-PI envs, disjoint from the 90 seeds, split by social dynamic.
- **Archive-size scaling curve** (`run_dev_scaling_curve.py`) — held-out score vs archive size to validate the "more experience → better" premise.
- **Retrieval by condition embedding** — currently skills are retrieved by scenario-level embedding; retrieval keyed on the condition field itself would be more precise when the archive grows large (deferred until 30+ entries).

---

## 13. Literature Anchors

- **OMNI-EPIC** (Faldor et al., ICLR 2025) — the physical task generation framework we adapt
- **SOTOPIA** (Zhou et al., 2024) — the social interaction evaluation framework; provides seeds, agent profiles, and the 7-dim evaluation dimensions
- **SOTOPIA-PI** (Wang et al., 2024) — provides the fine-tuning baseline and the 800 environment profiles for held-out evaluation
- **ExpeL** (Zhao et al., 2024) — the in-context baseline that extracts insights from a fixed set of trajectories
- **Dillard, Segrin & Harden (1989)** — primary/secondary goals framework (outcome = primary; constraint = secondary)
- **Clark & Delia (1979)** — instrumental / relational / identity goal types
- **Brown & Levinson (1987)** — face-threatening acts; the shortcut = the FTA
- **Burleson & Goldsmith** — person-centeredness in supportive communication (Region B / attunement scenarios)
