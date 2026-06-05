# Social OMNI-EPIC: Research Framing & Contribution

## The Core Claim

> **A model-agnostic, weight-free framework for inference-time social skill improvement: given any LLM at its current capability frontier, an open-ended curriculum self-calibrates to that model's difficulty level, generates structured social experience, and injects it as a skills chronicle at inference time — producing measurable improvements on held-out social scenarios without touching model weights.**

This is distinct from all existing work in a specific and defensible way:

---

## Why This Is Independent of Model Strength

The key mechanism is **Loop 1: difficulty calibration**. The curriculum does not generate scenarios at a fixed difficulty level — it ratchets up difficulty until the specific model being trained on fails. This means:

- GPT-4o-mini fails on different scenarios than GPT-5-mini
- The curriculum responds by generating scenarios that challenge THAT model specifically
- As models improve across generations, the curriculum generates harder, more nuanced scenarios
- The improvement signal (skills chronicle) is always calibrated to the model's current frontier

**Analogy:** Deliberate practice in human skill acquisition. An expert violinist and a beginner both improve from practice, but the practice material is different — the system self-selects the right difficulty level automatically.

This makes the contribution **scale-invariant**: it does not become irrelevant as frontier models improve. It just generates harder social scenarios and produces richer chronicles. The framework is a perpetual improvement engine, not a one-time fix for a specific model's weaknesses.

---

## Contribution Hierarchy

### Primary Contribution
An open-ended social curriculum framework with three components that collectively enable inference-time social skill improvement:

1. **Structured scenario generation** — three-part goal structure (outcome/constraint/shortcut) that operationalizes social difficulty via face-threat theory (Brown & Levinson) and goal theory (Dillard). Generates scenarios where naive behavior fails and skilled behavior succeeds.

2. **Difficulty-calibrated curriculum** — Loop 1 ratchets scenario difficulty until the specific learner model fails on attempt 1. This self-calibration ensures the curriculum always produces genuine learning events, regardless of model capability.

3. **Skills chronicle** — structured XML entries (Condition/Guidance/Type/Dimension) distilled from failures and successes, injected at inference time. More transferable than raw trajectory replay (ExpeL) because entries are abstracted to structural patterns, not scenario-specific scripts.

### Secondary Contribution
An evaluation methodology that goes beyond SOTOPIA-EVAL's goal dimension: the rubric-based AND gate (outcome ✓ + constraint ✓) explicitly penalizes hollow extraction — achieving the goal while violating the relational constraint — which SOTOPIA-EVAL's continuous GOAL score does not penalize sufficiently.

---

## How This Relates to Prior Work

| Method | Weight updates? | Fixed scenario distribution? | Self-calibrating difficulty? |
|--------|----------------|------------------------------|------------------------------|
| **SOTOPIA-PI** | Yes (QLoRA) | Yes (100 generated + 90 SOTOPIA) | No |
| **ExpeL** | No | Yes (fixed SOTOPIA trajectories) | No |
| **Reflexion** | No | Yes (same scenario retried) | No |
| **Ours** | No | No (open-ended generation) | **Yes** |

**vs. SOTOPIA-PI:** Their method requires weight access and a fixed training distribution. Ours works for API-only models, continually improves as more scenarios are added, and doesn't require a fine-tuning budget. The comparison is: can inference-time ICL from a self-generated curriculum approach the performance of fine-tuning?

**vs. ExpeL:** ExpeL extracts insights from a fixed set of SOTOPIA trajectories using UPVOTE/DOWNVOTE/EDIT/ADD operations. Ours generates scenarios beyond the SOTOPIA distribution, calibrates difficulty to the model, and produces structured chronicles via reflection + adversarial checking. The comparison is: does open-ended curriculum generation produce better inference-time guidance than fixed-distribution insight extraction?

**vs. Reflexion:** Reflexion reflects within a single task and doesn't accumulate cross-task knowledge. Ours accumulates a persistent skills chronicle across many scenarios, injected as prior experience into new scenarios.

---

## The Strongest Evaluable Claim for a Paper

**"An open-ended, difficulty-calibrated social curriculum produces skills chronicles that improve LLM social intelligence on held-out SOTOPIA scenarios at inference time, outperforming fixed-distribution ICL baselines (ExpeL), without weight updates."**

Supporting claims:
1. The curriculum generates scenarios that are genuinely challenging (the difficulty calibration loop produces biting failures)
2. The skills chronicle from those failures transfers to held-out scenarios (different surface context, same structural challenge)
3. The improvement is specifically on REL and GOAL dimensions — not just instrumental success but relational success (directly corresponds to the constraint/shortcut dynamic)
4. Open-ended curriculum > fixed SOTOPIA trajectories as a source of skills (the diversity and difficulty calibration add value over ExpeL)

---

## What Makes This Publishable at a Strong Venue

### Necessary conditions
1. **Headroom confirmed**: The learner model fails a meaningful fraction of held-out SOTOPIA scenarios at baseline (GOAL < 7 on SOTOPIA-HARD)
2. **Chronicle transfer confirmed**: Scenarios solved-after-biting produce chronicles that improve performance on HELD-OUT SOTOPIA scenarios (not the same scenarios)
3. **Baseline comparison**: Our method > ExpeL on SOTOPIA-HARD (GOAL and REL dimensions)
4. **Curriculum ablation**: Open-ended generated scenarios > random SOTOPIA scenarios as chronicle source (shows the curriculum generation adds value)

### Sufficient conditions (strengthens the paper)
5. **Scaling curve**: Performance improves as archive size grows (more past experience → better performance)
6. **Human eval on a subset**: Given SOTOPIA-PI's finding that GPT-4 eval overestimates fine-tuned models, some human evaluation on the hardest scenarios
7. **Qualitative analysis**: Show that the skills chronicle contains generalizable social principles, not scenario-specific scripts — the abstraction quality matters

### What we're NOT claiming
- We do not claim to beat SOTOPIA-PI's absolute numbers (fine-tuning has an inherent advantage)
- We do not claim to replace evaluation — SOTOPIA-EVAL is our metric, not a contribution
- We do not claim the approach is better for all social tasks — it's specifically designed for high-constraint relational scenarios where the shortcut/constraint dynamic is load-bearing
- We do not claim to produce "believable" agents in a human-facing sense — believability is already high at baseline (agents already sound natural); what fails is goal + relational success, which is what we target
- We do not claim the chronicle strategies generalize from LLM partners to human partners without further validation — that gap is real and untested
- The primary contribution is the **framework and its properties** (open-ended generation, difficulty calibration, structured distillation), not the size of the absolute performance delta; the delta is evidence the framework works, not the claim itself
- The most defensible practical context is AI-mediated practice settings (negotiation training, conflict resolution roleplay) where the agent plays a sparring partner, not a deployed conversational agent talking to real users

---

## Theoretical Grounding (Why Reviewers Should Believe It)

The three-part goal structure (outcome/constraint/shortcut) has direct grounding in social science:
- **Outcome** → Dillard's primary (influence) goal
- **Constraint** → Dillard's secondary goals (relational, identity) = Clark & Delia's three goal types
- **Shortcut** → Brown & Levinson's Face-Threatening Act (FTA)

This is not arbitrary prompt engineering. The structure operationalizes decades of social psychology research on what makes social situations genuinely difficult. Scenarios without a biting constraint (trivially satisfiable by polite behavior) are not social intelligence tests — they're cooperation tasks.

The skills chronicle format (abstract Condition + prescriptive Guidance) is grounded in cognitive apprenticeship literature: expertise is encoded as condition-action rules transferable across surface contexts, not as episodic memories of specific situations.
