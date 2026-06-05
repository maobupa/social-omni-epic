# Empirical Roadmap: From Current State to Publishable

## Where We Are Now

The curriculum pipeline (generation → difficulty calibration → skill learning → meta-reflection) is implemented and debugged. The primary blocker is that gpt-5-mini solves generated scenarios too easily in Loop 1, causing 100% discard rate. We have pending fixes (partner prompt, MoI/coherence tightening).

**Before fixing anything further: run the Step 1 sanity check.** It determines which path to take.

---

## Step 1: Baseline Sanity Check (1-2 hours)

**What:** Run gpt-5-mini on the original 90 SOTOPIA seeds using original flat goals (no structured goals, no chronicle, no rubric). Score with SOTOPIA-EVAL 7 dimensions. Record GOAL and REL.

**How:** Use `run_phase1.py` with `run_mode=phase1_no_memory` on the 90 seeds. This runs real SOTOPIA episodes with flat goals and produces SOTOPIA-EVAL scores. No chronicle injection.

**Why first:** If gpt-5-mini is already at ceiling (GOAL > 7.5), the entire debugging effort on the curriculum pipeline is solving the wrong problem for this model.

**Decision point:** Three outcomes, three paths.

---

## Path A: gpt-5-mini is at ceiling (GOAL > 7.5 on 90 seeds)

**What this means:** The model already performs at or above SOTOPIA-PI BC+SR level. There is no room to improve on general SOTOPIA scenarios.

**Options:**

**A1 — Switch to SOTOPIA-HARD as eval target**
The 14-20 hardest SOTOPIA scenarios (GPT-4 gets ~5.89 on these). Even strong models struggle here. Re-run Step 1 on SOTOPIA-HARD only. If GOAL < 7 on SOTOPIA-HARD, proceed with Path B using SOTOPIA-HARD as the held-out eval set.

**A2 — Switch learner model**
Use gpt-4o-mini or Mistral-7B as the learner. This is the SOTOPIA-PI setup. More headroom, clearer room for improvement. The contribution is still valid — "our framework improves weaker models at inference time, approaching GPT-5-mini performance without weight updates."

**A3 — Reframe the contribution**
If no model has meaningful headroom on SOTOPIA, reframe as a **data generation contribution**: the curriculum produces high-quality hard social scenarios that can be used for fine-tuning (SOTOPIA-PI style) by the community. This is still a useful paper but shifts the evaluation focus.

**Recommendation:** Try A1 first (SOTOPIA-HARD). If still at ceiling, A2.

---

## Path B: gpt-5-mini has headroom (GOAL 5-7 on target eval set)

This is the green light. Proceed in order:

### B1 — Fix the 100% discard rate (1-2 days)

Current problem: gpt-5-mini solves all generated scenarios on attempt 1, so Loop 1 discards everything.

Priority fixes (from the three-layer plan):
1. **Partner turn prompt** (`episode_runner.py`) — remove "OR reached a clear agreement" leave condition, add explicit goal-pursuit instruction. This is the highest-leverage change.
2. **MoI + coherence checks** — add ZOPA-too-large and constraint-trivially-satisfiable checks to catch structurally easy scenarios at generation time.

Verification: run `run_debug.py` and confirm at least one scenario produces `terminal_state: solved_after_biting` or `terminal_state: failed` instead of `discarded`.

### B2 — Minimum viable experiment (2-3 days)

Generate curriculum scenarios until you have **5 solved-after-biting events** (not 5 total scenarios — 5 that actually bit and produced skills). This is the minimum to confirm the architecture works end-to-end.

Then: evaluate gpt-5-mini on the held-out SOTOPIA target scenarios (from Step 1) **with vs. without** the skills chronicle injected. Run SOTOPIA-EVAL 7 dimensions on both conditions.

**Decision point:** Does the chronicle improve GOAL and/or REL?
- Yes → confirm the core claim, proceed to B3
- No → the skills are not transferring; investigate whether chronicle entries are too scenario-specific (check abstraction quality); may need to tune reflection prompts

### B3 — Scale up to publishable N (1-2 weeks)

Once B2 confirms the basic signal:
1. Generate N=50-100 solved-after-biting curriculum scenarios
2. Build the full skills chronicle
3. Evaluate on full held-out SOTOPIA-HARD set
4. Implement and run baselines

### B4 — Implement baselines

**Baseline 1 (floor): Vanilla gpt-5-mini**
No chronicle, no ICL. Already have from Step 1.

**Baseline 2 (main comparison): ExpeL-style**
Run 90 SOTOPIA episodes with gpt-5-mini, extract insights using UPVOTE/DOWNVOTE/EDIT/ADD from transcripts, inject insights at eval time. This is the "fixed-distribution ICL" baseline. Our method should beat this by showing open-ended curriculum > fixed SOTOPIA trajectories.

**Baseline 3 (retrieval-only ablation)**
Inject raw transcripts from the curriculum (no reflection, no chronicle distillation) at eval time. Tests whether the reflection + chronicle structure adds value over naive transcript retrieval.

**Baseline 4 (ceiling): SOTOPIA-PI BC+SR**
Their fine-tuned 7B model result. We probably won't beat this in absolute numbers, but showing inference-time ICL approaches fine-tuning performance is a strong result.

### B5 — Ablations

1. **No difficulty calibration**: Use random SOTOPIA seeds as the training scenarios (no Loop 1 difficulty calibration). Does the curriculum outperform this? This validates that difficulty calibration is necessary.

2. **Archive size scaling curve**: Plot SOTOPIA-HARD GOAL score vs. number of solved-after-biting scenarios in the archive. Should show monotonic improvement (more experience → better performance). This directly validates the "improvement scales with archive size" claim.

3. **Chronicle vs. raw transcript**: Already in B4 Baseline 3.

---

## Target Evaluation Setup

**Primary metric:** GOAL dimension on SOTOPIA-HARD (14-20 scenarios)
**Secondary metrics:** REL dimension (our method specifically targets this via constraint preservation)
**Evaluation protocol:** Same as SOTOPIA-PI — GPT-4 as evaluator, fixed partner model (GPT-3.5-turbo), both agents evaluated

**Held-out set management:**
- SOTOPIA-HARD scenarios: used ONLY for evaluation, never as curriculum seeds
- The 90-seed file (data/sotopia_90_seeds.jsonl): can be used as curriculum anchor seeds EXCEPT SOTOPIA-HARD scenarios
- The 884-entry file (data/sotopia_seeds/environment_profiles.jsonl): additional generation fodder

---

## Timeline Estimate

| Step | Time | Gating |
|------|------|--------|
| Step 1 (baseline check) | 1-2 hours | None — do first |
| B1 (fix discard rate) | 1-2 days | Step 1 shows headroom |
| B2 (minimum viable experiment) | 2-3 days | B1 produces biting scenarios |
| B3 (scale up) | 1-2 weeks | B2 confirms transfer |
| B4+B5 (baselines + ablations) | 1 week | B3 shows strong result |
| Paper writing | 1-2 weeks | All experiments done |

**If Step 1 shows gpt-5-mini is at ceiling:** add ~3 days to switch learner model and re-run Step 1 before proceeding.

---

## What NOT to Optimize Right Now

Until Step 1 and B2 are done, do NOT spend time on:
- Further prompt engineering of the curriculum generation (task_generator.py)
- Tightening rubric authoring instructions
- Coherence checker improvements beyond the three-layer fix
- Archive size scaling experiments
- External evaluation infrastructure

The curriculum pipeline is good enough to generate some biting scenarios once the partner prompt is fixed. Perfecting it before confirming the core claim is premature.

---

## The Minimum Publishable Result

The weakest result that would still make a publishable paper:

1. gpt-5-mini improves on SOTOPIA-HARD by ≥0.5 GOAL points with chronicle vs. without (effect size matters more than p-value given small N)
2. Our method > ExpeL-style baseline on GOAL or REL
3. At least one ablation showing open-ended curriculum > fixed distribution

If we can show this on SOTOPIA-HARD with N=50-100 curriculum scenarios, that's publishable at a workshop level. Add the scaling curve and human eval on 5-10 scenarios for a main conference (ACL, EMNLP, NeurIPS).
