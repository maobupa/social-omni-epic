# Social OMNI-EPIC: Experiment Plan

> **Audience:** implementation guide for running the comparison. The *why* behind every design choice lives in `evaluation_methodology.md`. This document is the *what* and *how*.

---

## 1. The Comparison

One question: does an open-ended difficulty-calibrated curriculum produce better inference-time social skill than extracting insights from a fixed set of SOTOPIA scenarios?

| Condition | Abbrev | Training source | ICL at test time |
|---|---|---|---|
| Vanilla LLM | **VAN** | — | Nothing |
| Chain-of-thought | **COT** | — | Fixed social reasoning prompt only |
| ExpeL (SOTOPIA seeds) | **EXP** | Runs episodes on 90 SOTOPIA seeds; extracts insights via UPVOTE/DOWNVOTE/EDIT/ADD | Full insight list + top-k similar SOTOPIA transcripts |
| **Ours** | **OUR** | Runs difficulty-calibrated curriculum on ~N generated scenarios (seeded from 90 SOTOPIA); extracts insights via UPVOTE/DOWNVOTE/EDIT/ADD applied to generated transcripts | Full insight list + top-k similar generated transcripts |
| SOTOPIA-PI (reference) | **SPI** | Published fine-tuning result | Published numbers only — do not re-run |

**Fixed across all live conditions:** same learner model, same partner model, same evaluator model, same turn budget, same test scenarios.

**One variable between EXP and OUR:** the curriculum source. Both use identical ExpeL extraction (UPVOTE/DOWNVOTE/EDIT/ADD) and identical retrieval format (full insight list + top-k similar transcripts). The only difference is where the training transcripts come from — fixed SOTOPIA seeds (EXP) vs open-ended difficulty-calibrated generated scenarios (OUR). This isolates the curriculum contribution cleanly: if OUR beats EXP, it is because generated scenarios are higher-quality ICL material, not because of a different knowledge representation.

---

## 2. Evaluation Sets

### 2.1 Primary eval: 90 SOTOPIA seeds

Use all 90 SOTOPIA seeds as evaluation scenarios for every condition.

**Contamination note:** ExpeL directly trains on these same scenarios (runs episodes on them, extracts insights from those transcripts). Our method generates *from* them as anchors but never runs curriculum episodes on them. This gives ExpeL a structural advantage on this eval set — the 90 seeds are closer to EXP's training distribution than OUR's. If OUR still outperforms EXP here, the result is stronger than it looks.

Report this asymmetry explicitly in the paper.

### 2.2 Held-out eval: subset of SOTOPIA-PI 800 environments

Carve a sealed held-out set from the 800 SOTOPIA-PI environments (disjoint from the 90 seeds). Neither EXP nor OUR has seen these during training.

**How to split:**
- Split *by social dynamic*, not by surface scenario (see `evaluation_methodology.md §3.4`)
- Target ~100–150 scenarios for statistical power
- Run `split_seeds_by_dynamic.py` (to be built) to carve this out
- **Reserve this set now. Never touch it during development or debugging.**

This is the clean comparison. Primary eval (§2.1) is the convenient one; held-out eval (§2.2) is the defensible one.

---

## 3. Training / Generation Phase

### 3.1 ExpeL condition

1. For each of the 90 SOTOPIA seeds, run the learner model as an agent with the seed's scenario + goals
2. Allow up to Z=3 Reflexion retries per seed (match ExpeL's standard protocol)
3. Collect success/failure transcript pairs
4. Run ExpeL's insight extraction (UPVOTE/DOWNVOTE/EDIT/ADD on an accumulated insight list)
5. Store: final insight list + all successful transcripts (for episode-level retrieval at test time)

Total training episodes: 90 seeds × ~3 attempts = ~270 conversations.

### 3.2 Our condition

1. Use 90 SOTOPIA seeds as generation anchors (UCB1 weighted; optionally weight toward SOTOPIA-HARD seeds where baseline GOAL < 7)
2. Run the two-loop curriculum (D=2, K=4) on each generated scenario
3. Archive solved-after-biting scenarios and their transcripts
4. Stop when N solved-after-biting scenarios are archived (see §4 on target N)
5. Run ExpeL's insight extraction (UPVOTE/DOWNVOTE/EDIT/ADD) on all archived transcripts — exactly the same extraction pipeline as EXP, applied to generated-scenario transcripts

Total training episodes: highly variable per scenario (1–7 conversations per generated scenario). Report the total episode count alongside archive size for transparency.

---

## 4. Target Archive Size (N)

The primary experiment uses **N = 60 solved-after-biting scenarios** as the OUR training set.

Rationale: this is roughly episode-budget-comparable to EXP's ~270 conversations (60 scenarios × ~4.5 average conversations per scenario including difficulty edits). Exact parity is not the goal — the comparison is knowledge quality, not compute — but gross parity is fair.

Additionally, run the archive-size ablation (§6) which requires N = 20, 40, 60, 80 checkpoints. Plan curriculum runs to checkpoint the archive at those sizes.

---

## 5. Retrieval Protocol at Test Time

Both EXP and OUR inject ICL context into the learner's prompt before each test episode. The retrieval must be symmetric.

### 5.1 ExpeL retrieval
- Inject: full insight list (all extracted rules) + top-k=3 most similar successful SOTOPIA transcripts, retrieved by scenario embedding similarity
- This matches ExpeL's standard inference protocol

### 5.2 Our retrieval
- Inject: full insight list (all extracted rules from generated transcripts) + top-k=3 most similar generated transcripts, retrieved by scenario embedding similarity
- Same format as EXP — the insight list and transcripts both come from the generated archive instead of the SOTOPIA training set

### 5.3 Vanilla and CoT
- VAN: no additions to the base agent prompt
- COT: prepend a fixed social reasoning instruction ("Think step by step about the other person's goals, the relational cost of each move, and what genuine buy-in would require")

---

## 6. Metrics

### 6.1 Primary metric
SOTOPIA-EVAL 7 dimensions (continuous, 0–10):
- `goal` — headline metric
- `relationship` (REL)
- `believability`, `knowledge`, `secret`, `social_rules`, `financial_and_material_benefits`

Report all 7. Lead with `goal` and `REL` (these directly correspond to the outcome/constraint structure). Do not threshold into a binary success rate as the headline.

### 6.2 Paired statistics
- For each test scenario: each condition runs M=3 independent episodes (stochastic)
- Report per-condition mean ± 95% CI across scenarios × episodes
- Paired t-test (or Wilcoxon) within scenarios (same scenario, different conditions)
- Primary comparison: OUR vs EXP on `goal` and `REL`

### 6.3 Human eval (subset)
Randomly sample 15–20 test scenarios. For each:
- Show the transcript to a human rater
- Rate on: goal achievement (binary), relational quality (1–5 Likert)
- Report inter-rater agreement (Cohen's κ)
- Report SOTOPIA-EVAL automated score vs human rating correlation on this subset

This validates that the automated metric is tracking what we claim.

---

## 7. Ablations

Run these after the main comparison is confirmed:

| Ablation | What it tests |
|---|---|
| **Archive-size curve** | OUR with N=20, 40, 60, 80 → plot held-out `goal` vs N. Tests "more experience → better performance" |
| **Difficulty calibration** | OUR with D=0 (no difficulty ratchet; accept any scenario that bites on first try) vs D=2. Tests whether the calibration ratchet adds value beyond just having more generated scenarios |
| **Structured chronicle vs ExpeL extraction** (optional) | OUR with structured XML chronicles (Condition/Guidance/Type/Dimension) injected at inference vs OUR with ExpeL-style extraction — both on the same generated scenarios. Tests whether structured distillation outperforms unstructured bullet rules given the same curriculum source |

Note: "Generated vs SOTOPIA source" is **not** an ablation — it is the main comparison (OUR vs EXP). The primary experiment already tests this directly.

The most important ablation is **Difficulty calibration** — it verifies that D=2 ratcheting, not simply having open-ended scenarios, drives the improvement. If D=0 matches D=2, the difficulty calibration loop is incidental; if D=2 > D=0, the ratchet is load-bearing.

---

## 8. Implementation Order

Build in this order to fail fast:

1. **Baseline eval script** (`run_baseline_eval.py`) — run vanilla LLM on all 90 SOTOPIA seeds, collect SOTOPIA-EVAL scores. Confirms: (a) the eval pipeline works end-to-end, (b) headroom exists (model fails a meaningful fraction).

2. **ExpeL implementation** — run SOTOPIA-seed episodes, implement insight extraction, build inference-time retrieval. Run on 90 seeds. This is the primary baseline.

3. **OUR curriculum** — already partially built. Run to N=60 solved-after-biting scenarios from the 90 seeds.

4. **External eval script** (`run_external_eval.py`) — run all conditions on the 90 SOTOPIA seeds (primary eval), then on the held-out set. Compute SOTOPIA-EVAL scores, paired stats.

5. **Ablations** — run after the main comparison is confirmed positive.

Do **not** touch the held-out SOTOPIA-PI set until step 4.

---

## 9. What a Positive Result Looks Like

### Minimum bar (necessary for publication)
- OUR > VAN on `goal` on held-out set (p < 0.05) — the method improves on the vanilla baseline
- OUR > EXP on `goal` on held-out set, or OUR ≈ EXP on primary eval (where EXP has the contamination advantage) — the curriculum is at least as good as fixed-seed insight extraction

### Strong result (sufficient for a good venue)
- OUR > EXP on held-out set (p < 0.05) on both `goal` and `REL` — clean win on the uncontaminated benchmark
- Archive-size curve shows a positive slope — scaling behavior validates the core premise
- D=2 > D=0 ablation confirms the difficulty calibration ratchet is load-bearing, not incidental

### What to do if OUR ≈ EXP
Since both conditions use the same extraction and retrieval format, OUR ≈ EXP means the generated scenarios are not producing higher-quality ICL material than the SOTOPIA seeds — the curriculum hypothesis would be under threat. Diagnose:
- Check whether the generated archive scenarios are actually harder (measured by solved-after-biting rate) than the SOTOPIA training seeds
- Check whether the generated transcripts, given the same extraction, produce richer / more diverse insight rules than SOTOPIA transcripts
- Check whether retrieval is finding relevant examples (embedding similarity scores and retrieved content)
- Consider the structured chronicle ablation — if structured format outperforms ExpeL extraction on generated scenarios, switch OUR to chronicle and treat the extraction format as a variable
