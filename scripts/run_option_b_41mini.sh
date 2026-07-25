#!/usr/bin/env bash
# Option B — end-to-end re-derivation with ONLY the learner changed to gpt-4.1-mini.
#   learner  = openai/gpt-4.1-mini   (the single manipulated variable)
#   generator/extractor/reflection = openai/gpt-5-mini   (teacher held fixed)
#   partner  = openai/gpt-5-mini     (environment held fixed)
#   judge    = google/gemini-3-flash-preview  (ruler held fixed; cross-lab)
#
# Rebuilds all three banks (Base90, Random90, Gen-90/SOE) with the 4.1-mini learner,
# then runs the 4-condition head-to-head. New _41mini output dirs — never touches the
# gpt-5-mini artifacts. Resumable: re-run and each stage picks up where it left off.
#
# DO NOT launch while the Option A eval (results/eval_comparison_41mini) is still running —
# they contend for the same OpenAI rate limits. Give the Gen-90 curriculum a clean window.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="uv run"

LEARNER=openai/gpt-4.1-mini
TEACHER=openai/gpt-5-mini            # generator / extractor / reflection
PARTNER=openai/gpt-5-mini            # fixed environment
JUDGE=google/gemini-3-flash-preview  # fixed ruler

LOGS=results/logs_option_b_41mini
mkdir -p "$LOGS"
ts() { date "+%Y-%m-%d %H:%M:%S"; }

# ---------------------------------------------------------------------------
echo "===== [1/5] Base90 bank (learner=$LEARNER, partner=$PARTNER) =====  $(ts)"
$PY scripts/run_baseline_eval.py \
    --seeds-path data/sotopia_90_seeds.jsonl \
    --model "$LEARNER" --partner-model "$PARTNER" --judge-model "$JUDGE" \
    --output-dir results/baseline_eval_base90_41mini \
    2>&1 | tee "$LOGS/1_base90.log"
$PY scripts/run_expel_chronicle.py \
    --seeds data/sotopia_90_seeds.jsonl \
    --baseline results/baseline_eval_base90_41mini \
    --model "$LEARNER" --reflection-model "$TEACHER" --partner-model "$PARTNER" --judge-model "$JUDGE" \
    --run-name Base90_ExpeL_41mini --out results/expel_phase0_Base90_ExpeL_41mini --resume \
    2>&1 | tee -a "$LOGS/1_base90.log"
$PY scripts/run_expel_baseline.py extract \
    --model "$TEACHER" --out results/expel_phase0_Base90_ExpeL_41mini \
    2>&1 | tee -a "$LOGS/1_base90.log"

# ---------------------------------------------------------------------------
echo "===== [2/5] Random90 bank (learner=$LEARNER, partner=$PARTNER) =====  $(ts)"
$PY scripts/run_baseline_eval.py \
    --seeds-path data/sotopia_baseline_90.jsonl \
    --model "$LEARNER" --partner-model "$PARTNER" --judge-model "$JUDGE" \
    --output-dir results/baseline_eval_random90_41mini \
    2>&1 | tee "$LOGS/2_random90.log"
$PY scripts/run_expel_chronicle.py \
    --seeds data/sotopia_baseline_90.jsonl \
    --baseline results/baseline_eval_random90_41mini \
    --model "$LEARNER" --reflection-model "$TEACHER" --partner-model "$PARTNER" --judge-model "$JUDGE" \
    --run-name Random90_ExpeL_41mini --out results/expel_phase0_Random90_ExpeL_41mini --resume \
    2>&1 | tee -a "$LOGS/2_random90.log"
$PY scripts/run_expel_baseline.py extract \
    --model "$TEACHER" --out results/expel_phase0_Random90_ExpeL_41mini \
    2>&1 | tee -a "$LOGS/2_random90.log"

# ---------------------------------------------------------------------------
# Gen-90 seeds from the 4.1-mini Base90 calibration (built in stage 1).
echo "===== [3/5] Gen-90 / SOE curriculum (learner=$LEARNER, generator=$TEACHER) =====  $(ts)"
$PY scripts/run_curriculum.py run_name=gen90_expel_41mini \
    model="$TEACHER" learner_model="$LEARNER" partner_model="$PARTNER" \
    seed_from_phase0_dir=results/expel_phase0_Base90_ExpeL_41mini \
    use_expel_memory=true stopping.N=90 iterations=250 \
    2>&1 | tee "$LOGS/3_gen90_curriculum.log"

echo "===== [4/5] Extract Gen-90 insights (extractor=$TEACHER) =====  $(ts)"
$PY scripts/run_expel_baseline.py extract \
    --model "$TEACHER" --out results/gen90_expel_41mini \
    2>&1 | tee "$LOGS/4_extract.log"

# ---------------------------------------------------------------------------
echo "===== [5/5] 4-condition head-to-head (learner=$LEARNER, partner=$PARTNER, judge=$JUDGE) =====  $(ts)"
$PY scripts/run_eval_comparison.py \
    --eval-seeds  data/eval_candidates.jsonl \
    --base-bank   results/expel_phase0_Base90_ExpeL_41mini \
    --gen-bank    results/gen90_expel_41mini \
    --random-bank results/expel_phase0_Random90_ExpeL_41mini \
    --conditions  vanilla,random90,expel_base,ours \
    --learner-model "$LEARNER" --partner-model "$PARTNER" --judge-model "$JUDGE" \
    --top-k 2 --max-turns 20 --concurrency 4 \
    --out results/eval_comparison_b_41mini \
    2>&1 | tee "$LOGS/5_eval.log"

echo "===== OPTION B COMPLETE =====  $(ts)"
echo "Results: results/eval_comparison_b_41mini/comparison_summary.json"
