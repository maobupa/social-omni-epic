#!/usr/bin/env bash
# Option B RESUME — stages 3-5 only (Base90 + Random90 banks already complete).
# Runs: Gen-90 curriculum -> extract -> final 4-condition eval.
# Same fixed config as the main driver: learner=gpt-4.1-mini, teacher/generator=gpt-5-mini,
# partner=gpt-5-mini, judge=gemini. Resumable (curriculum by run_name, eval by cached episodes).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="uv run"

LEARNER=openai/gpt-4.1-mini
TEACHER=openai/gpt-5-mini
PARTNER=openai/gpt-5-mini
JUDGE=google/gemini-3-flash-preview

LOGS=results/logs_option_b_41mini
mkdir -p "$LOGS"
ts() { date "+%Y-%m-%d %H:%M:%S"; }

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
