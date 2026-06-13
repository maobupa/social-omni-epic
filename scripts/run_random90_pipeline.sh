#!/usr/bin/env bash
# Random90 baseline pipeline (ExpeL), steps 1-3 — mirrors the ExpeL-Base90 bank.
# Stops at insights.json (like Base90); eval (step 4) is deferred.
# Launched under caffeinate+nohup; logs to results/random90_pipeline.log.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python

BASELINE_DIR=results/baseline_eval_random90
EXPEL_DIR=results/expel_phase0_Random90_ExpeL
SEEDS=data/sotopia_baseline_90.jsonl

echo "===== [1/3] vanilla attempt-1 baseline eval =====  $(date)"
$PY scripts/run_baseline_eval.py --seeds-path "$SEEDS" --output-dir "$BASELINE_DIR"

echo "===== [2/3] adapted ExpeL gather (reflexion + LP) =====  $(date)"
$PY scripts/run_expel_chronicle.py --seeds "$SEEDS" --baseline "$BASELINE_DIR" \
    --run-name Random90_ExpeL --out "$EXPEL_DIR" --resume

echo "===== [3/3] extract global insights =====  $(date)"
$PY scripts/run_expel_baseline.py extract --out "$EXPEL_DIR"

echo "===== STEPS 1-3 COMPLETE (eval deferred) =====  $(date)"
