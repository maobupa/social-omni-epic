#!/usr/bin/env bash
# End-to-end matrix ramp: phase-0 -> generate -> crossplay -> matrix, for one ramp level.
#
#   RAMP=smoke ./scripts/run_matrix_ramp.sh     # 3 seeds, one per band  (~1-2 h)
#   RAMP=pilot ./scripts/run_matrix_ramp.sh     # 20 stratified seeds
#   RAMP=full  ./scripts/run_matrix_ramp.sh     # all 90
#
#   DRY=1 RAMP=smoke ./scripts/run_matrix_ramp.sh    # stop after the free calibration audit
#
# Everything resumes. Deterministic child ids + resume-by-existence mean smoke output rolls into
# pilot, and pilot into full, generating only what is missing — PROVIDED no code or roster change
# happens in between. build_matrix.py checks commit provenance and warns if sets disagree.
#
# ─────────────────────────────────────────────────────────────────────────────────────────────
# WHY PHASE-0 IS RE-RUN RATHER THAN REUSED
#
# results/expel_phase0_Base90_ExpeL already has all 90 bands for gpt-5-mini — but it was produced
# with partner_model = gpt-5-mini (self-play), whereas the matrix freezes the partner at
# gpt-5.4-mini. Reusing it would mean gpt-5-mini's bands were measured against a different partner
# than gpt-4o-mini's, i.e. Row 0 (the saturation claim) would carry the exact confound the frozen-
# partner rule exists to prevent. On a 3- or 20-seed subset re-running is cheap, so we re-run.
#
# The old run is still used for ONE thing: choosing WHICH seeds to include (pick_ramp_seeds.py
# stratifies on its bands so all three operators fire). That is a coverage heuristic and does not
# need to be partner-matched. The bands actually used for calibration always come from the fresh run.
# ─────────────────────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="uv run"

RAMP="${RAMP:-smoke}"
DRY="${DRY:-0}"
CONCURRENCY="${CONCURRENCY:-4}"
OUT_ROOT="${OUT_ROOT:-results/matrix_v1}"
LOGS="$OUT_ROOT/logs"
mkdir -p "$LOGS" "$OUT_ROOT/phase0"
ts() { date +%Y-%m-%dT%H:%M:%S; }
say() { echo "[$(ts)] $*"; }

# --- frozen roster (mirrors run_grid_matrix.sh; see there for the reasoning) ------------------
LEARNERS=("gpt4omini:gpt-4o-mini" "gpt5mini:gpt-5-mini")   # the 2x2
GENERATOR="gpt-5.4"
PARTNER="gpt-5.4-mini"
GATES="gpt-4.1-mini"
JUDGE="google/gemini-3-flash-preview"
STRATIFY_FROM="results/expel_phase0_Base90_ExpeL"   # seed CHOICE only, never bands

say "ramp=$RAMP  learners=${LEARNERS[*]}  partner=$PARTNER  judge=$JUDGE  concurrency=$CONCURRENCY"

# ── Stage 0: the seed subset, in both identifier forms ────────────────────────────────────────
SEED_INDICES="$($PY scripts/pick_ramp_seeds.py "$RAMP" --bands-from "$STRATIFY_FROM" --emit indices 2>/dev/null)"
SEED_IDS="$($PY scripts/pick_ramp_seeds.py "$RAMP" --bands-from "$STRATIFY_FROM" --emit ids 2>/dev/null)"
N_SEEDS="$(echo "$SEED_IDS" | tr ',' '\n' | grep -c .)"
say "subset: $N_SEEDS seed(s)"
$PY scripts/pick_ramp_seeds.py "$RAMP" --bands-from "$STRATIFY_FROM" 2>/dev/null | sed 's/^/         /'

# ── Stage 1: phase-0 per learner = Row 0 AND the bands the operator is chosen from ────────────
for entry in "${LEARNERS[@]}"; do
  tag="${entry%%:*}"; model="${entry##*:}"
  p0="$OUT_ROOT/phase0/$tag"
  base="$OUT_ROOT/phase0_baseline/${tag}_${RAMP}"

  if [ -d "$p0/seeds" ] && [ "$(ls -1 "$p0/seeds" 2>/dev/null | wc -l | tr -d ' ')" -ge "$N_SEEDS" ]; then
    say "phase-0 $tag: already covers >= $N_SEEDS seed(s) — skipping"
    continue
  fi
  if [ "$DRY" = "1" ]; then
    say "phase-0 $tag: WOULD run baseline + K-loop over $N_SEEDS seed(s) (DRY=1, skipping)"
    continue
  fi

  say "phase-0 $tag: cold pass over $N_SEEDS seed(s)"
  mkdir -p "$(dirname "$base")"
  $PY scripts/run_baseline_eval.py \
      --seed-indices "$SEED_INDICES" \
      --learner-model "$model" --partner-model "$PARTNER" --judge-model "$JUDGE" \
      --output-dir "$base" \
      2>&1 | tee "$LOGS/1a_baseline_${tag}_${RAMP}.log" | tail -3

  say "phase-0 $tag: K-loop + LP + bands"
  # --out points straight at the matrix tree, and --resume means a later, larger ramp only adds
  # the seeds it is missing.
  $PY scripts/run_expel_chronicle.py \
      --baseline "$base" \
      --seed-indices "$SEED_INDICES" \
      --learner-model "$model" --reflection-model "$model" \
      --partner-model "$PARTNER" --judge-model "$JUDGE" \
      --success-rule goal \
      --out "$p0" --resume \
      2>&1 | tee "$LOGS/1b_phase0_${tag}_${RAMP}.log" | tail -5
done

# ── Stage 2: free audit — does every seed get the operator its band implies? ──────────────────
say "calibration audit (no episodes)"
for entry in "${LEARNERS[@]}"; do
  tag="${entry%%:*}"; model="${entry##*:}"
  [ -d "$OUT_ROOT/phase0/$tag/seeds" ] || { say "  $tag: no bands yet — skipping"; continue; }
  echo "  --- $tag ---"
  $PY scripts/run_grid_generate.py \
      --phase0-dir "$OUT_ROOT/phase0/$tag" \
      --learner-tag "$tag" --learner-model "$model" \
      --generator-model "$GENERATOR" --gates-model "$GATES" \
      --partner-model "$PARTNER" --judge-model "$JUDGE" \
      --out "$OUT_ROOT/sets/$tag" \
      --seed-ids "$SEED_IDS" --dry-run 2>/dev/null | sed 's/^/    /' | tail -4
done

if [ "$DRY" = "1" ]; then
  say "DRY=1 — stopping after the audit."
  exit 0
fi

# ── Stage 3: generate one set per learner ────────────────────────────────────────────────────
for entry in "${LEARNERS[@]}"; do
  tag="${entry%%:*}"; model="${entry##*:}"
  [ -d "$OUT_ROOT/phase0/$tag/seeds" ] || { say "no bands for $tag — cannot generate"; continue; }
  say "generating set for $tag ($N_SEEDS cell(s))"
  $PY scripts/run_grid_generate.py \
      --phase0-dir "$OUT_ROOT/phase0/$tag" \
      --learner-tag "$tag" --learner-model "$model" --reflection-model "$model" \
      --generator-model "$GENERATOR" --gates-model "$GATES" \
      --partner-model "$PARTNER" --judge-model "$JUDGE" \
      --out "$OUT_ROOT/sets/$tag" \
      --seed-ids "$SEED_IDS" --concurrency "$CONCURRENCY" \
      2>&1 | tee "$LOGS/3_generate_${tag}_${RAMP}.log" | tail -8

  # Abort the ramp if the generator is producing unwinnable scenarios faster than the gate filters
  # them: more compute will not fix that, only a better generation prompt will.
  y="$($PY - <<PYEOF 2>/dev/null
import json,pathlib
p=pathlib.Path("$OUT_ROOT/sets/$tag/grid_manifest.json")
print(json.loads(p.read_text()).get("oracle_yield") or 0 if p.exists() else 0)
PYEOF
)"
  awk -v y="$y" 'BEGIN{ if (y+0 < 0.25 && y+0 > 0) exit 1 }' || {
    say "ABORT: oracle yield for $tag is $y (< 0.25). Fix the generation prompt, not the compute."
    exit 1
  }
done

# ── Stage 4: crossplay (off-diagonal) + diagonal extraction ──────────────────────────────────
say "crossplay: all pairs"
$PY scripts/run_grid_crossplay.py --all --matrix-root "$OUT_ROOT" \
    --partner-model "$PARTNER" --judge-model "$JUDGE" --aux-model "$GATES" \
    --concurrency "$CONCURRENCY" \
    2>&1 | tee "$LOGS/4_crossplay_${RAMP}.log" | tail -12

# ── Stage 5: the figure ──────────────────────────────────────────────────────────────────────
$PY scripts/build_matrix.py --matrix-root "$OUT_ROOT" --csv 2>&1 | tee "$LOGS/5_matrix_${RAMP}.log"

say "ramp=$RAMP complete. Artifacts under $OUT_ROOT/analysis/."
say "Next: RAMP=pilot (then full) — resume means only the missing cells are generated."
