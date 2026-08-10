#!/usr/bin/env bash
# Matrix experiment driver — the ONE place the frozen roster lives.
#
# Design (see the matrix plan): a paired grid. For each learner M, every one of the 90 SOTOPIA
# seeds produces exactly one child, with the mutation operator set by that seed's band FOR M
# (too_easy -> escalate, frontier -> lateral, beyond_frontier -> relax). That band->operator
# mapping is the entire learner-calibration mechanism. Everything else is frozen.
#
# Roster verified reachable by `uv run scripts/probe_models.py` on 2026-08-10; raw output in
# results/matrix_v1/model_probe.json.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="uv run"

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN ROSTER — do not vary these across cells. If the partner or judge drifts
# with the learner, difficulty rises with learner strength and cancels the exact
# monotonicity the matrix exists to measure.
# ─────────────────────────────────────────────────────────────────────────────

# The matrix axis. Two generations of spread at the bottom; the generator sits above all three.
LEARNERS=(
  "gpt4omini:gpt-4o-mini"    # weak
  "gpt5mini:gpt-5-mini"      # mid   (already has a phase-0: results/expel_phase0_Base90_ExpeL)
  "gpt5:gpt-5"               # strong
)

GENERATOR="gpt-5.4"                          # + oracle. Stronger than every learner, frozen.
PARTNER="gpt-5.4-mini"                       # outside the ladder -> no self-play; see COST below
GATES="gpt-4.1-mini"                         # coherence + MOI; != generator, so it isn't self-grading
JUDGE="google/gemini-3-flash-preview"        # Sotopia-Eval + state check + LP. Cross-lab.
EMBED="text-embedding-3-small"

# Routing. Only the judge needs Lightning; everything else goes direct to OpenAI, which keeps
# Lightning spend below what gen90 used. Requires the FM api_key/base_url params (Stage B).
JUDGE_ROUTE="lightning"
DEFAULT_ROUTE="openai"

# Notes from the probe that affect config:
#   * The gpt-5 reasoning family (gpt-5, gpt-5-mini, gpt-5-nano) REJECTS an explicit
#     `temperature`. Harmless for learners — sotopia's LiteLLM path sets drop_params=True — and
#     each FM now owns its own model, so fm.py's sticky _temperature_supported flag can no longer
#     leak across roles.
#   * gpt-5.4 accepts temperature, so the generator keeps temperature=1.0.
#   * Anthropic via Lightning is unavailable (credit balance too low), so a Claude partner is out.
#   * openai/gpt-5.4 is NOT served on Lightning — the strong models are only reachable direct.
#   * gpt-5.6 is served and is stronger than the generator, but is deliberately UNUSED. The
#     generator is pinned to 5.4; anything above the learner ladder satisfies the requirement.
#
# COST — why the partner is 4.1-mini and not 4.1.
# Measured from results/gen90_expel transcripts, per 90-scenario run (300 episodes): the partner
# speaks 1,673 turns for ~134k output tokens but ~1.15M input tokens of conversation history, plus
# ~1.5M for the keyed preamble it re-reads every turn. Input dominates output ~20:1, so the partner
# is almost entirely a PROMPT cost, and it is the single largest line item in the run.
# Why gpt-5.4-mini specifically (probed 2026-08-10):
#   * mini-priced and OUTSIDE the learner ladder, so no self-play on any diagonal cell.
#   * fast -- 1.5-1.7 s/call, vs ~28 s for the gpt-5 reasoning family in gen90. The partner speaks
#     half of all turns, so its latency sets the wall clock.
#   * accepts `temperature` (the gpt-5 reasoning family does not).
#   * qualitatively holds a position better than gpt-4.1-mini on a spot check: 4.1-mini opened with
#     "I appreciate your enthusiasm to get involved!" -- the RLHF-cheerful register that leaks and
#     caves -- where 5.4-mini opened by refusing. Soft evidence, but it points the right way for a
#     role whose whole job is not conceding.
# Fidelity is enforced by the per-turn verifier (Stage D), not by model size -- that was the
# deliberate design choice. STEP-UP TRIGGER: if smoke shows a per-turn violation rate that
# resampling can't fix, move the partner up and re-measure.
# NOT gpt-5-nano despite the low per-token price: reasoning model, bills reasoning as output, slow.
#
# max_turns stays at 20.
#
# PROMPT CACHING -- worth trying, but the current template order defeats it. Automatic prefix
# caching only rewards a stable PREFIX, and _PARTNER_TURN_PROMPT_KEYED deliberately places the long
# static persona+key block AFTER {history} (episode_runner.py:112-124: "RECENCY > PRIMACY ...
# a disposition block buried before a long history section will be overridden by RLHF-cooperative
# defaults"). So the only cacheable prefix today is the short intro. Getting caching without losing
# recency means splitting into a static system message (persona + key, cacheable) plus a user
# message (history + a SHORT recency reminder) -- but sotopia sends one user message
# (generation_utils/generate.py builds messages=[{"role":"user",...}]), so that touches vendored
# code. Try it in Stage D alongside the verifier, and measure cached_tokens in the usage payload.

OUT_ROOT="results/matrix_v1"
LOGS="$OUT_ROOT/logs"
mkdir -p "$LOGS"
ts() { date +%Y-%m-%dT%H:%M:%S; }

# ─────────────────────────────────────────────────────────────────────────────
# Stages. Ramp: smoke (3 stratified seeds) -> pilot (20) -> full (90).
# Pass a seed subset with SEED_IDS=... ; deterministic child ids mean a later,
# larger run only generates what is missing.
# ─────────────────────────────────────────────────────────────────────────────
SEED_IDS="${SEED_IDS:-}"          # comma-separated env_pks; empty = all 90
CONCURRENCY="${CONCURRENCY:-3}"

echo "[$(ts)] roster: generator=$GENERATOR partner=$PARTNER gates=$GATES judge=$JUDGE"
echo "[$(ts)] learners: ${LEARNERS[*]}"
echo "[$(ts)] seeds: ${SEED_IDS:-all 90}  concurrency: $CONCURRENCY"

# --- Stage 1: per-learner seed bands (Row 0 of the figure) --------------------
# The bands ARE the calibration input -- the operator is chosen from them -- so this is a required
# step, not overhead, and it doubles as Row 0. gpt5mini can reuse the existing phase-0 run.
for entry in "${LEARNERS[@]}"; do
  tag="${entry%%:*}"; model="${entry##*:}"
  if [ -d "$OUT_ROOT/phase0/$tag/seeds" ]; then
    echo "[$(ts)] phase0/$tag already present -- skipping"
    continue
  fi
  # NOTE: results/expel_phase0_Base90_ExpeL is NOT reused as Row 0, even though it has all 90
  # bands for gpt-5-mini. It ran with partner_model = gpt-5-mini (self-play) rather than the frozen
  # gpt-5.4-mini, so its bands are not comparable with another learner's — and that mismatch would
  # sit directly in the saturation claim. It is still used to CHOOSE which seeds to include
  # (pick_ramp_seeds.py), which is a coverage heuristic and needs no partner matching.
  echo "[$(ts)] phase0 for $tag -- NOT YET AUTOMATED."
  echo "        run_baseline_eval.py then run_expel_chronicle.py with"
  echo "          --learner-model $model --partner-model $PARTNER --judge-model $JUDGE"
  echo "          --run-name matrix_phase0_$tag"
  echo "        then point $OUT_ROOT/phase0/$tag at its output."
done

# --- Stage 2: audit the calibration mapping, free -----------------------------
for entry in "${LEARNERS[@]}"; do
  tag="${entry%%:*}"; model="${entry##*:}"
  [ -d "$OUT_ROOT/phase0/$tag/seeds" ] || continue
  $PY scripts/run_grid_generate.py \
    --phase0-dir "$OUT_ROOT/phase0/$tag" \
    --learner-tag "$tag" --learner-model "$model" \
    --generator-model "$GENERATOR" --gates-model "$GATES" \
    --partner-model "$PARTNER" --judge-model "$JUDGE" \
    --out "$OUT_ROOT/sets/$tag" --dry-run 2>&1 | tail -3
done

# --- Stage 3: generate one set per learner -----------------------------------
# SEED_IDS is the ramp lever: smoke (3 stratified) -> pilot (20) -> full (90). Deterministic child
# ids mean a later, larger run generates only what is missing.
if [ "${RUN_GENERATE:-0}" = "1" ]; then
  for entry in "${LEARNERS[@]}"; do
    tag="${entry%%:*}"; model="${entry##*:}"
    [ -d "$OUT_ROOT/phase0/$tag/seeds" ] || { echo "[$(ts)] no bands for $tag -- skipping"; continue; }
    echo "[$(ts)] generating set for $tag"
    $PY scripts/run_grid_generate.py \
      --phase0-dir "$OUT_ROOT/phase0/$tag" \
      --learner-tag "$tag" --learner-model "$model" \
      --reflection-model "$model" \
      --generator-model "$GENERATOR" --gates-model "$GATES" \
      --partner-model "$PARTNER" --judge-model "$JUDGE" \
      --out "$OUT_ROOT/sets/$tag" \
      --concurrency "$CONCURRENCY" ${SEED_IDS:+--seed-ids "$SEED_IDS"} \
      2>&1 | tee "$LOGS/3_generate_$tag.log"
  done
else
  echo "[$(ts)] generation skipped (set RUN_GENERATE=1 to run it)"
fi

# --- Stage 4: crossplay + matrix ---------------------------------------------
if [ "${RUN_CROSSPLAY:-0}" = "1" ]; then
  $PY scripts/run_grid_crossplay.py --all --matrix-root "$OUT_ROOT" \
    --partner-model "$PARTNER" --judge-model "$JUDGE" --aux-model "$GATES" \
    --concurrency "$CONCURRENCY" 2>&1 | tee "$LOGS/4_crossplay.log"
fi

$PY scripts/build_matrix.py --matrix-root "$OUT_ROOT" --csv 2>&1 | tee "$LOGS/5_matrix.log" || true

echo "[$(ts)] done."
