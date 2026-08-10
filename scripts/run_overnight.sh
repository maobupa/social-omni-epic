#!/usr/bin/env bash
# Unattended overnight driver: finish the smoke, gate on it, then run the pilot. No prompts.
#
#   nohup caffeinate -i bash scripts/run_overnight.sh > results/matrix_v1/logs/overnight.log 2>&1 &
#
# Writes a running report to results/matrix_v1/OVERNIGHT_STATUS.md — read that first on waking.
#
# It STOPS ITSELF rather than asking, on any of three conditions. Each exists because continuing
# past it would burn hours producing output that cannot be interpreted:
#
#   1. oracle yield < 25%   the generator is producing unwinnable scenarios faster than the gate
#                           filters them. Only the generation prompt fixes that; more compute
#                           produces a bigger pile of broken scenarios.
#   2. hole rate  > 50%     more than half the cells failed to produce a scenario at all. The
#                           matrix would be mostly gaps, and the paired-drop rule would then
#                           discard almost everything.
#   3. zero cells advanced  the stage ran and produced nothing new — a wiring failure, not a slow
#                           run. Retrying it unattended would spin.
#
# Not -e: failures are handled explicitly so a single bad stage still writes a report.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY="uv run"

OUT_ROOT="${OUT_ROOT:-results/matrix_v1}"
LOGS="$OUT_ROOT/logs"
STATUS="$OUT_ROOT/OVERNIGHT_STATUS.md"
mkdir -p "$LOGS"
ts() { date +"%Y-%m-%d %H:%M:%S"; }

say() { echo "[$(ts)] $*"; }
report() { echo "$*" >> "$STATUS"; }

: > "$STATUS"
report "# Overnight matrix run"
report ""
report "Started $(ts) · commit \`$(git rev-parse --short HEAD 2>/dev/null || echo unknown)\`"
report ""
report "Read the FINAL VERDICT section at the bottom first."
report ""

# --- helpers ---------------------------------------------------------------------------------
yield_of() {   # $1 = set tag -> oracle yield, or "n/a"
  $PY - <<PYEOF 2>/dev/null || echo "n/a"
import json, pathlib
p = pathlib.Path("$OUT_ROOT/sets/$1/grid_manifest.json")
if not p.exists():
    print("n/a")
else:
    d = json.loads(p.read_text())
    y = d.get("oracle_yield")
    print("n/a" if y is None else f"{y:.3f}")
PYEOF
}

holes_of() {   # $1 = set tag -> "n_holes n_cells"
  $PY - <<PYEOF 2>/dev/null || echo "0 0"
import json, pathlib
p = pathlib.Path("$OUT_ROOT/sets/$1/grid_manifest.json")
if not p.exists():
    print("0 0")
else:
    d = json.loads(p.read_text())
    print(len(d.get("holes") or {}), d.get("n_cells", 0))
PYEOF
}

n_generated() { ls "$OUT_ROOT"/sets/*/bank/generated/*.json 2>/dev/null | wc -l | tr -d ' '; }

# --- Step 1: let whatever is already running finish ------------------------------------------
if pgrep -f "run_matrix_ramp" >/dev/null 2>&1; then
  say "a ramp is already running — waiting for it"
  report "## Step 1 — waiting for the in-flight smoke"
  while pgrep -f "run_matrix_ramp" >/dev/null 2>&1; do sleep 30; done
  say "in-flight ramp finished"
else
  say "no ramp running; starting from the smoke"
  report "## Step 1 — smoke"
  RAMP=smoke CONCURRENCY=4 bash scripts/run_matrix_ramp.sh \
      > "$LOGS/overnight_smoke.log" 2>&1
  say "smoke exited $?"
fi

$PY scripts/build_matrix.py --matrix-root "$OUT_ROOT" --csv > "$LOGS/overnight_matrix_smoke.log" 2>&1 || true

report ""
report '```'
sed -n '/^MATRIX/,/^GRADING/p' "$LOGS/overnight_matrix_smoke.log" 2>/dev/null | head -30 >> "$STATUS" || true
report '```'
report ""

# --- Step 2: gate on the smoke before spending pilot compute ---------------------------------
GO=1
GATE_NOTES=""
for tag in gpt4omini gpt5mini; do
  y="$(yield_of "$tag")"
  read -r nh nc <<<"$(holes_of "$tag")"
  say "smoke $tag: oracle_yield=$y holes=$nh/$nc"
  report "- **$tag** — oracle yield \`$y\`, holes \`$nh/$nc\`"
  if [ "$y" != "n/a" ]; then
    awk -v y="$y" 'BEGIN{exit (y+0 < 0.25) ? 1 : 0}' || {
      GO=0; GATE_NOTES="$GATE_NOTES
- **STOP: $tag oracle yield $y < 0.25.** The generator is producing unwinnable scenarios faster
  than the gate filters them. Fix the generation prompt (\`_GOAL_FORMAT_GUIDE\` /
  \`_PARTNER_KEY_RULES\` in task_generator.py) — more compute will not help. Inspect
  \`$OUT_ROOT/sets/$tag/rejected/\` for why they were rejected."
    }
  fi
  if [ "$nc" -gt 0 ] && [ "$((nh * 2))" -gt "$nc" ]; then
    GO=0; GATE_NOTES="$GATE_NOTES
- **STOP: $tag has $nh holes out of $nc cells (>50%).** The matrix would be mostly gaps and the
  paired-drop rule would discard nearly everything. Check \`$LOGS/overnight_smoke.log\` for the
  failing stage."
  fi
done

if [ "$GO" != "1" ]; then
  report ""
  report "## FINAL VERDICT — stopped before the pilot"
  report "$GATE_NOTES"
  report ""
  report "No pilot compute was spent. The smoke artifacts are intact under \`$OUT_ROOT\`."
  say "gate failed — not starting the pilot"
  exit 1
fi

report ""
report "Smoke gate passed. Proceeding to the pilot unattended."
report ""

# --- Step 3: the pilot -----------------------------------------------------------------------
BEFORE="$(n_generated)"
say "starting RAMP=pilot (20 seeds x 2 learners); smoke cells are reused, not regenerated"
report "## Step 2 — pilot (20 seeds x 2 learners)"
report ""
report "Started $(ts). The 3 smoke cells per learner are reused via deterministic ids, so only the"
report "missing 17 are generated."
report ""

RAMP=pilot CONCURRENCY=4 bash scripts/run_matrix_ramp.sh \
    > "$LOGS/overnight_pilot.log" 2>&1
PILOT_RC=$?
AFTER="$(n_generated)"
say "pilot exited $PILOT_RC ; generated $BEFORE -> $AFTER"

# --- Step 4: the report ----------------------------------------------------------------------
$PY scripts/build_matrix.py --matrix-root "$OUT_ROOT" --csv > "$LOGS/overnight_matrix_pilot.log" 2>&1 || true

report '```'
cat "$LOGS/overnight_matrix_pilot.log" 2>/dev/null | head -70 >> "$STATUS" || true
report '```'
report ""
report "## FINAL VERDICT"
report ""
if [ "$AFTER" = "$BEFORE" ]; then
  report "- **Nothing new was generated** ($BEFORE -> $AFTER scenarios). That is a wiring failure"
  report "  rather than a slow run — check \`$LOGS/overnight_pilot.log\`."
elif [ "$PILOT_RC" -ne 0 ]; then
  report "- Pilot exited non-zero ($PILOT_RC) after generating $((AFTER - BEFORE)) new scenario(s)."
  report "  Most likely an in-script abort (oracle yield). Completed cells persist, so a re-run"
  report "  resumes rather than restarts: \`RAMP=pilot bash scripts/run_matrix_ramp.sh\`"
else
  report "- Pilot completed. $((AFTER - BEFORE)) new scenario(s) generated ($BEFORE -> $AFTER)."
fi
report ""
report "### What to look at, in order"
report "1. The MATRIX table above — is there a diagonal? Cells are too_easy/frontier/beyond_frontier."
report "2. \`GENERATION HEALTH\` — oracle yield per set. This is the artifact rate, a headline number."
report "3. \`BULLDOZE CHECK\` — attempts passing with rel<0. gen-90 reference is 6/300, none of which"
report "   mattered. Materially higher means reinstate rel>=0 in the success rule."
report "4. \`GRADING\` — staged vs state-only disagreement. That is the deferred decision on whether"
report "   to grade on movement_conditions or on internal_state."
report "5. Did \`relax\` fire? The pilot includes all 5 beyond_frontier seeds, so it should have."
report "   \`grep -c relax $LOGS/overnight_pilot.log\`"
report ""
report "Artifacts: \`$OUT_ROOT/analysis/matrix.{json,csv}\`, per-set \`grid_manifest.json\`,"
report "rejects under \`$OUT_ROOT/sets/*/rejected/\`."
report ""
report "Finished $(ts)."
say "done — report at $STATUS"
