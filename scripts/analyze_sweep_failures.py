"""Two-stage LLM analysis of vanilla sweep failures.

Stage 1 — per-episode fingerprinting (batches of 20)
  For each of the 157 failed episodes: send interaction_type + scenario excerpt +
  learner_goal + judge reasoning to the LLM. Get back a short failure_tag (a
  3-5 word category slug) and a one-sentence explanation of what went wrong.

Stage 2 — synthesis
  Collect all 157 (failure_tag, why, interaction_type) tuples into one call.
  Ask the LLM to: (a) group into top recurring themes, (b) break down by
  interaction type, (c) write an overall diagnosis paragraph.

Outputs
--------
  results/vanilla_sweep/failure_fingerprints.jsonl   per-episode tags (Stage 1)
  results/vanilla_sweep/failure_patterns.json        structured synthesis (Stage 2)
  results/vanilla_sweep/failure_patterns.md          human-readable report

Usage
------
    python scripts/analyze_sweep_failures.py \\
        --failures results/vanilla_sweep/failures.jsonl \\
        --out      results/vanilla_sweep \\
        --model    openai/gpt-5-mini

    # Resume: skips Stage 1 if failure_fingerprints.jsonl already exists
    python scripts/analyze_sweep_failures.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM

BATCH_SIZE = 20   # failures per Stage-1 LLM call (~12K tokens per batch, well under limits)

# ---------------------------------------------------------------------------
# Stage 1: per-episode fingerprinting
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = (
    "You are analyzing failed social interaction episodes to identify WHY a language model "
    "failed to achieve its stated social goal. Focus on the model's behavioral mistakes, "
    "not on the difficulty of the scenario. Respond with ONLY valid JSON."
)

_STAGE1_USER = """Below are {n} failed social interaction episodes. The learner model (gpt-5-mini)
played Agent 1 but did NOT achieve its goal. For each episode I give you:
- interaction_type: the scenario category
- scenario: brief description of the situation
- learner_goal: what Agent 1 was trying to achieve
- failure_mode: rule-based bucket (for context only)
- evaluation_reasoning: the judge's per-dimension reasoning explaining why it failed

For EACH episode return a JSON object with:
  "scenario_id": the episode id
  "failure_tag": a 3-5 word slug naming the failure pattern (e.g. "repeated_same_offer",
                 "conceded_too_fast", "failed_to_build_rapport", "ignored_partner_objection",
                 "asked_too_aggressively", "missed_key_information", "gave_up_too_early",
                 "relationship_damaged_by_pressure") — invent new tags as needed, be consistent
  "why": one sentence describing what the model did wrong behaviorally

Return a JSON array of {n} objects in the same order as the input episodes:

EPISODES:
{episodes_block}
"""


def _format_episode_for_stage1(r: dict, idx: int) -> str:
    scenario_excerpt = (r.get("scenario") or "")[:300]
    goal_excerpt = (r.get("learner_goal") or "")[:200]
    reasoning_excerpt = (r.get("evaluation_reasoning") or "")[:900]
    return (
        f"--- Episode {idx+1} | id={r['scenario_id']} ---\n"
        f"interaction_type: {r.get('interaction_type') or 'unknown'}\n"
        f"failure_mode: {r.get('failure_mode', 'n/a')}\n"
        f"scenario: {scenario_excerpt}\n"
        f"learner_goal: {goal_excerpt}\n"
        f"evaluation_reasoning:\n{reasoning_excerpt}\n"
    )


def run_stage1(failures: list[dict], fm: FM, out_dir: Path, resume: bool) -> list[dict]:
    fp_path = out_dir / "failure_fingerprints.jsonl"

    # Resume: load already-tagged episodes
    done: dict[str, dict] = {}
    if resume and fp_path.exists():
        for line in fp_path.read_text().splitlines():
            line = line.strip()
            if line:
                fp = json.loads(line)
                done[fp["scenario_id"]] = fp
        print(f"[stage1] resume: {len(done)} fingerprints already on disk")

    remaining = [r for r in failures if r["scenario_id"] not in done]
    print(f"[stage1] fingerprinting {len(remaining)} failures in batches of {BATCH_SIZE} ...")

    with open(fp_path, "a") as fout:
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start: batch_start + BATCH_SIZE]
            episodes_block = "\n\n".join(
                _format_episode_for_stage1(r, i) for i, r in enumerate(batch)
            )
            user_prompt = _STAGE1_USER.format(
                n=len(batch),
                episodes_block=episodes_block,
            )
            try:
                fingerprints = fm.query_json(_STAGE1_SYSTEM, user_prompt, temperature=0.0)
                if not isinstance(fingerprints, list):
                    # Some models wrap in {"items": [...]} or similar
                    fingerprints = (fingerprints.get("items")
                                    or fingerprints.get("episodes")
                                    or list(fingerprints.values())[0]
                                    if isinstance(fingerprints, dict) else [])
            except Exception as e:
                print(f"  [stage1] batch {batch_start//BATCH_SIZE+1} ERROR: {e} — skipping")
                continue

            # Align returned fingerprints to input batch by position
            for i, fp_raw in enumerate(fingerprints[:len(batch)]):
                if not isinstance(fp_raw, dict):
                    continue
                # Enforce correct scenario_id (don't trust the LLM to copy it faithfully)
                fp = {
                    "scenario_id":    batch[i]["scenario_id"],
                    "interaction_type": batch[i].get("interaction_type", ""),
                    "failure_mode":   _classify_failure(batch[i]),
                    "failure_tag":    fp_raw.get("failure_tag", "unknown"),
                    "why":            fp_raw.get("why", ""),
                    "goal":           (batch[i].get("scores") or {}).get("goal"),
                    "relationship":   (batch[i].get("scores") or {}).get("relationship"),
                }
                done[fp["scenario_id"]] = fp
                fout.write(json.dumps(fp) + "\n")

            n_done = batch_start + len(batch)
            print(f"  [stage1] {n_done}/{len(remaining)} fingerprinted")

    all_fps = list(done.values())
    print(f"[stage1] complete — {len(all_fps)} fingerprints total")
    return all_fps


# ---------------------------------------------------------------------------
# Stage 2: synthesis
# ---------------------------------------------------------------------------

_STAGE2_SYSTEM = (
    "You are a researcher analyzing patterns in how a language model (gpt-5-mini) fails at "
    "social interaction tasks. You have per-episode failure tags and explanations. "
    "Your job is to synthesize these into actionable insights about the model's systematic "
    "weaknesses. Respond with ONLY valid JSON."
)

_STAGE2_USER = """I have {n} failed social interaction episodes where gpt-5-mini (vanilla, no memory)
failed to achieve its social goal. Below is the full list of per-episode failure tags and
one-sentence explanations, along with the interaction type and rule-based failure mode.

{fingerprints_block}

Please respond with a JSON object containing:

1. "top_patterns": list of the 6-10 most recurring failure themes. Each entry:
   {{
     "pattern_name": "short descriptive name",
     "count": approximate number of episodes fitting this theme,
     "description": "2-3 sentences: what the model does wrong and why it fails",
     "representative_tags": ["tag1", "tag2"],  // failure_tag values that map to this pattern
     "example_scenario_ids": ["id1", "id2"]    // 2-3 example episode ids
   }}

2. "by_interaction_type": for each interaction_type that has >= 3 failures, an entry:
   {{
     "interaction_type": "name",
     "n_failed": count,
     "dominant_pattern": "which top_pattern dominates here",
     "notes": "1-2 sentences on why this category is hard for vanilla"
   }}

3. "capability_gaps": list of 3-5 high-level social capabilities the model consistently lacks
   (e.g., "sustained empathic listening", "strategic concession timing"). Each entry:
   {{
     "capability": "name",
     "description": "what the model fails to do",
     "affected_patterns": ["pattern_name1", ...]
   }}

4. "overall_diagnosis": a paragraph (5-8 sentences) summarizing the core weaknesses of
   vanilla gpt-5-mini on social interaction tasks, what types of scenarios it handles vs
   struggles with, and the primary behavioral patterns that explain the failures.
"""


def run_stage2(fingerprints: list[dict], fm: FM, out_dir: Path) -> dict:
    print(f"[stage2] synthesizing {len(fingerprints)} fingerprints ...")

    lines = []
    for fp in fingerprints:
        lines.append(
            f"scenario_id={fp['scenario_id']}  interaction_type={fp['interaction_type']}  "
            f"failure_mode={fp['failure_mode']}  failure_tag={fp['failure_tag']}  "
            f"goal={fp['goal']}  rel={fp['relationship']}\n"
            f"  why: {fp['why']}"
        )
    fingerprints_block = "\n".join(lines)

    user_prompt = _STAGE2_USER.format(
        n=len(fingerprints),
        fingerprints_block=fingerprints_block,
    )

    synthesis = fm.query_json(_STAGE2_SYSTEM, user_prompt, temperature=0.0)
    print("[stage2] synthesis complete")
    return synthesis


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report_md(fingerprints: list[dict], synthesis: dict) -> str:
    lines = [
        "# Vanilla Sweep — Failure Pattern Analysis",
        "",
        f"**Episodes analyzed:** {len(fingerprints)} failed scenarios (out of 734 total, 78.6% pass rate)",
        f"**Model:** gpt-5-mini (vanilla, no ICL)",
        "",
    ]

    # Overall diagnosis
    diag = synthesis.get("overall_diagnosis", "")
    if diag:
        lines += ["## Overall Diagnosis", "", diag, ""]

    # Capability gaps
    gaps = synthesis.get("capability_gaps", [])
    if gaps:
        lines += ["## Core Capability Gaps", ""]
        for g in gaps:
            lines.append(f"**{g.get('capability', '?')}** — {g.get('description', '')}")
            affected = g.get("affected_patterns", [])
            if affected:
                lines.append(f"  *Affects patterns: {', '.join(affected)}*")
        lines.append("")

    # Top patterns
    patterns = synthesis.get("top_patterns", [])
    if patterns:
        lines += ["## Top Failure Patterns", ""]
        for i, p in enumerate(patterns, 1):
            lines += [
                f"### {i}. {p.get('pattern_name', '?')} (n≈{p.get('count', '?')})",
                "",
                p.get("description", ""),
                "",
                f"*Representative tags:* `{'`, `'.join(p.get('representative_tags', []))}`",
                "",
            ]

    # By interaction type
    by_type = synthesis.get("by_interaction_type", [])
    if by_type:
        lines += ["## Breakdown by Interaction Type", ""]
        if isinstance(by_type, dict):
            by_type = [{"interaction_type": k, **v} for k, v in by_type.items()]
        lines += [
            "| Type | N failed | Dominant pattern | Notes |",
            "|------|----------|-----------------|-------|",
        ]
        for entry in sorted(by_type, key=lambda x: -x.get("n_failed", 0)):
            lines.append(
                f"| {entry.get('interaction_type','?')} "
                f"| {entry.get('n_failed','?')} "
                f"| {entry.get('dominant_pattern','?')} "
                f"| {entry.get('notes','')[:80]} |"
            )
        lines.append("")

    # Failure tag frequency table
    from collections import Counter
    tag_counts = Counter(fp["failure_tag"] for fp in fingerprints)
    lines += [
        "## Failure Tag Frequency",
        "",
        "| Tag | Count |",
        "|-----|-------|",
    ]
    for tag, cnt in tag_counts.most_common():
        lines.append(f"| {tag} | {cnt} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Rule-based failure mode (same logic as sweep script, reproduced here)
# ---------------------------------------------------------------------------

def _classify_failure(rec: dict) -> str:
    scores = rec.get("scores") or {}
    goal = float(scores.get("goal") or 0.0)
    rel  = float(scores.get("relationship") or 0.0)
    goal_achieved = bool(rec.get("goal_achieved", False))
    if goal >= 7.0 and rel >= 0.0 and not goal_achieved:
        return "goal_score_ok_but_judge_rejected"
    if goal >= 7.0 and rel < 0.0:
        return "goal_ok_rel_negative"
    if goal < 7.0 and rel < 0.0:
        return "both_goal_and_rel_failed"
    if goal >= 5.0:
        return "goal_close_but_insufficient"
    if goal >= 3.0:
        return "goal_partial"
    return "goal_very_low"


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LLM analysis of vanilla sweep failures")
    ap.add_argument("--failures", default="results/vanilla_sweep/failures.jsonl")
    ap.add_argument("--out", default="results/vanilla_sweep")
    ap.add_argument("--model", default="openai/gpt-5-mini",
                    help="Model for the analysis (Stage 1 + 2)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip Stage 1 if failure_fingerprints.jsonl already exists")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--stage1-only", action="store_true",
                    help="Run Stage 1 (fingerprinting) only")
    ap.add_argument("--stage2-only", action="store_true",
                    help="Run Stage 2 (synthesis) only — requires fingerprints on disk")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LIGHTNING_AI_API_KEY"):
        print("ERROR: OPENAI_API_KEY or LIGHTNING_AI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    failures_path = Path(args.failures)
    if not failures_path.exists():
        print(f"ERROR: {failures_path} not found. Run run_vanilla_sotopia_sweep.py first.",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    using_lightning = bool(
        os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("LIGHTNING_AI_BASE_URL")
    )
    def _bare(m: str) -> str:
        return m.split("/", 1)[1] if (not using_lightning and m.startswith("openai/")) else m

    fm = FM(model=_bare(args.model))

    failures = [json.loads(l) for l in failures_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(failures)} failures from {failures_path}")

    # Stage 1
    fingerprints: list[dict] = []
    if not args.stage2_only:
        fingerprints = run_stage1(failures, fm, out_dir, resume=args.resume)
    else:
        fp_path = out_dir / "failure_fingerprints.jsonl"
        if not fp_path.exists():
            print(f"ERROR: {fp_path} not found — run Stage 1 first.", file=sys.stderr)
            sys.exit(1)
        fingerprints = [json.loads(l) for l in fp_path.read_text().splitlines() if l.strip()]
        print(f"[stage2-only] loaded {len(fingerprints)} fingerprints from disk")

    if args.stage1_only:
        print("Stage 1 done. Re-run without --stage1-only to run synthesis.")
        return

    # Stage 2
    synthesis = run_stage2(fingerprints, fm, out_dir)

    _atomic_write(out_dir / "failure_patterns.json",
                  json.dumps(synthesis, indent=2, default=str))
    _atomic_write(out_dir / "failure_patterns.md",
                  build_report_md(fingerprints, synthesis))

    print(f"\nOutputs written:")
    print(f"  {out_dir}/failure_fingerprints.jsonl   (per-episode tags)")
    print(f"  {out_dir}/failure_patterns.json        (structured synthesis)")
    print(f"  {out_dir}/failure_patterns.md          (human-readable report)")

    # Quick console summary
    top = synthesis.get("top_patterns", [])
    if top:
        print("\nTop failure patterns:")
        for p in top:
            print(f"  [{p.get('count','?'):>3}] {p.get('pattern_name','?')}")
    diag = synthesis.get("overall_diagnosis", "")
    if diag:
        print(f"\nDiagnosis preview:\n  {diag[:300]}...")


if __name__ == "__main__":
    main()
