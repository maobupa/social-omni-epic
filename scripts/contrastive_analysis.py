"""Contrastive analysis: what makes certain SOTOPIA scenarios hard for vanilla gpt-5-mini?

Approach
---------
For each category with enough pass/fail examples, show the LLM a set of passed
and failed scenarios side-by-side and ask: what structural properties of the
failed scenarios explain the difference?

Stage 1 — Per-category contrastive analysis
  For each eligible category (n >= 5 total, at least 1 pass + 1 fail):
    • Up to 5 failed episodes  (scenario + goal + scores + judge reasoning)
    • Up to 3 passed episodes  (scenario + goal + scores only)
    • One LLM call → structured difficulty factors for this category

Stage 2 — Cross-category synthesis
  Collect all per-category findings → one LLM call that:
    • Identifies universal difficulty dimensions across categories
    • Defines a difficulty taxonomy
    • Draws implications for what social skills vanilla LLMs lack

Outputs under <out>/
----------------------
  contrastive_by_category.json    per-category structured findings (Stage 1)
  contrastive_report.md           human-readable report
  contrastive_synthesis.json      cross-category taxonomy (Stage 2)

Usage
------
    python scripts/contrastive_analysis.py
    python scripts/contrastive_analysis.py --out results/vanilla_sweep --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM

# Categories we run contrastive analysis on — must have enough volume and mixed outcomes
MIN_TOTAL = 5
MIN_PASS = 1
MIN_FAIL = 1

MAX_FAILED_PER_CATEGORY = 5    # shown with full reasoning
MAX_PASSED_PER_CATEGORY = 3    # shown without reasoning (goal + scores enough)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episodes(episodes_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Return {interaction_type: {pass: [...], fail: [...]}}."""
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"pass": [], "fail": []})
    for f in sorted(episodes_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if "error" in r:
            continue
        t = r.get("interaction_type") or "unknown"
        key = "pass" if r.get("terminal_success") else "fail"
        by_type[t][key].append(r)
    return by_type


def eligible_categories(by_type: dict) -> list[str]:
    return sorted(
        [t for t, v in by_type.items()
         if len(v["pass"]) >= MIN_PASS
         and len(v["fail"]) >= MIN_FAIL
         and len(v["pass"]) + len(v["fail"]) >= MIN_TOTAL],
        key=lambda t: len(by_type[t]["fail"]),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Stage 1 prompts
# ---------------------------------------------------------------------------

_S1_SYSTEM = (
    "You are a researcher studying why a language model (gpt-5-mini, vanilla — no memory or ICL) "
    "fails at social interaction tasks. You will be shown passed and failed episodes from the same "
    "scenario category and must identify what structural properties of the scenarios explain the "
    "difference in outcomes. Focus on the SCENARIO properties, not just the model's behavior. "
    "Respond with ONLY valid JSON."
)

_S1_USER = """CATEGORY: {category}
Pass rate in this category: {n_pass}/{n_total} ({pass_rate:.0%})

I will show you {n_fail_shown} FAILED episodes and {n_pass_shown} PASSED episodes from this category.
The model (gpt-5-mini, vanilla) played Agent 1 in all cases.

=== FAILED EPISODES ===
{failed_block}

=== PASSED EPISODES ===
{passed_block}

Analyze the contrast. What structural properties of the FAILED scenarios made them harder?
Think about: goal conflict level, emotional stakes, number of conditions needed for success,
partner rigidity, whether the scenario requires sustained back-and-forth vs a single ask,
relational complexity, and anything else that distinguishes hard from easy cases.

Respond with a JSON object:
{{
  "category": "{category}",
  "pass_rate": {pass_rate_float},
  "difficulty_factors": [
    {{
      "factor": "short name",
      "description": "what makes scenarios with this property harder",
      "present_in_failed": "how this factor shows up in the failed cases",
      "absent_in_passed": "why the passed cases avoided or minimized this factor"
    }}
  ],
  "what_model_lacks": "2-3 sentences on the specific social skill or strategy the model is missing",
  "hard_scenario_signature": "1-2 sentences describing what a 'hard' scenario in this category looks like",
  "easy_scenario_signature": "1-2 sentences describing what an 'easy' scenario in this category looks like"
}}
"""


def _format_failed(r: dict, idx: int) -> str:
    scores = r.get("scores") or {}
    reasoning = (r.get("evaluation_reasoning") or "")[:700]
    return (
        f"[FAILED {idx+1}] scenario_id={r['scenario_id'][:20]}\n"
        f"Scenario: {(r.get('scenario') or '')[:300]}\n"
        f"Learner goal: {(r.get('learner_goal') or '')[:200]}\n"
        f"Scores: goal={scores.get('goal')}  rel={scores.get('relationship')}  "
        f"belief={scores.get('believability')}\n"
        f"Judge reasoning:\n{reasoning}\n"
    )


def _format_passed(r: dict, idx: int) -> str:
    scores = r.get("scores") or {}
    return (
        f"[PASSED {idx+1}] scenario_id={r['scenario_id'][:20]}\n"
        f"Scenario: {(r.get('scenario') or '')[:300]}\n"
        f"Learner goal: {(r.get('learner_goal') or '')[:200]}\n"
        f"Scores: goal={scores.get('goal')}  rel={scores.get('relationship')}  "
        f"belief={scores.get('believability')}\n"
    )


def run_stage1(
    by_type: dict,
    categories: list[str],
    fm: FM,
    out_dir: Path,
    resume: bool,
) -> list[dict]:
    cache_path = out_dir / "contrastive_by_category.json"
    cache: dict[str, dict] = {}
    if resume and cache_path.exists():
        for entry in json.loads(cache_path.read_text()):
            cache[entry["category"]] = entry
        print(f"[stage1] resume: {len(cache)} categories already done")

    results = []
    for cat in categories:
        if cat in cache:
            results.append(cache[cat])
            continue

        v = by_type[cat]
        failed_sample = v["fail"][:MAX_FAILED_PER_CATEGORY]
        passed_sample = v["pass"][:MAX_PASSED_PER_CATEGORY]
        n_total = len(v["pass"]) + len(v["fail"])
        pass_rate = len(v["pass"]) / n_total

        failed_block = "\n\n".join(_format_failed(r, i) for i, r in enumerate(failed_sample))
        passed_block = "\n\n".join(_format_passed(r, i) for i, r in enumerate(passed_sample))

        prompt = _S1_USER.format(
            category=cat,
            n_pass=len(v["pass"]), n_total=n_total, pass_rate=pass_rate,
            n_fail_shown=len(failed_sample), n_pass_shown=len(passed_sample),
            failed_block=failed_block,
            passed_block=passed_block,
            pass_rate_float=round(pass_rate, 3),
        )

        try:
            result = fm.query_json(_S1_SYSTEM, prompt, temperature=0.0)
            # Ensure category is correctly set (don't trust the LLM)
            result["category"] = cat
            result["n_pass"] = len(v["pass"])
            result["n_fail"] = len(v["fail"])
            print(f"  [stage1] {cat}: {len(result.get('difficulty_factors', []))} factors")
        except Exception as e:
            print(f"  [stage1] {cat} ERROR: {e}")
            result = {"category": cat, "error": str(e), "n_pass": len(v["pass"]), "n_fail": len(v["fail"])}

        cache[cat] = result
        results.append(result)

        # Checkpoint after each category
        _atomic_write(cache_path, json.dumps(list(cache.values()), indent=2, default=str))

    print(f"[stage1] complete — {len(results)} categories analyzed")
    return results


# ---------------------------------------------------------------------------
# Stage 2: cross-category synthesis
# ---------------------------------------------------------------------------

_S2_SYSTEM = (
    "You are a researcher synthesizing findings about why vanilla gpt-5-mini fails at social "
    "interaction tasks. You have per-category contrastive analyses showing what structural "
    "properties make scenarios hard. Your job is to identify universal difficulty dimensions "
    "and define a taxonomy of social scenario difficulty. Respond with ONLY valid JSON."
)

_S2_USER = """I have contrastive analyses from {n_categories} scenario categories showing what makes
certain social interaction scenarios hard for vanilla gpt-5-mini. Here are the findings:

{findings_block}

Based on these findings, respond with a JSON object:
{{
  "difficulty_taxonomy": [
    {{
      "dimension": "name of this difficulty dimension",
      "definition": "precise definition of what this dimension captures",
      "why_hard_for_vanilla": "what social reasoning skill vanilla LLMs lack to handle this",
      "categories_affected": ["list of categories where this dimension is primary"],
      "severity": "high / medium / low"
    }}
  ],
  "key_insight": "The single most important finding in 2-3 sentences — what does this tell us about vanilla LLM weaknesses in social intelligence?",
  "model_failure_profile": {{
    "handles_well": ["list of 3-4 scenario types or properties the model handles fine"],
    "systematically_fails": ["list of 3-4 scenario types or properties that reliably cause failure"]
  }},
  "implication_for_curriculum": "2-3 sentences on what kinds of scenarios a training curriculum should prioritize, based on this analysis"
}}
"""


def run_stage2(category_results: list[dict], fm: FM, out_dir: Path) -> dict:
    print(f"[stage2] synthesizing across {len(category_results)} categories ...")

    findings_block_parts = []
    for r in category_results:
        if "error" in r:
            continue
        factors = r.get("difficulty_factors", [])
        factor_lines = "\n".join(
            f"  - {f.get('factor','?')}: {f.get('description','')}"
            for f in factors
        )
        findings_block_parts.append(
            f"CATEGORY: {r['category']} (pass={r.get('n_pass',0)}, fail={r.get('n_fail',0)})\n"
            f"Hard scenario: {r.get('hard_scenario_signature','')}\n"
            f"Easy scenario: {r.get('easy_scenario_signature','')}\n"
            f"What model lacks: {r.get('what_model_lacks','')}\n"
            f"Difficulty factors:\n{factor_lines}"
        )

    findings_block = "\n\n".join(findings_block_parts)
    prompt = _S2_USER.format(
        n_categories=len(findings_block_parts),
        findings_block=findings_block,
    )

    synthesis = fm.query_json(_S2_SYSTEM, prompt, temperature=0.0)
    print("[stage2] complete")
    return synthesis


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report_md(category_results: list[dict], synthesis: dict) -> str:
    lines = [
        "# Contrastive Analysis: What Makes SOTOPIA Scenarios Hard for Vanilla gpt-5-mini?",
        "",
        "**Method:** For each category with mixed pass/fail outcomes, the LLM was shown "
        "up to 5 failed and 3 passed episodes side-by-side and asked to identify structural "
        "properties that explain the difference.",
        "",
    ]

    # Key insight up front
    key = synthesis.get("key_insight", "")
    if key:
        lines += ["## Key Insight", "", key, ""]

    # Difficulty taxonomy
    taxonomy = synthesis.get("difficulty_taxonomy", [])
    if taxonomy:
        lines += ["## Difficulty Taxonomy", ""]
        for dim in sorted(taxonomy, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("severity","low"), 1)):
            cats = ", ".join(dim.get("categories_affected", []))
            lines += [
                f"### {dim.get('dimension','?')}  `[{dim.get('severity','?')}]`",
                "",
                f"**Definition:** {dim.get('definition','')}",
                "",
                f"**Why hard for vanilla:** {dim.get('why_hard_for_vanilla','')}",
                "",
                f"**Categories:** {cats}",
                "",
            ]

    # Model failure profile
    profile = synthesis.get("model_failure_profile", {})
    if profile:
        lines += ["## Model Failure Profile", ""]
        lines += ["**Handles well:**"]
        for item in profile.get("handles_well", []):
            lines.append(f"- {item}")
        lines += ["", "**Systematically fails:**"]
        for item in profile.get("systematically_fails", []):
            lines.append(f"- {item}")
        lines.append("")

    # Curriculum implication
    impl = synthesis.get("implication_for_curriculum", "")
    if impl:
        lines += ["## Implication for Curriculum / Training", "", impl, ""]

    # Per-category findings
    lines += ["---", "", "## Per-Category Findings", ""]
    for r in category_results:
        if "error" in r:
            lines += [f"### {r['category']} — ERROR", "", r.get("error", ""), ""]
            continue
        n_pass, n_fail = r.get("n_pass", 0), r.get("n_fail", 0)
        pass_rate = n_pass / max(n_pass + n_fail, 1)
        lines += [
            f"### {r['category']}  (n={n_pass+n_fail}, pass={pass_rate:.0%})",
            "",
            f"**Hard scenario signature:** {r.get('hard_scenario_signature','')}",
            "",
            f"**Easy scenario signature:** {r.get('easy_scenario_signature','')}",
            "",
            f"**What the model lacks:** {r.get('what_model_lacks','')}",
            "",
            "**Difficulty factors:**",
            "",
        ]
        for f in r.get("difficulty_factors", []):
            lines += [
                f"- **{f.get('factor','?')}** — {f.get('description','')}",
                f"  - *In failed cases:* {f.get('present_in_failed','')}",
                f"  - *In passed cases:* {f.get('absent_in_passed','')}",
            ]
        lines.append("")

    return "\n".join(lines) + "\n"


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
    ap = argparse.ArgumentParser(description="Contrastive pass/fail analysis by interaction category")
    ap.add_argument("--episodes-dir", default="results/vanilla_sweep/episodes")
    ap.add_argument("--out", default="results/vanilla_sweep")
    ap.add_argument("--model", default="openai/gpt-5-mini")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip categories already in contrastive_by_category.json")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--stage2-only", action="store_true")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("LIGHTNING_AI_API_KEY"):
        print("ERROR: API key not set.", file=sys.stderr)
        sys.exit(1)

    episodes_dir = Path(args.episodes_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    using_lightning = bool(
        os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("LIGHTNING_AI_BASE_URL")
    )
    def _bare(m: str) -> str:
        return m.split("/", 1)[1] if (not using_lightning and m.startswith("openai/")) else m

    fm = FM(model=_bare(args.model))

    by_type = load_episodes(episodes_dir)
    categories = eligible_categories(by_type)
    print(f"Eligible categories: {categories}")

    # Stage 1
    category_results: list[dict] = []
    if not args.stage2_only:
        category_results = run_stage1(by_type, categories, fm, out_dir, resume=args.resume)
    else:
        cache_path = out_dir / "contrastive_by_category.json"
        if not cache_path.exists():
            print(f"ERROR: {cache_path} not found — run Stage 1 first.", file=sys.stderr)
            sys.exit(1)
        category_results = json.loads(cache_path.read_text())
        print(f"[stage2-only] loaded {len(category_results)} categories from disk")

    if args.stage1_only:
        print("Stage 1 done. Re-run without --stage1-only to synthesize.")
        return

    # Stage 2
    synthesis = run_stage2(category_results, fm, out_dir)
    _atomic_write(out_dir / "contrastive_synthesis.json",
                  json.dumps(synthesis, indent=2, default=str))
    _atomic_write(out_dir / "contrastive_report.md",
                  build_report_md(category_results, synthesis))

    print(f"\nOutputs:")
    print(f"  {out_dir}/contrastive_by_category.json")
    print(f"  {out_dir}/contrastive_synthesis.json")
    print(f"  {out_dir}/contrastive_report.md")

    # Quick console summary
    print("\nDifficulty taxonomy:")
    for dim in synthesis.get("difficulty_taxonomy", []):
        print(f"  [{dim.get('severity','?'):6}] {dim.get('dimension','?')}")
    print(f"\nKey insight: {synthesis.get('key_insight','')[:300]}")


if __name__ == "__main__":
    main()
