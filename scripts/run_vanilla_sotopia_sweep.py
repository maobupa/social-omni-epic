"""Vanilla sweep over all SOTOPIA scenarios to classify pass/fail on first try.

Each environment gets ONE attempt with a vanilla learner (no ICL memory injection).
The 150 held-out eval candidates (data/eval_candidates.jsonl) are skipped — those
have already been scored under the 5-condition comparison.

Consistency guarantees
-----------------------
  Model         openai/gpt-5-mini (learner + partner) — identical to run_eval_comparison.py
  Judge         google/gemini-3-flash-preview (cross-lab, same as run_eval_comparison.py)
  Success       terminal_success = GOAL≥7 ∧ REL≥0 ∧ judge_goal_achieved
                (episode_runner §3.2 — same gate as SOE loop + baseline eval)
  Memory        none (vanilla — no ICL injection)
  Attempts      1 per scenario (first-try classification)

Data sources
-------------
  90 training seeds   data/sotopia_90_seeds.jsonl  (pre-joined agent profiles)
  150 held-out eval   data/eval_candidates.jsonl   — EXCLUDED (already scored)
  Remaining non-seed  data/sotopia_seeds/environment_profiles.jsonl
                      Agents assigned deterministically from the 40-agent pool,
                      matched to the env relationship type via relationship_profiles.jsonl.
                      hash(env_pk) seeds the choice so results are stable across reruns.

Overnight safety
-----------------
  • Per-episode JSON is written atomically (temp + os.replace) — a crash mid-write
    leaves the previous file intact, never a corrupt partial.
  • summary.json and error_analysis.md are rewritten after every completed episode
    so the output is always consistent with what is on disk.
  • --resume skips any scenario whose episode file already exists (default on).
  • SIGINT / SIGTERM trigger a graceful shutdown: in-flight episodes finish, then
    the final summary is written before exit.

Usage
------
    # Full sweep of everything except the 150 held-out scenarios
    python scripts/run_vanilla_sotopia_sweep.py \\
        --out results/vanilla_sweep \\
        --concurrency 8 \\
        --resume

    # Only the 90 training seeds (quick sanity check)
    python scripts/run_vanilla_sotopia_sweep.py \\
        --out results/vanilla_sweep_seeds \\
        --env-split seeds --resume

    # Non-seed envs only
    python scripts/run_vanilla_sotopia_sweep.py \\
        --out results/vanilla_sweep_nonseed \\
        --env-split nonseed --resume

Outputs under <out>/
----------------------
  episodes/           per-episode JSON (transcript + 7-dim scores + reasoning)
  summary.json        aggregate: pass/fail lists, dim means, failure modes, by-source
  error_analysis.md   human-readable failure taxonomy + sample reasoning excerpts
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.data_models import SocialScenario
from social_omni_epic.episode_runner import clean_transcript, run_single_episode
from social_omni_epic.fm import FM
from social_omni_epic.seeds import _make_agent_profile, load_sotopia_seeds
from social_omni_epic.sotopia_bridge import RELATIONSHIP_STR_TO_INT, scenario_to_sotopia_profiles

GOAL_THRESHOLD = 7.0
REL_THRESHOLD = 0.0

SOTOPIA_DIMS = (
    "believability", "relationship", "knowledge", "secret",
    "social_rules", "financial_and_material_benefits", "goal",
)

RELATIONSHIP_INT_TO_STR = {v: k for k, v in RELATIONSHIP_STR_TO_INT.items()}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEEDS_DIR = DATA_DIR / "sotopia_seeds"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_eval_candidate_pks() -> set[str]:
    """Return the set of env_pks in the 150 held-out eval set (already scored)."""
    path = DATA_DIR / "eval_candidates.jsonl"
    if not path.exists():
        return set()
    pks: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pks.add(json.loads(line)["env_pk"])
    return pks


def _load_agent_pool() -> dict[str, dict]:
    """Load the 40 SOTOPIA benchmark agents keyed by PK."""
    path = SEEDS_DIR / "agent_profiles.jsonl"
    agents: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                agents[d["pk"]] = d
    return agents


def _load_relationship_pool() -> list[dict]:
    """Load the 120 pre-defined agent-pair relationship profiles."""
    path = SEEDS_DIR / "relationship_profiles.jsonl"
    rels: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rels.append(json.loads(line))
    return rels


def _pick_agents_for_env(
    env: dict,
    agent_pool: dict[str, dict],
    rel_pool: list[dict],
) -> tuple[dict, dict, str]:
    """Deterministically pick (agent_1_raw, agent_2_raw, background) for a non-seed env.

    Strategy:
      1. Filter relationship profiles to those whose 'relationship' int matches env's.
      2. Fall back to the full pool if none match.
      3. Index by hash(env_pk) % len(candidates) for stability across reruns.
    """
    rel_type = int(env.get("relationship", 0))
    candidates = [r for r in rel_pool if int(r["relationship"]) == rel_type] or rel_pool
    idx = int(hashlib.md5(env["pk"].encode()).hexdigest(), 16) % len(candidates)
    chosen = candidates[idx]
    return (
        agent_pool[chosen["agent_1_id"]],
        agent_pool[chosen["agent_2_id"]],
        chosen.get("background_story", ""),
    )


def _nonseed_env_to_scenario(
    env: dict,
    agent_pool: dict[str, dict],
    rel_pool: list[dict],
) -> SocialScenario:
    """Build a SocialScenario from a raw environment_profiles.jsonl entry."""
    a1_raw, a2_raw, background = _pick_agents_for_env(env, agent_pool, rel_pool)
    a1 = _make_agent_profile(a1_raw)
    a2 = _make_agent_profile(a2_raw)

    raw_goals = env.get("agent_goals") or ["", ""]
    agent_goals = (list(raw_goals) + ["", ""])[:2]

    rel_type = int(env.get("relationship", 0))
    rel_label = RELATIONSHIP_INT_TO_STR.get(rel_type, str(rel_type))
    env_pk = env["pk"]

    return SocialScenario(
        id=f"{env_pk}_p0",
        iteration=-1,
        scenario=env.get("scenario", ""),
        agent_profiles=[a1, a2],
        agent_goals=agent_goals,
        relationship=rel_label,
        relationship_background=background,
        tag=env.get("codename", "") or env.get("source", ""),
        interaction_type=env.get("source", ""),
        source="seed_sotopia_full",
        source_env_id=env_pk,
        source_scenario_id=env_pk,
    )


def load_all_scenarios(
    env_split: str = "all",
    exclude_pks: Optional[set[str]] = None,
) -> list[SocialScenario]:
    """Load scenarios according to env_split, skipping any in exclude_pks."""
    exclude_pks = exclude_pks or set()

    seeds_90: list[SocialScenario] = []
    if env_split in ("seeds", "all"):
        all_seeds = load_sotopia_seeds(
            seeds_path=str(DATA_DIR / "sotopia_90_seeds.jsonl"),
            both_perspectives=False,
        )
        seeds_90 = [s for s in all_seeds if s.source_env_id not in exclude_pks]
        skipped = len(all_seeds) - len(seeds_90)
        print(f"[data] seed scenarios: {len(seeds_90)} loaded"
              + (f", {skipped} excluded (held-out eval)" if skipped else ""))

    nonseed: list[SocialScenario] = []
    if env_split in ("nonseed", "all"):
        seed_pks = {s.source_env_id for s in seeds_90}
        env_path = SEEDS_DIR / "environment_profiles.jsonl"
        agent_pool = _load_agent_pool()
        rel_pool = _load_relationship_pool()
        with open(env_path) as f:
            all_envs = [json.loads(l) for l in f if l.strip()]
        skipped_nonseed = 0
        for env in all_envs:
            pk = env["pk"]
            if pk in seed_pks:
                continue  # already in seeds_90
            if pk in exclude_pks:
                skipped_nonseed += 1
                continue
            nonseed.append(_nonseed_env_to_scenario(env, agent_pool, rel_pool))
        print(f"[data] non-seed scenarios: {len(nonseed)} loaded"
              + (f", {skipped_nonseed} excluded (held-out eval)" if skipped_nonseed else ""))

    combined = seeds_90 + nonseed
    print(f"[data] total to run: {len(combined)}")
    return combined


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    """Write text to path atomically via a sibling temp file + os.replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Episode running
# ---------------------------------------------------------------------------

async def _run_one(
    scenario: SocialScenario,
    fm: FM,
    fm_judge: FM,
    args: argparse.Namespace,
) -> dict:
    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
    learner_goal = scenario.agent_goals[0] if scenario.agent_goals else ""

    result = await run_single_episode(
        env_profile=env_profile,
        agent_profiles=agent_profiles,
        fm=fm,
        learner_model=args.learner_model,
        partner_model=args.partner_model,
        memory_prompt="",       # vanilla — no ICL
        max_turns=args.max_turns,
        learner_goal=learner_goal,
        rubric=None,
        partner_profile=None,
        judge_self_consistency_k=1,
        partner_key=None,
        fm_judge=fm_judge,
    )

    scores = result.learner_scores or {}
    return {
        "scenario_id": scenario.id,
        "source_env_id": scenario.source_env_id,
        "source": scenario.source,
        "interaction_type": scenario.interaction_type,
        "scenario": scenario.scenario,
        "learner_goal": learner_goal,
        "scores": {k: scores.get(k) for k in SOTOPIA_DIMS},
        "goal_achieved": bool(scores.get("goal_achieved", False)),
        "terminal_success": bool(result.terminal_success),
        "num_turns": result.num_turns,
        "evaluation_reasoning": result.evaluation_reasoning,
        "transcript": clean_transcript(result.transcript),
    }


async def run_sweep(
    scenarios: list[SocialScenario],
    fm: FM,
    fm_judge: FM,
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict]:
    """Run all scenarios with bounded concurrency.

    Safety guarantees:
    - Per-episode files are written atomically so a mid-write crash never corrupts.
    - summary.json + error_analysis.md are rewritten after every completed episode.
    - A SIGINT/SIGTERM sets _shutdown which prevents new episodes from starting;
      in-flight episodes are allowed to complete before the process exits.
    """
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []
    results_lock = asyncio.Lock()
    _shutdown = False
    n_total = len(scenarios)
    n_done = 0

    def _handle_signal(sig, frame):
        nonlocal _shutdown
        _shutdown = True
        print(f"\n[sweep] Signal {sig} received — finishing in-flight episodes, then stopping.")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    async def _slot(scn: SocialScenario, idx: int) -> Optional[dict]:
        nonlocal n_done, _shutdown
        ep_path = episodes_dir / f"{scn.id}.json"

        # Resume: return cached result without consuming a semaphore slot.
        if ep_path.exists() and not args.overwrite:
            try:
                rec = json.loads(ep_path.read_text())
                async with results_lock:
                    results.append(rec)
                    n_done += 1
                    if n_done % 50 == 0 or n_done == n_total:
                        _flush_summary(results, args, out_dir)
                        print(f"[sweep] resumed {n_done}/{n_total} (from disk)")
                return rec
            except Exception:
                pass  # corrupt cache — fall through and re-run

        if _shutdown:
            return None

        async with sem:
            if _shutdown:
                return None
            try:
                rec = await _run_one(scn, fm, fm_judge, args)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"  [{idx+1:4d}/{n_total}] ERROR {scn.id[:14]}: {e}\n{tb[:300]}")
                rec = {
                    "scenario_id": scn.id,
                    "source_env_id": scn.source_env_id,
                    "source": scn.source,
                    "interaction_type": scn.interaction_type,
                    "scenario": scn.scenario,
                    "learner_goal": scn.agent_goals[0] if scn.agent_goals else "",
                    "error": str(e),
                    "scores": {},
                    "terminal_success": False,
                }

            _atomic_write(ep_path, json.dumps(rec, indent=2, default=str))

            async with results_lock:
                results.append(rec)
                n_done += 1
                status = "PASS" if rec.get("terminal_success") else (
                    "ERR " if "error" in rec else "FAIL"
                )
                goal = (rec.get("scores") or {}).get("goal", "?")
                rel  = (rec.get("scores") or {}).get("relationship", "?")
                print(f"  [{n_done:4d}/{n_total}] {status}  goal={goal}  rel={rel}"
                      f"  {scn.id[:16]}  {scn.interaction_type or scn.source}")
                # Rolling checkpoint: rewrite summary after every episode
                _flush_summary(results, args, out_dir)

        return rec

    await asyncio.gather(*[_slot(s, i) for i, s in enumerate(scenarios)])

    # Final flush after all coroutines complete
    _flush_summary(results, args, out_dir)
    return results


# ---------------------------------------------------------------------------
# Aggregation + error analysis (also used for rolling checkpoints)
# ---------------------------------------------------------------------------

def _mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 4) if xs else None


def _classify_failure(rec: dict) -> str:
    scores = rec.get("scores") or {}
    goal = float(scores.get("goal") or 0.0)
    rel  = float(scores.get("relationship") or 0.0)
    goal_achieved = bool(rec.get("goal_achieved", False))

    if goal >= GOAL_THRESHOLD and rel >= REL_THRESHOLD and not goal_achieved:
        return "goal_score_ok_but_judge_rejected"
    if goal >= GOAL_THRESHOLD and rel < REL_THRESHOLD:
        return "goal_ok_rel_negative"
    if goal < GOAL_THRESHOLD and rel < REL_THRESHOLD:
        return "both_goal_and_rel_failed"
    if goal >= 5.0:
        return "goal_close_but_insufficient"   # 5–6.9
    if goal >= 3.0:
        return "goal_partial"                  # 3–4.9
    return "goal_very_low"                     # <3


def build_summary(results: list[dict], args: argparse.Namespace) -> dict:
    clean  = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    passed = [r for r in clean if r.get("terminal_success")]
    failed = [r for r in clean if not r.get("terminal_success")]

    def _dim_stats(recs: list[dict]) -> dict:
        out: dict = {}
        for d in SOTOPIA_DIMS:
            vals = [float(r["scores"][d]) for r in recs
                    if r.get("scores") and r["scores"].get(d) is not None]
            out[d] = {"mean": _mean(vals), "n": len(vals)}
        return out

    failure_modes = Counter(_classify_failure(r) for r in failed)

    by_source: dict[str, dict] = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in clean:
        src = r.get("interaction_type") or r.get("source") or "unknown"
        by_source[src]["total"] += 1
        if r.get("terminal_success"):
            by_source[src]["passed"] += 1
    for src in by_source:
        t, p = by_source[src]["total"], by_source[src]["passed"]
        by_source[src]["pass_rate"] = round(p / t, 4) if t else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "learner_model": args.learner_model,
            "partner_model": args.partner_model,
            "judge_model": args.judge_model,
            "max_turns": args.max_turns,
            "memory": "none (vanilla)",
            "env_split": args.env_split,
            "success_label": (
                f"terminal_success = GOAL>={GOAL_THRESHOLD}"
                f" AND REL>={REL_THRESHOLD} AND judge_goal_achieved"
            ),
        },
        "n_total": len(results),
        "n_completed": len(clean),
        "n_errors": len(errors),
        "n_passed": len(passed),
        "n_failed": len(failed),
        "pass_rate": round(len(passed) / max(len(clean), 1), 4),
        "passed_scenario_ids": sorted(r["scenario_id"] for r in passed),
        "failed_scenario_ids": sorted(r["scenario_id"] for r in failed),
        "error_scenario_ids":  sorted(r["scenario_id"] for r in errors),
        "dim_stats_passed": _dim_stats(passed),
        "dim_stats_failed":  _dim_stats(failed),
        "failure_modes": dict(failure_modes.most_common()),
        "by_source": {k: dict(v) for k, v in sorted(
            by_source.items(), key=lambda x: -x[1]["pass_rate"]
        )},
    }


def build_error_analysis_md(summary: dict, failed: list[dict]) -> str:
    cfg = summary["config"]
    lines = [
        "# Vanilla Sweep — Error Analysis",
        "",
        f"**Date:** {summary['timestamp']}  ",
        f"**Learner / Partner:** `{cfg['learner_model']}`  ",
        f"**Judge:** `{cfg['judge_model']}`  ",
        f"**Memory:** {cfg['memory']}  ",
        f"**Env split:** `{cfg['env_split']}`  ",
        f"**Success label:** {cfg['success_label']}",
        f"**Progress:** {summary['n_completed']} / {summary['n_total']} completed"
        f" (including {summary['n_errors']} errors)",
        "",
        "## Overview",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Completed | {summary['n_completed']} |",
        f"| Passed    | {summary['n_passed']} ({summary['pass_rate']:.1%}) |",
        f"| Failed    | {summary['n_failed']} ({1-summary['pass_rate']:.1%}) |",
        f"| Errors    | {summary['n_errors']} |",
        "",
        "## Dimension Means: Pass vs Fail",
        "",
        "| Dimension | Pass mean | Fail mean | Δ (pass−fail) |",
        "|-----------|-----------|-----------|---------------|",
    ]
    for d in SOTOPIA_DIMS:
        p  = (summary["dim_stats_passed"].get(d) or {}).get("mean")
        f_ = (summary["dim_stats_failed"].get(d) or {}).get("mean")
        delta = round(p - f_, 3) if (p is not None and f_ is not None) else "n/a"
        lines.append(f"| {d} | {p} | {f_} | {delta} |")

    lines += [
        "",
        "## Failure Mode Taxonomy",
        "",
        "| Failure mode | Count |",
        "|-------------|-------|",
    ]
    for mode, cnt in sorted(summary["failure_modes"].items(), key=lambda x: -x[1]):
        lines.append(f"| {mode} | {cnt} |")

    lines += [
        "",
        "## Pass Rate by Interaction Type",
        "",
        "| Interaction type | N | Pass rate |",
        "|-----------------|---|-----------|",
    ]
    for src, st in sorted(summary["by_source"].items(), key=lambda x: -x[1]["pass_rate"]):
        lines.append(f"| {src} | {st['total']} | {st['pass_rate']:.1%} |")

    # Sample failure reasoning — one representative per failure mode
    if failed:
        lines += ["", "## Sample Failure Reasoning (one per mode)", ""]
        seen_modes: set[str] = set()
        for r in sorted(failed, key=lambda x: float((x.get("scores") or {}).get("goal") or 0)):
            mode = _classify_failure(r)
            if mode in seen_modes:
                continue
            seen_modes.add(mode)
            scores  = r.get("scores") or {}
            goal    = scores.get("goal", "?")
            rel     = scores.get("relationship", "?")
            lines += [
                f"### [{mode}] `{r['scenario_id'][:20]}` — goal={goal}, rel={rel}",
                "",
                f"**Learner goal:** {r.get('learner_goal', '')[:150]}",
                "",
                f"**Scenario (excerpt):** {r.get('scenario', '')[:250]}",
                "",
                "**Judge reasoning (excerpt):**",
                "```",
                (r.get("evaluation_reasoning") or "")[:700],
                "```",
                "",
            ]

    return "\n".join(lines) + "\n"


def _flush_summary(results: list[dict], args: argparse.Namespace, out_dir: Path) -> None:
    """Rewrite summary.json and error_analysis.md atomically from current results."""
    clean  = [r for r in results if "error" not in r]
    failed = [r for r in clean if not r.get("terminal_success")]
    summary = build_summary(results, args)
    _atomic_write(out_dir / "summary.json",
                  json.dumps(summary, indent=2, default=str))
    _atomic_write(out_dir / "error_analysis.md",
                  build_error_analysis_md(summary, failed))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Vanilla first-try sweep over SOTOPIA scenarios (excludes 150 held-out eval set)"
    )
    ap.add_argument("--out", default="results/vanilla_sweep")
    ap.add_argument(
        "--env-split", choices=["seeds", "nonseed", "all"], default="all",
        help="seeds = 90 training seeds only | nonseed = 794 additional | all = all 884",
    )
    ap.add_argument("--learner-model", default="openai/gpt-5-mini",
                    help="Model for learner + partner (default: openai/gpt-5-mini)")
    ap.add_argument("--partner-model", default=None,
                    help="Defaults to --learner-model")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview",
                    help="Cross-lab judge. Provider MUST differ from --learner-model.")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap scenarios count (for debugging)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip episodes already on disk (default: on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Ignore disk cache and re-run everything")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Alias for --no-resume")
    args = ap.parse_args()

    args.partner_model = args.partner_model or args.learner_model
    if args.overwrite:
        args.resume = False

    learner_provider = args.learner_model.split("/")[0]
    judge_provider   = args.judge_model.split("/")[0]
    if learner_provider == judge_provider:
        print(
            f"ERROR: judge provider '{judge_provider}' must differ from learner provider "
            f"'{learner_provider}' — eval must not be self-scored.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("LIGHTNING_AI_API_KEY")):
        print("ERROR: OPENAI_API_KEY or LIGHTNING_AI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_exclude = _load_eval_candidate_pks()
    print(f"[data] excluding {len(eval_exclude)} held-out eval_candidates env_pks")

    print(f"\nOutput dir  : {out_dir}")
    print(f"Learner     : {args.learner_model}")
    print(f"Partner     : {args.partner_model}")
    print(f"Judge       : {args.judge_model}")
    print(f"Env split   : {args.env_split}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Max turns   : {args.max_turns}")
    print(f"Resume      : {args.resume}\n")

    scenarios = load_all_scenarios(args.env_split, exclude_pks=eval_exclude)
    if args.limit:
        scenarios = scenarios[: args.limit]
        print(f"[data] limited to first {args.limit} scenarios")

    using_lightning = bool(
        os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("LIGHTNING_AI_BASE_URL")
    )

    def _bare(m: str) -> str:
        """Strip 'openai/' prefix when not routing through Lightning AI."""
        return m.split("/", 1)[1] if (not using_lightning and m.startswith("openai/")) else m

    fm       = FM(model=_bare(args.learner_model))
    fm_judge = FM(model=_bare(args.judge_model))

    results = asyncio.run(run_sweep(scenarios, fm, fm_judge, args, out_dir))

    clean  = [r for r in results if "error" not in r]
    passed = [r for r in clean if r.get("terminal_success")]
    failed = [r for r in clean if not r.get("terminal_success")]
    summary = build_summary(results, args)

    print("\n" + "=" * 60)
    print("VANILLA SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Total ran   : {len(clean)}")
    print(f"  Passed      : {len(passed)} ({summary['pass_rate']:.1%})")
    print(f"  Failed      : {len(failed)} ({1 - summary['pass_rate']:.1%})")
    print(f"  Errors      : {summary['n_errors']}")
    print(f"  Results     : {out_dir}")
    print(f"\n  Failure modes:")
    for mode, cnt in sorted(summary["failure_modes"].items(), key=lambda x: -x[1]):
        print(f"    {mode:<45} {cnt}")


if __name__ == "__main__":
    main()
