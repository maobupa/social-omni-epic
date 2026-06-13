"""Held-out 3-condition ICL comparison (the headline evaluation).

Runs the SAME held-out scenarios through a frozen learner under three in-context treatments,
scores each with a cross-lab 7-dim Sotopia-Eval judge, and reports paired deltas vs. Vanilla.

    Vanilla        : no memory injected (memory_prompt="")
    ExpeL-Base90   : ExpeL insights + retrieved fewshots from the raw-90-seed bank
    Ours / Gen90   : SAME mechanism, fewshots+insights from the generated bank

Controlled comparison: every condition runs the identical scenarios, learner, partner (vanilla,
no key/memory), judge, and prompt builder. The ONLY thing that differs between ExpeL-Base90 and
Ours is which bank the rules + fewshots come from. Vanilla isolates the no-memory floor.

Design (see docs/post_run_experiments.md):
  - Retrieval = ExpeL-native task-text cosine (faithful ExpeL `task_similarity`), top_k=2.
  - Prompt builder = expel_baseline.format_memory_prompt, IDENTICAL across conditions.
  - Judge = a different lab than the learner (breaks the self-eval monoculture).
  - Eval partner is vanilla (partner_key=None): the claim is transfer to STANDARD tasks.

Usage:
    python scripts/run_eval_comparison.py \
        --eval-seeds data/eval_candidates.jsonl \
        --base-bank results/expel_phase0_Base90_ExpeL \
        --gen-bank  results/gen90_expel \
        --conditions vanilla,expel_base,ours \
        --learner-model openai/gpt-5-mini \
        --partner-model openai/gpt-5-mini \
        --judge-model   google/gemini-3-flash-preview \
        --top-k 2 --max-turns 20 --concurrency 4 \
        --out results/eval_comparison

`ours` requires results/gen90_expel/insights.json — generate it first once the bank is complete:
    python scripts/run_expel_baseline.py extract --out results/gen90_expel
If insights.json is missing, the `ours` condition is skipped with a warning (the others still run).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
from social_omni_epic.seeds import load_sotopia_seeds
from social_omni_epic.episode_runner import run_single_episode
from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles
from social_omni_epic.expel_baseline import (
    trajectories_from_dict,
    embed_trajectory_bank,
    retrieve_fewshots,
    format_memory_prompt,
)

SOTOPIA_DIMS = (
    "believability", "relationship", "knowledge", "secret",
    "social_rules", "financial_and_material_benefits", "goal",
)
GOAL_THRESHOLD = 7.0
REL_THRESHOLD = 0.0


# --------------------------------------------------------------------------- #
# Memory builders (one per condition). A builder maps a scenario → memory_prompt.
# --------------------------------------------------------------------------- #

def _vanilla_builder(_scenario) -> str:
    """No memory — the no-ICL floor."""
    return ""


def _make_expel_builder(bank_dir: Path, fm: FM, top_k: int):
    """ExpeL-style builder: fixed insight rules + top_k retrieved successful fewshots.

    The trajectory bank is the bank's OWN successful trajectories; retrieval is task-text cosine
    (faithful ExpeL `task_similarity`). Identical mechanism for Base90 and Ours — only `bank_dir`
    differs. Rules + bank are loaded ONCE (closure); retrieval runs per scenario.
    """
    insights_path = bank_dir / "insights.json"
    traj_path = bank_dir / "trajectories.json"
    if not insights_path.exists():
        raise FileNotFoundError(
            f"{insights_path} missing — run `python scripts/run_expel_baseline.py extract "
            f"--out {bank_dir}` first."
        )
    if not traj_path.exists():
        raise FileNotFoundError(f"{traj_path} missing in {bank_dir}.")

    insights = json.loads(insights_path.read_text())
    rules = insights.get("all", {}).get("rules", [])
    succeeded, _failed, _idx2task = trajectories_from_dict(json.loads(traj_path.read_text()))
    bank = embed_trajectory_bank(succeeded, fm)  # embeds each trajectory's task text in place
    print(f"  loaded {len(rules)} insight rules + {len(bank)} successful trajectories "
          f"from {bank_dir.name}")

    def build(scenario) -> str:
        # Retrieve by the eval scenario's TASK TEXT (matches the bank's t.task embedding space).
        fewshots = retrieve_fewshots(
            query_text=scenario.scenario, bank=bank, fm=fm,
            top_k=top_k, exclude_scenario_id="",  # held-out scenarios are never in the bank
        )
        return format_memory_prompt(rules, fewshots, include_fewshots=True)

    return build


# --------------------------------------------------------------------------- #
# Episode running + scoring
# --------------------------------------------------------------------------- #

async def _run_one(scenario, memory_prompt, fm, fm_judge, args) -> dict:
    """Run one eval episode for one condition; return the learner's 7-dim scores + meta."""
    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
    learner_goal = scenario.agent_goals[0] if scenario.agent_goals else ""
    result = await run_single_episode(
        env_profile=env_profile,
        agent_profiles=agent_profiles,
        fm=fm,
        learner_model=args.learner_model,
        partner_model=args.partner_model,
        memory_prompt=memory_prompt,
        max_turns=args.max_turns,
        learner_goal=learner_goal,
        rubric=None,
        partner_profile=None,
        judge_self_consistency_k=1,
        partner_key=None,            # vanilla eval partner — claim is transfer to standard tasks
        fm_judge=fm_judge,           # cross-lab judge (mandatory)
    )
    scores = result.learner_scores or {}
    return {
        "scenario_id": scenario.id,
        "source_env_id": scenario.source_env_id,
        "scores": {k: scores.get(k) for k in SOTOPIA_DIMS},
        "overall_score": scores.get("overall_score"),
        "goal_achieved": bool(scores.get("goal_achieved", False)),
        "terminal_success": bool(getattr(result, "terminal_success", False)),
        "success_goal_rel": (float(scores.get("goal", 0.0)) >= GOAL_THRESHOLD
                             and float(scores.get("relationship", 0.0)) >= REL_THRESHOLD),
        "memory_chars": len(memory_prompt or ""),
        "num_turns": getattr(result, "num_turns", None),
    }


async def run_condition(name, scenarios, builder, fm, fm_judge, args, out_dir: Path) -> list[dict]:
    """Run all scenarios for one condition, with bounded concurrency. Per-episode files are
    written so a killed run resumes (skip scenarios already on disk)."""
    cond_dir = out_dir / name / "episodes"
    cond_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    async def _slot(scn):
        ep_path = cond_dir / f"{scn.id}.json"
        if ep_path.exists() and not args.overwrite:
            return json.loads(ep_path.read_text())
        async with sem:
            try:
                rec = await _run_one(scn, builder(scn), fm, fm_judge, args)
            except Exception as e:
                import traceback
                print(f"  [{name}] {scn.id[:8]} ERROR: {e}\n{traceback.format_exc()[:400]}")
                rec = {"scenario_id": scn.id, "error": str(e), "scores": {}}
            ep_path.write_text(json.dumps(rec, indent=2, default=str))
            return rec

    print(f"\n=== Condition: {name} ({len(scenarios)} scenarios, concurrency={args.concurrency}) ===")
    results = await asyncio.gather(*[_slot(s) for s in scenarios])
    return [r for r in results if r and not r.get("error")]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.fmean(xs), 4) if xs else None


def summarize(per_condition: dict[str, list[dict]]) -> dict:
    """Per-condition dimension means + success rate, and deltas vs. Vanilla (paired by scenario)."""
    summary = {"conditions": {}, "deltas_vs_vanilla": {}}
    for name, recs in per_condition.items():
        dim_means = {d: _mean([r["scores"].get(d) for r in recs]) for d in SOTOPIA_DIMS}
        summary["conditions"][name] = {
            "n": len(recs),
            "dim_means": dim_means,
            "overall_mean": _mean([r.get("overall_score") for r in recs]),
            "success_rate_goal_rel": _mean([1.0 if r.get("success_goal_rel") else 0.0 for r in recs]),
            "terminal_success_rate": _mean([1.0 if r.get("terminal_success") else 0.0 for r in recs]),
        }

    # Paired deltas: compare each condition to vanilla on the scenarios both share.
    base = {r["scenario_id"]: r for r in per_condition.get("vanilla", [])}
    if base:
        for name, recs in per_condition.items():
            if name == "vanilla":
                continue
            paired = [(base[r["scenario_id"]], r) for r in recs if r["scenario_id"] in base]
            if not paired:
                continue
            def _d(rec_r, rec_b, dim):  # paired difference on a dimension, null-safe
                return (rec_r["scores"].get(dim) or 0) - (rec_b["scores"].get(dim) or 0)
            summary["deltas_vs_vanilla"][name] = {
                "n_paired": len(paired),
                "goal": _mean([_d(r, b, "goal") for b, r in paired]),
                "relationship": _mean([_d(r, b, "relationship") for b, r in paired]),
                "overall": _mean([(r.get("overall_score") or 0) - (b.get("overall_score") or 0)
                                  for b, r in paired]),
                "success_rate_goal_rel": _mean(
                    [(1.0 if r.get("success_goal_rel") else 0.0)
                     - (1.0 if b.get("success_goal_rel") else 0.0) for b, r in paired]),
            }
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="3-condition held-out ICL comparison")
    ap.add_argument("--eval-seeds", default="data/eval_candidates.jsonl")
    ap.add_argument("--base-bank", default="results/expel_phase0_Base90_ExpeL",
                    help="bank dir for the ExpeL-Base90 condition (needs insights.json + trajectories.json)")
    ap.add_argument("--gen-bank", default="results/gen90_expel",
                    help="bank dir for the Ours condition (needs insights.json + trajectories.json)")
    ap.add_argument("--conditions", default="vanilla,expel_base,ours",
                    help="comma-separated subset of {vanilla,expel_base,ours}")
    ap.add_argument("--learner-model", default="openai/gpt-5-mini")
    ap.add_argument("--partner-model", default="openai/gpt-5-mini")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--top-k", type=int, default=2, help="fewshots retrieved per scenario (ExpeL default 2)")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="limit eval scenarios (debug)")
    ap.add_argument("--overwrite", action="store_true", help="re-run episodes even if cached on disk")
    ap.add_argument("--out", default="results/eval_comparison")
    args = ap.parse_args()

    # Honest-experiment guard: judge must be a different provider than the learner.
    if args.judge_model.split("/")[0] == args.learner_model.split("/")[0]:
        raise SystemExit(
            f"Judge provider ({args.judge_model}) must differ from learner provider "
            f"({args.learner_model}) — cross-lab judging breaks the self-eval monoculture."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]

    fm = FM(model=args.learner_model)
    fm_judge = FM(model=args.judge_model)

    scenarios = load_sotopia_seeds(args.eval_seeds, limit=args.limit, both_perspectives=False)
    print(f"Loaded {len(scenarios)} held-out eval scenarios from {args.eval_seeds}")

    # Build the memory builder for each requested condition (skip 'ours' if its bank isn't extracted).
    builders: dict = {}
    for cond in requested:
        if cond == "vanilla":
            builders["vanilla"] = _vanilla_builder
        elif cond == "expel_base":
            builders["expel_base"] = _make_expel_builder(Path(args.base_bank), fm, args.top_k)
        elif cond == "ours":
            try:
                builders["ours"] = _make_expel_builder(Path(args.gen_bank), fm, args.top_k)
            except FileNotFoundError as e:
                print(f"SKIPPING 'ours': {e}")
        else:
            print(f"Unknown condition '{cond}' — ignoring.")

    per_condition: dict[str, list[dict]] = {}
    for name, builder in builders.items():
        per_condition[name] = asyncio.run(
            run_condition(name, scenarios, builder, fm, fm_judge, args, out_dir)
        )

    summary = summarize(per_condition)
    summary["config"] = {
        "eval_seeds": args.eval_seeds, "n_scenarios": len(scenarios),
        "learner_model": args.learner_model, "partner_model": args.partner_model,
        "judge_model": args.judge_model, "top_k": args.top_k, "max_turns": args.max_turns,
        "base_bank": args.base_bank, "gen_bank": args.gen_bank,
        "conditions_run": list(per_condition.keys()),
        "retrieval": "expel_task_text_cosine", "fewshot_framing": "format_memory_prompt (identical across conditions)",
        "success_label": "goal>=7 and rel>=0",
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Console table
    print("\n" + "=" * 72)
    print("RESULTS (held-out, deltas vs. Vanilla)")
    print("=" * 72)
    for name, s in summary["conditions"].items():
        print(f"  {name:14} n={s['n']:3}  GOAL={s['dim_means'].get('goal')}  "
              f"REL={s['dim_means'].get('relationship')}  overall={s['overall_mean']}  "
              f"success(goal∧rel)={s['success_rate_goal_rel']}")
    for name, d in summary["deltas_vs_vanilla"].items():
        print(f"  Δ {name:12} (n={d['n_paired']})  ΔGOAL={d['goal']}  ΔREL={d['relationship']}  "
              f"Δoverall={d['overall']}  Δsuccess={d['success_rate_goal_rel']}")
    print(f"\nWrote {out_dir/'comparison_summary.json'} + per-episode files under {out_dir}/<condition>/")


if __name__ == "__main__":
    main()
