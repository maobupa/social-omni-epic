"""ExpeL baseline runner (Baseline 3, §5.1).

Faithful three-stage ExpeL pipeline (Zhao et al., AAAI 2024) on SOTOPIA seeds:

  1. gather  — Reflexion-style trajectory collection on the 90 training seeds
  2. extract — insight (rule) extraction via ADD/EDIT/AGREE/REMOVE operations
  3. eval    — run held-out / external scenarios with insights + retrieved
               few-shot trajectories injected (full-ExpeL configuration)

Run from project root. Typical end-to-end flow (insights from the 90 seeds,
eval on a SEPARATE social dataset you compile later):

  # 1. gather trajectories on the 90 seeds (Reflexion, up to 3 attempts each)
  python scripts/run_expel_baseline.py gather \
      --train-seeds data/sotopia_90_seeds.jsonl --out output/expel

  # 2. extract the insight list (single fold = train on all 90)
  python scripts/run_expel_baseline.py extract --out output/expel

  # 3. evaluate on your held-out scenarios, injecting insights + few-shots
  python scripts/run_expel_baseline.py eval \
      --eval-seeds data/my_eval_set.jsonl --out output/expel

To instead do ExpeL-faithful k-fold eval ON the 90 seeds themselves:
  python scripts/run_expel_baseline.py extract --out output/expel --k-folds 2
  python scripts/run_expel_baseline.py eval --out output/expel --in-distribution

`all` runs gather -> extract -> eval in one shot.

Outputs under <out>/:
  trajectories.json   gathered success/failure pool (stage 1)
  insights.json       extracted rule list(s) (stage 2)
  extract_log.txt     human-readable trace of the extraction operations
  eval/episodes/*.json, eval/summary.json, eval/analysis.md   (stage 3)
"""
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
from social_omni_epic.seeds import load_sotopia_seeds

GOAL_SUCCESS_THRESHOLD = 7.0  # SOTOPIA-PI success convention (matches run_baseline_eval.py)
REL_SUCCESS_THRESHOLD = 0.0   # relationship floor — success = GOAL≥7 ∧ REL≥0 (matches
                              # run_expel_chronicle._is_terminal_success and the bank labels)


def _is_success(scores: dict) -> bool:
    s = scores or {}
    return (float(s.get("goal", 0.0)) >= GOAL_SUCCESS_THRESHOLD
            and float(s.get("relationship", 0.0)) >= REL_SUCCESS_THRESHOLD)


# ---------------------------------------------------------------------------
# Reporting helpers (same shape as scripts/run_baseline_eval.py so ExpeL
# numbers drop straight into the baseline comparison)
# ---------------------------------------------------------------------------

def _mean_std(values):
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    std = (sum((v - m) ** 2 for v in values) / n) ** 0.5
    return round(m, 3), round(std, 3)


def _build_summary(results, learner_model, partner_model, seeds_path, config):
    dim_accum = defaultdict(list)
    by_source = defaultdict(list)
    failed, succeeded = [], []

    for r in results:
        if "error" in r:
            continue
        scores = r["scores"]
        goal = float(scores.get("goal", 0.0))
        for dim, val in scores.items():
            dim_accum[dim].append(float(val))
        by_source[r["source"]].append(goal)
        (succeeded if _is_success(scores) else failed).append(r["seed_idx"])

    overall = {}
    for dim, vals in dim_accum.items():
        m, s = _mean_std(vals)
        overall[dim] = {"mean": m, "std": s, "n": len(vals)}

    by_source_summary = {}
    for src, vals in by_source.items():
        m, s = _mean_std(vals)
        by_source_summary[src] = {"goal_mean": m, "goal_std": s, "n": len(vals)}

    return {
        "timestamp": datetime.now().isoformat(),
        "baseline": "expel",
        "config": config,
        "learner_model": learner_model,
        "partner_model": partner_model,
        "seeds_path": seeds_path,
        "n_completed": len([r for r in results if "error" not in r]),
        "n_errors": len([r for r in results if "error" in r]),
        "goal_success_threshold": GOAL_SUCCESS_THRESHOLD,
        "overall": overall,
        "by_source": by_source_summary,
        "success_rate": round(len(succeeded) / max(len(succeeded) + len(failed), 1), 3),
        "failed_seed_indices": sorted(failed),
        "succeeded_seed_indices": sorted(succeeded),
    }


def _build_analysis_md(summary):
    lines = [
        "# ExpeL Baseline Eval Analysis",
        "",
        f"**Timestamp:** {summary['timestamp']}  ",
        f"**Learner:** `{summary['learner_model']}`  **Partner:** `{summary['partner_model']}`  ",
        f"**N completed:** {summary['n_completed']}  **N errors:** {summary['n_errors']}  ",
        f"**Goal success threshold:** {summary['goal_success_threshold']}  ",
        f"**Config:** `{json.dumps(summary['config'])}`",
        "",
        "## Overall SOTOPIA-EVAL Scores",
        "",
        "| Dimension | Mean | Std | N |",
        "|-----------|------|-----|---|",
    ]
    for dim, st in summary["overall"].items():
        lines.append(f"| {dim} | {st['mean']:.3f} | {st['std']:.3f} | {st['n']} |")
    lines += [
        "",
        f"**GOAL success rate (≥{summary['goal_success_threshold']}):** {summary['success_rate']:.1%}",
        "",
        "## By Source (GOAL)",
        "",
        "| Source | N | GOAL mean ± std |",
        "|--------|---|-----------------|",
    ]
    for src, st in sorted(summary["by_source"].items(), key=lambda x: -x[1]["goal_mean"]):
        lines.append(f"| {src} | {st['n']} | {st['goal_mean']:.3f} ± {st['goal_std']:.3f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _require_key():
    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY or OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)


def stage_gather(args, fm, out_dir):
    from social_omni_epic.expel_baseline import gather_trajectories, trajectories_to_dict

    seeds = load_sotopia_seeds(seeds_path=args.train_seeds, limit=args.n_seeds,
                               both_perspectives=False)
    print(f"[gather] {len(seeds)} training seeds, up to {args.max_trials} attempts each")

    path = out_dir / "trajectories.json"
    # Per-seed checkpoint so a long unattended run survives crashes/restarts.
    # Atomic write (temp + replace) so a crash mid-write can't corrupt it.
    def _checkpoint(s, f, i, c):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(trajectories_to_dict(s, f, i, c), indent=2, default=str))
        os.replace(tmp, path)

    initial_state = None
    if args.resume and path.exists():
        initial_state = json.loads(path.read_text())
        done = len(initial_state.get("completed_idx", []))
        print(f"[gather] --resume: found checkpoint with {done} completed scenarios")

    succeeded, failed, idx2task, completed = gather_trajectories(
        scenarios=seeds, fm=fm,
        learner_model=args.learner_model, partner_model=args.partner_model,
        max_trials=args.max_trials, max_turns=args.max_turns,
        goal_threshold=GOAL_SUCCESS_THRESHOLD,
        on_log=print,
        initial_state=initial_state,
        on_progress=_checkpoint,
    )
    n_succ_tasks = sum(1 for v in succeeded.values() if v)
    n_fail_trajs = sum(len(v) for v in failed.values())
    n_succ_trajs = sum(len(v) for v in succeeded.values())
    print(f"[gather] tasks with ≥1 success: {n_succ_tasks}/{len(seeds)} | "
          f"success trajs: {n_succ_trajs} | failure trajs: {n_fail_trajs}")
    _checkpoint(succeeded, failed, idx2task, completed)  # final flush
    print(f"[gather] wrote {path}")
    return succeeded, failed, idx2task


def stage_extract(args, fm, out_dir, pool=None):
    from social_omni_epic.expel_baseline import extract_insights, trajectories_from_dict

    if pool is None:
        traj_path = out_dir / "trajectories.json"
        if not traj_path.exists():
            print(f"ERROR: {traj_path} not found — run `gather` first.", file=sys.stderr)
            sys.exit(1)
        succeeded, failed, idx2task = trajectories_from_dict(json.loads(traj_path.read_text()))
    else:
        succeeded, failed, idx2task = pool

    log_lines = []
    insights = extract_insights(
        succeeded, failed, idx2task, fm,
        max_num_rules=args.max_num_rules,
        success_critique_num=args.success_critique_num,
        k_folds=args.k_folds, seed=args.seed,
        on_log=lambda m: (print(m), log_lines.append(m)),
    )
    (out_dir / "insights.json").write_text(json.dumps(insights, indent=2))
    (out_dir / "extract_log.txt").write_text("\n".join(log_lines))
    n_rules = len(insights["all"]["rules"])
    print(f"[extract] combined rule set: {n_rules} insights")
    for i, r in enumerate(insights["all"]["rules"], 1):
        print(f"   {i}. {r}")
    return insights


def stage_eval(args, fm, out_dir, fm_judge=None):
    from social_omni_epic.episode_runner import clean_transcript, run_single_episode
    from social_omni_epic.expel_baseline import (
        embed_trajectory_bank, format_memory_prompt, retrieve_fewshots,
        trajectories_from_dict, _format_transcript,
    )
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    insights_path = out_dir / "insights.json"
    if not insights_path.exists():
        print(f"ERROR: {insights_path} not found — run `extract` first.", file=sys.stderr)
        sys.exit(1)
    insights = json.loads(insights_path.read_text())

    # Few-shot retrieval bank: the gathered TRAINING successes.
    traj_path = out_dir / "trajectories.json"
    succeeded, _failed, _idx2task = trajectories_from_dict(json.loads(traj_path.read_text()))
    bank = embed_trajectory_bank(succeeded, fm) if args.include_fewshots else []

    # Build (scenario, rules) eval list.
    in_distribution = args.in_distribution
    eval_path = args.eval_seeds or args.train_seeds
    eval_seeds = load_sotopia_seeds(seeds_path=eval_path, limit=args.n_seeds,
                                    both_perspectives=False)

    # Map each eval scenario -> the rule set to use.
    fold_rules_for_idx = {}
    if in_distribution and insights.get("folds"):
        for fold in insights["folds"]:
            for idx in fold["eval_idxs"]:
                fold_rules_for_idx[idx] = fold["rules"]
        print(f"[eval] in-distribution k-fold eval ({insights['k_folds']} folds)")
    combined_rules = insights["all"]["rules"]

    eval_dir = out_dir / "eval"
    episodes_dir = eval_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "include_fewshots": args.include_fewshots,
        "fewshot_top_k": args.fewshot_top_k,
        "in_distribution": in_distribution,
        "max_num_rules": args.max_num_rules,
        "n_rules_combined": len(combined_rules),
    }

    results = []
    for i, scenario in enumerate(eval_seeds):
        source = scenario.interaction_type or "unknown"
        safe = source.replace("/", "_").replace(" ", "_")
        ep_file = episodes_dir / f"eval_{i:03d}_{safe}.json"
        # Resume: reuse an already-scored episode instead of re-running it.
        if args.resume and ep_file.exists():
            try:
                results.append(json.loads(ep_file.read_text()))
                print(f"[{i+1:3d}/{len(eval_seeds)}] {source:<22} | RESUMED (cached)")
                continue
            except Exception:
                pass  # corrupt cache — fall through and re-run

        env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
        learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ""
        rules = fold_rules_for_idx.get(i, combined_rules)

        fewshots = []
        if args.include_fewshots and bank:
            fewshots = retrieve_fewshots(
                query_text=scenario.scenario, bank=bank, fm=fm,
                top_k=args.fewshot_top_k,
                exclude_scenario_id=scenario.id,  # guard against self-retrieval
            )
        memory_prompt = format_memory_prompt(rules, fewshots,
                                             include_fewshots=args.include_fewshots)

        error, scores, transcript = None, {}, []
        try:
            result = asyncio.run(run_single_episode(
                env_profile=env_profile, agent_profiles=agent_profiles, fm=fm,
                learner_model=args.learner_model, partner_model=args.partner_model,
                memory_prompt=memory_prompt, max_turns=args.max_turns,
                learner_goal=learner_goal, rubric=None, partner_profile=None,
                judge_self_consistency_k=1,
                fm_judge=fm_judge,  # cross-lab judge — eval must NOT be self-scored
            ))
            scores = result.learner_scores
            transcript = clean_transcript(result.transcript)
        except Exception as e:  # noqa: BLE001
            error = str(e)
            print(f"  [eval {i}] ERROR: {e}")

        goal = float(scores.get("goal", 0.0))
        completed = [float(r["scores"].get("goal", 0)) for r in results if "error" not in r]
        if scores:
            completed.append(goal)
        running = sum(completed) / len(completed) if completed else 0.0
        status = f"goal={goal:.1f}" if not error else "ERROR"
        print(f"[{i+1:3d}/{len(eval_seeds)}] {source:<22} | {status} | "
              f"n_rules={len(rules)} n_fewshot={len(fewshots)} | running_avg_goal={running:.2f}")

        record = {
            "seed_idx": i, "env_pk": scenario.source_env_id, "id": scenario.id,
            "source": source, "scenario": scenario.scenario,
            "learner_goal": learner_goal, "scores": scores,
            "n_rules_used": len(rules), "n_fewshots_used": len(fewshots),
            "transcript": transcript,
        }
        if error:
            record["error"] = error
        results.append(record)

        ep_file.write_text(json.dumps(record, indent=2, default=str))

        summary = _build_summary(results, args.learner_model, args.partner_model,
                                 eval_path, config)
        (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (eval_dir / "analysis.md").write_text(_build_analysis_md(summary))

    goal_stats = _build_summary(results, args.learner_model, args.partner_model,
                                eval_path, config)["overall"].get("goal", {})
    print(f"\n=== EXPEL BASELINE EVAL COMPLETE ===")
    print(f"  GOAL    : {goal_stats.get('mean', 0):.3f} ± {goal_stats.get('std', 0):.3f}")
    print(f"  Results : {eval_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ExpeL baseline (Baseline 3) on SOTOPIA")
    ap.add_argument("stage", choices=["gather", "extract", "eval", "all"])
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default: output/expel_<timestamp>)")
    ap.add_argument("--train-seeds", type=str, default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--eval-seeds", type=str, default=None,
                    help="Eval scenarios (default: same as --train-seeds)")
    ap.add_argument("--n-seeds", type=int, default=None, help="Limit to first N seeds")
    ap.add_argument("--model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--learner-model", type=str, default=None)
    ap.add_argument("--partner-model", type=str, default=None)
    ap.add_argument("--judge-model", type=str, default="google/gemini-3-flash-preview",
                    help="Cross-lab judge for eval scoring (key check / diagnostics). MUST differ "
                         "from the learner provider — eval must not be self-scored.")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume from existing checkpoints (skip gathered seeds / scored episodes)")
    # gather
    ap.add_argument("--max-trials", type=int, default=3,
                    help="Reflexion attempts per training scenario (ExpeL default depth)")
    # extract
    ap.add_argument("--max-num-rules", type=int, default=10, help="Target insight count")
    ap.add_argument("--success-critique-num", type=int, default=8,
                    help="Successes per batch in the success-critique stage")
    ap.add_argument("--k-folds", type=int, default=1,
                    help="1 = train on all (external eval); >=2 = ExpeL k-fold CV")
    ap.add_argument("--seed", type=int, default=42)
    # eval
    ap.add_argument("--include-fewshots", action="store_true", default=True,
                    help="Full-ExpeL: inject retrieved few-shot trajectories (default on)")
    ap.add_argument("--no-fewshots", dest="include_fewshots", action="store_false",
                    help="Insights-only ablation (ExpeL's own ablation)")
    ap.add_argument("--fewshot-top-k", type=int, default=2)
    ap.add_argument("--in-distribution", action="store_true", default=False,
                    help="Use per-fold rules to eval the held-out 90 seeds (needs --k-folds>=2)")
    args = ap.parse_args()

    # Episode agents go through Sotopia/litellm, which wants the provider-
    # prefixed id ("openai/gpt-5-mini"). The FM wrapper talks to the OpenAI
    # client directly, which wants the bare id ("gpt-5-mini") — UNLESS a
    # Lightning.ai base_url is configured, in which case the prefixed routing
    # form is correct. Normalize accordingly.
    args.learner_model = args.learner_model or args.model
    args.partner_model = args.partner_model or args.model

    _require_key()
    using_lightning = bool(os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("LIGHTNING_AI_BASE_URL"))

    def _bare(m: str) -> str:
        return m.split("/", 1)[1] if (not using_lightning and m.startswith("openai/")) else m

    fm = FM(model=_bare(args.model))
    # Separate cross-lab judge for eval scoring (gemini by default). Asserted distinct provider.
    fm_judge = FM(model=_bare(args.judge_model))
    if args.stage in ("eval", "all"):
        if str(args.judge_model).split("/")[0] == str(args.learner_model).split("/")[0]:
            print(f"ERROR: --judge-model provider must differ from the learner "
                  f"({args.learner_model}) — eval must not be self-scored.", file=sys.stderr)
            sys.exit(1)

    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"output/expel_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}  |  judge: {args.judge_model}")

    if args.stage == "gather":
        print("WARNING: the `gather` stage is DEPRECATED — it self-scores with the learner "
              "(no cross-lab judge) at a goal-only threshold. Use scripts/run_expel_chronicle.py "
              "for trajectory gathering (gemini judge, GOAL≥7 ∧ REL≥0, phase0 output + LP), then "
              "`run_expel_baseline.py extract --out <dir>`.", file=sys.stderr)
        stage_gather(args, fm, out_dir)
    elif args.stage == "extract":
        stage_extract(args, fm, out_dir)
    elif args.stage == "eval":
        stage_eval(args, fm, out_dir, fm_judge=fm_judge)
    elif args.stage == "all":
        print("WARNING: `all` uses the deprecated `gather` stage. Prefer "
              "run_expel_chronicle.py (gather) → run_expel_baseline.py extract → eval.",
              file=sys.stderr)
        pool = stage_gather(args, fm, out_dir)
        stage_extract(args, fm, out_dir, pool=pool)
        stage_eval(args, fm, out_dir, fm_judge=fm_judge)


if __name__ == "__main__":
    main()
