#!/usr/bin/env python3
"""Generate ONE scenario set for ONE learner as a paired grid: exactly one child per SOTOPIA seed.

This replaces the evolutionary search for benchmark-building. The difference is not "less
ambitious" — it is a different objective:

    a CURRICULUM wants to CONCENTRATE effort where learning yield is highest, so uneven coverage
    is correct;
    a BENCHMARK wants to SPREAD, because uneven coverage is a sampling defect. gen-90's Thompson
    sampler used only 47 of 90 root seeds and skewed the bank 2.2x toward deal-or-no-deal
    descendants / 0.4x away from social_chemistry (chi2=28.3, p=1.9e-4), so the resulting item set
    measures bargaining and calls it social intelligence.

So: no anchor selection. Every seed contributes exactly one item, which also makes set-to-set
comparison PAIRED (a filename join) rather than a comparison of two differently-shaped samples.

What still makes a set learner-relative — and this is the whole calibration mechanism — is the
mutation operator, chosen from that seed's phase-0 band FOR THIS LEARNER:

    too_easy -> escalate      frontier -> lateral      beyond_frontier -> relax

The same seed therefore yields a harder child for a strong learner and an easier one for a weak
one. Remove the operator and every row of the matrix becomes identical.

Pipeline per cell: build context -> generate batch -> gates -> ORACLE SOLVABILITY -> K-loop.
Rejected candidates are kept (rejected/) because the artifact rate is a headline number.

    # audit the band -> operator mapping for free, no episodes
    uv run scripts/run_grid_generate.py --phase0-dir results/expel_phase0_Base90_ExpeL \\
        --learner-tag gpt5mini --learner-model gpt-5-mini --out results/matrix_v1/sets/gpt5mini \\
        --dry-run

    # 3-seed smoke, then the full 90 (resume-by-existence means the second run adds only what is missing)
    uv run scripts/run_grid_generate.py ... --seed-ids <pk1>,<pk2>,<pk3>
    uv run scripts/run_grid_generate.py ...
"""
import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import numpy as np  # noqa: E402

from social_omni_epic.adversarial_agent import AdversarialAgent          # noqa: E402
from social_omni_epic.coherence_check import CoherenceChecker            # noqa: E402
from social_omni_epic.expel_export import flush_aggregates               # noqa: E402
from social_omni_epic.fm import make_fm                                       # noqa: E402
from social_omni_epic.generation_cell import (                           # noqa: E402
    Services, build_grid_context, load_phase0_annotated_seeds,
    operator_for_band, run_generation_cell, write_compute_report,
    write_lineage, write_quarantine,
)
from social_omni_epic.meta_reflection import MetaReflectionModule        # noqa: E402
from social_omni_epic.model_of_interestingness import ModelOfInterestingness  # noqa: E402
from social_omni_epic.oracle_gate import (                              # noqa: E402
    probe_solvability, write_oracle_record, write_rejection,
)
from social_omni_epic.reflection_module import ReflectionModule          # noqa: E402
from social_omni_epic.scenario_title import ScenarioTitleGenerator       # noqa: E402
from social_omni_epic.task_generator import TaskGenerator                # noqa: E402
from social_omni_epic.tracing_fm import print_info, print_step, print_warn  # noqa: E402

MAX_CONCURRENCY = 8   # OpenAI-direct for episodes; the judge is the Lightning-limited part


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase0-dir", required=True,
                    help="Directory with seeds/seed_*.json carrying this learner's bands. REQUIRED: "
                         "no band means no operator, which means no calibration.")
    ap.add_argument("--learner-tag", required=True, help="Short stable tag, e.g. gpt5mini")
    ap.add_argument("--learner-model", required=True)
    ap.add_argument("--reflection-model", default=None,
                    help="Writes the Reflexion string the learner reads. Defaults to (and is "
                         "asserted equal to) --learner-model; see the assert below for why.")
    ap.add_argument("--generator-model", default="gpt-5.4")
    ap.add_argument("--gates-model", default="gpt-4.1-mini", help="coherence + MOI")
    ap.add_argument("--partner-model", default="gpt-5.4-mini")
    ap.add_argument("--judge-model", default="google/gemini-3-flash-preview")
    ap.add_argument("--oracle-model", default=None, help="Defaults to --generator-model")
    ap.add_argument("--seeds-path", default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--gen-batch-size", type=int, default=3)
    ap.add_argument("--num-examples", type=int, default=3)
    ap.add_argument("--diversity-threshold", type=float, default=0.92)
    ap.add_argument("--oracle-tries", type=int, default=3)
    ap.add_argument("--gen-rounds", type=int, default=3,
                    help="Regeneration rounds when every candidate in a batch is rejected.")
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--seed-ids", default=None,
                    help="Comma-separated env_pks — the ramp lever (smoke 3 / pilot 20 / full 90).")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--retry-failed", action="store_true",
                    help="Re-open cells previously recorded as generation_failed / discarded.")
    ap.add_argument("--no-oracle", action="store_true",
                    help="Skip the solvability gate. For debugging only — it is the thing that "
                         "keeps unwinnable scenarios out.")
    ap.add_argument("--allow-partner-eq-learner", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the (env_pk, band, operator) rows and exit. Free audit of the "
                         "entire calibration claim before spending any episodes.")
    return ap


# Curriculum keys that are MEANINGLESS here. Rejected rather than ignored: silently accepting
# `stopping.N` or `anchor_selection` would let someone believe a search was running when it wasn't.
DEAD_KEYS = ("anchor-selection", "stopping-n", "iterations", "batch-size",
             "seed-prior", "child-prior-mass", "niches-refit-every", "mechanism-library")


def cell_seed(run_seed: int, learner_tag: str, env_pk: str) -> int:
    """Per-cell derived RNG seed. Independence comes from input scoping (build_grid_context); this
    is the guarantee it STAYS that way if anyone later adds sampling to the generation path."""
    h = hashlib.blake2b(f"{run_seed}|{learner_tag}|{env_pk}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

async def run_cell(seed, all_seeds, bands, svc, config, args, run_dir, oracle_kwargs) -> dict:
    """Generate -> gates -> oracle -> K-loop for one seed. Returns a checkpoint row."""
    env_pk = seed.source_env_id
    tag = f"[{args.learner_tag}/{env_pk[:8]}]"
    child_id = f"{env_pk}__{args.learner_tag}"
    rng = np.random.default_rng(cell_seed(args.random_seed, args.learner_tag, env_pk))
    band = bands.get(env_pk)
    operator = operator_for_band(band)

    row = {
        "env_pk": env_pk, "learner_tag": args.learner_tag, "child_id": child_id,
        "band": band, "operator": operator, "terminal_state": None,
        "n_rounds": 0, "n_rejected": 0, "n_oracle_rejected": 0,
        "oracle": None, "reason": None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    n_rejected = 0
    for gen_round in range(1, args.gen_rounds + 1):
        ctx = build_grid_context(
            seed, all_seeds, bands,
            n_examples=args.num_examples,
            n_failed_examples=2,
            child_id=child_id,
            rng=rng,
        )
        state, scenario, info = await run_generation_cell(
            ctx, svc, config, iteration=0, run_dir=run_dir, tag=tag,
        )
        # NOT gen_batch_size: the gate walk admits the FIRST candidate that passes and never
        # evaluates the rest, so counting the whole batch as "proposed" reports unused candidates as
        # rejections. On the smoke that made a clean 0-rejection set look like 3/9 = 0.33 yield,
        # one bad round away from tripping the 0.25 abort threshold for no reason.
        row["n_rounds"] += 1

        if state == "generation_failed" or scenario is None:
            n_rejected += 1
            write_rejection(run_dir, child_id, n_rejected, scenario, "generation",
                            info.get("reason"))
            row["reason"] = info.get("reason")
            if gen_round < args.gen_rounds:
                print_warn(f"{tag} generation failed ({info.get('reason')}) — round {gen_round + 1}")
                continue
            row.update(terminal_state="generation_failed", n_rejected=n_rejected)
            return row

        # --- oracle solvability gate -------------------------------------------------
        # Runs AFTER the episode here rather than before it, deliberately: run_generation_cell
        # already played the scenario, so we have its band. Rejecting a broken item post-hoc still
        # keeps it out of the set, and we avoid a second full episode on every admitted candidate.
        if not args.no_oracle:
            verdict = await probe_solvability(
                scenario, max_tries=args.oracle_tries, tag=tag, **oracle_kwargs
            )
            write_oracle_record(run_dir, scenario.id, verdict)
            row["oracle"] = verdict.to_dict()
            if not verdict.admit:
                n_rejected += 1
                row["n_oracle_rejected"] = row.get("n_oracle_rejected", 0) + 1
                write_rejection(run_dir, child_id, n_rejected, scenario, "oracle",
                                verdict.to_dict())
                # Remove the bank record written by run_generation_cell — it is not part of the set.
                stale = run_dir / "bank" / "generated" / f"{scenario.id}.json"
                if stale.exists():
                    stale.unlink()
                if gen_round < args.gen_rounds:
                    print_warn(f"{tag} oracle rejected (best goal {verdict.best_goal:.1f}) — "
                               f"regenerating, round {gen_round + 1}")
                    continue
                row.update(terminal_state="oracle_rejected", n_rejected=n_rejected,
                           reason=f"unwinnable after {args.gen_rounds} round(s)")
                return row

        row.update(terminal_state=state, n_rejected=n_rejected,
                   scenario_id=scenario.id,
                   lp_value=info.get("lp_value"), lp_inference=info.get("lp_inference"),
                   goal=(info.get("final_scores") or {}).get("goal"))
        if state == "discarded":
            write_quarantine(run_dir, child_id, "discarded", info)
        return row

    row.update(terminal_state="generation_failed", n_rejected=n_rejected)
    return row


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def main_async(args) -> int:
    from social_omni_epic.episode_runner import run_single_episode
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    reflection_model = args.reflection_model or args.learner_model
    oracle_model = args.oracle_model or args.generator_model

    # --- guards. Each of these has silently corrupted a run before. -------------------
    # The reflection writer must be the learner: recoverability means "did the learner recover after
    # being told what went wrong", so a stronger writer measures the teacher — unequally per row,
    # which bends the diagonal.
    if reflection_model != args.learner_model:
        print(f"ERROR: --reflection-model ({reflection_model}) must equal --learner-model "
              f"({args.learner_model}).", file=sys.stderr)
        return 1
    # No `partner or learner` fallback: a partner that scales with the learner makes difficulty rise
    # with learner strength, cancelling the exact monotonicity the matrix measures.
    if args.partner_model == args.learner_model and not args.allow_partner_eq_learner:
        print(f"ERROR: partner == learner ({args.partner_model}) → self-play on this diagonal cell. "
              f"Pass --allow-partner-eq-learner to override.", file=sys.stderr)
        return 1
    if args.judge_model.split("/")[0] == args.learner_model.split("/")[0]:
        print("ERROR: judge provider matches learner provider — not cross-lab.", file=sys.stderr)
        return 1
    concurrency = min(args.concurrency, MAX_CONCURRENCY)

    # --- FMs: one per role, so the judge can use Lightning while the rest go direct to OpenAI ----
    fm_gen = make_fm(model=args.generator_model, temperature=1.0)
    fm_reflect = make_fm(model=reflection_model, temperature=1.0)
    fm_gates = make_fm(model=args.gates_model, temperature=0.2)
    fm_judge = make_fm(model=args.judge_model, temperature=0.3)

    svc = Services(
        fm_generator=fm_gen, fm_judge=fm_judge,
        fm_reflection=fm_reflect, fm_gates=fm_gates,
        task_gen=TaskGenerator(fm_gen, num_examples=args.num_examples,
                               num_failed_examples=0, max_retries=3),
        moi=ModelOfInterestingness(fm_gates, num_examples=5),
        coherence_checker=CoherenceChecker(fm_gates),
        title_gen=ScenarioTitleGenerator(fm_gen),
        reflection_mod=ReflectionModule(fm_reflect),
        meta_mod=MetaReflectionModule(fm_reflect),
        adversarial=AdversarialAgent(fm_gates),
        run_single_episode=run_single_episode,
        scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
    )

    config = {
        "learner_model": args.learner_model, "partner_model": args.partner_model,
        "max_attempts": args.max_attempts, "max_turns": args.max_turns,
        "use_expel_memory": True, "chronicle_max_entries": 8,
        "gen_batch_size": args.gen_batch_size,
        "enable_moi": True, "enable_coherence_check": True, "coherence_max_retries": 2,
        "enable_diversity_gate": True,
        "diversity_similarity_threshold": args.diversity_threshold,
        "adversarial": {"re_reflect_on_rejection": True},
        "task_generator": {"num_examples": args.num_examples,
                           "num_episode_failed_examples": 2, "show_existing_types": True},
        # The grid has no Thompson posterior, so LP votes on a solved scenario are computed and
        # discarded. LP cannot change the band of a solved episode (`solved or lp>0 -> frontier`).
        "skip_lp_when_solved": True,
    }

    # --- per-learner seed bands ------------------------------------------------------
    all_seeds, bands = load_phase0_annotated_seeds(
        Path(args.phase0_dir), args.seeds_path, fm_gen,
    )
    uncal = [pk for pk, b in bands.items() if not b]
    if uncal:
        print(f"ERROR: {len(uncal)} seed(s) have no phase-0 band, so no operator can be chosen: "
              f"{uncal[:5]}", file=sys.stderr)
        return 1

    wanted = ([s.strip() for s in args.seed_ids.split(",") if s.strip()]
              if args.seed_ids else list(bands.keys()))
    seeds_by_pk = {s.source_env_id: s for s in all_seeds}
    todo = [seeds_by_pk[pk] for pk in wanted if pk in seeds_by_pk]
    missing = [pk for pk in wanted if pk not in seeds_by_pk]
    if missing:
        print_warn(f"unknown env_pk(s) skipped: {missing}")

    # A requested seed with no phase-0 band would fall through to operator_for_band(None) ==
    # "lateral", i.e. silently generate an UNCALIBRATED cell that looks identical to a calibrated
    # one in every artifact. Since phase-0 may legitimately cover only a subset (that is how the
    # smoke/pilot ramp stays cheap), this has to be an error rather than a warning.
    unbanded = [s.source_env_id for s in todo if not bands.get(s.source_env_id)]
    if unbanded:
        print(f"ERROR: {len(unbanded)} requested seed(s) have no band in {args.phase0_dir}, so no "
              f"operator can be chosen: {unbanded[:5]}\n"
              f"       Run phase-0 over these seeds first, or restrict --seed-ids to banded ones.",
              file=sys.stderr)
        return 1

    # --- dry run: audit the calibration claim for free -------------------------------
    if args.dry_run:
        counts: dict = {}
        print(f"\n{'env_pk':<30} {'band':<16} operator")
        for s in todo:
            b = bands.get(s.source_env_id)
            op = operator_for_band(b)
            counts[op] = counts.get(op, 0) + 1
            print(f"{s.source_env_id:<30} {str(b):<16} {op}")
        print(f"\n{len(todo)} cells. operators: {counts}")
        print(f"models: learner={args.learner_model} reflection={reflection_model} "
              f"generator={args.generator_model} gates={args.gates_model} "
              f"partner={args.partner_model} judge={args.judge_model} oracle={oracle_model}")
        return 0

    # --- resume: checkpoint pattern from audit_gen90_partner_leaks.py ----------------
    manifest = run_dir / "grid_manifest.json"
    ckpt = manifest.with_suffix(".jsonl")
    done: dict = {}
    if args.resume and ckpt.exists():
        for line in ckpt.read_text().splitlines():
            try:
                r = json.loads(line)
                done[r["env_pk"]] = r
            except Exception:
                pass
        reopen = {"generation_failed", "discarded"} if args.retry_failed else set()
        before = len(todo)
        todo = [s for s in todo
                if s.source_env_id not in done
                or done[s.source_env_id].get("terminal_state") in reopen]
        print_info(f"resuming: {before - len(todo)} cell(s) already done, {len(todo)} to go")
    elif ckpt.exists():
        ckpt.unlink()   # --no-resume TRUNCATES; it must never silently append twice

    oracle_kwargs = dict(
        run_single_episode=run_single_episode,
        scenario_to_sotopia_profiles=scenario_to_sotopia_profiles,
        fm=fm_gen, fm_judge=fm_judge,
        oracle_model=oracle_model, partner_model=args.partner_model,
        max_turns=args.max_turns,
    )

    print_step(f"grid: {len(todo)} cell(s) for {args.learner_tag} → {run_dir} "
               f"(concurrency {concurrency})")

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    t0 = time.time()

    with ckpt.open("a", encoding="utf-8") as fh:
        async def _slot(seed):
            async with sem:
                try:
                    row = await run_cell(seed, all_seeds, bands, svc, config, args,
                                         run_dir, oracle_kwargs)
                except Exception as e:
                    import traceback
                    print_warn(f"[{seed.source_env_id[:8]}] cell crashed: {e}\n{traceback.format_exc()}")
                    row = {"env_pk": seed.source_env_id, "learner_tag": args.learner_tag,
                           "terminal_state": "discarded", "reason": f"cell_exception: {e}"}
                async with write_lock:
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
                return row

        await asyncio.gather(*[_slot(s) for s in todo], return_exceptions=True)

    # --- final report, rebuilt FROM THE CHECKPOINT so a resumed run summarises everything ---
    rows = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    by_pk = {r["env_pk"]: r for r in rows}
    bands_out: dict = {}
    holes: dict = {}
    oracle_rejected = admitted = rounds = 0
    for r in by_pk.values():
        st = r.get("terminal_state")
        rounds += int(r.get("n_rounds") or 0)
        oracle_rejected += int(r.get("n_oracle_rejected") or 0)
        if st in ("too_easy", "frontier", "beyond_frontier"):
            bands_out[st] = bands_out.get(st, 0) + 1
            admitted += 1
        else:
            holes[r["env_pk"]] = st or "unknown"
    judged = admitted + oracle_rejected

    report = {
        "learner_tag": args.learner_tag,
        "models": {"learner": args.learner_model, "reflection": reflection_model,
                   "generator": args.generator_model, "gates": args.gates_model,
                   "partner": args.partner_model, "judge": args.judge_model,
                   "oracle": oracle_model},
        "commit": _git_commit(),
        "n_cells": len(by_pk),
        "band_counts": bands_out,
        "paired_complete": sorted(pk for pk, r in by_pk.items()
                                  if r.get("terminal_state") in
                                  ("too_easy", "frontier", "beyond_frontier")),
        "holes": holes,
        # Of the candidates the ORACLE actually judged, how many were winnable. This is the
        # artifact rate; it is not diluted by unused batch candidates.
        "oracle_yield": round(admitted / judged, 4) if judged else None,
        "n_oracle_judged": judged, "n_oracle_rejected": oracle_rejected,
        "n_admitted": admitted, "n_generation_rounds": rounds,
        "elapsed_s": round(time.time() - t0, 1),
        "config": {k: v for k, v in vars(args).items()},
    }
    manifest.write_text(json.dumps(report, indent=2, default=str))
    flush_aggregates(run_dir, learner_model=args.learner_model, judge_model=args.judge_model)
    write_lineage(run_dir, all_seeds + [])
    write_compute_report(run_dir, {"generator": fm_gen, "reflection": fm_reflect,
                                   "gates": fm_gates, "judge": fm_judge})

    print(f"\n=== {args.learner_tag} ===")
    print(f"  bands       {bands_out}")
    print(f"  holes       {len(holes)} {list(holes.items())[:5]}")
    print(f"  oracle      {admitted}/{judged} judged-and-admitted "
          f"(yield {report['oracle_yield'] if report['oracle_yield'] is not None else 'n/a'}); "
          f"{oracle_rejected} rejected as unwinnable over {rounds} generation round(s)")
    print(f"  elapsed     {report['elapsed_s']}s")
    if report["oracle_yield"] is not None and report["oracle_yield"] < 0.25:
        print("\nWARNING: oracle yield below 25%. The generator is producing unwinnable scenarios "
              "faster than the gate can filter them — fix the generation prompt rather than "
              "spending more compute.", file=sys.stderr)
    return 0


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    argv = sys.argv[1:]
    for dead in DEAD_KEYS:
        if any(a.startswith(f"--{dead}") for a in argv):
            print(f"ERROR: --{dead} is meaningless in the paired grid (there is no anchor search "
                  f"and N is fixed at one child per seed). Remove it.", file=sys.stderr)
            sys.exit(2)
    args = build_parser().parse_args(argv)
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
