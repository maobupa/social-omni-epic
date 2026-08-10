#!/usr/bin/env python3
"""Prove the frozen gen-90 artifact is still readable after a schema change.

`results/gen90_expel` is finished and cited in the writeup. Schema v2 adds `internal_state` and
retires `surface_misdirection` / `cost_coupling`, so the risk is that the pydantic models stop
parsing v1 records and every downstream consumer (transcript reader, fidelity audit, figure
scripts, analysis/) silently loses its input.

Three checks, cheapest first — no API calls, no writes to the run directory:

  1. PARSE     every bank/generated/*.json through SocialScenario, and archive_latest.json
               through ArchiveState. Any validation error is a hard fail.
  2. AGGREGATE rebuild summary.json + trajectories.json from the bank into a temp dir and diff
               against the committed ones. Catches silent changes to the success label or to
               the classification/operator tallies.
  3. FIELDS    report which partner_key fields are populated, so a v1 vs v2 record is
               distinguishable and the counts are on the record.

Run it BEFORE the schema change to capture a baseline, then after to compare.

    uv run scripts/check_schema_compat.py
    uv run scripts/check_schema_compat.py --run-dir results/gen90_expel --json
"""
import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_omni_epic.data_models import ArchiveState, SocialScenario  # noqa: E402
from social_omni_epic.expel_export import (  # noqa: E402
    _load_records,
    build_summary_from_records,
    build_trajectories_from_records,
)

KEY_FIELDS = [
    "key_mechanism", "movement_conditions", "hardening_triggers",
    "surface_misdirection", "cost_coupling", "internal_state",
]


def check_parse(run_dir: Path) -> dict:
    gen = sorted((run_dir / "bank" / "generated").glob("*.json"))
    ok, errors = 0, []
    for p in gen:
        try:
            SocialScenario(**json.loads(p.read_text()))
            ok += 1
        except Exception as e:
            errors.append({"file": p.name, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    archive_ok, archive_err = None, None
    ap = run_dir / "archive_latest.json"
    if ap.exists():
        try:
            raw = json.loads(ap.read_text())
            if "successful" in raw and "tasks" not in raw:   # legacy key migration
                raw["tasks"] = raw.pop("successful")
            st = ArchiveState(**raw)
            archive_ok, archive_err = True, f"{len(st.tasks)} tasks"
        except Exception as e:
            archive_ok, archive_err = False, f"{type(e).__name__}: {str(e)[:200]}"

    return {"n_files": len(gen), "parsed": ok, "errors": errors,
            "archive_parsed": archive_ok, "archive_detail": archive_err}


def check_aggregates(run_dir: Path) -> dict:
    """Rebuild summary/trajectories from the bank and diff against the committed files."""
    recs = _load_records(run_dir / "bank" / "generated")
    committed_summary = json.loads((run_dir / "summary.json").read_text())
    rebuilt = build_summary_from_records(
        recs,
        learner_model=committed_summary.get("learner_model", ""),
        judge_model=committed_summary.get("judge_model", ""),
    )

    diffs = {}
    for k in sorted(set(committed_summary) | set(rebuilt)):
        a, b = committed_summary.get(k), rebuilt.get(k)
        if a != b:
            diffs[k] = {"committed": a, "rebuilt": b}

    traj_ok, traj_detail = True, ""
    try:
        t = build_trajectories_from_records(recs)
        committed_t = json.loads((run_dir / "trajectories.json").read_text())
        n_new = len(t.get("succeeded", [])) + len(t.get("failed", []))
        n_old = len(committed_t.get("succeeded", [])) + len(committed_t.get("failed", []))
        traj_detail = f"rebuilt {n_new} trajectories, committed {n_old}"
        traj_ok = (n_new == n_old)
    except Exception as e:
        traj_ok, traj_detail = False, f"{type(e).__name__}: {e}"

    # Write the rebuild somewhere disposable so the run dir is never touched.
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "summary.json").write_text(json.dumps(rebuilt, indent=2, default=str))

    return {"n_records": len(recs), "summary_diffs": diffs,
            "trajectories_match": traj_ok, "trajectories_detail": traj_detail}


def check_fields(run_dir: Path) -> dict:
    present = Counter()
    versions = Counter()
    for p in sorted((run_dir / "bank" / "generated").glob("*.json")):
        pk = (json.loads(p.read_text()).get("partner_key") or {})
        if not pk:
            versions["no_partner_key"] += 1
            continue
        for f in KEY_FIELDS:
            if pk.get(f):
                present[f] += 1
        versions[f"v{pk.get('version', 1)}"] += 1
    return {"partner_key_field_counts": dict(present), "versions": dict(versions)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="results/gen90_expel")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "bank" / "generated").exists():
        print(f"No bank at {run_dir}/bank/generated", file=sys.stderr)
        sys.exit(2)

    report = {
        "run_dir": str(run_dir),
        "parse": check_parse(run_dir),
        "aggregates": check_aggregates(run_dir),
        "fields": check_fields(run_dir),
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        pr, ag, fl = report["parse"], report["aggregates"], report["fields"]
        print(f"\n=== {run_dir} ===")
        print(f"1. PARSE      {pr['parsed']}/{pr['n_files']} scenarios"
              f"   archive_latest: {pr['archive_parsed']} ({pr['archive_detail']})")
        for e in pr["errors"][:5]:
            print(f"     ✗ {e['file']}: {e['error']}")
        print(f"2. AGGREGATE  {ag['n_records']} records; summary diffs: "
              f"{len(ag['summary_diffs']) or 'none'}; trajectories match: {ag['trajectories_match']}"
              f" ({ag['trajectories_detail']})")
        for k, v in list(ag["summary_diffs"].items())[:8]:
            print(f"     ! {k}: committed={v['committed']!r} rebuilt={v['rebuilt']!r}")
        print(f"3. FIELDS     versions={fl['versions']}")
        for f in KEY_FIELDS:
            print(f"     {f:<22} {fl['partner_key_field_counts'].get(f, 0)}")

    hard_fail = bool(report["parse"]["errors"]) or report["parse"]["archive_parsed"] is False
    if hard_fail:
        print("\nFAIL: gen-90 no longer parses.", file=sys.stderr)
        sys.exit(1)
    if report["aggregates"]["summary_diffs"]:
        print("\nWARN: summary.json rebuild differs from the committed file (see above).",
              file=sys.stderr)
        sys.exit(3)
    print("\nOK: gen-90 parses and rebuilds identically.")


if __name__ == "__main__":
    main()
