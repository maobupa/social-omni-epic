#!/usr/bin/env python3
"""Assemble the model x scenario-set matrix and print it.

The figure this produces is the whole point of the redesign. Rows are scenario sets, columns are the
learner that played them, and Row 0 is the raw SOTOPIA seeds:

    scenario set        | learner W      | learner M      | learner S
    --------------------+----------------+----------------+---------------
    Row 0 raw SOTOPIA   | mostly easy    | easy           | VERY easy   <- saturation
    calibrated to W     | FRONTIER       | easy           | easy
    calibrated to M     | beyond         | FRONTIER       | easy
    calibrated to S     | beyond         | beyond         | FRONTIER

Frontier on the diagonal, easy above it, beyond below it, and Row 0 flat and saturating. That single
picture carries three claims at once: the human-authored benchmark runs out of headroom as models
improve (Row 0), our method puts the frontier wherever it is aimed (the diagonal), and the items
order models correctly rather than being noise (off-diagonal).

Also reported, because each is load-bearing for a different objection:
  * ORACLE YIELD      admitted/proposed per set. The artifact rate is a headline number — no
                      adaptive benchmark has published one, and the standing critique of the genre
                      (Bowman & Dahl) is precisely that they drift into artifacts.
  * VERDICT DISAGREE  how often the staged gate and the state-only verdict differ. The only case
                      they can is hollow performance (a listed condition ticked without the person
                      actually being reached). Rare -> the conservative gate is fine permanently;
                      common -> we have the evidence to switch.
  * HOLES + PAIRED-DROP  a seed missing from EITHER set is dropped from BOTH, so the comparison is
                      never silently 88-vs-90.

    uv run scripts/build_matrix.py
    uv run scripts/build_matrix.py --matrix-root results/matrix_v1 --csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BANDS = ("too_easy", "frontier", "beyond_frontier")


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_row0(root: Path) -> dict:
    """Row 0 = the raw SOTOPIA seeds, per learner. This comes free: calibrating the seeds against a
    learner is a REQUIRED input for generation (it is what supplies the band), so it is not overhead.
    """
    out = {}
    for d in sorted((root / "phase0").glob("*")):
        if not d.is_dir():
            continue
        s = read_json(d / "summary.json")
        if not s:
            continue
        counts = s.get("classification_counts") or {}
        n = sum(counts.values()) or 1
        out[d.name] = {
            "band_counts": counts,
            "n": sum(counts.values()),
            "too_easy_frac": round(counts.get("too_easy", 0) / n, 3),
        }
    return out


def load_cells(root: Path) -> dict:
    """crossplay/<set>__<learner>/summary.json for every cell that exists."""
    cells = {}
    for d in sorted((root / "crossplay").glob("*__*")):
        s = read_json(d / "summary.json")
        if not s:
            continue
        set_tag, _, learner_tag = d.name.partition("__")
        # Per-scenario rows let us do the paired drop and the disagreement rate.
        rows = [read_json(p) for p in sorted((d / "episodes").glob("*.json"))]
        s["_rows"] = [r for r in rows if r]
        cells[(set_tag, learner_tag)] = s
    return cells


def paired_seed_set(cells: dict) -> set:
    """Seeds present in EVERY cell. A seed missing anywhere is dropped everywhere.

    Without this a set with 2 generation holes silently gets compared 88-vs-90, and the difference
    reads as a difficulty effect.
    """
    per_cell = []
    for s in cells.values():
        per_cell.append({r.get("root_seed_env_pk") for r in s["_rows"]
                         if r.get("root_seed_env_pk")})
    if not per_cell:
        return set()
    common = per_cell[0]
    for s in per_cell[1:]:
        common &= s
    return common


def band_counts(rows: list, keep: set = None) -> Counter:
    c = Counter()
    for r in rows:
        if keep is not None and r.get("root_seed_env_pk") not in keep:
            continue
        if r.get("band"):
            c[r["band"]] += 1
    return c


def disagreement_rate(rows: list) -> tuple:
    """staged vs state-only verdict, over every attempt that recorded both."""
    n = dis = 0
    for r in rows:
        for v in (r.get("key_check_verdicts") or []):
            if not isinstance(v, dict) or "verdicts_disagree" not in v:
                continue
            n += 1
            dis += bool(v["verdicts_disagree"])
    return dis, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default="results/matrix_v1")
    ap.add_argument("--csv", action="store_true", help="also write analysis/matrix.csv")
    ap.add_argument("--no-paired-drop", action="store_true",
                    help="Report every cell on its own scenarios. Off by default because unequal "
                         "n across cells reads as a difficulty effect.")
    args = ap.parse_args()

    root = Path(args.matrix_root)
    row0 = load_row0(root)
    cells = load_cells(root)
    if not cells and not row0:
        print(f"nothing under {root} yet — run the grid and crossplay first.", file=sys.stderr)
        sys.exit(2)

    keep = None if args.no_paired_drop else (paired_seed_set(cells) or None)

    set_tags = sorted({s for s, _ in cells})
    learner_tags = sorted({l for _, l in cells} | set(row0))

    # ----- the table -----
    w = max([12] + [len(t) + 2 for t in learner_tags])
    print(f"\n{'=' * (22 + w * len(learner_tags))}")
    print(f"MATRIX  ({root})")
    if keep is not None:
        print(f"paired on {len(keep)} seed(s) common to all {len(cells)} cell(s)")
    print(f"{'=' * (22 + w * len(learner_tags))}")
    header = f"{'scenario set':<22}" + "".join(f"{t:<{w}}" for t in learner_tags)
    print(header)
    print("-" * len(header))

    def fmt(c: Counter, n: int) -> str:
        if not n:
            return "—"
        return "/".join(str(c.get(b, 0)) for b in BANDS)

    if row0:
        line = f"{'Row 0 raw SOTOPIA':<22}"
        for t in learner_tags:
            r = row0.get(t)
            line += f"{(fmt(Counter(r['band_counts']), r['n']) if r else '—'):<{w}}"
        print(line)

    for st in set_tags:
        line = f"{('calibrated→' + st):<22}"
        for lt in learner_tags:
            cell = cells.get((st, lt))
            if not cell:
                line += f"{'—':<{w}}"
                continue
            c = band_counts(cell["_rows"], keep)
            mark = "*" if st == lt else " "
            line += f"{fmt(c, sum(c.values())) + mark:<{w}}"
        print(line)
    print("\ncells are too_easy/frontier/beyond_frontier   * = diagonal")

    # ----- cold pass rate, the capability readout -----
    print(f"\n{'cold pass rate':<22}" + "".join(f"{t:<{w}}" for t in learner_tags))
    print("-" * len(header))
    for st in set_tags:
        line = f"{('calibrated→' + st):<22}"
        for lt in learner_tags:
            cell = cells.get((st, lt))
            if not cell:
                line += f"{'—':<{w}}"
                continue
            rows = [r for r in cell["_rows"]
                    if keep is None or r.get("root_seed_env_pk") in keep]
            n = len(rows)
            cold = sum(1 for r in rows if r.get("first_attempt_solved"))
            line += f"{(f'{cold}/{n}' if n else '—'):<{w}}"
        print(line)

    # ----- per-set generation health -----
    print("\nGENERATION HEALTH")
    for st in set_tags:
        mf = read_json(root / "sets" / st / "grid_manifest.json") or {}
        holes = mf.get("holes") or {}
        yield_ = mf.get("oracle_yield")
        print(f"  {st:<14} oracle_yield={yield_ if yield_ is not None else 'n/a':<8} "
              f"admitted={mf.get('n_admitted', '?')}/{mf.get('n_proposed', '?')}  "
              f"holes={len(holes)} {list(holes.items())[:3]}")
        if yield_ is not None and yield_ < 0.25:
            print("    ^ below 25%: the generator is producing unwinnable scenarios faster than "
                  "the gate filters them. Fix the prompt, not the compute.")

    # ----- staged vs state-only, the deferred decision -----
    print("\nGRADING: staged gate vs state-only verdict")
    tot_dis = tot_n = 0
    for (st, lt), cell in sorted(cells.items()):
        dis, n = disagreement_rate(cell["_rows"])
        tot_dis += dis
        tot_n += n
        if n:
            print(f"  {st}__{lt:<14} {dis}/{n} attempts disagree ({dis / n:.1%})")
    if tot_n:
        print(f"  OVERALL           {tot_dis}/{tot_n} ({tot_dis / tot_n:.1%})")
        print("  Disagreement = hollow performance: a listed condition ticked without the person "
              "actually being reached.")
        print("  Low  -> the conservative staged gate is fine permanently.")
        print("  High -> switch to state-only grading; the evidence is now in hand.")
    else:
        print("  no verdict data yet (needs cells generated with schema v2)")

    # ----- artifacts -----
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix_root": str(root),
        "paired_seeds": sorted(keep) if keep else None,
        "row0": row0,
        "cells": {
            f"{st}__{lt}": {
                "diagonal": st == lt,
                "band_counts": dict(band_counts(c["_rows"], keep)),
                "cold_pass": sum(1 for r in c["_rows"]
                                 if (keep is None or r.get("root_seed_env_pk") in keep)
                                 and r.get("first_attempt_solved")),
                "n": len([r for r in c["_rows"]
                          if keep is None or r.get("root_seed_env_pk") in keep]),
                "verdict_disagreement": disagreement_rate(c["_rows"]),
            }
            for (st, lt), c in cells.items()
        },
        "generation_health": {
            st: {k: v for k, v in (read_json(root / "sets" / st / "grid_manifest.json") or {}).items()
                 if k in ("oracle_yield", "n_admitted", "n_proposed", "holes", "band_counts",
                          "models", "commit")}
            for st in set_tags
        },
    }
    (out_dir / "matrix.json").write_text(json.dumps(payload, indent=2, default=str))

    if args.csv:
        with (out_dir / "matrix.csv").open("w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["set", "learner", "diagonal", "n", "too_easy", "frontier",
                         "beyond_frontier", "cold_pass"])
            for (st, lt), c in sorted(cells.items()):
                cnt = band_counts(c["_rows"], keep)
                rows = [r for r in c["_rows"]
                        if keep is None or r.get("root_seed_env_pk") in keep]
                wr.writerow([st, lt, st == lt, len(rows), cnt.get("too_easy", 0),
                             cnt.get("frontier", 0), cnt.get("beyond_frontier", 0),
                             sum(1 for r in rows if r.get("first_attempt_solved"))])
        print(f"\nwrote {out_dir / 'matrix.csv'}")
    print(f"wrote {out_dir / 'matrix.json'}")

    # ----- commit-provenance guard -----
    commits = {st: (read_json(root / "sets" / st / "grid_manifest.json") or {}).get("commit")
               for st in set_tags}
    distinct = {c for c in commits.values() if c and c != "unknown"}
    if len(distinct) > 1:
        print(f"\nWARNING: sets were generated at different commits {commits}. Pilot output only "
              f"rolls into a full run if nothing changed in between — otherwise the rows are not "
              f"comparable.", file=sys.stderr)


if __name__ == "__main__":
    main()
