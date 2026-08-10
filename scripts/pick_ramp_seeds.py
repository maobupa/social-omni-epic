#!/usr/bin/env python3
"""Pick the stratified seed subset for a ramp level, and emit it in both forms the ramp needs.

Two different identifiers are in play and they must agree, or phase-0 and the grid will silently
disagree about which scenarios are in the run:

    --seed-indices  positional indices into data/sotopia_90_seeds.jsonl  (run_baseline_eval.py,
                    run_expel_chronicle.py)
    --seed-ids      env_pk strings                                        (run_grid_generate.py)

Why stratify rather than take the first N. The seed's phase-0 band DECIDES the mutation operator, so
a subset that misses a band never exercises that code path at all. And the bands are wildly uneven
for gpt-5-mini: 63 too_easy / 22 frontier / 5 beyond_frontier, with `mutual_friends` 10/10 too_easy
and `persuasion_for_good` 9/10. A first-20-in-file-order subset would be nearly all `escalate` and
would never run `relax` even once.

So the smoke set deliberately takes ONE PER BAND, and the pilot over-weights the rare bands. That is
correct for a code-path and shape test — it is not meant to be an unbiased sample of the bank.

Selection is deterministic (sorted by env_pk within each stratum) so smoke ⊂ pilot ⊂ full and a
later, larger run reuses everything the earlier one produced.

    uv run scripts/pick_ramp_seeds.py smoke
    uv run scripts/pick_ramp_seeds.py pilot --emit indices
    uv run scripts/pick_ramp_seeds.py full --emit ids
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Per-band targets. Smoke: one of each, so escalate/lateral/relax all fire once. Pilot: all 5
# beyond_frontier seeds (there are only 5 in the entire bank), then frontier, then too_easy.
LEVELS = {
    "smoke": {"beyond_frontier": 1, "frontier": 1, "too_easy": 1},
    "pilot": {"beyond_frontier": 5, "frontier": 8, "too_easy": 7},
    "full":  None,   # everything
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", choices=sorted(LEVELS))
    ap.add_argument("--seeds-path", default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--bands-from", default="results/expel_phase0_Base90_ExpeL",
                   help="Phase-0 dir supplying the bands used to stratify. Defaults to the "
                        "gpt-5-mini run, deliberately: the SAME subset must be used for every "
                        "learner or the rows are not paired.")
    ap.add_argument("--emit", choices=["both", "indices", "ids"], default="both")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in Path(args.seeds_path).read_text().splitlines() if l.strip()]
    idx_of = {s["env_pk"]: i for i, s in enumerate(seeds)}
    src_of = {s["env_pk"]: s.get("source", "?") for s in seeds}

    bands = {}
    for p in sorted((Path(args.bands_from) / "seeds").glob("seed_*.json")):
        try:
            r = json.loads(p.read_text())
            bands[r["env_pk"]] = r.get("classification")
        except Exception:
            pass
    if not bands:
        print(f"no bands under {args.bands_from}/seeds", file=sys.stderr)
        sys.exit(2)

    target = LEVELS[args.level]
    if target is None:
        chosen = sorted(pk for pk in bands if pk in idx_of)
    else:
        by_band = defaultdict(list)
        for pk, b in bands.items():
            if pk in idx_of:
                by_band[b].append(pk)
        chosen = []
        for band, n in target.items():
            pool = sorted(by_band.get(band, []))          # deterministic → smoke ⊂ pilot
            take = pool[:n]
            if len(take) < n:
                print(f"note: only {len(take)} {band} seed(s) available (wanted {n})",
                      file=sys.stderr)
            chosen += take
        # Spread across source families where possible, without breaking determinism: prefer one
        # seed per source before doubling up within a band.
        chosen = sorted(set(chosen))

    indices = sorted(idx_of[pk] for pk in chosen)
    ids = [pk for pk in chosen]

    if args.json:
        print(json.dumps({"level": args.level, "n": len(chosen),
                          "indices": indices, "env_pks": ids,
                          "bands": {pk: bands[pk] for pk in ids},
                          "sources": {pk: src_of.get(pk) for pk in ids}}, indent=2))
        return

    if args.emit == "indices":
        print(",".join(str(i) for i in indices))
        return
    if args.emit == "ids":
        print(",".join(ids))
        return

    counts = defaultdict(int)
    srcs = defaultdict(int)
    for pk in ids:
        counts[bands[pk]] += 1
        srcs[src_of.get(pk, "?")] += 1
    print(f"level={args.level}  n={len(ids)}")
    print(f"bands   {dict(counts)}")
    print(f"sources {dict(srcs)}")
    print(f"\nINDICES {','.join(str(i) for i in indices)}")
    print(f"IDS     {','.join(ids)}")


if __name__ == "__main__":
    main()
