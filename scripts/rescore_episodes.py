"""Copy Gemini-judged attempt-1 scores from phase0 seeds into baseline episode JSONs.

run_phase0_calibration.py already called score_transcript(transcript1, fm_judge, ...)
on every baseline episode and stored the result in attempts[0]['scores']. This script
copies those scores into the original baseline episode files so downstream comparisons
use consistent judge-model scores without re-calling the API.

Reads from : results/phase0_Base90_Skills_Chronicle/seeds/
Writes to  : results/baseline_eval_20260604_222545_rescored/episodes/  (or --in-place)

Usage:
    # default: write rescored files to a new directory
    python scripts/rescore_episodes.py

    # write in-place (overwrites scores field in the original files)
    python scripts/rescore_episodes.py --in-place

    # explicit paths
    python scripts/rescore_episodes.py \\
        --phase0-seeds  results/phase0_Base90_Skills_Chronicle/seeds \\
        --episodes      results/baseline_eval_20260604_222545/episodes \\
        --out           results/baseline_eval_rescored/episodes
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(
        description="Copy Gemini attempt-1 scores from phase0 seeds into baseline episode JSONs"
    )
    ap.add_argument("--phase0-seeds", type=str,
                    default="results/phase0_Base90_Skills_Chronicle/seeds",
                    help="Directory of phase0 seed JSONs (source of Gemini-judged scores)")
    ap.add_argument("--episodes", type=str,
                    default="results/baseline_eval_20260604_222545/episodes",
                    help="Directory of baseline episode JSONs to update")
    ap.add_argument("--out", type=str, default=None,
                    help="Output directory (default: <episodes parent>_rescored/episodes)")
    ap.add_argument("--in-place", action="store_true", default=False,
                    help="Overwrite the original files instead of writing to --out")
    args = ap.parse_args()

    if args.in_place and args.out:
        print("ERROR: --in-place and --out are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    phase0_dir = Path(args.phase0_seeds)
    episodes_dir = Path(args.episodes)

    for d in (phase0_dir, episodes_dir):
        if not d.exists():
            print(f"ERROR: {d} does not exist", file=sys.stderr)
            sys.exit(1)

    if args.in_place:
        out_dir = None
    elif args.out:
        out_dir = Path(args.out)
    else:
        out_dir = episodes_dir.parent.parent / (episodes_dir.parent.name + "_rescored") / "episodes"

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Build index: seed_idx → phase0 attempt-1 scores + classification fields
    phase0_scores: dict[int, dict] = {}
    phase0_meta: dict[int, dict] = {}   # classification, terminal_success
    for p in phase0_dir.glob("seed_*.json"):
        try:
            d = json.loads(p.read_text())
            idx = int(d.get("seed_idx", -1))
            att1 = next((a for a in d.get("attempts", []) if a.get("attempt") == 1), None)
            if att1 and att1.get("scores"):
                phase0_scores[idx] = att1["scores"]
            phase0_meta[idx] = {
                "classification": d.get("classification"),
                "terminal_success": d.get("terminal_success"),
            }
        except Exception as e:
            print(f"  WARN: could not read {p.name}: {e}")

    print(f"Loaded Gemini scores for {len(phase0_scores)} seeds from {phase0_dir}")
    print(f"Episodes dir : {episodes_dir}")
    print(f"Output       : {'(in-place)' if args.in_place else out_dir}")
    print()

    n_ok = n_missing = n_unchanged = 0
    for ep_path in sorted(episodes_dir.glob("seed_*.json")):
        dest = (out_dir / ep_path.name) if out_dir else ep_path
        try:
            record = json.loads(ep_path.read_text())
        except Exception as e:
            print(f"  READ ERROR {ep_path.name}: {e}")
            continue

        idx = int(record.get("seed_idx", -1))
        new_scores = phase0_scores.get(idx)

        if new_scores is None:
            print(f"  [{idx:3d}] {ep_path.name} — MISSING in phase0 (kept original)")
            n_missing += 1
            if out_dir:
                dest.write_text(json.dumps(record, indent=2, default=str))
            continue

        old_goal = record.get("scores", {}).get("goal", "?")
        new_goal = new_scores.get("goal", "?")

        meta = phase0_meta.get(idx, {})
        changed = record.get("scores") != new_scores

        # Rebuild dict with phase0-matching field order
        KEY_ORDER = [
            "scenario_title", "social_dynamic", "target_perspective",
            "scenario", "learner_goal", "partner_goal",
            "seed_idx", "env_pk", "id", "source", "is_sotopia_hard",
            "classification", "n_attempts", "terminal_success", "scores",
        ]
        ordered: dict = {}
        for k in KEY_ORDER:
            if k == "classification":
                ordered[k] = meta.get("classification")
            elif k == "n_attempts":
                ordered[k] = None
            elif k == "terminal_success":
                # Derived from this episode's own scores (attempt-1 only); not copied from phase0
                # (phase0 terminal_success reflects the *final* attempt, which may differ)
                goal = new_scores.get("goal", 0) if new_scores else 0
                rel = new_scores.get("relationship", 0) if new_scores else 0
                ordered[k] = bool(goal >= 7.0 and rel >= 0.0)
            elif k == "scores":
                ordered[k] = new_scores
            elif k in record:
                ordered[k] = record[k]
        # append any remaining keys not in KEY_ORDER
        for k, v in record.items():
            if k not in ordered:
                ordered[k] = v

        if changed:
            n_ok += 1
        else:
            n_unchanged += 1

        dest.write_text(json.dumps(ordered, indent=2, default=str))
        print(f"  [{idx:3d}] {ep_path.name}  goal: {old_goal} → {new_goal}  cls: {meta.get('classification')}")

    print(f"\nDone. updated={n_ok}  unchanged={n_unchanged}  missing={n_missing}")
    if out_dir:
        print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
