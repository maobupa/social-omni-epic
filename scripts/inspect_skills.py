"""Inspect skills chronicles stored in an archive checkpoint.

Usage:
  python scripts/inspect_skills.py                              # output/debug/archive_latest.json
  python scripts/inspect_skills.py output/200_full/archive_latest.json
  python scripts/inspect_skills.py output/run/archive_latest.json --only-generated
  python scripts/inspect_skills.py output/run/archive_latest.json --id ec9870d7
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from social_omni_epic.skills_chronicle import SkillsChronicle

SEP = "─" * 80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?",
                    default="output/debug/archive_latest.json",
                    help="Path to archive_latest.json (default: output/debug/archive_latest.json)")
    ap.add_argument("--only-generated", action="store_true",
                    help="Skip seed scenarios (iteration < 0)")
    ap.add_argument("--only-with-chronicle", action="store_true",
                    help="Skip scenarios with empty skills_final_md")
    ap.add_argument("--id", type=str, default=None,
                    help="Show only the scenario whose id starts with this prefix")
    ap.add_argument("--raw", action="store_true",
                    help="Print the raw skills_final_md string instead of parsed entries")
    args = ap.parse_args()

    path = Path(args.archive)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    scenarios = data.get("successful", [])
    print(f"Archive: {path}  ({len(scenarios)} successful scenarios)\n")

    shown = 0
    for scn in scenarios:
        if args.only_generated and scn.get("iteration", -1) < 0:
            continue
        if args.id and not scn.get("id", "").startswith(args.id):
            continue

        md = scn.get("skills_final_md") or ""
        if args.only_with_chronicle and not md.strip():
            continue

        chronicle = SkillsChronicle.from_markdown(md)

        print(SEP)
        title = scn.get("scenario_title") or "(no title)"
        print(f"  {title}")
        print(f"  id        : {scn.get('id', '')}")
        print(f"  iteration : {scn.get('iteration', '?')}")
        print(f"  source    : {scn.get('source', '?')}")
        print(f"  type      : {scn.get('interaction_type', '?')}")
        print(f"  scenario  : {scn.get('scenario', '')[:120]}")
        print(f"  chronicle : {len(chronicle.entries)} entries")

        if not md.strip():
            print("  (no skills chronicle)")
        elif args.raw:
            print("\n" + md)
        else:
            for i, entry in enumerate(chronicle.entries, 1):
                print(f"\n  [{i}] {entry.entry_id}  [{entry.entry_type} · {entry.dimension}]")
                print(f"      Condition : {entry.condition.strip()[:120]}")
                print(f"      Guidance  : {entry.guidance.strip()[:200]}")
                if entry.provenance:
                    print(f"      Provenance: {entry.provenance}")

        print()
        shown += 1

    print(SEP)
    print(f"Showed {shown} / {len(scenarios)} scenarios.")


if __name__ == "__main__":
    main()
