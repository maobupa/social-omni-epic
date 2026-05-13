"""Print each generated scenario alongside the archive entries it was built from.

For runs after the provenance fix: uses `parent_example_ids` recorded at
generation time (exact prompt examples).
For older runs (smoke output before the fix): falls back to k-nearest-neighbor
lookup in embedding space.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

# allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.distance import cosine


def wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    if not text:
        return f"{indent}(empty)"
    return textwrap.fill(text.strip(), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def render(s: dict, label: str = "", show_embedding: bool = False) -> str:
    out = []
    head = f"[{label}] " if label else ""
    head += f"iter={s.get('iteration', '?')}  source={s.get('source', '?')}  "
    head += f"tag={s.get('tag', '')}  type={s.get('interaction_type', '')}"
    out.append(head)
    out.append("  scenario:")
    out.append(wrap(s.get("scenario", ""), indent="    "))
    out.append("  relationship: " + (s.get("relationship") or "")[:140])
    profiles = s.get("agent_profiles", [])
    for i, p in enumerate(profiles):
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        line = f"  agent {i+1}: {name} | {p.get('occupation', '')}"
        if p.get("big_five"):
            line += f" | {p['big_five'][:60]}"
        out.append(line)
        goals = s.get("agent_goals") or []
        if i < len(goals):
            out.append(wrap(f"goal: {goals[i]}", indent="      "))
    if s.get("moi_reasoning"):
        out.append("  MoI said: " + wrap(s["moi_reasoning"], indent="").strip())
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--k", type=int, default=2,
                    help="If parent_example_ids missing, show this many nearest neighbors instead.")
    ap.add_argument("--iter", type=int, default=None,
                    help="Show only this iteration (default: all generated).")
    args = ap.parse_args()

    archive = json.load(open(args.archive))
    successful = archive["successful"]
    by_id = {s["id"]: s for s in successful}
    embeddings = {s["id"]: np.array(s["embedding"]) for s in successful if s.get("embedding")}

    generated = [s for s in successful if s.get("iteration", -1) >= 0]
    if args.iter is not None:
        generated = [s for s in generated if s["iteration"] == args.iter]

    if not generated:
        print("No generated scenarios in this archive.")
        return

    for g in generated:
        print("=" * 100)
        print(render(g, label="GENERATED"))

        # Prefer recorded provenance
        parents = [by_id[pid] for pid in (g.get("parent_example_ids") or []) if pid in by_id]
        if parents:
            print(f"\n  Prompt examples used ({len(parents)}):")
            for p in parents:
                print("-" * 80)
                print(render(p, label="PROMPT_EXAMPLE"))
        elif g["id"] in embeddings:
            # Fallback: k nearest non-self entries
            q = embeddings[g["id"]]
            scored = [(cosine(q, e), sid) for sid, e in embeddings.items()
                      if sid != g["id"] and by_id[sid].get("iteration", -1) < g["iteration"]]
            scored.sort()
            print(f"\n  No parent ids recorded; showing {args.k} nearest neighbors instead:")
            for dist, sid in scored[:args.k]:
                print("-" * 80)
                print(f"  cosine distance: {dist:.4f}")
                print(render(by_id[sid], label="NEAREST"))
        print()


if __name__ == "__main__":
    main()
