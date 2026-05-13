"""Summary metrics over a finished run."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def summarize(archive_path: str):
    with open(archive_path) as f:
        archive = json.load(f)

    scenarios = archive["successful"]
    sources = Counter(s.get("source", "generated") for s in scenarios)
    types = Counter((s.get("interaction_type") or s.get("tag") or "unknown")
                    for s in scenarios)
    print(f"Total successful scenarios: {len(scenarios)}")
    print(f"Failed generations:         {len(archive.get('failed_generation', []))}")
    print(f"Failed interestingness:     {len(archive.get('failed_interestingness', []))}")
    print(f"\nBy source:")
    for k, v in sources.most_common():
        print(f"  {k}: {v}")
    print(f"\nBy interaction type:")
    for k, v in types.most_common():
        print(f"  {k}: {v}")

    embs = [s["embedding"] for s in scenarios if s.get("embedding")]
    if len(embs) > 2:
        from social_omni_epic.embedding_utils import compute_cell_coverage
        cov = compute_cell_coverage(np.array(embs))
        print(f"\nCell coverage (20x20 grid, PCA-2D): {cov:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    args = ap.parse_args()
    summarize(args.archive)
