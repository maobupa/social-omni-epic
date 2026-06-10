#!/usr/bin/env python3
"""Contamination check between seed bank, generated archive, and eval set (§10.2).

Checks two overlap signals:
  1. Exact ID match (source_scenario_id or source_env_id).
  2. Semantic cosine similarity > --cosine-threshold on abstract embeddings.

Usage:
  python scripts/check_contamination.py \\
      --seeds data/sotopia_90_seeds.jsonl \\
      --archive results/run_001/archive_latest.json \\
      --eval data/eval_set.jsonl \\
      --cosine-threshold 0.85
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_scenarios(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"[skip] {path} — file not found", file=sys.stderr)
        return []
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        items = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    elif suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        # Support archive format ({"tasks": [...]} or {"successful": [...]})
        if isinstance(data, dict):
            return data.get("tasks", data.get("successful", []))
        return data
    else:
        print(f"[skip] {path} — unsupported format {suffix}", file=sys.stderr)
        return []


def _get_ids(scenario: dict) -> set[str]:
    ids = set()
    for key in ("id", "source_scenario_id", "source_env_id"):
        v = scenario.get(key)
        if v:
            ids.add(str(v))
    return ids


def _get_embedding(scenario: dict) -> list[float] | None:
    emb = scenario.get("embedding")
    if emb and isinstance(emb, list):
        return emb
    return None


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return cosine similarity matrix of shape (len(a), len(b))."""
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    a_unit = a / np.where(a_norms == 0, 1.0, a_norms)
    b_unit = b / np.where(b_norms == 0, 1.0, b_norms)
    return a_unit @ b_unit.T


def check_contamination(
    seeds: list[dict],
    archive: list[dict],
    eval_set: list[dict],
    cosine_threshold: float = 0.85,
) -> dict:
    results = {
        "exact_id_overlaps": [],
        "semantic_overlaps": [],
        "summary": {},
    }

    # Build ID sets
    seed_ids: set[str] = set()
    for s in seeds:
        seed_ids.update(_get_ids(s))
    archive_ids: set[str] = set()
    for s in archive:
        archive_ids.update(_get_ids(s))

    eval_id_to_scenario: dict[str, dict] = {}
    for s in eval_set:
        for id_ in _get_ids(s):
            eval_id_to_scenario[id_] = s

    # Exact ID overlaps
    seed_overlap = seed_ids & set(eval_id_to_scenario.keys())
    archive_overlap = archive_ids & set(eval_id_to_scenario.keys())
    for id_ in sorted(seed_overlap):
        results["exact_id_overlaps"].append({"id": id_, "source": "seeds"})
    for id_ in sorted(archive_overlap):
        results["exact_id_overlaps"].append({"id": id_, "source": "archive"})

    # Semantic cosine overlaps
    eval_embeddings = [(i, _get_embedding(s)) for i, s in enumerate(eval_set)]
    eval_embs_valid = [(i, e) for i, e in eval_embeddings if e is not None]

    if eval_embs_valid:
        eval_arr = np.array([e for _, e in eval_embs_valid], dtype=float)

        for label, corpus in [("seeds", seeds), ("archive", archive)]:
            corpus_embs = [(i, _get_embedding(s)) for i, s in enumerate(corpus)]
            corpus_valid = [(i, e) for i, e in corpus_embs if e is not None]
            if not corpus_valid:
                continue
            corpus_arr = np.array([e for _, e in corpus_valid], dtype=float)
            sim = _cosine_matrix(corpus_arr, eval_arr)
            # Find all (corpus_idx, eval_idx) pairs above threshold
            above = np.argwhere(sim > cosine_threshold)
            for corpus_pos, eval_pos in above:
                corpus_i = corpus_valid[corpus_pos][0]
                eval_i = eval_embs_valid[eval_pos][0]
                results["semantic_overlaps"].append({
                    "source": label,
                    "source_idx": corpus_i,
                    "eval_idx": eval_i,
                    "cosine": float(sim[corpus_pos, eval_pos]),
                    "source_title": corpus[corpus_i].get("scenario_title", "")[:60],
                    "eval_title": eval_set[eval_i].get("scenario_title", "")[:60],
                })

    results["summary"] = {
        "n_seeds": len(seeds),
        "n_archive": len(archive),
        "n_eval": len(eval_set),
        "n_exact_overlaps": len(results["exact_id_overlaps"]),
        "n_semantic_overlaps": len(results["semantic_overlaps"]),
        "cosine_threshold": cosine_threshold,
        "contaminated": bool(results["exact_id_overlaps"] or results["semantic_overlaps"]),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Check dataset contamination")
    parser.add_argument("--seeds", default="data/sotopia_90_seeds.jsonl")
    parser.add_argument("--archive", default=None, help="Archive JSON (from checkpoint)")
    parser.add_argument("--eval", required=True, help="Eval set JSONL")
    parser.add_argument("--cosine-threshold", type=float, default=0.85)
    parser.add_argument("--output", default=None, help="Write results JSON here")
    args = parser.parse_args()

    seeds = _load_scenarios(args.seeds)
    archive = _load_scenarios(args.archive) if args.archive else []
    eval_set = _load_scenarios(args.eval)

    print(f"Loaded: {len(seeds)} seeds, {len(archive)} archive, {len(eval_set)} eval")

    results = check_contamination(seeds, archive, eval_set, args.cosine_threshold)
    summary = results["summary"]

    print(f"\n=== Contamination Report ===")
    print(f"Exact ID overlaps:    {summary['n_exact_overlaps']}")
    print(f"Semantic overlaps:    {summary['n_semantic_overlaps']}  (cosine > {args.cosine_threshold})")
    print(f"CONTAMINATED:         {summary['contaminated']}")

    if results["exact_id_overlaps"]:
        print("\nExact overlaps:")
        for o in results["exact_id_overlaps"]:
            print(f"  [{o['source']}] {o['id']}")

    if results["semantic_overlaps"]:
        print(f"\nTop semantic overlaps (cosine > {args.cosine_threshold}):")
        top = sorted(results["semantic_overlaps"], key=lambda x: -x["cosine"])[:10]
        for o in top:
            print(f"  [{o['source']}] {o['cosine']:.3f}  {o['source_title']!r} ↔ {o['eval_title']!r}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull report written to {args.output}")

    sys.exit(1 if summary["contaminated"] else 0)


if __name__ == "__main__":
    main()
