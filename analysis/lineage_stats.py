"""Numerical structural metrics on one or more lineage graphs.

You asked the right question: counting chains and fanout by eye on a 290-node
graph is hopeless. This script computes those things explicitly so you can
compare them across runs as numbers, not visual impressions.

Metrics produced per archive:
  -- Graph topology --
  num_nodes / num_edges          basic counts
  num_seeds / num_generated      breakdown
  num_failed_moi                 MoI-rejected scenarios (red triangles)
  weakly_connected_components   how many disconnected lineage trees
  num_orphans                   generated scenarios with NO parents (an LLM that
                                ignored the archive examples? a fallback hit?)
  branching_factor              mean out-degree of nodes that have any children
  leaf_count                    nodes with no children
  backbone_count                nodes with at least one child (intermediate
                                nodes in some lineage)
  max_chain_depth               longest path from any seed to any generated
                                scenario, in graph-edge steps. "Did the LLM
                                build 5 generations of revisions, or just
                                one-hop riffs on a seed?"
  median_chain_depth            same, but median across all generated nodes
  top_seed_fanout               max number of generated scenarios traceable
                                back (transitively) to any single seed
  fanout_gini                   inequality of seed reuse (0 = uniform, 1 = one
                                seed dominates)

  -- Spatial drift (uses embeddings, not just topology) --
  mean_dist_to_nearest_seed      avg cosine distance from each generated scenario
                                 to its closest seed. Higher = more exploration.
  mean_dist_to_parent            avg cosine distance to its closest parent.
                                 Higher = bigger conceptual jumps per generation.

Usage:
  python analysis/lineage_stats.py \
      --archive full=output/200_full/archive_latest.json \
      --archive no_moi=output/200_no_moi/archive_latest.json \
      --archive no_archive=output/200_no_archive/archive_latest.json \
      --out_csv output/comparison/lineage_stats.csv
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine


def parse_arg(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise ValueError(f"Expected name=path, got: {s}")
    name, p = s.split("=", 1)
    return name.strip(), Path(p.strip())


def load(path: Path) -> dict:
    return json.load(open(path))


def build_lineage_graph(archive: dict, inferred_k: int = 3) -> tuple[nx.DiGraph, dict]:
    """DiGraph with parent->child edges. Falls back to k-NN if provenance missing."""
    successful = archive.get("successful", [])
    failed_int = archive.get("failed_interestingness", [])
    all_nodes = successful + failed_int
    by_id = {s["id"]: s for s in all_nodes}

    ids_with_emb = [s["id"] for s in all_nodes if s.get("embedding")]
    id_to_emb = {sid: np.array(by_id[sid]["embedding"]) for sid in ids_with_emb}

    G = nx.DiGraph()
    for s in all_nodes:
        role = ("seed" if s.get("iteration", -1) < 0 else
                "failed_int" if s in failed_int else "generated")
        G.add_node(s["id"], role=role, iteration=s.get("iteration", -1),
                   embedding=s.get("embedding"))

    has_any_recorded_parents = any(s.get("parent_example_ids") for s in successful)
    for s in all_nodes:
        parent_ids = s.get("parent_example_ids") or []
        if (not parent_ids and not has_any_recorded_parents
                and s.get("iteration", -1) >= 0 and s["id"] in id_to_emb):
            # No provenance ever recorded → infer via k-NN to earlier iters
            own_emb = id_to_emb[s["id"]]
            own_iter = s.get("iteration", -1)
            candidates = [(sid, e) for sid, e in id_to_emb.items()
                          if sid != s["id"] and by_id[sid].get("iteration", -1) < own_iter]
            if candidates:
                scored = sorted(candidates,
                                key=lambda x: cosine(own_emb, x[1]))[:inferred_k]
                parent_ids = [sid for sid, _ in scored]
        for p in parent_ids:
            if p in G.nodes:
                G.add_edge(p, s["id"])
    return G, by_id


def transitive_seeds(G: nx.DiGraph) -> dict[str, set[str]]:
    """For each node, the set of seed ancestors reachable backwards."""
    out: dict[str, set[str]] = {}
    seed_ids = {n for n, d in G.nodes(data=True) if d.get("role") == "seed"}
    # iterate in topological order so we can build incrementally
    for n in nx.topological_sort(G):
        if n in seed_ids:
            out[n] = {n}
        else:
            anc: set[str] = set()
            for p in G.predecessors(n):
                anc |= out.get(p, set())
            out[n] = anc
    return out


def gini(values: list[float]) -> float:
    """Standard Gini coefficient. 0 = perfectly uniform, 1 = max inequality."""
    if not values or all(v == 0 for v in values):
        return 0.0
    v = sorted(values)
    n = len(v)
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2 * cum) / (n * sum(v)) - (n + 1) / n


def compute_metrics(archive: dict, G: nx.DiGraph, by_id: dict) -> dict[str, float]:
    seeds = [n for n, d in G.nodes(data=True) if d.get("role") == "seed"]
    generated = [n for n, d in G.nodes(data=True) if d.get("role") == "generated"]
    failed_int = [n for n, d in G.nodes(data=True) if d.get("role") == "failed_int"]

    # Treat as undirected for connectivity
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))

    # Orphans among generated
    orphans = [n for n in generated if G.in_degree(n) == 0]

    # Branching factor: avg out-degree among nodes with children
    out_degrees = [G.out_degree(n) for n in G.nodes() if G.out_degree(n) > 0]
    branching = mean(out_degrees) if out_degrees else 0.0

    # Backbone vs leaves
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    backbone = [n for n in G.nodes() if G.out_degree(n) > 0]

    # Chain depth: shortest path from any seed to each generated node
    # (graph-edge steps); ignore unreachable generations
    chain_depths: list[int] = []
    if seeds and generated:
        # Multi-source BFS
        dist = {s: 0 for s in seeds}
        frontier = list(seeds)
        while frontier:
            new_frontier = []
            for u in frontier:
                for v in G.successors(u):
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        new_frontier.append(v)
            frontier = new_frontier
        chain_depths = [dist[g] for g in generated if g in dist]

    # Seed fanout (transitive)
    seed_to_count: dict[str, int] = defaultdict(int)
    if generated:
        anc = transitive_seeds(G)
        for g in generated:
            for s in anc.get(g, set()):
                seed_to_count[s] += 1
    fanouts = list(seed_to_count.values())

    # Spatial metrics
    seed_embs = np.array([by_id[s]["embedding"] for s in seeds if by_id[s].get("embedding")])
    nearest_seed_dists: list[float] = []
    parent_dists: list[float] = []
    for g in generated:
        ge = by_id[g].get("embedding")
        if ge is None:
            continue
        ge_arr = np.array(ge)
        if seed_embs.size:
            # cosine distance to each seed, take min
            sims = seed_embs @ ge_arr / (
                np.linalg.norm(seed_embs, axis=1) * np.linalg.norm(ge_arr) + 1e-9)
            nearest_seed_dists.append(float(1 - sims.max()))
        # Closest parent distance
        pdists = []
        for p in G.predecessors(g):
            pe = by_id[p].get("embedding")
            if pe is not None:
                pdists.append(cosine(ge_arr, np.array(pe)))
        if pdists:
            parent_dists.append(min(pdists))

    return {
        "num_nodes":                    G.number_of_nodes(),
        "num_edges":                    G.number_of_edges(),
        "num_seeds":                    len(seeds),
        "num_generated":                len(generated),
        "num_failed_moi":               len(failed_int),
        "weakly_connected_components":  len(components),
        "num_orphans":                  len(orphans),
        "branching_factor":             round(branching, 3),
        "leaf_count":                   len(leaves),
        "backbone_count":               len(backbone),
        "max_chain_depth":              max(chain_depths) if chain_depths else 0,
        "median_chain_depth":           median(chain_depths) if chain_depths else 0,
        "top_seed_fanout":              max(fanouts) if fanouts else 0,
        "fanout_gini":                  round(gini(fanouts), 3) if fanouts else 0.0,
        "mean_dist_to_nearest_seed":    round(mean(nearest_seed_dists), 4) if nearest_seed_dists else 0.0,
        "mean_dist_to_parent":          round(mean(parent_dists), 4) if parent_dists else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", action="append", required=True,
                    help="name=path; repeatable")
    ap.add_argument("--out_csv", default=None,
                    help="Write the comparison table to this CSV (optional)")
    ap.add_argument("--inferred_k", type=int, default=3,
                    help="k-NN fallback parents per node when parent_example_ids missing")
    args = ap.parse_args()

    parsed = [parse_arg(a) for a in args.archive]
    rows: dict[str, dict] = {}
    for name, path in parsed:
        archive = load(path)
        G, by_id = build_lineage_graph(archive, inferred_k=args.inferred_k)
        rows[name] = compute_metrics(archive, G, by_id)
        # Note inferred-vs-recorded provenance
        any_recorded = any(s.get("parent_example_ids")
                           for s in archive.get("successful", []))
        rows[name]["provenance"] = "recorded" if any_recorded else "k-NN inferred"

    # Print as aligned table
    metrics = list(next(iter(rows.values())).keys())
    methods = list(rows.keys())
    width = max(len(m) for m in metrics) + 2
    col = max(max(len(str(rows[m][k])) for k in metrics) for m in methods) + 2
    col = max(col, max(len(m) for m in methods) + 2)
    header = " " * width + "".join(f"{m:>{col}}" for m in methods)
    print(header)
    print("-" * len(header))
    for k in metrics:
        line = f"{k:<{width}}"
        for m in methods:
            line += f"{str(rows[m][k]):>{col}}"
        print(line)

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric"] + methods)
            for k in metrics:
                w.writerow([k] + [rows[m][k] for m in methods])
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
