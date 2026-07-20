#!/usr/bin/env python3
"""Scenario-space expansion: the curriculum reaches social-dynamic regions the seeds never cover.

Two UMAP views of the SAME 180 scenarios (90 SOTOPIA seeds + 90 generated), differing only in
WHAT is embedded:

  (a) surface space    — the archive's stored embedding (full scenario text + goals; title-free).
  (b) structural space — a fresh embedding of the abstract SCENARIO_TITLE
                         ("social_dynamic | target_perspective"), so surface detail is stripped
                         and only the social-dynamic structure remains (scripts/embed_titles.py).

In both, seeds are grey rings and generated scenarios are coloured by generation iteration, so the
panels show expansion *beyond the seed manifold over time*. The structural view tests the
hypothesis that the curriculum diverges in social *structure*, not just surface wording.

Quantitative backbone (printed + in caption): for each generated scenario, cosine distance to its
nearest seed; the generated set sits further from the seed manifold in structural space than a
seed does from its own neighbours (permutation test).

Also emits a band-coloured structural map (the QD "map, not trophy case" coverage figure).

Outputs (results/analysis/):
  scenario_space.{pdf,png}            surface vs structural, by iteration
  scenario_space_bands.{pdf,png}      structural space, coloured by band
  scenario_space_divergence.csv       nearest-seed distances

Run:  python3 scripts/embed_titles.py && python3 scripts/plot_scenario_space.py
"""
from __future__ import annotations
import csv
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "results/gen90_expel/archive_latest.json")
TITLE_EMB = os.path.join(ROOT, "results/analysis/title_embeddings.json")
OUT_DIR = os.path.join(ROOT, "results/analysis")
SEED_SOURCES = {"seed_sotopia", "fallback_seed"}
SEED = 42


def load():
    tasks = json.load(open(ARCHIVE))["tasks"]
    by_id = {t["id"]: t for t in tasks}
    tcache = json.load(open(TITLE_EMB))
    # align structural embeddings to tasks (cache stores ids in archive order)
    title_vec = {i: v for i, v in zip(tcache["ids"], tcache["embeddings"])}

    rows = []
    for t in tasks:
        if not t.get("embedding") or t["id"] not in title_vec:
            continue
        is_seed = t.get("source") in SEED_SOURCES
        rows.append({
            "id": t["id"], "is_seed": is_seed,
            "iteration": 0 if is_seed else int(t.get("iteration") or 0),
            "band": t.get("classification"),
            "surface": np.asarray(t["embedding"], dtype=np.float32),
            "structural": np.asarray(title_vec[t["id"]], dtype=np.float32),
        })
    return rows


def umap2d(X):
    # Match the original recipe (output/200_full): UMAP defaults — euclidean, n_neighbors=15,
    # min_dist=0.1 — which spreads the clusters far apart instead of collapsing them.
    from umap import UMAP
    return UMAP(n_components=2, random_state=SEED).fit_transform(X)


def cosine_dist_matrix(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return 1.0 - An @ Bn.T


def nearest_seed_stats(rows, key):
    seeds = np.stack([r[key] for r in rows if r["is_seed"]])
    gen = np.stack([r[key] for r in rows if not r["is_seed"]])
    # generated -> nearest seed
    g2s = cosine_dist_matrix(gen, seeds).min(axis=1)
    # seed -> nearest *other* seed (within-manifold baseline)
    ss = cosine_dist_matrix(seeds, seeds)
    np.fill_diagonal(ss, np.inf)
    s2s = ss.min(axis=1)
    return g2s, s2s


def perm_test(a, b, n_perm=20000, seed=0):
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b]); na = len(a)
    rng = random.Random(seed); idx = list(range(len(pool))); hits = 0
    for _ in range(n_perm):
        rng.shuffle(idx)
        p = pool[idx]
        if abs(p[:na].mean() - p[na:].mean()) >= obs - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


# --------------------------------------------------------------------------------- panels
def scatter_by_iteration(ax, xy, rows, title, letter, cbar=False, fig=None):
    is_seed = np.array([r["is_seed"] for r in rows])
    it = np.array([r["iteration"] for r in rows])
    ax.scatter(xy[is_seed, 0], xy[is_seed, 1], marker="x", c=ps.GREY,
               s=42, linewidths=1.2, alpha=0.85, label="seeds (90)", zorder=2)
    sc = ax.scatter(xy[~is_seed, 0], xy[~is_seed, 1], c=it[~is_seed], cmap="viridis",
                    s=34, alpha=0.92, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_title(title, loc="left")
    ps.panel_label(ax, letter)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.tick_params(labelsize=7)
    ax.margins(0.06)
    if cbar and fig is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("generation iteration", fontsize=8)
        cb.outline.set_linewidth(0.6); cb.ax.tick_params(width=0.6, labelsize=7)
    return sc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(TITLE_EMB):
        raise SystemExit("run scripts/embed_titles.py first")
    rows = load()
    print(f"{len(rows)} scenarios ({sum(r['is_seed'] for r in rows)} seeds, "
          f"{sum(not r['is_seed'] for r in rows)} generated)")

    Xs = np.stack([r["surface"] for r in rows])
    Xt = np.stack([r["structural"] for r in rows])
    xy_s, xy_t = umap2d(Xs), umap2d(Xt)

    # divergence backbone
    print("\nnearest-seed cosine distance (generated->seed vs seed->seed):")
    div = {}
    for key, name in [("surface", "surface"), ("structural", "structural")]:
        g2s, s2s = nearest_seed_stats(rows, key)
        d, p = perm_test(g2s, s2s)
        div[name] = (g2s, s2s, d, p)
        print(f"  {name:<11}: gen->seed median={np.median(g2s):.3f}  "
              f"seed->seed median={np.median(s2s):.3f}  Δmean={d:.3f}  perm p={p:.4f}")
    with open(os.path.join(OUT_DIR, "scenario_space_divergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["space", "kind", "nearest_seed_cosine_dist"])
        for name in ("surface", "structural"):
            g2s, s2s, _, _ = div[name]
            for v in g2s:
                w.writerow([name, "generated_to_seed", f"{v:.4f}"])
            for v in s2s:
                w.writerow([name, "seed_to_seed", f"{v:.4f}"])

    # ---- figure 1: surface vs structural, by iteration ----
    ps.apply()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.0, 4.3),
                                   gridspec_kw={"wspace": 0.16})
    scatter_by_iteration(axa, xy_s, rows, "surface space  (full scenario)", "a")
    axa.legend(loc="best")
    scatter_by_iteration(axb, xy_t, rows, "structural space  (social dynamic)", "b",
                         cbar=True, fig=fig)
    _, _, ds, ps_ = div["structural"]
    fig.text(0.5, -0.04,
             rf"generated scenarios sit a median cosine {np.median(div['structural'][0]):.2f} from "
             rf"the nearest seed in structural space (vs {np.median(div['structural'][1]):.2f} "
             rf"seed-to-seed; perm $p={ps_:.3f}$)",
             ha="center", va="top", fontsize=7.2, color=ps.GREY)
    ps.save(fig, "scenario_space", OUT_DIR)

    # ---- figure 2: structural space, coloured by band (coverage map) ----
    fig2, ax2 = plt.subplots(figsize=(5.6, 4.6))
    is_seed = np.array([r["is_seed"] for r in rows])
    ax2.scatter(xy_t[is_seed, 0], xy_t[is_seed, 1], marker="x", c=ps.GREY,
                s=42, linewidths=1.2, alpha=0.8, zorder=2)
    for band in ps.BAND_ORDER:
        m = np.array([(not r["is_seed"]) and r["band"] == band for r in rows])
        if m.any():
            ax2.scatter(xy_t[m, 0], xy_t[m, 1], c=ps.BAND_COLOR[band], s=34, alpha=0.9,
                        edgecolors="white", linewidths=0.4, zorder=3, label=ps.BAND_LABEL[band])
    ax2.set_title("structural coverage by band", loc="left")
    ax2.set_xlabel("UMAP-1"); ax2.set_ylabel("UMAP-2")
    ax2.tick_params(labelsize=7); ax2.margins(0.06)
    handles = [Line2D([0], [0], marker="x", color=ps.GREY, markeredgecolor=ps.GREY,
                      label="seeds", markersize=6, linestyle="none")] + \
              [Line2D([0], [0], marker="o", color="w", markerfacecolor=ps.BAND_COLOR[b],
                      label=ps.BAND_LABEL[b], markersize=6) for b in ps.BAND_ORDER]
    ax2.legend(handles=handles, loc="best")
    ps.save(fig2, "scenario_space_bands", OUT_DIR)


if __name__ == "__main__":
    main()
