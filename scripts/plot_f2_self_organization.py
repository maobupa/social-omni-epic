#!/usr/bin/env python3
"""F2 — self-organization toward the frontier: the trend, and the mechanism that drives it.

Two panels answering two different questions about the gen90_expel run:

  (a) DOES it move toward the frontier?  Frontier-fraction of completed scenarios as the run
      proceeds (sliding window + Wilson 95% CI), against the raw-seed frontier rate (24%, the
      starting distribution). Trend reported as a point-biserial r with a seeded permutation p.
      Honest status: directional, marginal (two-sided p~0.05; one-sided ~0.026).

  (b) HOW does it move there?  The lineage self-correction mechanism. Each operator is conditioned
      on exactly one parent band (escalate<-too_easy, lateral<-frontier, relax<-beyond); the bars
      show where each operator's CHILDREN land. relax pulls beyond-frontier overshoots back to the
      frontier (57%), while escalate/lateral drift ~45% into beyond -- the cycle that produces (a).

Panel (a) alone is a borderline correlation a skeptic attributes to drift; panel (b) alone is a
conditional rate with no time axis. Together they are the Tier-1 "self-organizes toward the
frontier" claim.

Outputs (results/analysis/):
  f2_self_organization.pdf / .png
  f2_frontier_trend.csv      (sliding-window series)
  f2_operator_transition.csv (operator -> child-band composition)

Run:  python3 scripts/plot_f2_self_organization.py
"""
from __future__ import annotations
import csv
import glob
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(ROOT, "results/gen90_expel/metrics.json")
GEN_GLOB = os.path.join(ROOT, "results/gen90_expel/bank/generated/*.json")
BASE_GLOB = os.path.join(ROOT, "results/expel_phase0_Base90_ExpeL/seeds/*.json")
OUT_DIR = os.path.join(ROOT, "results/analysis")

BANDS = ps.BAND_ORDER
BAND_LABEL = ps.BAND_LABEL
BAND_COLOR = ps.BAND_COLOR
GREEN, GREY = ps.PALETTE["green"], ps.GREY
# each operator is bound to one parent band by construction
OP_ORDER = ["escalate", "lateral", "relax"]
OP_PARENT = {"escalate": "too_easy", "lateral": "frontier", "relax": "beyond_frontier"}
WINDOW = 21   # centred sliding window (odd) for panel (a)


# ----------------------------------------------------------------------------- data
def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def frontier_series():
    m = [r for r in json.load(open(METRICS)) if r.get("classification") in BANDS]
    m.sort(key=lambda r: r["iteration"])
    return [1 if r["classification"] == "frontier" else 0 for r in m]


def base_frontier_rate():
    recs = load(BASE_GLOB)
    return sum(r.get("classification") == "frontier" for r in recs) / len(recs)


def operator_child_composition():
    comp = defaultdict(Counter)
    for r in load(GEN_GLOB):
        comp[r.get("mutation_operator")][r.get("classification")] += 1
    return comp


# ------------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def sliding(series, w):
    half = w // 2
    xs, mean, lo, hi = [], [], [], []
    for i in range(len(series)):
        a, b = max(0, i - half), min(len(series), i + half + 1)
        win = series[a:b]
        p, l, h = wilson(sum(win), len(win))
        xs.append(i + 1); mean.append(p); lo.append(l); hi.append(h)
    return xs, mean, lo, hi


def point_biserial_trend(series, n_perm=20000, seed=0):
    y = list(series); x = list(range(len(y))); n = len(y)
    mx = sum(x) / n; my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)

    def corr(yy):
        myy = sum(yy) / n
        return sum((xi - mx) * (yi - myy) for xi, yi in zip(x, yy)) / \
            (sxx * sum((yi - myy) ** 2 for yi in yy)) ** 0.5

    r = corr(y); obs = abs(r); rng = random.Random(seed); hits = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if abs(corr(y)) >= obs - 1e-12:
            hits += 1
    return r, (hits + 1) / (n_perm + 1)


# ----------------------------------------------------------------------------- style
def panel_trend(ax, series, base_rate, r, p):
    xs, mean, lo, hi = sliding(series, WINDOW)
    ax.axhline(base_rate, ls=(0, (4, 3)), lw=1.0, color=GREY, zorder=2)
    ax.text(len(series), base_rate - 0.018, f"raw-seed frontier rate ({base_rate*100:.0f}%)",
            ha="right", va="top", fontsize=7.2, color=GREY)
    ax.fill_between(xs, lo, hi, color=GREEN, alpha=0.16, lw=0, zorder=2)
    ax.plot(xs, mean, color=GREEN, lw=1.9, zorder=3)
    # frontier-hit rug along the top
    for i, v in enumerate(series):
        if v:
            ax.plot([i + 1, i + 1], [1.005, 1.035], color=GREEN, lw=0.5, alpha=0.6,
                    clip_on=False, zorder=2)
    ax.set_xlim(1, len(series)); ax.set_ylim(0, 1.04)
    ax.set_xlabel("curriculum iteration"); ax.set_ylabel("frontier fraction")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("trend: does it move toward the frontier?", loc="left")
    ps.panel_label(ax, "a"); ps.despine(ax); ps.grid_y(ax)
    star = "*" if p < 0.05 else ""
    ax.annotate(rf"$r={r:+.2f}$, perm $p={p:.3f}${star}" + "\n" + rf"(1-sided $\approx${p/2:.3f})",
                xy=(0.035, 0.96), xycoords="axes fraction", ha="left", va="top", fontsize=7.6,
                color=ps.INK, linespacing=1.3)
    ax.annotate(f"sliding window = {WINDOW}", xy=(0.035, 0.04), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=6.8, color=GREY)


def panel_mechanism(ax, comp):
    ypos = {op: i for i, op in enumerate(reversed(OP_ORDER))}   # escalate at top
    bar_h = 0.6
    for op in OP_ORDER:
        n = sum(comp[op].values()); left = 0.0
        for band in BANDS:
            frac = comp[op].get(band, 0) / n
            if frac <= 0:
                continue
            ax.barh(ypos[op], frac, left=left, height=bar_h, color=BAND_COLOR[band],
                    edgecolor="white", linewidth=0.8, zorder=3)
            if frac >= 0.12:
                ax.text(left + frac / 2, ypos[op], f"{frac*100:.0f}%",
                        ha="center", va="center", color="white", fontsize=7.8,
                        fontweight="bold", zorder=4)
            left += frac
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(OP_ORDER) - 0.4)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"{op}\n(from {BAND_LABEL[OP_PARENT[op]]}, n={sum(comp[op].values())})"
                        for op in reversed(OP_ORDER)])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("child band (%)")
    ax.tick_params(axis="y", length=0); ps.despine(ax, left=False)
    ax.set_title("mechanism: the lineage self-correction", loc="left")
    ps.panel_label(ax, "b")
    handles = [Patch(facecolor=BAND_COLOR[b], edgecolor="white", label=BAND_LABEL[b]) for b in BANDS]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    series = frontier_series()
    base_rate = base_frontier_rate()
    comp = operator_child_composition()
    r, p = point_biserial_trend(series)

    print(f"frontier trend: r={r:+.3f}, perm p={p:.4f} (n={len(series)}); seed rate={base_rate:.3f}")
    print("operator -> child band:")
    for op in OP_ORDER:
        n = sum(comp[op].values())
        print(f"  {op:<9}(from {OP_PARENT[op]:<15} n={n}): " +
              ", ".join(f"{b}={comp[op].get(b,0)/n*100:.0f}%" for b in BANDS))

    with open(os.path.join(OUT_DIR, "f2_frontier_trend.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["iteration", "window_mean", "wilson_lo", "wilson_hi"])
        xs, mean, lo, hi = sliding(series, WINDOW)
        for row in zip(xs, mean, lo, hi):
            w.writerow([row[0]] + [f"{v:.4f}" for v in row[1:]])
    with open(os.path.join(OUT_DIR, "f2_operator_transition.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["operator", "parent_band", "child_band", "count", "fraction"])
        for op in OP_ORDER:
            n = sum(comp[op].values())
            for b in BANDS:
                w.writerow([op, OP_PARENT[op], b, comp[op].get(b, 0), f"{comp[op].get(b,0)/n:.4f}"])

    ps.apply()
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                   gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.34})
    panel_trend(axa, series, base_rate, r, p)
    panel_mechanism(axb, comp)
    ps.save(fig, "f2_self_organization", OUT_DIR)


if __name__ == "__main__":
    main()
