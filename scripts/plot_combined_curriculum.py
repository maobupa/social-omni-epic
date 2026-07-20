#!/usr/bin/env python3
"""Combined Tier-1 figure: what the LP curriculum produces, and that it self-organizes there.

Two panels, one story about the gen90_expel run, both measured through the identical pipeline:

  (a) The LP-calibrated curriculum reshapes the difficulty distribution. Band composition of the
      GENERATED scenario archive vs the raw SOTOPIA seeds (Base-90), measured through the same K=4
      episode loop / key-blind LP judge / learner+judge model -- so the contrast isolates the
      scenario distribution itself. chi^2 on the 3x2 table (df=2 -> exact p = exp(-chi2/2)).

  (b) It self-organizes toward the frontier over time. Frontier fraction of completed scenarios as
      the run proceeds (centred sliding window + Wilson 95% CI), against the raw-seed frontier rate
      (the starting distribution). Trend = point-biserial r with a seeded permutation p.

This merges the former F1 (panel a) and F2-panel-a (panel b) into a single publication figure with
a unified, top-tier-venue visual language. "bank" is rendered as "scenario archive" throughout.

Outputs (results/analysis/): fig_curriculum.{pdf,png}
Run:  python3 scripts/plot_combined_curriculum.py
"""
from __future__ import annotations
import glob
import json
import math
import os
import random
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_GLOB = os.path.join(ROOT, "results/expel_phase0_Base90_ExpeL/seeds/*.json")
GEN_GLOB = os.path.join(ROOT, "results/gen90_expel/bank/generated/*.json")
METRICS = os.path.join(ROOT, "results/gen90_expel/metrics.json")
OUT_DIR = os.path.join(ROOT, "results/analysis")

BANDS = ps.BAND_ORDER
BAND_COLOR = ps.BAND_COLOR
BAND_LABEL = ps.BAND_LABEL
GREEN, GREY, INK = ps.PALETTE["green"], ps.GREY, ps.INK
WINDOW = 21


# ----------------------------------------------------------------------------- data
def load(pattern):
    recs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    if not recs:
        raise SystemExit(f"no records: {pattern}")
    return recs


def comp(recs):
    return Counter(r.get("classification") for r in recs)


def frontier_series():
    m = [r for r in json.load(open(METRICS)) if r.get("classification") in BANDS]
    m.sort(key=lambda r: r["iteration"])
    return [1 if r["classification"] == "frontier" else 0 for r in m]


# ------------------------------------------------------------------------ statistics
def chi2_3x2(a, b):
    obs = [[a.get(k, 0) for k in BANDS], [b.get(k, 0) for k in BANDS]]
    rt = [sum(r) for r in obs]
    ct = [obs[0][j] + obs[1][j] for j in range(3)]
    n = sum(rt)
    chi2 = sum((obs[i][j] - rt[i] * ct[j] / n) ** 2 / (rt[i] * ct[j] / n)
               for i in range(2) for j in range(3))
    return chi2, math.exp(-chi2 / 2.0)


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
    mx = sum(x) / n
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


# ----------------------------------------------------------------------------- panels
def panel_letter(ax, letter, x_frac=0.0):
    """Bold panel letter pinned to the top-left of the axes, clear of the title."""
    ax.annotate(f"({letter})", xy=(x_frac, 1.0), xytext=(0, 17), xycoords="axes fraction",
                textcoords="offset points", ha="left", va="bottom",
                fontsize=10, fontweight="bold", color=INK, annotation_clip=False)


# Row geometry for panel (a): Generated-90 sits above Base-90.
ROWS = [
    ("Generated-90", "Generated-90", "LP curriculum"),
    ("Base90", "Base-90", "raw SOTOPIA seeds"),
]


def panel_composition(ax, composition, chi2):
    ypos = {"Generated-90": 1.0, "Base90": 0.0}
    bar_h = 0.52
    for key, _, _ in ROWS:
        c = composition[key]
        n = sum(c.values())
        left = 0.0
        for band in BANDS:
            frac = c.get(band, 0) / n
            if frac <= 0:
                continue
            ax.barh(ypos[key], frac, left=left, height=bar_h, color=BAND_COLOR[band],
                    edgecolor="white", linewidth=1.4, zorder=3)
            if frac >= 0.11:
                ax.text(left + frac / 2, ypos[key], f"{frac*100:.0f}%\n$n{{=}}{c.get(band,0)}$",
                        ha="center", va="center", color="white", fontsize=8.0,
                        fontweight="bold", linespacing=1.08, zorder=4)
            elif frac >= 0.03:
                ax.annotate(f"{frac*100:.0f}%", xy=(left + frac / 2, ypos[key] + bar_h / 2),
                            xytext=(left + frac / 2, ypos[key] + bar_h / 2 + 0.20),
                            ha="center", va="bottom", color=BAND_COLOR[band], fontsize=7.2,
                            fontweight="bold", zorder=4,
                            arrowprops=dict(arrowstyle="-", color=BAND_COLOR[band], lw=0.7,
                                            shrinkA=0, shrinkB=1))
            left += frac

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.62, 1.62)
    ax.set_yticks([1, 0])
    ax.set_yticklabels([])
    # two-line condition labels, with the descriptor in muted grey
    for key, title, sub in ROWS:
        ax.text(-0.025, ypos[key] + 0.085, title, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.6, fontweight="bold", color=INK)
        ax.text(-0.025, ypos[key] - 0.115, sub, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=7.4, color=GREY)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("share of scenario archive (%)")
    ax.tick_params(axis="y", length=0)
    ps.despine(ax, left=False)
    ax.spines["left"].set_visible(False)
    ax.set_title("the curriculum reshapes difficulty", loc="left", pad=18)
    panel_letter(ax, "a", x_frac=-0.20)
    ax.annotate(rf"$\chi^2(2)={chi2:.0f},\ p\!<\!10^{{-9}}$", xy=(1.0, 1.055),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=7.8, color=GREY)


def panel_trend(ax, series, base_rate, r, p):
    xs, mean, lo, hi = sliding(series, WINDOW)
    N = len(series)

    # raw-seed reference
    ax.axhline(base_rate, ls=(0, (4, 3)), lw=1.0, color=GREY, zorder=2)
    ax.text(N, base_rate - 0.022, f"raw-seed frontier rate ({base_rate*100:.0f}%)",
            ha="right", va="top", fontsize=7.2, color=GREY)

    # frontier-hit rug along the top
    for i, v in enumerate(series):
        if v:
            ax.plot([i + 1, i + 1], [1.005, 1.032], color=GREEN, lw=0.6, alpha=0.55,
                    clip_on=False, zorder=2)

    ax.fill_between(xs, lo, hi, color=GREEN, alpha=0.15, lw=0, zorder=2)
    ax.plot(xs, mean, color=GREEN, lw=2.1, zorder=4, solid_capstyle="round")
    # endpoint emphasis
    ax.scatter([xs[-1]], [mean[-1]], s=20, color=GREEN, zorder=5, edgecolor="white", linewidth=0.8)
    ax.annotate(f"{mean[-1]*100:.0f}%", xy=(xs[-1], mean[-1]), xytext=(-2, 6),
                textcoords="offset points", ha="right", va="bottom", fontsize=7.4,
                fontweight="bold", color=GREEN)

    ax.set_xlim(1, N)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("curriculum iteration")
    ax.set_ylabel("frontier fraction")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_title("self-organization toward the frontier", loc="left", pad=18)
    panel_letter(ax, "b", x_frac=-0.13)
    ps.despine(ax)
    ps.grid_y(ax)
    star = "*" if p < 0.05 else ""
    ax.annotate(rf"$r={r:+.2f}$, perm $p={p:.3f}${star}", xy=(0.035, 0.965),
                xycoords="axes fraction", ha="left", va="top", fontsize=7.8, color=INK)
    ax.annotate(rf"(1-sided $\approx${p/2:.3f}),  window $={WINDOW}$", xy=(0.035, 0.905),
                xycoords="axes fraction", ha="left", va="top", fontsize=6.9, color=GREY)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base, gen = load(BASE_GLOB), load(GEN_GLOB)
    composition = {"Base90": comp(base), "Generated-90": comp(gen)}
    chi2, p_comp = chi2_3x2(composition["Base90"], composition["Generated-90"])
    series = frontier_series()
    base_rate = sum(r.get("classification") == "frontier" for r in base) / len(base)
    r, p = point_biserial_trend(series)
    print(f"chi2={chi2:.1f} p={p_comp:.2e}  |  trend r={r:+.3f} perm p={p:.4f} "
          f"(n={len(series)}); seed rate={base_rate:.3f}")

    ps.apply()
    fig = plt.figure(figsize=(7.2, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.02], wspace=0.30,
                          left=0.085, right=0.985, top=0.84, bottom=0.30)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    panel_composition(axa, composition, chi2)
    panel_trend(axb, series, base_rate, r, p)

    # single shared band legend, centred beneath both panels
    handles = [Patch(facecolor=BAND_COLOR[b], edgecolor="white", label=BAND_LABEL[b]) for b in BANDS]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               frameon=False, handlelength=1.1, columnspacing=1.6)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"fig_curriculum.{ext}")
        fig.savefig(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
