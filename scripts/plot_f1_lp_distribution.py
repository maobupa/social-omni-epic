#!/usr/bin/env python3
"""F1 — the thesis figure: the LP-calibrated curriculum drains the trivial band into the frontier.

Band composition of the GENERATED bank (gen90_expel) vs the raw SOTOPIA seeds
(expel_phase0_Base90_ExpeL), both measured through the *identical* pipeline (same K=4 episode
loop, same key-blind LP judge, same learner + judge model) so the contrast isolates the scenario
distribution itself. One claim, one panel; the beyond-frontier overshoot is shown, not hidden.
chi^2 on the 3x2 table (df=2 -> exact p = exp(-chi2/2)).

The within-frontier LP grade (mean 0.42->0.55) is reported to CSV but NOT plotted: at n=22 vs 40
it is not significant (perm p~0.20) and would dilute the figure.

Outputs (results/analysis/): f1_lp_distribution.{pdf,png}, f1_composition.csv, f1_frontier_lp.csv
Run:  python3 scripts/plot_f1_lp_distribution.py
"""
from __future__ import annotations
import csv
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
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_GLOB = os.path.join(ROOT, "results/expel_phase0_Base90_ExpeL/seeds/*.json")
GEN_GLOB = os.path.join(ROOT, "results/gen90_expel/bank/generated/*.json")
OUT_DIR = os.path.join(ROOT, "results/analysis")
BANDS = ps.BAND_ORDER
LP_GRADES = [1 / 3, 1 / 2, 2 / 3, 1.0]


def load(p):
    recs = [json.load(open(f)) for f in sorted(glob.glob(p))]
    if not recs:
        raise SystemExit(f"no records: {p}")
    return recs


def comp(recs):
    return Counter(r.get("classification") for r in recs)


def frontier_lp(recs):
    return [r["lp_value"] for r in recs
            if r.get("classification") == "frontier" and r.get("lp_value") is not None]


def chi2_3x2(a, b):
    obs = [[a.get(k, 0) for k in BANDS], [b.get(k, 0) for k in BANDS]]
    rt = [sum(r) for r in obs]
    ct = [obs[0][j] + obs[1][j] for j in range(3)]
    n = sum(rt)
    chi2 = sum((obs[i][j] - rt[i] * ct[j] / n) ** 2 / (rt[i] * ct[j] / n)
               for i in range(2) for j in range(3))
    return chi2, math.exp(-chi2 / 2.0)


def perm_means(x, y, n_perm=20000, seed=0):
    obs = abs(sum(x) / len(x) - sum(y) / len(y))
    pool = x + y
    nx = len(x)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(sum(pool[:nx]) / nx - sum(pool[nx:]) / (len(pool) - nx)) >= obs - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


def draw(ax, composition, chi2):
    conds = ["Base90", "Generated90"]
    ylabels = {"Base90": "Base-90\nraw SOTOPIA seeds", "Generated90": "Generated-90\nLP curriculum"}
    ypos = {"Base90": 0.0, "Generated90": 1.0}
    bar_h = 0.58
    for cond in conds:
        n = sum(composition[cond].values())
        left = 0.0
        for band in BANDS:
            cnt = composition[cond].get(band, 0)
            frac = cnt / n
            if frac <= 0:
                continue
            ax.barh(ypos[cond], frac, left=left, height=bar_h, color=ps.BAND_COLOR[band],
                    edgecolor="white", linewidth=1.0, zorder=3)
            if frac >= 0.11:
                ax.text(left + frac / 2, ypos[cond], f"{frac*100:.0f}%\n$n{{=}}{cnt}$",
                        ha="center", va="center", color="white", fontsize=7.6,
                        fontweight="bold", linespacing=1.05, zorder=4)
            elif frac >= 0.035:
                ax.text(left + frac / 2, ypos[cond] + bar_h / 2 + 0.06, f"{frac*100:.0f}%",
                        ha="center", va="bottom", color=ps.BAND_COLOR[band], fontsize=7.0,
                        fontweight="bold", zorder=4)
            left += frac
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([ylabels[c] for c in conds])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("share of bank (%)")
    ax.tick_params(axis="y", length=0)
    ps.despine(ax, left=False)
    ax.annotate(rf"$\chi^2(2)={chi2:.0f},\ p\!<\!10^{{-9}}$", xy=(1.0, 1.30),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=7.6, color=ps.GREY)
    handles = [Patch(facecolor=ps.BAND_COLOR[b], edgecolor="white", label=ps.BAND_LABEL[b])
               for b in BANDS]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base, gen = load(BASE_GLOB), load(GEN_GLOB)
    composition = {"Base90": comp(base), "Generated90": comp(gen)}
    flp = {"Base90": frontier_lp(base), "Generated90": frontier_lp(gen)}
    chi2, p_comp = chi2_3x2(composition["Base90"], composition["Generated90"])
    dmean, p_lp = perm_means(flp["Base90"], flp["Generated90"])

    print(f"chi2={chi2:.1f} p={p_comp:.2e}  | frontier mean LP "
          f"{sum(flp['Base90'])/len(flp['Base90']):.3f} -> {sum(flp['Generated90'])/len(flp['Generated90']):.3f}"
          f" (perm p={p_lp:.3f})")
    with open(os.path.join(OUT_DIR, "f1_composition.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "band", "count", "fraction"])
        for cond in ("Base90", "Generated90"):
            n = sum(composition[cond].values())
            for b in BANDS:
                w.writerow([cond, b, composition[cond].get(b, 0), f"{composition[cond].get(b,0)/n:.4f}"])
    with open(os.path.join(OUT_DIR, "f1_frontier_lp.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "lp_grade", "count"])
        for cond in ("Base90", "Generated90"):
            binned = Counter(min(LP_GRADES, key=lambda g: abs(g - v)) for v in flp[cond])
            for g in LP_GRADES:
                w.writerow([cond, f"{g:.4f}", binned.get(g, 0)])

    ps.apply()
    fig, ax = plt.subplots(figsize=(4.6, 1.95))
    draw(ax, composition, chi2)
    ps.save(fig, "f1_lp_distribution", OUT_DIR)


if __name__ == "__main__":
    main()
