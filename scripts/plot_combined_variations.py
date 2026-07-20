#!/usr/bin/env python3
"""Several styled variations of the combined Tier-1 figure, for picking a direction.

Panel (a): band composition, Generated-90 (LP curriculum) vs Base-90 (raw SOTOPIA seeds).
Panel (b): frontier fraction over the curriculum (sliding window + Wilson 95% CI).

Each variation differs in LAYOUT (stacked vs side-by-side), FONT, COLOR theme, and SPACING.
Outputs (results/analysis/variations/): var_*.png   plus the chosen one can be promoted later.

Run:  python3 scripts/plot_combined_variations.py
"""
from __future__ import annotations
import glob
import json
import math
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_GLOB = os.path.join(ROOT, "results/expel_phase0_Base90_ExpeL/seeds/*.json")
GEN_GLOB = os.path.join(ROOT, "results/gen90_expel/bank/generated/*.json")
METRICS = os.path.join(ROOT, "results/gen90_expel/metrics.json")
OUT_DIR = os.path.join(ROOT, "results/analysis/variations")

BANDS = ["too_easy", "frontier", "beyond_frontier"]
BAND_LABEL = {"too_easy": "too easy", "frontier": "frontier", "beyond_frontier": "beyond frontier"}
WINDOW = 21


# ============================================================================ themes
# Each theme: fonts, ink/grey, the 3 band colors (light variant for soft fills), accent.
THEMES = {
    "okabe": {
        "name": "Okabe-Ito - clean sans",
        "family": "sans-serif",
        "sans": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathfont": "dejavusans",
        "ink": "#1A1A1A", "grey": "#8A8A8A", "grid": "#ECECEC",
        "bands": {"too_easy": "#0072B2", "frontier": "#009E73", "beyond_frontier": "#D55E00"},
        "accent": "#009E73",
        "base_x": 1.10,
    },
    "muted_serif": {
        "name": "Muted - camera-ready serif",
        "family": "serif",
        "sans": ["Helvetica", "Arial", "DejaVu Sans"],
        "serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathfont": "stix",
        "ink": "#23262B", "grey": "#9AA0A6", "grid": "#EDEEF0",
        "bands": {"too_easy": "#4C72B0", "frontier": "#2A9D8F", "beyond_frontier": "#E76F51"},
        "accent": "#2A9D8F",
        "base_x": 1.10,
    },
    "modern": {
        "name": "Modern - soft sans, airy",
        "family": "sans-serif",
        "sans": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathfont": "dejavusans",
        "ink": "#2B2D33", "grey": "#9097A1", "grid": "#EEF1F4",
        "bands": {"too_easy": "#3D5A80", "frontier": "#1B998B", "beyond_frontier": "#E07A5F"},
        "accent": "#1B998B",
        "base_x": 1.10,
    },
}


def apply_theme(t, base=9.0):
    mpl.rcParams.update({
        "font.family": t["family"],
        "font.sans-serif": t["sans"],
        "font.serif": t["serif"],
        "mathtext.fontset": t["mathfont"],
        "font.size": base,
        "axes.titlesize": base + 1.5,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1.2,
        "ytick.labelsize": base - 1.2,
        "legend.fontsize": base - 0.8,
        "text.color": t["ink"], "axes.labelcolor": t["ink"], "axes.edgecolor": t["ink"],
        "xtick.color": t["ink"], "ytick.color": t["ink"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
        "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.solid_capstyle": "round", "lines.antialiased": True,
        "patch.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


# ============================================================================== data
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


# ============================================================================ panels
ROWS = [("Generated-90", "Generated-90", "LP curriculum"),
        ("Base90", "Base-90", "raw SOTOPIA seeds")]


def panel_letter(ax, letter, t, dx=0.0, dy=20):
    ax.annotate(f"({letter})", xy=(dx, 1.0), xytext=(0, dy), xycoords="axes fraction",
                textcoords="offset points", ha="left", va="bottom",
                fontsize=mpl.rcParams["axes.titlesize"] + 1.5, fontweight="bold",
                color=t["ink"], annotation_clip=False)


def panel_composition(ax, composition, chi2, t):
    ypos = {"Generated-90": 1.0, "Base90": 0.0}
    bar_h = 0.46
    for key, _, _ in ROWS:
        c = composition[key]; n = sum(c.values()); left = 0.0
        for band in BANDS:
            frac = c.get(band, 0) / n
            if frac <= 0:
                continue
            ax.barh(ypos[key], frac, left=left, height=bar_h, color=t["bands"][band],
                    edgecolor="white", linewidth=1.6, zorder=3)
            if frac >= 0.11:
                ax.text(left + frac / 2, ypos[key], f"{frac*100:.0f}%\n$n{{=}}{c.get(band,0)}$",
                        ha="center", va="center", color="white",
                        fontsize=mpl.rcParams["font.size"] - 0.5, fontweight="bold",
                        linespacing=1.25, zorder=4)
            elif frac >= 0.03:
                # push the label to the OUTSIDE of the bar (away from the neighbouring bar)
                cx = left + frac / 2
                sgn = -1 if ypos[key] == 0 else 1   # bottom bar -> below, top bar -> above
                ax.annotate(f"{frac*100:.0f}%",
                            xy=(cx, ypos[key] + sgn * bar_h / 2),
                            xytext=(cx, ypos[key] + sgn * (bar_h / 2 + 0.34)),
                            ha="center", va="top" if sgn < 0 else "bottom",
                            color=t["bands"][band],
                            fontsize=mpl.rcParams["font.size"] - 1.5, fontweight="bold", zorder=4,
                            arrowprops=dict(arrowstyle="-", color=t["bands"][band], lw=0.8,
                                            shrinkA=0, shrinkB=1))
            left += frac
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.85, 1.85)
    ax.set_yticks([])
    for key, title, sub in ROWS:
        ax.text(-0.02, ypos[key] + 0.16, title, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=mpl.rcParams["font.size"] + 0.5,
                fontweight="bold", color=t["ink"])
        ax.text(-0.02, ypos[key] - 0.17, sub, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=mpl.rcParams["font.size"] - 1.5, color=t["grey"])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel("share of scenario archive (%)")
    ax.tick_params(axis="y", length=0)
    despine(ax, left=False)
    ax.spines["left"].set_visible(False)
    ax.annotate(rf"$\chi^2(2)={chi2:.0f},\ p<10^{{-9}}$", xy=(1.0, 1.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=mpl.rcParams["font.size"] - 1.0, color=t["grey"])


def panel_trend(ax, series, base_rate, r, p, t):
    xs, mean, lo, hi = sliding(series, WINDOW)
    N = len(series)
    acc = t["accent"]
    ax.axhline(base_rate, ls=(0, (5, 4)), lw=1.1, color=t["grey"], zorder=2)
    ax.text(N * t["base_x"] - N * 0.10, base_rate - 0.03,
            f"raw-seed frontier rate ({base_rate*100:.0f}%)",
            ha="right", va="top", fontsize=mpl.rcParams["font.size"] - 1.5, color=t["grey"])
    for i, v in enumerate(series):
        if v:
            ax.plot([i + 1, i + 1], [1.01, 1.045], color=acc, lw=0.7, alpha=0.5,
                    clip_on=False, zorder=2)
    ax.fill_between(xs, lo, hi, color=acc, alpha=0.14, lw=0, zorder=2)
    ax.plot(xs, mean, color=acc, lw=2.4, zorder=4)

    # stable early-vs-late anchors: equal-n half means (no edge-window artifact)
    h = N // 2
    m1 = sum(series[:h]) / h
    m2 = sum(series[h:]) / (N - h)
    for (x0, x1), mval in [((1, h), m1), ((h + 1, N), m2)]:
        ax.plot([x0, x1], [mval, mval], color=t["ink"], lw=1.4, ls=(0, (1, 1.4)),
                alpha=0.85, zorder=5)
    ax.annotate(f"first half\n{m1*100:.0f}%", xy=(h / 2, m1), xytext=(0, -4),
                textcoords="offset points", ha="center", va="top",
                fontsize=mpl.rcParams["font.size"] - 1.5, color=t["ink"], linespacing=1.15)
    ax.annotate(f"second half\n{m2*100:.0f}%", xy=((h + 1 + N) / 2, m2), xytext=(0, 7),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=mpl.rcParams["font.size"] - 1.5, fontweight="bold", color=acc,
                linespacing=1.15)
    ax.set_xlim(1, N)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("curriculum iteration")
    ax.set_ylabel("frontier fraction")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    despine(ax)
    ax.grid(axis="y", zorder=0)
    star = "*" if p < 0.05 else ""
    ax.annotate(rf"$r={r:+.2f}$,  perm $p={p:.3f}${star}", xy=(0.04, 0.96),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=mpl.rcParams["font.size"] - 0.5, color=t["ink"])
    ax.annotate(rf"1-sided $\approx${p/2:.3f}   $\cdot$   window $={WINDOW}$", xy=(0.04, 0.895),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=mpl.rcParams["font.size"] - 2.0, color=t["grey"])


# =========================================================================== render
def render(theme_key, layout, data, fname):
    t = THEMES[theme_key]
    composition, chi2, series, base_rate, r, p = data
    base_fs = 9.5 if layout == "stacked" else 9.0
    apply_theme(t, base=base_fs)

    if layout == "stacked":
        fig = plt.figure(figsize=(7.4, 6.3))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.95,
                              left=0.17, right=0.965, top=0.91, bottom=0.085)
        axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[1, 0])
        ta_pad, tb_pad = 22, 16
        let_dx_a, let_dx_b = -0.165, -0.105
        leg_anchor = (0.5, -0.55)
    else:  # side
        fig = plt.figure(figsize=(10.2, 3.9))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.26,
                              left=0.10, right=0.975, top=0.84, bottom=0.31)
        axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1])
        ta_pad, tb_pad = 24, 18
        let_dx_a, let_dx_b = -0.16, -0.11
        leg_anchor = (0.5, -0.27)

    panel_composition(axa, composition, chi2, t)
    panel_trend(axb, series, base_rate, r, p, t)
    axa.set_title("generated scenarios are harder than raw seeds", loc="left", pad=ta_pad)
    axb.set_title("frontier scenarios grow more common over the run", loc="left", pad=tb_pad)
    panel_letter(axa, "a", t, dx=let_dx_a, dy=ta_pad - 2)
    panel_letter(axb, "b", t, dx=let_dx_b, dy=tb_pad - 2)

    # band legend attached beneath panel (a) (bands belong to the composition panel)
    handles = [Patch(facecolor=t["bands"][b], edgecolor="white", label=BAND_LABEL[b]) for b in BANDS]
    leg = axa.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=leg_anchor,
                     frameon=False, handlelength=1.2, columnspacing=1.8,
                     title="scenario difficulty band", title_fontsize=base_fs - 1.0)
    leg.get_title().set_color(t["grey"])

    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", path, "  [", t["name"], "/", layout, "]")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base, gen = load(BASE_GLOB), load(GEN_GLOB)
    composition = {"Base90": comp(base), "Generated-90": comp(gen)}
    chi2, _ = chi2_3x2(composition["Base90"], composition["Generated-90"])
    series = frontier_series()
    base_rate = sum(r.get("classification") == "frontier" for r in base) / len(base)
    r, p = point_biserial_trend(series)
    data = (composition, chi2, series, base_rate, r, p)

    render("okabe", "stacked", data, "var1_okabe_stacked.png")
    render("muted_serif", "stacked", data, "var2_serif_stacked.png")
    render("modern", "side", data, "var3_modern_side.png")
    render("modern", "stacked", data, "var4_modern_stacked.png")


if __name__ == "__main__":
    main()
