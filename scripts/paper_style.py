"""Shared publication style for the Social-OMNI-EPIC figures (top-tier-venue grade).

One import, consistent typography/colour across every figure:

    import paper_style as ps
    ps.apply()
    fig, ax = plt.subplots(figsize=ps.WIDTH_2COL_HALF)
    ...
    ps.panel_label(ax, "a"); ps.despine(ax)

Design choices: clean sans (Helvetica/Arial, DejaVu Sans fallback); thin out-ticks; hairline
spines; whisper-light grid only where it aids reading; Okabe-Ito colour-blind-safe palette;
direct labelling over boxed legends. Switch FONT_FAMILY to "serif" for a Times camera-ready.
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- column geometry (inches) for a 2-column paper ----------------------------------------
WIDTH_1COL = 3.35           # single column
WIDTH_2COL = 7.0            # full text width
WIDTH_2COL_HALF = (3.45, 2.5)
WIDTH_2COL_WIDE = (7.0, 2.8)

FONT_FAMILY = "sans-serif"  # -> "serif" for a Times/CM look

# --- Okabe & Ito (2008) colour-blind-safe palette -----------------------------------------
PALETTE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
}
GREY = "#8A8A8A"
GRID = "#E9E9E9"
INK = "#1A1A1A"          # near-black for text/spines (softer than pure black)

# the three curriculum bands, used identically everywhere
BAND_COLOR = {"too_easy": PALETTE["blue"], "frontier": PALETTE["green"],
              "beyond_frontier": PALETTE["vermillion"]}
BAND_LABEL = {"too_easy": "too easy", "frontier": "frontier", "beyond_frontier": "beyond frontier"}
BAND_ORDER = ["too_easy", "frontier", "beyond_frontier"]


def apply():
    mpl.rcParams.update({
        # typography
        "font.family": FONT_FAMILY,
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "dejavusans" if FONT_FAMILY == "sans-serif" else "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "legend.fontsize": 7.6,
        "figure.titlesize": 9.5,
        # colour of text / lines
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        # spines & ticks: hairline, out-pointing, short
        "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        # grid: whisper-light, behind everything (enable per-axis)
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        # lines
        "lines.linewidth": 1.7, "lines.solid_capstyle": "round", "lines.antialiased": True,
        "patch.linewidth": 0.7,
        # legend
        "legend.frameon": False, "legend.handlelength": 1.1, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.3,
        # output
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,  # editable text in PDF
    })


def despine(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def panel_label(ax, letter, x=-0.02, y=1.04, **kw):
    """Bold lower-case panel letter, top-left, in the (a)/(b) convention."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold", color=INK, **kw)


def grid_y(ax):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem, out_dir):
    import os
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path)
        print("wrote", path)
