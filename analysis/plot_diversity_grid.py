"""Cell-occupancy heatmap of the scenario embedding space.

Renders a grid_size x grid_size grid where each cell's color encodes the number
of scenarios that fall in it after PCA-2D projection. Empty cells are white;
saturated cells are hot. Much more informative than a scalar cell-coverage value.

Adapted from omni-epic/analysis/plot_diversity.py::plot_archive_diversity.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap


def build_cmap(max_count: int = 10):
    palette = plt.cm.inferno
    colors = palette(np.linspace(0, 1, 12))
    colors = colors[1:][::-1]                  # skip black, reverse
    colors[0] = [1, 1, 1, 1]                   # white for empty cells
    cmap = ListedColormap(colors)
    bounds = np.linspace(0, max_count, max_count + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def discretize(embeddings_2d: np.ndarray, grid_size: int,
               extent: tuple[float, float, float, float]) -> np.ndarray:
    min_x, max_x, min_y, max_y = extent
    grid = np.zeros((grid_size, grid_size), dtype=int)
    x_bins = np.linspace(min_x, max_x, grid_size + 1)
    y_bins = np.linspace(min_y, max_y, grid_size + 1)
    for x, y in embeddings_2d:
        xi = min(max(np.digitize(x, x_bins) - 1, 0), grid_size - 1)
        yi = min(max(np.digitize(y, y_bins) - 1, 0), grid_size - 1)
        grid[yi, xi] += 1
    return grid


def plot_heatmap(grid: np.ndarray, extent: tuple, output_path: str,
                 title: str, max_count_cap: int = 10) -> float:
    min_x, max_x, min_y, max_y = extent
    cmap, norm = build_cmap(max_count_cap)
    capped = np.minimum(grid, max_count_cap)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(capped, cmap=cmap, norm=norm, interpolation="nearest",
              origin="lower", extent=[min_x, max_x, min_y, max_y])
    ax.set_aspect("auto")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Scenarios per cell",
                        ticks=np.arange(max_count_cap + 1))
    cbar.set_ticklabels([str(i) if i < max_count_cap else f"{max_count_cap}+"
                         for i in range(max_count_cap + 1)])

    gs = grid.shape[0]
    ax.set_xticks(np.linspace(min_x, max_x, gs + 1), minor=True)
    ax.set_yticks(np.linspace(min_y, max_y, gs + 1), minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.4, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    coverage = (grid > 0).sum() / grid.size
    ax.text(0.02, 0.98, f"Cell coverage: {coverage:.3f}\nTotal scenarios: {grid.sum()}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return coverage


def split_grids(scenarios: list[dict], embeddings_2d: np.ndarray,
                grid_size: int, extent: tuple) -> dict[str, np.ndarray]:
    """Return separate grids for seeds and generated subsets."""
    is_seed = np.array([s.get("source", "") in ("seed_sotopia", "fallback_seed")
                        for s in scenarios])
    grids = {
        "all":       discretize(embeddings_2d, grid_size, extent),
        "seeds":     discretize(embeddings_2d[is_seed], grid_size, extent),
        "generated": discretize(embeddings_2d[~is_seed], grid_size, extent),
    }
    return grids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: same as archive)")
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--max_count", type=int, default=10,
                    help="Cap the colorbar at this count; values above show as N+")
    args = ap.parse_args()

    archive = json.load(open(args.archive))
    scenarios = [s for s in archive["successful"] if s.get("embedding")]
    if len(scenarios) < 3:
        print("Not enough data."); return

    embs = np.array([s["embedding"] for s in scenarios])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    xy = pca.fit_transform(embs)

    extent = (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max())
    out_dir = Path(args.out_dir or Path(args.archive).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    grids = split_grids(scenarios, xy, args.grid_size, extent)
    for label, grid in grids.items():
        path = out_dir / f"diversity_grid_{label}_{args.grid_size}x{args.grid_size}.png"
        cov = plot_heatmap(grid, extent, str(path),
                           title=f"Cell occupancy ({label}) — {args.grid_size}x{args.grid_size}",
                           max_count_cap=args.max_count)
        print(f"  {label:>10}: coverage={cov:.3f}  total={grid.sum()}  -> {path}")


if __name__ == "__main__":
    main()
