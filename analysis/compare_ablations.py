"""Compare multiple archives in a common 2D space.

Pass several --archive <name>=<path> pairs. The script:
  1. Loads all archives, concatenates their embeddings, and fits ONE PCA on the
     union. All methods are projected through this shared model, so their cell
     coverage values are directly comparable.
  2. Produces, in --out_dir:
       compare_grid.png            -- side-by-side cell-occupancy heatmaps
       compare_coverage.png        -- bar chart of cell coverage per method
       compare_diversity_curve.png -- overlaid coverage-over-iteration lines
                                      (requires metrics.json next to each archive)

Adapted from omni-epic/analysis/plot_diversity.py.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.decomposition import PCA


def parse_archive_arg(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise ValueError(f"Expected name=path, got: {arg}")
    name, path = arg.split("=", 1)
    return name.strip(), Path(path.strip())


def load_one(path: Path, truncate_to: int | None = None) -> list[dict]:
    """Return the scenarios with embeddings, optionally truncated to a target size.

    Truncation preserves seeds (iteration < 0) and takes scenarios in iteration order,
    so a method with rejections doesn't lose its early/late distribution.
    """
    archive = json.load(open(path))
    scenarios = [s for s in archive["successful"] if s.get("embedding")]
    if truncate_to is None or len(scenarios) <= truncate_to:
        return scenarios
    seeds = [s for s in scenarios if s.get("iteration", -1) < 0]
    gens = sorted([s for s in scenarios if s.get("iteration", -1) >= 0],
                  key=lambda s: s.get("iteration", 0))
    keep_gens = truncate_to - len(seeds)
    if keep_gens < 0:
        return seeds[:truncate_to]
    return seeds + gens[:keep_gens]


def discretize(xy: np.ndarray, grid_size: int, extent: tuple) -> np.ndarray:
    min_x, max_x, min_y, max_y = extent
    grid = np.zeros((grid_size, grid_size), dtype=int)
    x_bins = np.linspace(min_x, max_x, grid_size + 1)
    y_bins = np.linspace(min_y, max_y, grid_size + 1)
    for x, y in xy:
        xi = min(max(np.digitize(x, x_bins) - 1, 0), grid_size - 1)
        yi = min(max(np.digitize(y, y_bins) - 1, 0), grid_size - 1)
        grid[yi, xi] += 1
    return grid


def build_cmap(max_count: int):
    palette = plt.cm.inferno
    colors = palette(np.linspace(0, 1, 12))
    colors = colors[1:][::-1]
    colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)
    bounds = np.linspace(0, max_count, max_count + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def plot_compare_grid(method_xy: dict[str, np.ndarray], grid_size: int,
                      extent: tuple, output_path: str, max_count: int = 10) -> dict[str, float]:
    cmap, norm = build_cmap(max_count)
    methods = list(method_xy.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n + 1, 6), squeeze=False)
    coverages: dict[str, float] = {}
    for ax, name in zip(axes[0], methods):
        grid = discretize(method_xy[name], grid_size, extent)
        coverages[name] = (grid > 0).sum() / grid.size
        capped = np.minimum(grid, max_count)
        min_x, max_x, min_y, max_y = extent
        ax.imshow(capped, cmap=cmap, norm=norm, interpolation="nearest",
                  origin="lower", extent=[min_x, max_x, min_y, max_y])
        ax.set_aspect("auto")
        ax.set_title(f"{name}\ncoverage={coverages[name]:.3f}  n={grid.sum()}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks(np.linspace(min_x, max_x, grid_size + 1), minor=True)
        ax.set_yticks(np.linspace(min_y, max_y, grid_size + 1), minor=True)
        ax.grid(which="minor", color="grey", linewidth=0.4, alpha=0.5)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[0].tolist(), label="Scenarios per cell",
                      ticks=np.arange(max_count + 1), fraction=0.04, pad=0.02)
    cb.set_ticklabels([str(i) if i < max_count else f"{max_count}+"
                       for i in range(max_count + 1)])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return coverages


def plot_coverage_bar(coverages: dict[str, float], output_path: str) -> None:
    methods = list(coverages.keys())
    values = [coverages[m] for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(methods)), 4.5))
    bars = ax.bar(methods, values, color=plt.get_cmap("tab10").colors[:len(methods)])
    ax.set_ylabel("Cell coverage (shared 2D space)")
    ax.set_ylim(0, 1)
    ax.set_title("Coverage by method")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_diversity_curves(metrics_paths: dict[str, Path], output_path: str) -> None:
    fig, (ax_cov, ax_size) = plt.subplots(1, 2, figsize=(14, 5))
    # Cycle styles so overlapping curves (e.g. no_moi vs no_archive on archive size,
    # which are mathematically identical) remain visually distinguishable.
    styles = [
        {"linestyle": "-",  "linewidth": 2.4, "alpha": 0.95},
        {"linestyle": "--", "linewidth": 2.0, "alpha": 0.80},
        {"linestyle": ":",  "linewidth": 2.6, "alpha": 0.75},
        {"linestyle": "-.", "linewidth": 2.0, "alpha": 0.70},
    ]
    palette = plt.get_cmap("tab10").colors
    for i, (name, path) in enumerate(metrics_paths.items()):
        if not path.exists():
            print(f"  [skip] no metrics.json at {path}")
            continue
        m = json.load(open(path))
        iters = [r["iteration"] for r in m]
        covs = [r["cell_coverage"] for r in m]
        sizes = [r["archive_size"] for r in m]
        style = styles[i % len(styles)]
        color = palette[i % len(palette)]
        ax_cov.plot(iters, covs, label=name, color=color, **style)
        ax_size.plot(iters, sizes, label=name, color=color, **style)
    ax_cov.set_xlabel("Iteration"); ax_cov.set_ylabel("Cell coverage (per-run PCA)")
    ax_cov.set_title("Diversity growth (per-run PCA — trajectories only, not levels)")
    ax_cov.legend()
    ax_cov.grid(alpha=0.3)
    ax_size.set_xlabel("Iteration"); ax_size.set_ylabel("Archive size")
    ax_size.set_title("Archive growth (no_moi / no_archive coincide exactly)")
    ax_size.legend()
    ax_size.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", action="append", required=True,
                    help="name=path pairs, e.g. --archive full=output/200_full/archive_latest.json")
    ap.add_argument("--out_dir", default="output/comparison")
    ap.add_argument("--grid_size", type=int, default=10)
    ap.add_argument("--max_count", type=int, default=10)
    ap.add_argument("--equal_size", action="store_true", default=True,
                    help="Truncate every archive to min(size) before computing coverage. "
                         "Defaults ON to avoid the volume confound — methods that accept "
                         "more scenarios naturally cover more cells.")
    ap.add_argument("--no_equal_size", dest="equal_size", action="store_false",
                    help="Disable equal-size truncation (use raw archive sizes).")
    args = ap.parse_args()

    parsed = [parse_archive_arg(a) for a in args.archive]
    raw_scenarios = {name: load_one(p) for name, p in parsed}
    raw_sizes = {name: len(v) for name, v in raw_scenarios.items()}

    if args.equal_size and len(raw_scenarios) > 1:
        target = min(raw_sizes.values())
        method_scenarios = {name: load_one(p, truncate_to=target)
                            for name, p in parsed}
        print(f"Equal-size mode: truncating every archive to {target} scenarios "
              f"(raw sizes: {raw_sizes})")
    else:
        method_scenarios = raw_scenarios
        print(f"Using raw archive sizes (volume confound NOT controlled): {raw_sizes}")

    # Fit PCA on the union of embeddings
    all_embs = np.concatenate(
        [np.array([s["embedding"] for s in v]) for v in method_scenarios.values()]
    )
    pca = PCA(n_components=2)
    pca.fit(all_embs)

    method_xy = {name: pca.transform(np.array([s["embedding"] for s in v]))
                 for name, v in method_scenarios.items()}

    union_xy = np.concatenate(list(method_xy.values()))
    extent = (union_xy[:, 0].min(), union_xy[:, 0].max(),
              union_xy[:, 1].min(), union_xy[:, 1].max())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Methods: {list(method_scenarios.keys())}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    coverages = plot_compare_grid(method_xy, args.grid_size, extent,
                                  str(out_dir / "compare_grid.png"),
                                  max_count=args.max_count)
    plot_coverage_bar(coverages, str(out_dir / "compare_coverage.png"))

    # Diversity-over-time curves use the in-run cell coverage that was logged.
    metrics_paths = {name: p.parent / "metrics.json" for name, p in parsed}
    plot_diversity_curves(metrics_paths, str(out_dir / "compare_diversity_curve.png"))

    # Dump numbers
    json.dump({"coverages": coverages,
               "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
               "n_scenarios_used": {name: len(v) for name, v in method_scenarios.items()},
               "n_scenarios_raw": raw_sizes,
               "equal_size_mode": args.equal_size},
              open(out_dir / "compare_summary.json", "w"), indent=2)

    print(f"\nWrote to {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
