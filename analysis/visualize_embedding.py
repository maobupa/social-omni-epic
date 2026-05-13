"""UMAP visualization of the archive."""
import argparse
import json
from collections import Counter
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# Substring -> umbrella label. Order matters: first match wins.
# Tuned to merge LLM-generated compound types ("commercial_lease_negotiation"
# -> "negotiation") and Sotopia source tags ("craigslist_bargains" -> "negotiation").
UMBRELLA_RULES: list[tuple[str, str]] = [
    ("negotiat", "negotiation"),
    ("bargain", "negotiation"),
    ("deal-or-no-deal", "negotiation"),
    ("persua", "persuasion"),
    ("mediat", "mediation"),
    ("conflict", "conflict"),
    ("dispute", "conflict"),
    ("competit", "competition"),
    ("rivalr", "competition"),
    ("cooperat", "cooperation"),
    ("collabor", "cooperation"),
    ("support", "support"),
    ("comfort", "support"),
    ("decept", "deception"),
    ("manipulat", "deception"),
    ("interview", "interview"),
    ("interrog", "interrogation"),
    ("apolog", "reconciliation"),
    ("reconcil", "reconciliation"),
    ("confront", "confrontation"),
    ("ethic", "ethical_dilemma"),
    ("moral", "ethical_dilemma"),
    ("dilemma", "ethical_dilemma"),
    ("polic", "policy_discussion"),
    ("discussion", "discussion"),
    ("conversation", "discussion"),
    ("planning", "planning"),
    ("decision", "decision_making"),
    ("mentor", "mentorship"),
    ("teach", "mentorship"),
    ("advice", "advising"),
    ("complaint", "complaint"),
    ("apology", "reconciliation"),
    ("social_iqa", "everyday_social"),
    ("social_chemistry", "everyday_social"),
    ("normbank", "norm_negotiation"),
    ("mutual_friend", "interpersonal"),
    ("hand-craft", "hand_crafted"),
    ("family", "family"),
    ("workplace", "workplace"),
    ("community", "community"),
    ("school", "education"),
    ("educat", "education"),
]


def umbrella_label(raw: str) -> str:
    if not raw:
        return "unknown"
    s = raw.lower().strip()
    for needle, label in UMBRELLA_RULES:
        if needle in s:
            return label
    return raw  # leave untouched if no rule matches


def collapse_labels(labels: list[str], top_k: Optional[int] = None,
                    use_umbrella: bool = True) -> tuple[list[str], list[str]]:
    """Returns (collapsed_labels, ordered_unique_labels).

    Steps: normalize case → optional umbrella mapping → optional top-K + "other".
    """
    if use_umbrella:
        mapped = [umbrella_label(l) for l in labels]
    else:
        mapped = [l.lower().strip() or "unknown" for l in labels]
    if top_k is not None:
        counts = Counter(mapped)
        keep = {t for t, _ in counts.most_common(top_k)}
        mapped = [m if m in keep else "other" for m in mapped]
    # ordered by frequency (most common first) for stable legend order
    ordered = [t for t, _ in Counter(mapped).most_common()]
    return mapped, ordered


def plot_scenario_space(archive_path: str, output_path: str,
                        top_k: Optional[int] = 10,
                        use_umbrella: bool = True):
    from umap import UMAP

    with open(archive_path) as f:
        archive = json.load(f)

    scenarios = archive["successful"]
    embeddings = np.array([s["embedding"] for s in scenarios if s.get("embedding")])
    iterations = np.array([s["iteration"] for s in scenarios if s.get("embedding")])
    sources = [s.get("source", "generated") for s in scenarios if s.get("embedding")]
    raw_types = [s.get("interaction_type") or s.get("tag") or "unknown"
                 for s in scenarios if s.get("embedding")]

    if len(embeddings) < 3:
        print("Not enough data to plot.")
        return

    interaction_types, ordered_types = collapse_labels(
        raw_types, top_k=top_k, use_umbrella=use_umbrella
    )
    print(f"Collapsed {len(set(raw_types))} raw types → {len(ordered_types)} groups")
    for t in ordered_types:
        print(f"  {sum(1 for x in interaction_types if x == t):4d}  {t}")

    reducer = UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    is_seed = np.array([s in ("seed_sotopia", "fallback_seed") for s in sources])
    is_gen = ~is_seed

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    if is_seed.any():
        axes[0].scatter(reduced[is_seed, 0], reduced[is_seed, 1],
                        c="lightgray", marker="x", s=60, alpha=0.9,
                        label=f"Seeds ({is_seed.sum()})", linewidths=1.5)
    if is_gen.any():
        sc = axes[0].scatter(reduced[is_gen, 0], reduced[is_gen, 1],
                             c=iterations[is_gen], cmap="plasma", s=30, alpha=0.7)
        plt.colorbar(sc, ax=axes[0], label="Generation Iteration")
    axes[0].legend()
    axes[0].set_title("Expansion Beyond Seed Task Space")
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    # Build color map: pick a perceptually distinct palette based on N groups.
    n = len(ordered_types)
    cmap = plt.get_cmap("tab10" if n <= 10 else ("tab20" if n <= 20 else "hsv"))
    type_to_color = {t: cmap(i / max(1, n)) for i, t in enumerate(ordered_types)}
    point_colors = [type_to_color[t] for t in interaction_types]

    axes[1].scatter(reduced[:, 0], reduced[:, 1], c=point_colors,
                    alpha=0.75, s=30, edgecolors="none")
    axes[1].set_title(f"Scenario Space (by interaction-type group, top {n})")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

    from matplotlib.lines import Line2D
    counts = Counter(interaction_types)
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=type_to_color[t],
               label=f"{t} ({counts[t]})", markersize=8)
        for t in ordered_types
    ]
    axes[1].legend(handles=legend_elements, loc="best", fontsize=8,
                   framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_diversity_over_time(metrics_path: str, output_path: str):
    with open(metrics_path) as f:
        metrics = json.load(f)
    if not metrics:
        print("No metrics to plot.")
        return
    iterations = [m["iteration"] for m in metrics]
    coverages = [m["cell_coverage"] for m in metrics]
    sizes = [m["archive_size"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(iterations, coverages, "b-", label="Cell Coverage")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Cell Coverage", color="b")
    ax2 = ax1.twinx()
    ax2.plot(iterations, sizes, "r--", label="Archive Size")
    ax2.set_ylabel("Archive Size", color="r")
    ax1.set_title("Scenario Diversity Over Iterations")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Show only the top-K interaction-type groups; lump the rest as 'other'. Use 0 for all.")
    ap.add_argument("--no_umbrella", action="store_true",
                    help="Disable substring-based umbrella grouping (use raw type strings).")
    args = ap.parse_args()

    from pathlib import Path
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top_k = None if args.top_k == 0 else args.top_k
    plot_scenario_space(args.archive, str(out_dir / "scenario_space.png"),
                        top_k=top_k, use_umbrella=not args.no_umbrella)
    if args.metrics:
        plot_diversity_over_time(args.metrics, str(out_dir / "diversity_over_time.png"))
