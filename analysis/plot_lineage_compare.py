"""Cross-condition lineage HTMLs in ONE frozen coordinate space.

Problem this solves: `plot_lineage.py --shared_with` re-fits UMAP per HTML on a
differently-ordered union and rescales per-run, so the *same* scenario lands at
different pixels across conditions (see analysis notes). On top of that, the
embeddings stored in each archive were computed during that run, and
`text-embedding-3-small` is not perfectly deterministic, so even byte-identical
seed texts drift.

This script fixes both at comparison time:
  1. Pool every scenario across the specified conditions.
  2. Deduplicate by the exact text fed to the embedder
     (mirrors SocialScenario.to_text_for_embedding). Identical scenarios
     (notably the 90 shared Sotopia seeds) collapse to ONE point.
  3. Re-embed each unique text exactly once (single OpenAI pass).
  4. Fit UMAP ONCE on those unique embeddings -> one global position map
     + one shared extent.
  5. Render one HTML per condition reusing those exact coordinates.

Result: any scenario is at the same pixel in every HTML, and the conditions are
directly comparable / overlay-able. Larger nodes + click-to-open detail popups
come for free from plot_lineage.render().

Usage:
    python analysis/plot_lineage_compare.py \
        --condition full=output/200_full/archive_latest.json \
        --condition no_moi=output/200_no_moi/archive_latest.json \
        --condition no_archive=output/200_no_archive/archive_latest.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from plot_lineage import build_graph, render  # noqa: E402


def embedding_text(s: dict) -> str:
    """Mirror SocialScenario.to_text_for_embedding for a raw archive dict.

    Must stay in sync with social_omni_epic/data_models.py:to_text_for_embedding
    so dedup keys match what the pipeline actually embedded.
    """
    profiles = s.get("agent_profiles") or []
    goals = s.get("agent_goals") or []
    parts = [
        f"Scenario: {s.get('scenario', '')}",
        f"Interaction type: {s.get('interaction_type', '')}",
        f"Relationship: {s.get('relationship', '')}",
    ]
    for i, goal in enumerate(goals):
        name = profiles[i].get("first_name", "") if i < len(profiles) else f"Agent{i}"
        parts.append(f"{name}'s goal: {goal}")
    for ag in profiles:
        parts.append(
            f"Character {ag.get('first_name', '')}: "
            f"{ag.get('occupation', '')}; {ag.get('big_five', '')}"
        )
    return "\n".join(parts)


def scale_to_canvas(xy: np.ndarray, spread: float) -> np.ndarray:
    """Same canvas mapping as plot_lineage.build_position_map."""
    width, height = 3200 * spread, 2400 * spread
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    out = xy.copy().astype(float)
    out[:, 0] = (out[:, 0] - xmin) / max(xmax - xmin, 1e-9) * width - width / 2
    out[:, 1] = (out[:, 1] - ymin) / max(ymax - ymin, 1e-9) * height - height / 2
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--condition", action="append", required=True, metavar="LABEL=ARCHIVE",
        help="Repeat per condition, e.g. --condition full=output/200_full/archive_latest.json",
    )
    ap.add_argument("--out_name", default="lineage_closest_shared.html",
                    help="HTML filename written into each condition's archive dir")
    ap.add_argument("--closest_only", action="store_true", default=True,
                    help="Collapse to the spatially-closest parent edge per node")
    ap.add_argument("--include_failed", action="store_true", default=True)
    ap.add_argument("--spread", type=float, default=2.0,
                    help="Canvas scale multiplier (matches plot_lineage)")
    ap.add_argument("--embed_batch", type=int, default=256,
                    help="Texts per OpenAI embeddings request")
    args = ap.parse_args()

    conditions: list[tuple[str, Path]] = []
    for spec in args.condition:
        if "=" not in spec:
            ap.error(f"--condition must be LABEL=ARCHIVE, got {spec!r}")
        label, path = spec.split("=", 1)
        conditions.append((label, Path(path)))

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import os
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set (looked in env and .env).")

    # 1-2. Pool all scenarios, dedupe by embedding text.
    archives: dict[str, dict] = {}
    text_to_idx: dict[str, int] = {}
    unique_texts: list[str] = []
    for label, path in conditions:
        archive = json.load(open(path))
        archives[label] = archive
        pool = archive.get("successful", []) + archive.get("failed_interestingness", [])
        for s in pool:
            t = embedding_text(s)
            if t not in text_to_idx:
                text_to_idx[t] = len(unique_texts)
                unique_texts.append(t)
    print(f"{len(conditions)} conditions, "
          f"{sum(len(json.load(open(p)).get('successful', [])) for _, p in conditions)} "
          f"successful scenarios pooled -> {len(unique_texts)} unique texts to embed")

    # 3. Re-embed each unique text exactly once.
    from social_omni_epic.fm import FM
    fm = FM()
    embeddings: list[list[float]] = []
    for i in range(0, len(unique_texts), args.embed_batch):
        chunk = unique_texts[i:i + args.embed_batch]
        embeddings.extend(fm.get_embeddings(chunk))
        print(f"  embedded {min(i + len(chunk), len(unique_texts))}/{len(unique_texts)}")
    emb_arr = np.array(embeddings)

    # 4. One UMAP fit over the unique embeddings -> global, frozen positions.
    from umap import UMAP
    reducer = UMAP(n_components=2, random_state=42, min_dist=0.5, n_neighbors=15)
    xy = scale_to_canvas(reducer.fit_transform(emb_arr), args.spread)
    pos_by_text = {t: (float(xy[idx, 0]), float(xy[idx, 1]))
                   for t, idx in text_to_idx.items()}

    # 5. Render one HTML per condition reusing those exact coordinates.
    for label, path in conditions:
        archive = archives[label]
        pool = archive.get("successful", []) + archive.get("failed_interestingness", [])
        positions_override = {s["id"]: pos_by_text[embedding_text(s)] for s in pool}
        G, _ = build_graph(
            archive,
            include_failed=args.include_failed,
            closest_only=args.closest_only,
            spread=args.spread,
            positions_override=positions_override,
        )
        max_iter = max((s.get("iteration", 0) for s in archive.get("successful", [])),
                       default=0)
        out = path.parent / args.out_name
        render(G, str(out), max_iter, physics=False)
        print(f"[{label}] wrote {out}  "
              f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print("\nAll conditions share one frozen UMAP space — positions are "
          "pixel-identical and directly comparable across the HTMLs.")


if __name__ == "__main__":
    main()
