"""Interactive HTML lineage graph of the scenario archive.

Each accepted scenario is a node. Edges go from prompt examples to the scenario
they helped generate (read from `parent_example_ids`). Failed-interestingness
scenarios are also rendered as dim nodes so dead branches are visible.

Adapted from omni-epic/analysis/visualize_taskgen.py:
- Node positions: UMAP of the embedding (omni-epic used t-SNE)
- Node color: gray for seeds, viridis-by-iteration for generated, red for failed
- Edges: parent_example_ids -> child; --closest_only collapses to one parent per node
- Hover text shows the scenario, agent_goals, and the MoI judge's reasoning

For runs predating the provenance fix, falls back to inferred edges (k-nearest
neighbor in embedding space, restricted to entries from earlier iterations).
"""
import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network


def load_archive(path: str) -> dict:
    return json.load(open(path))


def build_position_map(scenarios: list[dict], spread: float = 1.0,
                       extra_embeddings: Optional[list[list[float]]] = None,
                       shared_extent: Optional[tuple[float, float, float, float]] = None,
                       ) -> tuple[dict[str, tuple[float, float]], tuple[float, float, float, float]]:
    """UMAP-project embeddings to 2D, scale to a wide canvas.

    `spread` multiplies the canvas extent: 1.0 = 3200x2400, 2.0 = 6400x4800.
    `extra_embeddings` (optional) are added to the UMAP fitting set but get no
    nodes — used when multiple archives share one coordinate space.
    `shared_extent` (optional) forces the same (xmin, xmax, ymin, ymax)
    rescaling across multiple runs so their nodes land at comparable positions.

    Returns (positions, extent).
    """
    embs, ids = [], []
    for s in scenarios:
        if s.get("embedding"):
            embs.append(s["embedding"])
            ids.append(s["id"])
    if len(embs) < 3:
        return {}, (0., 0., 0., 0.)

    from umap import UMAP
    fit_input = np.array(embs + (extra_embeddings or []))
    reducer = UMAP(n_components=2, random_state=42, min_dist=0.5, n_neighbors=15)
    xy_all = reducer.fit_transform(fit_input)
    xy = xy_all[: len(embs)]

    if shared_extent is None:
        xmin, xmax = xy_all[:, 0].min(), xy_all[:, 0].max()
        ymin, ymax = xy_all[:, 1].min(), xy_all[:, 1].max()
        extent = (float(xmin), float(xmax), float(ymin), float(ymax))
    else:
        xmin, xmax, ymin, ymax = shared_extent
        extent = shared_extent

    width = 3200 * spread
    height = 2400 * spread
    xy[:, 0] = (xy[:, 0] - xmin) / max(xmax - xmin, 1e-9) * width - width / 2
    xy[:, 1] = (xy[:, 1] - ymin) / max(ymax - ymin, 1e-9) * height - height / 2
    return ({sid: (float(p[0]), float(p[1])) for sid, p in zip(ids, xy)}, extent)


def color_for(s: dict, max_iter: int, role: str) -> str:
    if role == "seed":
        return "#bdbdbd"
    if role == "failed_int":
        return "#e57373"  # soft red for "rejected as uninteresting"
    # generated -> viridis by iteration
    cmap = plt.get_cmap("viridis").reversed()
    it = max(s.get("iteration", 0), 0)
    rgba = cmap(it / max(max_iter, 1))
    return mcolors.to_hex(rgba)


def hover_html(s: dict) -> str:
    title = []
    title.append(f"<b>iter {s.get('iteration', '?')}</b>  "
                 f"<i>{s.get('source', '')}</i>")
    title.append(f"<b>type:</b> {s.get('interaction_type', '')}  "
                 f"<b>tag:</b> {s.get('tag', '')}")
    scn = (s.get("scenario") or "").strip()
    if len(scn) > 400:
        scn = scn[:400] + "..."
    title.append(f"<b>scenario:</b> {scn}")
    title.append(f"<b>relationship:</b> {s.get('relationship', '')[:120]}")
    for i, g in enumerate(s.get("agent_goals", [])):
        title.append(f"<b>goal {i+1}:</b> {g[:160]}")
    if s.get("moi_reasoning"):
        title.append(f"<b>MoI:</b> {s['moi_reasoning'][:300]}")
    return "<br>".join(title)


def closest_parent(positions: dict[str, tuple[float, float]],
                   child_id: str, parent_ids: list[str]) -> Optional[str]:
    if child_id not in positions:
        return None
    cx, cy = positions[child_id]
    best, best_d = None, float("inf")
    for pid in parent_ids:
        if pid not in positions:
            continue
        px, py = positions[pid]
        d = (px - cx) ** 2 + (py - cy) ** 2
        if d < best_d:
            best, best_d = pid, d
    return best


def infer_parents_via_knn(s: dict, by_id: dict[str, dict],
                          embedding_arr: np.ndarray, id_index: dict[str, int],
                          k: int = 3) -> list[str]:
    """For pre-provenance archives: pick K nearest entries from earlier iterations."""
    if s["id"] not in id_index or s.get("embedding") is None:
        return []
    own_idx = id_index[s["id"]]
    own_iter = s.get("iteration", -1)
    eligible = [(i, sid) for sid, i in id_index.items()
                if i != own_idx and by_id[sid].get("iteration", -1) < own_iter]
    if not eligible:
        return []
    target = embedding_arr[own_idx]
    dists = []
    for i, sid in eligible:
        v = embedding_arr[i]
        # cosine distance
        d = 1 - float(np.dot(target, v) / (np.linalg.norm(target) * np.linalg.norm(v) + 1e-9))
        dists.append((d, sid))
    dists.sort()
    return [sid for _, sid in dists[:k]]


def build_graph(archive: dict, include_failed: bool, closest_only: bool,
                inferred_k: int = 3, spread: float = 2.0,
                extra_embeddings: Optional[list[list[float]]] = None,
                shared_extent: Optional[tuple[float, float, float, float]] = None,
                ) -> tuple[nx.DiGraph, tuple[float, float, float, float]]:
    successful = archive.get("successful", [])
    failed_int = archive.get("failed_interestingness", []) if include_failed else []
    all_nodes = successful + failed_int
    by_id = {s["id"]: s for s in all_nodes}

    # If using inferred edges, build embedding lookup table
    ids_with_emb = [s["id"] for s in all_nodes if s.get("embedding")]
    id_index = {sid: i for i, sid in enumerate(ids_with_emb)}
    embedding_arr = np.array([by_id[sid]["embedding"] for sid in ids_with_emb]) \
        if ids_with_emb else np.zeros((0, 0))

    positions, extent = build_position_map(
        all_nodes, spread=spread,
        extra_embeddings=extra_embeddings,
        shared_extent=shared_extent,
    )
    max_iter = max((s.get("iteration", 0) for s in successful), default=0)

    G = nx.DiGraph()
    for s in successful:
        role = "seed" if s.get("iteration", -1) < 0 else "generated"
        if s["id"] not in positions:
            continue
        G.add_node(s["id"], role=role, scenario=s, color=color_for(s, max_iter, role),
                   title=hover_html(s), pos=positions[s["id"]])
    if include_failed:
        for s in failed_int:
            if s["id"] not in positions:
                continue
            G.add_node(s["id"], role="failed_int", scenario=s,
                       color=color_for(s, max_iter, "failed_int"),
                       title=hover_html(s), pos=positions[s["id"]])

    # Edges
    for s in all_nodes:
        if s["id"] not in G.nodes:
            continue
        parent_ids = s.get("parent_example_ids") or []
        if not parent_ids and s.get("iteration", -1) >= 0:
            # Pre-provenance fallback: infer via K-NN in earlier iterations
            parent_ids = infer_parents_via_knn(
                s, by_id, embedding_arr, id_index, k=inferred_k
            )
        parent_ids = [p for p in parent_ids if p in G.nodes]
        if closest_only and parent_ids:
            cp = closest_parent(positions, s["id"], parent_ids)
            if cp:
                G.add_edge(cp, s["id"])
        else:
            for p in parent_ids:
                G.add_edge(p, s["id"])
    return G, extent


def render(G: nx.DiGraph, output_path: str, max_iter: int,
           physics: bool = False) -> None:
    nt = Network(height="100vh", width="100%", directed=True,
                 notebook=False, cdn_resources="remote",
                 bgcolor="#fafafa", font_color="#222222")

    # vis.js options: physics off (positions come from UMAP), but allow user
    # to drag nodes. Hover highlighting + zoom-to-fit on load are below.
    nt.set_options("""
    {
      "physics": { "enabled": %s, "stabilization": { "iterations": 80 } },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": { "enabled": true },
        "tooltipDelay": 100,
        "zoomView": true
      },
      "nodes": {
        "borderWidth": 1,
        "borderWidthSelected": 3,
        "scaling": { "min": 6, "max": 30 }
      },
      "edges": {
        "smooth": { "type": "continuous", "roundness": 0.2 },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
        "color": { "inherit": false, "opacity": 0.55 }
      }
    }
    """ % ("true" if physics else "false"))

    for nid, attr in G.nodes(data=True):
        s = attr["scenario"]
        label = "seed" if attr["role"] == "seed" else f"i{s.get('iteration', '?')}"
        nt.add_node(
            nid, label=label, title=attr["title"], color=attr["color"],
            x=attr["pos"][0], y=attr["pos"][1],
            size=10 if attr["role"] == "seed" else (8 if attr["role"] == "failed_int" else 14),
            shape="dot" if attr["role"] != "failed_int" else "triangle",
        )
    for u, v in G.edges():
        nt.add_edge(u, v, color="#888888", width=0.8)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nt.write_html(output_path, notebook=False, open_browser=False)

    # --- Inject a richer overlay: legend, controls help, search box, click-to-highlight ---
    early_color = mcolors.to_hex(plt.get_cmap("viridis").reversed()(0.05))
    late_color = mcolors.to_hex(plt.get_cmap("viridis").reversed()(0.95))
    overlay = f"""
<style>
  #ui {{
    position: fixed; top: 12px; right: 12px; max-width: 320px;
    background: rgba(255,255,255,0.96); border: 1px solid #ccc;
    border-radius: 6px; padding: 12px; font-family: -apple-system, sans-serif;
    font-size: 12px; line-height: 1.45; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    z-index: 9999;
  }}
  #ui h4 {{ margin: 0 0 6px 0; }}
  #ui details {{ margin-top: 8px; }}
  #ui summary {{ cursor: pointer; user-select: none; font-weight: 600; }}
  #ui input {{ width: 100%; box-sizing: border-box; padding: 4px 6px;
    font-size: 12px; border: 1px solid #aaa; border-radius: 3px; }}
  .sw {{ display:inline-block; width:12px; height:12px; margin-right:6px;
    vertical-align: middle; border-radius: 50%; }}
  .tri {{ display:inline-block; width:0; height:0; vertical-align: middle;
    border-left:6px solid transparent; border-right:6px solid transparent;
    border-bottom:10px solid #e57373; margin-right:6px; }}
</style>
<div id="ui">
  <h4>Lineage</h4>
  <div><span class="sw" style="background:#bdbdbd"></span>seed (90)</div>
  <div><span class="sw" style="background:{early_color}"></span>early iter</div>
  <div><span class="sw" style="background:{late_color}"></span>late iter (max {max_iter})</div>
  <div><span class="tri"></span>MoI-rejected</div>

  <details open>
    <summary>How to read positions</summary>
    Layout is UMAP of scenario embeddings — nearby nodes are
    semantically similar. The seed cluster (gray) sits where it sits;
    bright colors drifting away from it = open-ended expansion.
  </details>

  <details>
    <summary>Controls</summary>
    Scroll = zoom. Drag empty area = pan. Drag a node = move it.<br>
    Click a node = highlight its parents &amp; children.<br>
    Hover a node = scenario preview.<br>
    Arrow keys / on-screen buttons also work.
  </details>

  <details>
    <summary>Search</summary>
    <input id="searchBox" placeholder="text in scenario / goal / type" />
    <div id="searchResults" style="margin-top:6px; max-height:140px; overflow-y:auto;"></div>
  </details>

  <details>
    <summary>Tips</summary>
    • Click a bright (late-iter) node to trace its provenance back to seeds.<br>
    • Many red triangles in one region = MoI persistently rejecting a pattern.<br>
    • A gray seed with many outgoing edges = a popular template the LLM keeps building from.
  </details>
</div>

<script>
  // wait for vis.js network to be ready
  function onReady(cb) {{
    if (typeof network !== 'undefined' && typeof nodes !== 'undefined') cb();
    else setTimeout(() => onReady(cb), 100);
  }}
  onReady(function() {{
    // Zoom to fit all nodes after load
    network.once('afterDrawing', function() {{ network.fit({{animation: false}}); }});

    // Click-to-highlight: dim everything except the clicked node + its 1-hop neighbors
    var allNodeIds = nodes.getIds();
    var allEdgeIds = edges.getIds();
    network.on('click', function(params) {{
      if (params.nodes.length === 0) {{
        // background click: restore
        nodes.update(allNodeIds.map(id => ({{id: id, opacity: 1, font: {{color: '#222'}}}})));
        edges.update(allEdgeIds.map(id => ({{id: id, color: {{opacity: 0.55}}}})));
        return;
      }}
      var sel = params.nodes[0];
      var kin = new Set([sel]);
      network.getConnectedNodes(sel).forEach(n => kin.add(n));
      nodes.update(allNodeIds.map(id => ({{id: id,
        opacity: kin.has(id) ? 1.0 : 0.15,
        font: {{color: kin.has(id) ? '#222' : '#bbb'}}
      }})));
      var connectedEdges = new Set(network.getConnectedEdges(sel));
      edges.update(allEdgeIds.map(id => ({{id: id,
        color: {{opacity: connectedEdges.has(id) ? 0.95 : 0.05}}
      }})));
    }});

    // Search: filter+jump to matching nodes
    document.getElementById('searchBox').addEventListener('input', function(e) {{
      var q = e.target.value.toLowerCase().trim();
      var results = document.getElementById('searchResults');
      if (!q) {{ results.innerHTML = ''; return; }}
      var matches = nodes.get().filter(function(n) {{
        return (n.title || '').toLowerCase().includes(q);
      }}).slice(0, 25);
      results.innerHTML = matches.map(function(n) {{
        return '<div style="cursor:pointer; padding:2px 0; border-bottom:1px solid #eee;" '
             + 'onclick="network.focus(\\'' + n.id + '\\', {{scale:1.5, animation:true}});'
             + 'network.selectNodes([\\'' + n.id + '\\']);">'
             + n.label + ' &mdash; ' + (n.title.match(/<b>scenario:<\\/b>([^<]+)/)
                 ? n.title.match(/<b>scenario:<\\/b>([^<]+)/)[1].slice(0,60) : '...')
             + '</div>';
      }}).join('');
    }});
  }});
</script>
"""
    with open(output_path, "a") as f:
        f.write(overlay)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--out", default=None,
                    help="HTML output path (default: alongside archive)")
    ap.add_argument("--closest_only", action="store_true",
                    help="Draw only the spatially-closest parent edge per node (cleaner)")
    ap.add_argument("--include_failed", action="store_true", default=True,
                    help="Show failed_interestingness nodes (default true)")
    ap.add_argument("--inferred_k", type=int, default=3,
                    help="When parent_example_ids missing, use top-K nearest earlier entries as edges")
    ap.add_argument("--spread", type=float, default=2.0,
                    help="Canvas scale multiplier — bump up if nodes overlap (try 3 or 4 for >300 nodes)")
    ap.add_argument("--physics", action="store_true",
                    help="Enable vis.js physics simulation (UMAP positions used as starting state)")
    ap.add_argument("--shared_with", default=None,
                    help="Comma-separated paths to OTHER archive JSONs. Their embeddings join "
                         "the UMAP fit, so positions in this HTML are comparable to those in "
                         "HTMLs generated for those other archives (call this flag on each).")
    args = ap.parse_args()

    archive = load_archive(args.archive)

    extra_embeddings: Optional[list[list[float]]] = None
    if args.shared_with:
        # Sort peers so the union is order-deterministic regardless of CLI ordering.
        peer_paths = sorted({p.strip() for p in args.shared_with.split(",") if p.strip()})
        extra_embeddings = []
        for op in peer_paths:
            other = load_archive(op)
            for s in other.get("successful", []) + other.get("failed_interestingness", []):
                if s.get("embedding"):
                    extra_embeddings.append(s["embedding"])
        print(f"Shared-UMAP mode: focal archive + {len(extra_embeddings)} extra peer embeddings "
              f"from {len(peer_paths)} archive(s) join the fit.")

    G, _ = build_graph(
        archive,
        include_failed=args.include_failed,
        closest_only=args.closest_only,
        inferred_k=args.inferred_k,
        spread=args.spread,
        extra_embeddings=extra_embeddings,
    )
    max_iter = max((s.get("iteration", 0) for s in archive["successful"]), default=0)
    out = args.out or str(Path(args.archive).parent /
                          (f"lineage_{'closest' if args.closest_only else 'all'}.html"))
    render(G, out, max_iter, physics=args.physics)
    print(f"Wrote {out}")
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
