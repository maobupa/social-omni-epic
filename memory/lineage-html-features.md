---
name: lineage-html-features
description: Interactive features baked into generated lineage HTMLs (node size, detail popups)
metadata:
  type: project
---

`analysis/plot_lineage.py` `render()` generates lineage HTMLs with:
- Enlarged nodes for easier clicking (generated=30, seed=22, MoI-rejected=18).
- Click a node -> draggable, scrollable detail popup showing full scenario,
  relationship, BOTH agent profiles (all fields), agent goals, MoI reasoning.
  Click same node or the X to close; multiple popups can coexist for comparison.
  Data injected as `window.SCN_DETAILS`; content set via textContent (no HTML
  injection). Existing parent/child highlight + hover preview preserved.

Any regeneration via `plot_lineage.py` or `plot_lineage_compare.py` carries
these automatically. Related: [[shared-lineage-comparison]]
