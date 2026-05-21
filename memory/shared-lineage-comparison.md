---
name: shared-lineage-comparison
description: How to generate cross-condition comparison lineage HTMLs in one frozen UMAP space
metadata:
  type: project
---

For comparing conditions (e.g. 200_full vs 200_no_moi vs 200_no_archive), use
`analysis/plot_lineage_compare.py` (NOT `plot_lineage.py --shared_with`).

**Why:** `--shared_with` only adds peers to the UMAP fitting set but re-fits
UMAP per HTML on a differently-ordered union with a per-run rescale extent, so
the same scenario lands at different pixels per condition. Also each archive's
stored embeddings were computed during its own run, and `text-embedding-3-small`
is not perfectly deterministic — byte-identical seed texts (the 90 shared
Sotopia seeds) got slightly different vectors in ~40/90 cases.

**How to apply:** The compare script pools all scenarios across conditions,
dedupes by `to_text_for_embedding()` text (collapsing the 90 shared seeds to one
point each), re-embeds each unique text exactly once, fits UMAP ONCE -> one
global position map + shared extent, then renders one HTML per condition reusing
those exact coords. Run:
`.venv/bin/python analysis/plot_lineage_compare.py --condition full=output/200_full/archive_latest.json --condition no_moi=... --condition no_archive=...`
Needs OPENAI_API_KEY (auto-loaded from .env). Use `.venv/bin/python` (system
python lacks matplotlib/umap). Verified: 90/90 shared seeds pixel-identical.

Note: the 90 Sotopia seeds are content-identical across runs; agent<->env
pairing is deterministic from Sotopia's episode logs (`data/sotopia_episodes_v1.jsonl`),
NOT random and not our choice. Only ids differ (fresh UUID per load, cosmetic).
Related: [[lineage-html-features]]
