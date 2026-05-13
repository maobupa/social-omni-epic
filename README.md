# social-omni-epic (Phase 0)

Open-ended social scenario generation with simulated learning. Adapts the
OMNI-EPIC loop to Sotopia-style social scenarios. Phase 0 runs generation only —
no actual agent execution — to show the generation pipeline produces an
ever-expanding, diverse space of scenarios.

## Setup

```bash
uv venv
uv pip install -r requirements.txt
```

### API key

Create a `.env` file in this project root containing:

```
OPENAI_API_KEY="sk-..."
```

`main.py` calls `dotenv.load_dotenv` on `social-omni-epic/.env` at startup.
The `.env` file is git-ignored.

### Seed data (Sotopia's scenarios)

The seed loader expects three JSONL files in `data/sotopia_seeds/`:

| file                          | rows  | purpose                                |
| ----------------------------- | ----- | -------------------------------------- |
| `environment_profiles.jsonl`  | ~884  | scenario text, agent_goals, source tag |
| `agent_profiles.jsonl`        | 40    | named characters with Big Five / MBTI  |
| `relationship_profiles.jsonl` | 120   | (optional, currently unused)           |

Plus, in `data/`, the episodes file which provides the env↔agent linkage:

| file                            | rows   | purpose                                                          |
| ------------------------------- | ------ | ---------------------------------------------------------------- |
| `sotopia_episodes_v1.jsonl`     | ~10k   | environment_id ↔ agent_ids pairings; defines the canonical 90    |

#### Obtaining the data

Sotopia ships its dataset as a Redis snapshot (`dump.rdb`), not as JSON. Our
extraction script downloads the snapshot and uses a short-lived Redis container
to convert it to JSONL. Requires Docker Desktop to be running.

```bash
.venv/bin/python scripts/extract_sotopia_seeds.py
```

This downloads `cmu-lti/sotopia-pi` from HuggingFace (256 MB), boots
`redis/redis-stack-server` with the RDB mounted, scans every redis-om
`JsonModel` key, and writes JSONL files. The container is removed at the end.

The episodes file (`sotopia_episodes_v1.jsonl`, ~180 MB) is a separate
HuggingFace download from `cmu-lti/sotopia`. Place it in `data/`.

Why sotopia-pi instead of sotopia? The original ICLR 2024 Sotopia dataset's Box
URL (listed in the sotopia repo's `published_datasets.json`) currently returns
HTTP 403. sotopia-pi is a superset (~884 envs) using the same schema; the seed
loader filters down to the canonical 90 by intersecting with the episodes file.

## Run

### Smoke test (5 iterations, ~30 sec, <$0.05)

```bash
.venv/bin/python main.py iterations=5 checkpoint_every=2 checkpoint_dir=output/smoke
```

### Full run (200 iterations, ~30 min, ~$4–5)

```bash
.venv/bin/python main.py
```

Uses defaults from `configs/social_omni_epic.yaml`: 200 iterations,
checkpoint every 25, output to `output/run_001`.

### Common overrides

Hydra lets you override any field with `key=value`:

```bash
# Different iteration count and output dir
.venv/bin/python main.py iterations=50 checkpoint_dir=output/quick

# Cheaper model
.venv/bin/python main.py model=gpt-4o-mini

# Use all 884 sotopia-pi seeds instead of the canonical 90
.venv/bin/python main.py restrict_to_episodes_v1=false

# Cap how many seeds get loaded (for fast experiments)
.venv/bin/python main.py seed_limit=20 iterations=10

# Adjust MoI behavior
.venv/bin/python main.py moi.num_examples=3 moi.min_archive_size=20

# Don't show the LLM the set of existing interaction_types
.venv/bin/python main.py task_generator.show_existing_types=false
```

### Example selection strategy

`task_generator.example_strategy` controls how the K archive entries shown in
each generation prompt are picked:

| value      | behavior                                                                                                              |
| ---------- | --------------------------------------------------------------------------------------------------------------------- |
| `knn`      | (default, OMNI-EPIC) pick one seed via `choose_probs`, then K-1 nearest neighbors. Examples form a tight local cluster. |
| `diverse`  | K independent weighted picks across the whole archive. Examples span multiple regions.                                |
| `farthest` | Greedy farthest-point: pick seed, then iteratively add the entry maximally distant from already-chosen examples.      |

Behaviorally:
- `knn` → depth-first exploration; LLM is asked to step just outside a tight cluster.
- `diverse` → breadth-first; LLM has to dodge K unrelated patterns at once, often landing in genuinely new regions.
- `farthest` → most aggressive diversity; maximum contrast between the examples shown.

```bash
.venv/bin/python main.py task_generator.example_strategy=knn       checkpoint_dir=output/knn
.venv/bin/python main.py task_generator.example_strategy=diverse   checkpoint_dir=output/diverse
.venv/bin/python main.py task_generator.example_strategy=farthest  checkpoint_dir=output/farthest
```

### Ablations

The three conditions described in the Phase 0 plan:

```bash
# Full system (default): archive + MoI
.venv/bin/python main.py checkpoint_dir=output/full

# No interestingness filter (archive-conditioned generation only)
.venv/bin/python main.py enable_moi=false checkpoint_dir=output/no_moi

# No archive conditioning at all (random/independent generation)
.venv/bin/python main.py use_archive=false enable_moi=false checkpoint_dir=output/no_archive
```

Strategies and ablations compose, so you can sweep them together:

```bash
for s in knn diverse farthest; do
  .venv/bin/python main.py task_generator.example_strategy=$s \
      iterations=200 checkpoint_dir=output/${s}_full
done
```

## Analysis

After a run completes:

```bash
RUN=output/run_001

# Numerical summary: counts by source, by interaction type, cell coverage
.venv/bin/python analysis/compute_metrics.py --archive $RUN/archive_latest.json

# Figures: UMAP scatter + diversity-over-time plot
.venv/bin/python analysis/visualize_embedding.py \
    --archive $RUN/archive_latest.json \
    --metrics $RUN/metrics.json \
    --out_dir $RUN

# Per-scenario audit: each generated scenario alongside the seeds it was built from
.venv/bin/python analysis/inspect_pairs.py --archive $RUN/archive_latest.json

# Restrict to one iteration
.venv/bin/python analysis/inspect_pairs.py --archive $RUN/archive_latest.json --iter 7

# Interactive HTML lineage graph (parent -> child edges, hover for details)
.venv/bin/python analysis/plot_lineage.py --archive $RUN/archive_latest.json --closest_only
.venv/bin/python analysis/plot_lineage.py --archive $RUN/archive_latest.json  # all edges

# More breathing room or vis.js spring physics
.venv/bin/python analysis/plot_lineage.py --archive $RUN/archive_latest.json --closest_only --spread 3.0
.venv/bin/python analysis/plot_lineage.py --archive $RUN/archive_latest.json --closest_only --physics

# Shared-UMAP across multiple runs so positions are visually comparable
.venv/bin/python analysis/plot_lineage.py \
    --archive output/200_full/archive_latest.json --closest_only --spread 2.5 \
    --shared_with output/200_no_moi/archive_latest.json,output/200_no_archive/archive_latest.json \
    --out output/200_full/lineage_closest_shared.html
# (run the same command three times, swapping which path is the --archive)

# Cell-occupancy heatmaps (all / seeds-only / generated-only)
.venv/bin/python analysis/plot_diversity_grid.py --archive $RUN/archive_latest.json --grid_size 10

# Cross-ablation comparison in a shared 2D space (run after all ablations finish)
.venv/bin/python analysis/compare_ablations.py \
    --archive full=output/200_full/archive_latest.json \
    --archive no_moi=output/200_no_moi/archive_latest.json \
    --archive no_archive=output/200_no_archive/archive_latest.json \
    --out_dir output/comparison

# Same comparison without size-normalization (shows the volume confound)
.venv/bin/python analysis/compare_ablations.py --no_equal_size \
    --archive full=output/200_full/archive_latest.json \
    --archive no_moi=output/200_no_moi/archive_latest.json \
    --archive no_archive=output/200_no_archive/archive_latest.json \
    --out_dir output/comparison_raw

# Numerical lineage metrics (chain depth, fanout gini, drift from seeds, ...)
.venv/bin/python analysis/lineage_stats.py \
    --archive full=output/200_full/archive_latest.json \
    --archive no_moi=output/200_no_moi/archive_latest.json \
    --archive no_archive=output/200_no_archive/archive_latest.json \
    --out_csv output/comparison/lineage_stats.csv
```

### Tool-by-tool flag reference

#### `analysis/visualize_embedding.py`
| flag             | default | meaning |
| ---------------- | ------- | ------- |
| `--archive`      | (req)   | Path to `archive_latest.json`. |
| `--metrics`      | none    | Path to `metrics.json` for the diversity-over-time plot. |
| `--out_dir`      | `.`     | Where to write the two PNGs. |
| `--top_k`        | `10`    | Show only the K most frequent interaction-type groups; the rest collapse to `"other"`. `0` = show all. |
| `--no_umbrella`  | off     | Disable substring-based umbrella grouping (use raw type strings). |

#### `analysis/plot_lineage.py`
| flag                | default     | meaning |
| ------------------- | ----------- | ------- |
| `--archive`         | (req)       | Path to `archive_latest.json`. |
| `--out`             | auto-named  | HTML output path. |
| `--closest_only`    | off         | Keep only the spatially-closest parent edge per node (cleaner). |
| `--include_failed`  | on          | Render `failed_interestingness` scenarios as red triangles. |
| `--inferred_k`      | `3`         | If `parent_example_ids` missing, infer parents via top-K nearest earlier entries. |
| `--spread`          | `2.0`       | Canvas-size multiplier; raise to 3-4 if nodes overlap with 300+ entries. |
| `--physics`         | off         | Turn on vis.js spring physics (UMAP positions become initial state — nodes will drift). |
| `--shared_with`     | none        | Comma-separated paths to other archive JSONs whose embeddings join the UMAP fit, so positions across multiple HTMLs are comparable. |

#### `analysis/plot_diversity_grid.py`
| flag           | default | meaning |
| -------------- | ------- | ------- |
| `--archive`    | (req)   | Path to `archive_latest.json`. |
| `--out_dir`    | archive dir | Where to write the heatmap PNGs. |
| `--grid_size`  | `10`    | 10 → 10×10 grid (100 cells). Use 20 for finer resolution. |
| `--max_count`  | `10`    | Cap the colorbar at this count; cells above show as `N+`. |

#### `analysis/compare_ablations.py`
| flag              | default | meaning |
| ----------------- | ------- | ------- |
| `--archive`       | (req, repeatable) | `name=path` pair; supply once per method. |
| `--out_dir`       | `output/comparison` | Where to write the comparison artifacts. |
| `--grid_size`     | `10`    | Grid resolution for `compare_grid.png`. |
| `--max_count`     | `10`    | Colorbar cap. |
| `--equal_size`    | on      | Truncate every archive to `min(size)` before computing coverage — controls for the "more scenarios = more cells filled" confound. |
| `--no_equal_size` | off     | Disable that truncation (use raw sizes). |

#### `analysis/lineage_stats.py`
| flag           | default | meaning |
| -------------- | ------- | ------- |
| `--archive`    | (req, repeatable) | `name=path` pair; supply once per method. |
| `--out_csv`    | none    | Also write the table to CSV. |
| `--inferred_k` | `3`     | k for the k-NN parent inference fallback when `parent_example_ids` is missing. |

#### `analysis/inspect_pairs.py`
| flag         | default | meaning |
| ------------ | ------- | ------- |
| `--archive`  | (req)   | Path to `archive_latest.json`. |
| `--k`        | `2`     | If parents weren't recorded, show this many nearest neighbors instead. |
| `--iter`     | none    | Restrict output to one iteration. |

#### `analysis/compute_metrics.py`
| flag         | default | meaning |
| ------------ | ------- | ------- |
| `--archive`  | (req)   | Path to `archive_latest.json`. |

## Output files

Every run writes to `checkpoint_dir` (default `output/run_001`):

| file                       | what it contains                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `archive_iter_N.json`      | Snapshot of the archive **as of iteration N**. Written every `checkpoint_every` iterations as a safety net — if the run crashes, you can resume from the most recent checkpoint instead of starting over. Same schema as `archive_latest.json`.                                                                                                                                                                                                                       |
| `archive_latest.json`      | The current archive state. Overwritten on every checkpoint. Schema: `{"successful": [SocialScenario...], "failed_generation": [{...raw FM output...}], "failed_interestingness": [SocialScenario...], "failed_tasks": []}`. Each `SocialScenario` has: `id`, `iteration` (−1 for seeds), `scenario`, `agent_profiles`, `agent_goals`, `relationship`, `tag`, `interaction_type`, `difficulty_tags`, `source`, `embedding` (1536-dim), `parent_example_ids`, `moi_reasoning`. |
| `metrics.json`             | Per-iteration time series. Each entry: `iteration`, `archive_size`, `cell_coverage`, `total_failed_gen`, `total_failed_interest`. Drives the `diversity_over_time.png` plot.                                                                                                                                                                                                                                                                                          |
| `main.log` (Hydra)         | Stdout/stderr captured by Hydra.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `hydra/`                   | Hydra's snapshot of the resolved config and overrides used for this run, so the run is exactly reproducible.                                                                                                                                                                                                                                                                                                                                                          |
| `scenario_space.png`       | (after running `visualize_embedding.py`) UMAP 2D projection. Left panel: seeds (gray ×) vs generated (colored by iteration). Right panel: same points colored by interaction type.                                                                                                                                                                                                                                                                                     |
| `diversity_over_time.png`  | (after running `visualize_embedding.py`) Blue solid line: cell coverage (left axis, ratio of occupied 20×20 PCA grid cells, 0–1). Red dashed: archive size (right axis, integer count).                                                                                                                                                                                                                                                                                |
| `lineage_closest.html` /<br>`lineage_all.html` | (after `plot_lineage.py`) Interactive pyvis network. Nodes = scenarios, positioned by UMAP, colored by iteration (gray = seed, viridis = generated, red triangle = MoI-rejected). Edges = parent → child from `parent_example_ids`. Closest variant keeps only the spatially-nearest parent edge per node for clarity; all variant draws every parent edge.                                                                                            |
| `diversity_grid_*.png`     | (after `plot_diversity_grid.py`) Cell-occupancy heatmap: 10×10 PCA grid, color = scenario count per cell. Three variants: `_all`, `_seeds`, `_generated`. All three share the same coordinate system (one PCA fit on the combined embeddings). Useful for the "expansion beyond seeds" story — the `_generated` heatmap should occupy different cells than `_seeds`.                                                              |
| `lineage_closest_shared.html` | (after `plot_lineage.py --shared_with`) Same as `lineage_closest.html` but positions come from a UMAP fit on the union of multiple archives, making node positions visually comparable across runs.                                                                                                                                                                                                                                  |
| `output/comparison/compare_grid.png`        | Three cell-occupancy heatmaps in the same PCA space, one per ablation. The headline cross-method figure.                                                                                                                                                                                                                                                                                                            |
| `output/comparison/compare_coverage.png`    | Bar chart of cell coverage per method. Paper-ready number.                                                                                                                                                                                                                                                                                                                                                          |
| `output/comparison/compare_diversity_curve.png` | Left panel: coverage-over-iteration (per-run PCA, so trajectories only, not absolute levels). Right panel: archive size over iteration. `no_moi` and `no_archive` coincide in the right panel by construction.                                                                                                                                                                                              |
| `output/comparison/compare_summary.json`    | Numbers behind those plots: `coverages`, `n_scenarios_used` (after equal-size truncation), `n_scenarios_raw`, `equal_size_mode`, `pca_explained_variance_ratio`.                                                                                                                                                                                                                                                  |
| `output/comparison/lineage_stats.csv`       | Per-method structural metrics: node/edge counts, weakly-connected-components, branching factor, chain depth, top-seed fanout, fanout Gini, distance from generated scenarios to nearest seed, distance from parent. Use the comparable metrics; treat metrics that depend on inferred-vs-recorded provenance with care (see "When metrics are comparable" note below).                                              |

### Why two archive files

`archive_iter_25.json` is a frozen snapshot — useful for "what did the archive
look like at iter 25?" comparisons. `archive_latest.json` always reflects the
most recent state; it's what analysis scripts read by default. Both are full
copies (no diffing). At checkpoint time, the same content is written to both.

### When metrics are comparable across ablations

Two pitfalls to keep in mind when comparing the three ablation runs:

1. **Volume confound.** `no_moi` and `no_archive` accept every generation, so
   they end with 290 archive entries; `full` ends with 241 (49 rejected by
   MoI). Cell coverage and any "fraction of space occupied" metric will favor
   the larger archive mechanically. `compare_ablations.py` truncates to
   `min(size)` by default (`--equal_size`); always report the size-controlled
   number, and only quote the raw number if you also report the size.

2. **Provenance source.** `full` and `no_moi` have `parent_example_ids` recorded
   at generation time. `no_archive` does not (the LLM wasn't shown any archive
   examples), so its lineage edges in `lineage_stats.py` are inferred via
   k-nearest-neighbor over earlier iterations. Metrics derived from edges —
   `mean_dist_to_parent`, `max_chain_depth`, `weakly_connected_components`,
   `top_seed_fanout` — are not apples-to-apples when one run is recorded and
   another is inferred. The non-edge-derived metrics
   (`mean_dist_to_nearest_seed`, cell coverage, counts) remain comparable.

## Ablation comparison

| Condition       | `use_archive` | `enable_moi` | expected behavior                                                  |
| --------------- | ------------- | ------------ | ------------------------------------------------------------------ |
| Full system     | true          | true         | High diversity, steady cell-coverage growth                        |
| No MoI          | true          | false        | Archive grows fast but coverage plateaus (near-duplicates accepted)|
| No archive      | false         | false        | Generation is unconditioned each iteration; LLM falls into patterns|

Comparing `metrics.json` and `scenario_space.png` across these three runs is
the core paper figure.

## Project layout

```
social-omni-epic/
├── configs/social_omni_epic.yaml   # Hydra defaults; all flags overridable from CLI
├── data/
│   ├── sotopia_seeds/              # extracted from dump.rdb (gitignored binary)
│   └── sotopia_episodes_v1.jsonl   # from HF cmu-lti/sotopia (gitignored)
├── social_omni_epic/               # the Python package
│   ├── data_models.py              # AgentProfile, SocialScenario (Pydantic, Sotopia-compatible)
│   ├── fm.py                       # OpenAI wrapper with retry/backoff
│   ├── task_generator.py           # archive-conditioned generation + example selection
│   ├── model_of_interestingness.py # MoI judge
│   ├── embedding_utils.py          # cosine retrieval + PCA cell-coverage metric
│   ├── validation.py               # schema checks for generated JSON
│   ├── archive.py                  # archive state + checkpointing
│   └── seeds.py                    # Sotopia → SocialScenario adapter
├── analysis/
│   ├── compute_metrics.py          # archive summary stats
│   ├── visualize_embedding.py      # UMAP + diversity-over-time plots
│   └── inspect_pairs.py            # per-scenario audit: generated + prompt examples
├── scripts/
│   └── extract_sotopia_seeds.py    # dump.rdb → JSONL via short-lived Docker
└── main.py                         # the Hydra-entry loop
```

## Phase 1 transition notes

Our `SocialScenario` uses Sotopia's exact field names. To run real agent
interactions, the next steps will be:

1. `pip install sotopia` and either start Redis or set `SOTOPIA_STORAGE_BACKEND=local`
2. Adapter: `SocialScenario → EnvironmentProfile + AgentProfile.save()` (~20-line function)
3. For each archived scenario, call `sotopia.run_async_server(...)` and collect `EpisodeLog`
4. Feed evaluation scores into the archive (replaces simulated-success in step 5 of the main loop)

No schema changes required.
