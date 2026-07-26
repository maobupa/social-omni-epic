# Gen-90 Transcript Reader

A self-contained HTML reader for inspecting the Gen-90 curriculum transcripts by hand —
list/select scenarios, see their category, and read the retry attempts side-by-side.

## Build

From the **repo root**:

```bash
uv run transcript_reader/build.py
```

This reads the per-scenario JSONs from each curriculum run and writes a single offline
file: `transcript_reader/reader.html`.

## Open

```bash
open transcript_reader/reader.html
```

Double-click works too — it's fully self-contained (no server, no internet).

## What it shows

- **Run toggle** — switch between `gpt-5-mini` (`results/gen90_expel`) and
  `gpt-4.1-mini` (`results/gen90_expel_41mini`) in-page.
- **Sidebar** — scenarios grouped and color-coded by category, filterable by text and category.
- **Categories** (derived from `classification` + `terminal_success`):
  - **Too easy** — solved on attempt 1
  - **Frontier · solved** — showed learning progress, eventually solved after retries
  - **Frontier · improved, not solved** — LP > 0 but never crossed the success bar
  - **Beyond frontier** — no learning progress (LP = 0)
- **Main pane** — the selected scenario's retry attempts as **parallel columns**: each column
  is one attempt's transcript (learner vs partner color-coded), with its score chips, a
  solved/not badge, and the **reflexion** written for the next attempt. A goal-per-attempt
  sparkline summarizes the trajectory.
- **Chronicle & insights** — the `📖 Chronicle` button opens a slide-over drawer with the
  scenario's reflexion chronicle and the run's global ExpeL insight rules.

## Controls

- Collapse the sidebar (`☰`) for full-width reading on a small screen.
- Show/hide individual attempts (checkboxes in the top bar).
- `↑`/`↓` arrows move between scenarios.

## Regenerate

Re-run `uv run transcript_reader/build.py` after any new curriculum run. The output
`reader.html` is gitignored (large, regenerable); `build.py` is the tracked source.
