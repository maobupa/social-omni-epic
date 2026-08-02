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
- **Partner key** — a collapsible block with the hidden ground truth driving the partner:
  mechanism, numbered movement conditions, hardening triggers, `surface_misdirection`
  (the partner's cover story) and `cost_coupling`.
- **Key-check verdict** — per attempt: PASS/FAIL plus `C1 ✓ met` / `T3 ⚡ tripped` chips
  (hover for the full text) and the judge's rationale. `solved` conjoins `key_check_passed`,
  so this is how you see why a GOAL-9 attempt still reads "not solved".
- **Insights** — the `🌐 Insights` button shows the run's global ExpeL rules (the same set is
  injected into every scenario — they are not scenario-specific).

## Review notes & checkmarks (shared between reviewers)

Pick who you are with the `as HX | HJ` toggle in the top bar (HX = Huanxing, HJ = Huijun).

- **📝 Notes** opens a side drawer with a free-text notepad for the selected scenario.
  Autosaves ~1s after you stop typing.
- **☐ Reviewed** stamps the scenario as checked by you, with a timestamp. Click again to
  un-check. The sidebar shows `✓HX` / `✓HJ` badges and a 📝 marker on scenarios with notes.

Everything lands in **`transcript_reader/review_notes.json`**, which **is tracked in git** —
commit it to hand your review off. Writes merge one scenario at a time and the file is written
with sorted keys, so two reviewers working on different scenarios produce clean, mergeable diffs.

Opening a scenario does **not** create a row — empty entries are pruned, and clearing your
notes + un-checking removes the entry again, so the file only ever contains real review state.

> Requires the server (no API key needed for notes). With it off, notes fall back to browser
> `localStorage` and are **not** shared — the badge in the drawer header turns 🔴 `local only`.

### Handing off

```bash
git add transcript_reader/review_notes.json && git commit -m "review notes" && git push
```

The other reviewer pulls, restarts their server, and reloads — notes and checkmarks appear.

## Controls

- `\` or `☰` collapses the sidebar for full-width transcript reading (remembered across reloads).
- Show/hide individual attempts (checkboxes in the top bar).
- `↑`/`↓` arrows move between scenarios · `Esc` closes the notes drawer.

## Regenerate

Re-run `uv run transcript_reader/build.py` after any new curriculum run. The output
`reader.html` is gitignored (large, regenerable); `build.py` is the tracked source.
