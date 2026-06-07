# Curriculum Generation — Quickstart

## What this does

Runs the two-loop curriculum on 90 SOTOPIA seeds, generating social scenarios and running episodes until scenarios are solved after biting (agent fails at least once, then succeeds). Saves results to `results/{run_name}/` and supports resume across manual batches.

---

## Setup

```bash
# From project root
cp .env.example .env          # add OPENAI_API_KEY or LIGHTNING_AI_API_KEY
poetry install
```

---

## Run

```bash
# Try 10 generation attempts
python scripts/run_curriculum.py run_name=run_001 iterations=10

# Continue from where you left off (auto-resumes from archive_latest.json)
python scripts/run_curriculum.py run_name=run_001 iterations=50

# Run until 90 solved-after-biting scenarios total
python scripts/run_curriculum.py run_name=run_001 stopping.N=90
```

`iterations` = number of generation attempts in this invocation, not number of successes.
`stopping.N` = stop early once this many solved-after-biting scenarios exist across all runs.

---

## Output

```
results/run_001/
  success/          ← solved-after-biting scenarios (used for eval retrieval)
    {id}.json
  failed/           ← bit but never solved across K attempts (task gen negative examples)
    {id}.json
  discarded/        ← too easy, never bit (analysis only)
    iter_N.json
  archive_latest.json   ← full archive state including Thompson priors; resume point
  archive_iter_N.json   ← periodic snapshots
  metrics.json          ← per-iteration log (terminal_state, goal, relationship, solved_count)
```

**Resume works by detecting `archive_latest.json`.** Thompson Sampling state (`n_i`, `n_solved`, `prior_alpha`, `prior_beta`) is fully preserved across runs. `solved_count` is derived from the number of files in `success/`.

---

## Terminal states

| State | Meaning | Saved to |
|---|---|---|
| `solved_after_biting` | Agent failed ≥1 attempt, then succeeded | `success/` |
| `failed` | Agent bit but never solved across K attempts | `failed/` |
| `discarded` | Scenario never bit (too easy); difficulty edits exhausted | `discarded/` |
| `generation_failed` | Scenario failed generation/coherence/diversity gate | nowhere |

---

## What each success JSON contains

Every file in `success/` is a full scenario with everything the eval script needs:

| Field | Used for |
|---|---|
| `embedding` | Cosine similarity search at eval time |
| `skills_final_md` | Chronicle injected into agent prompt as ICL |
| `scenario_title` | Structured embedding key (`social_dynamic \| perspective`) |
| `scenario`, `agent_goals` | Full scenario context |
| `goal_score` | Final SOTOPIA GOAL score from curriculum run |
| `prior_alpha`, `prior_beta` | Thompson prior (inherited from parent anchor) |

The `failed/` folder has the same structure. Eval retrieval uses `success/` only. The task generator uses both `success/` (positive examples) and `failed/` (negative examples — patterns that were too hard).

---

## Eval retrieval sketch

```python
import json
import numpy as np
from pathlib import Path

scenarios = [
    json.load(open(f))
    for f in Path("results/run_001/success").glob("*.json")
]
embeddings = np.array([s["embedding"] for s in scenarios])

# For a test scenario embedding, retrieve top-3 most similar chronicles
def retrieve_top_k(test_embedding, k=3):
    sims = embeddings @ test_embedding / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(test_embedding) + 1e-9
    )
    top_k = np.argsort(sims)[-k:][::-1]
    return [scenarios[i]["skills_final_md"] for i in top_k]
```

---

## Key config options

Edit `configs/social_omni_epic_curriculum.yaml` or pass as CLI overrides:

| Parameter | Default | Meaning |
|---|---|---|
| `run_name` | `run_001` | Output directory name under `results/` |
| `iterations` | `10` | Generation attempts this invocation |
| `batch_size` | `4` | Concurrent episodes per round (asyncio) |
| `stopping.N` | `null` | Stop when N solved-after-biting reached |
| `difficulty.D` | `2` | Max difficulty edits before discarding |
| `max_attempts` | `4` | K: max skill-loop episode attempts |
| `model` | `openai/gpt-5-mini` | LLM for generation and episodes |

---

## Monitoring progress

```bash
# Count solved so far
ls results/run_001/success | wc -l

# Tail metrics
python3 -c "
import json
log = json.load(open('results/run_001/metrics.json'))
for m in log[-10:]:
    print(m['iteration'], m['terminal_state'], 'solved='+str(m['solved_count']))
"
```
