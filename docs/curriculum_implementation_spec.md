# Social OMNI-EPIC Curriculum — Technical Implementation Spec

**Status:** Current implementation as of 2026-06-11. Skills Chronicle paused; ExpeL refactor described in §6.

---

## 1. Entry point: `scripts/run_curriculum.py`

### Top-level flow

```
_seed_archive()
└── for each SOTOPIA seed: archive.add_successful(scn)    # seeds as initial anchors

while global_iter < stopping_N:
    batch_anchors = [archive.thompson_select() × batch_size]
    gather( _run_one_scenario(anchor_idx, ...) × batch_size )   # concurrent
    for result in batch:
        archive.record_child_outcome(...)
        archive.add_task(scenario) / add_failed_task(...)
        save checkpoint
```

---

## 2. `_run_one_scenario(anchor_idx, archive, svc, config)` — L117–315

This is the per-iteration workhorse. One call = one generated scenario run through the full loop.

### 2.1 Anchor retrieval

```python
anchor = archive.state.tasks[anchor_idx]
```

Anchor is the parent scenario the new scenario will mutate from.

### 2.2 Mutation operator selection (L181–186)

```python
if anchor.classification in ("too_easy", None):  mutation_op = "escalate"
elif anchor.classification == "beyond_frontier":  mutation_op = "relax"
else:                                             mutation_op = "lateral"
```

Seeds have `classification=None` → treated as too_easy → first generation escalates from seeds.

### 2.3 Example selection for the generation prompt (L142–160)

```python
all_embs = archive.get_successful_embeddings()
examples, ex_idxs = svc.task_gen.select_examples(archive, choose_probs, num_examples=3, strategy="knn")
ep_failed_candidates = [s for s in archive.state.failed_tasks if s.skills_final_md]
```

- **KNN strategy** (default): pick one seed via Thompson weights, then 2 nearest neighbors by embedding cosine similarity.
- Up to `num_failed_examples=1` failed-but-chronicled scenarios as negative signal.

### 2.4 Scenario generation (L190–222)

```python
candidates = svc.task_gen.generate_batch_from_archive(
    examples, anchor=anchor, mutation_operator=mutation_op,
    failed_examples=..., episode_failed_examples=...,
    existing_types=..., batch_size=config.batch_size
)
```

LLM generates `batch_size=3` candidates at `temperature=1.0`. Each candidate is validated via `validate_scenario()`. Returns list of valid `SocialScenario` objects.

**MOI (Measure of Interestingness) ranking**: candidates are ranked; top-1 selected. Implementation inside `generate_batch_from_archive` (via internal scoring).

**Coherence gate**: if `validate_scenario` fails, `edit_scenario(intent="fix_coherence")` is called up to `max_retries=3` times.

**Diversity gate** (L208–213): cosine similarity vs all existing archive embeddings. If max_sim > 0.92, discard.

### 2.5 Target agent designation (L215)

```python
scenario = svc.task_gen.designate_target_agent(scenario)
```

Sets `scenario.target_agent_idx` (0 = learner), `target_perspective`, `target_agent_goal_abstract`.

### 2.6 K-attempt episode loop

```python
scenario, classification, outcome, final_scores, loop_info = await run_episode_k_loop(
    scenario=scenario, anchor=anchor, fm=svc.fm,
    reflection_mod=svc.reflection_mod, meta_mod=svc.meta_mod,
    adversarial=svc.adversarial, config=config
)
```

See §3 for full breakdown.

### 2.7 Archive update (L493–529)

```python
lp_improved = info.get("lp_improved_votes", 0)
lp_total    = info.get("lp_votes", 0)

if terminal_state == "too_easy":
    archive.record_child_outcome(anchor_idx, 0, K_VOTES_EQUIV)   # penalise anchor
elif terminal_state in ("frontier", "beyond_frontier"):
    archive.record_child_outcome(anchor_idx, lp_improved, max(lp_total, 1))
```

`K_VOTES_EQUIV = 6` — synthetic vote charge for too_easy children.

---

## 3. `curriculum.py::run_episode_k_loop()` — K-attempt learning loop

### 3.1 Setup

```python
K = config.get("k_attempts", 4)
max_entries = config.get("chronicle_max_entries", 8)

# Inherit parent's chronicle as starting knowledge
current_chronicle = SkillsChronicle.from_markdown(anchor.skills_final_md or "") if anchor else SkillsChronicle()
query_emb = fm.embed(scenario.target_agent_goal_abstract or "")
```

### 3.2 Attempt 1

```python
result = await _episode(scenario, current_chronicle)
```

**`_episode(scn, chronicle)`** runs:
1. `chronicle.format_for_prompt(query_emb, max_entries)` → `memory_prompt` string
2. `run_episode(scn, fm, memory_prompt=memory_prompt)` → `EpisodeResult(transcript, scores, terminal_success)`

**Too-easy fast path** (L172–200): if `result.terminal_success` on attempt 1:
- `analyze_too_easy(scenario, transcript1)` → `{slack_knob, rationale}` stored on scenario
- `lp_value = 0.0`, `lp_votes = K_VOTES_EQUIV` (pseudo-votes)
- `classification = "too_easy"`, `outcome_int = 1`
- Chronicle passed through unchanged (no reflection — nothing learned)
- **Returns immediately**

### 3.3 Attempts 2..K

For each attempt `k` in `2..K`:

**a) Reflection** (updates chronicle):
```python
ref_out = reflection_mod.reflect(
    chronicle=current_chronicle, scenario=scenario,
    transcripts=all_transcripts, edit_reasons=all_edit_reasons
)
```
`ReflectionOutput(updated_chronicle, edit_reasons)` — per-attempt edits to the SkillsChronicle.

**b) Adversarial check**:
```python
check = adversarial.check_reflection(
    chronicle=current_chronicle, updated=ref_out.updated_chronicle,
    edit_reasons=ref_out.edit_reasons, scenario=scenario,
    transcripts=all_transcripts
)
```
Five checks: evidence citation, condition abstraction, guidance abstraction, broadening limit, misdirection.

If `not check.approved` and `re_reflect=True`:
```python
ref_out = reflection_mod.synthesize_with_critique(
    chronicle=current_chronicle, critique=check.critique, ...
)
```

**c) Episode with updated chronicle**:
```python
current_chronicle = ref_out.updated_chronicle
result = await _episode(scenario, current_chronicle)
if result.terminal_success: solved = True; break
```

### 3.4 LP computation

```python
lp_result = await compute_lp(fm_judge, scenario, all_transcripts, learner_goal, relational_stakes)
```

`compute_lp()` in `lp_judge.py`:
- Pairs: `(attempt_1, attempt_j)` for `j = 2..K`
- Each pair voted twice (order swap): `"A=1,B=j"` and `"A=j,B=1"`
- Vote normalised to `"A" | "B" | "no_difference"`
- `pair_improved_votes = 2` if both agree j better; `= 0` if disagree or both agree j worse/tied
- `lp_value = sum(improved_votes) / sum(total_votes)`

Stored in `loop_info`:
```python
loop_info["lp_value"]          = lp_result.lp_value
loop_info["lp_votes"]          = lp_result.total_votes
loop_info["lp_improved_votes"] = lp_result.improved_votes
```

### 3.5 Classification

```python
if solved or lp_result.lp_value > 0:
    classification = "frontier"     # outcome_int = 2 (solved) or 3 (unsolved)
else:
    classification = "beyond_frontier"   # outcome_int = 4
```

Note: `lp_value > 0` means at least ONE judge comparison agreed j was better than attempt 1 in both presentation orders.

### 3.6 Meta-reflection synthesis

```python
final_chronicle = meta_mod.synthesize(
    chronicle_versions=all_versions, transcripts=all_transcripts,
    edit_reasons=all_edit_reasons, scenario=scenario, outcome=outcome,
    lp_votes=lp_result.votes, anchor=anchor
)
```

`meta_reflection.py` produces a single coherent `skills_final_md`:
- **Outcome 2** (solved): HEURISTIC-dominant, WARNINGs as contrast, max 3 entries
- **Outcome 3** (unsolved): WARNING-dominant, ≤1 untested hypothesis, max 3 entries

Followed by `adversarial.check_final()` (10 checks). Retry once with critique on failure.

```python
scenario.skills_final_md = final_chronicle.to_markdown()
```

---

## 4. `archive.py` — Thompson Sampling posterior

```
posterior ~ Beta(prior_alpha + alpha_votes, prior_beta + beta_votes)
```

| Event | alpha_votes change | beta_votes change |
|---|---|---|
| frontier child (lp=4/6) | +4 | +2 |
| beyond_frontier child (lp=0/6) | +0 | +6 |
| too_easy child | +0 | +K_VOTES_EQUIV (=6) |

**Selection**: `argmax(np.random.beta(α, β) for each task)`

**Warm start**: new child inherits `(prior_alpha, prior_beta) = parent's current posterior` via `child_prior_from_parent()`.

---

## 5. `task_generator.py` — Scenario mutation

### Mutation operators (`_EDIT_INTENTS`)

| Operator | When | What slots | Goal |
|---|---|---|---|
| `escalate` | anchor was too_easy | tighten 1–2 of (b)(c)(d) | harder, still solvable |
| `relax` | anchor was beyond_frontier | loosen (c) or (d) first | hard but winnable |
| `lateral` | anchor was frontier | mutate 1–2 of (a)(e)(f)(g) | same difficulty, new dynamic |

### Slot vocabulary
- **(a)** premise + characters
- **(b)** surface_misdirection discoverability
- **(c)** hardening_triggers congruence
- **(d)** cost_coupling cost
- **(e)** key_mechanism (from MECHANISM_LIBRARY)
- **(f)** power/information asymmetry
- **(g)** relationship type & stakes

### Generation prompt structure

```
SYSTEM: scenario designer persona + goal format guide + profile guide + mechanism library + mutation operator block

USER:
  === PARENT SCENARIO (mutate THIS one) ===
  {parent JSON + skills chronicle + classification}

  === RELATED ARCHIVE EXAMPLES (context only) ===
  FRONTIER EXEMPLARS — [3 KNN similar, with chronicles]
  TOO EASY — [if any in KNN pool]
  STRUCTURAL DEAD ENDS — [if any in KNN pool + episode_failed]

  Generate {batch_size} candidates as JSON array.
```

### `analyze_too_easy()` (called before `escalate`)

Diagnoses which knob is slack from attempt-1 transcript:
```python
{"slack_knob": "cooperative_alignment | surface_misdirection_too_obvious | 
                hardening_triggers_missing | cost_coupling_too_low | ...",
 "rationale": "one sentence"}
```
Stored in `scenario.too_easy_diagnosis` and shown in the parent block of the next generation prompt.

---

## 6. ExpeL Refactor Plan (Skills Chronicle → Reflexion)

**Motivation**: Main contribution is the generated curriculum. Skills Chronicle is complex, slow, and currently paused. Replace with ExpeL Reflexion strings — simpler, faster, same role.

### What changes in `curriculum.py::run_episode_k_loop()`

**Remove**:
- `SkillsChronicle` import and object management
- `reflection_mod.reflect()` call
- `adversarial.check_reflection()` call
- `meta_mod.synthesize()` + `adversarial.check_final()` call
- `validate_synthesis()` call
- `all_versions` list tracking
- `ReflectionModule`, `MetaReflectionModule`, `AdversarialAgent` parameters

**Replace with**:
```python
from social_omni_epic.expel_baseline import _reflect, _format_reflections

# In run_episode_k_loop():
reflexion_strings: list[str] = []

# Attempt 1 — no memory
result = await _episode(scenario, memory_prompt="")
if result.terminal_success: ...  # too_easy fast-path unchanged

# Attempts 2..K
for attempt in range(2, K + 1):
    memory_prompt = _format_reflections(reflexion_strings)
    result = await _episode(scenario, memory_prompt=memory_prompt)
    if result.terminal_success: solved = True; break
    if attempt < K:
        reflexion = _reflect(fm, task=scenario.scenario,
                             learner_goal=scenario.agent_goals[0],
                             transcript_text=format_transcript(all_transcripts[-1]),
                             goal_score=result.learner_scores.get("goal", 0.0))
        reflexion_strings.append(reflexion)

# Store as skills_final_md (task generator reads this field)
scenario.skills_final_md = _format_reflections(reflexion_strings)
loop_info["final_chronicle_md"] = scenario.skills_final_md
```

**What stays unchanged**:
- LP computation (`compute_lp()`)
- Classification logic
- Thompson Sampling
- `archive.record_child_outcome()`
- Task generator's `_format_scenario_for_prompt()` — it reads `skills_final_md` as a string; Reflexion strings work fine there
- All mutation operators

### After curriculum generation: run ExpeL gather + extract

Once 90 scenarios are generated (success/ folder), run `run_expel_chronicle.py` on them exactly as was done for Base90:

```bash
python scripts/run_expel_chronicle.py \
    --seeds results/<curriculum_run>/success/ \
    --run-name expel_generated90
```

This produces `insights.json` (global ExpeL rules) for the ICL evaluation baseline.

### What the refactor removes from `run_curriculum.py`

```python
# Remove these service initializations:
reflection_mod = ReflectionModule(fm)       # remove
meta_mod = MetaReflectionModule(fm)         # remove
adversarial = AdversarialAgent(fm)          # remove

# Remove from _run_one_scenario args and from curriculum call
```

`run_curriculum.py` no longer needs to instantiate or pass `reflection_mod`, `meta_mod`, `adversarial` to `run_episode_k_loop`.

---

## 7. Data flow summary

```
archive (90 seeds seeded)
    │
    ▼
thompson_select() → anchor_idx
    │
    ▼
generate_batch_from_archive(anchor, mutation_op, examples, failed)
    │  [LLM @ temp=1.0, batch_size=3]
    ▼
validate + coherence-fix + diversity-gate → scenario
    │
    ▼
run_episode_k_loop(scenario, anchor)
    ├── attempt 1: no memory
    │     └── too_easy? → return immediately
    ├── attempt 2..K: Reflexion memory [after ExpeL refactor]
    │     └── _reflect() after each failure
    ├── compute_lp() → lp_value, lp_votes, lp_improved_votes
    ├── classify: frontier / beyond_frontier
    └── scenario.skills_final_md = _format_reflections(reflexion_strings)
    │
    ▼
archive.record_child_outcome(anchor_idx, lp_improved, lp_total)
archive.add_task(scenario)   [all classifications go in archive as selectable anchors]
    │
    ▼
save success/NNN.json or failed/NNN.json
```

---

## 8. Key config parameters

| Config key | Default | Effect |
|---|---|---|
| `k_attempts` | 4 | Max attempts per scenario in K-loop |
| `batch_size` | 3 | Candidates generated per LLM call |
| `stopping_N` | 90 | Total scenarios to generate |
| `chronicle_max_entries` | 8 | Max chronicle entries injected into learner prompt |
| `similarity_threshold` | 0.92 | Diversity gate cosine threshold |
| `num_examples` | 3 | Archive examples shown in generation prompt |

---

## 9. Output structure

```
results/<run_name>/
├── success/          # frontier scenarios (solved or lp > 0)
│   └── {id}.json
├── failed/           # beyond_frontier scenarios (lp == 0, never solved)
│   └── {id}.json
├── discarded/        # too_easy or generation failures
│   └── iter_{N}.json
├── archive_latest.json   # full archive state + Thompson posteriors (resume point)
├── archive_iter_{N}.json # periodic checkpoint snapshots
└── metrics.json          # per-iteration: lp_value, classification, goal_score, archive_size
```
