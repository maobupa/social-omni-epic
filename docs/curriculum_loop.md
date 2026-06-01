# The Curriculum Loop: Generating Scenarios at the Frontier

> How the archive grows. This is the control-flow / algorithm spec. It sits between
> [`scenario_design_sweet_spot.md`](scenario_design_sweet_spot.md) (what a *good* scenario is)
> and [`evaluation_methodology.md`](evaluation_methodology.md) (how we *measure*). It is the
> authoritative reference for the expansion loop and the stopping condition.

---

## 0. The idea, and what we borrow from OMNI-EPIC

OMNI-EPIC (Faldor et al., ICLR 2025) matches task difficulty to the agent's current ability:
when its RL agent **fails** to learn a task, the environment generator **edits the task to be
easier** (fewer obstacles, simpler reward), looping up to a cap; if still unlearned, the task
is archived as **failed** and a new one is generated. Its progress metric, **ANNECS /
ANNECS-OMNI**, counts environments that are *novel + appropriately-difficult + **solved***.

We borrow the **adaptive-difficulty editor**, but run it in the **opposite direction**:

> OMNI-EPIC's RL agents start **incompetent**, so it ratchets tasks *down* to them.
> Our LLM learner starts **competent**, so we ratchet tasks ***up*** to it.
> Same principle — keep the task at the **frontier** of the agent's current capability —
> opposite default direction.

Because "the agent" here is **the learner *with its current chronicle***, the difficulty bar
**auto-escalates** as the chronicle grows: scenarios must get harder to keep biting. That is
exactly OMNI-EPIC's frontier-matching, applied to a competent, self-improving agent.

---

## 1. Control flow

```
while not stop():                                  # stopping condition, §6
    anchor   = select_anchor(archive)              # weighted by learning value (§4)
    examples = retrieve_similar(anchor, archive)
    scenario = task_generator(anchor, examples)    # 3-part goal, region-aware (design doc §6, §6b)
    if not coherence_gate(scenario) or not moi_gate(scenario):
        continue                                   # cheap pre-filters: discard + regenerate
    learner  = designate_learner(scenario, anchor) # by anchor-perspective similarity

    # ---- Loop 1: difficulty calibration (counter D) — ratchet UP until it bites ----
    bit = False
    for d in range(D):
        result = run_episode(scenario, chronicle)  # attempt-1, learner WITH current chronicle
        if not solved(result):                     # chronicle-agent failed → BITE
            bit = True
            break
        feedback  = analyze_too_easy(result, scenario)   # which difficulty knob is slack?
        scenario  = task_editor(scenario, feedback)      # raise that knob — social knobs only (§3)
        if not coherence_gate(scenario):                 # edit broke plausibility/ZOPA
            break
    if not bit:
        discard(scenario)                          # could not manufacture bite → not archived
        continue

    # The failing attempt-1 above is reused as attempt 1 of the skill loop (not wasted).

    # ---- Loop 2: skill learning (counter K = max_attempts) — push toward outcome=2 ----
    solved_after_biting = False
    for k in range(K):
        reflect_and_edit_chronicle(result)         # reflection → adversarial → synthesis
        result = run_episode(scenario, chronicle)
        if solved(result):
            solved_after_biting = True
            break

    if solved_after_biting:
        archive.add_solved(scenario, chronicle)    # COUNTS toward stopping (§6)
    else:
        archive.add_failed(scenario)               # archived for conditioning, NOT counted (§6)
```

This **replaces** the old policy (`if outcome in (1,2): add_successful else add_failed`).
Crucially, **"solved on attempt 1" no longer enters the archive as a success** — it triggers
the difficulty editor (Loop 1) instead. The old ambiguous `outcome=1` disappears: a scenario
is either edited until it bites, or discarded.

---

## 2. The two loops, two counters

| | Loop 1 — difficulty calibration | Loop 2 — skill learning |
|---|---|---|
| Edits | the **scenario** (harder) | the **chronicle** (`skills.md`) |
| Counter | **D** (≈ 3) | **K** = `max_attempts` |
| Goal | make attempt-1 *fail* (manufacture bite) | make a later attempt *succeed* (`outcome=2`) |
| Exhausted → | **discard** (unbiteable, too easy) | archive as **failed** (frontier / too hard) |
| Editor | `task_editor` (raises social difficulty) | `reflection → adversarial → synthesis` |

They chain: Loop 1 ends on a *failed* attempt-1; that transcript is attempt 1 of Loop 2, so no
rollout is wasted.

---

## 3. The task editor (difficulty UP) — guardrails

When a scenario is too easy, `analyze_too_easy` diagnoses *which knob is slack* and
`task_editor` raises **only the social-difficulty knobs** (design doc §5):

- **Shortcut salience** — make the constraint-violating shortcut more tempting/available.
- **Constraint bite** — make taking the shortcut cost more.
- **Partner resistance** — make the partner harder to move / quicker to react to the shortcut.
- **Partner stake** — give the partner more to lose by genuinely conceding.

**Do NOT** make it harder by adding facts, parties, or numeric complexity — that slides back
into the logic-puzzle trap (design doc §1). After each edit, **re-run the coherence + ZOPA /
achievability gate** so ratcheting up does not tip into *impossible* (no social ZOPA). The edit
keeps the same lineage/anchor; it is a new *version* of the same scenario, not a new node.

---

## 4. Outcomes and archive policy

| Terminal state | Meaning | Archived? | Counts toward stopping? | Use |
|---|---|---|---|---|
| **Discarded** | too easy; D edits couldn't make it bite | no | no | — |
| **Solved-after-biting** (`outcome=2`) | bit the chronicle-agent, then reflection solved it | yes | **yes** | stepping stone; chronicle grew |
| **Failed** | bit, but K exhausted without success | yes | **no** | generator conditioning ("beyond current frontier") |

**Anchor weighting.** Solved-after-biting scenarios are high-value anchors. Failed scenarios
are kept for conditioning but **down-weighted** as anchors (don't keep spawning from the
frontier-edge). Discarded scenarios never enter the archive.

---

## 5. Why this dissolves the counterfactual problem

Earlier we faced: *"if the learner solved it **with** the inherited chronicle, was the scenario
too easy, or did the chronicle do the work?"* — which seemed to require a without-chronicle
counterfactual per scenario.

This loop removes that need **during the curriculum**. Loop 1 ratchets up until the
*chronicle-equipped* agent **fails attempt-1**. So every archived **solved-after-biting**
scenario is, *by construction*, one the chronicle could **not** already handle → the chronicle
was load-bearing and learning demonstrably happened. **No per-scenario counterfactual is
needed in the loop.**

The with-vs-without-chronicle counterfactual is still run for the **final external claim**
(eval doc §3) — but that is the held-out comparison, not the curriculum.

---

## 6. Stopping condition (ANNECS-style)

Following OMNI-EPIC's ANNECS, **progress = novel + appropriately-difficult + *solved*.** Failed
scenarios are *not* progress.

- **Counts toward the stopping condition:** **solved-after-biting** (`outcome=2`) scenarios
  only. By the loop's construction these are exactly the genuine *learning events*.
- **Does NOT count:** discarded-too-easy; archived-failed. (Failed scenarios still live in the
  archive as generator conditioning — exactly OMNI-EPIC's use of failed tasks — and as honest
  records, but they never tick the counter.)
- **Primary stop = dev held-out plateau** (eval doc §3d scaling curve): periodically measure
  performance on a *dev* held-out set (not sealed subset B) vs. archive size; stop when it
  flattens. `N` is then *where the curve plateaus*, not set a priori.
- **`N` solved-after-biting** serves as a **budget ceiling** / proxy when a full scaling curve
  is too costly.

---

## 7. Cost and the caps

This *manufactures* sweet-spot scenarios instead of generate-and-discard, but it is the main
compute driver: up to **D** difficulty rollouts + up to **K** skill rollouts per archived
scenario (OMNI-EPIC pays the analogous env-edit + RL-training cost). So:

- Keep **D small** (≈ 3): a few difficulty edits, then give up and discard.
- **K = `max_attempts`** (≈ 3, eval doc): enough for meta-reflection to be non-trivial.
- The cheap LLM pre-filters (coherence, MoI) run *before* any rollout to kill obvious junk for
  free.

---

## 8. Relationship to existing code & docs

- **Replaces** the `outcome ∈ {1,2} → add_successful` policy in the expansion loop
  (`run_phase2.py`): `outcome=1` now routes into Loop 1 (difficulty editor), not the archive.
- **New component**: `task_editor` (+ `analyze_too_easy`) — a difficulty-raising sibling of the
  existing `task_generator`, constrained to the social knobs (§3).
- **Reuses**: `designate_target_agent` (learner designation), coherence/MoI gates (pre-filters),
  the reflection/adversarial/synthesis/meta loop (Loop 2), `record_child` (lineage).
- **Design grounding**: scenario shape and difficulty knobs — `scenario_design_sweet_spot.md`
  (§5, §6, §6b). Success signal and stopping/eval — `evaluation_methodology.md` (§2, §3b–§3d).
