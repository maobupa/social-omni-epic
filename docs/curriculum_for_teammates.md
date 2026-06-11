# Social OMNI-EPIC Curriculum — Overview for Teammates

## What we're building and why

Social intelligence is hard to improve via ICL because you need *good examples* — examples where the task was hard enough to require genuine skill, but the agent demonstrably learned from trying. Random scenarios from SOTOPIA are mostly too easy: a polite LLM just... solves them on the first try, so there's nothing to learn from.

Our core insight: **generate scenarios at the difficulty frontier**. A frontier scenario is one where the learner fails the first time but improves after seeing what went wrong. Those are the scenarios worth learning from. The curriculum loop is a system that automatically finds and populates that frontier.

---

## The big picture: what the loop does

We start with 90 SOTOPIA seed scenarios and run a loop that generates 90 *new* scenarios descended from them. At the end, we have a bank of frontier scenarios — scenarios we know are learnable-but-not-trivial — plus a record of how an agent improved on each one. We then run ExpeL on those 90 generated scenarios to extract transferable insights, and use those insights as ICL for our evaluation.

The loop has three interlocking mechanisms:

1. **Curriculum generation** — given a parent scenario, mutate it into a new child scenario using an LLM
2. **Difficulty measurement** — run the learner K times on the child; use an LP judge to measure if it improved
3. **Adaptive selection** — use Thompson Sampling to choose which parent to generate from next, biasing toward parents that have produced learnable children

---

## Mechanism 1: Mutation operators

Every scenario has 7 "slots":
- **(a)** who the characters are and what the surface situation is
- **(b)** how obvious the partner's real concern is (hard to read = difficult)
- **(c)** how easily the learner's natural moves cause the partner to shut down
- **(d)** how much satisfying the partner costs the learner's own goal
- **(e)** the underlying psychological mechanism (face needs, reactance, etc.)
- **(f)** the power/information asymmetry between parties
- **(g)** the relationship type and relational stakes

Slots (b)(c)(d) are the **difficulty knobs**. Slots (a)(e)(f)(g) are the **content/structure** knobs.

Three operators pick which slots to change based on how the parent performed:

- **`escalate`** — parent was too easy (learner solved it first try). Tighten 1–2 difficulty knobs. The `analyze_too_easy()` function first diagnoses *which* knob was slack (e.g. "partner had no real resistance" = hardening_triggers_missing), then the mutation operator is told to fix exactly that. Preserve characters and premise.

- **`relax`** — parent was unsolvable (learner made zero progress across all K attempts). Read the "what went wrong" chronicle entries and loosen exactly the slot that made it impossible. Prefer loosening (c) or (d) first. Goal: make it hard but winnable.

- **`lateral`** — parent was at the frontier (the sweet spot). Hold difficulty constant — don't touch (b)(c)(d). Instead mutate 1–2 of (a)(e)(f)(g) to explore a different social dynamic at the same difficulty level.

The generation prompt shows the LLM: (1) the parent scenario it must mutate, (2) 3 KNN-similar archive examples as structural context labeled frontier/too-easy/dead-end, and (3) exact mutation instructions. The LLM generates 3 candidates at once; valid ones pass a diversity gate (discard if > 0.92 cosine similarity to any existing scenario).

---

## Mechanism 2: LP (Learning Progress) measurement

After generating a new scenario, we run the learner K=4 times on it.

- **Attempt 1**: No memory. Clean slate.
- **Attempts 2–K**: Memory injected from the previous failure (Reflexion string: what went wrong and what to try next). Stop early on success.

After K attempts, we compute the LP score: a pairwise judge compares attempt 1 against each later attempt in *both presentation orders* (A=1st,B=later and A=later,B=1st). This order-swap trick eliminates position bias — if the two judges disagree on which is better, we call it a tie. 

```
LP = improved_votes / total_votes
   = (votes where j clearly beat attempt 1, agreed in both orders) / (total votes cast)
```

**Classification**:
- **too_easy**: learner solved it on attempt 1 → LP not computed, pseudo-votes (0/6) charged
- **frontier**: solved OR LP > 0 (any judge pair agreed j was better)
- **beyond_frontier**: LP = 0.0 exactly, never solved

The LP score, together with the classification, drives the Thompson posterior update in Mechanism 3.

---

## Mechanism 3: Thompson Sampling anchor selection

The archive is all scenarios we've generated so far (plus the original 90 seeds). We want to pick which parent to mutate next. The goal: bias toward parents that are *productive* — they tend to produce children at the frontier, not trivial children or impossible ones.

Each scenario is modelled as a Bernoulli arm with a Beta posterior:

```
posterior ~ Beta(prior_α + alpha_votes, prior_β + beta_votes)
```

After each generated child:
- **alpha_votes** += improved LP votes from that child
- **beta_votes** += non-improved LP votes from that child
- Too_easy children contribute (0, 6) — a fixed penalty even if no LP was computed

At each iteration, we sample one value from each scenario's Beta distribution and pick the argmax. Scenarios with productive histories get sampled high more often; scenarios whose children were trivial or hopeless accumulate beta weight and get picked less.

New children inherit their parent's current posterior as their own starting prior ("warm start") — so a child of a productive parent starts with an optimistic prior rather than a flat Beta(1,1).

---

## The full iteration cycle

```
1. Thompson select → choose parent (anchor)
2. Classify parent → pick mutation operator
3. Generate 3 candidate children (LLM, temp=1.0)
4. Validate + diversity gate → pick best candidate
5. Run K-attempt loop on the chosen candidate
6. Compute LP → classify child
7. Update parent's Beta posterior with child's LP votes
8. Store child in archive (all classifications are selectable as future anchors)
9. Save checkpoint
```

---

## What `skills_final_md` is and why it matters

After the K-attempt loop, we store a text summary of what was learned (or what failed) in `scenario.skills_final_md`. This field is read by the task generator when building the next generation's prompt. Specifically, frontier and beyond-frontier examples shown in the generation prompt include this text under the header `[Skills chronicle — what made this scenario hard / what the agent learned:]`.

This means the mutation operators can be intelligent: when told to `relax` a beyond-frontier scenario, the LLM sees the chronicle's account of *why* it was unsolvable and can target the right slot.

Currently this is a structured XML chronicle format (Skills Chronicle). After the ExpeL refactor (see §6 of the tech spec), it will be ExpeL Reflexion strings — simpler but functionally equivalent for this purpose.

---

## ExpeL refactor: why and what changes

**Why**: The Skills Chronicle requires three separate LLM calls per attempt (reflect, adversarial check, meta-synthesis) plus a complex structured format. The main research contribution is the *curriculum generation* (Thompson + mutation), not the memory format. Replacing it with ExpeL Reflexion strings:
- Cuts ~3 LLM calls per attempt per scenario
- Removes ReflectionModule, AdversarialAgent, MetaReflectionModule from the loop
- Keeps LP computation, Thompson Sampling, and mutation operators completely intact
- The `skills_final_md` field still gets populated (with Reflexion strings instead of XML entries), so the task generator works without changes

**What changes** (only in `curriculum.py::run_episode_k_loop`):
- Replace `SkillsChronicle` + `reflection_mod.reflect()` with `_reflect()` from `expel_baseline.py`
- Replace `chronicle.format_for_prompt()` with `_format_reflections()`
- Remove adversarial checks and meta-synthesis calls
- Everything else (LP, classification, Thompson, archive, mutation) is untouched

---

## Three-condition evaluation design

After generating the 90 curriculum scenarios, we run ExpeL gather + extract on them to produce global insights. The three conditions on 150 held-out SOTOPIA-PI scenarios:

| Condition | Memory injected | Source |
|---|---|---|
| **Vanilla** | None | — |
| **ExpeL-Base90** | 14 global rules | From 90 SOTOPIA seeds (already done) |
| **ExpeL-Generated90** | global rules | From 90 curriculum-generated scenarios |

The prediction: ExpeL-Generated90 > ExpeL-Base90 > Vanilla, because frontier scenarios produce more non-obvious, harder-won insights than easy seeds.

---

## Key files

| File | Role |
|---|---|
| `scripts/run_curriculum.py` | Orchestrator: Thompson loop, batch management, checkpointing |
| `social_omni_epic/curriculum.py` | K-attempt episode loop + LP computation + classification |
| `social_omni_epic/archive.py` | Beta posterior state + Thompson select |
| `social_omni_epic/lp_judge.py` | Pairwise LP judge (order-swap bias collapse) |
| `social_omni_epic/task_generator.py` | Scenario mutation + prompt construction |
| `social_omni_epic/expel_baseline.py` | `_reflect()`, `_format_reflections()` for ExpeL refactor |
| `scripts/run_expel_chronicle.py` | ExpeL gather on a scenario bank (run post-curriculum) |
| `results/expel_phase0_Base90_ExpeL/` | Already-done ExpeL on Base90 seeds |
