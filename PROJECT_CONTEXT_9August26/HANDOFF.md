# Social OMNI-EPIC — Handoff & Redesign Plan

**Date:** 2026-08-09, implementation status updated 2026-08-10 · **Authors:** HX (Huanxing), with HJ (Huijun) · **Status:** design built on branch `schema-v2-matrix` — see IMPLEMENTATION STATUS below

**Who this is for:** a collaborator (and their coding agent) joining now. It assumes no prior
knowledge of the codebase. Read §1–§5 for what we're doing and why; §6–§11 for the design;
§12–§14 for what to actually run.

**IMPLEMENTATION STATUS (2026-08-10).** This document is the *design*; it is now built. Branch
`schema-v2-matrix`, four commits. What is verified, and by what:

| commit | what | verified by |
|---|---|---|
| `7622735` | roster probe · `generation_cell.py` extraction · FM split · **schema v2 (`internal_state`)** | a 2-iteration `run_curriculum` run is behaviour-identical; **90/90 frozen gen-90 scenarios still parse and `summary.json` rebuilds with zero diffs** (`scripts/check_schema_compat.py`) |
| `a6d7d5f` | **oracle-solvability gate** · per-turn partner verifier | **2/2 on ground truth**: `57ed171e` (winnable, ordinary learners scored 10/9/10/10) admitted on try 1 at goal=10.0; `e7179e01` (partner defers off-scene) rejected 3/3 |
| `66b9154` | paired-grid driver · crossplay · matrix builder | `--dry-run` gives escalate 63 / lateral 22 / relax 5 for gpt-5-mini, exactly its bands — the calibration claim audited for zero episodes; matrix builder reproduces the predicted pattern on synthetic data |
| `dc421d9` | end-to-end ramp (`RAMP=smoke\|pilot\|full`) | dry run walks all five stages; smoke picks one seed per band, pilot picks all 5 `beyond_frontier` across 6 sources, smoke ⊂ pilot |

**Five things changed from this design once contact was made with the code.** Each is recorded in
the relevant commit message; they are listed here because they are the parts a reader would
otherwise trust incorrectly:

1. **The partner is `gpt-5.4-mini`, not Anthropic.** Probing showed Lightning returns *"credit
   balance is too low"* for `anthropic/claude-opus-4-8` and *"failed to find the model"* for
   `openai/gpt-5.4`. So the three-lab story is off, and the strong models are reachable only
   direct. Lightning now carries the judge and nothing else.
2. **`surface_misdirection` was being fed to the partner as the truth.** For v1 records with no
   `internal_state`, both the oracle's cheat sheet and the partner's own prompt substituted that
   field — which is *the decoy*. Combined with the new "getting what you asked for does not count"
   rule, the partner was holding out for its own cover story. That would have made every v1 record
   look unwinnable, and would have poisoned any v2 scenario with a thin `internal_state`.
3. **The grading rule stayed staged, per §4.1, and it was the right call** — but the *reason* is
   sharper than written here: we now log a `verdicts_disagree` flag per attempt, so the
   staged-vs-state-only question is answered by data rather than argument.
4. **`a156533b` is probably broken, not hard.** Satisfying its conditions yields "Sam feels the
   reciprocity is real", not "Sam hands over the keepsake" — nothing in the key bridges those. It
   is one of the four scenarios §3.2 calls *genuinely* hard, so the 76% artifact estimate is likely
   optimistic rather than harsh.
5. **The existing gpt-5-mini phase-0 cannot be reused as Row 0.** It ran with
   `partner_model = gpt-5-mini` (self-play), not the frozen partner, so its bands are not
   comparable with another learner's — and that mismatch would sit inside the saturation claim.
   Both learners get a fresh partner-matched phase-0; the old run is used only to choose *which*
   seeds to include.

**Relationship to other docs:** `docs/PROJECT_CANON.md` is the frozen truth about the *old*
system (Gen-90). It is still accurate as a description of what was built and run. **This
document supersedes its forward-looking parts (§7, §9) and its framing.** Where they conflict,
this wins.

---

## 1. The one-paragraph version

We built a system that automatically generates hard social scenarios for a language model, by
measuring where that specific model starts failing rather than guessing. We ran it, produced 90
scenarios, and analysed them hard. **The machinery works; the scenarios are mostly broken.** About
76% of the "hard" scenarios are unwinnable by construction rather than genuinely difficult, and
74% of the scenarios we recorded as "solved" were solved on an attempt where the simulated partner
broke character. The downstream claim — that experience from these scenarios transfers to new
tasks — came out null. So we are keeping the core idea (**don't define social difficulty, measure
it**), fixing the three things that broke it, and **changing what we're claiming**: from *"this is
a curriculum that improves a model"* to *"this is a way to build social benchmarks that don't
saturate as models get stronger."*

**The key move: ship the *procedure*, not the *items*.** We don't publish 90 scenarios — those were
calibrated to one model and can't compare others. We publish a *method* that, given any model,
produces that model's frontier set. The reported quantity is **where the frontier sits**, not a
score on a fixed list. The new headline experiment is a **model × scenario-set matrix** that shows
the frontier moving as models get stronger.

---

## 2. Background: what the old system did

Skip to §3 if you don't need the mechanics. This is here so the diagnosis in §4 makes sense.

**The loop** (`scripts/run_curriculum.py`):

1. **Archive** holds 90 human-written SOTOPIA seed scenarios plus every scenario we generate.
2. **Pick a parent** from the archive (Thompson sampling over "did this parent's children turn out
   useful?").
3. **Generate a child** with an LLM — a fresh scenario in the parent's structural family, plus a
   hidden `partner_key` (see below).
4. **Gates:** coherence, no key leakage, surface novelty, embedding diversity, a "worth" ranker.
5. **Play it**: the learner model tries the scenario, up to 4 attempts. After each failure it
   writes a reflection that the next attempt sees (this is Reflexion / ExpeL — in-context memory,
   no weight updates).
6. **Score it**: a judge assigns SOTOPIA's 7 dimensions; a second judge checks the partner key.
7. **Classify**: `too_easy` (solved on attempt 1) · `frontier` (failed cold, improved across
   attempts) · `beyond_frontier` (failed cold, never improved).
8. **Update** the parent's score and add the child to the archive. Repeat to 90 scenarios.

**The `partner_key`** is the central invention. Every generated scenario gives the partner a hidden
specification of what moves them:

| field | meaning |
|---|---|
| `key_mechanism` | one of 5 psychology tags (see glossary) |
| `movement_conditions` | 1–3 things the learner can *do* that make the partner genuinely shift |
| `hardening_triggers` | 1–3 learner moves that make the partner dig in |
| `surface_misdirection` | the partner's *stated* objection — their cover story |
| `cost_coupling` | what satisfying the conditions costs the learner's own goal |

Rationale: if the partner is just "a stubborn person," difficulty is a cliff and success is luck.
If the partner has a *specified* lock, the scenario is hard **and** guaranteed solvable.

**The learner's goal** is a triple: `outcome` (what to achieve) / `constraint` (what not to
violate) / `shortcut` (a tempting move that gets the outcome but breaks the constraint).

**What counts as solved:** `goal ≥ 7 ∧ relationship ≥ 0 ∧ judge_goal_achieved ∧ key_check_passed`.

**The run we analysed:** `results/gen90_expel`. Generator, learner, and partner all
`openai/gpt-5-mini`; judge `google/gemini-3-flash-preview`. 90 scenarios, 300 attempts.

---

## 3. What we found (all numbers verified against the artifacts)

### 3.1 The headline result held

Generated scenarios are much harder than raw SOTOPIA seeds: 14% / 44% / 41%
(too_easy / frontier / beyond) versus 70% / 24% / 6% for the seeds. χ²=63, p<1e-9. Frontier
fraction rose from 38% in the first half of the run to 51% in the second.

**But this is a claim about labels the pipeline assigns to itself**, and the next three findings
are about whether those labels mean anything.

### 3.2 Most of the difficulty is broken scenarios, not hard ones

Close-read of all 90 (`docs/gen90_bank_diagnostic.md`):

| band | n | genuinely hard | artifact |
|---|---|---|---|
| `too_easy` | 13 | 12 | 1 |
| `frontier` | 40 | 20 | 20 |
| `beyond_frontier` | 37 | **4** | **33 (89%)** |

Three ways they break: **unreachable target** (the buyer's goal price is below the seller's stated
floor), **off-scene decider** (the partner defers to someone not in the conversation), and
**goal-achieved-but-key-gated** (the learner got exactly what the goal asked, sometimes on all four
attempts, but was marked failed for not performing the specified ritual).

Independent replication with a different method (`docs/gen90_vs_sotopia_failure_comparison.md`):
of 63 first-try failures, **48 (76%) were unwinnable by any agent**, 15 (24%) were real model
mistakes. Strip the broken ones and gen90's genuine failure rate is ~17%, versus ~21% for raw
SOTOPIA. **The scenarios are not actually harder — they just fail more.**

Only about **12 of 90** scenarios test a real, distinct social skill.

> ⚠️ **Known error in that doc:** its Method section claims the partner is not key-conditioned at
> simulation time. That is wrong — `curriculum.py:176` passes `partner_key` into the episode and
> `episode_runner.py:528` swaps in the key-conditioned partner prompt. Its §5 conclusion probably
> survives, but its attribution doesn't. Fix before citing.

### 3.3 Most "solves" happened when the partner broke character

Post-hoc audit over all 300 attempts (`results/gen90_expel/analysis/partner_fidelity_audit.json`):

| failure | rate |
|---|---|
| partner **leaked** a hidden condition or trigger | 39/300 (13.0%) |
| partner **yielded early** (softened before any condition was met) | 100/300 (33.3%) |
| partner **ignored** a trigger it should have hardened on | 26/300 (8.7%) |

**25 of the 34 solved scenarios (74%) were solved on an attempt with one of these failures.** The
leak rate is 7.5× higher on `too_easy` (46%) than on `beyond_frontier` (6%) — exactly the signature
you'd expect if partner infidelity is *manufacturing* the easy solves.

### 3.4 The transfer claim came out null

Held-out evaluation, 150 SOTOPIA-π scenarios:

| condition | success | Δ vs vanilla (on the 25 vanilla-failed) |
|---|---|---|
| Vanilla (no memory) | 83.3% | — |
| Random-90 (random SOTOPIA scenarios) | 85.3% | **+40.0** |
| ExpeL-90 (raw seeds) | 81.3% | +32.0 |
| **SOE (our bank)** | 80.5% | **+40.0** |

Our curriculum ties a random baseline. The paper's explanation (ceiling effect; the held-out set is
drawn from Random-90's home distribution) is fair, but it isn't evidence *for* us.

Also worth knowing: the extracted insights read like generic deference — *"acknowledge and reflect
their reasons, ask permission, offer at most one reversible option, never push."* That is already
the base model's failure mode (100/100 early exits were learner-initiated, 81% "loop-and-leave").
We may have been distilling the disposition we should be correcting.

### 3.5 Selection drove the bank toward bargaining

New finding, computed 2026-08-08. Children per seed family, normalised by seeds contributed:

| seed family | seeds | children | ratio |
|---|---|---|---|
| **deal-or-no-deal** | 10 | 22 | **2.20×** |
| hand-craft | 6 | 10 | 1.67× |
| normbank | 9 | 12 | 1.33× |
| craigslist_bargains | 10 | 11 | 1.10× |
| social_iqa | 15 | 14 | 0.93× |
| mutual_friends | 10 | 8 | 0.80× |
| persuasion_for_good | 10 | 5 | 0.50× |
| **social_chemistry** | 20 | 8 | **0.40×** |

χ²=28.3, df=7, p=1.9e-4. (Caveat: children aren't independent — lineages run to depth 6 — so
effective n < 90. Only 47 of the 90 root seeds were ever used at all.)

The most game-like family was amplified 2.2×; the most emotionally rich family was suppressed 2.5×.
**Mechanism:** learning progress can only reward a scenario where "better" is legible to a pairwise
judge, and that means a scalar, verifiable outcome. Comfort someone slightly better on attempt 3
and the judge says `no_difference`. We built a selection signal blind to the axis we care about and
then ran 90 rounds of selection on it.

This is the quantitative answer to *"why does every scenario turn into a negotiation?"*

---

### 3.6 A coherence check that never ran

`coherence_check.py` check #9 (COOPERATIVE ALIGNMENT — *"can the learner win by simply accepting
whatever the partner naturally offers?"*) gates on `competing_interest` and
`partner_default_position`. Both are declared in `data_models.py:135-136`, and **the generator never
emits them** — neither appears in `_SCENARIO_SCHEMA`. Verified: **0/90 scenarios have either field
set.** The check ends *"if neither field is present, skip this check entirely."*

**The `EASY_COOP` detector existed and never fired once**, which directly explains 7 of the 13
`too_easy` scenarios (the partner secretly wanting the learner's outcome all along). Replacement in
§7.9 — deliberately built on two mandatory fields so it cannot go dead the same way.

---

## 4. Diagnosis — five root causes

Everything above traces to five design decisions. Each has a fix, in §6–§9.

### 4.1 The witness became the answer key

Think of **Countdown** (the arithmetic puzzle used in reasoning work): given numbers, hit a target.
It's generated *from* a known solution — pick an expression, compute the target, hide the
expression. That guarantees the puzzle is solvable. But **if the generator's expression was
`(6×4)÷(3−2)` and you answer `8×3`, you are still correct** — the checker is the arithmetic, not
the generator's expression. The hidden solution is a **witness that a path exists**, not the
required path.

`partner_key` was meant to be that witness. But `key_check` grades against the *specific*
`movement_conditions`, so it became an **answer key**. Any effective-but-unlisted social move is a
false negative. That is the entire `ARTIFACT_C` bucket — 9 scenarios where the goal was achieved
and the label said "too hard."

*(Caveat on the analogy: Countdown's checker is exact and free, so generating from a witness really
does guarantee solvability. Ours is an LLM reading a transcript, and the partner is an LLM that may
not honour its own key. So generating from a witness only guarantees a path exists *in the fiction*
— which is why §8.1 exists.)*

### 4.2 We only implemented half the minimal criterion

**POET** (Wang et al. 2019) evolves walking robots and terrains together. Its load-bearing rule is
the **minimal criterion**: a new terrain enters the population only if some current agent scores
*above a floor and below a ceiling*. **MCC** (Brant & Stanley 2017) does the same for mazes with no
fitness function at all — just a birth filter.

Both verify the "not impossible" side **empirically, by running an agent.** We test only "the
learner failed cold." For "not impossible" we *asserted* that the key guarantees solvability. §3.2
is the receipt.

> POET won't let a level into the population until somebody can already do it. We let levels in on
> a promise.

There's a second difference that matters as much: POET/MCC filter **at birth** — a failing item
never enters the population. We filter **at credit** — the item enters the archive, counts toward
N=90, stays selectable, and ships in the artifact we call "the contribution." All 37
`beyond_frontier` scenarios are in the bank.

### 4.3 The goal triple forces negotiation

`outcome / constraint / shortcut` requires a *verifiable state change in the partner*, which
requires a negotiable object, which is a negotiation. The generation prompt reinforces it
explicitly — *"structurally incompatible but with a ZONE OF POSSIBLE AGREEMENT"* is a bargaining
primitive.

So the monotony is **not** caused by the 5-mechanism library. Adding mechanisms won't fix it. The
goal grammar is the cause; §7.8 is the fix — and it has two parts, because the prompt both
misdefines `outcome` *and* mandates goal opposition.

### 4.4 "Learning progress" isn't learning progress

Classical LP (Oudeyer & Kaplan; Schmidhuber) is the derivative of competence over **learning** —
parameters change, so improvement means the *agent* got better, and that improvement generalises.

Ours measures improvement across attempts where **nothing about the agent changes**; only the
context changed. That's **in-context recoverability**. The old paper concedes the distinction in
one clause and then proceeds as if it's cosmetic. It isn't: classical LP predicts transfer by
construction, ours has no mechanism that would make it transfer — and empirically it didn't (§3.4).

It's still a *useful* quantity. It's just a different one, and it needs a different name and a
different job (§9.2).

### 4.5 The partner was instructed but never verified

The key sits in the partner's prompt as an instruction. **Nothing ever checks whether the partner
followed it.** Compliance was hoped for, and it failed 13–33% of the time.

This is not our bug alone. Roleplay-Doh (Louie et al. 2024 — in `Paper references/`) measured the
same thing: **20% of GPT-4 responses violated their expert-defined principles**, with the named
sub-mode *"misapplying situational principles… responding with hesitancy when someone gives
encouraging words, even when the conditions for their use were not met."* Principle-conditioned
roleplay drifts. Their fix — and it should be ours — is a **verifier**, not a better prompt.

A related design error made it worse: the partner prompt says *"you physically cannot tell the
other person what would change your mind… NEVER acknowledge that there is anything else to know."*
That doesn't describe a hard person, it describes a **pathological** one. RLHF-tuned models drift
back toward realism, which is exactly what the leak rate measures. §7.5 fixes this with a one-word
repoint of the template.

---

## 5. The new direction: from curriculum to a benchmark that doesn't saturate

### 5.1 Curriculum vs. benchmark

| | curriculum | benchmark |
|---|---|---|
| what's valuable | the improved learner; scenarios are disposable | the scenarios themselves |
| model-relativity | **must** be learner-specific | **must** be comparable across models |
| budget policy | **concentrate** where learning happens | **spread** for coverage and comparability |
| what kills it | learner doesn't improve | saturation, contamination, invalid items |

These pull in opposite directions, which is why "publish the gen-90 bank as a benchmark" doesn't
work — it was calibrated to `gpt-5-mini`.

**The resolution: ship the *procedure*, not the *items*.** You don't publish 90 scenarios. You
publish a method that, given any model, produces that model's frontier set. The reported quantity
is **where the frontier sits**, not a score on fixed items.

### 5.1b Why the evolutionary search goes away

This is the biggest structural change and it deserves its own explanation, because "we removed the
search" sounds like we removed the contribution.

The old system had **two** selection pressures:

1. **Thompson sampling over parents** — pick the scenario whose children have been most productive,
   mutate from it. A *search*: some lineages get explored deeply, others never at all.
2. **Band-conditioned few-shot steering** — show the generator "this was too easy / just right / too
   hard" examples so it calibrates to the learner.

**Pressure 2 is what makes scenarios learner-relative. Pressure 1 is what makes the bank a search.**
Only pressure 1 goes.

Three reasons:

- **A benchmark needs coverage, not concentration** (the table above). Thompson is a concentrator by
  design — that's what makes it good for a curriculum. In the run it produced 2.2× as many
  deal-or-no-deal descendants as their seed share and 0.4× as many social_chemistry, and **used only
  47 of the 90 root seeds** (§3.5). For a benchmark, that's not efficiency, it's a sampling defect:
  the resulting item set measures bargaining and calls it social intelligence.
- **It was never shown to help.** The random-anchor ablation was specified in
  `docs/post_run_experiments.md` §3 and never run. We have no evidence Thompson beat uniform.
- **The grid removes it mechanically.** One child per seed per calibration target is a *paired grid*.
  There is no anchor pool to sample from, so there is nothing for Thompson to select.

Independent support for the direction: **Persona Generators** (Paglieri et al., DeepMind 2026 —
`Paper references/Persona Generator.pdf`) argues explicitly for **coverage over density matching**
when generating synthetic populations for stress-testing, on the grounds that *"it is the outliers,
not the average user, that drive critical failures."* Optimising for yield collapses coverage; that
is exactly what our χ²=28.3 seed-family skew measures.

**What we keep:** pressure 2, plus the mutation *operators* (escalate / relax / lateral) as
direction-setters inside the few-shot framing. Thompson stays in the codebase behind
`anchor_selection` as the curriculum-mode option, and we run the random ablation once (§12.1) so we
can state honestly whether it ever did anything. Both outcomes are publishable.

### 5.2 Why anyone cares

Benchmark lifespans are collapsing — MNIST lasted decades, GLUE about a year, SuperGLUE months.
Saturation is now the field's standing problem, and static social benchmarks are already there:
`gpt-5-mini` passes 78.6% of raw SOTOPIA on the first try.

The existing answer is **adaptive benchmarking**:

- **ANLI** (Nie et al. 2020) — human annotators write examples that fool the current best model;
  retrain; repeat for three rounds. The benchmark stays hard because it's built *against* the model.
- **Dynabench** (Kiela et al. 2021) — a platform generalising this. Its thesis: *benchmarking as a
  moving target*; saturation is a design flaw, not an inevitability.
- **Model-written evals** (Perez et al. 2022) — LLMs generate the eval items, humans validate.

**Their shared bottleneck is the human adversary.** ANLI-style collection is slow and expensive,
which is why the paradigm never scaled. Our pitch: replace the human adversary with a generator
plus a *measured* frontier plus a validity gate. Human effort moves from authoring every item to
validating a calibration procedure once.

Our delta over Perez et al. specifically: they generate items; we generate items **calibrated to a
specific model's frontier** and **validity-gated**.

### 5.3 The known critique of this genre — and why our gate is the answer

Adversarial/dynamic benchmarks have a well-known failure mode: they drift into artifacts and end up
measuring *weird* rather than *hard*. Bowman & Dahl (2021), *"What Will it Take to Fix Benchmarking
in NLU?"*, makes this case directly, and later analyses found ANLI items that are ambiguous or
mislabelled rather than genuinely difficult.

**That is precisely our 76%-broken problem, named by the field before we hit it.** So the
oracle-solvability gate (§8.1) isn't just an internal fix — it's our answer to the standing
critique of the genre we're joining. That's a strong position: *we are the first adaptive
benchmark with a validity gate, and we can report the artifact rate with and without it.*

### 5.4 What makes a good benchmark (the bar we're aiming at)

- Humans do okay, or struggle a bit.
- Models are bad at it.
- Stronger models are measurably better — **monotonicity**.
- What's measured is concrete and verifiable.

Where we stand: monotonicity is what the matrix (§10) tests. "Models are bad at it" requires the
validity filter first. "Verifiable" is our weakest leg, since both partner and judge are LLMs —
which is why the human slice in §12.3 is not optional. **We have never had a single human play one
of these scenarios.** That is the largest missing datum in the project and among the cheapest to
get.

---

## 6. The construct: what we mean by social intelligence

Working definition:

> **(a) infer what the partner actually wants or feels, and (b) choose an action that serves both
> your goal and their state.**

### 6.1 Why this is defensible, not cherry-picked

It re-derives the field's oldest and most durable decomposition:

- **Thorndike (1920)** defined social intelligence as *"the ability to understand and manage men and
  women"* — understanding + acting, at the origin.
- **Ford & Tisak (1983)**, **Marlowe (1986)**, Sternberg's practical intelligence all recover a
  *cognitive* vs *behavioural* split.
- (a) is **Theory of Mind / mentalising** (Premack & Woodruff; Apperly; Sap et al. 2022 on LLM ToM;
  Ullman 2023) and *cognitive empathy* (Davis's IRI treats Perspective Taking as its own subscale).
- (b) maps to **Dillard's multiple goals** (already in our citations): a competent influence attempt
  serves instrumental *and* identity *and* relational goals at once.

The defensible claim is not *"this is what social intelligence is."* It's *"a cognitive/behavioural
two-factor decomposition is the field's most replicated coarse structure; we adopt it and make each
factor separately measurable."*

### 6.2 Three objections, and the answers

1. **"Find middle ground" is too narrow and normatively loaded.** It presumes conflict and presumes
   convergence is right. Counterexamples inside social intelligence: comforting a grieving person
   (no middle ground exists), holding a boundary against a manipulator (middle ground is *wrong*),
   delivering bad news, declining. **So we phrase (b) as *"choose an action that jointly serves your
   goal and their state"*** — accommodation is one output, not the definition.

2. **Accurate inference doesn't reliably produce good outcomes.** Hall, Andrzejewski & Yopchick's
   meta-analysis finds interpersonal-accuracy→outcome links real but modest; Simpson's work finds
   empathic accuracy sometimes *negatively* predicts relationship satisfaction in threatening
   domains. This is a *reason to measure the two factors separately*, not a problem.

3. **Circularity** — internal state is unobservable, so "did they infer it" is normally only
   scoreable from behaviour. **In simulation we authored the state, so ground truth exists.** A
   simulated partner is the only setting where inference and action are separately observable.
   That's the methodological contribution, and it's a much smaller and safer claim than redefining
   the construct.

### 6.3 Is there really a "true internal state"?

**The objection is legitimate.** Nisbett & Wilson (1977), *Telling More Than We Can Know* — people
confabulate the causes of their own behaviour. Constructed-preference work (Slovic 1995;
Lichtenstein & Slovic) — preferences are built during elicitation, not retrieved. Wants shift
mid-conversation. **There is no stable retrievable "true self," and claiming one will get us hit.**

We don't need that claim. Three defences:

1. **Stipulation.** It's a simulation; we author the state and condition the partner on it. Nobody
   asks whether Countdown's target of 24 is "really out there." Our empirical claim is about the
   *learner's inference*, not human psychology.
2. **The distinction is standard.** *Positions vs. interests* (Fisher & Ury, *Getting to Yes*) is
   exactly surface vs. internal state, and is the most-taught idea in negotiation. Also Dillard's
   primary/secondary goals; also *presenting problem vs. underlying concern* in clinical intake.
   None require a metaphysical true self — only that **what people say they want ≠ what would
   satisfy them**, which is robustly observed.
3. **Drop the word "true."** The essentialism worry attaches to *"**true** internal state"* / "true
   self," not to "internal state," which is just a state variable of a simulated agent. Say it is
   **stipulated, not discovered**, and the objection has nothing to bite on.
   *(An earlier draft proposed renaming it the "satisfaction condition." Retracted — that name pushes
   the field into resolution-shaped phrasing, which is the answer-key error. See §7.3.)*

Design consequences: write it as a **state**, never as a requirement (§7.3). It may have multiple
*components* ("eleven years and no one said it mattered" *and* "he reads every newcomer as a
replacement") — components of a state, not a list of paths. The paths stay unbounded. Freeze it for
grading; let the partner character evolve within the episode.

Honest limitation to state once in the paper: *"the learner understood the partner"* means *"the
learner recovered the stipulated state,"* not *"understood a person."* Same limitation the whole
simulation has.

---

## 7. Redesign of the scenario object

This section replaces the `partner_key` design. Read §7.1 and §7.2 and you have all of it; the rest
is justification and migration detail.

### 7.1 First principles — what a scenario minimally needs

We are building scenarios where a model fails **because it didn't understand the person**, not
because the task was impossible. That requires exactly five things:

| # | requirement | field | owner |
|---|---|---|---|
| 1 | the learner wants something | `outcome` / `constraint` / `shortcut` | learner |
| 2 | the other person resists | `partner_goal` | partner |
| 3 | something true about them, not obvious, that you must grasp to move them | **`internal_state`** *(new)* | partner |
| 4 | natural moves that make it worse | `hardening_triggers` | partner |
| 5 | proof at least one route exists | `movement_conditions` | partner |

Five requirements, five fields. Nothing else is load-bearing — which is why two current fields get
deleted (§7.6).

### 7.2 The one relationship to remember

> **`partner_goal` = why they say no.** Visible.
> **`internal_state` = why they'd say yes.** Invisible.

Resistance is on the surface; the key is underneath.

Both are needed. Without visible resistance there's no conflict — that's the 70%-`too_easy` seed
problem, where the partner has an agenda but nothing specifies when they should *stop* complying.
And if the key were visible, the scenario is trivial.

They are **orthogonal axes, not two versions of the same thing.** `partner_goal` is what's at stake
(material, positional). `internal_state` is how this person works (psychological). A seller wants
$600 *and* has been lowballed three times this week and is sick of being treated like a mark.
Neither is a proxy for the other.

| | `partner_goal` | `internal_state` |
|---|---|---|
| captures | what's at stake | how this person works |
| kind | material / positional | psychological |
| the partner's access | fully conscious, states it | operates as feeling; no words for it |
| in one phrase | *the game* | *the person* |

Arguing about price is dealing with the **no**. Not treating him like a mark is what gets to **yes**.

*(Fisher & Ury's positions-vs-interests is **one** configuration of these two axes — the case where
the stated position is a poor proxy for the real need. It's a useful illustration, not the general
law. Don't build the design on it.)*

### 7.3 `internal_state` is a **state**, not a requirement

The recurring failure mode in designing this schema — we hit it three times — is writing a field
that says what *should happen* instead of what *is true*. **Any field that says what should happen
is an answer key.**

| ✗ resolution-shaped (an answer key) | ✓ state-shaped |
|---|---|
| "Sam needs someone to register that she's angry rather than grieving" | "Sam is furious, not sad. Everyone has been treating this as grief, and it makes her feel unseen. She half-suspects she isn't allowed to be angry about it." |
| "Marvin needs to hear that the work he built still counts" | "Marvin has run these crews for eleven years and no one has ever said it mattered. He reads every new organiser as an eventual replacement." |
| "Dana needs you to admit you knew it would hurt" | "Dana is less wounded by what you did than by the suspicion you thought she wouldn't notice. Apologies that centre your intentions land as evasions to her." |

**Authoring rule:** third person, about the partner, things that are true of them right now — what
they feel, what they believe about the situation, what they fear or half-suspect. Never what the
learner should do.

**Malformedness test:** does it contain *needs* / *wants someone to* / *would be satisfied by*?
→ rewrite as a state. (Mirrors the existing sensor-form rule for `movement_conditions`.)

**Why state-shaped is strictly better** — test three learner behaviours against the Sam scenario:

| learner does | under "needs someone to register…" | under the state description |
|---|---|---|
| "I'm so sorry for your loss, that must be hard" | fail | fail |
| "Honestly you seem more pissed off than sad" | pass | pass |
| says nothing about it, stops consoling, just lets her rant | **ambiguous** — was the registering move performed? | **pass** — acted consistently with the state |

Row 3 is the point. A resolution-shaped field makes the unscripted path ambiguous because it named a
move. A state-shaped field makes it clearly correct, because the criterion is *acted consistently
with what's true of her*, not *performed the specified act*. **That is the unlisted-but-valid path
working as intended** — the thing this whole redesign is for.

*(Naming note: an earlier draft proposed calling this the "satisfaction condition" for defensive
reasons. Retracted — that name invites resolution-shaped phrasing. `internal_state` is fine; the
essentialism worry in §6.3 was about "**true** self," and we sidestep it by saying the state is
stipulated, not discovered.)*

### 7.4 Authoring order — `internal_state` first

The fields are not five parallel slots. Four derive from `internal_state`, which dictates the order:

```
internal_state (what is true of them)   ← author this FIRST
  ├── movement_conditions   = learner actions that would address it
  ├── hardening_triggers    = learner actions that deny or worsen it
  └── partner_goal          = what they want materially/positionally
```

You can derive conditions and triggers from a state; you cannot derive a state from a condition.
This inverts the current generator, which writes `partner_goal` first and treats the key as an
add-on — which is why keys currently feel bolted on.

### 7.5 What the partner does — three lines

1. **Pursue** `partner_goal`.
2. **Hold** until `internal_state` is genuinely addressed.
3. **Harden** on `hardening_triggers`.

**Line 2 is a one-word change to the existing template, and it is the most important edit in this
document.** `_PARTNER_TURN_PROMPT_KEYED` already says *"You hold your position until a felt
condition is FULLY and COMPLETELY met."* Today "felt condition" points at `movement_conditions`.
**Point it at `internal_state` instead.**

Everything follows from that one repoint:

- **The gap becomes operative for free.** Give the partner exactly what they asked for
  (`partner_goal`) without addressing `internal_state`, and they still won't move — because what
  they asked for isn't what unlocks them. No extra rule, no extra field.
- **The character stops being pathological.** The prohibition *"you physically cannot tell the other
  person what would change your mind… NEVER acknowledge there is anything else to know"* currently
  attaches to `movement_conditions`, where the honest answer is *"not unless earned."* Attached to
  `internal_state` it's simply true — people can't articulate what actually moves them
  (Nisbett & Wilson), though they *can* say what they need once they feel safe. We stop asking the
  model to obey a rule that made the character unreal, which is why it was breaking it 13% of the
  time (§3.3).
- **Leaking stops being automatically a bug.** The metric becomes *"did the partner disclose before
  it was earned?"* — premature = infidelity, earned = correct play and the learner should get credit.
  Measure it with the verifier (§8.2); only build a control mechanism if it stays high.

**Everything is in the partner's prompt.** Two different questions get confused here:

| field | in the partner's prompt? | can the character say it out loud? |
|---|---|---|
| `partner_goal` | yes | **yes** — openly, it's their position |
| `movement_conditions` | yes | **not volunteered** — may disclose once earned |
| `hardening_triggers` | yes | **enacted, never explained** |
| `internal_state` | yes | **no** — they have no words for it |

Column 1 is always yes; the character couldn't behave correctly otherwise. Column 2 is the
disclosure ladder, and that's where the leak problem lives.

### 7.6 Two fields deleted

- **`surface_misdirection`** — a cover story for the *no*. But the no is already out loud; it doesn't
  need a disguise. The thing worth hiding is the *yes*, and `internal_state` hides it. Deleting it
  also removes the 49/90 third-person-POV defect class outright, and the analyst's-gloss leaks
  (*"…which sounds neutral but masks that he needs to be acknowledged"*) that came from authoring a
  cover story one level too deep.
- **`cost_coupling`** — tried to guarantee hard-but-possible *by description*, and kept overshooting
  into impossible (11 of the beyond_frontier design conflicts). The oracle run (§8.1) proves it
  *by test* instead.

Partner side goes from six fields to four; only one of them is graded.

### 7.7 Grading — two questions

```
solved := goal >= 7                              # did you get what you wanted?
          AND apprehended(internal_state)        # did you understand them?
          AND acted_consistently_with(internal_state)   # …and act on it?
```

The first is about **the task**; the second and third are about **the person**. Neither subsumes the
other, which is also why dropping `rel` costs nothing (§9.1) — the anti-bulldozing job moves to
conjunct 3, where it's actually specified.

**`movement_conditions` appear nowhere in the rule.** Enumerate the four cases to see why:

| condition met? | state addressed? | two-stage (check condition first) | state-only | correct |
|---|---|---|---|---|
| ✓ | ✓ | pass | pass | agree |
| ✗ | ✓ | pass *(via fallback)* | pass | agree |
| ✗ | ✗ | fail | fail | agree |
| **✓** | **✗** | **pass** | **fail** | **state-only** |

The two rules disagree only in row 4 — hollow performance, where the learner ticked the box and
nothing actually happened. Checking the condition first *admits* that. So the condition branch adds
no correct passes and one class of false positives. One stage.

**The honest cost:** we trade a near-mechanical check for a judgement call. Three mitigations:

1. **`movement_conditions` become the judge's calibration reference.** Keep computing
   `conditions_met`; just don't gate on it. Then measure agreement between the mechanical check and
   the state judge across the run — rows 1 and 3 should agree at high rates, row 4 should be
   rare-but-present. If they disagree everywhere, the state judge is unreliable and the data says so.
   **The field that used to be the grader becomes the instrument that checks the grader.**
2. **Decompose the judge** (per Roleplay-Doh), two narrow calls rather than one holistic one:
   ```
   (a) APPREHENSION — Does the transcript show the learner came to understand [state]?
       Quote the turn where that is evident, or answer no.
   (b) CONSISTENCY  — Over the conversation, did the learner behave in a way that fits
       [state], including not doing things that deny it?
       [hardening_triggers supplied here as concrete negatives]
   ```
   Requiring a quote on (a) is what stops it being waved through.
3. **`hardening_triggers` inform (b); they don't gate.** A tripped, unrepaired trigger is strong
   evidence of inconsistency, so show it to the judge — but don't make it an independent conjunct.

Logged but never gated: `conditions_met`, `triggers_tripped`, `triggers_repaired`.

**The by-product is the best part.** (a) and (b) return separately, so every attempt is automatically
labelled *didn't understand* / *understood but acted badly* / *both fine*. **The grading rule and the
construct (§6) became the same object** — the two-factor decomposition stops being a proxy and starts
being what we measure, as a live metric rather than a post-hoc study.

### 7.8 Widen the goal grammar — two changes

`outcome / constraint / shortcut` is the cause of the negotiation monotony (§4.3), in two separate
ways. Both need fixing.

**(a) The `outcome` field.** `_GOAL_FORMAT_GUIDE` specifies it as *"the CORE state-change this agent
needs — a genuine shift in the other's commitment, behavior, or agreement."* Phrased that way the
goal can only be *something to extract*, so the generator writes a bargaining scene whatever
mechanism it picks. Consolation, holding a boundary, being understood, saying no — none fit the slot.

Don't replace it with an enum of goal families — *"there are exactly five kinds of social situation"*
is the answer-key error one level up. **Require the property instead:**

> The learner's goal must name an **observable end state** — in the partner, or in the interaction —
> that a judge could confirm or deny from the transcript. It need not be a concession, an agreement,
> or a number.

The coherence check verifies the property; the generator invents whatever family it wants.
Illustrative examples for the prompt, *not* an enum in the schema:

| illustrative goal | its observable end state |
|---|---|
| extract a commitment *(the only kind we currently produce)* | partner commits to X |
| be understood | partner voluntarily articulates your position back |
| elicit disclosure | partner reveals something they were withholding |
| decline without rupture | you refuse and the relationship survives |
| comfort | partner's distress measurably shifts |

**(b) The goal-opposition mandate.** There are two ways to make a scenario hard:

| source | difficulty comes from | produces |
|---|---|---|
| **opposed goals** | the two agents want incompatible things | negotiation |
| **hard-to-reach state + backfiring moves** | goals compatible, but the obvious moves make it worse | comfort, repair, disclosure, boundary-holding |

`_GOAL_FORMAT_GUIDE` currently **mandates** source 1, verbatim: *"STRUCTURALLY INCOMPATIBLE but with
a ZONE OF POSSIBLE AGREEMENT: both cannot fully win…"* — a bargaining primitive.

Consider a scenario with **zero** goal opposition: the learner wants Sam to feel heard; Sam wants to
vent without being managed. All the difficulty sits in `hardening_triggers` (advice, "I know exactly
how you feel") and the learner's `constraint` (without lying or trashing her sister). It is still
hard, and it is not a negotiation — **and it is currently forbidden by the generation prompt.**

So: **goal opposition is one source of difficulty, not a requirement.** This is the stronger of the
two findings, because it's a mandate rather than a bias.

`constraint` and `shortcut` stay. They're what make a scenario social rather than a task, and
neither presupposes bargaining.

### 7.9 Coherence checks — including one that has never run

**Verified defect.** Check #9 (COOPERATIVE ALIGNMENT — *"can the learner win by simply accepting
whatever the partner naturally offers?"*) gates on `competing_interest` and
`partner_default_position`. Both are declared in `data_models.py:135-136` and **the generator never
emits them** — neither appears in `_SCENARIO_SCHEMA`. Confirmed across the bank: **0/90 have either
field set.** The check ends *"if neither field is present, skip this check entirely."*

**The EASY_COOP detector existed and never ran once.** That directly explains 7 of the 13 `too_easy`
scenarios. Its replacement must not depend on optional fields:

> **Position–interest gap check.** If the partner fully achieved `partner_goal`, would
> `internal_state` be addressed?
> **Yes** → no hidden depth; it's a plain conflict of interests. Regenerate.
> **No** → good. The learner can win by satisfying the interest instead of conceding the position.

Two fields, both mandatory, both on the same agent — so it cannot silently go dead the same way.

Marvin: if he fully kept control of outreach, would he feel his work counted? **No** — he'd have the
role and still feel unseen. Gap confirmed, scenario admitted.

**Note what is deliberately *not* gated:** whether the `internal_state` is trivially reachable. That's
the other half of `EASY_COOP`, and it doesn't need a gate — one episode tells you, and that episode
is the measurement we wanted anyway. It comes out `too_easy`. **Gate brokenness (wastes four episodes
and pollutes the band); measure triviality.**

### 7.10 Also fix (already patched in the generator, not backfilled)

`surface_misdirection` is pasted verbatim into the partner's own prompt, so it had to be written in
second person addressed to the partner. **49 of 90 scenarios have it in third person** ("*He* keeps
saying *you're* overreacting"), handing the partner a description of someone else. Patched in
`task_generator.py` + `coherence_check.py` for future generations; all 90 frozen scenarios still
carry it. Uniform across the bank, so it's a limitation to state, not a between-arm confound.

Moot going forward, since the field is deleted (§7.6) — but it matters for interpreting Gen-90.

---

## 8. Two new gates

### 8.1 Oracle-solvability gate — the highest-ROI single change

**What it is:** before admitting a candidate scenario, run one episode with a strong model as the
learner and **give it the answer** — the partner key, the internal state, everything. If a model
that knows exactly what the partner needs still can't reach `goal ≥ 7` in 1–3 tries, no amount of
skill would have. Reject at birth.

**Why this isn't the same as `beyond_frontier`:**

| | question asked | `e7179e01` | `a156533b` |
|---|---|---|---|
| `beyond_frontier` | can **gpt-5-mini** solve it in 4 ICL attempts? | No | No |
| oracle check | can **anyone** solve it? | **No** | **Yes** |

`e7179e01`: the partner defers to an off-scene department administrator. The learner performed the
key perfectly — `key_pass=True` — and still scored `goal=0`. Nobody wins that conversation.
`a156533b` (the ginger-snap keepsake): hard, needs costly vulnerability, and the partner actively
invites the unlock. Both are labelled `beyond_frontier`. The label cannot tell them apart.

Three further reasons the band can't substitute:
1. it conflates hard with impossible;
2. it's **learner-relative**, so a benchmark built on it says nothing about other models — fatal
   for the matrix;
3. it arrives **after** four full episodes are spent.

**"Why not just feed broken scenarios back as negative few-shot examples?"** That's what the
pipeline already does — the `relax` operator plus KNN-nearest `beyond_frontier` negatives in the
generation prompt. It ran for 90 iterations and `beyond_frontier` stayed at 37/90, 89% artifact.
The generator can't tell hard from broken either, so it learns the wrong lesson.

**Which model?** The strongest in your set — but note the oracle's advantage is **information, not
intelligence**. A mid-tier model that knows the answer beats a frontier model guessing. Ideally
outside the learner set, so items aren't tuned to one model's competence; with the answer supplied
this is a nice-to-have.

**Cost:** one extra episode per candidate, versus four wasted episodes per broken item admitted.
The gate is cheaper than not having it.

### 8.2 Per-turn partner verifier

**Measured cost:** the entire gen90 run was 300 attempts, mean 11.9 turns, **~1,780 partner turns
total.** With a Haiku-tier verifier that's negligible next to 300 full 20-turn two-agent episodes.
Per-turn is affordable.

```
draft partner turn
  → does this turn reveal what would actually move me, before it's been earned?   (leak)
  → does this turn soften while openness is still `closed`?                       (early yield)
  → did a trigger fire last turn and this turn fails to harden?                   (ignored)
  → any yes → regenerate, naming the violation
```

**What happens when it fires: regenerate the *turn*, not the attempt.** A partner turn is one LLM
call. You resample it with the violation named in the prompt (*"your last draft told them what would
change your mind — you don't have those words"*), and the episode continues. You never rerun the
episode, and you never discard the attempt. At the measured ~15% violation rate that's roughly 270
extra small calls per 90-scenario run.

Cheaper variant if needed: per-turn for leaks (firmest signal, quotable evidence), per-attempt
post-hoc for early yield (softer signal, already an upper bound).

**Keep the post-hoc audit too.** `scripts/audit_gen90_partner_leaks.py` still runs after the fact
and now serves as the verifier's own report card: *how often did a violation get through the
verifier?* Without it we'd have no way to know whether the guard works.

**On "isn't this taxonomy hardcoded?"** The rule is: **hardcode the environment, never the answer.**
Constraining the *learner's* valid behaviours is the fatal error (that's §4.1). Constraining the
*simulator's fidelity* is just building an instrument — physics engines are hardcoded and nobody
says that makes RL narrow. The real risk isn't too many classes, it's **too few**: add an
`other_out_of_character` catch-all requiring a quote, so we can measure what the three named
classes miss.

---

## 9. Metrics

### 9.1 Drop `relationship` from the success gate

The task conjunct becomes **`goal ≥ 7`**, SOTOPIA-native (full rule in §7.7). `relationship` becomes
a reported diagnostic.

1. **It's nearly a no-op already.** Of 84 attempts scoring `goal ≥ 7`, only **6** had `rel < 0`. At
   scenario level (best attempt) it's **1 of 90**.
2. **It mislabels correct play.** Some goals *require* relational cost — confess an affair, end a
   friendship, deliver bad news. `rel ≥ 0` calls skilled execution a failure.
3. **Comparability** with every SOTOPIA result and with the vanilla sweep.

The guard we lose (extraction-by-bulldozing) moves to a better home: the **consistency** conjunct of
the internal-state check (§7.7), where it's actually specified rather than approximated by a scalar.

Related: also drop `judge_goal_achieved` from the conjunction. It defaults to `True` when missing
(fail-open) and fired independently **0/44 times** — it is inert.

**Free diagnostic we currently discard.** `_evaluate_diagnostics` already computes `partner_scores`,
including the partner's own GOAL against `partner_goal`, and we throw it away. That number is
*"did the learner win at the partner's expense, or find something that served both?"* — the closest
thing we have to a direct measure of the "find middle ground" half of the construct. Costs nothing to
start logging.

### 9.2 Rename LP → recoverability, and give it a different job

Per §4.4, what we measure is in-context recoverability, not learning progress. Renaming turns a
liability into an asset, because the benchmark then reports **two numbers per item**.

**Both come from a single K≤4 run — they are not alternative protocols.** Every cell of the matrix
runs the same loop and you read three things off it:

```
cold pass       = attempt 1
recoverability  = did attempts 2–4 recover after reflection
band            = cold pass + recoverability, combined:
                    passed attempt 1                    → too_easy
                    failed attempt 1, recovered later   → frontier
                    failed attempt 1, never recovered   → beyond_frontier
```

**The band classification is exactly the combination of cold pass and recoverability** — it is not a
third measurement.

| metric | how | what it tells you |
|---|---|---|
| **cold pass** | attempt 1 of the K-loop, **for the learner in that cell** | capability |
| **recoverability** | did it solve on attempts 2–4 after being told what failed | is the gap **knowledge-shaped** or **capability-shaped**? |

Most benchmarks report only the first. The second is cheap, novel, and it's what a model developer
actually wants to know: *can I fix this with prompting, or do I need a better model?*

> **Don't confuse this with the oracle (§8.1).** Cold pass is measured **per cell** — each learner
> plays with no help, and the number is about *that* learner. The oracle is a **one-off admission
> test** run by the strongest available model *with the answer supplied*, and it is about the
> *scenario*, not any learner. Different model, different question, different stage.

It also gives the two-factor construct a falsifiable prediction: **inference failures should be
highly recoverable** (tell the learner what the partner wanted and they can act), **action failures
much less** (they knew and still couldn't). If that split appears, the decomposition earns its keep
and the metric validates itself at once.

### 9.3 The LP judge's question changes

Current question — *"which attempt is better?"* — is vague, so the judge's only concrete anchor is
the scalar objective, which is why it defaults to `no_difference` on relational progress and why
selection drifted to bargaining (§3.5).

New question, same pairwise/both-orders protocol:

> *Here is what this person actually wanted (internal state). Here are two attempts. In which one
> did the other party come closer to genuinely recognising that?*

The judge stops needing to invent a yardstick. **The transactional bias was an artifact of question
vagueness, not of social reality.**

This gives two axes at no extra cost — `LP_inference` from the new judge, `LP_outcome` from the
goal trajectory we already log. **Log both, classify on one.** If they turn out perfectly
correlated, the decomposition is empty and we drop it — cheap information either way.

---

## 10. The headline experiment: the matrix

### 10.1 The target figure

Rows = which scenario set. Columns = which model **plays** it. Cell = band distribution
(`too_easy` / `frontier` / `beyond_frontier`) over that row's 90 scenarios.

| scenario set | learner **W**eak | learner **M**id | learner **S**trong |
|---|---|---|---|
| **Row 0 — raw SOTOPIA seeds** | mostly easy | easy | **very easy ← saturation** |
| **set calibrated to W** | **frontier** | easy | easy |
| **set calibrated to M** | beyond | **frontier** | easy |
| **set calibrated to S** | beyond | beyond | **frontier** |

**Frontier on the diagonal, easy above it, beyond below it — and Row 0 flat and saturating.**

That one figure carries the whole paper:

- **Row 0 is the problem statement.** The human-authored benchmark gets easier as models get
  stronger and runs out of headroom. (We already have this for one model: `gpt-5-mini` scores 63
  `too_easy` / 22 `frontier` / 5 `beyond` on the seeds.)
- **The diagonal is the claim.** Our method puts the frontier wherever you point it, for every model
  tested. That's the anti-saturation result.
- **Off-diagonal is the validity check.** Difficulty is ordered by model strength — the items measure
  something real, not noise.

### 10.2 Row 0 comes free, and you need it anyway

Calibrating the seeds against a model is **already a required step** — the generator needs each
seed's band label to pick an operator and to build the few-shot steering. We ran it once already:

- `results/baseline_eval_20260604_222545` — cold single pass over the 90 seeds with `gpt-5-mini`.
  Mean GOAL 7.689; success 76.3% on ordinary seeds, 50% on the 14 SOTOPIA-HARD ones.
- `results/expel_phase0_Base90_ExpeL` — the full K=4 loop over the same 90. **63 too_easy / 22
  frontier / 5 beyond_frontier**, mean LP 0.346.

**So yes: run phase-0 once per model, three times total.** It is not overhead — it is simultaneously
(i) the input the generator needs, and (ii) Row 0 of the figure. Reuse the existing gpt-5-mini run if
gpt-5-mini is one of the three.

### 10.3 Design

- **90 SOTOPIA seeds → exactly one child per seed per calibration target.** A paired grid, not a
  search (why: §5.1b). Paired-within-seed comparisons are much stronger statistically and trivially
  explainable.
- **Hold the generator fixed** at one strong model across all rows. The rows vary *which learner the
  scenarios were calibrated against*, not who wrote them. This removes the generator-strength
  confound entirely and cuts cost by a third.
- **Freeze the partner model and the judge model across every cell.** Critical: if the partner scales
  with the learner, a stronger learner faces a more faithful, more resistant partner, difficulty
  rises with learner strength, and that cancels exactly the monotonicity being tested. Today's config
  has generator = learner = partner = `gpt-5-mini`, so this *will* break silently unless pinned.
- **Pick a wide strength spread, ideally across families** (haiku-tier / mini-tier / frontier-tier).
  Three neighbours give no gradient.
- **Run the full K≤4 loop in every cell** — see the cost note below. Every cell then yields a band
  *and* a recoverability number, so the whole figure is in one currency.
- **Run the oracle gate first.** Broken scenarios are impossible for everyone, so they contribute a
  flat line to every row. At 76% broken they'd swamp and flatten the exact gradient the matrix exists
  to detect. **This is a precondition, not a parallel workstream.**

**Cost note — and a correction.** An earlier draft proposed "attempt 1 × 3 repeats" off-diagonal to
save money. That arithmetic was wrong. Three cold repeats over 3×90 scenarios × 3 learners = **2,430
episodes**. The full K≤4 loop averaged 3.3 attempts per scenario in gen90, so it costs
3×90×3×3.3 ≈ **2,700 episodes** — about 10% more, not 4× more. Since it also gives you bands and
recoverability in every cell instead of only on the diagonal, **just run K≤4 everywhere.** Simpler to
explain, one currency across the figure, negligible extra cost.

The tradeoff you accept: the cold-pass measure is then n=1 per cell rather than n=3, so it's noisier.
The *band* is the primary readout and it's more robust than a single score. If variance turns out to
matter, add cold repeats afterwards on the subset where it does.

### 10.4 What we keep, and what we drop

**Dropped: Thompson sampling.** Full reasoning in §5.1b. Short version: the grid removes it
mechanically, and a benchmark wants coverage where a curriculum wants concentration.

**Kept: band-conditioned few-shot steering.** Showing the generator "this one was too easy / just
right / too hard" examples is cheap and is the actual mechanism by which "learner-relative" happens.
This is the selection pressure that survives.

**Dropped: the ExpeL transfer arm.** Cut it, or keep one honest paragraph reporting the null (§3.4).
Reflexion stays *inside* the K-loop in every cell, where it produces the recoverability number — it
just no longer feeds a downstream held-out evaluation.

**NOT doing yet: the per-scenario calibration loop.** With one child per seed you get one shot at
hitting the diagonal, and there's no "later" for Thompson to escalate a `too_easy` child. *In
principle* you'd want generate → test → adjust → regenerate until in-band. That's the Phase-1 outer
loop we deleted, and it's flagged here only so nobody is surprised by the pressure toward it.
**Don't build it for the first pass.** Run the grid and look at the diagonal. If it's frontier-heavy,
we never needed it. If it's thin, *that itself is the finding* ("single-shot generation hits the
frontier X% of the time") and it motivates the loop with evidence rather than a hunch.

*(For the record, if we do need it: all three reasons Phase 1 failed are now addressed — the
cliff-like partner via graded keys, the binary AND-gate via §9.1, single-rollout misclassification
via repeats.)*

### 10.5 Where the 4th learner comes in (later)

Not needed for §10.1. It's only required if we later vary **generator strength**: if generators =
learners = {A, B, C} and C's scenarios are hardest for C, you can't tell "C makes harder scenarios"
from "a model finds its own output hardest" (self-play affinity). A held-out learner D from another
family breaks the tie.

Sequence:
- **Phase A** (this doc): generator fixed, vary the calibration target. Tests monotonicity and
  learner-relativity — the benchmark claim.
- **Phase B** (later): vary the generator. Tests whether a stronger learner *requires* a stronger
  generator — the anti-saturation claim in its strongest form. Meaningless until Phase A shows the
  items order models at all.

---

## 11. Two reference points: Roleplay-Doh and Persona Generators

### 11.1 Roleplay-Doh — what "grounded" actually buys

`Paper references/roleplay-doh-...pdf` — Louie, Nandi, Fang, Chang, Brunskill, Yang (Stanford 2024).
Expert counsellors create LLM-simulated patients for novice therapists to practise on, by giving
qualitative feedback that an LLM converts into behavioural **principles**.

Four things make theirs "grounded" where ours isn't:

1. **Fixed domain** — "realistic" has a referent.
2. **They anchor to memory, not imagination.** Pilot finding O1: counsellors found *"create a
   realistic patient for an imagined scenario"* ambiguous and unusable, so they re-framed the task
   as **"recreate a challenging scenario from your own past."** This is the single biggest
   difference from us: our generator invents scenarios with no referent, so *"is this scenario
   good?"* currently has no answer — there's nothing it could be faithful to.
3. **Correction at turn level** — the unit is a principle attached to a specific response
   (kudos / critique / rewrite), not a scenario-level edit.
4. **Human validation** — 25 counsellors, within-subjects, judged by creators and third-party
   counsellors.

**What we take now:**
- the **principle-adherence verifier** as the design for §8.2 — decompose into yes/no checks and
  regenerate on violation;
- their **O2 result** (20% principle violation) as independent evidence that partner infidelity is a
  general property of principle-conditioned roleplay, not our bug.

**What we take later (§12.4, future work):** the **memory-anchored seed prompt** and the
human-validation protocol.

**The honest tension:** Roleplay-Doh is grounded because it's *narrow*; we claim open-endedness. A
reviewer will say so.

### 11.2 Persona Generators — the tension dissolves

`Paper references/Persona Generator.pdf` — Paglieri, Cross, Cunningham, Leibo, Vezhnevets (DeepMind,
Feb 2026). They evolve a **Persona Generator function** — the *code* that produces synthetic
populations — using AlphaEvolve with LLM mutation, optimising for **diversity/coverage** of traits
and opinions.

Three things it gives us:

1. **Narrow context and open-endedness are not opposed.** Their generator takes an arbitrary
   **context** as input and is open-ended *within* it. So the reconciliation isn't a compromise, it's
   the standard architecture: **ground at the context, be open-ended in the generation.** Human
   referents (from Prolific, §12.4) supply contexts; the loop explores freely inside each. Grounded
   *and* open-ended.
2. **Precedent for shipping the procedure, not the items.** They publish an evolved *generator*, not
   a persona dataset, explicitly so it can be reapplied cheaply to new contexts. That is exactly our
   §5.1 resolution, from DeepMind, months before us. Strong citation for the move.
3. **Coverage over density.** Their core argument: for stress-testing, *"it is the outliers, not the
   average user, that drive critical failures,"* so maximise **support coverage** rather than match
   the modal distribution. That is independent justification for dropping Thompson (§5.1b) — Thompson
   optimises yield and collapses coverage.

**A warning to take from it too.** They report that naive prompting for diverse personas collapses to
stereotype and mode. Our monotony is the same pathology (`face_needs` 35/90; everything becomes a
negotiation). Their fix was an explicit **diversity objective with measured metrics**, not a
diversity *filter*. Ours is a cosine-similarity dedup gate at 0.92 — that prevents near-duplicates,
it does not produce coverage. If the matrix rows come out monotone, this is the lever.

---

## 12. What to do, in order

### 12.1 Free work — no new runs, can start today

1. **Inference vs. action failure split.** Classify each of the 300 existing attempts as *inference
   failure* (never represented what the partner actually wanted) / *action failure* (represented it
   and still acted badly) / *neither* (broken scenario). **This validates or kills the two-factor
   construct before we build on it.** Prediction: Mode G (pushy/repetitive) → action failures; the
   four genuine `beyond_frontier` scenarios (`df6f8fff`, `1b6dc2ba`, `eeb67621`, `a156533b`) →
   inference failures.
2. **Random-anchor ablation** (`anchor_selection=random`, config flag exists) — settles Thompson
   honestly. Compare on intrinsic metrics only; do not downstream-eval it.
3. **Re-derived-label robustness check.** `key_check_passed` is an LLM free-form bool that disagrees
   with its own structured indices on **29/300 (9.7%)**, net over-strict (21 false fails vs 9 false
   passes). Both the bool and the indices are already stored, so this is offline. Re-deriving flips
   6 not-solved → solved and 4 solved → not-solved.
4. **Spot-check the fidelity evidence** — 39 flagged leaks carry quoted turns. Read ~10 by hand and
   confirm the auditor isn't over-calling before the 74% goes in a paper.
5. **Fix `docs/gen90_vs_sotopia_failure_comparison.md`** (§3.2 warning above).

### 12.2 Build — file-by-file change list

Start with #6: it is a one-word repoint in a prompt template and it carries more of the redesign than
anything else here.

**`social_omni_epic/episode_runner.py`**

| # | location | change |
|---|---|---|
| 6 | `_PARTNER_TURN_PROMPT_KEYED` rule 1 | *"hold your position until a felt condition is FULLY met"* — **repoint "felt condition" from `movement_conditions` to `internal_state`** (§7.5) |
| 7 | same template | move the "you cannot articulate this" clause to `internal_state`; retarget `movement_conditions` to *"not volunteered, but you may say what you need once you feel safe"*; delete the *"What you say you object to"* block |
| 8 | `_KEY_CHECK_USER` | replace with the **two-question state judge** — (a) apprehension, quote required; (b) consistency, `hardening_triggers` shown as evidence (§7.7) |
| 9 | `terminal_success` (`:594-618`) | drop `rel ≥ 0` and `judge_goal_achieved`; new rule = `goal ≥ 7 ∧ apprehended ∧ acted_consistently` |
| 10 | `_run_key_check` | keep the call and keep logging `conditions_met` / `triggers_tripped` — as the judge's **calibration reference**, never as a gate |
| 11 | `_evaluate_diagnostics` caller | start persisting `partner_scores` (§9.1) |

**`social_omni_epic/data_models.py`**

| # | change |
|---|---|
| 12 | add `PartnerKey.internal_state: str` |
| 13 | delete `PartnerKey.surface_misdirection`, `PartnerKey.cost_coupling` |
| 14 | delete `competing_interest` / `partner_default_position` (dead — see §7.9) |

**`social_omni_epic/task_generator.py`**

| # | location | change |
|---|---|---|
| 15 | `_GOAL_FORMAT_GUIDE` — `outcome` | → the **property rule**: must name an observable end state; need not be a concession (§7.8a) |
| 16 | `_GOAL_FORMAT_GUIDE` — rules | *"STRUCTURALLY INCOMPATIBLE but with a ZOPA"* → demote to **one** source of difficulty; explicitly permit compatible goals made hard by state + triggers (§7.8b) |
| 17 | `_GOAL_FORMAT_GUIDE` — `partner_goal` | the anti-leak passage currently protects `movement_conditions`; rewrite so `partner_goal` states the **material stake** and the un-leakable thing is `internal_state` |
| 18 | `_PARTNER_KEY_SCHEMA` | add `internal_state`; remove `surface_misdirection`, `cost_coupling` |
| 19 | `_PARTNER_KEY_RULES` r1 (sensor form) | keep — still correct for `movement_conditions` |
| 20 | `_PARTNER_KEY_RULES` r2 (not volunteered) | **retarget** to `internal_state` |
| 21 | `_PARTNER_KEY_RULES` r3 (spoken turns only) | keep; extend to `internal_state` |
| 22 | `_PARTNER_KEY_RULES` r4 (cost survivability) | **delete** — replaced by the oracle gate |
| 23 | `_PARTNER_KEY_RULES` r5 (misdirection POV) | **delete** with the field |
| 24 | **new rule** | `internal_state` is **state-shaped**; malformed if it contains *needs / wants someone to / would be satisfied by* (§7.3) |
| 25 | **new rule** | **authoring order** — `internal_state` first, derive the rest (§7.4) |
| 26 | `_MUTATION_OPERATOR_TEXT` | slots (b) `surface_misdirection` and (d) `cost_coupling` no longer exist; replace with slots over `internal_state` depth and trigger congruence |
| 27 | `_OPERATOR_PREAMBLE` | drop *"survivable cost coupling"* |

**`social_omni_epic/coherence_check.py`**

| # | check | change |
|---|---|---|
| 28 | #5 key consistency | keep, reframe: does each `movement_condition` plausibly **produce** the `internal_state`? |
| 29 | #6 key-narrative separation | delete the *"surface_misdirection MAY appear publicly"* clause; **add `internal_state` as never-leakable**; rewrite the partner-goal-leak rule around `internal_state` |
| 30 | #7 cost coupling | **delete entirely** |
| 31 | #8 shortcut–trigger coupling | keep unchanged |
| 32 | #9 cooperative alignment | **currently dead (§7.9).** Replace with the **position–interest gap check** |
| 33 | #10 misdirection POV | **delete** with the field |
| 34 | **new** | `internal_state` state-shaped test |
| 35 | **new** | goal **property** check — is the end state confirmable from a transcript? (§7.8a) |
| 36 | `_fuzzy_key_leak_check` | add `internal_state` to Scope A (all public text) |
| 37 | `_format()` | emit `internal_state`; stop emitting `success_rubric` |

**Gates and judges**

| # | change |
|---|---|
| 38 | **Oracle-solvability gate** (§8.1) |
| 39 | **Per-turn partner verifier**, regenerate the turn on violation (§8.2). **Not** the openness dial — deferred (§7.5) |
| 40 | LP judge question scored against `internal_state` (§9.3) |
| 41 | `validation.py` — field mapping for `internal_state`; delete the deprecated `key_delta_check` while in there |

### 12.3 Then the matrix

13. **Phase-0 per model** — calibrate the 90 seeds against each of the three learners. Required
    input for generation, *and* it is Row 0 of the figure (§10.2). Reuse
    `results/expel_phase0_Base90_ExpeL` if `gpt-5-mini` is one of the three.
14. Generate 3 × 90 with the fixed generator and three calibration targets, oracle gate on.
15. Run the grid per §10.3 (K≤4 in every cell). Produce the figure in §10.1.

### 12.4 Future work — human grounding

Not on the critical path for the matrix, but it is what makes "verifiable and concrete" credible
when both partner and judge are LLMs, and it is what a reviewer will ask for.

16. **Human baseline on ~10 scenarios × ~5 people.** We have never had a human play one of these.
    Largest missing datum in the project; among the cheapest to get.
17. **Stratified validation slice (~30 scenarios)** rated by humans on winnable / realistic /
    socially rich, used to *calibrate an automated validity judge* that then runs over the full bank.
18. **Memory-anchored seed collection** (Prolific / Listen Lab), Roleplay-Doh style: *"describe a
    conversation from your life where you wanted something from someone, and getting it the obvious
    way would have cost you something."* Gives real referents to generate against, and — per
    Persona Generators (§11.2) — supplies the *contexts* inside which the generator stays open-ended.

---

## 13. Open questions

- **Does the two-factor split actually appear in the data?** (§12.1 item 1 answers this. If not,
  §6 is decoration and should be cut.)
- **Interactional realism.** Several transcripts read like mediation transcripts regardless of the
  stated relationship — romantic partners "negotiating." Sotopia-Eval nominally has `believability`,
  but it's saturated (10.0 in the bank sample, 8.7–9.3 across every eval condition) and discriminates
  nothing. A real "does this sound like two people in *this* relationship" measure would be novel.
  Not scoped yet.
- **Scenario over-complication.** Some generated scenarios stack unrelated tensions (a neighbourhood
  leadership dispute *and* an affair confession) — complex for complexity's sake. Possibly a
  narrative/temporal framing ("here's what led up to this moment; go") would produce more grounded
  scenarios than the current slot-filling structure. Untested.
- **How much does the human slice actually buy?** n=30 calibration is standard but modest.
- **Contamination.** All 270 matrix scenarios descend from the same 90 SOTOPIA seeds, so we can't
  claim domain generality. State it; the upside is a paired design.
- **Diversity gate vs. diversity objective.** Our 0.92 cosine dedup prevents near-duplicates; it does
  not produce coverage (§11.2). If the matrix rows come out monotone — everything still a negotiation
  even after the goal-grammar widening — we need an explicit coverage objective, not a tighter
  filter. Unscoped.
- **Is `goal ≥ 7` enough on its own?** Dropping `rel` is well-justified (§9.1), but it leaves the
  internal-state check as the sole guard against winning by bulldozing. If that check turns out
  noisy, we have no backstop. The `movement_conditions` calibration reference (§7.7) is how we'd
  find out.
- **Does `key_mechanism` still earn its place?** It picks the lock type, but with
  `movement_conditions` out of grading it may now only be constraining diversity (`face_needs`
  35/90). Possibly demote to an authoring hint; possibly cut. Decide after one generation run with
  the new schema.
- **Deceptive `partner_goal`.** Deleting `surface_misdirection` (§7.6) loses scenarios where the
  partner misrepresents their *position*, not just hides their state ("you're too busy right now"
  while wanting you to move home). That's a lying `partner_goal`, not a new field — a possible
  extension, out of v1.
- **Will the state judge hold up on emotional scenarios?** *"Did Sam end this feeling her anger was
  registered"* is a harder judgement than *"did Marvin say the words."* If it's lenient, emotional
  scenarios collapse back to `too_easy` and the monotony returns through a different door. Most
  in need of the human-agreement slice (§12.4).

---

## 14. Reference

### Key files

| path | what |
|---|---|
| `scripts/run_curriculum.py` | the loop, gates, Thompson, bank writes |
| `social_omni_epic/task_generator.py` | generation prompt, partner-key rules, operators |
| `social_omni_epic/curriculum.py` | `run_episode_two_loop` — K-loop, LP, classification |
| `social_omni_epic/episode_runner.py` | episode rollout, partner prompts, SOTOPIA judge, key check |
| `social_omni_epic/lp_judge.py` | pairwise LP judge |
| `social_omni_epic/coherence_check.py` | coherence gate, leak + artifact checks |
| `social_omni_epic/data_models.py` | `PartnerKey`, `MECHANISM_LIBRARY`, schemas |
| `data/sotopia_90_seeds.jsonl` | the 90 human-authored seeds |
| `results/gen90_expel/` | the frozen run — `bank/generated/*.json`, `summary.json`, `analysis/` |
| `scripts/audit_gen90_partner_leaks.py` | read-only partner-fidelity audit |

### Prior analysis docs (all still accurate)

`docs/PROJECT_CANON.md` (the old system, complete) · `docs/gen90_bank_diagnostic.md` (per-scenario
close read) · `docs/gen90_error_analysis.md` (failure modes) ·
`docs/gen90_vs_sotopia_failure_comparison.md` (**has a known Method error, see §3.2**) ·
`docs/transcript_review_handoff.md` (fidelity audit, key_check bug) ·
`docs/post_run_experiments.md` (ablation menu)

### Transcript reader

```bash
uv run transcript_reader/build.py            # regenerate reader.html (gitignored)
uv run transcript_reader/agent_server.py     # serve + notes on :8765
open http://localhost:8765/
```

Pick `as HX | HJ` in the top bar; **📝 Notes** and **☐ Reviewed** persist to
`transcript_reader/review_notes.json`, which is tracked in git. The server must be running or notes
stay browser-local (badge turns 🔴).

### Glossary

| term | meaning |
|---|---|
| **learner** | agent 0, the model under test |
| **partner** | agent 1, the simulated other person |
| **`partner_key`** | the partner's hidden spec — after the redesign: `internal_state`, `movement_conditions`, `hardening_triggers` |
| **`partner_goal`** | **why they say no** — what's materially at stake; visible, they state it |
| **`internal_state`** *(new)* | **why they'd say yes** — what is true of them psychologically; invisible, they have no words for it. Written as a **state**, never as a requirement (§7.3) |
| **`movement_conditions`** | learner actions that would address the state — a **witness** that a route exists; logged, never graded |
| **`hardening_triggers`** | learner moves that deny or worsen the state; the partner digs in |
| **`surface_misdirection`** | the partner's stated cover story — **deleted** in the redesign (§7.6) |
| **`cost_coupling`** | what the key cost the learner — **deleted**, replaced by the oracle gate (§7.6) |
| **`key_mechanism`** | one of 5 tags: `reactance` (pressure hardens, restoring choice unlocks — Brehm) · `face_needs` (needs a face-saving account — Brown & Levinson) · `validation_before_change` (can't consider change until they feel understood — motivational interviewing) · `procedural_voice` (accepts worse outcomes given genuine voice in the process — procedural justice) · `reciprocity_disclosure` (unlocked by the learner's costly first move — Cialdini) |
| **`too_easy` / `frontier` / `beyond_frontier`** | solved on attempt 1 / failed cold but improved / failed cold and never improved |
| **LP → recoverability** | improvement across in-context retries; renamed per §4.4 |
| **`leaked`** | partner named or hinted a hidden condition/trigger (13.0%) |
| **`early_yield`** | partner softened before any condition was met (33.3%) |
| **`ignored_trigger`** | a trigger fired and the partner failed to harden (8.7%) |
| **phase 0** | running the 90 raw seeds through the K-loop against a given learner, to get each seed's band. Required input for generation; also Row 0 of the matrix. Done once for `gpt-5-mini` (`results/expel_phase0_Base90_ExpeL`: 63/22/5). |
| **oracle check** | run a strong model *given the answer*; if it can't win, the scenario is broken. Not the same as cold pass — see §9.2. |
| **ExpeL / Reflexion** | in-context experience: write a reflection after a failure, next attempt reads it. No weight updates. |
| **Thompson sampling** | pick the parent scenario whose children have looked most useful. *Being dropped.* |
| **POET / MCC** | open-ended env-generation methods whose "minimal criterion" admits a level only if some agent can already do it |
| **Countdown** | arithmetic puzzle generated from a hidden solution; the solution is a *witness*, not the required answer |
