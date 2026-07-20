# Social OMNI-EPIC — Project Canon (Single Source of Truth)

*Status: 2026-06-12. Code frozen for the Gen-90 run (Patches 1–11 applied; see §4, §9). This
document supersedes the framing in all prior docs and is self-contained: a reader who has seen
nothing else should finish it understanding both the one-sentence idea and the implementation
down to the field level. Where this conflicts with an older doc, this wins.*

---

## 0. The one sentence

**To make an LLM better at social interaction without touching its weights, generate the
scenarios it is *just barely failing* — identified not by guessing difficulty but by *measuring
whether the model improves when given a chance to learn* — and feed that hard-won experience back
as in-context guidance.**

The unifying quantity is **learning progress (LP)**: a scenario is at the model's social frontier
*iff* the model fails it cold but improves across in-context retries. That one signal does triple
duty — it **defines** the frontier operationally, it **selects** which scenarios to evolve, and the
experience it certifies **is** the transferable artifact.

---

## 1. The succinct insight (the hard thinking)

The user's real question — *what is our NeurIPS-grade insight, and why doesn't the paper feel
intuitively simple?* — deserves a direct answer. Here it is.

### 1.1 The trap the dual-loop paper fell into

The Final Report had **two** headline contributions (an open-ended curriculum *and* a Skills
Chronicle ICL mechanism) plus a third construct (social-interestingness). Three ideas competing
for the spotlight is why it doesn't land as one clean thing. Great papers have **one** sentence a
reader remembers. We need to choose ours and demote the rest to supporting machinery.

### 1.2 Three candidate framings, and the winner

- **(A) "Self-calibrating open-ended curriculum for social skill."** True, but *not novel on its
  own* — POET/UED already self-calibrate curricula for RL. Saying "we did it for social" is a
  domain-transfer paper, not an insight paper.
- **(B) "Construct social difficulty from specified mechanisms (a hidden partner key), so
  solvability is guaranteed and difficulty is dialed."** This is the most *engineered* part, but
  "we built a difficulty grammar" is a methods contribution, not an idea — and it's the part the
  PI is (rightly) ambivalent about, because a hand-specified mechanism library reads as
  restrictive rather than deep.
- **(C, WINNER) "Don't define social difficulty — measure it. Learning progress is the
  operational definition of 'at the frontier,' and it is simultaneously the curriculum's selection
  signal and the certificate on the experience worth keeping."**

(C) is the strongest because it converts the field's central embarrassment — *social
intelligence is ill-defined and LLM social evals are gameable* — into the method. You cannot
predict, a priori, which scenario a given model finds hard (it is model-relative and
unobservable). So **stop predicting; measure.** A model that fails cold but improves with
experience is, by construction, standing at its own frontier. This is honest, general, and it is
exactly the principle that survived every debugging decision in this project (Patch 10: "operators
set *direction*; LP *verifies* where the child landed"; Patch 11: "if neither attempt makes
objective progress, there is no learning — answer no_difference").

### 1.3 Resolving the loop-vs-key tension (the PI's instinct is correct)

The PI's instinct — *the contribution is the loop, not the partner-key schema* — is right, and
here is the precise relationship:

> **The partner key is not the contribution; it is the precondition that makes the contribution
> measurable.** Learning progress is only meaningful if the task is *solvable in principle* — you
> cannot measure "improvement toward success" on a task with no success. The partner key
> guarantees solvability-by-construction (a hidden but satisfiable set of conditions), so LP
> measures *skill acquisition* rather than *noise on an impossible task*. It is scaffolding for the
> measurement, the way a calibrated thermometer is scaffolding for a temperature claim — essential,
> auditable, but not the discovery.

So: **lead with LP-as-frontier-measurement and the evolutionary loop it drives; present the
partner key as the instrument that makes LP well-defined; present ExpeL in-context memory as the
(pre-existing, not-novel) transfer mechanism we plug in.** That is one idea with two supports —
the simple shape great papers have.

### 1.4 The sentence, three lengths

- **Title-length:** *Measure the frontier, don't define it.*
- **Abstract-length:** *Social difficulty is continuous and model-relative, so it cannot be
  specified in advance; we use learning progress — does the model improve when allowed to learn
  in-context? — as an operational, model-relative definition of the social frontier, and as the
  selection signal for an evolutionary curriculum whose certified experience transfers back to the
  model at inference time.*
- **Course-length (this is "Curiosity in AI"):** learning progress is a **curiosity signal**
  (Oudeyer & Kaplan; Schmidhuber): the system is intrinsically driven toward scenarios of maximal
  *learnability*, which in the social domain is exactly the frontier of competence.

### 1.5 Is "open-endedness" the contribution?

Partly. Open-endedness is the *search strategy* (an ever-expanding bank, never converging) that
*exploits* the LP signal. The contribution is the **coupling**: LP makes social difficulty
measurable; open-ended evolutionary search makes the curriculum self-extending; ExpeL makes the
result usable without weight updates. None alone is new; the **thread that ties them** — LP as the
single currency for difficulty, selection, and transfer — is.

---

## 2. What the system does (one screen)

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │  ARCHIVE (a MAP, not a trophy case): 90 SOTOPIA seeds +       │
                 │  every completed generated scenario, each a Thompson arm      │
                 └─────────────────────────────────────────────────────────────┘
   Thompson select an anchor  ──▶ operator from anchor.classification:
        (Beta posteriors over          too_easy→escalate · frontier→lateral · beyond→relax
         "did my children learn?")
              │
              ▼
   GENERATE a fresh-surface child (LLM), inventing/мутating a hidden partner_key
              │
              ▼
   ADMISSION GATES  (single ownership, all fail-closed):
        coherence (key valid, no leak) · surface-novelty (no name reuse / clone)
        · embedding diversity (vs whole archive) · MOI worth-ranker (3 candidates)
              │
              ▼
   EPISODE K-LOOP (K=4 attempts, ExpeL within-episode reflexion as memory):
        attempt 1 cold → if solved: TOO_EASY (fast path)
        else reflect → retry → … ; partner role-plays the hidden key (holds it, hardens on triggers)
              │
              ▼
   MEASURE LP (cross-lab, key-blind pairwise judge: attempt 1 vs later, both orders)
        terminal_success = GOAL≥7 ∧ REL≥0 ∧ key_check_passed
              │
              ▼
   CLASSIFY:  too_easy (solved a1) · frontier (failed a1, LP>0) · beyond_frontier (failed a1, LP=0)
              │
              ▼
   UPDATE the ANCHOR's Beta posterior with LP pseudo-votes (Thompson learns which
   anchors breed frontier children) ; ADD child to archive as a new selectable arm
              │
              ▼
   Repeat until 90 completed generated scenarios.  The bank IS the contribution.
```

Then, offline: **extract ExpeL insights** from the bank's trajectories → an inference-time memory.
The headline evaluation compares that memory (**ExpeL-Generated90**) against ExpeL over the raw
seeds (**ExpeL-Base90**) and a no-memory **Vanilla** learner, on held-out SOTOPIA scenarios.

---

## 3. Intellectual lineage (threads we build on)

| Thread | Key refs | What we take | What we change / add |
|---|---|---|---|
| **Open-ended curricula via interestingness** | OMNI, OMNI-EPIC (Faldor, Zhang, Cully, Clune 2025) | the loop that generates tasks "learnable *and* interestingly novel"; FM-as-model-of-interestingness | replace physical RL envs with **social scenarios**; replace aesthetic-only interestingness with **frontier-ness measured by LP** |
| **Learning progress as intrinsic motivation / curiosity** | Oudeyer & Kaplan; Schmidhuber | LP as the curiosity signal — drive toward maximal *learnability*, not novelty or error | use LP as the *operational definition of the social frontier* and the curriculum's selection currency |
| **Quality-Diversity & minimal-criterion coevolution** | POET (Wang et al.); MAP-Elites; Novelty Search; MCC (Brant & Stanley) | archive-as-map; minimal criterion (POET/MCC); parent→child lineages | minimal criterion = **positive LP** (failed cold ∧ improved); mutation = FM structural edit over a theory-grounded grammar |
| **Unsupervised Environment Design** | PAIRED (Dennis 2020); PLR (Jiang 2021); regret-based ED (Parker-Holder 2022); MAESTRO (Samvelyan 2023) | the question "which environments are worth training on?"; difficulty as regret/learnability | answer with **measured LP** rather than predicted regret; no protagonist/antagonist game |
| **Parent selection rewarded by descendants' progress** | Darwin-Gödel-style self-improvement | Thompson over anchors, posterior updated by *children's* learning outcomes | anchors are scenarios; "fitness" = breeds frontier children |
| **Social simulation & evaluation** | SOTOPIA (Zhou 2024); SOTOPIA-π (Wang 2024); SOTOPIA-RL (Yu 2025) | seed bank, 7-dim Sotopia-Eval, dyadic goal-conflict format | we are the **weight-free, self-calibrating** alternative to π (QLoRA) and RL (reward design) |
| **In-context experiential learning** | Reflexion (Shinn 2023); ExpeL (Zhao 2024) | verbal RL within an episode (Reflexion); cross-task insight extraction (ExpeL) | ExpeL over a **self-generated frontier curriculum** instead of a fixed task set; **no** cross-lineage chronicle inheritance (paused) |
| **Social World Models & the ToM critique** | Zhou, Sap et al. 2025 (Social World Models); Sap 2022 (Neural ToM); Ullman 2023; Hu/Sosa/Ullman 2025 | the premise that LLMs hold a partial "social world model" with real failure modes | we *probe and grow* that model at its frontier rather than benchmark it statically |
| **Social-science grounding of difficulty** | Brown & Levinson 1987 (face); Dillard 1989 (primary/secondary goals); reactance, validation-before-change, etc. | the outcome/constraint/shortcut goal grammar and the 5-mechanism partner-key library | difficulty is **theory-grounded and auditable** (every key instantiates a documented phenomenon), not free-form "realism" |
| **Evaluation-validity critiques (the honest threats)** | "Misleading Success of Simulating Social Intelligence in LLMs"; "Can LLMs Keep a Secret?" | the warning that LLM-simulated social eval is gameable (pushover partners, lenient judges) | our **cross-lab judge**, **key check**, and planned **human sub-eval** are the direct mitigations; see §8.1 |
| **AI-mediated social practice (deployment grounding)** | Louie et al. 2024 (Roleplay-DoH), 2026 (counselor upskilling) | the legitimate use-case: a *sparring partner* for human skill training | positions the work where simulated practice is the point, not a claim about real-user chat |

---

## 4. History: how we got here

**Dual-loop era (→ Final Report, EDUC 234).** Difficulty was calibrated by an **outer loop**
(a `scenario_editor` tightened a scenario whenever the agent solved it first-try) and learning by
an **inner loop** (Skills Chronicle: reflect → abstract a `when/do/except` principle → retrieve
next attempt; with **chronicle inheritance** along lineages). Success was a **binary AND-gate**
(outcome ✓ ∧ constraint ✓). Construct: **social-interestingness** = INTERESTING ∧ LEARNABLE
collapsed (a scenario a capable LLM can't solve on priors but can solve with ICL).

**Why it failed (the diagnosis that drove everything).** Runs were **bimodal** — scenarios were
solved-first-try (discarded) or never solved (archived). The fail-then-succeed band (the entire
scientific claim) was nearly empty. Causes: (1) difficulty is a property of *(scenario, partner
policy, evaluation)* but only the scenario was calibrated — the "concede nothing" partner was a
cliff, not a dial; (2) the binary AND-gate destroyed the gradient; (3) single-rollout calibration
misclassifies a true-50% scenario half the time, and there was no path *back* from beyond the
frontier. A quieter failure: preference-tuned learners refuse the "tempting shortcut" (won't
blackmail), so failures were "partner refused," not "constraint violated."

**Phase-2 redesign (the current architecture).** Five moves: (1) difficulty moves into a **hidden
partner key** → solvable-by-construction, hard by discoverability/incongruence/cost; (2)
calibration signal becomes **learning progress (LP)**, a cross-lab pairwise judge; (3) the
per-scenario edit loop is **deleted** — calibration is now **selection pressure** (Thompson over
anchors) + **directed mutation** (escalate/relax/lateral); (4) success becomes the standard,
comparable **GOAL≥7 ∧ REL≥0** plus a **key-aware check**; (5) **Phase 0** calibrates the 90 seeds
to warm-start priors.

**Gen-90 launch decisions.** **The bank is the contribution.** Skills Chronicle + inheritance are
**paused** (`use_expel_memory: true`): in-episode memory is **ExpeL reflexion**, attempt 1 always
cold, so **LP is a stationary property of (scenario × base-learner × reflexion)**. Stopping = **90
completed** generated scenarios (too_easy ∪ frontier ∪ beyond). Headline comparison:
**ExpeL-Generated90 vs ExpeL-Base90 vs Vanilla**.

**Patches 1–11 (pre-freeze hardening; see the per-patch docs).**
- *Contract patches (1–7):* judge strictness on the gating goal-score; partner_goal leak scoping;
  off-channel artifact bans; sensor-form (not demand-form) movement conditions; artifact regex
  over key fields; cost-survivability; resistance-phrase variation.
- *Patch 10:* operators demoted to **direction-setters** with a universal **fresh-surface**
  mandate; the embedding diversity gate now owns surface novelty for **every** operator (the old
  `key_delta` path is deleted); a `surface_novelty_check`; a (now warn-only) direction-sanity
  monitor.
- *Patch 11:* LP judge answers **no_difference when neither attempt makes objective progress**
  (kills "politeness-frontier" inflation); key check treats an **offered/available** proscribed
  move as tripping the trigger. Both bias *against* the hypothesis (frontier/solved rarer).

---

## 5. Core definitions (precise)

- **Frontier (operational).** A scenario is at the learner's frontier iff it **fails attempt 1**
  (cold) **and has LP > 0** (improves across in-context retries). This is a **Minimal Criterion**
  (POET/MCC): a two-sided bar, neither trivial nor impossible.
- **Learning progress (LP).** `LP = improved_votes / total_votes ∈ [0,1]`, from a **key-blind**
  cross-lab judge comparing attempt 1 vs each later attempt in **both** presentation orders;
  order-swap disagreement collapses to `no_difference` (kills position bias). LP is coarse per
  scenario but Thompson aggregates it across a lineage.
- **Social-interestingness (legacy construct, now decomposed).** Originally INTERESTING ∧
  LEARNABLE. Now: *learnability* = measured by LP (not predicted); *worth/aesthetic* = the MOI
  ranker; *structural difficulty* = guaranteed by the partner key. The single word is retired in
  favor of the three measurable axes.
- **Terminal success.** `GOAL ≥ 7 ∧ REL ≥ 0 ∧ key_check_passed`. The third conjunct only binds on
  keyed (generated) scenarios; seeds have no key so it is vacuously true (this asymmetry is logged
  as `success_label`).
- **The three bands.** `too_easy` (solved attempt 1 → archived, LP≡0, full down-weight); `frontier`
  (failed attempt 1, LP>0 → the viable curriculum artifact); `beyond_frontier` (failed attempt 1,
  LP=0 → archived as a negative example *and* re-selectable for `relax`).
- **The archive is a map, not a trophy case.** Every completed scenario is kept regardless of band;
  beyond_frontier and too_easy are not failures, they are coordinates that shape future search.

---

## 6. The system in detail (implementation)

### 6.1 Scenario schema + partner key
A generated scenario (`SocialScenario`, `data_models.py`) is **role-asymmetric by construction**:
- **Agent 0 = learner**, holds the structured goal triple **outcome / constraint / shortcut**
  (Dillard primary goal / secondary goal / Brown-Levinson face-threatening act). The shortcut is
  surfaced privately as tempting `<extra_info>` and must mechanistically trip a hardening trigger.
- **Agent 1 = partner**, holds a natural-language `partner_goal` **plus the hidden `partner_key`**:
  - `key_mechanism` — one of {reactance, face_needs, validation_before_change, procedural_voice,
    reciprocity_disclosure} (closed, auditable, swappable; no completeness claim).
  - `movement_conditions` — concrete, behaviorally checkable things **the learner does** that make
    the partner genuinely shift. Authored in **sensor form** ("something shifts when the learner
    does X"), never **demand form** ("the partner insists on X") — else the partner enumerates them
    in turn 1 and destroys discoverability (Patch 4/Patch 10).
  - `hardening_triggers` — learner moves that lock the partner (reactance in action).
  - `surface_misdirection` — the partner's *stated* objection vs. its real lever (the
    discoverability dial; may appear publicly).
  - `cost_coupling` — what satisfying the key costs the learner's *own* stated goal (must make the
    outcome *harder/partial*, never *strictly unreachable* — survivability, Patch 6).
- `target_agent_idx = 0` always; lineage fields `parent_id`, `parent_classification`,
  `parent_scenario`, `root_seed_env_pk`, `lineage_depth`, `ancestor_ids`; descriptive
  `mutation_operator`, `mutated_slots` (self-report only, **not** a gate — Patch 10).

### 6.2 The loop (entry: `scripts/run_curriculum.py::_run_one_scenario`)
1. **Thompson-select** an anchor (`archive.thompson_select()`); operator from
   `anchor.classification`.
2. **Context:** lineage-excluded KNN exemplars + KNN-nearest beyond_frontier negatives.
3. **Generate** a `gen_batch_size`-candidate batch (`task_generator.generate_batch_from_archive`),
   each with a fresh surface and an invented/mutated key.
4. **Free key-leak filter** → **MOI worth-rank** (best-first).
5. **Admission walk** (§6.7): embed → coherence(+patch) → surface-novelty → diversity; admit the
   first survivor; stamp lineage (incl. `parent_classification`, `parent_scenario`), resolve the
   `_pX` perspective placeholder, designate target agent.
6. **Live-write** a `bank/generated/<id>.json` stub immediately; append turn-by-turn during the
   episode; finalize on completion (crash-safe, status `in_progress`→`completed`).
7. **Episode K-loop** (§6.6) → **LP** (§6.4) → **classification**.
8. **Sequential main-loop step:** update the anchor's Beta posterior with LP pseudo-votes
   (`record_child_outcome`), add the child to the archive, checkpoint, flush aggregates.

### 6.3 Operators — direction-setters (Patch 10)
A single shared `_OPERATOR_PREAMBLE` (injected once) carries the **fresh-surface** mandate: new
names/setting/occupation/stakes; never reuse the parent's; if the parent is a keyless seed, invent
the key from scratch. Each operator then sets only a **difficulty direction**:
- `escalate` (parent too_easy) — *harder*: deeper misdirection / higher cost / more
  counter-to-RLHF conditions. **Survivability clause** kept verbatim.
- `relax` (parent beyond_frontier) — *easier* along the dimension named by
  `beyond_frontier_diagnosis`.
- `lateral` (parent frontier) — *same* difficulty, different mechanism/asymmetry/relationship.

There are **no slot-preservation contracts** — that was Phase-1 difficulty micro-management. Where
the child lands is **LP's verdict**, accumulated into the anchor's posterior; an "escalate" that
fails to escalate produces a too_easy child, charges its anchor, and selection moves on.

### 6.4 LP judge (`lp_judge.py`, Patch 11)
Key-blind; compares attempt 1 vs each later attempt, both orders. **Patch 11 floor:** if *neither*
attempt makes meaningful progress toward the **stated objective**, tone/rapport/graceful-exit
differences do **not** make one better → `no_difference`. The word "stated objective" is
load-bearing: scenarios whose goal is itself relational (repair a friendship) still count
relational progress as objective progress. Errored votes are tagged and one-shot re-run; all-error
→ quarantine (no posterior charge for infrastructure failure).

### 6.5 Key check (`episode_runner.py`, Patch 11)
A separate temperature-0, fail-closed judge run **every attempt** (its verdict feeds reflection):
(1) did the actor *genuinely* satisfy ≥1 movement condition (not merely mention it)?; (2) did the
actor trip a hardening trigger never subsequently repaired — where **offering/presenting/making
available** the proscribed move counts as tripping it (Patch 11). `key_check_passed = (≥1
condition met) ∧ (no unrepaired trigger)`. This is *why* a `GOAL=10` attempt can still be
`beyond_frontier` (the partner-LLM caved, but the learner never honored the modeled psychology):
the key check is the guard against rewarding extraction-by-pushover.

### 6.6 Episode memory — ExpeL within-episode reflexion (no chronicle, no inheritance)
`use_expel_memory: true`. Attempt 1 cold; after each failure, `_reflect` writes a Reflexion string;
attempt *k* sees `_format_reflections` of attempts 1…*k*−1. The synthesized memory is
`skills_final_md` (raw material for extraction); it **never** enters generation prompts and is
**never** inherited across lineages. The legacy Skills Chronicle path remains behind
`use_expel_memory: false` for revivability only.

### 6.7 Gates — single-axis ownership, all fail-closed (Patch 10)
- **Solvability** → coherence gate (key exists/consistent/never leaks/cost real; quarantine on FM
  error; ≤2 patch retries; **break** to next anchor on rank-0 failure).
- **Surface novelty** → `surface_novelty_check` (deterministic: no anchor-name reuse; not a clone)
  **+** embedding diversity vs the whole archive (threshold 0.92). Universal across operators —
  this closes the old escalate-on-seed wide-open channel.
- **Worth** → MOI, a *ranker* over the batch (a ranker can't saturate to "yes"); fail-open.
- **Learnability** → measured by LP, never predicted by a judge.

### 6.8 Thompson, archive, priors (`archive.py`)
Each scenario is a Beta arm; `thompson_select()` = argmax of samples. A completed child charges its
**anchor**: too_easy → (0, K_VOTES_EQUIV); frontier → (improved, total); beyond → (0, total or
K_VOTES_EQUIV); generation_failed/discard → soft (0, 1) penalty (the emergent anti-mill brake).
Children inherit a **mass-capped** warm-start prior from the parent (`child_prior_mass=4.0`) so deep
lineages stay responsive to their own first votes. **Phase-0 seeding:** the 90 seeds carry phase0
classification + LP as *metadata*; classification picks the first operator; LP enters Thompson only
as a **soft asymmetric prior** (frontier → Beta(2,1), else Beta(1,1)); `alpha_votes=beta_votes=0`
(seed self-LP never becomes votes).

### 6.9 Bank, outputs, resume
`archive_latest.json` is the **canonical** resume state (Thompson posteriors); folders are
exports. `bank/generated/<id>.json` = every completed scenario = "Generated-N"; `trajectories.json`
+ `summary.json` rebuilt from the bank each checkpoint (resume-safe). Resume = same `run_name`.
`PROVENANCE.md` records the judge-version boundary (all Gen-90 children are post-Patch-11; phase0
seed priors are pre-patch — accepted, see §8).

---

## 7. Evaluation design

**Claim (strongest evaluable form):** *an open-ended, LP-calibrated social curriculum yields
in-context experience that improves a frozen LLM's held-out social performance more than the same
ICL mechanism applied to a fixed seed distribution — with no weight updates.*

**Conditions (controlled ablation isolating the *curriculum*, holding the ICL mechanism fixed):**
1. **Vanilla** — frozen learner, no memory.
2. **ExpeL-Base90** — ExpeL insights extracted from the raw 90 SOTOPIA seeds.
3. **ExpeL-Generated90** — ExpeL insights extracted from our 90-scenario bank.
4. *(separate workstream)* Random90; *(reference)* SOTOPIA-π fine-tuning (not our claim to beat).

**Protocol invariants:** one **frozen partner engine** across conditions; eval partners are vanilla
(no key, no chronicle — the claim is transfer to *standard* tasks); **cross-lab judge** (different
lab than the learner, breaks the self-eval monoculture); report **deltas vs. vanilla** (cancels
judge leniency + partner agreeableness); contamination check (ID + embedding overlap, seeds/bank
vs eval); per-condition compute accounting (the curriculum's budget is a superset of Base90's).

---

## 8. Open tensions & honest self-critique

These are the questions a strong reviewer (and the PI) will press. State them; don't hide them.

### 8.1 "Why pure LLM simulation with no human grounding?"
We are **not** ungrounded: the **seeds** are human-authored SOTOPIA scenarios, the **evaluation**
is Sotopia-Eval (validated against human judgment), and the **difficulty grammar** is grounded in
social-science theory (Brown & Levinson; Dillard). What *is* simulated is the partner's behavior
and the generated surface — and there is a real, named threat ("Misleading Success of Simulating
Social Intelligence," literally in our reading list): simulated social eval is gameable by pushover
partners and lenient judges. Our mitigations are concrete and already load-bearing: the **key
check** (a goal score from a caved partner doesn't count), the **cross-lab judge**, the **Patch-11
LP floor**, and a planned **human sub-eval on the hardest slice**. Honest positioning: the
defensible deployment is **AI-mediated practice** (a sparring partner for negotiation/counseling
training), not a claim about real-user conversation, and we **do not** claim LLM-partner strategies
transfer to human partners without further study.

### 8.2 "Does the partner key assume one correct solution? People are flexible."
The check is `≥1 of N` conditions, judged **semantically** (spirit, not string), so multiple
flexible executions pass. But the conditions are a **finite authored list**, so a genuinely
effective *unlisted* lever is a **false negative** — the known cost of "difficulty from a specified
model." Reframe: the key is the **partner agent's own programmed psychology** (the partner
role-plays the same key), so the check asks "did you unlock *this* modeled person via *their* real
levers?", not "is this the only way humans solve this." Flexibility lives at the **population
level** (5 mechanisms × surface variation × 90 scenarios), not within one key. *Lever if we want
more openness:* add a mechanism-level fallback clause ("also pass if the actor genuinely satisfied
the *spirit* of the `key_mechanism`") — trades identifiability for openness; deferred.

### 8.3 The LP-vs-success gap (real, and partly the point)
A scenario can be `frontier` by LP yet never reach GOAL≥7 (improving but unsolved). These yield no
success trajectory for ExpeL extraction (`n_frontier_unsolved`), so they are curriculum-valuable
but extraction-poor. Watch the rate in `summary.json`; if frontier is dominated by goal≈0
near-misses, the generator is setting goals too easy relative to their keys (a calibration finding,
not a bug). Conversely a `GOAL=10` can be `beyond_frontier` when the key was never honored — the
key check correctly overriding a lenient goal score.

### 8.4 The escalate calibration issue (live, in the current run)
With direction-only operators, escalate off `too_easy` currently **overshoots** to
`beyond_frontier` (and occasionally undershoots to `too_easy`), rarely landing frontier — so the
Gen-90 distribution skews beyond. This is partly real (the keyed success bar is genuinely high,
§3 of the gen90 read) and partly a step-size issue. The self-correcting answer is the lineage
cycle (beyond → relax walks back toward frontier; we observed a depth-2 relax child land frontier).
The pre-staged-but-unapplied fix is an escalate **step-size clause** ("aim just above the parent,
target the frontier, not maximal difficulty") + an overshoot arm on the direction monitor — to be
applied **only before a fresh run**, never mid-bank (provenance). The PI has chosen to run the
current bank to N=90 as-is and treat it as data.

### 8.5 "Is it simple enough?" — the meta-risk
The biggest threat to the paper is **conceptual clutter** (partner-key fields, gates, operators,
two judges). §1 is the antidote: in the writing, **the partner key, gates, and operators are
*Methods* details in service of one idea** (LP-measured self-calibration). If a reader can restate
the paper as "measure the frontier with learning progress, evolve toward it, feed the experience
back," the framing is working; if they restate it as "a system with a partner-key schema and three
operators," it is not.

---

## 9. Current state & immediate next steps

- **Frozen for Gen-90.** Patches 1–11 applied; direction-sanity is **warn-only** (won't halt).
  The `gen90_expel` run is in progress (phase0-seeded, `use_expel_memory=true`,
  `batch_size` per launch, `stopping.N=90`). Bank is uniformly post-Patch-11 (see `PROVENANCE.md`).
- **Known live issue:** escalate overshoot → beyond-heavy distribution (§8.4); accepted for this
  run.
- **After the bank completes:** (1) ExpeL `extract` over all 90 trajectories → `insights.json`
  (ExpeL-Generated90); (2) contamination check vs eval-150; (3) run the 3-condition eval with the
  frozen partner engine + cross-lab judge; (4) post-hoc niching/coverage figure; (5) draft with the
  single-run limitation sentence (cite Henderson et al.).
- **Deferred (documented, not implemented):** niche-first Thompson; cross-lineage chronicle
  inheritance; mechanism-library extension; mechanism-level key-check fallback; multi-seed variance
  runs; the escalate step-size fix (next fresh run only).

---

## 10. File map (where the truth lives in code)

- `scripts/run_curriculum.py` — the loop, gates, Thompson step, bank writes, direction monitor.
- `social_omni_epic/task_generator.py` — generation prompt, `_OPERATOR_PREAMBLE`, `_EDIT_INTENTS`,
  partner-key schema + authoring rules, `analyze_too_easy` / `analyze_beyond_frontier`.
- `social_omni_epic/curriculum.py` — `run_episode_two_loop` (K-loop, reflexion, LP, classification).
- `social_omni_epic/episode_runner.py` — episode rollout, partner prompts (keyed/native), Sotopia-
  Eval judge, key check, `clean_transcript`.
- `social_omni_epic/lp_judge.py` — pairwise LP judge (Patch 11 floor).
- `social_omni_epic/coherence_check.py` — coherence gate + fuzzy key-leak + artifact checks.
- `social_omni_epic/validation.py` — schema construction, `surface_novelty_check`, deprecated
  `key_delta_check`.
- `social_omni_epic/archive.py` — Beta arms, Thompson, priors, checkpoints.
- `social_omni_epic/expel_export.py` — bank records, trajectories.json, summary.json, live-write.
- `data/sotopia_90_seeds.jsonl` — the human-authored seed bank.
- **Docs lineage (chronological):** `curriculum_loop.md` → `research_framing.md` →
  `scenario_design_sweet_spot.md` → Final Report PDF → `PROJECT_STATE_AT_PROJREPORT.md` →
  `PHASE2_REFERENCE.md` + `IMPLEMENTATION_SPEC_phase2.md` → `Revised_Spec_Gen90_Launch.md` →
  contract patches → `pre_run_final_Patch.md` (Patch 10) → `patch11_lp_judge_keycheck.md` → **this
  canon**.
