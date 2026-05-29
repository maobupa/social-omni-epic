# Social Omni Epic: Intrinsically-Motivated Open-Ended Social Curriculum Generation with Persistent Inference-Time Knowledge Accumulation

**Huijun Mao, Huanxing Chen**
*Stanford University — PSYCH 240A / EDUC 234: Curiosity in Artificial Intelligence*

---

## 1. Motivation and Problem Statement

Large language models exhibit impressive social fluency in narrow, well-rehearsed interaction types but fail systematically on scenarios requiring strategic persistence, information asymmetry management, and adaptive goal pursuit under social pressure [1]. The dominant approaches to improving social intelligence either fine-tune model weights on curated interaction data [2] or provide structured representational scaffolding at inference time [3]. Both approaches share a fundamental limitation: they operate on fixed distributions of social scenarios, either encoding that distribution into parameters or assuming it is adequately represented by existing benchmarks.

This matters because social intelligence is inherently open-ended. The space of scenarios requiring sophisticated social reasoning is not bounded by any fixed dataset, and an agent trained or evaluated only on that dataset will exhibit systematic blind spots wherever the real world diverges from it. SOTOPIA-hard [1] surfaces this problem clearly: even GPT-4 achieves significantly lower goal completion than humans on challenging scenarios, and the gap is concentrated precisely in the scenarios that require combining multiple social reasoning demands simultaneously.

The core hypothesis of this project is: **an agent whose knowledge of social strategy is accumulated across a curriculum of open-endedly generated, intrinsically-interesting social scenarios will generalize more robustly than an agent whose knowledge comes from a fixed scenario distribution** — and this can be achieved entirely at inference time, without touching model weights.

Two prior lines of work make this hypothesis tractable. OMNI-EPIC [4] demonstrated that foundation models can generate open-ended physical task curricula that are simultaneously learnable and interesting, using a Model of Interestingness (MOI) to filter generated tasks. SOTOPIA [1] demonstrated that social scenario space can be procedurally generated and systematically evaluated. This project asks: can OMNI-EPIC's open-ended generation loop be extended to social scenario space, and can the resulting curriculum drive the accumulation of persistent, reusable social reasoning knowledge at inference time?

---

## 2. Background and Prior Work

### 2.1 SOTOPIA: Interactive Social Evaluation

SOTOPIA [1] is an open-ended simulation environment for evaluating social intelligence through interactive role-play. It contains 90 social scenarios (cooperative, competitive, mixed-motive) crossed with 40 richly characterized agents, producing a large combinatorial task space. Each episode assigns private social goals to two agents who interact in natural language to pursue those goals. Crucially, each agent's goal is private — unknown to the other agent — so agents must navigate genuine information asymmetry throughout the interaction.

Performance is evaluated per agent along seven dimensions by SOTOPIA-Eval: Goal completion (GOAL), Financial benefits (FIN), Relationship preservation (REL), Believability (BEL), Knowledge acquisition (KNO), Social rule adherence (SOC), and Secret-keeping (SEC). Scores are assigned individually to each agent, not to the episode as a whole — an episode can be a success for one agent and a failure for the other simultaneously.

SOTOPIA-Eval's reliability as an automated judge is non-uniform across dimensions. GPT-4-as-judge correlates well with human annotation on GOAL (r=0.71) and FIN (r=0.62) but poorly on SEC (r=0.22) and SOC (r=0.33) [1]. This non-uniformity is a structural feature of the evaluation oracle that must be explicitly accounted for in any system that uses SOTOPIA-Eval as a learning signal.

### 2.2 OMNI-EPIC: Open-Ended Physical Environment Generation

OMNI-EPIC [4] introduced a framework for generating open-ended task curricula for RL agents using foundation models. Its key components are: a Task Archive seeded with initial tasks, a Task Generator that proposes new tasks given the archive, a post-generation Model of Interestingness (MOI) that filters tasks by learnability and interestingness relative to similar archived tasks, and a Success Detector that determines whether an agent has mastered a proposed task. Successfully learned tasks enter the archive, driving open-ended expansion of the task space.

A pilot experiment adapting this framework to social scenario space — Social Omni Epic — across 200 generation iterations demonstrated that both architectural components are load-bearing in the social domain: removing the MOI filter reduces scenario coverage from 0.600 to 0.550, and removing the archive entirely reduces it to 0.530. These coverage properties establish that curiosity-driven filtering and archive-based context together produce meaningfully more diverse scenario exploration than either component alone.

### 2.3 Voyager: External Skill Libraries for Open-Ended Learning

Voyager [5] demonstrated that an LLM agent can accumulate reusable knowledge across open-ended episodes through a persistent external skill library — without any weight updates. Skills are executable JavaScript programs stored in a vector database and retrieved by cosine similarity at runtime. The quality gate is hard: skills are only stored after an independent verifier confirms task completion. Voyager established the key architectural principle this project extends: **inference-time performance can be improved by accumulating structured external knowledge across episodes rather than updating model parameters**.

The critical limitation Voyager faces in social domains is its verification oracle. Executable code either runs correctly or it does not. Social interaction success has no equivalent hard verification signal, which requires a fundamentally different approach to quality gating and knowledge storage.

### 2.4 Reflexion: Verbal Self-Reflection as Episodic Memory

Reflexion [6] demonstrated that agents can improve within episodes by accumulating verbal summaries of past failures as working memory, injected into subsequent attempts. Its key distinction from the present system is that reflection is episodic and local — verbal memory does not persist across episodes. This establishes the strongest memoryless baseline for inference-time improvement: how much can an agent improve through within-episode self-correction alone, without persistent cross-episode knowledge? This is Baseline 2 in the evaluation design (§5.1).

### 2.5 ExpeL: Experiential Learning from Trajectory Pools

ExpeL [7] demonstrated that a reflection module reading a pool of past trajectories — both successes and failures — can extract a set of natural language insights stored with frequency counts, importance scores, and provenance metadata. Retrieved at inference time, these insights improve performance on held-out tasks. ExpeL provides the direct ML precedent for this project's knowledge accumulation architecture and the primary schema reference for the skills chronicle. Related work on evolved prompt heuristics [8] similarly demonstrates that natural language critique of failure is a richer learning signal than scalar reward.

The key extension this project makes over ExpeL is structural: ExpeL operates on a fixed trajectory distribution drawn from a predetermined task set, whereas this project generates its own curriculum open-endedly. The quality of the accumulated knowledge is therefore a function of both the reflection architecture and the curriculum that feeds it — neither component is sufficient alone.

### 2.6 Cognitive Science Grounding

Three foundational frameworks justify the architecture:

**Tulving's episodic-semantic memory distinction** [9] establishes that humans do not retrieve raw episodic memories to guide behavior — they consolidate them into generalized semantic knowledge through an abstraction process. The meta-reflection module operationalizes exactly this consolidation: it transforms lossy episodic transcripts from multiple attempts into generalized semantic heuristics in the skills chronicle.

**ACT-R's production system architecture** [10] establishes that expert procedural knowledge is organized as condition-action rules that pattern-match against abstract situational features, not specific episodic details. This grounds the schema requirement that Condition fields be phrased in terms of abstract social dynamics, and that Guidance fields be phrased procedurally rather than declaratively. Declarative knowledge ("X causes Y") does not transfer across contexts the way procedural knowledge ("when X, do Y") does.

**Case-Based Reasoning** [11] provides the retrieve-reuse-revise-retain cycle that governs runtime behavior: a new scenario is addressed by retrieving structurally similar past cases, reusing their guidance, revising based on outcome, and retaining the result back to the archive. The skills chronicle is functionally a CBR case library where cases are social interaction episodes and solutions are behavioral heuristics.

**UCB1 for curriculum selection** [12] provides the principled mechanism for ensuring the archive is explored broadly rather than converging on a small set of frequently-chosen scenarios. UCB1's proven regret-minimization properties guarantee that no archived task is permanently neglected as the archive grows [13].

---

## 3. System Architecture Overview

The system operates as a closed loop with two interleaved processes: **curriculum generation** (Social Omni Epic loop) and **knowledge accumulation** (reflection and chronicle update). These are not sequential phases — each generated scenario that enters the archive also runs through the knowledge accumulation process, contributing entries to the skills chronicle that inform subsequent scenario runs.

```
┌──────────────────────────────────────────────────────────────┐
│                       TASK ARCHIVE                           │
│  (90 SOTOPIA seeds → grows open-endedly via UCB1 selection)  │
└───────────────────────┬──────────────────────────────────────┘
                        │ UCB1 selects Task X
                        │ + N similar tasks for proposer context
                        ▼
               ┌─────────────────┐
               │  TASK PROPOSER  │◄── MOI Filter + Verbalized Sampling
               └────────┬────────┘
                        │ Proposed scenario
                        ▼
               ┌─────────────────┐
               │   EVAL GATE     │ (Truly interesting?)
               └────────┬────────┘
                   Yes  │  No → regenerate
                        ▼
          Designate target agent by goal-similarity to Task X
          Inherit Task X's skills_final.md for target agent
                        │
                        ▼
               ┌─────────────────┐
               │   AGENT RUN     │◄── Target agent + inherited chronicle
               │                 │    Partner agent = vanilla GPT-4o
               └────────┬────────┘
                        │
           ┌────────────┼─────────────┐
           ▼            ▼             ▼
       Outcome 1    Outcome 2     Outcome 3
    (1st attempt  (fail → then  (fail all 5
       succeeds)    succeeds)    attempts)
           │            │             │
           ▼            ▼             ▼
      No reflection  REFLECTION   REFLECTION
      needed         MODULE ×≤5   MODULE ×≤5
                     (series,     (series,
                     cumulative)  cumulative)
                         │             │
                         ▼             ▼
                  META-REFLECTION (Success prompt)
                  META-REFLECTION (Failure prompt)
                         │             │
                         ▼             ▼
                  skills_final.md  (with HEURISTIC
                  and WARNING entries from all attempts)
                         │
                         ▼
                  SCENARIO_TITLE generation
                         │
                         ▼
              Add completed task to TASK ARCHIVE
              Update n_i, last_chosen, n_children
```

---

## 4. Module Specifications

### 4.1 Task Archive

The archive is seeded with SOTOPIA's 90 human-authored social scenarios. Each archived task stores the following:

**Scenario metadata:**
- Scenario description and agent goal descriptions
- `SCENARIO_TITLE` — pipe-separated string encoding social dynamic and target perspective (see §4.7)
- `SOCIAL_DYNAMIC` — left half of SCENARIO_TITLE, stored separately for independent retrieval
- `TARGET_PERSPECTIVE` — right half of SCENARIO_TITLE, stored separately for independent retrieval
- `GOAL_STRUCTURE` — COMPETITIVE / COOPERATIVE / MIXED, assigned programmatically from goal descriptions
- `INFO_POSITION` — INFORMED / UNINFORMED / SYMMETRIC from the target agent's perspective, assigned programmatically
- Target agent's abstracted goal description (one sentence, abstracting away scenario-specific details)

**Accumulated knowledge:**
- `skills_final.md` — the skills chronicle produced from running this scenario (linked by scenario ID)

**Selection bookkeeping (all updated programmatically):**
- `n_i` — number of times this task has been chosen as anchor task X
- `last_chosen` — iteration number of most recent selection as anchor task X
- `n_children` — number of descendant scenarios directly generated from this task

**UCB1-based task selection:** At each iteration, the system selects anchor task X by computing a selection score for every archived task:

```
score(task) = C × sqrt(ln(N) / n_i) - D × n_children
```

Where N is the total number of anchor task selections across all tasks to date, C controls the exploration-exploitation tradeoff (empirically tuned across C ∈ {0.5, 1.0, √2, 2.0}), and D penalizes tasks that have already generated many children to bias the loop toward less-explored branches of the archive tree. The task with the highest score is selected. This mechanism ensures underexplored and long-neglected tasks accumulate selection pressure over time, while tasks that have already spawned many descendants are deprioritized in favor of unexplored lineages.

The N similar tasks retrieved alongside task X (for Task Proposer context) are selected by SCENARIO_TITLE embedding similarity. They provide generative context to the Task Proposer but do not contribute to the new scenario's inherited skills chronicle.

### 4.2 Task Proposer Module

**Input:** Task X (the selected anchor task), N most similar tasks from the archive with their descriptions and success/failure logs.

**Function:** Propose the next learnable and interesting social scenario. Learnable means there exists a strategy that could succeed at the task. Interesting means the task explores social dynamics not already well-covered by the archive.

**Verbalized Sampling [4]:** The proposer is explicitly prompted to generate scenarios from low-probability regions of scenario space — interactions that a typical aligned LLM would not spontaneously generate. This operationalizes the curiosity-driven exploration principle: the system is biased toward scenarios at the frontier of its current competence, not toward scenarios it already handles well. Ablation 5 (§5.2) isolates this component's contribution by removing verbalized sampling while retaining the MOI filter.

**Model of Interestingness (MOI):** An LLM prompt evaluating candidate scenarios on two criteria simultaneously: (1) Is it socially interesting — does it require non-trivial reasoning about goals, beliefs, or norms? (2) Is it learnable — is there a discoverable strategy that would improve performance? The MOI compares the candidate against the M most similar archived tasks to enforce diversity. A scenario too similar to archived tasks fails the interestingness check regardless of its intrinsic complexity. Ablation 2 (§5.2) isolates this component's contribution.

### 4.3 Eval Gate

A separate LLM pass evaluates the proposed scenario for genuine interestingness before any agent is run on it. This is a lightweight binary pass/fail check functioning as a second filter after the MOI. Its purpose is to catch scenarios that passed the MOI mechanically but are degenerate in ways the proposer did not anticipate — trivially solved, socially implausible, or redundant with an existing archived task despite passing the similarity threshold. Failed proposals are returned to the Task Proposer for regeneration with the failure reason passed back as context.

### 4.4 Target Agent Designation and Chronicle Inheritance

**Target agent designation:** Each episode designates one agent as the target agent — the one whose skills chronicle is being built — and runs the other as a fixed vanilla GPT-4o partner, consistent with the experimental setup in S3AP [3] and SOTOPIA-hard evaluations [1].

For the 90 seed scenarios, the target agent is designated as Agent 1 by convention. For all generated descendant scenarios, the target agent is whichever of the two new agents has a goal description that embeds most similarly to Task X's target agent's abstracted goal description. This maintains coherent perspective through the lineage: the skills chronicle accumulates from a consistent structural vantage point across generations of descendant scenarios.

If the two agents' goal descriptions embed with equal similarity to Task X's target — a genuinely symmetric scenario — the target is designated as Agent 1 by convention and the symmetry is logged. Symmetric scenarios are informative: they indicate the generated task is cooperative or balanced in a way the parent may not have been, and the skills chronicle will naturally develop entries acknowledging this structural shift.

**Chronicle inheritance:** The new scenario inherits Task X's skills_final.md directly and only. The N similar tasks retrieved for the Task Proposer's context window do not contribute to the inherited chronicle — they serve the generation step and no further. Inheritance is a clean single-parent chain: each scenario's starting chronicle is exactly its direct parent's skills_final.md.

**Success evaluation:** Outcome 1/2/3 classification is determined exclusively by the target agent's GOAL and FIN scores against threshold. The partner agent's scores are logged as contextual metadata but do not gate the learning signal. This is consistent with SOTOPIA's per-agent evaluation design [1].

### 4.5 Agent Run

The target agent is a frozen LLM — no weight updates at any stage. At runtime it receives: the scenario description, its private social goal, and the inherited skills chronicle from Task X. The partner agent receives the same scenario description and its private social goal, with no skills chronicle.

Three outcomes are possible:

**Outcome 1 — Direct success:** The target agent passes SOTOPIA-Eval on GOAL and FIN dimensions on the first attempt. The inherited skills chronicle is validated without revision and becomes skills_final.md as-is. No reflection module is invoked.

**Outcome 2 — Fail then succeed:** The target agent fails at least once but succeeds within the 5-attempt ceiling. The Reflection Module is invoked after each failure in series. The Meta-Reflection Module (Success prompt) synthesizes across all attempts to produce skills_final.md.

**Outcome 3 — Fail throughout:** The target agent fails all 5 attempts. The Reflection Module is invoked after each failure in series. The Meta-Reflection Module (Failure prompt) synthesizes to produce skills_final.md containing predominantly WARNING-type entries documenting what failed and the likely structural reasons.

### 4.6 Reflection Module

**Input at attempt K:** The current skills chronicle (inherited from Task X, edited by reflections 1 through K-1), the task description, all transcripts from attempts 1 through K-1, all intermediate skills chronicle versions produced by prior reflections, all prior EditReasons, and the transcript from the most recent failed attempt K.

Retries run strictly in series with full cumulative context. Each attempt has strictly more information than the previous one. By attempt 5, the reflection module has seen every prior failure and every prior attempt to fix it. This is not branching — there is no parallel exploration of alternative revision strategies.

**Function:** Diagnose what went wrong in the most recent attempt given the cumulative history, and produce targeted edits to the skills chronicle. Operates in two steps within a single prompt.

**Step 1 — Diagnosis:** The module identifies which entries from the inherited skills_final.md were relevant to this scenario, whether they were applied by the agent, what specifically went wrong that existing entries did not anticipate, and whether any existing entry actively misdirected the agent's behavior.

**Step 2 — Edits:** For each entry touched, the module outputs the complete revised tag-block AND a `<EditReason>` tag justifying the change with specific reference to the failure transcript. For new entries, it must explain why no existing entry covered this case. The `<EditReason>` tag is read by the Adversarial Agent (§4.6.1) and stripped before the entry enters the final chronicle.

**Editing rules for the Condition field:**

The Condition field may be edited freely — narrowed, broadened, or rewritten — subject to two constraints. First, the abstraction constraint is hard: no proper nouns, specific occupations, or scenario-unique details. The adversarial agent enforces this. Second, broadening edits face conservative scrutiny proportional to the strength of evidence. Narrowing is low-risk and accepted when the EditReason cites specific evidence that the current scope is too broad. Rewriting at the same abstraction level is medium-risk. Broadening is high-risk: the EditReason must explicitly argue that the failure occurred under circumstances outside the current Condition's scope and cite specific transcript evidence. The adversarial agent applies stricter scrutiny to broadening edits.

Additionally, because descendant scenarios pass the MOI's diversity filter by exploring territory beyond the parent scenario, their Condition entries are expected to be at least as specific as the parent's. An adversarial agent check: if a broadened Condition would also describe the parent scenario's social dynamic, that is a signal the broadening is inappropriate.

**Other field editing:** All other fields may be edited freely with mandatory EditReason justification. The adversarial agent checks every EditReason for grounding in the failure transcript, not for direction of change.

**Support count update:** Support is incremented programmatically once per scenario episode — not once per attempt. Five retries within one episode increment Support by exactly 1 when the episode completes. Support is updated after the Meta-Reflection Module produces skills_final.md.

#### 4.6.1 Adversarial Stress-Testing

An adversarial LLM agent operates at two stages with different scopes:

**After each Reflection Module invocation:** Lightweight, targeted check on the specific edits proposed in that iteration. Checks: (1) Does the EditReason adequately justify the change with reference to the failure transcript? (2) Does the revised Condition remain appropriately abstract? (3) For broadening edits specifically: does the broadened Condition describe the parent scenario's social dynamic — if yes, flag as inappropriate. Edits failing this check are returned to the Reflection Module with specific critique.

**After the Meta-Reflection Module:** Broader consistency check across the entire resulting skills_final.md. Checks: internal contradictions between entries, synthesis drift from inherited chronicle, overall coherence, and whether the document's coverage is appropriately balanced between HEURISTIC and WARNING entries given the episode's outcome. This is a higher-bar check than the per-reflection check.

#### 4.6.2 SOTOPIA-Eval Reliability and Confidence Gating

Because SOTOPIA-Eval's oracle reliability is non-uniform, Confidence levels are assigned deterministically based on which dimension motivated the entry — not by LLM judgment:

| Dimension | Oracle Reliability (r with human) | Confidence Assigned |
|---|---|---|
| GOAL, FIN | > 0.60 | HIGH |
| REL, BEL | 0.45–0.56 | MEDIUM |
| KNO, SOC, SEC | < 0.40 | LOW |

**Confidence promotion:** LOW → MEDIUM when Support ≥ 5 with no contradicting failure logged against this entry. MEDIUM → HIGH when Support ≥ 10 with the same condition. This allows entries on less reliable dimensions to earn credibility through repeated corroboration across diverse episodes, partially compensating for oracle unreliability on any single episode.

**Confidence demotion:** Any failure in which the Reflection Module explicitly flags an entry's guidance as having actively misdirected the agent causes immediate one-level demotion. This is triggered by the EditReason — if an edit to an existing entry is accompanied by an EditReason citing active misdirection, the module also demotes the entry's Confidence by one level. Demotion is programmatic once the EditReason trigger is detected, not discretionary.

### 4.7 Meta-Reflection Module

The Meta-Reflection Module synthesizes across all attempts within a single scenario episode. It reads the full history of skills chronicle versions, all transcripts, all EditReasons, and the final outcome, and produces skills_final.md.

**The key distinction from the Reflection Module:** The Reflection Module is reactive and local — it responds to one failure at a time and produces targeted edits for the next attempt. The Meta-Reflection Module is synthetic and global — it reads across all attempts simultaneously and asks: given everything that happened, what is the most coherent and generalizable set of heuristics? It reconciles potentially contradictory edits from different reflection iterations, identifies which edits were noise and which were signal, and resolves contradictions into nuanced entries with appropriate conditional clauses in the Guidance field. An entry added after attempt 2 that was partially contradicted by what worked in attempt 4 is reconciled by the Meta-Reflection Module into a single entry with the appropriate exception clause — something neither individual Reflection Module call could produce because neither saw the full picture.

**Success prompt:** Emphasizes what worked, why it worked, and what conditions it applies to. Consolidates HEURISTIC entries from successful iterations while retaining WARNING entries from failed attempts as contextual contrast. The resulting skills_final.md contains both types, giving future agents both positive guidance and documented failure modes.

**Failure prompt:** Emphasizes structural reasons for failure across all attempts, what social dynamics made the scenario resistant to the strategies tried, and what a different approach might look like. Produces predominantly WARNING entries. These are not discarded — failure knowledge is informative for future scenario retrieval and agent behavior, and the skills chronicle is explicitly designed to carry both success and failure knowledge.

**Adversarial check:** After Meta-Reflection output, the adversarial agent runs the broader consistency check described in §4.6.1. Any claim in the new output that contradicts a claim in the inherited input must have an explicit EditReason before acceptance.

### 4.8 Skills Chronicle Schema

Each entry in skills_final.md uses tag-block structure: prose content inside lightweight markers, never JSON. This preserves natural language richness for LLM reasoning while enabling clean field extraction for embedding-based retrieval on the Condition field.

```
<Entry id="[scenario_id]_[entry_number]">

<Condition>
Abstract social dynamic pattern that makes this entry 
applicable. No proper nouns, specific occupations, or 
scenario-unique details. Phrased as a recognizable 
structural pattern that generalizes across surface variation.

Example: "When the party with a hidden budget constraint 
is being anchored by an opponent who is unaware of that 
constraint and is proposing prices above the hidden ceiling."
</Condition>

<Guidance>
1. [Default behavioral instruction — the general case]
2. [Exception: when [more specific condition], 
   do [override] instead]
3. [Exception: when [even more specific condition], 
   do [override] instead]
Note: Later clauses take precedence over earlier ones 
when their conditions apply.
</Guidance>

<Type>HEURISTIC | WARNING</Type>

<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>

<Confidence>HIGH | MEDIUM | LOW</Confidence>

<Support>[integer — number of distinct scenario episodes 
in which this entry was present in the inherited chronicle. 
Incremented once per episode, not once per retry attempt.]
</Support>

<Provenance>[scenario_ids and iteration numbers of the 
episodes that generated or corroborated this entry]
</Provenance>

</Entry>
```

**Field justifications:**

- **Condition:** Grounded in ACT-R's production system requirement that condition-side rules abstract over surface variation [10]. The abstraction mandate is enforced by the adversarial agent. Retrieval at evaluation time uses the Condition field embedding as the primary matching signal.

- **Guidance — clause structure:** Grounded in legal drafting conventions and ACT-R's defaults-and-exceptions pattern. Clause 1 states the default behavioral rule for the general case. Subsequent clauses state exceptions with increasing specificity. Later clauses take precedence when their conditions apply. The precedence note is inserted programmatically — not written by the LLM — ensuring consistent interpretation. New exceptions from later reflection iterations are appended as new numbered clauses; existing clauses are never renumbered or restructured.

- **Type:** Grounded in ExpeL's explicit success/failure trajectory comparison [7]. HEURISTIC entries encode what works; WARNING entries encode what fails and why. Both are present in skills_final.md for all episode outcomes. Collapsing them loses information about failure structure.

- **Dimension:** Encodes evaluation oracle provenance, making reliability visible at the entry level. Directly enables the Confidence gating mechanism.

- **Confidence:** Operationalizes SOTOPIA-Eval's known reliability gradient [1]. Turns a data-level limitation into an architectural feature. Assigned and updated deterministically, never by LLM judgment.

- **Support:** A passive count of distinct episode exposures, incremented once per episode regardless of outcome or whether the entry was demonstrably applied. Used as a tiebreaker for context-window-limited retrieval when two entries have similar Condition embeddings. Also enables evidence-based Confidence promotion for LOW-confidence entries on unreliable dimensions.

- **Provenance:** Grounded in CBR's source case tracking [11] and scientific reproducibility. Enables the adversarial agent to audit any entry's reasoning chain back to its originating episodes.

### 4.9 SCENARIO_TITLE Generation

After skills_final.md is produced, the system generates a SCENARIO_TITLE for the completed task. This is the primary retrieval key for the task in the archive.

**Structure:** A single field with a pipe-separated internal structure:

```
[Social dynamic description] | [Target perspective]
```

Both halves are embedded together as a single unit for primary retrieval. The left side captures scenario-structure similarity — finding the right neighborhood of the archive. The right side captures perspective similarity — ensuring retrieved chronicles were built from a compatible structural vantage point within that neighborhood.

**Why the perspective half is mandatory:** Two scenarios can have identical social structure with the roles reversed — a desperate seller and a patient buyer, versus a patient buyer and a desperate seller. These mirror-image scenarios have the same left-side description but are structurally opposite from any single agent's perspective. The right side of the pipe distinguishes them in embedding space, preventing retrieval from handing an agent a chronicle built from the opposing perspective.

**Examples:**

```
Zero-sum resource negotiation with asymmetric time pressure 
and anchoring behavior | skills from the uninformed, patient 
party managing concession timing against an anchoring opponent
```

```
Cooperative secret-keeping under social pressure from a 
trusted third party | skills from the secret-holder navigating 
loyalty conflict without damaging the relationship
```

**Stored separately for independent retrieval:** Beyond the unified SCENARIO_TITLE, the archive entry stores `SOCIAL_DYNAMIC` (left half alone) and `TARGET_PERSPECTIVE` (right half alone) as separate fields. These enable independent retrieval when needed — for instance, finding all scenarios with similar social dynamics regardless of perspective, or finding all scenarios from a similar target perspective regardless of scenario type. These fields live in the scenario archive entry, not in the skills chronicle.

**Generation prompt mandate:** The SCENARIO_TITLE generator is explicitly instructed to produce both halves. The adversarial agent flags any SCENARIO_TITLE missing either half as incomplete.

---

## 5. Evaluation Design

### 5.1 Baselines (Comparisons to Existing Methods)

**Baseline 1 — Vanilla LLM:** GPT-4o on SOTOPIA-hard with direct prompting and separately with CoT. No augmentation. Establishes the floor and the native model's typicality bias — the tendency toward overly cooperative, non-strategic behavior in the absence of accumulated social reasoning knowledge.

**Baseline 2 — Reflexion [6]:** Faithful implementation of Shinn et al. (2023) with proper verbal reflection accumulation within episodes, up to 5 retries, and no persistent cross-episode memory. The verbal reflection from each failed attempt is retained for subsequent attempts within the same episode but discarded at episode end. This is the strongest memoryless baseline. If the full system does not clearly outperform Reflexion, the persistent cross-episode memory architecture is not justified.

**Baseline 3 — ExpeL on fixed trajectories [7]:** ExpeL's insight extraction pipeline applied to a fixed set of SOTOPIA episodes — specifically the original 90 seed scenarios processed through their reflection architecture. Same base LLM, same evaluation scenarios. Tests whether open-ended curriculum generation adds value over fixed-distribution insight extraction. This baseline must be included given ExpeL's role as the primary schema precedent.

### 5.2 Ablations (Internal Component Validation)

Each ablation changes exactly one component while holding everything else constant.

**Ablation 1 — Fixed seed curriculum:** Full reflection and meta-reflection architecture, including all chronicle mechanisms, applied only to the original 90 seed scenarios rather than the Social Omni Epic generated curriculum. Isolates the contribution of open-ended curriculum generation from the reflection architecture. The most load-bearing ablation — if this matches the full system, the generation loop is not doing meaningful work and the paper's central claim is undermined.

**Ablation 2 — No MOI filter:** Full pipeline with Social Omni Epic curriculum generation, but without the Model of Interestingness filter. Scenarios enter the archive without the interestingness gate. Isolates the contribution of curiosity-driven filtering — whether generating diverse, interesting scenarios matters beyond simply generating more scenarios. Extends the pilot coverage evidence (0.600 vs. 0.550) to downstream agent performance.

**Ablation 3 — No meta-reflection:** The Meta-Reflection Module is replaced by simple concatenation of per-iteration reflection outputs. The skills chronicle contains the raw reflection text from each attempt rather than the synthesized output of the meta-reflector. Tests whether synthesis and cross-attempt reconciliation earn their LLM call cost over mere accumulation.

**Ablation 4 — Unweighted retrieval:** At evaluation time, entries are retrieved by Condition embedding similarity without Confidence weighting. HIGH and LOW confidence entries are treated identically. Tests whether the dimension-aware Confidence schema does real empirical work beyond intellectual justification.

**Ablation 5 — No verbalized sampling:** Full pipeline with MOI filter retained, but the Task Proposer generates from the default LLM distribution rather than being pushed toward low-probability regions of scenario space. The UMAP visualization of generated scenario spread is the primary evaluation metric for this ablation — it directly measures whether verbalized sampling expands the frontier of explored scenario space. Distinct from Ablation 2 because the MOI filter remains active; this isolates the low-probability sampling bias specifically.

### 5.3 Evaluation Scenarios

**Band A — In-distribution boundaries:** Complex variations of SOTOPIA archetypes held out from the generation loop. Tests the agent's ability to interpolate within known social dynamics using accumulated heuristics.

**Band B — Compositional generalization:** SOTOPIA-hard held-out scenarios requiring simultaneous application of multiple learned strategies. The primary evaluation benchmark. Human performance from the original SOTOPIA paper [1] is included as a reference ceiling on results figures — not as a target condition, but as a measure of how much of the human-AI gap the system closes.

**Band C (appendix) — Cross-benchmark transfer:** NegotiationArena [14], structurally close to SOTOPIA's competitive scenarios. Included as an exploratory analysis only. If transfer holds, it is a secondary finding. If it does not, it does not undermine the main claims.

---

## 6. Positioning in the Literature

This project sits at the intersection of three research traditions: open-ended learning [4, 13], inference-time cognitive architectures [5, 6, 7], and social intelligence evaluation [1, 2]. Its novel contribution relative to each:

**Relative to OMNI-EPIC [4]:** Extends open-ended environment generation from physical RL to social language agent domains. The key architectural challenge is the absence of a hard verification oracle — the paper's central methodological contribution is handling soft, dimension-stratified evaluation signals through the Confidence schema and dimension-aware gating.

**Relative to Voyager [5]:** Replaces executable skill verification with multi-attempt reflection and meta-synthesis. Handles the social domain's fundamentally different verification structure. Adds curriculum generation as a first-class component rather than assuming a fixed task distribution.

**Relative to ExpeL [7]:** Extends fixed-trajectory insight extraction to open-ended curriculum-driven accumulation. Adds dimension-aware confidence gating, adversarial stress-testing, the episodic-to-semantic consolidation architecture motivated by Tulving [9], and UCB1-based curriculum selection ensuring broad archive coverage. The most important extension is structural: ExpeL's insight quality is bounded by its fixed input distribution; this system's knowledge quality improves with the diversity of its self-generated curriculum.

**Relative to SOTOPIA [1]:** Uses SOTOPIA as evaluation infrastructure rather than research endpoint. The contribution is not a better benchmark but a better agent architecture — one that accumulates social reasoning knowledge across open-ended experience and deploys it at inference time without weight updates.

---

## References

[1] Zhou, X., Zhu, H., Mathur, L., et al. (2024). SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents. *ICLR 2024*. https://openreview.net/forum?id=mM7VurbA4r

[2] Wang, R., Yu, H., Zhang, W., et al. (2024). SOTOPIA-π: Interactive Learning of Socially Intelligent Language Agents. *arXiv:2403.08715*.

[3] Zhou, X., Liu, J., Yerukola, A., Kim, H., & Sap, M. (2026). Social World Models. *arXiv:2509.00559v2*.

[4] Faldor, M., Zhang, J., Cully, A., & Mouret, J.-B. (2024). OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code. *arXiv:2405.15568*.

[5] Wang, G., Xie, Y., Jiang, Y., et al. (2023). Voyager: An Open-Ended Embodied Agent with Large Language Models. *Transactions on Machine Learning Research*. arXiv:2305.16291.

[6] Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*. arXiv:2303.11366.

[7] Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., & Huang, G. (2024). ExpeL: LLM Agents Are Experiential Learners. *AAAI 2024*. arXiv:2308.10144.

[8] Agrawal, S., et al. (2025). GEPA: Generalizable Evolutionary Prompt Adaptation. *(Related work on evolved prompt heuristics from failure; inference-time adaptation literature)*.

[9] Tulving, E. (1972). Episodic and Semantic Memory. In E. Tulving & W. Donaldson (Eds.), *Organization of Memory* (pp. 381–403). Academic Press.

[10] Anderson, J. R. (1983). *The Architecture of Cognition*. Harvard University Press. *(ACT-R production system framework)*.

[11] Kolodner, J. L. (1993). *Case-Based Reasoning*. Morgan Kaufmann. *(CBR retrieve-reuse-revise-retain cycle)*.

[12] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning, 47*(2), 235–256. *(UCB1 theoretical foundation)*.

[13] Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2019). CURIOUS: Intrinsically motivated modular multi-goal reinforcement learning. *ICML 2019*. *(UCB-style curriculum selection in open-ended learning)*.

[14] Bianchi, F., Chia, P. J., Yuksekgonul, M., Tagliabue, J., Jurafsky, D., & Zou, J. (2024). How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis. *arXiv:2402.05863*.

[15] Oudeyer, P.-Y., Kaplan, F., & Hafner, V. (2007). Intrinsic Motivation Systems for Autonomous Mental Development. *IEEE Transactions on Evolutionary Computation, 11*(2), 265–286.

[16] Portelas, R., Colas, C., Weng, L., Hofmann, K., & Oudeyer, P.-Y. (2020). Automatic curriculum learning for deep RL: A short survey. *IJCAI 2020*.
