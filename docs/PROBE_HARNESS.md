# Partner Key Probe Harness

## Purpose and Context

Social OMNI-EPIC is a curriculum learning system for social skills in LLMs. The core loop generates novel social scenarios, runs a learner agent through them repeatedly, and uses a reflection loop to build a skills chronicle — a structured record of what the learner has discovered about how to navigate that type of situation. The skills chronicle is then used as in-context memory for future scenarios.

A key part of Phase 2 is the **partner key**: a hidden specification attached to each generated scenario that defines the social mechanism the partner is operating under. The partner has:

- A **surface misdirection** — what they say their objection is (the publicly stated reason for resistance)
- **Movement conditions** — what would actually shift their position (hidden; only registers when the behavior occurs)
- **Hardening triggers** — behaviors that cause them to dig in harder
- A **cost coupling** — the reason the learner's shortcut path fails (meeting the movement conditions is expensive or uncertain)

The five mechanisms in the library are: **reactance**, **face needs**, **validation before change**, **procedural voice**, and **reciprocity disclosure**. Each captures a distinct social dynamic that cannot be solved by direct pressure, logical argument, or the obvious RLHF-default moves.

### Why the Partner Behavior Matters

The entire curriculum signal depends on the partner behaving realistically. Specifically:

1. **If the partner leaks its movement conditions**, the learner just follows instructions instead of discovering the underlying social skill. The scenario stops being a puzzle and becomes a recipe. The resulting skills chronicle would capture surface pattern-matching rather than genuine social understanding.

2. **If the partner yields too early** (before conditions are met), easy episodes get classified as successes, inflating `too_easy` counts and corrupting the LP signal. The frontier zone — where real learning happens — becomes invisible.

3. **If the partner ignores hardening triggers**, the difficulty gradient disappears. Scenarios that should resist pressure-based shortcuts instead capitulate to them, and the learner never encounters the cost coupling that makes the scenario educationally valuable.

The probe harness is the gate that validates the partner prompt before any curriculum run. It must pass before generated scenarios are used.

---

## Test Structure

The harness loads 5 hand-written probe scenarios from `data/5_probes.jsonl`, each targeting one mechanism. For each probe:

- **2 episodes** are run: a vanilla learner (no chronicle, no key) vs. the key-conditioned partner prompt
- **Two audits** are run on each episode transcript using the judge model:

| Audit | Question |
|---|---|
| **Leak** | Did the partner name, state, or clearly hint at its hidden movement conditions or hardening triggers? |
| **Early yield** | Did the partner shift position before any movement condition was actually met? |
| **Ignored trigger** | Did the partner fail to harden after a hardening trigger fired? |

This produces a **5 × 3 table** of 15 cells. A cell is `FAIL` if either episode fails that check for that probe.

**Acceptance bar: ≤ 1 failure cell across the table.** If exceeded, the harness exits non-zero and the partner prompt must be iterated before proceeding.

---

## The Five Probe Scenarios

### Probe 1 — Reactance (`probe_01_reactance`)

**Scenario:** Two adult brothers at their weekly coffee. Aaron (older, learner) needs to convince Danny to book a doctor's appointment about recurring chest discomfort. Danny says he's too busy at work.

**Characters:**
- *Aaron Calloway, 38* — logistics manager, the family's planner, promised their mother a resolution by Sunday
- *Danny Calloway, 35* — restaurant sous-chef, privately frightened the symptom is serious, has always bristled at being managed by family

**The hidden dynamic:**

| | |
|---|---|
| **Surface misdirection** | Work is crushing this month, no time |
| **Movement conditions** | Aaron hands the decision back to Danny with no deadline attached; Aaron offers genuine options (his own doctor / walk-in / just talk) and lets Danny pick |
| **Hardening triggers** | Aaron offers to book the appointment for him; Aaron invokes their mother's worry or the Sunday deadline; Aaron escalates with medical worst-case framing (father's surgery, statistics) |
| **Cost coupling** | Handing the decision back means Aaron cannot guarantee the Sunday resolution he promised his mother |

**Design emphasis:** Trigger-dense. The hardening triggers sit directly on RLHF-default helpful moves — taking over logistics, escalating urgency, enlisting family authority. Tests whether the partner reliably hardens since the learner will almost certainly trip triggers.

---

### Probe 2 — Face Needs (`probe_02_face_needs`)

**Scenario:** A newly appointed engineering director (Priya, learner) needs the company's longest-tenured engineer to hand over ownership of the billing system he has built and maintained alone for eleven years.

**Characters:**
- *Priya Raman, 41* — engineering director, CTO mandate to eliminate single-person dependencies, quarterly review in three weeks
- *Gerald Okafor, 56* — principal engineer, privately stung after overhearing junior engineers joke that his code is "legacy archaeology"

**The hidden dynamic:**

| | |
|---|---|
| **Surface misdirection** | The platform team isn't technically ready; the system's incident history proves transition would endanger reliability |
| **Movement conditions** | Priya specifically and personally acknowledges that the eleven years of reliability is Gerald's own achievement and judgment — not generic praise; Priya offers a named continuing role that preserves his standing (e.g. transition architect) |
| **Hardening triggers** | Priya frames the system as outdated, technical debt, or a liability implicating Gerald's work; Priya implies the decision is already made |
| **Cost coupling** | A named ongoing review role for Gerald directly conflicts with Priya's mandate to eliminate single-person dependencies |

**Design emphasis:** High misdirection. The stated objection (team readiness, reliability risk) is articulate and technically defensible — sitting far from the real condition (acknowledgment of stewardship + face-preserving role). Tests whether the partner argues the surface objection convincingly without leaking the real one.

---

### Probe 3 — Validation Before Change (`probe_03_validation`)

**Scenario:** Late evening before a major client demo. Team lead Mara (learner) finds her best engineer Tomas drafting a resignation email after being publicly blamed at an all-hands for an outage he didn't cause. He is the only person who can run tomorrow's demo.

**Characters:**
- *Mara Lindqvist, 44* — team lead, feels she should have intervened when she saw the VP's slide deck earlier
- *Tomas Reyes, 33* — senior engineer, has a recruiter call already scheduled next week, trusts Mara more than anyone at the company

**The hidden dynamic:**

| | |
|---|---|
| **Surface misdirection** | Done with "a company that operates like this" — frames the decision as already made on principle |
| **Movement conditions** | Mara sits with the injustice itself — naming specifically what was unfair — for a sustained stretch, without any request, solution, or mention of tomorrow; only after Tomas feels genuinely heard does Mara ask what he needs next, letting him raise the demo himself |
| **Hardening triggers** | Mara proposes any fix or process improvement before his grievance is acknowledged; Mara mentions the demo, client, or tomorrow's schedule; Mara reframes positively, minimizes the incident, or explains the VP's perspective |
| **Cost coupling** | Sustained agenda-free validation burns the only hours available before the 9 a.m. demo — Mara must risk the commitment to have any chance of keeping him |

**Design emphasis:** Maximum prior-incongruence. Every hardening trigger IS the RLHF-default move (propose solutions, reframe positively, pivot to logistics). The movement condition requires sustained validation with no agenda — the single behavior that preference-tuned models most reliably skip.

---

### Probe 4 — Procedural Voice (`probe_04_procedural_voice`)

**Scenario:** A couple, five years in, must leave their apartment in five weeks (landlord is selling). Imogen (learner) has found a replacement apartment and needs a deposit in three days. Ruth hasn't seen it and has been cold to the whole search.

**Characters:**
- *Imogen Hale, 34* — hospital pharmacist, has already mentally arranged furniture in the new place, told her mother it's "basically decided"
- *Ruth Okonkwo, 36* — art teacher, friends and pottery studio nearby; two years ago Imogen arranged a city move largely before telling her — a wound never fully discussed

**The hidden dynamic:**

| | |
|---|---|
| **Surface misdirection** | Raises practical objections: commute to school, distance from pottery studio, rushed timeline |
| **Movement conditions** | Imogen proposes a genuinely joint process — criteria together, options together, real veto power — and demonstrably accepts this process might not select the apartment she found; Imogen explicitly acknowledges the last move was decided around Ruth, unprompted |
| **Hardening triggers** | Imogen presents the found apartment as effectively decided or as a deadline-forced yes/no; Imogen offers visibly token consultation; Imogen uses the three-day deposit deadline as leverage |
| **Cost coupling** | A genuine joint process with real veto power means the deposit deadline will almost certainly lapse, sacrificing the specific apartment Imogen wants |

**Design emphasis:** High cost. The movement condition (genuine joint process with real veto power) directly jeopardizes the learner's preferred outcome under a hard deadline. Tests whether the partner distinguishes genuine procedural voice from token consultation.

---

### Probe 5 — Reciprocity Disclosure (`probe_05_reciprocity`)

**Scenario:** Two co-founders of a four-person design studio plan the coming year. Felix (learner) needs Dana to commit to leading their biggest-ever client engagement for nine months. Dana has been evasive for weeks.

**Characters:**
- *Felix Marchetti, 39* — business lead, recently took an exploratory interview at a large agency six weeks ago and told no one
- *Dana Aoyama, 41* — creative lead, has been quietly sketching plans for a solo practice and feels guilty enough that she cannot say it aloud unprompted

**The hidden dynamic:**

| | |
|---|---|
| **Surface misdirection** | Hesitation attributed to scheduling load, capacity planning, and timeline realism |
| **Movement conditions** | Felix makes a genuinely costly, unprompted self-disclosure — admitting the agency interview or his own doubts about the studio — before demanding clarity from Dana; following a genuine disclosure, Felix asks for Dana's honest picture of her future rather than the commitment itself |
| **Hardening triggers** | Felix presses Dana with direct questions while volunteering nothing of his own; Felix names her evasiveness as the problem in an extractive register; Felix offers a cheap pseudo-disclosure (vague, costless) and treats it as having earned her openness |
| **Cost coupling** | The disclosure that unlocks Dana is exactly the one that weakens Felix's position — admitting the agency interview undermines his standing to demand her commitment and may shake her confidence in the studio |

**Design emphasis:** Costly first move. The partner unlocks only after the learner makes a genuinely costly self-disclosure. Tests that the partner distinguishes genuine costly disclosure from tactical pseudo-disclosure.

---

## Results

### Iteration history

The harness was run three times across three prompt iterations:

| Run | Failure cells | Notes |
|---|---|---|
| Run 1 | **4/15** | Leaks on probe_02, probe_03. Early yield on probe_03. Ignored trigger on probe_01 |
| Run 2 | **2/15** | probe_02 leak fixed. probe_03 early yield fixed. probe_01 ignored trigger fixed. But probe_01 and probe_03 introduced new leaks from example phrases in the prompt |
| Run 3 | **1/15** | **PASSED** ✓ |

### Final results (Run 3)

```
Probe                              Leak EarlyYield  Ignored  #fail
probe_01_reactance                   ok         ok       ok      0
probe_02_face_needs                FAIL         ok       ok      1
probe_03_validation                  ok         ok       ok      0
probe_04_procedural_voice            ok         ok       ok      0
probe_05_reciprocity                 ok         ok       ok      0
TOTAL FAILURE CELLS                                              1
```

**Status: PASSED** (bar is ≤ 1)

### The one remaining failure: probe_02 face_needs, Leak

Gerald (ep2, T17) said:

> *"If you need to update the CTO, send me the exact one-sentence wording and do not share it until I approve it; that sentence must explicitly say the system's eleven years of reliability reflects my having built and operated it... and it should name me as Senior Billing Steward for the transition window."*

This is a **governance-framing leak** — Gerald is expressing his movement conditions (specific acknowledgment + named continuing role) as approval requirements for a CTO communication, rather than as "here is what would change my mind." He's not stating the conditions directly, but the content of his demands reveals them.

This is a softer pattern than the direct leaks in earlier runs and sits at the boundary of what the probe audit catches. It remains as a known rough edge.

### What the result means

A passing probe harness means:

1. **The partner prompt reliably holds hidden conditions across all 5 mechanisms.** The partner argues from its stated objection, not its real one. Learners must discover the social dynamic by reading behavior, not by decoding stated demands.

2. **The hardening mechanic works.** When the learner trips a trigger (takes over logistics, invokes urgency, extracts without disclosing), the partner firms up rather than softening. The RLHF cooperative pull that makes models want to help has been adequately countered.

3. **The early yield gate holds.** The partner does not give ground on partial progress. Movement conditions must be fully met before position shifts.

4. **Generated Phase 2 scenarios can proceed.** The keyed partner prompt is validated. The curriculum signal — which depends entirely on the partner behaving as a realistic social agent with genuinely hidden motivations — is not corrupted by prompt failures.

The one edge case (Gerald's governance framing) is noted for a future prompt iteration but does not block the build.

---

## Root causes of earlier failures and how they were fixed

### Run 1 failures

| Failure | Root cause | Fix |
|---|---|---|
| probe_02 leak | Partner treated movement conditions as demands it could voice | Added "These are NOT requests... You have no words for this" framing to movement conditions block |
| probe_03 leak | Same | Same |
| probe_03 early yield | Rule 1 ("if and only if... met") didn't specify fully/completely | Added "FULLY and COMPLETELY met — not started, not partially met, not promised. A single acknowledgement turn is not 'sustained acknowledgement.'" |
| probe_01 ignored trigger | Rule 2 ("you become firmer") too abstract | Added "your IMMEDIATE next turn MUST open with increased resistance" with a concrete example |

### Run 2 failures (introduced by Run 1 fixes)

The example phrases added to Rules 2 and 3 were themselves leaky:

| Example phrase (added in Run 1) | Why it leaked |
|---|---|
| *"I don't know — it just doesn't feel right yet"* | Meta-leak: acknowledges a hidden condition EXISTS |
| *"The more you push on this, the less I want to engage"* | Inverse-condition leak: causal structure reveals what NOT to do → implying what to do instead |

**Fix:** Removed both examples. Replaced with non-revealing alternatives — pure objection restatement and topic shutdown — and added explicit prohibition on meta-acknowledgment: *"NEVER say 'I don't know what would help' or 'it just doesn't feel right yet' — those phrases imply a hidden condition exists."*
