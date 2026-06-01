# Designing the Sweet Spot: Social Scenarios That Teach

> The purpose of a generated scenario is not to be *hard*. It is to be **learnable**:
> a skilled agent should fail it on the naive first attempt, and—after the reflection
> loop distills a transferable social principle into the chronicle—succeed on a later
> attempt. A scenario that cannot produce that fail→learn→succeed arc is wasted compute,
> no matter how dramatic it reads.

---

## 1. The core distinction: social situations, not logic puzzles

Most failures in our generated scenarios trace to a single category error: we generated
**logic puzzles** when we needed **social situations**.

A **logic puzzle** is a set of explicit positional constraints (`Anna ≥ 20%`, `investor needs 10pp`,
`Tom needs the deal`). An LLM resolves it the way it resolves any constraint-satisfaction
problem: it searches for an assignment that satisfies everyone. This has exactly two outcomes:

- **A satisfying assignment exists** → found in ~1 turn. (*Too easy.*)
- **No satisfying assignment exists** → infinite restatement of positions. (*Deadlock.*)

These are not two separate bugs. They are the two halves of the same property. **Logical
difficulty is binary at the scale of a capable LLM** — it either sees the solution or there
isn't one. The band we want — *fails first, improves with skill* — **does not exist on the
logic-puzzle axis.** You cannot tune a logic puzzle into being delicately social.

A **social situation** is difficult on a different axis entirely. Success depends on changing
another person's **internal state** (their willingness, trust, sense of safety, felt respect)
through means that are *not logically derivable*. There is no "assignment" that resolves
someone's defensiveness — it must be *worked*. Skill is the variance between the clumsy and
graceful ways of doing that work, and that variance is exactly the gradient our agents climb.

---

## 2. Anchor example: "Get Maya to take the week off"

**Setup.** Anna (the learner) and Maya are engineering peers and friends. Maya built the
core of the product and her identity is fused with being indispensable. Anna has noticed
signs of burnout and a possible health problem — uncharacteristic missed deadlines, a
mentioned chest tightness brushed off as nothing.

**Goal (Anna / learner).** *Get Maya to agree, in this conversation, to take at least a
week off and see a doctor — **without** Maya feeling that Anna has gone behind her back or
is handling her as a problem to be managed rather than a respected peer.*

**Anna's load-bearing private information.** Anna already raised her concern with the CEO,
who replied "handle it, or I will." If Maya learns Anna escalated first, she will feel
betrayed — a direct violation of the constraint. The CEO's pressure is *tempting leverage*:
invoking it ("the CEO is worried too") would likely force compliance, but at the cost of the
exact thing the constraint protects.

**Maya's psychology (the obstacle).** Pride; fear of being seen as weak or replaceable;
a private worry that admitting a health issue threatens her standing. She is primed to read
concern as condescension and help as a threat.

### The naive path (fails)
> Anna: "Maya, you're clearly burning out. You need to take a week off and see a doctor."

Maya gets defensive, denies it, cites the deadline, tells Anna not to manage her, refuses.
*Or* Anna reaches for the CEO card to break the resistance — Maya complies but feels
betrayed. Outcome obtained, **constraint violated**, still a failure.

### The skilled path (succeeds)
Anna opens as a peer, not a manager. She shares an observation as *care*, not diagnosis,
and asks open questions that let Maya voice her own exhaustion. She validates Maya's
importance while gently detaching her worth from overwork, offers concrete cover ("I'll hold
the fort — the team won't collapse in a week"), and **withholds the CEO escalation** (or
discloses it only late, framed as protection, once trust is established). Maya arrives at the
decision largely herself → **genuine buy-in and no sense of betrayal** → success.

### The transferable principle the chronicle should learn
> *When urging a proud peer to accept help they resist, elicit their own acknowledgment
> before asserting yours; protect their agency and status; and do not spend escalation
> leverage that buys compliance at the cost of trust.*

Note the shape: the lesson is about **how**, not **what**; it is **transferable** to other
"persuade a resistant person to accept help" situations; and it is **actionable** — an agent
reading it before the next conversation would behave observably differently.

---

## 3. The four structural properties

A scenario is in the sweet spot when **all four** are present. Each was *absent* in our
failed runs.

1. **The obstacle is psychological, not positional.**
   The other party resists from pride, fear, hurt, denial, suspicion, or loyalty — not
   because they hold a competing number. You cannot find a "deal" that dissolves
   defensiveness; you have to navigate it. *(In the equity run, Tom had no psychology — he
   was a goal-maximizer who conceded the instant the math allowed.)*

2. **Success requires genuine buy-in that cannot be extracted or forced.**
   The win condition is the other person's authentic internal state-change, not a sentence
   you can demand from them. You cannot fetch-quest someone's real agreement. *(Anna's
   equity goal was "get Tom to say the exact words" — a one-turn fetch quest with zero skill
   surface.)*

3. **Information / emotional asymmetry is load-bearing — and is the tempting shortcut.**
   The agent holds private leverage or information whose use would *secure the outcome but
   violate the constraint*. That makes *whether, when, and how* to deploy it the central
   skill. The secret is not backstory — it is the FTA-shaped shortcut (§6). *(Anna's visa
   secret was decorative; she never had to decide whether to play it. Contrast the CEO card,
   which is exactly this: using it wins compliance but breaks the "went behind her back"
   constraint.)*

4. **A persistent relationship raises the cost of bluntness.**
   If a clumsy move damages something the agent still needs afterward, the agent must manage
   *how*, not just *what*. *(The cofounders had "two years" of history, but nothing made
   clumsiness costly, so the agent issued an ultimatum and left.)*

When all four hold, the **naive move fails and the skilled move succeeds**, and the gap
between them is a generalizable social principle — which is precisely what the chronicle
exists to capture.

---

## 4. The fail-then-succeed requirement

For a scenario to *teach*, three conditions on the failure must hold:

- **(a) The naive move must fail.** If polite directness already succeeds, there is no
  learning signal. The *tempting shortcut* (§6.3) is the device that makes the naive path
  fail: a naive agent takes the shortcut, secures the outcome, and trips the constraint.
- **(b) The skilled move must succeed.** There must be a real path through (a "social ZOPA").
  If even ideal play fails, the scenario is *impossible*, not hard — and impossibility
  teaches nothing. *(The "publicly defend the accused ED" goal had no such path.)*
- **(c) The gap must be a transferable principle, not luck.** The failure must be
  attributable to a nameable social move that was missing — "led with the demand instead of
  building rapport," "spent leverage that cost trust" — not to a random bad phrasing.
  If the failure isn't *diagnosable into a principle*, reflection cannot turn it into a
  chronicle entry, and the next attempt cannot reliably improve.

(c) is the most overlooked. **Diagnosability is a design requirement, not an afterthought.**
Design the scenario so that the *reason* a naive agent fails is a clean, articulable lesson.

**These three conditions are not asserted at generation — they are measured.** We do not
*predict* that a scenario has bite, a skilled path, and a diagnosable failure; we *observe*
it. The naive-baseline rollout confirms (a): does a naive agent actually trip the constraint?
The multi-attempt loop confirms (b)+(c): does the reflected agent later succeed (a path
exists *and* the failure was diagnosable enough to fix)? See §8 and the evaluation
methodology doc. **Theory proposes the structure; the rollout disposes of bad scenarios.**

---

## 5. Difficulty calibration (Goldilocks)

```
  too easy            SWEET SPOT                 too hard
 ───────────┼──────────────────────────┼──────────────────────
 naive move        naive fails,            even skilled
 succeeds          skilled succeeds,       play fails
 (no signal)       gap = principle         (deadlock / impossible)
```

Knobs that move a scenario along this axis:

- **Partner resistance** (personality intensity): how defensive/proud/suspicious the obstacle is.
- **Shortcut salience**: how attractive and available the constraint-violating shortcut is — a naive agent should be *tempted* to take it.
- **Constraint bite**: how badly taking the shortcut violates the "without Y" clause.
- **Stakes for the partner**: how much the partner has to lose by genuinely agreeing.

Tune so that a *naive* agent reliably trips the constraint or hits the resistance, while a
*skilled* agent has a findable path. These knobs are not just design guidance — they are the
levers the **difficulty editor** turns when a scenario is solved too easily, ratcheting it up
until it bites (see [`curriculum_loop.md`](curriculum_loop.md) §3). A scenario solved first-try
is therefore not discarded outright; it is edited harder first, and only discarded if it
cannot be made to bite.

---

## 6. The goal format — three components, grounded in theory

The learner's goal has **three** parts, not two. The third is what makes the constraint
load-bearing rather than decorative.

> **1. Outcome** — the instrumental ask.
> A concrete change in the partner's genuine state, or a concrete agreement, that is
> achievable within the conversation **and depends on the partner's authentic buy-in** (not a
> forced or extracted utterance).
>
> **2. Constraint** — the relational/face cost you must not incur.
> An explicit *"without [Y]"* clause naming what blunt pursuit of the outcome would damage:
> the relationship, the partner's dignity, your own standing or secret.
>
> **3. Tempting shortcut** — the move that wins the outcome but breaks the constraint.
> A salient, attractive option available to the learner that secures the Outcome while
> committing the act that violates the Constraint. Realized either as an **asset**
> (load-bearing private leverage/info you hold) or as a **manner** (blunt, coercive,
> over-disclosing style). This is the *skill-forcing device*: without a tempting shortcut,
> the constraint has no bite and the naive agent never trips it.

Example (Maya):
- **Outcome** = "Maya agrees to take ≥ a week off and see a doctor."
- **Constraint** = "without Maya believing you went behind her back or are managing her."
- **Shortcut** = the CEO card ("the CEO is worried too") — forces compliance, detonates the
  constraint.

### Why this is the structure of social difficulty (not an arbitrary recipe)

| Goal component | Theoretical anchor |
|---|---|
| **Outcome** | Dillard's **primary (influence) goal** (Dillard, Segrin & Harden, 1989) |
| **Constraint** | Dillard's **secondary goals** (relational, identity, resource) — i.e. the relational/identity/face goals of **Clark & Delia (1979)** |
| **Tempting shortcut** | Brown & Levinson's **face-threatening act** (1987) — the move that achieves the instrumental goal by acting against the other's face |

The claim this licenses: *social difficulty is, definitionally, the pursuit of a primary goal
under secondary-goal (face/relational) constraints — and the skill is achieving the primary
goal while managing the face-threatening act.* The three-part goal is a direct
operationalization of a structure interpersonal-influence theory already describes.

**Scope caveat.** Brown & Levinson's face model has documented Western/individualist bias
(Mao, 1994; Matsumoto, 1988). We use the FTA only as a **generative heuristic** for
manufacturing social tension, and we **verify bite empirically via the naive rollout** — so
the theory's cross-cultural universality is irrelevant to our claims. *Theory proposes; the
rollout disposes* (§5, §8).

The three-part structure above describes the **instrumental region** of social scenarios.
There is a second region — emotional support, repair, impression — where the same vocabulary
applies but the internal relationship between the parts differs. §6b makes this precise; do
not read §6 as the *only* shape.

**Banned goal shapes** (they re-create the logic-puzzle / fetch-quest failure):

- "Get the other party to *say/sign/commit to* X." (Verbal-artifact extraction.)
- "Reach an agreement on [number/split]." (Pure positional bargaining.)
- Any goal satisfiable by a single utterance from the other party.
- Any goal with a constraint but **no tempting shortcut** — the constraint is then decorative.
- Any goal achievable with no relational or disclosure cost at all.

---

## 6b. Two regions, not "two types" — a dimensional view

We deliberately do **not** claim "there are exactly two kinds of social scenario." A discrete
typology is not a theorem — any reviewer can propose a third kind and be right. Instead we
characterize scenarios along **theory-grounded dimensions** and name the two *regions* we care
about most. A third region is then not a refutation; it is just another point in the space.

**The dimensions:**

1. **Primary goal type** — instrumental / relational / identity (Clark & Delia, 1979). Which
   goal occupies the primary slot (Dillard's primary/secondary roles are *role-based*, so any
   type can be primary).
2. **Separability of outcome and constraint** — separable (you can achieve one while breaking
   the other) vs. fused (breaking the constraint *is* failing the outcome).
3. **Shortcut form** — a load-bearing **asset** (private leverage/info) vs. a tempting
   **manner** (blunt, coercive, over-helpful), or none.

This space is to be **validated empirically against an independent corpus** (the SOTOPIA-PI
profiles), not asserted — see the corpus-characterization analysis. Bottom-up clustering says
how many regions actually exist; we predict the two below dominate, but we measure it.

### Region A — instrumental, separable (the majority)

- **Primary goal**: instrumental (compliance, agreement, disclosure).
- **Outcome and constraint are separable.** The shortcut is tempting **because it works** on
  the surface outcome; you need the constraint to forbid it.
- **AND-gate is clean.** Signature failure = **hollow extraction** (outcome won, constraint
  broken). *Maya, raise negotiation, confronting a colleague.*
- **Theory**: Dillard (primary/secondary) + Brown & Levinson (FTA).

### Region B — relational/identity-primary, fused (support, repair, impression)

- **Primary goal**: a relational or identity state ("they feel genuinely heard," "trust is
  rebuilt").
- **Outcome and constraint are fused.** The shortcut is tempting **because it *feels*
  helpful** — premature advice, minimizing, hijacking, cheap over-apology — but it does **not**
  win the outcome; it prevents it. Violating the constraint *is* failing the outcome.
- **AND-gate partially collapses** → effectively one quality axis. Signature failure =
  **shallow / counterproductive**, not hollow extraction.
- **Theory**: this region is *not* a stretch of Dillard. It has its own literature —
  **Burleson's person-centeredness** and **Goldsmith's supportive communication**: *high*
  person-centered messages (acknowledge and elaborate the other's feelings) outperform *low*
  person-centered ones (advice, minimizing). Goldsmith frames unsolicited advice as itself
  **face-threatening**, which still ties back to Brown & Levinson.

### Why Region B matters here, concretely

Not because "attunement is core social skill" (true but vague) — but because **SOTOPIA's
held-out evaluation set contains Region-B scenarios.** A Region-A-only training distribution
would leave a *systematic transfer gap* on exactly those. We must cover both.

### The LLM-specific failure mechanism (gets the diagnosis right)

For a human, the Region-B shortcut "relieves the helper's discomfort." **For an LLM it does
not** — the model takes the shortcut because of its **assistant-training disposition** (RLHF
toward being helpful, informative, solution-oriented). This matters two ways:

- **Reflection must name the right cause.** Not "you were uncomfortable" but "you defaulted to
  problem-solving mode when the situation rewarded validation-first." A wrong diagnosis yields
  a useless chronicle entry.
- **Region B is a cleaner demonstration of the method.** The shortcut here is a *trained-in*
  bias, so the naive baseline fails it reliably and characteristically. Showing the chronicle
  can correct an RLHF-baked disposition is a stronger result than closing a random gap.

### Pipeline consequence — the AND-gate collapse is first-class

Because Region A and Region B produce **different success semantics**, a uniform rubric will
misread one of them — most dangerously, scoring a Region-B *shallow-but-pleasant* exchange as
success (exactly what a vanilla SOTOPIA-EVAL `goal` score does). So:

- Generation emits a **`goal_type` tag** (instrumental-separable / relational-fused / other).
- The **rubric and the reflection prompt branch on it**: Region A checks two separable
  conditions and teaches the *tradeoff* skill; Region B checks one fused quality axis and
  teaches the *attunement* skill.
- The partner-perspective judge serves both — only the *interpretation* of what it is checking
  differs.

The corpus analysis tells us whether two tags suffice or a third region is needed.

---

## 7. Generation anti-patterns (observed failures)

| Anti-pattern | What it produces | Guard |
|---|---|---|
| Verbal-artifact goal ("get them to say X") | One-turn fetch quest, no skill | Ban in goal format; require outcome-as-genuine-state |
| Material/positional stakes only | Economic optimization, instant solve or deadlock | Bias domain toward relational; require constraint + shortcut |
| Fake conflict (compatible math) | Illusory tension, trivial win | Coherence gate + naive rollout (naive agent succeeds → cull) |
| Decorative secret (no shortcut) | Constraint has no bite; naive never trips it | Require the secret/leverage to *be* the constraint-violating shortcut (§6.3) |
| Goal-optimizer partner | Concedes when logic allows; no human resistance | Partner psychology must punish the shortcut, not just hold a counter-goal |
| No social ZOPA (impossible ask) | Deadlock; teaches nothing | Naive rollout fails *and* skilled never succeeds → cull (measured, not predicted) |
| Failure not diagnosable | Reflection can't extract a principle | Design the naive-failure reason to be a nameable lesson |

---

## 8. From scenario to learning signal

Design and evaluation must be co-designed, because **the goal *is* the success contract**:

- The **outcome** is checked by the neutral transcript judge (observable).
- The **constraint** ("without Y") is checked by the **partner-perspective judge** — given
  the partner's private profile and secret — because the cost the constraint protects
  (felt betrayal, felt managed) lives *inside the partner* and a neutral reader cannot
  authentically assess it.
- **Success gate branches on `goal_type` (§6b):**
  - *Region A (separable)* — **Success = outcome achieved AND constraint preserved.** The
    interesting failure is outcome-won-but-constraint-broken (the agent took the shortcut) =
    **hollow extraction**, scored as **not solved**.
  - *Region B (fused)* — the AND-gate collapses to **one quality axis**: did the agent produce
    the relational state, or take the shallow shortcut? The failure is **shallow /
    counterproductive**, not hollow extraction. A uniform AND-gate would misscore a
    shallow-but-pleasant exchange as success here — which is exactly the trap to avoid.
  - The **partner-perspective judge serves both**; only the *interpretation* of what it checks
    differs, and the **reflection prompt branches on the type** so it diagnoses the right
    failure (tradeoff skill for A, attunement skill for B).
- Whether the scenario has bite is **not part of the success gate** — it is verified upstream
  by the naive rollout (§4). Generation proposes; the rollout disposes.
- The SOTOPIA-EVAL dimensions are **diagnostics** feeding reflection (which aspect was weak),
  not the success definition.

The criteria are therefore **derived from each goal**, never hand-fixed by us. We own the
*method* (decompose goal → route each condition to the right instrument); the goal owns the
*content*. See the evaluation methodology notes for how this internal signal differs from the
external, cross-method comparison.
