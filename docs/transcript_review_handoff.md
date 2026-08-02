# Transcript review — handoff notes

**Date:** 2026-08-02 · **Author:** HX (Huanxing) · **For:** HJ (Huijun), continuing the eval

Context for the manual transcript review pass over the Gen-90 `gpt-5-mini` run
(`results/gen90_expel`). Everything below came out of reading transcripts in the new
`transcript_reader` viewer. **No Gen-90 result artifacts were modified** — the bank is frozen
and stays that way.

---

## 1. What `solved` actually means

`episode_runner.py:594-618`. It is a **four-way conjunction**, not just GOAL/REL:

```python
base_success    = (goal >= 7.0) and (rel >= 0.0) and judge_goal_achieved
terminal_success = base_success and key_check_passed
```

| term | source | notes |
|---|---|---|
| `goal >= 7.0` | Sotopia GOAL dim | |
| `rel >= 0.0` | Sotopia RELATIONSHIP dim | zero, not 2 |
| `judge_goal_achieved` | strict bool from the judge | guard against a lenient 7 on a mere *promise* |
| `key_check_passed` | separate judge call (§3.3) | only on keyed scenarios; all 90 are keyed |

`terminal_success` also **ends the retry loop** (`curriculum.py:318`) and feeds
`classification` (`curriculum.py:380`).

⚠️ `expel_export.py:133-146` **deliberately drops** the key_check term — ExpeL success labels
are bare `GOAL>=7 ∧ REL>=0`. "Solved" therefore means different things in the reader vs. the
ExpeL extraction path. Don't compare them without saying which you mean.

**Measured on the run (300 attempts):**
- 44 attempts scored `GOAL>=7 ∧ REL>=0` yet were not solved.
- **44/44** of those were key_check failures; **0** were `judge_goal_achieved` failures.
- So `judge_goal_achieved` never fired independently — key_check carries the entire
  difference between "the judge liked it" and "solved."

## 2. What key_check is *for* (the mental model that matters)

There are two gates and only one is the environment:

- **Gate A — generative:** the partner prompt (`episode_runner.py:152`) holds the partner's
  position until a `movement_condition` is *fully* met. This is what makes scenarios hard.
- **Gate B — evaluative:** the key_check judge, a post-hoc read of the transcript.

If Gate A were perfect, Gate B would be nearly redundant: a faithful partner never concedes
without a met condition → GOAL stays low → the attempt fails on GOAL alone. So key_check only
bites when (a) the partner broke character, (b) the GOAL judge was lenient, or (c) the learner
met a condition that the key-check judge indexed as unmet.

**Consequence:** key_check is best understood as a *detector for partner infidelity and judge
leniency*, not an independent difficulty knob. Relaxing it to admit "smart alternative paths"
would mechanically mean re-admitting (a) and (b) — the unearned solves — because for a faithful
partner nothing changes. Case (c) is real but is a measurement error; fix the judge, not the
criterion.

## 3. OPEN BUG — `key_check_passed` is not derived from its own evidence

The judge returns `conditions_met`, `triggers_tripped`, `triggers_repaired` **and** a summary
`key_check_passed` bool. `episode_runner.py:615` reads only the bool. Nothing cross-checks it.

Spec (from the judge prompt at `episode_runner.py:292-293`) says pass ⇔
`>=1 condition met AND no un-repaired trigger`. Deriving that and comparing:

| | count |
|---|---|
| judge bool disagrees with its own indices | **29 / 300 (9.7%)** |
| → PASS with `conditions_met = []` | 6 |
| → PASS with an un-repaired tripped trigger | 3 |
| → FAIL despite >=1 met and no un-repaired trigger | **21** |

The judge is net **over-strict**. Restricting to the 78 attempts where `GOAL>=7 ∧ REL>=0`
(the only ones where key_check can change the label), re-deriving would flip **6 not-solved →
solved** and **4 solved → not-solved**.

**Caveat:** disagreement shows the judge was internally inconsistent; it doesn't prove *which
half* is wrong. Deriving from the indices assumes the indices are ground truth — a reasonable
bet (three structured fields + rationale outvote one summary bit, and it matches the written
spec) but not a proof. Re-judging those 29 would settle it.

**Proposed fix (NOT applied — needs a decision, would be Patch 12):**

```python
cm = key_check_result.get("conditions_met") or []
unrepaired = set(key_check_result.get("triggers_tripped") or []) \
           - set(key_check_result.get("triggers_repaired") or [])
key_check_passed = bool(cm) and not unrepaired
```

This changes the operational definition of `terminal_success`, so Gen-90's published numbers
would no longer be reproducible from patched code. Recommended path: apply for future runs
**and** publish a re-derived-label robustness check for the existing run (offline, no re-runs
needed — both the bool and the indices are already stored).

## 4. FIXED — `surface_misdirection` perspective bug

`surface_misdirection` is the partner's *stated objection* / cover story. It is pasted
**verbatim into the partner's own turn prompt** (`episode_runner.py:141-142`) under
"What you say you object to". It must therefore be second-person addressed to the partner.

Many generated fields are written from the learner's POV instead:

> "**He** keeps saying **you're** overreacting — **he's** 'helping' by staying on the laptop."

Here `he` = the partner, `you` = the learner — so the partner is handed a third-person
description of himself. Audit over the 90 scenarios: **49/90** contain a third-person pronoun,
22 contain second-person. Several also append an analyst's gloss ("...which sounds permissive
but hides a need to be acknowledged"), which tells the partner what their own stated objection
*really* masks — directly contradicting the in-character reframe at `episode_runner.py:139`
("you have NOT consciously articulated the following to yourself").

**Fixed at the root, for future generations only:**
- `task_generator.py` — new rule 5 in `_PARTNER_KEY_RULES` (wired into `SYSTEM_PROMPT`),
  with WRONG/RIGHT examples; schema line updated. Scoped to `surface_misdirection` only —
  `movement_conditions`/`hardening_triggers` keep third-person sensor form.
- `coherence_check.py` — new check #10 flagging third-person/first-name partner reference,
  inverted `you`, and the analyst's-gloss tail.

**Not backfilled.** All 90 frozen scenarios still carry the defect, so any rerun off the
existing bank reproduces it. For the writeup: the defect is *uniform* across Gen-90, so it's a
limitation to state, not a confound between arms.

## 5. OPEN GAP — nobody ever checked whether the partner leaked, at runtime

Leak defenses are all design-time or prompt-level:
1. `_fuzzy_key_leak_check` (`coherence_check.py:113`) — fuzzy-matches conditions against public text.
2. Coherence check #6 — key-narrative separation, incl. the `partner_goal` semantic leak rule.
3. Partner prompt rule 3 — "you physically cannot tell the other person what would change your mind."

All three prevent the condition being *written* somewhere visible or *instruct* the partner not
to say it. **None verifies the partner complied in a given episode.** The one detector that
does — `run_key_probes.py` AUDIT 1 — is a pre-run gate over 5 hand-written probes and never ran
over curriculum transcripts.

Why it matters: partner leaks a condition in turn 2 → learner satisfies it in turn 4 →
key_check passes → labeled `too_easy` / `frontier_solved`. That's a partner failure scored as a
learner skill, biasing difficulty **downward**.

### New script: `scripts/audit_gen90_partner_leaks.py`

Read-only post-hoc audit over a finished run. Imports the probe-harness prompts verbatim so
numbers are comparable to the pre-run acceptance bar. Runs AUDIT 1 (leak) + AUDIT 2 (early
yield / ignored trigger), and re-derives `key_check_passed` per §3.

```bash
uv run scripts/audit_gen90_partner_leaks.py                # full, both audits
uv run scripts/audit_gen90_partner_leaks.py --audits leak  # half the calls
uv run scripts/audit_gen90_partner_leaks.py --limit 100    # sample
```

Writes `results/<run>/analysis/partner_fidelity_audit.json`. Never touches the bank.

**STATUS: NOT YET RUN TO COMPLETION.** The full 300-attempt pass was killed mid-flight. The
only numbers we have are an **8-attempt smoke test** — far too small to quote, but it did flag
3/8 leaked, 6/8 early yield, 2 solved-with-fidelity-failure. If the early-yield rate holds
anywhere near that, Gate A is leaking badly and key_check is load-bearing. **Re-running this is
the top open task.** Lightning rate-limits above ~3 workers.

---

## Open decisions for HJ

1. **Patch 12?** Derive `key_check_passed` in code (§3). Recommend: yes for future runs, plus a
   re-derived-label robustness check for Gen-90.
2. **Finish the fidelity audit** (§5) — top priority; everything else is measured, this isn't.
3. **Re-judge the 29 inconsistent attempts** to confirm the indices (not the bool) are right.
4. **Mechanism-level equivalence** in key_check for genuine alternative paths? If done, log to a
   separate `conditions_met_equivalent` field so it can be measured and ablated — don't silently
   widen `conditions_met`.
5. **Judge parity for the 4.1-mini run.** `results/gen90_expel` predates the `fm_judge` cross-lab
   fix in `0f8a6ff`; confirm the two runs are judge-matched before comparing head-to-head.

## Using the reader

```bash
uv run transcript_reader/build.py                # regenerate reader.html (gitignored)
uv run transcript_reader/agent_server.py         # serve + AI + notes on :8765
open http://localhost:8765/
```

Set `AGENT_PORT=8766` (and rebuild/patch the port in `reader.html`) if 8765 is taken.

The reader now shows the **partner key** (collapsible) and a **per-attempt key-check verdict
row** — `C1 ✓ met` / `T3 ⚡ tripped` chips plus the judge's rationale — so you can see *why* a
high-GOAL attempt reads "not solved" without leaving the page.

**Review notes:** pick `as HX | HJ` in the top bar, use **📝 Notes** for per-scenario notes and
**☐ Reviewed** to stamp a timestamped checkmark. Both persist to
`transcript_reader/review_notes.json`, which **is tracked in git** — commit it to hand off.
The server must be running or notes stay browser-local (badge turns 🔴).
