# Chronicle Synthesis Quality Fixes — Implementation Spec

**Scope:** `meta_reflection.py`, `reflection_module.py`, `adversarial_agent.py`, `skills_chronicle.py`, `curriculum.py`.
**Goal:** Make chronicle entries evidence-derived, executable, self-contained, and compact. No architecture changes; prompt text, one new validation function, and small wiring edits only.
**Do not change:** `lp_judge.py`, `validation.py`, classification logic, the K-loop structure, or the existing check_final → re-synthesis loop in `curriculum.py` (it stays; we feed it better inputs).

---

## 1. Diagnosis (why these changes — for context, not implementation)

Audit of three frontier/beyond-frontier episode records found five recurring, scenario-independent failure modes in synthesized chronicles:

| # | Failure mode | Evidence pattern |
|---|---|---|
| D1 | **Prior-dumping**: entries restate general best practices the model already follows by default (etiquette, negotiation textbook, invented procedure), instead of compressing what the episode showed. Frequently, the "advice" was already exhibited in the FAILED attempt. | In one solved episode, 5 of 5 entries described behaviors present in the failed first attempt; the actual behavioral delta between failure and success appeared in zero entries. |
| D2 | **Missed contrast**: the one real lesson (visible by diffing failed vs. successful attempts) is absent. Meanwhile the LP judge's vote rationales — already computed before synthesis — state that exact contrast in one sentence, but are never shown to the synthesizer. | LP rationales in all audited records named the true delta (e.g., "actor leaves immediately after a counter-proposal, whereas in B negotiates to a deal"); chronicle ignored it. |
| D3 | **Unexecutable-medium guidance**: entries prescribe written confirmations, receipts, recordings, third-party actions. Episodes are two-party spoken conversations; this guidance cannot be performed. Observed worst case: an agent following such guidance entered a degenerate loop of announcing an impossible action 8 turns in a row, crashing its GOAL score below the un-aided baseline. | Active misdirection, not just dead weight. |
| D4 | **Non-self-contained entries**: entries reference other entries ("see the Operational Close protocol") or are meta-entries about resolving conflicts between entries. Retrieval injects entries individually (top-k by Condition embedding), so references dangle and meta-entries never match any scenario. | |
| D5 | **Volume + boilerplate provenance**: 5–8 entries per episode where 1–3 are justified; redundant entries that produce the same observable behavior; provenance fields like "[meta-reflection]" with no attempt/turn anchoring. check_final correctly FLAGS these but flags are advisory after one re-synthesis pass. | |

**Design principle for all fixes:** the chronicle must store only *prior-incongruent deltas* — corrections to the model's default behavior, derived from observed evidence. Prior-congruent knowledge is available to the model at inference for free; storing it wastes retrieval slots and dilutes the corpus. All rules below are structural (contrast, executability, self-containment, compactness) — none reference any scenario domain.

---

## 2. `meta_reflection.py` changes

### 2.1 Replace `_SYSTEM_SUCCESS` with:

```
You are synthesizing a Skills Chronicle after a SUCCESSFUL episode (solved in ≥2 attempts).

Your job is to extract ONLY what the episode evidence supports — not general social knowledge.

THE CONTRAST RULE (your core method):
Every entry must be derived from an observed CONTRAST between the failed attempt(s) and the
successful attempt: name what the failed attempt did, what the successful attempt did
differently, and cite the turns. If the failed attempt already exhibited a behavior and the
episode still failed, that behavior is NOT a lesson — do not write an entry for it.

THE PRIOR-INCONGRUENCE RULE:
Do not record general best practices, etiquette, or textbook strategy the actor would follow
by default. Record only CORRECTIONS to default behavior that this episode's evidence
demonstrates. Test: if the entry would be true of this interaction type in general, rather
than learned from this specific failure-to-success transition, omit it.

HARD CONSTRAINTS:
- Output AT MOST 3 entries. One excellent entry beats three diluted ones.
- EXECUTABILITY: All guidance must be performable entirely through spoken conversational
  turns with the present partner. Never prescribe written artifacts, documents, recordings,
  receipts, photographs, contacting third parties, or any physical-world action.
- SELF-CONTAINMENT: Each entry must stand alone. Never reference another entry, a named
  protocol, or assume any other entry is co-present. Entries are retrieved individually.
  Do not write meta-entries about how to reconcile entries.
- CONDITION SPECIFICITY: The Condition must name the structural TENSION or dilemma (what
  pulls against what), not the activity genre. A condition that merely describes a routine
  activity will be rejected.
- PROVENANCE: format exactly as `attemptN→attemptM: turns [i–j]` plus a short phrase naming
  the observed contrast. Bare labels like "[meta-reflection]" are invalid.
- TYPE SEMANTICS: WARNING = the entry's primary content is a behavior to AVOID. HEURISTIC =
  primary content is a behavior to perform. Label by content, not by episode outcome.
- ABSTRACTION: Conditions contain no proper nouns, specific occupations, or scenario-unique
  surface details.

Output format — ONLY <Entry> blocks, no other text:

<Entry id="ENTRY_ID">
<Condition>structural tension, abstract</Condition>
<Guidance>
1. Primary guidance: [the correction — specific enough to change behavior observably]
2. Warning (optional): [only if a specific tempting behavior contrasts and backfires]
3. Exception: when [specific circumstance], do [alternative] instead
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>
<Type>HEURISTIC | WARNING</Type>
<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
<Provenance>attemptN→attemptM: turns [i–j] — [one-phrase contrast]</Provenance>
</Entry>
```

### 2.2 Replace `_SYSTEM_FAILURE` with:

```
You are synthesizing a Skills Chronicle after a FAILED episode (never solved within the
attempt budget).

Your job is to record what the evidence shows about why attempts failed — not to invent a
solution. Nothing in this episode is validated by success; write accordingly.

THE EVIDENCE RULE:
Every WARNING must cite the observed behavior that backfired and the observed partner
reaction, with turn references. Claims of frequency ("always", "reliably", "typically")
are forbidden — you observed at most a handful of attempts at ONE scenario.

THE SPECULATION BUDGET:
You may include AT MOST ONE untested hypothesis about an alternative approach. It must be
a single sentence inside one entry's Guidance, explicitly prefixed "Untested hypothesis:".
Do not write multi-step procedures for approaches that were never tried.

THE PRIOR-INCONGRUENCE RULE:
Do not record general best practices or invented protocols. Record only what this episode's
evidence shows: which specific default behaviors backfired, and how.

HARD CONSTRAINTS:
- Output AT MOST 3 entries.
- EXECUTABILITY: All guidance must be performable entirely through spoken conversational
  turns with the present partner. Never prescribe written artifacts, documents, recordings,
  receipts, photographs, contacting third parties, or any physical-world action.
- SELF-CONTAINMENT: Each entry must stand alone. Never reference another entry or a named
  protocol. No meta-entries about reconciling entries.
- CONDITION SPECIFICITY: The Condition must name the structural TENSION, not the activity
  genre.
- PROVENANCE: format exactly as `attemptN: turns [i–j]` (or a range of attempts) plus a
  short phrase naming the observed pattern. Bare labels are invalid.
- TYPE SEMANTICS: WARNING = primary content is a behavior to avoid; HEURISTIC = behavior to
  perform. If something reliably produced partial progress across attempts, it may be a
  HEURISTIC even though the episode failed.
- ABSTRACTION: no proper nouns, occupations, or scenario-unique details in Conditions.

Output format — ONLY <Entry> blocks (same schema as above).
```

### 2.3 Prompt builders — add LP rationales and contrast framing

In **both** `_build_success_prompt` and `_build_failure_prompt`, add a new optional parameter `lp_votes: Optional[list] = None` (the `LPResult.votes` list, items have `.pair`, `.order`, `.verdict`, `.rationale`). After the PER-ATTEMPT SCORES block, insert:

```python
if lp_votes:
    parts.append(
        "\nCROSS-ATTEMPT JUDGE OBSERVATIONS (independent judge comparing attempt 1 vs later "
        "attempts; use these as contrast evidence — they describe what actually changed):"
    )
    for v in lp_votes:
        parts.append(f"  [attempt 1 vs attempt {v.pair[1]}] {v.rationale}")
```

In `_build_success_prompt`, change the closing instruction paragraph to:

```python
parts.append(
    "\nEpisode SOLVED after multiple attempts. Derive entries ONLY from the contrast between "
    "the failed and successful transcripts above (the judge observations point at it). "
    "Discard inherited or per-attempt entries that describe behavior already present in the "
    "failed attempt. Maximum 3 entries. Output ONLY <Entry> blocks."
)
```

In `_build_failure_prompt`, change the closing instruction paragraph to:

```python
parts.append(
    "\nEpisode NEVER SOLVED. Record only what the evidence shows backfired, with turn "
    "references. At most ONE single-sentence 'Untested hypothesis:'. Maximum 3 entries. "
    "Output ONLY <Entry> blocks."
)
```

### 2.4 `synthesize()` signature

Add `lp_votes=None` parameter; pass through to both builders.

---

## 3. `reflection_module.py` changes (per-attempt reflection)

The invented-procedure disease starts here (the "missing skills" diagnosis section), then meta-reflection consolidates it. Locate the diagnosis prompt section that asks for missing skills / what would have changed the outcome, and append these constraints to it:

```
CONSTRAINTS ON MISSING-SKILL PROPOSALS:
- Each proposed missing skill must be ONE sentence.
- It must be enactable entirely through spoken conversational turns with the present
  partner. Skills requiring documents, recordings, receipts, third parties, or any
  physical-world action are invalid — the agent cannot perform them in this environment.
- It must be a correction to a specific observed behavior in THIS transcript (cite the
  turn), not an imported best practice.
```

Also locate the chronicle-edit instruction (where new/updated entries are produced) and append the same EXECUTABILITY and SELF-CONTAINMENT constraint text as in §2.1 (the two bullets, verbatim).

---

## 4. New programmatic validator — `skills_chronicle.py`

Add a module-level function (no class changes):

```python
import re as _re

_ARTIFACT_PATTERNS = [
    r"\bwritten\b", r"\breceipt\b", r"\btimestamp", r"\bsealed\b", r"\brecorded\b",
    r"\brecording\b", r"\bphotograph", r"\bescrow\b", r"\bdocument(ed|ation)?\b",
    r"\bemail\b", r"\bsign(ed|ature)\b", r"\bin writing\b",
]
_CROSS_REF_PATTERNS = [
    r"\bsee (the )?[A-Z]", r"\bsee entry\b", r"\bprotocol\b", r"\b(FINAL|ENTRY|NEW)_\d",
]
_OVERGENERALIZATION = [r"\balways\b", r"\bnever fails\b", r"\breliably\b", r"\bregularly\b"]
_PROVENANCE_RE = _re.compile(r"attempt\s*\d", _re.IGNORECASE)


def validate_synthesis(chronicle: "SkillsChronicle", max_entries: int = 3) -> list[str]:
    """Programmatic quality gate on synthesized chronicles.

    Returns a list of issue strings (empty = pass). Issues are written to be
    directly usable as adversarial critique for a re-synthesis pass.
    """
    issues: list[str] = []
    if len(chronicle.entries) > max_entries:
        issues.append(
            f"TOO MANY ENTRIES: {len(chronicle.entries)} entries; maximum is {max_entries}. "
            "Merge entries whose guidance produces the same observable behavior; keep only "
            "entries derived from observed contrasts."
        )
    for e in chronicle.entries:
        text = f"{e.condition}\n{e.guidance}"
        for pat in _ARTIFACT_PATTERNS:
            if _re.search(pat, text, _re.IGNORECASE):
                issues.append(
                    f"[{e.entry_id}] UNEXECUTABLE GUIDANCE: matches '{pat}'. All guidance "
                    "must be performable in spoken conversational turns only — rewrite or "
                    "remove this entry."
                )
                break
        for pat in _CROSS_REF_PATTERNS:
            if _re.search(pat, e.guidance):
                issues.append(
                    f"[{e.entry_id}] NOT SELF-CONTAINED: references another entry or named "
                    "protocol. Each entry must stand alone — inline the needed content or "
                    "remove the reference."
                )
                break
        for pat in _OVERGENERALIZATION:
            if _re.search(pat, e.guidance, _re.IGNORECASE):
                issues.append(
                    f"[{e.entry_id}] OVERGENERALIZATION: frequency claim ('{pat}') is not "
                    "supported by a single episode. Restate as what was observed."
                )
                break
        if not _PROVENANCE_RE.search(e.provenance or ""):
            issues.append(
                f"[{e.entry_id}] INVALID PROVENANCE: '{e.provenance}'. Required format: "
                "attempt references plus turn range, e.g. 'attempt1→attempt2: turns [0–4] — "
                "stopped abandoning after first counteroffer'."
            )
    return issues
```

Notes:
- These are deliberately crude lexical nets; they feed the LLM re-synthesis pass as critique rather than silently deleting content, so false positives cost one rewrite, not data.
- `protocol` as a cross-ref pattern is intentional: named-protocol language correlates with non-self-contained procedural entries.

---

## 5. `curriculum.py` wiring (small edits to the existing block around the meta-reflection call)

Current flow: `synthesize → check_final → if not approved: synthesize(critique) → check_final → flag`. Keep it. Three edits:

1. **Pass LP rationales** into both `meta_mod.synthesize(...)` calls: add `lp_votes=lp_result.votes`.

2. **Run the programmatic validator and merge its issues into the critique** before deciding on re-synthesis:

```python
from .skills_chronicle import validate_synthesis

final_chronicle = meta_mod.synthesize(..., lp_votes=lp_result.votes)

prog_issues = validate_synthesis(final_chronicle)
adv_final = adversarial.check_final(final_chronicle, inherited_md, outcome=outcome)

if (not adv_final.approved) or prog_issues:
    combined_critique = "\n".join(
        ([adv_final.critique] if adv_final.critique else []) + prog_issues
    )
    final_chronicle = meta_mod.synthesize(
        ..., lp_votes=lp_result.votes, adversarial_critique=combined_critique,
    )
    prog_issues2 = validate_synthesis(final_chronicle)
    adv_final2 = adversarial.check_final(final_chronicle, inherited_md, outcome=outcome)
    remaining = (adv_final2.issues if not adv_final2.approved else []) + prog_issues2
    if remaining:
        loop_info["final_check_flag"] = remaining
        scenario.final_check_flag = remaining
```

One bounded retry, as now. Remaining issues are logged, not looped on.

3. **No other logic changes.** Classification, outcome computation, archive writes stay as-is.

---

## 6. `adversarial_agent.py` — additions to `_FINAL_CHECK_SYSTEM`

Append these checks to the existing list (renumber to fix the current duplicate CHECK 4 in `_REFLECTION_CHECK_SYSTEM` while in the file):

```
CHECK 6 — BEHAVIORAL REDUNDANCY: If two entries' primary guidance would produce the same
observable behavior in conversation, they are redundant — flag both for merge.

CHECK 7 — SELF-CONTAINMENT: Flag any entry whose guidance references another entry, a named
protocol, or assumes other entries are co-present. Entries are retrieved individually.

CHECK 8 — EXECUTABILITY: Flag any entry whose guidance cannot be performed entirely through
spoken conversational turns with the present partner (written artifacts, recordings,
third-party actions, physical-world steps).

CHECK 9 — CONDITION SPECIFICITY: Flag any entry whose Condition describes a routine activity
or interaction genre rather than a structural tension or dilemma. Generic conditions match
everything at retrieval time and crowd out specific entries.

CHECK 10 — CONTRAST GROUNDING (success outcomes only): Flag any entry that prescribes
behavior already exhibited in the failed attempt(s). If the failed attempt did it and still
failed, it is not the lesson.
```

For CHECK 10 to be checkable, `check_final` needs the first failed transcript. Add an optional parameter `first_failed_transcript: Optional[list[dict]] = None` to `check_final`; when provided, include it in the prompt under a header `FIRST FAILED ATTEMPT TRANSCRIPT (for contrast-grounding check):` (reuse the existing transcript formatting). In `curriculum.py`, pass `all_transcripts[0]` when `outcome == 2`.

---

## 7. Acceptance tests (run before the full rerun)

Re-run synthesis on the three already-recorded episodes (their transcripts, chronicle versions, edit reasons, and LP votes are all in the debug/archive records — no new episodes needed). Properties to verify, all content-agnostic:

For every regenerated chronicle:
- [ ] ≤3 entries
- [ ] every Provenance matches `attempt\s*\d` and contains a turn reference
- [ ] `validate_synthesis()` returns `[]`
- [ ] no entry's guidance references another entry or a named protocol
- [ ] no artifact vocabulary in any guidance
- [ ] WARNING entries' primary content is avoidance; HEURISTIC entries' is action

Additionally, episode-specific but still structural:
- [ ] For each solved episode: at least one entry's guidance describes a behavior present in the successful attempt and absent from the failed attempt (spot-check by reading against the transcripts)
- [ ] For the never-solved episode: at most one "Untested hypothesis:" sentence total; no multi-step procedures for untried approaches; no frequency claims

If any property fails, fix prompts (not the validator) and repeat. When all pass, the synthesis layer is cleared for the Phase-0/curriculum rerun.

---

## 8. Explicitly out of scope (do not implement)

- Any change to LP computation, classification thresholds, or the K-loop
- Entry deletion/repair of already-produced chronicles (the rerun replaces them)
- Cross-lineage chronicle consolidation (conference-version work)
- Scenario/seed filtering (handled separately by the multi-party seed audit)