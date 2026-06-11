"""Meta-Reflection Module (§4.7): cross-attempt synthesis after an episode ends.

Unlike per-attempt Reflection, MetaReflection reads ALL chronicle versions,
ALL transcripts, and ALL EditReasons simultaneously and produces a single
coherent skills_final.md.

Outcome values:
  1 — solved on first attempt (no reflection needed, caller skips this module)
  2 — solved after ≥2 attempts (HEURISTIC-dominant synthesis)
  3 — never solved (WARNING-dominant synthesis)

Key responsibilities:
  - Reconcile contradictory edits across attempts
  - Resolve contradictions into nuanced entries with exception clauses
  - For Outcome 2: consolidate what worked; retain WARNINGs as contrast
  - For Outcome 3: document structural failure; propose alternative approaches

Output: complete list of <Entry> blocks, parsed by SkillsChronicle.from_markdown().
"""
from copy import deepcopy
from typing import Optional

from .data_models import SocialScenario
from .fm import FM
from .reflection_module import _format_transcript
from .skills_chronicle import SkillsChronicle


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_SUCCESS = """You are synthesizing a Skills Chronicle after a SUCCESSFUL episode (solved in ≥2 attempts).

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
</Entry>"""


_SYSTEM_FAILURE = """You are synthesizing a Skills Chronicle after a FAILED episode (never solved within the
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
<Provenance>attemptN: turns [i–j] — [one-phrase observed pattern]</Provenance>
</Entry>"""


# ---------------------------------------------------------------------------
# Prompt builders (split by outcome)
# ---------------------------------------------------------------------------

def _common_header(
    scenario: SocialScenario,
    outcome: int,
    anchor_task: Optional[SocialScenario],
    attempt_scores: Optional[list[dict]],
) -> list[str]:
    parts: list[str] = []
    target_goal = (
        scenario.agent_goals[scenario.target_agent_idx]
        if scenario.target_agent_idx < len(scenario.agent_goals)
        else ""
    )
    parts.append(f"SCENARIO: {scenario.scenario}")
    parts.append(f"TARGET AGENT GOAL: {target_goal}")
    parts.append(f"INTERACTION TYPE: {scenario.interaction_type}")
    parts.append(f"OUTCOME: {'SOLVED after multiple attempts' if outcome == 2 else 'NEVER SOLVED'}")
    if attempt_scores:
        score_lines = [
            f"  Attempt {s['attempt']}: goal_diag={s['scores'].get('goal',0):.1f}  "
            f"overall_diag={s['scores'].get('overall_score',0):.2f}  "
            f"{'SOLVED' if s.get('solved') else 'failed'}"
            for s in attempt_scores
        ]
        parts.append("PER-ATTEMPT SCORES:\n" + "\n".join(score_lines))
    if anchor_task and anchor_task.social_dynamic:
        parts.append(
            f"PARENT SCENARIO SOCIAL DYNAMIC (for abstraction check): {anchor_task.social_dynamic}"
        )
    return parts


def _build_success_prompt(
    chronicle_versions: list[SkillsChronicle],
    edit_reasons: dict[str, str],
    scenario: SocialScenario,
    anchor_task: Optional[SocialScenario],
    attempt_scores: Optional[list[dict]],
    transcripts: Optional[list[list[dict]]] = None,
    lp_votes=None,
) -> str:
    parts = _common_header(scenario, 2, anchor_task, attempt_scores)

    if lp_votes:
        parts.append(
            "\nCROSS-ATTEMPT JUDGE OBSERVATIONS (independent judge comparing attempt 1 vs later "
            "attempts; use these as contrast evidence — they describe what actually changed):"
        )
        for v in lp_votes:
            parts.append(f"  [attempt 1 vs attempt {v.pair[1]}] {v.rationale}")

    final = chronicle_versions[-1].to_markdown() if chronicle_versions else ""
    parts.append("\nFINAL CHRONICLE (after all per-attempt reflections):")
    parts.append(final if final else "(empty)")

    if edit_reasons:
        parts.append("\nEDIT REASONS from per-attempt reflections (why entries were added/changed):")
        for eid, reason in edit_reasons.items():
            parts.append(f"  [{eid}]: {reason}")

    if transcripts and len(transcripts) >= 2:
        parts.append("\nFIRST ATTEMPT TRANSCRIPT (FAILED — shows baseline approach):")
        parts.append(_format_transcript(transcripts[0], 1, max_chars=2000))
        parts.append(f"\nFINAL ATTEMPT TRANSCRIPT (SUCCESSFUL, attempt {len(transcripts)}):")
        parts.append(_format_transcript(transcripts[-1], len(transcripts), max_chars=2000))

    parts.append(
        "\nEpisode SOLVED after multiple attempts. Derive entries ONLY from the contrast between "
        "the failed and successful transcripts above (the judge observations point at it). "
        "Discard inherited or per-attempt entries that describe behavior already present in the "
        "failed attempt. Maximum 3 entries. Output ONLY <Entry> blocks."
    )
    return "\n\n".join(parts)


def _build_failure_prompt(
    chronicle_versions: list[SkillsChronicle],
    transcripts: list[list[dict]],
    edit_reasons: dict[str, str],
    scenario: SocialScenario,
    anchor_task: Optional[SocialScenario],
    attempt_scores: Optional[list[dict]],
    lp_votes=None,
) -> str:
    parts = _common_header(scenario, 3, anchor_task, attempt_scores)

    if lp_votes:
        parts.append(
            "\nCROSS-ATTEMPT JUDGE OBSERVATIONS (independent judge comparing attempt 1 vs later "
            "attempts; use these as contrast evidence — they describe what actually changed):"
        )
        for v in lp_votes:
            parts.append(f"  [attempt 1 vs attempt {v.pair[1]}] {v.rationale}")

    final = chronicle_versions[-1].to_markdown() if chronicle_versions else ""
    parts.append("\nFINAL CHRONICLE (after all per-attempt reflections):")
    parts.append(final if final else "(empty)")

    if transcripts:
        parts.append("\nFIRST ATTEMPT TRANSCRIPT (FAILED):")
        parts.append(_format_transcript(transcripts[0], 1, max_chars=2000))
        if len(transcripts) > 1:
            parts.append(f"\nLAST ATTEMPT TRANSCRIPT (FAILED, attempt {len(transcripts)}):")
            parts.append(_format_transcript(transcripts[-1], len(transcripts), max_chars=2000))

    if edit_reasons:
        parts.append("\nEDIT REASONS from per-attempt reflections:")
        for eid, reason in edit_reasons.items():
            parts.append(f"  [{eid}]: {reason}")

    parts.append(
        "\nEpisode NEVER SOLVED. Record only what the evidence shows backfired, with turn "
        "references. At most ONE single-sentence 'Untested hypothesis:'. Maximum 3 entries. "
        "Output ONLY <Entry> blocks."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetaReflectionModule:
    def __init__(self, fm: FM, max_retries: int = 2):
        self.fm = fm
        self.max_retries = max_retries

    def synthesize(
        self,
        chronicle_versions: list[SkillsChronicle],
        transcripts: list[list[dict]],
        edit_reasons: dict[str, str],
        outcome: int,
        scenario: SocialScenario,
        anchor_task: Optional[SocialScenario] = None,
        attempt_scores: Optional[list[dict]] = None,
        adversarial_critique: str = "",
        lp_votes=None,
    ) -> SkillsChronicle:
        """Synthesize a final skills chronicle from all attempts.

        outcome: 2 = solved after ≥2 attempts; 3 = never solved.
        lp_votes: list[VoteRecord] from LPResult — injected as contrast evidence.
        Returns a new SkillsChronicle, or the last chronicle version on failure.
        """
        if outcome == 2:
            system = _SYSTEM_SUCCESS
            prompt = _build_success_prompt(
                chronicle_versions, edit_reasons, scenario, anchor_task, attempt_scores,
                transcripts=transcripts, lp_votes=lp_votes,
            )
        else:
            system = _SYSTEM_FAILURE
            prompt = _build_failure_prompt(
                chronicle_versions, transcripts, edit_reasons, scenario, anchor_task,
                attempt_scores, lp_votes=lp_votes,
            )

        if adversarial_critique:
            prompt = (
                prompt + "\n\nADVERSARIAL CRITIQUE FROM PRIOR SYNTHESIS PASS "
                "(address these issues in your output):\n" + adversarial_critique
            )

        fallback = deepcopy(chronicle_versions[-1]) if chronicle_versions else SkillsChronicle()

        for attempt in range(self.max_retries):
            try:
                llm_output = self.fm.query(system, prompt, temperature=0.3)
                result = SkillsChronicle.from_markdown(llm_output)
                if result.entries:
                    return result
            except Exception:
                pass

        return fallback
