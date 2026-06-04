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

_SYSTEM_SUCCESS = """You are a reflective coach synthesizing a Skills Chronicle after a SUCCESSFUL episode (solved in ≥2 attempts).

You have access to:
- All chronicle versions (one per reflection iteration)
- All transcripts (both failed and the final successful attempt)
- All EditReasons produced by intermediate reflections

Your job is to produce a FINAL, coherent Skills Chronicle — a set of <Entry> blocks — that:

1. CONSOLIDATE: Merge redundant entries that cover the same condition into one, combining the best Guidance from both.
2. RECONCILE: When earlier and later entries contradict each other, produce a single entry with an exception clause ("Exception: when X, do Y instead").
3. WEIGHT TOWARD HEURISTICS: The final chronicle should be predominantly HEURISTIC entries (what the agent should DO), not WARNING entries. Retain WARNING entries only where they provide essential contrast.
4. CAPTURE WHAT WORKED: The final Guidance should reflect what the successful attempt did differently from the failed ones.
5. RETAIN ABSTRACTION: Conditions must remain abstract — no proper nouns, specific occupations, scenario-unique details.

Output format — ONLY the <Entry> blocks, no other text:

<Entry id="ENTRY_ID">
<Condition>abstract structural pattern</Condition>
<Guidance>
1. Primary guidance: [what to do or not do — specific enough to change behavior observably]
2. Warning (optional): [only if a specific tempting behavior contrasts with the primary guidance and backfires]
3. Exception: when [a specific circumstance within the above Condition makes the primary guidance inappropriate], do [alternative] instead
(add further numbered Exception clauses as needed)
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>
<Type>HEURISTIC | WARNING</Type>
<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
<Provenance>[carry forward and append "meta-reflection"]</Provenance>
</Entry>

Only output <Entry> blocks. No <Diagnosis>, no <EditReason>, no commentary."""


_SYSTEM_FAILURE = """You are a reflective coach synthesizing a Skills Chronicle after a FAILED episode (never solved within the attempt budget).

You have access to:
- All chronicle versions (one per reflection iteration)
- All transcripts from all failed attempts
- All EditReasons produced by intermediate reflections

Your job is to produce a FINAL, coherent Skills Chronicle — a set of <Entry> blocks — that:

1. DOCUMENT STRUCTURAL RESISTANCE: Identify what made this scenario type persistently difficult. Add or strengthen WARNING entries that name the structural traps.
2. PROPOSE ALTERNATIVES: Where a strategy was tried and failed repeatedly, the Guidance should document what a DIFFERENT approach might look like (even untested), marked as a WARNING to flag uncertainty.
3. RECONCILE CONTRADICTIONS: When intermediate reflections produced contradictory edits, synthesize them into one entry with exception clauses.
4. WEIGHT TOWARD WARNINGS: The final chronicle should include substantial WARNING entries. HEURISTIC entries should be limited to what reliably worked across all attempts (if anything did).
5. RETAIN ABSTRACTION: Conditions must remain abstract — no proper nouns, occupations, or scenario-unique details.

Output format — ONLY the <Entry> blocks, no other text:

<Entry id="ENTRY_ID">
<Condition>abstract structural pattern</Condition>
<Guidance>
1. Primary guidance: [what to do or not do — specific enough to change behavior observably]
2. Warning (optional): [only if a specific tempting behavior contrasts with the primary guidance and backfires]
3. Exception: when [a specific circumstance within the above Condition makes the primary guidance inappropriate], do [alternative] instead
(add further numbered Exception clauses as needed)
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>
<Type>HEURISTIC | WARNING</Type>
<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
<Provenance>[carry forward and append "meta-reflection (failed)"]</Provenance>
</Entry>

Only output <Entry> blocks. No <Diagnosis>, no <EditReason>, no commentary."""


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
) -> str:
    """Outcome=2: per-attempt reflection already did the diagnosis work.
    Pass only the final chronicle + edit reasons for a lightweight cleanup pass."""
    parts = _common_header(scenario, 2, anchor_task, attempt_scores)

    final = chronicle_versions[-1].to_markdown() if chronicle_versions else ""
    parts.append("\nFINAL CHRONICLE (after all per-attempt reflections):")
    parts.append(final if final else "(empty)")

    if edit_reasons:
        parts.append("\nEDIT REASONS from per-attempt reflections (why entries were added/changed):")
        for eid, reason in edit_reasons.items():
            parts.append(f"  [{eid}]: {reason}")

    parts.append(
        "\nEpisode SOLVED after multiple attempts. The per-attempt reflections already "
        "diagnosed failures and updated the chronicle. Your job is a cleanup pass only: "
        "merge redundant entries covering the same condition, resolve any contradictions "
        "into exception clauses, ensure all Conditions are abstract (no proper nouns or "
        "scenario-unique details), and confirm Guidance is specific enough to change "
        "behavior observably. Output ONLY <Entry> blocks."
    )
    return "\n\n".join(parts)


def _build_failure_prompt(
    chronicle_versions: list[SkillsChronicle],
    transcripts: list[list[dict]],
    edit_reasons: dict[str, str],
    scenario: SocialScenario,
    anchor_task: Optional[SocialScenario],
    attempt_scores: Optional[list[dict]],
) -> str:
    """Outcome=3: per-attempt reflections may have drifted into wrong diagnoses across
    repeated failures. Show first + last transcript so meta-reflection can see the
    structural resistance pattern without re-reading all K attempts."""
    parts = _common_header(scenario, 3, anchor_task, attempt_scores)

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
        "\nEpisode NEVER SOLVED. Diagnose the structural resistance pattern that made "
        "this scenario type persistently difficult — what did every attempt get wrong, "
        "and what alternative approach might work? Add or strengthen WARNING entries "
        "that name the structural traps. Merge/reconcile contradictions from intermediate "
        "reflections into exception clauses. Conditions must remain abstract. "
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
    ) -> SkillsChronicle:
        """Synthesize a final skills chronicle from all attempts.

        outcome: 2 = solved after ≥2 attempts (cleanup pass — no transcripts needed).
                 3 = never solved (structural resistance diagnosis — first+last transcript).
        chronicle_versions: one entry per reflection step (includes initial).
        transcripts: all episode transcripts in order.

        Returns a new SkillsChronicle, or the last chronicle version on failure.
        """
        if outcome == 2:
            system = _SYSTEM_SUCCESS
            prompt = _build_success_prompt(
                chronicle_versions, edit_reasons, scenario, anchor_task, attempt_scores
            )
        else:
            system = _SYSTEM_FAILURE
            prompt = _build_failure_prompt(
                chronicle_versions, transcripts, edit_reasons, scenario, anchor_task,
                attempt_scores
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
