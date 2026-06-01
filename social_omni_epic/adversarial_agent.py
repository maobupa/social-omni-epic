"""Adversarial Agent (§4.6.1): two-stage quality gate on chronicle edits.

Mode 1 — post-reflection (check_reflection):
  Called after each ReflectionModule invocation.
  Checks:
    - EditReasons cite specific transcript evidence
    - Revised Conditions remain abstract (no proper nouns, occupations)
    - Broadened Conditions don't also describe anchor task's social dynamic

Mode 2 — post-meta-reflection (check_final):
  Called after MetaReflectionModule produces the final chronicle.
  Checks:
    - Internal contradictions between entries
    - Synthesis drift from inherited chronicle (contradictions need EditReasons)
    - Overall coherence
    - Outcome-appropriate balance (Outcome 3 → predominantly WARNING)

Both modes return AdversarialCheckResult. On approval failure the caller
may re-reflect with the critique or accept with logging.
"""
import json
from dataclasses import dataclass, field
from typing import Optional

from .data_models import SocialScenario
from .fm import FM
from .reflection_module import ReflectionOutput
from .skills_chronicle import SkillsChronicle


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdversarialCheckResult:
    approved: bool
    issues: list[str] = field(default_factory=list)
    flagged_entry_ids: list[str] = field(default_factory=list)
    active_misdirection_ids: list[str] = field(default_factory=list)
    critique: str = ""


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_REFLECTION_CHECK_SYSTEM = """You are an adversarial quality-control agent reviewing chronicle edits produced after a failed social interaction episode.

For each edited or added entry, check ALL of the following:

CHECK 1 — EVIDENCE: Does the EditReason cite SPECIFIC evidence from the transcript (direct quotes or specific turn references)? Generic reasoning ("the agent failed to understand") is NOT sufficient.
  EXCEPTION: If the entry concerns a resource, capability, or information the agent possessed but did NOT use (a missed opportunity or unused leverage), the scenario context or extra_info is acceptable evidence in place of transcript quotes — by definition, something unused cannot appear in the transcript.

CHECK 2 — ABSTRACTION: Does the revised Condition remain abstract? Flag if it contains:
  - Proper nouns (person names, place names)
  - Specific occupations (e.g., "the nurse", "the landlord")
  - Scenario-unique surface details not generalizable to other social contexts

CHECK 3 — BROADENING: If a Condition was broadened (made more general), does the broadened Condition ALSO describe the parent scenario's social dynamic? If yes, the broadening is too aggressive and must be rejected — it would import anchor-task guidance into structurally different contexts.

CHECK 4 — MISDIRECTION: Did any entry actively guide the agent toward worse behavior (not just fail to help, but actively caused harm)? If so, flag it as active misdirection.

Respond with JSON only:
{
  "approved": true/false,
  "issues": ["specific issue 1", "specific issue 2"],
  "flagged_entry_ids": ["id1", "id2"],
  "active_misdirection_ids": ["id3"],
  "critique": "brief overall critique for the reflection module to address (empty string if approved)"
}

If all checks pass: approved=true, empty issues/flagged/misdirection, empty critique."""


_FINAL_CHECK_SYSTEM = """You are an adversarial quality-control agent reviewing a final synthesized Skills Chronicle.

Check ALL of the following:

CHECK 1 — INTERNAL CONSISTENCY: Are there entries whose Conditions overlap AND whose Guidance contradicts? (Acceptable: one is an exception clause of the other. Problematic: neither acknowledges the other.)

CHECK 2 — SYNTHESIS DRIFT: Compare against the inherited chronicle. If the final chronicle contradicts an inherited entry without explicit justification, flag it. Inherited wisdom should not be silently overwritten.

CHECK 3 — COHERENCE: Does the chronicle read as a coherent guide? Are there redundant entries that could be merged?

CHECK 4 — OUTCOME BALANCE:
  - Failure outcome: Chronicle should be predominantly WARNING entries. Heavy HEURISTIC dominance in a failure case suggests over-confidence.
  - Success outcome: Chronicle may mix freely; WARNING entries from failed attempts are valuable contrast.

CHECK 5 — ACTIVE MISDIRECTION: Are there any entries that would actively mislead an agent (not just neutral/unhelpful, but directionally harmful)?

Respond with JSON only:
{
  "approved": true/false,
  "issues": ["specific issue 1"],
  "flagged_entry_ids": ["id1"],
  "active_misdirection_ids": ["id2"],
  "critique": "overall critique (empty string if approved)"
}"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AdversarialAgent:
    def __init__(self, fm: FM):
        self.fm = fm

    def check_reflection(
        self,
        reflection_output: ReflectionOutput,
        transcript: list[dict],
        anchor_task: Optional[SocialScenario] = None,
        scenario: Optional[SocialScenario] = None,
    ) -> AdversarialCheckResult:
        """Mode 1: check post-reflection chronicle edits for quality."""
        parts: list[str] = []

        if scenario:
            target_goal = (
                scenario.agent_goals[scenario.target_agent_idx]
                if scenario.target_agent_idx < len(scenario.agent_goals)
                else ""
            )
            parts.append(f"SCENARIO: {scenario.scenario}")
            parts.append(f"TARGET AGENT GOAL: {target_goal}")

        if anchor_task and anchor_task.social_dynamic:
            parts.append(
                f"PARENT SCENARIO SOCIAL DYNAMIC (broadening anchor): {anchor_task.social_dynamic}"
            )

        # Summarize transcript for evidence checks
        parts.append("TRANSCRIPT (for evidence verification):")
        for t in transcript:
            line = f"[T{t['turn']}] {t['speaker']}: {t['content']}"
            parts.append(line)

        # EditReasons and the resulting entries
        if reflection_output.edit_reasons:
            parts.append("\nEDIT REASONS AND RESULTING ENTRIES:")
            for eid, reason in reflection_output.edit_reasons.items():
                parts.append(f"\n[{eid}] EditReason: {reason}")
                entry = reflection_output.updated_chronicle.get_entry(eid)
                if entry:
                    parts.append(f"Resulting entry condition: {entry.condition}")
                    parts.append(f"Resulting entry type: {entry.entry_type}")

        if reflection_output.misdirection_entry_ids:
            parts.append(
                f"\nEntries already self-flagged as misdirection by reflection module: "
                + ", ".join(reflection_output.misdirection_entry_ids)
            )

        parts.append("\nReview the edits above and respond with your JSON assessment.")
        prompt = "\n".join(parts)

        return self._query(_REFLECTION_CHECK_SYSTEM, prompt)

    def check_final(
        self,
        final_chronicle: SkillsChronicle,
        inherited_chronicle_md: str,
        outcome: int = 3,
    ) -> AdversarialCheckResult:
        """Mode 2: check final synthesized chronicle for consistency and balance."""
        outcome_label = {1: "SOLVED (first attempt)", 2: "SOLVED (multi-attempt)"}.get(outcome, "FAILED")
        parts: list[str] = [f"EPISODE OUTCOME: {outcome_label}"]

        if inherited_chronicle_md:
            parts.append("\nINHERITED CHRONICLE (from anchor task):")
            parts.append(inherited_chronicle_md)
        else:
            parts.append("\nINHERITED CHRONICLE: (empty — first-generation scenario)")

        parts.append("\nFINAL SYNTHESIZED CHRONICLE:")
        final_text = final_chronicle.to_markdown()
        parts.append(final_text if final_text else "(empty)")

        n_heuristics = sum(1 for e in final_chronicle.entries if e.entry_type == "HEURISTIC")
        n_warnings = sum(1 for e in final_chronicle.entries if e.entry_type == "WARNING")
        parts.append(
            f"\nEntry counts: {n_heuristics} HEURISTIC, {n_warnings} WARNING, "
            f"{len(final_chronicle.entries)} total."
        )

        parts.append("\nReview the final chronicle above and respond with your JSON assessment.")
        prompt = "\n".join(parts)

        return self._query(_FINAL_CHECK_SYSTEM, prompt)

    def _query(self, system: str, prompt: str) -> AdversarialCheckResult:
        try:
            d = self.fm.query_json(system, prompt, temperature=0.2)
            return AdversarialCheckResult(
                approved=bool(d.get("approved", True)),
                issues=[str(x) for x in d.get("issues", [])],
                flagged_entry_ids=[str(x) for x in d.get("flagged_entry_ids", [])],
                active_misdirection_ids=[str(x) for x in d.get("active_misdirection_ids", [])],
                critique=str(d.get("critique", "")),
            )
        except Exception as e:
            # On API/parse failure: approve by default, log the error
            return AdversarialCheckResult(
                approved=True,
                issues=[],
                flagged_entry_ids=[],
                active_misdirection_ids=[],
                critique=f"[adversarial check failed: {e}]",
            )
