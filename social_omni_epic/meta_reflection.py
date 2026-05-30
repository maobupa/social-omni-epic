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
1. Primary guidance
2. Exception: when [condition], do [alternative] instead
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
1. Primary guidance (what was tried, what worked partially)
2. Warning: [structural trap to avoid]
3. Exception: when [condition], [alternative approach — untested but plausible]
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>
<Type>HEURISTIC | WARNING</Type>
<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
<Provenance>[carry forward and append "meta-reflection (failed)"]</Provenance>
</Entry>

Only output <Entry> blocks. No <Diagnosis>, no <EditReason>, no commentary."""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_meta_prompt(
    chronicle_versions: list[SkillsChronicle],
    transcripts: list[list[dict]],
    edit_reasons: dict[str, str],
    outcome: int,
    scenario: SocialScenario,
    anchor_task: Optional[SocialScenario],
) -> str:
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

    if anchor_task and anchor_task.social_dynamic:
        parts.append(
            f"PARENT SCENARIO SOCIAL DYNAMIC (for abstraction check): {anchor_task.social_dynamic}"
        )

    # Show chronicle evolution across versions
    if len(chronicle_versions) == 1:
        parts.append("\nINITIAL CHRONICLE (pre-episode):")
        text = chronicle_versions[0].to_markdown()
        parts.append(text if text else "(empty)")
    else:
        parts.append(f"\nCHRONICLE EVOLUTION ({len(chronicle_versions)} versions):")
        for i, cv in enumerate(chronicle_versions):
            label = "initial" if i == 0 else f"after attempt {i}"
            parts.append(f"--- Version {i} ({label}) ---")
            text = cv.to_markdown()
            parts.append(text if text else "(empty)")

    # All transcripts
    n = len(transcripts)
    if n:
        parts.append(f"\nALL TRANSCRIPTS ({n} attempt{'s' if n > 1 else ''}):")
        for i, t in enumerate(transcripts, 1):
            label = "FAILED" if (outcome == 3 or i < n) else "SUCCESSFUL"
            truncated = _format_transcript(t, i, max_chars=2000)
            parts.append(f"[{label}] {truncated}")

    # Accumulated edit reasons
    if edit_reasons:
        parts.append("\nALL EDIT REASONS (from intermediate reflections):")
        for eid, reason in edit_reasons.items():
            parts.append(f"  [{eid}]: {reason}")

    outcome_label = "SOLVED (multiple attempts)" if outcome == 2 else "FAILED (all attempts)"
    parts.append(
        f"\nEpisode outcome: {outcome_label}. "
        "Synthesize a final coherent Skills Chronicle from all the above. "
        "Reconcile contradictions. Output ONLY <Entry> blocks."
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
    ) -> SkillsChronicle:
        """Synthesize a final skills chronicle from all attempts.

        outcome: 2 = solved after ≥2 attempts, 3 = never solved.
        chronicle_versions: one entry per reflection step (includes initial).
        transcripts: all episode transcripts in order.

        Returns a new SkillsChronicle, or the last chronicle version on failure.
        """
        system = _SYSTEM_SUCCESS if outcome == 2 else _SYSTEM_FAILURE
        prompt = _build_meta_prompt(
            chronicle_versions, transcripts, edit_reasons, outcome, scenario, anchor_task
        )

        fallback = deepcopy(chronicle_versions[-1]) if chronicle_versions else SkillsChronicle()

        for attempt in range(self.max_retries):
            try:
                llm_output = self.fm.query(system, prompt, temperature=0.3)
                result = SkillsChronicle.from_markdown(llm_output)
                if result.entries:
                    return result
                # Empty parse — retry
            except Exception:
                pass

        return fallback
