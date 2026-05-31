"""Reflection Module (§4.6): per-attempt chronicle editing after a failed episode.

Each invocation at attempt K receives:
  - Current skills chronicle (inherited + edits from attempts 1..K-1)
  - Scenario description and goals
  - All transcripts from attempts 1..K-1
  - All intermediate chronicle versions
  - All prior EditReasons
  - The transcript from the most recent failed attempt K

It works in two steps within one prompt:
  1. Diagnosis — what went wrong, which entries were relevant/applied/misdirecting
  2. Edits — full revised <Entry> blocks with <EditReason> tags for each change

Output format from the LLM (parsed by this module):

  <Diagnosis>
  [free text analysis]
  </Diagnosis>

  <EditReason id="ENTRY_ID">reason citing specific transcript evidence</EditReason>
  <Entry id="ENTRY_ID">
  ... complete updated entry ...
  </Entry>

  <EditReason id="NEW_ID">why no existing entry covered this</EditReason>
  <Entry id="NEW_ID">
  ... new entry ...
  </Entry>

  <MisdirectionFlag id="ENTRY_ID"/>   (optional — flags active misdirection)
"""
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from .data_models import SocialScenario
from .fm import FM
from .skills_chronicle import (
    ChronicleEntry,
    SkillsChronicle,
    parse_chronicle,
)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReflectionOutput:
    updated_chronicle: SkillsChronicle
    edit_reasons: dict[str, str]           # {entry_id: reason_text}
    diagnosis: str
    misdirection_entry_ids: list[str]      # entries flagged as having actively misdirected


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """You are a reflective coach analyzing a failed social interaction episode to improve a Skills Chronicle.

A Skills Chronicle is a document of structured entries that guide an AI agent's social behavior. Each entry has a Condition (when to apply it) and Guidance (what to do).

Your task after a FAILED episode:

STEP 1 — DIAGNOSIS:
Write a <Diagnosis> block analyzing:
  - Which chronicle entries were relevant to this scenario
  - Whether the agent applied them
  - What specifically went wrong that existing entries did not anticipate
  - Whether any existing entry ACTIVELY MISDIRECTED the agent (caused worse behavior)

STEP 2 — EDITS:
For each entry you modify, output:
  <EditReason id="ENTRY_ID">Justification with SPECIFIC transcript evidence (quote directly)</EditReason>
  <Entry id="ENTRY_ID">
  <Condition>...</Condition>
  <Guidance>...</Guidance>
  <Type>HEURISTIC | WARNING</Type>
  <Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>
  <Provenance>[existing provenance, add ", attempt K"]</Provenance>
  </Entry>

For NEW entries (when no existing entry covers this case):
  <EditReason id="NEW_ID">Why no existing entry covered this</EditReason>
  <Entry id="NEW_ID">
  ... complete entry block ...
  </Entry>

For entries that ACTIVELY MISDIRECTED the agent, add after their edit block:
  <MisdirectionFlag id="ENTRY_ID"/>

CONDITION FIELD RULES (enforced strictly):
  - NO proper nouns, specific occupations, or scenario-unique details
  - Narrowing a Condition: low-risk, do freely
  - Rewriting at same abstraction: justify in EditReason
  - BROADENING a Condition: high-risk — EditReason MUST cite specific transcript evidence
    that the current scope is too narrow. If the broadened Condition would also apply to
    the parent/anchor scenario's social dynamic, that is a signal it is too broad.

Output ONLY the <Diagnosis>, <EditReason>, <Entry>, and <MisdirectionFlag> blocks. No other text."""


def _format_transcript(transcript: list[dict], attempt_num: int, max_chars: int = 3000) -> str:
    lines = [f"=== Attempt {attempt_num} Transcript ==="]
    for t in transcript:
        line = f"[T{t['turn']}] {t['speaker']}: {t['content']}"
        lines.append(line)
    text = "\n".join(lines)
    if len(text) > max_chars:
        head = max_chars // 3
        tail = max_chars - head
        text = text[:head] + "\n...[truncated]...\n" + text[-tail:]
    return text


def _build_prompt(
    chronicle: SkillsChronicle,
    scenario: SocialScenario,
    transcripts: list[list[dict]],
    prior_edit_reasons: dict[str, str],
    attempt_num: int,
    anchor_task: Optional[SocialScenario],
) -> str:
    parts: list[str] = []

    # Scenario context
    target_goal = (
        scenario.agent_goals[scenario.target_agent_idx]
        if scenario.target_agent_idx < len(scenario.agent_goals)
        else ""
    )
    parts.append(f"SCENARIO: {scenario.scenario}")
    parts.append(f"TARGET AGENT GOAL: {target_goal}")
    parts.append(f"INTERACTION TYPE: {scenario.interaction_type}")

    if anchor_task and anchor_task.social_dynamic:
        parts.append(f"PARENT SCENARIO SOCIAL DYNAMIC (for broadening check): {anchor_task.social_dynamic}")

    # Current chronicle
    if chronicle.entries:
        parts.append("\nCURRENT SKILLS CHRONICLE:")
        parts.append(chronicle.to_markdown())
    else:
        parts.append("\nCURRENT SKILLS CHRONICLE: (empty — no inherited entries)")

    # Prior transcripts (context)
    if len(transcripts) > 1:
        parts.append(f"\nPRIOR FAILED ATTEMPTS ({len(transcripts) - 1} attempts before this one):")
        for i, t in enumerate(transcripts[:-1], 1):
            parts.append(_format_transcript(t, i, max_chars=1500))

    # Prior edit reasons (what was already tried)
    if prior_edit_reasons:
        parts.append("\nPRIOR EDIT REASONS (changes already made in earlier reflections):")
        for eid, reason in prior_edit_reasons.items():
            parts.append(f"  [{eid}]: {reason}")

    # Latest failed transcript — the primary focus
    parts.append(f"\nMOST RECENT FAILED ATTEMPT (attempt {attempt_num} — primary focus):")
    if transcripts:
        parts.append(_format_transcript(transcripts[-1], attempt_num, max_chars=3500))

    parts.append(
        f"\nThis is attempt {attempt_num}. Diagnose the failure and produce targeted chronicle edits."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _extract_tag_content(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _parse_reflection_output(
    llm_output: str,
    existing_chronicle: SkillsChronicle,
    scenario_id: str,
    attempt_num: int,
) -> ReflectionOutput:
    """Parse the LLM output and return an updated SkillsChronicle."""
    updated = deepcopy(existing_chronicle)
    edit_reasons: dict[str, str] = {}
    misdirection_ids: list[str] = []

    diagnosis = _extract_tag_content(llm_output, "Diagnosis") or ""

    # Find all <MisdirectionFlag id="..."/> entries
    for m in re.finditer(r'<MisdirectionFlag\s+id="([^"]+)"\s*/>', llm_output):
        misdirection_ids.append(m.group(1).strip())

    # Find all (EditReason, Entry) pairs — EditReason always precedes its Entry
    # Pattern: <EditReason id="X">...</EditReason> ... <Entry id="X">...</Entry>
    edit_reason_re = re.compile(r'<EditReason\s+id="([^"]+)">(.*?)</EditReason>', re.DOTALL)
    entry_re = re.compile(r'<Entry\s+id="([^"]+)">(.*?)</Entry>', re.DOTALL)

    # Build lookup: entry_id → full entry tag-block text
    entries_in_output: dict[str, str] = {
        m.group(1).strip(): m.group(0) for m in entry_re.finditer(llm_output)
    }

    for m in edit_reason_re.finditer(llm_output):
        eid = m.group(1).strip()
        reason = m.group(2).strip()
        edit_reasons[eid] = reason

        if eid not in entries_in_output:
            continue  # EditReason without a matching Entry — skip

        # Parse the Entry block
        parsed = parse_chronicle(entries_in_output[eid])
        if not parsed:
            continue
        entry = parsed[0]

        existing = existing_chronicle.get_entry(eid)
        if existing is None:
            # New entry: set entry_id to include scenario context
            entry.entry_id = eid if "_" in eid else f"{scenario_id}_{eid}"
            if str(attempt_num) not in entry.provenance:
                entry.provenance = (
                    f"{entry.provenance}, {scenario_id} attempt {attempt_num}".strip(", ")
                )

        updated.upsert_entry(entry)

    return ReflectionOutput(
        updated_chronicle=updated,
        edit_reasons=edit_reasons,
        diagnosis=diagnosis,
        misdirection_entry_ids=misdirection_ids,
    )


# ---------------------------------------------------------------------------
# Synthesis prompt (reflection + adversarial critique → final edits)
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """You are the final arbiter synthesizing chronicle edits after a reflection-adversarial review cycle.

You have:
  1. The scenario and transcript showing what failed
  2. The reflection agent's diagnosis of why it failed and proposed chronicle edits
  3. The adversarial agent's critique of those proposed edits

Your task: produce the DEFINITIVE chronicle edits. You must:
  - Preserve the reflection's valid insights about what failed and why
  - Address legitimate adversarial concerns: evidence gaps (EditReason lacks transcript quotes), abstraction violations (Condition contains proper nouns or scenario-specific details), over-broad Conditions that would also describe the parent scenario
  - Discard adversarial objections that are overly strict or miss the point of the failure
  - You are NOT re-doing the reflection from scratch — you are refining the proposed edits using the critique as a quality filter

Output format is identical to the reflection output:
  <EditReason id="ENTRY_ID">Justification with SPECIFIC transcript evidence</EditReason>
  <Entry id="ENTRY_ID">
  ... complete entry ...
  </Entry>
  <MisdirectionFlag id="ENTRY_ID"/>   (optional)

Output ONLY <EditReason>, <Entry>, and <MisdirectionFlag> blocks. No <Diagnosis>, no commentary."""


def _build_synthesis_prompt(
    reflection_output: "ReflectionOutput",
    adversarial_critique: str,
    chronicle: "SkillsChronicle",
    scenario: "SocialScenario",
    transcripts: list[list[dict]],
    prior_edit_reasons: dict[str, str],
    attempt_num: int,
    anchor_task: Optional["SocialScenario"],
) -> str:
    parts: list[str] = []

    target_goal = (
        scenario.agent_goals[scenario.target_agent_idx]
        if scenario.target_agent_idx < len(scenario.agent_goals)
        else ""
    )
    parts.append(f"SCENARIO: {scenario.scenario}")
    parts.append(f"TARGET AGENT GOAL: {target_goal}")
    if anchor_task and anchor_task.social_dynamic:
        parts.append(f"PARENT SCENARIO SOCIAL DYNAMIC: {anchor_task.social_dynamic}")

    if chronicle.entries:
        parts.append("\nCURRENT SKILLS CHRONICLE (before this reflection pass):")
        parts.append(chronicle.to_markdown())
    else:
        parts.append("\nCURRENT SKILLS CHRONICLE: (empty)")

    parts.append(f"\nFAILED TRANSCRIPT (attempt {attempt_num}):")
    if transcripts:
        parts.append(_format_transcript(transcripts[-1], attempt_num, max_chars=3000))

    # Reflection output
    parts.append("\nREFLECTION AGENT'S DIAGNOSIS:")
    parts.append(reflection_output.diagnosis or "(no diagnosis)")

    parts.append("\nREFLECTION AGENT'S PROPOSED EDITS:")
    if reflection_output.edit_reasons:
        for eid, reason in reflection_output.edit_reasons.items():
            parts.append(f"  [{eid}] EditReason: {reason}")
            entry = reflection_output.updated_chronicle.get_entry(eid)
            if entry:
                parts.append(entry.to_tag_block())
    else:
        parts.append("(no edits proposed)")

    if reflection_output.misdirection_entry_ids:
        parts.append(
            f"\nEntries self-flagged as misdirection by reflection: "
            + ", ".join(reflection_output.misdirection_entry_ids)
        )

    parts.append(f"\nADVERSARIAL CRITIQUE:\n{adversarial_critique}")

    parts.append(
        "\nNow produce the final definitive chronicle edits, refining the reflection's "
        "proposals to satisfy the adversarial concerns where valid."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReflectionModule:
    def __init__(self, fm: FM, max_retries: int = 2):
        self.fm = fm
        self.max_retries = max_retries

    def reflect(
        self,
        chronicle: SkillsChronicle,
        scenario: SocialScenario,
        transcripts: list[list[dict]],
        prior_edit_reasons: dict[str, str],
        attempt_num: int,
        anchor_task: Optional[SocialScenario] = None,
    ) -> ReflectionOutput:
        """Run reflection after attempt_num has failed.

        transcripts: list of all episode transcripts so far (including the latest failure).
        prior_edit_reasons: accumulated {entry_id: reason} from previous reflections.
        """
        prompt = _build_prompt(
            chronicle, scenario, transcripts, prior_edit_reasons, attempt_num, anchor_task
        )
        for attempt in range(self.max_retries):
            try:
                llm_output = self.fm.query(_SYSTEM, prompt, temperature=0.4)
                result = _parse_reflection_output(
                    llm_output, chronicle, scenario.id, attempt_num
                )
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Return unchanged chronicle on persistent failure
                    return ReflectionOutput(
                        updated_chronicle=deepcopy(chronicle),
                        edit_reasons={},
                        diagnosis=f"[reflection failed: {e}]",
                        misdirection_entry_ids=[],
                    )

        return ReflectionOutput(
            updated_chronicle=deepcopy(chronicle),
            edit_reasons={},
            diagnosis="[reflection produced no output]",
            misdirection_entry_ids=[],
        )

    def synthesize_with_critique(
        self,
        reflection_output: ReflectionOutput,
        adversarial_critique: str,
        chronicle: SkillsChronicle,
        scenario: SocialScenario,
        transcripts: list[list[dict]],
        prior_edit_reasons: dict[str, str],
        attempt_num: int,
        anchor_task: Optional[SocialScenario] = None,
    ) -> ReflectionOutput:
        """Synthesize final chronicle edits from reflection output + adversarial critique.

        Rather than re-running reflection from scratch, this agent reads both
        the reflection's diagnosis/edits and the adversarial critique simultaneously
        and produces the definitive edits — preserving valid reflection insights while
        addressing legitimate adversarial concerns.
        """
        prompt = _build_synthesis_prompt(
            reflection_output, adversarial_critique, chronicle,
            scenario, transcripts, prior_edit_reasons, attempt_num, anchor_task
        )
        try:
            llm_output = self.fm.query(_SYNTHESIS_SYSTEM, prompt, temperature=0.3)
            return _parse_reflection_output(
                llm_output, chronicle, scenario.id, attempt_num
            )
        except Exception:
            return reflection_output  # fall back to original reflection on error
