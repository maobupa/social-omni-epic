"""Skills Chronicle: tag-block structured knowledge entries (§4.8).

Each entry serializes to:

<Entry id="SCENARIO_ID_N">

<Condition>
Abstract social dynamic pattern — no proper nouns, specific occupations, or
scenario-unique details. Phrased as a recognizable structural pattern.
</Condition>

<Guidance>
1. Default behavioral instruction — the general case
2. Exception: when [condition], do [override] instead
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>

<Type>HEURISTIC | WARNING</Type>

<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>

<Confidence>HIGH | MEDIUM | LOW</Confidence>

<Support>integer</Support>

<Provenance>scenario_ids and iteration numbers</Provenance>

</Entry>

Confidence assignment (§4.6.2) — deterministic, never LLM-assigned:
  GOAL, FIN        → HIGH
  REL, BEL         → MEDIUM
  KNO, SOC, SEC    → LOW

Promotion:
  LOW  → MEDIUM when Support ≥ 5  (no misdirection flag)
  MEDIUM → HIGH when Support ≥ 10 (no misdirection flag)

Demotion:
  One level when a Reflection EditReason flags active misdirection.
"""
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

PRECEDENCE_NOTE = (
    "Note: Later clauses take precedence over earlier ones when their conditions apply."
)

DIMENSION_TO_CONFIDENCE: dict[str, str] = {
    "GOAL": "HIGH",
    "FIN": "HIGH",
    "REL": "MEDIUM",
    "BEL": "MEDIUM",
    "KNO": "LOW",
    "SOC": "LOW",
    "SEC": "LOW",
}

VALID_TYPES = {"HEURISTIC", "WARNING"}
VALID_DIMS = set(DIMENSION_TO_CONFIDENCE)
VALID_CONFS = {"HIGH", "MEDIUM", "LOW"}

_CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_RANK_CONF = {v: k for k, v in _CONF_RANK.items()}

_PROMOTE_AT = {"LOW": 5, "MEDIUM": 10}  # support threshold → next level


@dataclass
class ChronicleEntry:
    entry_id: str
    condition: str
    guidance: str
    entry_type: str        # HEURISTIC | WARNING
    dimension: str         # GOAL | FIN | REL | BEL | KNO | SOC | SEC
    confidence: str        # HIGH | MEDIUM | LOW
    support: int = 0
    provenance: str = ""
    has_misdirection_flag: bool = False

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_tag_block(self) -> str:
        guidance_body = self.guidance.strip()
        if PRECEDENCE_NOTE not in guidance_body:
            guidance_body = guidance_body + "\n" + PRECEDENCE_NOTE
        return (
            f'<Entry id="{self.entry_id}">\n\n'
            f"<Condition>\n{self.condition.strip()}\n</Condition>\n\n"
            f"<Guidance>\n{guidance_body}\n</Guidance>\n\n"
            f"<Type>{self.entry_type}</Type>\n\n"
            f"<Dimension>{self.dimension}</Dimension>\n\n"
            f"<Confidence>{self.confidence}</Confidence>\n\n"
            f"<Support>{self.support}</Support>\n\n"
            f"<Provenance>{self.provenance}</Provenance>\n\n"
            f"</Entry>"
        )

    # ------------------------------------------------------------------
    # Confidence mechanics
    # ------------------------------------------------------------------

    @classmethod
    def initial_confidence(cls, dimension: str) -> str:
        return DIMENSION_TO_CONFIDENCE.get(dimension, "MEDIUM")

    def try_promote(self) -> bool:
        """Promote by one level if support threshold met and no misdirection flag."""
        if self.has_misdirection_flag:
            return False
        threshold = _PROMOTE_AT.get(self.confidence)
        if threshold is None:
            return False
        if self.support >= threshold:
            new_rank = _CONF_RANK[self.confidence] + 1
            new_conf = _RANK_CONF.get(new_rank)
            if new_conf:
                self.confidence = new_conf
                return True
        return False

    def demote(self) -> bool:
        """Demote by one level and set misdirection flag."""
        rank = _CONF_RANK.get(self.confidence, 0)
        if rank > 0:
            self.confidence = _RANK_CONF[rank - 1]
            self.has_misdirection_flag = True
            return True
        return False

    def increment_support(self) -> None:
        """Increment support by 1 (once per episode, not per attempt) then try promote."""
        self.support += 1
        self.try_promote()


# ---------------------------------------------------------------------------
# Chronicle container
# ---------------------------------------------------------------------------

class SkillsChronicle:
    """Ordered collection of ChronicleEntry objects for one scenario lineage."""

    def __init__(self, entries: Optional[list[ChronicleEntry]] = None):
        self.entries: list[ChronicleEntry] = entries or []

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Serialization / deserialization
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        if not self.entries:
            return ""
        return "\n\n---\n\n".join(e.to_tag_block() for e in self.entries)

    @classmethod
    def from_markdown(cls, text: str) -> "SkillsChronicle":
        return cls(parse_chronicle(text))

    # ------------------------------------------------------------------
    # Retrieval / prompt formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, max_entries: Optional[int] = None) -> str:
        """Format for injection into agent context window.

        Entries are sorted by (confidence rank DESC, support DESC) so the
        highest-quality guidance appears first. Truncates to max_entries
        if specified.
        """
        if not self.entries:
            return ""
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (_CONF_RANK.get(e.confidence, 0), e.support),
            reverse=True,
        )
        if max_entries is not None:
            sorted_entries = sorted_entries[:max_entries]
        blocks = [e.to_tag_block() for e in sorted_entries]
        return (
            "=== Skills Chronicle (prior experience — visible only to you) ===\n\n"
            + "\n\n---\n\n".join(blocks)
            + "\n\n=== End of Skills Chronicle ==="
        )

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: str) -> Optional[ChronicleEntry]:
        for e in self.entries:
            if e.entry_id == entry_id:
                return e
        return None

    def upsert_entry(self, entry: ChronicleEntry) -> None:
        """Replace existing entry with same id, or append if new."""
        for i, e in enumerate(self.entries):
            if e.entry_id == entry.entry_id:
                self.entries[i] = entry
                return
        self.entries.append(entry)

    def remove_entry(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.entry_id != entry_id]
        return len(self.entries) < before

    # ------------------------------------------------------------------
    # Episode-level updates
    # ------------------------------------------------------------------

    def increment_all_support(self) -> None:
        """Called once per episode after it completes (not per attempt)."""
        for e in self.entries:
            e.increment_support()

    def apply_misdirection_demotion(self, entry_id: str) -> bool:
        e = self.get_entry(entry_id)
        if e:
            return e.demote()
        return False


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _extract_tag(block: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_chronicle(text: str) -> list[ChronicleEntry]:
    """Parse a skills_final.md string into a list of ChronicleEntry objects."""
    entries: list[ChronicleEntry] = []
    entry_re = re.compile(r'<Entry\s+id="([^"]+)">(.*?)</Entry>', re.DOTALL)

    for m in entry_re.finditer(text):
        entry_id = m.group(1).strip()
        block = m.group(2)

        condition = _extract_tag(block, "Condition")

        # Strip the programmatically-added precedence note from guidance on parse
        guidance_raw = _extract_tag(block, "Guidance")
        guidance = guidance_raw.replace(PRECEDENCE_NOTE, "").strip()

        entry_type = _extract_tag(block, "Type").upper()
        dimension = _extract_tag(block, "Dimension").upper()
        confidence = _extract_tag(block, "Confidence").upper()
        support_str = _extract_tag(block, "Support")
        provenance = _extract_tag(block, "Provenance")

        # Validate / normalise
        if entry_type not in VALID_TYPES:
            entry_type = "HEURISTIC"
        if dimension not in VALID_DIMS:
            dimension = "GOAL"
        if confidence not in VALID_CONFS:
            confidence = DIMENSION_TO_CONFIDENCE.get(dimension, "MEDIUM")
        try:
            support = int(support_str)
        except (ValueError, TypeError):
            support = 0

        entries.append(
            ChronicleEntry(
                entry_id=entry_id,
                condition=condition,
                guidance=guidance,
                entry_type=entry_type,
                dimension=dimension,
                confidence=confidence,
                support=support,
                provenance=provenance,
            )
        )

    return entries
