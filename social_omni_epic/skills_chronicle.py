"""Skills Chronicle: tag-block structured knowledge entries (§4.8).

Each entry serializes to:

<Entry id="SCENARIO_ID_N">

<Condition>
Abstract social dynamic pattern — no proper nouns, specific occupations, or
scenario-unique details. Phrased as a recognizable structural pattern.
</Condition>

<Guidance>
1. Primary guidance: [how to behave when entering this type of interaction — can be what TO do or what NOT to do; must be specific enough that an agent reading it before a conversation would behave observably differently]
2. Warning (optional): only include if there is a specific tempting behavior that contrasts with the primary guidance and backfires in a non-obvious way — omit if the primary guidance already covers what to avoid, or if no clear contrasting trap exists
3. Exception: when [a specific circumstance within the above Condition makes the primary guidance inappropriate], do [alternative] instead
(add further numbered Exception clauses as needed)
Note: Later clauses take precedence over earlier ones when their conditions apply.
</Guidance>

<Type>HEURISTIC | WARNING</Type>

<Dimension>GOAL | FIN | REL | BEL | KNO | SOC | SEC</Dimension>

<Provenance>scenario_ids and iteration numbers</Provenance>

</Entry>

Retrieval at evaluation time uses the Condition field embedding as the primary
matching signal. Entries are injected in insertion order; context-window
truncation is a hard count cutoff via max_entries.
"""
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


PRECEDENCE_NOTE = (
    "Note: Later clauses take precedence over earlier ones when their conditions apply."
)

VALID_TYPES = {"HEURISTIC", "WARNING"}
VALID_DIMS = {"GOAL", "FIN", "REL", "BEL", "KNO", "SOC", "SEC"}


@dataclass
class ChronicleEntry:
    entry_id: str
    condition: str
    guidance: str
    entry_type: str   # HEURISTIC | WARNING
    dimension: str    # GOAL | FIN | REL | BEL | KNO | SOC | SEC
    provenance: str = ""
    condition_embedding: Optional[list[float]] = None  # cached at upsert time for relevance ranking

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
            f"<Provenance>{self.provenance}</Provenance>\n\n"
            f"</Entry>"
        )


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

    def format_for_prompt(
        self,
        max_entries: Optional[int] = None,
        query_embedding: Optional[list[float]] = None,
        fm=None,
    ) -> str:
        """Format for injection into agent context window.

        When query_embedding is provided (and optionally fm for lazy embedding
        computation), entries are ranked by cosine similarity of their Condition
        embedding to the query. The top max_entries are returned in their original
        relative order so the context reads chronologically. Falls back to
        insertion-order truncation when no query_embedding is given.
        """
        if not self.entries:
            return ""

        entries = list(self.entries)

        if query_embedding is not None:
            # Lazily compute missing condition embeddings if fm is available
            if fm is not None:
                missing = [e for e in entries if e.condition_embedding is None and e.condition]
                if missing:
                    try:
                        embs = fm.get_embeddings([e.condition for e in missing])
                        for e, emb in zip(missing, embs):
                            e.condition_embedding = emb
                    except Exception:
                        pass

            scored = [(
                _cosine(query_embedding, e.condition_embedding) if e.condition_embedding else 0.0,
                i,  # original position for stable sort
                e,
            ) for i, e in enumerate(entries)]
            scored.sort(key=lambda x: (-x[0], x[1]))

            if max_entries is not None:
                top_ids = {e.entry_id for _, _, e in scored[:max_entries]}
                entries = [e for e in self.entries if e.entry_id in top_ids]
            else:
                entries = [e for _, _, e in scored]
        elif max_entries is not None:
            entries = entries[:max_entries]

        blocks = [e.to_tag_block() for e in entries]
        return (
            "=== Skills Chronicle (prior experience — visible only to you) ===\n"
            "These lessons come from past interactions that may differ in setting or characters. "
            "Apply the underlying principle and adapt it to your current situation — do not treat them as literal scripts.\n\n"
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

    def upsert_entry(self, entry: ChronicleEntry, fm=None) -> None:
        """Replace existing entry with same id, or append if new.

        If fm is provided and the entry lacks a condition_embedding, compute
        and cache it now so format_for_prompt can rank by relevance later.
        """
        if fm is not None and entry.condition_embedding is None and entry.condition:
            try:
                entry.condition_embedding = fm.get_embeddings([entry.condition])[0]
            except Exception:
                pass
        for i, e in enumerate(self.entries):
            if e.entry_id == entry.entry_id:
                self.entries[i] = entry
                return
        self.entries.append(entry)

    def remove_entry(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.entry_id != entry_id]
        return len(self.entries) < before


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
        provenance = _extract_tag(block, "Provenance")

        if entry_type not in VALID_TYPES:
            entry_type = "HEURISTIC"
        if dimension not in VALID_DIMS:
            dimension = "GOAL"

        entries.append(
            ChronicleEntry(
                entry_id=entry_id,
                condition=condition,
                guidance=guidance,
                entry_type=entry_type,
                dimension=dimension,
                provenance=provenance,
            )
        )

    return entries
