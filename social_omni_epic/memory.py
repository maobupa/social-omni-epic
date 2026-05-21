"""In-context learning memory: RetrievalBank + SkillProfile.

Pure Python + JSON + our FM wrapper. Independent of Sotopia.
"""
import json
import math
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .data_models import SocialScenario


class BankEntry(BaseModel):
    scenario_id: str
    scenario_text: str
    interaction_type: str = ""
    embedding: list[float] = Field(default_factory=list)
    goal_score: float = 0.0
    overall_score: float = 0.0
    learner_scores: dict = Field(default_factory=dict)
    summary: str = ""  # FM-generated 3-5 sentence "what worked / what failed"
    num_turns: int = 0


SUMMARY_SYSTEM = (
    "You are a social-skills coach. You will read one social interaction and "
    "summarize, in 3-5 sentences, what the LEARNER agent did well and what "
    "they could have done differently. Focus on transferable lessons for "
    "future similar interactions. Be specific about tactics (e.g., 'asked "
    "open-ended question to surface partner's constraint'). Do NOT include "
    "the scenario details themselves; only the lessons."
)

SUMMARY_USER_TEMPLATE = """Scenario: {scenario}

Learner's private goal: {learner_goal}

Scores (0-10): goal={goal:.1f}, relationship={rel:.1f}, knowledge={kno:.1f}, overall={overall:.1f}

Transcript:
{transcript}

Write the 3-5 sentence lesson."""


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _truncate_transcript(transcript: list[dict], max_chars: int = 4000) -> str:
    parts = []
    for t in transcript:
        s = f"[turn {t['turn']}] {t['sender']} -> {t['receiver']}: {t['content']}"
        parts.append(s)
    text = "\n".join(parts)
    if len(text) > max_chars:
        # Keep first 1/3 and last 2/3 — the interesting bits are usually near the end.
        head_n = max_chars // 3
        tail_n = max_chars - head_n
        text = text[:head_n] + "\n...[truncated]...\n" + text[-tail_n:]
    return text


class RetrievalBank:
    """Append-only bank of episode summaries persisted to a JSON file."""

    def __init__(self, fm, bank_path: str):
        self.fm = fm
        self.bank_path = Path(bank_path)
        self.entries: list[BankEntry] = []
        if self.bank_path.exists():
            self.load()
        else:
            self.bank_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def size(self) -> int:
        return len(self.entries)

    def load(self) -> None:
        with open(self.bank_path) as f:
            data = json.load(f)
        self.entries = [BankEntry(**d) for d in data]

    def save(self) -> None:
        with open(self.bank_path, "w") as f:
            json.dump([e.model_dump() for e in self.entries], f, indent=2)

    def _summarize(self, scenario: SocialScenario, result) -> str:
        learner_goal = scenario.agent_goals[0] if scenario.agent_goals else ""
        user = SUMMARY_USER_TEMPLATE.format(
            scenario=scenario.scenario,
            learner_goal=learner_goal,
            goal=result.learner_scores.get("goal", 0.0),
            rel=result.learner_scores.get("relationship", 0.0),
            kno=result.learner_scores.get("knowledge", 0.0),
            overall=result.learner_scores.get("overall_score", 0.0),
            transcript=_truncate_transcript(result.transcript),
        )
        try:
            return self.fm.query(SUMMARY_SYSTEM, user, temperature=0.4).strip()
        except Exception as e:
            return f"[summary unavailable: {e}]"

    def add_episode(self, result, scenario: SocialScenario) -> BankEntry:
        summary = self._summarize(scenario, result)
        entry = BankEntry(
            scenario_id=scenario.id,
            scenario_text=scenario.scenario,
            interaction_type=scenario.interaction_type,
            embedding=scenario.embedding or [],
            goal_score=float(result.learner_scores.get("goal", 0.0)),
            overall_score=float(result.learner_scores.get("overall_score", 0.0)),
            learner_scores=dict(result.learner_scores),
            summary=summary,
            num_turns=result.num_turns,
        )
        self.entries.append(entry)
        self.save()
        return entry

    def retrieve(self, query_embedding: list[float], top_k: int = 3) -> list[BankEntry]:
        if not query_embedding or not self.entries:
            return []
        scored = [
            (_cosine(query_embedding, e.embedding), e)
            for e in self.entries
            if e.embedding
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def format_for_prompt(self, entries: list[BankEntry]) -> str:
        if not entries:
            return ""
        blocks = []
        for i, e in enumerate(entries, 1):
            blocks.append(
                f"[Past case {i}] (goal={e.goal_score:.1f}/10) "
                f"Scenario: {e.scenario_text[:160]}\n"
                f"Lesson: {e.summary}"
            )
        return "Retrieved lessons:\n" + "\n\n".join(blocks)


SKILL_PROFILE_SYSTEM = (
    "You are maintaining a living document of social skills the learner has "
    "demonstrated across many simulated interactions. Read the retrieval bank "
    "entries and produce an updated skill profile in markdown. The profile "
    "should: (1) list concrete tactics that worked, grouped by category "
    "(e.g., 'negotiation', 'emotional support'); (2) list recurring failure "
    "modes; (3) be at most ~400 words. Write in second person ('You tend to...'). "
    "Do NOT include scenario-specific names or contexts."
)

SKILL_PROFILE_USER = """Here are the most recent {n} episode summaries (newest first):

{entries}

Write the updated skill profile in markdown."""


class SkillProfile:
    """A single markdown document rewritten every `update_every` episodes."""

    def __init__(self, fm, profile_path: str, update_every: int = 10,
                 sample_size: int = 30):
        self.fm = fm
        self.profile_path = Path(profile_path)
        self.update_every = update_every
        self.sample_size = sample_size
        self._last_update_at = 0
        self.text = ""
        if self.profile_path.exists():
            self.text = self.profile_path.read_text()
        else:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)

    def format_for_prompt(self) -> str:
        if not self.text.strip():
            return ""
        return f"Your current social-skill profile:\n{self.text.strip()}"

    def maybe_update(self, bank: RetrievalBank) -> bool:
        if bank.size == 0:
            return False
        if bank.size - self._last_update_at < self.update_every:
            return False
        self._last_update_at = bank.size

        recent = list(reversed(bank.entries))[: self.sample_size]
        entries_str = "\n\n".join(
            f"- (goal={e.goal_score:.1f}) {e.summary}" for e in recent
        )
        user = SKILL_PROFILE_USER.format(n=len(recent), entries=entries_str)
        try:
            new_text = self.fm.query(SKILL_PROFILE_SYSTEM, user, temperature=0.4).strip()
            self.text = new_text
            self.profile_path.write_text(self.text)
            return True
        except Exception as e:
            print(f"  [SkillProfile update failed: {e}]")
            return False
