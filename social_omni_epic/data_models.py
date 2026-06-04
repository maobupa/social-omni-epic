from pydantic import BaseModel, Field
from typing import Optional
import uuid


class AgentProfile(BaseModel):
    """Mirrors Sotopia's AgentProfile using its exact field names."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    first_name: str
    last_name: str = ""
    age: int = 0
    gender_identity: str = ""
    occupation: str = ""
    big_five: str = ""
    moral_values: str = ""
    schwartz_portrait_value: str = ""
    decision_making_style: str = ""
    secret: str = ""
    mbti: str = ""
    public_info: str = ""


class StructuredGoal(BaseModel):
    """A learner/partner goal decomposed into the three social-difficulty components.

    The shortcut is a *generative/design* field (it drives generation, the naive failure
    mode, and the difficulty editor) — it is NOT a rubric check. It is evaluated only
    indirectly via the constraint check (taking the shortcut → constraint check fails).
    """
    outcome: str = ""       # the instrumental ask — a genuine state-change, not an extractable line
    constraint: str = ""    # the "without Y" relational/face cost the blunt path would incur
    shortcut: str = ""      # the tempting move that wins the outcome but breaks the constraint


class RubricCheck(BaseModel):
    """One machine-checkable success condition, authored at generation and frozen."""
    kind: str          # "outcome" | "constraint"
    question: str      # judged yes/no at eval time
    perspective: str   # "neutral" (transcript-observable) | "partner" (partner-internal state)


class SuccessRubric(BaseModel):
    checks: list[RubricCheck] = []


class SocialScenario(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = -1
    scenario: str
    agent_profiles: list[AgentProfile]
    agent_goals: list[str]                                   # RENDERED from structured_goals (Sotopia-facing)
    structured_goals: list[Optional[StructuredGoal]] = [None, None]  # symmetric — one per agent
    goal_type: Optional[str] = None                          # open descriptive label (analysis only)
    success_rubric: Optional[SuccessRubric] = None           # learner's checks; success = AND of all
    relationship: str = ""
    relationship_background: str = ""
    tag: str = ""
    interaction_type: str = ""
    difficulty_tags: list[str] = []
    source: str = "generated"
    source_env_id: str = ""          # env_pk for SOTOPIA seeds; "" for generated
    source_scenario_id: str = ""     # dedup key: env_pk for seeds, base UUID for generated
    embedding: Optional[list[float]] = None
    parent_example_ids: list[str] = []
    moi_reasoning: str = ""
    goal_score: Optional[float] = None
    progress_score: Optional[float] = None

    # Phase 2 — SCENARIO_TITLE (§4.9)
    scenario_title: Optional[str] = None       # "social dynamic | target perspective"
    social_dynamic: Optional[str] = None       # left half of pipe
    target_perspective: Optional[str] = None   # right half of pipe

    # Phase 2 — target agent designation (§4.4)
    target_agent_idx: int = 0
    target_agent_goal_abstract: Optional[str] = None

    # Phase 2 — skills chronicle (§4.8)
    skills_final_md: Optional[str] = None

    # Phase 2 — UCB1 bookkeeping (§4.1)
    n_i: int = 0           # times chosen as anchor task
    last_chosen: int = -1  # iteration index when last chosen
    n_children: int = 0    # number of descendant scenarios generated from this task

    def to_text_for_embedding(self) -> str:
        parts = [
            f"Scenario: {self.scenario}",
            f"Interaction type: {self.interaction_type}",
            f"Relationship: {self.relationship}",
        ]
        # Perspective-aware: label which goal belongs to the learner vs. partner.
        # The two perspectives of the same scenario must produce different texts.
        learner_idx = self.target_agent_idx
        partner_idx = 1 - learner_idx
        def _name(idx: int) -> str:
            return (self.agent_profiles[idx].first_name
                    if idx < len(self.agent_profiles) else f"Agent{idx}")
        if len(self.agent_goals) > learner_idx:
            parts.append(f"Learner goal ({_name(learner_idx)}): {self.agent_goals[learner_idx]}")
        if len(self.agent_goals) > partner_idx:
            parts.append(f"Partner goal ({_name(partner_idx)}): {self.agent_goals[partner_idx]}")
        for agent in self.agent_profiles:
            parts.append(
                f"Character {agent.first_name}: {agent.occupation}; {agent.big_five}"
            )
        return "\n".join(parts)


class ArchiveState(BaseModel):
    successful: list[SocialScenario] = []
    failed_generation: list[dict] = []
    failed_interestingness: list[SocialScenario] = []
    failed_tasks: list[SocialScenario] = []
