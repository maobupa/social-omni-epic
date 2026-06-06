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

    # Phase 2 — anchor selection bookkeeping (§4.1)
    n_i: int = 0            # times chosen as anchor
    last_chosen: int = -1   # iteration index when last chosen
    n_children: int = 0     # total descendant scenarios generated from this anchor
    n_solved: int = 0       # children that reached solved_after_biting

    # Hierarchical Thompson Sampling prior.
    # Seeds: flat Beta(1,1) — no evidence.
    # Generated children: parent's posterior at time of child creation.
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def to_text_for_embedding(self) -> str:
        parts = []
        # scenario_title is the primary structural retrieval key (abstract, no proper nouns).
        # Seeds that have been pre-titled populate this; others fall back to scenario text.
        if self.scenario_title:
            parts.append(f"Scenario type: {self.scenario_title}")
        parts += [
            f"Scenario: {self.scenario}",
            f"Interaction type: {self.interaction_type}",
            f"Relationship: {self.relationship}",
        ]
        # Perspective-aware goals (structural signal; drop character surface details).
        learner_idx = self.target_agent_idx
        partner_idx = 1 - learner_idx
        if len(self.agent_goals) > learner_idx:
            parts.append(f"Learner goal: {self.agent_goals[learner_idx]}")
        if len(self.agent_goals) > partner_idx:
            parts.append(f"Partner goal: {self.agent_goals[partner_idx]}")
        return "\n".join(parts)


class ArchiveState(BaseModel):
    successful: list[SocialScenario] = []
    failed_generation: list[dict] = []
    failed_interestingness: list[SocialScenario] = []
    failed_tasks: list[SocialScenario] = []
