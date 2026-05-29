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


class SocialScenario(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = -1
    scenario: str
    agent_profiles: list[AgentProfile]
    agent_goals: list[str]
    relationship: str = ""
    tag: str = ""
    interaction_type: str = ""
    difficulty_tags: list[str] = []
    source: str = "generated"
    embedding: Optional[list[float]] = None
    parent_example_ids: list[str] = []
    moi_reasoning: str = ""
    goal_score: Optional[float] = None
    progress_score: Optional[float] = None

    # Phase 2 — SCENARIO_TITLE (§4.9)
    scenario_title: Optional[str] = None       # "social dynamic | target perspective"
    social_dynamic: Optional[str] = None       # left half of pipe
    target_perspective: Optional[str] = None   # right half of pipe

    # Phase 2 — structural metadata (§4.1)
    goal_structure: Optional[str] = None       # COMPETITIVE | COOPERATIVE | MIXED
    info_position: Optional[str] = None        # INFORMED | UNINFORMED | SYMMETRIC

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
        for i, goal in enumerate(self.agent_goals):
            agent_name = (
                self.agent_profiles[i].first_name
                if i < len(self.agent_profiles)
                else f"Agent{i}"
            )
            parts.append(f"{agent_name}'s goal: {goal}")
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
