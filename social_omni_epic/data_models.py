from pydantic import BaseModel, Field
from typing import Optional
import uuid


# Theory-grounded mechanism library for PartnerKey.key_mechanism.
# These tags are the only valid values; extend only with explicit instruction.
MECHANISM_LIBRARY: dict[str, str] = {
    "reactance": (
        "Pressure, ultimatums, or removal of choice harden resistance; "
        "restoring autonomy and choice enables movement. (Brehm)"
    ),
    "face_needs": (
        "Movement requires a face-saving account, acknowledgment of competence/judgment, "
        "or an exit that preserves public identity. (Brown & Levinson)"
    ),
    "validation_before_change": (
        "The partner cannot consider change until they feel their position/emotion has been "
        "genuinely understood; premature problem-solving stalls or hardens. "
        "(Motivational interviewing)"
    ),
    "procedural_voice": (
        "The partner accepts substantively worse outcomes if given genuine voice in the process; "
        "imposed outcomes are rejected even when favorable. (Procedural justice)"
    ),
    "reciprocity_disclosure": (
        "Movement is unlocked by the learner's costly first move: a genuine concession, "
        "self-disclosure, or acceptance of risk. (Cialdini; social penetration theory)"
    ),
}


class PartnerKey(BaseModel):
    """Hidden ground-truth specification of what moves the partner.

    This is NEVER shown to the learner or included in the shared scenario description.
    It is injected into the partner's private turn context and used by the key-aware
    terminal judge. Absent on seed scenarios (they run native, without a key).
    """
    key_mechanism: str              # one tag from MECHANISM_LIBRARY — REQUIRED
    movement_conditions: list[str]  # 1-3 concrete conditions under which partner genuinely shifts
    hardening_triggers: list[str]   # 1-3 learner moves that lock the partner (reactance instantiations)
    surface_misdirection: str       # the partner's STATED objection (may appear in public scenario text)
    cost_coupling: str              # what satisfying movement_conditions costs the LEARNER's own goal


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

    # Phase 2 — PartnerKey (generated scenarios only; None for seeds)
    partner_key: Optional[PartnerKey] = None

    # Cooperative-alignment fields (optional; at least one should be present for generated scenarios).
    # competing_interest: family (a) — the learner's personal cost that full accommodation forfeits.
    # partner_default_position: family (a) and (b) — what the partner naturally offers without skilled
    #   engagement; must fall short of the learner's outcome for the scenario to have real difficulty.
    competing_interest: Optional[str] = None
    partner_default_position: Optional[str] = None

    # Per-attempt GOAL scores from the K-loop in order (attempt 1 first).
    # Populated after curriculum run; empty for seed scenarios and generation failures.
    goal_trajectory: list[float] = []

    # Phase 2 — anchor selection bookkeeping (§4.1)
    n_i: float = 0.0        # effective selection count (float to support weighted outcomes)
    last_chosen: int = -1   # iteration index when last chosen
    n_children: int = 0     # total descendant scenarios generated from this anchor
    n_solved: int = 0       # children that reached solved_after_biting

    # Hierarchical Thompson Sampling prior.
    # Seeds: flat Beta(1,1) — no evidence.
    # Generated children: parent's posterior at time of child creation.
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    # LP pseudo-vote accumulators (§5).
    # Posterior = Beta(prior_alpha + alpha_votes, prior_beta + beta_votes).
    alpha_votes: float = 0.0   # sum of improved_votes from children
    beta_votes: float = 0.0    # sum of (total_votes - improved_votes) from children

    # Phase 2 §1.5 — archive record fields (populated after curriculum run)
    lp_value: Optional[float] = None     # improved_votes/total_votes ∈ [0,1]; None pre-Phase-0
    lp_votes: int = 0                    # total vote count behind lp_value
    terminal_success: bool = False       # GOAL≥7 ∧ REL≥0 on final attempt (+ key check if keyed)
    n_attempts: int = 0                  # episode attempts run
    niche_id: Optional[int] = None      # k-means niche assignment (§6.4)
    mutation_operator: Optional[str] = None  # "escalate" | "relax" | "lateral" | None (seeds)
    mutated_slots: list[str] = []        # structural slots the generator mutated
    mutation_rationale: Optional[str] = None  # one-sentence rationale from the generator
    classification: Optional[str] = None  # "too_easy" | "frontier" | "beyond_frontier"
    too_easy_diagnosis: Optional[dict] = None   # {slack_knob, rationale} from analyze_too_easy
    final_check_flag: Optional[list[str]] = None  # adversarial check_final issues when not approved

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
    tasks: list[SocialScenario] = []          # all completed scenarios (§5 rename from successful)
    failed_generation: list[dict] = []
    failed_interestingness: list[SocialScenario] = []
    failed_tasks: list[SocialScenario] = []
    niche_counts: dict[int, int] = {}         # generations per niche (§6.4)
