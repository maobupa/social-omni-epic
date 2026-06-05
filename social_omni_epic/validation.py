import uuid as _uuid
from .data_models import (
    SocialScenario,
    AgentProfile,
    StructuredGoal,
    RubricCheck,
    SuccessRubric,
)


REQUIRED_TOP = [
    "scenario",
    "agent_profiles",
    "agent_structured_goals",
    "relationship",
    "interaction_type",
    "success_rubric",
]
REQUIRED_AGENT = ["first_name", "occupation", "big_five"]
REQUIRED_GOAL = ["outcome", "constraint", "shortcut"]
VALID_KINDS = {"outcome", "constraint"}
VALID_PERSPECTIVES = {"neutral", "partner"}


def render_agent_goal(sg: StructuredGoal) -> str:
    """Render a StructuredGoal into the Sotopia-facing goal string the agent actually sees.

    Single source of truth for `agent_goals`. The shortcut/leverage is surfaced in
    <extra_info> as *available and tempting* — deliberately NOT labeled as a trap (the agent
    must discover the tension itself). The generator authors `constraint` as the thing to
    avoid (so it reads after "without") and `shortcut` as the leverage phrased from the
    agent's POV.
    """
    outcome = (sg.outcome or "").strip().rstrip(".")
    goal = f"Your goal is to {outcome}" if outcome else "Your goal is to act in your interest"
    constraint = (sg.constraint or "").strip().rstrip(".")
    if constraint:
        goal += f", without {constraint}"
    goal += "."
    shortcut = (sg.shortcut or "").strip()
    if shortcut:
        goal += f" <extra_info>{shortcut}</extra_info>"
    return goal


def validate_scenario(d: dict) -> tuple[bool, str]:
    for f in REQUIRED_TOP:
        if f not in d:
            return False, f"Missing field: {f}"

    profiles = d["agent_profiles"]
    if not isinstance(profiles, list) or len(profiles) != 2:
        return False, "agent_profiles must be a list of exactly 2 profiles"
    for i, p in enumerate(profiles):
        if not isinstance(p, dict):
            return False, f"Agent profile {i} not a dict"
        for pf in REQUIRED_AGENT:
            if pf not in p or not p[pf]:
                return False, f"Agent profile {i} missing/empty field: {pf}"

    goals = d["agent_structured_goals"]
    if not isinstance(goals, list) or len(goals) != 2:
        return False, "agent_structured_goals must be a list of exactly 2 structured goals"
    for i, g in enumerate(goals):
        if not isinstance(g, dict):
            return False, f"Structured goal {i} not a dict"
        for gf in REQUIRED_GOAL:
            if gf not in g or not str(g[gf]).strip():
                return False, f"Structured goal {i} missing/empty field: {gf}"

    rubric = d["success_rubric"]
    checks = rubric.get("checks") if isinstance(rubric, dict) else None
    if not isinstance(checks, list) or not checks:
        return False, "success_rubric.checks must be a non-empty list"
    kinds = set()
    for i, c in enumerate(checks):
        if not isinstance(c, dict):
            return False, f"Rubric check {i} not a dict"
        if c.get("kind") not in VALID_KINDS:
            return False, f"Rubric check {i} kind must be one of {VALID_KINDS}"
        if not str(c.get("question", "")).strip():
            return False, f"Rubric check {i} missing question"
        kinds.add(c["kind"])
    if "outcome" not in kinds or "constraint" not in kinds:
        return False, "success_rubric must contain at least one 'outcome' and one 'constraint' check"
    if len(checks) > 3:
        return False, f"success_rubric has {len(checks)} checks — maximum is 3 (1 outcome + 1–2 constraint)"

    if len(d["scenario"]) < 50:
        return False, "scenario too short (< 50 chars)"

    return True, ""


def _coerce_perspective(kind: str, perspective) -> str:
    """Trust the generator's perspective; fall back to the natural default per kind."""
    p = str(perspective or "").strip().lower()
    if p in VALID_PERSPECTIVES:
        return p
    # outcome → transcript-observable by default; constraint → partner-internal by default
    return "neutral" if kind == "outcome" else "partner"


def _clean_str(s) -> str:
    """Strip null bytes and non-breaking spaces that LLMs occasionally generate as padding."""
    if not isinstance(s, str):
        return s
    return s.replace("\x00", "").replace(" ", " ").strip()


def _clean_dict(d: dict) -> dict:
    """Recursively clean string values in a dict."""
    return {k: (_clean_dict(v) if isinstance(v, dict) else
                [_clean_dict(i) if isinstance(i, dict) else _clean_str(i) if isinstance(i, str) else i for i in v] if isinstance(v, list) else
                _clean_str(v) if isinstance(v, str) else v)
            for k, v in d.items()}


def dict_to_scenario(d: dict) -> SocialScenario:
    d = _clean_dict(d)
    profiles = [AgentProfile(**p) for p in d["agent_profiles"]]
    structured = [StructuredGoal(**g) for g in d["agent_structured_goals"]]
    rendered_goals = [render_agent_goal(sg) for sg in structured]

    rubric = SuccessRubric(
        checks=[
            RubricCheck(
                kind=c["kind"],
                question=str(c["question"]).strip(),
                perspective=_coerce_perspective(c["kind"], c.get("perspective")),
            )
            for c in d["success_rubric"]["checks"]
        ]
    )

    base_id = str(_uuid.uuid4())
    return SocialScenario(
        id=f"{base_id}_pX",
        source_scenario_id=base_id,
        scenario=d["scenario"],
        agent_profiles=profiles,
        agent_goals=rendered_goals,
        structured_goals=structured,
        goal_type=d.get("goal_type"),
        success_rubric=rubric,
        relationship=d.get("relationship", ""),
        relationship_background=d.get("relationship_background", ""),
        tag=d.get("tag", d.get("interaction_type", "")),
        interaction_type=d.get("interaction_type", ""),
        difficulty_tags=d.get("difficulty_tags", []),
        source="generated",
    )
