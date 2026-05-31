import uuid as _uuid
from .data_models import SocialScenario, AgentProfile


REQUIRED_TOP = ["scenario", "agent_profiles", "agent_goals", "relationship", "interaction_type"]
REQUIRED_AGENT = ["first_name", "occupation", "big_five"]


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

    goals = d["agent_goals"]
    if not isinstance(goals, list) or len(goals) != 2:
        return False, "agent_goals must be a list of exactly 2 goals"

    if len(d["scenario"]) < 50:
        return False, "scenario too short (< 50 chars)"

    for i, g in enumerate(goals):
        if not isinstance(g, str) or len(g) < 20:
            return False, f"Goal {i} too short or not a string (< 20 chars)"

    return True, ""


def dict_to_scenario(d: dict) -> SocialScenario:
    profiles = [AgentProfile(**p) for p in d["agent_profiles"]]
    base_id = str(_uuid.uuid4())
    return SocialScenario(
        id=f"{base_id}_pX",
        source_scenario_id=base_id,
        scenario=d["scenario"],
        agent_profiles=profiles,
        agent_goals=d["agent_goals"],
        relationship=d.get("relationship", ""),
        relationship_background=d.get("relationship_background", ""),
        tag=d.get("tag", d.get("interaction_type", "")),
        interaction_type=d.get("interaction_type", ""),
        difficulty_tags=d.get("difficulty_tags", []),
        source="generated",
    )
