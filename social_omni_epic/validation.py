"""Schema validation and SocialScenario construction for Phase 2 generated scenarios.

Phase 2 schema invariants (enforced here):
  - structured_goals[0]  → learner's three-part triple (outcome / constraint / shortcut)
  - structured_goals[1]  → always None (partner gets a natural-language goal + partner_key)
  - partner_goal          → natural-language string rendered into agent_goals[1]
  - success_rubric        → NOT generated; GOAL≥7∧REL≥0 + key-aware check replaces it

SOTOPIA seeds and Phase-1 archive records bypass this validator (they load via checkpoint).
"""
import uuid as _uuid
from typing import Optional

from .data_models import (
    AgentProfile,
    MECHANISM_LIBRARY,
    PartnerKey,
    SocialScenario,
    StructuredGoal,
)


REQUIRED_TOP = [
    "scenario",
    "agent_profiles",
    "agent_structured_goals",
    "partner_goal",
    "relationship",
    "interaction_type",
    "partner_key",
]
REQUIRED_AGENT = ["first_name", "occupation", "big_five"]
REQUIRED_GOAL = ["outcome", "constraint", "shortcut"]
REQUIRED_KEY = ["key_mechanism", "movement_conditions", "hardening_triggers",
                "surface_misdirection", "cost_coupling"]
VALID_MECHANISMS = set(MECHANISM_LIBRARY.keys())  # derived; add new tags to data_models only


def render_agent_goal(sg: StructuredGoal) -> str:
    """Render a StructuredGoal into the Sotopia-facing goal string (learner / agent 0).

    Single source of truth for agent_goals[0]. The shortcut/leverage is surfaced in
    <extra_info> as available and tempting — NOT labeled as a trap (the agent must
    discover the tension itself). constraint is phrased as the thing to avoid so it
    reads naturally after "without".
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
    """Validate a generated scenario dict (Phase 2 schema).

    Returns (True, "") on success or (False, reason) on failure.
    """
    for f in REQUIRED_TOP:
        if f not in d:
            return False, f"Missing field: {f}"

    # --- agent_profiles ---
    profiles = d["agent_profiles"]
    if not isinstance(profiles, list) or len(profiles) != 2:
        return False, "agent_profiles must be a list of exactly 2 profiles"
    for i, p in enumerate(profiles):
        if not isinstance(p, dict):
            return False, f"Agent profile {i} not a dict"
        for pf in REQUIRED_AGENT:
            if pf not in p or not p[pf]:
                return False, f"Agent profile {i} missing/empty field: {pf}"

    # --- agent_structured_goals: [learner triple, null] ---
    goals = d["agent_structured_goals"]
    if not isinstance(goals, list) or len(goals) != 2:
        return False, "agent_structured_goals must be a list of exactly 2 entries"
    learner_goal = goals[0]
    partner_goal_entry = goals[1]
    if not isinstance(learner_goal, dict):
        return False, "agent_structured_goals[0] (learner) must be a dict"
    for gf in REQUIRED_GOAL:
        if gf not in learner_goal or not str(learner_goal[gf]).strip():
            return False, f"agent_structured_goals[0] missing/empty field: {gf}"
    if partner_goal_entry is not None:
        return False, "agent_structured_goals[1] must be null — partner has no structured triple"

    # --- partner_goal ---
    pg = d.get("partner_goal", "")
    if not isinstance(pg, str) or not pg.strip():
        return False, "partner_goal must be a non-empty string"

    # --- partner_key ---
    key = d.get("partner_key", {})
    if not isinstance(key, dict):
        return False, "partner_key must be a dict"
    for kf in REQUIRED_KEY:
        if kf not in key:
            return False, f"partner_key missing field: {kf}"
    if key.get("key_mechanism") not in VALID_MECHANISMS:
        return False, f"partner_key.key_mechanism must be one of {sorted(VALID_MECHANISMS)}"
    if not isinstance(key.get("movement_conditions"), list) or not key["movement_conditions"]:
        return False, "partner_key.movement_conditions must be a non-empty list"
    if not isinstance(key.get("hardening_triggers"), list) or not key["hardening_triggers"]:
        return False, "partner_key.hardening_triggers must be a non-empty list"

    # --- scenario text ---
    if len(d["scenario"]) < 50:
        return False, "scenario too short (< 50 chars)"

    return True, ""


def _clean_str(s) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\x00", "").replace(" ", " ").strip()


def _clean_dict(d: dict) -> dict:
    return {
        k: (
            _clean_dict(v) if isinstance(v, dict) else
            [_clean_dict(i) if isinstance(i, dict) else
             _clean_str(i) if isinstance(i, str) else i
             for i in v] if isinstance(v, list) else
            _clean_str(v) if isinstance(v, str) else v
        )
        for k, v in d.items()
    }


def dict_to_scenario(d: dict) -> SocialScenario:
    """Convert a validated Phase 2 scenario dict to a SocialScenario object.

    agent_goals[0] = rendered learner triple
    agent_goals[1] = partner_goal string (natural-language)
    structured_goals[1] = None (role invariant)
    partner_key = parsed from dict
    """
    d = _clean_dict(d)

    profiles = [AgentProfile(**p) for p in d["agent_profiles"]]

    # Learner structured goal (agent 0 only)
    learner_sg = StructuredGoal(**d["agent_structured_goals"][0])
    structured = [learner_sg, None]

    # agent_goals: learner from triple, partner from natural-language string
    partner_goal_text = str(d["partner_goal"]).strip()
    rendered_goals = [render_agent_goal(learner_sg), partner_goal_text]

    # Parse partner_key
    partner_key = PartnerKey(**d["partner_key"])

    base_id = str(_uuid.uuid4())
    return SocialScenario(
        id=f"{base_id}_pX",
        source_scenario_id=base_id,
        scenario=d["scenario"],
        agent_profiles=profiles,
        agent_goals=rendered_goals,
        structured_goals=structured,
        goal_type=d.get("goal_type"),
        relationship=d.get("relationship", ""),
        relationship_background=d.get("relationship_background", ""),
        tag=d.get("tag", d.get("interaction_type", "")),
        interaction_type=d.get("interaction_type", ""),
        difficulty_tags=d.get("difficulty_tags", []),
        partner_key=partner_key,
        mutation_operator=d.get("mutation_operator"),
        mutated_slots=d.get("mutated_slots", []),
        source="generated",
    )
