"""Schema validation and SocialScenario construction for Phase 2 generated scenarios.

Phase 2 schema invariants (enforced here):
  - structured_goals[0]  → learner's three-part triple (outcome / constraint / shortcut)
  - structured_goals[1]  → always None (partner gets a natural-language goal + partner_key)
  - partner_goal          → natural-language string rendered into agent_goals[1]
  - success_rubric        → NOT generated; GOAL≥7∧REL≥0 + key-aware check replaces it

SOTOPIA seeds and Phase-1 archive records bypass this validator (they load via checkpoint).
"""
import uuid as _uuid

from rapidfuzz import fuzz as _fuzz

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
# Schema v2. `internal_state` is now required and is the graded object; `surface_misdirection`
# and `cost_coupling` are retired (still Optional on the model so v1 records keep parsing, but
# no longer generated or validated).
REQUIRED_KEY = ["key_mechanism", "movement_conditions", "hardening_triggers", "internal_state"]

# MECHANISM_LIBRARY is now a set of EXAMPLES shown to the generator, not a closed enum.
# Enforcing it as an enum was a diversity bottleneck — gen-90 came out face_needs 35/90 vs
# reactance 3/90 — and under v2 the psychology lives in `internal_state` anyway, so forcing it
# into five buckets constrains the axis we are trying to widen. Same "require a property, don't
# enumerate categories" rule as the goal grammar. Kept importable for analysis/back-compat.
VALID_MECHANISMS = set(MECHANISM_LIBRARY.keys())


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
    # key_mechanism is a free-text psychology label (see VALID_MECHANISMS above) — presence only.
    if not isinstance(key.get("key_mechanism"), str) or not key["key_mechanism"].strip():
        return False, "partner_key.key_mechanism must be a non-empty string"
    if not isinstance(key.get("movement_conditions"), list) or not key["movement_conditions"]:
        return False, "partner_key.movement_conditions must be a non-empty list"
    if not isinstance(key.get("hardening_triggers"), list) or not key["hardening_triggers"]:
        return False, "partner_key.hardening_triggers must be a non-empty list"
    # internal_state must be a STATE, not a requirement. A field phrased as "X needs the learner
    # to ..." has smuggled the answer key back in one level up; the generator prompt says so and
    # this is the deterministic backstop. Checked here rather than only in the LLM coherence gate
    # because it is free and the failure is purely lexical.
    st = key.get("internal_state")
    if not isinstance(st, str) or len(st.strip()) < 20:
        return False, "partner_key.internal_state must be a descriptive string (>= 20 chars)"
    _low = st.lower()
    for phrase in ("needs the learner", "needs someone to", "wants the learner",
                   "would be satisfied by", "the learner must", "the learner should"):
        if phrase in _low:
            return False, (
                f"partner_key.internal_state is resolution-shaped (contains '{phrase}') — it must "
                "describe what is TRUE of the partner, not what the learner should do"
            )

    # --- scenario text ---
    if len(d["scenario"]) < 50:
        return False, "scenario too short (< 50 chars)"

    return True, ""


def surface_novelty_check(child: SocialScenario, anchor: SocialScenario) -> list[str]:
    """Cheap deterministic surface-novelty guard, applied to EVERY admitted child (Patch 10).

    Under the unified direction-setter operators, every child must have a fresh surface relative
    to its parent. The embedding diversity gate owns novelty against the whole archive; this is a
    free pre-check that catches the two most common lazy mutations before the embedding call:

    1. Reused character first name(s) — the strongest tell that the generator re-skinned rather
       than re-imagined (e.g. the Sasha/Emily lateral).
    2. Near-verbatim scenario text (partial_ratio > 90) — a clone that slipped the prompt.

    Returns a list of violation strings; empty = pass. Deliberately does NOT check mutated_slots
    (that is a descriptive self-report, not a contract — an empty-slots warning is emitted in the
    admission path instead).
    """
    violations: list[str] = []

    child_names = {p.first_name.strip().lower() for p in (child.agent_profiles or []) if p.first_name}
    anchor_names = {p.first_name.strip().lower() for p in (anchor.agent_profiles or []) if p.first_name}
    reused = child_names & anchor_names
    if reused:
        violations.append(
            f"SURFACE-NOVELTY: child reuses parent character first name(s) {reused} — "
            "every mutation must use a completely fresh surface (new names)"
        )

    child_text = (child.scenario or "").strip()
    anchor_text = (anchor.scenario or "").strip()
    if anchor_text and child_text:
        sim = _fuzz.partial_ratio(child_text.lower(), anchor_text.lower())
        if sim > 90:
            violations.append(
                f"SURFACE-NOVELTY: scenario text is near-verbatim to parent (partial_ratio={sim}) — "
                "this is a clone, not a mutation"
            )

    return violations


# NOTE: key_delta_check() and its _KEY_SLOT_FIELDS slot table were deleted here (schema v2).
# They implemented the old operator-conditional mutation-fidelity gate, which Patch 10 had
# already replaced with universal embedding diversity + surface_novelty_check, so the function
# was documented-deprecated and uncalled. Two of its three slots pointed at surface_misdirection
# and cost_coupling, which v2 retires. See docs/pre_run_final_Patch.md (Fault 2) for the history.

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

    # Learner structured goal (agent 0 only).
    # Normalize constraint: the renderer prepends "without ", so strip any leading "without "
    # the LLM wrote literally — otherwise the rendered goal reads "without without ...".
    sg_raw = dict(d["agent_structured_goals"][0])
    if isinstance(sg_raw.get("constraint"), str):
        c = sg_raw["constraint"].strip()
        if c.lower().startswith("without "):
            c = c[len("without "):].lstrip()
        sg_raw["constraint"] = c
    learner_sg = StructuredGoal(**sg_raw)
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
