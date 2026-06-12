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


# Slot label → partner_key field name(s). Only key slots (b/c/d) are mutation-fidelity
# checked; surface slots (a/e/f/g) are not in partner_key and are covered by the
# preservation side of the check instead.
_KEY_SLOT_FIELDS: dict[str, list[str]] = {
    "b": ["surface_misdirection"],
    "c": ["hardening_triggers"],   # list — joined before comparison
    "d": ["cost_coupling"],
}


def key_delta_check(
    child: SocialScenario,
    anchor: SocialScenario,
    threshold: int = 90,
    preservation_floor: int = 60,
) -> list[str]:
    """DEPRECATED (Patch 10) — no longer called by the production runner.

    Superseded by universal embedding diversity gating + surface_novelty_check, which removed
    the operator-conditional gate this check belonged to. Retained for documentation of the old
    escalate/relax mutation-fidelity design and possible post-hoc analysis. See
    docs/pre_run_final_Patch.md (Fault 2) for why the gate it served was removed.

    Two-sided mutation-fidelity check for escalate/relax children.

    Skip entirely when anchor has no partner_key (seed parents — any key is novel).

    Delta side: at least one mutated key-slot must differ from the parent
    (rapidfuzz ratio ≤ threshold). Catches LLM returning a lazy near-copy that
    ignored the mutation instruction.

    Preservation side: character first names must match the parent AND scenario text
    must clear a loose similarity floor (partial_ratio ≥ preservation_floor).
    Catches an "escalate" that actually swapped characters/premise — which is really
    an unlabeled lateral and must go through the embedding gate we just exempted
    escalate/relax from.

    Special violations:
    - mutated_slots declares no key slots at all (e.g. only ["a"]) → reject; the
      operator claims to have mutated difficulty knobs but didn't name any.
    - anchor has partner_key but child does not → schema error, reject.

    Returns a list of violation strings; empty = pass.
    """
    if anchor.partner_key is None:
        return []

    if child.partner_key is None:
        return ["KEY-DELTA: child has no partner_key but anchor does"]

    # Normalise mutated_slots: strip whitespace, lower-case
    mutated = [s.strip().lower() for s in (child.mutated_slots or [])]
    key_slots_declared = [s for s in mutated if s in _KEY_SLOT_FIELDS]

    if not key_slots_declared:
        # escalate/relax with zero declared key slots is a malformed mutation
        return [
            "KEY-DELTA: mutated_slots declares no partner_key slots (b/c/d); "
            "escalate/relax must mutate at least one difficulty knob"
        ]

    violations: list[str] = []

    # --- Delta side: require at least one declared slot to have actually changed ---
    all_similar = True
    for slot in key_slots_declared:
        fields = _KEY_SLOT_FIELDS[slot]
        for field in fields:
            child_val = getattr(child.partner_key, field, None)
            anchor_val = getattr(anchor.partner_key, field, None)
            # Lists (hardening_triggers) → join for comparison
            if isinstance(child_val, list):
                child_val = " | ".join(str(x) for x in child_val)
            if isinstance(anchor_val, list):
                anchor_val = " | ".join(str(x) for x in anchor_val)
            child_val = str(child_val or "").strip()
            anchor_val = str(anchor_val or "").strip()
            ratio = _fuzz.ratio(child_val.lower(), anchor_val.lower())
            if ratio <= threshold:
                all_similar = False
                break
        if not all_similar:
            break

    if all_similar:
        violations.append(
            f"KEY-DELTA: all declared key slots {key_slots_declared} are near-identical "
            f"to parent (fuzz ratio > {threshold}); mutation instruction was ignored"
        )

    # --- Preservation side: surface must stay close to the parent ---
    # Check 1: character first names
    child_names = {p.first_name.strip().lower() for p in (child.agent_profiles or []) if p.first_name}
    anchor_names = {p.first_name.strip().lower() for p in (anchor.agent_profiles or []) if p.first_name}
    if anchor_names and not child_names.issuperset(anchor_names):
        missing = anchor_names - child_names
        violations.append(
            f"KEY-DELTA-PRESERVATION: character first name(s) changed under escalate/relax "
            f"(missing from child: {missing}); this is an unlabeled lateral mutation"
        )

    # Check 2: scenario text surface similarity (loose — only catches wholesale rewrites)
    child_text = (child.scenario or "").strip()
    anchor_text = (anchor.scenario or "").strip()
    if anchor_text and child_text:
        surface_sim = _fuzz.partial_ratio(child_text.lower(), anchor_text.lower())
        if surface_sim < preservation_floor:
            violations.append(
                f"KEY-DELTA-PRESERVATION: scenario text drifted too far from parent "
                f"(partial_ratio={surface_sim} < {preservation_floor}); "
                "escalate/relax must preserve the surface premise"
            )

    return violations


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
