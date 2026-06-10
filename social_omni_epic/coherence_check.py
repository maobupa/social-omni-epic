import json
from dataclasses import dataclass, field
from typing import Optional

from .data_models import SocialScenario
from .fm import FM


SYSTEM_PROMPT = """You are a structural validator for social scenarios. Your job is NOT to judge quality or creativity — only to flag logical inconsistencies that would make the scenario internally broken.

Check the following things:

1. RELATIONSHIP CONSISTENCY: Does the relationship label match the relationship_background?
   - "stranger" → background must be empty or describe no prior history. Flag if background describes prior shared experiences.
   - "acquaintance" / "friend" / "romantic" / "family" → background must describe some plausible shared history. Flag if background is empty or contradicts the label.

2. CONSTRAINT PHRASING: Flag ONLY if the constraint field itself literally begins with the word "without" — that produces a broken double-"without" rendering. Do NOT flag for style, wordiness, or subjective phrasing preferences.

3. GOAL DISTINCTNESS: Are the two agent goals genuinely different?
   - Flag if both goals are identical or nearly identical (same objective, same stakes).
   - Do NOT flag opposing goals.

4. PROFILE-GOAL PLAUSIBILITY: Does each agent's occupation, personality (big_five), and decision_making_style make their goal believable?
   - Flag only obvious hard contradictions. Do NOT flag based on challenge or difficulty.

5. SCENARIO-INTERACTION MATCH: Does the scenario description actually describe the stated interaction_type?
   - Flag only if completely mismatched.

6. KEY EXISTENCE & CONSISTENCY (only when partner_key is present): Are movement_conditions concrete, behaviorally checkable, and consistent with the partner's profile and goal? Is each condition a genuine instantiation of the declared key_mechanism? Flag if movement_conditions are vague feelings rather than observable behaviors.

7. KEY-NARRATIVE SEPARATION (only when partner_key is present): Does any movement_condition, hardening_trigger, or the resolution of the surface_misdirection appear verbatim or near-verbatim in the shared scenario description or the learner's goal text? The surface_misdirection itself MAY appear publicly — it is the partner's public stance. Flag only if hidden conditions leak.

8. COST COUPLING (only when partner_key is present): Does satisfying the movement_conditions genuinely cost the learner something stated in their own goal? Flag if the cost_coupling is empty, circular, or makes the learner's accommodation essentially free.

9. SHORTCUT-TRIGGER COUPLING (only when partner_key is present): Does the learner's tempting shortcut plausibly trip at least one hardening_trigger? The shortcut should fail mechanistically (the partner hardens), not only by judge verdict. Flag if the shortcut and the hardening_triggers are unrelated.

10. COOPERATIVE ALIGNMENT: Can the learner win by simply accepting whatever the partner naturally offers, or by freely accommodating without sacrificing anything real?
    - If `competing_interest` is present: flag if full accommodation does NOT demonstrably forfeit this competing interest.
    - If `partner_default_position` is present: flag if the partner's default already satisfies the learner's outcome.
    - PASS if EITHER holds: (a) competing_interest genuinely bites under full accommodation, OR (b) partner_default_position falls short of the learner's outcome.
    - If neither field is present, skip this check entirely.

Return JSON: {"passed": true/false, "issues": ["specific issue 1", "specific issue 2", ...]}
Issues must be specific and actionable. If passed is true, issues must be empty. If passed is false, issues must contain at least one item."""


def _format(scenario: SocialScenario) -> str:
    profiles_summary = []
    for p in scenario.agent_profiles:
        profiles_summary.append({
            "first_name": p.first_name,
            "occupation": p.occupation,
            "big_five": p.big_five,
            "decision_making_style": p.decision_making_style,
        })
    out = {
        "scenario": scenario.scenario,
        "interaction_type": scenario.interaction_type,
        "relationship": scenario.relationship,
        "relationship_background": scenario.relationship_background,
        "agent_profiles": profiles_summary,
    }
    if any(sg is not None for sg in (scenario.structured_goals or [])):
        out["agent_structured_goals"] = [
            sg.model_dump() if sg else None for sg in scenario.structured_goals
        ]
        out["agent_secrets"] = [p.secret for p in scenario.agent_profiles]
        if scenario.success_rubric:
            out["success_rubric"] = scenario.success_rubric.model_dump()
    else:
        out["agent_goals"] = scenario.agent_goals
    if scenario.competing_interest:
        out["competing_interest"] = scenario.competing_interest
    if scenario.partner_default_position:
        out["partner_default_position"] = scenario.partner_default_position
    # Include partner_key for checks 6-9 (hidden from the scenario prompt; visible to the validator).
    if scenario.partner_key is not None:
        out["partner_key"] = scenario.partner_key.model_dump()
    return json.dumps(out, indent=2)


def _fuzzy_key_leak_check(scenario: SocialScenario) -> list[str]:
    """Non-LLM check: flag if any movement_condition or hardening_trigger string appears
    nearly verbatim in the public scenario text (rapidfuzz partial_ratio > 85).

    Returns a list of issue strings (empty = no leak detected).
    """
    if scenario.partner_key is None:
        return []
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return []

    public_text = " ".join([
        scenario.scenario or "",
        *[g or "" for g in scenario.agent_goals],
    ]).lower()

    issues = []
    for cond in scenario.partner_key.movement_conditions:
        if fuzz.partial_ratio(cond.lower(), public_text) > 85:
            issues.append(
                f"KEY-NARRATIVE SEPARATION: movement_condition appears near-verbatim in public text: {cond[:80]}"
            )
    for trig in scenario.partner_key.hardening_triggers:
        if fuzz.partial_ratio(trig.lower(), public_text) > 85:
            issues.append(
                f"KEY-NARRATIVE SEPARATION: hardening_trigger appears near-verbatim in public text: {trig[:80]}"
            )
    return issues


@dataclass
class CoherenceCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


class CoherenceChecker:
    def __init__(self, fm: FM):
        self.fm = fm

    def check(self, scenario: SocialScenario) -> CoherenceCheckResult:
        # Non-LLM key-leak pre-check (fast, free).
        fuzzy_issues = _fuzzy_key_leak_check(scenario)
        if fuzzy_issues:
            return CoherenceCheckResult(passed=False, issues=fuzzy_issues)

        user_prompt = (
            "Check this scenario for internal consistency:\n\n"
            + _format(scenario)
            + '\n\nReturn JSON: {"passed": true/false, "issues": [...]}'
        )
        try:
            d = self.fm.query_json(SYSTEM_PROMPT, user_prompt, temperature=0.2)
        except Exception as e:
            # Quarantine on error — never default to PASS (§6.1 failure policy).
            return CoherenceCheckResult(
                passed=False,
                issues=[f"quarantined: coherence check FM error: {e}"],
            )

        passed = bool(d.get("passed", True))
        issues = [str(i) for i in d.get("issues", []) if i]
        if not passed and not issues:
            issues = ["Scenario failed coherence check (no specific issues returned)."]
        return CoherenceCheckResult(passed=passed, issues=issues)
