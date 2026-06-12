import json
import re as _re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz as _fuzz

from .data_models import SocialScenario
from .fm import FM


# Off-channel artifact patterns for partner_key fields. Movement_conditions and
# hardening_triggers must be satisfiable/checkable in spoken turns only — a condition that
# requires paperwork (a "written guarantee", a "signed agreement", an "emailed confirmation")
# pushes the episode toward fabricated off-channel actions. Tightened vs. the chronicle-synthesis
# patterns to require artifact-REQUIREMENT phrasing, minimizing false positives on incidental words.
_KEY_ARTIFACT_PATTERNS = [
    r"\bin writing\b", r"\bwritten (guarantee|agreement|commitment|confirmation|assurance|contract|statement|sign[- ]?off)\b",
    r"\bsign(ed|s|ature)\b", r"\bsign[- ]?off\b", r"\bcontract\b", r"\bescrow\b",
    r"\b(send|sends|sent|email|emails|emailed|forward|forwards|forwarded) (a |an |the )?(email|document|confirmation|note|letter|memo)\b",
    r"\b(documented|documentation)\b", r"\breceipt\b", r"\bnotari", r"\bon paper\b",
]


SYSTEM_PROMPT = """You are a structural validator for social scenarios. Your job is NOT to judge quality or creativity — only to flag logical inconsistencies that would make the scenario internally broken.

Check the following things:

1. RELATIONSHIP CONSISTENCY: Does the relationship label match the relationship_background?
   - "stranger" → background must be empty or describe no prior history. Flag if background describes prior shared experiences.
   - "acquaintance" / "friend" / "romantic" / "family" → background must describe some plausible shared history. Flag if background is empty or contradicts the label.

2. GOAL DISTINCTNESS: Are the two agent goals genuinely different?
   - agent_structured_goals[1] = null is CORRECT and EXPECTED — the partner never has a structured triple; they have a partner_goal string and partner_key instead. Never flag this.
   - Flag if both goals are identical or nearly identical (same objective, same stakes).
   - Do NOT flag opposing goals.

3. PROFILE-GOAL PLAUSIBILITY: Does each agent's occupation, personality (big_five), and decision_making_style make their goal believable?
   - Flag only obvious hard contradictions. Do NOT flag based on challenge or difficulty.

4. SCENARIO-INTERACTION MATCH: Does the scenario description actually describe the stated interaction_type?
   - Flag only if completely mismatched.

5. KEY EXISTENCE & CONSISTENCY (only when partner_key is present): Are movement_conditions concrete, behaviorally checkable, and consistent with the partner's profile and goal? Is each condition a genuine instantiation of the declared key_mechanism? Flag if movement_conditions are vague feelings rather than observable behaviors.

6. KEY-NARRATIVE SEPARATION (only when partner_key is present): Do any movement_conditions appear verbatim or near-verbatim in the shared scenario description, relationship_background, or either agent's goal text?
   - movement_conditions reveal the path to success — flag if they appear in ANY public text.
   - hardening_triggers: flag ONLY if the partner's reaction/consequence is revealed in the scenario description, relationship_background, or the partner's goal text. Do NOT flag overlap with the learner's shortcut, constraint, or secret — by design, the shortcut must mechanistically trip a hardening_trigger (check #8), so near-verbatim overlap between the learner's shortcut and a hardening_trigger is CORRECT and REQUIRED, not a leak.
   - surface_misdirection MAY appear publicly — it is the partner's stated public stance.
   - PARTNER_GOAL LEAK (judge semantically, not lexically): Does the partner_goal state the CONDITIONS under which the partner would move, concede, or accept? The partner_goal should state the partner's stance and what they care about, NOT the specific terms that unlock them. If the partner_goal names the movement_conditions (e.g. "you will only accept a trial with your direct oversight and public credit" when those are the movement_conditions), flag it — that hands the learner the hidden key, collapsing the partial-observability the scenario depends on. The partner's public STANCE may appear; the conditions that actually move them may not.

7. COST COUPLING (only when partner_key is present): Does satisfying the movement_conditions genuinely cost the learner something stated in their own goal? Flag if the cost_coupling is empty, circular, or makes the learner's accommodation essentially free.
   - SURVIVABILITY: also flag the OPPOSITE failure — if satisfying the movement_conditions makes the learner's stated outcome STRICTLY UNREACHABLE (not merely harder or partial), the scenario is cost-stuck, not hard. Example: the learner's goal requires "at least 60% volume shift" but the only conditions that move the partner cap any change at a tiny pilot — there is then no path to any meaningful version of the outcome. A skilled actor who satisfies the conditions must still be able to reach a meaningful (even if partial) version of their goal.

8. SHORTCUT-TRIGGER COUPLING (only when partner_key is present): Does the learner's tempting shortcut plausibly trip at least one hardening_trigger? The shortcut should fail mechanistically (the partner hardens), not only by judge verdict. Flag if the shortcut and the hardening_triggers are unrelated.

9. COOPERATIVE ALIGNMENT: Can the learner win by simply accepting whatever the partner naturally offers, or by freely accommodating without sacrificing anything real?
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
        # The partner (agent 1) carries a natural-language partner_goal in agent_goals[1]
        # instead of a structured triple. Surface it explicitly — otherwise the validator
        # sees structured_goals[1]=null + a partner_key and wrongly reports "partner_goal missing".
        if len(scenario.agent_goals) > 1 and scenario.agent_goals[1]:
            out["partner_goal"] = scenario.agent_goals[1]
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
    """Non-LLM check: flag partner_key content leaking into public-visible text.

    Two scopes (see design note below):
    - movement_conditions: check against ALL public text including agent_goals[0]
      (rendered learner goal + shortcut). Leaking the path to success is a genuine break.
    - hardening_triggers: check ONLY against scenario description, relationship_background,
      and the partner's goal (agent_goals[1]). The learner's shortcut and secret are
      private and are EXPECTED to be mechanistically coupled with hardening_triggers
      (check #9 requires the shortcut to trip a trigger). Flagging that overlap would
      reject well-formed instances of the generative grammar.

    Returns a list of issue strings (empty = no leak detected).
    """
    if scenario.partner_key is None:
        return []

    # Scope A — everything both agents can see, including the learner's rendered goal
    # (which contains the shortcut in <extra_info>). Used for movement_conditions.
    full_public = " ".join([
        scenario.scenario or "",
        scenario.relationship_background or "",
        *[g or "" for g in scenario.agent_goals],
    ]).lower()

    # Scope B — only text the PARTNER can see: shared scenario + their own goal.
    # Excludes agent_goals[0] (learner goal + shortcut). Used for hardening_triggers.
    partner_goal = scenario.agent_goals[1] if len(scenario.agent_goals) > 1 else ""
    partner_public = " ".join([
        scenario.scenario or "",
        scenario.relationship_background or "",
        partner_goal,
    ]).lower()

    issues = []
    for cond in scenario.partner_key.movement_conditions:
        ratio = _fuzz.partial_ratio(cond.lower(), full_public)
        if ratio > 85:
            issues.append(
                f"KEY-NARRATIVE SEPARATION: movement_condition appears near-verbatim in public text "
                f"(ratio={ratio}): {cond[:80]}"
            )
    for trig in scenario.partner_key.hardening_triggers:
        ratio = _fuzz.partial_ratio(trig.lower(), partner_public)
        if ratio > 85:
            issues.append(
                f"KEY-NARRATIVE SEPARATION: hardening_trigger consequence revealed in scenario/partner-goal "
                f"(ratio={ratio}): {trig[:80]}"
            )

    # Off-channel artifact requirement: key conditions must be spoken-turn satisfiable.
    for label, items in (
        ("movement_condition", scenario.partner_key.movement_conditions),
        ("hardening_trigger", scenario.partner_key.hardening_triggers),
    ):
        for item in items:
            for pat in _KEY_ARTIFACT_PATTERNS:
                if _re.search(pat, item, _re.IGNORECASE):
                    issues.append(
                        f"OFF-CHANNEL ARTIFACT: {label} requires an off-channel artifact "
                        f"(matched '{pat}') — rewrite as a spoken-turn equivalent: {item[:80]}"
                    )
                    break
    return issues


def fuzzy_key_leak_check(scenario: SocialScenario) -> list[str]:
    """Public wrapper around the non-LLM key-leak pre-check (free).

    Used by the curriculum runner to drop key-leaking candidates before MOI ranking.
    Returns a list of issue strings (empty = no leak detected).
    """
    return _fuzzy_key_leak_check(scenario)


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
