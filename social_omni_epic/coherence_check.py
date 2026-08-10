import json
import re as _re
from dataclasses import dataclass, field

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

5. INTERNAL_STATE IS A STATE, NOT A REQUIREMENT (only when partner_key is present): internal_state must describe what is TRUE of the partner right now — what they feel, believe, fear, or half-suspect. Flag if it instead prescribes what the learner should do ("she needs someone to acknowledge...", "he wants to be told..."). That phrasing turns the field into an answer key and destroys the point, which is that many different behaviours could reach the same person. Also flag if it is so thin that no behavioural implication can be inferred from it.

6. THE KEY MUST BE DERIVABLE FROM THE STATE (only when partner_key is present): does each movement_condition plausibly PRODUCE the internal_state being addressed, and does each hardening_trigger plausibly DENY or worsen it? Flag any movement_condition that is unrelated to the internal_state — the conditions exist to prove at least one route to that state exists, so an unrelated condition witnesses nothing. Are movement_conditions concrete and behaviourally checkable (things the LEARNER does), rather than vague feelings or partner demands?

7. KEY-NARRATIVE SEPARATION (only when partner_key is present): does any hidden key content appear in text the learner can see?
   - internal_state is the graded object and must NEVER appear, in any paraphrase, in the scenario description, relationship_background, or either agent's goal text. This is the strictest rule here.
   - movement_conditions reveal a route to success — flag if they appear in ANY public text.
   - hardening_triggers: flag ONLY if the partner's reaction/consequence is revealed in the scenario description, relationship_background, or the partner's goal text. Do NOT flag overlap with the learner's shortcut, constraint, or secret — by design the shortcut must mechanistically trip a hardening_trigger (check #8), so near-verbatim overlap there is CORRECT and REQUIRED, not a leak.
   - PARTNER_GOAL LEAK (judge semantically, not lexically): partner_goal should state the partner's POSITION and stake — what they are materially after and would admit to. It must NOT state the conditions under which they would move, and must not describe the internal_state. If partner_goal reads like an explanation of what would satisfy them, flag it.

8. SHORTCUT-TRIGGER COUPLING (only when partner_key is present): Does the learner's tempting shortcut plausibly trip at least one hardening_trigger? The shortcut should fail mechanistically (the partner hardens), not only by judge verdict. Flag if the shortcut and the hardening_triggers are unrelated.

9. STATED-VS-ACTUAL GAP (only when partner_key is present): if the partner got exactly what their partner_goal asks for, would the internal_state be addressed?
   - If YES → FLAG IT. There is no hidden depth: the learner can win by simply handing over the stated thing, so the key is decoration. This is the single most common way these scenarios come out trivial.
   - If NO → pass. They can get precisely what they said they wanted and still be unsatisfied, and that gap is where the difficulty lives.
   NOTE: the gap does NOT require the two parties to want opposing things. Passing example with no opposition at all: partner_goal "get this off my chest without being managed"; internal_state "she is furious rather than sad, everyone has treated it as grief, and it makes her feel unseen." Letting her talk uninterrupted would not address that if she is still being handled. Do not flag a scenario merely for lacking a bargaining conflict.
   Passing example with opposition: partner_goal "remain the recognised local contact for the clean-up crews"; internal_state "he has run these crews for eleven years and no one ever said it mattered; he reads every new organiser as an eventual replacement." Keeping control of outreach would not make him feel his work counted.

10. GOAL IS CONFIRMABLE: could a reader of the finished transcript confirm or deny that the learner's `outcome` actually happened? It need not be a concession, an agreement, or a number — "the partner voluntarily says why they have been avoiding you" is perfectly good. Flag only outcomes that are unfalsifiable from a transcript ("make them feel better about themselves" with no observable marker).

11. WINNABLE IN CHANNEL: is the outcome reachable inside this two-party spoken conversation? Flag if it requires a decision that belongs to an absent third party (an executor, a committee, an administrator who must post something), or a numeric target below a floor the partner's own position makes immovable. These are the two ways a scenario becomes unwinnable-by-construction rather than hard.

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

    else:
        out["agent_goals"] = scenario.agent_goals
    if scenario.competing_interest:
        out["competing_interest"] = scenario.competing_interest
    if scenario.partner_default_position:
        out["partner_default_position"] = scenario.partner_default_position
    # Include partner_key for checks 5-8 and 10 (hidden from the scenario prompt; visible to the validator).
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
    # internal_state is the graded object in schema v2. If it appears in public text the learner is
    # simply told the answer, so this is checked first and against the widest scope.
    _state = (scenario.partner_key.internal_state or "").strip()
    if _state:
        ratio = _fuzz.partial_ratio(_state.lower(), full_public)
        if ratio > 80:
            issues.append(
                f"KEY-NARRATIVE SEPARATION: internal_state appears near-verbatim in public text "
                f"(ratio={ratio}): {_state[:80]}"
            )
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
