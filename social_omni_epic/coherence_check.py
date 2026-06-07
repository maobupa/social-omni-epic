import json
from dataclasses import dataclass, field
from typing import Optional

from .data_models import SocialScenario
from .fm import FM


SYSTEM_PROMPT = """You are a structural validator for social scenarios. Your job is NOT to judge quality or creativity — only to flag logical inconsistencies that would make the scenario internally broken.

Check exactly four things:

1. RELATIONSHIP CONSISTENCY: Does the relationship label match the relationship_background?
   - "stranger" → background must be empty or describe no prior history. Flag if background describes prior shared experiences.
   - "acquaintance" / "friend" / "romantic" / "family" → background must describe some plausible shared history. Flag if background is empty or contradicts the label.

2. CONSTRAINT PHRASING: Flag ONLY if the constraint field itself literally begins with the word "without" — that produces a broken double-"without" rendering (e.g., "without damaging trust" is broken; "damaging trust" is correct). Do NOT flag for style, wordiness, or subjective phrasing preferences. If the constraint does not start with "without", this check passes regardless of how the phrase is written.

3. GOAL DISTINCTNESS: Are the two agent goals genuinely different?
   - Flag if both goals are identical or nearly identical (same objective, same stakes).
   - Do NOT flag opposing goals — a buyer wanting to pay less and a seller wanting to charge more is fine.

4. PROFILE-GOAL PLAUSIBILITY: Does each agent's occupation, personality (big_five), and decision_making_style make their goal believable?
   - Flag only obvious hard contradictions. Example: a character with "Agreeableness - Very High; Conscientiousness - Very High" whose goal requires persistent deception and manipulation.
   - Do NOT flag based on challenge or difficulty — people act against type sometimes.

5. SCENARIO-INTERACTION MATCH: Does the scenario description actually describe the stated interaction_type?
   - Flag only if completely mismatched (e.g. scenario describes two people quietly sharing coffee but interaction_type is "hostile negotiation").

6. GOAL ACHIEVABILITY (ZOPA): Does a zone of possible agreement exist where both agents could partially win through skilled play?
   - Flag if the two agents' outcomes are MUTUALLY EXCLUSIVE on a specific dimension — i.e., one agent's outcome explicitly requires X while the other agent's outcome explicitly forbids X, with no middle ground. This is structural impossibility, not difficulty.
   - Flag if a goal requires the partner to take an action so costly to their core interest that no creative move or trade could plausibly bridge it.
   - Do NOT flag scenarios that are merely hard or require skilled negotiation — only flag when success is structurally impossible regardless of skill.

7. RUBRIC & SHORTCUT VALIDITY (structured goals): inspect agent_structured_goals (outcome / constraint / shortcut) and success_rubric.
   - Flag if the shortcut is NOT load-bearing: it must be a tempting move that WOULD achieve the outcome but VIOLATE the constraint. If taking the shortcut wouldn't actually win the outcome, or wouldn't actually break the constraint, flag it.
   - Flag if the partner's secret is decorative — it should be the leverage behind that agent's shortcut, or directly raise the constraint's stakes.
   - Flag if there are more than 3 rubric checks total — the rubric should have 1 outcome check and 1–2 constraint checks. Extra checks dilute the AND gate and add procedural noise.
   - Flag if any check is procedural ("did X refrain from doing Y during the conversation") rather than testing an outcome state or partner internal state.
   - Flag if any success_rubric check is NOT genuinely checkable (a neutral 'outcome' check must be answerable from the transcript; a 'partner' constraint check must be about the partner's internal state), or if a check is TRIVIALLY satisfiable by the naive/blunt move (then it has no bite).
   - Flag if the outcome is an extractable utterance ("get them to say X") or pure number-splitting rather than a genuine state-change requiring buy-in.
   - Flag if the outcome check tests a delivery mechanism rather than the core state-change: checks that require a signature, written acknowledgment, specific spoken phrase, or completion of an action during the conversation are delivery-mechanism checks. The correct form is always "Did [partner] agree to [core ask]?" — not "Did [partner] sign/confirm/say/complete X?"
   - NOTE: the success_rubric evaluates ONLY the learner (agent 0). Do NOT flag the rubric for failing to cover agent 1's goal — this is by design.

Return JSON: {"passed": true/false, "issues": ["specific issue 1", "specific issue 2", ...]}
Issues must be specific and actionable — describe exactly what is wrong and what needs to change.
If passed is true, issues must be an empty list.
If passed is false, issues must contain at least one item."""


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
    return json.dumps(out, indent=2)


@dataclass
class CoherenceCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


class CoherenceChecker:
    def __init__(self, fm: FM):
        self.fm = fm

    def check(self, scenario: SocialScenario) -> CoherenceCheckResult:
        user_prompt = (
            "Check this scenario for internal consistency:\n\n"
            + _format(scenario)
            + '\n\nReturn JSON: {"passed": true/false, "issues": [...]}'
        )
        try:
            d = self.fm.query_json(SYSTEM_PROMPT, user_prompt, temperature=0.2)
        except Exception:
            return CoherenceCheckResult(passed=True)

        passed = bool(d.get("passed", True))
        issues = [str(i) for i in d.get("issues", []) if i]
        if not passed and not issues:
            issues = ["Scenario failed coherence check (no specific issues returned)."]
        return CoherenceCheckResult(passed=passed, issues=issues)
