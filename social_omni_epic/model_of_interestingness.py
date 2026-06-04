import json
from .data_models import SocialScenario
from .fm import FM


SYSTEM_PROMPT = """You are an expert auditor of social scenarios used to train social skill. Judge a newly proposed scenario on THREE dimensions, and if it falls short, say concretely how to fix it.

DIMENSION 1 — SOCIAL TENSION (most important): Is this a genuinely social problem, not a logic puzzle?
A scenario PASSES tension only if, judging the structured goals:
  - the `constraint` genuinely BITES — pursuing the outcome bluntly really would damage the relationship/face;
  - the `shortcut` is genuinely TEMPTING — a naive agent would be drawn to take it and would thereby WIN the outcome but BREAK the constraint;
  - so the naive/default move (take the shortcut) FAILS the real goal, and success requires skilled HOW, not just the WHAT.
It FAILS tension if it is positional bargaining over a number, an extract-a-sentence fetch quest, or any case where a polite/blunt move already succeeds with no relational cost.

DIMENSION 2 — NOVELTY (vs archive): Is it interestingly new versus the shown archive scenarios?
FAILS if it re-skins an existing dynamic (same power structure / same tension / same strategic challenge) with new names or settings. PASSES if it explores a genuinely different social dynamic or adds real structural complexity. (If no archive scenarios are shown, do not fail on novelty.)

DIMENSION 3 — LEARNABILITY / ZOPA: Is there at least one discoverable path to success a skilled agent could find?
FAILS if the outcome is fixed regardless of skill, or requires the partner to fully capitulate with no possible middle ground. PASSES if creative framing, timing, disclosure, or trade-offs could plausibly move the partner.

Respond with ONLY valid JSON:
{"social_tension": true/false, "novel": true/false, "learnable": true/false,
 "reason": "concise explanation across the three dimensions",
 "suggested_edits": ["concrete change 1", "concrete change 2"]}

If all three are true, suggested_edits must be an empty list. Otherwise, suggested_edits must say
specifically what to change (especially to raise social tension) — without adding facts, parties, or
numeric complexity."""


def _format(s: SocialScenario) -> str:
    out = {
        "scenario": s.scenario,
        "interaction_type": s.interaction_type,
        "relationship": s.relationship,
        "difficulty_tags": s.difficulty_tags,
    }
    if any(sg is not None for sg in (s.structured_goals or [])):
        out["agent_structured_goals"] = [
            sg.model_dump() if sg else None for sg in s.structured_goals
        ]
        out["agent_secrets"] = [p.secret for p in s.agent_profiles]
        if s.goal_type:
            out["goal_type"] = s.goal_type
    else:
        out["agent_goals"] = s.agent_goals
    return json.dumps(out, indent=2)


class ModelOfInterestingness:
    def __init__(self, fm: FM, num_examples: int = 5, min_archive_size: int = 10):
        self.fm = fm
        self.num_examples = num_examples
        self.min_archive_size = min_archive_size

    def evaluate(
        self, new_scenario: SocialScenario, similar: list[SocialScenario]
    ) -> tuple[bool, str, list[str]]:
        """Audit the scenario. Returns (passed, reason, suggested_edits).

        passed = social_tension AND learnable AND (novel OR no archive shown). On error, passes
        (do not block the pipeline) with empty edits.
        """
        parts = ["NEW SCENARIO TO EVALUATE:", _format(new_scenario)]
        if similar:
            parts.append("\nMOST SIMILAR EXISTING SCENARIOS (for novelty comparison):")
            for i, s in enumerate(similar):
                parts.append(f"--- Existing {i + 1} ---")
                parts.append(_format(s))
        else:
            parts.append("\n(No existing scenarios yet — do not fail on novelty.)")
        parts.append(
            "\nAudit the NEW scenario on SOCIAL TENSION, NOVELTY, and LEARNABILITY. "
            'Respond with JSON: {"social_tension":..., "novel":..., "learnable":..., '
            '"reason":"...", "suggested_edits":[...]}'
        )
        try:
            d = self.fm.query_json(SYSTEM_PROMPT, "\n".join(parts), temperature=0.3)
        except Exception as e:
            return True, f"MoI error (defaulting to pass): {e}", []

        tension = bool(d.get("social_tension", True))
        novel = bool(d.get("novel", True)) if similar else True
        learnable = bool(d.get("learnable", True))
        reason = str(d.get("reason", ""))
        edits = [str(x) for x in d.get("suggested_edits", []) if str(x).strip()]
        passed = tension and novel and learnable
        return passed, reason, edits
