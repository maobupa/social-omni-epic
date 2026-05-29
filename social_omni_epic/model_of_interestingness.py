import json
from .data_models import SocialScenario
from .fm import FM


SYSTEM_PROMPT = """You are an expert judge evaluating a newly proposed social scenario on two dimensions simultaneously.

DIMENSION 1 — NOVELTY: Is the scenario genuinely different from the existing archived scenarios shown?
A scenario lacks novelty if it merely changes names, locations, or surface details while repeating the same underlying social dynamic, power structure, or strategic challenge.

DIMENSION 2 — VALIDITY: Is the scenario itself non-degenerate?
A scenario is invalid if it is:
- Trivially solved (no real social challenge requiring strategy)
- Socially implausible (the setup could not realistically occur)
- So vague or underspecified that no meaningful interaction could happen

Respond with a JSON object:
{"interesting": true/false, "valid": true/false, "reason": "concise explanation covering both dimensions"}

The scenario passes only if BOTH interesting AND valid are true."""


def _format(s: SocialScenario) -> str:
    return json.dumps({
        "scenario": s.scenario,
        "interaction_type": s.interaction_type,
        "relationship": s.relationship,
        "agent_goals": s.agent_goals,
        "difficulty_tags": s.difficulty_tags,
    }, indent=2)


class ModelOfInterestingness:
    def __init__(self, fm: FM, num_examples: int = 5, min_archive_size: int = 10):
        self.fm = fm
        self.num_examples = num_examples
        self.min_archive_size = min_archive_size

    def evaluate(self, new_scenario: SocialScenario,
                 similar: list[SocialScenario]) -> tuple[bool, str]:
        """Return (passed, reason). Passes only if novel AND valid."""
        parts = ["NEW SCENARIO TO EVALUATE:", _format(new_scenario)]
        if similar:
            parts.append("\nMOST SIMILAR EXISTING SCENARIOS (for novelty comparison):")
            for i, s in enumerate(similar):
                parts.append(f"--- Existing {i + 1} ---")
                parts.append(_format(s))
        else:
            parts.append("\n(No existing scenarios yet — only evaluate validity.)")
        parts.append(
            "\nEvaluate the NEW scenario on both NOVELTY and VALIDITY. "
            'Respond with JSON: {"interesting": true/false, "valid": true/false, "reason": "..."}'
        )
        try:
            d = self.fm.query_json(SYSTEM_PROMPT, "\n".join(parts), temperature=0.3)
        except Exception as e:
            return True, f"MoI error (defaulting to pass): {e}"

        interesting = bool(d.get("interesting", True))
        valid = bool(d.get("valid", True))
        reason = str(d.get("reason", ""))
        passed = interesting and valid
        return passed, reason
