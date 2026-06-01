import json
from .data_models import SocialScenario
from .fm import FM


SYSTEM_PROMPT = """You are an expert judge evaluating a newly proposed social scenario on two dimensions.

DIMENSION 1 — NOVELTY (interestingly new vs. archive): Is this scenario interestingly new compared to the shown archive scenarios?
A scenario FAILS novelty if it re-skins existing dynamics — same underlying power structure, same type of tension, same strategic challenge — just with different names, occupations, or settings.
A scenario PASSES novelty if it explores a genuinely different social dynamic, introduces a new type of conflict or asymmetry, or adds structural complexity not present in any shown scenario.

DIMENSION 2 — LEARNABILITY: Is there at least one viable social strategy the agent could discover and improve upon across attempts?
A scenario is NOT learnable if:
- The outcome is fixed regardless of what the agent does — the partner's position never shifts no matter how the agent frames, empathizes, or negotiates.
- It is structurally zero-sum with no possible accommodation (no strategy can succeed by construction).
- It requires only surface politeness — no real strategic depth means no learning signal.
- The learner's stated goal requires the partner to take an action that is against the partner's core stated interest AND there is no plausible zone of possible agreement — no creative move, trade, reframing, or partial outcome that skilled play could discover.
A learnable scenario has at least one discoverable path to success that a skilled agent could find — the specific approach does not matter, only that such a path exists.

Respond with a JSON object:
{"novel": true/false, "learnable": true/false, "reason": "concise explanation covering both dimensions"}

The scenario passes only if BOTH are true."""


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
            "\nEvaluate the NEW scenario on NOVELTY and LEARNABILITY. "
            'Respond with JSON: {"novel": true/false, "learnable": true/false, "reason": "..."}'
        )
        try:
            d = self.fm.query_json(SYSTEM_PROMPT, "\n".join(parts), temperature=0.3)
        except Exception as e:
            return True, f"MoI error (defaulting to pass): {e}"

        novel = bool(d.get("novel", True))
        learnable = bool(d.get("learnable", True))
        reason = str(d.get("reason", ""))
        passed = novel and learnable
        return passed, reason
