import json
from .data_models import SocialScenario
from .fm import FM


SYSTEM_PROMPT = """You are an expert judge of social scenario diversity. Given a NEW social scenario and several EXISTING scenarios that are most similar to it, determine whether the new scenario is INTERESTINGLY DIFFERENT from the existing ones.

A scenario is interestingly different if it introduces:
- A genuinely new type of social dynamic (not just a topic change)
- A different power structure or information asymmetry
- A novel moral or ethical dimension
- A different type of relationship or social context
- A meaningfully different strategic challenge for the agents

A scenario is NOT interestingly different if it merely:
- Changes names, locations, or surface details
- Repeats the same social dynamic with a different topic
- Is a minor variation of an existing scenario

Respond with a JSON object: {"is_interesting": true/false, "reasoning": "..."}"""


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
        parts = ["NEW SCENARIO:", _format(new_scenario),
                 "\nMOST SIMILAR EXISTING SCENARIOS:"]
        for i, s in enumerate(similar):
            parts.append(f"--- Existing {i+1} ---")
            parts.append(_format(s))
        parts.append(
            '\nIs the NEW scenario interestingly different from the EXISTING ones? '
            'Respond with JSON: {"is_interesting": true/false, "reasoning": "..."}'
        )
        try:
            d = self.fm.query_json(SYSTEM_PROMPT, "\n".join(parts), temperature=0.3)
        except Exception as e:
            return True, f"MoI error (defaulting to interesting): {e}"
        return bool(d.get("is_interesting", True)), str(d.get("reasoning", ""))
