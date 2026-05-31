"""Convert our SocialScenario / AgentProfile to Sotopia's Pydantic models.

We never call .save() and never touch Redis — these objects are passed
directly to ParallelSotopiaEnv and LLMAgent as in-memory Pydantic instances.
"""
from sotopia.database import (
    AgentProfile as SotopiaAgentProfile,
    EnvironmentProfile as SotopiaEnvironmentProfile,
)

from .data_models import SocialScenario


RELATIONSHIP_STR_TO_INT = {
    "strangers": 0,
    "know each other by name": 1,
    "acquaintances": 2,
    "friends": 3,
    "romantic relationship": 4,
    "family members": 5,
}


def _gender_pronoun(gender: str) -> str:
    g = (gender or "").lower()
    if g.startswith("woman") or g == "female":
        return "she/her"
    if g.startswith("man") or g == "male":
        return "he/him"
    return "they/them"


def _to_list(s) -> list[str]:
    """Sotopia stores moral_values / schwartz_personal_values as list[str].
    Our schema stores them as comma-joined strings. Round-trip safely."""
    if isinstance(s, list):
        return [str(x).strip() for x in s if str(x).strip()]
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _to_int_or_zero(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def scenario_to_sotopia_profiles(
    scenario: SocialScenario,
) -> tuple[SotopiaEnvironmentProfile, list[SotopiaAgentProfile]]:
    """Convert our SocialScenario to (EnvironmentProfile, [AgentProfile, AgentProfile])."""

    sotopia_agents: list[SotopiaAgentProfile] = []
    for ap in scenario.agent_profiles:
        sotopia_agents.append(
            SotopiaAgentProfile(
                first_name=ap.first_name or "Agent",
                last_name=ap.last_name or "",
                age=_to_int_or_zero(ap.age) or 30,
                gender=ap.gender_identity or "Nonbinary",
                gender_pronoun=_gender_pronoun(ap.gender_identity),
                occupation=ap.occupation or "",
                big_five=ap.big_five or "",
                moral_values=_to_list(ap.moral_values),
                schwartz_personal_values=_to_list(ap.schwartz_portrait_value),
                decision_making_style=ap.decision_making_style or "",
                secret=ap.secret or "",
                mbti=getattr(ap, "mbti", "") or "",
                personality_and_values=ap.public_info or "",
            )
        )

    rel_raw = scenario.relationship
    if isinstance(rel_raw, int):
        rel_int = rel_raw
    else:
        rel_int = RELATIONSHIP_STR_TO_INT.get((rel_raw or "").lower(), 2)

    # Append relationship background to scenario text so both agents share the
    # context of their prior history.
    scenario_text = scenario.scenario
    if scenario.relationship_background and scenario.relationship_background.strip():
        scenario_text = (
            scenario_text.rstrip()
            + f"\n\nBackground on their relationship: {scenario.relationship_background.strip()}"
        )

    env_profile = SotopiaEnvironmentProfile(
        scenario=scenario_text,
        agent_goals=list(scenario.agent_goals),
        relationship=rel_int,
        tag=scenario.tag or "",
    )
    return env_profile, sotopia_agents
