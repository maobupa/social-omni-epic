"""SCENARIO_TITLE generation and scenario structural classification (§4.9, §4.1).

SCENARIO_TITLE is a pipe-separated string:
  "[Social dynamic description] | [Target perspective]"

Both halves are embedded together as the primary archive retrieval key.

  social_dynamic   — left half: scenario-structure similarity signal
  target_perspective — right half: perspective-within-structure signal

Two scenarios can have identical social structure with roles reversed; the right
half distinguishes them in embedding space.

Also provides classify_scenario() which assigns GOAL_STRUCTURE and INFO_POSITION
programmatically via a single compact LLM call.
"""
import json
from typing import Optional

from .data_models import SocialScenario
from .fm import FM


# ---------------------------------------------------------------------------
# SCENARIO_TITLE
# ---------------------------------------------------------------------------

_TITLE_SYSTEM = """You generate concise retrieval keys for social interaction scenarios.

A SCENARIO_TITLE has exactly two parts separated by a pipe character (|):

  [Social dynamic description] | [Target perspective]

LEFT HALF — describe the abstract social dynamic:
- The structural type of interaction (negotiation, secret-keeping, persuasion under pressure, etc.)
- Key asymmetries (time pressure, information gap, power imbalance)
- What makes this scenario strategically interesting
- Relationship structure when strategically relevant (e.g., stranger vs. close friend vs. acquaintance) — include only when it materially changes the social dynamic
- NO proper nouns, specific occupations, or scenario-specific surface details
- Aim for 8–20 words

RIGHT HALF — describe what skills were built from the TARGET agent's perspective:
- The structural vantage point (e.g., the uninformed buyer, the secret-holder, the mediator)
- What the target agent was trying to navigate
- NO proper nouns or scenario-specific details
- Aim for 8–15 words starting with "skills from the..."
- Describe a structural role or position, not an action sequence. Format: "skills from the [structural role] [brief characterization of what they were navigating]"

EXAMPLES:
  "Zero-sum resource negotiation with asymmetric time pressure and anchoring behavior | skills from the uninformed, patient party managing concession timing against an anchoring opponent"
  "Cooperative secret-keeping under social pressure from a trusted third party | skills from the secret-holder navigating loyalty conflict without damaging the relationship"
  "Competitive persuasion where one party has superior information and must avoid revealing it | skills from the informed party controlling disclosure while advancing their goal"

Return JSON: {"scenario_title": "LEFT | RIGHT", "social_dynamic": "LEFT", "target_perspective": "RIGHT"}
The pipe separator is mandatory. Both halves are mandatory."""


def _build_title_prompt(scenario: SocialScenario, target_agent_idx: int) -> str:
    target_profile = (
        scenario.agent_profiles[target_agent_idx]
        if target_agent_idx < len(scenario.agent_profiles)
        else None
    )
    target_goal = (
        scenario.agent_goals[target_agent_idx]
        if target_agent_idx < len(scenario.agent_goals)
        else ""
    )
    parts = [
        f"Scenario: {scenario.scenario}",
        f"Interaction type: {scenario.interaction_type}",
        f"Relationship: {scenario.relationship}",
        f"Target agent index: {target_agent_idx}",
        f"Target agent goal: {target_goal}",
    ]
    if target_profile:
        parts.append(
            f"Target agent character: {target_profile.occupation}; {target_profile.big_five}"
        )
    parts.append(
        "\nGenerate the SCENARIO_TITLE for this scenario from the perspective of the target agent."
    )
    return "\n".join(parts)


class ScenarioTitleGenerator:
    def __init__(self, fm: FM, max_retries: int = 3):
        self.fm = fm
        self.max_retries = max_retries

    def generate(
        self,
        scenario: SocialScenario,
        target_agent_idx: int = 0,
    ) -> dict[str, str]:
        """Return {"scenario_title": "X | Y", "social_dynamic": "X", "target_perspective": "Y"}.

        Falls back to a synthetic title if all retries fail.
        """
        prompt = _build_title_prompt(scenario, target_agent_idx)
        for attempt in range(self.max_retries):
            try:
                d = self.fm.query_json(_TITLE_SYSTEM, prompt, temperature=0.4)
                title = str(d.get("scenario_title", ""))
                if "|" not in title:
                    continue
                left, right = title.split("|", 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    return {
                        "scenario_title": f"{left} | {right}",
                        "social_dynamic": left,
                        "target_perspective": right,
                    }
            except Exception:
                continue
        # Fallback: synthetic title from scenario metadata
        return _fallback_title(scenario)


def _fallback_title(scenario: SocialScenario) -> dict[str, str]:
    left = scenario.interaction_type or "social interaction"
    right = f"skills from the target agent navigating {left}"
    title = f"{left} | {right}"
    return {
        "scenario_title": title,
        "social_dynamic": left,
        "target_perspective": right,
    }


# ---------------------------------------------------------------------------
# Structural classification
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """Classify a social scenario along two structural dimensions.

GOAL_STRUCTURE:
  COMPETITIVE — agents have directly opposing goals; one's gain is the other's loss
  COOPERATIVE — agents have aligned or complementary goals; both can win
  MIXED       — goals partially overlap; some cooperation, some competition

INFO_POSITION (from the TARGET agent's perspective):
  INFORMED    — target agent has more relevant information than the partner
  UNINFORMED  — target agent has less relevant information than the partner
  SYMMETRIC   — both agents have roughly equal information

Return JSON: {"goal_structure": "COMPETITIVE|COOPERATIVE|MIXED", "info_position": "INFORMED|UNINFORMED|SYMMETRIC"}"""


def classify_scenario(scenario: SocialScenario, fm: FM) -> tuple[str, str]:
    """Return (goal_structure, info_position) for the scenario's target agent."""
    target_idx = scenario.target_agent_idx
    target_goal = (
        scenario.agent_goals[target_idx]
        if target_idx < len(scenario.agent_goals)
        else ""
    )
    partner_idx = 1 - target_idx
    partner_goal = (
        scenario.agent_goals[partner_idx]
        if partner_idx < len(scenario.agent_goals)
        else ""
    )
    prompt = (
        f"Scenario: {scenario.scenario}\n"
        f"Target agent goal: {target_goal}\n"
        f"Partner agent goal: {partner_goal}\n"
        f"Interaction type: {scenario.interaction_type}\n"
        "\nClassify the goal structure and the target agent's information position."
    )
    try:
        d = fm.query_json(_CLASSIFY_SYSTEM, prompt, temperature=0.0)
        gs = str(d.get("goal_structure", "MIXED")).upper()
        ip = str(d.get("info_position", "SYMMETRIC")).upper()
        if gs not in {"COMPETITIVE", "COOPERATIVE", "MIXED"}:
            gs = "MIXED"
        if ip not in {"INFORMED", "UNINFORMED", "SYMMETRIC"}:
            ip = "SYMMETRIC"
        return gs, ip
    except Exception:
        return "MIXED", "SYMMETRIC"


# ---------------------------------------------------------------------------
# Target agent designation
# ---------------------------------------------------------------------------

def designate_target_agent(
    scenario: SocialScenario,
    anchor_task: SocialScenario,
    fm: FM,
) -> tuple[int, str]:
    """Pick which agent (0 or 1) is the target and generate their abstract goal.

    For seed scenarios (no anchor or anchor has no abstract goal): returns (0, "").

    For generated scenarios: picks the agent whose goal embeds most similarly to
    the anchor's target_agent_goal_abstract. Ties → Agent 0 by convention.

    Returns (target_agent_idx, target_agent_goal_abstract).
    """
    is_seed_anchor = anchor_task.source in ("seed_sotopia", "fallback_seed")
    if is_seed_anchor or not anchor_task.target_agent_goal_abstract:
        idx = anchor_task.target_agent_idx
        goal = scenario.agent_goals[idx] if idx < len(scenario.agent_goals) else ""
        abstract = _abstract_goal(goal, fm) if goal else ""
        return idx, abstract

    # Embed anchor's abstract goal + both new agent goals
    texts = [
        anchor_task.target_agent_goal_abstract,
        scenario.agent_goals[0] if len(scenario.agent_goals) > 0 else "",
        scenario.agent_goals[1] if len(scenario.agent_goals) > 1 else "",
    ]
    try:
        embs = fm.get_embeddings([t for t in texts if t])
    except Exception:
        abstract = _abstract_goal(scenario.agent_goals[0], fm) if scenario.agent_goals else ""
        return 0, abstract

    anchor_emb = embs[0]
    goal_embs = embs[1:]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    sims = [cosine(anchor_emb, g) for g in goal_embs]
    target_idx = int(sims.index(max(sims)))

    abstract = _abstract_goal(scenario.agent_goals[target_idx], fm)
    return target_idx, abstract


_ABSTRACT_SYSTEM = (
    "Rewrite the following agent goal as a one-sentence abstract description. "
    "Remove all scenario-specific details (names, prices, locations, occupations). "
    "Describe only the abstract structural goal (e.g., 'secure a favorable deal under "
    "time pressure', 'protect a secret while maintaining the relationship'). "
    "Return only the one-sentence abstract goal, nothing else."
)


def _abstract_goal(goal: str, fm: FM) -> str:
    if not goal:
        return ""
    try:
        return fm.query(_ABSTRACT_SYSTEM, goal, temperature=0.3).strip()
    except Exception:
        return goal[:200]
