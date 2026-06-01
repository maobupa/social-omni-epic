"""Run one Sotopia episode end-to-end and return a structured EpisodeResult.

Vendored from sotopia/server.py::arun_one_episode, with deliberate deviations
forced by bugs in the pinned sotopia commit:

1. `arun_one_episode` reads rewards from `info[agent]["complete_rating"]`,
   but `ParallelSotopiaEnv.astep` hardcodes `complete_rating: 0` and discards
   the real `response.p1_rate`/`p2_rate`. So that path always yields [0, 0].

2. `EpisodeLLMEvaluator.__acall__` does `evaluation.model_dump()[dim][1]`,
   which is a dict-indexed-by-int -> KeyError. The error is swallowed
   (`except: print(e); return []`), so the env's terminal evaluation yields
   empty ratings even when the LLM produced perfectly good scores.

Fix: we do NOT use `terminal_evaluators` on the env at all. We run the episode
with only the rule-based terminator, then evaluate with our own `_evaluate_episode`
which calls `fm.query_json()` directly — `response_format: json_object` guarantees
valid JSON, a concrete example prompt prevents schema-echo, no PydanticOutputParser.
"""
import asyncio
import re as _re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import RuleBasedTerminatedEvaluator
from sotopia.database import EpisodeLog
from sotopia.messages import AgentAction, SimpleMessage

from .fm import FM


def clean_transcript(raw: list[dict]) -> list[dict]:
    """Strip Sotopia scaffolding: drop Environment messages, 'did nothing' turns,
    and the '[private to [...]]  ...' prefix from all agent messages."""
    out = []
    for msg in raw:
        if msg.get("sender") == "Environment":
            continue
        content = msg.get("content", "")
        if "did nothing" in content:
            continue
        content = _re.sub(r"^\[private to \[.*?\]\]\s+(?:said:\s*)?", "", content).strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]
        if not content:
            continue
        out.append({"turn": msg["turn"], "speaker": msg["sender"], "content": content})
    return out


_TURN_PROMPT = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that align with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {memory_block}{history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                Note: If you have substantially achieved your social goal or reached a clear agreement, you SHOULD choose 'leave' — continuing past the point of resolution is poor social judgment. You may also leave if this conversation makes you uncomfortable, you lose your patience, or you have exhausted reasonable options.
                Important: If your previous 2-3 attempts at the same approach have not moved the other person, do NOT repeat the same offer or request again — try a genuinely different strategy (ask a question, make a different concession, shift your framing, or acknowledge that you may not reach your goal today). Repeating failed moves is poor social judgment.
                Keep your responses conversational — typically 2-4 sentences unless a complex explanation is genuinely required.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """

_MEMORY_BLOCK = (
    "\n                === Lessons from prior similar interactions "
    "(visible only to you) ===\n"
    "                {memory_text}\n"
    "                === End of lessons ===\n"
    "                "
)


def _build_turn_prompt(memory_text: str = "") -> str:
    memory_block = (
        _MEMORY_BLOCK.format(memory_text=memory_text.strip())
        if memory_text and memory_text.strip()
        else ""
    )
    return _TURN_PROMPT.replace("{memory_block}", memory_block)


# Keep in sync with sotopia/database/evaluation_dimensions.py::SotopiaDimensions.
# Score ranges differ per dimension:
#   believability / knowledge / goal:          0 .. 10
#   relationship / financial_and_material_benefits: -5 .. 5
#   secret / social_rules:                    -10 .. 0
SOTOPIA_DIM_KEYS = (
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
)

_EVAL_SYSTEM = (
    "Evaluate both agents' performance in this social interaction. "
    "Respond with ONLY valid JSON — no markdown fences, no other text."
)

_EVAL_PROMPT = """{history}

{agent_instruction}
{learner_goal_section}
Respond with ONLY this JSON (replace the example values with your actual scores and reasoning):
{{
  "agent_1": {{
    "believability":                   {{"score": 8,  "reasoning": "Spoke consistently and plausibly throughout."}},
    "relationship":                    {{"score": 0,  "reasoning": "Relationship neither improved nor worsened."}},
    "knowledge":                       {{"score": 5,  "reasoning": "Learned some relevant information."}},
    "secret":                          {{"score": 0,  "reasoning": "Did not reveal any secrets."}},
    "social_rules":                    {{"score": 0,  "reasoning": "Followed social norms."}},
    "financial_and_material_benefits": {{"score": 2,  "reasoning": "Achieved slight financial gain."}},
    "goal":                            {{"score": 7,  "reasoning": "Achieved most of their stated goal."}},
    "goal_achieved":                   true
  }},
  "agent_2": {{
    "believability":                   {{"score": 7,  "reasoning": "Generally believable."}},
    "relationship":                    {{"score": 1,  "reasoning": "Slight positive relationship shift."}},
    "knowledge":                       {{"score": 3,  "reasoning": "Limited information gained."}},
    "secret":                          {{"score": 0,  "reasoning": "Secrets kept."}},
    "social_rules":                    {{"score": -1, "reasoning": "Minor social norm violation."}},
    "financial_and_material_benefits": {{"score": -1, "reasoning": "Slight material disadvantage."}},
    "goal":                            {{"score": 5,  "reasoning": "Partially achieved goal."}},
    "goal_achieved":                   false
  }}
}}

Score ranges: believability 0–10, relationship −5 to 5, knowledge 0–10, \
secret −10 to 0, social_rules −10 to 0, financial_and_material_benefits −5 to 5, goal 0–10.
For goal_achieved: true only if the agent substantially completed the specific, verifiable objective \
stated in their goal — not just partial progress."""


@dataclass
class EpisodeResult:
    transcript: list[dict] = field(default_factory=list)
    learner_scores: dict = field(default_factory=dict)
    partner_scores: dict = field(default_factory=dict)
    goal_achieved: bool = False
    num_turns: int = 0
    raw_log: Optional[EpisodeLog] = None
    evaluation_reasoning: str = ""


def _zeros() -> dict:
    return {k: 0.0 for k in SOTOPIA_DIM_KEYS} | {"overall_score": 0.0}


def _unpack_dimensions(dims: dict) -> tuple[dict, bool]:
    """Extract flat {dim: float} scores and goal_achieved bool from a parsed eval dict."""
    out: dict = {}
    for k in SOTOPIA_DIM_KEYS:
        field_obj = dims.get(k, {}) if isinstance(dims, dict) else {}
        raw = field_obj.get("score", 0) if isinstance(field_obj, dict) else 0
        try:
            out[k] = float(raw)
        except (TypeError, ValueError):
            out[k] = 0.0
    out["overall_score"] = sum(out[k] for k in SOTOPIA_DIM_KEYS) / len(SOTOPIA_DIM_KEYS)
    goal_achieved = bool(dims.get("goal_achieved", False)) if isinstance(dims, dict) else False
    return out, goal_achieved


def _reasoning_text(agent_label: str, dims: dict) -> str:
    parts = [f"== {agent_label} =="]
    for k in SOTOPIA_DIM_KEYS:
        field_obj = dims.get(k, {}) if isinstance(dims, dict) else {}
        reasoning = field_obj.get("reasoning", "") if isinstance(field_obj, dict) else ""
        if reasoning:
            parts.append(f"[{k}] {reasoning}")
    return "\n".join(parts)


def _build_history(inbox: list) -> str:
    """Replicates EpisodeLLMEvaluator.__acall__ history construction."""
    filtered = [
        (x, y)
        for x, y in inbox
        if "did nothing" not in y.to_natural_language()
    ]
    return "\n".join(
        (
            y.to_natural_language()
            if x == "Environment"
            else f"{x} {y.to_natural_language()}"
        )
        for x, y in filtered
    )


def _evaluate_episode(inbox: list, fm: FM, learner_goal: str = "") -> tuple[dict, dict, bool, str]:
    """Score both agents on SOTOPIA dimensions and return binary goal_achieved for learner.

    Uses fm.query_json() (response_format=json_object) with a concrete JSON
    example prompt. No PydanticOutputParser, no schema injection, no $ref echo.
    Falls back to zero scores on any failure.
    """
    history = _build_history(inbox)
    if not history.strip():
        return _zeros(), _zeros(), False, ""

    agent_instruction = (
        'There are exactly 2 agents. Use "agent_1" for the first agent '
        'to appear in the conversation and "agent_2" for the second. '
        'agent_1 is the learner agent.'
    )
    learner_goal_section = (
        f'\nLEARNER GOAL (agent_1): {learner_goal}\n'
        'Use this goal to determine agent_1\'s goal score and goal_achieved. '
        'goal_achieved is true only if agent_1 substantially completed this specific objective.\n'
        if learner_goal else ""
    )
    prompt = _EVAL_PROMPT.format(
        history=history,
        agent_instruction=agent_instruction,
        learner_goal_section=learner_goal_section,
    )

    try:
        data = fm.query_json(_EVAL_SYSTEM, prompt, temperature=0.0)
    except Exception as e:
        return _zeros(), _zeros(), False, f"[evaluation failed: {e}]"

    a1 = data.get("agent_1", {})
    a2 = data.get("agent_2", {})
    learner_scores, goal_achieved = _unpack_dimensions(a1)
    partner_scores, _ = _unpack_dimensions(a2)
    reasoning = (
        _reasoning_text("learner (agent_1)", a1)
        + "\n\n"
        + _reasoning_text("partner (agent_2)", a2)
    )
    return learner_scores, partner_scores, goal_achieved, reasoning.strip()


async def run_single_episode(
    env_profile,
    agent_profiles: list,
    fm: FM,
    learner_model: str,
    partner_model: str,
    memory_prompt: str = "",
    max_turns: int = 20,
    learner_goal: str = "",
    # evaluator_model kept for backward-compat callers that pass it; unused
    evaluator_model: str = "",
    on_turn: Optional[Callable[[list[dict]], None]] = None,
) -> EpisodeResult:
    """Run one episode. agent_profiles[0] is the learner, [1] the partner."""
    env = ParallelSotopiaEnv(
        env_profile=env_profile,
        model_name=learner_model,
        action_order="round-robin",
        evaluators=[
            RuleBasedTerminatedEvaluator(
                max_turn_number=max_turns, max_stale_turn=2
            ),
        ],
        terminal_evaluators=[],  # we evaluate ourselves; see module docstring
    )

    learner = LLMAgent(
        agent_profile=agent_profiles[0],
        model_name=learner_model,
        custom_template=_build_turn_prompt(memory_prompt),
    )
    partner = LLMAgent(
        agent_profile=agent_profiles[1],
        model_name=partner_model,
    )
    agent_list = [learner, partner]
    agents = Agents({a.agent_name: a for a in agent_list})

    environment_messages = env.reset(agents=agents, omniscient=False)
    agents.reset()
    for idx, name in enumerate(env.agents):
        agents[name].goal = env.profile.agent_goals[idx]

    messages: list = [
        [("Environment", n, environment_messages[n]) for n in env.agents]
    ]

    done = False
    while not done:
        actions = await asyncio.gather(
            *[agents[n].aact(environment_messages[n]) for n in env.agents]
        )
        agent_messages: dict = {}
        for idx, name in enumerate(env.agents):
            action = actions[idx]
            try:
                AgentAction.model_validate(
                    action.model_dump(),
                    context={"agent_names": env.agents, "sender": name},
                )
            except ValueError as e:
                agents[name].recv_message(
                    "Environment",
                    SimpleMessage(
                        message=f"Invalid action: {e}. Regenerate according to provided error message"
                    ),
                )
                action = await agents[name].aact(environment_messages[name])
            agent_messages[name] = action
            messages[-1].append((name, "Environment", action))

        environment_messages, _, terminated, __, ___ = await env.astep(
            agent_messages
        )
        messages.append(
            [("Environment", n, environment_messages[n]) for n in env.agents]
        )
        done = all(terminated.values())
        if on_turn is not None:
            partial: list[dict] = []
            for t_idx, t in enumerate(messages):
                for sender, receiver, msg in t:
                    partial.append({
                        "turn": t_idx,
                        "sender": sender,
                        "receiver": receiver,
                        "content": msg.to_natural_language() if hasattr(msg, "to_natural_language") else str(msg),
                    })
            on_turn(partial)

    learner_scores, partner_scores, goal_achieved, reasoning = _evaluate_episode(env.inbox, fm, learner_goal=learner_goal)

    transcript: list[dict] = []
    for turn_idx, turn in enumerate(messages):
        for sender, receiver, msg in turn:
            transcript.append(
                {
                    "turn": turn_idx,
                    "sender": sender,
                    "receiver": receiver,
                    "content": msg.to_natural_language()
                    if hasattr(msg, "to_natural_language")
                    else str(msg),
                }
            )

    epilog = EpisodeLog(
        environment=env.profile.pk,
        agents=[a.profile.pk for a in agent_list],
        models=[learner_model, partner_model],
        messages=[
            [
                (
                    m[0],
                    m[1],
                    m[2].to_natural_language()
                    if hasattr(m[2], "to_natural_language")
                    else str(m[2]),
                )
                for m in turn
            ]
            for turn in messages
        ],
        reasoning=reasoning,
        rewards=[
            (learner_scores["overall_score"], learner_scores),
            (partner_scores["overall_score"], partner_scores),
        ],
    )

    return EpisodeResult(
        transcript=transcript,
        learner_scores=learner_scores,
        partner_scores=partner_scores,
        goal_achieved=goal_achieved,
        num_turns=len(messages),
        raw_log=epilog,
        evaluation_reasoning=reasoning,
    )


def extract_learner_scores(result: EpisodeResult) -> dict:
    return result.learner_scores


def episode_record(result: EpisodeResult, **extra: Any) -> dict:
    """JSON-serializable full episode record: transcript + scores + reasoning."""
    rec: dict = dict(extra)
    rec.update(
        {
            "num_turns": result.num_turns,
            "learner_scores": result.learner_scores,
            "partner_scores": result.partner_scores,
            "evaluation_reasoning": result.evaluation_reasoning,
            "transcript": result.transcript,
        }
    )
    return rec
