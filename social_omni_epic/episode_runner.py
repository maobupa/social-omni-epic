"""Run one Sotopia episode end-to-end and return a structured EpisodeResult.

Vendored from sotopia/server.py::arun_one_episode, with two deliberate
deviations forced by bugs in the pinned sotopia commit:

1. `arun_one_episode` reads rewards from `info[agent]["complete_rating"]`,
   but `ParallelSotopiaEnv.astep` hardcodes `complete_rating: 0` and discards
   the real `response.p1_rate`/`p2_rate`. So that path always yields [0, 0].

2. `EpisodeLLMEvaluator.__acall__` does `evaluation.model_dump()[dim][1]`,
   which is a dict-indexed-by-int -> KeyError. The error is swallowed
   (`except: print(e); return []`), so the env's terminal evaluation yields
   empty ratings even when the LLM produced perfectly good scores.

Fix: we do NOT use `terminal_evaluators` on the env at all. We run the
episode with only the rule-based terminator, then evaluate ourselves by
calling `agenerate` directly with `EvaluationForAgents[SotopiaDimensions]`
as the output schema (this is exactly the LLM call Sotopia makes — same
rubric, same schema — we just skip Sotopia's broken tuple-reduction code
and read `.score` straight off the dimension objects).
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

from litellm.utils import supports_response_schema

from sotopia.agents import LLMAgent
from sotopia.agents.llm_agent import Agents
from sotopia.envs.parallel import ParallelSotopiaEnv
from sotopia.envs.evaluators import EvaluationForAgents, RuleBasedTerminatedEvaluator
from sotopia.database import EpisodeLog
from sotopia.database.evaluation_dimensions import SotopiaDimensions
from sotopia.generation_utils import PydanticOutputParser, agenerate
from sotopia.messages import AgentAction, SimpleMessage

from .memory_agent import build_custom_template_with_memory


# Keep in sync with sotopia/database/evaluation_dimensions.py::SotopiaDimensions.
# Score ranges differ per dimension (see SotopiaDimensions field docs):
#   believability/knowledge/goal: 0..10
#   relationship/financial_and_material_benefits: -5..5
#   secret/social_rules: -10..0
SOTOPIA_DIM_KEYS = (
    "believability",
    "relationship",
    "knowledge",
    "secret",
    "social_rules",
    "financial_and_material_benefits",
    "goal",
)

_EVAL_TEMPLATE = """{history}
Based on previous interactions, evaluate how well participants achieve their goals.
{agent_instruction}
Please follow the format:
{format_instructions}
"""


@dataclass
class EpisodeResult:
    transcript: list[dict] = field(default_factory=list)
    learner_scores: dict = field(default_factory=dict)
    partner_scores: dict = field(default_factory=dict)
    num_turns: int = 0
    raw_log: Optional[EpisodeLog] = None
    evaluation_reasoning: str = ""


def _zeros() -> dict:
    return {k: 0.0 for k in SOTOPIA_DIM_KEYS} | {"overall_score": 0.0}


def _unpack_dimensions(dims: Any) -> dict:
    """Turn a SotopiaDimensions instance into a flat {dim: float} dict.
    Each dimension field is a ReasoningScore object with .reasoning and .score."""
    out: dict = {}
    for k in SOTOPIA_DIM_KEYS:
        field_obj = getattr(dims, k, None)
        score = getattr(field_obj, "score", 0) if field_obj is not None else 0
        try:
            out[k] = float(score)
        except (TypeError, ValueError):
            out[k] = 0.0
    out["overall_score"] = sum(out[k] for k in SOTOPIA_DIM_KEYS) / len(SOTOPIA_DIM_KEYS)
    return out


def _reasoning_text(agent_label: str, dims: Any) -> str:
    parts = [f"== {agent_label} =="]
    for k in SOTOPIA_DIM_KEYS:
        field_obj = getattr(dims, k, None)
        if field_obj is not None:
            parts.append(f"[{k}] {getattr(field_obj, 'reasoning', '')}")
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


async def _evaluate_episode(
    inbox: list, evaluator_model: str
) -> tuple[dict, dict, str]:
    """Run the Sotopia 7-dimension evaluation ourselves. Returns
    (learner_scores, partner_scores, reasoning_text)."""
    history = _build_history(inbox)
    if not history.strip():
        return _zeros(), _zeros(), ""

    agent_instruction = (
        'There are exactly 2 agents. Under the "evaluations" field, use '
        'exactly these keys: ["agent_1", "agent_2"] (no other keys). '
        "agent_1 is the first agent to appear, agent_2 the second.\n"
    )
    use_structured = evaluator_model.startswith(
        "custom/structured"
    ) or supports_response_schema(model=evaluator_model)

    response: EvaluationForAgents[SotopiaDimensions] = await agenerate(
        model_name=evaluator_model,
        template=_EVAL_TEMPLATE,
        input_values=dict(history=history, agent_instruction=agent_instruction),
        output_parser=PydanticOutputParser(
            pydantic_object=EvaluationForAgents[SotopiaDimensions]
        ),
        temperature=0.0,
        structured_output=use_structured,
    )

    evals = list(response.evaluations.values())
    learner_dims = evals[0] if len(evals) > 0 else None
    partner_dims = evals[1] if len(evals) > 1 else None

    learner_scores = _unpack_dimensions(learner_dims) if learner_dims else _zeros()
    partner_scores = _unpack_dimensions(partner_dims) if partner_dims else _zeros()

    reasoning = ""
    if learner_dims is not None:
        reasoning += _reasoning_text("learner (agent_1)", learner_dims) + "\n\n"
    if partner_dims is not None:
        reasoning += _reasoning_text("partner (agent_2)", partner_dims)

    return learner_scores, partner_scores, reasoning.strip()


async def run_single_episode(
    env_profile,
    agent_profiles: list,
    learner_model: str,
    partner_model: str,
    evaluator_model: str,
    memory_prompt: str = "",
    max_turns: int = 20,
) -> EpisodeResult:
    """Run one episode. agent_profiles[0] is the learner, [1] the partner."""
    env = ParallelSotopiaEnv(
        env_profile=env_profile,
        model_name=evaluator_model,
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
        custom_template=build_custom_template_with_memory(memory_prompt),
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

    learner_scores, partner_scores, reasoning = await _evaluate_episode(
        env.inbox, evaluator_model
    )

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
        models=[evaluator_model, learner_model, partner_model],
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
        num_turns=len(messages),
        raw_log=epilog,
        evaluation_reasoning=reasoning,
    )


def extract_learner_scores(result: EpisodeResult) -> dict:
    return result.learner_scores


def episode_record(result: EpisodeResult, **extra: Any) -> dict:
    """JSON-serializable full episode record: transcript + scores + reasoning.
    `extra` lets callers attach context (iteration, scenario_id, ...)."""
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
