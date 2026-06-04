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
    learner_scores: dict = field(default_factory=dict)      # 7-dim diagnostics (not the gate)
    partner_scores: dict = field(default_factory=dict)
    rubric_results: list[dict] = field(default_factory=list)  # per-check: kind/verdict/confidence/rationale/n_agree/k
    outcome_achieved: bool = False                          # all outcome checks passed
    constraint_preserved: bool = False                      # all constraint checks passed
    goal_achieved: bool = False                             # alias = AND of all rubric checks (the gate)
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


def _evaluate_diagnostics(inbox: list, fm: FM, learner_goal: str = "") -> tuple[dict, dict, str]:
    """Score both agents on the SOTOPIA dimensions — DIAGNOSTICS ONLY.

    These 7-dim scores no longer gate success (the rubric does, see _evaluate_rubric). They
    feed reflection ("which facet was weak") and the external eval. Falls back to zeros.
    """
    history = _build_history(inbox)
    if not history.strip():
        return _zeros(), _zeros(), ""

    agent_instruction = (
        'There are exactly 2 agents. Use "agent_1" for the first agent '
        'to appear in the conversation and "agent_2" for the second. '
        'agent_1 is the learner agent.'
    )
    learner_goal_section = (
        f'\nLEARNER GOAL (agent_1): {learner_goal}\n'
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
        return _zeros(), _zeros(), f"[diagnostics failed: {e}]"

    a1 = data.get("agent_1", {})
    a2 = data.get("agent_2", {})
    learner_scores, _ = _unpack_dimensions(a1)
    partner_scores, _ = _unpack_dimensions(a2)
    reasoning = (
        _reasoning_text("learner (agent_1)", a1)
        + "\n\n"
        + _reasoning_text("partner (agent_2)", a2)
    )
    return learner_scores, partner_scores, reasoning.strip()


# ---------------------------------------------------------------------------
# Rubric evaluation — the success gate (per-check, perspective-routed)
# ---------------------------------------------------------------------------

_RUBRIC_NEUTRAL_SYSTEM = (
    "You are a neutral observer judging whether a specific thing happened in a conversation. "
    "Decide ONLY from what is observable in the transcript; do not speculate about private "
    "feelings. Respond with ONLY valid JSON."
)

_RUBRIC_PARTNER_SYSTEM = (
    "You ARE the second person in this conversation. Using your PRIVATE background and how the "
    "conversation actually landed for you, answer honestly from your own point of view — be "
    "candid, not polite or agreeable, and think about whether you would actually follow through. "
    "Respond with ONLY valid JSON."
)

_RUBRIC_USER = (
    "{context}\n\n"
    "QUESTION (a YES/true answer means the OTHER person succeeded): {question}\n"
    'Respond with ONLY this JSON: {{"verdict": true, "confidence": 0.8, "rationale": "one sentence"}}'
)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _judge_check(fm: FM, system: str, user: str, k: int) -> dict:
    """Run a single rubric check k times; majority vote. k=1 → single deterministic pass."""
    samples = []
    for _i in range(max(1, k)):
        temp = 0.0 if k <= 1 else 0.7
        try:
            d = fm.query_json(system, user, temperature=temp)
            samples.append(
                (bool(d.get("verdict", False)), str(d.get("rationale", "")), _safe_float(d.get("confidence")))
            )
        except Exception:
            continue
    if not samples:
        return {"verdict": False, "confidence": 0.0, "rationale": "[judge failed]", "n_agree": 0, "k": k}
    trues = sum(1 for v, _, _ in samples if v)
    verdict = trues * 2 >= len(samples)  # majority; ties resolve to True
    n_agree = sum(1 for v, _, _ in samples if v == verdict)
    rationale = next((r for v, r, _ in samples if v == verdict and r), samples[0][1])
    conf = round(sum(c for _, _, c in samples) / len(samples), 2)
    return {"verdict": verdict, "confidence": conf, "rationale": rationale, "n_agree": n_agree, "k": len(samples)}


def _evaluate_rubric(inbox: list, rubric, partner_profile, fm: FM, k: int = 3) -> list[dict]:
    """Evaluate each rubric check with the judge matching its perspective.

    neutral → single transcript-only pass. partner → k-sample self-consistency majority vote,
    conditioned on the partner's private profile + secret. Returns a list of per-check dicts.
    """
    history = _build_history(inbox)
    if not history.strip() or rubric is None or not getattr(rubric, "checks", None):
        return []

    partner_bg = ""
    partner_name = "you"
    if partner_profile is not None:
        partner_name = getattr(partner_profile, "first_name", "you") or "you"
        partner_bg = (getattr(partner_profile, "public_info", "") or "").strip()
        secret = (getattr(partner_profile, "secret", "") or "").strip()
        if secret:
            partner_bg += f"\nYour secret: {secret}"

    out: list[dict] = []
    for c in rubric.checks:
        if c.perspective == "partner":
            context = (
                f"YOUR PRIVATE BACKGROUND:\n{partner_bg}\n\n"
                f"THE CONVERSATION (you are {partner_name}):\n{history}"
            )
            r = _judge_check(fm, _RUBRIC_PARTNER_SYSTEM,
                             _RUBRIC_USER.format(context=context, question=c.question), k)
        else:
            context = f"TRANSCRIPT:\n{history}"
            r = _judge_check(fm, _RUBRIC_NEUTRAL_SYSTEM,
                             _RUBRIC_USER.format(context=context, question=c.question), 1)
        r["kind"] = c.kind
        r["question"] = c.question
        r["perspective"] = c.perspective
        out.append(r)
    return out


def _rollup(rubric_results: list[dict], kind: str) -> bool:
    """True iff there is ≥1 check of `kind` and all of them passed."""
    rs = [r for r in rubric_results if r.get("kind") == kind]
    return bool(rs) and all(r.get("verdict") for r in rs)


async def run_single_episode(
    env_profile,
    agent_profiles: list,
    fm: FM,
    learner_model: str,
    partner_model: str,
    memory_prompt: str = "",
    max_turns: int = 20,
    learner_goal: str = "",
    rubric=None,                       # SuccessRubric for the designated learner (the success gate)
    partner_profile=None,              # AgentProfile of the partner (for the partner-perspective judge)
    judge_self_consistency_k: int = 3,
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

    # Diagnostics (7-dim, not the gate) + the rubric gate (per-check, perspective-routed).
    learner_scores, partner_scores, reasoning = _evaluate_diagnostics(env.inbox, fm, learner_goal=learner_goal)
    rubric_results = _evaluate_rubric(env.inbox, rubric, partner_profile, fm, k=judge_self_consistency_k)
    outcome_achieved = _rollup(rubric_results, "outcome")
    constraint_preserved = _rollup(rubric_results, "constraint")
    goal_achieved = bool(rubric_results) and all(r.get("verdict") for r in rubric_results)

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
        rubric_results=rubric_results,
        outcome_achieved=outcome_achieved,
        constraint_preserved=constraint_preserved,
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
