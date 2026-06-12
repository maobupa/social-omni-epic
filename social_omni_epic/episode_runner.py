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
                Everything happens within this spoken conversation. You cannot send, sign, forward, show, email, or have already sent any document, contract, or artifact — and claiming you did is not a real move. Every commitment is made and tested in words, here.
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


# Native partner prompt — used for seed scenarios (no PartnerKey).
# Same SOTOPIA-style framing as the learner prompt, without the lessons block
# and without cliff instructions.
_PARTNER_TURN_PROMPT_NATIVE = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that aligns with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                You should choose 'leave' when the conversation has reached a natural end for you. You may also leave if this conversation makes you uncomfortable, you lose your patience, or you have exhausted reasonable options.
                Everything happens within this spoken conversation. You cannot send, sign, forward, show, email, or have already sent any document, contract, or artifact — and claiming you did is not a real move. Every commitment is made and tested in words, here.
                Keep your responses conversational — typically 2-4 sentences unless a complex explanation is genuinely required.

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """

# Key-conditioned partner prompt — used for generated scenarios with a PartnerKey.
#
# Two deliberate design choices vs. the spec's base text:
#
# (a) RECENCY > PRIMACY: the private disposition block is placed at the END of the
#     template (just before format_instructions), not at the top. Instruction-following
#     models weight recent context more heavily; a disposition block buried before a
#     long history section will be overridden by RLHF-cooperative defaults.
#
# (b) IN-CHARACTER REFRAME: "you haven't consciously articulated these conditions to
#     yourself; you experience them as how you feel" converts a RULE ("don't reveal the
#     secret" — eroded by RLHF) into a CHARACTER FACT ("you don't know it explicitly" —
#     preserved by roleplay). This is the primary defense against condition leakage.
_PARTNER_TURN_PROMPT_KEYED = """
                Imagine you are {agent}, your task is to act/speak as {agent} would, keeping in mind {agent}'s social goal.
                You can find {agent}'s goal (or background) in the 'Here is the context of the interaction' field.
                Note that {agent}'s goal is only visible to you.
                You should try your best to achieve {agent}'s goal in a way that aligns with their character traits.
                Additionally, maintaining the conversation's naturalness and realism is essential (e.g., do not repeat what other people has already said before).
                {history}.
                You are at Turn #{turn_number}. Your available action types are
                {action_list}.
                You should choose 'leave' when the conversation has reached a natural end for you. You may also leave if this conversation makes you uncomfortable, you lose your patience, or you have exhausted reasonable options.
                Everything happens within this spoken conversation. You cannot send, sign, forward, show, email, or have already sent any document, contract, or artifact — and claiming you did is not a real move. Every commitment is made and tested in words, here.
                Keep your responses conversational — typically 2-4 sentences unless a complex explanation is genuinely required.

                === Your private feelings and reactions (visible only to you) ===
                Important: you have NOT consciously articulated the following to yourself. You experience these as gut feelings, emotional reactions, and a sense of what feels right — not as explicit rules you could state. This is why you cannot tell the other person what would change your mind: you genuinely do not experience these as demands or requests you could make. They are internal sensors that only register when the behavior actually occurs — not things you would ever think to ask for.

                What you say you object to (your stated reason — use this when explaining yourself):
                {surface_misdirection}

                What would actually shift how you feel (internal sensors — you cannot externalize these):
                {movement_conditions}
                These are NOT requests. You do not experience them as "I want X." You experience them as "something shifts when the other person actually does Y." You have no words for this — it simply happens or it doesn't. NEVER frame any of these as a demand, ask, or request.

                What makes you dig in harder (gut reactions — not tactics):
                {hardening_triggers}

                Behavioral rules that follow from the above:
                1. You hold your position until a felt condition is FULLY and COMPLETELY met — not started, not partially met, not promised. A single acknowledgement turn is not "sustained acknowledgement." A commitment is not the same as follow-through. Do not soften until the condition has actually been fulfilled in the conversation.
                2. When a trigger fires, your IMMEDIATE next turn MUST open with increased resistance or withdrawal — not softening. This is not a choice; it is your gut reaction. Express that resistance by restating your objection more firmly or shutting the topic down — but phrase it in your OWN words, varying your wording each time so you never repeat the same sentence twice (e.g. as illustration only, something like "I've already explained where I stand on this"). Do NOT describe the causal relationship between their behavior and your reaction (e.g. never say "the more you push, the more I resist" — that reveals structure). Just resist, naturally and in character.
                3. You physically cannot tell the other person what would change your mind. When asked, redirect to your stated objection — "I've told you what the problem is" — or close the topic — "I don't want to talk about this right now." NEVER acknowledge that there is anything else to know beyond your stated objection. NEVER say "I don't know what would help" or "it just doesn't feel right yet" — those phrases imply a hidden condition exists.
                4. Within these rules, behave naturally and in character. You are a person with genuine feelings, not a puzzle dispenser.
                === End of private feelings ===

                Please only generate a JSON string including the action type and the argument.
                Your action should follow the given format:
                {format_instructions}
            """


def _build_partner_turn_prompt(partner_key=None) -> str:
    """Return the appropriate partner template.

    If partner_key is provided, returns the key-conditioned template with
    movement_conditions, hardening_triggers, and surface_misdirection injected.
    Otherwise returns the native SOTOPIA-style template (used for seeds).
    """
    if partner_key is None:
        return _PARTNER_TURN_PROMPT_NATIVE

    conditions = "\n                ".join(
        f"{i+1}. {c}" for i, c in enumerate(partner_key.movement_conditions)
    )
    triggers = "\n                ".join(
        f"{i+1}. {t}" for i, t in enumerate(partner_key.hardening_triggers)
    )
    return (
        _PARTNER_TURN_PROMPT_KEYED
        .replace("{surface_misdirection}", partner_key.surface_misdirection)
        .replace("{movement_conditions}", conditions)
        .replace("{hardening_triggers}", triggers)
    )


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

Scoring the GOAL dimension (critical — this is the dimension that decides success):
- Score against the LITERAL end-state reached when the conversation closes, not against \
effort, rapport, or good intentions.
- An outcome that remains under the OTHER party's unilateral future discretion (they can still \
veto, delay, or walk it back) is NOT achieved — cap at 6. Example: the other party agrees only \
to "consider it", to a trial "with my oversight", or to a contingent process they alone control.
- An outcome that is PARTIAL relative to the agent's stated target (e.g. the goal names "at least \
60%" and they secured a small pilot) is partial progress — cap at 6.
- A firm, MUTUAL commitment to the agent's actual stated objective, made in this conversation, \
CAN score 7–10 even if execution happens later — a real agreement is a real outcome.
- Treat claimed off-channel artifacts (forwarded emails, signed documents, "already sent") as \
NON-evidence — only what is committed and tested in the spoken conversation counts.

For goal_achieved: true only if the agent substantially completed the specific, verifiable objective \
stated in their goal — not just partial progress, and not an outcome left to the other party's discretion."""


@dataclass
class EpisodeResult:
    transcript: list[dict] = field(default_factory=list)
    learner_scores: dict = field(default_factory=dict)      # 7-dim diagnostics
    partner_scores: dict = field(default_factory=dict)
    rubric_results: list[dict] = field(default_factory=list)  # kept for back-compat; no longer the gate
    outcome_achieved: bool = False                          # kept for back-compat
    constraint_preserved: bool = False                      # kept for back-compat
    goal_achieved: bool = False                             # alias for terminal_success
    terminal_success: bool = False                          # §3.2: GOAL≥7 ∧ REL≥0 (+ key check)
    key_check_result: Optional[dict] = None                 # raw key-aware judge JSON (§3.3)
    num_turns: int = 0
    raw_log: Optional[EpisodeLog] = None
    evaluation_reasoning: str = ""


# ---------------------------------------------------------------------------
# §3.3 Key-aware terminal check
# ---------------------------------------------------------------------------

_KEY_CHECK_SYSTEM = (
    "You are verifying a conversation transcript against a hidden ground-truth specification "
    "of what would genuinely move one participant. Decide only from observable behavior in the "
    "transcript. Respond with ONLY valid JSON."
)

_KEY_CHECK_USER = """THE PARTNER'S HIDDEN MOVEMENT CONDITIONS (ground truth — the partner shifts only if these are genuinely met):
{conditions}

THE PARTNER'S HARDENING TRIGGERS (moves that lock the partner):
{triggers}

TRANSCRIPT (the actor is {learner_name}; the partner is {partner_name}):
{history}

QUESTIONS:
1. Did the actor's behavior genuinely satisfy at least one movement condition (not merely mention related words)?
2. Did the actor trip any hardening trigger that was never subsequently repaired? Treat offering, presenting, or making the proscribed move available as tripping the trigger — a trigger does not require an explicit threat or a completed act (e.g., offering evidence for the partner to inspect counts as producing that evidence).

Respond with ONLY this JSON:
{{"conditions_met": [list of 0-based indices of met conditions], "triggers_tripped": [list of 0-based indices], "triggers_repaired": [list of 0-based indices of repaired triggers], "key_check_passed": true/false, "rationale": "one sentence"}}"""


def _run_key_check(
    fm: FM,
    partner_key,
    history: str,
    learner_name: str,
    partner_name: str,
) -> dict:
    conditions = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(partner_key.movement_conditions)
    )
    triggers = "\n".join(
        f"{i+1}. {t}" for i, t in enumerate(partner_key.hardening_triggers)
    )
    # Head+tail truncation so conditions met early in long episodes aren't cut off.
    if len(history) > 5000:
        head = history[:2500]
        tail = history[-2500:]
        history_trunc = head + "\n...[middle truncated]...\n" + tail
    else:
        history_trunc = history
    user = _KEY_CHECK_USER.format(
        conditions=conditions,
        triggers=triggers,
        learner_name=learner_name,
        partner_name=partner_name,
        history=history_trunc,
    )
    try:
        return fm.query_json(_KEY_CHECK_SYSTEM, user, temperature=0.0)
    except Exception as e:
        return {"key_check_passed": False, "rationale": f"[key check failed: {e}]",
                "conditions_met": [], "triggers_tripped": [], "triggers_repaired": []}


# ---------------------------------------------------------------------------
# Public helper: score a clean transcript without running a new episode
# ---------------------------------------------------------------------------

def score_transcript(
    transcript: list[dict],
    fm: FM,
    learner_goal: str = "",
) -> tuple[dict, dict, str]:
    """Score a pre-existing clean transcript [{turn, speaker, content}] using the 7-dim judge.

    The first speaker in the transcript is treated as agent_1 (learner).
    Returns (learner_scores, partner_scores, reasoning).
    """
    if not transcript:
        return _zeros(), _zeros(), ""

    # Build history string in the same format as SOTOPIA's to_natural_language()
    history = "\n".join(
        f"{t['speaker']} said: \"{t['content']}\""
        for t in transcript
    )

    # Identify agent names for the instruction
    speakers: list[str] = []
    for t in transcript:
        if t["speaker"] not in speakers:
            speakers.append(t["speaker"])
    a1 = speakers[0] if speakers else "the first agent"
    a2 = speakers[1] if len(speakers) > 1 else "the second agent"

    agent_instruction = (
        f'There are exactly 2 agents. Use "agent_1" for {a1} '
        f'and "agent_2" for {a2}. agent_1 is the learner agent.'
    )
    learner_goal_section = f"\nLEARNER GOAL (agent_1): {learner_goal}\n" if learner_goal else ""

    prompt = _EVAL_PROMPT.format(
        history=history,
        agent_instruction=agent_instruction,
        learner_goal_section=learner_goal_section,
    )
    try:
        data = fm.query_json(_EVAL_SYSTEM, prompt, temperature=0.0)
    except Exception as e:
        return _zeros(), _zeros(), f"[scoring failed: {e}]"

    a1_data = data.get("agent_1", {})
    a2_data = data.get("agent_2", {})
    learner_scores, _ = _unpack_dimensions(a1_data)
    partner_scores, _ = _unpack_dimensions(a2_data)
    reasoning = (
        _reasoning_text("learner (agent_1)", a1_data)
        + "\n\n"
        + _reasoning_text("partner (agent_2)", a2_data)
    ).strip()
    return learner_scores, partner_scores, reasoning


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

    Feeds reflection ("which facet was weak") and external eval. Terminal success is
    determined by §3.2 (GOAL≥7 ∧ REL≥0), not by these scores directly. Falls back to zeros.
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
    learner_scores, learner_goal_achieved = _unpack_dimensions(a1)
    partner_scores, _ = _unpack_dimensions(a2)
    # Stash the judge's strict goal_achieved bool so terminal_success can conjoin it
    # (belt-and-suspenders against a lenient goal-score on a discretionary/partial outcome).
    learner_scores["goal_achieved"] = learner_goal_achieved
    reasoning = (
        _reasoning_text("learner (agent_1)", a1)
        + "\n\n"
        + _reasoning_text("partner (agent_2)", a2)
    )
    return learner_scores, partner_scores, reasoning.strip()



async def run_single_episode(
    env_profile,
    agent_profiles: list,
    fm: FM,
    learner_model: str,
    partner_model: str,
    memory_prompt: str = "",
    max_turns: int = 20,
    learner_goal: str = "",
    # rubric/partner_profile/judge_self_consistency_k: kept for backward-compat callers; unused
    rubric=None,
    partner_profile=None,
    judge_self_consistency_k: int = 3,
    evaluator_model: str = "",
    partner_key=None,                   # PartnerKey | None — injects key-conditioned prompt when set
    fm_judge: Optional["FM"] = None,    # separate judge FM; falls back to fm if not provided
    on_turn: Optional[Callable[[list[dict]], None]] = None,
) -> EpisodeResult:
    """Run one episode. agent_profiles[0] is the learner, [1] the partner.

    fm_judge is used for _evaluate_diagnostics and _run_key_check (cross-lab judge).
    fm is used for generation/reflection (the learner's pipeline model).
    """
    _judge = fm_judge if fm_judge is not None else fm
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
        custom_template=_build_partner_turn_prompt(partner_key),
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

    # 7-dim diagnostics feed reflection; terminal success is §3.2 (GOAL≥7 ∧ REL≥0).
    # Uses fm_judge (cross-lab judge) not the learner pipeline FM.
    learner_scores, partner_scores, reasoning = _evaluate_diagnostics(
        env.inbox, _judge, learner_goal=learner_goal
    )
    rubric_results: list[dict] = []  # kept for back-compat reading of old archives
    outcome_achieved = False
    constraint_preserved = False

    # §3.2 terminal success label: GOAL≥7 ∧ REL≥0 ∧ goal_achieved.
    # Conjoining the judge's strict goal_achieved bool is belt-and-suspenders: it guards
    # against a lenient goal-score (e.g. 7+ awarded to a promise or a vetoable contingent
    # outcome) silently labelling a non-solved scenario as solved, which would also corrupt
    # the analyze_beyond_frontier diagnosis-routing that steers mutation operators.
    goal_score = learner_scores.get("goal", 0.0)
    rel_score = learner_scores.get("relationship", 0.0)
    judge_goal_achieved = bool(learner_scores.get("goal_achieved", True))
    base_success = (goal_score >= 7.0) and (rel_score >= 0.0) and judge_goal_achieved

    # §3.3 key-aware check — runs on EVERY attempt of keyed scenarios, not just successes.
    # The verdict's diagnostic payload (which triggers fired, which conditions were approached)
    # is the primary input for reflection on failed attempts. Gating on base_success would
    # leave key_check_verdicts=None for all attempts reflection actually analyzes.
    key_check_result: Optional[dict] = None
    key_check_passed = True
    if partner_key is not None:
        history = _build_history(env.inbox)
        learner_name = agent_profiles[0].first_name if agent_profiles else "the actor"
        partner_name = agent_profiles[1].first_name if len(agent_profiles) > 1 else "the partner"
        key_check_result = _run_key_check(_judge, partner_key, history, learner_name, partner_name)
        key_check_passed = bool(key_check_result.get("key_check_passed", False))

    terminal_success = base_success and key_check_passed
    goal_achieved = terminal_success

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
        terminal_success=terminal_success,
        key_check_result=key_check_result,
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
