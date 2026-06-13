"""ExpeL baseline (Baseline 3, §5.1) — a faithful port of ExpeL's experiential
learning pipeline onto SOTOPIA social episodes.

ExpeL (Zhao et al., AAAI 2024) is a three-stage, inference-time-only method:

  1. EXPERIENCE GATHERING — a Reflexion-style agent attempts each training task
     with up to N retries, accumulating verbal reflections between attempts.
     This yields a pool of SUCCESS and FAILURE trajectories per task.

  2. INSIGHT EXTRACTION — an LLM reads (a) contrastive success-vs-failure pairs
     and (b) batches of successes, and maintains a flat, cross-task list of
     natural-language *insights* (rules) via four operations carrying importance
     counts: ADD (+2), EDIT (+1), AGREE (+1), REMOVE (-1, or -3 when the list is
     full). Rules whose count hits 0 are dropped; the list is kept sorted by
     count. This is `ExpelAgent.create_rules` in the original repo.

  3. EVALUATION — for each eval task, inject the extracted insight list PLUS the
     top-k most task-similar successful trajectories (dynamic few-shot) into the
     agent's prompt, then run.

This module ports that methodology onto social-omni-epic primitives:
  - trajectories come from `episode_runner.run_single_episode`
  - the success signal is SOTOPIA-Eval GOAL >= threshold
  - insights are injected via the learner's `memory_prompt`
  - retrieval reuses cosine similarity over FM embeddings

What is intentionally NOT ported (these are this project's extensions over
ExpeL, per §6 of the architecture doc, and must stay OUT of the baseline):
the structured chronicle schema, Confidence/Dimension gating, the adversarial
agent, UCB1 curriculum selection, and open-ended scenario generation. The
baseline operates only on the fixed 90 SOTOPIA seeds with a flat voted rule
list, exactly as ExpeL does.

The few-shot retrieval at eval time follows the full-ExpeL configuration
(insights + retrieved trajectories), which is the paper's headline agent.

Port fidelity notes:
  - `parse_rules` and `update_rules` are copied verbatim from
    `ExpeL/agent/expel.py` (domain-agnostic operation bookkeeping).
  - `random_divide_list` is copied verbatim from `ExpeL/utils.py`.
  - The operation-format block and the critique-summary suffixes are copied
    verbatim from `ExpeL/prompts/templates/human.py`; only the surrounding
    task-framing prose is rewritten from ReAct/Docstore to social interaction.
"""
from __future__ import annotations

import asyncio
import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .data_models import SocialScenario
from .episode_runner import clean_transcript, run_single_episode
from .fm import FM
from .sotopia_bridge import scenario_to_sotopia_profiles


# ===========================================================================
# Trajectory representation
# ===========================================================================

@dataclass
class ExpelTrajectory:
    """One attempt at one scenario, plus its outcome. The ExpeL analogue of a
    ReAct trajectory — here the 'trajectory' body is the cleaned conversation."""
    scenario_id: str
    task_idx: int
    task: str                      # scenario description (retrieval + critique key)
    learner_goal: str
    transcript_text: str           # formatted conversation body
    success: bool
    goal_score: float
    trial: int                     # 0-indexed attempt number within the episode
    reflections: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def to_critique_block(self) -> str:
        """The trajectory text fed into critique prompts. The SUCCESSFUL/FAILED
        label is supplied by the prompt template, so we only carry goal score."""
        return (
            f"Learner's private goal: {self.learner_goal}\n"
            f"GOAL score: {self.goal_score:.1f}/10\n"
            f"Conversation:\n{self.transcript_text}"
        )


def _format_transcript(transcript: list[dict]) -> str:
    """Render a cleaned transcript (list of {turn, speaker, content}) as text."""
    lines = []
    for t in transcript:
        speaker = t.get("speaker") or t.get("sender") or "?"
        lines.append(f"Turn {t.get('turn', '?')} [{speaker}]: {t.get('content', '')}")
    return "\n".join(lines) if lines else "(no dialogue)"


# ===========================================================================
# Prompts — operation format & suffixes copied verbatim from ExpeL; task-
# framing prose adapted from ReAct/Docstore to social interaction.
# ===========================================================================

# VERBATIM from ExpeL/prompts/templates/human.py (FORMAT_RULES_OPERATION_TEMPLATE)
_FORMAT_RULES_OPERATION_TEMPLATE = """<OPERATION> <RULE NUMBER>: <RULE>

The available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD <NEW RULE NUMBER>: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

# VERBATIM from ExpeL/prompts/templates/human.py (CRITIQUE_SUMMARY_SUFFIX)
_CRITIQUE_SUMMARY_SUFFIX = dict(
    full="""Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:
""",
    not_full="""Below are the operations you do to the above list of EXISTING RULES:
""",
)

# System instructions — adapted from ExpeL hotpotQA SYSTEM_CRITIQUE_* to the
# social domain. Operation semantics are unchanged.
_SYSTEM_CRITIQUE_COMPARE = (
    "You are an advanced reasoning agent that can add, edit or remove rules from "
    "your existing rule set, based on forming new critiques of past social "
    "interaction episodes. You will be given two previous episodes on the same "
    "social scenario, in which a learner agent pursued a private social goal by "
    "talking with a partner agent: one SUCCESSFUL and one FAILED trial. The "
    "learner failed the trial because it did not achieve its social goal (its "
    "SOTOPIA GOAL score was below threshold) — for example by being too blunt, "
    "too passive, conceding too early, damaging the relationship, or missing "
    "information it needed to surface."
)
_SYSTEM_CRITIQUE_ALL_SUCCESS = (
    "You are an advanced reasoning agent that can add, edit or remove rules from "
    "your existing rule set, based on forming new critiques of past social "
    "interaction episodes. You will be given several SUCCESSFUL episodes in which "
    "a learner agent achieved its private social goal by talking with a partner "
    "agent."
)
_SYSTEM_CRITIQUE_IMPROVE = (
    "You are an advanced reasoning agent that can add, edit or remove rules from "
    "your existing rule set, based on forming new critiques of past social "
    "interaction episodes. You will be given two previous episodes on the same "
    "social scenario, in which a learner agent pursued a private social goal by "
    "talking with a partner agent. In BOTH episodes the learner ultimately FAILED "
    "to achieve its goal (its SOTOPIA GOAL score stayed below threshold), but in "
    "one trial it did MEASURABLY BETTER (a higher GOAL score) than the other. "
    "Identify what the better trial did right that moved it closer and should be "
    "repeated, AND what was still missing that kept it from succeeding and must "
    "change. Do NOT treat the better trial as a full success — it fell short too."
)
_SYSTEM_REFLECTION = (
    "You will be given a previous episode in which you played a character pursuing "
    "a private social goal in a conversation. You were unsuccessful: you did not "
    "achieve your social goal (your SOTOPIA GOAL score was below threshold). In a "
    "few sentences, diagnose a possible reason for the failure and devise a new, "
    "concise, high-level plan that aims to mitigate the same failure next time. "
    "Use complete sentences. Do not exceed 4 sentences."
)

# Human critique templates — adapted from ExpeL human_critique_existing_rules_*
# ("trials"->"social interaction episodes", "Thought and Action"->"strategy and
# dialogue moves"). The trailing operation block is verbatim.
_HUMAN_COMPARE_TEMPLATE = (
    """Here are the two previous episodes to compare and critique:
SOCIAL SCENARIO:
{task}

SUCCESSFUL TRIAL:
{success_history}

FAILED TRIAL:
{fail_history}

Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting the successful trial with the failed trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques that capture how to behave so as to avoid similar failures when encountered with different social scenarios in the future. Have an emphasis on critiquing how to choose better strategy and dialogue moves. Follow the below format:

"""
    + _FORMAT_RULES_OPERATION_TEMPLATE
)

_HUMAN_IMPROVE_TEMPLATE = (
    """Here are the two previous episodes to compare and critique. BOTH fell short of the goal, but one improved on the other:
SOCIAL SCENARIO:
{task}

IMPROVED TRIAL (higher GOAL score, but still below threshold):
{success_history}

WORSE TRIAL:
{fail_history}

Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting the improved trial with the worse one, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques capturing BOTH what helped the learner do better AND what was still missing to fully achieve the goal, so they transfer to different social scenarios. Have an emphasis on critiquing how to choose better strategy and dialogue moves. Follow the below format:

"""
    + _FORMAT_RULES_OPERATION_TEMPLATE
)

_HUMAN_ALL_SUCCESS_TEMPLATE = (
    """Here are the successful episodes:
{success_history}

Here are the EXISTING RULES:
{existing_rules}

By examining the successful episodes, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights that capture what made these social interactions succeed, so they can be used as helpful tips on different social scenarios in the future. Have an emphasis on tips that help choose better strategy and dialogue moves. Follow the below format:

"""
    + _FORMAT_RULES_OPERATION_TEMPLATE
)

# Eval-time injection template — adapted from ExpeL RULE_TEMPLATE.
_RULE_INJECTION_TEMPLATE = (
    "The following are some experiences (in decreasing order of importance) you "
    "gathered on previous social interactions. Use these as references to help "
    "you pursue your social goal more effectively in this conversation:\n{rules}"
)


# ===========================================================================
# Rule bookkeeping — VERBATIM ports from ExpeL/agent/expel.py
# ===========================================================================

def parse_rules(llm_text: str) -> list[tuple[str, str]]:
    """VERBATIM port of ExpeL parse_rules."""
    pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
    matches = re.findall(pattern, llm_text)

    res = []
    banned_words = ['ADD', 'AGREE', 'EDIT']
    for operation, text in matches:
        text = text.strip()
        if text != '' and not any([w in text for w in banned_words]) and text.endswith('.'):
            if 'ADD' in operation:
                res.append(('ADD', text))
            else:
                res.append((operation.strip(), text))
    return res


def _retrieve_rule_index(rules, operation):
    operation_rule_text = operation[1]
    for i in range(len(rules)):
        if rules[i][0] in operation_rule_text:
            return i


def _is_existing_rule(rules, operation_rule_text):
    for i in range(len(rules)):
        if rules[i][0] in operation_rule_text:
            return True
    return False


def update_rules(rules: list[tuple[str, int]], operations: list[tuple[str, str]],
                 list_full: bool = False) -> list[tuple[str, int]]:
    """VERBATIM port of ExpeL update_rules. `rules` is a list of (text, count)."""
    # remove problematic operations
    delete_indices = []
    for i in range(len(operations)):
        operation, operation_rule_text = operations[i]
        operation_type = operation.split(' ')[0]
        rule_num = int(operation.split(' ')[1]) if ' ' in operation else None

        if operation_type == 'ADD':
            if _is_existing_rule(rules, operation_rule_text):
                delete_indices.append(i)
        else:
            if operation_type == 'EDIT':
                if _is_existing_rule(rules, operation_rule_text):
                    rule_num = _retrieve_rule_index(rules, (operation, operation_rule_text))
                    operations[i] = (f'AGREE {rule_num+1}', rules[rule_num][0])
                elif (rule_num is None) or (rule_num > len(rules)):
                    delete_indices.append(i)
            elif operation_type == 'REMOVE' or operation_type == 'AGREE':
                if not _is_existing_rule(rules, operation_rule_text):
                    delete_indices.append(i)

    operations = [operations[i] for i in range(len(operations)) if i not in delete_indices]

    for op in ['REMOVE', 'AGREE', 'EDIT', 'ADD']:  # Order is important
        for i in range(len(operations)):
            operation, operation_rule_text = operations[i]
            operation_type = operation.split(' ')[0]
            if operation_type != op:
                continue

            if operation_type == 'REMOVE':
                rule_index = _retrieve_rule_index(rules, (operation, operation_rule_text))
                remove_strength = 3 if list_full else 1
                rules[rule_index] = (rules[rule_index][0], rules[rule_index][1] - remove_strength)
            elif operation_type == 'AGREE':
                rule_index = _retrieve_rule_index(rules, (operation, operation_rule_text))
                rules[rule_index] = (rules[rule_index][0], rules[rule_index][1] + 1)
            elif operation_type == 'EDIT':
                rule_index = int(operation.split(' ')[1]) - 1
                rules[rule_index] = (operation_rule_text, rules[rule_index][1] + 1)
            elif operation_type == 'ADD':
                rules.append((operation_rule_text, 2))
    rules = [rules[i] for i in range(len(rules)) if rules[i][1] > 0]
    rules.sort(key=lambda x: x[1], reverse=True)

    return rules


def random_divide_list(lst: list, k: int) -> list[list]:
    """VERBATIM port of ExpeL random_divide_list — chunks of max length k."""
    random.shuffle(lst)
    if len(lst) <= k:
        return [lst]
    num_chunks = math.ceil(len(lst) / k)
    chunk_size = math.ceil(len(lst) / num_chunks)
    return [lst[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]


# ===========================================================================
# Reflexion-style reflection (between gathering attempts)
# ===========================================================================

def _format_reflections(reflections: list[str]) -> str:
    """ExpeL PREVIOUS_TRIALS_FORMATTER analogue — injected as the learner's
    memory_prompt on retry attempts."""
    if not reflections:
        return ""
    prefix = (
        "You have attempted this exact social scenario before but failed to "
        "achieve your goal. The following reflection(s) give a plan to avoid "
        "failing in the same way. Use them to improve your strategy.\nReflections:"
    )
    for r in reflections:
        prefix += f"\n- {r.strip()}"
    return prefix


def _reflect(fm: FM, task: str, learner_goal: str, transcript_text: str,
             goal_score: float) -> str:
    """Generate one Reflexion-style reflection after a failed attempt."""
    user = (
        f"SOCIAL SCENARIO:\n{task}\n\n"
        f"Your private goal: {learner_goal}\n\n"
        f"Your conversation (GOAL score {goal_score:.1f}/10 — below threshold):\n"
        f"{transcript_text}\n\nReflection:"
    )
    # temperature=None defers to the FM's default; some models (gpt-5-mini)
    # only accept the default temperature.
    return fm.query(_SYSTEM_REFLECTION, user, temperature=None).strip()


# ===========================================================================
# Stage 1 — Experience gathering
# ===========================================================================

def gather_trajectories(
    scenarios: list[SocialScenario],
    fm: FM,
    learner_model: str,
    partner_model: str,
    max_trials: int = 3,
    max_turns: int = 20,
    goal_threshold: float = 7.0,
    judge_self_consistency_k: int = 1,
    on_log: Optional[Callable[[str], None]] = None,
    initial_state: Optional[dict] = None,
    on_progress: Optional[Callable[[dict, dict, dict, set], None]] = None,
) -> tuple[dict[int, list[ExpelTrajectory]], dict[int, list[ExpelTrajectory]], dict[int, str], set]:
    """Reflexion-style gathering: attempt each scenario up to `max_trials` times,
    reflecting after each failure and stopping on first success.

    Crash-safe for long unattended runs:
      - `initial_state` (a previously written trajectories dict) lets the run
        RESUME — scenarios already fully gathered (recorded in `completed_idx`)
        are skipped. Partially-gathered scenarios (process died mid-attempts)
        are discarded and re-run cleanly from scratch.
      - `on_progress(succeeded, failed, idx2task, completed)` is called after
        EACH scenario completes, so a caller can checkpoint to disk per-seed.

    Returns (succeeded, failed, idx2task, completed), where succeeded/failed map
    a task index to its list of ExpelTrajectory, idx2task maps the index to the
    scenario description (the critique/retrieval key), and completed is the set
    of fully-gathered task indices.
    """
    log = on_log or (lambda _m: None)
    if initial_state:
        succeeded, failed, idx2task = trajectories_from_dict(initial_state)
        completed: set = set(initial_state.get("completed_idx", []))
        # Drop any partial entries for scenarios that did not fully complete
        # (e.g. a crash mid-attempt), so they re-run cleanly without duplicates.
        for i in list(idx2task.keys()):
            if i not in completed:
                succeeded.pop(i, None)
                failed.pop(i, None)
                idx2task.pop(i, None)
        if completed:
            log(f"  [gather] resuming — {len(completed)} scenarios already done, skipping them")
    else:
        succeeded, failed, idx2task, completed = {}, {}, {}, set()

    for idx, scenario in enumerate(scenarios):
        if idx in completed:
            continue
        env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
        learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ""
        idx2task[idx] = scenario.scenario
        succeeded.setdefault(idx, [])
        failed.setdefault(idx, [])
        reflections: list[str] = []

        for trial in range(max_trials):
            memory_prompt = _format_reflections(reflections)
            try:
                result = asyncio.run(run_single_episode(
                    env_profile=env_profile,
                    agent_profiles=agent_profiles,
                    fm=fm,
                    learner_model=learner_model,
                    partner_model=partner_model,
                    memory_prompt=memory_prompt,
                    max_turns=max_turns,
                    learner_goal=learner_goal,
                    rubric=None,
                    partner_profile=None,
                    judge_self_consistency_k=judge_self_consistency_k,
                ))
            except Exception as e:  # noqa: BLE001 — a single bad episode shouldn't abort the run
                log(f"  [gather idx={idx} trial={trial}] ERROR: {e}")
                break

            goal = float(result.learner_scores.get("goal", 0.0))
            success = goal >= goal_threshold
            transcript_text = _format_transcript(clean_transcript(result.transcript))
            traj = ExpelTrajectory(
                scenario_id=scenario.id,
                task_idx=idx,
                task=scenario.scenario,
                learner_goal=learner_goal,
                transcript_text=transcript_text,
                success=success,
                goal_score=goal,
                trial=trial,
                reflections=list(reflections),
            )
            log(f"  [gather idx={idx} trial={trial}] goal={goal:.1f} "
                f"{'SUCCESS' if success else 'FAIL'}")

            if success:
                succeeded[idx].append(traj)
                break
            failed[idx].append(traj)
            if trial < max_trials - 1:
                reflections.append(_reflect(fm, scenario.scenario, learner_goal,
                                            transcript_text, goal))

        # Scenario fully gathered — mark complete and checkpoint.
        completed.add(idx)
        if on_progress is not None:
            on_progress(succeeded, failed, idx2task, completed)

    return succeeded, failed, idx2task, completed


# ===========================================================================
# Stage 2 — Insight extraction (port of ExpelAgent.create_rules)
# ===========================================================================

def _critique(
    fm: FM,
    mode: str,                       # "compare" | "all_success" | "improve"
    existing_rules: list[str],
    rule_count: int,
    max_num_rules: int,
    success_history: str,
    fail_history: str = "",
    task: str = "",
) -> str:
    """One critique LLM call returning raw operation text (parsed by parse_rules)."""
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(existing_rules, 1)) or ""
    # ExpeL: 'full' suffix once the list reaches capacity, else 'not_full'.
    suffix = (_CRITIQUE_SUMMARY_SUFFIX["full"]
              if max_num_rules <= rule_count
              else _CRITIQUE_SUMMARY_SUFFIX["not_full"])

    if mode == "compare":
        system = _SYSTEM_CRITIQUE_COMPARE
        user = _HUMAN_COMPARE_TEMPLATE.format(
            task=task, success_history=success_history,
            fail_history=fail_history, existing_rules=rules_block,
        )
    elif mode == "improve":
        system = _SYSTEM_CRITIQUE_IMPROVE
        user = _HUMAN_IMPROVE_TEMPLATE.format(
            task=task, success_history=success_history,
            fail_history=fail_history, existing_rules=rules_block,
        )
    else:
        system = _SYSTEM_CRITIQUE_ALL_SUCCESS
        user = _HUMAN_ALL_SUCCESS_TEMPLATE.format(
            success_history=success_history, existing_rules=rules_block,
        )
    user = user + suffix
    # temperature=None defers to the FM default (gpt-5-mini rejects non-default
    # temperatures; Fix in fm.py also degrades gracefully on other models).
    return fm.query(system, user, temperature=None)


def _extract_one_fold(
    training_ids: list[int],
    succeeded: dict[int, list[ExpelTrajectory]],
    failed: dict[int, list[ExpelTrajectory]],
    idx2task: dict[int, str],
    fm: FM,
    max_num_rules: int,
    success_critique_num: int,
    on_log: Optional[Callable[[str], None]] = None,
    include_frontier_improvements: bool = True,
) -> list[tuple[str, int]]:
    """Build a rule list from one set of training task ids — the body of ExpeL
    create_rules: compare critiques first, then frontier-improvement critiques,
    then batched success critiques.

    When include_frontier_improvements (default), tasks that NEVER reached a
    success but whose later attempts improved on attempt 1 (objective GOAL gain)
    also feed the rule set via an "improve" critique — contrasting the best later
    (still-failed) attempt against attempt 1. This recovers the frontier-unsolved
    band that standard ExpeL discards. Tasks with no objective improvement (flat /
    beyond-frontier) are still excluded. The gate is objective GOAL gain rather
    than the LP label, to avoid the LP judge's noise."""
    log = on_log or (lambda _m: None)
    rule_items_with_count: list[tuple[str, int]] = []

    # ----- Compare critiques (success vs failure on the same task) -----
    for tid in training_ids:
        for succ in succeeded.get(tid, []):
            for fail in failed.get(tid, []):
                existing = [r for r, _ in rule_items_with_count]
                llm_out = _critique(
                    fm, "compare", existing,
                    rule_count=len(rule_items_with_count),
                    max_num_rules=max_num_rules,
                    success_history=succ.to_critique_block(),
                    fail_history=fail.to_critique_block(),
                    task=idx2task[tid],
                )
                ops = parse_rules(llm_out)
                rule_items_with_count = update_rules(
                    rule_items_with_count, ops,
                    list_full=(max_num_rules + 5 <= len(rule_items_with_count)),
                )
                log(f"    [compare tid={tid}] -> {len(rule_items_with_count)} rules")

    # ----- Frontier-improvement critiques (never solved, but improved) -----
    # For tasks with NO success, contrast the best later attempt (higher GOAL,
    # still below threshold) against attempt 1. Gate on objective GOAL gain so
    # genuinely flat (beyond-frontier) tasks are skipped.
    if include_frontier_improvements:
        for tid in training_ids:
            if succeeded.get(tid):
                continue  # has a success -> already covered by compare/all_success
            fails = failed.get(tid, [])
            if len(fails) < 2:
                continue
            base = min(fails, key=lambda t: t.trial)            # attempt 1
            later = [t for t in fails if t.trial > base.trial]
            if not later:
                continue
            best = max(later, key=lambda t: t.goal_score)       # best improved attempt
            if best.goal_score <= base.goal_score:
                continue  # no objective improvement -> beyond-frontier-like, skip
            existing = [r for r, _ in rule_items_with_count]
            llm_out = _critique(
                fm, "improve", existing,
                rule_count=len(rule_items_with_count),
                max_num_rules=max_num_rules,
                success_history=best.to_critique_block(),
                fail_history=base.to_critique_block(),
                task=idx2task[tid],
            )
            ops = parse_rules(llm_out)
            rule_items_with_count = update_rules(
                rule_items_with_count, ops,
                list_full=(max_num_rules + 5 <= len(rule_items_with_count)),
            )
            log(f"    [improve tid={tid}] base_goal={base.goal_score:.1f} "
                f"best_goal={best.goal_score:.1f} -> {len(rule_items_with_count)} rules")

    # ----- Success critiques (batches of successes) -----
    all_success = [
        (idx2task[tid], succeeded[tid][0].to_critique_block())
        for tid in training_ids if succeeded.get(tid)
    ]
    for chunk in random_divide_list(all_success, success_critique_num):
        if not chunk:
            continue
        success_trials = "\n\n".join(f"{task}\n{body}" for task, body in chunk)
        existing = [r for r, _ in rule_items_with_count]
        llm_out = _critique(
            fm, "all_success", existing,
            rule_count=len(rule_items_with_count),
            max_num_rules=max_num_rules,
            success_history=success_trials.strip(),
        )
        ops = parse_rules(llm_out)
        rule_items_with_count = update_rules(
            rule_items_with_count, ops,
            list_full=(max_num_rules + 5 <= len(rule_items_with_count)),
        )
        log(f"    [success chunk n={len(chunk)}] -> {len(rule_items_with_count)} rules")

    return rule_items_with_count


def _kfold_split(task_ids: list[int], k_folds: int, seed: int) -> list[list[int]]:
    """Deterministic k-fold split of task ids (ExpeL get_split_eval_idx_list analogue)."""
    ids = list(task_ids)
    random.Random(seed).shuffle(ids)
    return [ids[i::k_folds] for i in range(k_folds)]


def extract_insights(
    succeeded: dict[int, list[ExpelTrajectory]],
    failed: dict[int, list[ExpelTrajectory]],
    idx2task: dict[int, str],
    fm: FM,
    max_num_rules: int = 10,
    success_critique_num: int = 8,
    k_folds: int = 1,
    seed: int = 42,
    on_log: Optional[Callable[[str], None]] = None,
    include_frontier_improvements: bool = True,
) -> dict:
    """Run ExpeL insight extraction.

    k_folds == 1 (default): train on ALL tasks and return a single combined rule
        set. Use this when the eval scenarios are a SEPARATE set from the 90
        training seeds (the intended social-omni-epic flow). ExpeL's k-fold only
        exists because its eval tasks are drawn from the same pool; with a held-
        out eval set it is unnecessary.

    k_folds >= 2: ExpeL-faithful cross-validation. Returns per-fold rule sets and
        the fold assignment so that each held-out task is evaluated with insights
        extracted from the OTHER folds (no leakage). Use this for in-distribution
        eval on the 90 seeds themselves.

    Returns a dict:
        {
          "k_folds": int,
          "max_num_rules": int,
          "success_critique_num": int,
          "all": {"rules": [...], "rule_items_with_count": [[text, count], ...]},
          "folds": [{"eval_idxs": [...], "rules": [...],
                     "rule_items_with_count": [...]}, ...]   # only if k_folds>=2
        }
    """
    log = on_log or (lambda _m: None)
    all_ids = sorted(idx2task.keys())

    result: dict = {
        "k_folds": k_folds,
        "max_num_rules": max_num_rules,
        "success_critique_num": success_critique_num,
    }

    if k_folds <= 1:
        log("Extracting insights on ALL training tasks (single fold)...")
        items = _extract_one_fold(all_ids, succeeded, failed, idx2task, fm,
                                  max_num_rules, success_critique_num, on_log,
                                  include_frontier_improvements)
        result["all"] = {
            "rules": [r for r, _ in items],
            "rule_items_with_count": [[r, c] for r, c in items],
        }
        return result

    eval_idx_list = _kfold_split(all_ids, k_folds, seed)
    folds = []
    for k, eval_idxs in enumerate(eval_idx_list):
        log(f"Extracting insights for FOLD {k} (held out {len(eval_idxs)} tasks)...")
        training_ids = [i for i in all_ids if i not in set(eval_idxs)]
        items = _extract_one_fold(training_ids, succeeded, failed, idx2task, fm,
                                  max_num_rules, success_critique_num, on_log,
                                  include_frontier_improvements)
        folds.append({
            "eval_idxs": list(eval_idxs),
            "rules": [r for r, _ in items],
            "rule_items_with_count": [[r, c] for r, c in items],
        })
    result["folds"] = folds
    # Also provide a combined set trained on everything, for external eval reuse.
    items_all = _extract_one_fold(all_ids, succeeded, failed, idx2task, fm,
                                  max_num_rules, success_critique_num, on_log,
                                  include_frontier_improvements)
    result["all"] = {
        "rules": [r for r, _ in items_all],
        "rule_items_with_count": [[r, c] for r, c in items_all],
    }
    return result


# ===========================================================================
# Stage 3 — Eval-time retrieval + memory_prompt assembly
# ===========================================================================

def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_trajectory_bank(
    succeeded: dict[int, list[ExpelTrajectory]],
    fm: FM,
) -> list[ExpelTrajectory]:
    """Flatten successful trajectories into a retrievable bank and attach
    embeddings of their task text (ExpeL retrieves by task similarity)."""
    bank: list[ExpelTrajectory] = []
    for trajs in succeeded.values():
        for t in trajs:
            bank.append(t)
    if not bank:
        return bank
    embeddings = fm.get_embeddings([t.task for t in bank])
    for t, emb in zip(bank, embeddings):
        t.embedding = emb
    return bank


def retrieve_fewshots(
    query_text: str,
    bank: list[ExpelTrajectory],
    fm: FM,
    top_k: int = 2,
    exclude_scenario_id: str = "",
) -> list[ExpelTrajectory]:
    """Top-k task-similar successful trajectories (ExpeL fewshot_strategy=
    task_similarity). `exclude_scenario_id` guards against retrieving the eval
    task's own trajectory when the eval set overlaps the training set."""
    candidates = [t for t in bank if t.embedding and t.scenario_id != exclude_scenario_id]
    if not candidates:
        return []
    q = fm.get_embeddings([query_text])[0]
    scored = sorted(candidates, key=lambda t: _cosine(q, t.embedding), reverse=True)
    return scored[:top_k]


def format_memory_prompt(
    rules: list[str],
    fewshots: list[ExpelTrajectory],
    include_fewshots: bool = True,
) -> str:
    """Assemble the learner's memory_prompt: the insight list (always) plus, in
    the full-ExpeL configuration, the retrieved few-shot trajectories."""
    parts = []
    if rules:
        rules_str = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
        parts.append(_RULE_INJECTION_TEMPLATE.format(rules=rules_str))
    if include_fewshots and fewshots:
        blocks = []
        for j, t in enumerate(fewshots, 1):
            blocks.append(
                f"[Similar past interaction {j}] (a learner that ACHIEVED its goal)\n"
                f"Scenario: {t.task[:240]}\n"
                f"Goal: {t.learner_goal}\n"
                f"How it played out:\n{t.transcript_text}"
            )
        parts.append(
            "Here are some past interactions on similar scenarios where the "
            "learner succeeded. Use them as worked examples:\n" + "\n\n".join(blocks)
        )
    return "\n\n".join(parts)


# ===========================================================================
# (De)serialization helpers for the gathered trajectory pool
# ===========================================================================

def trajectories_to_dict(
    succeeded: dict[int, list[ExpelTrajectory]],
    failed: dict[int, list[ExpelTrajectory]],
    idx2task: dict[int, str],
    completed: Optional[set] = None,
) -> dict:
    def _ser(d):
        return {str(k): [t.__dict__ for t in v] for k, v in d.items()}
    return {
        "succeeded": _ser(succeeded),
        "failed": _ser(failed),
        "idx2task": {str(k): v for k, v in idx2task.items()},
        # completed_idx drives crash-safe resume in gather_trajectories.
        "completed_idx": sorted(int(i) for i in (completed or idx2task.keys())),
    }


def trajectories_from_dict(data: dict) -> tuple[
    dict[int, list[ExpelTrajectory]], dict[int, list[ExpelTrajectory]], dict[int, str]
]:
    def _deser(d):
        out = {}
        for k, v in d.items():
            out[int(k)] = [ExpelTrajectory(**rec) for rec in v]
        return out
    succeeded = _deser(data["succeeded"])
    failed = _deser(data["failed"])
    idx2task = {int(k): v for k, v in data["idx2task"].items()}
    return succeeded, failed, idx2task
