"""Shared curriculum engine: single K-attempt skill loop (Phase 2).

Loop 1 (difficulty calibration) is gone. Every scenario goes straight into the K-attempt
skill loop. Attempt-1 success → too_easy classification; post-loop LP computation determines
frontier vs. beyond_frontier. All classification results propagate to the scenario object so
archive.py's Thompson sampler can consume them directly.

Used by scripts/run_curriculum.py, scripts/run_debug.py, and scripts/run_phase2.py.
`config` may be an OmegaConf DictConfig or a plain dict — only `.get(...)` access is used.
"""
from copy import deepcopy

from .data_models import SocialScenario
from .skills_chronicle import SkillsChronicle

# Equivalent total_votes charged to too_easy children in the Thompson posterior.
# K=4 episodes × 1.5 vote pairs average = 6 votes. Improved = 0 (solved immediately ⇒
# no learning evidence), so the anchor is penalised by the full 6 failure-side votes.
K_VOTES_EQUIV = 6


def run_coherence_gate(
    scenario, coherence_checker, task_gen, fm, config, anchor, iteration,
) -> tuple:
    """Run the coherence check with patch-retry and re-embed. Returns (scenario_or_None, passed)."""
    if not config.get("enable_coherence_check", True):
        return scenario, True
    max_coherence = int(config.get("coherence_max_retries", 2))
    for _c in range(max_coherence + 1):
        c_result = coherence_checker.check(scenario)
        if c_result.passed:
            return scenario, True
        scenario = task_gen.patch_scenario(scenario, c_result.issues)
        if scenario is None:
            return None, False
        scenario.iteration = iteration
        scenario.parent_example_ids = [anchor.id] if anchor else []
        try:
            scenario.embedding = fm.get_embeddings([scenario.to_text_for_embedding()])[0]
        except Exception:
            return None, False
    return scenario, False


def build_episode_inputs(scenario: SocialScenario, scenario_to_sotopia_profiles):
    """Convert to sotopia profiles with the learner at index 0.

    Returns (env_profile, agent_profiles, learner_goal, partner_profile, success_rubric).
    success_rubric is kept in the return tuple for backward-compat callers; it is not used
    for gating (terminal_success = §3.2 GOAL≥7 ∧ REL≥0).
    """
    env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
    li = scenario.target_agent_idx
    if li == 1:
        agent_profiles = [agent_profiles[1], agent_profiles[0]]
        env_profile.agent_goals = [env_profile.agent_goals[1], env_profile.agent_goals[0]]
    learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ""
    partner_idx = 1 - li
    partner_profile = (
        scenario.agent_profiles[partner_idx] if partner_idx < len(scenario.agent_profiles) else None
    )
    return env_profile, agent_profiles, learner_goal, partner_profile, scenario.success_rubric


def _set_titles(scenario: SocialScenario, title_gen) -> None:
    try:
        title_data = title_gen.generate(scenario, scenario.target_agent_idx)
        scenario.scenario_title = title_data["scenario_title"]
        scenario.social_dynamic = title_data["social_dynamic"]
        scenario.target_perspective = title_data["target_perspective"]
    except Exception:
        pass


async def run_episode_k_loop(
    scenario: SocialScenario,
    anchor,
    task_gen,
    reflection_mod,
    meta_mod,
    adversarial,
    title_gen,
    coherence_checker,
    run_single_episode,
    scenario_to_sotopia_profiles,
    fm,
    config,
    on_attempt_done=None,
    on_turn=None,
    fm_judge=None,  # cross-lab judge FM; falls back to fm (breaks monoculture only when provided)
) -> tuple:
    """K-attempt skill-learning loop with LP-based classification.

    Returns (scenario, terminal_state, outcome_int, final_scores, loop_info).
    terminal_state ∈ {"too_easy", "frontier", "beyond_frontier"} or "discarded" on error.
    outcome_int: 1=too_easy, 2=frontier_solved, 3=frontier_unsolved, 4=beyond_frontier.
    """
    from .episode_runner import clean_transcript
    from .lp_judge import compute_lp, LPResult

    K = int(config.get("max_attempts", 4))
    max_entries = int(config.get("chronicle_max_entries", 8))
    re_reflect = bool(config.get("adversarial", {}).get("re_reflect_on_rejection", True))
    learner_model = config.get("learner_model")
    partner_model = config.get("partner_model")
    max_turns = int(config.get("max_turns", 20))

    current_chronicle = (
        SkillsChronicle.from_markdown(anchor.skills_final_md or "") if anchor else SkillsChronicle()
    )
    loop_info: dict = {"skill_attempts": [], "terminal_state": None, "outcome": 0}

    _judge = fm_judge if fm_judge is not None else fm

    # Compute query embedding once for relevance-ranked chronicle truncation (§8.2).
    # Falls back to positional truncation when abstract goal or FM embedding unavailable.
    _query_embedding: list[float] | None = None
    _abstract_goal_text = (scenario.target_agent_goal_abstract or
                           (scenario.agent_goals[0] if scenario.agent_goals else ""))
    if _abstract_goal_text:
        try:
            _query_embedding = fm.get_embeddings([_abstract_goal_text])[0]
        except Exception:
            pass

    async def _episode(scn: SocialScenario, chronicle: SkillsChronicle):
        env_profile, agent_profiles, learner_goal, _partner_profile, _rubric = build_episode_inputs(
            scn, scenario_to_sotopia_profiles
        )
        mem = chronicle.format_for_prompt(
            max_entries=max_entries,
            query_embedding=_query_embedding,
            fm=fm,
        )
        return await run_single_episode(
            env_profile=env_profile,
            agent_profiles=agent_profiles,
            fm=fm,
            learner_model=learner_model,
            partner_model=partner_model,
            memory_prompt=mem,
            max_turns=max_turns,
            learner_goal=learner_goal,
            partner_key=scn.partner_key,
            fm_judge=_judge,
            on_turn=on_turn,
        )

    # ------------------------------------------------------------------ #
    # Attempt 1                                                            #
    # ------------------------------------------------------------------ #
    try:
        result = await _episode(scenario, current_chronicle)
    except Exception as e:
        import traceback
        print(f"    [attempt 1] episode error: {e}\n{traceback.format_exc()}")
        return scenario, "discarded", 0, {}, loop_info

    transcript1 = clean_transcript(result.transcript)
    att_rec: dict = {
        "attempt": 1,
        "transcript_clean": transcript1,
        "diagnostics_scores": result.learner_scores,
        "solved": result.terminal_success,
        "key_check_result": result.key_check_result,
    }
    loop_info["skill_attempts"].append(att_rec)
    if on_attempt_done:
        on_attempt_done(loop_info)

    # ------------------------------------------------------------------ #
    # too_easy fast-path: solved on attempt 1                             #
    # ------------------------------------------------------------------ #
    if result.terminal_success:
        diagnosis = task_gen.analyze_too_easy(scenario, transcript1)
        scenario.classification = "too_easy"
        scenario.too_easy_diagnosis = {
            "slack_knob": diagnosis.get("slack_knob"),
            "rationale": diagnosis.get("rationale"),
        }
        scenario.n_attempts = 1
        scenario.terminal_success = True
        # LP pseudo-votes for too_easy: 0 improved out of K_VOTES_EQUIV total.
        scenario.lp_value = 0.0
        scenario.lp_votes = K_VOTES_EQUIV
        loop_info["terminal_state"] = "too_easy"
        loop_info["outcome"] = 1
        loop_info["too_easy_diagnosis"] = scenario.too_easy_diagnosis

        # Skip meta-reflection: no learning occurred (solved immediately, nothing to reflect on).
        # Pass inherited chronicle through unchanged so children see accurate lineage knowledge.
        # Synthesizing here would route outcome=1 into the FAILURE branch (outcome==2 is success)
        # and generate a WARNING-dominant chronicle for a trivially solved scenario — poisoning
        # all descendants' generation prompts.
        scenario.skills_final_md = current_chronicle.to_markdown()
        loop_info["final_chronicle_md"] = scenario.skills_final_md
        scenario.goal_score = float(result.learner_scores.get("goal", 0.0))
        scenario.goal_trajectory = [float(result.learner_scores.get("goal", 0.0))]
        _set_titles(scenario, title_gen)
        return scenario, "too_easy", 1, result.learner_scores, loop_info

    # ------------------------------------------------------------------ #
    # Skill loop: attempts 2 … K                                          #
    # ------------------------------------------------------------------ #
    all_transcripts: list = [transcript1]
    all_scores: list = [{"attempt": 1, "scores": result.learner_scores, "solved": False}]
    all_versions: list = [deepcopy(current_chronicle)]
    all_key_checks: list = [result.key_check_result]
    all_edit_reasons: dict = {}
    final_scores: dict = result.learner_scores
    solved = False

    for attempt in range(2, K + 1):
        ref_out = reflection_mod.reflect(
            chronicle=current_chronicle, scenario=scenario, transcripts=all_transcripts,
            prior_edit_reasons=all_edit_reasons, attempt_num=attempt - 1, anchor_task=anchor,
            key_check_verdicts=all_key_checks, attempt_scores=all_scores,
        )
        adv_result = adversarial.check_reflection(
            ref_out, all_transcripts[-1], anchor_task=anchor, scenario=scenario
        )
        if not adv_result.approved and re_reflect:
            ref_out = reflection_mod.synthesize_with_critique(
                reflection_output=ref_out, adversarial_critique=adv_result.critique,
                chronicle=current_chronicle, scenario=scenario, transcripts=all_transcripts,
                prior_edit_reasons=all_edit_reasons, attempt_num=attempt - 1, anchor_task=anchor,
            )
        current_chronicle = ref_out.updated_chronicle
        all_versions.append(deepcopy(current_chronicle))
        all_edit_reasons.update(ref_out.edit_reasons)

        try:
            result = await _episode(scenario, current_chronicle)
        except Exception as e:
            import traceback
            print(f"    [attempt {attempt}] episode error: {e}\n{traceback.format_exc()}")
            break

        transcript = clean_transcript(result.transcript)
        all_transcripts.append(transcript)
        all_scores.append({
            "attempt": attempt, "scores": result.learner_scores, "solved": result.terminal_success,
        })
        all_key_checks.append(result.key_check_result)
        final_scores = result.learner_scores

        att_rec = {
            "attempt": attempt,
            "transcript_clean": transcript,
            "diagnostics_scores": result.learner_scores,
            "solved": result.terminal_success,
            "key_check_result": result.key_check_result,
            "reflection_diagnosis": ref_out.diagnosis,
            "reflection_edit_reasons": ref_out.edit_reasons,
            "adversarial_approved": adv_result.approved,
            "chronicle_after_reflection": current_chronicle.to_markdown(),
        }
        loop_info["skill_attempts"].append(att_rec)
        if on_attempt_done:
            on_attempt_done(loop_info)

        if result.terminal_success:
            solved = True
            break

    # ------------------------------------------------------------------ #
    # LP computation                                                       #
    # ------------------------------------------------------------------ #
    env_profile_lp, _, learner_goal_lp, _, _ = build_episode_inputs(scenario, scenario_to_sotopia_profiles)
    learner_goal_text = env_profile_lp.agent_goals[0] if env_profile_lp.agent_goals else ""
    # Prepend relationship label so the LP judge has context even when background is empty.
    _rel_label = scenario.relationship or ""
    _rel_bg = scenario.relationship_background or ""
    relational_stakes = f"{_rel_label}: {_rel_bg}".strip(": ") if _rel_label else _rel_bg

    try:
        lp_result = await compute_lp(
            fm_judge=_judge,
            scenario=scenario,
            transcripts=all_transcripts,
            learner_goal=learner_goal_text,
            relational_stakes=relational_stakes,
        )
    except Exception as e:
        print(f"    [LP] compute_lp error: {e}")
        lp_result = LPResult(lp_value=0.0, improved_votes=0, total_votes=0, n_pairs=0)

    loop_info["lp_value"] = lp_result.lp_value
    loop_info["lp_votes"] = lp_result.total_votes
    loop_info["lp_improved_votes"] = lp_result.improved_votes

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #
    if solved or lp_result.lp_value > 0:
        classification = "frontier"
    else:
        classification = "beyond_frontier"

    scenario.classification = classification
    scenario.lp_value = lp_result.lp_value
    scenario.lp_votes = lp_result.total_votes
    scenario.terminal_success = solved
    scenario.n_attempts = len(all_scores)

    outcome = 2 if solved else (3 if classification == "frontier" else 4)
    loop_info["terminal_state"] = classification
    loop_info["outcome"] = outcome

    # ------------------------------------------------------------------ #
    # Meta-reflection                                                      #
    # ------------------------------------------------------------------ #
    final_chronicle = meta_mod.synthesize(
        chronicle_versions=all_versions, transcripts=all_transcripts, edit_reasons=all_edit_reasons,
        outcome=outcome, scenario=scenario, anchor_task=anchor, attempt_scores=all_scores,
    )

    inherited_md = (anchor.skills_final_md or "") if anchor else ""
    adv_final = adversarial.check_final(final_chronicle, inherited_md, outcome=outcome)
    if not adv_final.approved:
        final_chronicle = meta_mod.synthesize(
            chronicle_versions=all_versions, transcripts=all_transcripts,
            edit_reasons=all_edit_reasons, outcome=outcome, scenario=scenario,
            anchor_task=anchor, attempt_scores=all_scores,
            adversarial_critique=adv_final.critique,
        )
        adv_final2 = adversarial.check_final(final_chronicle, inherited_md, outcome=outcome)
        if not adv_final2.approved:
            loop_info["final_check_flag"] = adv_final2.issues
            scenario.final_check_flag = adv_final2.issues

    loop_info["final_chronicle_md"] = final_chronicle.to_markdown()
    scenario.skills_final_md = final_chronicle.to_markdown()
    scenario.goal_score = float(final_scores.get("goal", 0.0))
    scenario.goal_trajectory = [float(s["scores"].get("goal", 0.0)) for s in all_scores]
    _set_titles(scenario, title_gen)

    return scenario, classification, outcome, final_scores, loop_info


# Backward-compat alias for callers that import the old name.
run_episode_two_loop = run_episode_k_loop
