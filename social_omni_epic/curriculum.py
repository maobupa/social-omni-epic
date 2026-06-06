"""Shared curriculum engine: the difficulty loop (Loop 1) + skill loop (Loop 2).

Used by both scripts/run_phase2.py (the curriculum run) and scripts/run_debug.py (single-scenario
debug), so the two-loop logic lives in exactly one place. `config` may be an OmegaConf DictConfig
or a plain dict — only `.get(...)` access is used. `run_single_episode` and
`scenario_to_sotopia_profiles` are passed in (so this module has no top-level sotopia import).
"""
from copy import deepcopy

from .data_models import SocialScenario
from .skills_chronicle import SkillsChronicle


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
    """Convert to sotopia profiles with the learner at index 0; return the partner profile
    (our AgentProfile, for the partner-perspective judge) and the success rubric."""
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


async def run_episode_two_loop(
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
) -> tuple:
    """Difficulty-calibration loop (D edits, ≤D+1 scenario versions) then skill-learning loop
    (K episode attempts: the biting failure is attempt 1, then up to K-1 reflection-driven retries).

    Returns (scenario, terminal_state, outcome_int, final_scores, loop_info).
    terminal_state ∈ {"discarded","solved_after_biting","failed"}; outcome_int 0/2/3.
    """
    from .episode_runner import clean_transcript

    K = int(config.get("max_attempts", 4))
    D = int(config.get("difficulty", {}).get("D", 2))
    judge_k = int(config.get("judge", {}).get("self_consistency_k", 3))
    max_entries = int(config.get("chronicle_max_entries", 8))
    re_reflect = bool(config.get("adversarial", {}).get("re_reflect_on_rejection", True))
    learner_model = config.get("learner_model")
    partner_model = config.get("partner_model")
    max_turns = int(config.get("max_turns", 20))

    current_chronicle = SkillsChronicle.from_markdown(anchor.skills_final_md or "") if anchor else SkillsChronicle()
    loop_info: dict = {
        "n_difficulty_edits": 0, "bit": False,
        "difficulty_loop": [], "skill_attempts": [], "terminal_state": "discarded", "outcome": 0,
    }

    async def _episode(scn: SocialScenario, chronicle: SkillsChronicle):
        env_profile, agent_profiles, learner_goal, partner_profile, rubric = build_episode_inputs(
            scn, scenario_to_sotopia_profiles
        )
        mem = chronicle.format_for_prompt(max_entries=max_entries)
        return await run_single_episode(
            env_profile=env_profile,
            agent_profiles=agent_profiles,
            fm=fm,
            learner_model=learner_model,
            partner_model=partner_model,
            memory_prompt=mem,
            max_turns=max_turns,
            learner_goal=learner_goal,
            rubric=rubric,
            partner_profile=partner_profile,
            judge_self_consistency_k=judge_k,
            on_turn=on_turn,
        )

    # ---- Loop 1: difficulty calibration — ratchet up until attempt-1 fails (bites) ----
    result = None
    bit = False
    for d in range(D + 1):  # initial run + up to D edits
        try:
            result = await _episode(scenario, current_chronicle)
        except Exception as e:
            import traceback
            print(f"    [difficulty d={d}] episode error: {e}\n{traceback.format_exc()}")
            return scenario, "discarded", 0, {}, loop_info
        rec: dict = {
            "d": d,
            "attempt1_solved": bool(result.goal_achieved),
            "rubric_results": result.rubric_results,
        }
        if not result.goal_achieved:
            bit = True
            loop_info["difficulty_loop"].append(rec)
            if on_attempt_done:
                on_attempt_done(loop_info)
            break
        if d >= D:
            loop_info["difficulty_loop"].append(rec)
            if on_attempt_done:
                on_attempt_done(loop_info)
            break  # used all edits, still too easy
        feedback = task_gen.analyze_too_easy(scenario, clean_transcript(result.transcript))
        edited = task_gen.edit_scenario(
            scenario, [feedback.get("suggested_edit", "")], intent="raise_difficulty"
        )
        edited, ok = (
            run_coherence_gate(edited, coherence_checker, task_gen, fm, config, anchor, scenario.iteration)
            if edited is not None else (None, False)
        )
        rec.update({
            "slack_knob": feedback.get("slack_knob"),
            "suggested_edit": feedback.get("suggested_edit"),
            "re_gate_passed": bool(ok),
            "edited_structured_goals": [
                sg.model_dump() if sg else None
                for sg in (edited.structured_goals if edited else [])
            ],
        })
        loop_info["difficulty_loop"].append(rec)
        if on_attempt_done:
            on_attempt_done(loop_info)
        if not ok or edited is None:
            break
        scenario = edited
        loop_info["n_difficulty_edits"] += 1

    loop_info["bit"] = bit
    if not bit:
        return scenario, "discarded", 0, {}, loop_info

    # ---- Loop 2: skill learning — reuse the biting attempt 1 ----
    all_transcripts: list = [clean_transcript(result.transcript)]
    all_scores: list = [{"attempt": 1, "scores": result.learner_scores, "solved": False}]
    all_versions: list = [deepcopy(current_chronicle)]
    all_edit_reasons: dict = {}
    final_scores: dict = result.learner_scores
    solved = False

    for attempt in range(1, K + 1):
        if attempt > 1:
            try:
                result = await _episode(scenario, current_chronicle)
            except Exception as e:
                import traceback
                print(f"    [attempt {attempt}] episode error: {e}\n{traceback.format_exc()}")
                break
            all_transcripts.append(clean_transcript(result.transcript))
            all_scores.append({"attempt": attempt, "scores": result.learner_scores, "solved": bool(result.goal_achieved)})
            final_scores = result.learner_scores
        att_rec: dict = {
            "attempt": attempt,
            "transcript_clean": all_transcripts[-1],
            "rubric_results": result.rubric_results,
            "diagnostics_scores": result.learner_scores,
            "solved": bool(result.goal_achieved),
        }
        if result.goal_achieved:
            loop_info["skill_attempts"].append(att_rec)
            if on_attempt_done:
                on_attempt_done(loop_info)
            solved = True
            break
        if attempt < K:
            ref_out = reflection_mod.reflect(
                chronicle=current_chronicle, scenario=scenario, transcripts=all_transcripts,
                prior_edit_reasons=all_edit_reasons, attempt_num=attempt, anchor_task=anchor,
                rubric_results=result.rubric_results,
            )
            adv_result = adversarial.check_reflection(
                ref_out, all_transcripts[-1], anchor_task=anchor, scenario=scenario
            )
            if not adv_result.approved and re_reflect:
                ref_out = reflection_mod.synthesize_with_critique(
                    reflection_output=ref_out, adversarial_critique=adv_result.critique,
                    chronicle=current_chronicle, scenario=scenario, transcripts=all_transcripts,
                    prior_edit_reasons=all_edit_reasons, attempt_num=attempt, anchor_task=anchor,
                )
            att_rec["reflection_diagnosis"] = ref_out.diagnosis
            att_rec["reflection_edit_reasons"] = ref_out.edit_reasons
            att_rec["adversarial_approved"] = adv_result.approved
            current_chronicle = ref_out.updated_chronicle
            all_versions.append(deepcopy(current_chronicle))
            all_edit_reasons.update(ref_out.edit_reasons)
        loop_info["skill_attempts"].append(att_rec)
        if on_attempt_done:
            on_attempt_done(loop_info)

    outcome = 2 if solved else 3
    terminal_state = "solved_after_biting" if solved else "failed"
    loop_info["terminal_state"] = terminal_state
    loop_info["outcome"] = outcome

    # Meta-reflection runs for both success (consolidate) and failure (document the trap).
    final_chronicle = meta_mod.synthesize(
        chronicle_versions=all_versions, transcripts=all_transcripts, edit_reasons=all_edit_reasons,
        outcome=outcome, scenario=scenario, anchor_task=anchor, attempt_scores=all_scores,
    )
    loop_info["final_chronicle_md"] = final_chronicle.to_markdown()

    title_data = title_gen.generate(scenario, scenario.target_agent_idx)
    scenario.scenario_title = title_data["scenario_title"]
    scenario.social_dynamic = title_data["social_dynamic"]
    scenario.target_perspective = title_data["target_perspective"]
    scenario.skills_final_md = final_chronicle.to_markdown()
    scenario.goal_score = float(final_scores.get("goal", 0.0))

    return scenario, terminal_state, outcome, final_scores, loop_info
