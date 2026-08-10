"""Shared curriculum engine: single K-attempt skill loop (Phase 2).

Loop 1 (difficulty calibration) is gone. Every scenario goes straight into the K-attempt
skill loop. Attempt-1 success → too_easy classification; post-loop LP computation determines
frontier vs. beyond_frontier. All classification results propagate to the scenario object so
archive.py's Thompson sampler can consume them directly.

Used by scripts/run_curriculum.py, scripts/run_debug.py, and scripts/run_phase2.py.
`config` may be an OmegaConf DictConfig or a plain dict — only `.get(...)` access is used.
"""
from copy import deepcopy

from .data_models import SocialScenario, K_VOTES_EQUIV
from .skills_chronicle import SkillsChronicle

# K_VOTES_EQUIV (defined in data_models) = equivalent total_votes charged to too_easy
# children in the Thompson posterior. Re-exported here for back-compat importers.


def run_coherence_gate(
    scenario, coherence_checker, task_gen, fm, config, anchor, iteration,
) -> tuple:
    """Run the coherence check with patch-retry and re-embed.

    Returns (scenario_or_None, passed, issues) where issues is the list of
    failure reasons from the last check (empty on pass).
    """
    from .tracing_fm import print_info, print_warn
    if not config.get("enable_coherence_check", True):
        return scenario, True, []
    max_coherence = int(config.get("coherence_max_retries", 2))
    last_issues: list[str] = []
    for attempt in range(max_coherence + 1):
        c_result = coherence_checker.check(scenario)
        last_issues = c_result.issues
        if c_result.passed:
            if attempt > 0:
                print_info(f"  coherence: PASS after {attempt} patch(es)")
            return scenario, True, []
        is_fm_error = any("FM error" in i or "quarantined" in i for i in last_issues)
        label = "FM error" if is_fm_error else "issues"
        print_warn(f"  coherence attempt {attempt}: FAIL ({label}): {last_issues[0][:120]}")
        if attempt == max_coherence:
            break
        scenario = task_gen.patch_scenario(scenario, last_issues)
        if scenario is None:
            print_warn(f"  coherence: patch_scenario returned None — aborting")
            return None, False, last_issues
        scenario.iteration = iteration
        scenario.parent_example_ids = [anchor.id] if anchor else []
        try:
            scenario.embedding = fm.get_embeddings([scenario.to_text_for_embedding()])[0]
        except Exception as e:
            print_warn(f"  coherence: re-embed after patch failed: {e}")
            return None, False, last_issues
    print_warn(f"  coherence: FAIL after {max_coherence + 1} attempt(s) — dropping candidate")
    return scenario, False, last_issues


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
    fm_reflection=None,  # writes the Reflexion string the learner reads; see below. Falls back to fm.
) -> tuple:
    """K-attempt skill-learning loop with LP-based classification.

    Returns (scenario, terminal_state, outcome_int, final_scores, loop_info).
    terminal_state ∈ {"too_easy", "frontier", "beyond_frontier"} or "discarded" on error.
    outcome_int: 1=too_easy, 2=frontier_solved, 3=frontier_unsolved, 4=beyond_frontier.
    """
    from .episode_runner import clean_transcript
    from .lp_judge import compute_lp, LPResult, LPAllErrorsError

    K = int(config.get("max_attempts", 4))
    max_entries = int(config.get("chronicle_max_entries", 8))
    re_reflect = bool(config.get("adversarial", {}).get("re_reflect_on_rejection", True))
    learner_model = config.get("learner_model")
    partner_model = config.get("partner_model")
    max_turns = int(config.get("max_turns", 20))

    # Memory mechanism switch. ExpeL (default) replaces the Skills Chronicle in the
    # gen-90 loop with within-episode Reflexion strings; the chronicle modules stay
    # importable and are used when use_expel_memory is False.
    use_expel = bool(config.get("use_expel_memory", True))
    from .expel_baseline import _reflect, _format_reflections, _format_transcript

    # Chronicle is only constructed for the chronicle path. Under ExpeL the learner
    # starts attempt 1 with no memory (clean LP baseline); lineage knowledge is carried
    # to the generator via skills_final_md, not into the learner's within-episode memory.
    current_chronicle = (
        None if use_expel
        else (SkillsChronicle.from_markdown(anchor.skills_final_md or "") if anchor
              else SkillsChronicle())
    )
    loop_info: dict = {"skill_attempts": [], "terminal_state": None, "outcome": 0}

    _judge = fm_judge if fm_judge is not None else fm

    # The Reflexion writer must be the LEARNER's own model. Recoverability is defined as "did the
    # learner solve it after being told what went wrong" — if a stronger model writes the telling,
    # we measure the teacher, not the learner, and we do so *unequally* across learners (a weak
    # learner gains more from a strong teacher than a strong one does). In the matrix that would
    # bend the diagonal, which is the whole measurement. Gen-90 was safe only by coincidence:
    # config.model == learner_model there, so `fm` already was the learner.
    _reflect_fm = fm_reflection if fm_reflection is not None else fm

    # Learner goal text (stable across attempts) — needed for ExpeL reflection prompts.
    _, _, _learner_goal_text, _, _ = build_episode_inputs(scenario, scenario_to_sotopia_profiles)

    # Compute query embedding once for relevance-ranked chronicle truncation (§8.2).
    # Falls back to positional truncation when abstract goal or FM embedding unavailable.
    _query_embedding: list[float] | None = None
    if not use_expel:
        _abstract_goal_text = (scenario.target_agent_goal_abstract or
                               (scenario.agent_goals[0] if scenario.agent_goals else ""))
        if _abstract_goal_text:
            try:
                _query_embedding = fm.get_embeddings([_abstract_goal_text])[0]
            except Exception:
                pass

    def _chronicle_mem(chronicle: SkillsChronicle) -> str:
        return chronicle.format_for_prompt(
            max_entries=max_entries,
            query_embedding=_query_embedding,
            fm=fm,
        )

    async def _episode(scn: SocialScenario, memory_prompt: str):
        env_profile, agent_profiles, learner_goal, _partner_profile, _rubric = build_episode_inputs(
            scn, scenario_to_sotopia_profiles
        )
        return await run_single_episode(
            env_profile=env_profile,
            agent_profiles=agent_profiles,
            fm=fm,
            learner_model=learner_model,
            partner_model=partner_model,
            memory_prompt=memory_prompt,
            max_turns=max_turns,
            learner_goal=learner_goal,
            partner_key=scn.partner_key,
            fm_judge=_judge,
            on_turn=on_turn,
        )

    # ------------------------------------------------------------------ #
    # Attempt 1                                                            #
    # ------------------------------------------------------------------ #
    attempt1_mem = "" if use_expel else _chronicle_mem(current_chronicle)
    try:
        result = await _episode(scenario, attempt1_mem)
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

        # Skip synthesis: no learning occurred (solved immediately, nothing to reflect on).
        # ExpeL: empty memory (no reflexions gathered). Chronicle: pass the inherited
        # chronicle through unchanged so children see accurate lineage knowledge.
        # (Synthesizing here would route outcome=1 into the FAILURE branch — outcome==2 is
        # success — and produce WARNING-dominant memory for a trivially solved scenario.)
        scenario.skills_final_md = "" if use_expel else current_chronicle.to_markdown()
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
    all_key_checks: list = [result.key_check_result]
    final_scores: dict = result.learner_scores
    solved = False

    # Chronicle-path bookkeeping (unused under ExpeL).
    all_versions: list = [] if use_expel else [deepcopy(current_chronicle)]
    all_edit_reasons: dict = {}

    # ExpeL-path memory: within-episode Reflexion strings. Seed with a reflection on
    # attempt 1's failure (attempt 1 always failed here — the too_easy path returned
    # above). Attempt k>=2 sees the reflections from attempts 1..k-1.
    reflexion_strings: list[str] = []
    if use_expel and K >= 2:
        reflexion_strings.append(
            _reflect(_reflect_fm, task=scenario.scenario, learner_goal=_learner_goal_text,
                     transcript_text=_format_transcript(transcript1),
                     goal_score=float(result.learner_scores.get("goal", 0.0)))
        )

    for attempt in range(2, K + 1):
        if use_expel:
            mem = _format_reflections(reflexion_strings)
        else:
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
            mem = _chronicle_mem(current_chronicle)

        try:
            result = await _episode(scenario, mem)
        except Exception as e:
            import traceback
            print(f"    [attempt {attempt}] episode error: {e}\n{traceback.format_exc()}")
            loop_info["episode_error"] = f"attempt {attempt}: {e}"
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
        }
        if use_expel:
            att_rec["reflexion"] = reflexion_strings[-1] if reflexion_strings else ""
        else:
            att_rec.update({
                "reflection_diagnosis": ref_out.diagnosis,
                "reflection_edit_reasons": ref_out.edit_reasons,
                "adversarial_approved": adv_result.approved,
                "chronicle_after_reflection": current_chronicle.to_markdown(),
            })
        loop_info["skill_attempts"].append(att_rec)
        if on_attempt_done:
            on_attempt_done(loop_info)

        if result.terminal_success:
            solved = True
            break

        # ExpeL: reflect on this failure to inform the next attempt (skip after the last).
        if use_expel and attempt < K:
            reflexion_strings.append(
                _reflect(_reflect_fm, task=scenario.scenario, learner_goal=_learner_goal_text,
                         transcript_text=_format_transcript(transcript),
                         goal_score=float(result.learner_scores.get("goal", 0.0)))
            )

    # ------------------------------------------------------------------ #
    # Episode-error quarantine: an exception left LP uncomputable           #
    # ------------------------------------------------------------------ #
    # If the loop broke on an exception, the learner never solved, and fewer than 2 clean
    # transcripts exist, LP cannot be computed — do NOT charge the anchor a (0, K) penalty
    # for an infrastructure failure. Route to quarantine ("discarded"); the runner records
    # no votes, does not archive, and does not count it toward stopping.N.
    if loop_info.get("episode_error") and not solved and len(all_transcripts) < 2:
        loop_info["terminal_state"] = "discarded"
        loop_info["outcome"] = 0
        return scenario, "discarded", 0, final_scores, loop_info

    # ------------------------------------------------------------------ #
    # LP computation                                                       #
    # ------------------------------------------------------------------ #
    env_profile_lp, _, learner_goal_lp, _, _ = build_episode_inputs(scenario, scenario_to_sotopia_profiles)
    learner_goal_text = env_profile_lp.agent_goals[0] if env_profile_lp.agent_goals else ""
    # Prepend relationship label so the LP judge has context even when background is empty.
    _rel_label = scenario.relationship or ""
    _rel_bg = scenario.relationship_background or ""
    relational_stakes = f"{_rel_label}: {_rel_bg}".strip(": ") if _rel_label else _rel_bg

    # Optional saving: LP cannot change the band of a solved scenario. Classification below is
    # `solved or lp > 0 -> frontier`, so for a solved episode the LP votes are computed and then
    # discarded. Gen-90 solved 21 scenarios on a retry, i.e. ~126 judge votes spent for nothing.
    # Default OFF because the evolutionary loop feeds LP votes into the Thompson posterior, where
    # they do matter; the grid has no posterior, so it turns this on.
    _skip_lp = solved and bool(config.get("skip_lp_when_solved", False))
    if _skip_lp:
        lp_result = LPResult(lp_value=0.0, improved_votes=0, total_votes=0, n_pairs=0)
        loop_info["lp_skipped"] = "solved"
    else:
        try:
            lp_result = await compute_lp(
                fm_judge=_judge,
                scenario=scenario,
                transcripts=all_transcripts,
                learner_goal=learner_goal_text,
                relational_stakes=relational_stakes,
                # Second axis (logged, not gated): progress toward apprehending the partner. Only
                # available on keyed v2 scenarios; None for seeds and v1 replays.
                internal_state=(scenario.partner_key.internal_state
                                if scenario.partner_key is not None else None),
            )
        except LPAllErrorsError as e:
            # Every judge vote errored (even after the one-shot revote) → LP uncomputable.
            # Quarantine instead of misclassifying as beyond_frontier and charging the anchor.
            print(f"    [LP] all judge votes errored: {e} → quarantine")
            loop_info["episode_error"] = f"lp_all_errors: {e}"
            loop_info["terminal_state"] = "discarded"
            loop_info["outcome"] = 0
            return scenario, "discarded", 0, final_scores, loop_info
        except Exception as e:
            print(f"    [LP] compute_lp error: {e}")
            lp_result = LPResult(lp_value=0.0, improved_votes=0, total_votes=0, n_pairs=0)

    loop_info["lp_value"] = lp_result.lp_value
    loop_info["lp_inference"] = lp_result.lp_inference
    loop_info["lp_votes"] = lp_result.total_votes
    loop_info["lp_improved_votes"] = lp_result.improved_votes
    loop_info["n_error_votes"] = lp_result.n_error_votes

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

    # Beyond-frontier diagnosis for structured relax targeting (pure heuristic, no LLM call).
    if classification == "beyond_frontier":
        scenario.beyond_frontier_diagnosis = task_gen.analyze_beyond_frontier(
            scenario, all_key_checks, all_scores
        )
        loop_info["beyond_frontier_diagnosis"] = scenario.beyond_frontier_diagnosis

    # ------------------------------------------------------------------ #
    # Final memory synthesis                                               #
    # ------------------------------------------------------------------ #
    if use_expel:
        # ExpeL: the accumulated within-episode Reflexion strings ARE the lineage
        # memory the task generator reads from skills_final_md. No adversarial/meta pass.
        scenario.skills_final_md = _format_reflections(reflexion_strings)
        loop_info["final_chronicle_md"] = scenario.skills_final_md
    else:
        from .skills_chronicle import validate_synthesis

        first_failed_tx = all_transcripts[0] if all_transcripts else None

        final_chronicle = meta_mod.synthesize(
            chronicle_versions=all_versions, transcripts=all_transcripts, edit_reasons=all_edit_reasons,
            outcome=outcome, scenario=scenario, anchor_task=anchor, attempt_scores=all_scores,
            lp_votes=lp_result.votes,
        )

        inherited_md = (anchor.skills_final_md or "") if anchor else ""
        prog_issues = validate_synthesis(final_chronicle)
        adv_final = adversarial.check_final(
            final_chronicle, inherited_md, outcome=outcome,
            first_failed_transcript=first_failed_tx if outcome == 2 else None,
        )

        if (not adv_final.approved) or prog_issues:
            combined_critique = "\n".join(
                ([adv_final.critique] if adv_final.critique else []) + prog_issues
            )
            final_chronicle = meta_mod.synthesize(
                chronicle_versions=all_versions, transcripts=all_transcripts,
                edit_reasons=all_edit_reasons, outcome=outcome, scenario=scenario,
                anchor_task=anchor, attempt_scores=all_scores,
                lp_votes=lp_result.votes, adversarial_critique=combined_critique,
            )
            prog_issues2 = validate_synthesis(final_chronicle)
            adv_final2 = adversarial.check_final(
                final_chronicle, inherited_md, outcome=outcome,
                first_failed_transcript=first_failed_tx if outcome == 2 else None,
            )
            remaining = (adv_final2.issues if not adv_final2.approved else []) + prog_issues2
            if remaining:
                loop_info["final_check_flag"] = remaining
                scenario.final_check_flag = remaining

        loop_info["final_chronicle_md"] = final_chronicle.to_markdown()
        scenario.skills_final_md = final_chronicle.to_markdown()

    scenario.goal_score = float(final_scores.get("goal", 0.0))
    scenario.goal_trajectory = [float(s["scores"].get("goal", 0.0)) for s in all_scores]
    _set_titles(scenario, title_gen)

    return scenario, classification, outcome, final_scores, loop_info


# Backward-compat alias for callers that import the old name.
run_episode_two_loop = run_episode_k_loop
