"""Single-scenario debug runner for Social Omni Epic Phase 2.

Mirrors the exact pipeline of scripts/run_phase2.py for one scenario:
  1  Load seeds & select anchor
  2  Verbalized sampling generation
  3  Embed
  4  MoI (novelty + learnability)
  5  Coherence gate  (with feedback-retry loop)
  6  Diversity gate  (cosine similarity vs archive)
  7  Structural classification
  8  Designate target agent
  9  Inherit skills chronicle
  10 Multi-attempt episode loop
       10a  Run episode (or mock with --skip-episode)
       10b  Reflection
       10c  Adversarial check on reflection (+ re-reflect on rejection)
  11 Meta-reflection
  12 Adversarial final check
  13 SCENARIO_TITLE generation
  14 Summary

Saves a complete debug JSON log to --output-dir.

Run from project root:
  python scripts/run_debug.py --skip-episode            # no Sotopia needed
  python scripts/run_debug.py --seed-index 3            # full run, seed #3 as anchor
  python scripts/run_debug.py --no-show-prompts         # hide LLM prompts
  python scripts/run_debug.py --seed-limit 5            # load only 5 seed rows (10 entries)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import numpy as np

from social_omni_epic.adversarial_agent import AdversarialAgent
from social_omni_epic.archive import Archive
from social_omni_epic.coherence_check import CoherenceChecker
from social_omni_epic.embedding_utils import get_similar_scenarios
from social_omni_epic.meta_reflection import MetaReflectionModule
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.reflection_module import ReflectionModule
from social_omni_epic.scenario_title import (
    ScenarioTitleGenerator,
    designate_target_agent,
)
from social_omni_epic.seeds import load_sotopia_seeds_with_embeddings
from social_omni_epic.episode_runner import clean_transcript as _clean_transcript
from social_omni_epic.skills_chronicle import SkillsChronicle
from social_omni_epic.task_generator import TaskGenerator
from social_omni_epic.tracing_fm import (
    TracingFM,
    print_info,
    print_section,
    print_step,
    print_warn,
)


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def _scenario_dict(scenario) -> dict:
    """Serialize scenario to a debug-friendly dict including full agent profiles."""
    return {
        "id": scenario.id,
        "scenario": scenario.scenario,
        "interaction_type": scenario.interaction_type,
        "relationship": scenario.relationship,
        "relationship_background": scenario.relationship_background,
        "agent_goals": scenario.agent_goals,
        "agent_profiles": [
            {
                "name": f"{p.first_name} {p.last_name}".strip(),
                "age": p.age,
                "gender_identity": p.gender_identity,
                "occupation": p.occupation,
                "public_info": p.public_info,
                "secret": p.secret,
                "big_five": p.big_five,
                "mbti": getattr(p, "mbti", ""),
                "moral_values": p.moral_values,
                "schwartz_portrait_value": p.schwartz_portrait_value,
                "decision_making_style": p.decision_making_style,
            }
            for p in scenario.agent_profiles
        ],
    }


# ---------------------------------------------------------------------------
# Mock episode (for --skip-episode mode)
# ---------------------------------------------------------------------------

def _make_mock_episode_result(scenario, attempt: int):
    """Return a fake EpisodeResult-like object for testing without Sotopia."""
    from dataclasses import dataclass, field

    @dataclass
    class FakeResult:
        transcript: list[dict] = field(default_factory=list)
        learner_scores: dict = field(default_factory=dict)
        partner_scores: dict = field(default_factory=dict)
        num_turns: int = 0
        raw_log: object = None
        evaluation_reasoning: str = ""

    transcript = [
        {"turn": 1, "sender": "Agent0", "receiver": "Agent1",
         "content": "[mock] Hello, I wanted to discuss the matter at hand."},
        {"turn": 2, "sender": "Agent1", "receiver": "Agent0",
         "content": "[mock] I'm afraid we have quite different perspectives on this."},
        {"turn": 3, "sender": "Agent0", "receiver": "Agent1",
         "content": "[mock] Perhaps we can find common ground?"},
        {"turn": 4, "sender": "Agent1", "receiver": "Agent0",
         "content": "[mock] I don't think that's possible today."},
    ]
    # First attempt always fails; second attempt succeeds (for testing multi-attempt)
    goal_score = 8.5 if attempt >= 2 else 3.0
    scores = {
        "believability": 7.0,
        "relationship": 5.0,
        "knowledge": 5.0,
        "secret": 5.0,
        "social_rules": 6.0,
        "financial_and_material_benefits": 5.0,
        "goal": goal_score,
        "overall_score": (goal_score + 5.0 * 5) / 6,
    }
    return FakeResult(transcript=transcript, learner_scores=scores, num_turns=4,
                      evaluation_reasoning="[mock evaluation]")


# ---------------------------------------------------------------------------
# Debug pipeline
# ---------------------------------------------------------------------------

def run_debug_pipeline(args, out_path: Path) -> dict:
    """Run the full single-scenario debug pipeline. Returns debug_output dict."""

    def _flush(d: dict) -> None:
        """Write current debug_output state to disk so it can be tailed live."""
        out_path.write_text(json.dumps(d, indent=2, default=str))

    debug_output: dict = {
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
        "anchor": {},
        "generated_scenario": {},
        "moi_result": {},
        "target_agent_idx": 0,
        "target_agent_goal_abstract": "",
        "inherited_chronicle": {},
        "episode_results": [],
        "reflection_outputs": [],
        "adversarial_reflection_results": [],
        "meta_reflection": {},
        "adversarial_final": {},
        "scenario_title": "",
        "outcome": 3,
        "final_scores": {},
        "llm_traces": [],
    }

    # -----------------------------------------------------------------------
    # Step 1: Load seeds and set up archive
    # -----------------------------------------------------------------------
    print_step("Step 1: Load Seeds & Select Anchor")

    tfm = TracingFM(
        model=args.model,
        show_prompts=args.show_prompts,
        show_responses=args.show_responses,
        max_prompt_chars=args.max_chars,
    )

    # Tiny in-memory archive seeded with the loaded seeds
    archive = Archive(checkpoint_dir=args.output_dir)

    try:
        tfm.set_step("Step 1b: Embed seed scenarios")
        seeds = load_sotopia_seeds_with_embeddings(
            fm=tfm,
            seeds_path=args.seeds_path,
            limit=args.seed_limit,
            both_perspectives=True,
        )
    except FileNotFoundError as e:
        print_warn(f"Seeds not found ({e}). Using an empty archive — generation will be unconditioned.")
        seeds = []

    if seeds:
        for scn in seeds:
            archive.add_successful(scn)
        print_info(f"Loaded {len(seeds)} seeds (embeddings from cache or freshly computed). Archive size: {archive.size}.")

    # Pick anchor
    if archive.size == 0:
        print_warn("No seeds loaded — will run unconditioned generation, no episode anchor.")
        anchor = None
        anchor_idx = -1
    else:
        seed_idx = min(args.seed_index, archive.size - 1)
        anchor = archive.state.successful[seed_idx]
        anchor_idx = seed_idx
        inherited_entries = len(
            SkillsChronicle.from_markdown(anchor.skills_final_md or "").entries
        )
        print_info(
            f"Anchor: seed[{seed_idx}] '{anchor.scenario[:80]}...'\n"
            f"  skills_final_md: {inherited_entries} inherited chronicle entries"
        )
        debug_output["anchor"] = {
            "index": seed_idx,
            "id": anchor.id,
            "scenario": anchor.scenario,
            "interaction_type": anchor.interaction_type,
            "inherited_chronicle_entries": inherited_entries,
        }
        if anchor.skills_final_md:
            print_section(
                "Inherited Skills Chronicle",
                SkillsChronicle.from_markdown(anchor.skills_final_md).to_markdown() or "(empty)"
            )

    _flush(debug_output)

    # -----------------------------------------------------------------------
    # Step 2: Scenario generation (verbalized sampling)
    # -----------------------------------------------------------------------
    tfm.set_step("Step 2: Verbalized Sampling Scenario Generation")

    task_gen = TaskGenerator(tfm, num_examples=3, num_failed_examples=0, max_retries=2)
    existing_types = (
        list({s.interaction_type for s in archive.state.successful if s.interaction_type})
        if archive.size > 0 else []
    )

    examples = []
    if anchor and anchor.embedding:
        all_embs = archive.get_successful_embeddings()
        source_ids = [s.source_scenario_id for s in archive.state.successful]
        agent_idxs = [s.target_agent_idx for s in archive.state.successful]
        idxs = get_similar_scenarios(
            anchor.embedding, all_embs, num_returns=3,
            source_ids=source_ids, agent_idxs=agent_idxs,
            preferred_agent_idx=anchor.target_agent_idx,
        )
        examples = [archive.state.successful[i] for i in idxs]

    scenario = task_gen.generate_with_verbalized_sampling(
        examples, existing_types=existing_types, n_candidates=args.vs_candidates
    )
    if scenario is None:
        print_warn("Verbalized sampling failed — trying unconditioned generation.")
        scenario = task_gen.generate_unconditioned()
    if scenario is None:
        print_warn("Generation failed entirely. Aborting.")
        debug_output["llm_traces"] = tfm.get_traces()
        return debug_output

    print_info(f"Generated: '{scenario.scenario[:100]}'")
    print_info(f"  interaction_type: {scenario.interaction_type}")
    print_info(f"  relationship: {scenario.relationship}")
    debug_output["generated_scenario"] = _scenario_dict(scenario)
    _flush(debug_output)

    # -----------------------------------------------------------------------
    # Step 3: Embed
    # -----------------------------------------------------------------------
    tfm.set_step("Step 3: Embed Generated Scenario")
    scenario.embedding = tfm.get_embeddings([scenario.to_text_for_embedding()])[0]
    dim = len(scenario.embedding)
    print_info(f"Embedding dim: {dim}")

    # -----------------------------------------------------------------------
    # Step 4: MoI evaluation
    # -----------------------------------------------------------------------
    tfm.set_step("Step 4: Model of Interestingness")
    moi = ModelOfInterestingness(tfm, num_examples=5)
    similar: list = []
    if archive.size > 0:
        _src_ids = [s.source_scenario_id for s in archive.state.successful]
        _agt_idxs = [s.target_agent_idx for s in archive.state.successful]
        sim_idxs = get_similar_scenarios(
            scenario.embedding, archive.get_successful_embeddings(), num_returns=5,
            source_ids=_src_ids, agent_idxs=_agt_idxs,
            preferred_agent_idx=scenario.target_agent_idx,
        )
        similar = [archive.state.successful[i] for i in sim_idxs]
    is_interesting, moi_reason = moi.evaluate(scenario, similar)
    print_info(f"MoI: passed={is_interesting}")
    print_info(f"  reason: {moi_reason[:200]}")
    debug_output["moi_result"] = {"passed": is_interesting, "reason": moi_reason}
    if not is_interesting:
        print_warn("MoI rejected scenario. Continuing anyway for debug purposes.")
    _flush(debug_output)

    # -----------------------------------------------------------------------
    # Step 5: Coherence gate  (mirrors run_phase2.py step 5)
    # -----------------------------------------------------------------------
    tfm.set_step("Step 5: Coherence Gate")
    coherence_checker = CoherenceChecker(tfm)
    coherence_feedback = None
    passed_coherence = False
    for _c in range(args.coherence_max_retries + 1):
        c_result = coherence_checker.check(scenario)
        print_info(f"Coherence check #{_c + 1}: passed={c_result.passed}")
        if c_result.issues:
            for issue in c_result.issues:
                print_warn(f"  issue: {issue}")
        if c_result.passed:
            passed_coherence = True
            break
        coherence_feedback = c_result.issues
        print_warn(f"Coherence failed — patching scenario (retry {_c + 1}/{args.coherence_max_retries})")
        scenario = task_gen.patch_scenario(scenario, coherence_feedback)
        if scenario is None:
            print_warn("Regeneration returned None — aborting coherence retry loop.")
            break
        scenario.iteration = 0
        scenario.parent_example_ids = [anchor.id] if anchor else []
        try:
            scenario.embedding = tfm.get_embeddings([scenario.to_text_for_embedding()])[0]
        except Exception as e:
            print_warn(f"Re-embed failed: {e}")
            scenario = None
            break

    debug_output["coherence_result"] = {
        "passed": passed_coherence,
        "retries": _c,
        "final_issues": coherence_feedback or [],
    }
    # Update generated_scenario if coherence retry replaced the scenario
    if scenario is not None:
        debug_output["generated_scenario"] = _scenario_dict(scenario)
        _flush(debug_output)
    if not passed_coherence or scenario is None:
        print_warn("Coherence gate FAILED after all retries. Continuing anyway for debug purposes.")

    # -----------------------------------------------------------------------
    # Step 6: Diversity gate  (mirrors run_phase2.py step 6)
    # -----------------------------------------------------------------------
    print_step("Step 6: Diversity Gate")
    diversity_threshold = args.diversity_threshold
    diversity_passed = True
    max_sim = 0.0
    if scenario is not None and scenario.embedding and archive.size > 0:
        all_embs = archive.get_successful_embeddings()
        if all_embs:
            emb_arr = np.array(all_embs)
            s_emb = np.array(scenario.embedding)
            sims = emb_arr @ s_emb / (
                np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(s_emb) + 1e-9
            )
            max_sim = float(sims.max())
            diversity_passed = max_sim <= diversity_threshold
    print_info(f"Diversity: max_cosine_sim={max_sim:.4f}  threshold={diversity_threshold}  passed={diversity_passed}")
    if not diversity_passed:
        print_warn("Diversity gate FAILED (too similar to archive). Continuing anyway for debug purposes.")
    debug_output["diversity_result"] = {
        "passed": diversity_passed,
        "max_cosine_sim": max_sim,
        "threshold": diversity_threshold,
    }
    _flush(debug_output)

    # -----------------------------------------------------------------------
    # Step 7: Designate target agent
    # -----------------------------------------------------------------------
    tfm.set_step("Step 7: Designate Target Agent")
    if anchor:
        scenario.target_agent_idx, scenario.target_agent_goal_abstract = (
            designate_target_agent(scenario, anchor, tfm)
        )
    else:
        scenario.target_agent_idx = 0
        scenario.target_agent_goal_abstract = ""
    print_info(
        f"target_agent_idx={scenario.target_agent_idx}  "
        f"abstract_goal='{scenario.target_agent_goal_abstract[:80]}'"
    )
    debug_output["target_agent_idx"] = scenario.target_agent_idx
    debug_output["target_agent_goal_abstract"] = scenario.target_agent_goal_abstract

    # -----------------------------------------------------------------------
    # Step 9: Inherit chronicle
    # -----------------------------------------------------------------------
    print_step("Step 9: Inherit Skills Chronicle")
    current_chronicle = SkillsChronicle.from_markdown(
        (anchor.skills_final_md or "") if anchor else ""
    )
    n_inherited = len(current_chronicle.entries)
    print_info(f"Inherited {n_inherited} entries from anchor.")
    if current_chronicle.entries:
        print_section("Inherited Chronicle", current_chronicle.to_markdown())
    debug_output["inherited_chronicle"] = {
        "entry_count": n_inherited,
        "entries": [e.entry_id for e in current_chronicle.entries],
    }

    # -----------------------------------------------------------------------
    # Episode loop
    # -----------------------------------------------------------------------
    reflection_mod = ReflectionModule(tfm, max_retries=1)
    adversarial = AdversarialAgent(tfm)
    from social_omni_epic.success_detector import SuccessDetector
    success_detector = SuccessDetector(goal_threshold=args.goal_threshold)

    all_transcripts: list[list[dict]] = []
    all_scores: list[dict] = []
    all_versions: list[SkillsChronicle] = [deepcopy(current_chronicle)]
    all_edit_reasons: dict[str, str] = {}
    outcome = 3
    final_scores: dict = {}

    for attempt in range(1, args.max_attempts + 1):
        # -------------------------------------------------------------------
        # Step 10a: Run episode
        # -------------------------------------------------------------------
        tfm.set_step(f"Step 10a (attempt {attempt}): Run Episode")
        memory_prompt = current_chronicle.format_for_prompt(max_entries=8)
        if memory_prompt:
            print_section("Memory injected into agent", memory_prompt[:1000])

        if args.skip_episode:
            print_warn("--skip-episode: using mock transcript")
            episode_result = _make_mock_episode_result(scenario, attempt)
        else:
            try:
                from social_omni_epic.episode_runner import run_single_episode
                from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles
                env_profile, agent_profiles = scenario_to_sotopia_profiles(scenario)
                def _on_turn(partial_transcript: list[dict]) -> None:
                    debug_output["episode_results_partial"] = {
                        "attempt": attempt,
                        "transcript_so_far": _clean_transcript(partial_transcript),
                    }
                    _flush(debug_output)

                episode_result = asyncio.run(
                    run_single_episode(
                        env_profile=env_profile,
                        agent_profiles=agent_profiles,
                        fm=tfm,
                        learner_model=args.learner_model,
                        partner_model=args.partner_model,
                        memory_prompt=memory_prompt,
                        max_turns=args.max_turns,
                        on_turn=_on_turn,
                    )
                )
            except Exception as e:
                import traceback
                err_msg = traceback.format_exc()
                print_warn(f"Episode failed: {e}")
                debug_output.setdefault("episode_errors", []).append({
                    "attempt": attempt,
                    "error": str(e),
                    "traceback": err_msg,
                })
                _flush(debug_output)
                break

        debug_output.pop("episode_results_partial", None)
        clean = _clean_transcript(episode_result.transcript)
        all_transcripts.append(clean)
        final_scores = episode_result.learner_scores
        all_scores.append({"attempt": attempt, "scores": final_scores})

        # Display transcript
        transcript_text = "\n".join(
            f"[T{t['turn']}] {t['speaker']}: {t['content']}"
            for t in clean
        )
        print_section(f"Transcript (attempt {attempt})", transcript_text)

        # Scores table
        scores_text = "\n".join(
            f"  {k:38s}: {v:.1f}" for k, v in final_scores.items()
        )
        print_section("Scores", scores_text)

        solved = success_detector.is_solved(final_scores)
        print_info(f"Solved: {solved}  (goal={final_scores.get('goal', 0):.1f})")

        debug_output["episode_results"].append({
            "attempt": attempt,
            "transcript_clean": _clean_transcript(episode_result.transcript),
            "scores": final_scores,
            "solved": solved,
        })
        _flush(debug_output)

        if solved:
            outcome = 1 if attempt == 1 else 2
            final_scores = episode_result.learner_scores
            break

        if attempt >= args.max_attempts:
            break

        # -------------------------------------------------------------------
        # Step 10b: Reflection
        # -------------------------------------------------------------------
        tfm.set_step(f"Step 10b (attempt {attempt}): Reflection")
        ref_out = reflection_mod.reflect(
            chronicle=current_chronicle,
            scenario=scenario,
            transcripts=all_transcripts,
            prior_edit_reasons=all_edit_reasons,
            attempt_num=attempt,
            anchor_task=anchor,
        )
        print_info(f"Diagnosis: {ref_out.diagnosis[:300]}")
        if ref_out.edit_reasons:
            reasons_text = "\n".join(
                f"  [{eid}]: {r[:120]}" for eid, r in ref_out.edit_reasons.items()
            )
            print_section("Edit Reasons", reasons_text)
        if ref_out.misdirection_entry_ids:
            print_warn(f"Misdirection flags: {ref_out.misdirection_entry_ids}")

        debug_output["reflection_outputs"].append({
            "attempt": attempt,
            "diagnosis": ref_out.diagnosis,
            "edit_reasons": ref_out.edit_reasons,
            "misdirection_entry_ids": ref_out.misdirection_entry_ids,
            "updated_chronicle_entries": len(ref_out.updated_chronicle.entries),
        })

        # -------------------------------------------------------------------
        # Step 10c: Adversarial check on reflection
        # -------------------------------------------------------------------
        tfm.set_step(f"Step 10c (attempt {attempt}): Adversarial Check (Reflection)")
        adv_result = adversarial.check_reflection(
            ref_out, clean, anchor_task=anchor, scenario=scenario
        )
        status = "APPROVED" if adv_result.approved else "REJECTED"
        print_info(f"Adversarial: {status}")
        if adv_result.issues:
            print_section("Issues", "\n".join(f"  - {i}" for i in adv_result.issues))
        if not adv_result.approved and adv_result.critique:
            print_warn(f"Critique: {adv_result.critique[:300]}")
            print_warn("Synthesizing reflection + critique...")
            tfm.set_step(f"Step 10c (attempt {attempt}): Synthesis")
            ref_out = reflection_mod.synthesize_with_critique(
                reflection_output=ref_out,
                adversarial_critique=adv_result.critique,
                chronicle=current_chronicle,
                scenario=scenario,
                transcripts=all_transcripts,
                prior_edit_reasons=all_edit_reasons,
                attempt_num=attempt,
                anchor_task=anchor,
            )
            print_info("Synthesis complete.")

        debug_output["adversarial_reflection_results"].append({
            "attempt": attempt,
            "approved": adv_result.approved,
            "issues": adv_result.issues,
            "flagged_entry_ids": adv_result.flagged_entry_ids,
            "critique": adv_result.critique,
        })

        current_chronicle = ref_out.updated_chronicle
        all_versions.append(deepcopy(current_chronicle))
        all_edit_reasons.update(ref_out.edit_reasons)

    debug_output["outcome"] = outcome
    debug_output["final_scores"] = final_scores
    _flush(debug_output)

    # -----------------------------------------------------------------------
    # Step 11: Meta-reflection
    # -----------------------------------------------------------------------
    meta_mod = MetaReflectionModule(tfm, max_retries=1)
    if outcome == 1:
        print_step("Step 11: Meta-Reflection (skipped — solved on first attempt)")
        final_chronicle = current_chronicle
    elif not all_transcripts:
        print_step("Step 11: Meta-Reflection (skipped — no transcripts to learn from)")
        final_chronicle = current_chronicle
    else:
        tfm.set_step(f"Step 11: Meta-Reflection (outcome={outcome})")
        final_chronicle = meta_mod.synthesize(
            chronicle_versions=all_versions,
            transcripts=all_transcripts,
            edit_reasons=all_edit_reasons,
            outcome=outcome,
            scenario=scenario,
            anchor_task=anchor,
            attempt_scores=all_scores,
        )
        n_final = len(final_chronicle.entries)
        print_info(f"Meta-reflection produced {n_final} final chronicle entries.")
        if final_chronicle.entries:
            print_section("Final Chronicle (pre-adversarial)", final_chronicle.to_markdown())

    debug_output["meta_reflection"] = {
        "entry_count": len(final_chronicle.entries),
        "entries": [
            {
                "id": e.entry_id,
                "type": e.entry_type,
                "dimension": e.dimension,
                "condition": e.condition,
                "guidance": e.guidance,
                "provenance": e.provenance,
            }
            for e in final_chronicle.entries
        ],
    }
    _flush(debug_output)

    # Step 12 removed — adversarial final check dropped (meta-reflection entry-level
    # checks already validated individually; redundant final check had no downstream action)

    # -----------------------------------------------------------------------
    # Step 13: SCENARIO_TITLE
    # -----------------------------------------------------------------------
    tfm.set_step("Step 13: SCENARIO_TITLE Generation")
    title_gen = ScenarioTitleGenerator(tfm)
    title_data = title_gen.generate(scenario, scenario.target_agent_idx)
    scenario.scenario_title = title_data["scenario_title"]
    scenario.social_dynamic = title_data["social_dynamic"]
    scenario.target_perspective = title_data["target_perspective"]
    print_info(f"Title: {scenario.scenario_title}")
    debug_output["scenario_title"] = scenario.scenario_title
    debug_output["social_dynamic"] = scenario.social_dynamic
    debug_output["target_perspective"] = scenario.target_perspective

    # -----------------------------------------------------------------------
    # Step 14: Summary
    # -----------------------------------------------------------------------
    traces = tfm.get_traces()
    debug_output["llm_traces"] = traces

    outcome_labels = {1: "SOLVED (attempt 1)", 2: "SOLVED (multi-attempt)", 3: "FAILED"}
    total_elapsed = sum(t["elapsed_ms"] for t in traces)

    summary = (
        f"Outcome:     {outcome} — {outcome_labels.get(outcome, '?')}\n"
        f"LLM calls:   {len(traces)}\n"
        f"LLM time:    {total_elapsed/1000:.1f}s\n"
        f"Final goal:  {final_scores.get('goal', 0):.1f}\n"
        f"Scenario:    {scenario.scenario[:100]}"
    )
    print_section("Debug Summary", summary)

    return debug_output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-scenario debug runner for Social Omni Epic Phase 2"
    )
    parser.add_argument("--seed-index", type=int, default=0,
                        help="Which seed to use as anchor (default: 0)")
    parser.add_argument("--seed-limit", type=int, default=None,
                        help="How many seed rows to load (default: all 90 → 180 entries)")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="Max episode attempts (default: 2)")
    parser.add_argument("--vs-candidates", type=int, default=5,
                        help="Verbalized sampling candidates (default: 5)")
    parser.add_argument("--show-prompts", action=argparse.BooleanOptionalAction, default=True,
                        help="Show LLM prompts (default: True; --no-show-prompts to hide)")
    parser.add_argument("--show-responses", action=argparse.BooleanOptionalAction, default=True,
                        help="Show LLM responses (default: True)")
    parser.add_argument("--max-chars", type=int, default=800,
                        help="Truncate prompts/responses at N chars (default: 800)")
    parser.add_argument("--skip-episode", action="store_true",
                        help="Skip real Sotopia episode; use mock transcript")
    parser.add_argument("--goal-threshold", type=float, default=7.0)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--learner-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--partner-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--evaluator-model", type=str, default="openai/gpt-5-mini")
    parser.add_argument("--seeds-path", type=str, default="data/sotopia_90_seeds.jsonl")
    parser.add_argument("--coherence-max-retries", type=int, default=2,
                        help="Max coherence-gate regeneration attempts (default: 2)")
    parser.add_argument("--diversity-threshold", type=float, default=0.92,
                        help="Cosine similarity threshold for diversity gate (default: 0.92)")
    parser.add_argument("--output-dir", type=str, default="debug_log")
    args = parser.parse_args()

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY (or OPENAI_API_KEY) not set.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"debug_{timestamp}.json"
    out_path.write_text("{}")  # create file immediately so it can be opened before pipeline ends
    print(f"Debug log: {out_path}  (updates live after each step)")

    t_start = time.monotonic()
    debug_output = run_debug_pipeline(args, out_path)
    elapsed = time.monotonic() - t_start

    # Final write includes llm_traces which are only added at the very end
    out_path.write_text(json.dumps(debug_output, indent=2, default=str))
    print(f"\nDebug log finalised: {out_path}  (total wall time: {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
