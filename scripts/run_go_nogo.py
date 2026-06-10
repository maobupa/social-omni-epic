"""Go/no-go experiment: does the methodology have headroom + a reflect-act gap?

This is the existential checkpoint (evaluation_methodology.md §3b). On a small batch of freshly
generated *sweet-spot* dev scenarios (NOT the sealed external set), it measures:

  1. HEADROOM    — what fraction does the naive learner (no chronicle) fail, judged by the rubric?
  2. REFLECT-ACT GAP — among those it failed, does its OWN reflection (injected back as a chronicle)
                       let the SAME model solve the scenario on a retry?

Decision rule:
  - aces most scenarios naively       → no headroom → use a weaker learner or harder scenarios.
  - fails a healthy fraction AND       → GREEN: the method can work on this model.
    reflection-injected-back improves
  - fails but reflection doesn't help  → no reflect-act gap → method can't work on this model as-is.

Run from project root (needs sotopia + an API key):
  python scripts/run_go_nogo.py --n-scenarios 20
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import numpy as np

from social_omni_epic.archive import Archive
from social_omni_epic.coherence_check import CoherenceChecker
from social_omni_epic.embedding_utils import get_similar_scenarios
from social_omni_epic.fm import FM
from social_omni_epic.reflection_module import ReflectionModule
from social_omni_epic.scenario_title import designate_target_agent
from social_omni_epic.seeds import load_sotopia_seeds_with_embeddings
from social_omni_epic.skills_chronicle import SkillsChronicle
from social_omni_epic.success_detector import SuccessDetector
from social_omni_epic.task_generator import TaskGenerator


def _gen_one(fm, archive, task_gen, coherence_checker, n_examples, vs_candidates):
    """Generate one coherent, target-designated dev scenario from a random anchor."""
    idx = int(np.random.randint(archive.size))
    anchor = archive.state.tasks[idx]
    all_embs = archive.get_successful_embeddings()
    examples = [anchor]
    if anchor.embedding and all_embs:
        src = [s.source_scenario_id for s in archive.state.tasks]
        agt = [s.target_agent_idx for s in archive.state.tasks]
        ex_idx = get_similar_scenarios(anchor.embedding, all_embs, num_returns=n_examples,
                                       source_ids=src, agent_idxs=agt,
                                       preferred_agent_idx=anchor.target_agent_idx)
        examples = [archive.state.tasks[i] for i in ex_idx]
    scn = task_gen.generate_from_archive(examples)
    if scn is None:
        return None, None
    try:
        scn.embedding = fm.get_embeddings([scn.to_text_for_embedding()])[0]
    except Exception:
        return None, None
    c = coherence_checker.check(scn)
    if not c.passed:
        scn = task_gen.patch_scenario(scn, c.issues)
        if scn is None:
            return None, None
    scn.target_agent_idx, scn.target_agent_goal_abstract = designate_target_agent(scn, anchor, fm)
    return scn, anchor


def _episode_inputs(scn, scenario_to_sotopia_profiles):
    env_profile, agent_profiles = scenario_to_sotopia_profiles(scn)
    if scn.target_agent_idx == 1:
        agent_profiles = [agent_profiles[1], agent_profiles[0]]
        env_profile.agent_goals = [env_profile.agent_goals[1], env_profile.agent_goals[0]]
    learner_goal = env_profile.agent_goals[0] if env_profile.agent_goals else ""
    pidx = 1 - scn.target_agent_idx
    partner = scn.agent_profiles[pidx] if pidx < len(scn.agent_profiles) else None
    return env_profile, agent_profiles, learner_goal, partner


def main() -> None:
    ap = argparse.ArgumentParser(description="Go/no-go: headroom + reflect-act gap")
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--seeds-path", type=str, default="data/sotopia_90_seeds.jsonl")
    ap.add_argument("--seed-limit", type=int, default=None)
    ap.add_argument("--model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--learner-model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--partner-model", type=str, default="openai/gpt-5-mini")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--judge-k", type=int, default=3)
    ap.add_argument("--vs-candidates", type=int, default=5)
    ap.add_argument("--n-examples", type=int, default=3)
    ap.add_argument("--output", type=str, default="output/go_nogo.json")
    args = ap.parse_args()

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY (or OPENAI_API_KEY) not set.", file=sys.stderr)
        sys.exit(1)

    from social_omni_epic.episode_runner import run_single_episode, clean_transcript
    from social_omni_epic.sotopia_bridge import scenario_to_sotopia_profiles

    fm = FM(model=args.model)
    archive = Archive(checkpoint_dir="output/go_nogo_tmp")
    seeds = load_sotopia_seeds_with_embeddings(fm=fm, seeds_path=args.seeds_path,
                                               limit=args.seed_limit, both_perspectives=True)
    for s in seeds:
        archive.add_successful(s)
    if archive.size == 0:
        print("ERROR: no seeds to anchor generation.", file=sys.stderr)
        sys.exit(1)

    task_gen = TaskGenerator(fm, num_examples=args.n_examples, max_retries=3)
    coherence_checker = CoherenceChecker(fm)
    reflection_mod = ReflectionModule(fm)
    detector = SuccessDetector()

    async def _run(scn, chronicle):
        env_profile, agent_profiles, learner_goal, partner = _episode_inputs(scn, scenario_to_sotopia_profiles)
        return await run_single_episode(
            env_profile=env_profile, agent_profiles=agent_profiles, fm=fm,
            learner_model=args.learner_model, partner_model=args.partner_model,
            memory_prompt=chronicle.format_for_prompt(max_entries=8) if chronicle else "",
            max_turns=args.max_turns, learner_goal=learner_goal,
            rubric=scn.success_rubric, partner_profile=partner, judge_self_consistency_k=args.judge_k,
        )

    records = []
    naive_fail = 0
    reflected_solved = 0
    for i in range(args.n_scenarios):
        scn, anchor = _gen_one(fm, archive, task_gen, coherence_checker, args.n_examples, args.vs_candidates)
        if scn is None:
            print(f"[{i}] generation failed, skipping")
            continue
        rec = {"scenario": scn.scenario[:160], "goal_type": scn.goal_type}
        try:
            naive = asyncio.run(_run(scn, SkillsChronicle()))
        except Exception as e:
            print(f"[{i}] naive episode error: {e}")
            continue
        rec["naive_solved"] = bool(naive.goal_achieved)
        rec["naive_rubric"] = naive.rubric_results
        if naive.goal_achieved:
            rec["headroom"] = False  # too easy for this model
            records.append(rec)
            print(f"[{i}] naive SOLVED (too easy)")
            continue
        naive_fail += 1
        rec["headroom"] = True
        # reflect → inject → retry
        ref = reflection_mod.reflect(
            chronicle=SkillsChronicle(), scenario=scn, transcripts=[clean_transcript(naive.transcript)],
            prior_edit_reasons={}, attempt_num=1, anchor_task=anchor,
            rubric_results=naive.rubric_results,
        )
        try:
            retry = asyncio.run(_run(scn, ref.updated_chronicle))
        except Exception as e:
            print(f"[{i}] retry episode error: {e}")
            records.append(rec)
            continue
        rec["retry_solved"] = bool(retry.goal_achieved)
        rec["n_chronicle_entries"] = len(ref.updated_chronicle.entries)
        if retry.goal_achieved:
            reflected_solved += 1
        records.append(rec)
        print(f"[{i}] naive FAILED → reflect → retry {'SOLVED' if retry.goal_achieved else 'failed'}")

    n = len([r for r in records if "naive_solved" in r])
    headroom_frac = naive_fail / n if n else 0.0
    gap_frac = reflected_solved / naive_fail if naive_fail else 0.0
    summary = {
        "model": args.learner_model,
        "n_scenarios_evaluated": n,
        "naive_fail_fraction (headroom)": round(headroom_frac, 3),
        "reflect_act_gap (solved-on-retry / naive-failed)": round(gap_frac, 3),
        "decision": (
            "NO HEADROOM — aces scenarios; use a weaker learner or harder scenarios"
            if headroom_frac < 0.3 else
            ("GREEN — headroom + reflect-act gap present" if gap_frac >= 0.3 else
             "NO REFLECT-ACT GAP — fails but reflection doesn't help; method can't work on this model as-is")
        ),
    }
    out = {"timestamp": datetime.now().isoformat(), "args": vars(args),
           "summary": summary, "records": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    print("\n=== GO/NO-GO SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
