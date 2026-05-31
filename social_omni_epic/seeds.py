"""Load Sotopia seed SocialScenarios from the pre-assembled 90-seed file.

Primary source:
  data/sotopia_90_seeds.jsonl   — 90 scenarios with full agent profiles and
                                   relationship info already joined.

Each row has: env_pk, codename, scenario, agent_goals (2), agent_profiles (2),
relationship_type (int 0-5), relationship_label, relationship_background.

With both_perspectives=True (default), each row yields TWO archive entries —
one with target_agent_idx=0 and one with target_agent_idx=1. IDs are stable
and deterministic: {env_pk}_p0 and {env_pk}_p1. Both share source_scenario_id
= env_pk so perspective-aware retrieval can deduplicate correctly.

Fallback: if the primary file is missing, build_fallback_seeds() generates
seeds from short descriptions using the task generator.
"""
import json
from pathlib import Path
from typing import Optional

from .data_models import SocialScenario, AgentProfile


def _make_agent_profile(d: dict) -> AgentProfile:
    moral = d.get("moral_values", "")
    if isinstance(moral, list):
        moral = ", ".join(str(x) for x in moral)
    schwartz = d.get("schwartz_personal_values", "")
    if isinstance(schwartz, list):
        schwartz = ", ".join(str(x) for x in schwartz)
    return AgentProfile(
        first_name=d.get("first_name") or "Unknown",
        last_name=d.get("last_name", "") or "",
        age=d.get("age") or 0,
        gender_identity=d.get("gender") or d.get("gender_identity", "") or "",
        occupation=d.get("occupation", "") or "",
        big_five=d.get("big_five", "") or "",
        moral_values=moral,
        schwartz_portrait_value=schwartz,
        decision_making_style=d.get("decision_making_style", "") or "",
        secret=d.get("secret", "") or "",
        mbti=d.get("mbti", "") or "",
        public_info=d.get("public_info", "") or "",
    )


def load_sotopia_seeds(
    seeds_path: str = "data/sotopia_90_seeds.jsonl",
    limit: Optional[int] = None,
    both_perspectives: bool = True,
    # Legacy kwargs accepted but ignored
    data_dir: Optional[str] = None,
    episodes_path: Optional[str] = None,
    restrict_to_episodes_v1: bool = True,
) -> list[SocialScenario]:
    path = Path(seeds_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Expected pre-assembled seed file."
        )

    scenarios: list[SocialScenario] = []
    rows_read = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_read += 1

            profiles = [_make_agent_profile(p) for p in row.get("agent_profiles", [])]
            while len(profiles) < 2:
                profiles.append(AgentProfile(first_name=f"Agent{len(profiles)+1}"))

            agent_goals = row.get("agent_goals") or ["", ""]
            agent_goals = (list(agent_goals) + ["", ""])[:2]

            env_pk = row.get("env_pk", f"seed_{rows_read}")
            common = dict(
                iteration=-1,
                scenario=row.get("scenario", ""),
                agent_profiles=profiles,
                agent_goals=agent_goals,
                relationship=row.get("relationship_label", "") or str(row.get("relationship_type", "")),
                relationship_background=row.get("relationship_background", ""),
                tag=row.get("codename", "") or row.get("source", ""),
                interaction_type=row.get("source", ""),
                source="seed_sotopia",
                source_env_id=env_pk,
                source_scenario_id=env_pk,  # shared dedup key for both perspectives
            )

            for idx in ([0, 1] if both_perspectives else [0]):
                scenarios.append(SocialScenario(
                    id=f"{env_pk}_p{idx}",
                    target_agent_idx=idx,
                    **common,
                ))

            if limit is not None and rows_read >= limit:
                break

    return scenarios


FALLBACK_SEED_DESCRIPTIONS = [
    "Two coworkers must decide how to split credit for a joint project that one person contributed more to.",
    "Two strangers are stuck in an elevator and must work together to signal for help.",
    "A landlord confronts a tenant who has been subletting their apartment without permission.",
    "A teenager tries to convince their strict parent to let them go on a road trip with friends.",
    "A job interviewer suspects the candidate has fabricated part of their resume.",
    "A person must break the news to their best friend that the friend's partner has been seen on a dating app.",
    "Two food truck owners are parked next to each other at a festival, competing for customers.",
    "An international student asks their professor for a deadline extension.",
    "A senior doctor must address a junior resident who made a medical error.",
    "Two divorced parents meet to discuss changing their custody arrangement.",
]


def build_fallback_seeds(fm) -> list[SocialScenario]:
    from .task_generator import TaskGenerator
    gen = TaskGenerator(fm, num_examples=0, num_failed_examples=0, max_retries=3)
    out = []
    for desc in FALLBACK_SEED_DESCRIPTIONS:
        scn = gen.flesh_out_seed(desc)
        if scn is not None:
            scn.iteration = -1
            scn.source = "fallback_seed"
            scn.target_agent_idx = 0
            out.append(scn)
    return out
