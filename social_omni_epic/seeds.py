"""Load Sotopia environment + agent profiles as seed SocialScenarios.

Data layout (produced by scripts/extract_sotopia_seeds.py):
  data/sotopia_seeds/environment_profiles.jsonl  (~884 envs from sotopia-pi)
  data/sotopia_seeds/agent_profiles.jsonl        (~40 agent profiles)
  data/sotopia_seeds/relationship_profiles.jsonl (~120 pairings)

Plus an optional companion file:
  data/sotopia_episodes_v1.jsonl                 (episode logs; supplies the
                                                  env_id <-> agent_ids linkage
                                                  for the canonical ICLR 90)

The canonical Sotopia ICLR 2024 paper used 90 scenarios. sotopia-pi extended this
to ~884. We pick which subset to use based on `restrict_to_episodes_v1`:
  True  -> the 90 env_ids that appear in the episodes file (the canonical set)
  False -> all extracted env profiles
"""
import json
from pathlib import Path
from typing import Optional
from collections import OrderedDict

from .data_models import SocialScenario, AgentProfile


RELATIONSHIP_LABELS = {
    0: "strangers",
    1: "know each other by name",
    2: "acquaintances",
    3: "friends",
    4: "romantic relationship",
    5: "family members",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _moral_values_to_str(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return str(v or "")


def _make_agent_profile(d: dict) -> AgentProfile:
    """Map Sotopia agent record (with their field names) to our AgentProfile."""
    return AgentProfile(
        first_name=d.get("first_name") or "Unknown",
        last_name=d.get("last_name", "") or "",
        age=d.get("age") or 0,
        # Sotopia stores 'gender' (Woman/Man/Nonbinary) and 'gender_pronoun' separately.
        gender_identity=d.get("gender") or d.get("gender_identity", "") or "",
        occupation=d.get("occupation", "") or "",
        big_five=d.get("big_five", "") or "",
        moral_values=_moral_values_to_str(d.get("moral_values")),
        # Sotopia field is 'schwartz_personal_values' (list); our schema kept the
        # original prompt-engineering name 'schwartz_portrait_value'.
        schwartz_portrait_value=_moral_values_to_str(d.get("schwartz_personal_values")),
        decision_making_style=d.get("decision_making_style", "") or "",
        secret=d.get("secret", "") or "",
        mbti=d.get("mbti", "") or "",
        public_info=(d.get("public_info") or d.get("personality_and_values") or "")[:500],
    )


def _build_env_to_agents_from_episodes(episodes_path: Path) -> dict[str, list[str]]:
    """Return {env_id: [agent1_pk, agent2_pk]} from the first episode per env_id."""
    env_to_agents: dict[str, list[str]] = {}
    with open(episodes_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            env_id = d.get("environment_id")
            agent_ids = d.get("agent_ids")
            if env_id and agent_ids and env_id not in env_to_agents:
                env_to_agents[env_id] = list(agent_ids)[:2]
    return env_to_agents


def load_sotopia_seeds(
    data_dir: str = "data/sotopia_seeds",
    episodes_path: Optional[str] = "data/sotopia_episodes_v1.jsonl",
    restrict_to_episodes_v1: bool = True,
    limit: Optional[int] = None,
) -> list[SocialScenario]:
    data_dir_p = Path(data_dir)
    env_path = data_dir_p / "environment_profiles.jsonl"
    agent_path = data_dir_p / "agent_profiles.jsonl"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing {env_path}. Run scripts/extract_sotopia_seeds.py first.")

    env_rows = _read_jsonl(env_path)
    agent_rows = _read_jsonl(agent_path) if agent_path.exists() else []
    agent_db: dict[str, dict] = {a["pk"]: a for a in agent_rows if a.get("pk")}

    # env_id -> [agent_pk, agent_pk]
    env_to_agents: dict[str, list[str]] = {}
    if episodes_path:
        ep_path = Path(episodes_path)
        if ep_path.exists():
            env_to_agents = _build_env_to_agents_from_episodes(ep_path)

    # Optionally restrict to the canonical ICLR-90 set (the env_ids that appear in
    # the episodes file).
    if restrict_to_episodes_v1 and env_to_agents:
        canonical = set(env_to_agents.keys())
        env_rows = [e for e in env_rows if e.get("pk") in canonical]

    # Use ordered/deterministic agent fallback if env->agents missing
    agent_pks_in_order = list(agent_db.keys())

    scenarios: list[SocialScenario] = []
    for env in env_rows:
        env_pk = env.get("pk", "")
        agent_pks = env_to_agents.get(env_pk, [])

        if len(agent_pks) < 2 and agent_pks_in_order:
            # deterministic fallback: pick first two unused
            fallback = [pk for pk in agent_pks_in_order if pk not in agent_pks]
            agent_pks = (agent_pks + fallback)[:2]

        profiles = []
        for pk in agent_pks[:2]:
            ad = agent_db.get(pk, {"first_name": f"Agent_{pk[:6]}"})
            profiles.append(_make_agent_profile(ad))
        if len(profiles) < 2:
            profiles += [AgentProfile(first_name=f"Agent{i+1}")
                         for i in range(len(profiles), 2)]

        agent_goals = env.get("agent_goals") or ["", ""]
        if isinstance(agent_goals, dict):
            agent_goals = list(agent_goals.values())
        agent_goals = (list(agent_goals) + ["", ""])[:2]

        rel_raw = env.get("relationship")
        if isinstance(rel_raw, int):
            relationship = RELATIONSHIP_LABELS.get(rel_raw, str(rel_raw))
        else:
            relationship = str(rel_raw or "")

        scenarios.append(SocialScenario(
            iteration=-1,
            scenario=env.get("scenario", "") or "",
            agent_profiles=profiles,
            agent_goals=agent_goals,
            relationship=relationship,
            tag=env.get("source", "") or env.get("codename", "") or "",
            interaction_type=env.get("source", "") or "",
            source="seed_sotopia",
        ))

        if limit is not None and len(scenarios) >= limit:
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
            out.append(scn)
    return out
