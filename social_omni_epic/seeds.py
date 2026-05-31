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
"""
import json
import numpy as np
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
            f"Seed file not found: {path}\n"
            f"Expected data/sotopia_90_seeds.jsonl in the project root."
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
                source_scenario_id=env_pk,
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


def load_sotopia_seeds_with_embeddings(
    fm,
    seeds_path: str = "data/sotopia_90_seeds.jsonl",
    limit: Optional[int] = None,
    both_perspectives: bool = True,
) -> list[SocialScenario]:
    """Load seeds and attach embeddings, using a file cache to avoid re-embedding.

    Cache is stored as <seeds_path>.embeddings.npy + <seeds_path>.embeddings.meta.json.
    It is invalidated when the seeds file mtime or entry count changes.
    """
    seeds = load_sotopia_seeds(
        seeds_path=seeds_path, limit=limit, both_perspectives=both_perspectives
    )
    if not seeds:
        return seeds

    seeds_file = Path(seeds_path)
    cache_emb = seeds_file.with_suffix(".embeddings.npy")
    cache_meta = seeds_file.with_suffix(".embeddings.meta.json")

    mtime = seeds_file.stat().st_mtime
    n = len(seeds)

    # Try loading from cache
    if cache_emb.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text())
            if meta.get("mtime") == mtime and meta.get("count") == n:
                embs = np.load(cache_emb)
                if len(embs) == n:
                    for scn, emb in zip(seeds, embs):
                        scn.embedding = emb.tolist()
                    return seeds
        except Exception:
            pass  # cache corrupt — fall through to recompute

    # Compute and cache
    texts = [s.to_text_for_embedding() for s in seeds]
    embs = fm.get_embeddings(texts)
    for scn, emb in zip(seeds, embs):
        scn.embedding = emb
    np.save(cache_emb, np.array(embs))
    cache_meta.write_text(json.dumps({"mtime": mtime, "count": n}))

    return seeds
