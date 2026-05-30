"""Export the canonical SOTOPIA ICLR-2024 90 scenarios as a single resolved JSONL.

Each line contains:
  - full environment profile (scenario text, agent goals, relationship label)
  - full agent profiles for both agents
  - relationship background_story from relationship_profiles.jsonl (if found)

Output: data/sotopia_90_seeds.jsonl

Run from project root:
  python scripts/export_90_seeds.py
"""
import json
import sys
from pathlib import Path

DATA_DIR = Path("data/sotopia_seeds")
EPISODES_PATH = Path("data/sotopia_episodes_v1.jsonl")
OUT_PATH = Path("data/sotopia_90_seeds.jsonl")

RELATIONSHIP_LABELS = {
    0: "strangers",
    1: "know each other by name",
    2: "acquaintances",
    3: "friends",
    4: "romantic relationship",
    5: "family members",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    for p in [DATA_DIR / "environment_profiles.jsonl",
              DATA_DIR / "agent_profiles.jsonl",
              EPISODES_PATH]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    env_rows = read_jsonl(DATA_DIR / "environment_profiles.jsonl")
    agent_rows = read_jsonl(DATA_DIR / "agent_profiles.jsonl")
    agent_db: dict[str, dict] = {a["pk"]: a for a in agent_rows if a.get("pk")}

    # Load relationship profiles keyed by sorted agent-pair tuple
    rel_db: dict[tuple[str, str], str] = {}
    rel_path = DATA_DIR / "relationship_profiles.jsonl"
    if rel_path.exists():
        for r in read_jsonl(rel_path):
            a1, a2 = r.get("agent_1_id", ""), r.get("agent_2_id", "")
            story = r.get("background_story", "") or ""
            if a1 and a2 and story and story != "nan":
                rel_db[tuple(sorted([a1, a2]))] = story

    # Build env_id -> [agent_pk, agent_pk] from episodes (first occurrence per env)
    env_to_agents: dict[str, list[str]] = {}
    with open(EPISODES_PATH) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            env_id = d.get("environment_id")
            agent_ids = d.get("agent_ids")
            if env_id and agent_ids and env_id not in env_to_agents:
                env_to_agents[env_id] = list(agent_ids)[:2]

    canonical = set(env_to_agents.keys())
    env_rows = [e for e in env_rows if e.get("pk") in canonical]
    print(f"Canonical environments found: {len(env_rows)}")

    records = []
    missing_agents = 0
    missing_rel = 0

    for env in env_rows:
        env_pk = env.get("pk", "")
        agent_pks = env_to_agents.get(env_pk, [])

        agent_profiles = []
        for pk in agent_pks[:2]:
            a = agent_db.get(pk)
            if a is None:
                missing_agents += 1
                agent_profiles.append({"pk": pk})
                continue
            agent_profiles.append({
                "pk": pk,
                "first_name": a.get("first_name", ""),
                "last_name": a.get("last_name", ""),
                "age": a.get("age"),
                "gender": a.get("gender", ""),
                "gender_pronoun": a.get("gender_pronoun", ""),
                "occupation": a.get("occupation", ""),
                "big_five": a.get("big_five", ""),
                "moral_values": a.get("moral_values", ""),
                "schwartz_personal_values": a.get("schwartz_personal_values", ""),
                "decision_making_style": a.get("decision_making_style", ""),
                "secret": a.get("secret", ""),
                "mbti": a.get("mbti", ""),
                "public_info": a.get("public_info") or a.get("personality_and_values", ""),
            })

        rel_raw = env.get("relationship")
        relationship_label = (
            RELATIONSHIP_LABELS.get(rel_raw, str(rel_raw))
            if isinstance(rel_raw, int) else str(rel_raw or "")
        )

        background_story = ""
        if len(agent_pks) >= 2:
            key = tuple(sorted(agent_pks[:2]))
            background_story = rel_db.get(key, "")
        if not background_story:
            missing_rel += 1

        agent_goals = env.get("agent_goals") or ["", ""]
        if isinstance(agent_goals, dict):
            agent_goals = list(agent_goals.values())
        agent_goals = (list(agent_goals) + ["", ""])[:2]

        records.append({
            "env_pk": env_pk,
            "codename": env.get("codename", ""),
            "source": env.get("source", ""),
            "scenario": env.get("scenario", ""),
            "agent_goals": agent_goals,
            "relationship_type": rel_raw,
            "relationship_label": relationship_label,
            "relationship_background": background_story,
            "agent_pks": agent_pks[:2],
            "agent_profiles": agent_profiles,
            "age_constraint": env.get("age_constraint"),
            "occupation_constraint": env.get("occupation_constraint"),
            "agent_constraint": env.get("agent_constraint"),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} records to {OUT_PATH}")
    print(f"Missing agent profiles: {missing_agents}")
    print(f"Missing relationship background: {missing_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
