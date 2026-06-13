"""Build a runnable eval-candidate set from the sotopia-pi *expanded* scenarios.

The expanded environment set (`data/sotopia_seeds/environment_profiles.jsonl`,
~884 envs) is NOT directly runnable: unlike the canonical 90, those envs have
no agent pairing (episodes_v1 only covers the 90) and carry no resolved agent
profiles. This script turns sampled expanded envs into rows in the SAME schema
as `data/sotopia_90_seeds.jsonl`, so they load via `seeds.load_sotopia_seeds`
and run in episodes unchanged.

Pipeline (the first three funnel steps — NO model calls here):
  1. Load the ~884 expanded envs.
  2. Drop the 90 training env_pks (read from the seeds file) so the eval set is
     held out from ExpeL/SOE training. (The 884 is a superset of the 90.)
  3. Keep only runnable envs (non-empty scenario + two agent goals).
  4. Randomly sample N candidates (seeded, reproducible).
  5. Assign each env a SOTOPIA-faithful agent pairing by matching the env's
     `relationship` code against `relationship_profiles.jsonl`, then resolve both
     agents' full profiles from `agent_profiles.jsonl` (mirrors export_90_seeds).
  6. Write `data/eval_candidates.jsonl` in the canonical seed schema.

After this, generate retrieval titles (needed by the full SOE model; ignored by
the ExpeL baseline) with:
  python scripts/generate_seed_titles.py --seeds-path data/eval_candidates.jsonl --overwrite

Then later (when you have time) run the vanilla gpt-5-mini profiling pass to
measure difficulty and stratified-subsample to your final ~50-60.

Run from project root:
  python scripts/build_eval_candidates.py                       # 150, seed 42
  python scripts/build_eval_candidates.py --n 200 --seed 7
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/sotopia_seeds")
DEFAULT_SEEDS_PATH = Path("data/sotopia_90_seeds.jsonl")
DEFAULT_OUT = Path("data/eval_candidates.jsonl")

# Same mapping export_90_seeds.py uses, so relationship_label matches the 90.
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


def _resolve_agent(pk: str, agent_db: dict[str, dict]) -> dict:
    """Mirror export_90_seeds.py's agent dict exactly so load_sotopia_seeds
    parses these identically to the 90-seed file."""
    a = agent_db.get(pk)
    if a is None:
        return {"pk": pk}
    return {
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
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build runnable eval candidates from expanded sotopia-pi envs")
    ap.add_argument("--n", type=int, default=150, help="Number of candidates to sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    ap.add_argument("--seeds-path", type=Path, default=DEFAULT_SEEDS_PATH,
                    help="Training seeds file whose env_pks are excluded")
    ap.add_argument("--exclude", type=Path, action="append", default=[],
                    help="Additional seed files (canonical schema) whose env_pks are "
                         "ALSO excluded. Repeatable. Use to hold out previously built "
                         "sets (e.g. the eval-candidate 150) when sampling a fresh set.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in [DATA_DIR / "environment_profiles.jsonl",
              DATA_DIR / "agent_profiles.jsonl",
              DATA_DIR / "relationship_profiles.jsonl",
              args.seeds_path, *args.exclude]:
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    env_rows = read_jsonl(DATA_DIR / "environment_profiles.jsonl")
    agent_db = {a["pk"]: a for a in read_jsonl(DATA_DIR / "agent_profiles.jsonl") if a.get("pk")}
    rel_rows = read_jsonl(DATA_DIR / "relationship_profiles.jsonl")

    # relationship code -> list of (a1, a2, background_story)
    rel_by_code: dict[int, list[tuple[str, str, str]]] = {}
    for r in rel_rows:
        a1, a2 = r.get("agent_1_id"), r.get("agent_2_id")
        if a1 in agent_db and a2 in agent_db:
            rel_by_code.setdefault(r.get("relationship"), []).append(
                (a1, a2, r.get("background_story", "") or "")
            )

    # Step 2: exclude the training env_pks (the canonical 90) + any extra held-out
    #         sets passed via --exclude (e.g. the eval-candidate 150).
    train_pks = {json.loads(l)["env_pk"] for l in open(args.seeds_path) if l.strip()}
    for ex_path in args.exclude:
        ex_pks = {json.loads(l)["env_pk"] for l in open(ex_path) if l.strip()}
        print(f"  excluding {len(ex_pks)} env_pks from {ex_path}")
        train_pks |= ex_pks

    # Step 3: keep only held-out, runnable envs.
    eligible = []
    for e in env_rows:
        if e.get("pk") in train_pks:
            continue
        goals = e.get("agent_goals") or []
        if isinstance(goals, dict):
            goals = list(goals.values())
        if not e.get("scenario") or len([g for g in goals if g]) < 2:
            continue
        if e.get("relationship") not in rel_by_code:  # need a pairing for this relationship
            continue
        eligible.append(e)

    print(f"Expanded envs: {len(env_rows)} | excluded 90 training | eligible held-out: {len(eligible)}")
    if len(eligible) < args.n:
        print(f"WARNING: only {len(eligible)} eligible (< requested {args.n}); using all.")

    # Step 4: random sample (seeded).
    rng = random.Random(args.seed)
    sampled = rng.sample(eligible, min(args.n, len(eligible)))

    # Step 5: assign agent pairing + resolve profiles.
    records = []
    for env in sampled:
        env_pk = env.get("pk", "")
        rel_code = env.get("relationship")
        a1, a2, background = rng.choice(rel_by_code[rel_code])
        agent_pks = [a1, a2]
        agent_profiles = [_resolve_agent(pk, agent_db) for pk in agent_pks]

        goals = env.get("agent_goals") or ["", ""]
        if isinstance(goals, dict):
            goals = list(goals.values())
        goals = (list(goals) + ["", ""])[:2]

        records.append({
            "env_pk": env_pk,
            "codename": env.get("codename", ""),
            "source": env.get("source", ""),
            "scenario": env.get("scenario", ""),
            "agent_goals": goals,
            "relationship_type": rel_code,
            "relationship_label": RELATIONSHIP_LABELS.get(rel_code, str(rel_code)),
            "relationship_background": background,
            "agent_pks": agent_pks,
            "agent_profiles": agent_profiles,
            "age_constraint": env.get("age_constraint"),
            "occupation_constraint": env.get("occupation_constraint"),
            "agent_constraint": env.get("agent_constraint"),
            "is_sotopia_hard": False,   # these are expanded envs, not the hard set
            # scenario_title/social_dynamic/target_perspective added later by
            # scripts/generate_seed_titles.py
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} candidates -> {args.out}")
    print(f"  by source        : {dict(Counter(r['source'] for r in records))}")
    print(f"  by relationship  : {dict(Counter(r['relationship_label'] for r in records))}")
    n_missing = sum(1 for r in records for a in r['agent_profiles'] if 'first_name' not in a)
    print(f"  unresolved agent profiles: {n_missing}")
    print("\nNext:")
    print(f"  python scripts/generate_seed_titles.py --seeds-path {args.out} --overwrite")
    return 0


if __name__ == "__main__":
    sys.exit(main())
