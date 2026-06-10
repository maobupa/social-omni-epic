"""Pre-generate SCENARIO_TITLE for all 90 SOTOPIA seeds.

Writes scenario_title, social_dynamic, and target_perspective fields directly into:
  - data/sotopia_90_seeds.jsonl        (in-place, one field added per row)
  - results/baseline_eval_*/episodes/  (in-place, one field added per episode JSON)

Uses the identical ScenarioTitleGenerator prompt as the curriculum pipeline.

Run from project root:
  python scripts/generate_seed_titles.py
  python scripts/generate_seed_titles.py --overwrite   # regenerate even if title exists
  python scripts/generate_seed_titles.py --episodes-dir results/baseline_eval_20260604_222545/episodes
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from social_omni_epic.fm import FM
from social_omni_epic.scenario_title import ScenarioTitleGenerator
from social_omni_epic.seeds import load_sotopia_seeds


DEFAULT_SEEDS_PATH = "data/sotopia_90_seeds.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate SCENARIO_TITLE for seed scenarios")
    parser.add_argument("--seeds-path", default=DEFAULT_SEEDS_PATH)
    parser.add_argument("--episodes-dir", default=None,
                        help="Path to baseline_eval episodes dir to also update (optional)")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate titles even if already present")
    args = parser.parse_args()

    if not (os.getenv("LIGHTNING_AI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("ERROR: LIGHTNING_AI_API_KEY (or OPENAI_API_KEY) not set.", file=sys.stderr)
        sys.exit(1)

    seeds_path = Path(args.seeds_path)
    if not seeds_path.exists():
        print(f"ERROR: seeds file not found: {seeds_path}", file=sys.stderr)
        sys.exit(1)

    # Load seeds (single perspective — title is per-scenario, not per-agent)
    seeds = load_sotopia_seeds(seeds_path=str(seeds_path), both_perspectives=False)
    print(f"Loaded {len(seeds)} seeds")

    fm = FM(model=args.model, temperature=0.4)
    title_gen = ScenarioTitleGenerator(fm, max_retries=3)

    # Read all jsonl rows into memory keyed by env_pk
    rows: list[dict] = []
    with open(seeds_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows_by_pk = {r.get("env_pk", f"seed_{i}"): r for i, r in enumerate(rows)}

    # Generate titles
    titles: dict[str, dict] = {}
    n_generated = 0
    n_skipped = 0

    for i, scenario in enumerate(seeds):
        env_pk = scenario.source_env_id or scenario.id
        row = rows_by_pk.get(env_pk, {})

        if not args.overwrite and row.get("scenario_title"):
            n_skipped += 1
            titles[env_pk] = {
                "scenario_title": row["scenario_title"],
                "social_dynamic": row.get("social_dynamic", ""),
                "target_perspective": row.get("target_perspective", ""),
            }
            continue

        print(f"[{i+1}/{len(seeds)}] {env_pk} ({scenario.tag or scenario.source})")
        title_data = title_gen.generate(scenario, target_agent_idx=scenario.target_agent_idx)
        titles[env_pk] = title_data
        n_generated += 1
        print(f"  → {title_data['scenario_title']}")

    print(f"\nGenerated: {n_generated}  Skipped (existing): {n_skipped}")

    # Write titles back into seeds jsonl in-place
    for row in rows:
        env_pk = row.get("env_pk", "")
        if env_pk in titles:
            row.update(titles[env_pk])

    with open(seeds_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Updated {seeds_path}")

    # Optionally update baseline_eval episode JSONs
    if args.episodes_dir:
        episodes_dir = Path(args.episodes_dir)
        if not episodes_dir.exists():
            print(f"WARNING: episodes dir not found: {episodes_dir}")
        else:
            ep_files = sorted(episodes_dir.glob("*.json"))
            n_ep = 0
            for ep_file in ep_files:
                ep = json.loads(ep_file.read_text())
                env_pk = ep.get("env_pk", "")
                if env_pk in titles:
                    if not args.overwrite and ep.get("scenario_title"):
                        continue
                    ep.update(titles[env_pk])
                    ep_file.write_text(json.dumps(ep, indent=2))
                    n_ep += 1
            print(f"Updated {n_ep} episode files in {episodes_dir}")


if __name__ == "__main__":
    main()
