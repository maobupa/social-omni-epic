"""Phase 0 main loop: open-ended social scenario generation with simulated learning.

Run from project root:
  python scripts/run_phase0.py
"""
import json
import os
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

from social_omni_epic.archive import Archive
from social_omni_epic.fm import FM
from social_omni_epic.task_generator import TaskGenerator
from social_omni_epic.model_of_interestingness import ModelOfInterestingness
from social_omni_epic.embedding_utils import (
    get_similar_scenarios,
    compute_cell_coverage,
)
from social_omni_epic.seeds import (
    load_sotopia_seeds,
    build_fallback_seeds,
)


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent / "configs"),
    config_name="social_omni_epic",
)
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    np.random.seed(config.random_seed)

    fm = FM(model=config.model, temperature=config.temperature)
    archive = Archive(checkpoint_dir=config.checkpoint_dir)
    task_gen = TaskGenerator(
        fm,
        num_examples=config.task_generator.num_examples,
        num_failed_examples=config.task_generator.num_failed_examples,
        max_retries=config.task_generator.max_retries,
    )
    moi = ModelOfInterestingness(
        fm,
        num_examples=config.moi.num_examples,
        min_archive_size=config.moi.min_archive_size,
    )

    # === Seed the archive ===
    seed_scenarios = []
    try:
        seed_scenarios = load_sotopia_seeds(
            data_dir=config.seed_data_dir,
            episodes_path=config.get("episodes_path"),
            restrict_to_episodes_v1=config.get("restrict_to_episodes_v1", True),
            limit=config.get("seed_limit"),
        )
        print(f"Loaded {len(seed_scenarios)} Sotopia seed scenarios")
    except FileNotFoundError:
        if config.use_fallback_seeds_if_missing:
            print("Sotopia seeds not found, using FM-generated fallback seeds...")
            seed_scenarios = build_fallback_seeds(fm)
            print(f"Built {len(seed_scenarios)} fallback seeds")
        else:
            print("ERROR: Sotopia seeds missing and fallback disabled.", file=sys.stderr)
            sys.exit(1)

    if seed_scenarios:
        print("Embedding seed scenarios...")
        texts = [s.to_text_for_embedding() for s in seed_scenarios]
        # Batch to be safe
        embs: list[list[float]] = []
        BATCH = 100
        for i in range(0, len(texts), BATCH):
            embs.extend(fm.get_embeddings(texts[i:i + BATCH]))
        for scn, e in zip(seed_scenarios, embs):
            scn.embedding = e
            archive.add_successful(scn)

    print(f"Archive initialized with {archive.size} scenarios")

    choose_probs = np.ones(archive.size) if archive.size > 0 else np.array([])
    metrics_log: list[dict] = []

    for iteration in tqdm(range(config.iterations), desc="Generating"):
        # === STEP 1: Generate ===
        examples: list = []
        example_indices: list[int] = []
        if config.use_archive and archive.size > 0:
            examples, example_indices = task_gen.select_examples(
                archive, choose_probs, config.task_generator.num_examples,
                strategy=config.task_generator.get("example_strategy", "knn"),
            )
            choose_probs = choose_probs + 1
            for idx in example_indices:
                if idx < len(choose_probs):
                    choose_probs[idx] = 0

            failed = archive.state.failed_interestingness[-config.task_generator.num_failed_examples:] \
                if config.task_generator.num_failed_examples > 0 else []
            existing_types = list({s.interaction_type for s in archive.state.successful
                                   if s.interaction_type}) \
                if config.task_generator.get("show_existing_types", True) else None
            scenario = task_gen.generate_from_archive(
                examples, failed, existing_types=existing_types
            )
        else:
            scenario = task_gen.generate_unconditioned()

        if scenario is None:
            archive.add_failed_generation({"iteration": iteration, "reason": "validation_failed_after_retries"})
            print(f"[iter {iteration}] FAILED generation (validation)")
            continue

        scenario.iteration = iteration
        scenario.parent_example_ids = [examples[i].id for i in range(len(examples))]

        # === STEP 2: Embed ===
        try:
            scenario.embedding = fm.get_embeddings([scenario.to_text_for_embedding()])[0]
        except Exception as e:
            archive.add_failed_generation({"iteration": iteration, "reason": f"embedding_error: {e}"})
            print(f"[iter {iteration}] FAILED embedding: {e}")
            continue

        # === STEP 3: MoI ===
        is_interesting = True
        moi_reason = ""
        if config.enable_moi and archive.size >= config.moi.min_archive_size:
            sim_indices = get_similar_scenarios(
                scenario.embedding,
                archive.get_successful_embeddings(),
                num_returns=config.moi.num_examples,
            )
            similar = [archive.state.successful[i] for i in sim_indices]
            is_interesting, moi_reason = moi.evaluate(scenario, similar)
            scenario.moi_reasoning = moi_reason

        if not is_interesting:
            archive.add_failed_interestingness(scenario)
            print(f"[iter {iteration}] REJECTED (uninteresting) | type={scenario.interaction_type} | "
                  f"reason={moi_reason[:80]}")
            continue

        # === STEP 4: Simulated success ===
        archive.add_successful(scenario)
        choose_probs = np.append(choose_probs, 1.0)

        # === STEP 5: Metrics ===
        if archive.size > 1:
            embs = np.array(archive.get_successful_embeddings())
            coverage = compute_cell_coverage(embs)
            metrics_log.append({
                "iteration": iteration,
                "archive_size": archive.size,
                "cell_coverage": coverage,
                "total_failed_gen": len(archive.state.failed_generation),
                "total_failed_interest": len(archive.state.failed_interestingness),
            })

        snippet = scenario.scenario[:60].replace("\n", " ")
        print(f"[iter {iteration}] ACCEPTED: \"{snippet}...\" | type={scenario.interaction_type} | "
              f"archive={archive.size}")

        if iteration % config.checkpoint_every == 0 and iteration > 0:
            archive.save_checkpoint(iteration)
            with open(Path(config.checkpoint_dir) / "metrics.json", "w") as f:
                json.dump(metrics_log, f, indent=2)

    # Final save
    archive.save_checkpoint(config.iterations)
    with open(Path(config.checkpoint_dir) / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Done. Final archive size: {archive.size}")


if __name__ == "__main__":
    main()
