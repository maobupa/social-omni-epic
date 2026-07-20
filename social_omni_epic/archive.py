import json
from pathlib import Path

import numpy as np

from .data_models import SocialScenario, ArchiveState, K_VOTES_EQUIV  # single source of truth


class Archive:
    def __init__(self, checkpoint_dir: str):
        self.state = ArchiveState()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._total_selections: int = 0

    # ------------------------------------------------------------------
    # Adding scenarios
    # ------------------------------------------------------------------

    def add_task(self, scenario: SocialScenario) -> None:
        """Add any completed scenario (too_easy / frontier / beyond_frontier) to the task store."""
        self.state.tasks.append(scenario)

    def add_successful(self, scenario: SocialScenario) -> None:
        """Backward-compat alias for add_task (seeds use this path)."""
        self.add_task(scenario)

    def add_failed_generation(self, info: dict) -> None:
        self.state.failed_generation.append(info)

    def add_failed_interestingness(self, scenario: SocialScenario) -> None:
        self.state.failed_interestingness.append(scenario)

    def add_failed_task(self, scenario: SocialScenario) -> None:
        self.state.failed_tasks.append(scenario)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_successful_embeddings(self) -> list[list[float]]:
        return [s.embedding for s in self.state.tasks if s.embedding is not None]

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, iteration: int) -> None:
        path = self.checkpoint_dir / f"archive_iter_{iteration}.json"
        with open(path, "w") as f:
            f.write(self.state.model_dump_json(indent=2))
        latest = self.checkpoint_dir / "archive_latest.json"
        with open(latest, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def load_checkpoint(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        # Migrate old checkpoints that stored tasks under "successful".
        if "successful" in data and "tasks" not in data:
            data["tasks"] = data.pop("successful")
        self.state = ArchiveState(**data)
        self._total_selections = sum(s.n_i for s in self.state.tasks)

    # ------------------------------------------------------------------
    # Thompson Sampling (LP pseudo-vote posterior, §5)
    #
    # Each task is modelled as a Bernoulli arm:
    #   posterior ~ Beta(prior_alpha + alpha_votes, prior_beta + beta_votes)
    #
    # alpha_votes accumulates improved_votes from generated children.
    # beta_votes  accumulates (total_votes - improved_votes) from children.
    # too_easy children contribute (0, K_VOTES_EQUIV) — penalise the anchor.
    #
    # All tasks (too_easy / frontier / beyond_frontier) are selectable
    # anchors so that the relax and lateral operators have appropriate targets.
    # ------------------------------------------------------------------

    def thompson_select(self) -> int:
        """Return archive index selected by Thompson Sampling over LP pseudo-vote posteriors."""
        if self.size == 0:
            return -1
        samples = []
        for task in self.state.tasks:
            alpha = task.prior_alpha + task.alpha_votes
            beta_param = task.prior_beta + task.beta_votes
            samples.append(np.random.beta(alpha, max(beta_param, 1e-9)))
        return int(np.argmax(samples))

    def random_select(self) -> int:
        """Uniform-random anchor selection — the ablation baseline for Thompson (DOF 1).

        Ignores all posteriors; every arm is equally likely. Posterior updates still happen
        (record_child_outcome runs as usual) but are unused for selection, so this is a clean
        ceteris-paribus swap of ONLY the selection rule. Uses np.random (seeded at startup) for
        reproducibility. See docs/post_run_experiments.md §3.
        """
        if self.size == 0:
            return -1
        return int(np.random.randint(self.size))

    def select_anchor(self, mode: str = "thompson") -> int:
        """Dispatch anchor selection by mode: 'thompson' (default) or 'random' (ablation)."""
        return self.random_select() if mode == "random" else self.thompson_select()

    def record_selection(self, idx: int, iteration: int) -> None:
        """Mark task at idx as selected; increment n_i for bookkeeping."""
        self._total_selections += 1
        task = self.state.tasks[idx]
        task.n_i += 1.0
        task.last_chosen = iteration

    def record_child_outcome(self, parent_idx: int, improved_votes: int, total_votes: int) -> None:
        """Accumulate a child's LP pseudo-votes onto the parent's posterior.

        Call once per generated child after its K-attempt loop completes:
          - frontier child:       pass lp_result.improved_votes, lp_result.total_votes
          - too_easy child:       pass 0, K_VOTES_EQUIV
          - beyond_frontier child: pass 0, lp_result.total_votes (or K_VOTES_EQUIV if 0)
        """
        if 0 <= parent_idx < self.size:
            p = self.state.tasks[parent_idx]
            p.alpha_votes += improved_votes
            p.beta_votes += max(total_votes - improved_votes, 0)

    def record_child(self, parent_idx: int) -> None:
        """Increment n_children for bookkeeping."""
        if 0 <= parent_idx < self.size:
            self.state.tasks[parent_idx].n_children += 1

    def record_solved_child(self, parent_idx: int) -> None:
        """Increment n_solved for bookkeeping (separate from the LP posterior)."""
        if 0 <= parent_idx < self.size:
            self.state.tasks[parent_idx].n_solved += 1

    def child_prior_from_parent(self, parent_idx: int,
                                child_prior_mass: float = 4.0) -> tuple[float, float]:
        """Return (prior_alpha, prior_beta) a child inherits from this parent.

        The child starts with the parent's current posterior as its prior, giving it a
        warm start that reflects structural similarity to the parent. The total mass is
        capped at `child_prior_mass` so deep-lineage children don't inherit so much
        posterior mass that it drowns their own first ~6 votes — the mean (structural
        similarity signal) is preserved, responsiveness to the child's own evidence is not.
        """
        if not (0 <= parent_idx < self.size):
            return 1.0, 1.0
        p = self.state.tasks[parent_idx]
        alpha = p.prior_alpha + p.alpha_votes
        beta_param = p.prior_beta + p.beta_votes
        m = alpha + beta_param
        if child_prior_mass > 0 and m > child_prior_mass:
            scale = child_prior_mass / m
            alpha *= scale
            beta_param *= scale
        return float(alpha), float(beta_param)

    def update_niche_count(self, niche_id: int) -> None:
        """Increment the generation count for a niche (used for metrics logging)."""
        self.state.niche_counts[niche_id] = self.state.niche_counts.get(niche_id, 0) + 1

    @property
    def size(self) -> int:
        return len(self.state.tasks)
