import json
from pathlib import Path

import numpy as np

from .data_models import SocialScenario, ArchiveState


class Archive:
    def __init__(self, checkpoint_dir: str):
        self.state = ArchiveState()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._total_selections: int = 0

    def add_successful(self, scenario: SocialScenario):
        self.state.successful.append(scenario)

    def add_failed_generation(self, info: dict):
        self.state.failed_generation.append(info)

    def add_failed_interestingness(self, scenario: SocialScenario):
        self.state.failed_interestingness.append(scenario)

    def add_failed_task(self, scenario: SocialScenario):
        self.state.failed_tasks.append(scenario)

    def get_successful_embeddings(self) -> list[list[float]]:
        return [s.embedding for s in self.state.successful if s.embedding is not None]

    def save_checkpoint(self, iteration: int):
        path = self.checkpoint_dir / f"archive_iter_{iteration}.json"
        with open(path, "w") as f:
            f.write(self.state.model_dump_json(indent=2))
        latest = self.checkpoint_dir / "archive_latest.json"
        with open(latest, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def load_checkpoint(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.state = ArchiveState(**data)
        self._total_selections = sum(s.n_i for s in self.state.successful)

    # ------------------------------------------------------------------
    # Hierarchical Thompson Sampling selection (§4.1)
    #
    # Each archive entry is modelled as a Bernoulli arm with unknown
    # success rate p_i (probability of producing a solved-after-biting
    # child).  We maintain a Beta posterior over p_i:
    #
    #   posterior_i ~ Beta(prior_alpha_i + n_solved_i,
    #                      prior_beta_i  + n_i - n_solved_i)
    #
    # Original seeds start with a flat prior: Beta(1, 1).
    # Generated children inherit the parent's posterior at birth, giving
    # them a warm start that reflects structural similarity to a proven
    # productive seed rather than treating them as completely unknown.
    #
    # At each selection step, sample once from every posterior and pick
    # the argmax.  This naturally balances exploration (high uncertainty
    # → wide distribution → occasionally sampled high) and exploitation
    # (high solved rate → distribution peaks near 1 → consistently
    # sampled high), with no tunable constants.
    # ------------------------------------------------------------------

    def thompson_select(self) -> int:
        """Return archive index selected by hierarchical Thompson Sampling."""
        if self.size == 0:
            return -1
        samples = []
        for task in self.state.successful:
            alpha = task.prior_alpha + task.n_solved
            beta_param = task.prior_beta + (task.n_i - task.n_solved)
            samples.append(np.random.beta(alpha, beta_param))
        return int(np.argmax(samples))

    def record_selection(self, idx: int, iteration: int) -> None:
        """Mark task at idx as selected. Increments n_i immediately so subsequent
        Thompson picks within the same batch see the updated distribution."""
        self._total_selections += 1
        task = self.state.successful[idx]
        task.n_i += 1.0
        task.last_chosen = iteration

    def record_outcome_weight(self, parent_idx: int, extra_n_i: float) -> None:
        """Post-result adjustment to n_i after outcome is known.

        Positive extra_n_i adds to the failure side of the Beta posterior (stronger
        downward pressure on the anchor's estimated success rate). Negative reduces it.

        Use cases:
          generation_failed → extra_n_i = -0.5  (generator may be at fault, halve the penalty)
          structural failure → extra_n_i = +1.0  (structurally unlearnable, double the penalty)
        """
        if 0 <= parent_idx < self.size:
            self.state.successful[parent_idx].n_i += extra_n_i

    def record_child(self, parent_idx: int) -> None:
        """Increment n_children for the parent that spawned a new task."""
        if 0 <= parent_idx < self.size:
            self.state.successful[parent_idx].n_children += 1

    def record_solved_child(self, parent_idx: int) -> None:
        """Increment n_solved for the parent that produced a solved-after-biting child."""
        if 0 <= parent_idx < self.size:
            self.state.successful[parent_idx].n_solved += 1

    def child_prior_from_parent(self, parent_idx: int) -> tuple[float, float]:
        """Return (prior_alpha, prior_beta) a child should inherit from this parent.

        The child's prior is the parent's current posterior — reflecting that a
        child generated from a productive seed is structurally likely to be
        productive itself, rather than starting from total ignorance.
        """
        if not (0 <= parent_idx < self.size):
            return 1.0, 1.0
        p = self.state.successful[parent_idx]
        alpha = p.prior_alpha + p.n_solved
        beta_param = p.prior_beta + (p.n_i - p.n_solved)
        return float(alpha), float(beta_param)

    @property
    def size(self) -> int:
        return len(self.state.successful)
