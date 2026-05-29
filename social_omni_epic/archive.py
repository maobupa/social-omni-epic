import json
import math
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
    # UCB1 selection (§4.1)
    # score(task) = C * sqrt(ln(N) / n_i) - D * n_children
    # Tasks with n_i == 0 always get inf → chosen first.
    # ------------------------------------------------------------------

    def ucb1_select(self, C: float = 1.0, D: float = 0.1) -> int:
        """Return archive index of the task selected as anchor by UCB1."""
        n = self.size
        if n == 0:
            return -1
        N = max(self._total_selections, 1)
        scores = []
        for task in self.state.successful:
            if task.n_i == 0:
                scores.append(float("inf"))
            else:
                exploration = C * math.sqrt(math.log(N) / task.n_i)
                penalty = D * task.n_children
                scores.append(exploration - penalty)
        return int(np.argmax(scores))

    def record_selection(self, idx: int, iteration: int) -> None:
        """Mark task at idx as selected; update UCB1 bookkeeping."""
        self._total_selections += 1
        task = self.state.successful[idx]
        task.n_i += 1
        task.last_chosen = iteration

    def record_child(self, parent_idx: int) -> None:
        """Increment n_children for the parent that spawned a new task."""
        if 0 <= parent_idx < self.size:
            self.state.successful[parent_idx].n_children += 1

    @property
    def size(self) -> int:
        return len(self.state.successful)
