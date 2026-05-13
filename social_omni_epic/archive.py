import json
from pathlib import Path
from .data_models import SocialScenario, ArchiveState


class Archive:
    def __init__(self, checkpoint_dir: str):
        self.state = ArchiveState()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def add_successful(self, scenario: SocialScenario):
        self.state.successful.append(scenario)

    def add_failed_generation(self, info: dict):
        self.state.failed_generation.append(info)

    def add_failed_interestingness(self, scenario: SocialScenario):
        self.state.failed_interestingness.append(scenario)

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

    @property
    def size(self) -> int:
        return len(self.state.successful)
