"""Convert Sotopia dimension scores into solved/unsolved + learning progress.

`progress_score` is a heuristic, not a Sotopia metric. Provenance:
  - the *role* (a continuous signal feeding a learning-progress curriculum)
    comes from the OMNI / OMNI-EPIC open-ended-learning lineage;
  - restricting the blend to goal + relationship + knowledge is motivated by
    the sotopia-rl finding that those three dimensions are the meaningful,
    learnable ones;
  - the per-dimension weights are a config knob, not derived from any paper.

Sotopia's own dimensions have DIFFERENT ranges, so each is normalized to
[0, 1] by its own range before the weighted blend — otherwise relationship
(-5..5) would contribute asymmetrically and the score could never reach 1.0.
"""
from collections import defaultdict, deque
from typing import Deque, Optional


# Score ranges per Sotopia dimension (see SotopiaDimensions field docs).
DIM_RANGES: dict[str, tuple[float, float]] = {
    "believability": (0.0, 10.0),
    "relationship": (-5.0, 5.0),
    "knowledge": (0.0, 10.0),
    "secret": (-10.0, 0.0),
    "social_rules": (-10.0, 0.0),
    "financial_and_material_benefits": (-5.0, 5.0),
    "goal": (0.0, 10.0),
}


def normalize_dimension(name: str, value: float) -> float:
    """Map a raw Sotopia dimension score to [0, 1] using its own range."""
    lo, hi = DIM_RANGES.get(name, (0.0, 10.0))
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


class SuccessDetector:
    def __init__(
        self,
        goal_threshold: float = 7.0,
        weights: dict | None = None,
        lp_window: int = 10,
    ):
        # Weighted blend over the three sotopia-rl "meaningful" dimensions.
        self.weights = weights or {"goal": 0.6, "relationship": 0.2, "knowledge": 0.2}
        self.goal_threshold = goal_threshold
        self.lp_window = lp_window
        self._history: dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.lp_window)
        )

    def is_solved(self, scores: dict, goal_achieved: Optional[bool] = None) -> bool:
        """Return True if the episode is solved.

        Prefers the evaluator's binary goal_achieved judgment when available.
        Falls back to goal score >= threshold when goal_achieved is None.
        """
        if goal_achieved is not None:
            return goal_achieved
        return float(scores.get("goal", 0.0)) >= self.goal_threshold

    def compute_progress_score(self, scores: dict) -> float:
        """Weighted blend of per-dimension-normalized scores, in [0, 1].

        Each dimension is first normalized to [0, 1] by its own range, then
        combined with `self.weights`. Dividing by the weight sum keeps the
        result in [0, 1] even if the weights don't sum to 1.
        """
        total = 0.0
        weight_sum = 0.0
        for k, w in self.weights.items():
            total += w * normalize_dimension(k, scores.get(k, 0.0))
            weight_sum += w
        return total / weight_sum if weight_sum > 0 else 0.0

    def record_and_compute_learning_progress(
        self, interaction_type: str, progress: float
    ) -> float:
        """Slope of progress over the recent window for this interaction_type.

        Returns the difference between the mean of the second half and the
        first half of the window. Positive = learning, negative = regressing.
        """
        key = interaction_type or "_default"
        self._history[key].append(progress)
        window = list(self._history[key])
        if len(window) < 4:
            return 0.0
        mid = len(window) // 2
        first = window[:mid]
        second = window[mid:]
        return (sum(second) / len(second)) - (sum(first) / len(first))
