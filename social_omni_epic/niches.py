"""Deduplication pre-filter and niche assignment via k-means (§6.4).

Dedup: cosine similarity > 0.92 between a candidate and any archive member → reject.
Niche: k-means on archive embeddings, k = max(8, min(15, archive_size // 8)), refit every 10.
"""
from __future__ import annotations

import numpy as np


def cosine_dedup(
    candidate_embedding: list[float],
    archive_embeddings: list[list[float]],
    threshold: float = 0.92,
) -> bool:
    """Return True if the candidate is too similar to any archive member (should be rejected).

    Returns False (pass) when archive is empty or when the max cosine similarity is below threshold.
    """
    if not archive_embeddings:
        return False
    cand = np.array(candidate_embedding, dtype=float)
    cand_norm = np.linalg.norm(cand)
    if cand_norm == 0:
        return False
    cand_unit = cand / cand_norm

    arch = np.array(archive_embeddings, dtype=float)
    norms = np.linalg.norm(arch, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    arch_unit = arch / norms

    similarities = arch_unit @ cand_unit
    return bool(np.max(similarities) > threshold)


class NicheManager:
    """Assign generated scenarios to niches via k-means on abstract embeddings.

    Refits every `refit_every` new scenarios (default 10). Niche IDs are stable
    across refits within a run (centroids are stored); after refit, all existing
    assignments are invalidated and must be reassigned if needed (we only use the
    current assignment for new scenarios — historical records keep their stored niche_id).
    """

    def __init__(self, refit_every: int = 10):
        self.refit_every = refit_every
        self._centroids: np.ndarray | None = None
        self._n_since_refit: int = 0
        self._total_assigned: int = 0

    @property
    def n_niches(self) -> int:
        return len(self._centroids) if self._centroids is not None else 0

    def fit(self, embeddings: list[list[float]]) -> None:
        """Recompute k-means centroids from the given embeddings."""
        if len(embeddings) < 2:
            self._centroids = None
            return
        arr = np.array(embeddings, dtype=float)
        n = len(arr)
        k = max(8, min(15, n // 8))
        k = min(k, n)

        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
            km.fit(arr)
            self._centroids = km.cluster_centers_
        except Exception:
            # Fallback: random centroids if sklearn not available.
            indices = np.random.choice(n, size=k, replace=False)
            self._centroids = arr[indices]

        self._n_since_refit = 0

    def assign(self, embedding: list[float]) -> int:
        """Return the niche id (0-indexed) closest to this embedding.

        Returns 0 if no centroids are available yet.
        """
        if self._centroids is None or len(self._centroids) == 0:
            return 0
        vec = np.array(embedding, dtype=float)
        diffs = self._centroids - vec
        dists = np.linalg.norm(diffs, axis=1)
        niche_id = int(np.argmin(dists))
        self._n_since_refit += 1
        self._total_assigned += 1
        return niche_id

    def should_refit(self) -> bool:
        """True when enough new scenarios have been assigned to warrant a refit."""
        return self._n_since_refit >= self.refit_every

    def state_dict(self) -> dict:
        """Serializable state for checkpoint persistence."""
        return {
            "centroids": self._centroids.tolist() if self._centroids is not None else None,
            "n_since_refit": self._n_since_refit,
            "total_assigned": self._total_assigned,
            "refit_every": self.refit_every,
        }

    def load_state_dict(self, d: dict) -> None:
        centroids = d.get("centroids")
        self._centroids = np.array(centroids, dtype=float) if centroids else None
        self._n_since_refit = int(d.get("n_since_refit", 0))
        self._total_assigned = int(d.get("total_assigned", 0))
        self.refit_every = int(d.get("refit_every", self.refit_every))
