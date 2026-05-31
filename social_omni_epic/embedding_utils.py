import numpy as np
from typing import Optional


def get_similar_scenarios(
    query_embedding: list[float],
    archive_embeddings: list[list[float]],
    num_returns: int = 5,
    source_ids: Optional[list[str]] = None,
    agent_idxs: Optional[list[int]] = None,
    preferred_agent_idx: Optional[int] = None,
) -> list[int]:
    """Return indices of the top-K most similar archive entries.

    When source_ids is provided, perspective-aware deduplication runs BEFORE
    cosine similarity: for each source_scenario_id, only one entry enters the
    candidate pool. The entry matching preferred_agent_idx is preferred; if
    neither or both match, the first-seen entry is kept.
    """
    n = len(archive_embeddings)
    if n == 0:
        return []

    # Build candidate pool: at most one entry per source_scenario_id.
    if source_ids is not None:
        seen: dict[str, int] = {}   # source_id -> chosen archive index
        for i in range(n):
            sid = source_ids[i]
            if sid not in seen:
                seen[sid] = i
            elif (preferred_agent_idx is not None
                  and agent_idxs is not None
                  and agent_idxs[i] == preferred_agent_idx):
                seen[sid] = i   # replace with perspective-matching entry
        candidate_indices = list(seen.values())
    else:
        candidate_indices = list(range(n))

    if not candidate_indices:
        return []

    # Cosine similarity on the reduced candidate pool.
    emb_arr = np.array([archive_embeddings[i] for i in candidate_indices], dtype=float)
    q = np.array(query_embedding, dtype=float)
    norms = np.linalg.norm(emb_arr, axis=1) * np.linalg.norm(q) + 1e-9
    sims = emb_arr @ q / norms
    sorted_pos = np.argsort(-sims)
    return [candidate_indices[int(p)] for p in sorted_pos[:num_returns]]


def compute_cell_coverage(embeddings: np.ndarray, num_bins: int = 20) -> float:
    from sklearn.decomposition import PCA

    if len(embeddings) < 2:
        return 0.0

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    mins = reduced.min(axis=0)
    maxs = reduced.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normalized = (reduced - mins) / ranges

    cells = set()
    for point in normalized:
        x = min(int(point[0] * num_bins), num_bins - 1)
        y = min(int(point[1] * num_bins), num_bins - 1)
        cells.add((x, y))

    return len(cells) / (num_bins * num_bins)
