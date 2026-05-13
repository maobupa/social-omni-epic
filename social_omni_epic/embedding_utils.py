import numpy as np
from scipy.spatial.distance import cosine


def get_similar_scenarios(
    query_embedding: list[float],
    archive_embeddings: list[list[float]],
    num_returns: int = 5,
) -> list[int]:
    if not archive_embeddings:
        return []
    distances = [cosine(query_embedding, emb) for emb in archive_embeddings]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:num_returns].tolist()


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
