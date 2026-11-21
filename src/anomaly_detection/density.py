"""Density-based outlier detection: k-nearest-neighbour distance and DBSCAN.

DBSCAN calls a point NOISE when too few other points lie within distance eps. The distance
to the k-th nearest neighbour is the continuous version of the same idea, so it can be
scored like Isolation Forest.
"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def knn_distance_scores(Z_reference, Z_query, k: int, query_is_reference: bool = False):
    """Distance from each query point to its k-th nearest REFERENCE point.

    When the query set is the reference set itself, a point's own zero distance is skipped.
    Higher = more isolated.
    """
    nn = NearestNeighbors(n_neighbors=k + 1 if query_is_reference else k).fit(Z_reference)
    distances, _ = nn.kneighbors(Z_query)
    return distances[:, -1]


def choose_eps(train_knn_scores, percentile: float = 95.0) -> float:
    """Rule fixed in advance: eps = a high percentile of the training k-distances."""
    return float(np.percentile(train_knn_scores, percentile))


def fit_dbscan(Z_train, eps: float, min_samples: int) -> DBSCAN:
    return DBSCAN(eps=eps, min_samples=min_samples).fit(Z_train)


def dbscan_noise_flags(model: DBSCAN, Z_new, eps: float) -> np.ndarray:
    """True where a new point would be NOISE: farther than eps from every core sample."""
    if len(model.components_) == 0:
        return np.ones(len(Z_new), dtype=bool)
    nn = NearestNeighbors(n_neighbors=1).fit(model.components_)
    distances, _ = nn.kneighbors(Z_new)
    return distances[:, 0] > eps
