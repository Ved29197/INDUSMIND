"""K-means operating regimes.

Segments machines into groups with similar operating conditions. Absolute air and process
temperatures are deliberately left out: air temperature drifts with the row order (a hidden
time axis), so clusters built on it would mostly be time periods, not operating conditions.
The temperature GAP is kept, because cooling margin matters mechanically.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.classification.splits import RANDOM_STATE
from src.feature_engineering.physical import add_physical_features

REGIME_FEATURES = [
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "power_W", "wear_torque", "temp_diff_K",
]


def regime_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """The six operating-condition features, in original units."""
    return add_physical_features(X)[REGIME_FEATURES]


def choose_k(Z, k_range=range(2, 9), seed: int = RANDOM_STATE, sample_size: int = 3000) -> pd.DataFrame:
    """Inertia and silhouette for each candidate number of clusters.

    Silhouette runs from -1 to 1: above about 0.5 means clearly separated groups, below
    about 0.25 means weak structure (a convenient partition, not natural groups).
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(sample_size, len(Z)), random_state=seed)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def fit_kmeans(Z, k: int, seed: int = RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)


def cluster_stability(Z, k: int, seeds=(0, 1, 2, 3, 4)) -> float:
    """Mean agreement (adjusted Rand index) between clusterings from different random starts.
    Near 1: the same groups come back every time. Much lower: the grouping is arbitrary."""
    runs = [KMeans(n_clusters=k, n_init=1, random_state=s).fit(Z).labels_ for s in seeds]
    scores = [adjusted_rand_score(runs[i], runs[j])
              for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return float(np.mean(scores))


def time_concentration(udi, labels) -> pd.DataFrame:
    """How spread each cluster is along the row order (UDI).

    max_decile_share is the largest fraction of a cluster's rows falling in one tenth of the
    UDI range: 0.10 means perfectly spread out, near 1.0 means the cluster is a time period.
    """
    deciles = pd.qcut(pd.Series(np.asarray(udi)), 10, labels=False)
    frame = pd.DataFrame({"cluster": np.asarray(labels), "decile": deciles.values, "udi": np.asarray(udi)})
    rows = []
    for cluster, group in frame.groupby("cluster"):
        shares = group["decile"].value_counts(normalize=True)
        rows.append({"cluster": cluster, "rows": len(group), "median_udi": group["udi"].median(),
                     "max_decile_share": shares.max(), "busiest_decile": int(shares.idxmax())})
    return pd.DataFrame(rows)