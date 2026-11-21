import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.density import (
    choose_eps, dbscan_noise_flags, fit_dbscan, knn_distance_scores,
)
from src.anomaly_detection.regimes import (
    REGIME_FEATURES, choose_k, cluster_stability, fit_kmeans, regime_matrix, time_concentration,
)
from src.data_processing import schema


@pytest.fixture(scope="module")
def blobs():
    """Three clearly separated groups in 2D."""
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10]])
    return np.vstack([c + rng.normal(0, 0.5, (150, 2)) for c in centres])


def test_regime_features_exclude_absolute_temperatures_and_banned_columns():
    assert "Air temperature [K]" not in REGIME_FEATURES
    assert "Process temperature [K]" not in REGIME_FEATURES
    assert set(REGIME_FEATURES).isdisjoint(schema.LEAKAGE_COLUMNS + [schema.TARGET])
    X = pd.DataFrame({"Type": ["L"], "Air temperature [K]": [300.0], "Process temperature [K]": [310.0],
                      "Rotational speed [rpm]": [1500], "Torque [Nm]": [40.0], "Tool wear [min]": [100]})
    out = regime_matrix(X)
    assert list(out.columns) == REGIME_FEATURES and out["temp_diff_K"].iloc[0] == pytest.approx(10.0)


def test_choose_k_finds_three_separated_groups(blobs):
    table = choose_k(blobs, range(2, 7))
    assert int(table.sort_values("silhouette").iloc[-1]["k"]) == 3
    assert table["silhouette"].between(-1, 1).all()
    assert table["inertia"].is_monotonic_decreasing


def test_stability_high_for_real_groups_low_for_no_structure(blobs):
    uniform = np.random.default_rng(1).random((450, 2))
    assert cluster_stability(blobs, 3) > 0.95
    assert cluster_stability(uniform, 6) < cluster_stability(blobs, 3)


def test_time_concentration_detects_a_time_period_cluster():
    udi = np.arange(1, 1001)
    time_aligned = np.where(udi <= 100, 0, 1)             # cluster 0 is the first tenth of the rows
    result = time_concentration(udi, time_aligned).set_index("cluster")
    assert result.loc[0, "max_decile_share"] == pytest.approx(1.0)
    spread = np.tile([0, 1], 500)                          # alternates through time
    assert time_concentration(udi, spread)["max_decile_share"].max() < 0.2


def test_knn_distance_ranks_an_isolated_point_highest(blobs):
    query = np.vstack([blobs[:5], [[30.0, 30.0]]])
    scores = knn_distance_scores(blobs, query, k=10)
    assert scores.argmax() == 5


def test_knn_distance_skips_the_point_itself_for_reference_rows(blobs):
    own = knn_distance_scores(blobs, blobs, k=5, query_is_reference=True)
    assert (own > 0).all()


def test_dbscan_flags_far_points_as_noise_and_near_points_as_clustered(blobs):
    k = 8
    eps = choose_eps(knn_distance_scores(blobs, blobs, k, query_is_reference=True), 95)
    model = fit_dbscan(blobs, eps, k)
    flags = dbscan_noise_flags(model, np.array([[0.1, 0.1], [10.2, -0.1], [40.0, 40.0]]), eps)
    assert list(flags) == [False, False, True]
    assert (model.labels_ == -1).mean() < 0.1


def test_choose_eps_is_a_percentile():
    scores = np.arange(100, dtype=float)
    assert choose_eps(scores, 95) == pytest.approx(np.percentile(scores, 95))


def test_kmeans_is_reproducible(blobs):
    a, b = fit_kmeans(blobs, 3).labels_, fit_kmeans(blobs, 3).labels_
    assert (a == b).all()
    