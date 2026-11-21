import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection.isolation import (
    anomaly_scores, build_isolation_pipeline, cutoff_for_fraction, lift_at_cutoff,
)
from src.data_processing import schema
from src.feature_engineering.physical import PHYSICAL_FEATURES


@pytest.fixture(scope="module")
def readings():
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame(
        {
            "Type": rng.choice(list("LMH"), n, p=[0.6, 0.3, 0.1]),
            "Air temperature [K]": 300 + rng.normal(0, 2, n),
            "Process temperature [K]": 310 + rng.normal(0, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 150, n),
            "Torque [Nm]": np.clip(rng.normal(40, 8, n), 5, 70),
            "Tool wear [min]": rng.integers(0, 250, n).astype(float),
        }
    )
    return X[schema.FEATURE_COLUMNS]


def test_an_obvious_outlier_gets_the_top_score(readings):
    X = readings.copy()
    X.loc[X.index[0], "Torque [Nm]"] = 400.0
    pipe = build_isolation_pipeline(use_physical=True, n_estimators=100).fit(X)
    scores = anomaly_scores(pipe, X)
    assert scores.argmax() == 0 and len(scores) == len(X)


def test_feature_sets_and_type_exclusion(readings):
    for use_physical in (False, True):
        pipe = build_isolation_pipeline(use_physical, n_estimators=10).fit(readings)
        names = list(pipe[:-1].transform(readings).columns)
        assert set(schema.NUMERIC_FEATURES) <= set(names)
        assert (set(PHYSICAL_FEATURES) <= set(names)) is use_physical
        assert not any(name.startswith("Type") for name in names)
        assert set(names).isdisjoint(schema.LEAKAGE_COLUMNS + [schema.TARGET])


def test_scores_are_reproducible_with_a_seed(readings):
    a = anomaly_scores(build_isolation_pipeline(n_estimators=20).fit(readings), readings)
    b = anomaly_scores(build_isolation_pipeline(n_estimators=20).fit(readings), readings)
    assert np.allclose(a, b)


def test_cutoff_flags_about_the_requested_fraction():
    scores = np.arange(1000, dtype=float)
    cutoff = cutoff_for_fraction(scores, 0.05)
    assert abs((scores >= cutoff).mean() - 0.05) < 0.005


def test_lift_for_a_perfectly_aligned_score():
    y = np.array([0] * 90 + [1] * 10)
    stats = lift_at_cutoff(y, y.astype(float), cutoff=1.0)
    assert stats["flagged"] == 10 and stats["failures_flagged"] == 10
    assert stats["precision"] == 1 and stats["recall"] == 1
    assert stats["lift"] == pytest.approx(10.0)          # 1.0 / base rate 0.10


def test_lift_is_about_one_for_a_random_score():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.05).astype(int)
    stats = lift_at_cutoff(y, rng.random(20000), cutoff=0.9)
    assert 0.7 < stats["lift"] < 1.3