import pickle

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import (
    Calibrator, calibrate_oof, calibration_summary, reliability_table,
)
from src.evaluation.thresholds import (
    best_cost_threshold, cost_curve, theoretical_threshold, threshold_for_recall,
)


@pytest.fixture(scope="module")
def overconfident():
    """True risks are small, but the model's scores overstate them 5x."""
    rng = np.random.default_rng(0)
    true_p = rng.beta(0.6, 12, 20000)
    y = pd.Series((rng.random(20000) < true_p).astype(int))
    scores = pd.Series(np.clip(true_p * 5, 0, 1), name="model")
    return y, scores


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_improves_an_overconfident_model(overconfident, method):
    y, scores = overconfident
    calibrated = calibrate_oof(scores, y, method)
    assert calibrated.notna().all() and calibrated.between(0, 1).all()
    assert calibrated.index.equals(scores.index)
    raw = calibration_summary(y, scores)
    after = calibration_summary(y, calibrated)
    assert after["brier"] < raw["brier"]
    assert after["ece"] < raw["ece"]


def test_invalid_calibration_method_rejected():
    with pytest.raises(ValueError):
        Calibrator("magic")


def test_calibrator_survives_pickling(overconfident):
    y, scores = overconfident
    cal = Calibrator("sigmoid").fit(scores, y)
    restored = pickle.loads(pickle.dumps(cal))
    assert np.allclose(cal.predict(scores[:50]), restored.predict(scores[:50]))


def test_reliability_table_accounts_for_every_row(overconfident):
    y, scores = overconfident
    table = reliability_table(y, scores)
    assert table["rows"].sum() == len(y)
    assert table["observed_rate"].between(0, 1).all()


def test_cost_curve_counts_are_consistent():
    y = np.array([0] * 90 + [1] * 10)
    p = np.linspace(0, 1, 100)
    curve = cost_curve(y, p, cost_fn=10, cost_fp=1)
    assert ((curve["tp"] + curve["fn"]) == 10).all()
    assert ((curve["fp"] + curve["tn"]) == 90).all()
    assert (curve["total_cost"] == curve["fn"] * 10 + curve["fp"] * 1).all()


def test_theoretical_threshold_formula():
    assert theoretical_threshold(cost_fn=10, cost_fp=1) == pytest.approx(1 / 11)
    assert theoretical_threshold(cost_fn=1, cost_fp=1) == pytest.approx(0.5)


def test_best_threshold_matches_theory_for_calibrated_probabilities():
    rng = np.random.default_rng(1)
    p = rng.beta(0.5, 10, 300000)
    y = (rng.random(300000) < p).astype(int)      # p is perfectly calibrated by construction
    for ratio in (5, 10, 25):
        best = best_cost_threshold(cost_curve(y, p, cost_fn=ratio, cost_fp=1))
        assert abs(best["threshold"] - theoretical_threshold(ratio, 1)) <= 0.03


def test_threshold_for_recall_is_the_highest_that_qualifies():
    rng = np.random.default_rng(2)
    p = rng.random(5000)
    y = (rng.random(5000) < p ** 3).astype(int)
    row = threshold_for_recall(y, p, 0.9)
    assert row["recall"] >= 0.9
    higher = cost_curve(y, p, 1, 1)
    higher = higher[higher["threshold"] > row["threshold"]]
    assert (higher["recall"] < 0.9).all()


def test_unreachable_recall_target_raises():
    y = np.array([0, 1, 1])
    p = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        threshold_for_recall(y, p, 0.9, thresholds=[0.5])