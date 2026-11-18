import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.classification.final_evaluation import hdf_episode_mask
from src.classification.final_model import fit_final_model
from src.classification.splits import stratified_split
from src.data_processing import schema
from src.data_processing.loader import get_features_and_target
from src.evaluation.bootstrap import bootstrap_ci


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(5)
    n = 2000
    air = 300 + np.cumsum(rng.normal(0, 0.05, n))
    d = pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"L{i}" for i in range(n)],
            "Type": rng.choice(list("LMH"), n, p=[0.6, 0.3, 0.1]),
            "Air temperature [K]": air,
            "Process temperature [K]": air + 9.6 + rng.normal(0, 1.0, n),
            "Rotational speed [rpm]": rng.normal(1500, 170, n).astype(int),
            "Torque [Nm]": np.clip(rng.normal(40, 10, n), 4, 70),
            "Tool wear [min]": rng.integers(0, 250, n),
        }
    )
    power = d["Torque [Nm]"] * d["Rotational speed [rpm]"] * 2 * np.pi / 60
    d["HDF"] = (((d["Process temperature [K]"] - air) < 8.6) & (d["Rotational speed [rpm]"] < 1380)).astype(int)
    d["PWF"] = ((power < 3500) | (power > 9000)).astype(int)
    d["OSF"] = (d["Tool wear [min]"] * d["Torque [Nm]"] > 11000).astype(int)
    d["TWF"] = 0
    d["RNF"] = 0
    d[schema.TARGET] = (d[["HDF", "PWF", "OSF"]].sum(axis=1) > 0).astype(int)
    return d


@pytest.fixture(scope="module")
def fitted(df):
    X_train, X_test, y_train, y_test = stratified_split(df)
    model = fit_final_model(X_train, y_train, "Random Forest", n_estimators=20)
    return model, X_train, X_test, y_train, y_test


def test_final_model_gives_valid_calibrated_probabilities(fitted):
    model, X_train, X_test, y_train, y_test = fitted
    p = model.predict_calibrated(X_test)
    assert len(p) == len(X_test) and ((p >= 0) & (p <= 1)).all()
    assert model.oof_calibrated.notna().all() and model.oof_raw.index.equals(X_train.index)
    assert average_precision_score(y_test, p) > y_test.mean()   # beats the base rate


@pytest.mark.parametrize("name", ["Random Forest", "XGBoost"])
def test_final_model_survives_pickling(df, name):
    X_train, X_test, y_train, _ = stratified_split(df)
    model = fit_final_model(X_train, y_train, name, n_estimators=10)
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(model.predict_calibrated(X_test[:50]), restored.predict_calibrated(X_test[:50]))


def test_bootstrap_interval_is_sensible_and_reproducible():
    rng = np.random.default_rng(0)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(int)
    point = average_precision_score(y, p)
    low, high = bootstrap_ci(y, p, average_precision_score, n_boot=200, seed=1)
    assert low < point < high
    assert (low, high) == bootstrap_ci(y, p, average_precision_score, n_boot=200, seed=1)


def test_bootstrap_interval_is_narrow_for_a_perfect_classifier():
    y = np.array([0] * 200 + [1] * 50)
    low, high = bootstrap_ci(y, y.astype(float), average_precision_score, n_boot=100)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)


def test_hdf_episode_mask_covers_first_to_last_event_and_leaves_no_hdf_outside(df):
    mask = hdf_episode_mask(df)
    hdf_udi = df.loc[df["HDF"] == 1, "UDI"]
    assert df.loc[mask, "UDI"].min() == hdf_udi.min() and df.loc[mask, "UDI"].max() == hdf_udi.max()
    assert df.loc[~mask, "HDF"].sum() == 0
    X_train, y_train = get_features_and_target(df[~mask])
    assert len(X_train) == (~mask).sum()
    