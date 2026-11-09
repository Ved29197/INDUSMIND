import numpy as np
import pandas as pd
import pytest

from src.classification.pipelines import build_preprocessor
from src.classification.splits import blocked_split, make_cv, stratified_split
from src.data_processing import schema
from src.evaluation.metrics import evaluate_binary
from src.feature_engineering.physical import PHYSICAL_FEATURES, add_physical_features


@pytest.fixture
def big_df():
    """1000 valid rows with about 3.4% failures, shuffled row order."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"L{i}" for i in range(n)],
            "Type": rng.choice(list("LMH"), n, p=[0.6, 0.3, 0.1]),
            "Air temperature [K]": 300 + rng.normal(0, 2, n),
            "Process temperature [K]": 310 + rng.normal(0, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 170, n).astype(int),
            "Torque [Nm]": np.clip(rng.normal(40, 10, n), 4, 70),
            "Tool wear [min]": rng.integers(0, 250, n),
        }
    )
    for mode in schema.FAILURE_MODE_COLUMNS:
        df[mode] = 0
    fail_idx = rng.choice(n, 34, replace=False)
    df.loc[fail_idx, "TWF"] = 1
    df[schema.TARGET] = (df[schema.FAILURE_MODE_COLUMNS].sum(axis=1) > 0).astype(int)
    return df.sample(frac=1, random_state=1)  # shuffle row order


def test_physical_features_known_values():
    row = pd.DataFrame(
        {
            "Air temperature [K]": [300.0], "Process temperature [K]": [309.0],
            "Rotational speed [rpm]": [1500], "Torque [Nm]": [40.0], "Tool wear [min]": [100],
        }
    )
    out = add_physical_features(row)
    assert out["temp_diff_K"].iloc[0] == pytest.approx(9.0)
    assert out["power_W"].iloc[0] == pytest.approx(40 * 1500 * 2 * np.pi / 60)
    assert out["wear_torque"].iloc[0] == pytest.approx(4000.0)


def test_physical_features_need_no_label_columns_and_do_not_mutate():
    row = pd.DataFrame(
        {
            "Air temperature [K]": [300.0], "Process temperature [K]": [309.0],
            "Rotational speed [rpm]": [1500], "Torque [Nm]": [40.0], "Tool wear [min]": [100],
        }
    )
    out = add_physical_features(row)
    assert set(out.columns) - set(row.columns) == set(PHYSICAL_FEATURES)
    assert "temp_diff_K" not in row.columns


def test_stratified_split_sizes_and_failure_rate(big_df):
    X_train, X_test, y_train, y_test = stratified_split(big_df)
    assert len(X_train) == 800 and len(X_test) == 200
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_features_exclude_banned_columns(big_df):
    X_train, X_test, _, _ = stratified_split(big_df)
    banned = set(schema.LEAKAGE_COLUMNS + [schema.TARGET])
    assert banned.isdisjoint(X_train.columns) and banned.isdisjoint(X_test.columns)


def test_split_is_reproducible(big_df):
    a = stratified_split(big_df)[1].index
    b = stratified_split(big_df)[1].index
    assert list(a) == list(b)


def test_blocked_split_test_set_is_strictly_later(big_df):
    X_train, X_test, _, _ = blocked_split(big_df)
    train_udi = big_df.loc[X_train.index, "UDI"]
    test_udi = big_df.loc[X_test.index, "UDI"]
    assert train_udi.max() < test_udi.min()
    assert len(X_test) == 200


def test_cv_folds_keep_failures_in_every_fold(big_df):
    _, _, y_train, _ = stratified_split(big_df)
    for _, val_idx in make_cv().split(np.zeros(len(y_train)), y_train):
        assert y_train.iloc[val_idx].sum() > 0


def test_preprocessor_with_and_without_physical_features(big_df):
    X_train, X_test, _, _ = stratified_split(big_df)
    with_phys = build_preprocessor(use_physical=True).fit(X_train)
    without = build_preprocessor(use_physical=False).fit(X_train)
    out_with, out_without = with_phys.transform(X_test), without.transform(X_test)
    assert set(PHYSICAL_FEATURES) <= set(out_with.columns)
    assert set(PHYSICAL_FEATURES).isdisjoint(out_without.columns)
    assert not out_with.isna().any().any()
    assert {"Type_L", "Type_M", "Type_H"} <= set(out_with.columns)


def test_scaler_learns_from_training_data_only(big_df):
    X_train, X_test, _, _ = stratified_split(big_df)
    pre = build_preprocessor(use_physical=False, scale=True).fit(X_train)
    train_out = pre.transform(X_train)
    assert abs(train_out["Torque [Nm]"].mean()) < 1e-9       # centred on train
    assert abs(pre.transform(X_test)["Torque [Nm]"].mean()) > 1e-9  # test not forced to 0


def test_metrics_for_a_useless_model():
    y = np.array([0] * 96 + [1] * 4)
    m = evaluate_binary(y, np.full(100, 0.04))
    assert m["accuracy"] == pytest.approx(0.96)
    assert m["recall"] == 0 and m["false_negative_rate"] == 1
    assert m["pr_auc"] == pytest.approx(0.04)   # equals prevalence
    assert m["roc_auc"] == pytest.approx(0.5)


def test_metrics_for_a_perfect_model():
    y = np.array([0, 0, 0, 1, 1])
    m = evaluate_binary(y, np.array([0.1, 0.2, 0.3, 0.9, 0.8]))
    assert m["recall"] == 1 and m["precision"] == 1 and m["false_negative_rate"] == 0
    assert m["pr_auc"] == 1 and m["fp"] == 0 and m["fn"] == 0