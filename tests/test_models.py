import numpy as np
import pandas as pd
import pytest

from src.classification.cross_validation import METRIC_COLUMNS, compare_models, cross_validate_model
from src.classification.models import get_models
from src.classification.pipelines import build_model_pipeline
from src.classification.splits import stratified_split
from src.data_processing import schema


@pytest.fixture(scope="module")
def train_data():
    """1500 rows where failures follow simple physical rules plus a little noise."""
    rng = np.random.default_rng(3)
    n = 1500
    air = 300 + rng.normal(0, 2, n)
    df = pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"L{i}" for i in range(n)],
            "Type": rng.choice(list("LMH"), n, p=[0.6, 0.3, 0.1]),
            "Air temperature [K]": air,
            "Process temperature [K]": air + 9.5 + rng.normal(0, 1.2, n),
            "Rotational speed [rpm]": rng.normal(1500, 170, n).astype(int),
            "Torque [Nm]": np.clip(rng.normal(40, 10, n), 4, 70),
            "Tool wear [min]": rng.integers(0, 250, n),
        }
    )
    power = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
    df["HDF"] = (
        ((df["Process temperature [K]"] - air) < 8.6) & (df["Rotational speed [rpm]"] < 1380)
    ).astype(int)
    df["PWF"] = ((power < 3500) | (power > 9000)).astype(int)
    df["OSF"] = (df["Tool wear [min]"] * df["Torque [Nm]"] > 11000).astype(int)
    df["TWF"] = 0
    df["RNF"] = 0
    df[schema.TARGET] = (df[["HDF", "PWF", "OSF"]].sum(axis=1) > 0).astype(int)
    X_train, _, y_train, _ = stratified_split(df)
    return X_train, y_train


def test_all_expected_models_present(train_data):
    _, y = train_data
    assert list(get_models(y)) == [
        "Baseline (prior)", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost",
    ]


def test_imbalance_handling_is_configured(train_data):
    _, y = train_data
    models = get_models(y)
    ratio = (len(y) - y.sum()) / y.sum()
    assert models["Logistic Regression"].class_weight == "balanced"
    assert models["Decision Tree"].class_weight == "balanced"
    assert models["Random Forest"].class_weight == "balanced_subsample"
    assert models["XGBoost"].scale_pos_weight == pytest.approx(ratio)


def test_pipeline_column_names_are_xgboost_safe(train_data):
    X, y = train_data
    pipe = build_model_pipeline(get_models(y, n_estimators=5)["XGBoost"]).fit(X, y)
    names = pipe[:-1].transform(X).columns
    assert not any(ch in name for name in names for ch in "[]<")


def test_out_of_fold_predictions_cover_every_row_once(train_data):
    X, y = train_data
    model = get_models(y, n_estimators=10)["Random Forest"]
    folds, oof = cross_validate_model("Random Forest", model, X, y, use_physical=True, n_splits=3)
    assert len(folds) == 3
    assert oof.notna().all() and oof.index.equals(X.index)
    assert oof.between(0, 1).all()


def test_compare_models_table_shape_and_baseline_behaviour(train_data):
    X, y = train_data
    summary, oofs = compare_models(X, y, use_physical=True, n_estimators=15, n_splits=3)
    assert len(summary) == 5 and set(oofs) == set(summary["model"])
    for col in METRIC_COLUMNS:
        assert f"{col}_mean" in summary and f"{col}_std" in summary
    base = summary.set_index("model").loc["Baseline (prior)"]
    assert base["recall_mean"] == 0 and base["roc_auc_mean"] == pytest.approx(0.5)
    rf = summary.set_index("model").loc["Random Forest"]
    assert rf["pr_auc_mean"] > base["pr_auc_mean"]   # a real model beats the floor