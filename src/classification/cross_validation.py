"""Cross-validated comparison of the candidate models.

Works on TRAINING data only. The test set is never passed in here.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.classification.models import NEEDS_SCALING, get_models
from src.classification.pipelines import build_model_pipeline
from src.classification.splits import make_cv
from src.evaluation.metrics import evaluate_binary

METRIC_COLUMNS = ["pr_auc", "roc_auc", "recall", "precision", "f1", "false_negative_rate"]


def cross_validate_model(name, model, X, y, use_physical: bool, n_splits: int = 5):
    """Return (per-fold metrics DataFrame, out-of-fold probabilities Series).

    Every training row is predicted exactly once, by a model that never saw it.
    Precision, recall and F1 here use the default 0.5 threshold; threshold
    selection comes in Milestone 4.3.
    """
    scale = name in NEEDS_SCALING
    oof = pd.Series(np.nan, index=X.index, name=name)
    fold_rows = []
    for train_idx, val_idx in make_cv(n_splits).split(X, y):
        pipe = build_model_pipeline(clone(model), use_physical=use_physical, scale=scale)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        oof.iloc[val_idx] = probs
        metrics = evaluate_binary(y.iloc[val_idx], probs)
        fold_rows.append({k: metrics[k] for k in METRIC_COLUMNS})
    return pd.DataFrame(fold_rows), oof


def compare_models(X, y, use_physical: bool, n_estimators: int = 300, n_splits: int = 5):
    """Cross-validate every model. Returns (summary DataFrame, {name: OOF probs})."""
    rows, oofs = [], {}
    for name, model in get_models(y, n_estimators=n_estimators).items():
        folds, oof = cross_validate_model(name, model, X, y, use_physical, n_splits)
        row = {"model": name, "physical_features": use_physical}
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = folds[col].mean()
            row[f"{col}_std"] = folds[col].std()
        rows.append(row)
        oofs[name] = oof
    return pd.DataFrame(rows), oofs