"""Isolation Forest anomaly detection.

Unsupervised: it never sees failure labels. It scores how UNUSUAL a machine's readings are
compared with the bulk of the data. Unusual is not the same as failed; labels are used only
afterwards, as an external yardstick.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.classification.splits import RANDOM_STATE
from src.data_processing import schema
from src.feature_engineering.physical import PHYSICAL_FEATURES, add_physical_features


def build_isolation_pipeline(use_physical: bool = True, n_estimators: int = 300,
                             seed: int = RANDOM_STATE) -> Pipeline:
    """Numeric readings (optionally plus physical features) -> Isolation Forest.

    Product Type is left out on purpose: as a rare category, 'H' machines would look
    unusual for reasons that have nothing to do with how they are running.
    """
    columns = list(schema.NUMERIC_FEATURES)
    steps = []
    if use_physical:
        steps.append(("physical", FunctionTransformer(add_physical_features)))
        columns += PHYSICAL_FEATURES
    select = ColumnTransformer(
        [("numeric", "passthrough", columns)], verbose_feature_names_out=False
    ).set_output(transform="pandas")
    steps.append(("select", select))
    steps.append(("forest", IsolationForest(n_estimators=n_estimators, random_state=seed, n_jobs=-1)))
    return Pipeline(steps)


def anomaly_scores(pipeline: Pipeline, X) -> np.ndarray:
    """Higher = more unusual. (scikit-learn's own score is the negative of this.)"""
    return -pipeline.score_samples(X)


def cutoff_for_fraction(reference_scores, fraction: float) -> float:
    """Score above which the top `fraction` of REFERENCE rows lie. Uses no labels."""
    return float(np.quantile(reference_scores, 1 - fraction))


def lift_at_cutoff(y_true, scores, cutoff: float) -> dict:
    """How well the flagged (score >= cutoff) machines line up with real failures."""
    y = np.asarray(y_true)
    flagged = np.asarray(scores) >= cutoff
    n_flagged = int(flagged.sum())
    failures_flagged = int((flagged & (y == 1)).sum())
    prevalence = y.mean()
    precision = failures_flagged / n_flagged if n_flagged else float("nan")
    return {
        "flagged": n_flagged,
        "failures_flagged": failures_flagged,
        "precision": precision,
        "recall": failures_flagged / y.sum() if y.sum() else float("nan"),
        "lift": precision / prevalence if n_flagged else float("nan"),
    }