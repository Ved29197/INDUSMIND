"""Probability calibration: make "30% risk" mean about 30%.

Class weights (used to find rare failures) distort predicted probabilities.
A calibrator learns a mapping from raw model scores to honest probabilities.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

RELIABILITY_BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw probabilities to calibrated ones.

    method="sigmoid": Platt scaling (a 2-parameter S-curve; stable with few failures).
    method="isotonic": a flexible step function (needs more data; can overfit).
    """

    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError("method must be 'sigmoid' or 'isotonic'")
        self.method = method

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y)
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        else:
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return self.model_.predict(p)
        return self.model_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def calibrate_oof(p_oof: pd.Series, y: pd.Series, method: str, n_splits: int = 5, seed: int = 42):
    """Honestly calibrated out-of-fold probabilities.

    The calibrator is fitted on some rows and applied to different rows, so the
    calibrated values are never scored on data the calibrator learned from.
    """
    out = pd.Series(np.nan, index=p_oof.index, name=p_oof.name)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in cv.split(p_oof, y):
        cal = Calibrator(method).fit(p_oof.iloc[fit_idx], y.iloc[fit_idx])
        out.iloc[apply_idx] = cal.predict(p_oof.iloc[apply_idx])
    return out


def reliability_table(y_true, y_prob, bins=RELIABILITY_BINS) -> pd.DataFrame:
    """Per probability band: how many rows, mean predicted risk, observed failure rate."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df["band"] = pd.cut(df["p"], bins, include_lowest=True)
    table = (
        df.groupby("band", observed=True)
        .agg(rows=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    table["band"] = table["band"].astype(str)
    return table


def calibration_summary(y_true, y_prob) -> dict:
    """Brier score, skill versus always predicting the base rate, ECE and log loss."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    brier = brier_score_loss(y, p)
    base_rate = y.mean()
    brier_ref = base_rate * (1 - base_rate)   # Brier score of always predicting the base rate
    table = reliability_table(y, p)
    ece = float(np.sum(table["rows"] * (table["mean_predicted"] - table["observed_rate"]).abs()) / len(y))
    return {
        "brier": brier,
        "brier_skill": 1 - brier / brier_ref,
        "ece": ece,
        "log_loss": log_loss(y, np.clip(p, _EPS, 1 - _EPS)),
    }