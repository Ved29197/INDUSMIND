"""Choosing the decision threshold from costs, not from habit (0.5).

A missed failure (false negative) and an unnecessary inspection (false positive)
cost different amounts. The threshold should reflect that.
"""
import numpy as np
import pandas as pd


def theoretical_threshold(cost_fn: float, cost_fp: float) -> float:
    """For CALIBRATED probabilities the cost-minimising threshold is
    cost_fp / (cost_fp + cost_fn): act when risk x cost_fn > (1 - risk) x cost_fp."""
    return cost_fp / (cost_fp + cost_fn)


def cost_curve(y_true, y_prob, cost_fn: float, cost_fp: float, thresholds=None) -> pd.DataFrame:
    """Confusion counts, recall, precision and total cost at each threshold."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    positives = int(y.sum())
    negatives = len(y) - positives
    rows = []
    for t in thresholds:
        pred = p >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = positives - tp
        rows.append(
            {
                "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": negatives - fp,
                "recall": tp / positives if positives else float("nan"),
                "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
                "total_cost": fn * cost_fn + fp * cost_fp,
            }
        )
    return pd.DataFrame(rows)


def best_cost_threshold(curve: pd.DataFrame) -> pd.Series:
    """The row of the curve with the lowest total cost (lowest threshold on ties)."""
    return curve.loc[curve["total_cost"].idxmin()]


def threshold_for_recall(y_true, y_prob, target_recall: float, thresholds=None) -> pd.Series:
    """The HIGHEST threshold whose recall still meets the target (fewest false alarms)."""
    curve = cost_curve(y_true, y_prob, cost_fn=1, cost_fp=1, thresholds=thresholds)
    ok = curve[curve["recall"] >= target_recall]
    if ok.empty:
        raise ValueError(f"No threshold reaches recall {target_recall}")
    return ok.loc[ok["threshold"].idxmax()]