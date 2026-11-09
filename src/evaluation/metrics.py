"""Evaluation metrics for an imbalanced failure-prediction problem."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return the full metric set for predicted failure probabilities.

    Accuracy is included only so it can be shown next to the metrics that matter.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positives = tp + fn
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_negative_rate": fn / positives if positives else float("nan"),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "prevalence": positives / len(y_true),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }