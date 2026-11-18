"""Bootstrap confidence intervals: how much a test metric could vary by luck."""
import numpy as np


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Resample the test rows with replacement and recompute the metric.

    Returns (low, high) covering the middle (1 - alpha) of the resampled values.
    Resamples that contain only one class are skipped (the metric is undefined).
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric_fn(y[idx], p[idx]))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)
