"""Evaluate Isolation Forest against failure labels (labels are NOT used for fitting).

Run from the project root:
    python -m src.anomaly_detection.evaluate_isolation

Expectations were written down first: docs/results/anomaly_expectations.md.
No tuning: both feature sets use the same settings, and the flag cutoff comes from the
training scores only.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.anomaly_detection.isolation import (
    anomaly_scores, build_isolation_pipeline, cutoff_for_fraction, lift_at_cutoff,
)
from src.classification.splits import stratified_split
from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH, load_ai4i
from src.evaluation.bootstrap import bootstrap_ci
from src.feature_engineering.physical import PHYSICAL_FEATURES, add_physical_features

RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
FLAG_FRACTION = 0.05          # flag the 5% most unusual machines
CAUSES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def main():
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, X_test, y_train, y_test = stratified_split(df)
    print(f"Train {len(X_train)} rows ({int(y_train.sum())} failures), "
          f"test {len(X_test)} rows ({int(y_test.sum())} failures). "
          f"Base failure rate {y_train.mean():.2%}.\n")

    rows, fitted = [], {}
    for use_physical in (False, True):
        pipe = build_isolation_pipeline(use_physical).fit(X_train)     # labels never used
        s_train, s_test = anomaly_scores(pipe, X_train), anomaly_scores(pipe, X_test)
        cutoff = cutoff_for_fraction(s_train, FLAG_FRACTION)
        fitted[use_physical] = (pipe, s_test, cutoff)
        for name, s, y in (("train (in-sample)", s_train, y_train), ("test", s_test, y_test)):
            stats = lift_at_cutoff(y, s, cutoff)
            row = {
                "physical_features": use_physical, "split": name,
                "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
                "base_rate": y.mean(), **stats,
            }
            if name == "test":
                row["pr_auc_ci_low"], row["pr_auc_ci_high"] = bootstrap_ci(
                    y, s, average_precision_score, 1000)
            rows.append(row)

    table = pd.DataFrame(rows)
    show = table[["physical_features", "split", "pr_auc", "roc_auc", "flagged",
                  "failures_flagged", "precision", "recall", "lift"]]
    print(f"1) ISOLATION FOREST vs failure labels (flagging the top {FLAG_FRACTION:.0%} most unusual)")
    print(show.round(3).to_string(index=False))
    test_rows = table[table["split"] == "test"]
    for _, r in test_rows.iterrows():
        print(f"   test PR-AUC (physical={r['physical_features']}): {r['pr_auc']:.3f} "
              f"95% CI [{r['pr_auc_ci_low']:.3f}, {r['pr_auc_ci_high']:.3f}]  "
              f"(random guessing = {r['base_rate']:.3f})")

    # Which kinds of failure does 'unusual' catch? (cause flags used for analysis only)
    pipe, s_test, cutoff = fitted[True]
    flagged = s_test >= cutoff
    causes = df.loc[X_test.index, CAUSES]
    failed = (y_test.values == 1)
    print("\n2) WHICH FAILURES LOOK UNUSUAL? (test set, physical features)")
    cause_rows = []
    for cause in CAUSES:
        mask = failed & (causes[cause].values == 1)
        cause_rows.append({"group": cause, "failures": int(mask.sum()),
                           "flagged_as_unusual": int((mask & flagged).sum()),
                           "share_flagged": (mask & flagged).sum() / mask.sum() if mask.sum() else np.nan})
    none_mask = failed & (causes.sum(axis=1).values == 0)
    cause_rows.append({"group": "no cause flagged", "failures": int(none_mask.sum()),
                       "flagged_as_unusual": int((none_mask & flagged).sum()),
                       "share_flagged": (none_mask & flagged).sum() / none_mask.sum() if none_mask.sum() else np.nan})
    cause_table = pd.DataFrame(cause_rows)
    print(cause_table.round(3).to_string(index=False))
    print("   (a failure can have several causes, so groups can overlap)")

    # What do unusual machines that did NOT fail look like?
    phys = add_physical_features(X_test)[PHYSICAL_FEATURES]
    groups = {
        "flagged, failed": flagged & failed,
        "flagged, did NOT fail": flagged & ~failed,
        "not flagged": ~flagged,
    }
    profile = pd.DataFrame({name: phys[mask].median() for name, mask in groups.items()}).T
    profile.insert(0, "machines", [int(mask.sum()) for mask in groups.values()])
    print("\n3) MEDIAN PHYSICAL READINGS BY GROUP (test set)")
    print(profile.round(1).to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "isolation_forest_eval.csv", index=False)
    cause_table.to_csv(RESULTS_DIR / "isolation_forest_by_cause.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()