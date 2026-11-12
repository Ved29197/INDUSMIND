"""Compare the four models with and without physical features.

Run from the project root:
    python -m src.classification.compare
Uses the TRAINING split only. The test set stays untouched until the final evaluation.
"""
import time

import pandas as pd

from src.classification.cross_validation import METRIC_COLUMNS, compare_models
from src.classification.splits import stratified_split
from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH, load_ai4i

OUTPUT_PATH = PROJECT_ROOT / "docs" / "results" / "cv_comparison.csv"


def format_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact 'mean +/- std' view for printing."""
    shown = summary[["model", "physical_features"]].copy()
    for col in METRIC_COLUMNS:
        shown[col] = [
            f"{m:.3f} +/- {s:.3f}"
            for m, s in zip(summary[f"{col}_mean"], summary[f"{col}_std"])
        ]
    return shown


def main():
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, _, y_train, _ = stratified_split(df)   # the test part is discarded here
    print(f"Training rows: {len(X_train)}, failures: {int(y_train.sum())}")
    print("5-fold stratified cross-validation on the TRAINING set only.\n")

    parts = []
    for use_physical in (False, True):
        start = time.time()
        summary, _ = compare_models(X_train, y_train, use_physical=use_physical)
        parts.append(summary)
        label = "WITH" if use_physical else "WITHOUT"
        print(f"{label} physical features ({time.time() - start:.0f}s)")
        print(format_table(summary).drop(columns="physical_features").to_string(index=False))
        print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()