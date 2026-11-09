"""Baseline: a 'model' that always predicts the training failure rate.

Run from the project root:
    python -m src.classification.baseline
Every real model must beat these numbers to be worth using.
"""
from sklearn.dummy import DummyClassifier

from src.classification.splits import blocked_split, stratified_split
from src.data_processing.loader import SAMPLE_PATH, load_ai4i
from src.evaluation.metrics import evaluate_binary


def main():
    df, _ = load_ai4i(SAMPLE_PATH)

    X_train, X_test, y_train, y_test = stratified_split(df)
    print("STRATIFIED SPLIT (primary)")
    print(f"  train rows {len(X_train)}, failures {int(y_train.sum())} ({y_train.mean():.2%})")
    print(f"  test  rows {len(X_test)}, failures {int(y_test.sum())} ({y_test.mean():.2%})")

    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    metrics = evaluate_binary(y_test, baseline.predict_proba(X_test)[:, 1])
    print("\nBASELINE (always predicts the training failure rate), test set")
    for name, value in metrics.items():
        print(f"  {name:22s} {value:.4f}" if isinstance(value, float) else f"  {name:22s} {value}")

    Xb_train, Xb_test, yb_train, yb_test = blocked_split(df)
    print("\nTIME-BLOCKED SPLIT (robustness check)")
    print(f"  train rows {len(Xb_train)}, failures {int(yb_train.sum())}, "
          f"HDF events {int(df.loc[yb_train.index, 'HDF'].sum())}")
    print(f"  test  rows {len(Xb_test)}, failures {int(yb_test.sum())}, "
          f"HDF events {int(df.loc[yb_test.index, 'HDF'].sum())}")


if __name__ == "__main__":
    main()