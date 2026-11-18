"""Build the deployable model artifact from TRAINING data and the frozen configuration.

Run from the project root:
    python -m src.classification.build_artifact

Writes models/failure_model.joblib and models/failure_model.json.
This is not a new evaluation: the test-set score is recomputed once only to confirm that
the saved model reproduces the numbers recorded in docs/results/test_evaluation.csv.
"""
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score

from src.classification.artifact import save_failure_model
from src.classification.final_model import fit_final_model
from src.classification.splits import RANDOM_STATE, stratified_split
from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH, load_ai4i

MODEL_NAME = "Random Forest"      # frozen choice (docs/results/frozen_choices.md)
USE_PHYSICAL = True
CALIBRATION = "isotonic"
EVALUATION_CSV = PROJECT_ROOT / "docs" / "results" / "test_evaluation.csv"


def recorded_test_results():
    """Test metrics recorded in Milestone 4.4 for the frozen model."""
    if not EVALUATION_CSV.exists():
        return {}
    table = pd.read_csv(EVALUATION_CSV)
    row = table[table["model"] == MODEL_NAME].iloc[0]
    return {
        "test_pr_auc": round(float(row["test_pr_auc"]), 3),
        "test_pr_auc_ci_low": round(float(row["test_pr_auc_ci_low"]), 3),
        "test_pr_auc_ci_high": round(float(row["test_pr_auc_ci_high"]), 3),
        "test_roc_auc": round(float(row["test_roc_auc"]), 3),
        "cv_pr_auc": round(float(row["cv_pr_auc"]), 3),
    }


def main():
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, X_test, y_train, y_test = stratified_split(df)
    model = fit_final_model(X_train, y_train, MODEL_NAME, USE_PHYSICAL, method=CALIBRATION)

    recorded = recorded_test_results()
    rebuilt = float(average_precision_score(y_test, model.predict_calibrated(X_test)))
    if recorded:
        gap = abs(rebuilt - recorded["test_pr_auc"])
        status = "MATCHES" if gap < 0.005 else "DIFFERS FROM"
        print(f"Consistency check: rebuilt test PR-AUC {rebuilt:.3f} {status} "
              f"recorded {recorded['test_pr_auc']:.3f}")

    metadata = {
        "model": MODEL_NAME,
        "calibration": CALIBRATION,
        "physical_features": USE_PHYSICAL,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": int(len(X_train)),
        "training_failures": int(y_train.sum()),
        "base_failure_rate": round(float(y_train.mean()), 4),
        "random_seed": RANDOM_STATE,
        "recorded_test_results": recorded,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "data": "UCI AI4I 2020 (synthetic); see docs/data_dictionary.md",
        "limitations": [
            "Synthetic, partly rule-generated data; one split; about 68 test failures.",
            "Cannot predict failure modes absent from its training data.",
            "Output is an estimated risk, not a guaranteed failure.",
        ],
    }
    model_path, metadata_path = save_failure_model(model, metadata)
    size_mb = model_path.stat().st_size / 1e6
    print(f"Saved {model_path.name}: {size_mb:.1f} MB; metadata: {metadata_path.name}")
    if size_mb > 25:
        print("WARNING: larger than 25 MB. Do not commit it; tell your mentor.")


if __name__ == "__main__":
    main()