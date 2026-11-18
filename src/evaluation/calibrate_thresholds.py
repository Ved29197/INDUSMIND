"""Calibration and cost-based thresholds on the TRAINING set (out-of-fold predictions).

Run from the project root:
    python -m src.classification.calibrate_thresholds
The test set is never used here.

Cost units are ILLUSTRATIVE placeholders (one unnecessary inspection = 1 unit).
Real cost assumptions are entered in the maintenance decision engine later.
"""
import pandas as pd

from src.classification.cross_validation import cross_validate_model
from src.classification.models import get_models
from src.classification.splits import stratified_split
from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH, load_ai4i
from src.evaluation.calibration import calibrate_oof, calibration_summary, reliability_table
from src.evaluation.thresholds import (
    best_cost_threshold, cost_curve, theoretical_threshold, threshold_for_recall,
)

RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
COST_RATIOS = [5, 10, 25, 50]      # cost of a missed failure / cost of an unneeded inspection
TARGET_RECALL = 0.90


def main():
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, _, y_train, _ = stratified_split(df)   # test part discarded
    print(f"Training rows: {len(X_train)}, failures: {int(y_train.sum())}\n")

    models = get_models(y_train)
    variants, summary_rows = {}, []
    for name in MODELS:
        _, raw = cross_validate_model(name, models[name], X_train, y_train, use_physical=True)
        candidates = {
            "raw": raw,
            "sigmoid": calibrate_oof(raw, y_train, "sigmoid"),
            "isotonic": calibrate_oof(raw, y_train, "isotonic"),
        }
        briers = {}
        for variant, probs in candidates.items():
            stats = calibration_summary(y_train, probs)
            summary_rows.append({"model": name, "variant": variant, **stats})
            briers[variant] = stats["brier"]
        best = min(briers, key=briers.get)
        variants[name] = (raw, candidates[best], best)

    summary = pd.DataFrame(summary_rows)
    print("1) CALIBRATION (out-of-fold; lower Brier/ECE/log loss is better, higher skill is better)")
    print(summary.round(4).to_string(index=False))
    print("\nBest variant by Brier score:",
          {name: variants[name][2] for name in MODELS})

    print("\n2) RELIABILITY: Random Forest, raw vs best calibrated")
    raw_rf, cal_rf, best_rf = variants["Random Forest"]
    print("raw:\n", reliability_table(y_train, raw_rf).round(3).to_string(index=False))
    print(f"\n{best_rf}:\n", reliability_table(y_train, cal_rf).round(3).to_string(index=False))

    print("\n3) COST-BASED THRESHOLDS (illustrative cost units; 1 unit = one unneeded inspection)")
    rows = []
    positives = int(y_train.sum())
    negatives = len(y_train) - positives
    for name in ["Random Forest", "XGBoost"]:
        _, probs, variant = variants[name]
        for ratio in COST_RATIOS:
            curve = cost_curve(y_train, probs, cost_fn=ratio, cost_fp=1)
            best = best_cost_threshold(curve)
            rows.append({
                "model": name, "calibration": variant, "cost_ratio": ratio,
                "theoretical_threshold": round(theoretical_threshold(ratio, 1), 3),
                "best_threshold": best["threshold"], "recall": best["recall"],
                "precision": best["precision"], "missed": int(best["fn"]),
                "false_alarms": int(best["fp"]), "total_cost": best["total_cost"],
                "never_inspect_cost": positives * ratio, "inspect_all_cost": negatives,
            })
    thresholds = pd.DataFrame(rows)
    print(thresholds.round(3).to_string(index=False))

    print(f"\n4) ALTERNATIVE POLICY: highest threshold with recall >= {TARGET_RECALL:.0%}")
    for name in ["Random Forest", "XGBoost"]:
        _, probs, variant = variants[name]
        row = threshold_for_recall(y_train, probs, TARGET_RECALL)
        print(f"  {name} ({variant}): threshold {row['threshold']:.2f}, "
              f"recall {row['recall']:.3f}, precision {row['precision']:.3f}, "
              f"false alarms {int(row['fp'])}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "calibration_summary.csv", index=False)
    thresholds.to_csv(RESULTS_DIR / "threshold_analysis.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()