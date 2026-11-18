"""One-time test-set evaluation plus robustness checks.

Run from the project root:
    python -m src.classification.final_evaluation

Choices were frozen BEFORE this was run (docs/results/frozen_choices.md).
The test set is used here exactly once, for reporting. Nothing is re-tuned afterwards.
Cost units are illustrative (1 unit = one unneeded inspection).
"""
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.classification.final_model import fit_final_model
from src.classification.splits import blocked_split, stratified_split
from src.data_processing.loader import (
    PROJECT_ROOT, SAMPLE_PATH, get_features_and_target, load_ai4i,
)
from src.evaluation.bootstrap import bootstrap_ci
from src.evaluation.calibration import calibration_summary
from src.evaluation.thresholds import cost_curve, theoretical_threshold

RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
PRIMARY, CHALLENGER = "Random Forest", "XGBoost"
COST_RATIOS = [5, 10, 25, 50]
REFERENCE_RATIO = 10          # cost ratio used for the robustness checks
N_BOOT = 1000


def outcome_at(y, p, threshold, ratio):
    """Confusion counts and cost at one threshold."""
    return cost_curve(y, p, cost_fn=ratio, cost_fp=1, thresholds=[threshold]).iloc[0]


def recall_at(threshold):
    return lambda yt, pp: ((pp >= threshold) & (yt == 1)).sum() / yt.sum()


def hdf_episode_mask(df):
    """Rows between the first and last heat-dissipation failure (by UDI)."""
    hdf_udi = df.loc[df["HDF"] == 1, "UDI"]
    return df["UDI"].between(hdf_udi.min(), hdf_udi.max())


def evaluate_on_test(model, X_test, y_test, y_train):
    """Print and return test metrics for a fitted FailureModel."""
    raw = model.predict_raw(X_test)
    cal = model.predict_calibrated(X_test)
    cv_pr = average_precision_score(y_train, model.oof_raw)
    pr = average_precision_score(y_test, cal)
    pr_lo, pr_hi = bootstrap_ci(y_test, cal, average_precision_score, N_BOOT)
    roc = roc_auc_score(y_test, cal)
    roc_lo, roc_hi = bootstrap_ci(y_test, cal, roc_auc_score, N_BOOT)
    raw_cal, after_cal = calibration_summary(y_test, raw), calibration_summary(y_test, cal)

    print(f"\n{model.name} (physical features: {model.use_physical}), "
          f"{len(y_test)} test rows, {int(y_test.sum())} failures")
    print(f"  PR-AUC   cross-validation (train) {cv_pr:.3f} | test {pr:.3f}  95% CI [{pr_lo:.3f}, {pr_hi:.3f}]")
    print(f"  ROC-AUC  test {roc:.3f}  95% CI [{roc_lo:.3f}, {roc_hi:.3f}]")
    print(f"  Brier    raw {raw_cal['brier']:.4f} -> calibrated {after_cal['brier']:.4f} "
          f"(skill {raw_cal['brier_skill']:.3f} -> {after_cal['brier_skill']:.3f}); "
          f"ECE {raw_cal['ece']:.4f} -> {after_cal['ece']:.4f}")

    rows = []
    for ratio in COST_RATIOS:
        t = theoretical_threshold(ratio, 1)
        o = outcome_at(y_test, cal, t, ratio)
        never, inspect_all = int(y_test.sum()) * ratio, len(y_test) - int(y_test.sum())
        rec_lo, rec_hi = bootstrap_ci(y_test, cal, recall_at(t), N_BOOT)
        rows.append({
            "model": model.name, "cost_ratio": ratio, "threshold": round(t, 3),
            "recall": o["recall"], "recall_ci_low": rec_lo, "recall_ci_high": rec_hi,
            "precision": o["precision"], "missed": int(o["fn"]), "false_alarms": int(o["fp"]),
            "total_cost": o["total_cost"], "never_inspect_cost": never,
            "inspect_all_cost": inspect_all,
            "saving_vs_best_simple_policy": 1 - o["total_cost"] / min(never, inspect_all),
            "test_pr_auc": pr, "test_pr_auc_ci_low": pr_lo, "test_pr_auc_ci_high": pr_hi,
            "cv_pr_auc": cv_pr, "test_roc_auc": roc,
        })
    table = pd.DataFrame(rows)
    show = table[["cost_ratio", "threshold", "recall", "precision", "missed", "false_alarms",
                  "total_cost", "never_inspect_cost", "inspect_all_cost",
                  "saving_vs_best_simple_policy"]]
    print("  Thresholds from costs alone (no tuning on test data):")
    print(show.round(3).to_string(index=False))
    return table


def check_blocked(df):
    """Train on the earliest 80% of rows (by UDI), test on the latest 20%."""
    Xb_train, Xb_test, yb_train, yb_test = blocked_split(df)
    model = fit_final_model(Xb_train, yb_train, PRIMARY)
    cal = model.predict_calibrated(Xb_test)
    t = theoretical_threshold(REFERENCE_RATIO, 1)
    o = outcome_at(yb_test, cal, t, REFERENCE_RATIO)
    pr = average_precision_score(yb_test, cal)
    lo, hi = bootstrap_ci(yb_test, cal, average_precision_score, N_BOOT)
    causes = df.loc[yb_test.index, ["TWF", "HDF", "PWF", "OSF", "RNF"]].sum().to_dict()
    print(f"\nTIME-BLOCKED SPLIT ({PRIMARY}): {len(yb_test)} later rows, {int(yb_test.sum())} failures")
    print(f"  causes in the later period: {causes}")
    print(f"  PR-AUC {pr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  at threshold {t:.3f}: recall {o['recall']:.3f}, precision {o['precision']:.3f}, "
          f"missed {int(o['fn'])}, false alarms {int(o['fp'])}")
    return {"check": "time_blocked", "rows": len(yb_test), "failures": int(yb_test.sum()),
            "pr_auc": pr, "pr_auc_ci_low": lo, "pr_auc_ci_high": hi,
            "recall": o["recall"], "precision": o["precision"],
            "missed": int(o["fn"]), "false_alarms": int(o["fp"])}


def check_unseen_heat_failures(df):
    """Train with the whole heat-dissipation episode removed; test on that episode."""
    mask = hdf_episode_mask(df)
    X_train, y_train = get_features_and_target(df[~mask])
    X_hold, y_hold = get_features_and_target(df[mask])
    hdf_fail = (df.loc[y_hold.index, "HDF"] == 1) & (y_hold == 1)
    other_fail = (df.loc[y_hold.index, "HDF"] == 0) & (y_hold == 1)
    t = theoretical_threshold(REFERENCE_RATIO, 1)
    print(f"\nUNSEEN HEAT FAILURES: trained on {len(X_train)} rows with no HDF events; "
          f"held out {len(X_hold)} rows holding {int(hdf_fail.sum())} HDF failures "
          f"and {int(other_fail.sum())} other failures")
    rows = []
    for use_physical in (False, True):
        model = fit_final_model(X_train, y_train, PRIMARY, use_physical=use_physical)
        cal = model.predict_calibrated(X_hold)
        flagged = cal >= t
        rec_hdf = flagged[hdf_fail.values].mean()
        rec_other = flagged[other_fail.values].mean() if other_fail.any() else float("nan")
        false_alarms = int((flagged & (y_hold.values == 0)).sum())
        print(f"  physical features {str(use_physical):5s}: recall on HDF failures {rec_hdf:.3f}, "
              f"on other failures {rec_other:.3f}, false alarms {false_alarms}")
        rows.append({"check": f"unseen_hdf_physical_{use_physical}", "rows": len(X_hold),
                     "failures": int(y_hold.sum()), "recall_hdf": rec_hdf,
                     "recall_other": rec_other, "false_alarms": false_alarms})
    return rows


def check_risk_plateau(model, X_train, y_train, low=0.04, high=0.10):
    """What do machines in the ~5% calibrated-risk band have in common?"""
    cal = model.oof_calibrated
    band = (cal >= low) & (cal <= high)
    wear = X_train["Tool wear [min]"]
    high_wear = wear >= 198
    print(f"\nRISK PLATEAU ({low:.0%} to {high:.0%} calibrated risk, out-of-fold, training rows)")
    print(f"  rows in band: {int(band.sum())}, observed failure rate {y_train[band].mean():.3f}")
    print(f"  share of band with tool wear >= 198 min: {high_wear[band].mean():.3f}, "
          f"mean wear {wear[band].mean():.0f} min")
    print(f"  rows with tool wear >= 198 min: {int(high_wear.sum())}, "
          f"of which in band: {int((band & high_wear).sum())}; "
          f"their failure rate {y_train[high_wear].mean():.3f}")
    return {"check": "risk_plateau", "rows": int(band.sum()),
            "observed_rate": y_train[band].mean(),
            "share_high_wear": high_wear[band].mean()}


def main():
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, X_test, y_train, y_test = stratified_split(df)

    print("=" * 70)
    print("A. ONE-TIME TEST-SET EVALUATION (choices frozen before this run)")
    primary = fit_final_model(X_train, y_train, PRIMARY)
    tables = [evaluate_on_test(primary, X_test, y_test, y_train)]
    challenger = fit_final_model(X_train, y_train, CHALLENGER)
    tables.append(evaluate_on_test(challenger, X_test, y_test, y_train))

    print("\n" + "=" * 70)
    print("B. ROBUSTNESS CHECKS")
    robustness = [check_blocked(df)]
    robustness += check_unseen_heat_failures(df)
    robustness.append(check_risk_plateau(primary, X_train, y_train))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(RESULTS_DIR / "test_evaluation.csv", index=False)
    pd.DataFrame(robustness).to_csv(RESULTS_DIR / "robustness_checks.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()