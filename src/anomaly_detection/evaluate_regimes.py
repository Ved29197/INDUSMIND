"""K-means operating regimes and DBSCAN outliers, evaluated against failure labels.

Run from the project root:
    python -m src.anomaly_detection.evaluate_regimes

Expectations were written first: docs/results/regime_expectations.md.
Everything is fitted on the training machines WITHOUT labels; test machines are only assigned.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection.density import (
    choose_eps, dbscan_noise_flags, fit_dbscan, knn_distance_scores,
)
from src.anomaly_detection.isolation import (
    anomaly_scores, build_isolation_pipeline, cutoff_for_fraction, lift_at_cutoff,
)
from src.anomaly_detection.regimes import (
    REGIME_FEATURES, choose_k, cluster_stability, fit_kmeans, regime_matrix, time_concentration,
)
from src.classification.splits import stratified_split
from src.data_processing import schema
from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH, load_ai4i
from src.evaluation.bootstrap import bootstrap_ci
from src.feature_engineering.physical import PHYSICAL_FEATURES, add_physical_features

RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
K_CANDIDATES = range(2, 9)
K_FLOOR = 3                  # rule fixed in advance: best silhouette among k >= 3
DBSCAN_FEATURES = list(schema.NUMERIC_FEATURES) + PHYSICAL_FEATURES   # same 8 as Isolation Forest
EPS_PERCENTILE = 95.0
MIN_SAMPLES = 2 * len(DBSCAN_FEATURES)
CAUSES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def regime_section(df, X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(regime_matrix(X_train))
    Z_train = scaler.transform(regime_matrix(X_train))
    Z_test = scaler.transform(regime_matrix(X_test))

    selection = choose_k(Z_train, K_CANDIDATES)
    best_k = int(selection[selection["k"] >= K_FLOOR].sort_values("silhouette").iloc[-1]["k"])
    print("1) CHOOSING THE NUMBER OF REGIMES (training machines)")
    print(selection.round(3).to_string(index=False))
    print(f"   chosen k = {best_k} (best silhouette among k >= {K_FLOOR}); "
          f"stability across random starts (1.0 = identical) = {cluster_stability(Z_train, best_k):.3f}")

    km = fit_kmeans(Z_train, best_k)
    labels_train, labels_test = km.labels_, km.predict(Z_test)
    base = y_train.mean()

    phys_train = regime_matrix(X_train)
    profile = phys_train.groupby(labels_train).median()
    profile.columns = ["rpm", "torque_Nm", "wear_min", "power_W", "wear_x_torque", "gap_K"]
    profile.insert(0, "train_rows", pd.Series(labels_train).value_counts().sort_index().values)
    profile["train_fail_rate"] = pd.Series(y_train.values).groupby(labels_train).mean().values
    test_counts = pd.DataFrame({"c": labels_test, "y": y_test.values}).groupby("c")["y"].agg(["size", "sum", "mean"])
    profile["test_rows"] = test_counts["size"].reindex(profile.index).fillna(0).astype(int).values
    profile["test_failures"] = test_counts["sum"].reindex(profile.index).fillna(0).astype(int).values
    profile["test_fail_rate"] = test_counts["mean"].reindex(profile.index).values
    profile["lift_train"] = profile["train_fail_rate"] / base
    profile.index.name = "regime"
    print(f"\n2) OPERATING REGIMES (cluster medians in original units; base failure rate {base:.2%})")
    print(profile.round(3).to_string())

    timing = time_concentration(df.loc[X_train.index, "UDI"], labels_train)
    print("\n3) ARE THE REGIMES JUST TIME PERIODS? (max_decile_share: 0.10 = spread evenly)")
    print(timing.round(3).to_string(index=False))
    return profile, timing


def density_section(df, X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(add_physical_features(X_train)[DBSCAN_FEATURES])
    Z_train = scaler.transform(add_physical_features(X_train)[DBSCAN_FEATURES])
    Z_test = scaler.transform(add_physical_features(X_test)[DBSCAN_FEATURES])

    k = MIN_SAMPLES
    s_train = knn_distance_scores(Z_train, Z_train, k, query_is_reference=True)
    s_test = knn_distance_scores(Z_train, Z_test, k)
    eps = choose_eps(s_train, EPS_PERCENTILE)
    model = fit_dbscan(Z_train, eps, MIN_SAMPLES)
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    noise_train = model.labels_ == -1
    noise_test = dbscan_noise_flags(model, Z_test, eps)
    print(f"\n4) DBSCAN (eps = {eps:.3f} = {EPS_PERCENTILE:.0f}th percentile of training k-distances; "
          f"min_samples = {MIN_SAMPLES}; features = same 8 as Isolation Forest)")
    print(f"   clusters found: {n_clusters}; noise points: train {noise_train.mean():.1%} "
          f"({int(noise_train.sum())}), test {noise_test.mean():.1%} ({int(noise_test.sum())})")
    print("   (points just outside the dense core but near it are border points, not noise, "
          "so DBSCAN noise is much rarer than the top 5% of k-distances)")
    noise_stats = lift_at_cutoff(y_test, noise_test.astype(float), 1.0)
    print(f"   DBSCAN noise on test: flagged {noise_stats['flagged']}, failures among them "
          f"{noise_stats['failures_flagged']}")

    # Like-for-like with Isolation Forest: flag the top 5% by k-distance (cutoff from training)
    cutoff = cutoff_for_fraction(s_train, 0.05)
    knn_flag = s_test >= cutoff
    stats = lift_at_cutoff(y_test, s_test, cutoff)
    pr = average_precision_score(y_test, s_test)
    lo, hi = bootstrap_ci(y_test, s_test, average_precision_score, 1000)
    print(f"\n5) k-DISTANCE SCORE, top 5% flagged (test): flagged {stats['flagged']}, failures among "
          f"them {stats['failures_flagged']}, precision {stats['precision']:.3f}, "
          f"recall {stats['recall']:.3f}, lift {stats['lift']:.1f}")
    print(f"   test PR-AUC {pr:.3f}  95% CI [{lo:.3f}, {hi:.3f}] (random = {y_test.mean():.3f})")

    iso = build_isolation_pipeline(use_physical=True).fit(X_train)
    iso_cut = cutoff_for_fraction(anomaly_scores(iso, X_train), 0.05)
    iso_flag = anomaly_scores(iso, X_test) >= iso_cut
    both = int((iso_flag & knn_flag).sum())
    union = int((iso_flag | knn_flag).sum())
    print(f"\n6) OVERLAP WITH ISOLATION FOREST (test, both flag the top 5%): IF flagged "
          f"{int(iso_flag.sum())}, k-distance flagged {int(knn_flag.sum())}, both {both} "
          f"(Jaccard overlap {both / union if union else float('nan'):.2f})")

    causes = df.loc[X_test.index, CAUSES]
    failed = y_test.values == 1
    rows = []
    for cause in CAUSES:
        mask = failed & (causes[cause].values == 1)
        rows.append({"group": cause, "failures": int(mask.sum()),
                     "isolation_forest": int((mask & iso_flag).sum()),
                     "knn_distance_top5": int((mask & knn_flag).sum()),
                     "dbscan_noise": int((mask & noise_test).sum())})
    cause_table = pd.DataFrame(rows)
    print("\n7) WHICH FAILURES DO THE DETECTORS FLAG? (test set)")
    print(cause_table.to_string(index=False))
    return {"eps": eps, "min_samples": MIN_SAMPLES, "clusters": n_clusters,
            "dbscan_noise_train": noise_train.mean(), "dbscan_noise_test": noise_test.mean(),
            "dbscan_noise_flagged_test": noise_stats["flagged"],
            "dbscan_noise_failures_test": noise_stats["failures_flagged"],
            "knn_flagged": stats["flagged"], "knn_failures_flagged": stats["failures_flagged"],
            "knn_precision": stats["precision"], "knn_recall": stats["recall"], "knn_lift": stats["lift"],
            "knn_pr_auc": pr, "knn_pr_auc_ci_low": lo, "knn_pr_auc_ci_high": hi,
            "overlap_with_isolation_forest": both}, cause_table


def main():
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, X_test, y_train, y_test = stratified_split(df)
    print(f"Train {len(X_train)} rows ({int(y_train.sum())} failures), test {len(X_test)} rows "
          f"({int(y_test.sum())} failures).\n")
    profile, timing = regime_section(df, X_train, X_test, y_train, y_test)
    summary, cause_table = density_section(df, X_train, X_test, y_train, y_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(RESULTS_DIR / "regime_profile.csv")
    timing.to_csv(RESULTS_DIR / "regime_time_concentration.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "dbscan_eval.csv", index=False)
    cause_table.to_csv(RESULTS_DIR / "detectors_by_cause.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
    