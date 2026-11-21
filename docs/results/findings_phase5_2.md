# Phase 5.2 findings: operating regimes (K-means) and density outliers (DBSCAN)

Setup: fitted on 8,000 training machines without labels; 2,000 test machines (68 failures)
only assigned or scored. Regimes use six operating-condition features; absolute air and
process temperatures are excluded because air temperature drifts with the row order.
k chosen by the rule fixed in advance (best silhouette among k >= 3). Expectations were
committed before the evaluation (regime_expectations.md).

## Regimes
- Silhouette 0.270 at k=3 and 0.23 to 0.27 for every other k: weak structure, so the regimes
  are a convenient partition, not natural groups. Stability across random starts 0.990.
- Regime 0, heavy load and worn tool (median 44.8 Nm, 166 min wear): failure rate 6.0% (train)
  and 6.7% (test). It holds about 37% of test machines but 72% of test failures (49 of 68).
- Regime 1, fast with light torque: 1.6% and 1.7%. Regime 2, fresh tool and normal load:
  2.3% and 1.3% (the lowest-failure regime swaps between train and test; 10 vs 9 test
  failures, so this is noise).
- No regime is a time period (largest-decile share 0.106 to 0.114).
- Regimes are weak predictors on their own (lift 1.8x for the worst regime); they are context,
  filters and planning categories.

## Density-based outliers
- DBSCAN finds one dense cluster; noise is 1.4% of training and 1.0% of test machines.
- Test noise: 20 machines, 10 failed (precision 0.50, recall 0.15, lift 14.7x; small sample).
  Nine of the 10 are power-window failures.
- k-distance top 5% (88 machines, 22 failures): PR-AUC 0.312 (CI 0.207 to 0.422), lift 7.4x;
  overlap with Isolation Forest 0.59 (67 machines in common).

## Blind spot
- Heat-dissipation failures flagged: Isolation Forest 4 of 29, k-distance 4 of 29, DBSCAN
  noise 1 of 29. Heat failures need a combination of a small temperature gap and low speed,
  each normal alone. Together with the unseen-episode test in Phase 4, this shows that only
  supervised learning (and engineering-limit checks) cover this failure mode.

## Expectations
Met: 1 (silhouette), 3 (no time-period regime), 4 (DBSCAN noise), 5 (PR-AUC and overlap),
6 (heat failures rarely flagged). Partly met: 2 (strong regime differences; lowest regime
swaps on test within noise).

## Limitations
Small test counts per regime (49, 10, 9 failures) and for DBSCAN noise (20 machines);
synthetic data; one split.