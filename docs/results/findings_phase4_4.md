# Phase 4.4 findings: one-time test evaluation and robustness

Choices were frozen and committed before the test set was evaluated (see
frozen_choices.md; git history shows the order). Test set: 2,000 rows, 68 failures.
All results use illustrative cost units and a single split of synthetic data.

## Test results (Random Forest, physical features, isotonic calibration)
- PR-AUC 0.847 (95% CI 0.752 to 0.926) vs cross-validation 0.891. ROC-AUC 0.980.
- Thresholds from costs alone (no tuning). At cost ratio 10 (threshold 0.091): recall
  0.853 (CI 0.757 to 0.929), precision 0.906, 10 missed, 6 false alarms.
- Total cost is 84% to 86% below the better of "never inspect" and "inspect everything"
  at cost ratios 5, 10, 25 and 50.
- Between ratios 10 and 25, 5 more failures are caught at the price of 107 more false
  alarms (about 21 per catch), so break-even is a cost ratio of roughly 18 to 21.

## Challenger
- XGBoost: test PR-AUC 0.860 (CI 0.774 to 0.929). Intervals overlap with the Random
  Forest, and the order was reversed in cross-validation, so there is no real difference.
  The frozen rule applies: the primary model is not replaced because of test results.

## Frozen expectations: all four met
1. Test PR-AUC within two standard deviations of cross-validation (at the lower edge).
2. Savings 83.5% to 86.0% against the expected 82% to 86%.
3. Unseen heat failures mostly missed (recall 0.087 with physical features, 0.165 without).
4. ~5% risk band is mostly high-wear machines (63.7%; observed failure rate 5.3%).

## Robustness
- Time-blocked split (train on earliest 80%, test on latest 20%): 39 failures, no heat
  failures in that period. PR-AUC 0.777 (CI 0.656 to 0.883), recall 0.718, precision 0.683.
  Cause flags in that period: TWF 10, PWF 10, OSF 21, HDF 0, RNF 0. Random tool-wear
  failures make up a larger share than in the full data, which caps achievable recall at
  roughly 74% to 79%. The drop is mostly the mix of failure types, not a collapse of the model.
- Unseen heat failures: with the whole heat-dissipation episode removed from training, the
  model catches 8.7% to 16.5% of heat failures but about 68% to 70% of other failures in
  the same window.

## Limitations to state wherever results are shown
- Synthetic, partly rule-generated data; one split; 68 test failures, so intervals are wide.
- The model cannot predict failure modes absent from its training data. A mechanical
  engineer's physical limits should act as an override in the decision engine.
- Cost units are placeholders, not plant costs.