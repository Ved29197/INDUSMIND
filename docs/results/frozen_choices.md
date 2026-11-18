# Frozen choices (committed BEFORE the test set was evaluated)

## Choices
- Data: AI4I 2020, stratified 80/20 split with seed 42 (8,000 train / 2,000 test).
- Features: Type, five sensor columns, three physical features (temp_diff_K, power_W,
  wear_torque). Banned: UDI, Product ID, TWF, HDF, PWF, OSF, RNF.
- Primary model: Random Forest, settings as in src/classification/models.py
  (300 trees, class_weight balanced_subsample).
- Challenger: XGBoost, same file. Reported for transparency; it will NOT replace the
  primary model because of test results.
- Calibration: isotonic, fitted on out-of-fold training predictions.
- Thresholds: nothing is tuned. They come from cost ratios as C_FP / (C_FP + C_FN),
  with illustrative ratios 5, 10, 25 and 50.

## Rules
1. The test set is evaluated once, with the code in this commit.
2. No change to model, features, hyperparameters or calibration after seeing test
   results. If results disappoint, they are reported as they are.
3. Uncertainty is reported with bootstrap 95% confidence intervals (1,000 resamples).

## Expectations stated in advance
- Test PR-AUC lands within about two standard deviations of cross-validation
  (Random Forest cross-validated PR-AUC 0.892 +/- 0.027, so roughly 0.84 to 0.95).
  A value below 0.80 triggers an investigation of data differences, not retuning.
- Savings versus the best simple policy are similar to the cross-validation estimate
  (82% to 86%) but noisier, because the test set holds only about 68 failures.
- Unseen-heat-failure test: low recall on heat failures, because a model cannot learn a
  failure mode it never saw.
- The ~5% risk plateau consists mostly of high-wear machines.