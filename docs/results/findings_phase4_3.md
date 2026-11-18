# Phase 4.3 findings: calibration and cost-based thresholds

Setup: out-of-fold predictions on the 8,000 training rows (271 failures), physical
features on. Test set untouched. Calibrators were fitted on some rows and applied to
others. Cost units are illustrative (1 unit = one unneeded inspection).

## Calibration
1. Raw class-weighted models are poorly calibrated. Brier skill (0 = no better than
   always predicting 3.39%): Logistic Regression -2.34, Decision Tree -0.02, XGBoost
   0.61, Random Forest 0.71.
2. Calibration repairs this. Best by Brier: Random Forest + isotonic (Brier 0.0075,
   skill 0.77, ECE 0.0015). XGBoost sigmoid and isotonic are effectively tied.
3. Raw Random Forest overstated mid-range risk (predicted 27%, observed 4.1%) and
   understated high risk (predicted 86%, observed 97%).
4. Isotonic output is a step function: many machines share identical probabilities, and
   sparse bands (10 rows) are too noisy to interpret. Keep the raw score as a tie-breaker.

## Thresholds
5. Threshold = C_FP / (C_FP + C_FN) for calibrated probabilities. At cost ratio 10 that
   is about 9%; at 25 about 4%.
6. At cost ratios 5 to 50 the model's total cost is 82-86% below the cheaper of "never
   inspect" and "inspect everything". Optimistic: thresholds were chosen on the same
   out-of-fold data. The test set will show the honest figure.
7. Between ratio 10 and 25 the Random Forest catches 22 more failures at the price of 399
   more false alarms (about 18 per catch). Break-even cost ratio is therefore about 18:
   inspect the roughly 5%-risk group only when a failure costs more than about 18
   inspections. Hypothesis to verify: this group is the high-wear machines.
8. A "catch 90% of failures" target implies a high cost ratio (here 435 false alarms,
   precision 0.37). Recall targets hide cost assumptions; the decision engine makes them
   explicit.
9. At ratio 10 recall is 0.845, close to the 84.7% of failures that follow deterministic
   rules: the model recovers the rule-determined failures and nothing more until
   inspecting the 5% group pays off.

## Decision (proposed, to confirm in 4.4)
Primary: Random Forest + physical features + isotonic calibration. Challenger: XGBoost.
No single threshold is frozen; the threshold comes from user-editable costs in the
decision engine.