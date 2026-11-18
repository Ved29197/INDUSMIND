# Phase 4.2 findings: model comparison

Setup: 8,000 training rows, 271 failures. 5-fold stratified cross-validation on the
training set only; test set untouched. Class weights for imbalance; default settings,
no tuning.

## PR-AUC (mean +/- std)
| Model | Without physical features | With physical features |
|---|---|---|
| Baseline (always the failure rate) | 0.034 | 0.034 |
| Logistic Regression | 0.441 +/- 0.033 | 0.459 +/- 0.047 |
| Decision Tree | 0.619 +/- 0.054 | 0.850 +/- 0.037 |
| Random Forest | 0.731 +/- 0.030 | 0.892 +/- 0.027 |
| XGBoost | 0.743 +/- 0.051 | 0.855 +/- 0.042 |

## Findings
1. Every real model beats the baseline by a wide margin.
2. ROC-AUC flatters weak models (Logistic Regression: ROC-AUC 0.92, PR-AUC 0.46), so
   PR-AUC is the headline metric.
3. Trees clearly beat Logistic Regression. Failures depend on threshold windows and
   combinations of conditions, which a single straight boundary cannot express.
4. Physical features help trees a lot (Decision Tree +0.23, Random Forest +0.16,
   XGBoost +0.11) and Logistic Regression barely (+0.02, within noise). Trees split on
   one feature at a time, so products and differences such as torque x speed and
   process - air temperature are hard for them to build unaided. Caveat: these
   features encode the documented generating rules.
5. With physical features, Random Forest, XGBoost and the Decision Tree are not clearly
   separated (gaps are about the size of the fold-to-fold spread). No winner is
   declared by decimals.
6. Precision and recall at the default 0.5 threshold are provisional: class weights
   shift each model's probability scale, so differences there mostly reflect scale,
   not skill.
7. About 85% of failures follow deterministic rules. Random Forest recall of 0.83 at
   precision 0.92 is close to that ceiling, so a high score on this dataset mostly means
   the model rediscovered the generator's rules, not real-world predictive skill. The
   remaining failures can only be caught at the cost of many false alarms (Decision
   Tree: recall 0.92 but precision 0.40).

## Correction to an earlier expectation
I expected physical features to help Logistic Regression most. They helped it least:
power failure is a two-sided window and heat failure needs two conditions together,
neither of which a linear model can represent.

## Decision
Carry Random Forest and XGBoost (with physical features) forward to threshold
selection and calibration.