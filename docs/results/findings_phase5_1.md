# Phase 5.1 findings: Isolation Forest anomaly detection

Setup: fitted on 8,000 training machines without labels; Product Type excluded; the top 5%
most unusual machines flagged using a cutoff taken from the training scores; evaluated
against failure labels on 2,000 test machines (68 failures). No tuning. Expectations were
committed before the evaluation (anomaly_expectations.md).

## Results (test set)
- With physical features: PR-AUC 0.302 (95% CI 0.199 to 0.415) against 0.034 for random
  guessing; lift 8.6x; of 92 flagged machines, 27 failed (precision 0.29, recall 0.40).
- Without physical features: PR-AUC 0.146 (CI 0.088 to 0.232); lift 2.7x.
- Training and test results are similar (0.321 and 0.302), as expected for a method that
  never used the labels.

## What looks unusual
- Power-window failures 92% flagged (12 of 13), overstrain 69% (11 of 16).
- Heat-dissipation failures only 14% (4 of 29), tool-wear 20% (2 of 10). Heat failures are
  the largest group in the test set (29 of 68) and need a combination of a small temperature
  gap and low speed, each of which is normal on its own.
- Unusual machines that did not fail are mostly lightly loaded (median power 4,174 W,
  wear x torque 2,527); unusual machines that failed are mostly heavily loaded (8,425 W,
  9,730). Unusual is not the same as failed.

## Expectations: all four met
Test PR-AUC between 0.10 and 0.50; physical features help; power and overstrain flagged
more than heat and tool wear (RNF not assessable, none in the test set); precision below 0.5.

## Conclusions
1. Anomaly detection finds stressed operating conditions but misses failures whose readings
   look normal one at a time. It cannot replace the supervised failure model (the Random
   Forest flagged 64 machines and caught 58 failures; the detector flagged 92 and caught 27).
2. Engineered physical features matter more here than in supervised learning, because no
   labels help the method find combinations.
3. Its value is as a label-free screen for new equipment, unusual operating conditions and
   data problems. Engineering-limit checks remain the safeguard for modes that look normal.

## Limitations
Only 68 test failures (wide intervals); 2 failures with no cause and no RNF failures in the
test set; synthetic data.