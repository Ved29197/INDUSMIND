# Isolation Forest: expectations written before running the evaluation

Setup: fit on the 8,000 training machines without labels; flag the top 5% most unusual by
the training-score cutoff; evaluate against failure labels on the 2,000 test machines.
No tuning. Cause flags are used only for after-the-fact analysis.

## Expectations
1. Test PR-AUC is clearly above the base failure rate (0.034, so above about 0.10) but well
   below the supervised model's 0.847 and below 0.50. Unusual is not the same as failed.
2. Adding physical features improves the PR-AUC.
3. Power-window (PWF) and overstrain (OSF) failures are flagged as unusual more often than
   heat (HDF), tool-wear (TWF) or random (RNF) failures. Power and wear x torque have
   extreme tails; the heat failure needs a specific combination, and tool-wear and random
   failures are mostly stochastic.
4. Most flagged machines are not failures (precision among flagged machines below 0.5).