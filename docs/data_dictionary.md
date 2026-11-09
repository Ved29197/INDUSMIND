# INDUSMIND Data Dictionary: AI4I 2020 Predictive Maintenance Dataset

## Provenance
- **Source:** UCI Machine Learning Repository, dataset 601, donated 2020.
- **Author:** S. Matzka. Introductory paper: "Explainable Artificial Intelligence for
  Predictive Maintenance Applications", 2020 Third International Conference on
  Artificial Intelligence for Industries.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Citation:** AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
  Learning Repository. https://doi.org/10.24432/C5HS5C
- **Nature:** synthetic. It imitates industrial predictive-maintenance data and is NOT
  real plant data.
- **Committed copy:** `data/sample/ai4i2020.csv` (unchanged from the UCI file).

## Columns

| Column | Role | Type | Unit | Description | Check in code |
|---|---|---|---|---|---|
| UDI | Identifier | int | - | Row number 1 to 10,000. Rows follow a hidden time order. Never a feature. | Must be unique (error) |
| Product ID | Identifier | text | - | Quality letter plus serial number. Never a feature. | Required |
| Type | Feature | category | - | Product quality variant L, M or H. | Must be L, M or H (error) |
| Air temperature [K] | Feature | float | K | Documented as a random walk around 300 K with a 2 K spread. | Range 250 to 350 (warning) |
| Process temperature [K] | Feature | float | K | Documented as air temperature + 10 K + its own random walk (1 K spread). | Range 250 to 400; should exceed air temperature (warnings) |
| Rotational speed [rpm] | Feature | int | rpm | Documented as derived from a fixed 2,860 W power with noise. | Range 0 to 10,000 (warning) |
| Torque [Nm] | Feature | float | Nm | Documented as normal around 40 Nm, spread 10 Nm. | Range 0 to 200 (warning) |
| Tool wear [min] | Feature | int | min | Accumulated wear. Quality variants H, M, L add 5, 3, 2 minutes per documentation. | Range 0 to 500 (warning) |
| Machine failure | Target | 0/1 | - | 1 if any of the five failure modes occurred. | Only 0 and 1 (error) |
| TWF | Cause flag | 0/1 | - | Tool wear failure. | Only 0 and 1 (error) |
| HDF | Cause flag | 0/1 | - | Heat dissipation failure: temperature gap below 8.6 K and speed below 1,380 rpm. | Only 0 and 1 (error) |
| PWF | Cause flag | 0/1 | - | Power failure: power (torque x angular speed) below 3,500 W or above 9,000 W. | Only 0 and 1 (error) |
| OSF | Cause flag | 0/1 | - | Overstrain: tool wear x torque above 11,000 (L), 12,000 (M) or 13,000 (H) min-Nm. | Only 0 and 1 (error) |
| RNF | Cause flag | 0/1 | - | Random failure, documented as a 0.1% chance per process. | Only 0 and 1 (error) |

## Feature policy (leakage control)
- **Allowed features:** Type and the five sensor/process columns.
- **Banned:** UDI and Product ID (identifiers; UDI also encodes a hidden time order),
  and TWF, HDF, PWF, OSF, RNF (known only after a failure; they reproduce the label with
  94.8% precision and 97.3% recall).
- Derived features (temperature gap, mechanical power, wear x torque) use sensor columns
  only and are allowed. They encode the documented generating rules, so model
  performance with and without them is reported separately.

## Observed facts (from our own EDA, notebooks/01_AI4I_EDA.ipynb)
- 10,000 rows, 14 columns, no missing values.
- 339 failures (3.39%). A model that never predicts failure scores 96.61% accuracy.
- Cause flags raised: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19.
- Type counts: L 6,000, M 2,997, H 1,003.

## Known data issues
1. **Label inconsistency:** 9 failures have no cause flagged; 18 rows have RNF flagged
   but Machine failure = 0 (RNF is flagged 19 times, 1 with a recorded failure).
2. **Documentation mismatch:** the UCI page describes a 50/30/20 product mix; the data is
   about 60/30/10.
3. **Hidden time order:** air temperature has lag-1 autocorrelation 0.999 along UDI.
   Heat-dissipation failures cluster in UDI 4001 to 5000 (108 of the 115 HDF events fall
   in the 5th decile of UDI).
4. **Mostly rule-generated failures:** HDF, PWF and OSF follow their threshold rules
   exactly (every flagged row meets the rule, no unflagged row does). 287 of 339 failures
   (84.7%) are explained this way. Tool-wear failures are stochastic: all 46 occur at
   wear of 198 minutes or more, but only 46 of 894 such rows fail this way.
5. **Synthetic data:** results demonstrate methods, not real-world performance.