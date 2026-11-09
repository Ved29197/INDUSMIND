# INDUSMIND: Project Status

## Completed
- Phase 2: public repo, Codespaces workflow, Streamlit app deployed.
- Phase 3: reproducible AI4I download, schema + validator + leakage guard (tested),
  EDA notebook, Dataset Explorer page (demo mode and safe CSV upload), data dictionary.

## Verified facts to carry forward
- 10,000 rows; 339 failures (3.39%); accuracy baseline 96.61%.
- Banned features: UDI, Product ID, TWF, HDF, PWF, OSF, RNF.
- Hidden time order in UDI; HDF episode at UDI 4001 to 5000.
- 84.7% of failures follow deterministic rules; TWF and RNF are stochastic.

## Phase 4 decisions already made
- Primary evaluation: 80/20 stratified split; 5-fold stratified CV on the training set.
- Robustness check: time-blocked split, reported alongside.
- Threshold chosen from cost trade-offs, not a default 0.5.
- Models compared with and without engineered physical features.

## Open items
- Key findings text cell in notebook 01 to be rewritten in own words.