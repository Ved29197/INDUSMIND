# INDUSMIND
AI-powered industrial predictive intelligence and maintenance decision platform.

# INDUSMIND

**AI-powered industrial predictive intelligence and maintenance decision platform.**

Live app: <https://indusmind-9kf5dbpiojzkab98mj39oi.streamlit.app/>

> **Important:** this project runs on the public, synthetic UCI AI4I 2020 dataset. It
> demonstrates methods and decision logic. It does **not** represent real plant data and
> makes no claim of validated real-world savings.

## What it will do
Predict equipment failure probability, flag abnormal behaviour, explain risk drivers,
and turn probabilities into maintenance decisions (preventive vs. reactive, ranked by
cost and limited by crew and budget).

## Status
| Phase | Scope | Status |
|---|---|---|
| 2 | Repo, Codespaces, deployed Streamlit skeleton | Done |
| 3 | Data download, validation, EDA, Dataset Explorer page | Done |
| 4 | Failure prediction and validation framework | Planned |
| 5 | Anomaly detection and SHAP explainability | Planned |
| 6 | Maintenance decision engine | Planned |
| 7 | Dashboard integration and demo mode | Planned |
| 8 | PDF report, final tests and documentation | Planned |

## Run it yourself
```bash
git clone https://github.com/Ved29197/INDUSMIND.git
cd INDUSMIND
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
streamlit run app.py
```
The demo dataset is included in `data/sample/`. To re-download the original file:
`python -m src.data_processing.download`

## Data
- **Source:** UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset.
  Licence: CC BY 4.0. Citation: AI4I 2020 Predictive Maintenance Dataset [Dataset].
  (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- **Details, column roles, leakage policy and known issues:** see
  [docs/data_dictionary.md](docs/data_dictionary.md).

## Repository layout
`pages/` Streamlit pages, `src/` tested logic, `notebooks/` experiments, `tests/`
automated tests, `data/sample/` demo data, `docs/` documentation.

## Security notes
No API keys or secrets are used. Uploaded CSVs are size- and row-limited, validated, and
held in memory only for the session.