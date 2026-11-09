# INDUSMIND
AI-powered industrial predictive intelligence and maintenance decision platform.

INDUSMIND/
├── app.py                  # entry point and home page
├── pages/                  # one file per Streamlit page (added in later phases)
├── src/                    # all reusable logic
│   ├── data_processing/    # loading and validation
│   ├── classification/     # failure models
│   ├── anomaly_detection/
│   ├── explainability/     # SHAP
│   ├── optimization/       # decision engine
│   ├── evaluation/         # metrics and threshold logic
│   └── reporting/          # PDF generation
├── notebooks/              # Colab experiments
├── data/{raw,processed,sample}/
├── models/                 # small saved artifacts only
├── tests/                  # automated tests
├── docs/                   # architecture, data dictionary, screenshots
├── requirements.txt
├── .gitignore
└── README.md