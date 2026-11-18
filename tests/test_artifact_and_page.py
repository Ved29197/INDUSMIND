import subprocess
import sys

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.classification import artifact
from src.classification.artifact import (
    MODEL_PATH, load_failure_model, load_metadata, save_failure_model,
)
from src.classification.final_model import fit_final_model
from src.classification.splits import stratified_split
from src.data_processing.loader import PROJECT_ROOT, SAMPLE_PATH, load_ai4i

PAGE = str(PROJECT_ROOT / "pages" / "04_Failure_Prediction.py")


@pytest.fixture(scope="module")
def tiny_artifact(tmp_path_factory):
    """A small real model saved to a temporary folder."""
    if not SAMPLE_PATH.exists():
        pytest.skip("demo sample not present")
    df, _ = load_ai4i(SAMPLE_PATH)
    X_train, X_test, y_train, _ = stratified_split(df)
    model = fit_final_model(X_train, y_train, "Random Forest", n_estimators=15)
    folder = tmp_path_factory.mktemp("artifact")
    paths = save_failure_model(
        model, {"model": "Random Forest", "base_failure_rate": 0.034},
        folder / "m.joblib", folder / "m.json",
    )
    return model, X_test.head(40), paths


def test_saved_model_reproduces_predictions(tiny_artifact):
    model, X, (model_path, metadata_path) = tiny_artifact
    restored = load_failure_model(model_path)
    assert np.allclose(model.predict_calibrated(X), restored.predict_calibrated(X))
    assert load_metadata(metadata_path)["model"] == "Random Forest"


def test_saved_model_loads_without_xgboost(tiny_artifact):
    """The deployed app must not depend on training-only libraries."""
    _, X, (model_path, _) = tiny_artifact
    code = (
        "import sys; sys.modules['xgboost'] = None\n"
        "from pathlib import Path\n"
        "from src.classification.artifact import load_failure_model\n"
        f"m = load_failure_model(Path(r'{model_path}'))\n"
        "print('loaded')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=PROJECT_ROOT)
    assert "loaded" in result.stdout, result.stderr


def test_missing_model_gives_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_artifact"):
        load_failure_model(tmp_path / "nope.joblib")


needs_artifact = pytest.mark.skipif(
    not (MODEL_PATH.exists() and SAMPLE_PATH.exists()), reason="model artifact not built"
)


@needs_artifact
def test_page_opens_and_shows_risk_without_errors():
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Estimated failure risk" in labels and "Risk category" in labels
    assert any("not a prediction" in w.value for w in at.warning)


@needs_artifact
def test_page_risk_responds_to_inputs():
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_file(PAGE, default_timeout=120).run()
    first = next(m.value for m in at.metric if m.label == "Estimated failure risk")
    for widget in at.number_input:     # heavy load and a worn tool
        if widget.label == "Tool wear [min]":
            widget.set_value(250)
        if widget.label == "Torque [Nm]":
            widget.set_value(60.0)
    at.run()
    assert not at.exception
    second = next(m.value for m in at.metric if m.label == "Estimated failure risk")
    assert first != second


def test_page_shows_error_not_crash_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(artifact, "MODEL_PATH", tmp_path / "missing.joblib")
    st.cache_resource.clear()
    st.cache_data.clear()
    if not SAMPLE_PATH.exists():
        pytest.skip("demo sample not present")
    at = AppTest.from_file(PAGE, default_timeout=60).run()
    assert not at.exception
    assert any("build_artifact" in e.value for e in at.error)