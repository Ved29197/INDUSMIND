"""Saving and loading the final model artifact.

Light on purpose: imports no training code, so the deployed app never needs XGBoost.
"""
import json

import joblib

from src.classification.failure_model import FailureModel  # noqa: F401  (needed to unpickle)
from src.data_processing.loader import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "failure_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "failure_model.json"


def save_failure_model(model, metadata: dict, model_path=None, metadata_path=None):
    """Write the model (compressed) and its metadata file."""
    model_path = model_path or MODEL_PATH
    metadata_path = metadata_path or METADATA_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metadata_path


def load_failure_model(model_path=None):
    """Load the saved model, with a clear message if it has not been built."""
    model_path = model_path or MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Build it with: python -m src.classification.build_artifact"
        )
    return joblib.load(model_path)


def load_metadata(metadata_path=None) -> dict:
    metadata_path = metadata_path or METADATA_PATH
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return json.loads(metadata_path.read_text())