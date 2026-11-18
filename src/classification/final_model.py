"""The final failure model: pipeline + calibrator, built from TRAINING data only."""
from sklearn.base import clone

from src.classification.cross_validation import cross_validate_model
from src.classification.failure_model import FailureModel
from src.classification.models import NEEDS_SCALING, get_models
from src.classification.pipelines import build_model_pipeline
from src.evaluation.calibration import Calibrator, calibrate_oof


def fit_final_model(X_train, y_train, name: str = "Random Forest", use_physical: bool = True,
                    n_estimators: int = 300, method: str = "isotonic") -> FailureModel:
    """Fit the model on all training rows and learn its calibrator.

    1. Cross-validate to get out-of-fold raw scores for every training row.
    2. Fit the calibrator on those out-of-fold scores (they come from models that
       never saw the row they scored).
    3. Fit the final pipeline on all training rows.
    """
    model = get_models(y_train, n_estimators=n_estimators)[name]
    _, oof_raw = cross_validate_model(name, model, X_train, y_train, use_physical)
    oof_calibrated = calibrate_oof(oof_raw, y_train, method)
    calibrator = Calibrator(method).fit(oof_raw, y_train)
    pipeline = build_model_pipeline(
        clone(model), use_physical=use_physical, scale=name in NEEDS_SCALING
    ).fit(X_train, y_train)
    return FailureModel(name, use_physical, pipeline, calibrator, oof_raw, oof_calibrated)