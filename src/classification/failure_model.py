"""The FailureModel container.

Kept in its own light module (no XGBoost, no training code) so the deployed app
can load a saved model without installing any training-only dependencies.
"""
from dataclasses import dataclass

import pandas as pd

from src.evaluation.calibration import Calibrator


@dataclass
class FailureModel:
    """A fitted pipeline plus the calibrator that turns its scores into probabilities."""

    name: str
    use_physical: bool
    pipeline: object
    calibrator: Calibrator
    oof_raw: pd.Series          # out-of-fold raw scores on the training rows
    oof_calibrated: pd.Series   # honestly calibrated out-of-fold probabilities

    def predict_raw(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_calibrated(self, X):
        return self.calibrator.predict(self.predict_raw(X))