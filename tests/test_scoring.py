import numpy as np
import pandas as pd
import pytest

from src.classification.scoring import (
    engineering_checks, risk_category, typical_values, what_if_contributions,
)
from src.data_processing import schema


def machine(**overrides):
    base = {"Type": "L", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
            "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 100}
    base.update(overrides)
    return base


def status(checks, name):
    return checks.set_index("Check").loc[name, "Status"]


def test_risk_category_boundaries():
    assert risk_category(0.10, threshold=0.09, base_rate=0.034) == "Inspect"
    assert risk_category(0.09, threshold=0.09, base_rate=0.034) == "Inspect"   # at threshold
    assert risk_category(0.05, threshold=0.09, base_rate=0.034) == "Elevated"
    assert risk_category(0.01, threshold=0.09, base_rate=0.034) == "Low"


def test_cooling_margin_needs_both_conditions():
    assert status(engineering_checks(machine(**{"Process temperature [K]": 308.0, "Rotational speed [rpm]": 1300})), "Cooling margin") == "ALERT"
    assert status(engineering_checks(machine(**{"Process temperature [K]": 308.0, "Rotational speed [rpm]": 1500})), "Cooling margin") == "Within limit"
    assert status(engineering_checks(machine(**{"Rotational speed [rpm]": 1300})), "Cooling margin") == "Within limit"


def test_power_window_both_sides():
    assert status(engineering_checks(machine()), "Power window") == "Within limit"           # about 6,283 W
    assert status(engineering_checks(machine(**{"Torque [Nm]": 70.0, "Rotational speed [rpm]": 3000})), "Power window") == "ALERT"
    assert status(engineering_checks(machine(**{"Torque [Nm]": 10.0, "Rotational speed [rpm]": 1500})), "Power window") == "ALERT"


def test_overstrain_limit_depends_on_type():
    heavy = {"Tool wear [min]": 250, "Torque [Nm]": 50.0}          # wear x torque = 12,500
    assert status(engineering_checks(machine(Type="L", **heavy)), "Overstrain") == "ALERT"
    assert status(engineering_checks(machine(Type="M", **heavy)), "Overstrain") == "ALERT"
    assert status(engineering_checks(machine(Type="H", **heavy)), "Overstrain") == "Within limit"


def test_tool_wear_alert_starts_at_200():
    assert status(engineering_checks(machine(**{"Tool wear [min]": 199})), "Tool wear") == "Within limit"
    assert status(engineering_checks(machine(**{"Tool wear [min]": 200})), "Tool wear") == "ALERT"


def test_typical_values_use_median_and_mode():
    X = pd.DataFrame([machine(Type="L", **{"Torque [Nm]": 10.0}),
                      machine(Type="L", **{"Torque [Nm]": 30.0}),
                      machine(Type="M", **{"Torque [Nm]": 90.0})])[schema.FEATURE_COLUMNS]
    typical = typical_values(X)
    assert typical["Type"] == "L" and typical["Torque [Nm]"] == 30.0
    assert list(typical.index) == schema.FEATURE_COLUMNS


class StubModel:
    """Risk rises with torque only, so contributions are easy to predict."""

    def predict_calibrated(self, X):
        return np.clip(X["Torque [Nm]"].to_numpy(dtype=float) / 100, 0, 1)


def test_what_if_contributions_with_a_known_model():
    typical = pd.Series(machine(**{"Torque [Nm]": 40.0}))[schema.FEATURE_COLUMNS]
    row = pd.DataFrame([machine(**{"Torque [Nm]": 70.0})])[schema.FEATURE_COLUMNS]
    contrib = what_if_contributions(StubModel(), row, typical)
    assert contrib["Torque [Nm]"] == pytest.approx(0.30)       # 0.70 versus 0.40
    others = contrib.drop("Torque [Nm]")
    assert (others.abs() < 1e-12).all()


def test_what_if_is_zero_for_a_typical_machine():
    typical = pd.Series(machine())[schema.FEATURE_COLUMNS]
    row = pd.DataFrame([machine()])[schema.FEATURE_COLUMNS]
    assert (what_if_contributions(StubModel(), row, typical).abs() < 1e-12).all()