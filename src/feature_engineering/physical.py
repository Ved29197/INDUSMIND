"""Physical features derived from sensor readings only.

These use NO label or cause-flag columns, so they cannot leak the answer.
They do encode the documented failure rules (temperature gap, mechanical
power, overstrain proxy), so we report model results with and without them.
"""
import numpy as np
import pandas as pd

PHYSICAL_FEATURES = ["temp_diff_K", "power_W", "wear_torque"]


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with three engineered columns added."""
    out = df.copy()
    out["temp_diff_K"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    # power (W) = torque (Nm) x angular speed (rad/s); rpm -> rad/s is x 2*pi/60
    out["power_W"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * 2 * np.pi / 60
    out["wear_torque"] = out["Tool wear [min]"] * out["Torque [Nm]"]
    return out