"""Column definitions for the AI4I 2020 dataset.

Keeping these in one place means the validator, the models and the app all
agree on which columns are features, which are the target, and which must
never be used for prediction (leakage).
"""

TARGET = "Machine failure"

# Identifiers: no physical meaning, must not be used as features.
ID_COLUMNS = ["UDI", "Product ID"]

# Failure modes are the target split by cause. They are only known AFTER a
# failure happens, so using them to predict failure would be leakage.
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LEAKAGE_COLUMNS = ID_COLUMNS + FAILURE_MODE_COLUMNS

REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET] + FAILURE_MODE_COLUMNS

VALID_TYPES = {"L", "M", "H"}

# Deliberately WIDE physical bounds. They catch unit mistakes and corrupt
# values (e.g. Celsius instead of Kelvin), not unusual-but-valid readings.
PLAUSIBLE_RANGES = {
    "Air temperature [K]": (250, 350),
    "Process temperature [K]": (250, 400),
    "Rotational speed [rpm]": (0, 10000),
    "Torque [Nm]": (0, 200),
    "Tool wear [min]": (0, 500),
}