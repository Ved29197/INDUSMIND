import pandas as pd
import pytest

from src.data_processing import schema
from src.data_processing.loader import get_features_and_target, validate_ai4i


@pytest.fixture
def valid_df():
    """A tiny, fully valid dataset (6 rows, 1 consistent failure)."""
    return pd.DataFrame(
        {
            "UDI": [1, 2, 3, 4, 5, 6],
            "Product ID": ["M1", "L2", "L3", "H4", "M5", "L6"],
            "Type": ["M", "L", "L", "H", "M", "L"],
            "Air temperature [K]": [298.1, 298.2, 298.3, 298.4, 298.5, 298.6],
            "Process temperature [K]": [308.6, 308.7, 308.8, 308.9, 309.0, 309.1],
            "Rotational speed [rpm]": [1551, 1408, 1498, 1433, 1408, 1500],
            "Torque [Nm]": [42.8, 46.3, 49.4, 39.5, 40.0, 41.0],
            "Tool wear [min]": [0, 3, 5, 7, 9, 240],
            "Machine failure": [0, 0, 0, 0, 0, 1],
            "TWF": [0, 0, 0, 0, 0, 1],
            "HDF": [0, 0, 0, 0, 0, 0],
            "PWF": [0, 0, 0, 0, 0, 0],
            "OSF": [0, 0, 0, 0, 0, 0],
            "RNF": [0, 0, 0, 0, 0, 0],
        }
    )


def test_valid_data_passes(valid_df):
    report = validate_ai4i(valid_df)
    assert report.passed
    assert report.warnings == []
    assert report.info["n_failures"] == 1


def test_missing_column_is_error(valid_df):
    report = validate_ai4i(valid_df.drop(columns=["Torque [Nm]"]))
    assert not report.passed
    assert any("Missing required columns" in e for e in report.errors)


def test_missing_values_are_error(valid_df):
    valid_df.loc[0, "Torque [Nm]"] = None
    assert not validate_ai4i(valid_df).passed


def test_unknown_type_is_error(valid_df):
    valid_df.loc[0, "Type"] = "X"
    assert not validate_ai4i(valid_df).passed


def test_non_binary_target_is_error(valid_df):
    valid_df.loc[0, "Machine failure"] = 2
    assert not validate_ai4i(valid_df).passed


def test_duplicate_udi_is_error(valid_df):
    valid_df.loc[1, "UDI"] = 1
    assert not validate_ai4i(valid_df).passed


def test_celsius_instead_of_kelvin_is_warning(valid_df):
    valid_df.loc[0, "Air temperature [K]"] = 25.0
    report = validate_ai4i(valid_df)
    assert any("Air temperature" in w for w in report.warnings)


def test_label_inconsistency_is_warning_not_error(valid_df):
    valid_df.loc[5, "TWF"] = 0  # failure with no mode flagged
    report = validate_ai4i(valid_df)
    assert report.passed
    assert report.info["failures_with_no_mode_flag"] == 1
    assert any("inconsistency" in w for w in report.warnings)


def test_features_exclude_leakage_columns(valid_df):
    X, y = get_features_and_target(valid_df)
    assert set(X.columns).isdisjoint(schema.LEAKAGE_COLUMNS + [schema.TARGET])
    assert len(X) == len(y) == 6