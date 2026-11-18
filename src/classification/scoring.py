"""Helpers that turn a model score into something a maintenance manager can act on.

Light on purpose (no training code), so the deployed app can import it.
"""
import numpy as np
import pandas as pd

from src.data_processing import schema

OVERSTRAIN_LIMITS = {"L": 11000, "M": 12000, "H": 13000}


def risk_category(p: float, threshold: float, base_rate: float) -> str:
    """Inspect: at or above the cost-based threshold. Elevated: above the average
    failure rate. Low: otherwise."""
    if p >= threshold:
        return "Inspect"
    if p >= base_rate:
        return "Elevated"
    return "Low"


def typical_values(features: pd.DataFrame) -> pd.Series:
    """A 'typical machine': median of each numeric input, most common Type."""
    typical = features[schema.NUMERIC_FEATURES].median().astype(object)  # object: it will also hold the text Type
    typical["Type"] = features["Type"].mode().iloc[0]
    return typical[schema.FEATURE_COLUMNS]


def what_if_contributions(model, row: pd.DataFrame, typical: pd.Series) -> pd.Series:
    """Risk change when ONE input at a time is replaced by its typical value.

    Positive: this input raises the machine's risk compared with a typical machine.
    A simple what-if view, not a full explanation (SHAP comes in Phase 5).
    """
    base = float(model.predict_calibrated(row)[0])
    variants = [row.assign(**{col: typical[col]}) for col in schema.FEATURE_COLUMNS]
    risks = model.predict_calibrated(pd.concat(variants, ignore_index=True))
    return pd.Series(base - risks, index=schema.FEATURE_COLUMNS)


def engineering_checks(row: dict) -> pd.DataFrame:
    """Physical reference limits from the AI4I documentation, as engineering overrides.

    For a real plant, replace these with the equipment's own design limits.
    """
    gap = row["Process temperature [K]"] - row["Air temperature [K]"]
    rpm = row["Rotational speed [rpm]"]
    torque = row["Torque [Nm]"]
    wear = row["Tool wear [min]"]
    power = torque * rpm * 2 * np.pi / 60
    limit = OVERSTRAIN_LIMITS[row["Type"]]
    wear_torque = wear * torque
    checks = [
        ("Cooling margin", f"gap {gap:.1f} K at {rpm:.0f} rpm",
         "gap below 8.6 K together with speed below 1,380 rpm", gap < 8.6 and rpm < 1380),
        ("Power window", f"{power:,.0f} W", "outside 3,500 to 9,000 W",
         power < 3500 or power > 9000),
        ("Overstrain", f"wear x torque = {wear_torque:,.0f}",
         f"above {limit:,} for type {row['Type']}", wear_torque > limit),
        ("Tool wear", f"{wear:.0f} min", "200 min or more", wear >= 200),
    ]
    return pd.DataFrame(
        [(name, value, rule, "ALERT" if hit else "Within limit") for name, value, rule, hit in checks],
        columns=["Check", "Current value", "Alert condition", "Status"],
    )