"""Load and validate the AI4I 2020 dataset.

Validation separates ERRORS (the data cannot be used safely) from WARNINGS
(the data is usable but something deserves a human look).
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.data_processing import schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = ["VALIDATION " + ("PASSED" if self.passed else "FAILED")]
        lines += [f"  ERROR:   {m}" for m in self.errors]
        lines += [f"  WARNING: {m}" for m in self.warnings]
        lines += [f"  INFO:    {k} = {v}" for k, v in self.info.items()]
        return "\n".join(lines)


def validate_ai4i(df: pd.DataFrame) -> ValidationReport:
    """Run all data-quality checks and return a report (never raises)."""
    report = ValidationReport()

    # 1. Schema: required columns must exist, otherwise nothing else can run.
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {missing}")
        return report
    extra = [c for c in df.columns if c not in schema.REQUIRED_COLUMNS]
    if extra:
        report.warnings.append(f"Unexpected extra columns (ignored): {extra}")

    # 2. Missing values in required columns.
    nulls = df[schema.REQUIRED_COLUMNS].isna().sum()
    for col, n in nulls[nulls > 0].items():
        report.errors.append(f"Column '{col}' has {n} missing values")

    # 3. Numeric columns must really be numeric.
    numeric_cols = (
        ["UDI"] + schema.NUMERIC_FEATURES + [schema.TARGET] + schema.FAILURE_MODE_COLUMNS
    )
    numeric_ok = []
    for col in numeric_cols:
        if is_numeric_dtype(df[col]):
            numeric_ok.append(col)
        else:
            report.errors.append(f"Column '{col}' is not numeric")

    # 4. Product type must be L, M or H.
    bad_types = set(df["Type"].dropna().unique()) - schema.VALID_TYPES
    if bad_types:
        report.errors.append(f"Unknown Type values: {sorted(map(str, bad_types))}")

    # 5. Target and failure-mode flags must be binary 0/1.
    for col in [schema.TARGET] + schema.FAILURE_MODE_COLUMNS:
        if col in numeric_ok and not set(df[col].dropna().unique()) <= {0, 1}:
            report.errors.append(f"Column '{col}' must contain only 0 and 1")

    # 6. Duplicates.
    if "UDI" in numeric_ok:
        dup_ids = int(df["UDI"].duplicated().sum())
        if dup_ids:
            report.errors.append(f"{dup_ids} duplicate UDI values (should be unique)")
    dup_readings = int(df.drop(columns=schema.ID_COLUMNS).duplicated().sum())
    if dup_readings:
        report.warnings.append(
            f"{dup_readings} rows have identical readings to another row (different IDs)"
        )

    # 7. Plausible physical ranges (warnings only).
    for col, (lo, hi) in schema.PLAUSIBLE_RANGES.items():
        if col in numeric_ok:
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out:
                report.warnings.append(
                    f"{n_out} values in '{col}' outside plausible range [{lo}, {hi}]"
                )

    # 8. Physical sanity: process temperature should exceed air temperature.
    air, proc = "Air temperature [K]", "Process temperature [K]"
    if air in numeric_ok and proc in numeric_ok:
        n_bad = int((df[proc] <= df[air]).sum())
        if n_bad:
            report.warnings.append(f"{n_bad} rows where process temp <= air temp")

    # 9. Label consistency: Machine failure vs failure-mode flags.
    flags_ok = all(c in numeric_ok for c in [schema.TARGET] + schema.FAILURE_MODE_COLUMNS)
    if flags_ok:
        failed = df[schema.TARGET] == 1
        any_mode = df[schema.FAILURE_MODE_COLUMNS].sum(axis=1) > 0
        fail_no_mode = int((failed & ~any_mode).sum())
        mode_no_fail = int((~failed & any_mode).sum())
        report.info["failures_with_no_mode_flag"] = fail_no_mode
        report.info["mode_flag_without_failure"] = mode_no_fail
        if fail_no_mode or mode_no_fail:
            report.warnings.append(
                f"Label inconsistency: {fail_no_mode} failures with no mode flagged, "
                f"{mode_no_fail} rows with a mode flagged but no failure"
            )
        report.info["n_failures"] = int(failed.sum())
        report.info["failure_rate_pct"] = round(100 * failed.mean(), 2)

    report.info["n_rows"] = len(df)
    report.info["type_counts"] = df["Type"].value_counts().to_dict()
    return report


def load_ai4i(path=DEFAULT_PATH, strict: bool = True):
    """Read the CSV and validate it. Returns (DataFrame, ValidationReport).

    With strict=True a failed validation raises ValueError, so bad data
    can never silently reach a model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data_processing.download"
        )
    df = pd.read_csv(path)
    report = validate_ai4i(df)
    if strict and not report.passed:
        raise ValueError(report.summary())
    return df, report


def get_features_and_target(df: pd.DataFrame):
    """Return (X, y) using only legitimate features. Fails loudly on leakage."""
    X = df[schema.FEATURE_COLUMNS].copy()
    y = df[schema.TARGET].copy()
    leaked = set(X.columns) & set(schema.LEAKAGE_COLUMNS + [schema.TARGET])
    assert not leaked, f"Leakage columns found in features: {leaked}"
    return X, y