"""Preprocessing pipeline.

Everything that learns from data (scaling, encoding) lives inside the pipeline,
so during cross-validation it is fitted on each training fold only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data_processing import schema
from src.feature_engineering.physical import PHYSICAL_FEATURES, add_physical_features


def build_preprocessor(use_physical: bool = True, scale: bool = False) -> Pipeline:
    """Build the feature pipeline.

    use_physical: add temp_diff_K, power_W, wear_torque.
    scale: standardise numeric columns (needed for logistic regression, not trees).
    """
    numeric = list(schema.NUMERIC_FEATURES)
    steps = []
    if use_physical:
        steps.append(("physical", FunctionTransformer(add_physical_features)))
        numeric += PHYSICAL_FEATURES

    columns = ColumnTransformer(
        [
            ("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
            ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    steps.append(("columns", columns))
    return Pipeline(steps)