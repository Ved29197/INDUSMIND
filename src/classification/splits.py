"""Train/test splitting and cross-validation setup.

All splitting goes through get_features_and_target(), so identifiers and
cause flags can never enter the features.
"""
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data_processing.loader import get_features_and_target

RANDOM_STATE = 42  # fixed seed so every run gives identical splits


def stratified_split(df, test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Primary split: random, with the same failure rate in train and test."""
    X, y = get_features_and_target(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def blocked_split(df, test_fraction: float = 0.2):
    """Robustness split: train on the earliest rows (by UDI), test on the latest.

    Rows have a hidden time order, so this mimics predicting a later period.
    """
    ordered = df.sort_values("UDI")
    cut = int(len(ordered) * (1 - test_fraction))
    X_train, y_train = get_features_and_target(ordered.iloc[:cut])
    X_test, y_test = get_features_and_target(ordered.iloc[cut:])
    return X_train, X_test, y_train, y_test


def make_cv(n_splits: int = 5, seed: int = RANDOM_STATE) -> StratifiedKFold:
    """Stratified k-fold for use on the TRAINING set only."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)