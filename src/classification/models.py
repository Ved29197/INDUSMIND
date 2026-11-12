"""The candidate models, each with its class-imbalance handling.

Settings are deliberately plain defaults. We compare first and tune later, because
tuning on only ~270 training failures mostly fits noise.

Class weights make a model care more about the rare failures. They also distort
predicted probabilities, so probabilities are calibrated separately (Milestone 4.3).
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.classification.splits import RANDOM_STATE

NEEDS_SCALING = {"Logistic Regression"}  # trees do not care about feature scale


def get_models(y_train, n_estimators: int = 300, seed: int = RANDOM_STATE) -> dict:
    """Return {name: untrained model}. y_train supplies only the class ratio."""
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    return {
        "Baseline (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=n_estimators, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / n_pos, eval_metric="aucpr",
            n_jobs=2, random_state=seed,
        ),
    }