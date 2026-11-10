from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import make_numeric_preprocessor


def make_candidate_models(feature_columns: list[str], random_state: int = 77) -> dict[str, Pipeline]:
    """Use the credit-assignment comparison pattern, adapted to rare APS failures."""
    def pipe(estimator, scale: bool = True):
        return Pipeline(
            [
                ("preprocess", make_numeric_preprocessor(feature_columns, scale=scale)),
                ("model", estimator),
            ]
        )

    return {
        "majority_baseline": pipe(DummyClassifier(strategy="prior")),
        "logistic_regression": pipe(
            LogisticRegression(
                max_iter=500,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
        ),
        "decision_tree": pipe(
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
            ),
            scale=False,
        ),
        "random_forest": pipe(
            RandomForestClassifier(
                n_estimators=50,
                max_depth=16,
                min_samples_leaf=5,
                max_features="sqrt",
                max_samples=0.75,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
            scale=False,
        ),
    }
