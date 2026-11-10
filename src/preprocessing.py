from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_numeric_preprocessor(feature_columns: list[str], scale: bool = True) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        [("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
