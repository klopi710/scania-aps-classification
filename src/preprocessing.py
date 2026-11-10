from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip feature extremes without deleting rows or learning from validation data.

    Approximately symmetric features use a mean +/- z-score bound. Skewed features
    use Tukey's IQR bound. The transformer is fitted inside each model pipeline,
    so cross-validation learns bounds only from the training fold.
    """

    def __init__(
        self,
        skewness_threshold: float = 1.0,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        self.skewness_threshold = skewness_threshold
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2:
            raise ValueError("OutlierClipper expects a two-dimensional numeric matrix")

        self.n_features_in_ = values.shape[1]
        self.feature_methods_ = []
        self.skewness_ = np.full(self.n_features_in_, np.nan, dtype=float)
        self.lower_bounds_ = np.full(self.n_features_in_, -np.inf, dtype=float)
        self.upper_bounds_ = np.full(self.n_features_in_, np.inf, dtype=float)

        for index in range(self.n_features_in_):
            column = values[:, index]
            observed = column[np.isfinite(column)]
            if observed.size < 3:
                self.feature_methods_.append("none")
                continue

            skewness = float(pd.Series(observed).skew())
            self.skewness_[index] = skewness
            if abs(skewness) <= self.skewness_threshold:
                method = "z-score"
                mean = float(np.mean(observed))
                std = float(np.std(observed, ddof=1))
                lower = mean - self.z_threshold * std
                upper = mean + self.z_threshold * std
            else:
                method = "IQR"
                q1, q3 = np.quantile(observed, [0.25, 0.75])
                spread = float(q3 - q1)
                lower = float(q1 - self.iqr_multiplier * spread)
                upper = float(q3 + self.iqr_multiplier * spread)

            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                self.lower_bounds_[index] = lower
                self.upper_bounds_[index] = upper
                self.feature_methods_.append(method)
            else:
                self.feature_methods_.append("none")

        return self

    def transform(self, X):
        if not hasattr(self, "lower_bounds_"):
            raise RuntimeError("OutlierClipper must be fitted before transform")
        values = np.asarray(X, dtype=float).copy()
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Input shape does not match the fitted OutlierClipper")
        for index, (lower, upper) in enumerate(zip(self.lower_bounds_, self.upper_bounds_)):
            observed = np.isfinite(values[:, index])
            values[observed, index] = np.clip(values[observed, index], lower, upper)
        return values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_outlier_audit(X, skewness_threshold: float = 1.0) -> pd.DataFrame:
    """Return a development-only diagnostic of the selected clipping rule."""
    feature_names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.asarray(X).shape[1])]
    clipper = OutlierClipper(skewness_threshold=skewness_threshold).fit(X)
    values = np.asarray(X, dtype=float)
    rows = []
    for index, feature in enumerate(feature_names):
        observed = values[:, index]
        valid = np.isfinite(observed)
        flagged = valid & ((observed < clipper.lower_bounds_[index]) | (observed > clipper.upper_bounds_[index]))
        method = clipper.feature_methods_[index]
        rows.append(
            {
                "feature": feature,
                "skewness": clipper.skewness_[index],
                "distribution_rule": "approximately symmetric" if method == "z-score" else "skewed/non-normal" if method == "IQR" else "insufficient variation",
                "method": method,
                "lower_bound": clipper.lower_bounds_[index],
                "upper_bound": clipper.upper_bounds_[index],
                "observed_values": int(valid.sum()),
                "flagged_count": int(flagged.sum()),
                "flagged_rate": float(flagged.sum() / valid.sum()) if valid.any() else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("flagged_rate", ascending=False).reset_index(drop=True)


def make_numeric_preprocessor(feature_columns: list[str], scale: bool = True) -> ColumnTransformer:
    steps = [
        ("outlier_clipper", OutlierClipper()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        [("numeric", numeric_pipeline, feature_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
