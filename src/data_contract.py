from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "class"
POSITIVE_LABEL = "pos"
NEGATIVE_LABEL = "neg"
CSV_METADATA_ROWS = 20
EXPECTED_ROWS = 60_000
EXPECTED_FEATURES = 170


def load_raw_data(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a UCI APS CSV, skipping the 20-line licence preamble."""
    return pd.read_csv(path, skiprows=CSV_METADATA_ROWS, na_values=["na"], nrows=nrows)


def validate_schema(frame: pd.DataFrame, require_full_training_size: bool = False) -> None:
    if frame.empty:
        raise ValueError("The dataset is empty")
    if frame.columns[0] != TARGET_COLUMN:
        raise ValueError(f"Expected first column {TARGET_COLUMN!r}")
    if len(frame.columns) != EXPECTED_FEATURES + 1:
        raise ValueError(f"Expected 171 columns, got {len(frame.columns)}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names detected")
    if not frame[TARGET_COLUMN].isin([NEGATIVE_LABEL, POSITIVE_LABEL]).all():
        raise ValueError("Unexpected target labels")
    if require_full_training_size and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} training rows, got {len(frame)}")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("The target contains missing values")
    feature_types = frame.drop(columns=[TARGET_COLUMN]).dtypes
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in feature_types):
        raise ValueError("All operational predictors must be numeric after parsing")


def make_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_schema(frame)
    X = frame.drop(columns=[TARGET_COLUMN]).copy()
    y = frame[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).astype(int)
    return X, y


def contract_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    X, y = make_xy(frame)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "features": int(X.shape[1]),
        "positive_failures": int(y.sum()),
        "negative_unrelated_failures": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
        "missing_cells": int(X.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_features": int((X.nunique(dropna=False) <= 1).sum()),
    }
