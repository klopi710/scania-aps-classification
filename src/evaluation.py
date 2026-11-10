from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _arrays(y_true: Iterable[int], y_score: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(list(y_true), dtype=int), np.asarray(list(y_score), dtype=float)


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float = 0.5) -> dict[str, float | int]:
    y_true, y_score = _arrays(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.column_stack([1 - y_score, y_score]), labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_score)),
        "inspection_rate": float(y_pred.mean()),
    }


def oof_predictions(model, X, y, random_state: int = 77, n_splits: int = 2) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]


def evaluate_model_set(models: dict, X_dev, y_dev, X_test, y_test, random_state: int = 77):
    rows = []
    predictions = {}
    for name, model in models.items():
        dev_score = oof_predictions(model, X_dev, y_dev, random_state=random_state)
        fitted_model = model.fit(X_dev, y_dev)
        test_score = fitted_model.predict_proba(X_test)[:, 1]
        dev_metrics = binary_metrics(y_dev, dev_score)
        test_metrics = binary_metrics(y_test, test_score)
        rows.append({"model": name, "split": "development OOF", **dev_metrics})
        rows.append({"model": name, "split": "official frozen test", **test_metrics})
        predictions[name] = {"development": dev_score, "frozen_test": test_score}
        del fitted_model
    return pd.DataFrame(rows), predictions, {}
