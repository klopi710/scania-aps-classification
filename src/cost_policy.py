from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


COST_UNNECESSARY_INSPECTION = 10.0
COST_MISSED_FAILURE = 500.0


def challenge_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float,
    cost_fp: float = COST_UNNECESSARY_INSPECTION,
    cost_fn: float = COST_MISSED_FAILURE,
) -> dict[str, float | int]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    baseline_cost = float(y_true.sum() * cost_fn)
    total_cost = float(fp * cost_fp + fn * cost_fn)
    return {
        "threshold": float(threshold),
        "inspections": int(tp + fp),
        "inspection_rate": float((tp + fp) / len(y_true)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "unnecessary_inspection_cost": float(fp * cost_fp),
        "missed_failure_cost": float(fn * cost_fn),
        "total_cost": total_cost,
        "all_negative_baseline_cost": baseline_cost,
        "cost_avoided_vs_all_negative": float(baseline_cost - total_cost),
    }


def top_k_metrics(y_true: Iterable[int], y_score: Iterable[float], fraction: float) -> dict[str, float | int]:
    y_true = np.asarray(list(y_true), dtype=int)
    y_score = np.asarray(list(y_score), dtype=float)
    k = max(1, int(round(len(y_true) * fraction)))
    order = np.argsort(-y_score, kind="mergesort")
    selected = np.zeros(len(y_true), dtype=int)
    selected[order[:k]] = 1
    selected_score = np.where(selected == 1, 1.0, 0.0)
    result = challenge_metrics(y_true, selected_score, 0.5)
    result["top_k_fraction"] = float(fraction)
    return result


def threshold_table(y_true: Iterable[int], y_score: Iterable[float], points: list[dict]) -> pd.DataFrame:
    rows = []
    for point in points:
        row = challenge_metrics(y_true, y_score, float(point["threshold"]))
        row["operating_point"] = point["name"]
        row["reason"] = point["reason"]
        rows.append(row)
    return pd.DataFrame(rows)


def cost_sweep(y_true: Iterable[int], y_score: Iterable[float], thresholds: Iterable[float]) -> pd.DataFrame:
    return pd.DataFrame([challenge_metrics(y_true, y_score, float(t)) for t in thresholds])
