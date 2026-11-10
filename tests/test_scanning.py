import numpy as np
import pandas as pd

from src.cost_policy import challenge_metrics, top_k_metrics
from src.data_contract import make_xy, validate_schema
from src.preprocessing import OutlierClipper, build_outlier_audit, make_numeric_preprocessor


def tiny_frame():
    columns = ["class", "f1", "f2"] + [f"f{i}" for i in range(3, 171)]
    rows = []
    for i in range(6):
        row = ["pos" if i in (0, 3) else "neg", float(i), np.nan if i == 2 else float(i + 1)]
        row.extend([float(i + j) for j in range(3, 171)])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def test_contract_maps_positive_class_and_schema():
    frame = tiny_frame()
    validate_schema(frame)
    X, y = make_xy(frame)
    assert X.shape == (6, 170)
    assert y.tolist() == [1, 0, 0, 1, 0, 0]


def test_cost_policy_penalises_missed_failure_more():
    metrics = challenge_metrics([1, 0], [0.1, 0.1], threshold=0.5)
    assert metrics["fn"] == 1
    assert metrics["total_cost"] == 500
    assert metrics["missed_failure_cost"] == 500


def test_top_k_selects_exact_capacity():
    metrics = top_k_metrics([1, 0, 1, 0, 0], [0.2, 0.9, 0.8, 0.1, 0.3], fraction=0.4)
    assert metrics["inspections"] == 2
    assert metrics["tp"] == 1


def test_preprocessor_is_constructed_for_numeric_features():
    preprocessor = make_numeric_preprocessor(["f1", "f2"])
    transformed = preprocessor.fit_transform(pd.DataFrame({"f1": [1, np.nan], "f2": [2, 3]}))
    assert transformed.shape[0] == 2
    assert np.isfinite(transformed).all()


def test_outlier_rules_use_iqr_for_skewed_and_z_score_for_symmetric_data():
    rng = np.random.default_rng(77)
    symmetric = rng.normal(size=500)
    skewed = rng.exponential(size=500)
    frame = pd.DataFrame({"symmetric": symmetric, "skewed": skewed})

    audit = build_outlier_audit(frame)
    rules = dict(zip(audit["feature"], audit["method"]))
    assert rules["symmetric"] == "z-score"
    assert rules["skewed"] == "IQR"

    clipper = OutlierClipper().fit(frame)
    transformed = clipper.transform(pd.DataFrame({"symmetric": [100.0], "skewed": [100.0]}))
    assert np.isfinite(transformed).all()
    assert transformed[0, 0] < 100.0
    assert transformed[0, 1] < 100.0
