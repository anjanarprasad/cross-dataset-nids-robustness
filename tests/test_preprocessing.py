"""Tests for feature preparation and dataset splitting."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    ENGINEERED_FEATURES,
    FLOW_COLUMNS,
    REMOVED_FEATURES,
    SPECIAL_FEATURE,
    add_flow_features,
    feature_matrix,
    fit_source_scaler,
    load_feature_schema,
    split_source,
    split_target,
    transform_features,
)


@pytest.fixture
def feature_names():
    raw = list(FLOW_COLUMNS) + [SPECIAL_FEATURE] + list(REMOVED_FEATURES)
    raw += [f"RAW_{i:02d}" for i in range(31)]
    retained = [name for name in raw if name not in REMOVED_FEATURES]
    return raw, retained + list(ENGINEERED_FEATURES)


@pytest.fixture
def sample_flows(feature_names):
    raw_features, _ = feature_names
    rng = np.random.default_rng(17)
    n = 120
    data = {name: rng.integers(1, 1000, n).astype(float) for name in raw_features}
    data["FLOW_DURATION_MILLISECONDS"] = np.arange(1, n + 1, dtype=float)
    data[SPECIAL_FEATURE] = np.linspace(-500, 500, n)
    frame = pd.DataFrame(data)
    frame["Label"] = np.arange(n) % 2
    frame["Attack"] = np.where(frame["Label"] == 1, "Attack", "Benign")
    return add_flow_features(frame)


def test_engineered_feature_values_and_input_unchanged():
    flows = pd.DataFrame({
        "IN_BYTES": [100.0, -5.0],
        "OUT_BYTES": [50.0, 0.0],
        "IN_PKTS": [4.0, -2.0],
        "OUT_PKTS": [2.0, 0.0],
        "FLOW_DURATION_MILLISECONDS": [9.0, -10.0],
    })
    original = flows.copy(deep=True)
    result = add_flow_features(flows)

    pd.testing.assert_frame_equal(flows, original)
    assert all(name in result.columns for name in ENGINEERED_FEATURES)
    assert result.loc[0, "FE_LOG_TOTAL_BYTES"] == pytest.approx(np.log1p(150))
    assert result.loc[0, "FE_LOG_TOTAL_PKTS"] == pytest.approx(np.log1p(6))
    assert result.loc[0, "FE_BYTE_ASYMMETRY"] == pytest.approx(50 / 151)
    assert result.loc[0, "FE_PKT_ASYMMETRY"] == pytest.approx(2 / 7)
    assert result.loc[0, "FE_LOG_AVG_PKT_SIZE"] == pytest.approx(np.log1p(150 / 7))
    assert result.loc[0, "FE_LOG_DURATION"] == pytest.approx(np.log1p(9))
    np.testing.assert_array_equal(result.loc[1, list(ENGINEERED_FEATURES)].to_numpy(), np.zeros(6))


def test_missing_flow_column_raises():
    with pytest.raises(ValueError, match="Missing flow columns"):
        add_flow_features(pd.DataFrame({"IN_BYTES": [1]}))


def test_feature_schema_is_consistent(tmp_path, feature_names):
    raw, final = feature_names
    schema = {
        "raw_features": raw,
        "retained_raw_features": [name for name in raw if name not in REMOVED_FEATURES],
        "engineered_features": list(ENGINEERED_FEATURES),
        "final_features_in_order": final,
    }
    path = tmp_path / "feature_schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    assert load_feature_schema(path) == schema

    schema["final_features_in_order"] = final[::-1]
    path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(ValueError, match="Feature names or feature order"):
        load_feature_schema(path)


def test_feature_matrix_has_43_columns_in_schema_order(sample_flows, feature_names):
    _, final = feature_names
