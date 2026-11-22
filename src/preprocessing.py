"""Feature preparation and dataset splitting for the  NIDS experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


EXCLUDED_COLUMNS = (
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"
)
REMOVED_FEATURES = (
    "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"
)
ENGINEERED_FEATURES = (
    "FE_LOG_TOTAL_BYTES", "FE_LOG_TOTAL_PKTS",
    "FE_BYTE_ASYMMETRY", "FE_PKT_ASYMMETRY",
    "FE_LOG_AVG_PKT_SIZE", "FE_LOG_DURATION"
)
SPECIAL_FEATURE = "SRC_TO_DST_SECOND_BYTES"
FLOW_COLUMNS = (
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS"
)


def load_feature_schema(path: str | Path) -> dict[str, Any]:
    """Read and check the feature order saved during data preparation."""
    with Path(path).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    raw = schema["raw_features"]
    retained = schema["retained_raw_features"]
    added = schema["engineered_features"]
    names = schema["final_features_in_order"]

    if len(raw) != 41 or len(retained) != 37 or len(names) != 43:
        raise ValueError("Expected 41 raw, 37 retained and 43 final features")
    if retained != [name for name in raw if name not in REMOVED_FEATURES]:
        raise ValueError("Retained feature list does not match the exclusion rule")
    if added != list(ENGINEERED_FEATURES) or names != retained + added:
        raise ValueError("Feature names or feature order do not match")
    if SPECIAL_FEATURE not in names or len(names) != len(set(names)):
        raise ValueError("Invalid feature schema")

    return schema


def add_flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create six flow features using the same formulas as the notebooks."""
    missing = set(FLOW_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing flow columns: {sorted(missing)}")

    result = frame.copy()
    in_bytes = result["IN_BYTES"].clip(lower=0)
    out_bytes = result["OUT_BYTES"].clip(lower=0)
    in_pkts = result["IN_PKTS"].clip(lower=0)
    out_pkts = result["OUT_PKTS"].clip(lower=0)

    total_bytes = in_bytes + out_bytes
    total_pkts = in_pkts + out_pkts

    result["FE_LOG_TOTAL_BYTES"] = np.log1p(total_bytes)
    result["FE_LOG_TOTAL_PKTS"] = np.log1p(total_pkts)
    result["FE_BYTE_ASYMMETRY"] = (
        (in_bytes - out_bytes) / (total_bytes + 1)
    )
    result["FE_PKT_ASYMMETRY"] = (
        (in_pkts - out_pkts) / (total_pkts + 1)
    )
    result["FE_LOG_AVG_PKT_SIZE"] = np.log1p(
        total_bytes / (total_pkts + 1)
    )
    result["FE_LOG_DURATION"] = np.log1p(
        result["FLOW_DURATION_MILLISECONDS"].clip(lower=0)
    )
    return result


def feature_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Read the 43 input features in their saved order."""
    if len(feature_names) != 43 or len(set(feature_names)) != 43:
        raise ValueError("Expected 43 unique feature names")
    missing = set(feature_names).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    x = frame.loc[:, feature_names].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x).all():

