"""Feature preparation and dataset splitting for the CSC786 NIDS experiments."""

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
        raise ValueError("Input features contain missing or infinite values")
    return x


def _before_scaling(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    x = feature_matrix(frame, feature_names)
    x[:, feature_names.index(SPECIAL_FEATURE)] = np.arcsinh(
        x[:, feature_names.index(SPECIAL_FEATURE)]
    )
    return x


def fit_source_scaler(
    source_train: pd.DataFrame, feature_names: list[str]
) -> StandardScaler:
    """Fit standardization using source training records only."""
    return StandardScaler().fit(_before_scaling(source_train, feature_names))


def transform_features(
    frame: pd.DataFrame,
    feature_names: list[str],
    source_scaler: StandardScaler,
) -> np.ndarray:
    """Apply the saved source scaler and return 43 float32 model features."""
    x = source_scaler.transform(_before_scaling(frame, feature_names))
    x = np.arcsinh(x).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Transformed features contain missing or infinite values")
    return x


def split_source(
    source: pd.DataFrame, raw_features: list[str], seed: int = 13
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group identical raw-feature records before the 70/15/15 split."""
    groups = pd.util.hash_pandas_object(
        source[raw_features], index=False
    ).to_numpy()
    train_idx, rest_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed
    ).split(source, groups=groups))
    val_local, test_local = next(GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    ).split(source.iloc[rest_idx], groups=groups[rest_idx]))
    return (
        source.iloc[train_idx].copy(),
        source.iloc[rest_idx[val_local]].copy(),
        source.iloc[rest_idx[test_local]].copy(),
    )


def split_target(
    target: pd.DataFrame,
    raw_features: list[str],
    feature_names: list[str],
    seed: int = 786,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split CIC into an adaptation pool and a unique development set."""
    groups = pd.util.hash_pandas_object(
        target[raw_features], index=False
    ).to_numpy()
    adapt_idx, dev_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=seed
    ).split(target, groups=groups))

    adaptation = target.iloc[adapt_idx].copy()
    development = target.iloc[dev_idx].copy()
    pool_hash = pd.util.hash_pandas_object(
        adaptation[feature_names], index=False
    ).to_numpy()
    dev_hash = pd.util.hash_pandas_object(
        development[feature_names], index=False
    ).to_numpy()

    unique = ~pd.Series(dev_hash).duplicated(keep="first").to_numpy()
    not_in_pool = ~np.isin(dev_hash, np.unique(pool_hash))
    development = development.iloc[np.flatnonzero(unique & not_in_pool)].copy()
    return adaptation, development
