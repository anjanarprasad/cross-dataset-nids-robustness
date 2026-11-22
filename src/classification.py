"""Prediction functions for the adapted network intrusion detection models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, StandardScaler

from .feature_encoder import (
    EMBEDDING_FEATURES,
    INPUT_FEATURES,
    build_classifier_features,
    extract_embeddings,
    load_compact_model,
    load_source_model,
)
from .preprocessing import (
    add_flow_features,
    load_feature_schema,
    transform_features,
)


CLASSIFIER_FEATURES = INPUT_FEATURES + EMBEDDING_FEATURES


def _check_matrix(values: np.ndarray, columns: int) -> np.ndarray:
    """Check the shape and values of a preprocessed feature matrix."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"Expected a matrix with {columns} columns")
    if not np.isfinite(matrix).all():
        raise ValueError("Features contain missing or infinite values")
    return matrix


def _check_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Attack threshold must be between 0 and 1")
    return value


@dataclass
class AdaptedNIDS:
    """Run a trained neural encoder and adapted multiclass classifier."""

    encoder: torch.nn.Module
    source_scaler: StandardScaler
    embedding_scaler: RobustScaler
    classifier: LogisticRegression
    feature_names: tuple[str, ...]
    batch_size: int = 2048

    def __post_init__(self) -> None:
        if len(self.feature_names) != INPUT_FEATURES:
            raise ValueError("Expected 43 model feature names")
        if len(set(self.feature_names)) != INPUT_FEATURES:
            raise ValueError("Model feature names must be unique")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

        if getattr(self.source_scaler, "n_features_in_", None) != INPUT_FEATURES:
            raise ValueError("The source scaler must be fitted on 43 features")
        if getattr(self.embedding_scaler, "n_features_in_", None) != EMBEDDING_FEATURES:
            raise ValueError("The embedding scaler must be fitted on 32 features")
        if getattr(self.classifier, "n_features_in_", None) != CLASSIFIER_FEATURES:
            raise ValueError("The adapted classifier must use 75 features")

        self.classes_ = np.asarray(self.classifier.classes_, dtype=str)
        if "Benign" not in self.classes_:
            raise ValueError("The adapted classifier must contain the Benign class")
        self._benign_index = int(np.flatnonzero(self.classes_ == "Benign")[0])
        self.encoder.eval()

    def classifier_features(self, x43: np.ndarray) -> np.ndarray:
        """Build the 75 inputs used by the adapted classifier."""
        features = _check_matrix(x43, INPUT_FEATURES)
        embeddings = extract_embeddings(
            self.encoder, features, batch_size=self.batch_size
        )
        return build_classifier_features(
            features, embeddings, self.embedding_scaler
        )

    def predict_proba(self, x43: np.ndarray) -> np.ndarray:
        """Return class probabilities in classifier.classes_ order."""
        features = self.classifier_features(x43)
        if len(features) == 0:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        probabilities = np.asarray(
            self.classifier.predict_proba(features), dtype=np.float64
        )
        if probabilities.shape != (len(features), len(self.classes_)):
            raise ValueError("Classifier probability output has an unexpected shape")
        if not np.isfinite(probabilities).all():
            raise ValueError("Classifier produced invalid probabilities")
        return probabilities

    def predict_preprocessed(
        self,
        x43: np.ndarray,
        attack_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Predict attack types and compute the probability of non-benign traffic.

        The attack alert is optional. Its threshold is provided by the caller
        rather than estimated from the evaluated records.
        """
        probabilities = self.predict_proba(x43)
        labels = self.classes_[probabilities.argmax(axis=1)] if len(probabilities) else np.array([], dtype=str)
        attack_scores = 1.0 - probabilities[:, self._benign_index]

        result = pd.DataFrame({
            "Predicted_Category": labels,
            "Category_Confidence": probabilities.max(axis=1) if len(probabilities) else np.array([], dtype=float),
            "Attack_Score": attack_scores,
        })
        if attack_threshold is not None:
            threshold = _check_threshold(attack_threshold)
            result["Attack_Alert"] = attack_scores >= threshold
        return result

    def predict_flows(
        self,
        records: pd.DataFrame,
        attack_threshold: float | None = None,
    ) -> pd.DataFrame:
        """Prepare flow records and return their predictions."""
        if not isinstance(records, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame of network flows")
        if records.empty:
            return self.predict_preprocessed(
                np.empty((0, INPUT_FEATURES), dtype=np.float32),
                attack_threshold=attack_threshold,
            )
        flow_data = add_flow_features(records)
        x43 = transform_features(
            flow_data, list(self.feature_names), self.source_scaler
        )
        return self.predict_preprocessed(x43, attack_threshold=attack_threshold)


def load_detector(
    *,
    feature_schema_path: str | Path,
    source_scaler_path: str | Path,
    encoder_checkpoint_path: str | Path,
    embedding_scaler_path: str | Path,
    classifier_path: str | Path,
    model_kind: str = "teacher",
    device: str = "cpu",
    batch_size: int = 2048,
) -> AdaptedNIDS:
    """Load fitted preprocessing, encoder, and classification artifacts.

    ``teacher`` loads the source encoder. ``compact`` loads a pruned or
    distilled encoder with a saved hidden width of 16 or 32.
    """
    schema = load_feature_schema(feature_schema_path)
    names: Sequence[str] = schema["final_features_in_order"]

    # Joblib files should come from a trusted experiment run.
    source_scaler = joblib.load(source_scaler_path)
    embedding_scaler = joblib.load(embedding_scaler_path)
    classifier = joblib.load(classifier_path)

    if model_kind == "teacher":
        encoder = load_source_model(encoder_checkpoint_path, names, device=device)
    elif model_kind == "compact":
        encoder = load_compact_model(encoder_checkpoint_path, device=device)
    else:
        raise ValueError("model_kind must be 'teacher' or 'compact'")

    return AdaptedNIDS(
        encoder=encoder,
        source_scaler=source_scaler,
        embedding_scaler=embedding_scaler,
        classifier=classifier,
        feature_names=tuple(names),
        batch_size=batch_size,
    )
