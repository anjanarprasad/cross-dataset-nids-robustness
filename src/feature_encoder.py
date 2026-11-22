"""Neural feature encoders used in the  intrusion detection experiments."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler


SOURCE_CLASSES = (
    "Benign", "Exploits", "Fuzzers", "Generic", "Reconnaissance",
    "DoS", "Backdoor", "Analysis", "Shellcode", "Worms",
)
INPUT_FEATURES = 43
EMBEDDING_FEATURES = 32


class RareAwareNet(nn.Module):
    """Source model with a 43-to-32 encoder and 10-class prediction head."""

    def __init__(self, input_dim: int = INPUT_FEATURES, n_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, EMBEDDING_FEATURES),
            nn.GELU(),
        )
        # The decoder is used during source-only denoising pretraining.
        self.decoder = nn.Sequential(
            nn.Linear(EMBEDDING_FEATURES, 64),
            nn.GELU(),
            nn.Linear(64, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(EMBEDDING_FEATURES),
            nn.Linear(EMBEDDING_FEATURES, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.classifier(z), z


class CompactNIDS(nn.Module):
    """Smaller encoder used for structured pruning and distillation."""

    def __init__(self, width: int, n_classes: int = 10):
        super().__init__()
        if width not in (16, 32):
            raise ValueError("Expected a compact hidden width of 16 or 32")
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_FEATURES, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, EMBEDDING_FEATURES),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(EMBEDDING_FEATURES),
            nn.Linear(EMBEDDING_FEATURES, n_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.classifier(z), z


def encoder_parameters(model: nn.Module) -> int:
    """Count parameters in the neural encoder, excluding classifier and decoder."""
    return sum(param.numel() for param in model.encoder.parameters())


def load_source_model(
    checkpoint_path: str | Path,
    feature_names: Sequence[str],
    device: str = "cpu",
) -> RareAwareNet:
    """Load the trained UNSW model and check its feature and class order."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("input_features") != list(feature_names):
        raise ValueError("Feature names or feature order do not match the checkpoint")
    if checkpoint.get("class_names") != list(SOURCE_CLASSES):
        raise ValueError("Source attack categories do not match the checkpoint")
    if len(feature_names) != INPUT_FEATURES:
        raise ValueError("Expected 43 input features")

    model = RareAwareNet().cpu()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def load_compact_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> CompactNIDS:
    """Load a pruned or distilled model saved by the compression notebook."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    width = int(checkpoint["hidden_width"])
    n_classes = int(checkpoint["n_classes"])
    if n_classes != len(SOURCE_CLASSES):
        raise ValueError("Unexpected number of source attack classes")

    model = CompactNIDS(width=width, n_classes=n_classes)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def extract_embeddings(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """Extract 32 neural features from preprocessed 43-feature inputs."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != INPUT_FEATURES:
        raise ValueError("Expected an N x 43 feature matrix")
    if not np.isfinite(x).all():
        raise ValueError("The input contains missing or infinite values")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(x) == 0:
        return np.empty((0, EMBEDDING_FEATURES), dtype=np.float32)

    model.eval()
    device = next(model.parameters()).device
    parts = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], device=device)
        z = model.encoder(batch)
        parts.append(z.cpu().numpy())
    return np.concatenate(parts).astype(np.float32, copy=False)


def fit_embedding_scaler(source_embeddings: np.ndarray) -> RobustScaler:
    """Fit the embedding transform on UNSW training examples only."""
    z = np.asarray(source_embeddings, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != EMBEDDING_FEATURES or len(z) == 0:
        raise ValueError("Expected nonempty source embeddings with 32 columns")
    if not np.isfinite(z).all():
        raise ValueError("Source embeddings contain missing or infinite values")
    return RobustScaler(quantile_range=(10, 90)).fit(np.arcsinh(z))


def transform_embeddings(
    embeddings: np.ndarray,
    source_embedding_scaler: RobustScaler,
) -> np.ndarray:
    """Apply the source-fitted scaler and clip embedding values to [-8, 8]."""
    z = np.asarray(embeddings, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != EMBEDDING_FEATURES:
        raise ValueError("Expected an N x 32 embedding matrix")
    if not np.isfinite(z).all():
        raise ValueError("Embeddings contain missing or infinite values")
    if len(z) == 0:
        return np.empty((0, EMBEDDING_FEATURES), dtype=np.float32)
    scaled = source_embedding_scaler.transform(np.arcsinh(z))
    return np.clip(scaled, -8, 8).astype(np.float32)


def build_classifier_features(
    flow_features: np.ndarray,
    embeddings: np.ndarray,
    source_embedding_scaler: RobustScaler,
) -> np.ndarray:
    """Combine 43 flow features with 32 scaled neural features (75 total)."""
    x = np.asarray(flow_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != INPUT_FEATURES:
        raise ValueError("Expected an N x 43 flow-feature matrix")
    if not np.isfinite(x).all():
        raise ValueError("Flow features contain missing or infinite values")
    z = transform_embeddings(embeddings, source_embedding_scaler)
    if len(x) != len(z):
        raise ValueError("Flow features and embeddings have different row counts")
    return np.column_stack((x, z)).astype(np.float32)
