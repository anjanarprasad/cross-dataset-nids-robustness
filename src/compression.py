"""INT8 quantization, structured pruning, and distillation for NIDS encoders."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from .feature_encoder import (
    CompactNIDS,
    EMBEDDING_FEATURES,
    INPUT_FEATURES,
    RareAwareNet,
)


@dataclass
class TrainingResult:
    """The selected student model and its source-validation results."""

    model: CompactNIDS
    best_epoch: int
    validation_macro_f1: float
    history: list[dict[str, float]]


def dynamic_int8(teacher: RareAwareNet) -> nn.Module:
    """Quantize linear layers of a CPU copy without changing the teacher."""
    model = copy.deepcopy(teacher).cpu().eval()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    return quantized.eval()


def prune_encoder(teacher: RareAwareNet, width: int) -> CompactNIDS:
    """Keep the hidden units with the largest outgoing L2 weight norms."""
    if width not in (16, 32):
        raise ValueError("Pruned hidden width must be 16 or 32")

    teacher = teacher.cpu().eval()
    compact = CompactNIDS(width=width, n_classes=teacher.classifier[1].out_features)

    # Column j of the second linear layer carries the output of hidden unit j.
    outgoing = teacher.encoder[3].weight.detach()
    importance = torch.linalg.vector_norm(outgoing, ord=2, dim=0)
    keep = torch.topk(importance, width).indices.sort().values

    with torch.no_grad():
        compact.encoder[0].weight.copy_(teacher.encoder[0].weight[keep])
        compact.encoder[0].bias.copy_(teacher.encoder[0].bias[keep])
        compact.encoder[1].weight.copy_(teacher.encoder[1].weight[keep])
        compact.encoder[1].bias.copy_(teacher.encoder[1].bias[keep])
        compact.encoder[3].weight.copy_(teacher.encoder[3].weight[:, keep])
        compact.encoder[3].bias.copy_(teacher.encoder[3].bias)
        compact.classifier.load_state_dict(teacher.classifier.state_dict())

    return compact.eval()


def source_class_weights(labels: np.ndarray, n_classes: int = 10) -> torch.Tensor:
    """Use the same capped inverse-frequency weights as source training."""
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or not len(y):
        raise ValueError("Expected nonempty one-dimensional source labels")
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError("Source labels must be integer class IDs")
    counts = np.bincount(y, minlength=n_classes)
    weights = np.clip((counts.max() / np.maximum(counts, 1)) ** 0.25, 1, 8)
    return torch.as_tensor(weights, dtype=torch.float32)


def rare_aware_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted focal-style loss used for the source classifier."""
    class_weights = weights.to(device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = F.softmax(logits, dim=1).gather(
        1, labels[:, None]
    ).squeeze(1).clamp(1e-7, 1.0)
    sample_weights = class_weights[labels]
    return (sample_weights * (1 - pt).pow(1.5) * ce).sum() / sample_weights.sum()


@torch.inference_mode()
def teacher_signals(
    teacher: RareAwareNet,
    source_features: np.ndarray,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect teacher logits and 32D features for UNSW training records."""
    x = np.asarray(source_features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != INPUT_FEATURES or not len(x):
        raise ValueError("Expected a nonempty N x 43 feature matrix")
    if not np.isfinite(x).all() or batch_size < 1:
        raise ValueError("Invalid source features or batch size")

    teacher = teacher.cpu().eval()
    logits_parts, embedding_parts = [], []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size]))
        logits, embeddings = teacher(batch)
        logits_parts.append(logits.cpu().numpy())
        embedding_parts.append(embeddings.cpu().numpy())
    return (
        np.vstack(logits_parts).astype(np.float32),
        np.vstack(embedding_parts).astype(np.float32),
    )


@torch.inference_mode()
def source_validation_f1(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int = 10,
    batch_size: int = 2048,
) -> float:
    """Compute macro-F1 across all source categories, including missing ones."""
    model.eval()
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != INPUT_FEATURES or len(x) != len(y) or not len(x):
        raise ValueError("Source validation arrays have inconsistent shapes")
    if not np.isfinite(x).all() or batch_size < 1:
        raise ValueError("Invalid validation features or batch size")

    predictions = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(x[start:start + batch_size]))
        logits, _ = model(batch)
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return float(f1_score(
        y, np.concatenate(predictions), labels=np.arange(n_classes),
        average="macro", zero_division=0,
    ))


def _check_source_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check the feature dimensions and source class IDs."""
    xt = np.asarray(x_train, dtype=np.float32)
    yt = np.asarray(y_train, dtype=np.int64)
    xv = np.asarray(x_val, dtype=np.float32)
    yv = np.asarray(y_val, dtype=np.int64)
    for x, y in ((xt, yt), (xv, yv)):
        if (x.ndim != 2 or x.shape[1] != INPUT_FEATURES or not len(x)
                or y.shape != (len(x),) or not np.isfinite(x).all()):
            raise ValueError("Expected nonempty N x 43 features and aligned labels")
        if np.any((y < 0) | (y >= n_classes)):
            raise ValueError("Unexpected source attack category ID")
    return xt, yt, xv, yv


def _fit_student(
    student: CompactNIDS,
    dataset: TensorDataset,
    x_val: np.ndarray,
    y_val: np.ndarray,
    weights: torch.Tensor,
    method: str,
    epochs: int,
    seed: int,
    lr: float,
    batch_size: int,
    temperature: float = 2.0,
) -> TrainingResult:
    """Train on UNSW and select the best epoch on UNSW validation."""
    if epochs < 1 or batch_size < 1 or lr <= 0 or temperature <= 0:
        raise ValueError("Invalid training configuration")
    if method not in ("pruning", "distillation"):
        raise ValueError("Unknown student training method")

    torch.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        student.train()
        running = []
        for batch in loader:
            logits, embeddings = student(batch[0])
            supervised = rare_aware_loss(logits, batch[1], weights)
            if method == "distillation":
                teacher_logits, teacher_embeddings = batch[2], batch[3]
                soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
                soft_student = F.log_softmax(logits / temperature, dim=1)
                soft_loss = F.kl_div(
                    soft_student, soft_teacher, reduction="batchmean"
                ) * temperature**2
                feature_loss = F.smooth_l1_loss(embeddings, teacher_embeddings)
                loss = 0.40 * supervised + 0.40 * soft_loss + 0.20 * feature_loss
            else:
                loss = supervised

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            optimizer.step()
            running.append(float(loss.item()))

        score = source_validation_f1(
            student, x_val, y_val, n_classes=student.classifier[1].out_features
        )
        history.append({
            "epoch": epoch,
            "training_loss": float(np.mean(running)),
            "source_validation_macro_f1": score,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(student.state_dict())

    student.load_state_dict(best_state)
    return TrainingResult(student.cpu().eval(), best_epoch, best_score, history)


def train_pruned(
    teacher: RareAwareNet,
    width: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int | None = None,
    epochs: int = 5,
    lr: float = 5e-4,
    batch_size: int = 512,
) -> TrainingResult:
    """Fine-tune a 16- or 32-unit pruned model on UNSW labels."""
    xt, yt, xv, yv = _check_source_data(
        x_train, y_train, x_val, y_val, teacher.classifier[1].out_features
    )
    if seed is None:
        seed = 44001 + width
    torch.manual_seed(seed)
    student = prune_encoder(teacher, width)
    dataset = TensorDataset(torch.from_numpy(xt), torch.from_numpy(yt))
    weights = source_class_weights(yt, student.classifier[1].out_features)
    return _fit_student(
        student, dataset, xv, yv, weights, "pruning", epochs, seed, lr, batch_size
    )


def train_distilled(
    teacher: RareAwareNet,
    width: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int | None = None,
    epochs: int = 8,
    lr: float = 5e-4,
    batch_size: int = 512,
    temperature: float = 2.0,
) -> TrainingResult:
    """Train a compact model from source labels and frozen teacher signals."""
    xt, yt, xv, yv = _check_source_data(
        x_train, y_train, x_val, y_val, teacher.classifier[1].out_features
    )
    if seed is None:
        seed = 44031 + width
    teacher_logits, teacher_embeddings = teacher_signals(teacher, xt)
    if teacher_embeddings.shape != (len(xt), EMBEDDING_FEATURES):
        raise ValueError("Teacher embedding size does not match the student")
    torch.manual_seed(seed)
    student = CompactNIDS(width=width, n_classes=teacher.classifier[1].out_features)
    dataset = TensorDataset(
        torch.from_numpy(xt),
        torch.from_numpy(yt),
        torch.from_numpy(teacher_logits),
        torch.from_numpy(teacher_embeddings),
    )
    weights = source_class_weights(yt, student.classifier[1].out_features)
    return _fit_student(
        student, dataset, xv, yv, weights, "distillation", epochs, seed,
        lr, batch_size, temperature=temperature,
    )


def student_checkpoint(
    result: TrainingResult,
    method: str,
    temperature: float = 2.0,
) -> dict:
    """Build a checkpoint compatible with load_compact_model()."""
    if method not in ("structured_pruning", "knowledge_distillation"):
        raise ValueError("Unknown checkpoint method")
    checkpoint = {
        "state_dict": result.model.state_dict(),
        "hidden_width": result.model.encoder[0].out_features,
        "n_classes": result.model.classifier[1].out_features,
        "best_epoch": result.best_epoch,
        "source_validation_macro_f1": result.validation_macro_f1,
        "method": method,
    }
    if method == "knowledge_distillation":
        checkpoint["temperature"] = temperature
        checkpoint["loss_weights"] = {
            "supervised": 0.40,
            "teacher_logits": 0.40,
            "features": 0.20,
        }
    return checkpoint
