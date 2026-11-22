"""Tests for structured pruning, knowledge distillation, and INT8 inference."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compression import (
    TrainingResult,
    dynamic_int8,
    prune_encoder,
    rare_aware_loss,
    source_class_weights,
    source_validation_f1,
    student_checkpoint,
    teacher_signals,
    train_distilled,
    train_pruned,
)
from src.feature_encoder import (
    CompactNIDS,
    EMBEDDING_FEATURES,
    INPUT_FEATURES,
    RareAwareNet,
    encoder_parameters,
    load_compact_model,
)


@pytest.fixture
def teacher():
    torch.manual_seed(786)
    return RareAwareNet().eval()


@pytest.fixture
def source_arrays():
    rng = np.random.default_rng(786)
    x_train = rng.normal(size=(40, INPUT_FEATURES)).astype(np.float32)
    x_val = rng.normal(size=(20, INPUT_FEATURES)).astype(np.float32)
    y_train = np.tile(np.arange(10), 4).astype(np.int64)
    y_val = np.tile(np.arange(10), 2).astype(np.int64)
    return x_train, y_train, x_val, y_val


def test_teacher_and_compact_parameter_counts(teacher):
    assert encoder_parameters(teacher) == 5024
    assert encoder_parameters(CompactNIDS(width=32)) == 2528
    assert encoder_parameters(CompactNIDS(width=16)) == 1280


@pytest.mark.parametrize("width", [16, 32])
def test_pruning_retains_top_outgoing_weight_norms(teacher, width):
    with torch.no_grad():
        # Produce a known ordering rather than relying on random weights.
        teacher.encoder[3].weight.zero_()
        teacher.encoder[3].weight[0, :] = torch.arange(1, 65, dtype=torch.float32)
    student = prune_encoder(teacher, width)
    keep = torch.arange(64 - width, 64)

    torch.testing.assert_close(student.encoder[0].weight, teacher.encoder[0].weight[keep])
    torch.testing.assert_close(student.encoder[0].bias, teacher.encoder[0].bias[keep])
    torch.testing.assert_close(student.encoder[1].weight, teacher.encoder[1].weight[keep])
    torch.testing.assert_close(student.encoder[1].bias, teacher.encoder[1].bias[keep])
    torch.testing.assert_close(student.encoder[3].weight, teacher.encoder[3].weight[:, keep])
    torch.testing.assert_close(student.encoder[3].bias, teacher.encoder[3].bias)


@pytest.mark.parametrize("width", [16, 32])
def test_pruned_model_preserves_classifier_and_output_shapes(teacher, width):
    student = prune_encoder(teacher, width)
    for name, value in teacher.classifier.state_dict().items():
        torch.testing.assert_close(student.classifier.state_dict()[name], value)
    x = torch.randn(7, INPUT_FEATURES)
    logits, embedding = student(x)
    assert logits.shape == (7, 10)
    assert embedding.shape == (7, EMBEDDING_FEATURES)
    assert not student.training


@pytest.mark.parametrize("width", [8, 24, 64])
def test_pruning_rejects_unsupported_width(teacher, width):
    with pytest.raises(ValueError, match="16 or 32"):
        prune_encoder(teacher, width)


def test_pruning_does_not_modify_teacher_parameters(teacher):
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    prune_encoder(teacher, 16)
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_dynamic_int8_is_a_separate_cpu_model(teacher):
    before = {name: value.clone() for name, value in teacher.state_dict().items()}
    quantized = dynamic_int8(teacher)
    assert quantized is not teacher
    assert not quantized.training
    assert any(isinstance(layer, torch.ao.nn.quantized.dynamic.Linear) for layer in quantized.modules())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_dynamic_int8_produces_finite_outputs(teacher):
    quantized = dynamic_int8(teacher)
    x = torch.randn(11, INPUT_FEATURES)
    with torch.inference_mode():
        logits, z = quantized(x)
    assert logits.shape == (11, 10)
    assert z.shape == (11, EMBEDDING_FEATURES)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(z).all()


def test_dynamic_int8_predictions_are_close_to_fp32(teacher):
    x = torch.randn(20, INPUT_FEATURES)
    quantized = dynamic_int8(teacher)
    with torch.inference_mode():
        original, _ = teacher(x)
        compressed, _ = quantized(x)
    # Dynamic quantization is approximate, not bitwise identical to FP32.
    torch.testing.assert_close(compressed, original, rtol=0.08, atol=0.08)


def test_class_weights_match_capped_inverse_frequency_rule():
    labels = np.array([0] * 16 + [1] * 4 + [2], dtype=np.int64)
    weights = source_class_weights(labels, n_classes=4)
    expected = np.clip(
        (16 / np.maximum(np.array([16, 4, 1, 0]), 1)) ** 0.25, 1, 8
    )
    np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-6)
    assert weights.dtype == torch.float32


@pytest.mark.parametrize("labels", [[], [-1, 0], [0, 10], [[0, 1]]])
def test_class_weights_reject_invalid_labels(labels):
    with pytest.raises(ValueError):
        source_class_weights(np.asarray(labels), n_classes=10)


def test_rare_aware_loss_matches_formula_and_supports_gradients():
    logits = torch.tensor([[1.4, -0.5], [-0.2, 0.8]], requires_grad=True)
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 2.0])
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = F.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    expected = (weights[labels] * (1 - pt).pow(1.5) * ce).sum() / weights[labels].sum()
    actual = rare_aware_loss(logits, labels, weights)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_teacher_signals_match_forward_pass_with_batches(teacher, source_arrays):
    x_train = source_arrays[0]
    logits, z = teacher_signals(teacher, x_train, batch_size=7)
    with torch.inference_mode():
        expected_logits, expected_z = teacher(torch.from_numpy(x_train))
    assert logits.dtype == np.float32
    assert z.dtype == np.float32
    assert logits.shape == (len(x_train), 10)
    assert z.shape == (len(x_train), EMBEDDING_FEATURES)
    np.testing.assert_allclose(logits, expected_logits.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(z, expected_z.numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "bad_x,batch_size",
    [
        (np.empty((0, INPUT_FEATURES)), 4),
        (np.zeros((4, INPUT_FEATURES - 1)), 4),
        (np.ones((4, INPUT_FEATURES)) * np.nan, 4),
        (np.ones((4, INPUT_FEATURES)) * np.inf, 4),
        (np.zeros((4, INPUT_FEATURES)), 0),
    ],
)
def test_teacher_signals_rejects_bad_input(teacher, bad_x, batch_size):
    with pytest.raises(ValueError):
        teacher_signals(teacher, bad_x, batch_size=batch_size)


def test_source_validation_includes_missing_categories(teacher):
    # A constant class-0 predictor achieves F1=1 for class 0 but zero
    # for each of the other nine source classes.
    class AlwaysBenign(nn.Module):
        def forward(self, x):
            scores = torch.zeros((len(x), 10))
            scores[:, 0] = 1
            return scores, torch.zeros((len(x), EMBEDDING_FEATURES))

    model = AlwaysBenign()
    x = np.zeros((12, INPUT_FEATURES), dtype=np.float32)
    labels = np.zeros(12, dtype=np.int64)
    assert source_validation_f1(model, x, labels) == pytest.approx(0.1)


@pytest.mark.parametrize("method,width", [("pruning", 16), ("pruning", 32), ("distillation", 16), ("distillation", 32)])
def test_one_epoch_student_training(teacher, source_arrays, method, width):
    args = (teacher, width, *source_arrays)
    if method == "pruning":
        result = train_pruned(*args, epochs=1, seed=73, batch_size=10)
    else:
        result = train_distilled(*args, epochs=1, seed=73, batch_size=10)

    assert isinstance(result, TrainingResult)
    assert isinstance(result.model, CompactNIDS)
    assert result.best_epoch == 1
    assert len(result.history) == 1
    assert not result.model.training
    assert 0 <= result.validation_macro_f1 <= 1
    assert result.history[0]["source_validation_macro_f1"] == result.validation_macro_f1
    assert np.isfinite(result.history[0]["training_loss"])
    assert encoder_parameters(result.model) == {16: 1280, 32: 2528}[width]


@pytest.mark.parametrize("method", ["pruning", "distillation"])
def test_training_selects_valid_checkpoint_over_two_epochs(teacher, source_arrays, method):
    args = (teacher, 16, *source_arrays)
    fn = train_pruned if method == "pruning" else train_distilled
    result = fn(*args, epochs=2, seed=51, batch_size=20)
    assert result.best_epoch in (1, 2)
    assert result.validation_macro_f1 == max(
        row["source_validation_macro_f1"] for row in result.history
    )
    final_score = source_validation_f1(result.model, source_arrays[2], source_arrays[3])
    assert final_score == pytest.approx(result.validation_macro_f1)


@pytest.mark.parametrize("method", ["pruning", "distillation"])
def test_training_is_reproducible_with_fixed_seed(teacher, source_arrays, method):
    fn = train_pruned if method == "pruning" else train_distilled
    args = (teacher, 16, *source_arrays)
    a = fn(*args, epochs=1, seed=97, batch_size=10)
    b = fn(*args, epochs=1, seed=97, batch_size=10)
    assert a.history == b.history
    for name, tensor in a.model.state_dict().items():
        torch.testing.assert_close(tensor, b.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "bad_config",
    [{"epochs": 0}, {"batch_size": 0}, {"lr": 0}, {"lr": -0.1}],
)
def test_pruning_rejects_invalid_training_settings(teacher, source_arrays, bad_config):
    with pytest.raises(ValueError, match="Invalid training configuration"):
        train_pruned(teacher, 16, *source_arrays, **bad_config)


def test_distillation_rejects_invalid_temperature(teacher, source_arrays):
    with pytest.raises(ValueError, match="Invalid training configuration"):
        train_distilled(teacher, 16, *source_arrays, temperature=0)


@pytest.mark.parametrize("invalid", ["features", "labels", "classes"])
def test_student_training_rejects_invalid_source_data(teacher, source_arrays, invalid):
    x_train, y_train, x_val, y_val = source_arrays
    if invalid == "features":
        x_train = x_train[:, :-1]
    elif invalid == "labels":
        y_train = y_train[:-1]
    else:
        y_train = y_train.copy()
        y_train[0] = 12
    with pytest.raises(ValueError):
        train_pruned(teacher, 16, x_train, y_train, x_val, y_val, epochs=1)


@pytest.mark.parametrize("method", ["structured_pruning", "knowledge_distillation"])
def test_student_checkpoint_round_trip(teacher, source_arrays, method, tmp_path):
    train = train_pruned if method == "structured_pruning" else train_distilled
    result = train(teacher, 16, *source_arrays, epochs=1, seed=51, batch_size=10)
    checkpoint = student_checkpoint(result, method)
    assert checkpoint["method"] == method
    assert checkpoint["hidden_width"] == 16
    assert checkpoint["n_classes"] == 10
    assert checkpoint["best_epoch"] == result.best_epoch
    if method == "knowledge_distillation":
        assert checkpoint["temperature"] == 2.0
        assert checkpoint["loss_weights"] == {
            "supervised": 0.40, "teacher_logits": 0.40, "features": 0.20
        }
    else:
        assert "temperature" not in checkpoint

    path = tmp_path / "student.pt"
    torch.save(checkpoint, path)
    reloaded = load_compact_model(path)
    sample = torch.from_numpy(source_arrays[2][:5])
    with torch.inference_mode():
        expected, z_expected = result.model(sample)
        actual, z_actual = reloaded(sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(z_actual, z_expected, rtol=0, atol=0)


def test_checkpoint_rejects_unknown_method(teacher):
    result = TrainingResult(
        model=prune_encoder(teacher, 16),
        best_epoch=1,
        validation_macro_f1=0.25,
        history=[],
    )
    with pytest.raises(ValueError, match="Unknown checkpoint method"):
        student_checkpoint(result, "unknown")
