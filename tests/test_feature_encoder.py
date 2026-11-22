"""Tests for source and compact neural feature encoders."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_encoder import (
    EMBEDDING_FEATURES,
    INPUT_FEATURES,
    SOURCE_CLASSES,
    CompactNIDS,
    RareAwareNet,
    build_classifier_features,
    encoder_parameters,
    extract_embeddings,
    fit_embedding_scaler,
    load_compact_model,
    load_source_model,
    transform_embeddings,
)


@pytest.fixture
def input_features():
    rng = np.random.default_rng(786)
    return rng.normal(size=(37, INPUT_FEATURES)).astype(np.float32)


@pytest.fixture
def source_model():
    torch.manual_seed(786)
    return RareAwareNet().eval()


@pytest.fixture
def source_embeddings():
    rng = np.random.default_rng(51)
    return rng.normal(size=(60, EMBEDDING_FEATURES)).astype(np.float32)


def test_source_model_dimensions_and_parameter_count(source_model, input_features):
    logits, embeddings = source_model(torch.from_numpy(input_features))
    assert logits.shape == (37, len(SOURCE_CLASSES))
    assert embeddings.shape == (37, EMBEDDING_FEATURES)
    assert encoder_parameters(source_model) == 5024
    torch.testing.assert_close(embeddings, source_model.encoder(torch.from_numpy(input_features)))


@pytest.mark.parametrize("width,expected_parameters", [(32, 2528), (16, 1280)])
def test_compact_model_dimensions_and_parameter_count(width, expected_parameters, input_features):
    model = CompactNIDS(width=width).eval()
    logits, embeddings = model(torch.from_numpy(input_features))
    assert logits.shape == (37, len(SOURCE_CLASSES))
    assert embeddings.shape == (37, EMBEDDING_FEATURES)
    assert encoder_parameters(model) == expected_parameters


def test_compact_model_rejects_unsupported_width():
    with pytest.raises(ValueError, match="16 or 32"):
        CompactNIDS(width=8)


def test_embedding_extraction_matches_direct_encoder(source_model, input_features):
    expected = source_model.encoder(torch.from_numpy(input_features)).detach().numpy()
    actual = extract_embeddings(source_model, input_features, batch_size=7)
    assert actual.shape == (37, EMBEDDING_FEATURES)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        extract_embeddings(source_model, input_features, batch_size=37),
        actual,
        rtol=1e-6,
        atol=1e-6,
    )


def test_embedding_extraction_works_for_compact_encoder(input_features):
    torch.manual_seed(73)
    model = CompactNIDS(width=16)
    model.train()
    result = extract_embeddings(model, input_features, batch_size=10)
    assert result.shape == (37, EMBEDDING_FEATURES)
    assert np.isfinite(result).all()
    assert not model.training


def test_embedding_extraction_empty_input(source_model):
    empty = extract_embeddings(source_model, np.empty((0, INPUT_FEATURES)))
    assert empty.shape == (0, EMBEDDING_FEATURES)
    assert empty.dtype == np.float32


@pytest.mark.parametrize(
    "features,batch_size,error",
    [
        (np.zeros((5, 42), dtype=np.float32), 4, "N x 43"),
        (np.zeros((43,), dtype=np.float32), 4, "N x 43"),
        (np.full((5, 43), np.nan, dtype=np.float32), 4, "missing or infinite"),
        (np.full((5, 43), np.inf, dtype=np.float32), 4, "missing or infinite"),
        (np.zeros((5, 43), dtype=np.float32), 0, "positive"),
    ],
)
def test_embedding_extraction_rejects_invalid_inputs(source_model, features, batch_size, error):
    with pytest.raises(ValueError, match=error):
        extract_embeddings(source_model, features, batch_size=batch_size)


def test_source_checkpoint_round_trip(tmp_path, source_model, input_features):
    names = [f"feature_{index:02d}" for index in range(INPUT_FEATURES)]
    checkpoint_path = tmp_path / "source_encoder_checkpoint.pt"
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "input_features": names,
            "class_names": list(SOURCE_CLASSES),
            "best_epoch": 3,
        },
        checkpoint_path,
    )
    restored = load_source_model(checkpoint_path, names)
    assert isinstance(restored, RareAwareNet)
    assert not restored.training
    np.testing.assert_allclose(
        extract_embeddings(restored, input_features),
        extract_embeddings(source_model, input_features),
        rtol=0,
        atol=0,
    )


def test_source_checkpoint_rejects_feature_or_class_mismatch(tmp_path, source_model):
    names = [f"feature_{index:02d}" for index in range(INPUT_FEATURES)]
    checkpoint = {
        "state_dict": source_model.state_dict(),
        "input_features": names,
        "class_names": list(SOURCE_CLASSES),
    }
    path = tmp_path / "source.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="feature order"):
        load_source_model(path, names[::-1])

    checkpoint["class_names"] = list(SOURCE_CLASSES[::-1])
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Source attack categories"):
        load_source_model(path, names)


def test_source_checkpoint_rejects_invalid_feature_count(tmp_path, source_model):
    names = [f"feature_{index:02d}" for index in range(INPUT_FEATURES - 1)]
    path = tmp_path / "source.pt"
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "input_features": names,
            "class_names": list(SOURCE_CLASSES),
        },
        path,
    )
    with pytest.raises(ValueError, match="43 input features"):
        load_source_model(path, names)


@pytest.mark.parametrize("width", [16, 32])
def test_compact_checkpoint_round_trip(tmp_path, width, input_features):
    torch.manual_seed(width)
    model = CompactNIDS(width=width).eval()
    path = tmp_path / f"pruned_{width}_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_width": width,
            "n_classes": len(SOURCE_CLASSES),
            "method": "structured_pruning",
        },
        path,
    )
    restored = load_compact_model(path)
    assert isinstance(restored, CompactNIDS)
    assert restored.encoder[0].out_features == width
    assert not restored.training
    np.testing.assert_allclose(
        extract_embeddings(restored, input_features),
        extract_embeddings(model, input_features),
        rtol=0,
        atol=0,
    )


def test_compact_checkpoint_rejects_wrong_class_count(tmp_path):
    model = CompactNIDS(width=16)
    path = tmp_path / "compact.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_width": 16,
            "n_classes": len(SOURCE_CLASSES) - 1,
        },
        path,
    )
    with pytest.raises(ValueError, match="source attack classes"):
        load_compact_model(path)


def test_embedding_scaler_matches_reference(source_embeddings):
    from sklearn.preprocessing import RobustScaler

    scaler = fit_embedding_scaler(source_embeddings)
    reference = RobustScaler(quantile_range=(10, 90)).fit(np.arcsinh(source_embeddings))
    np.testing.assert_allclose(scaler.center_, reference.center_)
    np.testing.assert_allclose(scaler.scale_, reference.scale_)
    actual = transform_embeddings(source_embeddings, scaler)
    expected = np.clip(reference.transform(np.arcsinh(source_embeddings)), -8, 8)
    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert actual.dtype == np.float32
    assert np.max(np.abs(actual)) <= 8


def test_target_embedding_transform_does_not_refit_source_scaler(source_embeddings):
    scaler = fit_embedding_scaler(source_embeddings[:40])
    source_center = scaler.center_.copy()
    source_scale = scaler.scale_.copy()
    target = source_embeddings[40:].copy() * 100_000
    target[0, 0] = 1e12
    target[0, 1] = -1e12
    transformed = transform_embeddings(target, scaler)
    assert transformed.shape == (20, EMBEDDING_FEATURES)
    assert np.max(np.abs(transformed)) <= 8
    np.testing.assert_array_equal(scaler.center_, source_center)
    np.testing.assert_array_equal(scaler.scale_, source_scale)


def test_embedding_scaler_rejects_invalid_inputs(source_embeddings):
    with pytest.raises(ValueError, match="nonempty"):
        fit_embedding_scaler(np.empty((0, EMBEDDING_FEATURES)))
    with pytest.raises(ValueError, match="32 columns"):
        fit_embedding_scaler(source_embeddings[:, :-1])
    corrupt = source_embeddings.copy()
    corrupt[2, 4] = np.inf
    with pytest.raises(ValueError, match="missing or infinite"):
        fit_embedding_scaler(corrupt)
    with pytest.raises(ValueError, match="N x 32"):
        transform_embeddings(corrupt[:, :-1], fit_embedding_scaler(source_embeddings))


def test_classifier_features_have_75_columns_in_correct_order(input_features, source_embeddings):
    scaler = fit_embedding_scaler(source_embeddings)
    embeddings = source_embeddings[:len(input_features)]
    combined = build_classifier_features(input_features, embeddings, scaler)
    expected_embeddings = transform_embeddings(embeddings, scaler)
    assert combined.shape == (37, 75)
    assert combined.dtype == np.float32
    np.testing.assert_array_equal(combined[:, :INPUT_FEATURES], input_features)
    np.testing.assert_array_equal(combined[:, INPUT_FEATURES:], expected_embeddings)


def test_classifier_features_support_empty_batches(source_embeddings):
    scaler = fit_embedding_scaler(source_embeddings)
    combined = build_classifier_features(
        np.empty((0, INPUT_FEATURES)),
        np.empty((0, EMBEDDING_FEATURES)),
        scaler,
    )
    assert combined.shape == (0, 75)
    assert combined.dtype == np.float32


def test_classifier_features_reject_mismatched_and_nonfinite_inputs(input_features, source_embeddings):
    scaler = fit_embedding_scaler(source_embeddings)
    with pytest.raises(ValueError, match="different row counts"):
        build_classifier_features(input_features, source_embeddings, scaler)
    with pytest.raises(ValueError, match="N x 43"):
        build_classifier_features(input_features[:, :-1], source_embeddings[:37], scaler)
    corrupt = input_features.copy()
    corrupt[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        build_classifier_features(corrupt, source_embeddings[:37], scaler)
    corrupt_embeddings = source_embeddings[:37].copy()
    corrupt_embeddings[0, 0] = np.inf
    with pytest.raises(ValueError, match="missing or infinite"):
        build_classifier_features(input_features, corrupt_embeddings, scaler)
