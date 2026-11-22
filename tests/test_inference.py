"""Tests for the adapted NIDS inference pipeline and CSV interface."""

import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_encoder import (
    SOURCE_CLASSES,
    CompactNIDS,
    RareAwareNet,
    build_classifier_features,
    extract_embeddings,
    fit_embedding_scaler,
)
from src.inference import (
    MODEL_VARIANTS,
    InferenceSummary,
    _source_checkpoint,
    load_inference_model,
    main,
    predict_csv,
    predict_records,
)
from src.preprocessing import (
    ENGINEERED_FEATURES,
    FLOW_COLUMNS,
    REMOVED_FEATURES,
    SPECIAL_FEATURE,
    add_flow_features,
    fit_source_scaler,
    transform_features,
)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """Build fitted, compatible model artifacts without using project datasets."""
    root = tmp_path_factory.mktemp("nids_inference")
    data_dir = root / "data"
    source_dir = data_dir / "reproduction_runs" / "source_adaptation"
    compression_dir = data_dir / "reproduction_runs" / "model_compression"
    source_dir.mkdir(parents=True)
    compression_dir.mkdir(parents=True)

    raw = list(FLOW_COLUMNS) + [SPECIAL_FEATURE] + list(REMOVED_FEATURES)
    raw += [f"MEASURE_{i}" for i in range(31)]
    names = [column for column in raw if column not in REMOVED_FEATURES]
    names += list(ENGINEERED_FEATURES)
    assert len(raw) == 41 and len(names) == 43
    schema = {
        "raw_features": raw,
        "retained_raw_features": [c for c in raw if c not in REMOVED_FEATURES],
        "engineered_features": list(ENGINEERED_FEATURES),
        "final_features_in_order": names,
    }
    (data_dir / "csc786_feature_schema.json").write_text(
        json.dumps(schema), encoding="utf-8"
    )

    rng = np.random.default_rng(786)
    records = pd.DataFrame(rng.gamma(2.0, 8.0, (170, len(raw))), columns=raw)
    processed = add_flow_features(records.iloc[:130])
    source_scaler = fit_source_scaler(processed, names)
    x43 = transform_features(processed, names, source_scaler)
    joblib.dump(source_scaler, source_dir / "source_standard_scaler.joblib")

    torch.manual_seed(786)
    teacher = RareAwareNet().eval()
    teacher_checkpoint = data_dir / "cell_44C_multiclass_source.pt"
    torch.save(
        {
            "state_dict": teacher.state_dict(),
            "input_features": names,
            "class_names": list(SOURCE_CLASSES),
        },
        teacher_checkpoint,
    )
    z32 = extract_embeddings(teacher, x43)
    embedding_scaler = fit_embedding_scaler(z32)
    x75 = build_classifier_features(x43, z32, embedding_scaler)
    labels = np.where(x43[:, 0] > np.median(x43[:, 0]), "Bot", "Benign")
    classifier = LogisticRegression(max_iter=600, random_state=13).fit(x75, labels)
    joblib.dump(embedding_scaler, source_dir / "source_embedding_scaler.joblib")
    joblib.dump(classifier, source_dir / "adapted_classifier_seed51.joblib")

    adjusted_classifier = copy.deepcopy(classifier)
    adjusted_classifier.intercept_ = adjusted_classifier.intercept_ + 0.5
    joblib.dump(embedding_scaler, compression_dir / "teacher_fp32_embedding_scaler.joblib")
    joblib.dump(adjusted_classifier, compression_dir / "teacher_fp32_classifier.joblib")

    for variant in ("pruned_32", "pruned_16", "distilled_32", "distilled_16"):
        width = int(variant.rsplit("_", 1)[-1])
        model = CompactNIDS(width=width).eval()
        z32 = extract_embeddings(model, x43)
        scaler = fit_embedding_scaler(z32)
        joblib.dump(scaler, compression_dir / f"{variant}_embedding_scaler.joblib")
        joblib.dump(classifier, compression_dir / f"{variant}_classifier.joblib")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden_width": width,
                "n_classes": len(SOURCE_CLASSES),
            },
            compression_dir / f"{variant}_checkpoint.pt",
        )

    csv_path = root / "flows.csv"
    records.to_csv(csv_path, index=False)
    return {
        "root": root,
        "data_dir": data_dir,
        "source_dir": source_dir,
        "compression_dir": compression_dir,
        "records": records,
        "csv_path": csv_path,
        "source_checkpoint": teacher_checkpoint,
    }


@pytest.fixture(scope="module")
def teacher(case):
    return load_inference_model(case["data_dir"], "teacher", batch_size=19)


@pytest.mark.parametrize("variant", MODEL_VARIANTS)
def test_load_each_model_variant(case, variant):
    model = load_inference_model(case["data_dir"], variant)
    sample = model.predict_flows(case["records"].iloc[:7])
    assert sample.shape == (7, 3)
    assert np.isfinite(sample["Attack_Score"]).all()
    assert model.encoder.training is False


def test_teacher_and_compression_teacher_use_matching_classifier(case):
    teacher_model = load_inference_model(case["data_dir"], "teacher")
    compression_model = load_inference_model(case["data_dir"], "teacher_fp32")
    x43 = np.zeros((4, 43), dtype=np.float32)
    p_teacher = teacher_model.predict_proba(x43)
    p_compression = compression_model.predict_proba(x43)
    assert not np.allclose(p_teacher, p_compression)


def test_teacher_int8_quantizes_linear_layers(case):
    model = load_inference_model(case["data_dir"], "teacher_int8")
    dynamic_linear = torch.ao.nn.quantized.dynamic.Linear
    assert any(isinstance(layer, dynamic_linear) for layer in model.encoder.modules())
    assert model.encoder.training is False


def test_model_prediction_columns_and_original_row_order(teacher, case):
    result = predict_records(teacher, case["records"].iloc[:23])
    assert result.columns.tolist() == [
        "Row_Index", "Predicted_Category", "Category_Confidence", "Attack_Score"
    ]
    assert result["Row_Index"].tolist() == list(range(23))
    assert set(result["Predicted_Category"]).issubset(set(teacher.classes_))
    assert result["Attack_Score"].between(0, 1).all()
    assert result["Category_Confidence"].between(0, 1).all()


def test_attack_score_uses_benign_class_probability(teacher, case):
    records = case["records"].iloc[:11]
    x43 = transform_features(add_flow_features(records), list(teacher.feature_names), teacher.source_scaler)
    probabilities = teacher.predict_proba(x43)
    result = predict_records(teacher, records)
    benign = int(np.flatnonzero(teacher.classes_ == "Benign")[0])
    np.testing.assert_allclose(result["Attack_Score"], 1 - probabilities[:, benign])


@pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 1.0])
def test_threshold_creates_expected_alerts(teacher, case, threshold):
    result = predict_records(teacher, case["records"].iloc[:20], attack_threshold=threshold)
    assert result["Attack_Alert"].dtype == bool
    assert result["Attack_Alert"].tolist() == (result["Attack_Score"] >= threshold).tolist()


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(teacher, case, threshold):
    with pytest.raises(ValueError, match="threshold"):
        predict_records(teacher, case["records"].iloc[:3], attack_threshold=threshold)


def test_threshold_does_not_change_category_predictions(teacher, case):
    frame = case["records"].iloc[:18]
    no_threshold = predict_records(teacher, frame)
    with_threshold = predict_records(teacher, frame, attack_threshold=0.2)
    assert "Attack_Alert" not in no_threshold
    assert no_threshold["Predicted_Category"].tolist() == with_threshold["Predicted_Category"].tolist()
    np.testing.assert_allclose(no_threshold["Attack_Score"], with_threshold["Attack_Score"])


def test_prediction_does_not_modify_input(teacher, case):
    records = case["records"].iloc[:8].copy()
    original = records.copy(deep=True)
    predict_records(teacher, records)
    pd.testing.assert_frame_equal(records, original)


def test_prediction_does_not_require_labels(teacher, case):
    frame = case["records"].iloc[:12].copy()
    with_labels = frame.assign(Label=1, Attack="Bot")
    expected = predict_records(teacher, frame)
    actual = predict_records(teacher, with_labels)
    pd.testing.assert_frame_equal(expected, actual)


def test_empty_input_returns_headers_and_zero_rows(teacher, case):
    empty = case["records"].iloc[0:0]
    result = predict_records(teacher, empty, attack_threshold=0.6)
    assert result.empty
    assert result.columns.tolist() == [
        "Row_Index", "Predicted_Category", "Category_Confidence", "Attack_Score", "Attack_Alert"
    ]


def test_csv_matches_direct_predictions_across_chunks(teacher, case, tmp_path):
    output = tmp_path / "nested" / "predictions.csv"
    summary = predict_csv(teacher, case["csv_path"], output, chunksize=13, attack_threshold=0.4)
    actual = pd.read_csv(output)
    expected = predict_records(teacher, case["records"], attack_threshold=0.4)
    assert summary.rows == len(case["records"])
    assert summary.output_path == output
    assert actual["Row_Index"].tolist() == list(range(len(actual)))
    assert actual["Predicted_Category"].tolist() == expected["Predicted_Category"].tolist()
    assert actual["Attack_Alert"].tolist() == expected["Attack_Alert"].tolist()
    np.testing.assert_allclose(actual["Attack_Score"], expected["Attack_Score"], atol=1e-7)
    np.testing.assert_allclose(actual["Category_Confidence"], expected["Category_Confidence"], atol=1e-7)


@pytest.mark.parametrize("chunksize", [1, 17, 1000])
def test_csv_chunk_sizes_produce_same_fp32_results(teacher, case, tmp_path, chunksize):
    output = tmp_path / f"pred_{chunksize}.csv"
    predict_csv(teacher, case["csv_path"], output, chunksize=chunksize)
    actual = pd.read_csv(output)
    expected = predict_records(teacher, case["records"])
    assert "Attack_Alert" not in actual
    assert actual["Predicted_Category"].tolist() == expected["Predicted_Category"].tolist()
    np.testing.assert_allclose(actual["Attack_Score"], expected["Attack_Score"], atol=1e-6)


def test_empty_csv_produces_empty_predictions(teacher, case, tmp_path):
    source = tmp_path / "empty.csv"
    case["records"].iloc[:0].to_csv(source, index=False)
    output = tmp_path / "empty_predictions.csv"
    result = predict_csv(teacher, source, output)
    assert result.rows == 0
    assert pd.read_csv(output).empty
    assert pd.read_csv(output).columns.tolist() == [
        "Row_Index", "Predicted_Category", "Category_Confidence", "Attack_Score"
    ]


def test_csv_does_not_overwrite_input(teacher, case):
    with pytest.raises(ValueError, match="different"):
        predict_csv(teacher, case["csv_path"], case["csv_path"])


@pytest.mark.parametrize("chunksize", [0, -5])
def test_invalid_csv_chunk_size(teacher, case, tmp_path, chunksize):
    output = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="positive"):
        predict_csv(teacher, case["csv_path"], output, chunksize=chunksize)
    assert not output.exists()


def test_missing_csv_input(teacher, tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_csv(teacher, tmp_path / "missing.csv", tmp_path / "output.csv")


def test_invalid_flow_data_does_not_leave_partial_csv(teacher, case, tmp_path):
    invalid = case["records"].drop(columns=["FLOW_DURATION_MILLISECONDS"])
    source = tmp_path / "invalid.csv"
    output = tmp_path / "output.csv"
    invalid.to_csv(source, index=False)
    with pytest.raises(ValueError, match="Missing flow columns"):
        predict_csv(teacher, source, output, chunksize=7)
    assert not output.exists()
    assert not list(tmp_path.glob(".nids_*.csv"))


def test_existing_output_is_preserved_if_new_run_fails(teacher, case, tmp_path):
    invalid = case["records"].copy()
    invalid.loc[31, "IN_BYTES"] = np.nan
    source = tmp_path / "invalid_in_second_chunk.csv"
    output = tmp_path / "results.csv"
    invalid.to_csv(source, index=False)
    output.write_text("previous verified result\n", encoding="utf-8")
    with pytest.raises(ValueError):
        predict_csv(teacher, source, output, chunksize=25)
    assert output.read_text(encoding="utf-8") == "previous verified result\n"
    assert not list(tmp_path.glob(".nids_*.csv"))


def test_inference_summary_throughput(tmp_path):
    summary = InferenceSummary(120, 4.0, tmp_path / "predictions.csv")
    assert summary.rows_per_second == pytest.approx(30.0)
    assert InferenceSummary(0, 0.0, tmp_path / "empty.csv").rows_per_second == 0.0


@pytest.mark.parametrize("variant", ["unknown", "pruned_8", "TEACHER"])
def test_unknown_model_variant(case, variant):
    with pytest.raises(ValueError, match="Unknown model"):
        load_inference_model(case["data_dir"], model=variant)


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_inference_batch_size(case, size):
    with pytest.raises(ValueError, match="positive"):
        load_inference_model(case["data_dir"], batch_size=size)


def test_int8_rejects_non_cpu_device(case):
    with pytest.raises(ValueError, match="CPU only"):
        load_inference_model(case["data_dir"], "teacher_int8", device="cuda")


def test_missing_model_artifacts_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inference_model(tmp_path, "teacher")


def test_missing_source_checkpoint_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        _source_checkpoint(tmp_path)


@pytest.mark.parametrize("name", ["../checkpoint.pt", "/tmp/checkpoint.pt", "subdir/model.pt"])
def test_run_configuration_rejects_unsafe_checkpoint_path(tmp_path, name):
    config_dir = tmp_path / "reproduction_runs" / "source_adaptation"
    config_dir.mkdir(parents=True)
    (config_dir / "run_config.json").write_text(json.dumps({"source_checkpoint": name}))
    with pytest.raises(ValueError, match="Invalid source checkpoint"):
        _source_checkpoint(tmp_path)


def test_run_configuration_selects_named_source_checkpoint(case, tmp_path):
    source = case["source_checkpoint"]
    checkpoint = tmp_path / "custom_source.pt"
    checkpoint.write_bytes(source.read_bytes())
    config_dir = tmp_path / "reproduction_runs" / "source_adaptation"
    config_dir.mkdir(parents=True)
    (config_dir / "run_config.json").write_text(
        json.dumps({"source_checkpoint": checkpoint.name})
    )
    assert _source_checkpoint(tmp_path) == checkpoint


def test_main_runs_csv_inference(case, tmp_path, capsys):
    output = tmp_path / "cli_output.csv"
    result = main([
        "--data-dir", str(case["data_dir"]),
        "--input", str(case["csv_path"]),
        "--output", str(output),
        "--model", "teacher", "--chunksize", "23", "--threshold", "0.4",
    ])
    assert result == 0
    assert len(pd.read_csv(output)) == len(case["records"])
    printed = capsys.readouterr().out
    assert "Records:" in printed
    assert "Throughput:" in printed


def test_csv_does_not_change_source_scaler(teacher, case, tmp_path):
    before = teacher.source_scaler.mean_.copy()
    destination = tmp_path / "predictions.csv"
    predict_csv(teacher, case["csv_path"], destination, chunksize=29)
    np.testing.assert_array_equal(teacher.source_scaler.mean_, before)
