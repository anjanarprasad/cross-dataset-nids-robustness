"""Batch inference for the adapted network intrusion detection models."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .classification import AdaptedNIDS, load_detector


MODEL_VARIANTS = (
    "teacher",
    "teacher_fp32",
    "teacher_int8",
    "pruned_32",
    "pruned_16",
    "distilled_32",
    "distilled_16",
)


@dataclass(frozen=True)
class InferenceSummary:
    """Processing statistics for one CSV inference run."""

    rows: int
    elapsed_seconds: float
    output_path: Path

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


def _source_checkpoint(data_dir: Path) -> Path:
    """Locate the full source model checkpoint saved during training."""
    source_dir = data_dir / "reproduction_runs" / "source_adaptation"
    config_path = source_dir / "run_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as stream:
            checkpoint_name = json.load(stream)["source_checkpoint"]
        if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
            raise ValueError("Invalid source checkpoint filename in run configuration")
        choices = [data_dir / checkpoint_name, source_dir / checkpoint_name]
    else:
        choices = [
            data_dir / "cell_44C_multiclass_source.pt",
            source_dir / "source_encoder_checkpoint.pt",
        ]
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Source model checkpoint not found: "
        + ", ".join(str(path) for path in choices)
    )


def _required_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required model artifact not found: {path}")
    return path


def load_inference_model(
    data_dir: str | Path,
    model: str = "teacher",
    *,
    device: str = "cpu",
    batch_size: int = 2048,
) -> AdaptedNIDS:
    """Load the encoder, source scaler, and matching CIC classifier.

    The teacher model uses the classifier saved after target adaptation.
    Compression variants use the classifiers fitted for their own features.
    INT8 quantization uses the compression teacher's fitted classifier.
    """
    if model not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_VARIANTS}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if model == "teacher_int8" and device != "cpu":
        raise ValueError("Dynamic INT8 inference is supported on CPU only")

    root = Path(data_dir).expanduser()
    source_dir = root / "reproduction_runs" / "source_adaptation"
    compression_dir = root / "reproduction_runs" / "model_compression"
    source_scaler = source_dir / "source_standard_scaler.joblib"
    feature_schema = root / "csc786_feature_schema.json"

    if model == "teacher":
        embedding_scaler = source_dir / "source_embedding_scaler.joblib"
        classifier = source_dir / "adapted_classifier_seed51.joblib"
        checkpoint = _source_checkpoint(root)
        kind = "teacher"
    elif model in {"teacher_fp32", "teacher_int8"}:
        embedding_scaler = compression_dir / "teacher_fp32_embedding_scaler.joblib"
        classifier = compression_dir / "teacher_fp32_classifier.joblib"
        checkpoint = _source_checkpoint(root)
        kind = "teacher"
    else:
        embedding_scaler = compression_dir / f"{model}_embedding_scaler.joblib"
        classifier = compression_dir / f"{model}_classifier.joblib"
        checkpoint = compression_dir / f"{model}_checkpoint.pt"
        kind = "compact"

    paths = [feature_schema, source_scaler, embedding_scaler, classifier, checkpoint]
    for path in paths:
        _required_file(path)

    predictor = load_detector(
        feature_schema_path=feature_schema,
        source_scaler_path=source_scaler,
        encoder_checkpoint_path=checkpoint,
        embedding_scaler_path=embedding_scaler,
        classifier_path=classifier,
        model_kind=kind,
        device=device,
        batch_size=batch_size,
    )

    if model == "teacher_int8":
        predictor.encoder = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(predictor.encoder), {nn.Linear}, dtype=torch.qint8
        ).cpu().eval()

    return predictor


def predict_records(
    predictor: AdaptedNIDS,
    records: pd.DataFrame,
    *,
    attack_threshold: float | None = None,
) -> pd.DataFrame:
    """Return attack-category predictions and attack scores in input order."""
    result = predictor.predict_flows(records, attack_threshold=attack_threshold)
    result.insert(0, "Row_Index", np.arange(len(result), dtype=np.int64))
    return result


def predict_csv(
    predictor: AdaptedNIDS,
    input_path: str | Path,
    output_path: str | Path,
    *,
    chunksize: int = 10000,
    attack_threshold: float | None = None,
) -> InferenceSummary:
    """Score network-flow CSV records in chunks and save a prediction table.

    The output contains a zero-based row index and model predictions only.
    A threshold must be provided explicitly; it is not fitted using input labels.
    Dynamic INT8 activation quantization may cause small batch-dependent differences.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be positive")

    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    _required_file(source)
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output files must be different")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    row_count = 0
    start = time.perf_counter()

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=".nids_", suffix=".csv",
            dir=destination.parent, delete=False, encoding="utf-8", newline=""
        ) as temporary:
            temporary_path = Path(temporary.name)
            header_written = False
            for chunk in pd.read_csv(source, chunksize=chunksize):
                predictions = predict_records(
                    predictor, chunk, attack_threshold=attack_threshold
                )
                predictions["Row_Index"] += row_count
                predictions.to_csv(temporary, index=False, header=not header_written)
                row_count += len(predictions)
                header_written = True

            if not header_written:
                empty = predict_records(
                    predictor, pd.DataFrame(), attack_threshold=attack_threshold
                )
                empty.to_csv(temporary, index=False)

        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return InferenceSummary(
        rows=row_count,
        elapsed_seconds=time.perf_counter() - start,
        output_path=destination,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict network intrusion categories from flow records."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_VARIANTS, default="teacher")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--chunksize", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args(argv)

    predictor = load_inference_model(
        args.data_dir, args.model, device=args.device, batch_size=args.batch_size
    )
    summary = predict_csv(
        predictor, args.input, args.output,
        chunksize=args.chunksize, attack_threshold=args.threshold,
    )
    print(f"Records: {summary.rows:,}")
    print(f"Elapsed: {summary.elapsed_seconds:.3f} s")
    print(f"Throughput: {summary.rows_per_second:.1f} records/s")
    print(f"Predictions: {summary.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
