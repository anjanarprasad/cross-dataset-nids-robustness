"""Evaluation metrics for the adapted network intrusion detection models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_ALERT_FRACTION = 0.125
DEFAULT_BOOTSTRAP_SEED = 44047


def _binary_labels(values: Sequence[int]) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Binary labels must be a non-empty one-dimensional array")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary labels must contain only 0 and 1")
    return labels.astype(np.int64)


def _categories(values: Sequence[str], name: str) -> np.ndarray:
    labels = np.asarray(values, dtype=str)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return labels


def _scores(values: Sequence[float], n: int) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (n,) or not np.isfinite(scores).all():
        raise ValueError("Attack scores must be finite and match the number of records")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Attack scores must be probabilities between 0 and 1")
    return scores


def _taxonomy(attack_categories: Sequence[str]) -> list[str]:
    labels = [str(label) for label in attack_categories]
    if not labels or len(labels) != len(set(labels)) or "Benign" in labels:
        raise ValueError("Supply unique attack categories, excluding Benign")
    return labels


def _evaluation_arrays(
    true_binary: Sequence[int],
    true_categories: Sequence[str],
    predicted_categories: Sequence[str],
    attack_scores: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binary = _binary_labels(true_binary)
    truth = _categories(true_categories, "True categories")
    predicted = _categories(predicted_categories, "Predicted categories")
    n = len(binary)
    if truth.shape != (n,) or predicted.shape != (n,):
        raise ValueError("Category labels must match the number of records")
    if np.any(truth[binary == 0] != "Benign") or np.any(
        truth[binary == 1] == "Benign"
    ):
        raise ValueError("Binary labels and true attack categories do not agree")
    return binary, truth, predicted, _scores(attack_scores, n)


def attack_scores_from_proba(
    probabilities: np.ndarray,
    classes: Sequence[str],
    benign_label: str = "Benign",
) -> np.ndarray:
    """Return P(attack) = 1 - P(Benign) for a multiclass classifier."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = _categories(classes, "Classifier classes")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Probability columns must match classifier classes")
    if len(set(labels)) != len(labels) or np.sum(labels == benign_label) != 1:
        raise ValueError("Classifier classes must contain one Benign label")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Class probabilities must be finite and nonnegative")
    if np.any(~np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-6)):
        raise ValueError("Class probabilities must sum to one")
    index = int(np.flatnonzero(labels == benign_label)[0])
    return np.clip(1.0 - probabilities[:, index], 0.0, 1.0)


def alert_budget_metrics(
    true_binary: Sequence[int],
    attack_scores: Sequence[float],
    fraction: float = DEFAULT_ALERT_FRACTION,
) -> dict[str, int | float]:
    """Measure precision and recall among the highest-scoring alerts."""
    binary = _binary_labels(true_binary)
    scores = _scores(attack_scores, len(binary))
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Alert fraction must be greater than 0 and at most 1")
    if binary.sum() == 0:
        raise ValueError("At least one attack is required to calculate recall")

    budget = int(np.ceil(len(binary) * fraction))
    # Stable sorting makes equal-score selection deterministic.
    top = np.argsort(-scores, kind="stable")[:budget]
    found = int(binary[top].sum())
    return {
        "Alert_Budget_Fraction": float(fraction),
        "Alert_Budget_Records": budget,
        "Attacks_Found_At_Budget": found,
        "Precision_At_Budget": float(found / budget),
        "Recall_At_Budget": float(found / binary.sum()),
    }


def threshold_metrics(
    true_binary: Sequence[int],
    attack_scores: Sequence[float],
    threshold: float,
) -> dict[str, int | float]:
    """Calculate binary metrics at a threshold chosen before evaluation."""
    binary = _binary_labels(true_binary)
    scores = _scores(attack_scores, len(binary))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    alerts = scores >= threshold
    tp = int(np.count_nonzero(alerts & (binary == 1)))
    fp = int(np.count_nonzero(alerts & (binary == 0)))
    fn = int(np.count_nonzero(~alerts & (binary == 1)))
    tn = int(np.count_nonzero(~alerts & (binary == 0)))
    return {
        "Threshold": float(threshold),
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
        "Binary_Precision": float(precision_score(binary, alerts, zero_division=0)),
        "Binary_Recall": float(recall_score(binary, alerts, zero_division=0)),
        "Binary_F1": float(f1_score(binary, alerts, zero_division=0)),
    }


def evaluate_predictions(
    true_binary: Sequence[int],
    true_categories: Sequence[str],
    predicted_categories: Sequence[str],
    attack_scores: Sequence[float],
    *,
    attack_categories: Sequence[str],
    alert_fraction: float = DEFAULT_ALERT_FRACTION,
    threshold: float | None = None,
) -> dict[str, int | float]:
    """Evaluate multiclass recognition and binary attack detection.

    The attack taxonomy must be fixed independently of the evaluated data.
    In particular, a frozen holdout must not determine its own taxonomy.
    """
    binary, truth, predicted, scores = _evaluation_arrays(
        true_binary, true_categories, predicted_categories, attack_scores
    )
    labels = _taxonomy(attack_categories)
    if len(np.unique(binary)) != 2:
        raise ValueError("Both benign and attack records are needed for PR-AUC and ROC-AUC")

    attack_mask = binary == 1
    correct = int(np.count_nonzero(predicted[attack_mask] == truth[attack_mask]))
    result: dict[str, int | float] = {
        "Evaluation_Records": int(len(binary)),
        "Attack_Records": int(binary.sum()),
        "Attack_Prevalence": float(binary.mean()),
        "Multiclass_Macro_F1": float(
            f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "Exact_Attack_Type_Accuracy": float(correct / binary.sum()),
        "Correct_Attack_Type_Predictions": correct,
        "Binary_PR_AUC": float(average_precision_score(binary, scores)),
        "Binary_ROC_AUC": float(roc_auc_score(binary, scores)),
    }
    result.update(alert_budget_metrics(binary, scores, fraction=alert_fraction))
    if threshold is not None:
        result.update(threshold_metrics(binary, scores, threshold))
    return result


def category_results(
    true_binary: Sequence[int],
    true_categories: Sequence[str],
    predicted_categories: Sequence[str],
    *,
    attack_categories: Sequence[str],
    supported_categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Calculate per-attack precision, recall, F1, and record counts."""
    binary = _binary_labels(true_binary)
    truth = _categories(true_categories, "True categories")
    predicted = _categories(predicted_categories, "Predicted categories")
    if truth.shape != binary.shape or predicted.shape != binary.shape:
        raise ValueError("Category labels must match binary labels")
    if np.any(truth[binary == 0] != "Benign") or np.any(truth[binary == 1] == "Benign"):
        raise ValueError("Binary labels and true attack categories do not agree")

    # Include previously unseen true categories, even when their recall is zero.
    categories = sorted(set(_taxonomy(attack_categories)) | set(truth[binary == 1]))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=categories, zero_division=0
    )
    output = pd.DataFrame({
        "Attack_Category": categories,
        "Test_Records": support.astype(int),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    })
    if supported_categories is not None:
        supported = set(supported_categories)
        output["Supported_By_Model"] = output["Attack_Category"].isin(supported)
    return output


def paired_category_bootstrap(
    true_binary: Sequence[int],
    true_categories: Sequence[str],
    baseline_predictions: Sequence[str],
    baseline_scores: Sequence[float],
    candidate_predictions: Sequence[str],
    candidate_scores: Sequence[float],
    *,
    attack_categories: Sequence[str],
    n_bootstrap: int = 400,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Paired, category-stratified percentile intervals on development data.

    Both models use identical resampled rows. These conditional intervals do
    not account for model selection and are not independent holdout evidence.
    """
    binary, truth, baseline, base_scores = _evaluation_arrays(
        true_binary, true_categories, baseline_predictions, baseline_scores
    )
    _, _, candidate, new_scores = _evaluation_arrays(
        binary, truth, candidate_predictions, candidate_scores
    )
    labels = _taxonomy(attack_categories)
    if len(np.unique(binary)) != 2:
        raise ValueError("Both binary classes are required for bootstrap PR-AUC")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
        raise ValueError("n_bootstrap must be an integer of at least 2")

    def metrics(indices: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        f1 = f1_score(
            truth[indices], predictions[indices], labels=labels,
            average="macro", zero_division=0,
        )
        ap = average_precision_score(binary[indices], scores[indices])
        return float(f1), float(ap)

    full = np.arange(len(binary))
    base_f1, base_ap = metrics(full, baseline, base_scores)
    new_f1, new_ap = metrics(full, candidate, new_scores)
    groups = [np.flatnonzero(truth == name) for name in np.unique(truth)]
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(n_bootstrap):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True) for group in groups
        ])
        bootstrap_base_f1, bootstrap_base_ap = metrics(indices, baseline, base_scores)
        bootstrap_new_f1, bootstrap_new_ap = metrics(indices, candidate, new_scores)
        rows.append({
            "Iteration": iteration,
            "Macro_F1_Delta": bootstrap_new_f1 - bootstrap_base_f1,
            "PR_AUC_Delta": bootstrap_new_ap - bootstrap_base_ap,
        })

    samples = pd.DataFrame(rows)
    f1_low, f1_high = np.quantile(samples["Macro_F1_Delta"], [0.025, 0.975])
    ap_low, ap_high = np.quantile(samples["PR_AUC_Delta"], [0.025, 0.975])
    summary = {
        "Observed_F1_Delta": new_f1 - base_f1,
        "F1_CI_Lower": float(f1_low),
        "F1_CI_Upper": float(f1_high),
        "Observed_AP_Delta": new_ap - base_ap,
        "AP_CI_Lower": float(ap_low),
        "AP_CI_Upper": float(ap_high),
    }
    return summary, samples
