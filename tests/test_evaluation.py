"""Tests for network intrusion detection metrics and uncertainty estimates."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (
    DEFAULT_ALERT_FRACTION,
    DEFAULT_BOOTSTRAP_SEED,
    alert_budget_metrics,
    attack_scores_from_proba,
    category_results,
    evaluate_predictions,
    paired_category_bootstrap,
    threshold_metrics,
)


@pytest.fixture
def predictions():
    truth = np.array([
        "Benign", "Benign", "DoS", "DoS", "DDoS", "DDoS",
        "Scan", "Scan", "Benign", "DoS", "Benign", "Scan",
    ])
    predicted = np.array([
        "Benign", "DoS", "DoS", "DDoS", "DDoS", "Benign",
        "Scan", "Scan", "Benign", "DoS", "Benign", "DoS",
    ])
    binary = (truth != "Benign").astype(int)
    scores = np.array([
        0.01, 0.80, 0.90, 0.80, 0.70, 0.15,
        0.65, 0.60, 0.03, 0.95, 0.04, 0.50,
    ])
    taxonomy = ["DoS", "DDoS", "Scan"]
    return binary, truth, predicted, scores, taxonomy


def test_attack_scores_follow_benign_class_column():
    probabilities = np.array([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    scores = attack_scores_from_proba(probabilities, ["DoS", "Benign", "Scan"])
    np.testing.assert_allclose(scores, [0.2, 0.8])


def test_attack_scores_with_benign_first():
    scores = attack_scores_from_proba(
        [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], ["Benign", "DoS"]
    )
    np.testing.assert_allclose(scores, [0.0, 1.0, 0.75])


def test_attack_scores_accept_custom_benign_label():
    scores = attack_scores_from_proba(
        [[0.3, 0.7]], ["Normal", "Attack"], benign_label="Normal"
    )
    np.testing.assert_allclose(scores, [0.7])


@pytest.mark.parametrize("probabilities,classes", [
    ([[0.2, 0.8]], ["DoS", "Scan"]),
    ([[0.2, 0.8]], ["Benign", "Benign"]),
    ([[0.5, 0.5]], ["Benign", "Benign", "DoS"]),
    ([[0.2, 0.7]], ["Benign", "DoS"]),
    ([[-0.1, 1.1]], ["Benign", "DoS"]),
    ([[float("nan"), 1.0]], ["Benign", "DoS"]),
    ([[float("inf"), 0.0]], ["Benign", "DoS"]),
    ([0.1, 0.9], ["Benign", "DoS"]),
])
def test_attack_scores_reject_invalid_probabilities(probabilities, classes):
    with pytest.raises(ValueError):
        attack_scores_from_proba(probabilities, classes)


def test_default_alert_fraction():
    assert DEFAULT_ALERT_FRACTION == 0.125


def test_alert_budget_uses_ceiling_and_highest_scores(predictions):
    binary, _, _, scores, _ = predictions
    result = alert_budget_metrics(binary, scores, fraction=0.25)
    assert result["Alert_Budget_Records"] == 3
    assert result["Attacks_Found_At_Budget"] == 2
    assert result["Precision_At_Budget"] == pytest.approx(2 / 3)
    assert result["Recall_At_Budget"] == pytest.approx(2 / 8)


def test_alert_budget_default_selects_two_of_twelve(predictions):
    binary, _, _, scores, _ = predictions
    result = alert_budget_metrics(binary, scores)
    assert result["Alert_Budget_Records"] == 2
    assert result["Attacks_Found_At_Budget"] == 2
    assert result["Precision_At_Budget"] == 1.0


def test_alert_budget_ties_have_stable_row_order():
    result = alert_budget_metrics([0, 1, 1, 0], [0.9, 0.9, 0.2, 0.1], 0.25)
    assert result["Alert_Budget_Records"] == 1
    assert result["Attacks_Found_At_Budget"] == 0


def test_alert_budget_full_fraction(predictions):
    binary, _, _, scores, _ = predictions
    result = alert_budget_metrics(binary, scores, 1.0)
    assert result["Alert_Budget_Records"] == len(binary)
    assert result["Precision_At_Budget"] == pytest.approx(binary.mean())
    assert result["Recall_At_Budget"] == 1.0


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_alert_budget_rejects_invalid_fraction(fraction, predictions):
    binary, _, _, scores, _ = predictions
    with pytest.raises(ValueError):
        alert_budget_metrics(binary, scores, fraction)


def test_alert_budget_needs_attack_records():
    with pytest.raises(ValueError, match="At least one attack"):
        alert_budget_metrics([0, 0, 0], [0.1, 0.2, 0.3])


def test_threshold_counts_and_scores(predictions):
    binary, _, _, scores, _ = predictions
    result = threshold_metrics(binary, scores, 0.6)
    assert (result["True_Positives"], result["False_Positives"]) == (6, 1)
    assert (result["False_Negatives"], result["True_Negatives"]) == (2, 3)
    assert result["Binary_Precision"] == pytest.approx(6 / 7)
    assert result["Binary_Recall"] == pytest.approx(6 / 8)
    assert result["Binary_F1"] == pytest.approx(0.8)


def test_threshold_includes_scores_equal_to_threshold():
    metrics = threshold_metrics([1, 0], [0.6, 0.4], 0.6)
    assert metrics["True_Positives"] == 1
    assert metrics["False_Positives"] == 0


def test_threshold_extremes(predictions):
    binary, _, _, scores, _ = predictions
    all_alerts = threshold_metrics(binary, scores, 0)
    no_alerts = threshold_metrics(binary, scores, 1)
    assert all_alerts["True_Positives"] == 8
    assert all_alerts["False_Positives"] == 4
    assert no_alerts["True_Positives"] == 0
    assert no_alerts["True_Negatives"] == 4
    assert no_alerts["Binary_Precision"] == 0.0


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_threshold_rejects_invalid_values(predictions, threshold):
    binary, _, _, scores, _ = predictions
    with pytest.raises(ValueError, match="Threshold"):
        threshold_metrics(binary, scores, threshold)


def test_evaluate_predictions_matches_reference_metrics(predictions):
    binary, truth, predicted, scores, taxonomy = predictions
    result = evaluate_predictions(
        binary, truth, predicted, scores, attack_categories=taxonomy
    )
    assert result["Evaluation_Records"] == 12
    assert result["Attack_Records"] == 8
    assert result["Attack_Prevalence"] == pytest.approx(8 / 12)
    assert result["Correct_Attack_Type_Predictions"] == 5
    assert result["Exact_Attack_Type_Accuracy"] == pytest.approx(5 / 8)
    assert result["Multiclass_Macro_F1"] == pytest.approx(
        f1_score(truth, predicted, labels=taxonomy, average="macro", zero_division=0)
    )
    assert result["Binary_PR_AUC"] == pytest.approx(
        average_precision_score(binary, scores)
    )
    assert result["Binary_ROC_AUC"] == pytest.approx(roc_auc_score(binary, scores))
    assert "Threshold" not in result


def test_evaluate_predictions_uses_declared_taxonomy_not_only_observed_labels():
    truth = ["Benign", "DoS", "Benign", "DoS"]
    predicted = ["Benign", "DoS", "Benign", "Benign"]
    result = evaluate_predictions(
        [0, 1, 0, 1], truth, predicted, [0.1, 0.9, 0.2, 0.3],
        attack_categories=["DoS", "Scan"],
    )
    do_s_f1 = f1_score(truth, predicted, labels=["DoS"], average="macro")
    assert result["Multiclass_Macro_F1"] == pytest.approx(do_s_f1 / 2)


def test_evaluation_with_threshold_adds_alert_counts(predictions):
    binary, truth, predicted, scores, taxonomy = predictions
    result = evaluate_predictions(
        binary, truth, predicted, scores,
        attack_categories=taxonomy, alert_fraction=0.25, threshold=0.6,
    )
    assert result["True_Positives"] == 6
    assert result["Alert_Budget_Records"] == 3
    assert result["Precision_At_Budget"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("binary,truth,predicted,scores", [
    ([0, 1], ["Benign", "DoS"], ["Benign"], [0.1, 0.9]),
    ([0, 1], ["Benign", "DoS"], ["Benign", "DoS"], [0.1]),
    ([0, 2], ["Benign", "DoS"], ["Benign", "DoS"], [0.1, 0.9]),
    ([0, 1], ["DoS", "Benign"], ["Benign", "DoS"], [0.1, 0.9]),
    ([0, 1], ["Benign", "Benign"], ["Benign", "DoS"], [0.1, 0.9]),
    ([0, 1], ["Benign", "DoS"], ["Benign", "DoS"], [0.1, -0.1]),
    ([0, 1], ["Benign", "DoS"], ["Benign", "DoS"], [0.1, float("nan")]),
    ([0, 1], ["Benign", "DoS"], ["Benign", "DoS"], [0.1, 1.1]),
])
def test_evaluation_rejects_bad_arrays(binary, truth, predicted, scores):
    with pytest.raises(ValueError):
        evaluate_predictions(binary, truth, predicted, scores, attack_categories=["DoS"])


@pytest.mark.parametrize("taxonomy", [[], ["DoS", "DoS"], ["Benign", "DoS"]])
def test_evaluation_rejects_invalid_taxonomy(predictions, taxonomy):
    binary, truth, predicted, scores, _ = predictions
    with pytest.raises(ValueError):
        evaluate_predictions(binary, truth, predicted, scores, attack_categories=taxonomy)


@pytest.mark.parametrize("binary,truth,predicted,scores", [
    ([0, 0], ["Benign", "Benign"], ["Benign", "Benign"], [0.1, 0.2]),
    ([1, 1], ["DoS", "DoS"], ["DoS", "DoS"], [0.8, 0.9]),
])
def test_evaluation_requires_both_binary_classes(binary, truth, predicted, scores):
    with pytest.raises(ValueError, match="Both benign and attack"):
        evaluate_predictions(binary, truth, predicted, scores, attack_categories=["DoS"])


def test_category_precision_recall_support(predictions):
    binary, truth, predicted, _, taxonomy = predictions
    result = category_results(binary, truth, predicted, attack_categories=taxonomy)
    assert set(result.columns) == {"Attack_Category", "Test_Records", "Precision", "Recall", "F1"}
    dos = result.set_index("Attack_Category").loc["DoS"]
    assert dos["Test_Records"] == 3
    assert dos["Precision"] == pytest.approx(0.5)
    assert dos["Recall"] == pytest.approx(2 / 3)
    ddos = result.set_index("Attack_Category").loc["DDoS"]
    assert ddos["Test_Records"] == 2
    assert ddos["Recall"] == pytest.approx(1 / 2)
    scan = result.set_index("Attack_Category").loc["Scan"]
    assert scan["Precision"] == pytest.approx(1.0)
    assert scan["Recall"] == pytest.approx(2 / 3)


def test_category_results_includes_unseen_taxonomy_category(predictions):
    binary, truth, predicted, _, taxonomy = predictions
    result = category_results(
        binary, truth, predicted, attack_categories=taxonomy + ["Rare"]
    ).set_index("Attack_Category")
    assert result.loc["Rare", "Test_Records"] == 0
    assert result.loc["Rare", "Recall"] == 0.0


def test_category_results_includes_unexpected_true_attack_category():
    result = category_results(
        [0, 1, 1], ["Benign", "DoS", "Novel"], ["Benign", "DoS", "DoS"],
        attack_categories=["DoS"]
    ).set_index("Attack_Category")
    assert result.loc["Novel", "Test_Records"] == 1
    assert result.loc["Novel", "Recall"] == 0.0


def test_category_results_marks_model_supported_categories(predictions):
    binary, truth, predicted, _, taxonomy = predictions
    result = category_results(
        binary, truth, predicted,
        attack_categories=taxonomy, supported_categories=["DoS", "Scan"],
    ).set_index("Attack_Category")
    assert bool(result.loc["DoS", "Supported_By_Model"])
    assert bool(result.loc["Scan", "Supported_By_Model"])
    assert not bool(result.loc["DDoS", "Supported_By_Model"])


def test_category_results_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="Category labels"):
        category_results([0, 1], ["Benign"], ["Benign", "DoS"], attack_categories=["DoS"])


def test_category_results_rejects_binary_category_disagreement():
    with pytest.raises(ValueError, match="do not agree"):
        category_results([0, 1], ["DoS", "Benign"], ["DoS", "Benign"], attack_categories=["DoS"])


def test_bootstrap_identical_models_have_zero_deltas(predictions):
    binary, truth, predicted, scores, taxonomy = predictions
    summary, samples = paired_category_bootstrap(
        binary, truth, predicted, scores, predicted, scores,
        attack_categories=taxonomy, n_bootstrap=30, seed=123,
    )
    assert len(samples) == 30
    assert samples["Iteration"].tolist() == list(range(30))
    assert (samples[["Macro_F1_Delta", "PR_AUC_Delta"]] == 0).all().all()
    assert all(value == 0 for value in summary.values())


def test_bootstrap_detects_improvement():
    truth = np.array(["Benign"] * 8 + ["DoS"] * 8 + ["Scan"] * 8)
    binary = (truth != "Benign").astype(int)
    baseline = np.array(["Benign"] * len(truth))
    candidate = truth.copy()
    baseline_scores = np.full(len(truth), 0.5)
    candidate_scores = np.where(binary == 1, 0.9, 0.1)
    summary, samples = paired_category_bootstrap(
        binary, truth, baseline, baseline_scores, candidate, candidate_scores,
        attack_categories=["DoS", "Scan"], n_bootstrap=35, seed=123,
    )
    assert summary["Observed_F1_Delta"] == pytest.approx(1.0)
    assert summary["Observed_AP_Delta"] == pytest.approx(1 - binary.mean())
    assert summary["F1_CI_Lower"] > 0
    assert summary["AP_CI_Lower"] > 0
    assert (samples["Macro_F1_Delta"] > 0).all()
    assert (samples["PR_AUC_Delta"] > 0).all()


def test_bootstrap_same_seed_is_reproducible(predictions):
    binary, truth, predicted, scores, taxonomy = predictions
    improved = truth.copy()
    improved_scores = np.where(binary == 1, 0.9, 0.1)
    args = (binary, truth, predicted, scores, improved, improved_scores)
    summary_a, samples_a = paired_category_bootstrap(
        *args, attack_categories=taxonomy, n_bootstrap=40, seed=7
    )
    summary_b, samples_b = paired_category_bootstrap(
        *args, attack_categories=taxonomy, n_bootstrap=40, seed=7
    )
    assert summary_a == summary_b
    pd.testing.assert_frame_equal(samples_a, samples_b)
    _, samples_c = paired_category_bootstrap(
        *args, attack_categories=taxonomy, n_bootstrap=40, seed=11
    )
    assert not samples_a.equals(samples_c)


def test_bootstrap_summary_confidence_bounds_are_ordered(predictions):
    binary, truth, predicted, scores, taxonomy = predictions
    summary, _ = paired_category_bootstrap(
        binary, truth, predicted, scores, truth, np.where(binary == 1, 0.9, 0.1),
        attack_categories=taxonomy, n_bootstrap=50, seed=DEFAULT_BOOTSTRAP_SEED,
    )
    assert summary["F1_CI_Lower"] <= summary["F1_CI_Upper"]
    assert summary["AP_CI_Lower"] <= summary["AP_CI_Upper"]
    assert -1 <= summary["Observed_F1_Delta"] <= 1
    assert -1 <= summary["Observed_AP_Delta"] <= 1


@pytest.mark.parametrize("n_bootstrap", [0, 1, -10, 2.5, "400"])
def test_bootstrap_rejects_invalid_iteration_counts(predictions, n_bootstrap):
    binary, truth, predicted, scores, taxonomy = predictions
    with pytest.raises(ValueError, match="n_bootstrap"):
        paired_category_bootstrap(
            binary, truth, predicted, scores, predicted, scores,
            attack_categories=taxonomy, n_bootstrap=n_bootstrap,
        )


def test_bootstrap_rejects_unmatched_candidate_predictions(predictions):
    binary, truth, predicted, scores, taxonomy = predictions
    with pytest.raises(ValueError):
        paired_category_bootstrap(
            binary, truth, predicted, scores, predicted[:-1], scores,
            attack_categories=taxonomy, n_bootstrap=5,
        )


def test_bootstrap_rejects_single_binary_class():
    with pytest.raises(ValueError, match="Both binary classes"):
        paired_category_bootstrap(
            [1, 1], ["DoS", "DoS"], ["DoS", "DoS"], [0.8, 0.9],
            ["DoS", "DoS"], [0.8, 0.9],
            attack_categories=["DoS"], n_bootstrap=5,
        )
