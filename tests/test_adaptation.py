"""Tests for target-label selection and CIC classifier adaptation."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adaptation import (
    CLASSIFIER_FEATURES,
    DEFAULT_SEEDS,
    INITIAL_LABELS,
    LABEL_BUDGET,
    POLICY_FEATURES,
    acquire_labels,
    compare_selection_methods,
    evaluate_adaptation,
    fit_adapted_classifier,
    initial_sample,
    make_policy,
    random_selection,
)


@pytest.fixture
def target_pool():
    rng = np.random.default_rng(786)
    names = np.array(
        ["Benign"] * 48 + ["DDoS"] * 36 + ["DoS"] * 24 + ["Bot"] * 12
    )
    names = names[rng.permutation(len(names))]
    binary = (names != "Benign").astype(np.int64)
    policy_features = rng.normal(size=(len(names), POLICY_FEATURES)).astype(np.float32)
    policy_features[:, 0] += 1.5 * binary
    policy_features[:, 1] += np.select(
        [names == "DDoS", names == "DoS", names == "Bot"],
        [2.0, -2.0, 0.8],
        default=0.0,
    )
    classifier_features = rng.normal(size=(len(names), CLASSIFIER_FEATURES)).astype(np.float32)
    classifier_features[:, :POLICY_FEATURES] = policy_features
    return policy_features, classifier_features, binary, names


@pytest.fixture
def initial_indices(target_pool):
    return initial_sample(len(target_pool[0]), count=8, seed=51)


@pytest.fixture
def starting_weights():
    torch.manual_seed(19)
    return deepcopy(make_policy().state_dict())


def test_experiment_defaults():
    assert POLICY_FEATURES == 43
    assert CLASSIFIER_FEATURES == 75
    assert INITIAL_LABELS == 50
    assert LABEL_BUDGET == 1000
    assert DEFAULT_SEEDS == (51, 73, 97)


def test_initial_sample_is_unique_reproducible_and_in_range():
    first = initial_sample(120, count=50, seed=51)
    second = initial_sample(120, count=50, seed=51)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (50,)
    assert len(np.unique(first)) == 50
    assert np.all((0 <= first) & (first < 120))
    assert not np.array_equal(first, initial_sample(120, count=50, seed=73))


@pytest.mark.parametrize("size,count", [(0, 1), (10, 0), (10, 11)])
def test_initial_sample_rejects_invalid_sizes(size, count):
    with pytest.raises(ValueError, match="Initial sample size"):
        initial_sample(size, count=count)


def test_random_selection_preserves_start_and_budget(initial_indices):
    result = random_selection(120, initial_indices, seed=51, budget=30)
    np.testing.assert_array_equal(result[:len(initial_indices)], initial_indices)
    assert len(result) == 30
    assert len(np.unique(result)) == 30
    assert result.min() >= 0 and result.max() < 120
    np.testing.assert_array_equal(
        result, random_selection(120, initial_indices, seed=51, budget=30)
    )
    assert not np.array_equal(result, random_selection(120, initial_indices, seed=73, budget=30))


def test_random_selection_allows_no_additional_reviews(initial_indices):
    actual = random_selection(120, initial_indices, seed=51, budget=len(initial_indices))
    np.testing.assert_array_equal(actual, initial_indices)


@pytest.mark.parametrize(
    "initial,budget",
    [([0, 0], 5), ([], 5), ([-1, 0], 5), ([0, 120], 5), ([0, 1], 1), ([0, 1], 121)],
)
def test_random_selection_rejects_invalid_indices_or_budget(initial, budget):
    with pytest.raises(ValueError):
        random_selection(120, initial, seed=51, budget=budget)


def test_policy_network_shape_and_invalid_width():
    model = make_policy()
    assert model(torch.zeros(5, 43)).shape == (5, 1)
    assert model[0].in_features == 43
    assert model[0].out_features == 32
    with pytest.raises(ValueError, match="positive"):
        make_policy(n_features=0)


@pytest.mark.parametrize("diversity", [False, True])
def test_bandit_selection_preserves_start_budget_and_seed(
    target_pool, initial_indices, starting_weights, diversity
):
    policy_features, _, binary, names = target_pool
    args = (policy_features, binary, names, initial_indices)
    result = acquire_labels(
        *args, seed=51, starting_weights=starting_weights,
        diversity=diversity, budget=18, candidate_count=24,
    )
    repeated = acquire_labels(
        *args, seed=51, starting_weights=starting_weights,
        diversity=diversity, budget=18, candidate_count=24,
    )
    np.testing.assert_array_equal(result, repeated)
    np.testing.assert_array_equal(result[:len(initial_indices)], initial_indices)
    assert len(result) == 18
    assert len(np.unique(result)) == 18
    assert np.all((result >= 0) & (result < len(binary)))


def test_bandit_selection_does_not_require_extra_labels_at_initial_budget(
    target_pool, initial_indices, starting_weights
):
    policy_features, _, binary, names = target_pool
    selected = acquire_labels(
        policy_features, binary, names, initial_indices,
        seed=51, starting_weights=starting_weights, diversity=True,
        budget=len(initial_indices),
    )
    np.testing.assert_array_equal(selected, initial_indices)


@pytest.mark.parametrize(
    "change,error",
    [
        ("wrong_features", "43 columns"),
        ("nan_features", "missing or infinite"),
        ("wrong_label_length", "same row count"),
        ("invalid_label", "only 0 and 1"),
        ("duplicate_initial", "unique"),
        ("invalid_initial", "outside the adaptation pool"),
        ("budget_too_small", "Label budget"),
        ("epsilon", "epsilon"),
        ("candidate_count", "candidate_count"),
    ],
)
def test_bandit_rejects_invalid_inputs(
    target_pool, initial_indices, starting_weights, change, error
):
    features, _, binary, names = target_pool
    features, binary, names, initial = (
        features.copy(), binary.copy(), names.copy(), initial_indices.copy()
    )
    options = {"budget": 14, "epsilon": 0.2, "candidate_count": 16}
    if change == "wrong_features":
        features = features[:, :-1]
    elif change == "nan_features":
        features[0, 0] = np.nan
    elif change == "wrong_label_length":
        names = names[:-1]
    elif change == "invalid_label":
        binary[0] = 2
    elif change == "duplicate_initial":
        initial[0] = initial[1]
    elif change == "invalid_initial":
        initial[0] = 120
    elif change == "budget_too_small":
        options["budget"] = len(initial) - 1
    elif change == "epsilon":
        options["epsilon"] = 1.1
    elif change == "candidate_count":
        options["candidate_count"] = 0
    with pytest.raises(ValueError, match=error):
        acquire_labels(
            features, binary, names, initial,
            seed=51, starting_weights=starting_weights, diversity=False, **options,
        )


def test_classifier_groups_categories_below_review_threshold(target_pool):
    _, classifier_features, binary, names = target_pool
    selected = np.concatenate([
        np.flatnonzero(names == "Benign")[:10],
        np.flatnonzero(names == "DDoS")[:8],
        np.flatnonzero(names == "DoS")[:6],
        np.flatnonzero(names == "Bot")[:2],
    ])
    classifier, supported = fit_adapted_classifier(
        classifier_features, binary, names, selected
    )
    assert supported == {"DDoS", "DoS"}
    assert set(classifier.classes_) == {"Benign", "DDoS", "DoS", "Other_Attack"}
    assert classifier.n_features_in_ == CLASSIFIER_FEATURES
    probabilities = classifier.predict_proba(classifier_features[:5])
    assert probabilities.shape == (5, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-6)

    _, lower_threshold_supported = fit_adapted_classifier(
        classifier_features, binary, names, selected, min_category_examples=2
    )
    assert "Bot" in lower_threshold_supported


@pytest.mark.parametrize(
    "change,error",
    [
        ("wrong_features", "75 input features"),
        ("nan_features", "missing or infinite"),
        ("wrong_name_length", "row counts differ"),
        ("invalid_binary", "only 0 and 1"),
        ("duplicate_selected", "unique and nonempty"),
        ("invalid_selected", "outside the adaptation pool"),
        ("min_examples", "positive"),
    ],
)
def test_classifier_rejects_invalid_inputs(target_pool, change, error):
    _, features, binary, names = target_pool
    features, binary, names, indices = (
        features.copy(), binary.copy(), names.copy(), np.arange(50),
    )
    min_examples = 3
    if change == "wrong_features":
        features = features[:, :-1]
    elif change == "nan_features":
        features[0, 0] = np.inf
    elif change == "wrong_name_length":
        names = names[:-1]
    elif change == "invalid_binary":
        binary[0] = 2
    elif change == "duplicate_selected":
        indices[0] = indices[1]
    elif change == "invalid_selected":
        indices[0] = len(features)
    elif change == "min_examples":
        min_examples = 0
    with pytest.raises(ValueError, match=error):
        fit_adapted_classifier(
            features, binary, names, indices, min_category_examples=min_examples,
        )


def test_evaluation_matches_reference_multiclass_and_binary_metrics():
    class FixedClassifier:
        classes_ = np.array(["Benign", "DDoS", "DoS"])

        def predict_proba(self, features):
            return np.array([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.3, 0.6, 0.1],
                [0.7, 0.2, 0.1],
            ])

    classifier = FixedClassifier()
    binary = np.array([0, 1, 1, 0, 1])
    attack_names = np.array(["ignored", "DDoS", "DoS", "ignored", "DDoS"])
    reviewed_binary = np.array([0, 1, 1, 0])
    indices = np.array([0, 1, 2])
    result = evaluate_adaptation(
        classifier, np.zeros((5, CLASSIFIER_FEATURES)), binary, attack_names,
        reviewed_binary, {"DDoS", "DoS"}, indices,
    )
    scores = 1 - classifier.predict_proba(None)[:, 0]
    predictions = classifier.classes_[classifier.predict_proba(None).argmax(axis=1)]
    expected_names = np.array(["Benign", "DDoS", "DoS", "Benign", "DDoS"])
    assert result["Macro_F1"] == pytest.approx(
        f1_score(expected_names, predictions, labels=["DDoS", "DoS"], average="macro")
    )
    assert result["Binary_PR_AUC"] == pytest.approx(average_precision_score(binary, scores))
    assert result["Exact_Type_Accuracy"] == pytest.approx(2 / 3)
    assert result["Reviewed_Attacks"] == 2
    assert result["Supported_Types"] == 2


@pytest.mark.parametrize(
    "change,error",
    [
        ("wrong_features", "75 columns"),
        ("wrong_names", "different row counts"),
        ("invalid_binary", "only 0 and 1"),
        ("invalid_review_labels", "only 0 and 1"),
        ("invalid_selected", "outside the reviewed pool"),
    ],
)
def test_evaluation_rejects_invalid_inputs(target_pool, change, error):
    _, features, binary, names = target_pool
    features, binary, names, reviewed, selected = (
        features.copy(), binary.copy(), names.copy(), binary.copy(), np.arange(20)
    )
    if change == "wrong_features":
        features = features[:, :-1]
    elif change == "wrong_names":
        names = names[:-1]
    elif change == "invalid_binary":
        binary[0] = 2
    elif change == "invalid_review_labels":
        reviewed[0] = 2
    elif change == "invalid_selected":
        selected[0] = len(reviewed)
    with pytest.raises(ValueError, match=error):
        evaluate_adaptation(
            None, features, binary, names, reviewed, {"DDoS"}, selected
        )


def test_method_comparison_uses_shared_initial_labels_and_equal_budgets(target_pool):
    policy_features, classifier_features, binary, names = target_pool
    rng = np.random.default_rng(900)
    dev_names = np.array(["Benign"] * 12 + ["DDoS"] * 12 + ["DoS"] * 12 + ["Bot"] * 12)
    dev_binary = (dev_names != "Benign").astype(np.int64)
    dev_features = rng.normal(size=(len(dev_names), CLASSIFIER_FEATURES)).astype(np.float32)
    dev_features[:, 0] += 1.5 * dev_binary

    results, selections = compare_selection_methods(
        policy_features, classifier_features, binary, names,
        dev_features, dev_binary, dev_names,
        seeds=(51, 73), label_budget=56,
    )
    assert len(results) == 6
    assert set(results["Seed"]) == {51, 73}
    assert set(results["Method"]) == {"Random", "Standard_RL", "Diversity_Aware_RL"}
    assert results[["Macro_F1", "Binary_PR_AUC", "Exact_Type_Accuracy"]].notna().all().all()
    common_initial = initial_sample(len(policy_features))
    for seed in (51, 73):
        for method in ("Random", "Standard_RL", "Diversity_Aware_RL"):
            selection = selections[(seed, method)]
            assert len(selection) == 56
            assert len(np.unique(selection)) == 56
            np.testing.assert_array_equal(selection[:INITIAL_LABELS], common_initial)
            row = results.loc[(results.Seed == seed) & (results.Method == method)].iloc[0]
            assert int(row.Reviewed_Attacks) == int(binary[selection].sum())
