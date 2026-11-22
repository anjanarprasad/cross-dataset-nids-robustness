"""Target-label selection and classifier adaptation for the  NIDS study."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from torch.distributions import Categorical


POLICY_FEATURES = 43
CLASSIFIER_FEATURES = 75
DEFAULT_SEEDS = (51, 73, 97)
LABEL_BUDGET = 1000
INITIAL_LABELS = 50
MIN_CATEGORY_EXAMPLES = 3


def initial_sample(n_records: int, count: int = INITIAL_LABELS, seed: int = 78643) -> np.ndarray:
    """Choose the shared starting records for all three sampling methods."""
    if not 0 < count <= n_records:
        raise ValueError("Initial sample size must be between 1 and the pool size")
    return np.random.default_rng(seed).choice(n_records, size=count, replace=False)


def make_policy(n_features: int = POLICY_FEATURES, device: str = "cpu") -> nn.Sequential:
    """Create the contextual-bandit policy used for target sample selection."""
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    return nn.Sequential(
        nn.Linear(n_features, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    ).to(device)


def _check_pool(
    features: np.ndarray,
    binary_labels: Sequence[int],
    attack_names: Sequence[str],
    initial_indices: Sequence[int],
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    binary = np.asarray(binary_labels, dtype=np.int64)
    names = np.asarray(attack_names, dtype=str)
    initial = np.asarray(initial_indices, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != POLICY_FEATURES:
        raise ValueError("Policy features must have 43 columns")
    if not np.isfinite(x).all():
        raise ValueError("Policy features contain missing or infinite values")
    if len(binary) != len(x) or len(names) != len(x):
        raise ValueError("Pool features and labels must have the same row count")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("Binary labels must contain only 0 and 1")
    if initial.ndim != 1 or len(initial) == 0 or len(initial) != len(np.unique(initial)):
        raise ValueError("Initial indices must be a nonempty list of unique row indices")
    if initial.min() < 0 or initial.max() >= len(x):
        raise ValueError("Initial indices are outside the adaptation pool")
    if not len(initial) <= budget <= len(x):
        raise ValueError("Label budget must be between the initial count and pool size")
    return x, binary, names, initial


def random_selection(
    n_records: int,
    initial_indices: Sequence[int],
    seed: int,
    budget: int = LABEL_BUDGET,
) -> np.ndarray:
    """Select remaining records at random after the shared starting sample."""
    initial = np.asarray(initial_indices, dtype=np.int64)
    if initial.ndim != 1 or not len(initial) or len(np.unique(initial)) != len(initial):
        raise ValueError("Initial indices must be unique and nonempty")
    if initial.min() < 0 or initial.max() >= n_records:
        raise ValueError("Initial indices are outside the adaptation pool")
    if not len(initial) <= budget <= n_records:
        raise ValueError("Invalid label budget")

    rng = np.random.default_rng(seed + 100)
    remaining = np.setdiff1d(np.arange(n_records), initial)
    return np.concatenate([
        initial,
        rng.choice(remaining, size=budget - len(initial), replace=False),
    ])


def acquire_labels(
    policy_features: np.ndarray,
    binary_labels: Sequence[int],
    attack_names: Sequence[str],
    initial_indices: Sequence[int],
    seed: int,
    starting_weights: dict[str, torch.Tensor],
    diversity: bool,
    budget: int = LABEL_BUDGET,
    epsilon: float = 0.20,
    candidate_count: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Train the bandit while selecting target records for label review."""
    x, binary, names, initial = _check_pool(
        policy_features, binary_labels, attack_names, initial_indices, budget
    )
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")

    torch.manual_seed(seed + 100)
    rng = np.random.default_rng(seed + 100)
    policy = make_policy(x.shape[1], device=device)
    policy.load_state_dict(starting_weights, strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)

    feature_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    available = np.ones(len(x), dtype=bool)
    available[initial] = False
    selected = list(initial)
    counts = Counter(names[i] for i in initial if binary[i] == 1)
    baseline = 0.0

    while len(selected) < budget:
        candidates = rng.choice(
            np.flatnonzero(available),
            size=min(candidate_count, int(available.sum())),
            replace=False,
        )
        logits = policy(feature_tensor[candidates]).flatten()
        probabilities = (
            (1.0 - epsilon) * torch.softmax(logits, dim=0)
            + epsilon / len(candidates)
        )
        distribution = Categorical(probs=probabilities)
        action = distribution.sample()
        index = int(candidates[action.item()])

        # Only the selected record provides feedback to the policy.
        if binary[index] == 1:
            category = names[index]
            previous = counts[category]
            reward = 2.0 + 6.0 / np.sqrt(1.0 + previous) if diversity else 2.0
            counts[category] += 1
        else:
            reward = -1.0

        loss = (
            -(reward - baseline) * distribution.log_prob(action)
            - 0.001 * distribution.entropy()
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        optimizer.step()
        baseline = 0.95 * baseline + 0.05 * reward
        selected.append(index)
        available[index] = False

    return np.asarray(selected, dtype=np.int64)


def fit_adapted_classifier(
    classifier_features: np.ndarray,
    binary_labels: Sequence[int],
    attack_names: Sequence[str],
    selected_indices: Sequence[int],
    min_category_examples: int = MIN_CATEGORY_EXAMPLES,
) -> tuple[LogisticRegression, set[str]]:
    """Fit a weighted classifier from the reviewed CIC records."""
    x = np.asarray(classifier_features, dtype=np.float32)
    binary = np.asarray(binary_labels, dtype=np.int64)
    names = np.asarray(attack_names, dtype=str)
    selected = np.asarray(selected_indices, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != CLASSIFIER_FEATURES:
        raise ValueError("Adapted classifier requires 75 input features")
    if len(x) != len(binary) or len(x) != len(names):
        raise ValueError("Feature and label row counts differ")
    if not np.isfinite(x).all():
        raise ValueError("Classifier features contain missing or infinite values")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("Binary labels must contain only 0 and 1")
    if selected.ndim != 1 or len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("Selected indices must be unique and nonempty")
    if selected.min() < 0 or selected.max() >= len(x):
        raise ValueError("Selected indices are outside the adaptation pool")
    if min_category_examples < 1:
        raise ValueError("min_category_examples must be positive")

    reviewed_binary = binary[selected]
    reviewed_names = np.where(reviewed_binary == 0, "Benign", names[selected])
    attack_counts = pd.Series(reviewed_names[reviewed_binary == 1]).value_counts()
    supported = set(attack_counts[attack_counts >= min_category_examples].index)

    training_labels = np.asarray([
        name if (name == "Benign" or name in supported) else "Other_Attack"
        for name in reviewed_names
    ])
    counts = pd.Series(training_labels).value_counts()
    sample_weight_map = {
        name: min(5.0, (counts.max() / count) ** 0.25)
        for name, count in counts.items()
    }
    sample_weights = np.asarray([sample_weight_map[name] for name in training_labels])

    classifier = LogisticRegression(solver="lbfgs", max_iter=1200, random_state=13)
    classifier.fit(x[selected], training_labels, sample_weight=sample_weights)
    return classifier, supported


def evaluate_adaptation(
    classifier: LogisticRegression,
    development_features: np.ndarray,
    development_binary: Sequence[int],
    development_attack_names: Sequence[str],
    reviewed_binary: Sequence[int],
    supported_categories: set[str],
    selected_indices: Sequence[int],
) -> dict[str, float | int]:
    """Measure multiclass and binary performance on the CIC development set."""
    x = np.asarray(development_features, dtype=np.float32)
    binary = np.asarray(development_binary, dtype=np.int64)
    names = np.asarray(development_attack_names, dtype=str)
    selected = np.asarray(selected_indices, dtype=np.int64)
    reviewed_binary = np.asarray(reviewed_binary, dtype=np.int64)

    if x.ndim != 2 or x.shape[1] != CLASSIFIER_FEATURES:
        raise ValueError("Development features must have 75 columns")
    if len(x) != len(binary) or len(x) != len(names):
        raise ValueError("Development features and labels have different row counts")
    if not np.isin(binary, [0, 1]).all():
        raise ValueError("Development binary labels must contain only 0 and 1")
    if not np.isin(reviewed_binary, [0, 1]).all():
        raise ValueError("Reviewed binary labels must contain only 0 and 1")
    if selected.ndim != 1 or len(selected) == 0 or selected.min() < 0 or selected.max() >= len(reviewed_binary):
        raise ValueError("Selected indices are outside the reviewed pool")

    true_names = np.where(binary == 0, "Benign", names)
    categories = sorted(set(true_names) - {"Benign"})
    probabilities = classifier.predict_proba(x)
    predicted = classifier.classes_[probabilities.argmax(axis=1)]
    classes = list(classifier.classes_)
    if "Benign" not in classes:
        raise ValueError("Classifier must include the Benign class")
    attack_scores = 1.0 - probabilities[:, classes.index("Benign")]
    actual_attacks = binary == 1

    return {
        "Macro_F1": float(f1_score(
            true_names, predicted, labels=categories,
            average="macro", zero_division=0,
        )),
        "Binary_PR_AUC": float(average_precision_score(binary, attack_scores)),
        "Exact_Type_Accuracy": float(np.mean(predicted[actual_attacks] == true_names[actual_attacks])),
        "Reviewed_Attacks": int(reviewed_binary[selected].sum()),
        "Supported_Types": len(supported_categories),
    }


def compare_selection_methods(
    policy_features: np.ndarray,
    classifier_pool_features: np.ndarray,
    pool_binary: Sequence[int],
    pool_attack_names: Sequence[str],
    development_features: np.ndarray,
    development_binary: Sequence[int],
    development_attack_names: Sequence[str],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    label_budget: int = LABEL_BUDGET,
    device: str = "cpu",
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    """Compare random selection and two RL rewards using the same label budget."""
    initial = initial_sample(len(policy_features))
    rows = []
    selections = {}

    for seed in seeds:
        torch.manual_seed(int(seed))
        starting_weights = deepcopy(make_policy(device=device).state_dict())
        methods = {
            "Random": random_selection(len(policy_features), initial, int(seed), label_budget),
            "Standard_RL": acquire_labels(
                policy_features, pool_binary, pool_attack_names, initial,
                int(seed), starting_weights, diversity=False,
                budget=label_budget, device=device,
            ),
            "Diversity_Aware_RL": acquire_labels(
                policy_features, pool_binary, pool_attack_names, initial,
                int(seed), starting_weights, diversity=True,
                budget=label_budget, device=device,
            ),
        }

        for method, indices in methods.items():
            classifier, supported = fit_adapted_classifier(
                classifier_pool_features, pool_binary, pool_attack_names, indices,
            )
            metrics = evaluate_adaptation(
                classifier, development_features, development_binary,
                development_attack_names, pool_binary, supported, indices,
            )
            rows.append({"Seed": int(seed), "Method": method, **metrics})
            selections[(int(seed), method)] = indices

    return pd.DataFrame(rows), selections
