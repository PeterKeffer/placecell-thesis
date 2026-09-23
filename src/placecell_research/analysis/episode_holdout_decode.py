"""Shared episode-held-out classification for categorical analysis probes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


@dataclass(frozen=True, slots=True)
class EpisodeHoldoutClassification:
    predictions: np.ndarray
    test_labels: np.ndarray
    train_labels: np.ndarray
    train_episode_ids: np.ndarray
    test_episode_ids: np.ndarray
    train_sample_count: int
    test_sample_count: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    majority_chance: float


def _sample_indices(
    indices: np.ndarray,
    *,
    maximum_samples: int,
    random_number_generator: np.random.Generator,
) -> np.ndarray:
    if maximum_samples <= 0 or len(indices) <= maximum_samples:
        return indices
    return np.sort(
        random_number_generator.choice(indices, size=maximum_samples, replace=False)
    )


def decode_labels_episode_holdout(
    features: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    *,
    train_fraction: float,
    random_seed: int,
    maximum_samples: int,
    max_iter: int,
    class_weight: str | dict | None,
    classifier_random_state: int | None,
) -> tuple[EpisodeHoldoutClassification | None, str | None]:
    """Fit a categorical decoder while keeping every episode wholly in one split."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    episode_ids = np.asarray(episode_ids)
    if features.ndim != 2:
        raise ValueError(f"Expected features [samples, dimensions], got {features.shape}.")
    if labels.shape != (len(features),) or episode_ids.shape != (len(features),):
        raise ValueError(
            "Labels and episode_ids must each have one entry per feature row; "
            f"got features={features.shape}, labels={labels.shape}, "
            f"episode_ids={episode_ids.shape}."
        )

    unique_episode_ids = np.unique(episode_ids)
    if len(unique_episode_ids) < 2:
        return None, "fewer_than_two_eligible_episodes"
    random_number_generator = np.random.default_rng(int(random_seed))
    shuffled_episode_ids = random_number_generator.permutation(unique_episode_ids)
    train_episode_count = min(
        len(shuffled_episode_ids) - 1,
        max(1, int(len(shuffled_episode_ids) * float(train_fraction))),
    )
    train_episode_ids = shuffled_episode_ids[:train_episode_count]
    test_episode_ids = shuffled_episode_ids[train_episode_count:]
    train_indices = np.flatnonzero(np.isin(episode_ids, train_episode_ids))
    test_indices = np.flatnonzero(np.isin(episode_ids, test_episode_ids))
    train_indices = _sample_indices(
        train_indices,
        maximum_samples=int(maximum_samples),
        random_number_generator=random_number_generator,
    )
    test_indices = _sample_indices(
        test_indices,
        maximum_samples=int(maximum_samples),
        random_number_generator=random_number_generator,
    )
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    if len(np.unique(train_labels)) < 2 or len(test_indices) == 0:
        return None, "degenerate_episode_split"

    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=int(max_iter),
        random_state=classifier_random_state,
    )
    classifier.fit(features[train_indices], train_labels)
    predictions = classifier.predict(features[test_indices])
    _test_values, test_counts = np.unique(test_labels, return_counts=True)
    return (
        EpisodeHoldoutClassification(
            predictions=predictions,
            test_labels=test_labels,
            train_labels=train_labels,
            train_episode_ids=train_episode_ids,
            test_episode_ids=test_episode_ids,
            train_sample_count=int(len(train_indices)),
            test_sample_count=int(len(test_indices)),
            accuracy=float(accuracy_score(test_labels, predictions)),
            balanced_accuracy=float(balanced_accuracy_score(test_labels, predictions)),
            macro_f1=float(f1_score(test_labels, predictions, average="macro")),
            majority_chance=float(np.max(test_counts) / np.sum(test_counts)),
        ),
        None,
    )
