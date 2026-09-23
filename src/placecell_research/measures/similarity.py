"""Population similarity against distance between places, for pairs from different episodes."""

from __future__ import annotations

import numpy as np

EPISODES = 128
STRIDE = 4
PAIR_DRAWS = 8_000_000
PAIR_SEED = 0
COLUMN_MINIMUM_PAIRS = 4000
DISTANCE_LIMIT = 78.0
SAME_HEADING_DEGREES = 45.0
OPPOSITE_HEADING_DEGREES = 135.0
MEAN_EDGES = np.concatenate([np.arange(0.0, 20.0, 1.0), np.arange(20.0, DISTANCE_LIMIT + 1, 3.0)])
MEAN_CENTERS = 0.5 * (MEAN_EDGES[:-1] + MEAN_EDGES[1:])


def cosine_similarity(vectors: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    return np.einsum("ij,ij->i", vectors[left], vectors[right]) / (norms[left] * norms[right])


def binned_mean(
    distance: np.ndarray,
    similarity: np.ndarray,
    edges: np.ndarray,
    selection: np.ndarray | None = None,
) -> np.ndarray:
    """Mean similarity per distance bin, NaN where the bin holds too few pairs."""
    share = 1.0 if selection is None else float(selection.mean())
    if selection is not None:
        distance, similarity = distance[selection], similarity[selection]
    widths = np.diff(edges)
    index = np.digitize(distance, edges) - 1
    totals = np.bincount(index, weights=similarity, minlength=len(edges) - 1)[: len(edges) - 1]
    counts = np.bincount(index, minlength=len(edges) - 1)[: len(edges) - 1]
    enough = counts >= COLUMN_MINIMUM_PAIRS * widths * share
    return np.where(enough, totals / np.maximum(counts, 1), np.nan)


def half_distance(centers: np.ndarray, means: np.ndarray) -> float:
    """Where the mean curve first falls below half its first finite value, interpolated."""
    finite = np.isfinite(means)
    centers, means = centers[finite], means[finite]
    if not means.size:
        return float("nan")
    target = 0.5 * means[0]
    below = np.flatnonzero(means < target)
    if below.size == 0:
        return float("nan")
    first = below[0]
    if first == 0:
        return float(centers[0])
    above, drop = means[first - 1], means[first - 1] - means[first]
    span = centers[first] - centers[first - 1]
    return float(centers[first - 1] + span * (above - target) / drop)


def samples(
    representations: dict[str, np.ndarray],
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Every STRIDE-th valid step of the first EPISODES episodes."""
    episodes = min(EPISODES, valid.shape[0])
    keep = valid[:episodes, ::STRIDE]
    episode = np.broadcast_to(np.arange(episodes)[:, None], keep.shape)[keep]
    flat = {
        name: values[:episodes, ::STRIDE][keep].astype(np.float32)
        for name, values in representations.items()
    }
    return (
        flat,
        position_xy[:episodes, ::STRIDE][keep].astype(np.float64),
        heading[:episodes, ::STRIDE][keep].astype(np.float64),
        episode,
    )


def similarity_half_distances(
    representations: dict[str, np.ndarray],
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Half-distance of mean cosine similarity for all, same- and opposite-heading pairs."""
    flat, position, step_heading, episode = samples(representations, position_xy, heading, valid)
    generator = np.random.default_rng(PAIR_SEED)
    left = generator.integers(0, len(position), PAIR_DRAWS)
    right = generator.integers(0, len(position), PAIR_DRAWS)
    different_episode = episode[left] != episode[right]
    left, right = left[different_episode], right[different_episode]
    offset = position[left] - position[right]
    distance = np.hypot(offset[:, 0], offset[:, 1])
    gap = step_heading[left] - step_heading[right]
    heading_difference = np.abs((gap + np.pi) % (2.0 * np.pi) - np.pi)
    selections = {
        "all": None,
        "same_heading": heading_difference < np.deg2rad(SAME_HEADING_DEGREES),
        "opposite_heading": heading_difference > np.deg2rad(OPPOSITE_HEADING_DEGREES),
    }
    result = {"similarity_pairs": int(len(left))}
    for name, vectors in flat.items():
        similarity = cosine_similarity(vectors, left, right)
        for band, selection in selections.items():
            means = binned_mean(distance, similarity, MEAN_EDGES, selection)
            result[f"similarity_half_distance_{name}_{band}"] = half_distance(MEAN_CENTERS, means)
    return result
