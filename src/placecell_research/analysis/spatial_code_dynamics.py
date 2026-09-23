"""Temporal persistence and room-transition diagnostics for spatial codes."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import AnalysisInput, AnalysisResult

_WALLGAP_ROOM_BOUNDS = {
    "MiniWorld-WallGapAsym-v0": {
        "northern_courtyard": (-6.0, 6.0, 2.0, 12.0),
        "central_corridor": (-2.0, 2.0, -2.0, 2.0),
        "southern_yard_left": (-8.0, -2.0, -10.0, 1.5),
        "southern_yard_right": (2.0, 8.0, -10.0, 1.5),
    },
    "MiniWorld-WallGapAsymLarge-v0": {
        "northern_courtyard": (-18.0, 18.0, 6.0, 36.0),
        "central_corridor": (-6.0, 6.0, -6.0, 6.0),
        "southern_yard_left": (-24.0, -6.0, -30.0, 4.5),
        "southern_yard_right": (6.0, 24.0, -30.0, 4.5),
    },
}


def wallgap_room_ids(position_xy: np.ndarray, *, env_id: str) -> np.ndarray:
    """Assign WallGap positions to structural rooms, with corridor boundaries taking priority."""
    positions = np.asarray(position_xy)
    if positions.ndim < 2 or positions.shape[-1] != 2:
        raise ValueError("position_xy must have shape [..., 2].")
    if env_id not in _WALLGAP_ROOM_BOUNDS:
        raise ValueError(f"No structural room definition for environment {env_id!r}.")

    room_ids = np.full(positions.shape[:2], "outside", dtype="<U24")
    x_position = positions[..., 0]
    y_position = positions[..., 1]
    bounds_by_room = _WALLGAP_ROOM_BOUNDS[env_id]
    room_order = (
        "northern_courtyard",
        "southern_yard_left",
        "southern_yard_right",
        "central_corridor",
    )
    for room_name in room_order:
        min_x, max_x, min_y, max_y = bounds_by_room[room_name]
        inside = (
            (x_position >= min_x)
            & (x_position <= max_x)
            & (y_position >= min_y)
            & (y_position <= max_y)
        )
        room_ids[inside] = room_name
    return room_ids


def _cosine_distances(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    previous_norm = np.linalg.norm(previous, axis=-1)
    current_norm = np.linalg.norm(current, axis=-1)
    denominator = previous_norm * current_norm
    both_nonzero = denominator > 1e-12
    cosine = np.zeros_like(denominator, dtype=np.float64)
    cosine[both_nonzero] = (
        np.sum(previous[both_nonzero] * current[both_nonzero], axis=-1) / denominator[both_nonzero]
    )
    cosine = np.clip(cosine, -1.0, 1.0)
    distance = 1.0 - cosine
    both_zero = (previous_norm <= 1e-12) & (current_norm <= 1e-12)
    distance[both_zero] = 0.0
    return distance


def _nearest_matched_values(
    target_displacement: np.ndarray,
    candidate_displacement: np.ndarray,
    candidate_values: np.ndarray,
) -> np.ndarray:
    """Match each target to the candidate with the nearest spatial displacement."""
    order = np.argsort(candidate_displacement)
    sorted_displacement = candidate_displacement[order]
    insertion = np.searchsorted(sorted_displacement, target_displacement)
    upper = np.clip(insertion, 0, len(order) - 1)
    lower = np.clip(insertion - 1, 0, len(order) - 1)
    choose_lower = np.abs(target_displacement - sorted_displacement[lower]) <= np.abs(
        target_displacement - sorted_displacement[upper]
    )
    matched_sorted_index = np.where(choose_lower, lower, upper)
    return candidate_values[order[matched_sorted_index]]


def _metric_threshold_suffix(threshold: float) -> str:
    return f"{threshold:.6g}".replace("-", "m").replace(".", "p")


def compute_lag_code_dynamics(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lag: int,
    support_epsilon: float = 1e-8,
    cosine_event_thresholds: tuple[float, ...] = (0.01, 0.05, 0.1),
    room_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure code change over one lag without pairing across episode boundaries."""
    values = np.asarray(representation)
    positions = np.asarray(position_xy)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("representation and valid_mask must align as [episodes, time, features].")
    if positions.shape != (*values.shape[:2], 2):
        raise ValueError("position_xy must align with representation as [episodes, time, 2].")
    if lag < 1 or lag >= values.shape[1]:
        raise ValueError("lag must be positive and smaller than the sequence length.")
    if support_epsilon < 0.0:
        raise ValueError("support_epsilon must be non-negative.")
    if any(threshold < 0.0 for threshold in cosine_event_thresholds):
        raise ValueError("cosine event thresholds must be non-negative.")
    if room_ids is not None and np.asarray(room_ids).shape != values.shape[:2]:
        raise ValueError("room_ids must align with representation batch and time axes.")

    pair_valid = valid[:, :-lag] & valid[:, lag:]
    pair_valid &= np.isfinite(positions[:, :-lag]).all(axis=-1)
    pair_valid &= np.isfinite(positions[:, lag:]).all(axis=-1)
    previous = values[:, :-lag][pair_valid]
    current = values[:, lag:][pair_valid]
    previous_position = positions[:, :-lag][pair_valid]
    current_position = positions[:, lag:][pair_valid]
    pair_count = int(previous.shape[0])
    if pair_count == 0:
        return {"pair_count": 0.0}

    cosine_distance = _cosine_distances(previous, current)
    l2_distance = np.linalg.norm(current - previous, axis=-1)
    mean_norm = 0.5 * (np.linalg.norm(previous, axis=-1) + np.linalg.norm(current, axis=-1))
    normalized_l2_distance = l2_distance / np.maximum(mean_norm, 1e-12)

    previous_support = np.abs(previous) > support_epsilon
    current_support = np.abs(current) > support_epsilon
    support_changed = np.any(previous_support != current_support, axis=-1)
    support_intersection = np.logical_and(previous_support, current_support).sum(axis=-1)
    support_union = np.logical_or(previous_support, current_support).sum(axis=-1)
    support_jaccard_distance = np.zeros(pair_count, dtype=np.float64)
    nonempty_union = support_union > 0
    support_jaccard_distance[nonempty_union] = 1.0 - (
        support_intersection[nonempty_union] / support_union[nonempty_union]
    )
    position_displacement = np.linalg.norm(current_position - previous_position, axis=-1)

    metrics = {
        "pair_count": float(pair_count),
        "cosine_distance_mean": float(np.mean(cosine_distance)),
        "cosine_distance_median": float(np.median(cosine_distance)),
        "cosine_distance_p90": float(np.quantile(cosine_distance, 0.9)),
        "normalized_l2_distance_mean": float(np.mean(normalized_l2_distance)),
        "support_change_fraction": float(np.mean(support_changed)),
        "support_jaccard_distance_mean": float(np.mean(support_jaccard_distance)),
        "position_displacement_mean": float(np.mean(position_displacement)),
    }
    for threshold in cosine_event_thresholds:
        event_fraction = float(np.mean(cosine_distance >= threshold))
        suffix = _metric_threshold_suffix(threshold)
        metrics[f"cosine_event_fraction_ge_{suffix}"] = event_fraction
        if event_fraction > 0.0:
            metrics[f"cosine_event_mean_interval_steps_ge_{suffix}"] = 1.0 / event_fraction

    if room_ids is not None:
        rooms = np.asarray(room_ids)
        previous_room = rooms[:, :-lag][pair_valid]
        current_room = rooms[:, lag:][pair_valid]
        known_room = (previous_room != "outside") & (current_room != "outside")
        room_transition = known_room & (previous_room != current_room)
        same_room = known_room & (previous_room == current_room)
        metrics["room_transition_pair_count"] = float(np.sum(room_transition))
        metrics["same_room_pair_count"] = float(np.sum(same_room))
        if np.any(room_transition):
            metrics["room_transition_cosine_distance_mean"] = float(
                np.mean(cosine_distance[room_transition])
            )
            metrics["room_transition_position_displacement_mean"] = float(
                np.mean(position_displacement[room_transition])
            )
            metrics["room_transition_support_change_fraction"] = float(
                np.mean(support_changed[room_transition])
            )
        if np.any(same_room):
            metrics["same_room_cosine_distance_mean"] = float(np.mean(cosine_distance[same_room]))
            metrics["same_room_position_displacement_mean"] = float(
                np.mean(position_displacement[same_room])
            )
            metrics["same_room_support_change_fraction"] = float(
                np.mean(support_changed[same_room])
            )
        if np.any(room_transition) and np.any(same_room):
            matched_cosine_distance = _nearest_matched_values(
                position_displacement[room_transition],
                position_displacement[same_room],
                cosine_distance[same_room],
            )
            matched_support_change = _nearest_matched_values(
                position_displacement[room_transition],
                position_displacement[same_room],
                support_changed[same_room].astype(np.float64),
            )
            matched_cosine_mean = float(np.mean(matched_cosine_distance))
            transition_cosine_mean = float(np.mean(cosine_distance[room_transition]))
            metrics["movement_matched_same_room_cosine_distance_mean"] = matched_cosine_mean
            metrics["room_transition_cosine_distance_excess"] = (
                transition_cosine_mean - matched_cosine_mean
            )
            if matched_cosine_mean > 1e-12:
                metrics["room_transition_cosine_distance_ratio"] = (
                    transition_cosine_mean / matched_cosine_mean
                )
            metrics["movement_matched_same_room_support_change_fraction"] = float(
                np.mean(matched_support_change)
            )
    return metrics


def support_dwell_lengths(
    representation: np.ndarray,
    valid_mask: np.ndarray,
    *,
    support_epsilon: float = 1e-8,
) -> np.ndarray:
    """Return lengths of contiguous runs with an unchanged active support set."""
    values = np.asarray(representation)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("representation and valid_mask must align as [episodes, time, features].")
    if support_epsilon < 0.0:
        raise ValueError("support_epsilon must be non-negative.")

    support = np.abs(values) > support_epsilon
    dwell_lengths: list[int] = []
    for episode_support, episode_valid in zip(support, valid, strict=False):
        previous_support: np.ndarray | None = None
        current_length = 0
        for time_support, is_valid in zip(episode_support, episode_valid, strict=False):
            if not is_valid:
                if current_length:
                    dwell_lengths.append(current_length)
                previous_support = None
                current_length = 0
                continue
            if previous_support is not None and np.array_equal(time_support, previous_support):
                current_length += 1
            else:
                if current_length:
                    dwell_lengths.append(current_length)
                current_length = 1
            previous_support = time_support
        if current_length:
            dwell_lengths.append(current_length)
    return np.asarray(dwell_lengths, dtype=np.int64)


def shuffled_support_dwell_lengths(
    representation: np.ndarray,
    valid_mask: np.ndarray,
    *,
    support_epsilon: float = 1e-8,
    seed: int = 0,
) -> np.ndarray:
    """support_dwell_lengths after permuting time WITHIN each episode."""
    values = np.asarray(representation)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("representation and valid_mask must align as [episodes, time, features].")
    if support_epsilon < 0.0:
        raise ValueError("support_epsilon must be non-negative.")

    generator = np.random.default_rng(seed)
    dwell_lengths: list[int] = []
    for episode_values, episode_valid in zip(values, valid, strict=False):
        valid_indices = np.flatnonzero(episode_valid)
        if valid_indices.size == 0:
            continue
        support = np.abs(episode_values[valid_indices]) > support_epsilon
        generator.shuffle(support, axis=0)
        if support.shape[0] == 1:
            dwell_lengths.append(1)
            continue
        changed = np.any(support[1:] != support[:-1], axis=1)
        boundaries = np.flatnonzero(changed) + 1
        edges = np.concatenate(([0], boundaries, [support.shape[0]]))
        dwell_lengths.extend(np.diff(edges).tolist())
    return np.asarray(dwell_lengths, dtype=np.int64)


def _safe_file_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )


@dataclass(slots=True)
class SpatialCodeDynamicsModule:
    """Measure temporal code persistence and concentration at structural room transitions."""

    name: str = "spatial_code_dynamics"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        configured_lags = config.get("spatial_code_dynamics_lags", [1, 2, 4, 8, 16, 32, 64])
        lags = sorted(
            {
                int(lag)
                for lag in configured_lags
                if 0 < int(lag) < analysis_input.representation.shape[1]
            }
        )
        support_epsilon = float(config.get("spatial_code_dynamics_support_epsilon", 1e-8))
        cosine_event_thresholds = tuple(
            float(threshold)
            for threshold in config.get(
                "spatial_code_dynamics_event_thresholds",
                [0.01, 0.05, 0.1],
            )
        )
        env_id = str(analysis_input.metadata.get("env_id", ""))
        room_ids = (
            wallgap_room_ids(analysis_input.position_xy, env_id=env_id)
            if env_id in _WALLGAP_ROOM_BOUNDS
            else None
        )

        metrics: dict[str, float] = {}
        rows: list[dict[str, float]] = []
        for lag in lags:
            lag_metrics = compute_lag_code_dynamics(
                analysis_input.representation,
                analysis_input.position_xy,
                analysis_input.valid_mask,
                lag=lag,
                support_epsilon=support_epsilon,
                cosine_event_thresholds=cosine_event_thresholds,
                room_ids=room_ids,
            )
            rows.append({"lag": float(lag), **lag_metrics})
            metrics.update(
                {f"lag_{lag}_{metric_name}": value for metric_name, value in lag_metrics.items()}
            )

        dwell_lengths = support_dwell_lengths(
            analysis_input.representation,
            analysis_input.valid_mask,
            support_epsilon=support_epsilon,
        )
        if dwell_lengths.size:
            observed_mean_steps = float(np.mean(dwell_lengths))
            metrics.update(
                {
                    "support_dwell_count": float(dwell_lengths.size),
                    "support_dwell_mean_steps": observed_mean_steps,
                    "support_dwell_median_steps": float(np.median(dwell_lengths)),
                    "support_dwell_p90_steps": float(np.quantile(dwell_lengths, 0.9)),
                    "support_dwell_max_steps": float(np.max(dwell_lengths)),
                }
            )
            shuffled_lengths = shuffled_support_dwell_lengths(
                analysis_input.representation,
                analysis_input.valid_mask,
                support_epsilon=support_epsilon,
                seed=int(config.get("dwell_shuffle_seed", 0)),
            )
            if shuffled_lengths.size:
                shuffled_mean_steps = float(np.mean(shuffled_lengths))
                metrics["support_dwell_mean_steps_shuffled"] = shuffled_mean_steps
                metrics["support_dwell_ratio_to_shuffled"] = (
                    observed_mean_steps / shuffled_mean_steps
                    if shuffled_mean_steps > 0.0
                    else float("nan")
                )

        table_directory = output_dir / self.name
        table_directory.mkdir(parents=True, exist_ok=True)
        table_path = table_directory / (
            f"lag_dynamics__{_safe_file_component(analysis_input.source_name)}__"
            f"{_safe_file_component(analysis_input.split_name)}.csv"
        )
        field_names = sorted({key for row in rows for key in row})
        with table_path.open("w", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(rows)

        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={},
            tables={"lag_dynamics": table_path},
            metadata={
                "lags": lags,
                "support_epsilon": support_epsilon,
                "cosine_event_thresholds": list(cosine_event_thresholds),
                "room_transition_analysis": room_ids is not None,
            },
        )
