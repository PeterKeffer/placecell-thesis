"""Linear probes for representation content beyond XY position."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.evaluation.decode import chunked_ridge_fit_predict

from ..numerics.rate_map_kernels import compute_spatial_bin_assignments
from .base import AnalysisInput, AnalysisResult
from .helpers import write_csv
from .world_overlay import resolve_plot_bounds, resolve_world_overlay

ProbeKind = Literal["regression", "heading"]


@dataclass(frozen=True, slots=True)
class _ProbeTarget:
    name: str
    values: np.ndarray
    kind: ProbeKind


@dataclass(frozen=True, slots=True)
class _ProbeScore:
    score: float
    shuffle_score: float | None
    train_size: int
    validation_size: int
    train_episode_count: int
    validation_episode_count: int


@dataclass(slots=True)
class ProbingModule:
    """Episode-held-out linear probes for heading, temporal, and geometry variables."""

    name: str = "probing"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        targets = _build_targets(analysis_input, config)
        if not targets:
            raise ValueError("ProbingModule found no configured probe targets.")

        metrics: dict[str, float] = {}
        metadata: dict[str, object] = {"probe_split_mode": "episode", "probe_splits": {}}
        table_rows: list[list[object]] = []
        for target in targets:
            features, target_values, episode_ids = _flatten_probe_arrays(analysis_input, target)
            score = _fit_and_score_probe(
                features,
                target_values,
                episode_ids,
                target.kind,
                train_fraction=float(config.get("probing_train_fraction", 0.8)),
                ridge_alpha=float(config.get("probing_ridge_alpha", 1e-3)),
                include_shuffle=bool(config.get("probing_include_shuffle", True)),
                random_seed=int(config["probing_shuffle_seed"]),
            )
            metrics[f"probe_{target.name}_score"] = score.score
            if score.shuffle_score is not None:
                metrics[f"probe_{target.name}_shuffle_score"] = score.shuffle_score
            metadata["probe_splits"][target.name] = {
                "kind": target.kind,
                "train_size": score.train_size,
                "validation_size": score.validation_size,
                "train_episode_count": score.train_episode_count,
                "validation_episode_count": score.validation_episode_count,
            }
            table_rows.append(
                [
                    target.name,
                    target.kind,
                    score.score,
                    "" if score.shuffle_score is None else score.shuffle_score,
                    score.train_size,
                    score.validation_size,
                    score.train_episode_count,
                    score.validation_episode_count,
                ]
            )

        module_dir = output_dir / self.name
        figure_path = module_dir / (
            f"probe_scores__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        table_path = module_dir / (
            f"probe_scores__{analysis_input.source_name}__{analysis_input.split_name}.csv"
        )
        _save_probe_score_figure(figure_path, metrics)
        write_csv(
            table_path,
            [
                "probe",
                "kind",
                "score",
                "shuffle_score",
                "train_size",
                "validation_size",
                "train_episode_count",
                "validation_episode_count",
            ],
            table_rows,
        )

        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={"probe_scores": figure_path},
            tables={"probe_scores": table_path},
            metadata=metadata,
        )


def _valid_mask_array(analysis_input: AnalysisInput) -> np.ndarray:
    valid_mask = np.asarray(analysis_input.valid_mask, dtype=bool)
    expected_shape = analysis_input.position_xy.shape[:2]
    if valid_mask.shape != expected_shape:
        raise ValueError(f"Expected valid_mask shape {expected_shape}, got {valid_mask.shape}.")
    return valid_mask


def _build_targets(analysis_input: AnalysisInput, config: dict) -> list[_ProbeTarget]:
    targets: list[_ProbeTarget] = []
    if bool(config.get("probing_include_heading", True)) and analysis_input.heading is not None:
        targets.append(_heading_target(analysis_input))
    if bool(config.get("probing_include_time", True)):
        targets.append(_time_since_start_target(analysis_input))
    max_horizon = analysis_input.position_xy.shape[1] - 1
    for horizon in config.get("probing_future_horizons", [1, 5, 10, 20]):
        if int(horizon) > max_horizon:
            continue
        targets.append(
            _shifted_position_target(
                analysis_input,
                int(horizon),
                name=f"future_xy_k{int(horizon)}",
            )
        )
    for horizon in config.get("probing_past_horizons", [1, 5, 10]):
        if int(horizon) > max_horizon:
            continue
        targets.append(
            _shifted_position_target(
                analysis_input,
                -int(horizon),
                name=f"past_xy_k{int(horizon)}",
            )
        )
    if bool(config.get("probing_include_wall_distance", True)):
        wall_distance = _nearest_wall_distance_target(analysis_input)
        if wall_distance is not None:
            targets.append(wall_distance)
    if bool(config.get("probing_include_landmark_distances", True)):
        targets.extend(_landmark_distance_targets(analysis_input))
    if bool(config.get("probing_include_novelty", True)):
        targets.append(_novelty_target(analysis_input, config))
    return targets


def _heading_target(analysis_input: AnalysisInput) -> _ProbeTarget:
    heading = np.asarray(analysis_input.heading, dtype=np.float32)
    expected_shape = analysis_input.position_xy.shape[:2]
    if heading.shape != expected_shape:
        heading = heading.reshape(expected_shape)
    values = np.stack([np.cos(heading), np.sin(heading)], axis=-1).astype(np.float32)
    return _ProbeTarget(name="heading", values=values, kind="heading")


def _time_since_start_target(analysis_input: AnalysisInput) -> _ProbeTarget:
    episodes, steps = analysis_input.position_xy.shape[:2]
    denominator = float(max(1, steps - 1))
    values = np.broadcast_to(
        (np.arange(steps, dtype=np.float32) / denominator)[None, :, None],
        (episodes, steps, 1),
    ).copy()
    return _ProbeTarget(name="time_since_start", values=values, kind="regression")


def _shifted_position_target(
    analysis_input: AnalysisInput,
    horizon: int,
    *,
    name: str,
) -> _ProbeTarget:
    if horizon == 0:
        raise ValueError("Probe horizon must be non-zero.")
    positions = np.asarray(analysis_input.position_xy, dtype=np.float32)
    valid_mask = _valid_mask_array(analysis_input)
    shifted = np.full_like(positions, np.nan, dtype=np.float32)
    if horizon > 0:
        shifted[:, :-horizon] = np.where(
            valid_mask[:, horizon:, None],
            positions[:, horizon:],
            np.nan,
        )
    else:
        offset = abs(horizon)
        shifted[:, offset:] = np.where(
            valid_mask[:, :-offset, None],
            positions[:, :-offset],
            np.nan,
        )
    return _ProbeTarget(name=name, values=shifted, kind="regression")


def _nearest_wall_distance_target(analysis_input: AnalysisInput) -> _ProbeTarget | None:
    overlay = resolve_world_overlay(
        str(analysis_input.metadata.get("env_id", "")),
        analysis_input.metadata.get("env_kwargs"),
    )
    if overlay is None or not overlay.segments:
        return None
    distances = distance_to_segments(analysis_input.position_xy, overlay.segments)
    return _ProbeTarget(
        name="distance_nearest_wall",
        values=distances[..., None].astype(np.float32, copy=False),
        kind="regression",
    )


def _landmark_distance_targets(analysis_input: AnalysisInput) -> list[_ProbeTarget]:
    overlay = resolve_world_overlay(
        str(analysis_input.metadata.get("env_id", "")),
        analysis_input.metadata.get("env_kwargs"),
    )
    if overlay is None:
        return []
    targets: list[_ProbeTarget] = []
    for layer in overlay.landmarks:
        if not layer.positions:
            continue
        distances = distance_to_points(analysis_input.position_xy, layer.positions)
        targets.append(
            _ProbeTarget(
                name=f"distance_{_sanitize_probe_name(layer.label)}",
                values=distances[..., None].astype(np.float32, copy=False),
                kind="regression",
            )
        )
    return targets


def _novelty_target(analysis_input: AnalysisInput, config: dict) -> _ProbeTarget:
    valid_mask = _valid_mask_array(analysis_input)
    positions = np.asarray(analysis_input.position_xy, dtype=np.float32)
    episodes, steps = positions.shape[:2]
    values = np.full((episodes, steps, 1), np.nan, dtype=np.float32)
    env_id = str(analysis_input.metadata.get("env_id", ""))
    bounds = resolve_plot_bounds(env_id, positions.reshape(-1, 2))
    num_bins_x = int(config["probing_novelty_num_bins_x"])
    num_bins_y = int(config["probing_novelty_num_bins_y"])
    for episode_index in range(episodes):
        linear_bins, _, _, _ = compute_spatial_bin_assignments(
            positions[episode_index],
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
        )
        last_seen: dict[int, int] = {}
        for step_index, bin_index in enumerate(linear_bins.tolist()):
            if not valid_mask[episode_index, step_index]:
                continue
            previous_step = last_seen.get(int(bin_index))
            if previous_step is None:
                values[episode_index, step_index, 0] = float(step_index + 1)
            else:
                values[episode_index, step_index, 0] = float(step_index - previous_step)
            last_seen[int(bin_index)] = step_index
    return _ProbeTarget(name="novelty_steps_since_bin_visit", values=values, kind="regression")


def distance_to_segments(
    positions: np.ndarray,
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
) -> np.ndarray:
    flat_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    starts = np.asarray([segment[0] for segment in segments], dtype=np.float32)
    ends = np.asarray([segment[1] for segment in segments], dtype=np.float32)
    segment_vectors = ends - starts
    segment_lengths = np.maximum(np.sum(np.square(segment_vectors), axis=1), 1e-12)
    relative_positions = flat_positions[:, None, :] - starts[None, :, :]
    projection = (
        np.sum(relative_positions * segment_vectors[None, :, :], axis=2)
        / segment_lengths[None, :]
    )
    clipped_projection = np.clip(projection, 0.0, 1.0)
    closest_points = (
        starts[None, :, :]
        + clipped_projection[..., None] * segment_vectors[None, :, :]
    )
    distances = np.linalg.norm(flat_positions[:, None, :] - closest_points, axis=2)
    return np.min(distances, axis=1).reshape(positions.shape[:2]).astype(np.float32)


def distance_to_points(
    positions: np.ndarray,
    points: tuple[tuple[float, float], ...],
) -> np.ndarray:
    flat_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    point_array = np.asarray(points, dtype=np.float32)
    distances = np.linalg.norm(flat_positions[:, None, :] - point_array[None, :, :], axis=2)
    return np.min(distances, axis=1).reshape(positions.shape[:2]).astype(np.float32)


def _sanitize_probe_name(name: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in name.strip())
    return normalized.strip("_").lower() or "landmark"


def _flatten_probe_arrays(
    analysis_input: AnalysisInput,
    target: _ProbeTarget,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_shape = analysis_input.position_xy.shape[:2]
    values = target.values
    if values.shape[:2] != expected_shape:
        raise ValueError(
            f"Probe target {target.name!r} must start with shape {expected_shape}, "
            f"got {values.shape}."
        )
    if values.ndim == 2:
        values = values[..., None]

    flat_features = analysis_input.representation.reshape(
        -1,
        analysis_input.representation.shape[-1],
    )
    flat_values = values.reshape(-1, values.shape[-1]).astype(np.float32, copy=False)
    flat_valid_mask = _valid_mask_array(analysis_input).reshape(-1)
    flat_episode_ids = np.repeat(
        np.arange(expected_shape[0], dtype=np.int64),
        expected_shape[1],
    )
    finite_rows = (
        flat_valid_mask
        & np.all(np.isfinite(flat_features), axis=1)
        & np.all(np.isfinite(flat_values), axis=1)
    )
    if int(np.count_nonzero(finite_rows)) < 4:
        raise ValueError(f"Probe target {target.name!r} has fewer than four valid samples.")
    return (
        flat_features[finite_rows].astype(np.float32, copy=False),
        flat_values[finite_rows],
        flat_episode_ids[finite_rows],
    )


def episode_split_indices(
    episode_ids: np.ndarray,
    *,
    train_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_episode_ids = np.unique(episode_ids)
    if len(unique_episode_ids) < 2:
        raise ValueError("Probing requires at least two episodes for episode-level holdout.")
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(unique_episode_ids)
    split = int(max(1, min(len(shuffled) - 1, round(len(shuffled) * train_fraction))))
    train_episode_ids = set(int(episode_id) for episode_id in shuffled[:split])
    train_mask = np.asarray([int(episode_id) in train_episode_ids for episode_id in episode_ids])
    indices = np.arange(len(episode_ids), dtype=np.int64)
    return indices[train_mask], indices[~train_mask]


def _fit_and_score_probe(
    features: np.ndarray,
    target_values: np.ndarray,
    episode_ids: np.ndarray,
    kind: ProbeKind,
    *,
    train_fraction: float,
    ridge_alpha: float,
    include_shuffle: bool,
    random_seed: int,
) -> _ProbeScore:
    train_indices, validation_indices = episode_split_indices(
        episode_ids,
        train_fraction=train_fraction,
        random_seed=random_seed,
    )
    predictions = chunked_ridge_fit_predict(
        features,
        target_values,
        train_indices,
        validation_indices,
        ridge_alpha,
    )
    score = _score_predictions(target_values[validation_indices], predictions, kind)

    shuffle_score: float | None = None
    if include_shuffle:
        rng = np.random.default_rng(random_seed + 1)
        shuffled_values = target_values[rng.permutation(len(target_values))]
        shuffle_predictions = chunked_ridge_fit_predict(
            features,
            shuffled_values,
            train_indices,
            validation_indices,
            ridge_alpha,
        )
        shuffle_score = _score_predictions(
            shuffled_values[validation_indices],
            shuffle_predictions,
            kind,
        )

    return _ProbeScore(
        score=score,
        shuffle_score=shuffle_score,
        train_size=int(len(train_indices)),
        validation_size=int(len(validation_indices)),
        train_episode_count=int(len(np.unique(episode_ids[train_indices]))),
        validation_episode_count=int(len(np.unique(episode_ids[validation_indices]))),
    )


def _score_predictions(
    target_values: np.ndarray,
    predictions: np.ndarray,
    kind: ProbeKind,
) -> float:
    if kind == "heading":
        return _mean_heading_cosine(target_values, predictions)
    return _r2_score(target_values, predictions)


def _mean_heading_cosine(target_values: np.ndarray, predictions: np.ndarray) -> float:
    target_norm = np.maximum(np.linalg.norm(target_values, axis=1), 1e-8)
    prediction_norm = np.maximum(np.linalg.norm(predictions, axis=1), 1e-8)
    target_unit = target_values / target_norm[:, None]
    prediction_unit = predictions / prediction_norm[:, None]
    return float(np.mean(np.sum(target_unit * prediction_unit, axis=1)))


def _r2_score(target_values: np.ndarray, predictions: np.ndarray) -> float:
    residual_sum = float(np.sum(np.square(target_values - predictions)))
    centered = target_values - target_values.mean(axis=0, keepdims=True)
    total_sum = float(np.sum(np.square(centered)))
    if total_sum <= 1e-12:
        return 0.0
    return float(1.0 - residual_sum / total_sum)


def _save_probe_score_figure(path: Path, metrics: dict[str, float]) -> Path:
    observed_items = [
        (key.removeprefix("probe_").removesuffix("_score"), value)
        for key, value in sorted(metrics.items())
        if (
            key.startswith("probe_")
            and key.endswith("_score")
            and not key.endswith("_shuffle_score")
        )
    ]
    if not observed_items:
        raise ValueError("Cannot render probe figure without observed scores.")
    labels = [name for name, _value in observed_items]
    observed = np.asarray([value for _name, value in observed_items], dtype=np.float32)
    shuffled = np.asarray(
        [
            metrics.get(f"probe_{name}_shuffle_score", np.nan)
            for name in labels
        ],
        dtype=np.float32,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure_height = max(3.8, 0.36 * len(labels) + 1.2)
    figure, axis = plt.subplots(figsize=(7.2, figure_height))
    y_positions = np.arange(len(labels), dtype=np.float32)
    axis.barh(y_positions - 0.18, observed, height=0.32, color="#3366AA", label="observed")
    if np.any(np.isfinite(shuffled)):
        axis.barh(y_positions + 0.18, shuffled, height=0.32, color="#AAAAAA", label="shuffled")
    axis.axvline(0.0, color="#202020", linewidth=0.8)
    axis.set_yticks(y_positions, labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("Probe score (R2, heading uses mean cosine)")
    axis.set_title("Episode-held-out linear probes")
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.4, alpha=0.8)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
