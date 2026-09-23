"""Decode error as a function of time since the last landmark sighting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.evaluation.decode import (
    chunked_ridge_predict,
    episode_level_decode_skip_reason,
    fit_position_ridge_decoder,
)

from .base import AnalysisInput, AnalysisResult
from .figures import apply_publication_style, despine
from .world_overlay import resolve_world_overlay

FOV_TOTAL_ANGLE_DEGREES = 60.0
MAX_VISIBLE_DISTANCE = 20.0
_BIN_LOWER_EDGES = (1, 5, 17, 65)
_BIN_LABELS = ("0", "1-4", "5-16", "17-64", ">64")
_BIN_METRIC_SUFFIXES = ("0", "1_4", "5_16", "17_64", "over_64")
_EPS = 1e-9


def _cross_z(a_x: np.ndarray, a_y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray) -> np.ndarray:
    return a_x * b_y - a_y * b_x


def landmark_visibility(
    positions_xy: np.ndarray,
    headings: np.ndarray,
    landmarks_xy: np.ndarray,
    wall_segments: np.ndarray,
) -> np.ndarray:
    """Boolean visibility matrix [num_steps, num_landmarks]."""
    positions = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    forward = np.stack(
        [np.cos(np.asarray(headings, dtype=np.float64).reshape(-1)),
         -np.sin(np.asarray(headings, dtype=np.float64).reshape(-1))],
        axis=1,
    )
    walls_a = np.asarray(wall_segments, dtype=np.float64)[:, 0, :]
    walls_b = np.asarray(wall_segments, dtype=np.float64)[:, 1, :]
    wall_direction = walls_b - walls_a
    cos_half_fov = math.cos(math.radians(FOV_TOTAL_ANGLE_DEGREES) / 2.0)

    visible = np.zeros((len(positions), len(landmarks_xy)), dtype=bool)
    for landmark_index, landmark in enumerate(np.asarray(landmarks_xy, dtype=np.float64)):
        delta = landmark[None, :] - positions
        distance = np.linalg.norm(delta, axis=1)
        in_range = (distance > _EPS) & (distance < MAX_VISIBLE_DISTANCE)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_offset = (delta * forward).sum(axis=1) / distance
        candidates = np.flatnonzero(in_range & (cos_offset >= cos_half_fov))
        if candidates.size == 0:
            continue
        agent = positions[candidates][:, None, :]
        sight = delta[candidates][:, None, :]
        to_wall_a = walls_a[None, :, :] - agent
        to_wall_b = walls_b[None, :, :] - agent
        side_a = _cross_z(sight[..., 0], sight[..., 1], to_wall_a[..., 0], to_wall_a[..., 1])
        side_b = _cross_z(sight[..., 0], sight[..., 1], to_wall_b[..., 0], to_wall_b[..., 1])
        agent_from_a = agent - walls_a[None, :, :]
        landmark_from_a = landmark[None, :] - walls_a
        side_agent = _cross_z(
            wall_direction[None, :, 0], wall_direction[None, :, 1],
            agent_from_a[..., 0], agent_from_a[..., 1],
        )
        side_landmark = _cross_z(
            wall_direction[:, 0], wall_direction[:, 1],
            landmark_from_a[:, 0], landmark_from_a[:, 1],
        )
        blocked = np.any(
            (side_a * side_b < 0.0) & (side_agent * side_landmark[None, :] < 0.0), axis=1
        )
        visible[candidates[~blocked], landmark_index] = True
    return visible


def _steps_since_last_sighting(visible_any: np.ndarray) -> np.ndarray:
    """Per-step steps since the last sighting, NaN before an episode's first sighting."""
    num_episodes, num_steps = visible_any.shape
    step_axis = np.arange(num_steps)
    last_seen = np.where(visible_any, step_axis[None, :], -1)
    last_seen = np.maximum.accumulate(last_seen, axis=1)
    steps_since = (step_axis[None, :] - last_seen).astype(np.float64)
    steps_since[last_seen < 0] = np.nan
    return steps_since


def _skip(reason: str) -> AnalysisResult:
    return AnalysisResult(
        metrics={"landmark_visibility_error_skipped": 1.0},
        per_unit_metrics={},
        figures={},
        tables={},
        metadata={"landmark_visibility_error_skip_reason": reason},
    )


@dataclass(slots=True)
class LandmarkVisibilityErrorModule:
    """Held-out decode error vs steps since the last landmark was in view."""

    name: str = "landmark_visibility_error"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        representation = analysis_input.representation
        if representation.ndim != 3:
            raise ValueError(f"Expected representation [N, T, D], got {representation.shape}.")
        num_episodes, num_steps, _ = representation.shape
        if analysis_input.heading is None:
            return _skip("Heading is required to reconstruct landmark visibility.")
        env_id = str(analysis_input.metadata.get("env_id", ""))
        world_overlay = resolve_world_overlay(env_id, analysis_input.metadata.get("env_kwargs"))
        if world_overlay is None:
            return _skip(f"No world overlay registered for env_id '{env_id}'.")
        if world_overlay.y_down:
            return _skip(
                f"Overlay for '{env_id}' is a tile-grid backend; heading convention unverified."
            )
        landmark_list = [
            position
            for layer in world_overlay.landmarks
            if layer.label != "goal"
            for position in layer.positions
        ]
        if not landmark_list:
            return _skip(f"Overlay for '{env_id}' has no landmark positions.")
        landmarks_xy = np.asarray(landmark_list, dtype=np.float64)
        wall_segments = np.asarray(world_overlay.segments, dtype=np.float64)

        flat_codes = representation.reshape(-1, representation.shape[-1])
        flat_positions = analysis_input.position_xy.reshape(-1, 2)
        flat_headings = analysis_input.heading.reshape(-1)
        if analysis_input.valid_mask is None:
            valid_rows = np.arange(num_episodes * num_steps, dtype=np.int64)
        else:
            valid_rows = np.flatnonzero(
                analysis_input.valid_mask.reshape(-1).astype(bool, copy=False)
            ).astype(np.int64, copy=False)
        codes = flat_codes[valid_rows].astype(np.float32, copy=False)
        positions = flat_positions[valid_rows].astype(np.float32, copy=False)
        episode_ids = valid_rows // num_steps

        skip_reason = episode_level_decode_skip_reason(episode_ids)
        if skip_reason is not None:
            return _skip(skip_reason)

        visible_per_landmark = landmark_visibility(
            positions, flat_headings[valid_rows], landmarks_xy, wall_segments
        )
        visible_any_grid = np.zeros((num_episodes * num_steps,), dtype=bool)
        visible_any_grid[valid_rows] = visible_per_landmark.any(axis=1)
        steps_since_grid = _steps_since_last_sighting(
            visible_any_grid.reshape(num_episodes, num_steps)
        ).reshape(-1)
        fraction_visible = float(visible_any_grid[valid_rows].mean())
        if fraction_visible == 0.0:
            return _skip("No landmark sighting under the FOV/distance/occlusion model.")

        decoder = fit_position_ridge_decoder(
            codes,
            positions,
            train_fraction=float(config.get("decode_train_fraction", 0.8)),
            alpha=float(config.get("decode_ridge_alpha", 1e-3)),
            episode_ids=episode_ids,
        )
        validation_indices = decoder.validation_indices
        predictions = chunked_ridge_predict(decoder.ridge, codes, validation_indices)
        errors = np.linalg.norm(predictions - positions[validation_indices], axis=1)
        steps_since = steps_since_grid[valid_rows[validation_indices]]
        uncensored = np.isfinite(steps_since)
        censored_fraction = float(1.0 - uncensored.mean()) if len(steps_since) else float("nan")
        errors, steps_since = errors[uncensored], steps_since[uncensored]
        if len(errors) < 2:
            return _skip("Too few uncensored held-out steps after the first sighting.")

        bin_index = np.digitize(steps_since, _BIN_LOWER_EDGES)
        bin_means = np.full(len(_BIN_LABELS), np.nan)
        bin_quartiles = np.full((2, len(_BIN_LABELS)), np.nan)
        bin_counts = np.zeros(len(_BIN_LABELS), dtype=np.int64)
        for index in range(len(_BIN_LABELS)):
            in_bin = bin_index == index
            bin_counts[index] = int(in_bin.sum())
            if bin_counts[index]:
                bin_means[index] = float(errors[in_bin].mean())
                bin_quartiles[:, index] = np.percentile(errors[in_bin], [25.0, 75.0])

        log_steps = np.log1p(steps_since)
        if len(np.unique(log_steps)) >= 2:
            slope = float(np.polyfit(log_steps, errors, deg=1)[0])
        else:
            slope = float("nan")

        module_dir = output_dir / self.name
        figure_path = (
            module_dir
            / f"landmark_visibility_error__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        apply_publication_style()
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        bin_axis = np.arange(len(_BIN_LABELS))
        plot_mask = np.isfinite(bin_means)
        axis.fill_between(
            bin_axis[plot_mask],
            bin_quartiles[0, plot_mask],
            bin_quartiles[1, plot_mask],
            color="#4C72B0",
            alpha=0.2,
            linewidth=0.0,
            label="interquartile range",
        )
        axis.plot(
            bin_axis[plot_mask],
            bin_means[plot_mask],
            color="#4C72B0",
            linewidth=1.6,
            marker="o",
            markersize=4.0,
            label="mean over held-out steps",
        )
        axis.set_xticks(bin_axis)
        axis.set_xticklabels(_BIN_LABELS)
        axis.set_xlabel("Steps since last landmark sighting")
        axis.set_ylabel("Decode error (arena units)")
        axis.legend(frameon=False, fontsize=8)
        despine(axis)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        metrics = {
            f"mean_error_steps_since_{suffix}": float(bin_means[index])
            for index, suffix in enumerate(_BIN_METRIC_SUFFIXES)
        }
        metrics["error_vs_log_steps_since_slope"] = slope
        metrics["fraction_steps_landmark_visible"] = fraction_visible
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={"landmark_visibility_error_curve": figure_path},
            tables={},
            metadata={
                "env_id": env_id,
                "fov_total_angle_degrees": FOV_TOTAL_ANGLE_DEGREES,
                "max_visible_distance": MAX_VISIBLE_DISTANCE,
                "num_landmarks": int(len(landmarks_xy)),
                "num_wall_segments": int(len(wall_segments)),
                "steps_since_bin_labels": list(_BIN_LABELS),
                "steps_since_bin_counts": [int(count) for count in bin_counts],
                "censored_heldout_fraction": censored_fraction,
                "num_validation_steps_uncensored": int(len(errors)),
            },
        )
