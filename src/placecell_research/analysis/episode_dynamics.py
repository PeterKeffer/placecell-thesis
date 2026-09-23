"""Example-episode dynamics visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .base import AnalysisInput, AnalysisResult
from .figures import figure_to_rgb_array, save_gif
from .world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    resolve_world_overlay,
)


def _normalize_observation_frame(
    rgb_frame: np.ndarray | None, latent_frame: np.ndarray | None
) -> np.ndarray:
    if rgb_frame is not None:
        frame = rgb_frame
        if frame.ndim == 3 and frame.shape[0] in {1, 3}:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.max() > 1.0:
            frame = np.clip(frame / 255.0, 0.0, 1.0)
        return frame.astype(np.float32, copy=False)
    if latent_frame is not None:
        vector = latent_frame.astype(np.float32, copy=False)
        vector = vector.reshape(1, -1)
        vector_min = float(vector.min())
        vector_max = float(vector.max())
        if vector_max - vector_min < 1e-8:
            normalized = np.zeros_like(vector)
        else:
            normalized = (vector - vector_min) / (vector_max - vector_min)
        return normalized
    return np.zeros((16, 16), dtype=np.float32)


def _top_unit_indices(representation: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.mean(np.abs(representation), axis=0)
    top_k = max(1, min(top_k, scores.shape[0]))
    return np.argsort(scores)[::-1][:top_k]


def _selected_unit_indices(representation: np.ndarray, requested_top_k: int) -> np.ndarray:
    if requested_top_k <= 0 or requested_top_k >= representation.shape[1]:
        return np.arange(representation.shape[1], dtype=np.int32)
    return _top_unit_indices(representation, top_k=requested_top_k).astype(np.int32, copy=False)


def _select_episode_index(
    valid_mask: np.ndarray,
    *,
    requested_episode_index: int,
    random_seed: int | None,
) -> int:
    valid_lengths = valid_mask.sum(axis=1).astype(int)
    if valid_lengths.size == 0 or int(valid_lengths.max(initial=0)) == 0:
        raise ValueError("No valid timesteps available for example-episode visualization.")
    if requested_episode_index >= 0:
        return min(requested_episode_index, valid_mask.shape[0] - 1)
    valid_episode_indices = np.flatnonzero(valid_lengths > 0)
    if valid_episode_indices.size == 0:
        raise ValueError("No valid episodes are available for example-episode visualization.")
    random_number_generator = (
        np.random.default_rng() if random_seed is None else np.random.default_rng(random_seed)
    )
    return int(random_number_generator.choice(valid_episode_indices))


def _choose_unit_grid_shape(num_units: int) -> tuple[int, int]:
    if num_units <= 0:
        return 0, 0
    best_rows = 1
    best_columns = num_units
    best_score = float("inf")
    for rows in range(1, num_units + 1):
        columns = int(np.ceil(num_units / rows))
        aspect_score = abs((columns / rows) - 1.25)
        waste_score = (rows * columns - num_units) * 0.05
        score = aspect_score + waste_score
        if score < best_score:
            best_rows = rows
            best_columns = columns
            best_score = score
    return best_rows, best_columns


def _activation_grid(values: np.ndarray, *, rows: int, columns: int) -> np.ndarray:
    grid = np.full((rows, columns), np.nan, dtype=np.float32)
    flat_grid = grid.reshape(-1)
    flat_grid[: values.size] = values.astype(np.float32, copy=False)
    return grid


def _activation_style(values: np.ndarray) -> tuple[str, colors.Normalize]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "turbo", colors.Normalize(vmin=0.0, vmax=1.0)
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    if min_value < -1e-6:
        limit = max(abs(min_value), abs(max_value), 1e-6)
        return "seismic", colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    return "turbo", colors.Normalize(vmin=0.0, vmax=max(max_value, 1e-6))


def _draw_heading_arrow(
    axis: plt.Axes,
    positions: np.ndarray,
    headings: np.ndarray | None,
    current_index: int,
) -> None:
    if headings is None or current_index >= len(headings):
        return
    heading_value = float(headings[current_index])
    if not np.isfinite(heading_value):
        return
    span = np.ptp(positions, axis=0)
    arrow_length = max(float(np.max(span)) * 0.06, 0.2)
    origin_x = float(positions[current_index, 0])
    origin_y = float(positions[current_index, 1])
    delta_x = float(np.cos(heading_value) * arrow_length)
    delta_y = float(np.sin(heading_value) * arrow_length)
    axis.arrow(
        origin_x,
        origin_y,
        delta_x,
        delta_y,
        color="#f28e2b",
        width=max(arrow_length * 0.04, 0.01),
        head_width=max(arrow_length * 0.18, 0.08),
        head_length=max(arrow_length * 0.18, 0.08),
        length_includes_head=True,
        zorder=5,
    )


def _draw_heading_compass(axis: plt.Axes, heading_value: float | None) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-1.1, 1.1)
    axis.set_ylim(-1.1, 1.1)
    axis.axis("off")
    compass = plt.Circle((0.0, 0.0), 1.0, edgecolor="#7a7a7a", facecolor="none", linewidth=1.0)
    axis.add_patch(compass)
    axis.text(0.0, 1.05, "N", ha="center", va="bottom", fontsize=8)
    axis.text(1.08, 0.0, "E", ha="left", va="center", fontsize=8)
    axis.text(0.0, -1.08, "S", ha="center", va="top", fontsize=8)
    axis.text(-1.08, 0.0, "W", ha="right", va="center", fontsize=8)
    if heading_value is None or not np.isfinite(heading_value):
        axis.set_title("Head direction unavailable", fontsize=9)
        return
    delta_x = float(np.cos(heading_value) * 0.82)
    delta_y = float(np.sin(heading_value) * 0.82)
    axis.arrow(
        0.0,
        0.0,
        delta_x,
        delta_y,
        color="#f28e2b",
        width=0.04,
        head_width=0.18,
        head_length=0.18,
        length_includes_head=True,
        zorder=3,
    )
    heading_degrees = float(np.degrees(heading_value))
    axis.set_title(f"Head direction {heading_degrees:.1f} deg", fontsize=9)


def _episode_frame(
    source_name: str,
    split_name: str,
    env_id: str | None,
    positions: np.ndarray,
    headings: np.ndarray | None,
    current_index: int,
    observation_frame: np.ndarray,
    current_values: np.ndarray,
    selected_unit_indices: np.ndarray,
    activation_grid_rows: int,
    activation_grid_columns: int,
    activation_cmap_name: str,
    activation_norm: colors.Normalize,
) -> np.ndarray:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.6))

    axes[0, 0].imshow(
        observation_frame, cmap="magma" if observation_frame.ndim == 2 else None, aspect="auto"
    )
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].plot(positions[:, 0], positions[:, 1], color="#5b84c4", alpha=0.7)
    axes[0, 1].scatter(
        positions[current_index, 0], positions[current_index, 1], color="#d04d4d", s=45
    )
    world_overlay = resolve_world_overlay(env_id)
    if world_overlay is not None:
        draw_world_segments_on_axis(axes[0, 1], world_overlay.segments, line_color="#404040")
        draw_landmarks_on_axis(axes[0, 1], world_overlay)
    apply_plot_bounds(axes[0, 1], env_id=env_id, position_xy=positions)
    _draw_heading_arrow(axes[0, 1], positions, headings, current_index)
    axes[0, 1].set_title(f"Trajectory step {current_index + 1}/{len(positions)}")
    axes[0, 1].set_xlabel(POSITION_X_LABEL)
    axes[0, 1].set_ylabel(POSITION_Y_LABEL)
    axes[0, 1].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    heading_value = (
        None
        if headings is None or current_index >= len(headings)
        else float(headings[current_index])
    )
    _draw_heading_compass(axes[1, 0], heading_value)

    activation_grid = _activation_grid(
        current_values,
        rows=activation_grid_rows,
        columns=activation_grid_columns,
    )
    image = axes[1, 1].imshow(
        np.ma.masked_invalid(activation_grid),
        origin="lower",
        aspect="equal",
        cmap=activation_cmap_name,
        norm=activation_norm,
    )
    axes[1, 1].set_title(f"Neuron activations ({len(selected_unit_indices)} units)")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    figure.colorbar(image, ax=axes[1, 1], shrink=0.8, fraction=0.05, pad=0.02)
    figure.suptitle(
        f"{source_name} example episode ({split_name})\n"
        f"all-neuron activation grid, step {current_index + 1}/{len(positions)}",
        fontsize=12,
    )
    figure.tight_layout()
    return figure_to_rgb_array(figure)


@dataclass(slots=True)
class EpisodeDynamicsModule:
    """Animate one example episode with inputs and representation dynamics."""

    name: str = "episode_dynamics"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def required_batch_keys(self) -> set[str]:
        return {"rgb", "latent"}

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        requested_episode_index = int(config.get("example_episode_index", -1))
        random_seed = config.get("example_episode_random_seed")
        episode_index = _select_episode_index(
            analysis_input.valid_mask,
            requested_episode_index=requested_episode_index,
            random_seed=None if random_seed in {None, ""} else int(random_seed),
        )
        valid_lengths = analysis_input.valid_mask.sum(axis=1).astype(int)
        episode_length = int(valid_lengths[episode_index])

        top_k = int(config.get("example_episode_top_k", 0))
        episode_representation = analysis_input.representation[episode_index, :episode_length]
        selected_unit_indices = _selected_unit_indices(
            episode_representation, requested_top_k=top_k
        )
        selected_activations = episode_representation[:, selected_unit_indices]
        activation_grid_rows, activation_grid_columns = _choose_unit_grid_shape(
            len(selected_unit_indices)
        )
        activation_cmap_name, activation_norm = _activation_style(selected_activations)
        positions = analysis_input.position_xy[episode_index, :episode_length]
        headings = (
            None
            if analysis_input.heading is None
            else analysis_input.heading[episode_index, :episode_length]
        )
        rgb_episode = (
            None
            if analysis_input.rgb is None
            else analysis_input.rgb[episode_index, :episode_length]
        )
        latent_episode = (
            None
            if analysis_input.latent is None
            else analysis_input.latent[episode_index, :episode_length]
        )

        module_dir = output_dir / self.name
        gif_path = (
            module_dir
            / f"example_episode__{analysis_input.source_name}__{analysis_input.split_name}.gif"
        )
        summary_path = (
            module_dir
            / f"example_episode_summary__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )

        frames: list[np.ndarray] = []
        for timestep in range(episode_length):
            observation_frame = _normalize_observation_frame(
                None if rgb_episode is None else rgb_episode[timestep],
                None if latent_episode is None else latent_episode[timestep],
            )
            frames.append(
                _episode_frame(
                    analysis_input.source_name,
                    analysis_input.split_name,
                    str(analysis_input.metadata.get("env_id", "")),
                    positions,
                    headings,
                    timestep,
                    observation_frame,
                    selected_activations[timestep],
                    selected_unit_indices,
                    activation_grid_rows,
                    activation_grid_columns,
                    activation_cmap_name,
                    activation_norm,
                )
            )
        save_gif(
            gif_path, frames, duration=float(config.get("example_episode_frame_duration", 0.14))
        )

        summary_figure, axes = plt.subplots(1, 4, figsize=(15, 4))
        summary_figure.suptitle(
            f"{analysis_input.source_name} example episode summary", fontsize=12
        )
        summary_observation = _normalize_observation_frame(
            None if rgb_episode is None else rgb_episode[0],
            None if latent_episode is None else latent_episode[0],
        )
        axes[0].imshow(
            summary_observation,
            cmap="magma" if summary_observation.ndim == 2 else None,
            aspect="auto",
        )
        axes[0].set_title("First input")
        axes[0].axis("off")
        axes[1].plot(positions[:, 0], positions[:, 1], color="#5b84c4")
        axes[1].scatter(positions[0, 0], positions[0, 1], color="#4caf50", s=40, label="start")
        axes[1].scatter(positions[-1, 0], positions[-1, 1], color="#d04d4d", s=40, label="end")
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        if world_overlay is not None:
            draw_world_segments_on_axis(axes[1], world_overlay.segments, line_color="#404040")
            draw_landmarks_on_axis(axes[1], world_overlay)
        apply_plot_bounds(
            axes[1],
            env_id=str(analysis_input.metadata.get("env_id", "")),
            position_xy=positions,
        )
        _draw_heading_arrow(axes[1], positions, headings, len(positions) - 1)
        axes[1].set_title("Episode trajectory")
        axes[1].set_xlabel(POSITION_X_LABEL)
        axes[1].set_ylabel(POSITION_Y_LABEL)
        axes[1].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)
        axes[1].legend(loc="best")

        final_heading = None if headings is None or len(headings) == 0 else float(headings[-1])
        _draw_heading_compass(axes[2], final_heading)

        activation_image = axes[3].imshow(
            selected_activations.T,
            origin="lower",
            aspect="auto",
            cmap=activation_cmap_name,
            norm=activation_norm,
        )
        axes[3].set_title("Selected-unit activations over time")
        axes[3].set_xlabel("time step")
        axes[3].set_ylabel("unit rank")
        axes[3].set_yticks([])
        summary_figure.colorbar(activation_image, ax=axes[3], shrink=0.8)
        summary_figure.tight_layout()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_figure.savefig(summary_path, dpi=160)
        plt.close(summary_figure)

        return AnalysisResult(
            metrics={
                "selected_episode_index": float(episode_index),
                "selected_episode_length": float(episode_length),
                "mean_top_unit_activation": float(np.mean(np.abs(selected_activations))),
                "selected_unit_count": float(len(selected_unit_indices)),
            },
            per_unit_metrics={
                "top_unit_indices": selected_unit_indices.astype(np.float32),
                "selected_unit_indices": selected_unit_indices.astype(np.float32),
            },
            figures={
                "example_episode_gif": gif_path,
                "example_episode_summary": summary_path,
            },
            tables={},
        )
