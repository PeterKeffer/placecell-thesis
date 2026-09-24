"""Combined multi-source example-episode visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from .base import AnalysisInput, AnalysisResult
from .episode_dynamics import (
    activation_style,
    arrange_activation_grid,
    choose_unit_grid_shape,
    draw_heading_arrow,
    normalize_observation_frame,
    select_episode_index,
    select_unit_indices,
)
from .figures import figure_to_rgb_array
from .world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    resolve_world_overlay,
)


@dataclass(frozen=True, slots=True)
class _EpisodeSourceView:
    """Per-source activation view derived from one aligned episode."""

    label: str
    source_name: str
    activations: np.ndarray
    selected_unit_indices: np.ndarray
    grid_rows: int
    grid_columns: int
    cmap_name: str
    norm: colors.Normalize


def _display_label(label: str) -> str:
    return str(label).replace(".", " ").replace("_", " ").strip().title()


def _panel_grid_shape(total_panels: int) -> tuple[int, int]:
    if total_panels <= 0:
        return 0, 0
    if total_panels <= 3:
        return 1, total_panels
    return int(np.ceil(total_panels / 3.0)), 3


def _validate_aligned_inputs(inputs: list[AnalysisInput]) -> None:
    if not inputs:
        raise ValueError("Combined example-episode visualization needs at least one source.")
    reference_input = inputs[0]
    for analysis_input in inputs[1:]:
        if analysis_input.valid_mask.shape != reference_input.valid_mask.shape:
            raise ValueError(
                "Combined example-episode inputs must share the same valid-mask shape."
            )
        if analysis_input.position_xy.shape != reference_input.position_xy.shape:
            raise ValueError("Combined example-episode inputs must share the same position shape.")
        if not np.array_equal(analysis_input.valid_mask, reference_input.valid_mask):
            raise ValueError(
                "Combined example-episode inputs must share the same valid-mask values."
            )
        if not np.allclose(
            analysis_input.position_xy,
            reference_input.position_xy,
            atol=1e-5,
            equal_nan=True,
        ):
            raise ValueError("Combined example-episode inputs must share the same positions.")
        if analysis_input.split_name != reference_input.split_name:
            raise ValueError("Combined example-episode inputs must share the same split name.")


def _shared_observation_arrays(
    inputs: list[AnalysisInput],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    rgb = next(
        (analysis_input.rgb for analysis_input in inputs if analysis_input.rgb is not None), None
    )
    latent = next(
        (analysis_input.latent for analysis_input in inputs if analysis_input.latent is not None),
        None,
    )
    return rgb, latent


def _build_source_views(
    inputs: list[AnalysisInput],
    *,
    episode_index: int,
    episode_length: int,
    requested_top_k: int,
) -> list[_EpisodeSourceView]:
    source_views: list[_EpisodeSourceView] = []
    for analysis_input in inputs:
        episode_representation = analysis_input.representation[episode_index, :episode_length]
        selected_unit_indices = select_unit_indices(
            episode_representation,
            requested_top_k=requested_top_k,
        )
        selected_activations = episode_representation[:, selected_unit_indices]
        grid_rows, grid_columns = choose_unit_grid_shape(len(selected_unit_indices))
        cmap_name, norm = activation_style(selected_activations)
        source_views.append(
            _EpisodeSourceView(
                label=analysis_input.label,
                source_name=analysis_input.source_name,
                activations=selected_activations,
                selected_unit_indices=selected_unit_indices,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                cmap_name=cmap_name,
                norm=norm,
            )
        )
    return source_views


def _copy_activation_norm(norm: colors.Normalize) -> colors.Normalize:
    if isinstance(norm, colors.TwoSlopeNorm):
        return colors.TwoSlopeNorm(vmin=norm.vmin, vcenter=norm.vcenter, vmax=norm.vmax)
    return colors.Normalize(vmin=norm.vmin, vmax=norm.vmax, clip=norm.clip)


def _draw_topdown_panel(
    axis: plt.Axes,
    *,
    env_id: str | None,
    positions: np.ndarray,
    headings: np.ndarray | None,
    current_index: int,
) -> None:
    axis.plot(
        positions[:, 0], positions[:, 1], color="#B6C4D8", linewidth=1.4, alpha=0.65, zorder=2
    )
    axis.plot(
        positions[: current_index + 1, 0],
        positions[: current_index + 1, 1],
        color="#3568B8",
        linewidth=1.8,
        alpha=0.95,
        zorder=3,
    )
    axis.scatter(
        positions[current_index, 0], positions[current_index, 1], color="#D04D4D", s=42, zorder=4
    )
    world_overlay = resolve_world_overlay(env_id)
    if world_overlay is not None:
        draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#404040")
        draw_landmarks_on_axis(axis, world_overlay, marker_size=18.0)
    apply_plot_bounds(axis, env_id=env_id, position_xy=positions)
    draw_heading_arrow(axis, positions, headings, current_index)
    axis.set_title("Top-Down Map", fontsize=11)
    axis.set_xlabel(POSITION_X_LABEL)
    axis.set_ylabel(POSITION_Y_LABEL)
    axis.grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)


class _EpisodeFrameRenderer:
    """Persistent figure for the per-timestep GIF frames."""

    def __init__(
        self,
        *,
        positions: np.ndarray,
        headings: np.ndarray | None,
        first_observation_frame: np.ndarray,
        source_views: list[_EpisodeSourceView],
        env_id: str | None,
        split_name: str,
    ) -> None:
        self.positions = positions
        self.headings = headings
        self.source_views = source_views
        self.split_name = split_name
        total_panels = 2 + len(source_views)
        rows, columns = _panel_grid_shape(total_panels)
        self.figure, axes = plt.subplots(
            rows, columns, figsize=(columns * 4.8, rows * 3.8), squeeze=False
        )
        flat_axes = axes.reshape(-1)

        input_axis = flat_axes[0]
        self.input_image = input_axis.imshow(
            first_observation_frame,
            cmap="magma" if first_observation_frame.ndim == 2 else None,
            aspect="auto",
        )
        input_axis.set_title("Input", fontsize=11)
        input_axis.axis("off")

        topdown_axis = flat_axes[1]
        self.topdown_axis = topdown_axis
        topdown_axis.plot(
            positions[:, 0], positions[:, 1], color="#B6C4D8", linewidth=1.4, alpha=0.65, zorder=2
        )
        (self.progress_line,) = topdown_axis.plot(
            positions[:1, 0], positions[:1, 1], color="#3568B8", linewidth=1.8, alpha=0.95, zorder=3
        )
        self.position_marker = topdown_axis.scatter(
            positions[0, 0], positions[0, 1], color="#D04D4D", s=42, zorder=4
        )
        world_overlay = resolve_world_overlay(env_id)
        if world_overlay is not None:
            draw_world_segments_on_axis(topdown_axis, world_overlay.segments, line_color="#404040")
            draw_landmarks_on_axis(topdown_axis, world_overlay, marker_size=18.0)
        apply_plot_bounds(topdown_axis, env_id=env_id, position_xy=positions)
        self.heading_arrow: plt.Artist | None = None
        topdown_axis.set_title("Top-Down Map", fontsize=11)
        topdown_axis.set_xlabel(POSITION_X_LABEL)
        topdown_axis.set_ylabel(POSITION_Y_LABEL)
        topdown_axis.grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

        self.activation_images = []
        for panel_index, source_view in enumerate(source_views, start=2):
            axis = flat_axes[panel_index]
            activation_grid = arrange_activation_grid(
                source_view.activations[0],
                rows=source_view.grid_rows,
                columns=source_view.grid_columns,
            )
            self.activation_images.append(
                axis.imshow(
                    np.ma.masked_invalid(activation_grid),
                    origin="lower",
                    aspect="equal",
                    cmap=source_view.cmap_name,
                    norm=_copy_activation_norm(source_view.norm),
                )
            )
            axis.set_title(
                f"{_display_label(source_view.label)}\n"
                f"{len(source_view.selected_unit_indices)} units",
                fontsize=10,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("#F2F2F2")

        for panel_index in range(total_panels, len(flat_axes)):
            flat_axes[panel_index].axis("off")

        self.suptitle = self.figure.suptitle(
            f"Combined Example Episode ({split_name})\nstep 1/{len(positions)}",
            fontsize=14,
        )
        self.figure.tight_layout()

    def render(self, observation_frame: np.ndarray, current_index: int) -> np.ndarray:
        self.input_image.set_data(observation_frame)
        if observation_frame.ndim == 2:
            self.input_image.autoscale()
        self.progress_line.set_data(
            self.positions[: current_index + 1, 0],
            self.positions[: current_index + 1, 1],
        )
        self.position_marker.set_offsets(self.positions[current_index])
        if self.heading_arrow is not None:
            self.heading_arrow.remove()
            self.heading_arrow = None
        patches_before = len(self.topdown_axis.patches)
        draw_heading_arrow(self.topdown_axis, self.positions, self.headings, current_index)
        if len(self.topdown_axis.patches) > patches_before:
            self.heading_arrow = self.topdown_axis.patches[-1]
        for image, source_view in zip(self.activation_images, self.source_views, strict=False):
            image.set_data(
                np.ma.masked_invalid(
                    arrange_activation_grid(
                        source_view.activations[current_index],
                        rows=source_view.grid_rows,
                        columns=source_view.grid_columns,
                    )
                )
            )
        self.suptitle.set_text(
            f"Combined Example Episode ({self.split_name})\n"
            f"step {current_index + 1}/{len(self.positions)}"
        )
        return figure_to_rgb_array(self.figure, close=False)

    def close(self) -> None:
        plt.close(self.figure)


def _render_summary(
    path: Path,
    *,
    positions: np.ndarray,
    headings: np.ndarray | None,
    observation_frame: np.ndarray,
    source_views: list[_EpisodeSourceView],
    env_id: str | None,
    split_name: str,
) -> Path:
    rows, columns = _panel_grid_shape(2 + len(source_views))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4.8, rows * 3.8), squeeze=False)
    flat_axes = axes.reshape(-1)

    flat_axes[0].imshow(
        observation_frame,
        cmap="magma" if observation_frame.ndim == 2 else None,
        aspect="auto",
    )
    flat_axes[0].set_title("First input", fontsize=11)
    flat_axes[0].axis("off")

    trajectory_axis = flat_axes[1]
    _draw_topdown_panel(
        trajectory_axis,
        env_id=env_id,
        positions=positions,
        headings=headings,
        current_index=len(positions) - 1,
    )
    trajectory_axis.scatter(positions[0, 0], positions[0, 1], color="#4CAF50", s=36, zorder=5)
    trajectory_axis.scatter(positions[-1, 0], positions[-1, 1], color="#D04D4D", s=36, zorder=5)
    trajectory_axis.set_title("Episode trajectory", fontsize=11)

    for panel_index, source_view in enumerate(source_views, start=2):
        axis = flat_axes[panel_index]
        activation_image = axis.imshow(
            source_view.activations.T,
            origin="lower",
            aspect="auto",
            cmap=source_view.cmap_name,
            norm=_copy_activation_norm(source_view.norm),
        )
        axis.set_title(f"{_display_label(source_view.label)}\nover time", fontsize=10)
        axis.set_xlabel("time step")
        axis.set_ylabel("unit rank")
        axis.set_yticks([])
        figure.colorbar(activation_image, ax=axis, fraction=0.046, pad=0.03)

    for panel_index in range(2 + len(source_views), len(flat_axes)):
        flat_axes[panel_index].axis("off")

    figure.suptitle(f"Combined Example Episode Summary ({split_name})", fontsize=14)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


@dataclass(slots=True)
class CombinedEpisodeDynamicsModule:
    """Render one synchronized multi-source example-episode GIF."""

    name: str = "combined_episode_dynamics"
    cost_tier: str = "standard"

    def required_batch_keys(self) -> set[str]:
        return {"rgb", "latent"}

    def run(
        self,
        inputs: list[AnalysisInput],
        labels: list[str],
        output_dir: Path,
        config: dict,
    ) -> AnalysisResult:
        del labels
        _validate_aligned_inputs(inputs)
        requested_episode_index = int(config.get("example_episode_index", -1))
        random_seed = config.get("example_episode_random_seed")
        episode_index = select_episode_index(
            inputs[0].valid_mask,
            requested_episode_index=requested_episode_index,
            random_seed=None if random_seed in {None, ""} else int(random_seed),
        )
        episode_length = int(inputs[0].valid_mask[episode_index].sum())
        positions = inputs[0].position_xy[episode_index, :episode_length]
        headings = (
            None if inputs[0].heading is None else inputs[0].heading[episode_index, :episode_length]
        )
        rgb, latent = _shared_observation_arrays(inputs)
        requested_top_k = int(config.get("example_episode_top_k", 0))
        source_views = _build_source_views(
            inputs,
            episode_index=episode_index,
            episode_length=episode_length,
            requested_top_k=requested_top_k,
        )

        module_dir = output_dir / self.name
        gif_path = module_dir / f"combined_example_episode__{inputs[0].split_name}.gif"
        summary_path = module_dir / f"combined_example_episode_summary__{inputs[0].split_name}.png"

        first_observation_frame = normalize_observation_frame(
            None if rgb is None else rgb[episode_index, 0],
            None if latent is None else latent[episode_index, 0],
        )
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = _EpisodeFrameRenderer(
            positions=positions,
            headings=headings,
            first_observation_frame=first_observation_frame,
            source_views=source_views,
            env_id=str(inputs[0].metadata.get("env_id", "")),
            split_name=inputs[0].split_name,
        )
        try:
            with imageio.get_writer(
                gif_path,
                mode="I",
                duration=float(config.get("example_episode_frame_duration", 0.14)),
            ) as gif_writer:
                for timestep in range(episode_length):
                    observation_frame = normalize_observation_frame(
                        None if rgb is None else rgb[episode_index, timestep],
                        None if latent is None else latent[episode_index, timestep],
                    )
                    gif_writer.append_data(renderer.render(observation_frame, timestep))
        finally:
            renderer.close()

        _render_summary(
            summary_path,
            positions=positions,
            headings=headings,
            observation_frame=first_observation_frame,
            source_views=source_views,
            env_id=str(inputs[0].metadata.get("env_id", "")),
            split_name=inputs[0].split_name,
        )

        selected_unit_counts = [
            len(source_view.selected_unit_indices) for source_view in source_views
        ]
        return AnalysisResult(
            metrics={
                "selected_episode_index": float(episode_index),
                "selected_episode_length": float(episode_length),
                "source_count": float(len(source_views)),
                "mean_selected_unit_count": float(np.mean(selected_unit_counts))
                if selected_unit_counts
                else 0.0,
            },
            per_unit_metrics={},
            figures={
                "combined_example_episode_gif": gif_path,
                "combined_example_episode_summary": summary_path,
            },
            tables={},
            metadata={
                "source_labels": [source_view.label for source_view in source_views],
                "source_names": [source_view.source_name for source_view in source_views],
                "selected_unit_counts": selected_unit_counts,
            },
        )
