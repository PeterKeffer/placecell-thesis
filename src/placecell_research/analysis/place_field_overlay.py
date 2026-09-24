"""Per-step overlay of active place cells' full rate-map fields as a GIF."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib.patches import Patch

from .base import AnalysisInput, AnalysisResult
from .episode_dynamics import _draw_heading_arrow, _select_episode_index
from .figures import figure_to_rgb_array, save_gif
from .helpers import get_or_compute_rate_maps
from .world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    finalize_arena_axis,
    overlay_bounds,
    resolve_world_overlay,
)

logger = logging.getLogger(__name__)

_BACKGROUND_RGB = (0.05, 0.05, 0.07)
_BLEND_MODES = ("max", "additive", "alpha")
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def _kwinners_k_from_fraction(num_units: int, k_fraction: float | None) -> int:
    """Top-k count for a fraction, mirroring KWinnersSparsifier; 0 means disabled."""
    if k_fraction is None or float(k_fraction) <= 0.0:
        return 0
    return max(1, int(round(int(num_units) * float(k_fraction))))


def _active_unit_indices(
    step_activation: np.ndarray,
    *,
    threshold: float,
    max_cells: int,
    kwinners_k: int = 0,
) -> tuple[np.ndarray, int]:
    """Return active-unit indices (strongest first) and how many were dropped."""
    activation = np.asarray(step_activation)
    active = np.flatnonzero(activation > float(threshold))
    if active.size == 0:
        return active.astype(np.int64), 0
    descending_order = np.argsort(activation[active], kind="stable")[::-1]
    ordered = active[descending_order]
    limit = int(kwinners_k) if int(kwinners_k) > 0 else int(max_cells)
    dropped = 0
    if limit > 0 and ordered.size > limit:
        dropped = int(ordered.size - limit)
        ordered = ordered[:limit]
    return ordered.astype(np.int64), dropped


def _assign_frame_colors(active_indices: np.ndarray) -> np.ndarray:
    """Give the active cells maximally separated hues for this frame."""
    active_indices = np.asarray(active_indices)
    count = active_indices.size
    if count == 0:
        return np.zeros((0, 3), dtype=float)
    seed_hue = (active_indices.astype(np.float64) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    order = np.argsort(seed_hue, kind="stable")
    hues = np.empty(count, dtype=np.float64)
    hues[order] = np.arange(count, dtype=np.float64) / count
    return np.array([mpl_colors.hsv_to_rgb((hue, 0.95, 1.0)) for hue in hues], dtype=float)


def _normalize_fields(rate_maps: np.ndarray) -> np.ndarray:
    """Self-normalize each unit's rate map to its own peak; NaN and negatives become 0."""
    fields = np.nan_to_num(np.asarray(rate_maps, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fields = np.clip(fields, 0.0, None)
    peaks = fields.reshape(fields.shape[0], -1).max(axis=1)
    safe_peaks = np.where(peaks > 1e-12, peaks, 1.0).astype(np.float32)
    return (fields / safe_peaks[:, None, None]).astype(np.float32)


def _composite_active_fields(
    normalized_fields: np.ndarray,
    colors: np.ndarray,
    *,
    blend_mode: str,
    background: np.ndarray,
) -> np.ndarray:
    """Blend per-cell colored field layers into one RGB image over a fixed background."""
    background_rgb = np.asarray(background, dtype=np.float32).reshape(1, 1, 3)
    normalized_fields = np.asarray(normalized_fields, dtype=np.float32)
    height, width = normalized_fields.shape[1], normalized_fields.shape[2]
    if normalized_fields.shape[0] == 0:
        return np.broadcast_to(background_rgb, (height, width, 3)).astype(np.float32).copy()
    colors = np.asarray(colors, dtype=np.float32)
    layers = normalized_fields[..., None] * colors[:, None, None, :]
    if blend_mode == "max":
        image = np.maximum(layers.max(axis=0), background_rgb)
    elif blend_mode == "additive":
        image = background_rgb + layers.sum(axis=0)
    elif blend_mode == "alpha":
        image = np.broadcast_to(background_rgb, (height, width, 3)).astype(np.float32).copy()
        for layer_index in range(layers.shape[0]):
            alpha = normalized_fields[layer_index][..., None]
            image = layers[layer_index] + image * (1.0 - alpha)
    else:
        raise ValueError(f"Unknown blend_mode {blend_mode!r}; expected one of {_BLEND_MODES}.")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _frame_step_indices(episode_length: int, max_frames: int) -> np.ndarray:
    """Pick step indices to render, striding long episodes while keeping both endpoints."""
    if episode_length <= 0:
        return np.zeros(0, dtype=np.int64)
    if max_frames is None or int(max_frames) <= 0 or episode_length <= int(max_frames):
        return np.arange(episode_length, dtype=np.int64)
    sampled = np.linspace(0, episode_length - 1, num=int(max_frames))
    return np.unique(np.round(sampled).astype(np.int64))


def _render_overlay_frame(
    *,
    composite_image: np.ndarray,
    extent: tuple[float, float, float, float],
    positions: np.ndarray,
    headings: np.ndarray | None,
    current_index: int,
    active_unit_indices: np.ndarray,
    active_colors: np.ndarray,
    world_overlay,
    source_name: str,
    split_name: str,
    step_number: int,
    total_steps: int,
) -> np.ndarray:
    figure, axis = plt.subplots(figsize=(7.4, 6.6))
    axis.imshow(
        composite_image,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        zorder=1,
    )
    if world_overlay is not None:
        draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#c9c9c9")
        draw_landmarks_on_axis(axis, world_overlay, marker_size=18.0)
    axis.plot(
        positions[:, 0], positions[:, 1], color="#9fb4d8", linewidth=0.8, alpha=0.30, zorder=2
    )
    axis.plot(
        positions[: current_index + 1, 0],
        positions[: current_index + 1, 1],
        color="#e6e6e6",
        linewidth=1.0,
        alpha=0.55,
        zorder=3,
    )
    axis.scatter(
        positions[current_index, 0],
        positions[current_index, 1],
        color="#ffffff",
        edgecolor="#000000",
        linewidth=0.6,
        s=46,
        zorder=5,
    )
    _draw_heading_arrow(axis, positions, headings, current_index)
    finalize_arena_axis(
        axis,
        x_bounds=(extent[0], extent[1]),
        y_bounds=(extent[2], extent[3]),
        world_overlay=world_overlay,
    )
    axis.set_xlabel(POSITION_X_LABEL)
    axis.set_ylabel(POSITION_Y_LABEL)
    axis.set_title(
        f"{source_name} active place fields ({split_name})\n"
        f"step {step_number}/{total_steps}  ·  {active_unit_indices.size} cells on",
        fontsize=11,
    )
    if active_unit_indices.size > 0:
        legend_handles = [
            Patch(color=active_colors[position], label=f"unit {int(unit_index)}")
            for position, unit_index in enumerate(active_unit_indices)
        ]
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
            ncol=1 if active_unit_indices.size <= 12 else 2,
            framealpha=0.9,
            title="active units",
            title_fontsize=8,
        )
    figure.tight_layout()
    return figure_to_rgb_array(figure)


@dataclass(slots=True)
class PlaceFieldOverlayModule:
    """Animate one example run, overlaying each active cell's global rate-map field."""

    name: str = "place_field_overlay"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_bins_x = int(config.get("num_bins_x", 60))
        num_bins_y = int(config.get("num_bins_y", 60))
        smoothing_sigma = float(config.get("smoothing_sigma", 0.3))
        min_occupancy = float(config.get("min_occupancy", 1e-6))
        active_threshold = float(config.get("place_field_overlay_active_threshold", 0.0))
        blend_mode = str(config.get("place_field_overlay_blend_mode", "max"))
        requested_max_cells = int(config["place_field_overlay_max_cells"])
        max_frames = int(config.get("place_field_overlay_max_frames", 240))
        frame_duration = float(config["place_field_overlay_frame_duration"])

        env_id = str(analysis_input.metadata.get("env_id", ""))
        world_overlay = resolve_world_overlay(env_id, analysis_input.metadata.get("env_kwargs"))
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None

        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=world_bounds,
        )
        normalized_fields = _normalize_fields(rate_map_result.rate_maps)
        (x_bounds, y_bounds) = rate_map_result.bounds
        extent = (x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1])

        effective_max_cells = max(0, requested_max_cells)
        num_units = int(analysis_input.representation.shape[-1])
        raw_kwinners_fraction = config.get("place_field_overlay_kwinners_k_fraction")
        kwinners_k = _kwinners_k_from_fraction(
            num_units,
            None if raw_kwinners_fraction in {None, ""} else float(raw_kwinners_fraction),
        )
        active_selection_limit = kwinners_k if kwinners_k > 0 else effective_max_cells

        random_seed = config.get("example_episode_random_seed")
        episode_index = _select_episode_index(
            analysis_input.valid_mask,
            requested_episode_index=int(config.get("example_episode_index", -1)),
            random_seed=None if random_seed in {None, ""} else int(random_seed),
        )
        episode_length = int(analysis_input.valid_mask[episode_index].sum())
        episode_representation = analysis_input.representation[episode_index, :episode_length]
        positions = analysis_input.position_xy[episode_index, :episode_length]
        headings = (
            None
            if analysis_input.heading is None
            else analysis_input.heading[episode_index, :episode_length]
        )

        frame_steps = _frame_step_indices(episode_length, max_frames)
        frames: list[np.ndarray] = []
        max_active_cells = 0
        dropped_cell_count = 0
        for step in frame_steps:
            active_units, dropped = _active_unit_indices(
                episode_representation[int(step)],
                threshold=active_threshold,
                max_cells=effective_max_cells,
                kwinners_k=kwinners_k,
            )
            dropped_cell_count += dropped
            max_active_cells = max(max_active_cells, int(active_units.size))
            active_colors = _assign_frame_colors(active_units)
            composite_image = _composite_active_fields(
                normalized_fields[active_units],
                active_colors,
                blend_mode=blend_mode,
                background=np.asarray(_BACKGROUND_RGB, dtype=np.float32),
            )
            frames.append(
                _render_overlay_frame(
                    composite_image=composite_image,
                    extent=extent,
                    positions=positions,
                    headings=headings,
                    current_index=int(step),
                    active_unit_indices=active_units,
                    active_colors=active_colors,
                    world_overlay=world_overlay,
                    source_name=analysis_input.source_name,
                    split_name=analysis_input.split_name,
                    step_number=int(step) + 1,
                    total_steps=episode_length,
                )
            )

        if dropped_cell_count > 0 and kwinners_k > 0:
            logger.info(
                "place_field_overlay applied k-winners display (k=%d) for %s; showing the top %d "
                "active cells per frame (%d weaker activations excluded across %d frames).",
                kwinners_k,
                analysis_input.source_name,
                kwinners_k,
                dropped_cell_count,
                len(frame_steps),
            )
        elif dropped_cell_count > 0:
            logger.warning(
                "place_field_overlay capped active cells to %d/frame for %s; dropped %d "
                "weakest cell-activations across %d frames.",
                effective_max_cells,
                analysis_input.source_name,
                dropped_cell_count,
                len(frame_steps),
            )

        module_dir = output_dir / self.name
        gif_path = (
            module_dir
            / f"place_field_overlay__{analysis_input.source_name}__{analysis_input.split_name}.gif"
        )
        save_gif(gif_path, frames, duration=frame_duration)

        return AnalysisResult(
            metrics={
                "selected_episode_index": float(episode_index),
                "selected_episode_length": float(episode_length),
                "frames_rendered": float(len(frame_steps)),
                "max_active_cells_in_frame": float(max_active_cells),
                "dropped_cell_count": float(dropped_cell_count),
                "active_selection_k": float(active_selection_limit),
            },
            per_unit_metrics={},
            figures={"place_field_overlay_gif": gif_path},
            tables={},
            metadata={
                "blend_mode": blend_mode,
                "active_threshold": active_threshold,
                "max_cells_per_frame": effective_max_cells,
                "active_selection_mode": "kwinners" if kwinners_k > 0 else "threshold",
                "kwinners_k": kwinners_k,
                "env_id": env_id,
            },
        )
