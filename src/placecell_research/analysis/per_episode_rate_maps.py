"""Per-trajectory (per-episode) rate maps rendered beside the pooled map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.occupancy import gate_episodes_by_coverage
from ..numerics.rate_map_kernels import (
    compute_rate_maps,
    flatten_positions,
    infer_bounds,
)
from .base import AnalysisInput, AnalysisResult
from .figures import save_figure
from .helpers import get_or_compute_rate_maps
from .rate_map_rendering import draw_unit_rate_map
from .world_overlay import overlay_bounds, resolve_world_overlay

_EPS = 1e-9


@dataclass(slots=True)
class PerEpisodeRateMapsModule:
    """Render pooled vs single-episode rate maps for the most within-episode-tuned units."""

    name: str = "per_episode_rate_maps"
    cost_tier: str = "heavy"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_units = int(analysis_input.representation.shape[2])
        num_bins_x = int(config.get("per_episode_num_bins_x", 20))
        num_bins_y = int(config.get("per_episode_num_bins_y", 20))
        smoothing_sigma = float(config.get("per_episode_smoothing_sigma", 1.5))
        min_occupancy = float(config.get("per_episode_min_occupancy", 1e-6))
        minimum_valid_steps = int(config.get("per_episode_minimum_valid_steps", 200))
        minimum_visited_fraction = float(config.get("per_episode_minimum_visited_fraction", 0.05))
        top_k = int(config.get("per_episode_rate_maps_top_k", 8))
        max_episode_columns = int(config.get("per_episode_rate_maps_num_episodes", 6))
        render_dpi = int(
            config.get("per_episode_rate_maps_render_dpi", config.get("render_dpi", 160))
        )

        all_valid_positions = flatten_positions(
            analysis_input.position_xy, analysis_input.valid_mask
        )
        if all_valid_positions.size == 0:
            return _empty_result("no valid positions")
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        bounds = (
            overlay_bounds(world_overlay)
            if world_overlay is not None
            else infer_bounds(all_valid_positions)
        )

        qualifying = gate_episodes_by_coverage(
            analysis_input.position_xy,
            analysis_input.valid_mask,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
            minimum_valid_steps=minimum_valid_steps,
            minimum_visited_fraction=minimum_visited_fraction,
        ).qualifying_episodes
        if qualifying.size == 0:
            return _empty_result(
                "no episodes pass the valid-steps / coverage gates "
                f"(per_episode_minimum_visited_fraction={minimum_visited_fraction})"
            )
        selected_episodes = _evenly_spaced(qualifying, max_episode_columns)

        per_episode_maps = np.stack(
            [
                compute_rate_maps(
                    analysis_input.representation[episode_index : episode_index + 1],
                    analysis_input.position_xy[episode_index : episode_index + 1],
                    None
                    if analysis_input.valid_mask is None
                    else analysis_input.valid_mask[episode_index : episode_index + 1],
                    num_bins_x=num_bins_x,
                    num_bins_y=num_bins_y,
                    smoothing_sigma=smoothing_sigma,
                    min_occupancy=min_occupancy,
                    bounds=bounds,
                ).rate_maps
                for episode_index in selected_episodes
            ],
            axis=0,
        )
        pooled_maps = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=smoothing_sigma,
            min_occupancy=min_occupancy,
            bounds=bounds,
        ).rate_maps

        episode_peak = np.nanmax(
            np.nan_to_num(per_episode_maps, nan=0.0).reshape(
                per_episode_maps.shape[0], num_units, -1
            ),
            axis=2,
        )
        mean_episode_peak = np.nanmean(episode_peak, axis=0)
        pooled_peak = np.nanmax(np.nan_to_num(pooled_maps, nan=0.0).reshape(num_units, -1), axis=1)
        ranked_units = np.argsort(np.nan_to_num(mean_episode_peak, nan=0.0))[::-1]
        ranked_units = ranked_units[:top_k] if top_k > 0 else ranked_units

        figures = _render_pages(
            output_dir / self.name,
            analysis_input=analysis_input,
            per_episode_maps=per_episode_maps,
            pooled_maps=pooled_maps,
            unit_indices=ranked_units,
            selected_episodes=selected_episodes,
            bounds=bounds,
            world_overlay=world_overlay,
            mean_episode_peak=mean_episode_peak,
            pooled_peak=pooled_peak,
            page_size=int(config.get("per_episode_rate_maps_page_size", 6)),
            render_dpi=render_dpi,
        )

        reanchoring_ratio = mean_episode_peak / np.where(pooled_peak > _EPS, pooled_peak, np.nan)
        return AnalysisResult(
            metrics={
                "per_episode_rate_maps_units_rendered": float(ranked_units.size),
                "per_episode_rate_maps_episodes_shown": float(selected_episodes.size),
                "per_episode_rate_maps_qualifying_episodes": float(qualifying.size),
                "median_reanchoring_ratio": float(np.nanmedian(reanchoring_ratio)),
            },
            per_unit_metrics={
                "mean_episode_peak_rate": mean_episode_peak.astype(np.float32, copy=False),
                "pooled_peak_rate": pooled_peak.astype(np.float32, copy=False),
                "reanchoring_ratio": reanchoring_ratio.astype(np.float32, copy=False),
            },
            figures=figures,
            tables={},
            metadata={
                "per_episode_rate_maps_selected_episodes": selected_episodes.tolist(),
                "per_episode_rate_maps_num_bins_x": num_bins_x,
                "per_episode_rate_maps_num_bins_y": num_bins_y,
            },
        )


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or values.size <= count:
        return values
    picks = np.linspace(0, values.size - 1, count).round().astype(int)
    return values[np.unique(picks)]


def _render_pages(
    output_dir: Path,
    *,
    analysis_input: AnalysisInput,
    per_episode_maps: np.ndarray,
    pooled_maps: np.ndarray,
    unit_indices: np.ndarray,
    selected_episodes: np.ndarray,
    bounds,
    world_overlay,
    mean_episode_peak: np.ndarray,
    pooled_peak: np.ndarray,
    page_size: int,
    render_dpi: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    page_size = max(1, int(page_size))
    num_columns = 1 + selected_episodes.size
    for page_index, start in enumerate(range(0, unit_indices.size, page_size)):
        page_units = unit_indices[start : start + page_size].astype(int, copy=False)
        figure, axes = plt.subplots(
            len(page_units),
            num_columns,
            figsize=(2.05 * num_columns, max(2.3, 2.2 * len(page_units))),
            squeeze=False,
        )
        figure.suptitle(
            "Per-trajectory rate maps -- "
            f"{analysis_input.source_name.replace('_', ' ').replace('.', ' ')} "
            f"({analysis_input.split_name})",
            fontsize=12,
        )
        for row, unit_index in enumerate(page_units):
            ratio = mean_episode_peak[unit_index] / max(float(pooled_peak[unit_index]), _EPS)
            _draw_map(
                axes[row][0],
                pooled_maps[unit_index],
                bounds,
                world_overlay,
                title=f"unit {unit_index} pooled  (x{ratio:.1f})"
                if row == 0
                else f"unit {unit_index} pooled",
            )
            for column, episode_index in enumerate(selected_episodes, start=1):
                _draw_map(
                    axes[row][column],
                    per_episode_maps[column - 1, unit_index],
                    bounds,
                    world_overlay,
                    title=f"episode {int(episode_index)}" if row == 0 else "",
                )
        figure.subplots_adjust(
            left=0.04, right=0.99, bottom=0.04, top=0.9, wspace=0.12, hspace=0.22
        )
        path = (
            output_dir
            / f"per_episode_rate_maps__{analysis_input.source_name}"
            f"__{analysis_input.split_name}__{start}_{start + len(page_units)}.png"
        )
        save_figure(figure, path, dpi=render_dpi)
        plt.close(figure)
        figures[f"per_episode_rate_maps_page_{page_index}"] = path
    return figures


def _draw_map(axis, rate_map, bounds, world_overlay, *, title: str) -> None:
    draw_unit_rate_map(axis, rate_map, bounds=bounds, world_overlay=world_overlay)
    if title:
        axis.set_title(title, fontsize=8)


def _empty_result(reason: str) -> AnalysisResult:
    """Nothing was rendered."""
    return AnalysisResult(
        metrics={
            "per_episode_rate_maps_units_rendered": 0.0,
            "per_episode_rate_maps_episodes_shown": 0.0,
            "per_episode_rate_maps_qualifying_episodes": 0.0,
            "median_reanchoring_ratio": float("nan"),
        },
        per_unit_metrics={},
        figures={},
        tables={},
        metadata={"reason": reason},
    )
