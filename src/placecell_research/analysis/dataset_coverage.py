"""Dataset coverage heatmap analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from ..numerics.rate_map_kernels import flatten_positions
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_occupancy
from .world_overlay import (
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    overlay_bounds,
    resolve_world_overlay,
    style_arena_axes,
)


@dataclass(slots=True)
class DatasetCoverageModule:
    """Render an occupancy heatmap showing where the dataset covers space."""

    name: str = "dataset_coverage"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        occupancy_result = get_or_compute_occupancy(
            analysis_input,
            num_bins_x=int(config.get("num_bins_x", 60)),
            num_bins_y=int(config.get("num_bins_y", 60)),
            smoothing_sigma=float(config.get("smoothing_sigma", 0.4)),
            bounds=world_bounds,
        )
        positions = flatten_positions(
            analysis_input.position_xy,
            analysis_input.valid_mask,
        )
        module_dir = output_dir / self.name
        figure_path = (
            module_dir
            / f"dataset_coverage__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        figure, axis = plt.subplots(figsize=(5, 5))
        x_bounds, y_bounds = occupancy_result.bounds
        occupancy = np.asarray(occupancy_result.occupancy, dtype=np.float32)
        occupied_mask = np.asarray(occupancy_result.raw_occupancy, dtype=np.float32) > 0.0
        positive_occupancy = occupancy[occupied_mask]
        occupancy_heatmap = np.ma.masked_where(~occupied_mask, occupancy)
        if positive_occupancy.size > 0:
            occupancy_norm = colors.PowerNorm(
                gamma=0.55,
                vmin=float(positive_occupancy.min()),
                vmax=float(positive_occupancy.max()),
            )
        else:
            occupancy_norm = colors.Normalize(vmin=0.0, vmax=1.0)
        axis.imshow(
            occupancy_heatmap,
            origin="lower",
            cmap="YlOrRd",
            norm=occupancy_norm,
            extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
            aspect="equal",
        )
        if world_overlay is not None:
            draw_world_segments_on_axis(axis, world_overlay.segments)
            draw_landmarks_on_axis(axis, world_overlay)
        apply_plot_bounds(
            axis,
            env_id=str(analysis_input.metadata.get("env_id", "")),
            position_xy=positions,
        )
        axis.set_title(f"Dataset occupancy heatmap ({analysis_input.split_name})")
        style_arena_axes(axis)
        coverage_colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=occupancy_norm, cmap="YlOrRd"),
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )
        coverage_colorbar.set_label("Valid steps per spatial bin", fontsize=9)
        coverage_colorbar.ax.tick_params(labelsize=7)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        occupied_bins_fraction = float(np.mean(occupancy_result.raw_occupancy > 0))
        return AnalysisResult(
            metrics={
                "occupied_bins_fraction": occupied_bins_fraction,
                "max_occupancy": float(occupancy_result.occupancy.max())
                if occupancy_result.occupancy.size
                else 0.0,
                "total_valid_steps": float(positions.shape[0]),
            },
            per_unit_metrics={},
            figures={"dataset_coverage": figure_path},
            tables={},
            metadata={
                "visualization": "occupancy_heatmap",
            },
        )
