"""XY decode analysis module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.evaluation.decode import (
    episode_level_decode_skip_reason,
    linear_decode_position,
    nonlinear_decode_position,
)
from placecell_research.numerics.error_metrics import RMSE_AGGREGATION

from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    sample_valid_steps,
)
from .base import AnalysisInput, AnalysisResult
from .timing import log_timing, record_timing
from .world_overlay import (
    draw_world_segments_on_axis,
    finalize_arena_axis,
    resolve_plot_bounds,
    resolve_world_overlay,
    style_arena_axes,
)


@dataclass(slots=True)
class DecodeXYModule:
    """Offline XY decoding analysis."""

    name: str = "decode_xy"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        decode_max_samples = int(config.get("decode_max_samples", 0))
        section_started_at = perf_counter()
        sampled_steps = sample_valid_steps(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
            max_samples=decode_max_samples,
            random_seed=int(config.get("decode_random_seed", 0)),
        )
        record_timing(timing_seconds, "sample_valid_steps", section_started_at)
        flattened = sampled_steps.values
        flat_positions = sampled_steps.positions
        episode_ids = sampled_steps.episode_ids
        section_started_at = perf_counter()
        decode_skip_reason = episode_level_decode_skip_reason(episode_ids)
        record_timing(timing_seconds, "episode_decode_guard", section_started_at)
        if decode_skip_reason is not None:
            log_timing(
                self.name,
                analysis_input.source_name,
                analysis_input.split_name,
                timing_seconds,
                config=config,
            )
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "decode_skipped": True,
                    "decode_skip_reason": decode_skip_reason,
                    "valid_sample_count": int(len(episode_ids)),
                    "valid_episode_count": int(len(np.unique(episode_ids))),
                    "decode_max_samples": decode_max_samples,
                    "decode_sampled_valid_steps": int(len(episode_ids)),
                    "decode_total_valid_steps": int(sampled_steps.total_valid_steps),
                    "decode_timing_seconds": timing_seconds,
                },
            )
        section_started_at = perf_counter()
        decode = linear_decode_position(
            flattened,
            flat_positions,
            train_fraction=float(config.get("decode_train_fraction", 0.8)),
            alpha=float(config.get("decode_ridge_alpha", 1e-3)),
            include_shuffle=bool(config.get("decode_include_shuffle", True)),
            episode_ids=episode_ids,
        )
        record_timing(timing_seconds, "linear_decode", section_started_at)
        nonlinear_decode = None
        if bool(config["decode_nonlinear_enabled"]):
            section_started_at = perf_counter()
            nonlinear_decode = nonlinear_decode_position(
                flattened,
                flat_positions,
                train_fraction=float(config.get("decode_train_fraction", 0.8)),
                hidden_sizes=tuple(
                    int(hidden_size)
                    for hidden_size in config.get("decode_nonlinear_hidden_sizes", [128, 128])
                ),
                max_epochs=int(config.get("decode_nonlinear_max_epochs", 200)),
                batch_size=int(config.get("decode_nonlinear_batch_size", 1024)),
                max_train_samples=int(config.get("decode_nonlinear_max_train_samples", 65_536)),
                max_validation_samples=int(
                    config.get("decode_nonlinear_max_validation_samples", 16_384)
                ),
                random_seed=int(config.get("decode_nonlinear_random_seed", 0)),
                include_shuffle=False,
                episode_ids=episode_ids,
            )
            record_timing(timing_seconds, "nonlinear_decode", section_started_at)
        module_dir = output_dir / self.name
        figure_path = (
            module_dir / f"xy_decode__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        bias_figure_path = (
            module_dir
            / f"xy_decode_bias__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        env_id = str(analysis_input.metadata.get("env_id", ""))
        section_started_at = perf_counter()
        bounds = resolve_plot_bounds(
            env_id,
            np.concatenate([decode.targets, decode.predictions], axis=0),
        )
        if bounds is None:
            raise ValueError("Could not resolve XY decode plot bounds.")
        num_bins_x = int(config["decode_error_num_bins_x"])
        num_bins_y = int(config["decode_error_num_bins_y"])
        per_step_squared_error = np.mean(
            np.square(decode.predictions - decode.targets, dtype=np.float64), axis=1
        )
        linear_bins, _, _, resolved_bounds = compute_spatial_bin_assignments(
            decode.targets,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
        )
        squared_error_sum = np.bincount(
            linear_bins,
            weights=per_step_squared_error,
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float32, copy=False)
        sample_count = np.bincount(
            linear_bins,
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float32, copy=False)
        rmse_map = np.full(num_bins_x * num_bins_y, np.nan, dtype=np.float32)
        visited_mask = sample_count > 0
        rmse_map[visited_mask] = np.sqrt(
            squared_error_sum[visited_mask] / sample_count[visited_mask]
        )
        rmse_map = rmse_map.reshape(num_bins_y, num_bins_x)

        bias_sum_x = np.bincount(
            linear_bins,
            weights=(decode.predictions[:, 0] - decode.targets[:, 0]),
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float32, copy=False)
        bias_sum_y = np.bincount(
            linear_bins,
            weights=(decode.predictions[:, 1] - decode.targets[:, 1]),
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float32, copy=False)
        mean_target_x = np.bincount(
            linear_bins,
            weights=decode.targets[:, 0],
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float32, copy=False)
        mean_target_y = np.bincount(
            linear_bins,
            weights=decode.targets[:, 1],
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float32, copy=False)
        mean_bias_x = np.zeros((num_bins_x * num_bins_y,), dtype=np.float32)
        mean_bias_y = np.zeros((num_bins_x * num_bins_y,), dtype=np.float32)
        mean_target_x_per_bin = np.zeros((num_bins_x * num_bins_y,), dtype=np.float32)
        mean_target_y_per_bin = np.zeros((num_bins_x * num_bins_y,), dtype=np.float32)
        np.divide(bias_sum_x, sample_count, out=mean_bias_x, where=visited_mask)
        np.divide(bias_sum_y, sample_count, out=mean_bias_y, where=visited_mask)
        np.divide(mean_target_x, sample_count, out=mean_target_x_per_bin, where=visited_mask)
        np.divide(mean_target_y, sample_count, out=mean_target_y_per_bin, where=visited_mask)
        mean_bias_magnitude = np.sqrt(np.square(mean_bias_x) + np.square(mean_bias_y)).astype(
            np.float32, copy=False
        )
        bias_min_samples = int(config.get("decode_bias_min_samples_per_bin", 8))
        bias_arrow_mask = visited_mask & (sample_count >= bias_min_samples)

        x_bounds, y_bounds = resolved_bounds
        record_timing(timing_seconds, "error_maps", section_started_at)
        section_started_at = perf_counter()
        figure, axis = plt.subplots(figsize=(6.6, 6.0))
        image = axis.imshow(
            np.ma.masked_invalid(rmse_map),
            origin="lower",
            extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
            aspect="equal",
            cmap="inferno",
        )
        world_overlay = resolve_world_overlay(env_id, analysis_input.metadata.get("env_kwargs"))
        if world_overlay is not None:
            draw_world_segments_on_axis(
                axis,
                world_overlay.segments,
                line_color="#F2F2F2",
                line_width=1.2,
            )
        finalize_arena_axis(
            axis,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            world_overlay=world_overlay,
        )
        axis.set_title(
            f"{analysis_input.source_name} XY decode error\n"
            f"overall RMSE {decode.rmse:.3f} | R2 {decode.r2:.3f}"
        )
        style_arena_axes(axis)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.86)
        colorbar.set_label("Spatial decode RMSE")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        bias_figure, bias_axis = plt.subplots(figsize=(6.6, 6.0))
        bias_image = bias_axis.imshow(
            np.ma.masked_invalid(rmse_map),
            origin="lower",
            extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
            aspect="equal",
            cmap="inferno",
        )
        if world_overlay is not None:
            draw_world_segments_on_axis(
                bias_axis,
                world_overlay.segments,
                line_color="#F2F2F2",
                line_width=1.2,
            )
        if np.any(bias_arrow_mask):
            bias_axis.quiver(
                mean_target_x_per_bin[bias_arrow_mask],
                mean_target_y_per_bin[bias_arrow_mask],
                mean_bias_x[bias_arrow_mask],
                mean_bias_y[bias_arrow_mask],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="#54F0FF",
                edgecolor="#111111",
                linewidth=0.25,
                width=0.004,
                alpha=0.95,
            )
        finalize_arena_axis(
            bias_axis,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            world_overlay=world_overlay,
        )
        bias_axis.set_title(
            f"{analysis_input.source_name} XY decode bias\n"
            f"overall RMSE {decode.rmse:.3f} | "
            f"arrows shown for bins with >= {bias_min_samples} samples"
        )
        style_arena_axes(bias_axis)
        bias_colorbar = bias_figure.colorbar(bias_image, ax=bias_axis, shrink=0.86)
        bias_colorbar.set_label("Spatial decode RMSE")
        bias_figure.tight_layout()
        bias_figure.savefig(bias_figure_path, dpi=160)
        plt.close(bias_figure)
        record_timing(timing_seconds, "render_figures", section_started_at)
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )

        metrics = {
            "decode_rmse": decode.rmse,
            "decode_mae": decode.mae,
            "decode_r2": decode.r2,
            "mean_decode_bias_magnitude": float(np.mean(mean_bias_magnitude[bias_arrow_mask]))
            if np.any(bias_arrow_mask)
            else 0.0,
            "max_decode_bias_magnitude": float(np.max(mean_bias_magnitude[bias_arrow_mask]))
            if np.any(bias_arrow_mask)
            else 0.0,
        }
        if nonlinear_decode is not None:
            metrics.update(
                {
                    "nonlinear_decode_rmse": nonlinear_decode.rmse,
                    "nonlinear_decode_mae": nonlinear_decode.mae,
                    "nonlinear_decode_r2": nonlinear_decode.r2,
                }
            )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={
                "decode_error_heatmap": figure_path,
                "decode_scatter": figure_path,
                "decode_bias_arrows": bias_figure_path,
            },
            tables={},
            metadata={
                "visualization": "spatial_rmse_heatmap",
                "rmse_aggregation": RMSE_AGGREGATION,
                "decode_error_num_bins_x": num_bins_x,
                "decode_error_num_bins_y": num_bins_y,
                "decode_bias_min_samples_per_bin": bias_min_samples,
                "decode_max_samples": decode_max_samples,
                "decode_sampled_valid_steps": int(len(episode_ids)),
                "decode_total_valid_steps": int(sampled_steps.total_valid_steps),
                "decode_timing_seconds": timing_seconds,
            },
        )
