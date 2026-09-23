"""Per-episode gridness analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from ..numerics.occupancy import gate_episodes_by_coverage
from ..numerics.rate_map_kernels import (
    compute_rate_maps,
    flatten_positions,
    gridness_score,
    infer_bounds,
)
from .base import AnalysisInput, AnalysisResult
from .timing import log_timing, record_timing


@dataclass(slots=True)
class PerEpisodeGridnessModule:
    """Compute gridness within episodes before aggregating across them."""

    name: str = "per_episode_gridness"
    cost_tier: str = "heavy"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        del output_dir
        timing_seconds: dict[str, float] = {}

        per_episode_num_bins_x = int(config.get("per_episode_num_bins_x", 20))
        per_episode_num_bins_y = int(config.get("per_episode_num_bins_y", 20))
        per_episode_smoothing_sigma = float(config.get("per_episode_smoothing_sigma", 1.5))
        per_episode_min_occupancy = float(config.get("per_episode_min_occupancy", 1e-6))
        minimum_visited_fraction = float(config.get("per_episode_minimum_visited_fraction", 0.05))
        minimum_valid_steps = int(config.get("per_episode_minimum_valid_steps", 200))

        section_started_at = perf_counter()
        all_valid_positions = flatten_positions(
            analysis_input.position_xy,
            analysis_input.valid_mask,
        )
        if all_valid_positions.size == 0:
            raise ValueError("Cannot compute per-episode gridness without any valid positions.")
        shared_bounds = infer_bounds(all_valid_positions)
        record_timing(timing_seconds, "flatten_positions", section_started_at)

        section_started_at = perf_counter()
        coverage_gate = gate_episodes_by_coverage(
            analysis_input.position_xy,
            analysis_input.valid_mask,
            num_bins_x=per_episode_num_bins_x,
            num_bins_y=per_episode_num_bins_y,
            bounds=shared_bounds,
            minimum_valid_steps=minimum_valid_steps,
            minimum_visited_fraction=minimum_visited_fraction,
        )
        record_timing(timing_seconds, "coverage_gate", section_started_at)

        section_started_at = perf_counter()
        num_episodes, _, num_units = analysis_input.representation.shape
        episode_scores = np.full((num_episodes, num_units), np.nan, dtype=np.float32)
        episodes_used = int(coverage_gate.qualifying_episodes.size)

        for episode_index in coverage_gate.qualifying_episodes.tolist():
            episode_slice = slice(episode_index, episode_index + 1)
            episode_rate_maps = compute_rate_maps(
                analysis_input.representation[episode_slice],
                analysis_input.position_xy[episode_slice],
                None
                if analysis_input.valid_mask is None
                else analysis_input.valid_mask[episode_slice],
                num_bins_x=per_episode_num_bins_x,
                num_bins_y=per_episode_num_bins_y,
                smoothing_sigma=per_episode_smoothing_sigma,
                min_occupancy=per_episode_min_occupancy,
                bounds=shared_bounds,
            ).rate_maps
            episode_scores[episode_index] = np.asarray(
                [
                    gridness_score(np.nan_to_num(rate_map, nan=0.0))
                    for rate_map in episode_rate_maps
                ],
                dtype=np.float32,
            )
        record_timing(timing_seconds, "episode_loop", section_started_at)

        section_started_at = perf_counter()
        if episodes_used > 0:
            per_episode_gridness = np.nanmean(
                episode_scores,
                axis=0,
            ).astype(np.float32, copy=False)
            per_episode_gridness_std = np.nanstd(
                episode_scores,
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            per_episode_gridness = np.zeros(num_units, dtype=np.float32)
            per_episode_gridness_std = np.zeros(num_units, dtype=np.float32)
        record_timing(timing_seconds, "aggregate_scores", section_started_at)

        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )

        return AnalysisResult(
            metrics={
                "mean_per_episode_gridness": (
                    float(np.mean(per_episode_gridness)) if len(per_episode_gridness) else 0.0
                ),
                "max_per_episode_gridness": (
                    float(np.max(per_episode_gridness)) if len(per_episode_gridness) else 0.0
                ),
                "per_episode_gridness_episodes_used": episodes_used,
                "per_episode_gridness_episodes_skipped_short": (
                    coverage_gate.episodes_skipped_short
                ),
                "per_episode_gridness_episodes_skipped_sparse": (
                    coverage_gate.episodes_skipped_sparse
                ),
            },
            per_unit_metrics={
                "per_episode_gridness": per_episode_gridness,
                "per_episode_gridness_std": per_episode_gridness_std,
            },
            figures={},
            tables={},
            metadata={
                "per_episode_gridness_bounds": shared_bounds,
                "per_episode_gridness_num_bins_x": per_episode_num_bins_x,
                "per_episode_gridness_num_bins_y": per_episode_num_bins_y,
                "per_episode_gridness_minimum_valid_steps": minimum_valid_steps,
                "per_episode_gridness_minimum_visited_fraction": minimum_visited_fraction,
                "per_episode_gridness_timing_seconds": timing_seconds,
            },
        )
