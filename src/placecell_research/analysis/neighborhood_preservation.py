"""World<->code neighborhood-preservation metrics (no UMAP in the loop)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from ..numerics.rate_map_kernels import flatten_valid_steps
from .base import AnalysisInput, AnalysisResult
from .helpers import (
    compute_trustworthiness,
    subsample_indices,
)
from .timing import log_timing, record_timing
from .topology_umap import pool_by_spatial_bin


def _world_code_trustworthiness(
    true_space: np.ndarray,
    other_space: np.ndarray,
    neighbor_count: int,
) -> float:
    """World<->code trustworthiness with the shared k-clamp guard."""
    return compute_trustworthiness(
        true_space, other_space, neighbor_count, degenerate_value=float("nan")
    )


def _distance_rank_spearman(
    features: np.ndarray,
    positions: np.ndarray,
) -> float:
    """Spearman correlation of pairwise L2 code distance vs world distance."""
    if len(features) < 3:
        return float("nan")
    code_distances = pdist(features, metric="euclidean")
    world_distances = pdist(positions, metric="euclidean")
    return float(spearmanr(code_distances, world_distances).statistic)


@dataclass(slots=True)
class NeighborhoodPreservationModule:
    """World<->code neighborhood preservation, trustworthiness, continuity, rank correlation."""

    name: str = "neighborhood_preservation"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        section_started_at = perf_counter()
        features, positions = flatten_valid_steps(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
        )
        record_timing(timing_seconds, "flatten_valid_steps", section_started_at)
        if len(features) < 4:
            raise ValueError("neighborhood_preservation requires at least 4 valid timesteps.")
        features = features.astype(np.float32, copy=False)
        positions = positions.astype(np.float32, copy=False)

        neighbor_count = int(config.get("neighborhood_preservation_neighbors", 15))
        max_points = max(4, int(config.get("neighborhood_preservation_max_points", 4096)))
        random_seed = int(config.get("neighborhood_preservation_random_seed", 0))

        metrics: dict[str, float] = {"neighbors_k": float(neighbor_count)}

        section_started_at = perf_counter()
        step_indices = subsample_indices(len(features), max_points, random_seed)
        step_features = features[step_indices]
        step_positions = positions[step_indices]
        metrics["step_world_code_trustworthiness"] = _world_code_trustworthiness(
            step_positions, step_features, neighbor_count
        )
        metrics["step_world_code_continuity"] = _world_code_trustworthiness(
            step_features, step_positions, neighbor_count
        )
        metrics["world_code_distance_spearman"] = _distance_rank_spearman(
            step_features, step_positions
        )
        metrics["step_points"] = float(len(step_features))
        record_timing(timing_seconds, "step_metrics", section_started_at)

        section_started_at = perf_counter()
        pooled = pool_by_spatial_bin(
            analysis_input,
            features,
            positions,
            num_bins_x=int(config.get("neighborhood_preservation_pool_num_bins_x", 20)),
            num_bins_y=int(config.get("neighborhood_preservation_pool_num_bins_y", 20)),
            min_count=int(config.get("neighborhood_preservation_pool_min_count", 2)),
        )
        if pooled is not None:
            pooled_features, pooled_positions, _ = pooled
            metrics["pooled_world_code_trustworthiness"] = _world_code_trustworthiness(
                pooled_positions, pooled_features, neighbor_count
            )
            metrics["pooled_world_code_continuity"] = _world_code_trustworthiness(
                pooled_features, pooled_positions, neighbor_count
            )
            metrics["pooled_points"] = float(len(pooled_features))
        record_timing(timing_seconds, "pooled_metrics", section_started_at)

        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={"neighborhood_preservation_timing_seconds": timing_seconds},
        )
