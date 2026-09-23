"""Coarse-region decoding: classify a representation into spatial region ids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.rate_map_kernels import compute_spatial_bin_assignments
from .base import AnalysisInput, AnalysisResult
from .episode_holdout_decode import decode_labels_episode_holdout


@dataclass(slots=True)
class DecodeRegionModule:
    """Multiclass coarse-region classifier from a representation, episode-held-out."""

    name: str = "decode_region"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_regions_x = int(config.get("decode_region_num_regions_x", 4))
        num_regions_y = int(config.get("decode_region_num_regions_y", 4))
        train_fraction = float(config.get("decode_train_fraction", 0.8))
        random_seed = int(config.get("decode_random_seed", 0))

        representation = np.asarray(analysis_input.representation)
        positions = np.asarray(analysis_input.position_xy)
        valid = np.asarray(analysis_input.valid_mask, dtype=bool)
        episodes, time = valid.shape
        episode_index = np.broadcast_to(np.arange(episodes)[:, None], (episodes, time))

        flat_valid = valid.reshape(-1)
        codes = representation.reshape(episodes * time, -1)[flat_valid]
        flat_positions = positions.reshape(episodes * time, 2)[flat_valid]
        flat_episode = episode_index.reshape(-1)[flat_valid]

        if codes.shape[0] < 4 or len(np.unique(flat_episode)) < 2:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "decode_skipped": True,
                    "decode_skip_reason": "insufficient_valid_samples",
                },
            )

        region_ids, _x_edges, _y_edges, _bounds = compute_spatial_bin_assignments(
            flat_positions, num_bins_x=num_regions_x, num_bins_y=num_regions_y, bounds=None
        )

        result, _skip_reason = decode_labels_episode_holdout(
            codes,
            region_ids,
            flat_episode,
            train_fraction=train_fraction,
            random_seed=random_seed,
            maximum_samples=0,
            max_iter=int(config.get("decode_region_max_iter", 200)),
            class_weight=None,
            classifier_random_state=None,
        )
        if result is None:
            return AnalysisResult(
                metrics={},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "decode_skipped": True,
                    "decode_skip_reason": "degenerate_split_or_labels",
                },
            )

        return AnalysisResult(
            metrics={
                "region_decode_accuracy": result.accuracy,
                "region_decode_macro_f1": result.macro_f1,
                "region_decode_chance": result.majority_chance,
                "region_decode_num_regions": float(num_regions_x * num_regions_y),
            },
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={
                "decode_region_num_regions_x": num_regions_x,
                "decode_region_num_regions_y": num_regions_y,
                "decode_sampled_valid_steps": int(codes.shape[0]),
            },
        )
