from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.per_episode_gridness import PerEpisodeGridnessModule
from placecell_research.numerics.rate_map_kernels import (
    compute_rate_maps,
    flatten_positions,
    infer_bounds,
)


def test_per_episode_gridness_averages_valid_episodes_and_tracks_skips(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "placecell_research.analysis.per_episode_gridness.gridness_score",
        lambda rate_map: float(np.nan_to_num(rate_map, nan=0.0).mean()),
    )

    positions = np.asarray(
        [
            [
                [-1.0, -1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [-1.0, 1.0],
                [0.0, 1.0],
            ],
            [
                [-1.0, -1.0],
                [0.0, -1.0],
                [1.0, -1.0],
                [-1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            [
                [-1.0, -1.0],
                [-1.0, -1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )
    representation = np.asarray(
        [
            [
                [0.1, 0.8],
                [0.2, 0.7],
                [0.3, 0.6],
                [0.4, 0.5],
                [0.5, 0.4],
                [0.6, 0.3],
                [0.7, 0.2],
                [0.8, 0.1],
            ],
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.5, 0.5],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
            ],
            [
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.1, 0.9],
            ],
        ],
        dtype=np.float32,
    )
    valid_mask = np.asarray(
        [
            [True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True],
            [True, True, False, False, False, False, False, False],
        ],
        dtype=bool,
    )
    analysis_input = AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=valid_mask,
        source_name="encoder.hidden_state",
        label="encoder_hidden_state",
        split_name="validation",
    )
    config = {
        "num_bins_x": 4,
        "num_bins_y": 4,
        "smoothing_sigma": 0.0,
        "min_occupancy": 1e-6,
        "per_episode_num_bins_x": 4,
        "per_episode_num_bins_y": 4,
        "per_episode_smoothing_sigma": 0.0,
        "per_episode_min_occupancy": 1e-6,
        "per_episode_minimum_visited_fraction": 0.25,
        "per_episode_minimum_valid_steps": 4,
    }

    result = PerEpisodeGridnessModule().run(analysis_input, tmp_path, config)

    shared_bounds = infer_bounds(flatten_positions(positions, valid_mask))
    episode_scores = []
    for episode_index in (0, 1):
        episode_rate_maps = compute_rate_maps(
            representation[episode_index : episode_index + 1],
            positions[episode_index : episode_index + 1],
            valid_mask[episode_index : episode_index + 1],
            num_bins_x=4,
            num_bins_y=4,
            smoothing_sigma=0.0,
            min_occupancy=1e-6,
            bounds=shared_bounds,
        ).rate_maps
        episode_scores.append(np.nan_to_num(episode_rate_maps, nan=0.0).mean(axis=(1, 2)))
    expected_per_episode = np.mean(np.stack(episode_scores, axis=0), axis=0)
    expected_std = np.std(np.stack(episode_scores, axis=0), axis=0)
    np.testing.assert_allclose(
        result.per_unit_metrics["per_episode_gridness"],
        expected_per_episode,
    )
    np.testing.assert_allclose(result.per_unit_metrics["per_episode_gridness_std"], expected_std)
    assert "global_gridness" not in result.per_unit_metrics
    assert "mean_global_gridness" not in result.metrics
    assert result.metrics["per_episode_gridness_episodes_used"] == 2
    assert result.metrics["per_episode_gridness_episodes_skipped_short"] == 1
    assert result.metrics["per_episode_gridness_episodes_skipped_sparse"] == 0
    assert set(result.metadata["per_episode_gridness_timing_seconds"]) >= {
        "flatten_positions",
        "episode_loop",
        "aggregate_scores",
    }
