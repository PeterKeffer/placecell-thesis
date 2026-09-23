from __future__ import annotations

import numpy as np

from placecell_research.downstream.feature_sources import transform_place_code_batch
from placecell_research.downstream.place_code_stats import (
    PlaceCodeStats,
    PlaceCodeStatsAccumulator,
    compute_place_code_stats,
)


def test_place_code_stats_preserve_sparse_active_rms_scale() -> None:
    codes = np.asarray(
        [
            [0.0, 2.0, 0.0],
            [0.0, 4.0, -3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    stats = compute_place_code_stats(codes)

    np.testing.assert_allclose(stats.active_rms, [1.0, np.sqrt(10.0), 3.0], rtol=1e-6)


def test_place_code_stats_backed_transform_modes() -> None:
    codes = np.asarray(
        [
            [0.0, 2.0, -4.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    stats = PlaceCodeStats(
        mean=np.asarray([0.0, 1.0, -2.0], dtype=np.float32),
        std=np.asarray([1.0, 2.0, 4.0], dtype=np.float32),
        active_rms=np.asarray([1.0, 4.0, 2.0], dtype=np.float32),
        sample_count=2,
        active_count=np.asarray([0, 1, 1], dtype=np.int64),
    )

    active_rms = transform_place_code_batch(codes, mode="active_rms", stats=stats)
    zscore = transform_place_code_batch(codes, mode="zscore", stats=stats)

    np.testing.assert_allclose(active_rms, [[0.0, 0.5, -2.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(zscore, [[0.0, 0.5, -0.5], [0.0, -0.5, 0.5]])


def test_streaming_place_code_stats_match_batch_stats() -> None:
    codes = np.asarray(
        [
            [0.0, 2.0, 0.0],
            [0.0, 4.0, -3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    accumulator = PlaceCodeStatsAccumulator(feature_dim=3)

    accumulator.update(codes[:1])
    accumulator.update(codes[1:])

    streaming_stats = accumulator.to_stats()
    batch_stats = compute_place_code_stats(codes)
    np.testing.assert_allclose(streaming_stats.mean, batch_stats.mean, rtol=1e-6)
    np.testing.assert_allclose(streaming_stats.std, batch_stats.std, rtol=1e-6)
    np.testing.assert_allclose(streaming_stats.active_rms, batch_stats.active_rms, rtol=1e-6)
    np.testing.assert_array_equal(streaming_stats.active_count, batch_stats.active_count)
    assert streaming_stats.sample_count == batch_stats.sample_count
