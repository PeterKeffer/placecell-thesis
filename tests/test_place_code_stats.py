from __future__ import annotations

import numpy as np

from placecell_research.downstream.feature_sources import transform_place_code_batch
from placecell_research.downstream.place_code_stats import PlaceCodeStats


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
