from __future__ import annotations

import numpy as np

from placecell_research.numerics.rate_map_kernels import skaggs_spatial_information


def test_skaggs_spatial_information_is_zero_for_uniform_rate_map() -> None:
    occupancy = np.asarray([[1.0, 1.0]], dtype=np.float32)
    rate_map = np.asarray([[3.0, 3.0]], dtype=np.float32)

    score = skaggs_spatial_information(rate_map, occupancy)

    assert abs(score) < 1e-6


def test_skaggs_spatial_information_is_one_bit_for_two_bin_selective_cell() -> None:
    occupancy = np.asarray([[1.0, 1.0]], dtype=np.float32)
    rate_map = np.asarray([[2.0, 0.0]], dtype=np.float32)

    score = skaggs_spatial_information(rate_map, occupancy)

    assert np.isclose(score, 1.0, atol=1e-5)


def test_skaggs_spatial_information_supports_batched_rate_maps() -> None:
    occupancy = np.asarray([[1.0, 1.0]], dtype=np.float32)
    rate_maps = np.asarray(
        [
            [[2.0, 0.0]],
            [[3.0, 3.0]],
        ],
        dtype=np.float32,
    )

    scores = skaggs_spatial_information(rate_maps, occupancy)

    assert scores.shape == (2,)
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    assert abs(float(scores[1])) < 1e-6


def test_skaggs_spatial_information_is_unsupported_for_signed_rate_maps() -> None:
    occupancy = np.asarray([[1.0, 1.0]], dtype=np.float32)
    rate_map = np.asarray([[0.5, -0.25]], dtype=np.float32)

    score = skaggs_spatial_information(rate_map, occupancy)

    assert np.isnan(score)


def test_skaggs_spatial_information_marks_signed_maps_as_nan_in_batches() -> None:
    occupancy = np.asarray([[1.0, 1.0]], dtype=np.float32)
    rate_maps = np.asarray(
        [
            [[2.0, 0.0]],
            [[0.5, -0.25]],
        ],
        dtype=np.float32,
    )

    scores = skaggs_spatial_information(rate_maps, occupancy)

    assert scores.shape == (2,)
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    assert np.isnan(scores[1])


def test_skaggs_spatial_information_tolerates_tiny_negative_contamination() -> None:
    occupancy = np.ones((1, 100), dtype=np.float32)
    clipped_rate_map = np.zeros((1, 100), dtype=np.float32)
    clipped_rate_map[0, 0] = 2.0
    near_nonnegative_rate_map = clipped_rate_map.copy()
    near_nonnegative_rate_map[0, 1] = -0.05

    contaminated_score = skaggs_spatial_information(near_nonnegative_rate_map, occupancy)
    clipped_score = skaggs_spatial_information(clipped_rate_map, occupancy)

    assert np.isclose(contaminated_score, clipped_score, atol=1e-6)
