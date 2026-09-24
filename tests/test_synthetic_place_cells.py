"""Tests for the Sorscher-style synthetic place-cell baseline representation."""

from __future__ import annotations

import numpy as np
import pytest

from placecell_research.downstream.synthetic_place_cells import (
    build_synthetic_place_cell_centers,
    resolve_environment_reachable_regions,
    resolve_environment_xz_bounds,
    resolve_synthetic_sigmas,
    synthetic_place_cell_code,
)

WALLGAP_LARGE_BOUNDS = (-24.0, 24.0, -30.0, 36.0)


def test_code_rows_sum_to_zero():
    rng = np.random.default_rng(0)
    centers = rng.uniform([-10.0, -10.0], [10.0, 10.0], size=(64, 2)).astype(np.float32)
    positions = np.array([[0.0, 0.0], [5.0, -3.0], [-8.0, 9.0]], dtype=np.float32)
    code = synthetic_place_cell_code(positions, centers, sigma_center=2.0, sigma_surround=4.0)
    assert code.shape == (3, 64)
    np.testing.assert_allclose(code.sum(axis=1), 0.0, atol=1e-5)


def test_code_peaks_positive_at_nearest_center():
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    positions = np.array([[10.0, 0.0]], dtype=np.float32)
    code = synthetic_place_cell_code(positions, centers, sigma_center=2.0, sigma_surround=4.0)
    assert int(np.argmax(code[0])) == 1
    assert code[0, 1] > 0.0


def test_code_uses_second_coordinate_z():
    centers = np.array([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32)
    near_first = synthetic_place_cell_code(np.array([[0.0, 0.0]], np.float32), centers, 2.0, 4.0)
    near_second = synthetic_place_cell_code(np.array([[0.0, 10.0]], np.float32), centers, 2.0, 4.0)
    assert int(np.argmax(near_first[0])) == 0
    assert int(np.argmax(near_second[0])) == 1


def test_l2_normalization_makes_unit_rows_and_preserves_pattern():
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0]], dtype=np.float32)
    positions = np.array([[1.0, 2.0], [8.0, 1.0]], dtype=np.float32)
    raw = synthetic_place_cell_code(positions, centers, 2.0, 4.0)
    normed = synthetic_place_cell_code(positions, centers, 2.0, 4.0, normalization="l2")
    assert float(np.linalg.norm(raw, axis=1).max()) < 0.6
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-5)
    assert (np.argmax(raw, axis=1) == np.argmax(normed, axis=1)).all()


def test_centers_within_bounds_and_reproducible():
    centers = build_synthetic_place_cell_centers(WALLGAP_LARGE_BOUNDS, num_cells=512, seed=0)
    assert centers.shape == (512, 2)
    assert centers[:, 0].min() >= -24.0 and centers[:, 0].max() <= 24.0
    assert centers[:, 1].min() >= -30.0 and centers[:, 1].max() <= 36.0
    again = build_synthetic_place_cell_centers(WALLGAP_LARGE_BOUNDS, num_cells=512, seed=0)
    np.testing.assert_array_equal(centers, again)
    different = build_synthetic_place_cell_centers(WALLGAP_LARGE_BOUNDS, num_cells=512, seed=1)
    assert not np.array_equal(centers, different)


def test_resolve_bounds_known_explicit_unknown():
    assert resolve_environment_xz_bounds("MiniWorld-WallGapAsymLarge-v0") == WALLGAP_LARGE_BOUNDS
    assert resolve_environment_xz_bounds("anything", [1.0, 2.0, 3.0, 4.0]) == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        resolve_environment_xz_bounds("Unknown-env-v0")


def test_resolve_sigmas_auto_and_explicit():
    characteristic = 0.5 * ((24.0 - -24.0) + (36.0 - -30.0))
    auto_center, auto_surround = resolve_synthetic_sigmas(
        WALLGAP_LARGE_BOUNDS, sigma_center=0.0, surround_scale=2.0
    )
    assert auto_center == pytest.approx((0.2 / 2.2) * characteristic, rel=1e-6)
    assert auto_surround == pytest.approx(2.0 * auto_center, rel=1e-6)
    ex_center, ex_surround = resolve_synthetic_sigmas(
        WALLGAP_LARGE_BOUNDS, sigma_center=3.0, surround_scale=2.0
    )
    assert ex_center == pytest.approx(3.0)
    assert ex_surround == pytest.approx(6.0)


def test_feature_source_reads_position_and_dim():
    from placecell_research.downstream.feature_sources import (
        StepContext,
        SyntheticPlaceCellsFeatureSource,
    )

    centers = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    source = SyntheticPlaceCellsFeatureSource(centers=centers, sigma_center=2.0, sigma_surround=4.0)
    assert source.name == "synthetic_place_cells"
    assert source.feature_dim == 2
    context = StepContext(
        rgb=np.zeros((3, 4, 4), dtype=np.uint8),
        position_xy=np.array([10.0, 0.0], dtype=np.float32),
        heading=0.0,
        kinematics=None,
        goal_position_xy=None,
    )
    out = source.extract(context, None)
    assert out.shape == (2,)
    assert int(np.argmax(out)) == 1


def _synthetic_observation(num_cells: int):
    from placecell_research.config.downstream_schema import (
        DownstreamObservationConfig,
        SyntheticPlaceCellsConfig,
    )

    return DownstreamObservationConfig(
        feature_sources=["synthetic_place_cells"],
        synthetic_place_cells=SyntheticPlaceCellsConfig(num_cells=num_cells, seed=0),
    )


def test_build_feature_extractor_synthetic_only(tmp_path):
    from placecell_research.artifacts.registry import ArtifactRegistry
    from placecell_research.config.downstream_schema import DownstreamModelConfig
    from placecell_research.downstream.runtime import build_feature_extractor

    extractor = build_feature_extractor(
        artifact_registry=ArtifactRegistry(tmp_path),
        models=DownstreamModelConfig(),
        observation=_synthetic_observation(16),
        goal_candidate_positions_xy=[],
        goal_rbf_sigma=1.5,
        device="cpu",
        env_id="MiniWorld-WallGapAsymLarge-v0",
    )
    assert extractor is not None
    assert extractor.feature_dim == 16


def test_single_and_vectorized_paths_match(tmp_path):
    from placecell_research.artifacts.registry import ArtifactRegistry
    from placecell_research.config.downstream_schema import DownstreamModelConfig
    from placecell_research.downstream.feature_sources import StepContext
    from placecell_research.downstream.runtime import build_feature_extractor
    from placecell_research.downstream.shared_feature_vec_env import build_shared_feature_pipeline

    observation = _synthetic_observation(32)
    env_id = "MiniWorld-WallGapAsymLarge-v0"
    extractor = build_feature_extractor(
        artifact_registry=ArtifactRegistry(tmp_path),
        models=DownstreamModelConfig(),
        observation=observation,
        env_id=env_id,
        goal_candidate_positions_xy=[],
        goal_rbf_sigma=1.5,
        device="cpu",
    )
    pipeline = build_shared_feature_pipeline(
        artifact_registry=ArtifactRegistry(tmp_path),
        models=DownstreamModelConfig(),
        observation=observation,
        env_id=env_id,
        goal_candidate_positions_xy=[],
        goal_rbf_sigma=1.5,
        device="cpu",
    )
    assert pipeline.feature_dim == 32

    position = np.array([3.0, -7.0], dtype=np.float32)
    context = StepContext(
        rgb=np.zeros((3, 4, 4), dtype=np.uint8),
        position_xy=position,
        heading=0.0,
        kinematics=None,
        goal_position_xy=None,
    )
    single = extractor.extract(context, None)
    batched = pipeline.encode(
        rgb_batch=np.zeros((1, 3, 4, 4), dtype=np.uint8),
        positions_xy=position.reshape(1, 2),
        headings=np.zeros(1, dtype=np.float32),
        kinematics_batch=np.zeros((1, 4), dtype=np.float32),
        goal_positions_xy=None,
        goal_place_codes=None,
        previous_actions=np.zeros(1, dtype=np.int64),
        indices=np.array([0], dtype=np.int64),
    )
    assert single.shape == (32,)
    assert batched.shape == (1, 32)
    np.testing.assert_allclose(single, batched[0], atol=1e-6)


def test_wallgap_centers_land_only_in_reachable_rooms():
    rooms = resolve_environment_reachable_regions("MiniWorld-WallGapAsymLarge-v0")
    assert rooms is not None
    centers = build_synthetic_place_cell_centers(
        WALLGAP_LARGE_BOUNDS, num_cells=512, seed=0, reachable_regions=rooms
    )
    assert centers.shape == (512, 2)
    inside = np.zeros(len(centers), dtype=bool)
    for min_x, max_x, min_z, max_z in rooms:
        inside |= (
            (centers[:, 0] >= min_x)
            & (centers[:, 0] <= max_x)
            & (centers[:, 1] >= min_z)
            & (centers[:, 1] <= max_z)
        )
    assert inside.all()


def test_reachable_sampling_is_seed_reproducible():
    rooms = resolve_environment_reachable_regions("MiniWorld-WallGapAsymLarge-v0")
    first = build_synthetic_place_cell_centers(
        WALLGAP_LARGE_BOUNDS, num_cells=64, seed=7, reachable_regions=rooms
    )
    second = build_synthetic_place_cell_centers(
        WALLGAP_LARGE_BOUNDS, num_cells=64, seed=7, reachable_regions=rooms
    )
    np.testing.assert_array_equal(first, second)


def test_unknown_environment_falls_back_to_the_bounding_box():
    assert resolve_environment_reachable_regions("simple") is None
    centers = build_synthetic_place_cell_centers(WALLGAP_LARGE_BOUNDS, num_cells=32, seed=1)
    assert centers.shape == (32, 2)


def test_reachable_regions_accepts_a_numpy_array():
    rooms = np.asarray(
        resolve_environment_reachable_regions("MiniWorld-WallGapAsymLarge-v0"), dtype=np.float64
    )
    centers = build_synthetic_place_cell_centers(
        WALLGAP_LARGE_BOUNDS, num_cells=32, seed=0, reachable_regions=rooms
    )
    assert centers.shape == (32, 2)
