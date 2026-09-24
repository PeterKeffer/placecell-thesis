"""Tests for the Solstad-style synthetic grid-cell baseline representation."""

from __future__ import annotations

import numpy as np
import pytest

from placecell_research.downstream.synthetic_grid_cells import (
    build_synthetic_grid_cell_bank,
    synthetic_grid_cell_code,
)

WALLGAP_LARGE_BOUNDS = (-24.0, 24.0, -30.0, 36.0)
WALLGAP_MAX_SIDE = 66.0


def _bank(num_cells=512, num_modules=4, period_ratio=1.42, seed=0):
    return build_synthetic_grid_cell_bank(
        WALLGAP_LARGE_BOUNDS,
        num_cells=num_cells,
        num_modules=num_modules,
        min_period=0.0,
        period_ratio=period_ratio,
        orientation_degrees=0.0,
        orientation_jitter_degrees=0.0,
        seed=seed,
    )


def test_bank_shapes_and_reproducible():
    wave_vectors, phases = _bank(num_cells=10, num_modules=4, seed=0)
    assert wave_vectors.shape == (10, 3, 2)
    assert phases.shape == (10, 2)
    wave_again, phase_again = _bank(num_cells=10, num_modules=4, seed=0)
    np.testing.assert_array_equal(wave_vectors, wave_again)
    np.testing.assert_array_equal(phases, phase_again)
    _, phase_other = _bank(num_cells=10, num_modules=4, seed=1)
    assert not np.array_equal(phases, phase_other)


def test_cells_split_across_modules_summing_to_num_cells():
    wave_vectors, phases = _bank(num_cells=513, num_modules=4, seed=0)
    assert wave_vectors.shape[0] == 513
    assert phases.shape[0] == 513


def test_periods_geometric_and_anchored_to_arena_max_side():
    wave_vectors, _ = _bank(num_cells=512, num_modules=4, period_ratio=1.42, seed=0)
    magnitudes = np.linalg.norm(wave_vectors[:, 0, :], axis=1)
    periods = 4.0 * np.pi / (np.sqrt(3.0) * magnitudes)
    unique_periods = np.unique(np.round(periods, 4))
    assert unique_periods.size == 4
    assert unique_periods.max() == pytest.approx(WALLGAP_MAX_SIDE, rel=1e-4)
    ratios = unique_periods[1:] / unique_periods[:-1]
    np.testing.assert_allclose(ratios, 1.42, rtol=1e-4)


def test_code_range_is_zero_to_one():
    wave_vectors, phases = _bank()
    rng = np.random.default_rng(0)
    positions = rng.uniform([-24.0, -30.0], [24.0, 36.0], size=(200, 2)).astype(np.float32)
    code = synthetic_grid_cell_code(positions, wave_vectors, phases)
    assert code.shape == (200, 512)
    assert code.min() >= -1e-5
    assert code.max() <= 1.0 + 1e-5


def test_cell_peaks_at_its_phase():
    wave_vectors, phases = _bank(num_cells=1, num_modules=1, seed=0)
    at_phase = synthetic_grid_cell_code(phases[0:1], wave_vectors, phases)
    assert at_phase[0, 0] == pytest.approx(1.0, abs=1e-5)
    period = 4.0 * np.pi / (np.sqrt(3.0) * np.linalg.norm(wave_vectors[0, 0]))
    offset = phases[0:1] + np.array([[period / 2.0, 0.0]], dtype=np.float32)
    away = synthetic_grid_cell_code(offset, wave_vectors, phases)
    assert away[0, 0] < at_phase[0, 0]


def test_code_uses_second_coordinate_z():
    wave_vectors, phases = _bank(num_cells=64, num_modules=2, seed=0)
    a = synthetic_grid_cell_code(np.array([[0.0, 0.0]], np.float32), wave_vectors, phases)
    b = synthetic_grid_cell_code(np.array([[0.0, 12.0]], np.float32), wave_vectors, phases)
    assert not np.allclose(a, b)


def test_l2_normalization_makes_unit_rows():
    wave_vectors, phases = _bank(num_cells=64, num_modules=2, seed=0)
    positions = np.array([[1.0, 2.0], [8.0, -5.0]], dtype=np.float32)
    normed = synthetic_grid_cell_code(positions, wave_vectors, phases, normalization="l2")
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, atol=1e-5)


def test_unknown_normalization_raises():
    wave_vectors, phases = _bank(num_cells=4, num_modules=1, seed=0)
    with pytest.raises(ValueError):
        synthetic_grid_cell_code(
            np.zeros((1, 2), np.float32), wave_vectors, phases, normalization="bogus"
        )


def test_population_code_is_injective_over_arena():
    wave_vectors, phases = _bank(num_cells=512, num_modules=4, seed=0)
    xs = np.linspace(-24.0, 24.0, 40)
    zs = np.linspace(-30.0, 36.0, 40)
    grid_x, grid_z = np.meshgrid(xs, zs)
    reference_positions = np.stack([grid_x.ravel(), grid_z.ravel()], axis=1).astype(np.float32)
    reference_codes = synthetic_grid_cell_code(reference_positions, wave_vectors, phases)

    rng = np.random.default_rng(3)
    queries = rng.uniform([-24.0, -30.0], [24.0, 36.0], size=(50, 2)).astype(np.float32)
    query_codes = synthetic_grid_cell_code(queries, wave_vectors, phases)
    distances = np.linalg.norm(query_codes[:, None, :] - reference_codes[None, :, :], axis=2)
    decoded = reference_positions[np.argmin(distances, axis=1)]
    decode_error = np.linalg.norm(decoded - queries, axis=1)
    reference_spacing = np.hypot(xs[1] - xs[0], zs[1] - zs[0])
    assert np.median(decode_error) < 3.0 * reference_spacing


def test_feature_source_reads_position_and_dim():
    from placecell_research.downstream.feature_sources import (
        StepContext,
        SyntheticGridCellsFeatureSource,
    )

    wave_vectors, phases = _bank(num_cells=8, num_modules=2, seed=0)
    source = SyntheticGridCellsFeatureSource(wave_vectors=wave_vectors, phases=phases)
    assert source.name == "synthetic_grid_cells"
    assert source.feature_dim == 8
    context = StepContext(
        rgb=np.zeros((3, 4, 4), dtype=np.uint8),
        position_xy=np.array([3.0, -7.0], dtype=np.float32),
        heading=0.0,
        kinematics=None,
        goal_position_xy=None,
    )
    out = source.extract(context, None)
    assert out.shape == (8,)
    expected = synthetic_grid_cell_code(np.array([[3.0, -7.0]], np.float32), wave_vectors, phases)[
        0
    ]
    np.testing.assert_allclose(out, expected, atol=1e-6)


def _grid_observation(num_cells: int):
    from placecell_research.config.downstream_schema import (
        DownstreamObservationConfig,
        SyntheticGridCellsConfig,
    )

    return DownstreamObservationConfig(
        feature_sources=["synthetic_grid_cells"],
        synthetic_grid_cells=SyntheticGridCellsConfig(num_cells=num_cells, num_modules=4, seed=0),
    )


def _grid_goal_observation(num_cells: int):
    from placecell_research.config.downstream_schema import (
        DownstreamObservationConfig,
        SyntheticGridCellsConfig,
    )

    return DownstreamObservationConfig(
        feature_sources=["synthetic_grid_cells", "goal_grid_code"],
        synthetic_grid_cells=SyntheticGridCellsConfig(num_cells=num_cells, num_modules=4, seed=0),
    )


def test_goal_grid_source_reads_goal_position_and_dim():
    from placecell_research.downstream.feature_sources import (
        GoalGridCellsFeatureSource,
        StepContext,
    )

    wave_vectors, phases = _bank(num_cells=8, num_modules=2, seed=0)
    source = GoalGridCellsFeatureSource(wave_vectors=wave_vectors, phases=phases)
    assert source.name == "goal_grid_code"
    assert source.feature_dim == 8
    goal = np.array([5.0, -3.0], dtype=np.float32)
    context = StepContext(
        rgb=np.zeros((3, 4, 4), dtype=np.uint8),
        position_xy=np.array([1.0, 1.0], dtype=np.float32),
        heading=0.0,
        kinematics=None,
        goal_position_xy=goal,
    )
    out = source.extract(context, None)
    expected = synthetic_grid_cell_code(goal.reshape(1, 2), wave_vectors, phases)[0]
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_goal_grid_source_requires_a_goal():
    from placecell_research.downstream.feature_sources import (
        GoalGridCellsFeatureSource,
        StepContext,
    )

    wave_vectors, phases = _bank(num_cells=8, num_modules=2, seed=0)
    source = GoalGridCellsFeatureSource(wave_vectors=wave_vectors, phases=phases)
    context = StepContext(
        rgb=np.zeros((3, 4, 4), dtype=np.uint8),
        position_xy=np.array([1.0, 1.0], dtype=np.float32),
        heading=0.0,
        kinematics=None,
        goal_position_xy=None,
    )
    with pytest.raises(ValueError):
        source.extract(context, None)


def test_current_and_goal_grid_share_basis(tmp_path):
    from placecell_research.artifacts.registry import ArtifactRegistry
    from placecell_research.config.downstream_schema import DownstreamModelConfig
    from placecell_research.downstream.feature_sources import StepContext
    from placecell_research.downstream.runtime import build_feature_extractor

    extractor = build_feature_extractor(
        artifact_registry=ArtifactRegistry(tmp_path),
        models=DownstreamModelConfig(),
        observation=_grid_goal_observation(16),
        env_id="MiniWorld-WallGapAsymLarge-v0",
        goal_candidate_positions_xy=[],
        goal_rbf_sigma=1.5,
        device="cpu",
    )
    assert extractor.feature_dim == 32

    a = np.array([3.0, -7.0], dtype=np.float32)
    b = np.array([-11.0, 20.0], dtype=np.float32)
    at_a_goal_b = extractor.extract(
        StepContext(
            rgb=np.zeros((3, 4, 4), np.uint8),
            position_xy=a,
            heading=0.0,
            kinematics=None,
            goal_position_xy=b,
        ),
        None,
    )
    at_b_goal_a = extractor.extract(
        StepContext(
            rgb=np.zeros((3, 4, 4), np.uint8),
            position_xy=b,
            heading=0.0,
            kinematics=None,
            goal_position_xy=a,
        ),
        None,
    )
    current_at_a, goal_is_b = at_a_goal_b[:16], at_a_goal_b[16:]
    current_at_b, goal_is_a = at_b_goal_a[:16], at_b_goal_a[16:]
    np.testing.assert_allclose(goal_is_b, current_at_b, atol=1e-6)
    np.testing.assert_allclose(goal_is_a, current_at_a, atol=1e-6)


def test_paired_grid_code_extractor_preserves_aligned_interactions():
    import gymnasium as gym
    import torch

    from placecell_research.config.downstream_schema import (
        DownstreamObservationConfig,
        SyntheticGridCellsConfig,
    )
    from placecell_research.downstream.feature_extractors import PairedGridCodeExtractor

    observation_config = DownstreamObservationConfig(
        feature_sources=["synthetic_grid_cells", "goal_grid_code", "heading_sin_cos"],
        synthetic_grid_cells=SyntheticGridCellsConfig(num_cells=2, num_modules=1),
    )
    extractor = PairedGridCodeExtractor(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        observation_config,
    )
    observation = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.5, -0.5]])

    output = extractor(observation)

    assert extractor.features_dim == 10
    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0, 3.0, 4.0, -2.0, -2.0, 3.0, 8.0, 0.5, -0.5]]),
    )


def test_split_position_goal_infers_runtime_goal_place_code_width():
    import gymnasium as gym
    import torch

    from placecell_research.config.downstream_schema import (
        DownstreamObservationConfig,
        SyntheticPlaceCellsConfig,
    )
    from placecell_research.downstream.feature_extractors import SplitPositionGoalExtractor

    observation_config = DownstreamObservationConfig(
        feature_sources=["synthetic_place_cells", "goal_place_code", "heading_sin_cos"],
        synthetic_place_cells=SyntheticPlaceCellsConfig(num_cells=4),
    )
    extractor = SplitPositionGoalExtractor(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
        observation_config,
        embed_dim=3,
    )

    output = extractor(torch.zeros((2, 9)))

    assert extractor.features_dim == 14
    assert output.shape == (2, 14)


def test_paired_grid_code_policy_wiring_uses_custom_extractor():
    from placecell_research.config.downstream_schema import (
        DownstreamObservationConfig,
        DownstreamRunConfig,
        DownstreamTrainingConfig,
        SyntheticGridCellsConfig,
    )
    from placecell_research.downstream.feature_extractors import PairedGridCodeExtractor
    from placecell_research.downstream.sb3_algorithms import _build_policy_kwargs

    config = DownstreamRunConfig(
        observation=DownstreamObservationConfig(
            feature_sources=["synthetic_grid_cells", "goal_grid_code", "heading_sin_cos"],
            synthetic_grid_cells=SyntheticGridCellsConfig(num_cells=8, num_modules=2),
        ),
        training=DownstreamTrainingConfig(feature_extractor="paired_grid_code"),
    )

    policy_kwargs = _build_policy_kwargs(config, policy_name="MlpPolicy")

    assert policy_kwargs is not None
    assert policy_kwargs["features_extractor_class"] is PairedGridCodeExtractor
    assert policy_kwargs["features_extractor_kwargs"] == {
        "observation_config": config.observation,
    }


def test_goal_grid_single_and_vectorized_paths_match(tmp_path):
    from placecell_research.artifacts.registry import ArtifactRegistry
    from placecell_research.config.downstream_schema import DownstreamModelConfig
    from placecell_research.downstream.feature_sources import StepContext
    from placecell_research.downstream.runtime import build_feature_extractor
    from placecell_research.downstream.shared_feature_vec_env import build_shared_feature_pipeline

    observation = _grid_goal_observation(16)
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
    goal = np.array([-11.0, 20.0], dtype=np.float32)
    single = extractor.extract(
        StepContext(
            rgb=np.zeros((3, 4, 4), np.uint8),
            position_xy=position,
            heading=0.0,
            kinematics=None,
            goal_position_xy=goal,
        ),
        None,
    )
    batched = pipeline.encode(
        rgb_batch=np.zeros((1, 3, 4, 4), dtype=np.uint8),
        positions_xy=position.reshape(1, 2),
        headings=np.zeros(1, dtype=np.float32),
        kinematics_batch=np.zeros((1, 4), dtype=np.float32),
        goal_positions_xy=goal.reshape(1, 2),
        goal_place_codes=None,
        previous_actions=np.zeros(1, dtype=np.int64),
        indices=np.array([0], dtype=np.int64),
    )
    assert single.shape == (32,)
    assert batched.shape == (1, 32)
    np.testing.assert_allclose(single, batched[0], atol=1e-6)


def test_build_feature_extractor_grid_only(tmp_path):
    from placecell_research.artifacts.registry import ArtifactRegistry
    from placecell_research.config.downstream_schema import DownstreamModelConfig
    from placecell_research.downstream.runtime import build_feature_extractor

    extractor = build_feature_extractor(
        artifact_registry=ArtifactRegistry(tmp_path),
        models=DownstreamModelConfig(),
        observation=_grid_observation(16),
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

    observation = _grid_observation(32)
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
