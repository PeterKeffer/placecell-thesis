"""Cognitive-map geometry: representational distance tracks geodesic vs Euclidean."""

from __future__ import annotations

import numpy as np
import pytest

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.cognitive_map_geometry import (
    CognitiveMapGeometryModule,
    _pairs_cross_wall,
    _partial_spearman_controls,
)
from placecell_research.analysis.registry import ANALYSIS_MODULES
from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay


def _u_corridor() -> tuple[np.ndarray, np.ndarray]:
    """Positions tracing a U, plus arc length (walked distance) at each step."""
    left_arm = [(0.0, float(y)) for y in np.arange(0.0, 10.0, 0.1)]
    top = [(float(x), 10.0) for x in np.arange(0.0, 4.0, 0.1)]
    right_arm = [(4.0, float(y)) for y in np.arange(10.0, 0.0, -0.1)]
    points = left_arm + top + right_arm
    arc = 0.1 * np.arange(len(points), dtype=np.float32)
    return np.asarray(points, dtype=np.float32), arc


def _gaussian_bumps(coordinate: np.ndarray, num_bumps: int, width: float) -> np.ndarray:
    centers = np.linspace(coordinate.min(), coordinate.max(), num_bumps)
    return np.exp(-((coordinate[:, None] - centers[None, :]) ** 2) / (2 * width**2)).astype(
        np.float32
    )


def _analysis_input(
    representation: np.ndarray, positions: np.ndarray, latent: np.ndarray | None = None
) -> AnalysisInput:
    representation = np.tile(representation, (3, 1))
    positions = np.tile(positions, (3, 1))
    steps = positions.shape[0]
    latent_array = None if latent is None else np.tile(latent, (3, 1))[None, :, :]
    return AnalysisInput(
        representation=representation[None, :, :],
        position_xy=positions[None, :, :],
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="test",
        split_name="test",
        latent=latent_array,
        metadata={"env_id": "synthetic-u-corridor"},
    )


def _euclidean_visual_latent(positions: np.ndarray) -> np.ndarray:
    """A visual latent that is a function of straight-line position (Euclidean-like)."""
    x_bumps = _gaussian_bumps(positions[:, 0], num_bumps=5, width=1.5)
    y_bumps = _gaussian_bumps(positions[:, 1], num_bumps=5, width=1.5)
    return (x_bumps[:, :, None] * y_bumps[:, None, :]).reshape(positions.shape[0], 25)


_CONFIG = {"cognitive_map_num_bins_x": 12, "cognitive_map_num_bins_y": 12}


def test_pairs_cross_wall_detects_straddling_segments() -> None:
    segments = (((0.0, -1.0), (0.0, 1.0)),)
    centers_i = np.array([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 5.0]])
    centers_j = np.array([[1.0, 0.0], [-0.5, 0.0], [1.0, 5.0]])
    crosses = _pairs_cross_wall(centers_i, centers_j, segments)
    assert crosses.tolist() == [True, False, False]


def test_partial_spearman_is_zero_when_a_control_fully_explains_the_target() -> None:
    predictor = np.asarray([0.2, 1.1, 0.4, 2.0, 0.7])
    control = np.asarray([3.0, 1.0, 4.0, 2.0, 5.0])

    assert _partial_spearman_controls(control, predictor, [control]) == 0.0


def _arena_walk(env_id: str, steps: int, seed: int) -> np.ndarray:
    overlay = resolve_world_overlay(env_id)
    assert overlay is not None
    (x_low, x_high), (y_low, y_high) = overlay_bounds(overlay)
    rng = np.random.default_rng(seed)
    step_scale = 0.06 * min(x_high - x_low, y_high - y_low)
    position = np.array([(x_low + x_high) / 2.0, (y_low + y_high) / 2.0])
    walk = np.empty((steps, 2), dtype=np.float32)
    for t in range(steps):
        walk[t] = position
        position = np.clip(
            position + rng.normal(0.0, step_scale, size=2), [x_low, y_low], [x_high, y_high]
        )
    return walk


def test_wall_scatter_renders_with_real_walls(tmp_path) -> None:
    env_id = "MiniWorld-WallGapAsymLarge-v0"
    positions = _arena_walk(env_id, steps=800, seed=0)
    centers = positions[np.linspace(0, len(positions) - 1, 12).astype(int)]
    squared_distance = np.sum((positions[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    representation = np.exp(-squared_distance / (2.0 * 4.0**2)).astype(np.float32)

    analysis_input = AnalysisInput(
        representation=representation[None, :, :],
        position_xy=positions[None, :, :],
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, len(positions)), dtype=bool),
        source_name="encoder.place_codes",
        label="test",
        split_name="test",
        metadata={"env_id": env_id},
    )
    result = CognitiveMapGeometryModule().run(
        analysis_input, tmp_path, {"cognitive_map_num_bins_x": 14, "cognitive_map_num_bins_y": 14}
    )

    assert "cognitive_map_wall_separated_pair_fraction" in result.metrics
    assert 0.0 < result.metrics["cognitive_map_wall_separated_pair_fraction"] < 1.0
    assert "cognitive_map_wall_repr_gap_matched" in result.metrics
    assert np.isfinite(result.metrics["cognitive_map_wall_repr_gap_matched"])
    assert result.figures["cognitive_map_wall_scatter"].exists()


def test_cognitive_map_geometry_is_registered() -> None:
    assert "cognitive_map_geometry" in ANALYSIS_MODULES
    assert ANALYSIS_MODULES["cognitive_map_geometry"]().name == "cognitive_map_geometry"


def test_path_code_reads_as_topological(tmp_path) -> None:
    positions, arc = _u_corridor()
    representation = _gaussian_bumps(arc, num_bumps=6, width=1.2)
    module = CognitiveMapGeometryModule()
    result = module.run(_analysis_input(representation, positions), tmp_path, _CONFIG)

    metrics = result.metrics
    assert metrics["cognitive_map_topological_index"] > 0.05
    assert (
        metrics["cognitive_map_partial_spearman_geodesic"]
        > metrics["cognitive_map_partial_spearman_euclidean"]
    )
    assert all(path.exists() for path in result.figures.values())


def test_required_batch_keys_includes_latent() -> None:
    assert CognitiveMapGeometryModule().required_batch_keys() == {"latent"}


@pytest.mark.parametrize("kind", ["all_zero", "dead_dims"])
def test_degenerate_code_reports_nan_instead_of_aborting_the_stage(tmp_path, kind: str) -> None:
    """A collapsed code has no population-vector correlation, so every RDM entry is NaN."""
    positions, _arc = _u_corridor()
    representation = np.zeros((positions.shape[0], 64), dtype=np.float32)
    if kind == "dead_dims":
        representation[:, :5] = np.random.default_rng(0).random((positions.shape[0], 5)) * 1e-20

    result = CognitiveMapGeometryModule().run(
        _analysis_input(representation, positions), tmp_path, _CONFIG
    )

    assert result.metrics["cognitive_map_skipped"] == 1.0
    assert np.isnan(result.metrics["cognitive_map_topological_index"])
    assert np.isnan(result.metrics["cognitive_map_spearman_repr_geodesic"])
    assert result.figures == {}
    assert "degenerate" in result.metadata["cognitive_map_skipped_reason"]


def test_too_few_spatial_bins_still_raises(tmp_path) -> None:
    """The degenerate-code path must not swallow a genuinely unusable binning."""
    positions = np.repeat(np.asarray([[0.0, 0.0]], dtype=np.float32), 40, axis=0)
    representation = _gaussian_bumps(np.arange(len(positions), dtype=np.float32), 6, 1.2)

    with pytest.raises(ValueError):
        CognitiveMapGeometryModule().run(
            _analysis_input(representation, positions), tmp_path, _CONFIG
        )


def test_rdm_figure_rendered_without_latent(tmp_path) -> None:
    positions, arc = _u_corridor()
    representation = _gaussian_bumps(arc, num_bumps=6, width=1.2)
    result = CognitiveMapGeometryModule().run(
        _analysis_input(representation, positions), tmp_path, _CONFIG
    )
    assert result.figures["cognitive_map_rdm"].exists()
    assert "cognitive_map_spearman_repr_visual" not in result.metrics


def test_rdm_visual_baseline_separates_path_from_visual_lookup(tmp_path) -> None:
    positions, arc = _u_corridor()
    visual_latent = _euclidean_visual_latent(positions)
    path_code = _gaussian_bumps(arc, num_bumps=6, width=1.2)

    module = CognitiveMapGeometryModule()
    path = module.run(
        _analysis_input(path_code, positions, latent=visual_latent), tmp_path, _CONFIG
    )
    lookup = module.run(
        _analysis_input(visual_latent, positions, latent=visual_latent), tmp_path, _CONFIG
    )

    partial_key = "cognitive_map_partial_spearman_geodesic_given_visual"
    partial_key_full = "cognitive_map_partial_spearman_geodesic_given_euclidean_visual"
    assert np.isfinite(path.metrics[partial_key])
    assert np.isfinite(path.metrics[partial_key_full])
    assert path.metrics[partial_key] > lookup.metrics[partial_key]
    assert (
        lookup.metrics["cognitive_map_spearman_repr_visual"]
        > path.metrics["cognitive_map_spearman_repr_visual"]
    )
    assert path.figures["cognitive_map_rdm"].exists()


def test_position_code_reads_as_more_metric_than_path_code(tmp_path) -> None:
    positions, arc = _u_corridor()
    path_code = _gaussian_bumps(arc, num_bumps=6, width=1.2)
    x_bumps = _gaussian_bumps(positions[:, 0], num_bumps=3, width=1.5)
    y_bumps = _gaussian_bumps(positions[:, 1], num_bumps=3, width=2.5)
    position_code = (x_bumps[:, :, None] * y_bumps[:, None, :]).reshape(positions.shape[0], 9)

    module = CognitiveMapGeometryModule()
    path_result = module.run(_analysis_input(path_code, positions), tmp_path, _CONFIG)
    position_result = module.run(_analysis_input(position_code, positions), tmp_path, _CONFIG)

    assert (
        position_result.metrics["cognitive_map_topological_index"]
        < path_result.metrics["cognitive_map_topological_index"]
    )
