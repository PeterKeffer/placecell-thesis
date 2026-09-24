from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.probing import distance_to_segments
from placecell_research.analysis.registry import ANALYSIS_MODULES
from placecell_research.analysis.selectivity import (
    SelectivityPartitionModule,
    SpatialCodeTypeModule,
)
from placecell_research.analysis.world_overlay import overlay_bounds, resolve_world_overlay


def _selectivity_input(
    env_id: str = "", *, position_scale: float = 5.0, with_latent: bool = False
) -> AnalysisInput:
    """Synthetic episodes with one pure place, heading, speed, and noise unit."""
    rng = np.random.default_rng(0)
    episodes, steps = 16, 80
    positions = rng.uniform(-position_scale, position_scale, size=(episodes, steps, 2)).astype(
        np.float32
    )
    heading = rng.uniform(-np.pi, np.pi, size=(episodes, steps)).astype(np.float32)
    speed = rng.uniform(0.1, 1.0, size=(episodes, steps)).astype(np.float32)
    angular_velocity = rng.uniform(-0.5, 0.5, size=(episodes, steps)).astype(np.float32)
    kinematics = np.stack([speed, angular_velocity], axis=-1).astype(np.float32)

    center = np.array([0.4 * position_scale, 0.4 * position_scale], dtype=np.float32)
    squared_distance = np.sum((positions - center) ** 2, axis=-1)
    place_unit = np.exp(-squared_distance / (2.0 * (0.6 * position_scale) ** 2))
    heading_unit = np.cos(heading)
    speed_unit = (speed - speed.mean()) / speed.std()
    noise_unit = rng.normal(size=(episodes, steps)).astype(np.float32)

    noise = 0.02 * rng.normal(size=(episodes, steps, 4)).astype(np.float32)
    representation = np.stack(
        [place_unit, heading_unit, speed_unit, noise_unit], axis=-1
    ).astype(np.float32)
    representation = representation + noise

    latent = (
        rng.normal(size=(episodes, steps, 8)).astype(np.float32) if with_latent else None
    )
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=heading,
        kinematics=kinematics,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder_place_cells",
        label="encoder",
        split_name="test",
        latent=latent,
        metadata={"env_id": env_id},
    )


_PARTITION_CONFIG = {
    "selectivity_position_centers_x": 5,
    "selectivity_position_centers_y": 5,
    "selectivity_min_r2": 0.05,
    "selectivity_mixed_dominance": 0.6,
}


def test_selectivity_modules_are_registered() -> None:
    assert ANALYSIS_MODULES["selectivity_partition"] is SelectivityPartitionModule
    assert ANALYSIS_MODULES["spatial_code_type"] is SpatialCodeTypeModule


def test_partition_recovers_planted_selectivity(tmp_path: Path) -> None:
    result = SelectivityPartitionModule().run(_selectivity_input(), tmp_path, _PARTITION_CONFIG)

    categories = {
        unit: category
        for unit, category in enumerate(_read_categories(result))
    }
    assert categories[0] == "place_like"
    assert categories[1] == "hd_like"
    assert categories[2] == "speed_like"
    assert categories[3] == "untuned"

    assert result.per_unit_metrics["full_r2"][0] > 0.5
    assert result.per_unit_metrics["unique_r2_position"][0] > result.per_unit_metrics[
        "unique_r2_heading"
    ][0]
    assert {"position", "heading", "speed", "angular_velocity", "time"} == set(
        result.metadata["selectivity_groups"]
    )
    assert abs(sum(_category_fractions(result).values()) - 1.0) < 1e-6
    assert result.figures["selectivity_partition"].exists()
    assert result.figures["selectivity_decomposition"].exists()
    assert result.figures["selectivity_unit_scatter"].exists()
    assert result.tables["selectivity_partition"].exists()


def test_selectivity_partition_requests_latent() -> None:
    assert SelectivityPartitionModule().required_batch_keys() == {"latent"}


def test_partition_adds_visual_source_when_latent_present(tmp_path: Path) -> None:
    result = SelectivityPartitionModule().run(
        _selectivity_input(with_latent=True), tmp_path, _PARTITION_CONFIG
    )
    assert "visual" in result.metadata["selectivity_groups"]
    assert "mean_unique_r2_visual" in result.metrics
    assert "fraction_visual_like" in result.metrics
    assert abs(sum(_category_fractions(result).values()) - 1.0) < 1e-6
    assert result.figures["selectivity_decomposition"].exists()


def _code_type_input(env_id: str = "MiniWorld-WallGapAsymLarge-v0") -> AnalysisInput:
    """Unit 0 is a place bump; unit 1 is a pure function of nearest-wall distance."""
    overlay = resolve_world_overlay(env_id)
    assert overlay is not None and overlay.segments
    (x_low, x_high), (y_low, y_high) = overlay_bounds(overlay)
    rng = np.random.default_rng(1)
    episodes, steps = 16, 80
    x = rng.uniform(x_low, x_high, size=(episodes, steps)).astype(np.float32)
    y = rng.uniform(y_low, y_high, size=(episodes, steps)).astype(np.float32)
    positions = np.stack([x, y], axis=-1).astype(np.float32)

    wall_distance = distance_to_segments(positions, overlay.segments)
    center_x, center_y = 0.3 * x_high, 0.3 * y_high
    sigma = 0.2 * (x_high - x_low)
    place_unit = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * sigma**2))
    boundary_unit = (wall_distance - wall_distance.mean()) / max(wall_distance.std(), 1e-6)
    noise = 0.02 * rng.normal(size=(episodes, steps, 2)).astype(np.float32)
    representation = np.stack([place_unit, boundary_unit], axis=-1).astype(np.float32) + noise

    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder_place_cells",
        label="encoder",
        split_name="test",
        metadata={"env_id": env_id},
    )


def test_spatial_code_type_flags_boundary_unit_not_place_unit(tmp_path: Path) -> None:
    result = SpatialCodeTypeModule().run(
        _code_type_input(),
        tmp_path,
        {"spatial_code_min_r2": 0.05, "spatial_code_sufficiency_threshold": 0.8},
    )

    boundary_sufficiency = result.per_unit_metrics["boundary_vector_sufficiency"]
    assert boundary_sufficiency[1] >= 0.8
    assert boundary_sufficiency[0] < boundary_sufficiency[1]
    assert boundary_sufficiency[0] < 0.8
    assert np.isfinite(result.metrics["mean_r2_allocentric_place"])
    assert "fraction_boundary_vector_sufficient" in result.metrics
    assert result.figures["spatial_code_type"].exists()
    assert result.tables["spatial_code_type"].exists()


def _sparse_code_input() -> AnalysisInput:
    """k-winners-like codes: each step keeps the top-3 of 40 units, the rest exact zero."""
    rng = np.random.default_rng(2)
    episodes, steps, units = 16, 80, 40
    positions = rng.uniform(-5.0, 5.0, size=(episodes, steps, 2)).astype(np.float32)
    heading = rng.uniform(-np.pi, np.pi, size=(episodes, steps)).astype(np.float32)
    kinematics = np.stack(
        [
            rng.uniform(0.1, 1.0, size=(episodes, steps)),
            rng.uniform(-0.5, 0.5, size=(episodes, steps)),
        ],
        axis=-1,
    ).astype(np.float32)

    centers = rng.uniform(-5.0, 5.0, size=(units, 2)).astype(np.float32)
    squared_distance = np.sum((positions[..., None, :] - centers) ** 2, axis=-1)
    bumps = np.exp(-squared_distance / (2.0 * 1.5**2))
    winners = np.argsort(-bumps, axis=-1)[..., :3]
    keep = np.zeros_like(bumps, dtype=bool)
    np.put_along_axis(keep, winners, True, axis=-1)
    representation = np.where(keep, bumps, 0.0).astype(np.float32)

    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=heading,
        kinematics=kinematics,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder_place_cells",
        label="encoder",
        split_name="test",
        metadata={"env_id": ""},
    )


def test_partition_handles_sparse_exact_zero_codes(tmp_path: Path) -> None:
    result = SelectivityPartitionModule().run(_sparse_code_input(), tmp_path, _PARTITION_CONFIG)

    assert np.all(np.isfinite(result.per_unit_metrics["full_r2"]))
    assert np.all(np.isfinite(result.per_unit_metrics["unique_r2_position"]))
    assert result.metrics["fraction_place_like"] > 0.0


def test_spatial_code_type_without_overlay_uses_place_only(tmp_path: Path) -> None:
    result = SpatialCodeTypeModule().run(
        _selectivity_input("", position_scale=15.0),
        tmp_path,
        {"spatial_code_min_r2": 0.05},
    )

    assert result.metadata["spatial_code_models"] == ["allocentric_place"]
    assert "mean_r2_allocentric_place" in result.metrics
    assert "r2_boundary_vector" not in result.per_unit_metrics


def _read_categories(result) -> list[str]:
    table_path = result.tables["selectivity_partition"]
    lines = table_path.read_text().strip().splitlines()
    return [row.split(",")[-1] for row in lines[1:]]


def _category_fractions(result) -> dict[str, float]:
    return {
        key.removeprefix("fraction_"): value
        for key, value in result.metrics.items()
        if key.startswith("fraction_")
    }
