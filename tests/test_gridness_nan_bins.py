"""Regression tests: gridness must survive NaN (unvisited) rate-map bins."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.gridness import GridnessModule
from placecell_research.numerics.rate_map_kernels import gridness_score


def hex_bump_map(size: int = 48, spacing: float = 12.0, sigma: float = 2.5) -> np.ndarray:
    """Hexagonal grid of localized Gaussian fields (a high-gridness ground truth)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    basis_a = spacing * np.array([1.0, 0.0])
    basis_b = spacing * np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])
    field = np.zeros((size, size), dtype=np.float64)
    span = range(-4, size // int(spacing) + 5)
    for i in span:
        for j in span:
            cx = i * basis_a[0] + j * basis_b[0]
            cy = i * basis_a[1] + j * basis_b[1]
            field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return field


def test_gridness_score_finite_with_unvisited_nan_bins():
    rate_map = hex_bump_map()
    clean = gridness_score(rate_map)
    assert np.isfinite(clean)
    assert clean > 0.3

    masked = rate_map.copy()
    masked[::7, ::5] = np.nan
    masked_score = gridness_score(masked)
    assert np.isfinite(masked_score)


def _gridness_input_with_unvisited_bins() -> AnalysisInput:
    """One episode sampled on a coarse raster; high-res binning leaves NaN bins."""
    field = hex_bump_map(size=64, spacing=12.0, sigma=2.0)
    coords = [(x, y) for y in range(0, 8) for x in range(0, 8)]
    steps = len(coords)
    positions = np.zeros((1, steps, 2), dtype=np.float32)
    representation = np.zeros((1, steps, 2), dtype=np.float32)
    for step, (cx, cy) in enumerate(coords):
        positions[0, step] = (cx, cy)
        representation[0, step, 0] = field[cy * 8, cx * 8]
        representation[0, step, 1] = float((cx + cy) % 2)
    return AnalysisInput(
        representation=representation,
        position_xy=positions,
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, steps), dtype=bool),
        source_name="grid.hidden_state",
        label="grid_cells",
        split_name="validation",
    )


def test_gridness_module_aggregates_finite_with_unvisited_bins(tmp_path: Path):
    config = {"num_bins_x": 20, "num_bins_y": 20, "smoothing_sigma": 0.0}
    result = GridnessModule().run(_gridness_input_with_unvisited_bins(), tmp_path, config)
    assert np.isfinite(result.metrics["mean_gridness"])
    assert np.isfinite(result.metrics["max_gridness"])
    for key in (
        "num_grid_units",
        "num_grid_units_strong",
        "fraction_grid",
        "gridness_scorable_units",
    ):
        assert key in result.metrics
        assert np.isfinite(result.metrics[key])
    assert result.metrics["max_gridness"] >= result.metrics["mean_gridness"]
