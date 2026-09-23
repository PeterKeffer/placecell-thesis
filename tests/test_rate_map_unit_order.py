"""Spatial (field-position) ordering of rate-map units."""

from __future__ import annotations

import numpy as np

from placecell_research.analysis.rate_map_computation import (
    _field_center_xy,
    _order_indices_by_field_position,
)

_BOUNDS = ((0.0, 10.0), (0.0, 10.0))


def _blob(
    ny: int, nx: int, *, center_row: float, center_col: float, sigma: float = 1.6
) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    squared_distance = (xx - center_col) ** 2 + (yy - center_row) ** 2
    return np.exp(-squared_distance / (2.0 * sigma**2)).astype(np.float32)


def test_field_center_xy_locates_blob_and_reports_empty() -> None:
    ny, nx = 20, 20
    center = _field_center_xy(_blob(ny, nx, center_row=5, center_col=15), _BOUNDS)
    assert center is not None
    x, y = center
    assert abs(x - 7.75) < 0.5, f"x center off: {x}"
    assert abs(y - 2.75) < 0.5, f"y center off: {y}"
    assert _field_center_xy(np.zeros((ny, nx), dtype=np.float32), _BOUNDS) is None


def test_field_center_xy_ignores_negative_activity() -> None:
    ny, nx = 20, 20
    rate_map = _blob(ny, nx, center_row=4, center_col=4)
    rate_map[15:, 15:] = -1.0
    center = _field_center_xy(rate_map, _BOUNDS)
    assert center is not None
    x, y = center
    assert x < 5.0 and y < 5.0, f"center should track the positive lobe, got {(x, y)}"


def test_field_position_order_groups_colocated_fields() -> None:
    ny, nx = 20, 20
    top_left = _blob(ny, nx, center_row=3, center_col=3)
    bottom_right = _blob(ny, nx, center_row=16, center_col=16)
    rate_maps = np.stack([top_left, bottom_right, 0.9 * top_left, 0.9 * bottom_right])
    selected = np.array([0, 1, 2, 3])

    ordered = _order_indices_by_field_position(selected, rate_maps, _BOUNDS)

    order = ordered.tolist()
    assert sorted(order) == [0, 1, 2, 3], "ordering must be a permutation of the input"
    position = {int(unit): index for index, unit in enumerate(order)}
    assert abs(position[0] - position[2]) == 1, f"co-located 0,2 must be adjacent: {order}"
    assert abs(position[1] - position[3]) == 1, f"co-located 1,3 must be adjacent: {order}"


def test_field_position_order_appends_unlocatable_units_last() -> None:
    ny, nx = 20, 20
    rate_maps = np.stack(
        [
            _blob(ny, nx, center_row=3, center_col=3),
            np.zeros((ny, nx), dtype=np.float32),
            _blob(ny, nx, center_row=16, center_col=16),
        ]
    )
    selected = np.array([0, 1, 2])

    ordered = _order_indices_by_field_position(selected, rate_maps, _BOUNDS)

    assert sorted(ordered.tolist()) == [0, 1, 2], "ordering must be a permutation of the input"
    assert ordered.tolist()[-1] == 1, f"silent unit must sort last: {ordered.tolist()}"
