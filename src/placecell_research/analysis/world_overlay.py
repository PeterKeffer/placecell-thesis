"""Static environment geometry and landmark overlays for analysis figures."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

WorldPoint = tuple[float, float]
WorldSegment = tuple[WorldPoint, WorldPoint]

POSITION_X_LABEL = "World x"
POSITION_Y_LABEL = "World y"


def style_arena_axes(axis: plt.Axes, *, label_x: bool = True, label_y: bool = True) -> None:
    """Apply the shared Nature-style treatment to an arena/position heatmap axis."""
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlabel(POSITION_X_LABEL if label_x else "")
    axis.set_ylabel(POSITION_Y_LABEL if label_y else "")


@dataclass(frozen=True, slots=True)
class LandmarkLayer:
    label: str
    marker: str
    color: str
    positions: tuple[WorldPoint, ...]


@dataclass(frozen=True, slots=True)
class WorldOverlay:
    env_id: str
    segments: tuple[WorldSegment, ...]
    landmarks: tuple[LandmarkLayer, ...] = ()
    y_down: bool = False


def _wall_gap_asym_large_segments() -> tuple[WorldSegment, ...]:
    return (
        ((-18.0, 6.0), (-18.0, 36.0)),
        ((18.0, 6.0), (18.0, 36.0)),
        ((-18.0, 36.0), (18.0, 36.0)),
        ((-18.0, 6.0), (-3.0, 6.0)),
        ((3.0, 6.0), (18.0, 6.0)),
        ((-6.0, 6.0), (-3.0, 6.0)),
        ((3.0, 6.0), (6.0, 6.0)),
        ((-6.0, -6.0), (6.0, -6.0)),
        ((-6.0, 6.0), (-6.0, 3.0)),
        ((-6.0, -3.0), (-6.0, -6.0)),
        ((6.0, 6.0), (6.0, 3.0)),
        ((6.0, -3.0), (6.0, -6.0)),
        ((-24.0, -30.0), (-6.0, -30.0)),
        ((-24.0, 4.5), (-6.0, 4.5)),
        ((-24.0, -30.0), (-24.0, 4.5)),
        ((-6.0, 4.5), (-6.0, 3.0)),
        ((-6.0, -3.0), (-6.0, -30.0)),
        ((6.0, -30.0), (24.0, -30.0)),
        ((6.0, 4.5), (24.0, 4.5)),
        ((24.0, -30.0), (24.0, 4.5)),
        ((6.0, 4.5), (6.0, 3.0)),
        ((6.0, -3.0), (6.0, -30.0)),
    )


WALLGAP_ASYM_LARGE_OVERLAY = WorldOverlay(
    env_id="MiniWorld-WallGapAsymLarge-v0",
    segments=_wall_gap_asym_large_segments(),
    landmarks=(
        LandmarkLayer(
            label="courtyard_trees",
            marker="^",
            color="#2E8B57",
            positions=(
                (-15.0, 9.0),
                (-14.0, 14.0),
                (-16.0, 19.0),
                (-13.0, 23.0),
                (-15.0, 28.0),
                (-17.0, 33.0),
                (-5.0, 11.0),
                (-8.0, 18.0),
                (-3.0, 25.0),
                (2.0, 13.0),
                (7.0, 22.0),
                (4.0, 31.0),
            ),
        ),
        LandmarkLayer(
            label="courtyard_cones",
            marker="o",
            color="#FF8C00",
            positions=(
                (12.0, 8.0),
                (14.0, 12.0),
                (13.0, 17.0),
                (15.0, 21.0),
                (12.0, 25.0),
                (14.0, 29.0),
                (16.0, 32.0),
                (11.0, 35.0),
            ),
        ),
        LandmarkLayer(
            label="duckies",
            marker="D",
            color="#FFD54F",
            positions=((-2.0, -4.0), (4.0, 4.0), (-3.0, 2.0), (2.0, -3.0)),
        ),
        LandmarkLayer(
            label="office_desks",
            marker="p",
            color="#00CED1",
            positions=((-4.0, 3.0), (3.0, -2.0)),
        ),
        LandmarkLayer(
            label="left_pines",
            marker="v",
            color="#006400",
            positions=(
                (-21.0, -8.0),
                (-18.0, -15.0),
                (-22.0, -21.0),
                (-15.0, -25.0),
                (-20.0, -28.0),
                (-17.0, 2.0),
            ),
        ),
        LandmarkLayer(
            label="office_chairs",
            marker="h",
            color="#FF00FF",
            positions=((-10.0, 0.0), (-19.0, -10.0), (-11.0, -18.0), (-16.0, -24.0)),
        ),
        LandmarkLayer(
            label="barrels",
            marker="s",
            color="#8B4513",
            positions=(
                (-10.0, 32.0),
                (8.0, 10.0),
                (0.0, 28.0),
                (-9.0, -5.0),
                (-12.0, -12.0),
                (-8.0, -20.0),
                (-14.0, -27.0),
                (9.0, -6.0),
                (15.0, -11.0),
                (12.0, -18.0),
                (18.0, -22.0),
                (10.0, -27.0),
            ),
        ),
        LandmarkLayer(
            label="barriers",
            marker="P",
            color="#7F7F7F",
            positions=(
                (5.0, 16.0),
                (-2.0, 24.0),
                (-13.0, -3.0),
                (-20.0, -16.0),
                (-9.0, -26.0),
                (8.0, -3.0),
                (20.0, -14.0),
                (14.0, -25.0),
            ),
        ),
        LandmarkLayer(
            label="goal",
            marker="X",
            color="#D62728",
            positions=((18.0, -22.0),),
        ),
        LandmarkLayer(
            label="medkit",
            marker="P",
            color="#FFFFFF",
            positions=((21.0, -10.0),),
        ),
    ),
)


_WORLD_OVERLAYS: dict[str, WorldOverlay] = {
    "MiniWorld-WallGapAsymLarge-v0": WALLGAP_ASYM_LARGE_OVERLAY,
}


def resolve_world_overlay(
    env_id: str | None, env_kwargs: dict | None = None
) -> WorldOverlay | None:
    if not env_id:
        return None
    env_id = str(env_id)
    overlay = _WORLD_OVERLAYS.get(env_id)
    if overlay is not None:
        return overlay
    overlay = _build_jaxenstein_overlay(env_id)
    if overlay is not None:
        _WORLD_OVERLAYS[env_id] = overlay
    return overlay


def _build_jaxenstein_overlay(env_id: str) -> WorldOverlay | None:
    if env_id.startswith("MiniWorld-"):
        return None
    from placecell_research.envs.jaxenstein_maps import (
        CUSTOM_JAXENSTEIN_MAPS,
        build_jaxenstein_env,
        supports_goal_override,
    )

    is_custom_map = env_id in CUSTOM_JAXENSTEIN_MAPS
    try:
        is_known_map = is_custom_map or supports_goal_override(env_id)
    except ImportError:
        return None
    if not is_known_map:
        return None
    try:
        wall_grid = np.asarray(build_jaxenstein_env(env_id).maze.wall_grid)
        return world_overlay_from_wall_grid(env_id, wall_grid)
    except ImportError as exc:
        warnings.warn(
            f"JAXenstein overlay for known env_id {env_id!r} needs the optional [jax] backend, "
            f"but it could not be imported: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    except (AttributeError, RuntimeError, ValueError) as exc:
        warnings.warn(
            f"Failed to build JAXenstein overlay for known env_id {env_id!r}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def world_overlay_from_wall_grid(
    env_id: str,
    wall_grid: np.ndarray,
    *,
    cell_size: float = 1.0,
    origin: WorldPoint = (0.0, 0.0),
) -> WorldOverlay:
    """Build a WorldOverlay from a boolean tile grid (e.g."""
    grid = np.asarray(wall_grid)
    if grid.ndim != 2:
        raise ValueError(f"wall_grid must be 2D [rows, cols], got shape {grid.shape}.")
    grid = grid.astype(bool)
    rows, cols = grid.shape
    origin_x, origin_y = float(origin[0]), float(origin[1])

    def is_wall(row: int, col: int) -> bool:
        return 0 <= row < rows and 0 <= col < cols and bool(grid[row, col])

    segments: list[WorldSegment] = []
    for row, col in zip(*np.nonzero(grid), strict=False):
        row, col = int(row), int(col)
        x0, y0 = origin_x + col * cell_size, origin_y + row * cell_size
        x1, y1 = x0 + cell_size, y0 + cell_size
        if not is_wall(row - 1, col):
            segments.append(((x0, y0), (x1, y0)))
        if not is_wall(row + 1, col):
            segments.append(((x0, y1), (x1, y1)))
        if not is_wall(row, col - 1):
            segments.append(((x0, y0), (x0, y1)))
        if not is_wall(row, col + 1):
            segments.append(((x1, y0), (x1, y1)))
    return WorldOverlay(env_id=str(env_id), segments=tuple(segments), y_down=True)


def overlay_bounds(
    world_overlay: WorldOverlay, padding_fraction: float = 0.04
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_points = [point[0] for segment in world_overlay.segments for point in segment]
    y_points = [point[1] for segment in world_overlay.segments for point in segment]
    for layer in world_overlay.landmarks:
        x_points.extend(point[0] for point in layer.positions)
        y_points.extend(point[1] for point in layer.positions)
    min_x = float(min(x_points))
    max_x = float(max(x_points))
    min_y = float(min(y_points))
    max_y = float(max(y_points))
    pad_x = max(1e-6, (max_x - min_x) * padding_fraction)
    pad_y = max(1e-6, (max_y - min_y) * padding_fraction)
    return ((min_x - pad_x, max_x + pad_x), (min_y - pad_y, max_y + pad_y))


def _fallback_bounds_from_positions(
    position_xy: np.ndarray,
    *,
    padding_fraction: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if position_xy.ndim != 2 or position_xy.shape[1] != 2:
        flattened_positions = position_xy.reshape(-1, 2)
    else:
        flattened_positions = position_xy
    x_values = flattened_positions[:, 0]
    y_values = flattened_positions[:, 1]
    x_span = max(1e-6, float(x_values.max() - x_values.min()))
    y_span = max(1e-6, float(y_values.max() - y_values.min()))
    return (
        (
            float(x_values.min() - padding_fraction * x_span),
            float(x_values.max() + padding_fraction * x_span),
        ),
        (
            float(y_values.min() - padding_fraction * y_span),
            float(y_values.max() + padding_fraction * y_span),
        ),
    )


def resolve_plot_bounds(
    env_id: str | None,
    position_xy: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    world_overlay = resolve_world_overlay(env_id)
    if world_overlay is not None:
        return overlay_bounds(world_overlay)
    if position_xy is None or position_xy.size == 0:
        return None
    return _fallback_bounds_from_positions(position_xy)


def apply_plot_bounds(
    axis: plt.Axes,
    *,
    env_id: str | None,
    position_xy: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    bounds = resolve_plot_bounds(env_id, position_xy)
    if bounds is None:
        return None
    x_bounds, y_bounds = bounds
    finalize_arena_axis(
        axis,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        world_overlay=resolve_world_overlay(env_id),
    )
    return bounds


def draw_world_segments_on_axis(
    axis: plt.Axes,
    world_segments: Sequence[WorldSegment],
    *,
    line_color: str = "white",
    line_width: float = 1.3,
    alpha: float = 0.92,
    zorder: int = 6,
) -> None:
    if not world_segments:
        return
    line_collection = LineCollection(
        [[start_point, end_point] for start_point, end_point in world_segments],
        colors=line_color,
        linewidths=line_width,
        alpha=alpha,
        zorder=zorder,
    )
    axis.add_collection(line_collection)


def orient_topdown_yaxis(axis: plt.Axes, world_overlay: WorldOverlay | None) -> None:
    if world_overlay is None or not world_overlay.y_down:
        return
    low, high = sorted(axis.get_ylim())
    axis.set_ylim(high, low)


def finalize_arena_axis(
    axis: plt.Axes,
    *,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    world_overlay: WorldOverlay | None,
) -> None:
    axis.set_xlim(*x_bounds)
    axis.set_ylim(*y_bounds)
    axis.set_aspect("equal")
    orient_topdown_yaxis(axis, world_overlay)


def draw_landmarks_on_axis(
    axis: plt.Axes,
    world_overlay: WorldOverlay,
    *,
    marker_size: float = 22.0,
    edge_color: str = "black",
    edge_line_width: float = 0.35,
    alpha: float = 0.92,
    zorder: int = 7,
) -> None:
    for layer in world_overlay.landmarks:
        if not layer.positions:
            continue
        x_positions, y_positions = zip(*layer.positions, strict=False)
        axis.scatter(
            x_positions,
            y_positions,
            s=marker_size,
            marker=layer.marker,
            color=layer.color,
            edgecolors=edge_color,
            linewidths=edge_line_width,
            alpha=alpha,
            zorder=zorder,
        )
