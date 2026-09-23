"""Custom JAXenstein ASCII map for the museum environment, built via RayMazeEnv.from_ascii."""

from __future__ import annotations


def _carve_floor(grid: list[list[str]], x0: int, y0: int, x1: int, y1: int) -> None:
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            grid[y][x] = "."


def _carve_room(
    grid: list[list[str]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: str,
) -> None:
    for x in range(x0, x1 + 1):
        grid[y0][x] = color
        grid[y1][x] = color
    for y in range(y0, y1 + 1):
        grid[y][x0] = color
        grid[y][x1] = color
    for y in range(y0 + 1, y1):
        for x in range(x0 + 1, x1):
            grid[y][x] = "."


def _build_museum_gallery() -> str:
    width, height = 43, 25
    grid = [["#"] * width for _ in range(height)]

    top_rooms = [
        (3, 3, 11, 10, "1"), (13, 3, 21, 10, "2"), (23, 3, 31, 10, "3"), (33, 3, 39, 10, "4"),
    ]
    bottom_rooms = [
        (3, 14, 11, 21, "5"), (13, 14, 21, 21, "6"), (23, 14, 31, 21, "7"), (33, 14, 39, 21, "8"),
    ]
    for x0, y0, x1, y1, color in top_rooms + bottom_rooms:
        _carve_room(grid, x0, y0, x1, y1, color)

    _carve_floor(grid, 1, 1, 41, 2)
    _carve_floor(grid, 1, 22, 41, 23)
    _carve_floor(grid, 1, 1, 2, 23)
    _carve_floor(grid, 40, 1, 41, 23)
    _carve_floor(grid, 3, 11, 39, 13)

    for x0, _y0, x1, _y1, _c in top_rooms:
        cx = (x0 + x1) // 2
        grid[10][cx] = grid[10][cx + 1] = "."
        grid[3][cx] = grid[3][cx + 1] = "."
    for x0, _y0, x1, _y1, _c in bottom_rooms:
        cx = (x0 + x1) // 2
        grid[14][cx] = grid[14][cx + 1] = "."
        grid[21][cx] = grid[21][cx + 1] = "."

    for spawn_x in (6, 21, 36):
        grid[12][spawn_x] = "S"
    grid[18][36] = "G"
    keys = [(6, 6, "r"), (9, 8, "b"), (16, 6, "y"), (19, 8, "r"), (26, 6, "b"), (29, 8, "y"),
            (36, 6, "r"), (38, 8, "b"), (6, 17, "y"), (9, 19, "r"), (16, 17, "b"), (19, 19, "y"),
            (26, 17, "r"), (29, 19, "b"), (38, 19, "y")]
    for kx, ky, kc in keys:
        if grid[ky][kx] == ".":
            grid[ky][kx] = kc
    return "\n".join("".join(row) for row in grid)


CUSTOM_JAXENSTEIN_MAPS: dict[str, str] = {
    "museum-gallery": _build_museum_gallery(),
}


_ENV_BUILD_CACHE: dict = {}


def _base_ascii(env_id: str) -> str:
    if env_id in CUSTOM_JAXENSTEIN_MAPS:
        return CUSTOM_JAXENSTEIN_MAPS[env_id]
    raise ValueError(f"no base ASCII map for env_id {env_id!r}")


def _ascii_with_goal(base_ascii: str, goal_tile: tuple[int, int]) -> str:
    """Return the map with its G moved to goal_tile = (col, row)."""
    col, row = goal_tile
    grid = [list(line) for line in base_ascii.strip("\n").split("\n")]
    for grid_row in grid:
        for c, ch in enumerate(grid_row):
            if ch == "G":
                grid_row[c] = "."
    grid[row][col] = "G"
    if not any("S" in grid_row for grid_row in grid):
        for spawn_row_index, grid_row in enumerate(grid):
            for spawn_col_index, ch in enumerate(grid_row):
                if ch == ".":
                    grid[spawn_row_index][spawn_col_index] = "S"
                    return "\n".join("".join(line) for line in grid)
        raise ValueError("goal relocation removed every spawn and no replacement floor tile exists")
    return "\n".join("".join(line) for line in grid)


def supports_goal_override(env_id: str) -> bool:
    return env_id in CUSTOM_JAXENSTEIN_MAPS


def build_jaxenstein_env(
    env_id: str,
    goal_tile: tuple[int, int] | None = None,
    episode_horizon: int | None = None,
):
    """Build a JAXenstein env from a custom ASCII map, optionally with the goal moved."""
    key = (env_id, goal_tile, episode_horizon)
    cached = _ENV_BUILD_CACHE.get(key)
    if cached is not None:
        return cached
    from jaxenstein.env import RayMazeEnv

    horizon_kwargs = {} if episode_horizon is None else {"episode_horizon": int(episode_horizon)}
    ascii_map = _base_ascii(env_id)
    if goal_tile is not None:
        ascii_map = _ascii_with_goal(ascii_map, goal_tile)
    env = RayMazeEnv.from_ascii(ascii_map, **horizon_kwargs)
    _ENV_BUILD_CACHE[key] = env
    return env
