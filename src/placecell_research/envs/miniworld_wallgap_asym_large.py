"""Large asymmetric WallGap environment registered as MiniWorld-WallGapAsymLarge-v0."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable

import numpy as np
from gymnasium import spaces, utils

try:
    from miniworld.envs.miniworld_env import MiniWorldEnv
except ModuleNotFoundError:
    from miniworld.miniworld import MiniWorldEnv  # type: ignore

from miniworld.entity import Box, MeshEnt

_MESH_FLOOR_OFFSET_FRACTIONS = {
    "duckie": -0.07,
    "medkit": -0.545,
}
_DEFAULT_GOAL_POSITION_XY = np.asarray([18.0, -22.0], dtype=np.float32)
_GOAL_BOX_SIZE = 1.2
_GOAL_RADIUS = math.sqrt(_GOAL_BOX_SIZE * _GOAL_BOX_SIZE + _GOAL_BOX_SIZE * _GOAL_BOX_SIZE) / 2.0
_ALLOWED_REGION_BOUNDS_BY_NAME = {
    "courtyard_north": (-13.5, 13.5, 18.0, 33.0),
    "courtyard_south": (-13.5, 13.5, 9.0, 15.0),
    "corridor_indoor": (-3.0, 3.0, -3.0, 3.0),
    "woodplank_yard": (-21.0, -9.0, -27.0, 1.5),
    "concrete_yard": (9.0, 21.0, -27.0, 1.5),
}
_FULL_ROOM_BOUNDS_BY_NAME = {
    "northern_courtyard": (-18.0, 18.0, 6.0, 36.0),
    "central_corridor": (-6.0, 6.0, -6.0, 6.0),
    "southern_yard_left": (-24.0, -6.0, -30.0, 4.5),
    "southern_yard_right": (6.0, 24.0, -30.0, 4.5),
}


def _mesh_floor_y(mesh_name: str, height: float) -> float:
    return _MESH_FLOOR_OFFSET_FRACTIONS.get(mesh_name, 0.0) * float(height)


def _mesh_position(
    mesh_name: str, height: float, x_position: float, z_position: float
) -> np.ndarray:
    return np.array([x_position, _mesh_floor_y(mesh_name, height), z_position], dtype=float)


def _sample_spawn_bounds_by_area(
    bounds_by_region: dict[str, tuple[float, float, float, float]],
    valid_regions: list[str],
    random_uniform,
) -> tuple[float, float, float, float]:
    weighted_regions: list[tuple[str, float]] = []
    total_area = 0.0
    for region_name in valid_regions:
        min_x, max_x, min_z, max_z = bounds_by_region[region_name]
        area = max(0.0, float(max_x - min_x)) * max(0.0, float(max_z - min_z))
        if area <= 0.0:
            continue
        weighted_regions.append((region_name, area))
        total_area += area
    if not weighted_regions:
        return bounds_by_region[valid_regions[0]]
    threshold = float(random_uniform(0.0, total_area))
    cumulative_area = 0.0
    for region_name, area in weighted_regions:
        cumulative_area += area
        if threshold <= cumulative_area:
            return bounds_by_region[region_name]
    return bounds_by_region[weighted_regions[-1][0]]


class WallGapAsymLarge(MiniWorldEnv, utils.EzPickle):
    """A larger landmark-dense asymmetric WallGap layout with the same 3-action interface."""

    goal_object_radius = float(_GOAL_RADIUS)

    def __init__(
        self,
        max_episode_steps: int = 700,
        randomize_agent_start: bool | None = None,
        spawn_regions: Iterable[str] | None = None,
        spawn_region_xz: tuple[float, float, float, float] | None = None,
        spawn_yaw_range_degrees: tuple[float, float] = (-180.0, 180.0),
        goal_position_xy: tuple[float, float] | None = None,
        render_goal_object: bool = True,
        reward_on_goal: bool = True,
        terminate_on_goal: bool = True,
        spawn_area_weighted: bool = True,
        place_landmark_objects: bool = True,
        forward_step: float | None = None,
        turn_step: int | None = None,
        **kwargs,
    ) -> None:
        init_kwargs = dict(kwargs)
        init_kwargs.setdefault("domain_rand", False)

        self.randomize_agent_start = (
            True if randomize_agent_start is None else bool(randomize_agent_start)
        )
        self.spawn_region_xz = tuple(spawn_region_xz) if spawn_region_xz is not None else None
        self.spawn_yaw_range_degrees = tuple(spawn_yaw_range_degrees)
        self.goal_position_xy = (
            _DEFAULT_GOAL_POSITION_XY.copy()
            if goal_position_xy is None
            else np.asarray(goal_position_xy, dtype=np.float32)
        )
        self.render_goal_object = bool(render_goal_object)
        self.reward_on_goal = bool(reward_on_goal)
        self.terminate_on_goal = bool(terminate_on_goal)
        self.spawn_area_weighted = bool(spawn_area_weighted)
        self.place_landmark_objects = bool(place_landmark_objects)
        self.goal_tolerance: float | None = None
        default_regions = ["all"]
        self.spawn_regions = (
            [region.lower() for region in spawn_regions]
            if spawn_regions is not None
            else default_regions
        )
        self.allowed_spawn_bounds_by_name = dict(_ALLOWED_REGION_BOUNDS_BY_NAME)
        self.full_room_bounds_by_name = dict(_FULL_ROOM_BOUNDS_BY_NAME)

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **init_kwargs)
        utils.EzPickle.__init__(
            self,
            max_episode_steps,
            randomize_agent_start,
            self.spawn_regions,
            self.spawn_region_xz,
            self.spawn_yaw_range_degrees,
            self.goal_position_xy,
            self.render_goal_object,
            self.reward_on_goal,
            self.terminate_on_goal,
            self.spawn_area_weighted,
            self.place_landmark_objects,
            forward_step,
            turn_step,
            **init_kwargs,
        )
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        if forward_step is not None:
            forward_step_value = float(forward_step)
            self.params.set(
                "forward_step", forward_step_value, forward_step_value, forward_step_value
            )
        else:
            self.params.set("forward_step", 0.35, 0.30, 0.40)
        if turn_step is not None:
            turn_step_value = float(turn_step)
            self.params.set("turn_step", turn_step_value, turn_step_value, turn_step_value)
        else:
            self.params.set("turn_step", 20.0, 16.0, 24.0)

    def set_goal_position_xy(self, goal_position_xy: np.ndarray | tuple[float, float]) -> None:
        self.goal_position_xy = np.asarray(goal_position_xy, dtype=np.float32).reshape(2)
        if hasattr(self, "box") and self.box is not None:
            self.box.pos = np.asarray(
                [self.goal_position_xy[0], 0.0, self.goal_position_xy[1]], dtype=float
            )

    def get_goal_position_xy(self) -> np.ndarray:
        if self.goal_position_xy is not None:
            return np.asarray(self.goal_position_xy, dtype=np.float32).reshape(2)
        if hasattr(self, "box") and self.box is not None:
            box_position = np.asarray(
                getattr(self.box, "pos", [0.0, 0.0, 0.0]), dtype=np.float32
            ).reshape(-1)
            if box_position.size >= 3:
                return np.asarray([box_position[0], box_position[2]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _gen_world(self) -> None:
        northern_courtyard = self.add_rect_room(
            min_x=-18.0,
            max_x=18.0,
            min_z=6.0,
            max_z=36.0,
            wall_tex="brick_wall",
            floor_tex="grass",
            no_ceiling=True,
        )
        central_corridor = self.add_rect_room(
            min_x=-6.0,
            max_x=6.0,
            min_z=-6.0,
            max_z=6.0,
            wall_tex="metal_grill",
            floor_tex="concrete_tiles",
            no_ceiling=False,
        )
        southern_yard_left = self.add_rect_room(
            min_x=-24.0,
            max_x=-6.0,
            min_z=-30.0,
            max_z=4.5,
            wall_tex="cinder_blocks",
            floor_tex="wood_planks",
            no_ceiling=True,
            wall_height=10.5,
        )
        southern_yard_right = self.add_rect_room(
            min_x=6.0,
            max_x=24.0,
            min_z=-30.0,
            max_z=4.5,
            wall_tex="stucco",
            floor_tex="concrete",
            no_ceiling=True,
        )

        self.connect_rooms(northern_courtyard, central_corridor, min_x=-3.0, max_x=3.0)
        self.connect_rooms(central_corridor, southern_yard_left, min_z=-3.0, max_z=3.0)
        self.connect_rooms(central_corridor, southern_yard_right, min_z=-3.0, max_z=3.0)

        if self.place_landmark_objects:
            courtyard_tree_positions = [
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
            ]
            courtyard_tree_heights = [3.0, 3.5, 2.8, 3.2, 3.0, 3.6, 2.9, 3.3, 3.1, 2.7, 3.4, 3.0]
            courtyard_tree_directions = [0.5, 1.2, 0.8, 2.1, 1.7, 0.3, 2.5, 1.9, 0.6, 2.8, 1.1, 2.3]
            for (x_position, z_position), height, direction in zip(
                courtyard_tree_positions,
                courtyard_tree_heights,
                courtyard_tree_directions,
                strict=False,
            ):
                self.place_entity(
                    MeshEnt(mesh_name="tree", height=float(height)),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            courtyard_cone_positions = [
                (12.0, 8.0),
                (14.0, 12.0),
                (13.0, 17.0),
                (15.0, 21.0),
                (12.0, 25.0),
                (14.0, 29.0),
                (16.0, 32.0),
                (11.0, 35.0),
            ]
            courtyard_cone_heights = [0.75, 0.9, 0.7, 0.85, 0.8, 0.95, 0.75, 0.9]
            for (x_position, z_position), height in zip(
                courtyard_cone_positions, courtyard_cone_heights, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="cone", height=float(height)),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=0.0,
                )

            courtyard_barrel_positions = [(-10.0, 32.0), (8.0, 10.0), (0.0, 28.0)]
            courtyard_barrel_directions = [0.3, 1.1, 2.2]
            for (x_position, z_position), direction in zip(
                courtyard_barrel_positions, courtyard_barrel_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="barrel", height=1.2),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            courtyard_barrier_positions = [(5.0, 16.0), (-2.0, 24.0)]
            courtyard_barrier_directions = [0.7, 2.4]
            for (x_position, z_position), direction in zip(
                courtyard_barrier_positions, courtyard_barrier_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="barrier", height=1.0),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            corridor_desk_positions = [(-4.0, 3.0), (3.0, -2.0)]
            corridor_desk_directions = [math.pi / 6, -math.pi / 5]
            for (x_position, z_position), direction in zip(
                corridor_desk_positions, corridor_desk_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="office_desk", height=1.0),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            corridor_duckie_positions = [(-2.0, -4.0), (4.0, 4.0), (-3.0, 2.0), (2.0, -3.0)]
            corridor_duckie_directions = [math.pi / 3, -math.pi / 4, 0.2, 1.4]
            for (x_position, z_position), direction in zip(
                corridor_duckie_positions, corridor_duckie_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="duckie", height=0.5),
                    pos=_mesh_position("duckie", 0.5, x_position, z_position),
                    dir=float(direction),
                )

            left_yard_pine_positions = [
                (-21.0, -8.0),
                (-18.0, -15.0),
                (-22.0, -21.0),
                (-15.0, -25.0),
                (-20.0, -28.0),
                (-17.0, 2.0),
            ]
            left_yard_pine_heights = [4.0, 4.5, 3.8, 4.2, 4.0, 4.3]
            left_yard_pine_directions = [math.pi / 4, -math.pi / 6, 0.9, 2.1, 1.3, -1.7]
            for (x_position, z_position), height, direction in zip(
                left_yard_pine_positions,
                left_yard_pine_heights,
                left_yard_pine_directions,
                strict=False,
            ):
                self.place_entity(
                    MeshEnt(mesh_name="tree_pine", height=float(height)),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            left_yard_barrel_positions = [
                (-9.0, -5.0),
                (-12.0, -12.0),
                (-8.0, -20.0),
                (-14.0, -27.0),
            ]
            left_yard_barrel_directions = [1.5, -0.4, 2.6, 0.8]
            for (x_position, z_position), direction in zip(
                left_yard_barrel_positions, left_yard_barrel_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="barrel", height=1.3),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            left_yard_chair_positions = [
                (-10.0, 0.0),
                (-19.0, -10.0),
                (-11.0, -18.0),
                (-16.0, -24.0),
            ]
            left_yard_chair_directions = [2.1, -2.4, 0.6, -0.9]
            for (x_position, z_position), direction in zip(
                left_yard_chair_positions, left_yard_chair_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="office_chair", height=1.2),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            left_yard_barrier_positions = [(-13.0, -3.0), (-20.0, -16.0), (-9.0, -26.0)]
            left_yard_barrier_directions = [0.9, -1.6, 2.5]
            for (x_position, z_position), direction in zip(
                left_yard_barrier_positions, left_yard_barrier_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="barrier", height=1.1),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            right_yard_barrel_positions = [
                (9.0, -6.0),
                (15.0, -11.0),
                (12.0, -18.0),
                (18.0, -22.0),
                (10.0, -27.0),
            ]
            right_yard_barrel_directions = [0.4, -1.1, 1.9, 2.7, -2.2]
            for (x_position, z_position), direction in zip(
                right_yard_barrel_positions, right_yard_barrel_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="barrel", height=1.2),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            right_yard_barrier_positions = [(8.0, -3.0), (20.0, -14.0), (14.0, -25.0)]
            right_yard_barrier_directions = [1.8, -0.5, 2.2]
            for (x_position, z_position), direction in zip(
                right_yard_barrier_positions, right_yard_barrier_directions, strict=False
            ):
                self.place_entity(
                    MeshEnt(mesh_name="barrier", height=1.0),
                    pos=np.array([x_position, 0.0, z_position], dtype=float),
                    dir=float(direction),
                )

            self.place_entity(
                MeshEnt(mesh_name="medkit", height=0.6),
                pos=_mesh_position("medkit", 0.6, 21.0, -10.0),
                dir=0.0,
            )
        self.place_entity(
            MeshEnt(mesh_name="building", height=40.0),
            pos=np.array([60.0, 0.0, 54.0], dtype=float),
            dir=-math.pi / 2,
        )

        if self.render_goal_object:
            self.box = self.place_entity(
                Box(color="red", size=_GOAL_BOX_SIZE),
                pos=np.asarray(
                    [self.goal_position_xy[0], 0.0, self.goal_position_xy[1]], dtype=float
                ),
                dir=0.0,
            )
        else:
            self.box = None

        if self.spawn_region_xz is not None:
            min_x, max_x, min_z, max_z = self.spawn_region_xz
        else:
            selected_bounds: list[tuple[float, float, float, float]] = []
            for region_name in self.spawn_regions:
                if region_name == "all":
                    selected_bounds.extend(self.full_room_bounds_by_name.values())
                elif region_name in self.allowed_spawn_bounds_by_name:
                    selected_bounds.append(self.allowed_spawn_bounds_by_name[region_name])
                elif region_name in self.full_room_bounds_by_name:
                    selected_bounds.append(self.full_room_bounds_by_name[region_name])
            if not selected_bounds:
                selected_bounds = list(self.full_room_bounds_by_name.values())
            if not selected_bounds:
                raise ValueError(
                    "Unsupported spawn regions. Allowed subsets: "
                    f"{sorted(self.allowed_spawn_bounds_by_name)}, "
                    f"full rooms: {sorted(self.full_room_bounds_by_name)}, and 'all'."
                )
            if self.randomize_agent_start:
                if self.spawn_area_weighted:
                    bounds_by_name = {
                        f"candidate_{index}": bounds for index, bounds in enumerate(selected_bounds)
                    }
                    min_x, max_x, min_z, max_z = _sample_spawn_bounds_by_area(
                        bounds_by_name,
                        list(bounds_by_name.keys()),
                        self.np_random.uniform,
                    )
                else:
                    region_index = int(self.np_random.integers(0, len(selected_bounds)))
                    min_x, max_x, min_z, max_z = selected_bounds[region_index]
            else:
                min_x, max_x, min_z, max_z = selected_bounds[0]

        if self.randomize_agent_start:
            self._place_agent_continuous(min_x, max_x, min_z, max_z)
            return

        center_x = 0.5 * (min_x + max_x)
        center_z = 0.5 * (min_z + max_z)
        yaw_min, yaw_max = self.spawn_yaw_range_degrees
        fixed_yaw = math.radians(0.5 * (yaw_min + yaw_max))
        self.place_agent(pos=np.array([center_x, 0.0, center_z], dtype=float), dir=fixed_yaw)

    def _place_agent_continuous(
        self, min_x: float, max_x: float, min_z: float, max_z: float
    ) -> None:
        yaw_low_degrees, yaw_high_degrees = self.spawn_yaw_range_degrees
        candidate_position = np.array(
            [0.5 * (min_x + max_x), 0.0, 0.5 * (min_z + max_z)], dtype=float
        )
        found_valid_position = False
        for _ in range(200):
            candidate_position = np.array(
                [
                    float(self.np_random.uniform(min_x, max_x)),
                    0.0,
                    float(self.np_random.uniform(min_z, max_z)),
                ],
                dtype=float,
            )
            if not self.intersect(self.agent, candidate_position, self.agent.radius):
                found_valid_position = True
                break
        if not found_valid_position:
            raise RuntimeError(
                "WallGapAsymLarge could not find a collision-free random spawn position within the "
                "retry budget. "
                "Adjust spawn regions, reduce clutter, or increase the valid free-space support."
            )
        random_yaw = float(
            self.np_random.uniform(math.radians(yaw_low_degrees), math.radians(yaw_high_degrees))
        )
        self.place_agent(pos=candidate_position, dir=random_yaw)

    def _goal_reached(self) -> bool:
        goal_xy = np.asarray(self.get_goal_position_xy(), dtype=np.float32).reshape(2)
        agent_position = np.asarray(
            getattr(self.agent, "pos", np.zeros(3, dtype=float)), dtype=np.float32
        ).reshape(-1)
        if agent_position.size < 3:
            return False
        agent_xy = np.asarray([agent_position[0], agent_position[2]], dtype=np.float32)
        distance = float(np.linalg.norm(goal_xy - agent_xy))
        goal_tolerance = getattr(self, "goal_tolerance", None)
        effective_radius = _GOAL_RADIUS if goal_tolerance is None else float(goal_tolerance)
        return distance < effective_radius + float(self.agent.radius) + 1.1 * float(
            self.max_forward_step
        )

    def step(self, action):
        base_step = getattr(super(), "step", None)
        if base_step is not None:
            observation, reward, terminated, truncated, info = base_step(action)
        else:
            miniworld_env_module = sys.modules.get("miniworld.envs.miniworld_env")
            if miniworld_env_module is None:
                miniworld_env_module = sys.modules.get("miniworld.miniworld")
            miniworld_env_class = (
                None
                if miniworld_env_module is None
                else getattr(
                    miniworld_env_module,
                    "MiniWorldEnv",
                    None,
                )
            )
            fallback_step = (
                None if miniworld_env_class is None else getattr(miniworld_env_class, "step", None)
            )
            if fallback_step is None:
                raise AttributeError("MiniWorldEnv.step is unavailable for WallGapAsymLarge.step.")
            observation, reward, terminated, truncated, info = fallback_step(self, action)
        if self._goal_reached():
            if self.reward_on_goal:
                reward += self._reward()
            if self.terminate_on_goal:
                terminated = True
        info = dict(info)
        info["goal_position_xy"] = [float(value) for value in self.get_goal_position_xy()]
        return observation, reward, terminated, truncated, info


def _register_env() -> None:
    from gymnasium.envs.registration import register
    from gymnasium.error import Error

    try:
        register(
            id="MiniWorld-WallGapAsymLarge-v0",
            entry_point="placecell_research.envs.miniworld_wallgap_asym_large:WallGapAsymLarge",
            max_episode_steps=700,
        )
    except Error as exc:
        if "Cannot re-register" not in str(exc):
            raise


_register_env()
