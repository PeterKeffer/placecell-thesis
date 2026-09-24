"""Raw dataset collection."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from placecell_research.collection.parallel_runtime import (
    iter_parallel_episode_batches,
    resolve_collection_worker_count,
)
from placecell_research.collection.policies import ContinuousRandomPolicy, RandomDiscretePolicy
from placecell_research.collection.previews import (
    write_kinematics_histogram,
    write_position_density_map,
    write_sample_frames,
    write_trajectory_gif,
)
from placecell_research.collection.safety import render_topdown_with_timeout
from placecell_research.datasets.schema import (
    ACTIONS_KEY,
    CONTINUOUS_ACTIONS_KEY,
    HEADING_KEY,
    KINEMATICS_KEY,
    LENGTH_KEY,
    POSITION_KEY,
    RGB_KEY,
    SOURCE_SEED_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    VALID_MASK_KEY,
    DatasetSummary,
    observation_array_key,
)
from placecell_research.datasets.zarr_io import DatasetZarrStreamWriter
from placecell_research.envs import assert_environment_supports
from placecell_research.envs.base import ContinuousMotion
from placecell_research.envs.builder import build_environment
from placecell_research.tracking.progress import ProgressUpdate

if TYPE_CHECKING:
    from placecell_research.config.schema import CollectionConfig, EnvironmentConfig


@dataclass
class CollectionResult:
    """Collection manifest summary plus preview location."""

    arrays: dict[str, np.ndarray]
    summary: DatasetSummary
    preview_dir: Path


CollectionProgressCallback = Callable[[ProgressUpdate], None]
_PASSIVE_COLLECTION_GOAL_TASK_ENV_IDS = {"MiniWorld-WallGapAsymLarge-v0"}
_DEFAULT_COLLECTION_GOAL_POSITION_BY_ENV_ID = {
    "MiniWorld-WallGapAsymLarge-v0": np.asarray([18.0, -22.0], dtype=np.float32),
}


def _should_skip_env_close_on_slurm(collection_config: CollectionConfig) -> bool:
    return bool(os.environ.get("SLURM_JOB_ID")) and bool(
        collection_config.safety.skip_env_close_on_slurm
    )


def _close_collection_adapter(adapter: Any, collection_config: CollectionConfig) -> None:
    if _should_skip_env_close_on_slurm(collection_config):
        return
    adapter.close()


@dataclass
class _PreviewAccumulator:
    sample_frame_limit: int = 16
    gif_frame_limit: int = 64
    sample_frames: list[np.ndarray] = field(default_factory=list)
    gif_frames: list[np.ndarray] = field(default_factory=list)
    gif_headings: list[float] = field(default_factory=list)
    position_chunks: list[np.ndarray] = field(default_factory=list)
    kinematics_chunks: list[np.ndarray] = field(default_factory=list)
    total_valid_frames: int = 0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def add_episode(self, episode_arrays: dict[str, np.ndarray]) -> None:
        valid_steps = np.asarray(episode_arrays[VALID_MASK_KEY], dtype=bool)
        valid_count = int(valid_steps.sum())
        if valid_count <= 0:
            return
        valid_heading = np.asarray(episode_arrays[HEADING_KEY], dtype=np.float32)[valid_steps]
        valid_positions = np.asarray(episode_arrays[POSITION_KEY])[valid_steps]
        valid_kinematics = np.asarray(episode_arrays[KINEMATICS_KEY])[valid_steps]
        self.position_chunks.append(valid_positions.astype(np.float32, copy=False))
        self.kinematics_chunks.append(valid_kinematics.astype(np.float32, copy=False))
        if RGB_KEY not in episode_arrays:
            return
        valid_rgb = np.asarray(episode_arrays[RGB_KEY])[valid_steps]
        for episode_frame_index, frame in enumerate(valid_rgb):
            frame_uint8 = np.asarray(frame, dtype=np.uint8)
            self.total_valid_frames += 1
            if len(self.gif_frames) < self.gif_frame_limit:
                self.gif_frames.append(frame_uint8.copy())
                self.gif_headings.append(float(valid_heading[episode_frame_index]))
            if len(self.sample_frames) < self.sample_frame_limit:
                self.sample_frames.append(frame_uint8.copy())
                continue
            reservoir_index = int(self.rng.integers(0, self.total_valid_frames))
            if reservoir_index < self.sample_frame_limit:
                self.sample_frames[reservoir_index] = frame_uint8.copy()

    def write(
        self,
        preview_dir: Path,
        *,
        topdown_frame: np.ndarray | None,
        goal_positions_xy: np.ndarray | None,
    ) -> None:
        positions = np.concatenate(self.position_chunks, axis=0)
        kinematics = np.concatenate(self.kinematics_chunks, axis=0)
        write_position_density_map(
            positions,
            preview_dir / "topdown_map.png",
            topdown_frame=topdown_frame,
            goal_positions_xy=goal_positions_xy,
        )
        write_kinematics_histogram(kinematics, preview_dir / "kinematics_histogram.png")
        if not self.sample_frames or not self.gif_frames:
            return
        write_sample_frames(np.stack(self.sample_frames, axis=0), preview_dir / "sample_frames.png")
        write_trajectory_gif(
            np.stack(self.gif_frames, axis=0),
            preview_dir / "trajectory_samples.gif",
            headings=np.asarray(self.gif_headings, dtype=np.float32),
        )


def _environment_config_for_collection(
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
) -> EnvironmentConfig:
    is_goal_task_env = str(environment_config.env_id) in _PASSIVE_COLLECTION_GOAL_TASK_ENV_IDS
    if not collection_config.spawn_regions and not is_goal_task_env:
        return environment_config
    env_kwargs = dict(environment_config.env_kwargs)
    if collection_config.spawn_regions:
        env_kwargs["spawn_regions"] = [str(region) for region in collection_config.spawn_regions]
    if is_goal_task_env:
        env_kwargs["render_goal_object"] = False
        env_kwargs["reward_on_goal"] = False
        env_kwargs["terminate_on_goal"] = False
    return replace(environment_config, env_kwargs=env_kwargs)


def _compute_kinematics(position_xy: np.ndarray, heading: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(position_xy)
    deltas[1:] = position_xy[1:] - position_xy[:-1]
    step_displacement = np.linalg.norm(deltas, axis=-1)
    heading_delta = np.zeros_like(heading)
    heading_delta[1:] = heading[1:] - heading[:-1]
    heading_delta = (heading_delta + np.pi) % (2 * np.pi) - np.pi
    return np.stack(
        [
            step_displacement.astype(np.float32),
            heading_delta.astype(np.float32),
            np.sin(heading).astype(np.float32),
            np.cos(heading).astype(np.float32),
        ],
        axis=-1,
    )


def _resolve_action_names(adapter: Any) -> list[str] | None:
    action_space = getattr(adapter, "action_space", None)
    if action_space is not None:
        names = getattr(action_space, "names", ())
        if names:
            return [str(action_name) for action_name in names]
    action_names = getattr(adapter, "action_names", None)
    if action_names is None:
        return None
    if callable(action_names):
        resolved = action_names()
    else:
        resolved = action_names
    return [str(action_name) for action_name in resolved]


def _collect_one_episode(
    adapter,
    collection_config: CollectionConfig,
    episode_seed: int,
    episode_index: int,
) -> dict[str, np.ndarray | bool | int]:
    sampler: Any
    if collection_config.policy == "continuous_random":
        sampler = ContinuousRandomPolicy(collection_config.continuous_motion)
    elif collection_config.policy == "motion_bouts":
        from .policies import MotionBoutPolicy

        sampler = MotionBoutPolicy(
            _resolve_action_names(adapter), collection_config.bout_switch_probability
        )
    else:
        sampler = RandomDiscretePolicy(
            adapter.action_space.count,
            seed=episode_seed,
            mode=collection_config.policy,
            action_names=_resolve_action_names(adapter),
            action_probabilities=collection_config.action_probabilities,
        )
    sampler.reset(adapter, episode_seed, episode_index)
    max_steps = int(collection_config.episode_length)
    first_observation = adapter.reset(seed=episode_seed)
    saved_modality_names = tuple(
        name
        for name in sorted(first_observation.modalities)
        if name != "rgb" or bool(collection_config.save_rgb)
    )
    positions = np.zeros((max_steps, 2), dtype=np.float32)
    heading = np.zeros((max_steps,), dtype=np.float32)
    modality_arrays = {
        name: np.zeros(
            (max_steps, *first_observation.require_modality(name).shape),
            dtype=first_observation.require_modality(name).dtype,
        )
        for name in saved_modality_names
    }
    actions = np.full((max_steps,), adapter.action_space.count, dtype=np.int64)
    continuous_actions = (
        np.zeros((max_steps, 2), dtype=np.float32)
        if collection_config.policy == "continuous_random" else None
    )
    valid_steps = np.zeros((max_steps,), dtype=bool)
    terminated = False
    truncated = False

    current_observation = first_observation
    steps_taken = 0
    for step_index in range(max_steps):
        for modality_name, modality_array in modality_arrays.items():
            modality_array[step_index] = current_observation.require_modality(modality_name)
        positions[step_index] = current_observation.position_xy
        heading[step_index] = current_observation.heading
        valid_steps[step_index] = True
        action = sampler.sample(current_observation)
        steps_taken = step_index + 1
        if isinstance(action, ContinuousMotion):
            continuous_actions[step_index] = [action.forward_distance, action.turn_radians]
            result = adapter.step_motion(action)
        else:
            actions[step_index] = action
            result = adapter.step(action)
        current_observation = result.observation
        terminated = result.terminated
        truncated = result.truncated
        if terminated or truncated:
            break

    kinematics = _compute_kinematics(positions[:steps_taken], heading[:steps_taken])
    padded_kinematics = np.zeros((max_steps, kinematics.shape[-1]), dtype=np.float32)
    padded_kinematics[:steps_taken] = kinematics
    episode_payload: dict[str, np.ndarray | bool | int] = {
        "actions": actions,
        "position_xy": positions,
        "heading": heading,
        "kinematics": padded_kinematics,
        "valid_steps": valid_steps,
        "length": steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "source_seed": episode_seed,
    }
    if continuous_actions is not None:
        episode_payload[CONTINUOUS_ACTIONS_KEY] = continuous_actions
    for modality_name, modality_array in modality_arrays.items():
        episode_payload[observation_array_key(modality_name)] = modality_array
    return episode_payload


def _collect_episode_batch(
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
    episode_seeds: Iterable[int],
    startup_delay_seconds: float = 0.0,
    on_episode_complete: Callable[[int], None] | None = None,
) -> tuple[list[dict[str, np.ndarray | bool | int]], int]:
    seed_list = [int(seed) for seed in episode_seeds]
    if not seed_list:
        raise ValueError("Episode batch must contain at least one seed.")
    if startup_delay_seconds > 0.0:
        time.sleep(float(startup_delay_seconds))

    collection_environment_config = _environment_config_for_collection(
        environment_config,
        collection_config,
    )
    adapter = build_environment(collection_environment_config, seed=seed_list[0])
    try:
        assert_environment_supports(
            adapter,
            consumer="dataset collection",
            required=("position_xy", "heading", "discrete_actions"),
            required_modalities=("rgb",) if collection_config.save_rgb else (),
        )
        episodes: list[dict[str, np.ndarray | bool | int]] = []
        for episode_seed in seed_list:
            episodes.append(
                _collect_one_episode(
                    adapter, collection_config, episode_seed, episode_index=episode_seed
                )
            )
            if on_episode_complete is not None:
                on_episode_complete(1)
        return episodes, int(adapter.action_space.count)
    finally:
        _close_collection_adapter(adapter, collection_config)


def _episode_arrays_for_storage(
    episode: dict[str, np.ndarray | bool | int],
) -> dict[str, np.ndarray]:
    arrays = {
        ACTIONS_KEY: np.asarray(episode["actions"], dtype=np.int64),
        POSITION_KEY: np.asarray(episode["position_xy"], dtype=np.float32),
        HEADING_KEY: np.asarray(episode["heading"], dtype=np.float32),
        KINEMATICS_KEY: np.asarray(episode["kinematics"], dtype=np.float32),
        VALID_MASK_KEY: np.asarray(episode["valid_steps"], dtype=bool),
        LENGTH_KEY: np.asarray(episode["length"], dtype=np.int32),
        TERMINATED_KEY: np.asarray(episode["terminated"], dtype=bool),
        TRUNCATED_KEY: np.asarray(episode["truncated"], dtype=bool),
        SOURCE_SEED_KEY: np.asarray(episode["source_seed"], dtype=np.int64),
    }
    for key, value in episode.items():
        if isinstance(key, str) and (
            key.startswith("observations/") or key == CONTINUOUS_ACTIONS_KEY
        ):
            arrays[key] = np.asarray(value)
    return arrays


def _observation_modalities_from_episode(episode: dict[str, np.ndarray | bool | int]) -> list[str]:
    modalities: list[str] = []
    for key in episode:
        if isinstance(key, str) and key.startswith("observations/"):
            modalities.append(key.split("/", 1)[1])
    return sorted(modalities)


def _resolve_collection_goal_positions_xy(
    environment_config: EnvironmentConfig,
) -> np.ndarray | None:
    env_id = str(environment_config.env_id)
    if env_id not in _PASSIVE_COLLECTION_GOAL_TASK_ENV_IDS:
        return None
    configured_goal_position = environment_config.env_kwargs.get("goal_position_xy")
    if configured_goal_position is not None:
        return np.asarray(configured_goal_position, dtype=np.float32).reshape(1, 2)
    default_goal_position = _DEFAULT_COLLECTION_GOAL_POSITION_BY_ENV_ID.get(env_id)
    if default_goal_position is None:
        return None
    return np.asarray(default_goal_position, dtype=np.float32).reshape(1, 2)


@dataclass(slots=True)
class _DatasetWriteSession:
    output_path: Path
    env_id: str
    total_episodes: int
    episode_length: int
    _writer: DatasetZarrStreamWriter | None = None
    _summary: DatasetSummary | None = None
    _num_actions: int | None = None
    _episodes_written: int = 0

    def append_episode(
        self,
        episode_arrays: dict[str, np.ndarray],
        *,
        num_actions: int,
    ) -> None:
        resolved_num_actions = int(num_actions)
        if self._num_actions is None:
            self._num_actions = resolved_num_actions
        elif self._num_actions != resolved_num_actions:
            raise ValueError(
                "Inconsistent action-space sizes across collection writes: "
                f"{self._num_actions} vs {resolved_num_actions}."
            )
        if self._writer is None:
            self._summary = DatasetSummary(
                env_id=self.env_id,
                num_episodes=int(self.total_episodes),
                episode_length=int(self.episode_length),
                num_actions=resolved_num_actions,
                modalities=_observation_modalities_from_episode(episode_arrays),
            )
            self._writer = DatasetZarrStreamWriter(
                self.output_path,
                _streaming_array_specs(episode_arrays, total_episodes=self.total_episodes),
                self._summary,
            ).open()
        self._writer.write_episode(self._episodes_written, episode_arrays)
        self._episodes_written += 1

    def finalize(self) -> DatasetSummary:
        if self._writer is None or self._summary is None:
            raise ValueError("Dataset write session did not receive any episodes.")
        if self._episodes_written != int(self.total_episodes):
            raise RuntimeError(
                "Dataset write session received "
                f"{self._episodes_written} episodes for {self.total_episodes} expected episodes."
            )
        self._writer.finalize()
        return self._summary

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _streaming_array_specs(
    first_episode_arrays: dict[str, np.ndarray],
    *,
    total_episodes: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    return {
        key: ((int(total_episodes), *tuple(int(size) for size in value.shape)), value.dtype)
        for key, value in first_episode_arrays.items()
    }


def _append_episode_to_outputs(
    write_session: _DatasetWriteSession,
    preview_accumulator: _PreviewAccumulator,
    episode: dict[str, np.ndarray | bool | int],
    *,
    num_actions: int,
) -> None:
    episode_arrays = _episode_arrays_for_storage(episode)
    write_session.append_episode(episode_arrays, num_actions=num_actions)
    preview_accumulator.add_episode(episode_arrays)


def _collect_raw_dataset_vectorized(
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
    output_dir: Path,
    collection_seed: int,
    progress_callback: CollectionProgressCallback | None = None,
) -> CollectionResult:
    """Vectorized JAXenstein collection: vmap+scan GPU rollout written to the same Zarr schema."""
    from placecell_research.collection.jax_policies import JaxOUPolicy
    from placecell_research.collection.jax_rollout import make_jaxenstein_rollout
    from placecell_research.envs.jaxenstein_adapter import JAXENSTEIN_NAV_ACTION_NAMES

    terminate_on_goal = bool(environment_config.env_kwargs.get("terminate_on_goal", True))
    uniform_spawn = bool(environment_config.env_kwargs.get("uniform_spawn", False))
    horizon = int(collection_config.episode_length)
    total_episodes = int(collection_config.episodes)
    batch_size = int(collection_config.num_parallel_envs) or total_episodes
    num_actions = len(JAXENSTEIN_NAV_ACTION_NAMES)
    rollout = make_jaxenstein_rollout(
        env_id=str(environment_config.env_id), policy=JaxOUPolicy(),
        terminate_on_goal=terminate_on_goal,
        randomize_agent_start=environment_config.randomize_agent_start,
        uniform_spawn=uniform_spawn,
    )

    started_at = time.perf_counter()
    completed_episodes = 0

    def _emit_progress(detail: str) -> None:
        if progress_callback is None:
            return
        progress_callback(
            ProgressUpdate(
                completed=completed_episodes,
                total=total_episodes,
                elapsed_seconds=time.perf_counter() - started_at,
                unit_name="episodes",
                detail=detail,
            )
        )

    write_session = _DatasetWriteSession(
        output_path=output_dir / "dataset.zarr",
        env_id=str(environment_config.env_id),
        total_episodes=total_episodes,
        episode_length=horizon,
    )
    preview_accumulator = _PreviewAccumulator()
    _emit_progress("collect episodes")
    try:
        for batch_start in range(0, total_episodes, batch_size):
            batch_envs = min(batch_size, total_episodes - batch_start)
            episodes = rollout(
                num_envs=batch_envs,
                horizon=horizon,
                base_seed=collection_seed + batch_start,
            )
            for episode in episodes:
                _append_episode_to_outputs(
                    write_session,
                    preview_accumulator,
                    episode,
                    num_actions=num_actions,
                )
                completed_episodes += 1
            _emit_progress("collect episodes")
        _emit_progress("write dataset.zarr")
        summary = write_session.finalize()
    finally:
        write_session.close()
    preview_dir = output_dir / "previews"
    _emit_progress("write previews")
    preview_accumulator.write(preview_dir, topdown_frame=None, goal_positions_xy=None)
    _emit_progress("dataset collection complete")
    return CollectionResult(arrays={}, summary=summary, preview_dir=preview_dir)


def collect_raw_dataset(
    environment_config: EnvironmentConfig,
    collection_config: CollectionConfig,
    output_dir: Path,
    collection_seed: int,
    progress_callback: CollectionProgressCallback | None = None,
) -> CollectionResult:
    """Collect a raw dataset with mandatory state and optional observation modalities."""
    use_vectorized_jaxenstein = (
        str(environment_config.kind).strip().lower() == "jaxenstein"
        and collection_config.vectorized
    )
    if use_vectorized_jaxenstein:
        return _collect_raw_dataset_vectorized(
            environment_config,
            collection_config,
            output_dir,
            collection_seed,
            progress_callback,
        )
    collection_environment_config = _environment_config_for_collection(
        environment_config,
        collection_config,
    )
    seeds = [collection_seed + episode_index for episode_index in range(collection_config.episodes)]
    started_at = time.perf_counter()
    completed_episodes = 0

    def _emit_progress(*, detail: str | None = None) -> None:
        if progress_callback is None:
            return
        progress_callback(
            ProgressUpdate(
                completed=completed_episodes,
                total=len(seeds),
                elapsed_seconds=time.perf_counter() - started_at,
                unit_name="episodes",
                detail=detail,
            )
        )

    def _record_episode_completion(batch_size: int) -> None:
        nonlocal completed_episodes
        completed_episodes = min(len(seeds), completed_episodes + int(batch_size))
        _emit_progress(detail="collect episodes")

    _emit_progress(detail="collect episodes")
    topdown_frame = None
    if bool(collection_config.save_topdown):
        topdown_frame = render_topdown_with_timeout(
            collection_environment_config,
            collection_seed,
            collection_config.safety,
        )

    if not seeds:
        raise ValueError("Raw dataset collection requires at least one episode.")
    write_session = _DatasetWriteSession(
        output_path=output_dir / "dataset.zarr",
        env_id=collection_environment_config.env_id,
        total_episodes=len(seeds),
        episode_length=int(collection_config.episode_length),
    )
    preview_accumulator = _PreviewAccumulator()

    worker_count = resolve_collection_worker_count(collection_config)
    try:
        if worker_count > 1 or collection_config.safety.isolate_cuda_miniworld:
            _emit_progress(detail=f"collect episodes ({worker_count} shard workers)")
            pending_batches = {}
            next_shard_index = 0
            for batch in iter_parallel_episode_batches(
                collection_environment_config,
                collection_config,
                seeds,
                batch_collector=_collect_episode_batch,
                on_episode_complete=_record_episode_completion,
            ):
                pending_batches[batch.shard_index] = batch
                while next_shard_index in pending_batches:
                    ordered_batch = pending_batches.pop(next_shard_index)
                    for episode in ordered_batch.episodes:
                        _append_episode_to_outputs(
                            write_session,
                            preview_accumulator,
                            episode,
                            num_actions=ordered_batch.num_actions,
                        )
                    next_shard_index += 1
            if pending_batches:
                raise RuntimeError(
                    "Parallel dataset collection finished with unresolved shard ordering state."
                )
        else:
            adapter = build_environment(collection_environment_config, seed=seeds[0])
            try:
                assert_environment_supports(
                    adapter,
                    consumer="dataset collection",
                    required=("position_xy", "heading", "discrete_actions"),
                    required_modalities=("rgb",) if collection_config.save_rgb else (),
                )
                num_actions = int(adapter.action_space.count)
                for episode_seed in seeds:
                    episode = _collect_one_episode(
                        adapter, collection_config, episode_seed, episode_index=episode_seed
                    )
                    _append_episode_to_outputs(
                        write_session,
                        preview_accumulator,
                        episode,
                        num_actions=num_actions,
                    )
                    _record_episode_completion(1)
            finally:
                _close_collection_adapter(adapter, collection_config)
        _emit_progress(detail="write dataset.zarr")
        summary = write_session.finalize()
    finally:
        write_session.close()
    preview_dir = output_dir / "previews"
    _emit_progress(detail="write previews")
    preview_accumulator.write(
        preview_dir,
        topdown_frame=topdown_frame,
        goal_positions_xy=_resolve_collection_goal_positions_xy(collection_environment_config),
    )
    _emit_progress(detail="dataset collection complete")
    return CollectionResult(arrays={}, summary=summary, preview_dir=preview_dir)
