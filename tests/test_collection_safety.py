from __future__ import annotations

import multiprocessing as mp
import pickle
import signal
from pathlib import Path
from queue import Empty

import numpy as np
import pytest

from placecell_research.collection import collector, parallel_runtime
from placecell_research.collection import safety as collection_safety
from placecell_research.collection.safety import _render_topdown_worker, configure_start_method
from placecell_research.config.schema import (
    CollectionConfig,
    CollectionSafetyConfig,
    EnvironmentConfig,
)
from placecell_research.datasets.schema import (
    HEADING_KEY,
    KINEMATICS_KEY,
    POSITION_KEY,
    RGB_KEY,
    SOURCE_SEED_KEY,
    VALID_MASK_KEY,
)
from placecell_research.envs import DiscreteActionSpace, EnvironmentCapabilities


def test_spawn_context_is_enforced_for_collection() -> None:
    safety = CollectionSafetyConfig(multiprocessing_start_method="spawn")
    context = configure_start_method(safety)
    assert isinstance(context, mp.context.SpawnContext)


def test_collection_config_has_no_unused_teleport_threshold() -> None:
    assert not hasattr(CollectionConfig(), "teleport_threshold")


def test_topdown_worker_and_arguments_are_spawn_picklable() -> None:
    assert "<locals>" not in _render_topdown_worker.__qualname__
    pickle.dumps(_render_topdown_worker)
    pickle.dumps(EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"))
    pickle.dumps(CollectionSafetyConfig())


def test_topdown_worker_skips_env_close_on_slurm_when_configured(monkeypatch) -> None:
    close_calls: list[int] = []
    sent_payloads: list[object] = []
    connection_closed: list[bool] = []

    class FakeAdapter:
        def render_topdown(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:
            close_calls.append(1)

    class FakeConnection:
        def send(self, payload) -> None:
            sent_payloads.append(payload)

        def close(self) -> None:
            connection_closed.append(True)

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        "placecell_research.envs.builder.build_environment",
        lambda environment_config, seed: FakeAdapter(),
    )
    safety = CollectionSafetyConfig(skip_env_close_on_slurm=True)

    _render_topdown_worker(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"),
        11,
        safety,
        FakeConnection(),
    )

    assert len(sent_payloads) == 1
    assert close_calls == []
    assert connection_closed == [True]


def test_collect_episode_batch_reuses_one_environment_for_multiple_episodes(monkeypatch) -> None:
    build_calls: list[int] = []
    close_calls: list[int] = []

    class FakeObservation:
        def __init__(self) -> None:
            self.position_xy = np.zeros(2, dtype=np.float32)
            self.heading = 0.0
            self.info = {}
            self.modalities = {"rgb": np.zeros((3, 8, 8), dtype=np.uint8)}

        @property
        def rgb(self):
            return self.modalities["rgb"]

        def require_modality(self, name: str):
            return self.modalities[name]

    class FakeStepResult:
        def __init__(self) -> None:
            self.observation = FakeObservation()
            self.terminated = True
            self.truncated = False

    class FakeAdapter:
        action_space = DiscreteActionSpace(count=3)
        capabilities = EnvironmentCapabilities(observation_modalities=("rgb",))

        @property
        def num_actions(self) -> int:
            return self.action_space.count

        def reset(self, seed: int | None = None):
            del seed
            return FakeObservation()

        def step(self, action: int):
            del action
            return FakeStepResult()

        def close(self) -> None:
            close_calls.append(1)

    def fake_build_environment(environment_config: EnvironmentConfig, seed: int):
        del environment_config, seed
        build_calls.append(1)
        return FakeAdapter()

    monkeypatch.setattr(collector, "build_environment", fake_build_environment)

    episodes, num_actions = collector._collect_episode_batch(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"),
        CollectionConfig(episodes=3, episode_length=4),
        [11, 12, 13],
    )

    assert len(episodes) == 3
    assert num_actions == 3
    assert len(build_calls) == 1
    assert len(close_calls) == 1


def test_collect_episode_batch_skips_env_close_on_slurm_when_configured(monkeypatch) -> None:
    close_calls: list[int] = []

    class FakeObservation:
        def __init__(self) -> None:
            self.position_xy = np.zeros(2, dtype=np.float32)
            self.heading = 0.0
            self.info = {}
            self.modalities = {"rgb": np.zeros((3, 8, 8), dtype=np.uint8)}

        @property
        def rgb(self):
            return self.modalities["rgb"]

        def require_modality(self, name: str):
            return self.modalities[name]

    class FakeStepResult:
        def __init__(self) -> None:
            self.observation = FakeObservation()
            self.terminated = True
            self.truncated = False

    class FakeAdapter:
        action_space = DiscreteActionSpace(count=3)
        capabilities = EnvironmentCapabilities(observation_modalities=("rgb",))

        def reset(self, seed: int | None = None):
            del seed
            return FakeObservation()

        def step(self, action: int):
            del action
            return FakeStepResult()

        def close(self) -> None:
            close_calls.append(1)

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        collector, "build_environment", lambda environment_config, seed: FakeAdapter()
    )

    collection_config = CollectionConfig(episodes=1, episode_length=1)
    collection_config.safety.skip_env_close_on_slurm = True

    episodes, num_actions = collector._collect_episode_batch(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"),
        collection_config,
        [11],
    )

    assert len(episodes) == 1
    assert num_actions == 3
    assert close_calls == []


def test_collect_raw_dataset_forwards_collection_spawn_regions_without_mutating_environment(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    progress_details: list[str | None] = []

    def fake_render_topdown(
        environment_config: EnvironmentConfig, seed: int, safety: CollectionSafetyConfig
    ):
        del seed, safety
        captured["topdown_spawn_regions"] = environment_config.env_kwargs.get("spawn_regions")
        captured["topdown_reward_on_goal"] = environment_config.env_kwargs.get("reward_on_goal")
        captured["topdown_terminate_on_goal"] = environment_config.env_kwargs.get(
            "terminate_on_goal"
        )
        return None

    class FakeObservation:
        def __init__(self) -> None:
            self.position_xy = np.zeros(2, dtype=np.float32)
            self.heading = 0.0
            self.info = {}
            self.modalities = {"rgb": np.zeros((3, 8, 8), dtype=np.uint8)}

        @property
        def rgb(self):
            return self.modalities["rgb"]

        def require_modality(self, name: str):
            return self.modalities[name]

    class FakeStepResult:
        def __init__(self) -> None:
            self.observation = FakeObservation()
            self.terminated = True
            self.truncated = False

    class FakeAdapter:
        action_space = DiscreteActionSpace(count=3)
        capabilities = EnvironmentCapabilities(observation_modalities=("rgb",))

        @property
        def num_actions(self) -> int:
            return self.action_space.count

        def reset(self, seed: int | None = None):
            del seed
            return FakeObservation()

        def step(self, action: int):
            del action
            return FakeStepResult()

        def close(self) -> None:
            return None

    def fake_build_environment(environment_config: EnvironmentConfig, seed: int):
        del seed
        captured["batch_spawn_regions"] = environment_config.env_kwargs.get("spawn_regions")
        captured["batch_reward_on_goal"] = environment_config.env_kwargs.get("reward_on_goal")
        captured["batch_terminate_on_goal"] = environment_config.env_kwargs.get("terminate_on_goal")
        return FakeAdapter()

    monkeypatch.setattr(collector, "render_topdown_with_timeout", fake_render_topdown)
    monkeypatch.setattr(collector, "build_environment", fake_build_environment)

    class FakeDatasetZarrStreamWriter:
        def __init__(self, output_path, array_specs, manifest_summary) -> None:
            captured["dataset_path"] = output_path
            captured["array_specs"] = array_specs
            captured["saved_summary"] = manifest_summary
            captured["saved_episodes"] = []

        def open(self):
            return self

        def write_episode(self, episode_index, arrays) -> None:
            captured["saved_episodes"].append((episode_index, arrays))

        def finalize(self) -> None:
            captured["finalized"] = True

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(collector, "DatasetZarrStreamWriter", FakeDatasetZarrStreamWriter)
    monkeypatch.setattr(
        collector._PreviewAccumulator,
        "write",
        lambda *args, **kwargs: None,
    )

    environment = EnvironmentConfig(
        env_id="MiniWorld-WallGapAsymLarge-v0",
        env_kwargs={"forward_step": 0.3},
    )
    collection_config = CollectionConfig(
        episodes=1,
        episode_length=1,
        spawn_regions=["courtyard_north", "concrete_yard"],
    )
    collection_config.safety.isolate_cuda_miniworld = False

    collector.collect_raw_dataset(
        environment,
        collection_config,
        tmp_path,
        collection_seed=11,
        progress_callback=lambda update: progress_details.append(update.detail),
    )

    assert captured["topdown_spawn_regions"] == ["courtyard_north", "concrete_yard"]
    assert captured["batch_spawn_regions"] == ["courtyard_north", "concrete_yard"]
    assert captured["topdown_reward_on_goal"] is False
    assert captured["topdown_terminate_on_goal"] is False
    assert captured["batch_reward_on_goal"] is False
    assert captured["batch_terminate_on_goal"] is False
    assert str(captured["dataset_path"]).endswith("dataset.zarr")
    assert captured["array_specs"]["observations/rgb"][0] == (1, 1, 3, 8, 8)
    assert len(captured["saved_episodes"]) == 1
    assert captured["finalized"] is True
    assert environment.env_kwargs == {"forward_step": 0.3}
    assert progress_details == [
        "collect episodes",
        "collect episodes",
        "write dataset.zarr",
        "write previews",
        "dataset collection complete",
    ]


def test_collection_environment_config_forces_passive_goal_task_for_wallgap_envs() -> None:
    environment = EnvironmentConfig(
        env_id="MiniWorld-WallGapAsym-v0",
        env_kwargs={"forward_step": 0.3, "reward_on_goal": True, "terminate_on_goal": True},
    )
    collection_config = CollectionConfig(episodes=1, episode_length=1)

    collection_environment = collector._environment_config_for_collection(
        environment, collection_config
    )

    assert collection_environment.env_kwargs["forward_step"] == 0.3
    assert collection_environment.env_kwargs["render_goal_object"] is False
    assert collection_environment.env_kwargs["reward_on_goal"] is False
    assert collection_environment.env_kwargs["terminate_on_goal"] is False
    assert environment.env_kwargs == {
        "forward_step": 0.3,
        "reward_on_goal": True,
        "terminate_on_goal": True,
    }


def test_collect_parallel_auto_shard_size_is_bounded_for_lower_process_churn(monkeypatch) -> None:
    submitted_shards: list[list[int]] = []
    queued_messages: list[dict[str, object]] = []

    class FakeQueue:
        def get(self, timeout: float | None = None):
            del timeout
            if queued_messages:
                return queued_messages.pop(0)
            raise Empty

        def close(self) -> None:
            return None

        def cancel_join_thread(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        def __init__(self) -> None:
            self.exitcode = 0

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

        def terminate(self) -> None:
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

    def fake_launch_worker(
        context,
        result_queue,
        *,
        batch_collector,
        shard_index: int,
        environment_config: EnvironmentConfig,
        collection_config: CollectionConfig,
        shard: list[int],
        startup_delay_seconds: float,
    ):
        del context, result_queue, batch_collector
        del environment_config, collection_config, startup_delay_seconds
        shard_list = list(shard)
        submitted_shards.append(shard_list)
        queued_messages.append(
            {
                "status": "ok",
                "shard_index": shard_index,
                "episodes": [{} for _ in shard_list],
                "num_actions": 3,
            }
        )
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=shard_list,
            process=FakeProcess(),
            started_at=0.0,
        )

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: 0.0)

    collection_config = CollectionConfig(episodes=1024)
    collection_config.safety.num_workers = 4
    collection_config.safety.worker_max_episodes = 0

    results = list(
        parallel_runtime.iter_parallel_episode_batches(
            EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
            collection_config,
            list(range(1024)),
            batch_collector=collector._collect_episode_batch,
        )
    )
    episodes = [episode for batch in results for episode in batch.episodes]
    num_actions = results[0].num_actions

    assert len(episodes) == 1024
    assert num_actions == 3
    assert submitted_shards
    assert max(len(shard) for shard in submitted_shards) == 32


def test_collect_parallel_progress_messages_reset_watchdog_and_emit_progress(monkeypatch) -> None:
    progress_updates: list[int] = []

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = [
                {"status": "progress", "shard_index": 0, "completed": 1},
                {
                    "status": "ok",
                    "shard_index": 0,
                    "episodes": [{}, {}],
                    "num_actions": 3,
                },
            ]

        def get(self, timeout: float | None = None):
            del timeout
            if self.messages:
                return self.messages.pop(0)
            raise Empty

        def close(self) -> None:
            return None

        def cancel_join_thread(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        exitcode = None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            self.exitcode = 0

        def terminate(self) -> None:
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

    def fake_launch_worker(
        context,
        result_queue,
        *,
        batch_collector,
        shard_index: int,
        environment_config: EnvironmentConfig,
        collection_config: CollectionConfig,
        shard: list[int],
        startup_delay_seconds: float,
    ):
        del context, result_queue, batch_collector
        del environment_config, collection_config, startup_delay_seconds
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=list(shard),
            process=FakeProcess(),
            started_at=0.0,
        )

    monotonic_values = iter([0.0, 299.0, 599.0, 599.0])

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: next(monotonic_values))

    collection_config = CollectionConfig(episodes=2)
    collection_config.safety.num_workers = 1
    collection_config.safety.watchdog_timeout_seconds = 300.0

    results = list(
        parallel_runtime.iter_parallel_episode_batches(
            EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
            collection_config,
            [123, 124],
            batch_collector=collector._collect_episode_batch,
            on_episode_complete=progress_updates.append,
        )
    )

    assert progress_updates == [1]
    assert len(results) == 1
    assert len(results[0].episodes) == 2


def test_collect_parallel_terminates_hung_worker_when_watchdog_expires(monkeypatch) -> None:
    class FakeQueue:
        def get(self, timeout: float | None = None):
            del timeout
            raise Empty

        def close(self) -> None:
            return None

        def cancel_join_thread(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.exitcode = None
            self.terminated = False

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

        def terminate(self) -> None:
            self.terminated = True
            self.exitcode = -15

        def kill(self) -> None:
            self.terminated = True
            self.exitcode = -9

    launched_processes: list[FakeProcess] = []

    def fake_launch_worker(
        context,
        result_queue,
        *,
        batch_collector,
        shard_index: int,
        environment_config: EnvironmentConfig,
        collection_config: CollectionConfig,
        shard: list[int],
        startup_delay_seconds: float,
    ):
        del context, result_queue, batch_collector
        del environment_config, collection_config, startup_delay_seconds
        process = FakeProcess()
        launched_processes.append(process)
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=list(shard),
            process=process,
            started_at=0.0,
        )

    monotonic_values = iter([301.0, 301.0])

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: next(monotonic_values))

    def fake_killpg(process_group_id: int, signal_number: int) -> None:
        del process_group_id, signal_number
        launched_processes[0].terminated = True
        launched_processes[0].exitcode = -15

    monkeypatch.setattr(parallel_runtime.os, "killpg", fake_killpg)

    collection_config = CollectionConfig(episodes=1)
    collection_config.safety.num_workers = 1
    collection_config.safety.watchdog_timeout_seconds = 300.0

    try:
        list(
            parallel_runtime.iter_parallel_episode_batches(
                EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
                collection_config,
                [123],
                batch_collector=collector._collect_episode_batch,
            )
        )
    except TimeoutError as exc:
        assert "shard 0" in str(exc)
        assert "seed 123" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected the watchdog to time out the hung worker.")

    assert len(launched_processes) == 1
    assert launched_processes[0].terminated is True


def test_collect_parallel_terminates_workers_when_rss_watchdog_exceeds_limit(monkeypatch) -> None:
    class FakeQueue:
        def get(self, timeout: float | None = None):
            del timeout
            raise Empty

        def close(self) -> None:
            return None

        def cancel_join_thread(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.exitcode = None
            self.terminated = False

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def terminate(self) -> None:
            self.terminated = True
            self.exitcode = -15

        def kill(self) -> None:
            self.terminated = True
            self.exitcode = -9

    launched_processes: list[FakeProcess] = []

    def fake_launch_worker(
        context,
        result_queue,
        *,
        batch_collector,
        shard_index: int,
        environment_config: EnvironmentConfig,
        collection_config: CollectionConfig,
        shard: list[int],
        startup_delay_seconds: float,
    ):
        del context, result_queue, batch_collector
        del environment_config, collection_config, startup_delay_seconds
        process = FakeProcess()
        launched_processes.append(process)
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=list(shard),
            process=process,
            started_at=0.0,
        )

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(parallel_runtime, "_read_process_rss_mb", lambda pid: 2048.0)

    def fake_killpg(process_group_id: int, signal_number: int) -> None:
        del process_group_id, signal_number
        launched_processes[0].terminated = True
        launched_processes[0].exitcode = -15

    monkeypatch.setattr(parallel_runtime.os, "killpg", fake_killpg)

    collection_config = CollectionConfig(episodes=1)
    collection_config.safety.num_workers = 1
    collection_config.safety.memory_watchdog_limit_mb = 1024

    with pytest.raises(MemoryError, match="Parallel collection RSS watchdog"):
        list(
            parallel_runtime.iter_parallel_episode_batches(
                EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
                collection_config,
                [123],
                batch_collector=collector._collect_episode_batch,
            )
        )

    assert len(launched_processes) == 1
    assert launched_processes[0].terminated is True


def test_collection_memory_watchdog_uses_explicit_limit_before_slurm_memory(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "8192")
    collection_config = CollectionConfig(episodes=1)
    collection_config.safety.memory_watchdog_limit_mb = 1024

    assert parallel_runtime._resolve_collection_memory_watchdog_limit_mb(collection_config) == 1024


def test_collection_memory_watchdog_derives_limit_from_slurm_memory(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "10240")
    collection_config = CollectionConfig(episodes=1)
    collection_config.safety.memory_watchdog_limit_mb = 0

    assert parallel_runtime._resolve_collection_memory_watchdog_limit_mb(collection_config) == 9216


def test_collect_parallel_cancels_queue_feeder_join_on_shutdown(monkeypatch) -> None:
    queue_calls: list[str] = []

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = [
                {
                    "status": "ok",
                    "shard_index": 0,
                    "episodes": [{}],
                    "num_actions": 3,
                }
            ]

        def get(self, timeout: float | None = None):
            del timeout
            if self.messages:
                return self.messages.pop(0)
            raise Empty

        def close(self) -> None:
            queue_calls.append("close")

        def cancel_join_thread(self) -> None:
            queue_calls.append("cancel_join_thread")

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        exitcode = 0

        def join(self, timeout: float | None = None) -> None:
            del timeout

    def fake_launch_worker(
        context,
        result_queue,
        *,
        batch_collector,
        shard_index: int,
        environment_config: EnvironmentConfig,
        collection_config: CollectionConfig,
        shard: list[int],
        startup_delay_seconds: float,
    ):
        del (
            context,
            result_queue,
            batch_collector,
            environment_config,
            collection_config,
            startup_delay_seconds,
        )
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=list(shard),
            process=FakeProcess(),
            started_at=0.0,
        )

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: 0.0)

    collection_config = CollectionConfig(episodes=1)
    collection_config.safety.num_workers = 1

    results = list(
        parallel_runtime.iter_parallel_episode_batches(
            EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
            collection_config,
            [123],
            batch_collector=collector._collect_episode_batch,
        )
    )

    assert len(results) == 1
    assert queue_calls == ["cancel_join_thread", "close"]


def test_collect_parallel_terminates_reported_worker_when_join_times_out(monkeypatch) -> None:
    process_state: dict[str, bool] = {"terminated": False}

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = [
                {
                    "status": "ok",
                    "shard_index": 0,
                    "episodes": [{}],
                    "num_actions": 3,
                }
            ]

        def get(self, timeout: float | None = None):
            del timeout
            if self.messages:
                return self.messages.pop(0)
            raise Empty

        def close(self) -> None:
            return None

        def cancel_join_thread(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.exitcode = None

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def terminate(self) -> None:
            process_state["terminated"] = True
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

    def fake_launch_worker(
        context,
        result_queue,
        *,
        batch_collector,
        shard_index: int,
        environment_config: EnvironmentConfig,
        collection_config: CollectionConfig,
        shard: list[int],
        startup_delay_seconds: float,
    ):
        del (
            context,
            result_queue,
            batch_collector,
            environment_config,
            collection_config,
            startup_delay_seconds,
        )
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=list(shard),
            process=FakeProcess(),
            started_at=0.0,
        )

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: 0.0)

    def fake_killpg(process_group_id: int, signal_number: int) -> None:
        del process_group_id, signal_number
        process_state["terminated"] = True

    monkeypatch.setattr(parallel_runtime.os, "killpg", fake_killpg)

    collection_config = CollectionConfig(episodes=1)
    collection_config.safety.num_workers = 1

    results = list(
        parallel_runtime.iter_parallel_episode_batches(
            EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
            collection_config,
            [123],
            batch_collector=collector._collect_episode_batch,
        )
    )

    assert len(results) == 1
    assert process_state == {"terminated": True}


def test_collect_worker_does_not_swallow_process_exit_exceptions() -> None:
    queued_messages: list[dict[str, object]] = []

    class FakeQueue:
        def put(self, payload: dict[str, object]) -> None:
            queued_messages.append(payload)

    def interrupted_collector(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        parallel_runtime._collect_episode_batch_worker(
            FakeQueue(),
            interrupted_collector,
            EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
            CollectionConfig(episodes=1),
            shard_index=0,
            episode_seeds=[123],
            startup_delay_seconds=0.0,
        )

    assert queued_messages == []


def test_collect_worker_starts_new_session_inside_spawned_child(monkeypatch) -> None:
    session_calls: list[bool] = []
    queued_messages: list[dict[str, object]] = []

    class FakeQueue:
        def put(self, payload: dict[str, object]) -> None:
            queued_messages.append(payload)

    def fake_collector(*args, **kwargs):
        del args, kwargs
        return [], 3

    monkeypatch.setattr(parallel_runtime.mp, "parent_process", lambda: object())
    monkeypatch.setattr(parallel_runtime.os, "setsid", lambda: session_calls.append(True))

    parallel_runtime._collect_episode_batch_worker(
        FakeQueue(),
        fake_collector,
        EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
        CollectionConfig(episodes=1),
        shard_index=0,
        episode_seeds=[123],
        startup_delay_seconds=0.0,
    )

    assert session_calls == [True]
    assert queued_messages[0]["status"] == "ok"


def test_terminate_collection_process_signals_process_group_before_pid(monkeypatch) -> None:
    signals: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 456

        def __init__(self) -> None:
            self.exitcode = None
            self.terminated = False

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def terminate(self) -> None:
            self.terminated = True
            self.exitcode = -15

        def kill(self) -> None:
            self.exitcode = -9

    process = FakeProcess()

    def fake_killpg(process_group_id: int, signal_number: int) -> None:
        signals.append((process_group_id, signal_number))
        process.exitcode = -15

    monkeypatch.setattr(parallel_runtime.os, "killpg", fake_killpg)

    parallel_runtime._terminate_collection_process(process)

    assert signals == [(456, signal.SIGTERM)]
    assert process.terminated is False


def test_render_topdown_kills_worker_when_terminate_does_not_finish(monkeypatch) -> None:
    process_state: dict[str, bool] = {"terminated": False, "killed": False}
    connection_state: dict[str, bool] = {"parent_closed": False, "child_closed": False}

    class FakeConnection:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            connection_state[f"{self.name}_closed"] = True

        def poll(self) -> bool:
            return False

    class FakeProcess:
        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

        def is_alive(self) -> bool:
            return not process_state["killed"]

        def terminate(self) -> None:
            process_state["terminated"] = True

        def kill(self) -> None:
            process_state["killed"] = True

    class FakeContext:
        def Pipe(self, duplex: bool):
            del duplex
            return FakeConnection("parent"), FakeConnection("child")

        def Process(self, *, target, args):
            del target, args
            return FakeProcess()

    monkeypatch.setattr(collection_safety, "configure_start_method", lambda safety: FakeContext())

    result = collection_safety.render_topdown_with_timeout(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
        seed=123,
        safety=CollectionSafetyConfig(render_timeout_seconds=0.01),
    )

    assert result is None
    assert process_state == {"terminated": True, "killed": True}
    assert connection_state == {"parent_closed": True, "child_closed": True}


def test_collect_raw_dataset_allows_coordinate_only_collection_when_rgb_is_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeObservation:
        def __init__(self) -> None:
            self.position_xy = np.zeros(2, dtype=np.float32)
            self.heading = 0.0
            self.info = {}
            self.modalities: dict[str, np.ndarray] = {}

        @property
        def rgb(self):
            return None

        def require_modality(self, name: str):
            raise KeyError(name)

    class FakeStepResult:
        def __init__(self) -> None:
            self.observation = FakeObservation()
            self.terminated = True
            self.truncated = False

    class FakeAdapter:
        action_space = DiscreteActionSpace(count=3)
        capabilities = EnvironmentCapabilities(observation_modalities=())

        @property
        def num_actions(self) -> int:
            return self.action_space.count

        def reset(self, seed: int | None = None):
            del seed
            return FakeObservation()

        def step(self, action: int):
            del action
            return FakeStepResult()

        def render_topdown(self):
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        collector, "build_environment", lambda environment_config, seed: FakeAdapter()
    )
    monkeypatch.setattr(
        collector, "render_topdown_with_timeout", lambda environment_config, seed, safety: None
    )
    monkeypatch.setattr(
        collector._PreviewAccumulator,
        "write",
        lambda *args, **kwargs: None,
    )

    class FakeDatasetZarrStreamWriter:
        def __init__(self, output_path, array_specs, manifest_summary) -> None:
            del output_path
            captured["array_specs"] = array_specs
            captured["episodes"] = []
            captured["summary"] = manifest_summary

        def open(self):
            return self

        def write_episode(self, episode_index, arrays) -> None:
            captured["episodes"].append((episode_index, arrays))

        def finalize(self) -> None:
            captured["finalized"] = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(collector, "DatasetZarrStreamWriter", FakeDatasetZarrStreamWriter)

    collection_config = CollectionConfig(episodes=1, episode_length=2, save_rgb=False)
    collection_config.safety.isolate_cuda_miniworld = False
    collector.collect_raw_dataset(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"),
        collection_config,
        tmp_path,
        collection_seed=5,
    )

    array_specs = captured["array_specs"]
    assert "observations/rgb" not in array_specs
    assert captured["summary"].modalities == []
    saved_episode = captured["episodes"][0][1]
    assert "observations/rgb" not in saved_episode
    assert captured["finalized"] is True


def test_collect_raw_dataset_writes_previews_after_dataset_finalize(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeObservation:
        def __init__(self) -> None:
            self.position_xy = np.zeros(2, dtype=np.float32)
            self.heading = 0.0
            self.info = {}
            self.modalities: dict[str, np.ndarray] = {}

        def require_modality(self, name: str):
            raise KeyError(name)

    class FakeStepResult:
        def __init__(self) -> None:
            self.observation = FakeObservation()
            self.terminated = True
            self.truncated = False

    class FakeAdapter:
        action_space = DiscreteActionSpace(count=3)
        capabilities = EnvironmentCapabilities(observation_modalities=())

        def reset(self, seed: int | None = None):
            del seed
            return FakeObservation()

        def step(self, action: int):
            del action
            return FakeStepResult()

        def close(self) -> None:
            return None

    class FakeDatasetZarrStreamWriter:
        def __init__(self, output_path, array_specs, manifest_summary) -> None:
            del output_path, array_specs, manifest_summary

        def open(self):
            return self

        def write_episode(self, episode_index, arrays) -> None:
            del episode_index, arrays
            events.append("write")

        def finalize(self) -> None:
            events.append("finalize")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        collector,
        "build_environment",
        lambda environment_config, seed: FakeAdapter(),
    )
    monkeypatch.setattr(
        collector,
        "render_topdown_with_timeout",
        lambda environment_config, seed, safety: None,
    )
    monkeypatch.setattr(collector, "DatasetZarrStreamWriter", FakeDatasetZarrStreamWriter)
    monkeypatch.setattr(
        collector._PreviewAccumulator,
        "write",
        lambda self, preview_dir, **kwargs: events.append("preview"),
    )

    collection_config = CollectionConfig(episodes=1, episode_length=1, save_rgb=False)
    collection_config.safety.isolate_cuda_miniworld = False
    collector.collect_raw_dataset(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"),
        collection_config,
        tmp_path,
        collection_seed=5,
    )

    assert events == ["write", "finalize", "preview"]


def test_preview_accumulator_extracts_valid_positions_and_samples_frames() -> None:
    accumulator = collector._PreviewAccumulator()
    accumulator.add_episode(
        {
            POSITION_KEY: np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            HEADING_KEY: np.asarray([0.25, 0.5], dtype=np.float32),
            KINEMATICS_KEY: np.zeros((2, 4), dtype=np.float32),
            VALID_MASK_KEY: np.asarray([True, False], dtype=bool),
            RGB_KEY: np.zeros((2, 3, 4, 4), dtype=np.uint8),
        }
    )

    positions = np.concatenate(accumulator.position_chunks, axis=0)
    np.testing.assert_allclose(positions, np.asarray([[1.0, 2.0]]))
    assert len(accumulator.sample_frames) == 1


def test_collect_raw_dataset_serial_path_skips_env_close_on_slurm_when_configured(
    monkeypatch,
    tmp_path: Path,
) -> None:
    close_calls: list[int] = []

    class FakeObservation:
        def __init__(self) -> None:
            self.position_xy = np.zeros(2, dtype=np.float32)
            self.heading = 0.0
            self.info = {}
            self.modalities: dict[str, np.ndarray] = {}

        def require_modality(self, name: str):
            raise KeyError(name)

    class FakeStepResult:
        def __init__(self) -> None:
            self.observation = FakeObservation()
            self.terminated = True
            self.truncated = False

    class FakeAdapter:
        action_space = DiscreteActionSpace(count=3)
        capabilities = EnvironmentCapabilities(observation_modalities=())

        def reset(self, seed: int | None = None):
            del seed
            return FakeObservation()

        def step(self, action: int):
            del action
            return FakeStepResult()

        def close(self) -> None:
            close_calls.append(1)

    class FakeDatasetZarrStreamWriter:
        def __init__(self, output_path, array_specs, manifest_summary) -> None:
            del output_path, array_specs, manifest_summary

        def open(self):
            return self

        def write_episode(self, episode_index, arrays) -> None:
            del episode_index, arrays

        def finalize(self) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(
        collector,
        "build_environment",
        lambda environment_config, seed: FakeAdapter(),
    )
    monkeypatch.setattr(
        collector,
        "render_topdown_with_timeout",
        lambda environment_config, seed, safety: None,
    )
    monkeypatch.setattr(
        collector._PreviewAccumulator,
        "write",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(collector, "DatasetZarrStreamWriter", FakeDatasetZarrStreamWriter)

    collection_config = CollectionConfig(episodes=1, episode_length=1, save_rgb=False)
    collection_config.safety.num_workers = 1
    collection_config.safety.isolate_cuda_miniworld = False
    collection_config.safety.skip_env_close_on_slurm = True

    collector.collect_raw_dataset(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsym-v0"),
        collection_config,
        tmp_path,
        collection_seed=5,
    )

    assert close_calls == []


def test_collect_raw_dataset_isolates_default_single_worker_collection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"episodes": []}

    monkeypatch.setattr(
        collector,
        "render_topdown_with_timeout",
        lambda environment_config, seed, safety: None,
    )
    monkeypatch.setattr(
        collector._PreviewAccumulator,
        "write",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        collector,
        "build_environment",
        lambda environment_config, seed: (_ for _ in ()).throw(
            AssertionError("isolated single-worker collection must not build env in parent")
        ),
    )

    def make_episode(source_seed: int) -> dict[str, np.ndarray | bool | int]:
        return {
            "actions": np.asarray([0], dtype=np.int64),
            "position_xy": np.asarray([[float(source_seed), 0.0]], dtype=np.float32),
            "heading": np.asarray([0.0], dtype=np.float32),
            "kinematics": np.zeros((1, 4), dtype=np.float32),
            "valid_steps": np.asarray([True], dtype=bool),
            "length": 1,
            "terminated": True,
            "truncated": False,
            "source_seed": source_seed,
            "observations/rgb": np.zeros((1, 3, 4, 4), dtype=np.uint8),
        }

    def fake_iter_parallel_episode_batches(
        environment_config,
        collection_config,
        seeds,
        *,
        batch_collector,
        on_episode_complete=None,
    ):
        del environment_config, collection_config, batch_collector, on_episode_complete
        assert list(seeds) == [10, 11]
        yield parallel_runtime.ParallelEpisodeBatch(
            shard_index=0,
            seeds=[10, 11],
            episodes=[make_episode(10), make_episode(11)],
            num_actions=3,
        )

    class FakeDatasetZarrStreamWriter:
        def __init__(self, output_path, array_specs, manifest_summary) -> None:
            del output_path, array_specs, manifest_summary

        def open(self):
            return self

        def write_episode(self, episode_index, arrays) -> None:
            captured["episodes"].append((episode_index, int(arrays[SOURCE_SEED_KEY])))

        def finalize(self) -> None:
            captured["finalized"] = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        collector,
        "iter_parallel_episode_batches",
        fake_iter_parallel_episode_batches,
    )
    monkeypatch.setattr(collector, "DatasetZarrStreamWriter", FakeDatasetZarrStreamWriter)

    collection_config = CollectionConfig(episodes=2, episode_length=1)
    collection_config.safety.num_workers = 1
    collection_config.safety.isolate_cuda_miniworld = True

    collector.collect_raw_dataset(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
        collection_config,
        tmp_path,
        collection_seed=10,
    )

    assert captured["episodes"] == [(0, 10), (1, 11)]
    assert captured["finalized"] is True


def test_collect_raw_dataset_streams_parallel_batches_in_deterministic_episode_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {"episodes": []}

    monkeypatch.setattr(
        collector,
        "render_topdown_with_timeout",
        lambda environment_config, seed, safety: None,
    )
    monkeypatch.setattr(
        collector._PreviewAccumulator,
        "write",
        lambda *args, **kwargs: None,
    )

    def make_episode(source_seed: int) -> dict[str, np.ndarray | bool | int]:
        return {
            "actions": np.asarray([0], dtype=np.int64),
            "position_xy": np.asarray([[float(source_seed), 0.0]], dtype=np.float32),
            "heading": np.asarray([0.0], dtype=np.float32),
            "kinematics": np.zeros((1, 4), dtype=np.float32),
            "valid_steps": np.asarray([True], dtype=bool),
            "length": 1,
            "terminated": True,
            "truncated": False,
            "source_seed": source_seed,
            "observations/rgb": np.zeros((1, 3, 4, 4), dtype=np.uint8),
        }

    def fake_iter_parallel_episode_batches(
        environment_config,
        collection_config,
        seeds,
        *,
        batch_collector,
        on_episode_complete=None,
    ):
        del environment_config, collection_config, seeds, batch_collector, on_episode_complete
        yield parallel_runtime.ParallelEpisodeBatch(
            shard_index=1,
            seeds=[12, 13],
            episodes=[make_episode(12), make_episode(13)],
            num_actions=3,
        )
        yield parallel_runtime.ParallelEpisodeBatch(
            shard_index=0,
            seeds=[10, 11],
            episodes=[make_episode(10), make_episode(11)],
            num_actions=3,
        )

    class FakeDatasetZarrStreamWriter:
        def __init__(self, output_path, array_specs, manifest_summary) -> None:
            del output_path, array_specs, manifest_summary

        def open(self):
            return self

        def write_episode(self, episode_index, arrays) -> None:
            captured["episodes"].append((episode_index, int(arrays[SOURCE_SEED_KEY])))

        def finalize(self) -> None:
            captured["finalized"] = True

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        collector,
        "iter_parallel_episode_batches",
        fake_iter_parallel_episode_batches,
    )
    monkeypatch.setattr(collector, "DatasetZarrStreamWriter", FakeDatasetZarrStreamWriter)

    collection_config = CollectionConfig(episodes=4, episode_length=1)
    collection_config.safety.num_workers = 2
    collector.collect_raw_dataset(
        EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
        collection_config,
        tmp_path,
        collection_seed=10,
    )

    assert captured["episodes"] == [(0, 10), (1, 11), (2, 12), (3, 13)]
    assert captured["finalized"] is True


def test_collection_worker_count_takes_a_positive_setting_literally(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    collection_config = CollectionConfig(episodes=8)
    collection_config.safety.num_workers = 3

    assert parallel_runtime.resolve_collection_worker_count(collection_config) == 3


def test_collection_worker_count_auto_leaves_one_allocated_cpu_for_the_parent(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    collection_config = CollectionConfig(episodes=8)
    collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS

    assert parallel_runtime.resolve_collection_worker_count(collection_config) == 15


@pytest.mark.parametrize("allocated_cpus", ["", "1", "not-a-number"])
def test_collection_worker_count_auto_defaults_to_one_without_a_usable_allocation(
    monkeypatch,
    allocated_cpus: str,
) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", allocated_cpus)
    collection_config = CollectionConfig(episodes=8)
    collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS

    assert parallel_runtime.resolve_collection_worker_count(collection_config) == 1


def test_collection_worker_count_auto_defaults_to_one_off_slurm(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    collection_config = CollectionConfig(episodes=8)
    collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS

    assert parallel_runtime.resolve_collection_worker_count(collection_config) == 1


def test_collection_worker_count_auto_never_exceeds_the_allocation(monkeypatch) -> None:
    """Staying inside the cgroup is the whole point of the guard."""
    for allocated_cpus in (1, 2, 4, 8, 16, 112):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(allocated_cpus))
        collection_config = CollectionConfig(episodes=8)
        collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS

        resolved = parallel_runtime.resolve_collection_worker_count(collection_config)

        assert 1 <= resolved <= allocated_cpus


def _run_parallel_collection_recording_concurrency(
    monkeypatch,
    collection_config: CollectionConfig,
    seeds: list[int],
) -> tuple[int, int]:
    """Drive the real scheduler with fake processes; return (peak in flight, episodes yielded)."""
    queued_messages: list[dict[str, object]] = []
    in_flight = 0
    peak_in_flight = 0

    class FakeQueue:
        def get(self, timeout: float | None = None):
            del timeout
            if not queued_messages:
                raise Empty
            nonlocal in_flight
            in_flight -= 1
            return queued_messages.pop(0)

        def close(self) -> None:
            return None

        def cancel_join_thread(self) -> None:
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

    class FakeProcess:
        def __init__(self) -> None:
            self.exitcode = 0
            self.pid = 4321

        def join(self, timeout: float | None = None) -> None:
            del timeout

    def fake_launch_worker(context, result_queue, *, shard_index, shard, **kwargs):
        del context, result_queue, kwargs
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        queued_messages.append(
            {
                "status": "ok",
                "shard_index": shard_index,
                "episodes": [{"source_seed": seed} for seed in shard],
                "num_actions": 3,
            }
        )
        return parallel_runtime._ParallelCollectionWorker(
            shard_index=shard_index,
            seeds=list(shard),
            process=FakeProcess(),
            started_at=0.0,
        )

    monkeypatch.setattr(parallel_runtime, "configure_start_method", lambda safety: FakeContext())
    monkeypatch.setattr(parallel_runtime, "_launch_collection_worker", fake_launch_worker)
    monkeypatch.setattr(parallel_runtime.time, "monotonic", lambda: 0.0)

    batches = list(
        parallel_runtime.iter_parallel_episode_batches(
            EnvironmentConfig(env_id="MiniWorld-WallGapAsymLarge-v0"),
            collection_config,
            seeds,
            batch_collector=collector._collect_episode_batch,
        )
    )
    return peak_in_flight, sum(len(batch.episodes) for batch in batches)


def test_collect_parallel_auto_width_reaches_the_scheduler(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    collection_config = CollectionConfig(episodes=1024)
    collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS
    collection_config.safety.worker_max_episodes = 16

    peak_in_flight, episodes = _run_parallel_collection_recording_concurrency(
        monkeypatch, collection_config, list(range(1024))
    )

    assert peak_in_flight == 15
    assert episodes == 1024


def test_collect_parallel_auto_width_stays_serial_off_slurm(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    collection_config = CollectionConfig(episodes=64)
    collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS
    collection_config.safety.worker_max_episodes = 8

    peak_in_flight, episodes = _run_parallel_collection_recording_concurrency(
        monkeypatch, collection_config, list(range(64))
    )

    assert peak_in_flight == 1
    assert episodes == 64


def test_collect_parallel_shards_partition_the_seeds_in_order_at_every_width(monkeypatch) -> None:
    """Data identity, the config-level half: width changes packing, never seed order."""
    seeds = list(range(200))
    partitions = []
    for allocated_cpus in (1, 2, 13, 17):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", str(allocated_cpus))
        collection_config = CollectionConfig(episodes=len(seeds))
        collection_config.safety.num_workers = parallel_runtime.AUTO_COLLECTION_WORKERS
        max_workers = parallel_runtime.resolve_collection_worker_count(collection_config)
        shards = parallel_runtime._resolve_collection_shards(seeds, collection_config, max_workers)
        partitions.append([seed for shard in shards for seed in shard])

    assert all(partition == seeds for partition in partitions)
