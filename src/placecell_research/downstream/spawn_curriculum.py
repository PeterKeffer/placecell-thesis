"""Episode-indexed spawn curriculum helpers for downstream RL."""

from __future__ import annotations

from bisect import bisect_right

from placecell_research.config.downstream_schema import DownstreamSpawnCurriculumPhaseConfig


class SpawnCurriculumScheduler:
    """Stateless curriculum lookup keyed by completed episode count."""

    def __init__(self, phases: list[DownstreamSpawnCurriculumPhaseConfig]) -> None:
        if not phases:
            raise ValueError("SpawnCurriculumScheduler requires at least one phase.")
        self._phases = list(phases)
        self._start_episodes = [int(phase.start_episode) for phase in self._phases]
        if self._start_episodes[0] != 0:
            raise ValueError("Spawn curriculum must start at episode 0.")
        if self._start_episodes != sorted(self._start_episodes):
            raise ValueError("Spawn curriculum phases must be sorted by start_episode.")
        if len(self._start_episodes) != len(set(self._start_episodes)):
            raise ValueError("Spawn curriculum phases must not repeat start_episode values.")

    @property
    def phases(self) -> list[DownstreamSpawnCurriculumPhaseConfig]:
        return list(self._phases)

    def phase_index(self, completed_episodes: int) -> int:
        if completed_episodes < 0:
            raise ValueError("completed_episodes must be non-negative.")
        return max(0, bisect_right(self._start_episodes, int(completed_episodes)) - 1)

    def active_phase(self, completed_episodes: int) -> DownstreamSpawnCurriculumPhaseConfig:
        return self._phases[self.phase_index(completed_episodes)]


def format_phase_transition(
    phase: DownstreamSpawnCurriculumPhaseConfig,
    phase_index: int,
    episode: int,
    timestep: int,
) -> str:
    region_text = "default" if not phase.spawn_regions else ",".join(phase.spawn_regions)
    region_bounds_text = (
        "default"
        if phase.spawn_region_xz is None
        else ",".join(f"{float(value):.2f}" for value in phase.spawn_region_xz)
    )
    max_spawn_text = (
        "default" if phase.max_spawn_distance is None else f"{float(phase.max_spawn_distance):.2f}"
    )
    goal_tolerance_text = (
        "default" if phase.goal_tolerance is None else f"{float(phase.goal_tolerance):.2f}"
    )
    goal_schedule_text = "default" if phase.goal_schedule is None else str(phase.goal_schedule)
    goal_interval_text = (
        "default"
        if phase.goal_change_interval_episodes is None
        else str(int(phase.goal_change_interval_episodes))
    )
    goal_index_text = "default" if phase.goal_index is None else str(int(phase.goal_index))
    separator = "=" * 88
    return "\n".join(
        [
            separator,
            (
                f"Spawn curriculum transition -> phase[{int(phase_index)}] '{phase.name}' "
                f"at episode {int(episode)} and timestep {int(timestep)}"
            ),
            (
                "settings: "
                f"spawn_regions={region_text} | "
                f"spawn_region_xz={region_bounds_text} | "
                f"max_spawn_distance={max_spawn_text} | "
                f"goal_tolerance={goal_tolerance_text} | "
                f"goal_schedule={goal_schedule_text} | "
                f"goal_change_interval_episodes={goal_interval_text} | "
                f"goal_index={goal_index_text}"
            ),
            separator,
        ]
    )
