"""Lightweight downstream trajectory diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from ..analysis.figures import figure_to_rgb_array, save_gif
from ..analysis.world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    overlay_bounds,
    resolve_world_overlay,
)
from ..collection.previews import draw_goal_markers


def _position_xy_from_info(info: dict[str, Any], key: str = "position_xy") -> list[float] | None:
    raw_position = info.get(key)
    if raw_position is None:
        return None
    position = np.asarray(raw_position, dtype=np.float32).reshape(-1)
    if position.shape[0] < 2:
        return None
    return [float(position[0]), float(position[1])]


def _heading_from_info(info: dict[str, Any]) -> float | None:
    raw_heading = info.get("heading")
    if raw_heading is None:
        return None
    return float(raw_heading)


def _headings_to_array(headings: list[float | None], length: int) -> np.ndarray | None:
    if not any(heading is not None for heading in headings):
        return None
    padded_headings = list(headings[:length])
    if len(padded_headings) < length:
        padded_headings.extend([None] * (length - len(padded_headings)))
    return np.asarray(
        [np.nan if heading is None else float(heading) for heading in padded_headings],
        dtype=np.float32,
    )


def _serializable_headings(headings: np.ndarray | None) -> list[float | None] | None:
    if headings is None:
        return None
    return [
        None if not np.isfinite(float(heading)) else float(heading)
        for heading in np.asarray(headings, dtype=np.float32).reshape(-1)
    ]


def _heading_at(headings: np.ndarray | None, index: int) -> float | None:
    if headings is None or index >= len(headings):
        return None
    heading = float(headings[index])
    return heading if np.isfinite(heading) else None


def _trajectory_quality_metrics(
    positions_xy: list[list[float]],
    goal_position_xy: list[float] | None,
) -> dict[str, float | None]:
    positions = np.asarray(positions_xy, dtype=np.float32).reshape(-1, 2)
    if positions.shape[0] < 2:
        path_length = 0.0
    else:
        deltas = np.diff(positions, axis=0)
        step_distances = np.linalg.norm(deltas, axis=1)
        path_length = float(np.sum(step_distances))
    straight_line_distance = float(np.linalg.norm(positions[-1] - positions[0]))
    path_efficiency = None if path_length <= 1e-8 else float(straight_line_distance / path_length)
    final_goal_distance = None
    if goal_position_xy is not None:
        goal_xy = np.asarray(goal_position_xy, dtype=np.float32).reshape(2)
        final_goal_distance = float(np.linalg.norm(positions[-1] - goal_xy))
    stuck_step_fraction = 0.0
    if positions.shape[0] >= 2:
        step_distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        stuck_step_fraction = float(np.mean(step_distances < 1e-3))
    return {
        "path_length": path_length,
        "straight_line_distance": straight_line_distance,
        "path_efficiency": path_efficiency,
        "final_goal_distance": final_goal_distance,
        "stuck_step_fraction": stuck_step_fraction,
    }


class TrainingTrajectoryRecorder:
    """Stream episode trajectories and write compact training diagnostics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        env_id: str,
        num_envs: int,
        max_rendered_trajectories: int = 200,
        max_recorded_episodes: int | None = None,
        file_prefix: str = "training",
        write_gif: bool = False,
        individual_trajectory_gif_count: int = 0,
        individual_trajectory_gif_max_frames: int = 64,
    ) -> None:
        self.output_dir = output_dir
        self.env_id = env_id
        self.num_envs = max(1, int(num_envs))
        self.max_rendered_trajectories = max(1, int(max_rendered_trajectories))
        self.max_recorded_episodes = (
            None if max_recorded_episodes is None else max(1, int(max_recorded_episodes))
        )
        self.file_prefix = str(file_prefix)
        self.write_gif = bool(write_gif)
        self.individual_trajectory_gif_count = max(0, int(individual_trajectory_gif_count))
        self.individual_trajectory_gif_max_frames = max(
            2, int(individual_trajectory_gif_max_frames)
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_jsonl_path = self.output_dir / f"{self.file_prefix}_trajectories.jsonl"
        self._current_positions: list[list[list[float]]] = [[] for _ in range(self.num_envs)]
        self._current_headings: list[list[float | None]] = [[] for _ in range(self.num_envs)]
        self._episode_start_positions: list[list[float]] = []
        self._episode_successes: list[bool] = []
        self._rendered_trajectories: list[np.ndarray] = []
        self._rendered_headings: list[np.ndarray | None] = []
        self._rendered_successes: list[bool] = []
        self._rendered_goals: list[list[float] | None] = []
        self._episode_count = 0
        self._render_sample_rng = np.random.default_rng(0)

    def observe_step(
        self,
        *,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        timestep: int,
    ) -> None:
        done_flags = np.asarray(dones, dtype=bool).reshape(-1)
        for env_index in range(min(self.num_envs, len(infos))):
            info = infos[env_index] if isinstance(infos[env_index], dict) else {}
            position_xy = _position_xy_from_info(info)
            if position_xy is not None:
                self._current_positions[env_index].append(position_xy)
                self._current_headings[env_index].append(_heading_from_info(info))
            if env_index < len(done_flags) and bool(done_flags[env_index]):
                next_start_position_xy = _position_xy_from_info(info, key="next_start_position_xy")
                self._finish_episode(
                    env_index=env_index,
                    info=info,
                    timestep=timestep,
                    next_start_position_xy=next_start_position_xy,
                )
                if next_start_position_xy is not None:
                    self._current_positions[env_index] = [next_start_position_xy]
                    self._current_headings[env_index] = [None]

    def _finish_episode(
        self,
        *,
        env_index: int,
        info: dict[str, Any],
        timestep: int,
        next_start_position_xy: list[float] | None,
    ) -> None:
        positions_xy = self._current_positions[env_index]
        if not positions_xy:
            return
        if (
            self.max_recorded_episodes is not None
            and self._episode_count >= self.max_recorded_episodes
        ):
            self._current_positions[env_index] = []
            self._current_headings[env_index] = []
            return
        self._episode_count += 1
        trajectory = np.asarray(positions_xy, dtype=np.float32)
        headings = _headings_to_array(self._current_headings[env_index], len(positions_xy))
        goal_position_xy = _position_xy_from_info(info, key="goal_position_xy")
        success = bool(info.get("is_success", False))
        self._episode_start_positions.append(positions_xy[0])
        self._episode_successes.append(success)
        self._maybe_sample_rendered_trajectory(trajectory, headings, success, goal_position_xy)
        quality_metrics = _trajectory_quality_metrics(positions_xy, goal_position_xy)
        payload = {
            "episode": int(self._episode_count),
            "env_index": int(env_index),
            "timestep": int(timestep),
            "start_position_xy": positions_xy[0],
            "final_position_xy": positions_xy[-1],
            "next_start_position_xy": next_start_position_xy,
            "goal_position_xy": goal_position_xy,
            "success": success,
            "positions_xy": positions_xy,
            "headings": _serializable_headings(headings),
            **quality_metrics,
        }
        with self.trajectory_jsonl_path.open("a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._current_positions[env_index] = []
        self._current_headings[env_index] = []

    def _maybe_sample_rendered_trajectory(
        self,
        trajectory: np.ndarray,
        headings: np.ndarray | None,
        success: bool,
        goal_position_xy: list[float] | None,
    ) -> None:
        if len(self._rendered_trajectories) < self.max_rendered_trajectories:
            self._rendered_trajectories.append(trajectory.copy())
            self._rendered_headings.append(None if headings is None else headings.copy())
            self._rendered_successes.append(success)
            self._rendered_goals.append(goal_position_xy)
            return
        replacement_index = int(self._render_sample_rng.integers(0, self._episode_count))
        if replacement_index >= self.max_rendered_trajectories:
            return
        self._rendered_trajectories[replacement_index] = trajectory.copy()
        self._rendered_headings[replacement_index] = None if headings is None else headings.copy()
        self._rendered_successes[replacement_index] = success
        self._rendered_goals[replacement_index] = goal_position_xy

    def close(self) -> None:
        if self._episode_start_positions:
            self._write_start_heatmap()
        if self._rendered_trajectories:
            self._write_start_outcome_scatter()
            self._write_trajectory_overlay()
            if self.write_gif:
                self._write_trajectory_gif()
                self._write_individual_trajectory_gifs()

    def _prepare_axis(
        self,
        axis: plt.Axes,
        *,
        positions_xy: np.ndarray,
        draw_overlay: bool = True,
    ) -> None:
        world_overlay = resolve_world_overlay(self.env_id)
        if draw_overlay and world_overlay is not None:
            draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#404040")
            draw_landmarks_on_axis(axis, world_overlay)
        apply_plot_bounds(axis, env_id=self.env_id, position_xy=positions_xy)
        axis.set_xlabel(POSITION_X_LABEL)
        axis.set_ylabel(POSITION_Y_LABEL)
        axis.grid(color="#d6d6d6", linewidth=0.35, alpha=0.5)

    def _draw_heading_arrow(
        self,
        axis: plt.Axes,
        *,
        position_xy: np.ndarray,
        heading: float | None,
    ) -> None:
        if heading is None:
            return
        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        arrow_length = max(
            min(abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0])) * 0.045, 0.3
        )
        axis.arrow(
            float(position_xy[0]),
            float(position_xy[1]),
            np.cos(float(heading)) * arrow_length,
            np.sin(float(heading)) * arrow_length,
            color="#f28e2b",
            width=max(arrow_length * 0.045, 0.015),
            head_width=max(arrow_length * 0.22, 0.12),
            head_length=max(arrow_length * 0.22, 0.12),
            length_includes_head=True,
            zorder=6,
        )

    def _write_start_heatmap(self) -> Path:
        starts = np.asarray(self._episode_start_positions, dtype=np.float32).reshape(-1, 2)
        figure, axis = plt.subplots(1, 1, figsize=(6, 5))
        self._prepare_axis(axis, positions_xy=starts, draw_overlay=False)
        x_edges, y_edges = _spatial_bin_edges(self.env_id, starts, num_bins=32)
        counts, _, _ = np.histogram2d(starts[:, 0], starts[:, 1], bins=[x_edges, y_edges])
        occupied = counts > 0.0
        occupancy_heatmap = np.ma.masked_where(~occupied.T, counts.T)
        positive_counts = counts[occupied]
        occupancy_norm: colors.Normalize
        if positive_counts.size:
            occupancy_norm = colors.PowerNorm(
                gamma=0.55,
                vmin=float(positive_counts.min()),
                vmax=float(positive_counts.max()),
            )
        else:
            occupancy_norm = colors.Normalize(vmin=0.0, vmax=1.0)
        axis.imshow(
            occupancy_heatmap,
            origin="lower",
            extent=(float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])),
            cmap="YlOrRd",
            norm=occupancy_norm,
            interpolation="nearest",
            aspect="equal",
        )
        world_overlay = resolve_world_overlay(self.env_id)
        if world_overlay is not None:
            draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#404040")
            draw_landmarks_on_axis(axis, world_overlay)
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=occupancy_norm, cmap="YlOrRd"),
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("completed episode starts per spatial bin", fontsize=9)
        colorbar.ax.tick_params(labelsize=7)
        figure_label = "Training" if self.file_prefix == "training" else "Eval"
        axis.set_title(f"{figure_label} completed episode starts (n={len(starts)})")
        figure.tight_layout()
        filename = (
            "start_position_heatmap.png"
            if self.file_prefix == "training"
            else f"{self.file_prefix}_start_position_heatmap.png"
        )
        path = self.output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def _write_start_outcome_scatter(self) -> Path:
        starts = np.asarray(self._episode_start_positions, dtype=np.float32).reshape(-1, 2)
        successes = np.asarray(self._episode_successes, dtype=bool)
        figure, axis = plt.subplots(1, 1, figsize=(6, 5))
        self._prepare_axis(axis, positions_xy=starts)
        failure_starts = starts[~successes]
        success_starts = starts[successes]
        if failure_starts.size:
            axis.scatter(
                failure_starts[:, 0],
                failure_starts[:, 1],
                color="#d62728",
                s=38,
                alpha=0.85,
                label="failed",
            )
        if success_starts.size:
            axis.scatter(
                success_starts[:, 0],
                success_starts[:, 1],
                color="#2ca02c",
                s=38,
                alpha=0.85,
                label="reached goal",
            )
        figure_label = "Training" if self.file_prefix == "training" else "Eval"
        axis.set_title(f"{figure_label} start outcomes (n={len(starts)})")
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        filename = (
            "start_position_outcomes.png"
            if self.file_prefix == "training"
            else f"{self.file_prefix}_start_position_outcomes.png"
        )
        path = self.output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def _write_trajectory_overlay(self) -> Path:
        all_positions = np.concatenate(self._rendered_trajectories, axis=0)
        figure, axis = plt.subplots(1, 1, figsize=(6, 5))
        self._prepare_axis(axis, positions_xy=all_positions)
        for trajectory in self._rendered_trajectories:
            axis.plot(
                trajectory[:, 0], trajectory[:, 1], color="#4c72b0", linewidth=0.8, alpha=0.25
            )
            axis.scatter(trajectory[0, 0], trajectory[0, 1], color="#2ca02c", s=8, alpha=0.45)
        goal_positions = [
            goal_position_xy
            for goal_position_xy in self._rendered_goals
            if goal_position_xy is not None
        ]
        draw_goal_markers(
            axis, None if not goal_positions else np.asarray(goal_positions), marker_size=110.0
        )
        figure_label = "Training" if self.file_prefix == "training" else "Eval"
        axis.set_title(
            f"{figure_label} trajectories (sampled {len(self._rendered_trajectories)} episodes)"
        )
        figure.tight_layout()
        filename = (
            "trajectories.png"
            if self.file_prefix == "training"
            else f"{self.file_prefix}_trajectories.png"
        )
        path = self.output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    def _render_single_trajectory_frame(
        self,
        *,
        trajectory: np.ndarray,
        headings: np.ndarray | None,
        success: bool,
        goal_position_xy: list[float] | None,
        episode_index: int,
    ) -> np.ndarray:
        figure, axis = plt.subplots(1, 1, figsize=(6, 5))
        self._prepare_axis(axis, positions_xy=trajectory)
        line_color = "#2ca02c" if success else "#d62728"
        axis.plot(trajectory[:, 0], trajectory[:, 1], color=line_color, linewidth=2.0, alpha=0.9)
        axis.scatter(trajectory[0, 0], trajectory[0, 1], color="#2ca02c", s=42, label="start")
        axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color="#d62728", s=42, label="final")
        self._draw_heading_arrow(
            axis, position_xy=trajectory[-1], heading=_heading_at(headings, len(trajectory) - 1)
        )
        draw_goal_markers(
            axis,
            None if goal_position_xy is None else np.asarray(goal_position_xy, dtype=np.float32),
            marker_size=130.0,
        )
        status = "success" if success else "failure"
        axis.set_title(
            f"Eval trajectory {episode_index + 1}/{len(self._rendered_trajectories)} | {status}"
        )
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        frame = figure_to_rgb_array(figure)
        plt.close(figure)
        return frame

    def _write_trajectory_gif(self) -> Path:
        frames = [
            self._render_single_trajectory_frame(
                trajectory=trajectory,
                headings=self._rendered_headings[index],
                success=self._rendered_successes[index],
                goal_position_xy=self._rendered_goals[index],
                episode_index=index,
            )
            for index, trajectory in enumerate(self._rendered_trajectories)
        ]
        path = self.output_dir / f"{self.file_prefix}_trajectory_gallery.gif"
        return save_gif(path, frames, duration=0.9)

    def _render_individual_trajectory_frame(
        self,
        *,
        trajectory: np.ndarray,
        headings: np.ndarray | None,
        success: bool,
        goal_position_xy: list[float] | None,
        trajectory_index: int,
        frame_index: int,
    ) -> np.ndarray:
        figure, axis = plt.subplots(1, 1, figsize=(6, 5))
        self._prepare_axis(axis, positions_xy=trajectory)
        partial_trajectory = trajectory[: frame_index + 1]
        line_color = "#2ca02c" if success else "#d62728"
        axis.plot(
            partial_trajectory[:, 0],
            partial_trajectory[:, 1],
            color=line_color,
            linewidth=2.0,
            alpha=0.9,
        )
        axis.scatter(trajectory[0, 0], trajectory[0, 1], color="#2ca02c", s=42, label="start")
        axis.scatter(
            partial_trajectory[-1, 0],
            partial_trajectory[-1, 1],
            color="#1f77b4",
            s=38,
            label="current",
        )
        self._draw_heading_arrow(
            axis,
            position_xy=partial_trajectory[-1],
            heading=_heading_at(headings, frame_index),
        )
        if frame_index == trajectory.shape[0] - 1:
            axis.scatter(trajectory[-1, 0], trajectory[-1, 1], color="#d62728", s=42, label="final")
        draw_goal_markers(
            axis,
            None if goal_position_xy is None else np.asarray(goal_position_xy, dtype=np.float32),
            marker_size=130.0,
        )
        status = "success" if success else "failure"
        axis.set_title(f"Eval trajectory {trajectory_index + 1} | {status}")
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        frame = figure_to_rgb_array(figure)
        plt.close(figure)
        return frame

    def _trajectory_frame_indices(self, trajectory: np.ndarray) -> list[int]:
        frame_count = min(int(trajectory.shape[0]), self.individual_trajectory_gif_max_frames)
        if frame_count <= 1:
            return [0]
        return sorted(
            {int(index) for index in np.linspace(0, trajectory.shape[0] - 1, frame_count)}
        )

    def _write_individual_trajectory_gifs(self) -> list[Path]:
        paths: list[Path] = []
        trajectory_count = min(
            self.individual_trajectory_gif_count, len(self._rendered_trajectories)
        )
        for trajectory_index in range(trajectory_count):
            trajectory = self._rendered_trajectories[trajectory_index]
            frames = [
                self._render_individual_trajectory_frame(
                    trajectory=trajectory,
                    headings=self._rendered_headings[trajectory_index],
                    success=self._rendered_successes[trajectory_index],
                    goal_position_xy=self._rendered_goals[trajectory_index],
                    trajectory_index=trajectory_index,
                    frame_index=frame_index,
                )
                for frame_index in self._trajectory_frame_indices(trajectory)
            ]
            path = self.output_dir / f"{self.file_prefix}_trajectory_{trajectory_index + 1:02d}.gif"
            paths.append(save_gif(path, frames, duration=0.08))
        return paths


class RGBInputRecorder:
    def __init__(
        self,
        *,
        output_dir: Path,
        file_prefix: str = "eval_rgb_input",
        max_recorded_episodes: int = 5,
        max_frames_per_gif: int = 96,
    ) -> None:
        self.output_dir = output_dir
        self.file_prefix = str(file_prefix)
        self.max_recorded_episodes = max(1, int(max_recorded_episodes))
        self.max_frames_per_gif = max(2, int(max_frames_per_gif))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_by_episode: dict[int, list[np.ndarray]] = {}
        self._completed_episode_frames: list[list[np.ndarray]] = []

    def observe_step(
        self,
        *,
        infos: list[dict[str, Any]],
        dones: np.ndarray,
        episode_indices: np.ndarray,
    ) -> None:
        if len(self._completed_episode_frames) >= self.max_recorded_episodes:
            return
        done_flags = np.asarray(dones, dtype=bool).reshape(-1)
        for env_index, info in enumerate(infos):
            if len(self._completed_episode_frames) >= self.max_recorded_episodes:
                return
            if env_index >= len(episode_indices):
                continue
            episode_index = int(episode_indices[env_index])
            frame = _rgb_frame_from_info(info)
            if frame is not None:
                self._frames_by_episode.setdefault(episode_index, []).append(frame)
            if env_index < len(done_flags) and bool(done_flags[env_index]):
                frames = self._frames_by_episode.pop(episode_index, [])
                if frames:
                    self._completed_episode_frames.append(frames)

    def observe_observation(
        self,
        *,
        observation: Any,
        episode_indices: np.ndarray,
    ) -> None:
        if len(self._completed_episode_frames) >= self.max_recorded_episodes:
            return
        frames = _rgb_frames_from_observation(observation)
        if frames is None:
            return
        for env_index, frame in enumerate(frames):
            if len(self._completed_episode_frames) >= self.max_recorded_episodes:
                return
            if env_index >= len(episode_indices):
                continue
            episode_index = int(episode_indices[env_index])
            self._frames_by_episode.setdefault(episode_index, []).append(frame)

    def close(self) -> list[Path]:
        paths: list[Path] = []
        for episode_index, frames in enumerate(
            self._completed_episode_frames[: self.max_recorded_episodes],
            start=1,
        ):
            path = self.output_dir / f"{self.file_prefix}_{episode_index:02d}.gif"
            paths.append(
                save_gif(path, _sample_rgb_frames(frames, self.max_frames_per_gif), duration=0.08)
            )
        return paths


def _rgb_frame_from_info(info: dict[str, Any]) -> np.ndarray | None:
    raw_frame = info.get("rgb_frame")
    if raw_frame is None:
        return None
    return _rgb_frame_to_hwc(raw_frame, copy=True)


def _rgb_frames_from_observation(observation: Any) -> list[np.ndarray] | None:
    array = np.asarray(observation)
    if array.ndim == 3:
        frame = _rgb_frame_to_hwc(array, copy=True)
        return None if frame is None else [frame]
    if array.ndim != 4:
        return None
    frames = [_rgb_frame_to_hwc(array[env_index], copy=True) for env_index in range(array.shape[0])]
    if any(frame is None for frame in frames):
        return None
    return [frame for frame in frames if frame is not None]


def _rgb_frame_to_hwc(raw_frame: Any, *, copy: bool) -> np.ndarray | None:
    frame = np.asarray(raw_frame)
    if frame.ndim != 3:
        return None
    if frame.shape[0] in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] != 3:
        return None
    return frame.astype(np.uint8, copy=copy)


def _sample_rgb_frames(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
    if len(frames) <= int(max_frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, int(max_frames), dtype=np.int64)
    return [frames[int(index)] for index in indices]


def _spatial_bin_edges(
    env_id: str,
    positions_xy: np.ndarray,
    *,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    world_overlay = resolve_world_overlay(env_id)
    bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
    if bounds is None:
        x_values = positions_xy[:, 0]
        y_values = positions_xy[:, 1]
        x_span = max(1.0, float(np.max(x_values) - np.min(x_values)))
        y_span = max(1.0, float(np.max(y_values) - np.min(y_values)))
        bounds = (
            (float(np.min(x_values) - 0.05 * x_span), float(np.max(x_values) + 0.05 * x_span)),
            (float(np.min(y_values) - 0.05 * y_span), float(np.max(y_values) + 0.05 * y_span)),
        )
    x_bounds, y_bounds = bounds
    return (
        np.linspace(float(x_bounds[0]), float(x_bounds[1]), int(num_bins) + 1),
        np.linspace(float(y_bounds[0]), float(y_bounds[1]), int(num_bins) + 1),
    )
