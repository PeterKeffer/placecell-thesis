"""Visual previews for downstream RL rollouts and training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config.downstream_schema import (
    DownstreamModelConfig,
    DownstreamPreviewConfig,
)
from placecell_research.downstream.feature_sources import AELatentFeatureSource
from placecell_research.downstream.runtime import resolve_downstream_model_artifacts

from ..analysis.figures import figure_to_rgb_array, save_gif
from ..analysis.world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    resolve_world_overlay,
)
from ..collection.previews import draw_goal_markers, overlay_heading_indicator


@dataclass(slots=True)
class PreviewEpisodeRecord:
    episode_index: int
    total_return: float
    success: bool
    positions_xy: np.ndarray
    goal_position_xy: np.ndarray | None
    rgb_frames: list[np.ndarray]
    reconstruction_frames: list[np.ndarray]
    frame_indices: np.ndarray | None = None
    headings: np.ndarray | None = None


@dataclass(slots=True)
class PreviewArtifactBundle:
    combined_gif_path: Path
    rgb_gif_path: Path | None
    reconstruction_gif_path: Path | None
    trajectory_summary_path: Path | None


class PreviewScheduler:
    """Episode-count trigger for periodic downstream previews."""

    def __init__(self, every_n_episodes: int) -> None:
        self.every_n_episodes = max(1, int(every_n_episodes))
        self.next_trigger_episode = self.every_n_episodes

    def should_capture(self, completed_episodes: int) -> bool:
        if completed_episodes < self.next_trigger_episode:
            return False
        while self.next_trigger_episode <= completed_episodes:
            self.next_trigger_episode += self.every_n_episodes
        return True


def build_reconstruction_source(
    *,
    artifact_registry: ArtifactRegistry,
    models: DownstreamModelConfig,
    device: str,
) -> AELatentFeatureSource | None:
    resolved_artifacts = resolve_downstream_model_artifacts(
        artifact_registry=artifact_registry,
        models=models,
        require_vision_encoder=False,
    )
    if resolved_artifacts.vision_encoder is None:
        return None
    return AELatentFeatureSource(
        vision_encoder_path=resolved_artifacts.vision_encoder.path, device=device
    )


def _pause_frames(frames: list[np.ndarray], count: int = 3) -> list[np.ndarray]:
    if not frames:
        return []
    return [frames[-1]] * max(0, int(count))


def _render_trajectory_axis(
    axis: plt.Axes,
    *,
    env_id: str,
    positions_xy: np.ndarray,
    headings: np.ndarray | None,
    goal_position_xy: np.ndarray | None,
    upto_index: int | None = None,
    title: str,
) -> None:
    world_overlay = resolve_world_overlay(env_id)
    if world_overlay is not None:
        draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#404040")
        draw_landmarks_on_axis(axis, world_overlay)
    if upto_index is None:
        upto_index = len(positions_xy) - 1
    clipped_positions = positions_xy[: upto_index + 1]
    axis.plot(
        clipped_positions[:, 0], clipped_positions[:, 1], color="#4c72b0", linewidth=2.0, alpha=0.9
    )
    axis.scatter(
        clipped_positions[0, 0], clipped_positions[0, 1], color="#2ca02c", s=38, label="start"
    )
    axis.scatter(
        clipped_positions[-1, 0], clipped_positions[-1, 1], color="#d62728", s=42, label="current"
    )
    if headings is not None and upto_index < len(headings):
        heading_value = float(headings[upto_index])
        span = np.ptp(positions_xy, axis=0)
        arrow_length = max(float(np.max(span)) * 0.06, 0.2)
        axis.arrow(
            clipped_positions[-1, 0],
            clipped_positions[-1, 1],
            np.cos(heading_value) * arrow_length,
            np.sin(heading_value) * arrow_length,
            color="#f28e2b",
            width=max(arrow_length * 0.04, 0.01),
            head_width=max(arrow_length * 0.18, 0.08),
            head_length=max(arrow_length * 0.18, 0.08),
            length_includes_head=True,
            zorder=5,
        )
    draw_goal_markers(axis, goal_position_xy, marker_size=150.0)
    current_xy = clipped_positions[-1]
    annotation_lines = [f"current=({float(current_xy[0]):.2f}, {float(current_xy[1]):.2f})"]
    if goal_position_xy is not None:
        goal_xy = np.asarray(goal_position_xy, dtype=np.float32)
        goal_distance = float(np.linalg.norm(current_xy - goal_xy))
        annotation_lines.append(f"goal=({float(goal_xy[0]):.2f}, {float(goal_xy[1]):.2f})")
        annotation_lines.append(f"distance={goal_distance:.2f}")
    if headings is not None and upto_index < len(headings):
        annotation_lines.append(f"heading={float(headings[upto_index]):.2f} rad")
    apply_plot_bounds(axis, env_id=env_id, position_xy=positions_xy)
    axis.set_title(title)
    axis.set_xlabel(POSITION_X_LABEL)
    axis.set_ylabel(POSITION_Y_LABEL)
    axis.grid(color="#d6d6d6", linewidth=0.35, alpha=0.5)
    axis.legend(loc="best", fontsize=8)
    axis.text(
        0.02,
        0.02,
        "\n".join(annotation_lines),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )


def _render_combined_preview_frame(
    *,
    env_id: str,
    episode_label: str,
    rgb_frame: np.ndarray,
    reconstruction_frame: np.ndarray | None,
    positions_xy: np.ndarray,
    headings: np.ndarray | None,
    goal_position_xy: np.ndarray | None,
    timestep_index: int,
    total_return: float,
    success: bool,
) -> np.ndarray:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))

    current_heading = None
    if headings is not None and timestep_index < len(headings):
        current_heading = float(headings[timestep_index])
    axes[0].imshow(overlay_heading_indicator(rgb_frame, current_heading))
    axes[0].set_title("Environment RGB")
    axes[0].axis("off")

    if reconstruction_frame is None:
        axes[1].text(
            0.5, 0.5, "AE reconstruction\nnot available", ha="center", va="center", fontsize=12
        )
        axes[1].set_title("AE Decode")
        axes[1].set_axis_off()
    else:
        axes[1].imshow(overlay_heading_indicator(reconstruction_frame, current_heading))
        axes[1].set_title("Decoded AE Latent")
        axes[1].axis("off")

    _render_trajectory_axis(
        axes[2],
        env_id=env_id,
        positions_xy=positions_xy,
        headings=headings,
        goal_position_xy=goal_position_xy,
        upto_index=timestep_index,
        title=f"Trajectory step {timestep_index + 1}/{len(positions_xy)}",
    )

    status_text = "success" if success else "incomplete"
    current_xy = np.asarray(
        positions_xy[min(timestep_index, len(positions_xy) - 1)], dtype=np.float32
    )
    goal_summary = ""
    if goal_position_xy is not None:
        goal_xy = np.asarray(goal_position_xy, dtype=np.float32)
        goal_distance = float(np.linalg.norm(current_xy - goal_xy))
        goal_summary = (
            f" | current=({float(current_xy[0]):.2f}, {float(current_xy[1]):.2f})"
            f" | goal=({float(goal_xy[0]):.2f}, {float(goal_xy[1]):.2f})"
            f" | distance={goal_distance:.2f}"
        )
    figure.suptitle(
        f"{episode_label} | return={total_return:.2f} | {status_text}{goal_summary}",
        fontsize=12,
    )
    figure.tight_layout()
    return figure_to_rgb_array(figure)


def _write_trajectory_summary(
    *,
    env_id: str,
    episodes: list[PreviewEpisodeRecord],
    path: Path,
) -> Path:
    columns = max(1, len(episodes))
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 4), squeeze=False)
    for axis, episode in zip(axes[0], episodes, strict=False):
        _render_trajectory_axis(
            axis,
            env_id=env_id,
            positions_xy=episode.positions_xy,
            headings=episode.headings,
            goal_position_xy=episode.goal_position_xy,
            title=f"Episode {episode.episode_index} | return={episode.total_return:.2f}",
        )
        axis.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"steps={len(episode.positions_xy) - 1}",
                    f"success={episode.success}",
                    f"final=({float(episode.positions_xy[-1, 0]):.2f}, "
                    f"{float(episode.positions_xy[-1, 1]):.2f})",
                    (
                        "goal=(none)"
                        if episode.goal_position_xy is None
                        else (
                            f"goal=({float(episode.goal_position_xy[0]):.2f}, "
                            f"{float(episode.goal_position_xy[1]):.2f})"
                        )
                    ),
                    (
                        ""
                        if episode.goal_position_xy is None
                        else "distance="
                        + format(
                            float(
                                np.linalg.norm(episode.positions_xy[-1] - episode.goal_position_xy)
                            ),
                            ".2f",
                        )
                    ),
                ]
            ).strip(),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _capture_preview_episode(
    *,
    env,
    action_fn: Callable[[Any], int],
    episode_index: int,
    reset_seed: int,
    max_steps_per_episode: int,
    max_animation_frames: int,
    reconstruction_source: AELatentFeatureSource | None,
) -> PreviewEpisodeRecord:
    observation, info = env.reset(seed=reset_seed)
    total_return = 0.0
    positions_xy = [np.asarray(info["position_xy"], dtype=np.float32)]
    goal_position_xy = info.get("goal_position_xy")
    goal_xy_array = (
        None if goal_position_xy is None else np.asarray(goal_position_xy, dtype=np.float32)
    )
    headings = [float(info.get("heading", 0.0))]
    rgb_frames: list[np.ndarray] = []
    reconstruction_frames = []
    frame_indices: list[int] = []
    frame_stride = max(
        1, int(np.ceil((int(max_steps_per_episode) + 1) / int(max_animation_frames)))
    )

    def capture_animation_frame(frame_index: int) -> None:
        rgb_frame = env.current_rgb_frame
        rgb_frames.append(rgb_frame)
        frame_indices.append(int(frame_index))
        if reconstruction_source is not None:
            reconstruction = reconstruction_source.reconstruct_rgb(rgb_frame)
            if reconstruction is not None:
                reconstruction_frames.append(reconstruction)

    capture_animation_frame(0)

    terminated = False
    truncated = False
    final_info = dict(info)
    steps = 0
    while not terminated and not truncated and steps < max_steps_per_episode:
        action = int(action_fn(observation))
        observation, reward, terminated, truncated, final_info = env.step(action)
        total_return += float(reward)
        positions_xy.append(np.asarray(final_info["position_xy"], dtype=np.float32))
        headings.append(float(final_info.get("heading", headings[-1])))
        steps += 1
        if steps % frame_stride == 0:
            capture_animation_frame(steps)

    if frame_indices[-1] != steps:
        capture_animation_frame(steps)

    success = bool(final_info.get("is_success", total_return > 0.0))
    return PreviewEpisodeRecord(
        episode_index=episode_index,
        total_return=total_return,
        success=success,
        positions_xy=np.asarray(positions_xy, dtype=np.float32),
        goal_position_xy=goal_xy_array,
        rgb_frames=rgb_frames,
        reconstruction_frames=reconstruction_frames,
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        headings=np.asarray(headings, dtype=np.float32),
    )


def write_episode_previews(
    *,
    env_id: str,
    preview_name: str,
    episodes: list[PreviewEpisodeRecord],
    output_dir: Path,
    preview_config: DownstreamPreviewConfig,
) -> PreviewArtifactBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    reconstruction_frames: list[np.ndarray] = []
    has_reconstructions = any(episode.reconstruction_frames for episode in episodes)

    for episode in episodes:
        max_frame_count = len(episode.rgb_frames)
        for frame_index in range(max_frame_count):
            timestep_index = frame_index
            if episode.frame_indices is not None and frame_index < len(episode.frame_indices):
                timestep_index = int(episode.frame_indices[frame_index])
            rgb_frame = episode.rgb_frames[frame_index]
            heading_value = None
            if episode.headings is not None and timestep_index < len(episode.headings):
                heading_value = float(episode.headings[timestep_index])
            reconstruction_frame = None
            if frame_index < len(episode.reconstruction_frames):
                reconstruction_frame = episode.reconstruction_frames[frame_index]
                reconstruction_frames.append(
                    overlay_heading_indicator(reconstruction_frame, heading_value)
                )
            rgb_frames.append(overlay_heading_indicator(rgb_frame, heading_value))
            combined_frames.append(
                _render_combined_preview_frame(
                    env_id=env_id,
                    episode_label=f"Episode {episode.episode_index}",
                    rgb_frame=rgb_frame,
                    reconstruction_frame=reconstruction_frame,
                    positions_xy=episode.positions_xy,
                    headings=episode.headings,
                    goal_position_xy=episode.goal_position_xy,
                    timestep_index=min(timestep_index, len(episode.positions_xy) - 1),
                    total_return=episode.total_return,
                    success=episode.success,
                )
            )
        combined_frames.extend(_pause_frames(combined_frames))
        rgb_frames.extend(_pause_frames(rgb_frames))
        if reconstruction_frames:
            reconstruction_frames.extend(_pause_frames(reconstruction_frames))

    combined_gif_path = save_gif(
        output_dir / f"policy_preview__{preview_name}.gif",
        combined_frames,
        duration=float(preview_config.gif_frame_duration),
    )
    rgb_gif_path = None
    if preview_config.save_rgb_gif:
        rgb_gif_path = save_gif(
            output_dir / f"policy_preview_rgb__{preview_name}.gif",
            rgb_frames,
            duration=float(preview_config.gif_frame_duration),
        )
    reconstruction_gif_path = None
    if preview_config.save_ae_reconstruction_gif and has_reconstructions and reconstruction_frames:
        reconstruction_gif_path = save_gif(
            output_dir / f"policy_preview_ae_reconstruction__{preview_name}.gif",
            reconstruction_frames,
            duration=float(preview_config.gif_frame_duration),
        )
    trajectory_summary_path = None
    if preview_config.save_trajectory_summary:
        trajectory_summary_path = _write_trajectory_summary(
            env_id=env_id,
            episodes=episodes,
            path=output_dir / f"policy_preview_trajectory_summary__{preview_name}.png",
        )
    return PreviewArtifactBundle(
        combined_gif_path=combined_gif_path,
        rgb_gif_path=rgb_gif_path,
        reconstruction_gif_path=reconstruction_gif_path,
        trajectory_summary_path=trajectory_summary_path,
    )


def capture_policy_preview(
    *,
    env_factory: Callable[[], Any],
    action_fn: Callable[[Any], int],
    env_id: str,
    preview_name: str,
    output_dir: Path,
    preview_config: DownstreamPreviewConfig,
    seed: int,
    reconstruction_source: AELatentFeatureSource | None,
) -> PreviewArtifactBundle:
    env = env_factory()
    try:
        episodes = [
            _capture_preview_episode(
                env=env,
                action_fn=action_fn,
                episode_index=index,
                reset_seed=int(seed) + index,
                max_steps_per_episode=int(preview_config.max_steps_per_episode),
                max_animation_frames=int(preview_config.max_animation_frames),
                reconstruction_source=reconstruction_source,
            )
            for index in range(int(preview_config.episodes_per_preview))
        ]
    finally:
        env.close()
    return write_episode_previews(
        env_id=env_id,
        preview_name=preview_name,
        episodes=episodes,
        output_dir=output_dir,
        preview_config=preview_config,
    )
