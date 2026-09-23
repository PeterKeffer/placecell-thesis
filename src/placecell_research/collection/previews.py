"""Preview generation for collection and vision stages."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from ..figure_style import despine

DEFAULT_PREVIEW_GIF_FRAME_DURATION_SECONDS = 0.12


def _to_display_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[0] in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=2)
    if np.issubdtype(array.dtype, np.floating):
        if float(np.nanmax(array)) <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
    return np.asarray(array, dtype=np.uint8)


def _blend_color(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> np.ndarray:
    if not np.any(mask):
        return frame
    output = frame.astype(np.float32, copy=True)
    color_array = np.asarray(color, dtype=np.float32)
    output[mask] = (1.0 - alpha) * output[mask] + alpha * color_array
    return np.asarray(np.clip(output, 0.0, 255.0), dtype=np.uint8)


def _paint_line(
    frame: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    alpha: float = 1.0,
) -> np.ndarray:
    x0, y0 = start_xy
    x1, y1 = end_xy
    sample_count = max(int(np.hypot(x1 - x0, y1 - y0)) * 2, 2)
    xs = np.linspace(x0, x1, sample_count)
    ys = np.linspace(y0, y1, sample_count)
    output = frame
    radius = max(1, int(thickness))
    for x_coord, y_coord in zip(xs, ys, strict=False):
        x_index = int(round(x_coord))
        y_index = int(round(y_coord))
        x_start = max(0, x_index - radius)
        x_end = min(frame.shape[1], x_index + radius + 1)
        y_start = max(0, y_index - radius)
        y_end = min(frame.shape[0], y_index + radius + 1)
        if x_start >= x_end or y_start >= y_end:
            continue
        yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
        local_mask = (xx - x_coord) ** 2 + (yy - y_coord) ** 2 <= radius**2
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[y_start:y_end, x_start:x_end] = local_mask
        output = _blend_color(output, mask, color, alpha=alpha)
    return output


def _normalize_goal_positions_xy(
    goal_positions_xy: np.ndarray | list[list[float]] | list[float] | None,
) -> np.ndarray | None:
    if goal_positions_xy is None:
        return None
    goal_array = np.asarray(goal_positions_xy, dtype=np.float32)
    if goal_array.size == 0:
        return None
    if goal_array.ndim == 1:
        if goal_array.shape[0] != 2:
            raise ValueError("Goal positions must contain x/y pairs.")
        goal_array = goal_array.reshape(1, 2)
    if goal_array.ndim != 2 or goal_array.shape[1] != 2:
        raise ValueError("Goal positions must have shape (n, 2).")
    return goal_array


def draw_goal_markers(
    axis: plt.Axes,
    goal_positions_xy: np.ndarray | list[list[float]] | list[float] | None,
    *,
    marker_size: float = 150.0,
) -> None:
    normalized_goal_positions = _normalize_goal_positions_xy(goal_positions_xy)
    if normalized_goal_positions is None:
        return
    label_count = len(normalized_goal_positions)
    for goal_index, goal_xy in enumerate(normalized_goal_positions):
        goal_x = float(goal_xy[0])
        goal_y = float(goal_xy[1])
        goal_label = "goal" if label_count == 1 else f"goal {goal_index + 1}"
        axis.scatter(
            goal_x,
            goal_y,
            color="#ff8c00",
            edgecolors="white",
            linewidths=1.4,
            marker="*",
            s=marker_size,
            zorder=7,
            label=goal_label if goal_index == 0 else None,
        )
        axis.annotate(
            goal_label,
            xy=(goal_x, goal_y),
            xytext=(6, 6),
            textcoords="offset points",
            color="#7a3e00",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#ffcf99", "pad": 0.25},
            zorder=8,
        )


def overlay_heading_indicator(frame: np.ndarray, heading: float | None) -> np.ndarray:
    """Overlay a compact head-direction compass in the top-left corner."""
    display_frame = _to_display_frame(frame)
    if heading is None or not np.isfinite(float(heading)):
        return display_frame

    height, width = display_frame.shape[:2]
    radius = int(np.clip(min(height, width) * 0.08, 8, 18))
    margin = max(4, radius // 2)
    center_x = margin + radius + 2
    center_y = margin + radius + 2
    box_half_extent = radius + 6
    x_start = max(0, center_x - box_half_extent)
    x_end = min(width, center_x + box_half_extent)
    y_start = max(0, center_y - box_half_extent)
    y_end = min(height, center_y + box_half_extent)

    output = display_frame.astype(np.float32, copy=True)
    output[y_start:y_end, x_start:x_end] = 0.4 * output[
        y_start:y_end, x_start:x_end
    ] + 0.6 * np.asarray([24.0, 24.0, 24.0], dtype=np.float32)
    output = np.asarray(np.clip(output, 0.0, 255.0), dtype=np.uint8)

    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    ring_mask = np.abs(distance - radius) <= 1.5
    center_mask = distance <= max(2, radius // 5)
    output = _blend_color(output, ring_mask, (255, 255, 255), alpha=0.95)
    output = _blend_color(output, center_mask, (255, 255, 255), alpha=0.95)

    heading_value = float(heading)
    arrow_length = radius - 2
    end_x = center_x + np.cos(heading_value) * arrow_length
    end_y = center_y - np.sin(heading_value) * arrow_length
    output = _paint_line(
        output,
        (center_x, center_y),
        (end_x, end_y),
        (255, 99, 71),
        thickness=max(2, radius // 4),
        alpha=0.95,
    )
    arrow_head_length = max(4, radius // 3)
    head_angle = np.pi / 7
    left_x = end_x - np.cos(heading_value - head_angle) * arrow_head_length
    left_y = end_y + np.sin(heading_value - head_angle) * arrow_head_length
    right_x = end_x - np.cos(heading_value + head_angle) * arrow_head_length
    right_y = end_y + np.sin(heading_value + head_angle) * arrow_head_length
    output = _paint_line(
        output,
        (end_x, end_y),
        (left_x, left_y),
        (255, 99, 71),
        thickness=max(2, radius // 5),
        alpha=0.95,
    )
    output = _paint_line(
        output,
        (end_x, end_y),
        (right_x, right_y),
        (255, 99, 71),
        thickness=max(2, radius // 5),
        alpha=0.95,
    )
    return output


def write_sample_frames(frames: np.ndarray, output_path: Path, max_frames: int = 16) -> None:
    """Write a deterministic grid of sampled frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = min(max_frames, frames.shape[0])
    frame_indices = np.linspace(0, frames.shape[0] - 1, total, dtype=int)
    selected = frames[frame_indices]
    columns = int(np.ceil(np.sqrt(total)))
    rows = int(np.ceil(total / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    axes_array = np.asarray(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")
    for axis, frame in zip(axes_array.flat, selected, strict=False):
        axis.imshow(_to_display_frame(frame))
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_trajectory_gif(
    frames: np.ndarray,
    output_path: Path,
    max_frames: int = 64,
    *,
    headings: np.ndarray | None = None,
) -> None:
    """Write a compact preview GIF for a trajectory slice."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frames.shape[0] > max_frames:
        frame_indices = np.linspace(0, frames.shape[0] - 1, max_frames, dtype=int)
        frames = frames[frame_indices]
        if headings is not None:
            headings = np.asarray(headings)[frame_indices]
    rgb_frames = [
        overlay_heading_indicator(
            frame,
            None if headings is None else float(headings[index]),
        )
        for index, frame in enumerate(frames)
    ]
    imageio.mimsave(output_path, rgb_frames, duration=DEFAULT_PREVIEW_GIF_FRAME_DURATION_SECONDS)


def write_reconstruction_grid(
    source_frames: np.ndarray,
    reconstructed_frames: np.ndarray,
    output_path: Path,
    max_frames: int = 4,
) -> None:
    """Write an input-vs-reconstruction grid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    available_frames = min(int(source_frames.shape[0]), int(reconstructed_frames.shape[0]))
    total = min(max_frames, available_frames)
    if total <= 0:
        raise ValueError("Reconstruction grid requires at least one frame.")
    frame_indices = np.linspace(0, available_frames - 1, total, dtype=int)
    figure, axes = plt.subplots(2, total, figsize=(2.4 * total, 4.8))
    axes_array = np.asarray(axes).reshape(2, total)
    for column, frame_index in enumerate(frame_indices):
        axes_array[0, column].imshow(_to_display_frame(source_frames[frame_index]))
        axes_array[0, column].set_title("Input")
        axes_array[0, column].axis("off")
        axes_array[1, column].imshow(_to_display_frame(reconstructed_frames[frame_index]))
        axes_array[1, column].set_title("Reconstruction")
        axes_array[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_side_by_side_reconstruction_gif(
    source_frames: np.ndarray,
    reconstructed_frames: np.ndarray,
    output_path: Path,
    max_frames: int = 64,
    separator_width: int = 4,
    *,
    headings: np.ndarray | None = None,
) -> None:
    """Write a side-by-side input/reconstruction preview GIF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = min(int(source_frames.shape[0]), int(reconstructed_frames.shape[0]))
    if total_frames <= 0:
        raise ValueError("Reconstruction GIF requires at least one frame.")
    frame_indices = np.arange(total_frames, dtype=int)
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    if headings is not None:
        headings = np.asarray(headings)[frame_indices]
    rendered_frames: list[np.ndarray] = []
    separator: np.ndarray | None = None
    for rendered_index, frame_index in enumerate(frame_indices):
        source_frame = _to_display_frame(source_frames[frame_index])
        reconstructed_frame = _to_display_frame(reconstructed_frames[frame_index])
        if separator is None or separator.shape[0] != source_frame.shape[0]:
            separator = np.full((source_frame.shape[0], separator_width, 3), 255, dtype=np.uint8)
        combined_frame = np.concatenate([source_frame, separator, reconstructed_frame], axis=1)
        rendered_frames.append(
            overlay_heading_indicator(
                combined_frame,
                None if headings is None else float(headings[rendered_index]),
            )
        )
    imageio.mimsave(
        output_path, rendered_frames, duration=DEFAULT_PREVIEW_GIF_FRAME_DURATION_SECONDS
    )


def write_position_density_map(
    positions: np.ndarray,
    output_path: Path,
    topdown_frame: np.ndarray | None = None,
    *,
    goal_positions_xy: np.ndarray | list[list[float]] | list[float] | None = None,
) -> None:
    """Render trajectory occupancy density in world coordinates."""
    del topdown_frame
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(positions[:, 0], positions[:, 1], s=2, alpha=0.25)
    draw_goal_markers(axis, goal_positions_xy, marker_size=180.0)
    axis.set_aspect("equal")
    axis.set_title("Trajectory density")
    axis.set_xlabel("World x")
    axis.set_ylabel("World y")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_kinematics_histogram(kinematics: np.ndarray, output_path: Path) -> None:
    """Write step_displacement and angular velocity histograms."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(kinematics[:, 0], bins=40)
    axes[0].set_title("Step displacement")
    axes[1].hist(kinematics[:, 1], bins=40)
    axes[1].set_title("Angular velocity")
    despine(axes[0])
    despine(axes[1])
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
