"""Transition-graph eigendecomposition and representation alignment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    flatten_positions,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .timing import log_timing, record_timing
from .world_overlay import (
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    overlay_bounds,
    resolve_world_overlay,
    style_arena_axes,
)


def _assign_bins_with_shared_bounds(
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    linear_bins, _x_edges, _y_edges, _resolved_bounds = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    return linear_bins


def _flatten_valid_episode_bins(
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[list[np.ndarray], np.ndarray]:
    num_episodes = position_xy.shape[0]
    episode_bins: list[np.ndarray] = []
    occupancy_counts = np.zeros(num_bins_x * num_bins_y, dtype=np.float64)
    for episode_index in range(num_episodes):
        episode_positions = position_xy[episode_index]
        if valid_mask is not None:
            episode_positions = episode_positions[valid_mask[episode_index]]
        if episode_positions.size == 0:
            episode_bins.append(np.zeros((0,), dtype=np.int32))
            continue
        linear_bins = _assign_bins_with_shared_bounds(
            episode_positions,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
        )
        episode_bins.append(linear_bins)
        occupancy_counts += np.bincount(
            linear_bins,
            minlength=num_bins_x * num_bins_y,
        ).astype(np.float64, copy=False)
    return episode_bins, occupancy_counts


def _build_transition_counts(
    episode_bins: list[np.ndarray],
    *,
    num_bins: int,
    include_self_transitions: bool,
) -> tuple[np.ndarray, int]:
    transition_counts = np.zeros((num_bins, num_bins), dtype=np.float64)
    transitions_used = 0
    for bins in episode_bins:
        if bins.size < 2:
            continue
        from_bins = bins[:-1]
        to_bins = bins[1:]
        if not include_self_transitions:
            nonself_mask = from_bins != to_bins
            from_bins = from_bins[nonself_mask]
            to_bins = to_bins[nonself_mask]
        if from_bins.size == 0:
            continue
        np.add.at(transition_counts, (from_bins, to_bins), 1.0)
        transitions_used += int(from_bins.size)
    return transition_counts, transitions_used


def _masked_correlation(first_map: np.ndarray, second_map: np.ndarray) -> float:
    overlap_mask = np.isfinite(first_map) & np.isfinite(second_map)
    if int(overlap_mask.sum()) < 3:
        return 0.0
    first_values = first_map[overlap_mask].astype(np.float64, copy=False)
    second_values = second_map[overlap_mask].astype(np.float64, copy=False)
    first_values = first_values - float(first_values.mean())
    second_values = second_values - float(second_values.mean())
    first_norm = float(np.linalg.norm(first_values))
    second_norm = float(np.linalg.norm(second_values))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return 0.0
    return float(np.dot(first_values, second_values) / (first_norm * second_norm))


def _orient_mode(mode_vector: np.ndarray) -> np.ndarray:
    pivot_index = int(np.argmax(np.abs(mode_vector)))
    pivot_value = float(mode_vector[pivot_index])
    if pivot_value < 0.0:
        return -mode_vector
    return mode_vector


def _transition_laplacian_modes(
    transition_counts: np.ndarray,
    *,
    occupancy_counts: np.ndarray,
    num_bins_x: int,
    num_bins_y: int,
    num_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active_bin_mask = (
        (occupancy_counts > 0.0)
        & ((transition_counts.sum(axis=0) + transition_counts.sum(axis=1)) > 0.0)
    )
    active_bin_indices = np.flatnonzero(active_bin_mask)
    if active_bin_indices.size < 2:
        raise ValueError(
            "Transition-geometry analysis needs at least two spatial bins connected by transitions."
        )

    active_counts = transition_counts[np.ix_(active_bin_indices, active_bin_indices)]
    outbound = active_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        active_counts,
        outbound,
        out=np.zeros_like(active_counts, dtype=np.float64),
        where=outbound > 0.0,
    )
    symmetric_affinity = 0.5 * (transition_matrix + transition_matrix.T)
    degrees = symmetric_affinity.sum(axis=1)
    connected_mask = degrees > 1e-12
    if int(np.count_nonzero(connected_mask)) < 2:
        raise ValueError(
            "Transition-geometry analysis found too few connected spatial bins after "
            "symmetrization."
        )
    if not np.all(connected_mask):
        symmetric_affinity = symmetric_affinity[np.ix_(connected_mask, connected_mask)]
        active_bin_indices = active_bin_indices[connected_mask]
        degrees = symmetric_affinity.sum(axis=1)

    inverse_sqrt_degrees = np.zeros_like(degrees, dtype=np.float64)
    positive_degree_mask = degrees > 1e-12
    inverse_sqrt_degrees[positive_degree_mask] = 1.0 / np.sqrt(degrees[positive_degree_mask])
    normalized_affinity = (
        inverse_sqrt_degrees[:, None] * symmetric_affinity * inverse_sqrt_degrees[None, :]
    )
    normalized_laplacian = (
        np.eye(normalized_affinity.shape[0], dtype=np.float64) - normalized_affinity
    )
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)

    nontrivial_indices = np.flatnonzero(eigenvalues > 1e-8)
    if nontrivial_indices.size == 0:
        raise ValueError("Transition-geometry analysis could not find any nontrivial graph modes.")
    selected_indices = nontrivial_indices[: max(1, int(num_modes))]
    selected_vectors = np.asarray(
        [
            _orient_mode(eigenvectors[:, index].astype(np.float32, copy=False))
            for index in selected_indices
        ],
        dtype=np.float32,
    )

    mode_maps = np.full(
        (selected_vectors.shape[0], num_bins_y * num_bins_x),
        np.nan,
        dtype=np.float32,
    )
    mode_maps[:, active_bin_indices] = selected_vectors
    return (
        eigenvalues[selected_indices].astype(np.float32, copy=False),
        selected_indices.astype(np.int32, copy=False),
        mode_maps.reshape(selected_vectors.shape[0], num_bins_y, num_bins_x),
    )


def _compute_mode_alignment(
    rate_maps: np.ndarray,
    mode_maps: np.ndarray,
    displayed_mode_ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_units = rate_maps.shape[0]
    num_modes = mode_maps.shape[0]
    signed_correlations = np.zeros((num_units, num_modes), dtype=np.float32)
    for mode_index in range(num_modes):
        mode_map = mode_maps[mode_index]
        for unit_index in range(num_units):
            signed_correlations[unit_index, mode_index] = _masked_correlation(
                rate_maps[unit_index],
                mode_map,
            )
    absolute_correlations = np.abs(signed_correlations)
    if absolute_correlations.shape[1] == 0:
        return (
            np.zeros((num_units,), dtype=np.float32),
            np.zeros((num_units,), dtype=np.float32),
            signed_correlations,
        )
    best_mode_columns = np.argmax(absolute_correlations, axis=1)
    best_mode_abs_correlation = absolute_correlations[
        np.arange(num_units, dtype=np.int32),
        best_mode_columns,
    ].astype(np.float32, copy=False)
    best_mode_rank = displayed_mode_ranks[best_mode_columns].astype(np.float32, copy=False)
    return best_mode_abs_correlation, best_mode_rank, signed_correlations


@dataclass(frozen=True, slots=True)
class _TransitionGeometryCore:
    bounds: tuple[tuple[float, float], tuple[float, float]]
    occupancy_counts: np.ndarray
    transitions_used: int
    selected_eigenvalues: np.ndarray
    selected_mode_indices: np.ndarray
    mode_maps: np.ndarray


@dataclass(frozen=True, slots=True)
class _TransitionGeometryAlignment:
    best_mode_abs_correlation: np.ndarray
    best_mode_rank: np.ndarray
    best_mode_abs_correlation_per_mode: np.ndarray
    top_unit_scores: np.ndarray
    top_k: int


@dataclass(slots=True)
class _TransitionGeometryModuleBase:
    """Inspect graph eigenmodes induced by experienced spatial transitions."""

    name: str = "transition_geometry_bundle"
    cost_tier: str = "standard"
    emit_graph_metrics: bool = True
    emit_alignment_metrics: bool = True
    emit_panel: bool = True

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        num_bins_x = int(config.get("transition_geometry_num_bins_x", 24))
        num_bins_y = int(config.get("transition_geometry_num_bins_y", 24))
        num_modes = int(config.get("transition_geometry_num_modes", 6))
        alignment_top_k = int(config.get("transition_geometry_alignment_top_k", 16))
        include_self_transitions = bool(
            config.get("transition_geometry_include_self_transitions", False)
        )
        valid_positions = flatten_positions(
            analysis_input.position_xy,
            analysis_input.valid_mask,
        )
        if valid_positions.size == 0:
            raise ValueError("Transition-geometry analysis needs at least one valid position.")
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )

        core_cache_key = (
            "transition_geometry_core",
            num_bins_x,
            num_bins_y,
            num_modes,
            include_self_transitions,
            str(analysis_input.metadata.get("env_id", "")),
        )

        def build_core() -> _TransitionGeometryCore:
            section_started_at = perf_counter()
            bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
            if bounds is None:
                _linear_bins, _x_edges, _y_edges, bounds = compute_spatial_bin_assignments(
                    valid_positions,
                    num_bins_x=num_bins_x,
                    num_bins_y=num_bins_y,
                )
            episode_bins, occupancy_counts = _flatten_valid_episode_bins(
                analysis_input.position_xy,
                analysis_input.valid_mask,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                bounds=bounds,
            )
            record_timing(timing_seconds, "prepare_bins", section_started_at)

            section_started_at = perf_counter()
            transition_counts, transitions_used = _build_transition_counts(
                episode_bins,
                num_bins=num_bins_x * num_bins_y,
                include_self_transitions=include_self_transitions,
            )
            if transitions_used <= 0:
                raise ValueError(
                    "Transition-geometry analysis found no valid bin-to-bin transitions. "
                    "Try enabling self-transitions or using coarser transition bins."
                )
            record_timing(timing_seconds, "transition_counts", section_started_at)

            section_started_at = perf_counter()
            selected_eigenvalues, selected_mode_indices, mode_maps = (
                _transition_laplacian_modes(
                    transition_counts,
                    occupancy_counts=occupancy_counts,
                    num_bins_x=num_bins_x,
                    num_bins_y=num_bins_y,
                    num_modes=num_modes,
                )
            )
            record_timing(timing_seconds, "eigendecomposition", section_started_at)
            return _TransitionGeometryCore(
                bounds=bounds,
                occupancy_counts=occupancy_counts,
                transitions_used=transitions_used,
                selected_eigenvalues=selected_eigenvalues,
                selected_mode_indices=selected_mode_indices,
                mode_maps=mode_maps,
            )

        core = analysis_input.get_cached_metric(core_cache_key, build_core)
        bounds = core.bounds
        occupancy_counts = core.occupancy_counts
        transitions_used = core.transitions_used
        selected_eigenvalues = core.selected_eigenvalues
        selected_mode_indices = core.selected_mode_indices
        mode_maps = core.mode_maps

        best_mode_abs_correlation = np.zeros(
            (analysis_input.representation.shape[-1],),
            dtype=np.float32,
        )
        best_mode_rank = np.zeros_like(best_mode_abs_correlation)
        best_mode_abs_correlation_per_mode = np.zeros((mode_maps.shape[0],), dtype=np.float32)
        top_unit_scores = np.zeros((0,), dtype=np.float32)
        top_k = min(max(alignment_top_k, 1), len(best_mode_abs_correlation))
        if self.emit_alignment_metrics or self.emit_panel:
            smoothing_sigma = float(config.get("smoothing_sigma", 0.4))
            min_occupancy = float(config.get("min_occupancy", 1e-6))
            alignment_cache_key = (
                "transition_geometry_alignment",
                core_cache_key,
                alignment_top_k,
                smoothing_sigma,
                min_occupancy,
            )

            def build_alignment() -> _TransitionGeometryAlignment:
                section_started_at = perf_counter()
                rate_map_result = get_or_compute_rate_maps(
                    analysis_input,
                    num_bins_x=num_bins_x,
                    num_bins_y=num_bins_y,
                    smoothing_sigma=smoothing_sigma,
                    min_occupancy=min_occupancy,
                    bounds=bounds,
                )
                record_timing(timing_seconds, "rate_maps", section_started_at)
                section_started_at = perf_counter()
                displayed_mode_ranks = np.arange(
                    1,
                    mode_maps.shape[0] + 1,
                    dtype=np.int32,
                )
                best_correlation, best_rank, signed_correlations = _compute_mode_alignment(
                    rate_map_result.rate_maps,
                    mode_maps,
                    displayed_mode_ranks,
                )
                absolute_correlations = np.abs(signed_correlations)
                per_mode = (
                    absolute_correlations.max(axis=0).astype(np.float32, copy=False)
                    if absolute_correlations.size
                    else np.zeros((0,), dtype=np.float32)
                )
                resolved_top_k = min(max(alignment_top_k, 1), len(best_correlation))
                top_scores = (
                    np.sort(best_correlation)[-resolved_top_k:]
                    if best_correlation.size
                    else np.zeros((0,), dtype=np.float32)
                )
                record_timing(timing_seconds, "mode_alignment", section_started_at)
                return _TransitionGeometryAlignment(
                    best_mode_abs_correlation=best_correlation,
                    best_mode_rank=best_rank,
                    best_mode_abs_correlation_per_mode=per_mode,
                    top_unit_scores=top_scores,
                    top_k=resolved_top_k,
                )

            alignment = analysis_input.get_cached_metric(
                alignment_cache_key,
                build_alignment,
            )
            best_mode_abs_correlation = alignment.best_mode_abs_correlation
            best_mode_rank = alignment.best_mode_rank
            best_mode_abs_correlation_per_mode = (
                alignment.best_mode_abs_correlation_per_mode
            )
            top_unit_scores = alignment.top_unit_scores
            top_k = alignment.top_k

        figures: dict[str, Path] = {}
        if self.emit_panel:
            section_started_at = perf_counter()
            module_dir = output_dir / self.name
            figure_path = (
                module_dir
                / (
                    f"transition_geometry__{analysis_input.source_name}"
                    f"__{analysis_input.split_name}.png"
                )
            )
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            columns = 2 if int(mode_maps.shape[0]) == 1 else min(3, int(mode_maps.shape[0]))
            mode_rows = int(ceil(mode_maps.shape[0] / columns))
            figure = plt.figure(figsize=(4.8 * columns, 3.6 * (mode_rows + 1)))
            grid_spec = figure.add_gridspec(
                mode_rows + 1,
                columns,
                height_ratios=[1.0] + [1.15] * mode_rows,
            )
            spectrum_axis = figure.add_subplot(grid_spec[0, : max(1, columns - 1)])
            histogram_axis = figure.add_subplot(grid_spec[0, columns - 1])
            spectrum_indices = np.arange(1, len(selected_eigenvalues) + 1, dtype=np.int32)
            spectrum_axis.plot(
                spectrum_indices,
                selected_eigenvalues,
                color="#1F77B4",
                linewidth=2.0,
                marker="o",
                markersize=4.0,
            )
            spectrum_axis.set_title("Selected Nontrivial Graph Eigenvalues")
            spectrum_axis.set_xlabel("Mode rank")
            spectrum_axis.set_ylabel("Normalized Laplacian eigenvalue")
            spectrum_axis.grid(alpha=0.3)

            if best_mode_abs_correlation.size > 0:
                histogram_axis.hist(
                    best_mode_abs_correlation,
                    bins=min(24, max(6, len(best_mode_abs_correlation) // 4)),
                    color="#C44E52",
                    alpha=0.85,
                )
            histogram_axis.set_title("Best Unit-to-Mode |corr|")
            histogram_axis.set_xlabel("absolute correlation")
            histogram_axis.set_ylabel("unit count")
            histogram_axis.grid(alpha=0.3)

            for mode_offset, mode_map in enumerate(mode_maps):
                row_index = 1 + mode_offset // columns
                column_index = mode_offset % columns
                axis = figure.add_subplot(grid_spec[row_index, column_index])
                finite_mask = np.isfinite(mode_map)
                finite_values = mode_map[finite_mask]
                max_abs_value = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
                masked_mode_map = np.ma.masked_where(~finite_mask, mode_map)
                x_bounds, y_bounds = bounds
                axis.imshow(
                    masked_mode_map,
                    origin="lower",
                    cmap="coolwarm",
                    vmin=-max(max_abs_value, 1e-6),
                    vmax=max(max_abs_value, 1e-6),
                    extent=(x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]),
                    aspect="equal",
                )
                if world_overlay is not None:
                    draw_world_segments_on_axis(axis, world_overlay.segments, line_color="#202020")
                    draw_landmarks_on_axis(axis, world_overlay)
                apply_plot_bounds(
                    axis,
                    env_id=str(analysis_input.metadata.get("env_id", "")),
                    position_xy=valid_positions,
                )
                axis.set_title(
                    f"Mode {mode_offset + 1} (lambda={selected_eigenvalues[mode_offset]:.3f})\n"
                    f"best |corr|={best_mode_abs_correlation_per_mode[mode_offset]:.3f}",
                    fontsize=10,
                )
                style_arena_axes(axis)

            for empty_index in range(mode_maps.shape[0], mode_rows * columns):
                row_index = 1 + empty_index // columns
                column_index = empty_index % columns
                empty_axis = figure.add_subplot(grid_spec[row_index, column_index])
                empty_axis.axis("off")

            figure.suptitle(
                f"{analysis_input.source_name} transition geometry\n"
                f"{transitions_used} transitions | "
                f"{int(np.count_nonzero(occupancy_counts))} visited bins | "
                f"top-{top_k} mean |corr|="
                f"{float(top_unit_scores.mean()) if top_unit_scores.size else 0.0:.3f}",
                fontsize=12,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
            figure.savefig(figure_path, dpi=160)
            plt.close(figure)
            figures["transition_geometry"] = figure_path
            record_timing(timing_seconds, "render_figure", section_started_at)
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )

        graph_metrics = {
            "transition_geometry_transitions_used": float(transitions_used),
            "transition_geometry_visited_bin_fraction": float(
                np.count_nonzero(occupancy_counts) / max(len(occupancy_counts), 1)
            ),
            "transition_geometry_connected_bin_count": float(
                np.count_nonzero(np.any(np.isfinite(mode_maps), axis=0))
                if mode_maps.shape[0]
                else 0
            ),
            "transition_geometry_selected_mode_count": float(mode_maps.shape[0]),
            "transition_geometry_first_nontrivial_eigenvalue": float(selected_eigenvalues[0]),
        }
        alignment_metrics = {
            "transition_geometry_mean_top_unit_mode_abs_correlation": float(top_unit_scores.mean())
            if top_unit_scores.size
            else 0.0,
            "transition_geometry_max_unit_mode_abs_correlation": float(
                best_mode_abs_correlation.max()
            )
            if best_mode_abs_correlation.size
            else 0.0,
            "transition_geometry_mean_mode_best_unit_abs_correlation": float(
                best_mode_abs_correlation_per_mode.mean()
            )
            if best_mode_abs_correlation_per_mode.size
            else 0.0,
        }
        metrics = {}
        if self.emit_graph_metrics:
            metrics.update(graph_metrics)
        if self.emit_alignment_metrics:
            metrics.update(alignment_metrics)
        per_unit_metrics = (
            {
                "best_transition_mode_abs_correlation": best_mode_abs_correlation,
                "best_transition_mode_rank": best_mode_rank,
            }
            if self.emit_alignment_metrics
            else {}
        )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit_metrics,
            figures=figures,
            tables={},
            metadata={
                "transition_geometry_bounds": bounds,
                "transition_geometry_num_bins_x": num_bins_x,
                "transition_geometry_num_bins_y": num_bins_y,
                "transition_geometry_include_self_transitions": include_self_transitions,
                "transition_geometry_selected_eigenvalues": selected_eigenvalues.tolist(),
                "transition_geometry_selected_mode_indices": selected_mode_indices.tolist(),
                "transition_geometry_timing_seconds": timing_seconds,
            },
        )


@dataclass(slots=True)
class TransitionGeometryGraphModule(_TransitionGeometryModuleBase):
    """Compute transition graph occupancy and eigenspectrum metrics."""

    name: str = "transition_geometry_graph"
    emit_alignment_metrics: bool = False
    emit_panel: bool = False


@dataclass(slots=True)
class TransitionGeometryAlignmentModule(_TransitionGeometryModuleBase):
    """Compute rate-map alignment to transition-graph eigenmodes."""

    name: str = "transition_geometry_alignment"
    emit_graph_metrics: bool = False
    emit_panel: bool = False


@dataclass(slots=True)
class TransitionGeometryPanelModule(_TransitionGeometryModuleBase):
    """Render transition-geometry eigenspectrum and mode maps."""

    name: str = "transition_geometry_panel"
    emit_graph_metrics: bool = False
    emit_alignment_metrics: bool = False
