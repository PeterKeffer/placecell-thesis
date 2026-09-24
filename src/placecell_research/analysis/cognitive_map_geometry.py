"""Cognitive-map geometry: a topological map or a metric lookup?"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

from ..numerics.rate_map_kernels import (
    compute_rate_maps,
    compute_spatial_bin_assignments,
    flatten_positions,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .timing import log_timing, record_timing
from .transition_geometry import build_transition_counts, flatten_valid_episode_bins
from .world_overlay import (
    apply_plot_bounds,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    overlay_bounds,
    resolve_world_overlay,
    style_arena_axes,
)

logger = logging.getLogger(__name__)

_REPRESENTATION_METRIC_NAMES = (
    "cognitive_map_topological_index",
    "cognitive_map_partial_spearman_geodesic",
    "cognitive_map_partial_spearman_euclidean",
    "cognitive_map_spearman_repr_geodesic",
    "cognitive_map_spearman_repr_euclidean",
    "cognitive_map_spearman_geodesic_euclidean",
    "cognitive_map_detour_pair_fraction",
)


def _bin_centers(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    num_bins_x: int,
    num_bins_y: int,
) -> np.ndarray:
    """Center (x, y) of every linear bin, indexed as y * num_bins_x + x."""
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    grid_y, grid_x = np.meshgrid(y_centers, x_centers, indexing="ij")
    return np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1).astype(np.float64)


def _partial_spearman(r_target_a: float, r_target_b: float, r_ab: float) -> float:
    """Partial correlation of target with A controlling for B, from pairwise r."""
    denominator = np.sqrt(max(1.0 - r_target_b**2, 1e-12) * max(1.0 - r_ab**2, 1e-12))
    return float((r_target_a - r_target_b * r_ab) / denominator)


def _partial_spearman_controls(
    target: np.ndarray, predictor: np.ndarray, controls: list[np.ndarray]
) -> float:
    """Partial Spearman of target~predictor controlling for >=1 variables."""
    target_ranks = rankdata(target)
    predictor_ranks = rankdata(predictor)
    target_scale = np.sum((target_ranks - target_ranks.mean()) ** 2)
    predictor_scale = np.sum((predictor_ranks - predictor_ranks.mean()) ** 2)
    if controls:
        design = np.column_stack([np.ones(len(target_ranks))] + [rankdata(c) for c in controls])
        target_ranks = target_ranks - design @ np.linalg.lstsq(design, target_ranks, rcond=None)[0]
        predictor_ranks = predictor_ranks - design @ np.linalg.lstsq(
            design, predictor_ranks, rcond=None
        )[0]
    target_centered = target_ranks - target_ranks.mean()
    predictor_centered = predictor_ranks - predictor_ranks.mean()
    denominator = np.sqrt(np.sum(target_centered**2) * np.sum(predictor_centered**2))
    original_scale = np.sqrt(target_scale * predictor_scale)
    if denominator <= 1e-12 * max(original_scale, 1.0):
        return 0.0
    return float(np.sum(target_centered * predictor_centered) / denominator)


def _pairs_cross_wall(
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
) -> np.ndarray:
    """Whether the straight segment between each location pair crosses any wall segment."""
    pair_a = centers_i[:, None, :]
    pair_b = centers_j[:, None, :]
    wall_c = np.asarray([segment[0] for segment in segments], dtype=np.float64)[None, :, :]
    wall_d = np.asarray([segment[1] for segment in segments], dtype=np.float64)[None, :, :]

    def cross(origin: np.ndarray, point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
        return (point_a[..., 0] - origin[..., 0]) * (point_b[..., 1] - origin[..., 1]) - (
            point_a[..., 1] - origin[..., 1]
        ) * (point_b[..., 0] - origin[..., 0])

    d1 = cross(wall_c, wall_d, pair_a)
    d2 = cross(wall_c, wall_d, pair_b)
    d3 = cross(pair_a, pair_b, wall_c)
    d4 = cross(pair_a, pair_b, wall_d)
    straddles_wall = (d1 > 0) != (d2 > 0)
    straddles_pair = (d3 > 0) != (d4 > 0)
    return np.any(straddles_wall & straddles_pair, axis=1)


def _distance_stratified_wall_gap(
    euclidean_pairs: np.ndarray,
    representational_pairs: np.ndarray,
    wall_separated: np.ndarray,
    *,
    num_bins: int = 10,
    min_per_group: int = 8,
) -> tuple[float, list[tuple[float, float, float, int, int]], float, bool]:
    same_region = ~wall_separated
    raw_gap = float(
        np.median(representational_pairs[wall_separated])
        - np.median(representational_pairs[same_region])
    )
    edges = np.unique(np.quantile(euclidean_pairs, np.linspace(0.0, 1.0, num_bins + 1)))
    per_bin: list[tuple[float, float, float, int, int]] = []
    gaps: list[float] = []
    for index in range(edges.size - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        is_last = index == edges.size - 2
        in_bin = (euclidean_pairs >= low) & (
            euclidean_pairs <= high if is_last else euclidean_pairs < high
        )
        separated_in_bin = in_bin & wall_separated
        same_in_bin = in_bin & same_region
        n_sep = int(separated_in_bin.sum())
        n_same = int(same_in_bin.sum())
        if n_sep >= min_per_group and n_same >= min_per_group:
            median_separated = float(np.median(representational_pairs[separated_in_bin]))
            median_same = float(np.median(representational_pairs[same_in_bin]))
            gaps.append(median_separated - median_same)
            per_bin.append((0.5 * (low + high), median_separated, median_same, n_sep, n_same))
    matched = len(gaps) >= 2
    mean_gap = float(np.mean(gaps)) if matched else raw_gap
    return mean_gap, per_bin, raw_gap, matched


def _representational_distance(bin_codes: np.ndarray) -> np.ndarray:
    """1 - population-vector correlation between per-bin code vectors."""
    centered = bin_codes - bin_codes.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, np.nan)
    normalized = centered / safe_norms
    correlation = normalized @ normalized.T
    return 1.0 - correlation


def _largest_connected_component(adjacency: np.ndarray) -> np.ndarray:
    """Indices (into the active-bin array) of the largest connected component."""
    component_count, labels = connected_components(adjacency, directed=False)
    if component_count <= 1:
        return np.arange(adjacency.shape[0], dtype=np.int64)
    largest_label = np.argmax(np.bincount(labels))
    return np.flatnonzero(labels == largest_label).astype(np.int64)


@dataclass(slots=True)
class CognitiveMapGeometryModule:
    """Test whether representational distance tracks geodesic vs Euclidean distance."""

    name: str = "cognitive_map_geometry"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def required_batch_keys(self) -> set[str]:
        return {"latent"}

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        num_bins_x = int(config.get("cognitive_map_num_bins_x", 20))
        num_bins_y = int(config.get("cognitive_map_num_bins_y", 20))
        min_occupancy = float(config.get("min_occupancy", 1e-6))
        max_scatter_points = int(config.get("cognitive_map_max_scatter_points", 4000))
        render_rdm = bool(config.get("cognitive_map_render_rdm", True))

        section_started_at = perf_counter()
        valid_positions = flatten_positions(analysis_input.position_xy, analysis_input.valid_mask)
        if valid_positions.size == 0:
            raise ValueError("Cognitive-map geometry needs at least one valid position.")

        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        if bounds is None:
            _bins, _x_edges, _y_edges, bounds = compute_spatial_bin_assignments(
                valid_positions, num_bins_x=num_bins_x, num_bins_y=num_bins_y
            )
        _bins, x_edges, y_edges, bounds = compute_spatial_bin_assignments(
            valid_positions, num_bins_x=num_bins_x, num_bins_y=num_bins_y, bounds=bounds
        )
        centers = _bin_centers(x_edges, y_edges, num_bins_x, num_bins_y)
        record_timing(timing_seconds, "prepare_bins", section_started_at)

        section_started_at = perf_counter()
        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=0.0,
            min_occupancy=min_occupancy,
            bounds=bounds,
        )
        num_units = rate_map_result.rate_maps.shape[0]
        bin_codes = rate_map_result.rate_maps.reshape(num_units, num_bins_x * num_bins_y).T
        visited = np.isfinite(bin_codes).all(axis=1)
        record_timing(timing_seconds, "rate_maps", section_started_at)

        section_started_at = perf_counter()
        episode_bins, occupancy_counts = flatten_valid_episode_bins(
            analysis_input.position_xy,
            analysis_input.valid_mask,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=bounds,
        )
        transition_counts, transitions_used = build_transition_counts(
            episode_bins, num_bins=num_bins_x * num_bins_y, include_self_transitions=False
        )
        if transitions_used <= 0:
            raise ValueError("Cognitive-map geometry found no bin-to-bin transitions.")

        active = np.flatnonzero(visited & (occupancy_counts > 0.0))
        if active.size < 8:
            raise ValueError(
                f"Cognitive-map geometry needs >=8 visited bins, found {int(active.size)}. "
                "Use coarser cognitive_map_num_bins."
            )
        active_centers = centers[active]
        euclidean_full = cdist(active_centers, active_centers)
        symmetric_counts = transition_counts[np.ix_(active, active)]
        symmetric_counts = symmetric_counts + symmetric_counts.T
        adjacency = symmetric_counts > 0.0
        weighted_graph = np.where(adjacency, euclidean_full, 0.0)
        record_timing(timing_seconds, "transition_graph", section_started_at)

        section_started_at = perf_counter()
        component = _largest_connected_component(adjacency)
        if component.size < 8:
            raise ValueError(
                f"Largest connected bin component has {int(component.size)} bins (<8). "
                "Transitions are too fragmented for a geodesic comparison."
            )
        component_graph = weighted_graph[np.ix_(component, component)]
        geodesic = shortest_path(component_graph, method="D", directed=False)
        euclidean = euclidean_full[np.ix_(component, component)]
        component_codes = bin_codes[active][component]
        representational = _representational_distance(component_codes)
        component_centers = active_centers[component]
        visual_rdm = self._latent_rdm(
            analysis_input,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            min_occupancy=min_occupancy,
            bounds=bounds,
            active=active,
            component=component,
        )
        record_timing(timing_seconds, "geodesic", section_started_at)

        section_started_at = perf_counter()
        upper_i, upper_j = np.triu_indices(component.size, k=1)
        geodesic_all = geodesic[upper_i, upper_j]
        euclidean_all = euclidean[upper_i, upper_j]
        representational_all = representational[upper_i, upper_j]
        finite = (
            np.isfinite(geodesic_all)
            & np.isfinite(euclidean_all)
            & np.isfinite(representational_all)
        )
        geodesic_pairs = geodesic_all[finite]
        euclidean_pairs = euclidean_all[finite]
        representational_pairs = representational_all[finite]
        if geodesic_pairs.size < 16:
            spatial_pair_count = int(
                (np.isfinite(geodesic_all) & np.isfinite(euclidean_all)).sum()
            )
            if spatial_pair_count < 16:
                raise ValueError(
                    "Too few finite bin pairs for a cognitive-map geometry correlation."
                )
            reason = (
                "representation degenerate: only "
                f"{int(geodesic_pairs.size)} of {spatial_pair_count} spatial bin pairs have a "
                "finite representational distance, so the population-vector correlations are "
                "undefined (the code carries no per-bin variation)."
            )
            logger.warning(
                "cognitive_map_geometry reporting NaN for %s (%s): %s",
                analysis_input.source_name,
                analysis_input.split_name,
                reason,
            )
            record_timing(timing_seconds, "correlations", section_started_at)
            log_timing(
                self.name,
                analysis_input.source_name,
                analysis_input.split_name,
                timing_seconds,
                config=config,
            )
            return AnalysisResult(
                metrics={
                    **{name: float("nan") for name in _REPRESENTATION_METRIC_NAMES},
                    "cognitive_map_component_bin_count": float(component.size),
                    "cognitive_map_pair_count": float(geodesic_pairs.size),
                    "cognitive_map_skipped": 1.0,
                },
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "cognitive_map_num_bins_x": num_bins_x,
                    "cognitive_map_num_bins_y": num_bins_y,
                    "cognitive_map_bounds": bounds,
                    "cognitive_map_timing_seconds": timing_seconds,
                    "cognitive_map_skipped_reason": reason,
                },
            )

        r_repr_geodesic = float(spearmanr(representational_pairs, geodesic_pairs)[0])
        r_repr_euclidean = float(spearmanr(representational_pairs, euclidean_pairs)[0])
        r_geodesic_euclidean = float(spearmanr(geodesic_pairs, euclidean_pairs)[0])
        partial_geodesic = _partial_spearman(
            r_repr_geodesic, r_repr_euclidean, r_geodesic_euclidean
        )
        partial_euclidean = _partial_spearman(
            r_repr_euclidean, r_repr_geodesic, r_geodesic_euclidean
        )
        topological_index = partial_geodesic - partial_euclidean
        detour_excess = geodesic_pairs - euclidean_pairs
        detour_fraction = float(np.mean(detour_excess > euclidean_pairs * 0.25))

        visual_metrics: dict[str, float] = {}
        if visual_rdm is not None:
            visual_all = visual_rdm[upper_i, upper_j]
            finite_visual = finite & np.isfinite(visual_all)
            representational_v = representational_all[finite_visual]
            geodesic_v = geodesic_all[finite_visual]
            euclidean_v = euclidean_all[finite_visual]
            visual_v = visual_all[finite_visual]
            if visual_v.size >= 16:
                visual_metrics = {
                    "cognitive_map_spearman_repr_visual": float(
                        spearmanr(representational_v, visual_v)[0]
                    ),
                    "cognitive_map_partial_spearman_geodesic_given_visual": (
                        _partial_spearman_controls(representational_v, geodesic_v, [visual_v])
                    ),
                    "cognitive_map_partial_spearman_geodesic_given_euclidean_visual": (
                        _partial_spearman_controls(
                            representational_v, geodesic_v, [euclidean_v, visual_v]
                        )
                    ),
                }
        wall_separated = None
        wall_stratification: (
            tuple[float, list[tuple[float, float, float, int, int]], float, bool] | None
        ) = None
        wall_metrics: dict[str, float] = {}
        if world_overlay is not None and world_overlay.segments:
            centers_i = component_centers[upper_i[finite]]
            centers_j = component_centers[upper_j[finite]]
            wall_separated = _pairs_cross_wall(centers_i, centers_j, world_overlay.segments)
            if wall_separated.any() and (~wall_separated).any():
                wall_metrics["cognitive_map_wall_separated_pair_fraction"] = float(
                    wall_separated.mean()
                )
                mean_gap, per_bin, raw_gap, matched = _distance_stratified_wall_gap(
                    euclidean_pairs, representational_pairs, wall_separated
                )
                wall_stratification = (mean_gap, per_bin, raw_gap, matched)
                wall_metrics["cognitive_map_wall_repr_gap_matched"] = mean_gap
                wall_metrics["cognitive_map_wall_repr_gap_raw"] = raw_gap
                wall_metrics["cognitive_map_wall_matched_bin_count"] = float(len(per_bin))
            else:
                wall_separated = None
        record_timing(timing_seconds, "correlations", section_started_at)

        figures = self._render_figure(
            output_dir=output_dir,
            analysis_input=analysis_input,
            representational_pairs=representational_pairs,
            euclidean_pairs=euclidean_pairs,
            geodesic_pairs=geodesic_pairs,
            component_centers=component_centers,
            geodesic=geodesic,
            bounds=bounds,
            valid_positions=valid_positions,
            world_overlay=world_overlay,
            topological_index=topological_index,
            partial_geodesic=partial_geodesic,
            partial_euclidean=partial_euclidean,
            max_scatter_points=max_scatter_points,
            timing_seconds=timing_seconds,
        )
        if wall_separated is not None and wall_stratification is not None:
            figures.update(
                self._render_wall_scatter_figure(
                    output_dir=output_dir,
                    analysis_input=analysis_input,
                    euclidean_pairs=euclidean_pairs,
                    representational_pairs=representational_pairs,
                    wall_separated=wall_separated,
                    stratification=wall_stratification,
                    max_scatter_points=max_scatter_points,
                    timing_seconds=timing_seconds,
                )
            )
        if render_rdm:
            figures.update(
                self._render_rdm_figure(
                    output_dir=output_dir,
                    analysis_input=analysis_input,
                    representational=representational,
                    geodesic=geodesic,
                    euclidean=euclidean,
                    visual_rdm=visual_rdm,
                    component_centers=component_centers,
                    topological_index=topological_index,
                    visual_metrics=visual_metrics,
                    timing_seconds=timing_seconds,
                )
            )
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )

        metrics = {
            "cognitive_map_topological_index": topological_index,
            "cognitive_map_partial_spearman_geodesic": partial_geodesic,
            "cognitive_map_partial_spearman_euclidean": partial_euclidean,
            "cognitive_map_spearman_repr_geodesic": r_repr_geodesic,
            "cognitive_map_spearman_repr_euclidean": r_repr_euclidean,
            "cognitive_map_spearman_geodesic_euclidean": r_geodesic_euclidean,
            "cognitive_map_detour_pair_fraction": detour_fraction,
            "cognitive_map_component_bin_count": float(component.size),
            "cognitive_map_pair_count": float(geodesic_pairs.size),
            **visual_metrics,
            **wall_metrics,
        }
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures=figures,
            tables={},
            metadata={
                "cognitive_map_num_bins_x": num_bins_x,
                "cognitive_map_num_bins_y": num_bins_y,
                "cognitive_map_bounds": bounds,
                "cognitive_map_timing_seconds": timing_seconds,
            },
        )

    def _render_wall_scatter_figure(
        self,
        *,
        output_dir: Path,
        analysis_input: AnalysisInput,
        euclidean_pairs: np.ndarray,
        representational_pairs: np.ndarray,
        wall_separated: np.ndarray,
        stratification: tuple[float, list[tuple[float, float, float, int, int]], float, bool],
        max_scatter_points: int,
        timing_seconds: dict[str, float],
    ) -> dict[str, Path]:
        section_started_at = perf_counter()
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_path = module_dir / (
            f"cognitive_map_wall_scatter__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )

        same_region = ~wall_separated
        mean_gap, per_bin, _raw_gap, matched = stratification

        def sample(mask: np.ndarray) -> np.ndarray:
            indices = np.flatnonzero(mask)
            if indices.size > max_scatter_points:
                stride = int(np.ceil(indices.size / max_scatter_points))
                return indices[::stride]
            return indices

        figure, axis = plt.subplots(figsize=(6.4, 5.2))
        same_sample = sample(same_region)
        separated_sample = sample(wall_separated)
        axis.scatter(
            euclidean_pairs[same_sample],
            representational_pairs[same_sample],
            s=6, alpha=0.16, color="#7F7F7F", label="same region",
        )
        axis.scatter(
            euclidean_pairs[separated_sample],
            representational_pairs[separated_sample],
            s=6, alpha=0.22, color="#D62728", label="wall between",
        )
        if per_bin:
            centers = [item[0] for item in per_bin]
            median_separated = [item[1] for item in per_bin]
            median_same = [item[2] for item in per_bin]
            axis.plot(
                centers, median_same, color="#3F3F3F", marker="o", markersize=3,
                linewidth=1.8, label="median, same region",
            )
            axis.plot(
                centers, median_separated, color="#B22222", marker="o", markersize=3,
                linewidth=1.8, label="median, wall between",
            )
        axis.set_xlabel("Euclidean distance (world units)")
        axis.set_ylabel("representational distance (1 - corr)")
        note = (
            f"distance-matched gap {mean_gap:+.3f} (mean over {len(per_bin)} distance bins)"
            if matched
            else f"gap {mean_gap:+.3f} (too few pairs per bin to stratify; unmatched)"
        )
        axis.set_title(
            "Wall-separated vs same-region pairs, controlled for distance\n" + note,
            fontsize=10,
        )
        axis.legend(loc="lower right", fontsize=7, framealpha=0.9)
        axis.grid(alpha=0.3)
        figure.suptitle(
            f"{analysis_input.source_name}: does a wall make places look far?", fontsize=12
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        record_timing(timing_seconds, "render_wall_scatter", section_started_at)
        return {"cognitive_map_wall_scatter": figure_path}

    def _latent_rdm(
        self,
        analysis_input: AnalysisInput,
        *,
        num_bins_x: int,
        num_bins_y: int,
        min_occupancy: float,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        active: np.ndarray,
        component: np.ndarray,
    ) -> np.ndarray | None:
        """1 - population-vector correlation between per-bin AE-latent vectors."""
        if analysis_input.latent is None:
            return None
        latent_rate_maps = compute_rate_maps(
            analysis_input.latent,
            analysis_input.position_xy,
            analysis_input.valid_mask,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            smoothing_sigma=0.0,
            min_occupancy=min_occupancy,
            bounds=bounds,
        ).rate_maps
        latent_codes = latent_rate_maps.reshape(
            latent_rate_maps.shape[0], num_bins_x * num_bins_y
        ).T
        return _representational_distance(latent_codes[active][component])

    def _render_rdm_figure(
        self,
        *,
        output_dir: Path,
        analysis_input: AnalysisInput,
        representational: np.ndarray,
        geodesic: np.ndarray,
        euclidean: np.ndarray,
        visual_rdm: np.ndarray | None,
        component_centers: np.ndarray,
        topological_index: float,
        visual_metrics: dict[str, float],
        timing_seconds: dict[str, float],
    ) -> dict[str, Path]:
        section_started_at = perf_counter()
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_path = module_dir / (
            f"cognitive_map_rdm__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )

        seed_index = int(np.argmin(component_centers[:, 0] + component_centers[:, 1]))
        order = np.argsort(geodesic[seed_index], kind="stable")

        def reorder(matrix: np.ndarray) -> np.ndarray:
            return matrix[np.ix_(order, order)]

        def normalized(matrix: np.ndarray) -> np.ndarray:
            finite_values = matrix[np.isfinite(matrix)]
            scale = float(finite_values.max()) if finite_values.size else 1.0
            return matrix / scale if scale > 0 else matrix

        colormap = "viridis"
        panels: list[tuple[str | None, np.ndarray | None, str]] = [
            ("neural RDM (1 - pop-vec corr)", reorder(representational), colormap),
            ("geodesic distance (walked)", reorder(normalized(geodesic)), colormap),
            ("Euclidean distance (crow-flies)", reorder(normalized(euclidean)), colormap),
        ]
        if visual_rdm is not None:
            panels.append(("AE-latent RDM (visual input)", reorder(visual_rdm), colormap))
        else:
            panels.append((None, None, colormap))

        figure, axes = plt.subplots(2, 2, figsize=(9.6, 9.8))
        for axis, (title, matrix, cmap) in zip(axes.flat, panels, strict=False):
            if matrix is None:
                axis.text(
                    0.5, 0.5, "AE-latent not collected\n(visual baseline off)",
                    ha="center", va="center", fontsize=10,
                )
                axis.set_axis_off()
                continue
            image = axis.imshow(matrix, cmap=cmap, origin="upper")
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("location j (ordered by walking distance)")
            axis.set_ylabel("location i")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="dissimilarity")

        subtitle = f"topological index = {topological_index:+.3f}"
        if visual_metrics:
            subtitle += (
                "\npartial r(repr, geodesic | euclidean, visual) = "
                f"{visual_metrics['cognitive_map_partial_spearman_geodesic_given_euclidean_visual']:+.3f}"
            )
        figure.suptitle(
            f"{analysis_input.source_name} representational similarity matrix\n{subtitle}",
            fontsize=12,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        record_timing(timing_seconds, "render_rdm_figure", section_started_at)
        return {"cognitive_map_rdm": figure_path}

    def _render_figure(
        self,
        *,
        output_dir: Path,
        analysis_input: AnalysisInput,
        representational_pairs: np.ndarray,
        euclidean_pairs: np.ndarray,
        geodesic_pairs: np.ndarray,
        component_centers: np.ndarray,
        geodesic: np.ndarray,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        valid_positions: np.ndarray,
        world_overlay,
        topological_index: float,
        partial_geodesic: float,
        partial_euclidean: float,
        max_scatter_points: int,
        timing_seconds: dict[str, float],
    ) -> dict[str, Path]:
        section_started_at = perf_counter()
        module_dir = output_dir / self.name
        module_dir.mkdir(parents=True, exist_ok=True)
        figure_name = (
            f"cognitive_map_geometry__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        figure_path = module_dir / figure_name

        if representational_pairs.size > max_scatter_points:
            stride = int(np.ceil(representational_pairs.size / max_scatter_points))
            sample = slice(None, None, stride)
        else:
            sample = slice(None)

        figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
        scatter_axis, map_axis = axes

        scatter_axis.scatter(
            euclidean_pairs[sample],
            representational_pairs[sample],
            s=5,
            alpha=0.25,
            color="#7F7F7F",
            label="Euclidean",
        )
        scatter_axis.scatter(
            geodesic_pairs[sample],
            representational_pairs[sample],
            s=5,
            alpha=0.25,
            color="#1F77B4",
            label="geodesic (walked)",
        )
        scatter_axis.set_xlabel("spatial distance (world units)")
        scatter_axis.set_ylabel("representational distance (1 - corr)")
        scatter_axis.set_title(
            f"topological index = {topological_index:+.3f}\n"
            f"partial r: geodesic {partial_geodesic:+.3f} | euclidean {partial_euclidean:+.3f}",
            fontsize=10,
        )
        scatter_axis.legend(loc="lower right", fontsize=8, framealpha=0.9)
        scatter_axis.grid(alpha=0.3)

        seed_index = int(np.argmin(component_centers[:, 0] + component_centers[:, 1]))
        seed_distances = geodesic[seed_index]
        finite_seed = np.isfinite(seed_distances)
        scatter = map_axis.scatter(
            component_centers[finite_seed, 0],
            component_centers[finite_seed, 1],
            c=seed_distances[finite_seed],
            s=28,
            cmap="viridis",
            marker="s",
        )
        map_axis.scatter(
            component_centers[seed_index, 0],
            component_centers[seed_index, 1],
            color="#D62728",
            s=70,
            marker="*",
            label="seed",
        )
        figure.colorbar(scatter, ax=map_axis, fraction=0.046, pad=0.04, label="geodesic distance")
        if world_overlay is not None:
            draw_world_segments_on_axis(map_axis, world_overlay.segments, line_color="#202020")
            draw_landmarks_on_axis(map_axis, world_overlay)
        apply_plot_bounds(
            map_axis,
            env_id=str(analysis_input.metadata.get("env_id", "")),
            position_xy=valid_positions,
        )
        map_axis.set_title("Geodesic distance from a corner (walls force detours)", fontsize=10)
        style_arena_axes(map_axis)

        figure.suptitle(
            f"{analysis_input.source_name} cognitive-map geometry "
            f"({geodesic_pairs.size} bin pairs)",
            fontsize=12,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        record_timing(timing_seconds, "render_figure", section_started_at)
        return {"cognitive_map_geometry": figure_path}
