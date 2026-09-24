"""UMAP topology analysis for spatial representations."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from scipy.sparse.linalg import ArpackError
from scipy.spatial.distance import pdist, squareform

from placecell_research.utils.angles import wrap_radians

from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    flatten_valid_steps,
    flatten_vector_field,
)
from .base import AnalysisInput, AnalysisResult
from .helpers import (
    compute_trustworthiness,
    subsample_indices,
)
from .timing import log_timing, record_timing
from .transition_geometry import build_transition_counts, flatten_valid_episode_bins
from .world_overlay import (
    POSITION_X_LABEL,
    POSITION_Y_LABEL,
    draw_landmarks_on_axis,
    draw_world_segments_on_axis,
    finalize_arena_axis,
    resolve_plot_bounds,
    resolve_world_overlay,
)

logger = logging.getLogger(__name__)

_DEGENERATE_ACTIVATION_TOLERANCE = 1e-8


class _DegenerateRepresentationError(ValueError):
    """The feature matrix is collapsed, so every embedding of it would be undefined."""


def _ensure_embeddable_features(features: np.ndarray, *, embedder: str) -> None:
    """Reject a collapsed feature matrix before it reaches an eigensolver."""
    if not np.isfinite(features).all():
        raise _DegenerateRepresentationError(
            f"representation degenerate: the {embedder} input holds non-finite activations."
        )
    largest_activation = float(np.abs(features).max()) if features.size else 0.0
    if largest_activation <= _DEGENERATE_ACTIVATION_TOLERANCE:
        raise _DegenerateRepresentationError(
            f"representation degenerate: the largest activation reaching {embedder} is "
            f"{largest_activation:.3e}, at or below {_DEGENERATE_ACTIVATION_TOLERANCE:.0e}, "
            "so the code is effectively zero."
        )
    largest_deviation = float(np.abs(features - features.mean(axis=0, keepdims=True)).max())
    if largest_deviation <= _DEGENERATE_ACTIVATION_TOLERANCE:
        raise _DegenerateRepresentationError(
            f"representation degenerate: every sample reaching {embedder} carries the same code "
            f"(largest deviation from the population mean is {largest_deviation:.3e}), so all "
            "pairwise distances are zero."
        )


def _degenerate_skip_reason(error: Exception, *, embedder: str) -> str:
    """One sentence saying why an embedding was skipped, for the log and the report."""
    if isinstance(error, ArpackError):
        return (
            f"representation degenerate: ARPACK could not solve the {embedder} eigenproblem "
            f"({error}), which is what a rank-deficient, effectively zero code produces."
        )
    return str(error)


def _load_umap_estimator() -> type[Any]:
    try:
        return importlib.import_module("umap").UMAP
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "topology_umap requires the optional runtime dependency `umap-learn`. "
            "Reinstall the environment so the updated project dependencies are available."
        ) from error


def _resolve_heading_values(flattened_heading: np.ndarray | None) -> np.ndarray | None:
    if flattened_heading is None:
        return None
    heading_values = np.asarray(flattened_heading, dtype=np.float32)
    if heading_values.ndim == 1:
        return _wrap_scalar_headings(heading_values)
    if heading_values.ndim == 2 and heading_values.shape[-1] == 1:
        return _wrap_scalar_headings(heading_values[:, 0])
    if heading_values.ndim == 2 and heading_values.shape[-1] >= 2:
        return np.arctan2(heading_values[:, 1], heading_values[:, 0]).astype(np.float32, copy=False)
    return _wrap_scalar_headings(heading_values.reshape(len(heading_values), -1)[:, 0])


def _wrap_scalar_headings(heading_values: np.ndarray) -> np.ndarray:
    return wrap_radians(heading_values)


def _fit_umap_embedding(features: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    _ensure_embeddable_features(features, embedder="UMAP")
    inference_threads = torch.get_num_threads()
    try:
        estimator = _load_umap_estimator()(
            n_components=2,
            n_neighbors=max(2, min(int(config.get("umap_n_neighbors", 25)), len(features) - 1)),
            min_dist=float(config.get("umap_min_dist", 0.1)),
            metric=str(config.get("umap_metric", "euclidean")),
            random_state=int(config.get("umap_random_seed", 42)),
        )
        return np.asarray(estimator.fit_transform(features), dtype=np.float32)
    finally:
        torch.set_num_threads(inference_threads)


def _cosine_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances, suitable as a precomputed dissimilarity for MDS."""
    from sklearn.metrics.pairwise import cosine_distances

    distances = cosine_distances(features.astype(np.float64, copy=False))
    np.fill_diagonal(distances, 0.0)
    return np.clip(distances, 0.0, None)


class _DisconnectedKNNGraphError(ValueError):
    """Isomap's neighbor graph is disconnected, so a geodesic embedding is degenerate."""


def _ensure_connected_knn_graph(features: np.ndarray, *, n_neighbors: int, metric: str) -> None:
    """Detect a disconnected k-NN graph before Isomap builds geodesics over it."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(features, n_neighbors=n_neighbors, metric=metric, mode="connectivity")
    num_components, _ = connected_components(graph, directed=False)
    if num_components > 1:
        raise _DisconnectedKNNGraphError(
            f"Isomap requires a connected k-NN graph but found {num_components} components "
            f"across {len(features)} pooled bins at n_neighbors={n_neighbors}. Raise "
            "isomap_n_neighbors or lower manifold_pool_min_count."
        )


def _fit_mds_embedding(features: np.ndarray, *, metric: str, random_seed: int) -> np.ndarray:
    """Metric MDS to 2D."""
    from sklearn.manifold import MDS

    dissimilarity = "precomputed" if metric == "cosine" else "euclidean"
    estimator = MDS(
        n_components=2,
        dissimilarity=dissimilarity,
        random_state=random_seed,
        n_init=4,
        normalized_stress=False,
    )
    inputs = _cosine_distance_matrix(features) if metric == "cosine" else features
    return np.asarray(estimator.fit_transform(inputs), dtype=np.float32)


def _fit_isomap_embedding(features: np.ndarray, *, metric: str, n_neighbors: int) -> np.ndarray:
    """Isomap to 2D after verifying the neighbor graph is connected."""
    from sklearn.manifold import Isomap

    safe_neighbors = max(2, min(int(n_neighbors), len(features) - 1))
    _ensure_connected_knn_graph(features, n_neighbors=safe_neighbors, metric=metric)
    estimator = Isomap(n_components=2, n_neighbors=safe_neighbors, metric=metric)
    return np.asarray(estimator.fit_transform(features), dtype=np.float32)


def _fit_pooled_manifold_embedding(
    features: np.ndarray, *, method: str, config: dict[str, Any]
) -> np.ndarray:
    """Dispatch pooled bin-mean features to the requested distance-faithful embedder."""
    _ensure_embeddable_features(features, embedder=method.upper())
    metric = str(config.get("manifold_metric", "euclidean"))
    if method == "mds":
        return _fit_mds_embedding(
            features, metric=metric, random_seed=int(config.get("manifold_random_seed", 42))
        )
    if method == "isomap":
        return _fit_isomap_embedding(
            features, metric=metric, n_neighbors=int(config.get("isomap_n_neighbors", 10))
        )
    raise ValueError(f"Unknown manifold embedding method: {method!r}.")


def pool_by_spatial_bin(
    analysis_input: AnalysisInput,
    features: np.ndarray,
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pooled = _pool_by_spatial_bin_with_metadata(
        analysis_input,
        features,
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        min_count=min_count,
    )
    if pooled is None:
        return None
    pooled_features, pooled_positions, pooled_counts, _kept_bins, _bounds = pooled
    return pooled_features, pooled_positions, pooled_counts


def _pool_by_spatial_bin_with_metadata(
    analysis_input: AnalysisInput,
    features: np.ndarray,
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    min_count: int,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[tuple[float, float], tuple[float, float]],
    ]
    | None
):
    """Bin means for one target, computed once and shared by every module that pools it."""
    return analysis_input.get_cached_metric(
        ("pool_by_spatial_bin", num_bins_x, num_bins_y, min_count),
        lambda: _compute_pooled_spatial_bins(
            features,
            positions,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            min_count=min_count,
        ),
    )


def _compute_pooled_spatial_bins(
    features: np.ndarray,
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    min_count: int,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[tuple[float, float], tuple[float, float]],
    ]
    | None
):
    linear_bins, _, _, bounds = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
    )
    num_bins = num_bins_x * num_bins_y
    counts = np.bincount(linear_bins, minlength=num_bins).astype(np.int32, copy=False)
    keep_bins = np.flatnonzero(counts >= max(1, int(min_count)))
    if len(keep_bins) < 4:
        return None

    feature_sums = np.zeros((num_bins, features.shape[-1]), dtype=np.float32)
    position_sums = np.zeros((num_bins, 2), dtype=np.float32)
    np.add.at(feature_sums, linear_bins, features)
    np.add.at(position_sums, linear_bins, positions)

    pooled_features = feature_sums[keep_bins] / counts[keep_bins, None]
    pooled_positions = position_sums[keep_bins] / counts[keep_bins, None]
    pooled_counts = counts[keep_bins].astype(np.float32, copy=False)
    return (
        pooled_features.astype(np.float32, copy=False),
        pooled_positions.astype(np.float32, copy=False),
        pooled_counts,
        keep_bins.astype(np.int32, copy=False),
        bounds,
    )


@dataclass(frozen=True, slots=True)
class _PooledTransitionGraph:
    edge_pairs: np.ndarray
    edge_counts: np.ndarray
    metrics: dict[str, float]


def _greedy_distance_matched_non_edges(
    edge_pairs: np.ndarray,
    non_edge_pairs: np.ndarray,
    positions: np.ndarray,
    *,
    max_distance_error: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match edges to unique non-edges within an absolute physical-distance caliper."""
    if max_distance_error < 0.0:
        raise ValueError("Physical-distance matching caliper must be non-negative.")
    if len(edge_pairs) == 0 or len(non_edge_pairs) == 0:
        empty_pairs = np.zeros((0, 2), dtype=np.int32)
        return empty_pairs, empty_pairs.copy(), np.zeros((0,), dtype=np.float64)

    edge_world_distances = np.linalg.norm(
        positions[edge_pairs[:, 0]] - positions[edge_pairs[:, 1]], axis=1
    )
    non_edge_world_distances = np.linalg.norm(
        positions[non_edge_pairs[:, 0]] - positions[non_edge_pairs[:, 1]], axis=1
    )
    available = np.ones(len(non_edge_pairs), dtype=bool)
    matched_edge_indices: list[int] = []
    matched_non_edge_indices: list[int] = []
    match_errors: list[float] = []
    for edge_index in range(len(edge_pairs)):
        candidates = np.flatnonzero(available)
        if len(candidates) == 0:
            break
        distance_errors = np.abs(
            non_edge_world_distances[candidates] - edge_world_distances[edge_index]
        )
        selected_candidate = int(np.argmin(distance_errors))
        selected_index = int(candidates[selected_candidate])
        selected_error = float(distance_errors[selected_candidate])
        if selected_error > max_distance_error:
            continue
        available[selected_index] = False
        matched_edge_indices.append(edge_index)
        matched_non_edge_indices.append(selected_index)
        match_errors.append(selected_error)

    return (
        edge_pairs[np.asarray(matched_edge_indices, dtype=np.int32)],
        non_edge_pairs[np.asarray(matched_non_edge_indices, dtype=np.int32)],
        np.asarray(match_errors, dtype=np.float64),
    )


def _build_pooled_transition_graph(
    analysis_input: AnalysisInput,
    pooled_features: np.ndarray,
    pooled_positions: np.ndarray,
    pooled_bin_indices: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    min_transition_count: int,
    representation_metric: str,
    physical_match_caliper: float = 0.25,
) -> _PooledTransitionGraph:
    episode_bins, _occupancy_counts = flatten_valid_episode_bins(
        analysis_input.position_xy,
        analysis_input.valid_mask,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    transition_counts, transitions_used = build_transition_counts(
        episode_bins,
        num_bins=num_bins_x * num_bins_y,
        include_self_transitions=False,
    )
    pooled_transition_counts = transition_counts[np.ix_(pooled_bin_indices, pooled_bin_indices)]
    symmetric_counts = pooled_transition_counts + pooled_transition_counts.T
    upper_triangle = np.triu(np.ones_like(symmetric_counts, dtype=bool), k=1)
    observed_transition_pairs = np.argwhere(upper_triangle & (symmetric_counts > 0.0))
    displayed_edge_pairs = np.argwhere(
        upper_triangle & (symmetric_counts >= max(1, int(min_transition_count)))
    ).astype(np.int32, copy=False)
    displayed_edge_counts = (
        symmetric_counts[displayed_edge_pairs[:, 0], displayed_edge_pairs[:, 1]].astype(
            np.float32, copy=False
        )
        if len(displayed_edge_pairs) > 0
        else np.zeros((0,), dtype=np.float32)
    )

    non_edge_pairs = np.argwhere(upper_triangle & (symmetric_counts == 0.0)).astype(
        np.int32, copy=False
    )
    matched_edge_pairs, matched_non_edge_pairs, physical_match_errors = (
        _greedy_distance_matched_non_edges(
            displayed_edge_pairs,
            non_edge_pairs,
            pooled_positions,
            max_distance_error=physical_match_caliper,
        )
    )
    if len(matched_edge_pairs) > 0:
        represented_distances = squareform(
            pdist(pooled_features.astype(np.float64, copy=False), metric=representation_metric)
        )
        edge_distances = represented_distances[matched_edge_pairs[:, 0], matched_edge_pairs[:, 1]]
        non_edge_distances = represented_distances[
            matched_non_edge_pairs[:, 0], matched_non_edge_pairs[:, 1]
        ]
        finite = np.isfinite(edge_distances) & np.isfinite(non_edge_distances)
        edge_distances = edge_distances[finite]
        non_edge_distances = non_edge_distances[finite]
        physical_match_errors = physical_match_errors[finite]
    else:
        edge_distances = np.zeros((0,), dtype=np.float64)
        non_edge_distances = np.zeros((0,), dtype=np.float64)

    if len(edge_distances) > 0:
        edge_median = float(np.median(edge_distances))
        non_edge_median = float(np.median(non_edge_distances))
        distance_gap = non_edge_median - edge_median
    else:
        edge_median = float("nan")
        non_edge_median = float("nan")
        distance_gap = float("nan")

    return _PooledTransitionGraph(
        edge_pairs=displayed_edge_pairs,
        edge_counts=displayed_edge_counts,
        metrics={
            "pooled_umap_transition_steps_used": float(transitions_used),
            "pooled_umap_transition_edge_count": float(len(displayed_edge_pairs)),
            "pooled_umap_observed_transition_edge_count": float(len(observed_transition_pairs)),
            "pooled_umap_matched_non_edge_count": float(len(edge_distances)),
            "pooled_umap_transition_unmatched_edge_count": float(
                len(displayed_edge_pairs) - len(edge_distances)
            ),
            "pooled_umap_transition_edge_distance_median": edge_median,
            "pooled_umap_matched_non_edge_distance_median": non_edge_median,
            "pooled_umap_transition_distance_gap": distance_gap,
            "pooled_umap_transition_physical_match_caliper": float(physical_match_caliper),
            "pooled_umap_transition_physical_match_error_mean": (
                float(physical_match_errors.mean())
                if len(physical_match_errors) > 0
                else float("nan")
            ),
        },
    )


def _position_colors(
    positions: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Map [N, 2] positions to RGB in [0, 1] using shared world-map bounds."""
    x_bounds, y_bounds = bounds
    x_span = max(float(x_bounds[1] - x_bounds[0]), 1e-8)
    y_span = max(float(y_bounds[1] - y_bounds[0]), 1e-8)
    normalized_x = np.clip((positions[:, 0] - x_bounds[0]) / x_span, 0.0, 1.0)
    normalized_y = np.clip((positions[:, 1] - y_bounds[0]) / y_span, 0.0, 1.0)
    red = normalized_x
    green = normalized_y
    blue = 1.0 - 0.5 * (normalized_x + normalized_y)
    return np.stack([red, green, blue], axis=1).astype(np.float32, copy=False)


def _embedding_bounds(embedding: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = float(np.min(embedding[:, 0])), float(np.max(embedding[:, 0]))
    y_min, y_max = float(np.min(embedding[:, 1])), float(np.max(embedding[:, 1]))
    x_padding = max((x_max - x_min) * 0.02, 1e-6)
    y_padding = max((y_max - y_min) * 0.02, 1e-6)
    return ((x_min - x_padding, x_max + x_padding), (y_min - y_padding, y_max + y_padding))


_HEADING_PANEL_TITLE = "Colored by heading"
_HEADING_COLORBAR_TICKS = [-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi]
_HEADING_COLORBAR_LABELS = ["W -180", "S -90", "E 0", "N 90", "W 180"]


def _scatter_axis(
    axis: plt.Axes,
    embedding: np.ndarray,
    color_values: np.ndarray,
    *,
    title: str,
    cmap: str,
    point_size: float,
    embedding_label: str = "UMAP",
) -> None:
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color_values,
        s=point_size,
        cmap=cmap,
        alpha=0.85,
        linewidths=0.0,
    )
    axis.set_title(title)
    axis.set_xlabel(f"{embedding_label} 1")
    axis.set_ylabel(f"{embedding_label} 2")
    axis.grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)
    if title == _HEADING_PANEL_TITLE:
        scatter.set_clim(-np.pi, np.pi)
        colorbar = plt.colorbar(
            scatter, ax=axis, fraction=0.046, pad=0.04, ticks=_HEADING_COLORBAR_TICKS
        )
        colorbar.ax.set_yticklabels(_HEADING_COLORBAR_LABELS)
        colorbar.set_label("heading (deg, 0=East/+x, CCW)", fontsize=8)
    else:
        plt.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)


def _write_embedding_figure(
    output_path: Path,
    embedding: np.ndarray,
    positions: np.ndarray,
    *,
    env_id: str,
    title_prefix: str,
    trustworthiness_score: float,
    third_panel_values: np.ndarray,
    third_panel_title: str,
    third_panel_cmap: str,
    embedding_label: str = "UMAP",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = resolve_plot_bounds(env_id, positions)
    if bounds is None:
        raise ValueError("Could not resolve embedding position plot bounds.")
    world_overlay = resolve_world_overlay(env_id)
    position_colors = _position_colors(positions, bounds)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    point_size = 10.0 if len(embedding) <= 2500 else 4.5

    axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=position_colors,
        s=point_size,
        alpha=0.88,
        linewidths=0.0,
    )
    axes[0].set_title(f"{embedding_label} colored by true XY")
    axes[0].set_xlabel(f"{embedding_label} 1")
    axes[0].set_ylabel(f"{embedding_label} 2")
    axes[0].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    axes[1].scatter(
        positions[:, 0],
        positions[:, 1],
        c=position_colors,
        s=point_size,
        alpha=0.88,
        linewidths=0.0,
    )
    if world_overlay is not None:
        draw_world_segments_on_axis(axes[1], world_overlay.segments, line_color="#404040")
        draw_landmarks_on_axis(axes[1], world_overlay)
    x_bounds, y_bounds = bounds
    finalize_arena_axis(axes[1], x_bounds=x_bounds, y_bounds=y_bounds, world_overlay=world_overlay)
    axes[1].set_title("Actual world position color key")
    axes[1].set_xlabel(POSITION_X_LABEL)
    axes[1].set_ylabel(POSITION_Y_LABEL)
    axes[1].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    _scatter_axis(
        axes[2],
        embedding,
        third_panel_values,
        title=third_panel_title,
        cmap=third_panel_cmap,
        point_size=point_size,
        embedding_label=embedding_label,
    )
    figure.suptitle(
        f"{title_prefix} | n={len(embedding)} | trustworthiness={trustworthiness_score:.3f}",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_world_by_representation_figure(
    output_path: Path,
    embedding: np.ndarray,
    positions: np.ndarray,
    *,
    env_id: str,
    title_prefix: str,
    embedding_label: str = "UMAP",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    world_bounds = resolve_plot_bounds(env_id, positions)
    if world_bounds is None:
        raise ValueError("Could not resolve embedding position plot bounds.")
    world_overlay = resolve_world_overlay(env_id)
    embedding_bounds = _embedding_bounds(embedding)
    embedding_colors = _position_colors(embedding, embedding_bounds)
    point_size = 10.0 if len(embedding) <= 2500 else 4.5

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=embedding_colors,
        s=point_size,
        alpha=0.88,
        linewidths=0.0,
    )
    axes[0].set_title(f"Representation {embedding_label} color key")
    axes[0].set_xlabel(f"{embedding_label} 1")
    axes[0].set_ylabel(f"{embedding_label} 2")
    axes[0].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    axes[1].scatter(
        positions[:, 0],
        positions[:, 1],
        c=embedding_colors,
        s=point_size,
        alpha=0.88,
        linewidths=0.0,
    )
    if world_overlay is not None:
        draw_world_segments_on_axis(axes[1], world_overlay.segments, line_color="#404040")
        draw_landmarks_on_axis(axes[1], world_overlay)
    x_bounds, y_bounds = world_bounds
    finalize_arena_axis(axes[1], x_bounds=x_bounds, y_bounds=y_bounds, world_overlay=world_overlay)
    axes[1].set_title("World map colored by representation")
    axes[1].set_xlabel(POSITION_X_LABEL)
    axes[1].set_ylabel(POSITION_Y_LABEL)
    axes[1].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    figure.suptitle(title_prefix, fontsize=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _add_transition_edges(
    axis: plt.Axes,
    coordinates: np.ndarray,
    edge_pairs: np.ndarray,
    edge_counts: np.ndarray,
) -> None:
    if len(edge_pairs) == 0:
        return
    segments = np.stack(
        [coordinates[edge_pairs[:, 0]], coordinates[edge_pairs[:, 1]]],
        axis=1,
    )
    log_counts = np.log1p(edge_counts)
    max_log_count = max(float(log_counts.max()), 1e-8)
    line_widths = 0.35 + 1.65 * log_counts / max_log_count
    axis.add_collection(
        LineCollection(
            segments,
            colors="#303030",
            linewidths=line_widths,
            alpha=0.22,
            zorder=1,
        )
    )


def _write_transition_graph_figure(
    output_path: Path,
    embedding: np.ndarray,
    positions: np.ndarray,
    pooled_counts: np.ndarray,
    transition_graph: _PooledTransitionGraph,
    *,
    env_id: str,
    title_prefix: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bounds = resolve_plot_bounds(env_id, positions)
    if bounds is None:
        raise ValueError("Could not resolve transition-graph position plot bounds.")
    world_overlay = resolve_world_overlay(env_id)
    position_colors = _position_colors(positions, bounds)
    log_occupancy = np.log1p(pooled_counts)
    max_log_occupancy = max(float(log_occupancy.max()), 1e-8)
    node_sizes = 18.0 + 58.0 * log_occupancy / max_log_occupancy

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    _add_transition_edges(
        axes[0], embedding, transition_graph.edge_pairs, transition_graph.edge_counts
    )
    axes[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=position_colors,
        s=node_sizes,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        zorder=2,
    )
    axes[0].set_title("Latent UMAP with experienced transitions")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    _add_transition_edges(
        axes[1], positions, transition_graph.edge_pairs, transition_graph.edge_counts
    )
    axes[1].scatter(
        positions[:, 0],
        positions[:, 1],
        c=position_colors,
        s=node_sizes,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        zorder=2,
    )
    if world_overlay is not None:
        draw_world_segments_on_axis(axes[1], world_overlay.segments, line_color="#404040")
        draw_landmarks_on_axis(axes[1], world_overlay)
    x_bounds, y_bounds = bounds
    finalize_arena_axis(axes[1], x_bounds=x_bounds, y_bounds=y_bounds, world_overlay=world_overlay)
    axes[1].set_title("Same transition graph in physical space")
    axes[1].set_xlabel(POSITION_X_LABEL)
    axes[1].set_ylabel(POSITION_Y_LABEL)
    axes[1].grid(color="#D6D6D6", linewidth=0.35, alpha=0.45)

    distance_gap = transition_graph.metrics["pooled_umap_transition_distance_gap"]
    figure.suptitle(
        f"{title_prefix} | original-space matched distance gap={distance_gap:.3f}",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.01,
        "UMAP coordinates use pooled population codes only; empirical transitions are overlaid "
        "after fitting. Node size shows occupancy; edge width shows transition count.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _run_pooled_embedding(
    analysis_input: AnalysisInput,
    output_dir: Path,
    config: dict[str, Any],
    *,
    embed_fn: Callable[[np.ndarray], np.ndarray],
    label: str,
    metric_prefix: str,
    embedding_label: str,
    figure_stem: str,
    num_bins_x: int,
    num_bins_y: int,
    min_count: int,
    max_points: int,
    trustworthiness_neighbors: int,
    random_seed: int,
    timing_metadata_key: str,
    timing_seconds: dict[str, float] | None = None,
    flattened_features: np.ndarray | None = None,
    flattened_positions: np.ndarray | None = None,
    require_pool: bool = True,
    include_feature_dim: bool = False,
    include_transition_graph: bool = False,
) -> AnalysisResult:
    """Pool codes by spatial bin, embed with embed_fn, score, and render figures."""
    timing_seconds = {} if timing_seconds is None else timing_seconds

    if flattened_features is None or flattened_positions is None:
        section_started_at = perf_counter()
        flattened_features, flattened_positions = flatten_valid_steps(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
        )
        record_timing(timing_seconds, "flatten_valid_steps", section_started_at)
        if len(flattened_features) < 4:
            raise ValueError(f"{figure_stem} requires at least 4 valid timesteps.")

    env_id = str(analysis_input.metadata.get("env_id", ""))
    module_dir = output_dir / figure_stem

    section_started_at = perf_counter()
    pooled = _pool_by_spatial_bin_with_metadata(
        analysis_input,
        flattened_features.astype(np.float32, copy=False),
        flattened_positions.astype(np.float32, copy=False),
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        min_count=min_count,
    )
    record_timing(timing_seconds, "pool_spatial_bins", section_started_at)
    if pooled is None:
        if require_pool:
            raise ValueError(f"{figure_stem} requires at least 4 populated spatial bins.")
        return AnalysisResult(
            metrics={},
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={timing_metadata_key: timing_seconds},
        )
    pooled_features, pooled_positions, pooled_counts, pooled_bin_indices, pooling_bounds = pooled

    section_started_at = perf_counter()
    pooled_indices = subsample_indices(len(pooled_features), max_points, random_seed)
    pooled_features = pooled_features[pooled_indices]
    pooled_positions = pooled_positions[pooled_indices]
    pooled_counts = pooled_counts[pooled_indices]
    pooled_bin_indices = pooled_bin_indices[pooled_indices]
    record_timing(timing_seconds, "subsample_pooled_bins", section_started_at)

    section_started_at = perf_counter()
    try:
        pooled_embedding = embed_fn(pooled_features)
    except (_DisconnectedKNNGraphError, _DegenerateRepresentationError, ArpackError) as error:
        record_timing(timing_seconds, metric_prefix, section_started_at)
        reason = _degenerate_skip_reason(error, embedder=label)
        logger.warning(
            "%s skipped for %s (%s): %s",
            figure_stem,
            analysis_input.source_name,
            analysis_input.split_name,
            reason,
        )
        skipped_metrics = {
            f"{metric_prefix}_skipped": 1.0,
            f"{metric_prefix}_trustworthiness": float("nan"),
            f"{metric_prefix}_points": float(len(pooled_features)),
            f"{metric_prefix}_mean_bin_count": float(pooled_counts.mean()),
        }
        if include_feature_dim:
            skipped_metrics[f"{metric_prefix}_feature_dim"] = float(pooled_features.shape[-1])
        return AnalysisResult(
            metrics=skipped_metrics,
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={
                timing_metadata_key: timing_seconds,
                "topology_manifold_skipped_reason": reason,
            },
        )
    record_timing(timing_seconds, metric_prefix, section_started_at)

    section_started_at = perf_counter()
    pooled_trustworthiness = compute_trustworthiness(
        pooled_features,
        pooled_embedding,
        neighbor_count=trustworthiness_neighbors,
    )
    record_timing(timing_seconds, "pooled_trustworthiness", section_started_at)

    pooled_figure_path = module_dir / (
        f"{figure_stem}__{analysis_input.source_name}__{analysis_input.split_name}.png"
    )
    section_started_at = perf_counter()
    _write_embedding_figure(
        pooled_figure_path,
        pooled_embedding,
        pooled_positions,
        env_id=env_id,
        title_prefix=f"{analysis_input.source_name} spatially pooled {label}",
        trustworthiness_score=pooled_trustworthiness,
        third_panel_values=pooled_counts,
        third_panel_title="Colored by pooled sample count",
        third_panel_cmap="magma",
        embedding_label=embedding_label,
    )
    record_timing(timing_seconds, "render_pooled_figure", section_started_at)

    pooled_world_by_representation_path = module_dir / (
        f"{figure_stem}_world_by_representation__"
        f"{analysis_input.source_name}__{analysis_input.split_name}.png"
    )
    section_started_at = perf_counter()
    _write_world_by_representation_figure(
        pooled_world_by_representation_path,
        pooled_embedding,
        pooled_positions,
        env_id=env_id,
        title_prefix=f"{analysis_input.source_name} pooled {label} representation map",
        embedding_label=embedding_label,
    )
    record_timing(timing_seconds, "render_pooled_world_by_representation", section_started_at)

    metrics = {
        f"{metric_prefix}_trustworthiness": pooled_trustworthiness,
        f"{metric_prefix}_points": float(len(pooled_features)),
        f"{metric_prefix}_mean_bin_count": float(pooled_counts.mean()),
    }
    if include_feature_dim:
        metrics[f"{metric_prefix}_feature_dim"] = float(pooled_features.shape[-1])
    figures = {
        f"{metric_prefix}_embedding": pooled_figure_path,
        f"{metric_prefix}_world_by_representation": pooled_world_by_representation_path,
    }
    if include_transition_graph:
        section_started_at = perf_counter()
        transition_graph = _build_pooled_transition_graph(
            analysis_input,
            pooled_features,
            pooled_positions,
            pooled_bin_indices,
            num_bins_x=num_bins_x,
            num_bins_y=num_bins_y,
            bounds=pooling_bounds,
            min_transition_count=int(config.get("umap_transition_min_count", 2)),
            representation_metric=str(config.get("umap_transition_metric", "correlation")),
            physical_match_caliper=float(
                config.get("umap_transition_physical_match_caliper", 0.25)
            ),
        )
        metrics.update(transition_graph.metrics)
        record_timing(timing_seconds, "pooled_transition_graph", section_started_at)

        transition_figure_path = module_dir / (
            f"{figure_stem}_transition_graph__"
            f"{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        section_started_at = perf_counter()
        _write_transition_graph_figure(
            transition_figure_path,
            pooled_embedding,
            pooled_positions,
            pooled_counts,
            transition_graph,
            env_id=env_id,
            title_prefix=f"{analysis_input.source_name} pooled UMAP",
        )
        record_timing(timing_seconds, "render_pooled_transition_graph", section_started_at)
        figures[f"{metric_prefix}_transition_graph"] = transition_figure_path
    return AnalysisResult(
        metrics=metrics,
        per_unit_metrics={},
        figures=figures,
        tables={},
        metadata={timing_metadata_key: timing_seconds},
    )


@dataclass(slots=True)
class _TopologyUMAPModuleBase:
    """Visualize whether local latent neighborhoods preserve spatial structure."""

    name: str = "topology_umap_bundle"
    cost_tier: str = "standard"
    emit_steps: bool = True
    emit_pooled: bool = True

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        section_started_at = perf_counter()
        flattened_features, flattened_positions = flatten_valid_steps(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
        )
        record_timing(timing_seconds, "flatten_valid_steps", section_started_at)
        if len(flattened_features) < 4:
            raise ValueError("topology_umap requires at least 4 valid timesteps.")

        env_id = str(analysis_input.metadata.get("env_id", ""))
        random_seed = int(config.get("umap_random_seed", 42))
        module_dir = output_dir / self.name
        metrics: dict[str, float] = {}
        figures: dict[str, Path] = {}
        metadata: dict[str, Any] = {}
        if self.emit_steps:
            section_started_at = perf_counter()
            step_indices = subsample_indices(
                len(flattened_features),
                max(4, int(config.get("umap_max_points", 4096))),
                random_seed,
            )
            step_features = flattened_features[step_indices].astype(np.float32, copy=False)
            step_positions = flattened_positions[step_indices].astype(np.float32, copy=False)
            flattened_heading = _resolve_heading_values(
                flatten_vector_field(analysis_input.heading, analysis_input.valid_mask)
            )
            if flattened_heading is None:
                step_third_panel = np.linspace(0.0, 1.0, num=len(step_features), dtype=np.float32)
                step_third_title = "Colored by sample order"
                step_third_cmap = "cividis"
            else:
                step_third_panel = flattened_heading[step_indices]
                step_third_title = _HEADING_PANEL_TITLE
                step_third_cmap = "twilight"
            record_timing(timing_seconds, "subsample_steps", section_started_at)

            section_started_at = perf_counter()
            try:
                step_embedding = _fit_umap_embedding(step_features, config)
            except (_DegenerateRepresentationError, ArpackError) as error:
                record_timing(timing_seconds, "step_umap", section_started_at)
                reason = _degenerate_skip_reason(error, embedder="UMAP")
                logger.warning(
                    "topology_umap_steps skipped for %s (%s): %s",
                    analysis_input.source_name,
                    analysis_input.split_name,
                    reason,
                )
                metrics.update(
                    {
                        "step_umap_skipped": 1.0,
                        "step_umap_trustworthiness": float("nan"),
                        "step_umap_points": float(len(step_features)),
                        "step_umap_feature_dim": float(step_features.shape[-1]),
                    }
                )
                metadata["topology_umap_steps_skipped_reason"] = reason
            else:
                record_timing(timing_seconds, "step_umap", section_started_at)
                section_started_at = perf_counter()
                step_trustworthiness = compute_trustworthiness(
                    step_features,
                    step_embedding,
                    neighbor_count=int(config.get("umap_trustworthiness_neighbors", 15)),
                )
                record_timing(timing_seconds, "step_trustworthiness", section_started_at)

                step_figure_path = module_dir / (
                    f"topology_umap_steps__{analysis_input.source_name}__"
                    f"{analysis_input.split_name}.png"
                )
                section_started_at = perf_counter()
                _write_embedding_figure(
                    step_figure_path,
                    step_embedding,
                    step_positions,
                    env_id=env_id,
                    title_prefix=f"{analysis_input.source_name} step-wise UMAP",
                    trustworthiness_score=step_trustworthiness,
                    third_panel_values=step_third_panel,
                    third_panel_title=step_third_title,
                    third_panel_cmap=step_third_cmap,
                )
                record_timing(timing_seconds, "render_step_figure", section_started_at)

                step_world_by_representation_path = module_dir / (
                    f"topology_umap_steps_world_by_representation__"
                    f"{analysis_input.source_name}__{analysis_input.split_name}.png"
                )
                section_started_at = perf_counter()
                _write_world_by_representation_figure(
                    step_world_by_representation_path,
                    step_embedding,
                    step_positions,
                    env_id=env_id,
                    title_prefix=f"{analysis_input.source_name} step-wise representation map",
                )
                record_timing(
                    timing_seconds,
                    "render_step_world_by_representation",
                    section_started_at,
                )

                metrics.update(
                    {
                        "step_umap_trustworthiness": step_trustworthiness,
                        "step_umap_points": float(len(step_features)),
                        "step_umap_feature_dim": float(step_features.shape[-1]),
                    }
                )
                figures["step_umap_embedding"] = step_figure_path
                figures["step_umap_world_by_representation"] = step_world_by_representation_path

        pooled_mode = str(config.get("umap_pool_mode", "spatial_bins"))
        if self.emit_pooled and pooled_mode == "spatial_bins":
            pooled_result = _run_pooled_embedding(
                analysis_input,
                output_dir,
                config,
                embed_fn=lambda features: _fit_umap_embedding(features, config),
                label="UMAP",
                metric_prefix="pooled_umap",
                embedding_label="UMAP",
                figure_stem="topology_umap_pooled",
                num_bins_x=int(config.get("umap_pool_num_bins_x", 20)),
                num_bins_y=int(config.get("umap_pool_num_bins_y", 20)),
                min_count=int(config.get("umap_pool_min_count", 2)),
                max_points=max(4, int(config.get("umap_max_points", 4096))),
                trustworthiness_neighbors=int(config.get("umap_trustworthiness_neighbors", 15)),
                random_seed=random_seed,
                timing_metadata_key="topology_umap_timing_seconds",
                timing_seconds=timing_seconds,
                flattened_features=flattened_features,
                flattened_positions=flattened_positions,
                require_pool=False,
                include_transition_graph=True,
            )
            metrics.update(pooled_result.metrics)
            figures.update(pooled_result.figures)
            skipped_reason = pooled_result.metadata.get("topology_manifold_skipped_reason")
            if skipped_reason is not None:
                metadata["topology_umap_pooled_skipped_reason"] = skipped_reason

        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={},
            figures=figures,
            tables={},
            metadata={"topology_umap_timing_seconds": timing_seconds, **metadata},
        )


@dataclass(slots=True)
class TopologyUMAPStepsModule(_TopologyUMAPModuleBase):
    """Visualize sampled valid timesteps with UMAP."""

    name: str = "topology_umap_steps"
    emit_pooled: bool = False


@dataclass(slots=True)
class TopologyUMAPPooledModule(_TopologyUMAPModuleBase):
    """Visualize spatially pooled representations with UMAP."""

    name: str = "topology_umap_pooled"
    emit_steps: bool = False


@dataclass(slots=True)
class _PooledManifoldEmbeddingModuleBase:
    """Embed spatially pooled population codes with a distance-faithful manifold method."""

    name: str = "topology_manifold_pooled"
    cost_tier: str = "standard"
    method: str = "mds"
    embedding_label: str = "MDS"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}
        result = _run_pooled_embedding(
            analysis_input,
            output_dir,
            config,
            embed_fn=lambda features: _fit_pooled_manifold_embedding(
                features, method=self.method, config=config
            ),
            label=self.embedding_label,
            metric_prefix=f"pooled_{self.method}",
            embedding_label=self.embedding_label,
            figure_stem=self.name,
            num_bins_x=int(config.get("manifold_pool_num_bins_x", 20)),
            num_bins_y=int(config.get("manifold_pool_num_bins_y", 20)),
            min_count=int(config.get("manifold_pool_min_count", 2)),
            max_points=max(4, int(config.get("manifold_max_points", 4096))),
            trustworthiness_neighbors=int(config.get("manifold_trustworthiness_neighbors", 15)),
            random_seed=int(config.get("manifold_random_seed", 42)),
            timing_metadata_key="topology_manifold_timing_seconds",
            timing_seconds=timing_seconds,
            require_pool=True,
            include_feature_dim=True,
        )
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )
        return result


@dataclass(slots=True)
class TopologyMDSPooledModule(_PooledManifoldEmbeddingModuleBase):
    """Metric MDS on spatially pooled codes: the distance-preserving geometry view."""

    name: str = "topology_mds_pooled"
    method: str = "mds"
    embedding_label: str = "MDS"


@dataclass(slots=True)
class TopologyIsomapPooledModule(_PooledManifoldEmbeddingModuleBase):
    """Isomap on spatially pooled codes: geodesic-distance geometry along the data graph."""

    name: str = "topology_isomap_pooled"
    method: str = "isomap"
    embedding_label: str = "Isomap"
