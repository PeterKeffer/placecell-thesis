"""Object-vector (OVC) and boundary-vector (BVC) cell metrics + visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..numerics.rate_map_kernels import flatten_valid_steps  # noqa: E402
from .base import AnalysisInput, AnalysisResult  # noqa: E402
from .reanchoring_gridness import benjamini_hochberg  # noqa: E402
from .world_overlay import resolve_world_overlay  # noqa: E402


def _point_segment_vectors(
    points: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to a segment, and the foot (nearest point on the segment)."""
    ab = seg_b - seg_a
    ab_sq = float(ab @ ab)
    if ab_sq <= 1e-12:
        foot = np.broadcast_to(seg_a, points.shape)
    else:
        t = np.clip(((points - seg_a) @ ab) / ab_sq, 0.0, 1.0)
        foot = seg_a + t[:, None] * ab
    diff = points - foot
    return np.sqrt((diff**2).sum(axis=1)), foot


def _nearest_boundary_vectors(
    points: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each point: distance to nearest wall and ALLOCENTRIC bearing from wall to the agent."""
    best_distance = np.full(points.shape[0], np.inf)
    best_foot = np.zeros_like(points)
    for seg_a, seg_b in segments:
        distance, foot = _point_segment_vectors(points, np.asarray(seg_a), np.asarray(seg_b))
        closer = distance < best_distance
        best_distance[closer] = distance[closer]
        best_foot[closer] = foot[closer]
    vector = points - best_foot
    return best_distance, np.arctan2(vector[:, 1], vector[:, 0])


def _nearest_object_vectors(
    points: np.ndarray, landmarks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each point: distance to nearest object and ALLOCENTRIC bearing from object to agent."""
    difference = points[:, None, :] - landmarks[None, :, :]
    distance = np.sqrt((difference**2).sum(axis=-1))
    nearest = np.argmin(distance, axis=1)
    rows = np.arange(points.shape[0])
    vector = difference[rows, nearest]
    return distance[rows, nearest], np.arctan2(vector[:, 1], vector[:, 0])


def _vector_rate_maps(
    distances: np.ndarray,
    angles: np.ndarray,
    activations: np.ndarray,
    *,
    num_distance_bins: int,
    num_angle_bins: int,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit mean rate in (distance, allocentric-angle) bins."""
    distance_bin = np.clip(
        (distances / max(max_distance, 1e-9) * num_distance_bins).astype(np.int64),
        0,
        num_distance_bins - 1,
    )
    angle_fraction = (angles + np.pi) / (2.0 * np.pi)
    angle_bin = (angle_fraction * num_angle_bins).astype(np.int64) % num_angle_bins
    flat_bin = distance_bin * num_angle_bins + angle_bin
    total_bins = num_distance_bins * num_angle_bins
    occupancy = np.bincount(flat_bin, minlength=total_bins).astype(np.float64)
    rectified = np.clip(activations, a_min=0.0, a_max=None)
    sums = np.zeros((total_bins, rectified.shape[1]), dtype=np.float64)
    np.add.at(sums, flat_bin, rectified)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(occupancy[:, None] > 0, sums / occupancy[:, None], 0.0)
    maps = mean.T.reshape(rectified.shape[1], num_distance_bins, num_angle_bins)
    return maps, occupancy.reshape(num_distance_bins, num_angle_bins)


def _spatial_information(maps: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Skaggs spatial information (bits) of each unit over the vector-space occupancy."""
    probability = occupancy.ravel() / max(occupancy.sum(), 1e-9)
    rates = maps.reshape(maps.shape[0], -1)
    mean_rate = (rates * probability[None, :]).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(mean_rate > 0, rates / mean_rate, 0.0)
        log_ratio = np.log2(ratio, where=ratio > 0, out=np.zeros_like(ratio))
        contribution = probability[None, :] * rates * log_ratio
    return contribution.sum(axis=1).astype(np.float32)


def _split_half_reliability(
    distances: np.ndarray,
    angles: np.ndarray,
    activations: np.ndarray,
    *,
    num_distance_bins: int,
    num_angle_bins: int,
    max_distance: float,
) -> np.ndarray:
    """Per-unit correlation between vector maps built from even- and odd-indexed samples."""
    even = np.arange(distances.shape[0]) % 2 == 0
    binning = dict(
        num_distance_bins=num_distance_bins,
        num_angle_bins=num_angle_bins,
        max_distance=max_distance,
    )
    maps_a, occupancy_a = _vector_rate_maps(
        distances[even], angles[even], activations[even], **binning
    )
    maps_b, occupancy_b = _vector_rate_maps(
        distances[~even], angles[~even], activations[~even], **binning
    )
    shared = ((occupancy_a > 0) & (occupancy_b > 0)).ravel()
    if shared.sum() < 3:
        return np.zeros(activations.shape[1], dtype=np.float32)
    flat_a = maps_a.reshape(maps_a.shape[0], -1)[:, shared]
    flat_b = maps_b.reshape(maps_b.shape[0], -1)[:, shared]
    reliability = np.zeros(activations.shape[1], dtype=np.float32)
    for unit in range(flat_a.shape[0]):
        if flat_a[unit].std() < 1e-9 or flat_b[unit].std() < 1e-9:
            continue
        reliability[unit] = float(np.corrcoef(flat_a[unit], flat_b[unit])[0, 1])
    return reliability


class _VectorScoreCore:
    """Shared OVC/BVC scoring; reference_kind picks walls vs objects."""

    def __init__(self, *, name: str, reference_kind: str, cost_tier: str = "standard") -> None:
        self.name = name
        self.reference_kind = reference_kind
        self.cost_tier = cost_tier

    def required_representations(self) -> set[str]:
        return set()

    def _reference_geometry(self, env_id: str, env_kwargs=None):
        overlay = resolve_world_overlay(env_id, env_kwargs)
        if overlay is None:
            return None
        if self.reference_kind == "boundary":
            return np.asarray(overlay.segments, dtype=np.float64) if overlay.segments else None
        points = [point for layer in overlay.landmarks for point in layer.positions]
        return np.asarray(points, dtype=np.float64) if points else None

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        env_id = str(analysis_input.metadata.get("env_id", ""))
        geometry = self._reference_geometry(env_id, analysis_input.metadata.get("env_kwargs"))
        activations, positions = flatten_valid_steps(
            analysis_input.representation, analysis_input.position_xy, analysis_input.valid_mask
        )
        num_units = int(analysis_input.representation.shape[-1])
        empty = AnalysisResult(
            metrics={f"{self.name}_units_scored": 0.0},
            per_unit_metrics={},
            figures={},
            tables={},
            metadata={"skipped_reason": f"no {self.reference_kind} geometry for env_id={env_id!r}"},
        )
        if geometry is None or activations.shape[0] < 16:
            return empty

        if self.reference_kind == "boundary":
            distances, angles = _nearest_boundary_vectors(positions, geometry)
        else:
            distances, angles = _nearest_object_vectors(positions, geometry)

        num_distance_bins = int(config.get("vector_num_distance_bins", 12))
        num_angle_bins = int(config.get("vector_num_angle_bins", 24))
        max_distance = float(config.get("vector_max_distance", 0.0)) or float(
            np.percentile(distances, 95.0)
        )
        binning = dict(
            num_distance_bins=num_distance_bins,
            num_angle_bins=num_angle_bins,
            max_distance=max_distance,
        )
        maps, occupancy = _vector_rate_maps(distances, angles, activations, **binning)
        spatial_information = _spatial_information(maps, occupancy)
        reliability = _split_half_reliability(distances, angles, activations, **binning)

        flat_peak = maps.reshape(num_units, -1).argmax(axis=1)
        peak_distance_bin = flat_peak // num_angle_bins
        peak_distance = (peak_distance_bin + 0.5) / num_distance_bins * max_distance
        peak_direction = ((flat_peak % num_angle_bins) + 0.5) / num_angle_bins * 2.0 * np.pi - np.pi

        min_distance_fraction = float(config.get("vector_min_peak_distance_fraction", 0.15))
        min_peak_distance = min_distance_fraction * max_distance
        reliability_threshold = float(config.get("vector_reliability_threshold", 0.4))
        significant = self._shuffle_significance(
            distances, angles, activations, spatial_information, config=config, binning=binning
        )
        significance_gate = np.ones(num_units, dtype=bool) if significant is None else significant

        is_reliable = reliability >= reliability_threshold
        is_vector = is_reliable & (peak_distance >= min_peak_distance) & significance_gate
        is_border_like = is_reliable & (peak_distance < min_peak_distance)

        figure_path = self._render(
            output_dir,
            analysis_input,
            maps=maps,
            occupancy=occupancy,
            reliability=reliability,
            peak_distance=peak_distance,
            peak_direction=peak_direction,
            is_vector=is_vector,
            max_distance=max_distance,
        )

        return AnalysisResult(
            metrics={
                f"{self.name}_units_scored": float(num_units),
                f"mean_{self.name}_reliability": float(reliability.mean()),
                f"max_{self.name}_reliability": float(reliability.max()),
                f"mean_{self.name}_spatial_info": float(spatial_information.mean()),
                f"max_{self.name}_spatial_info": float(spatial_information.max()),
                f"fraction_{self.reference_kind}_vector_cells": float(is_vector.mean()),
                f"fraction_{self.reference_kind}_border_like": float(is_border_like.mean()),
                f"num_{self.name}_significant": (
                    float("nan") if significant is None else float(int(significant.sum()))
                ),
                f"mean_{self.name}_peak_distance": float(peak_distance.mean()),
                f"{self.name}_reference_count": float(len(geometry)),
            },
            per_unit_metrics={
                f"{self.name}_reliability": reliability,
                f"{self.name}_spatial_info": spatial_information,
                f"{self.name}_peak_distance": peak_distance.astype(np.float32),
                f"{self.name}_peak_direction": peak_direction.astype(np.float32),
            },
            figures={self.name: figure_path} if figure_path is not None else {},
            tables={},
            metadata={
                "max_distance": max_distance,
                "reference_count": int(len(geometry)),
                "shuffle_significance_tested": significant is not None,
            },
        )

    def _shuffle_significance(
        self, distances, angles, activations, spatial_information, *, config, binning
    ) -> np.ndarray | None:
        """Roll-null on spatial info for the top candidates only (cost-bounded), BH-FDR."""
        shuffle_count = int(config.get("vector_shuffle_count", 0))
        if shuffle_count <= 0 or activations.shape[0] < 32:
            return None
        num_units = activations.shape[1]
        top_k = min(int(config.get("vector_shuffle_top_k", 32)), num_units)
        candidates = np.argsort(spatial_information)[::-1][:top_k]
        candidate_activations = activations[:, candidates]
        sample_count = activations.shape[0]
        null_max = np.zeros(shuffle_count, dtype=np.float64)
        stride = sample_count // (shuffle_count + 1) + 1
        for shuffle_index in range(shuffle_count):
            offset = 1 + (shuffle_index * stride) % (sample_count - 1)
            rolled = np.roll(candidate_activations, offset, axis=0)
            rolled_maps, rolled_occupancy = _vector_rate_maps(distances, angles, rolled, **binning)
            null_max[shuffle_index] = _spatial_information(rolled_maps, rolled_occupancy).max()
        threshold = float(np.percentile(null_max, 95.0))
        p_values = np.ones(num_units, dtype=np.float64)
        for candidate in candidates:
            exceedances = int((null_max >= spatial_information[candidate]).sum())
            p_values[candidate] = (1.0 + exceedances) / (1.0 + shuffle_count)
        significant = np.zeros(num_units, dtype=bool)
        significant[candidates] = benjamini_hochberg(p_values[candidates])
        significant &= spatial_information > threshold
        return significant

    def _render(
        self,
        output_dir: Path,
        analysis_input: AnalysisInput,
        *,
        maps,
        occupancy,
        reliability,
        peak_distance,
        peak_direction,
        is_vector,
        max_distance,
    ) -> Path | None:
        order = np.argsort(reliability)[::-1]
        top = order[:11]
        figure, axes = plt.subplots(3, 4, figsize=(13, 9))
        flat_axes = axes.ravel()
        extent = (-180.0, 180.0, 0.0, float(max_distance))
        for panel_index, unit in enumerate(top):
            axis = flat_axes[panel_index]
            axis.imshow(maps[unit], origin="lower", aspect="auto", extent=extent, cmap="Reds")
            mark = "VECTOR" if is_vector[unit] else "-"
            axis.set_title(
                f"u{unit} r={reliability[unit]:.2f} d*={peak_distance[unit]:.1f} {mark}", fontsize=8
            )
            axis.set_xlabel("allocentric bearing (deg)", fontsize=7)
            axis.set_ylabel("distance", fontsize=7)
            axis.tick_params(labelsize=6)
        summary = flat_axes[11]
        colors = np.where(is_vector, "tab:red", "0.6")
        summary.scatter(peak_distance, reliability, s=10, c=colors)
        summary.axhline(0.4, color="0.4", linewidth=0.8, linestyle="--")
        summary.set_xlabel("peak distance to reference", fontsize=7)
        summary.set_ylabel("split-half reliability", fontsize=7)
        summary.set_title(f"{is_vector.sum()} {self.reference_kind}-vector cells", fontsize=8)
        summary.tick_params(labelsize=6)
        figure.suptitle(
            f"{self.name}  {analysis_input.label}  ({analysis_input.split_name})  "
            f"top units by (distance x bearing) reliability",
            fontsize=10,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        path = output_dir / f"{self.name}__{analysis_input.split_name}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        return path


class BoundaryVectorScoreModule(_VectorScoreCore):
    """Boundary-vector cells (BVC): tuned to (distance, allocentric direction) from walls."""

    def __init__(self, name: str = "boundary_vector_score") -> None:
        super().__init__(name=name, reference_kind="boundary")


class ObjectVectorScoreModule(_VectorScoreCore):
    """Object-vector cells (OVC): tuned to (distance, allocentric direction) from objects."""

    def __init__(self, name: str = "object_vector_score") -> None:
        super().__init__(name=name, reference_kind="object")
