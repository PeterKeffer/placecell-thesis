"""Head-direction tuning with a spatial-independence gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..numerics.circular import _EPS, circular_vector, peak_to_mean
from ..numerics.place_cell_quality import benjamini_hochberg
from ..numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    scatter_add_over_units,
)
from ..numerics.work_blocks import run_over_index_blocks
from .base import AnalysisInput, AnalysisResult
from .directionality import _valid_steps_per_episode
from .helpers import write_csv
from .shift_nulls import _circular_shift_step_layout, _draw_circular_shift_offsets
from .world_overlay import overlay_bounds, resolve_world_overlay

_TWO_PI = 2.0 * np.pi
_NULL_SCRATCH_BYTES = 128 * 1024 * 1024
_NULL_MIN_SHIFT_FRACTION = 0.05
_NULL_RNG_SEED = 0


@dataclass(slots=True)
class HeadDirectionTuningModule:
    """Per-unit HD candidates: heading selectivity that generalizes across place."""

    name: str = "head_direction_tuning"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        num_units = int(analysis_input.representation.shape[-1])
        num_heading_bins = int(config.get("head_direction_num_bins", 36))
        num_bins_x = int(config["head_direction_num_bins_x"])
        num_bins_y = int(config["head_direction_num_bins_y"])
        vector_length_threshold = float(config.get("head_direction_vector_length_threshold", 0.5))
        spatial_coverage_threshold = float(
            config.get("head_direction_spatial_coverage_threshold", 0.2)
        )
        position_invariance_threshold = float(
            config.get("head_direction_position_invariance_threshold", 0.6)
        )
        hd_cell_score_threshold = float(config.get("head_direction_hd_cell_score_threshold", 0.1))
        min_active_spatial_bins = int(config.get("head_direction_min_active_spatial_bins", 4))

        if analysis_input.heading is None:
            per_unit = _empty_per_unit_metrics(num_units)
        else:
            world_overlay = resolve_world_overlay(
                str(analysis_input.metadata.get("env_id", "")),
                analysis_input.metadata.get("env_kwargs"),
            )
            world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
            per_unit = _head_direction_metrics_per_unit(
                representation=analysis_input.representation,
                position_xy=analysis_input.position_xy,
                heading=analysis_input.heading,
                valid_mask=analysis_input.valid_mask,
                num_heading_bins=num_heading_bins,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                bounds=world_bounds,
                min_occupancy_per_bin=int(config.get("head_direction_min_occupancy_per_bin", 5)),
                min_heading_bins=int(config["head_direction_min_heading_bins"]),
                min_heading_quadrants=int(config.get("head_direction_min_heading_quadrants", 3)),
                min_active_spatial_bins=min_active_spatial_bins,
                active_threshold_fraction=float(config["head_direction_active_threshold_fraction"]),
                min_fire_rate=float(config.get("head_direction_min_fire_rate", 0.01)),
                vector_length_threshold=vector_length_threshold,
                spatial_coverage_threshold=spatial_coverage_threshold,
                position_invariance_threshold=position_invariance_threshold,
                hd_cell_score_threshold=hd_cell_score_threshold,
                num_null_shuffles=int(config.get("head_direction_null_shuffles", 999)),
            )

        assessable = per_unit["assessable"].astype(bool, copy=False)
        num_assessable = int(np.count_nonzero(assessable))
        finite_vector_lengths = per_unit["heading_vector_length"][assessable]
        metrics = {
            "median_heading_vector_length": (
                float(np.median(finite_vector_lengths)) if num_assessable else float("nan")
            ),
            "mean_heading_vector_length": (
                float(finite_vector_lengths.mean()) if num_assessable else float("nan")
            ),
            "mean_position_invariance_score": _mean_assessable(
                per_unit["position_invariance_score"], assessable
            ),
            "mean_spatial_coverage_fraction": _mean_assessable(
                per_unit["spatial_coverage_fraction"], assessable
            ),
            "fraction_hd_candidate": _fraction_assessable(per_unit["is_hd_candidate"], assessable),
            "fraction_conjunctive_candidate": _fraction_assessable(
                per_unit["is_conjunctive_candidate"], assessable
            ),
            "mean_heading_vector_length_excess": _mean_assessable(
                per_unit["heading_vector_length_excess"], assessable
            ),
            "fraction_heading_selective_significant": _fraction_assessable(
                per_unit["heading_vector_length_significant"], assessable
            ),
            "fraction_hd_candidate_significant": _fraction_assessable(
                per_unit["is_hd_candidate_significant"], assessable
            ),
        }
        table_path = _write_per_unit_table(output_dir, analysis_input, per_unit)
        figures: dict[str, Path] = {}
        if analysis_input.heading is not None and num_assessable > 0:
            rose_path = _render_preferred_direction_rose(
                output_dir,
                analysis_input,
                preferred_heading_rad=per_unit["preferred_heading_rad"],
                heading_vector_length=per_unit["heading_vector_length"],
                assessable=assessable,
                strong_threshold=vector_length_threshold,
            )
            if rose_path is not None:
                figures["preferred_direction_rose"] = rose_path
        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics=per_unit,
            figures=figures,
            tables={"per_unit_metrics": table_path},
            metadata={
                "head_direction_assessable_unit_count": num_assessable,
                "head_direction_total_unit_count": num_units,
                "head_direction_num_bins": num_heading_bins,
                "head_direction_num_bins_x": num_bins_x,
                "head_direction_num_bins_y": num_bins_y,
                "head_direction_vector_length_threshold": vector_length_threshold,
                "head_direction_spatial_coverage_threshold": spatial_coverage_threshold,
                "head_direction_position_invariance_threshold": position_invariance_threshold,
                "head_direction_hd_cell_score_threshold": hd_cell_score_threshold,
                "head_direction_min_active_spatial_bins": min_active_spatial_bins,
            },
        )


def _mean_assessable(values: np.ndarray, assessable: np.ndarray) -> float:
    selected = values[assessable]
    selected = selected[np.isfinite(selected)]
    return float(selected.mean()) if selected.size else float("nan")


def _fraction_assessable(values: np.ndarray, assessable: np.ndarray) -> float:
    selected = values[assessable].astype(np.float64, copy=False)
    return float(selected.mean()) if selected.size else float("nan")


def _empty_per_unit_metrics(num_units: int) -> dict[str, np.ndarray]:
    nan_values = np.full(num_units, np.nan, dtype=np.float32)
    zero_values = np.zeros(num_units, dtype=np.float32)
    return {
        "heading_vector_length": nan_values.copy(),
        "heading_vector_length_null_mean": nan_values.copy(),
        "heading_vector_length_null_95": nan_values.copy(),
        "heading_vector_length_excess": nan_values.copy(),
        "heading_vector_length_null_p": nan_values.copy(),
        "heading_vector_length_significant": zero_values.copy(),
        "preferred_heading_rad": nan_values.copy(),
        "heading_peak_to_mean": nan_values.copy(),
        "position_invariance_score": nan_values.copy(),
        "spatial_coverage_fraction": nan_values.copy(),
        "active_spatial_bin_count": zero_values.copy(),
        "normalized_spatial_information": nan_values.copy(),
        "hd_cell_score": nan_values.copy(),
        "assessable": zero_values.copy(),
        "is_hd_candidate": zero_values.copy(),
        "is_hd_candidate_significant": zero_values.copy(),
        "is_conjunctive_candidate": zero_values.copy(),
    }


def _head_direction_metrics_per_unit(
    *,
    representation: np.ndarray,
    position_xy: np.ndarray,
    heading: np.ndarray,
    valid_mask: np.ndarray | None,
    num_heading_bins: int,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    min_occupancy_per_bin: int,
    min_heading_bins: int,
    min_heading_quadrants: int,
    min_active_spatial_bins: int,
    active_threshold_fraction: float,
    min_fire_rate: float,
    vector_length_threshold: float,
    spatial_coverage_threshold: float,
    position_invariance_threshold: float,
    hd_cell_score_threshold: float,
    num_null_shuffles: int,
) -> dict[str, np.ndarray]:
    num_units = int(representation.shape[-1])
    per_unit = _empty_per_unit_metrics(num_units)
    codes = representation.reshape(-1, num_units)
    positions = position_xy.reshape(-1, 2)
    headings = heading.reshape(-1)
    if valid_mask is not None:
        keep = valid_mask.reshape(-1).astype(bool, copy=False)
        codes, positions, headings = codes[keep], positions[keep], headings[keep]
    if codes.shape[0] == 0:
        return per_unit

    rates = np.clip(codes, 0.0, None).astype(np.float64)
    fire_rate = (np.abs(codes) > 1e-4).mean(axis=0)
    heading_bins = (
        np.floor((headings % _TWO_PI) / (_TWO_PI / num_heading_bins)).astype(int)
    ) % num_heading_bins
    heading_centers = (
        (np.arange(num_heading_bins, dtype=np.float64) + 0.5) * _TWO_PI
    ) / num_heading_bins
    heading_vectors = np.exp(1j * heading_centers)
    heading_bin_quadrant = np.floor((heading_centers % _TWO_PI) / (np.pi / 2.0)).astype(int) % 4

    spatial_bins, _, _, _ = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    num_spatial_bins = num_bins_x * num_bins_y

    combined_bins = spatial_bins * num_heading_bins + heading_bins
    combined_occupancy = (
        np.bincount(combined_bins, minlength=num_spatial_bins * num_heading_bins)
        .astype(np.float64)
        .reshape(num_spatial_bins, num_heading_bins)
    )
    combined_activity = scatter_add_over_units(
        combined_bins, rates, num_spatial_bins * num_heading_bins
    ).reshape(num_spatial_bins, num_heading_bins, num_units)

    occupied_spatial = combined_occupancy.sum(axis=1)
    occupied_spatial_mask = occupied_spatial >= float(min_occupancy_per_bin)
    occupied_spatial_count = int(np.count_nonzero(occupied_spatial_mask))
    heading_occupancy = combined_occupancy.sum(axis=0)
    sampled_heading_mask = heading_occupancy >= float(min_occupancy_per_bin)

    with np.errstate(divide="ignore", invalid="ignore"):
        heading_mean = combined_activity.sum(axis=0) / heading_occupancy[:, None]
        spatial_mean = combined_activity.sum(axis=1) / occupied_spatial[:, None]
        local_heading_mean = combined_activity / combined_occupancy[:, :, None]

    local_sampled_heading = combined_occupancy >= float(min_occupancy_per_bin)
    heading_resolved_bins = local_sampled_heading.sum(axis=1) >= min_heading_bins

    for unit_index in range(num_units):
        if fire_rate[unit_index] < min_fire_rate:
            continue
        valid_heading_bins = sampled_heading_mask & np.isfinite(heading_mean[:, unit_index])
        if int(np.count_nonzero(valid_heading_bins)) < min_heading_bins:
            continue
        covered_quadrants = int(np.unique(heading_bin_quadrant[valid_heading_bins]).size)
        if covered_quadrants < int(min_heading_quadrants):
            continue
        unit_heading_rates = heading_mean[valid_heading_bins, unit_index]
        vector_length, preferred_heading = circular_vector(
            unit_heading_rates,
            heading_vectors[valid_heading_bins],
        )
        if not np.isfinite(vector_length):
            continue
        per_unit["heading_vector_length"][unit_index] = vector_length
        per_unit["preferred_heading_rad"][unit_index] = preferred_heading
        per_unit["heading_peak_to_mean"][unit_index] = peak_to_mean(unit_heading_rates)
        per_unit["assessable"][unit_index] = 1.0

        unit_spatial_mean = spatial_mean[:, unit_index]
        finite_spatial = occupied_spatial_mask & np.isfinite(unit_spatial_mean)
        if not np.any(finite_spatial):
            continue
        peak_spatial_rate = float(np.nanmax(unit_spatial_mean[finite_spatial]))
        if peak_spatial_rate <= _EPS:
            continue
        active_spatial = finite_spatial & (
            unit_spatial_mean >= active_threshold_fraction * peak_spatial_rate
        )
        active_count = int(np.count_nonzero(active_spatial))
        spatial_coverage = active_count / max(1, occupied_spatial_count)
        per_unit["active_spatial_bin_count"][unit_index] = float(active_count)
        per_unit["spatial_coverage_fraction"][unit_index] = float(spatial_coverage)
        normalized_spatial_information = _normalized_spatial_information(
            unit_spatial_mean,
            occupied_spatial,
            finite_spatial,
        )
        per_unit["normalized_spatial_information"][unit_index] = normalized_spatial_information

        position_invariance = _position_invariance_score(
            local_heading_mean[:, :, unit_index],
            local_sampled_heading,
            heading_resolved_bins,
            active_spatial,
            unit_spatial_mean,
            heading_vectors,
            min_active_spatial_bins=min_active_spatial_bins,
        )
        per_unit["position_invariance_score"][unit_index] = position_invariance
        finite_invariance = position_invariance if np.isfinite(position_invariance) else 0.0
        hd_cell_score = float(
            vector_length
            * spatial_coverage
            * finite_invariance
            * (1.0 - normalized_spatial_information)
        )
        per_unit["hd_cell_score"][unit_index] = hd_cell_score

        heading_selective = vector_length >= vector_length_threshold
        spatially_broad = (
            spatial_coverage >= spatial_coverage_threshold
            and active_count >= min_active_spatial_bins
        )
        position_independent = position_invariance >= position_invariance_threshold
        score_high = hd_cell_score >= hd_cell_score_threshold
        is_hd = bool(heading_selective and spatially_broad and position_independent and score_high)
        per_unit["is_hd_candidate"][unit_index] = float(is_hd)
        per_unit["is_conjunctive_candidate"][unit_index] = float(heading_selective and not is_hd)

    assessable_units = per_unit["assessable"].astype(bool, copy=False)
    if num_null_shuffles > 0 and assessable_units.any():
        observed_lengths = per_unit["heading_vector_length"]
        null_matrix = _heading_vector_length_null_matrix(
            rates,
            heading_bins=heading_bins,
            heading_occupancy=heading_occupancy,
            sampled_heading_mask=sampled_heading_mask,
            heading_vectors=heading_vectors,
            episode_lengths=_valid_steps_per_episode(representation, valid_mask),
            num_null_shuffles=num_null_shuffles,
        )
        null_finite = np.isfinite(null_matrix)
        draw_counts = null_finite.sum(axis=0)
        calibrated = assessable_units & np.isfinite(observed_lengths) & (draw_counts > 0)
        exceed_counts = np.sum(null_finite & (null_matrix >= observed_lengths[None, :]), axis=0)
        per_unit["heading_vector_length_null_p"][calibrated] = (
            (1.0 + exceed_counts[calibrated]) / (1.0 + draw_counts[calibrated])
        ).astype(np.float32)
        for unit_index in np.flatnonzero(calibrated):
            unit_null = null_matrix[null_finite[:, unit_index], unit_index]
            per_unit["heading_vector_length_null_mean"][unit_index] = float(unit_null.mean())
            per_unit["heading_vector_length_null_95"][unit_index] = float(
                np.percentile(unit_null, 95.0)
            )
        per_unit["heading_vector_length_excess"] = (
            observed_lengths - per_unit["heading_vector_length_null_mean"]
        )
        significant = benjamini_hochberg(per_unit["heading_vector_length_null_p"])
        per_unit["heading_vector_length_significant"] = significant.astype(np.float32, copy=False)
        per_unit["is_hd_candidate_significant"] = (
            per_unit["is_hd_candidate"].astype(bool, copy=False) & significant
        ).astype(np.float32, copy=False)

    return per_unit


def _heading_vector_length_null_matrix(
    rates: np.ndarray,
    *,
    heading_bins: np.ndarray,
    heading_occupancy: np.ndarray,
    sampled_heading_mask: np.ndarray,
    heading_vectors: np.ndarray,
    episode_lengths: np.ndarray,
    num_null_shuffles: int,
) -> np.ndarray:
    """[num_null_shuffles, num_units] resultant lengths under the circular-shift null."""
    num_samples, num_units = rates.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_occupancy = np.where(sampled_heading_mask, 1.0 / heading_occupancy, 0.0)
    bin_weights = np.stack(
        [
            inverse_occupancy,
            inverse_occupancy * heading_vectors.real,
            inverse_occupancy * heading_vectors.imag,
        ]
    )

    rng = np.random.default_rng(_NULL_RNG_SEED)
    shuffle_offsets = [
        _draw_circular_shift_offsets(episode_lengths, rng, _NULL_MIN_SHIFT_FRACTION)
        for _ in range(num_null_shuffles)
    ]
    step_layout = _circular_shift_step_layout(episode_lengths, np.arange(num_samples))
    null_matrix = np.full((num_null_shuffles, num_units), np.nan, dtype=np.float32)

    batch_size = _null_shuffle_batch(num_samples, num_null_shuffles)
    for start in range(0, num_null_shuffles, batch_size):
        batch = min(batch_size, num_null_shuffles - start)
        shifted_heading_bins = _shifted_heading_labels(
            heading_bins,
            step_layout=step_layout,
            offsets=shuffle_offsets[start : start + batch],
        )
        sample_weights = _batch_sample_weights(bin_weights, shifted_heading_bins)

        projected = sample_weights @ rates
        total_rates, cosine_parts, sine_parts = projected[0::3], projected[1::3], projected[2::3]
        has_rate = total_rates > _EPS
        null_matrix[start : start + batch][has_rate] = (
            np.hypot(cosine_parts[has_rate], sine_parts[has_rate]) / total_rates[has_rate]
        )
    return null_matrix


def _shifted_heading_labels(
    heading_bins: np.ndarray,
    *,
    step_layout: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    offsets: list[np.ndarray],
) -> np.ndarray:
    """[batch, num_samples] heading label each sample's rate is scored against after a roll."""
    step_episode, episode_start, episode_length, step_within_episode = step_layout
    return np.stack(
        [
            heading_bins[
                episode_start + ((step_within_episode + shift[step_episode]) % episode_length)
            ]
            for shift in offsets
        ]
    )


def _batch_sample_weights(bin_weights: np.ndarray, shifted_heading_bins: np.ndarray) -> np.ndarray:
    """Per-sample (total, cosine, sine) weight rows, three per shuffle in the batch."""
    batch, num_samples = shifted_heading_bins.shape
    sample_weights = np.empty((3 * batch, num_samples), dtype=np.float64)

    def gather_shuffle_rows(shuffle_indices: range) -> None:
        for shuffle_index in shuffle_indices:
            sample_weights[3 * shuffle_index : 3 * shuffle_index + 3] = bin_weights[
                :, shifted_heading_bins[shuffle_index]
            ]

    run_over_index_blocks(batch, gather_shuffle_rows)
    return sample_weights


def _null_shuffle_batch(num_samples: int, num_null_shuffles: int) -> int:
    """Shuffles per matmul, capped so the batch scratch stays inside _NULL_SCRATCH_BYTES."""
    scratch_bytes_per_shuffle = num_samples * 40
    return max(1, min(num_null_shuffles, _NULL_SCRATCH_BYTES // scratch_bytes_per_shuffle))


def _normalized_spatial_information(
    spatial_mean: np.ndarray,
    occupancy: np.ndarray,
    finite_spatial: np.ndarray,
) -> float:
    rates = spatial_mean[finite_spatial].astype(np.float64, copy=False)
    occupancies = occupancy[finite_spatial].astype(np.float64, copy=False)
    if rates.size <= 1 or float(np.sum(occupancies)) <= _EPS:
        return 0.0
    probabilities = occupancies / float(np.sum(occupancies))
    mean_rate = float(np.sum(probabilities * rates))
    if mean_rate <= _EPS:
        return 0.0
    rate_ratio = rates / mean_rate
    positive = rate_ratio > _EPS
    spatial_information = float(
        np.sum(probabilities[positive] * rate_ratio[positive] * np.log2(rate_ratio[positive]))
    )
    maximum_information = float(np.log2(rates.size))
    if maximum_information <= _EPS:
        return 0.0
    return float(np.clip(spatial_information / maximum_information, 0.0, 1.0))


def _position_invariance_score(
    local_heading_mean: np.ndarray,
    local_sampled_heading: np.ndarray,
    heading_resolved_bins: np.ndarray,
    active_spatial: np.ndarray,
    spatial_mean: np.ndarray,
    heading_vectors: np.ndarray,
    *,
    min_active_spatial_bins: int,
) -> float:
    """Rate-weighted agreement of the local heading preference across a unit's active place bins."""
    candidate_bins = active_spatial & heading_resolved_bins
    if int(np.count_nonzero(candidate_bins)) < min_active_spatial_bins:
        return float("nan")
    local_rates = np.where(
        local_sampled_heading[candidate_bins], local_heading_mean[candidate_bins], 0.0
    )
    total_rates = local_rates.sum(axis=1)
    resolved = total_rates > _EPS
    if int(np.count_nonzero(resolved)) < min_active_spatial_bins:
        return float("nan")
    weights = np.maximum(spatial_mean[candidate_bins][resolved], 0.0)
    total_weight = float(weights.sum())
    if total_weight <= _EPS:
        return float("nan")
    local_vectors = (local_rates[resolved] * heading_vectors).sum(axis=1) / total_rates[resolved]
    preferred_vectors = np.exp(1j * np.angle(local_vectors))
    return float(np.abs(np.sum(weights * preferred_vectors) / total_weight))


def _write_per_unit_table(
    output_dir: Path,
    analysis_input: AnalysisInput,
    per_unit: dict[str, np.ndarray],
) -> Path:
    rows = []
    for unit_index in range(len(per_unit["heading_vector_length"])):
        rows.append(
            [
                unit_index,
                _format_float(per_unit["heading_vector_length"][unit_index]),
                _format_float(per_unit["heading_vector_length_excess"][unit_index]),
                _format_float(per_unit["heading_vector_length_null_p"][unit_index]),
                bool(per_unit["heading_vector_length_significant"][unit_index]),
                _format_float(per_unit["preferred_heading_rad"][unit_index]),
                _format_float(per_unit["heading_peak_to_mean"][unit_index]),
                _format_float(per_unit["position_invariance_score"][unit_index]),
                _format_float(per_unit["spatial_coverage_fraction"][unit_index]),
                _format_float(per_unit["active_spatial_bin_count"][unit_index]),
                _format_float(per_unit["normalized_spatial_information"][unit_index]),
                _format_float(per_unit["hd_cell_score"][unit_index]),
                bool(per_unit["assessable"][unit_index]),
                bool(per_unit["is_hd_candidate"][unit_index]),
                bool(per_unit["is_hd_candidate_significant"][unit_index]),
                bool(per_unit["is_conjunctive_candidate"][unit_index]),
            ]
        )
    return write_csv(
        output_dir
        / "head_direction_tuning"
        / f"head_direction_tuning_per_unit__{analysis_input.source_name}"
        f"__{analysis_input.split_name}.csv",
        [
            "unit_index",
            "heading_vector_length",
            "heading_vector_length_excess",
            "heading_vector_length_null_p",
            "heading_vector_length_significant",
            "preferred_heading_rad",
            "heading_peak_to_mean",
            "position_invariance_score",
            "spatial_coverage_fraction",
            "active_spatial_bin_count",
            "normalized_spatial_information",
            "hd_cell_score",
            "assessable",
            "is_hd_candidate",
            "is_hd_candidate_significant",
            "is_conjunctive_candidate",
        ],
        rows,
    )


def _format_float(value: float | np.floating) -> float | str:
    return float(value) if np.isfinite(value) else "nan"


def _render_preferred_direction_rose(
    output_dir: Path,
    analysis_input: AnalysisInput,
    *,
    preferred_heading_rad: np.ndarray,
    heading_vector_length: np.ndarray,
    assessable: np.ndarray,
    strong_threshold: float,
) -> Path | None:
    """Population polar histogram of per-unit preferred headings."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    finite = assessable.astype(bool, copy=False) & np.isfinite(preferred_heading_rad)
    angles_all = preferred_heading_rad[finite].astype(np.float64)
    if angles_all.size == 0:
        return None
    strong = finite & (heading_vector_length >= strong_threshold)
    angles_strong = preferred_heading_rad[strong].astype(np.float64)

    num_petals = 24
    petal_color = "#4C72B0"

    def _resultant(angles):
        if angles.size == 0:
            return 0.0, 0.0
        mean_vector = np.exp(1j * angles).mean()
        return float(np.angle(mean_vector)), float(np.abs(mean_vector))

    def _draw(axis, angles, title):
        edges = np.linspace(0.0, _TWO_PI, num_petals + 1)
        counts, _ = np.histogram(angles % _TWO_PI, bins=edges)
        percent = counts / max(counts.sum(), 1) * 100.0
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.bar(
            centers,
            percent,
            width=edges[1] - edges[0],
            color=petal_color,
            edgecolor="white",
            linewidth=0.6,
            align="center",
            alpha=0.9,
            zorder=2,
        )
        r_max = max(float(percent.max()) if percent.size else 1.0, 1.0)
        mean_angle, mean_r = _resultant(angles)
        axis.annotate(
            "",
            xy=(mean_angle, mean_r * r_max),
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#1A1A1A", lw=2.0),
            zorder=5,
        )
        axis.set_rlabel_position(100)
        step = 5.0 if r_max <= 15 else 10.0
        rticks = np.arange(step, r_max + step, step)
        axis.set_yticks(rticks)
        axis.set_yticklabels([f"{int(t)}%" for t in rticks], fontsize=7, color="#666666")
        axis.set_ylim(0.0, r_max * 1.05)
        axis.set_thetagrids(
            range(0, 360, 45), labels=[f"{d}°" for d in range(0, 360, 45)], fontsize=8
        )
        axis.grid(color="#CFCFCF", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color("#CFCFCF")
            spine.set_linewidth(0.5)
        axis.set_title(f"{title}\nmean R = {mean_r:.2f}", fontsize=10, pad=12)

    figure, axes = plt.subplots(
        1, 2, figsize=(9.6, 5.4), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    source_label = analysis_input.source_name.replace("_", " ").replace(".", " ")
    figure.suptitle(
        f"Preferred heading  ·  {source_label} ({analysis_input.split_name})", fontsize=12
    )
    _draw(axes[0], angles_all, f"all units (n = {angles_all.size})")
    _draw(
        axes[1],
        angles_strong,
        f"strongly directional, R >= {strong_threshold:g}  (n = {angles_strong.size})",
    )

    path = (
        output_dir
        / "head_direction_tuning"
        / f"preferred_direction_rose__{analysis_input.source_name}__{analysis_input.split_name}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
