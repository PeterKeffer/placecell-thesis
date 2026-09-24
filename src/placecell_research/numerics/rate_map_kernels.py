from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label, rotate

from .work_blocks import run_over_index_blocks


@dataclass(slots=True)
class RateMapComputation:
    """Computed rate maps plus occupancy, smoothed and raw."""

    rate_maps: np.ndarray
    occupancy: np.ndarray
    raw_occupancy: np.ndarray
    reliability_maps: np.ndarray | None
    visited_episode_counts: np.ndarray | None
    bounds: tuple[tuple[float, float], tuple[float, float]]


@dataclass(slots=True)
class SampledValidSteps:
    """A bounded sample of valid sequence rows plus aligned metadata."""

    values: np.ndarray
    positions: np.ndarray
    episode_ids: np.ndarray
    total_valid_steps: int


@dataclass(slots=True)
class PlaceMetricRateMapPreparation:
    """Support decision plus clipped maps for near-nonnegative place metrics."""

    supported_mask: np.ndarray
    clipped_rate_maps: np.ndarray
    negative_bin_fraction: np.ndarray
    negative_peak_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class PlaceMetricSettings:
    """The signed-support tolerances and the field threshold, as one run configured them."""

    negative_tolerance: float = 1e-8
    max_negative_bin_fraction: float = 0.01
    max_negative_peak_fraction: float = 0.05
    field_threshold_fraction: float = 0.2


DEFAULT_PLACE_METRIC_SETTINGS = PlaceMetricSettings()


def read_float_setting(analysis_config: Any, field_name: str, default: float) -> float:
    """One float setting from a mapping or a typed config section."""
    if isinstance(analysis_config, Mapping):
        return float(analysis_config.get(field_name, default))
    return float(getattr(analysis_config, field_name, default))


def resolve_place_metric_settings(analysis_config: Any) -> PlaceMetricSettings:
    return PlaceMetricSettings(
        negative_tolerance=read_float_setting(
            analysis_config,
            "place_metric_negative_tolerance", DEFAULT_PLACE_METRIC_SETTINGS.negative_tolerance
        ),
        max_negative_bin_fraction=read_float_setting(
            analysis_config,
            "place_metric_max_negative_bin_fraction",
            DEFAULT_PLACE_METRIC_SETTINGS.max_negative_bin_fraction,
        ),
        max_negative_peak_fraction=read_float_setting(
            analysis_config,
            "place_metric_max_negative_peak_fraction",
            DEFAULT_PLACE_METRIC_SETTINGS.max_negative_peak_fraction,
        ),
        field_threshold_fraction=read_float_setting(
            analysis_config,
            "place_field_threshold_fraction",
            DEFAULT_PLACE_METRIC_SETTINGS.field_threshold_fraction,
        ),
    )


def _all_steps_valid(valid_mask: np.ndarray | None) -> bool:
    return valid_mask is None or bool(np.all(valid_mask))


def flatten_valid_steps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten [N, T, D] sequences and filter invalid steps."""
    flat_representation = representation.reshape(-1, representation.shape[-1])
    flat_positions = position_xy.reshape(-1, 2)
    if _all_steps_valid(valid_mask):
        return flat_representation, flat_positions
    flattened_mask = valid_mask.reshape(-1).astype(bool, copy=False)
    return flat_representation[flattened_mask], flat_positions[flattened_mask]


def sample_valid_steps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    max_samples: int,
    random_seed: int = 0,
) -> SampledValidSteps:
    """Sample valid sequence rows without first materializing all valid rows."""
    if representation.ndim != 3:
        raise ValueError(f"Expected representation [N, T, D], got {representation.shape}.")
    if position_xy.shape[:2] != representation.shape[:2] or position_xy.shape[-1] != 2:
        raise ValueError(
            "position_xy must match representation sequence shape and end in XY, "
            f"got representation={representation.shape}, position_xy={position_xy.shape}."
        )

    num_episodes, num_steps, _ = representation.shape
    total_rows = num_episodes * num_steps
    all_steps_valid = valid_mask is None
    if valid_mask is not None:
        if valid_mask.shape != (num_episodes, num_steps):
            raise ValueError(
                f"Expected valid_mask shape {(num_episodes, num_steps)}, got {valid_mask.shape}."
            )
        all_steps_valid = bool(np.all(valid_mask))

    if all_steps_valid:
        total_valid_steps = total_rows
        if int(max_samples) <= 0 or total_valid_steps <= int(max_samples):
            flat_values = representation.reshape(-1, representation.shape[-1])
            flat_positions = position_xy.reshape(-1, 2)
            return SampledValidSteps(
                values=flat_values.astype(np.float32, copy=False),
                positions=flat_positions.astype(np.float32, copy=False),
                episode_ids=np.repeat(np.arange(num_episodes, dtype=np.int64), num_steps),
                total_valid_steps=total_valid_steps,
            )
        valid_indices = np.arange(total_rows, dtype=np.int64)
    else:
        valid_indices = np.flatnonzero(valid_mask.reshape(-1).astype(bool, copy=False)).astype(
            np.int64,
            copy=False,
        )
        total_valid_steps = int(valid_indices.size)

    if int(max_samples) > 0 and total_valid_steps > int(max_samples):
        rng = np.random.default_rng(int(random_seed))
        valid_indices = np.sort(rng.choice(valid_indices, size=int(max_samples), replace=False))

    flat_values = representation.reshape(-1, representation.shape[-1])
    flat_positions = position_xy.reshape(-1, 2)
    return SampledValidSteps(
        values=flat_values[valid_indices].astype(np.float32, copy=False),
        positions=flat_positions[valid_indices].astype(np.float32, copy=False),
        episode_ids=(valid_indices // num_steps).astype(np.int64, copy=False),
        total_valid_steps=total_valid_steps,
    )


def place_metric_support_rule(
    *,
    negative_tolerance: float,
    max_negative_bin_fraction: float,
    max_negative_peak_fraction: float,
) -> str:
    """Describe when a rate map is treated as near-nonnegative for place metrics."""
    return (
        "Skaggs SI and field summaries are reported for units whose negative bins stay below "
        f"{negative_tolerance:g} in magnitude or, if negatives remain, occupy at most "
        f"{max_negative_bin_fraction:.3f} of finite bins and peak at at most "
        f"{max_negative_peak_fraction:.3f} of the unit's positive peak. Supported units clip "
        "those tiny negative bins to zero only for place-metric computation."
    )


def prepare_place_metric_rate_maps(
    rate_maps: np.ndarray,
    *,
    negative_tolerance: float = 1e-8,
    max_negative_bin_fraction: float = 0.01,
    max_negative_peak_fraction: float = 0.05,
) -> PlaceMetricRateMapPreparation:
    """Resolve which rate maps are near-nonnegative enough for place-field metrics."""
    if rate_maps.ndim == 2:
        batched_rate_maps = rate_maps[None, :, :]
    elif rate_maps.ndim == 3:
        batched_rate_maps = rate_maps
    else:
        raise ValueError(
            f"prepare_place_metric_rate_maps expects a [Y, X] or [N, Y, X] array, got "
            f"{rate_maps.shape}."
        )

    finite_mask = np.isfinite(batched_rate_maps)
    negative_mask = finite_mask & (batched_rate_maps < -float(negative_tolerance))
    if negative_tolerance >= 0 and batched_rate_maps.size and not negative_mask.any():
        unit_count = batched_rate_maps.shape[0]
        return PlaceMetricRateMapPreparation(
            supported_mask=np.full(
                unit_count,
                0.0 <= float(max_negative_bin_fraction)
                and 0.0 <= float(max_negative_peak_fraction),
                dtype=bool,
            ),
            clipped_rate_maps=batched_rate_maps.astype(np.float32, order="C", copy=True),
            negative_bin_fraction=np.zeros(unit_count, dtype=np.float32),
            negative_peak_fraction=np.zeros(unit_count, dtype=np.float32),
        )
    finite_counts = np.count_nonzero(finite_mask, axis=(1, 2)).astype(np.float32, copy=False)
    negative_counts = np.count_nonzero(negative_mask, axis=(1, 2)).astype(np.float32, copy=False)
    negative_bin_fraction = np.divide(
        negative_counts,
        finite_counts,
        out=np.zeros_like(negative_counts, dtype=np.float32),
        where=finite_counts > 0,
    )

    positive_peak = np.max(
        np.where(finite_mask, np.clip(batched_rate_maps, a_min=0.0, a_max=None), 0.0),
        axis=(1, 2),
    ).astype(np.float32, copy=False)

    negative_peak = np.max(
        np.where(negative_mask, -batched_rate_maps, 0.0),
        axis=(1, 2),
    ).astype(np.float32, copy=False)

    negative_peak_fraction = np.zeros_like(negative_peak, dtype=np.float32)
    has_positive_peak = positive_peak > float(negative_tolerance)
    negative_peak_fraction[has_positive_peak] = (
        negative_peak[has_positive_peak] / positive_peak[has_positive_peak]
    ).astype(np.float32, copy=False)
    materially_negative_without_positive_peak = (~has_positive_peak) & (
        negative_peak > float(negative_tolerance)
    )
    negative_peak_fraction[materially_negative_without_positive_peak] = np.inf

    supported_mask = (negative_bin_fraction <= float(max_negative_bin_fraction)) & (
        negative_peak_fraction <= float(max_negative_peak_fraction)
    )
    clipped_rate_maps = batched_rate_maps.copy()
    clipped_rate_maps[negative_mask & supported_mask[:, None, None]] = 0.0
    return PlaceMetricRateMapPreparation(
        supported_mask=supported_mask.astype(bool, copy=False),
        clipped_rate_maps=clipped_rate_maps.astype(np.float32, copy=False),
        negative_bin_fraction=negative_bin_fraction,
        negative_peak_fraction=negative_peak_fraction,
    )


def flatten_vector_field(
    values: np.ndarray | None, valid_mask: np.ndarray | None
) -> np.ndarray | None:
    """Flatten an auxiliary array alongside a valid mask."""
    if values is None:
        return None
    flattened = values.reshape(-1, *values.shape[2:]) if values.ndim > 2 else values.reshape(-1)
    if _all_steps_valid(valid_mask):
        return flattened
    flattened_mask = valid_mask.reshape(-1).astype(bool, copy=False)
    return flattened[flattened_mask]


def flatten_positions(position_xy: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """Flatten XY positions and keep only valid steps."""
    flat_positions = position_xy.reshape(-1, position_xy.shape[-1])
    if _all_steps_valid(valid_mask):
        return flat_positions
    flattened_mask = valid_mask.reshape(-1).astype(bool, copy=False)
    return flat_positions[flattened_mask]


def infer_bounds(position_xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded XY bounds."""
    x_values = position_xy[:, 0]
    y_values = position_xy[:, 1]
    x_span = max(1e-6, float(x_values.max() - x_values.min()))
    y_span = max(1e-6, float(y_values.max() - y_values.min()))
    return (
        (float(x_values.min() - 0.05 * x_span), float(x_values.max() + 0.05 * x_span)),
        (float(y_values.min() - 0.05 * y_span), float(y_values.max() + 0.05 * y_span)),
    )


def compute_spatial_bin_assignments(
    positions: np.ndarray,
    *,
    num_bins_x: int,
    num_bins_y: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[float, float], tuple[float, float]]]:
    """Map XY positions to flattened spatial bins."""
    if positions.size == 0:
        raise ValueError("Cannot compute spatial bins without any valid positions.")
    resolved_bounds = infer_bounds(positions) if bounds is None else bounds
    x_edges = np.linspace(
        resolved_bounds[0][0], resolved_bounds[0][1], num_bins_x + 1, dtype=np.float32
    )
    y_edges = np.linspace(
        resolved_bounds[1][0], resolved_bounds[1][1], num_bins_y + 1, dtype=np.float32
    )
    x_bin_indices = np.clip(
        np.searchsorted(x_edges, positions[:, 0], side="right") - 1,
        0,
        num_bins_x - 1,
    ).astype(np.int32, copy=False)
    y_bin_indices = np.clip(
        np.searchsorted(y_edges, positions[:, 1], side="right") - 1,
        0,
        num_bins_y - 1,
    ).astype(np.int32, copy=False)
    linear_bins = (y_bin_indices * num_bins_x + x_bin_indices).astype(np.int32, copy=False)
    return linear_bins, x_edges, y_edges, resolved_bounds


_SCATTER_CHUNK_ELEMENT_BUDGET = 1 << 23


def scatter_add_over_units(
    bin_indices: np.ndarray,
    rates: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """Sum per-step, per-unit rates into bins (vectorized np.add.at replacement)."""
    num_steps, num_units = int(rates.shape[0]), int(rates.shape[1])
    chunk_width = max(1, min(num_units, _SCATTER_CHUNK_ELEMENT_BUDGET // max(1, num_steps)))
    chunk_starts = list(range(0, num_units, chunk_width))
    bin_indices = bin_indices.astype(np.int64, copy=False)
    summed = np.empty((num_bins, num_units), dtype=np.float64)

    def accumulate_chunks(chunk_indices: range) -> None:
        for chunk_index in chunk_indices:
            start = chunk_starts[chunk_index]
            stop = min(start + chunk_width, num_units)
            width = stop - start
            flat_codes = bin_indices[:, None] * width + np.arange(width, dtype=np.int64)
            summed[:, start:stop] = np.bincount(
                flat_codes.reshape(-1),
                weights=np.ascontiguousarray(rates[:, start:stop], dtype=np.float64).reshape(-1),
                minlength=num_bins * width,
            ).reshape(num_bins, width)

    run_over_index_blocks(len(chunk_starts), accumulate_chunks)
    return summed


def compute_rate_maps(
    representation: np.ndarray,
    position_xy: np.ndarray,
    valid_mask: np.ndarray | None,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    unit_chunk_size: int = 64,
) -> RateMapComputation:
    """Compute smoothed occupancy-normalized rate maps."""
    values, positions = flatten_valid_steps(representation, position_xy, valid_mask)
    if values.size == 0:
        raise ValueError("Cannot compute rate maps without any valid timesteps.")
    linear_bins, _, _, bounds = compute_spatial_bin_assignments(
        positions,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    num_units = values.shape[-1]
    num_bins = num_bins_x * num_bins_y
    raw_occupancy = (
        np.bincount(linear_bins, minlength=num_bins)
        .reshape(num_bins_y, num_bins_x)
        .astype(np.float32, copy=False)
    )
    occupancy = raw_occupancy
    if smoothing_sigma > 0.0:
        occupancy = gaussian_filter(raw_occupancy, sigma=smoothing_sigma)

    activity_maps = np.zeros((num_units, num_bins), dtype=np.float32)
    chunk_size = max(1, min(int(unit_chunk_size), num_units))
    linear_bins_int = linear_bins.astype(np.int64, copy=False)
    for start_index in range(0, num_units, chunk_size):
        stop_index = min(start_index + chunk_size, num_units)
        chunk_values = values[:, start_index:stop_index].astype(np.float32, copy=False)
        chunk_width = stop_index - start_index
        flat_bin_indices = (
            linear_bins_int[:, None] + np.arange(chunk_width, dtype=np.int64)[None, :] * num_bins
        )
        chunk_activity = np.bincount(
            flat_bin_indices.reshape(-1),
            weights=chunk_values.reshape(-1).astype(np.float64, copy=False),
            minlength=chunk_width * num_bins,
        ).reshape(chunk_width, num_bins)
        activity_maps[start_index:stop_index] = chunk_activity.astype(np.float32, copy=False)
    activity_maps = activity_maps.reshape(num_units, num_bins_y, num_bins_x)
    if smoothing_sigma > 0.0:
        activity_maps = gaussian_filter(
            activity_maps, sigma=(0.0, smoothing_sigma, smoothing_sigma)
        )

    safe_occupancy = np.where(occupancy >= min_occupancy, occupancy, np.nan)
    rate_maps = (activity_maps / safe_occupancy[None, :, :]).astype(np.float32, copy=False)
    return RateMapComputation(
        rate_maps=rate_maps,
        occupancy=occupancy.astype(np.float32),
        raw_occupancy=raw_occupancy,
        reliability_maps=None,
        visited_episode_counts=None,
        bounds=bounds,
    )


def compute_place_field_mask(
    rate_map: np.ndarray, threshold_fraction: float
) -> tuple[np.ndarray, int, float]:
    """Return a NaN-safe field mask plus connected-component count and area."""
    finite_rate_map = np.nan_to_num(rate_map, nan=0.0)
    peak_value = float(np.max(finite_rate_map)) if finite_rate_map.size else 0.0
    if peak_value <= 1e-8:
        empty_mask = np.zeros_like(finite_rate_map, dtype=bool)
        return empty_mask, 0, 0.0
    mask = finite_rate_map >= threshold_fraction * peak_value
    _, field_count = label(mask)
    return mask, int(field_count), float(mask.sum())


def compute_largest_field_metrics(
    rate_map: np.ndarray, threshold_fraction: float
) -> tuple[float, float, float]:
    """Return (largest_component_area, coherence, equivalent_radius) for one rate map."""
    finite_rate_map = np.nan_to_num(rate_map, nan=0.0)
    peak_value = float(np.max(finite_rate_map)) if finite_rate_map.size else 0.0
    if peak_value <= 1e-8:
        return 0.0, 0.0, 0.0
    mask = finite_rate_map >= threshold_fraction * peak_value
    labeled, field_count = label(mask)
    if field_count == 0:
        return 0.0, 0.0, 0.0
    component_sizes = np.bincount(labeled.ravel())[1:]
    largest_area = float(component_sizes.max())
    total_area = float(mask.sum())
    coherence = largest_area / total_area if total_area > 0 else 0.0
    radius = float(np.sqrt(largest_area / np.pi))
    return largest_area, coherence, radius


def rowwise_correlation(
    first_values: np.ndarray,
    second_values: np.ndarray,
    epsilon: float = 1e-8,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute one Pearson correlation per row for matching [units, samples] arrays."""
    if first_values.shape != second_values.shape:
        raise ValueError(
            "rowwise_correlation requires matching shapes, "
            f"got {first_values.shape} and {second_values.shape}."
        )
    if valid_mask is None:
        first_centered = first_values - first_values.mean(axis=1, keepdims=True)
        second_centered = second_values - second_values.mean(axis=1, keepdims=True)
    else:
        if valid_mask.shape != first_values.shape:
            raise ValueError(
                "rowwise_correlation valid_mask must match values, "
                f"got {valid_mask.shape} and {first_values.shape}."
            )
        finite_mask = (
            valid_mask.astype(bool, copy=False)
            & np.isfinite(first_values)
            & np.isfinite(second_values)
        )
        counts = finite_mask.sum(axis=1, keepdims=True)
        safe_counts = np.maximum(counts, 1)
        first_mean = (
            np.where(finite_mask, first_values, 0.0).sum(axis=1, keepdims=True) / safe_counts
        )
        second_mean = (
            np.where(finite_mask, second_values, 0.0).sum(axis=1, keepdims=True) / safe_counts
        )
        first_centered = np.where(finite_mask, first_values - first_mean, 0.0)
        second_centered = np.where(finite_mask, second_values - second_mean, 0.0)
    numerator = np.sum(first_centered * second_centered, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(first_centered), axis=1) * np.sum(np.square(second_centered), axis=1)
    )
    correlations = np.zeros(first_values.shape[0], dtype=np.float32)
    valid = denominator > epsilon
    correlations[valid] = (numerator[valid] / denominator[valid]).astype(np.float32, copy=False)
    return correlations


def correlation_to_vector(
    values: np.ndarray, vector: np.ndarray, epsilon: float = 1e-8
) -> np.ndarray:
    """Correlate every row of a [units, samples] matrix with one [samples] vector."""
    if values.ndim != 2:
        raise ValueError(
            f"correlation_to_vector expects a 2D [units, samples] array, got {values.shape}."
        )
    if vector.ndim != 1 or values.shape[1] != vector.shape[0]:
        raise ValueError(
            f"correlation_to_vector requires [units, samples] against [samples], got "
            f"{values.shape} and {vector.shape}."
        )
    centered_values = values - values.mean(axis=1, keepdims=True)
    centered_vector = vector - vector.mean()
    return correlation_from_centered_parts(
        centered_values @ centered_vector,
        np.sum(np.square(centered_values), axis=1),
        float(np.square(centered_vector).sum()),
        epsilon=epsilon,
    )


def correlation_from_centered_parts(
    cross_term: np.ndarray,
    values_sum_of_squares: np.ndarray,
    vector_sum_of_squares: float,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Finish a rowwise Pearson correlation from the cross term and the two sums of squares."""
    denominator = np.sqrt(values_sum_of_squares * vector_sum_of_squares)
    correlations = np.zeros(cross_term.shape[0], dtype=np.float32)
    valid = denominator > epsilon
    correlations[valid] = (cross_term[valid] / denominator[valid]).astype(np.float32, copy=False)
    return correlations


def skaggs_spatial_information(
    rate_map: np.ndarray,
    occupancy: np.ndarray,
    epsilon: float = 1e-10,
    negative_tolerance: float = 1e-8,
    max_negative_bin_fraction: float = 0.01,
    max_negative_peak_fraction: float = 0.05,
) -> float | np.ndarray:
    """Compute Skaggs spatial information in bits/spike for one map or a batch of maps."""
    occupancy_probability = occupancy / max(epsilon, float(occupancy.sum()))
    if rate_map.ndim == 2:
        prepared = prepare_place_metric_rate_maps(
            rate_map,
            negative_tolerance=negative_tolerance,
            max_negative_bin_fraction=max_negative_bin_fraction,
            max_negative_peak_fraction=max_negative_peak_fraction,
        )
        if not bool(prepared.supported_mask[0]):
            return float("nan")
        clipped_rate_map = prepared.clipped_rate_maps[0]
        mean_rate = float(np.nansum(occupancy_probability * clipped_rate_map))
        if mean_rate <= epsilon:
            return 0.0
        normalized_rate = np.clip(clipped_rate_map / mean_rate, epsilon, None)
        info = occupancy_probability * normalized_rate * np.log2(normalized_rate)
        return float(np.nansum(info))
    if rate_map.ndim != 3:
        raise ValueError(
            f"skaggs_spatial_information expects a [Y, X] or [N, Y, X] array, got {rate_map.shape}."
        )
    prepared = prepare_place_metric_rate_maps(
        rate_map,
        negative_tolerance=negative_tolerance,
        max_negative_bin_fraction=max_negative_bin_fraction,
        max_negative_peak_fraction=max_negative_peak_fraction,
    )
    occupancy_probability = occupancy_probability[None, :, :]
    clipped_rate_maps = prepared.clipped_rate_maps
    supports_skaggs = prepared.supported_mask
    mean_rate = np.nansum(occupancy_probability * clipped_rate_maps, axis=(1, 2))
    valid = supports_skaggs & (mean_rate > epsilon)
    safe_mean_rate = np.where(valid, mean_rate, 1.0)
    normalized_rate = np.clip(clipped_rate_maps / safe_mean_rate[:, None, None], epsilon, None)
    info = occupancy_probability * normalized_rate * np.log2(normalized_rate)
    scores = np.nansum(info, axis=(1, 2)).astype(np.float32, copy=False)
    scores[~supports_skaggs] = np.nan
    scores[supports_skaggs & ~valid] = 0.0
    return scores


def autocorrelogram(rate_map: np.ndarray) -> np.ndarray:
    """Compute a normalized spatial autocorrelogram."""
    rate_map = np.nan_to_num(rate_map, nan=0.0)
    centered = rate_map - rate_map.mean()
    fft = np.fft.fftn(centered)
    autocorr = np.fft.ifftn(fft * np.conj(fft)).real
    autocorr = np.fft.fftshift(autocorr)
    max_abs = np.max(np.abs(autocorr))
    return autocorr / max(max_abs, 1e-8)


def gridness_from_autocorrelogram(
    autocorr: np.ndarray,
    inner_radius: float = 3.0,
    outer_radius: float | None = None,
) -> float:
    """Sargolini-style hexagonal gridness scored on a given autocorrelogram."""
    center_y, center_x = np.array(autocorr.shape) // 2
    yy, xx = np.indices(autocorr.shape)
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    if outer_radius is None:
        outer_radius = min(autocorr.shape) / 2.5
    ring_mask = (radius > inner_radius) & (radius < outer_radius)
    if ring_mask.sum() < 20:
        return 0.0

    reference = autocorr[ring_mask]
    correlations: dict[int, float] = {}
    for angle in (30, 60, 90, 120, 150):
        rotated = rotate(autocorr, angle=angle, reshape=False, order=1, mode="nearest")
        candidate = rotated[ring_mask]
        if np.std(reference) < 1e-8 or np.std(candidate) < 1e-8:
            correlations[angle] = 0.0
        else:
            correlations[angle] = float(np.corrcoef(reference, candidate)[0, 1])
    return float(
        (correlations[60] + correlations[120]) / 2.0
        - (correlations[30] + correlations[90] + correlations[150]) / 3.0
    )


def gridness_score(rate_map: np.ndarray) -> float:
    """Approximate Sargolini-style gridness from the rate-map autocorrelogram."""
    return gridness_from_autocorrelogram(autocorrelogram(rate_map))


def gridness_summary_metrics(scores: np.ndarray) -> dict[str, float]:
    """Population gridness aggregates shared by the offline module and the in-loop eval."""
    scores = np.asarray(scores, dtype=np.float32)
    finite_scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        mean_gridness = max_gridness = 0.0
    elif finite_scores.size == 0:
        mean_gridness = max_gridness = float("nan")
    else:
        mean_gridness = float(finite_scores.mean())
        max_gridness = float(finite_scores.max())
    return {
        "mean_gridness": mean_gridness,
        "max_gridness": max_gridness,
        "gridness_scorable_units": float(finite_scores.size),
        "num_grid_units": float((finite_scores > 0.3).sum()),
        "num_grid_units_strong": float((finite_scores > 0.5).sum()),
        "fraction_grid": float((finite_scores > 0.3).mean()) if finite_scores.size else 0.0,
    }


def lifetime_activity_fraction(unit_values: np.ndarray, epsilon: float = 1e-8) -> float:
    """Treves-Rolls activity ratio (E[x])^2 / E[x^2] over a unit's lifetime."""
    mean_value = float(unit_values.mean())
    mean_square = float(np.square(unit_values).mean())
    if mean_square <= epsilon:
        return 0.0
    return float((mean_value**2) / mean_square)


def population_activity_fraction(batch_values: np.ndarray, epsilon: float = 1e-8) -> float:
    """Treves-Rolls activity ratio across the population, averaged over samples."""
    sample_means = batch_values.mean(axis=1)
    sample_mean_squares = np.square(batch_values).mean(axis=1)
    valid = sample_mean_squares > epsilon
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.square(sample_means[valid]) / sample_mean_squares[valid]))
