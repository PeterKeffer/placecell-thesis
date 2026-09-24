"""Lightweight representation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from placecell_research.numerics.circular import circular_vector, peak_to_mean
from placecell_research.numerics.rate_map_kernels import scatter_add_over_units


def compute_heading_tuning_shape_eval_metrics(
    *,
    representation_array: np.ndarray,
    valid_array: np.ndarray,
    heading_array: np.ndarray,
    num_heading_bins: int = 36,
    min_occupancy_per_bin: int = 5,
    min_heading_bins: int = 8,
    min_fire_rate: float = 0.01,
) -> dict[str, float]:
    """Global heading-tuning shape metrics for distinguishing one-peak vs 180-degree tuning."""
    num_units = int(representation_array.shape[-1])
    valid_flat = valid_array.reshape(-1).astype(bool)
    codes = representation_array.reshape(-1, num_units)[valid_flat]
    headings = heading_array.reshape(-1)[valid_flat]
    if codes.shape[0] == 0:
        return {
            "validation.heading_tuning_total_units": float(num_units),
            "validation.heading_tuning_assessable_units": 0.0,
        }

    rates = np.clip(codes, 0.0, None).astype(np.float64)
    fire_rate = (codes != 0).mean(axis=0)
    heading_bins = (
        np.floor((headings % (2.0 * np.pi)) / ((2.0 * np.pi) / num_heading_bins)).astype(int)
    ) % num_heading_bins
    occupancy = np.bincount(heading_bins, minlength=num_heading_bins).astype(np.float64)
    activity = scatter_add_over_units(heading_bins, rates, num_heading_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        heading_mean = activity / occupancy[:, None]

    sampled = occupancy >= float(min_occupancy_per_bin)
    heading_centers = (np.arange(num_heading_bins, dtype=np.float64) + 0.5) * (
        2.0 * np.pi / num_heading_bins
    )
    first_vectors = np.exp(1j * heading_centers)
    second_vectors = np.exp(2j * heading_centers)
    bin_width_deg = 360.0 / float(num_heading_bins)

    first_harmonic: list[float] = []
    second_harmonic: list[float] = []
    activation_corridor_width_deg: list[float] = []
    peak_to_mean_values: list[float] = []
    opposite_peak_ratio: list[float] = []
    single_peak_flags: list[bool] = []
    opposite_peak_flags: list[bool] = []

    for unit_index in range(num_units):
        if fire_rate[unit_index] < min_fire_rate:
            continue
        valid_bins = sampled & np.isfinite(heading_mean[:, unit_index])
        if int(np.count_nonzero(valid_bins)) < min_heading_bins:
            continue
        unit_rates = heading_mean[:, unit_index]
        valid_rates = unit_rates[valid_bins]
        total_rate = float(valid_rates.sum())
        peak_rate = float(valid_rates.max())
        mean_rate = float(valid_rates.mean())
        if total_rate <= 1e-9 or peak_rate <= 1e-9 or mean_rate <= 1e-9:
            continue

        first_r = circular_vector(valid_rates, first_vectors[valid_bins])[0]
        second_r = circular_vector(valid_rates, second_vectors[valid_bins])[0]
        peak_bin = int(np.nanargmax(np.where(valid_bins, unit_rates, np.nan)))
        opposite_bin = (peak_bin + num_heading_bins // 2) % num_heading_bins
        opposite_rate = unit_rates[opposite_bin] if valid_bins[opposite_bin] else np.nan
        opposite_ratio = (
            float(opposite_rate / peak_rate) if np.isfinite(opposite_rate) else float("nan")
        )
        above_half = valid_bins & (unit_rates >= 0.5 * peak_rate)
        run_count = count_circular_true_runs(above_half)
        main_run_length = circular_true_run_length(above_half, peak_bin)

        first_harmonic.append(first_r)
        second_harmonic.append(second_r)
        activation_corridor_width_deg.append(float(main_run_length * bin_width_deg))
        peak_to_mean_values.append(peak_to_mean(valid_rates))
        if np.isfinite(opposite_ratio):
            opposite_peak_ratio.append(opposite_ratio)
            opposite_peak_flags.append(bool(opposite_ratio >= 0.5 and second_r > first_r))
        single_peak_flags.append(bool(run_count == 1 and first_r >= second_r))

    metrics = {
        "validation.heading_tuning_total_units": float(num_units),
        "validation.heading_tuning_assessable_units": float(len(first_harmonic)),
    }
    if first_harmonic:
        median_corridor_deg = float(np.median(activation_corridor_width_deg))
        metrics.update(
            {
                "validation.heading_tuning_first_harmonic_median_r": float(
                    np.median(first_harmonic)
                ),
                "validation.heading_tuning_second_harmonic_median_r": float(
                    np.median(second_harmonic)
                ),
                "validation.heading_tuning_median_half_width_deg": median_corridor_deg,
                "validation.heading_activation_corridor_median_deg": median_corridor_deg,
                "validation.heading_tuning_median_peak_to_mean": float(
                    np.median(peak_to_mean_values)
                ),
                "validation.heading_tuning_single_peak_fraction": float(np.mean(single_peak_flags)),
            }
        )
    if opposite_peak_ratio:
        metrics["validation.heading_tuning_median_opposite_peak_ratio"] = float(
            np.median(opposite_peak_ratio)
        )
        metrics["validation.heading_tuning_opposite_peak_fraction"] = float(
            np.mean(opposite_peak_flags)
        )
        metrics["validation.heading_tuning_180_candidate_fraction"] = float(
            np.mean(opposite_peak_flags)
        )
    return metrics


def count_circular_true_runs(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    if np.all(mask):
        return 1
    return int(np.count_nonzero(mask & ~np.roll(mask, 1)))


def circular_true_run_length(mask: np.ndarray, index: int) -> int:
    if not bool(mask[index]):
        return 0
    length = 1
    cursor = (index - 1) % mask.size
    while cursor != index and bool(mask[cursor]):
        length += 1
        cursor = (cursor - 1) % mask.size
    cursor = (index + 1) % mask.size
    while cursor != index and bool(mask[cursor]):
        length += 1
        cursor = (cursor + 1) % mask.size
    return min(length, int(mask.size))


_SPARSITY_CHUNK_ELEMENTS = 4 << 20

_SPARSITY_KEYS = ("mean_activation", "fraction_active", "active_units_mean", "active_units_std")


def summarize_code_sparsity(codes: np.ndarray, active_threshold: float = 1e-4) -> dict[str, Any]:
    """Return stable code sparsity summaries for [N, D] or [B, T, D] arrays."""
    if codes.ndim < 2:
        raise ValueError(f"Expected codes with at least 2 dims, got {codes.shape}.")
    flattened = codes.reshape(-1, codes.shape[-1])
    num_rows, num_units = flattened.shape
    if flattened.size == 0:
        return dict.fromkeys(_SPARSITY_KEYS, float("nan"))
    active_per_row = np.empty(num_rows, dtype=np.int64)
    absolute_total = 0.0
    rows_per_chunk = max(1, _SPARSITY_CHUNK_ELEMENTS // num_units)
    for start in range(0, num_rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, num_rows)
        absolute_chunk = np.abs(flattened[start:stop])
        absolute_total += float(absolute_chunk.sum(dtype=np.float64))
        active_per_row[start:stop] = np.count_nonzero(absolute_chunk > active_threshold, axis=1)
    total_elements = float(num_rows) * float(num_units)
    return {
        "mean_activation": absolute_total / total_elements,
        "fraction_active": float(active_per_row.sum()) / total_elements,
        "active_units_mean": float(active_per_row.mean()),
        "active_units_std": float(active_per_row.std()),
    }


def summarize_topk_scores(
    scores: np.ndarray,
    top_k: int,
    selection_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Return a few consistent aggregate scores."""
    if scores.ndim != 1:
        raise ValueError("Scores must be 1D.")
    empty = {
        "mean_score": float("nan"),
        "mean_top_k": float("nan"),
        "max_score": float("nan"),
        "mean_top_k_selected": float("nan"),
    }
    finite_mask = np.isfinite(scores)
    finite_scores = scores[finite_mask]
    if len(finite_scores) == 0:
        return empty
    count = max(1, min(top_k, len(finite_scores)))
    top_values = np.sort(finite_scores)[-count:]

    selected_mean = float("nan")
    if selection_scores is not None:
        if selection_scores.shape != scores.shape:
            raise ValueError(
                "selection_scores must have the same shape as scores; got "
                f"{selection_scores.shape} vs {scores.shape}."
            )
        usable = finite_mask & np.isfinite(selection_scores)
        if usable.any():
            usable_indices = usable.nonzero()[0]
            order = np.argsort(selection_scores[usable_indices])
            chosen = usable_indices[order[-max(1, min(top_k, len(usable_indices))) :]]
            selected_mean = float(scores[chosen].mean())

    return {
        "mean_score": float(finite_scores.mean()),
        "mean_top_k": float(top_values.mean()),
        "max_score": float(top_values.max()),
        "mean_top_k_selected": selected_mean,
    }


def topk_spatial_information(scores: np.ndarray, top_k: int) -> dict[str, float]:
    return summarize_topk_scores(scores, top_k)


def place_code_quality(
    decode_r2: float,
    fraction_place_cells: float,
    field_coverage: float,
) -> dict[str, float]:
    """One readable [0, 1] number for "how good is this place code", plus its factors."""
    accuracy = float(np.clip(decode_r2, 0.0, 1.0)) if np.isfinite(decode_r2) else float("nan")
    factors = np.asarray([accuracy, fraction_place_cells, field_coverage], dtype=np.float64)
    all_factors_finite = bool(np.all(np.isfinite(factors)))
    quality = float(np.cbrt(np.prod(factors))) if all_factors_finite else float("nan")
    return {
        "place_code_accuracy": accuracy,
        "place_code_quality": quality,
    }
