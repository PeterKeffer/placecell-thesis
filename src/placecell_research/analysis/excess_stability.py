"""Excess temporal stability: observed stability vs memoryless spatial twins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.rate_map_kernels import (
    compute_rate_maps,
    compute_spatial_bin_assignments,
    infer_bounds,
)
from .base import AnalysisInput, AnalysisResult
from .figures import apply_publication_style, despine

DEFAULT_LAG = 16
_VARIANCE_EPS = 1.0e-12
_RATIO_EPS = 1.0e-12
_UNIT_CHUNK_SIZE = 32
_NUM_HEADING_QUADRANTS = 4
_TWO_PI = 2.0 * np.pi
_SWEEP_BIN_COUNTS = (20, 40)
_TWIN_SPLIT_SEED = 0


def variance_normalized_lag_stability(
    activity: np.ndarray,
    valid: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Per-unit Wyss stability Psi at lag; NaN for units without variance or pairs."""
    step_valid = valid[..., None] & np.isfinite(activity)
    weights = step_valid.astype(np.float32)
    masked = np.where(step_valid, activity, 0.0).astype(np.float32, copy=False)
    counts = weights.sum(axis=(0, 1), dtype=np.float64)
    safe_counts = np.clip(counts, 1.0, None)
    means = masked.sum(axis=(0, 1), dtype=np.float64) / safe_counts
    centered = (masked - means[None, None, :].astype(np.float32)) * weights
    variances = np.square(centered).sum(axis=(0, 1), dtype=np.float64) / safe_counts

    pair_valid = step_valid[:, lag:] & step_valid[:, :-lag]
    pair_counts = pair_valid.sum(axis=(0, 1)).astype(np.float64)
    if pair_valid.size:
        squared_change = np.where(
            pair_valid,
            np.square(activity[:, lag:] - activity[:, :-lag]),
            0.0,
        )
        numerators = squared_change.sum(axis=(0, 1), dtype=np.float64)
    else:
        numerators = np.zeros_like(pair_counts)
    psi = (numerators / np.clip(pair_counts, 1.0, None)) / (variances + _VARIANCE_EPS)
    psi[(variances <= _VARIANCE_EPS) | (pair_counts == 0)] = np.nan
    return psi


def heading_quadrants(heading: np.ndarray) -> np.ndarray:
    """Heading quadrant labels in {0..3}, same convention as directionality.py."""
    return (np.floor((heading % _TWO_PI) / (np.pi / 2.0)).astype(int)) % _NUM_HEADING_QUADRANTS


@dataclass(slots=True)
class TwinPsi:
    """Per-unit Psi for the observed activity and its two memoryless twins."""

    psi_observed: np.ndarray
    psi_xy_twin: np.ndarray
    psi_pose_twin: np.ndarray


def compute_twin_psi(
    representation: np.ndarray,
    positions: np.ndarray,
    heading: np.ndarray | None,
    valid: np.ndarray,
    lag: int,
    *,
    fit_episodes: np.ndarray,
    eval_episodes: np.ndarray,
    num_bins_x: int,
    num_bins_y: int,
    smoothing_sigma: float,
    min_occupancy: float,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> TwinPsi:
    """Psi on eval_episodes for observed activity and twins fit on fit_episodes."""
    num_units = representation.shape[-1]
    fit_valid = valid[fit_episodes]
    quadrants = heading_quadrants(heading) if heading is not None else None

    num_bins = num_bins_y * num_bins_x
    xy_maps = compute_rate_maps(
        representation[fit_episodes],
        positions[fit_episodes],
        fit_valid,
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        smoothing_sigma=smoothing_sigma,
        min_occupancy=min_occupancy,
        bounds=bounds,
    ).rate_maps
    flat_xy_maps = xy_maps.reshape(num_units, num_bins)

    flat_pose_maps = None
    if quadrants is not None:
        pose_maps = np.full(
            (_NUM_HEADING_QUADRANTS, num_units, num_bins_y, num_bins_x),
            np.nan,
            dtype=np.float32,
        )
        for quadrant in range(_NUM_HEADING_QUADRANTS):
            quadrant_valid = fit_valid & (quadrants[fit_episodes] == quadrant)
            if not quadrant_valid.any():
                continue
            pose_maps[quadrant] = compute_rate_maps(
                representation[fit_episodes],
                positions[fit_episodes],
                quadrant_valid,
                num_bins_x=num_bins_x,
                num_bins_y=num_bins_y,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
                bounds=bounds,
            ).rate_maps
        flat_pose_maps = pose_maps.transpose(1, 0, 2, 3).reshape(
            num_units, _NUM_HEADING_QUADRANTS * num_bins
        )

    eval_representation = representation[eval_episodes]
    eval_valid = valid[eval_episodes]
    num_eval_episodes, num_steps = eval_valid.shape
    linear_bins, _, _, _ = compute_spatial_bin_assignments(
        positions[eval_episodes].reshape(-1, 2),
        num_bins_x=num_bins_x,
        num_bins_y=num_bins_y,
        bounds=bounds,
    )
    pose_bins = None
    if flat_pose_maps is not None:
        pose_bins = quadrants[eval_episodes].reshape(-1) * num_bins + linear_bins

    psi_observed = np.full(num_units, np.nan)
    psi_xy_twin = np.full(num_units, np.nan)
    psi_pose_twin = np.full(num_units, np.nan)
    for start_index in range(0, num_units, _UNIT_CHUNK_SIZE):
        stop_index = min(start_index + _UNIT_CHUNK_SIZE, num_units)
        chunk_width = stop_index - start_index
        xy_twin_chunk = (
            flat_xy_maps[start_index:stop_index, linear_bins]
            .reshape(chunk_width, num_eval_episodes, num_steps)
            .transpose(1, 2, 0)
        )
        pose_twin_chunk = None
        shared_invalid = ~np.isfinite(xy_twin_chunk)
        if flat_pose_maps is not None:
            pose_twin_chunk = (
                flat_pose_maps[start_index:stop_index, pose_bins]
                .reshape(chunk_width, num_eval_episodes, num_steps)
                .transpose(1, 2, 0)
            )
            shared_invalid |= ~np.isfinite(pose_twin_chunk)
        observed_chunk = np.where(
            shared_invalid, np.nan, eval_representation[..., start_index:stop_index]
        )
        psi_observed[start_index:stop_index] = variance_normalized_lag_stability(
            observed_chunk, eval_valid, lag
        )
        psi_xy_twin[start_index:stop_index] = variance_normalized_lag_stability(
            np.where(shared_invalid, np.nan, xy_twin_chunk), eval_valid, lag
        )
        if pose_twin_chunk is not None:
            psi_pose_twin[start_index:stop_index] = variance_normalized_lag_stability(
                np.where(shared_invalid, np.nan, pose_twin_chunk), eval_valid, lag
            )
    return TwinPsi(psi_observed, psi_xy_twin, psi_pose_twin)


def _measurable_denominator(denominator: np.ndarray) -> np.ndarray:
    """Twins whose Psi stands above the guard the division adds to it."""
    return np.isfinite(denominator) & (np.abs(denominator) > _RATIO_EPS)


def _count_unmeasurable(denominator: np.ndarray) -> int:
    """Units whose Psi is defined but too small to divide by."""
    return int(np.count_nonzero(np.isfinite(denominator) & ~_measurable_denominator(denominator)))


def _finite_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.where(
        np.isfinite(numerator) & _measurable_denominator(denominator),
        numerator / (denominator + _RATIO_EPS),
        np.nan,
    )


def _finite_median(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    return float(np.median(finite_values)) if finite_values.size else float("nan")


def _ratio_summaries(twin_psi: TwinPsi, suffix: str) -> dict[str, float]:
    return {
        f"median_excess_stability_ratio{suffix}": _finite_median(
            _finite_ratio(twin_psi.psi_observed, twin_psi.psi_xy_twin)
        ),
        f"median_excess_vs_pose_twin{suffix}": _finite_median(
            _finite_ratio(twin_psi.psi_observed, twin_psi.psi_pose_twin)
        ),
        f"median_pose_vs_xy_twin{suffix}": _finite_median(
            _finite_ratio(twin_psi.psi_pose_twin, twin_psi.psi_xy_twin)
        ),
    }


def _resolve_transition_kinematics(
    analysis_input: AnalysisInput,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-transition (|step displacement|, |heading delta|), each (episodes, time - 1)."""
    kinematics = analysis_input.kinematics
    if kinematics is not None and kinematics.ndim == 3 and kinematics.shape[-1] >= 2:
        return np.abs(kinematics[:, 1:, 0]), np.abs(kinematics[:, 1:, 1])
    if analysis_input.heading is None:
        return None
    displacement = np.linalg.norm(
        analysis_input.position_xy[:, 1:] - analysis_input.position_xy[:, :-1], axis=-1
    )
    heading_delta = analysis_input.heading[:, 1:] - analysis_input.heading[:, :-1]
    heading_delta = (heading_delta + np.pi) % (2.0 * np.pi) - np.pi
    return np.abs(displacement), np.abs(heading_delta)


def _rotation_variance_fraction(
    representation: np.ndarray,
    valid: np.ndarray,
    analysis_input: AnalysisInput,
) -> tuple[float, int]:
    """Fraction of the pooled one-step <(dA)^2> on rotation-dominated transitions."""
    transition_kinematics = _resolve_transition_kinematics(analysis_input)
    if transition_kinematics is None:
        return float("nan"), 0
    displacement, heading_delta = transition_kinematics
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if not pair_valid.any():
        return float("nan"), 0
    displacement_rms = float(np.sqrt(np.mean(np.square(displacement[pair_valid]))))
    heading_delta_rms = float(np.sqrt(np.mean(np.square(heading_delta[pair_valid]))))
    if heading_delta_rms <= 0.0:
        return 0.0, 0
    if displacement_rms <= 0.0:
        rotation_dominated = heading_delta > 0.0
    else:
        rotation_dominated = heading_delta / heading_delta_rms > displacement / displacement_rms
    squared_change_per_step = np.zeros(pair_valid.shape, dtype=np.float64)
    num_units = representation.shape[-1]
    for start_index in range(0, num_units, _UNIT_CHUNK_SIZE):
        chunk = representation[..., start_index : start_index + _UNIT_CHUNK_SIZE]
        squared_change_per_step += np.square(chunk[:, 1:] - chunk[:, :-1], dtype=np.float64).sum(
            axis=-1
        )
    total = float(squared_change_per_step[pair_valid].sum())
    if total <= 0.0:
        return float("nan"), int((rotation_dominated & pair_valid).sum())
    rotation_total = float(squared_change_per_step[rotation_dominated & pair_valid].sum())
    return rotation_total / total, int((rotation_dominated & pair_valid).sum())


@dataclass(slots=True)
class ExcessStabilityModule:
    """Per-unit Psi_obs vs held-out xy/pose twins at a fixed lag, plus rotation split."""

    name: str = "excess_stability"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(
        self, analysis_input: AnalysisInput, output_dir: Path, config: dict[str, Any]
    ) -> AnalysisResult:
        representation = np.asarray(analysis_input.representation)
        if representation.ndim != 3:
            raise ValueError(f"Expected representation [N, T, D], got {representation.shape}.")
        num_episodes, num_steps, _ = representation.shape
        if analysis_input.valid_mask is None:
            valid = np.ones((num_episodes, num_steps), dtype=bool)
        else:
            valid = np.asarray(analysis_input.valid_mask, dtype=bool)
        lag = int(config.get("excess_stability_lag", DEFAULT_LAG))

        clipped_negative_fraction = float((representation[valid] < 0).mean())
        representation = np.clip(representation, 0.0, None)
        positions = np.asarray(analysis_input.position_xy)
        heading = analysis_input.heading
        if heading is not None:
            heading = np.asarray(heading)

        num_fit_episodes = num_episodes // 2
        if num_fit_episodes == 0:
            fit_episodes = eval_episodes = np.arange(num_episodes)
        else:
            episode_order = np.random.default_rng(_TWIN_SPLIT_SEED).permutation(num_episodes)
            fit_episodes = np.sort(episode_order[:num_fit_episodes])
            eval_episodes = np.sort(episode_order[num_fit_episodes:])
        bounds = infer_bounds(positions.reshape(-1, 2)[valid.reshape(-1)])

        num_bins_x = int(config.get("num_bins_x", 60))
        num_bins_y = int(config.get("num_bins_y", 60))
        smoothing_sigma = float(config["smoothing_sigma"])
        min_occupancy = float(config.get("min_occupancy", 1e-6))

        def twin_psi_at(bins_x: int, bins_y: int) -> TwinPsi:
            return compute_twin_psi(
                representation,
                positions,
                heading,
                valid,
                lag,
                fit_episodes=fit_episodes,
                eval_episodes=eval_episodes,
                num_bins_x=bins_x,
                num_bins_y=bins_y,
                smoothing_sigma=smoothing_sigma,
                min_occupancy=min_occupancy,
                bounds=bounds,
            )

        main_psi = twin_psi_at(num_bins_x, num_bins_y)
        xy_ratios = _finite_ratio(main_psi.psi_observed, main_psi.psi_xy_twin)
        pose_ratios = _finite_ratio(main_psi.psi_observed, main_psi.psi_pose_twin)
        pose_vs_xy_ratios = _finite_ratio(main_psi.psi_pose_twin, main_psi.psi_xy_twin)
        finite_xy_ratios = xy_ratios[np.isfinite(xy_ratios)]
        if finite_xy_ratios.size:
            fraction_below = float(np.mean(finite_xy_ratios < 0.9))
            fraction_above = float(np.mean(finite_xy_ratios > 1.1))
        else:
            fraction_below = float("nan")
            fraction_above = float("nan")

        rotation_fraction, num_rotation_steps = _rotation_variance_fraction(
            representation, valid, analysis_input
        )

        metrics = {
            **_ratio_summaries(main_psi, ""),
            "fraction_units_ratio_below_0_9": fraction_below,
            "fraction_units_ratio_above_1_1": fraction_above,
            "fraction_units_dropped": float(np.mean(~np.isfinite(xy_ratios))),
            "num_units_unmeasurable_xy_twin": float(_count_unmeasurable(main_psi.psi_xy_twin)),
            "num_units_unmeasurable_pose_twin": float(_count_unmeasurable(main_psi.psi_pose_twin)),
            "rotation_variance_fraction": rotation_fraction,
        }
        for sweep_bins in _SWEEP_BIN_COUNTS:
            metrics.update(
                _ratio_summaries(twin_psi_at(sweep_bins, sweep_bins), f"_bins{sweep_bins}")
            )

        module_dir = output_dir / self.name
        figure_path = (
            module_dir / f"excess_stability__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        apply_publication_style()
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        if finite_xy_ratios.size:
            axis.hist(finite_xy_ratios, bins=40, color="#4C72B0", alpha=0.85)
        axis.axvline(1.0, color="#444444", linewidth=0.8, linestyle="--")
        axis.set_xlabel(f"Excess stability ratio vs xy twin (lag {lag}, held-out)")
        axis.set_ylabel("Units")
        despine(axis)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        return AnalysisResult(
            metrics=metrics,
            per_unit_metrics={
                "excess_stability_ratio": xy_ratios,
                "excess_vs_pose_twin_ratio": pose_ratios,
                "pose_vs_xy_twin_ratio": pose_vs_xy_ratios,
                "psi_observed": main_psi.psi_observed,
                "psi_xy_twin": main_psi.psi_xy_twin,
                "psi_pose_twin": main_psi.psi_pose_twin,
            },
            figures={"excess_stability_histogram": figure_path},
            tables={},
            metadata={
                "lag": lag,
                "num_units_with_finite_ratio": int(finite_xy_ratios.size),
                "held_out_twin": bool(num_fit_episodes > 0),
                "twin_split_seed": _TWIN_SPLIT_SEED,
                "num_twin_fit_episodes": int(len(fit_episodes)),
                "num_twin_eval_episodes": int(len(eval_episodes)),
                "sweep_bin_counts": list(_SWEEP_BIN_COUNTS),
                "smoothing_sigma": smoothing_sigma,
                "clipped_negative_fraction": clipped_negative_fraction,
                "pose_twin_available": bool(heading is not None),
                "num_rotation_dominated_steps": num_rotation_steps,
                "rotation_criterion": (
                    "RMS-normalized |heading delta| exceeds RMS-normalized "
                    "|step displacement| on the transition arriving at each step"
                ),
            },
        )
