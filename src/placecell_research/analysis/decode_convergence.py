"""Within-episode decode convergence."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from placecell_research.evaluation.decode import (
    chunked_ridge_predict,
    episode_level_decode_skip_reason,
    fit_position_ridge_decoder,
)

from .base import AnalysisInput, AnalysisResult
from .figures import apply_publication_style, despine

TRAINED_HORIZON_STEPS = 2048

STEP_WINDOWS: tuple[tuple[int, int], ...] = (
    (1, 8),
    (9, 64),
    (1, TRAINED_HORIZON_STEPS),
    (TRAINED_HORIZON_STEPS + 1, 2 * TRAINED_HORIZON_STEPS),
)

ROLLING_MEDIAN_WINDOW = 5
CONVERGENCE_THRESHOLD_FACTOR = 1.1
MIN_EPISODES_PER_STEP = 2


def _rolling_median(curve: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling nanmedian with edge padding, same length as curve."""
    half_window = window // 2
    padded = np.pad(curve, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def _pooled_mean(errors: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(errors[mask])) if np.any(mask) else float("nan")


def window_metric_name(first_step: int, last_step: int) -> str:
    """Metric key for one pooled step window, 1-based and inclusive at both ends."""
    return f"mean_error_steps_{first_step}_{last_step}"


def emitted_step_windows(num_steps: int) -> tuple[tuple[int, int], ...]:
    """The windows an episode of num_steps fills completely."""
    return tuple(window for window in STEP_WINDOWS if window[1] <= num_steps)


@dataclass(slots=True)
class DecodeConvergenceModule:
    """Step-aligned XY decode error within held-out episodes."""

    name: str = "decode_convergence"
    cost_tier: str = "light"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        representation = analysis_input.representation
        if representation.ndim != 3:
            raise ValueError(f"Expected representation [N, T, D], got {representation.shape}.")
        num_episodes, num_steps, _ = representation.shape
        flat_codes = representation.reshape(-1, representation.shape[-1])
        flat_positions = analysis_input.position_xy.reshape(-1, 2)
        if analysis_input.valid_mask is None:
            valid_rows = np.arange(num_episodes * num_steps, dtype=np.int64)
        else:
            valid_rows = np.flatnonzero(
                analysis_input.valid_mask.reshape(-1).astype(bool, copy=False)
            ).astype(np.int64, copy=False)
        codes = flat_codes[valid_rows].astype(np.float32, copy=False)
        positions = flat_positions[valid_rows].astype(np.float32, copy=False)
        episode_ids = valid_rows // num_steps
        step_ids = valid_rows % num_steps

        skip_reason = episode_level_decode_skip_reason(episode_ids)
        if skip_reason is not None:
            return AnalysisResult(
                metrics={"decode_convergence_skipped": 1.0},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={"decode_convergence_skip_reason": skip_reason},
            )

        decoder = fit_position_ridge_decoder(
            codes,
            positions,
            train_fraction=float(config.get("decode_train_fraction", 0.8)),
            alpha=float(config.get("decode_ridge_alpha", 1e-3)),
            episode_ids=episode_ids,
        )
        validation_indices = decoder.validation_indices
        predictions = chunked_ridge_predict(decoder.ridge, codes, validation_indices)
        validation_errors = np.linalg.norm(predictions - positions[validation_indices], axis=1)
        validation_step_ids = step_ids[validation_indices]
        validation_episode_ids = episode_ids[validation_indices]

        unique_validation_episodes, episode_rows = np.unique(
            validation_episode_ids, return_inverse=True
        )
        error_matrix = np.full((len(unique_validation_episodes), num_steps), np.nan)
        error_matrix[episode_rows, validation_step_ids] = validation_errors

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            episode_counts_per_step = np.sum(np.isfinite(error_matrix), axis=0)
            median_curve = np.nanmedian(error_matrix, axis=0)
            mean_curve = np.nanmean(error_matrix, axis=0)
            quartile_low, quartile_high = np.nanpercentile(error_matrix, [25.0, 75.0], axis=0)
        step_unsupported = episode_counts_per_step < MIN_EPISODES_PER_STEP
        for curve in (median_curve, mean_curve, quartile_low, quartile_high):
            curve[step_unsupported] = np.nan

        if not np.any(np.isfinite(median_curve)):
            return AnalysisResult(
                metrics={"decode_convergence_skipped": 1.0},
                per_unit_metrics={},
                figures={},
                tables={},
                metadata={
                    "decode_convergence_skip_reason": (
                        f"no step has at least {MIN_EPISODES_PER_STEP} held-out episodes"
                    ),
                    "num_validation_episodes": int(len(unique_validation_episodes)),
                    "num_steps": int(num_steps),
                },
            )

        last_quarter_start = num_steps - max(num_steps // 4, 1)
        windows = emitted_step_windows(num_steps)
        window_metrics = {
            window_metric_name(first_step, last_step): _pooled_mean(
                validation_errors,
                (validation_step_ids >= first_step - 1) & (validation_step_ids <= last_step - 1),
            )
            for first_step, last_step in windows
        }
        asymptote_error = _pooled_mean(validation_errors, validation_step_ids >= last_quarter_start)
        initial_error = window_metrics.get(window_metric_name(*STEP_WINDOWS[0]), float("nan"))
        if np.isfinite(asymptote_error) and asymptote_error > 0.0:
            error_ratio_initial_over_asymptote = initial_error / asymptote_error
        else:
            error_ratio_initial_over_asymptote = float("nan")

        rolling_median_curve = _rolling_median(median_curve, ROLLING_MEDIAN_WINDOW)
        convergence_step = float("nan")
        if np.isfinite(asymptote_error):
            threshold = CONVERGENCE_THRESHOLD_FACTOR * asymptote_error
            below_threshold = np.flatnonzero(
                np.isfinite(rolling_median_curve) & (rolling_median_curve <= threshold)
            )
            if below_threshold.size:
                convergence_step = float(below_threshold[0] + 1)

        module_dir = output_dir / self.name
        figure_path = (
            module_dir / f"decode_convergence__{analysis_input.source_name}"
            f"__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        apply_publication_style()
        figure, axis = plt.subplots(figsize=(6.0, 4.0))
        step_axis = np.arange(1, num_steps + 1)
        plot_mask = np.isfinite(median_curve)
        axis.fill_between(
            step_axis[plot_mask],
            quartile_low[plot_mask],
            quartile_high[plot_mask],
            color="#4C72B0",
            alpha=0.2,
            linewidth=0.0,
            label="interquartile range",
        )
        axis.plot(
            step_axis[plot_mask],
            median_curve[plot_mask],
            color="#4C72B0",
            linewidth=1.6,
            label="median over episodes",
        )
        axis.plot(
            step_axis[plot_mask],
            mean_curve[plot_mask],
            color="#DD8452",
            linewidth=1.0,
            label="mean over episodes",
        )
        if np.isfinite(asymptote_error):
            axis.axhline(
                asymptote_error,
                color="#444444",
                linewidth=0.8,
                linestyle="--",
                label="asymptote (last quarter)",
            )
        axis.set_xscale("log")
        axis.set_xlabel("Step within episode")
        axis.set_ylabel("Decode error (arena units)")
        axis.legend(frameon=False, fontsize=8)
        despine(axis)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)

        return AnalysisResult(
            metrics={
                **window_metrics,
                "asymptote_error": asymptote_error,
                "convergence_step": convergence_step,
                "error_ratio_initial_over_asymptote": error_ratio_initial_over_asymptote,
            },
            per_unit_metrics={},
            figures={"decode_convergence_curve": figure_path},
            tables={},
            metadata={
                "num_validation_episodes": int(len(unique_validation_episodes)),
                "num_steps": int(num_steps),
                "emitted_step_windows": [list(window) for window in windows],
                "last_quarter_start_step": int(last_quarter_start + 1),
                "rolling_median_window": ROLLING_MEDIAN_WINDOW,
                "convergence_threshold_factor": CONVERGENCE_THRESHOLD_FACTOR,
                "min_episodes_per_step": MIN_EPISODES_PER_STEP,
                "decode_train_size": int(len(decoder.train_indices)),
                "decode_validation_size": int(len(validation_indices)),
            },
        )
