"""Redundancy and effective-dimensionality analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.rate_map_kernels import sample_valid_steps
from .base import AnalysisInput, AnalysisResult
from .figures import despine
from .timing import log_timing, record_timing


def _safe_probability(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    total = float(values.sum())
    if total <= epsilon:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip(values / total, epsilon, 1.0)


@dataclass(slots=True)
class RedundancyMetricsModule:
    """Summarize effective dimensionality, redundancy, and dead units."""

    name: str = "redundancy_metrics"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        timing_seconds: dict[str, float] = {}

        section_started_at = perf_counter()
        sampled_steps = sample_valid_steps(
            analysis_input.representation,
            analysis_input.position_xy,
            analysis_input.valid_mask,
            max_samples=int(config.get("redundancy_max_samples", 4096)),
            random_seed=int(config.get("redundancy_random_seed", 0)),
        )
        record_timing(timing_seconds, "sample_valid_steps", section_started_at)
        sampled = sampled_steps.values
        if sampled.size == 0:
            raise ValueError("Redundancy analysis needs at least one valid timestep.")
        section_started_at = perf_counter()
        centered = sampled - sampled.mean(axis=0, keepdims=True)
        unit_std = sampled.std(axis=0, dtype=np.float32)
        unit_peak_abs = np.max(np.abs(sampled), axis=0).astype(np.float32, copy=False)
        dead_unit_mask = ((unit_std <= 1e-6) | (unit_peak_abs <= 1e-6)).astype(bool, copy=False)
        record_timing(timing_seconds, "moments", section_started_at)

        section_started_at = perf_counter()
        covariance = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
        eigenvalues = np.linalg.eigvalsh(covariance.astype(np.float64, copy=False))
        eigenvalues = np.clip(np.sort(eigenvalues)[::-1], 0.0, None)
        explained_variance_ratio = (
            eigenvalues / float(eigenvalues.sum())
            if float(eigenvalues.sum()) > 1e-12
            else np.zeros_like(eigenvalues, dtype=np.float64)
        )
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        nonzero_probabilities = _safe_probability(eigenvalues)
        effective_rank = float(
            np.exp(-np.sum(nonzero_probabilities * np.log(nonzero_probabilities)))
        )
        participation_ratio = float(
            (np.square(eigenvalues.sum()) / np.square(eigenvalues).sum())
            if np.square(eigenvalues).sum() > 1e-12
            else 0.0
        )
        record_timing(timing_seconds, "eigendecomposition", section_started_at)

        section_started_at = perf_counter()
        if centered.shape[1] > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation_matrix = np.corrcoef(centered, rowvar=False)
            upper_triangle = correlation_matrix[np.triu_indices(centered.shape[1], k=1)]
            upper_triangle = upper_triangle[np.isfinite(upper_triangle)]
        else:
            upper_triangle = np.zeros((0,), dtype=np.float64)
        absolute_pairwise_correlation = np.abs(upper_triangle)
        record_timing(timing_seconds, "pairwise_correlations", section_started_at)

        section_started_at = perf_counter()
        module_dir = output_dir / self.name
        figure_path = (
            module_dir
            / f"redundancy_metrics__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))

        variance_axis = axes[0]
        x_values = np.arange(1, len(cumulative_explained_variance) + 1, dtype=np.int32)
        variance_axis.plot(x_values, cumulative_explained_variance, color="#1167B1", linewidth=2.0)
        variance_axis.set_title("Cumulative Explained Variance")
        variance_axis.set_xlabel("Principal component")
        variance_axis.set_ylabel("Cumulative variance ratio")
        variance_axis.set_ylim(0.0, 1.02)
        variance_axis.grid(alpha=0.3)

        correlation_axis = axes[1]
        if absolute_pairwise_correlation.size > 0:
            correlation_axis.hist(
                absolute_pairwise_correlation,
                bins=32,
                color="#C44E52",
                alpha=0.85,
            )
        correlation_axis.set_title("Pairwise Unit Correlation")
        correlation_axis.set_xlabel("|corr(unit_i, unit_j)|")
        correlation_axis.set_ylabel("Pair count")
        correlation_axis.grid(alpha=0.3)

        despine(variance_axis)
        despine(correlation_axis)

        figure.suptitle(
            f"{analysis_input.source_name} redundancy metrics\n"
            f"dead {int(np.count_nonzero(dead_unit_mask))}/{len(dead_unit_mask)}"
            f" | eff rank {effective_rank:.1f}"
            f" | PR {participation_ratio:.1f}",
            fontsize=12,
        )
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        record_timing(timing_seconds, "render_figure", section_started_at)
        log_timing(
            self.name,
            analysis_input.source_name,
            analysis_input.split_name,
            timing_seconds,
            config=config,
        )

        return AnalysisResult(
            metrics={
                "dead_unit_fraction": (
                    float(np.mean(dead_unit_mask)) if len(dead_unit_mask) else 0.0
                ),
                "active_unit_count": float(len(dead_unit_mask) - np.count_nonzero(dead_unit_mask)),
                "effective_rank": effective_rank,
                "participation_ratio": participation_ratio,
                "mean_abs_pairwise_unit_correlation": float(np.mean(absolute_pairwise_correlation))
                if absolute_pairwise_correlation.size
                else 0.0,
                "median_abs_pairwise_unit_correlation": float(
                    np.median(absolute_pairwise_correlation)
                )
                if absolute_pairwise_correlation.size
                else 0.0,
            },
            per_unit_metrics={
                "unit_std": unit_std.astype(np.float32, copy=False),
                "unit_peak_abs_activation": unit_peak_abs.astype(np.float32, copy=False),
                "dead_unit_mask": dead_unit_mask.astype(np.float32, copy=False),
            },
            figures={"redundancy_metrics": figure_path},
            tables={},
            metadata={
                "visualization": "redundancy_summary",
                "redundancy_sample_count": int(sampled.shape[0]),
                "redundancy_timing_seconds": timing_seconds,
            },
        )
