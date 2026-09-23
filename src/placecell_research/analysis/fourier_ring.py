"""Fourier ring score: is there a band-pass ring in the population spatial power spectrum?"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..numerics.fourier_ring import radial_power_spectrum, ring_metrics
from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .world_overlay import overlay_bounds, resolve_world_overlay


@dataclass(slots=True)
class FourierRingModule:
    """Population band-pass-ring detection on the spatial power spectrum of a representation."""

    name: str = "fourier_ring"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(
        self, analysis_input: AnalysisInput, output_dir: Path, config: dict[str, Any]
    ) -> AnalysisResult:
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=int(config.get("fourier_ring_num_bins", 48)),
            num_bins_y=int(config.get("fourier_ring_num_bins", 48)),
            smoothing_sigma=float(config.get("fourier_ring_smoothing_sigma", 0.4)),
            min_occupancy=float(config.get("min_occupancy", 1e-6)),
            bounds=world_bounds,
        )
        band_pass_ratio = float(config.get("fourier_ring_band_pass_ratio", 1.2))
        rate_maps = np.asarray(rate_map_result.rate_maps, dtype=np.float64)

        population = ring_metrics(rate_maps, band_pass_ratio=band_pass_ratio)
        per_unit_score = np.array(
            [ring_metrics(unit_map, band_pass_ratio=band_pass_ratio)["ring_score"]
             for unit_map in rate_maps],
            dtype=np.float32,
        )
        per_unit_peak = np.array(
            [ring_metrics(unit_map, band_pass_ratio=band_pass_ratio)["ring_peak_frequency"]
             for unit_map in rate_maps],
            dtype=np.float32,
        )
        fraction_band_pass = (
            float(((per_unit_score >= band_pass_ratio) & (per_unit_peak > 1.0)).mean())
            if len(per_unit_score) else 0.0
        )
        frequencies, radial_power = radial_power_spectrum(rate_maps)
        figure_path = self._render(
            analysis_input, output_dir, frequencies, radial_power, population
        )
        return AnalysisResult(
            metrics={
                "ring_score": population["ring_score"],
                "ring_peak_frequency": population["ring_peak_frequency"],
                "ring_peak_relative_power": population["ring_peak_relative_power"],
                "is_band_pass": population["is_band_pass"],
                "fraction_units_band_pass": fraction_band_pass,
            },
            per_unit_metrics={"ring_score": per_unit_score, "ring_peak_frequency": per_unit_peak},
            figures={"fourier_ring": figure_path},
            tables={},
            metadata={
                "radial_frequencies": frequencies.tolist(),
                "radial_power": radial_power.tolist(),
            },
        )

    def _render(
        self,
        analysis_input: AnalysisInput,
        output_dir: Path,
        frequencies: np.ndarray,
        radial_power: np.ndarray,
        population: dict[str, float],
    ) -> Path:
        """Plot the radial power spectrum P(|k|): a peak at |k| > 1 IS the band-pass ring."""
        module_dir = output_dir / self.name
        figure_path = (
            module_dir
            / f"fourier_ring__{analysis_input.source_name}__{analysis_input.split_name}.png"
        )
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        axis.plot(frequencies[1:], radial_power[1:], color="#1167B1", linewidth=2.0)
        peak_frequency = float(population["ring_peak_frequency"])
        axis.axvline(peak_frequency, color="#C44E52", linestyle="--", linewidth=1.2)
        axis.set_xlabel("spatial frequency |k| (cycles / arena)")
        axis.set_ylabel("mean power (DC removed)")
        axis.grid(alpha=0.3)
        verdict = "RING (band-pass)" if population["is_band_pass"] >= 1.0 else "no ring (low-pass)"
        figure.suptitle(
            f"{analysis_input.source_name} Fourier ring | peak |k|={peak_frequency:.0f} | "
            f"ring_score={population['ring_score']:.2f} | {verdict}",
            fontsize=11,
        )
        figure.tight_layout()
        figure.savefig(figure_path, dpi=160)
        plt.close(figure)
        return figure_path
