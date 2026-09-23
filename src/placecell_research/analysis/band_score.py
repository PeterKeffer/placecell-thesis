"""Band/stripe-cell score: 1D spatial periodicity x directional anisotropy of the rate map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import AnalysisInput, AnalysisResult
from .helpers import get_or_compute_rate_maps
from .world_overlay import overlay_bounds, resolve_world_overlay


def band_score(rate_map: np.ndarray) -> float:
    """Periodicity x directional anisotropy of a single rate map."""
    grid = np.nan_to_num(np.asarray(rate_map, dtype=np.float64), nan=0.0)
    grid = grid - grid.mean()
    side = min(grid.shape)
    power = np.abs(np.fft.fftshift(np.fft.fft2(grid))) ** 2
    height, width = power.shape
    center_y, center_x = height // 2, width // 2
    rows, cols = np.indices((height, width))
    radius = np.sqrt((rows - center_y) ** 2 + (cols - center_x) ** 2)
    frequency_mask = (radius >= 2) & (radius <= side // 2 - 1)
    total_power = power[frequency_mask].sum()
    if total_power < 1e-9:
        return 0.0
    peak_index = int(np.argmax(np.where(frequency_mask, power, 0.0)))
    peak_y, peak_x = np.unravel_index(peak_index, power.shape)
    peak_radius = np.sqrt((peak_y - center_y) ** 2 + (peak_x - center_x) ** 2)
    ring_mask = frequency_mask & (np.abs(radius - peak_radius) <= 1.5)
    ring_power = power[ring_mask].sum()
    periodicity = ring_power / total_power
    angle = np.arctan2(rows - center_y, cols - center_x)
    peak_angle = np.arctan2(peak_y - center_y, peak_x - center_x)
    angle_to_peak = np.angle(np.exp(1j * (angle - peak_angle)))
    on_axis = ring_mask & (
        (np.abs(angle_to_peak) <= np.pi / 6) | (np.abs(np.abs(angle_to_peak) - np.pi) <= np.pi / 6)
    )
    anisotropy = power[on_axis].sum() / (ring_power + 1e-9)
    return float(periodicity * anisotropy)


@dataclass(slots=True)
class BandScoreModule:
    """Per-unit band/stripe-cell scoring."""

    name: str = "band_score"
    cost_tier: str = "standard"

    def required_representations(self) -> set[str]:
        return set()

    def run(self, analysis_input: AnalysisInput, output_dir: Path, config: dict) -> AnalysisResult:
        world_overlay = resolve_world_overlay(
            str(analysis_input.metadata.get("env_id", "")),
            analysis_input.metadata.get("env_kwargs"),
        )
        world_bounds = overlay_bounds(world_overlay) if world_overlay is not None else None
        rate_map_result = get_or_compute_rate_maps(
            analysis_input,
            num_bins_x=int(config.get("num_bins_x", 60)),
            num_bins_y=int(config.get("num_bins_y", 60)),
            smoothing_sigma=float(config.get("smoothing_sigma", 0.4)),
            min_occupancy=float(config.get("min_occupancy", 1e-6)),
            bounds=world_bounds,
        )
        scores = np.asarray(
            [band_score(rate_map) for rate_map in rate_map_result.rate_maps], dtype=np.float32
        )
        threshold = float(config.get("band_score_threshold", 0.35))
        return AnalysisResult(
            metrics={
                "mean_band_score": float(scores.mean()) if len(scores) else 0.0,
                "max_band_score": float(scores.max()) if len(scores) else 0.0,
                "fraction_band": float((scores > threshold).mean()) if len(scores) else 0.0,
            },
            per_unit_metrics={"band_score": scores},
            figures={},
            tables={},
        )
