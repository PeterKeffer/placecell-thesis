"""Radially averaged spatial power spectrum and the band-pass ring it may contain."""

from __future__ import annotations

import numpy as np


def radial_power_spectrum(rate_maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population radially-averaged spatial power spectrum."""
    maps = np.asarray(rate_maps, dtype=np.float64)
    if maps.ndim == 2:
        maps = maps[None]
    _, height, width = maps.shape
    center_y, center_x = height // 2, width // 2
    rows, cols = np.indices((height, width))
    radius = np.round(np.sqrt((rows - center_y) ** 2 + (cols - center_x) ** 2)).astype(int)
    max_radius = int(min(center_y, center_x))

    accumulated_power = np.zeros((height, width), dtype=np.float64)
    used_units = 0
    for unit_map in maps:
        finite = np.isfinite(unit_map)
        if int(finite.sum()) < 4:
            continue
        filled = np.where(finite, unit_map, float(unit_map[finite].mean()))
        filled = filled - filled.mean()
        accumulated_power += np.abs(np.fft.fftshift(np.fft.fft2(filled))) ** 2
        used_units += 1

    frequencies = np.arange(max_radius + 1, dtype=np.float64)
    if used_units == 0:
        return frequencies, np.zeros(max_radius + 1, dtype=np.float64)
    mean_power = accumulated_power / used_units
    radial_power = np.array(
        [float(mean_power[radius == r].mean()) if np.any(radius == r) else 0.0
         for r in range(max_radius + 1)],
        dtype=np.float64,
    )
    return frequencies, radial_power


def ring_metrics(rate_maps: np.ndarray, *, band_pass_ratio: float = 1.2) -> dict[str, float]:
    """Band-pass-ring metrics from the radial power spectrum."""
    frequencies, power = radial_power_spectrum(rate_maps)
    empty = {
        "ring_score": 0.0,
        "ring_peak_frequency": 0.0,
        "ring_peak_relative_power": 0.0,
        "is_band_pass": 0.0,
    }
    if power.shape[0] < 3 or power[1:].sum() <= 0.0:
        return empty
    nonzero_power = power[1:]
    peak_index = int(np.argmax(nonzero_power)) + 1
    peak_power = float(power[peak_index])
    low_frequency_power = float(power[1])
    ring_score = peak_power / (low_frequency_power + 1e-12)
    return {
        "ring_score": float(ring_score),
        "ring_peak_frequency": float(frequencies[peak_index]),
        "ring_peak_relative_power": float(peak_power / float(nonzero_power.sum())),
        "is_band_pass": float(peak_index > 1 and ring_score >= band_pass_ratio),
    }
