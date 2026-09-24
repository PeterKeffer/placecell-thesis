"""Per-neuron heading/rate-map overlay figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.heading_rate_map_overlay import (
    HeadingRateMapOverlayModule,
    _compute_heading_rate_map_overlay_metrics,
)
from placecell_research.analysis.registry import ANALYSIS_MODULES


def _build_input() -> AnalysisInput:
    heading_centers = (
        (np.arange(16, dtype=np.float32) + 0.5) * (2.0 * np.pi / 16.0)
    )
    positions: list[tuple[float, float]] = []
    headings: list[float] = []
    directional_field: list[float] = []
    omnidirectional_field: list[float] = []
    preferred_heading = np.pi / 2.0

    for x in range(5):
        for y in range(5):
            is_field_position = x == 2 and y == 2
            for heading in heading_centers:
                tuning = float(np.exp(3.0 * np.cos(float(heading) - preferred_heading)))
                positions.append((float(x), float(y)))
                headings.append(float(heading))
                directional_field.append(tuning if is_field_position else 0.0)
                omnidirectional_field.append(1.0 if is_field_position else 0.0)

    steps = len(positions)
    representation = np.stack(
        [directional_field, omnidirectional_field], axis=1
    ).reshape(1, steps, 2)
    return AnalysisInput(
        representation=representation.astype(np.float32),
        position_xy=np.asarray(positions, dtype=np.float32).reshape(1, steps, 2),
        heading=np.asarray(headings, dtype=np.float32).reshape(1, steps),
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder",
        split_name="test",
    )


def _config() -> dict[str, float | int | bool]:
    return {
        "heading_rate_map_overlay_num_bins_x": 5,
        "heading_rate_map_overlay_num_bins_y": 5,
        "heading_rate_map_overlay_num_heading_bins": 16,
        "heading_rate_map_overlay_min_occupancy_per_heading_bin": 1,
        "heading_rate_map_overlay_top_k": 2,
        "heading_rate_map_overlay_page_size": 2,
        "heading_rate_map_overlay_render_dpi": 80,
        "heading_rate_map_overlay_smoothing_sigma": 0.0,
    }


def test_heading_rate_map_overlay_is_registered_for_config_use() -> None:
    assert "heading_rate_map_overlay" in ANALYSIS_MODULES
    assert ANALYSIS_MODULES["heading_rate_map_overlay"]().name == "heading_rate_map_overlay"


def test_heading_rate_map_overlay_metrics_expose_corridor_and_support(analysis_settings) -> None:
    analysis_input = _build_input()

    metrics = _compute_heading_rate_map_overlay_metrics(
        analysis_input,
        config=analysis_settings(**_config()),
        bounds=((-0.5, 4.5), (-0.5, 4.5)),
    )

    assert metrics.heading_bin_rates.shape == (2, 16)
    assert metrics.heading_bin_occupancy.shape == (16,)
    assert float(metrics.heading_bin_occupancy.sum()) == 400.0
    assert metrics.local_preferred_heading.shape == (2, 5, 5)
    assert metrics.local_support_fraction.shape == (2, 5, 5)
    assert metrics.activation_corridor_deg[0] < metrics.activation_corridor_deg[1]
    assert metrics.peak_to_mean[0] > metrics.peak_to_mean[1]
    assert metrics.local_support_fraction[0, 2, 2] == 1.0


def test_heading_rate_map_overlay_writes_paged_figure_and_table(
    tmp_path: Path, analysis_settings
) -> None:
    result = HeadingRateMapOverlayModule().run(
        _build_input(), tmp_path, analysis_settings(**_config())
    )

    assert result.figures["heading_rate_map_overlay_page_0"].exists()
    assert result.tables["per_unit_metrics"].exists()
    assert result.metrics["units_rendered"] == 2.0
    assert "activation_corridor_deg" in result.per_unit_metrics

    table_text = result.tables["per_unit_metrics"].read_text()
    assert "activation_corridor_deg" in table_text
    assert "heading_support_fraction" in table_text
