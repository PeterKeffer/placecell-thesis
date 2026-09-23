"""Layout invariants for the publication rate-map figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from placecell_research.analysis.rate_map_export import (
    _render_summary_panel,
    _render_support_map,
)
from placecell_research.analysis.rate_map_rendering import (
    _build_rate_map_grid_figure,
    _build_summary_panel_figure,
    _panel_metric_cmap_name,
)

_DIVERGING_CMAPS = {"coolwarm", "bwr", "seismic", "RdBu", "RdBu_r"}


def test_reliability_map_uses_sequential_colormap() -> None:
    for metric in ("quantile_thresholded_reliability", "thresholded_reliability"):
        cmap = _panel_metric_cmap_name(panel_metric_name=metric)
        assert cmap not in _DIVERGING_CMAPS, f"{metric} should be sequential, got {cmap}"
        assert cmap == "viridis"


def _fake_rate_maps(num_units: int = 3, ny: int = 14, nx: int = 12) -> np.ndarray:
    yy, xx = np.mgrid[0:ny, 0:nx]
    maps = np.zeros((num_units, ny, nx), dtype=np.float32)
    for unit_index in range(num_units):
        center_x = nx * (0.6 + 0.05 * unit_index)
        center_y = ny * 0.7
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        maps[unit_index] = 0.1 * np.exp(-(squared_distance / (2 * 2.0**2)))
    return maps


_BOUNDS = ((-7.0, 7.0), (-8.0, 8.0))


def _heatmap_overlay_texts(axis: plt.Axes) -> list[str]:
    return [text_artist.get_text() for text_artist in axis.texts]


def test_grid_metrics_render_as_caption_not_overlay() -> None:
    rate_maps = _fake_rate_maps()
    figure, heatmap_axes = _build_rate_map_grid_figure(
        source_name="encoder.place_codes",
        split_name="test",
        bounds=_BOUNDS,
        rate_maps=rate_maps,
        ranked_indices=np.array([0, 1, 2]),
        spatial_information_bits=np.array([0.61, 0.42, 0.30]),
        mean_panel_metric_inside_fields=np.array([0.61, 0.55, 0.40]),
        world_overlay=None,
        mean_spatial_information_bits=0.45,
        shared_color_scale=False,
        colormap_mode="turbo",
        panel_metric_name="quantile_thresholded_reliability",
        title_prefix="rate maps",
        show_captions=True,
    )
    try:
        assert heatmap_axes, "expected at least one heatmap cell"
        for axis in heatmap_axes:
            assert not any("SI" in text for text in _heatmap_overlay_texts(axis)), (
                "metric values must not be painted on top of the heatmap"
            )
        assert any("SI" in axis.get_xlabel() for axis in heatmap_axes), (
            "metrics should appear as a caption below the map"
        )
    finally:
        plt.close(figure)


def test_grid_clean_variant_drops_the_caption() -> None:
    rate_maps = _fake_rate_maps()
    figure, heatmap_axes = _build_rate_map_grid_figure(
        source_name="encoder.place_codes",
        split_name="test",
        bounds=_BOUNDS,
        rate_maps=rate_maps,
        ranked_indices=np.array([0, 1, 2]),
        spatial_information_bits=np.array([0.61, 0.42, 0.30]),
        mean_panel_metric_inside_fields=np.array([0.61, 0.55, 0.40]),
        world_overlay=None,
        mean_spatial_information_bits=0.45,
        shared_color_scale=False,
        colormap_mode="turbo",
        panel_metric_name="quantile_thresholded_reliability",
        title_prefix="rate maps",
        show_captions=False,
    )
    try:
        assert all(axis.get_xlabel() == "" for axis in heatmap_axes), (
            "clean grid must be maps-only (no metric caption)"
        )
    finally:
        plt.close(figure)


def test_panel_metrics_render_off_the_heatmap() -> None:
    rate_maps = _fake_rate_maps()
    panel_metric_maps = np.clip(rate_maps / (rate_maps.max() + 1e-9), 0.0, 1.0)
    figure, rate_axes = _build_summary_panel_figure(
        source_name="encoder.place_codes",
        split_name="test",
        bounds=_BOUNDS,
        rate_maps=rate_maps,
        panel_metric_maps=panel_metric_maps,
        ranked_indices=np.array([0, 1, 2]),
        field_masks=(rate_maps > rate_maps.max() * 0.5),
        spatial_information_bits=np.array([0.61, 0.42, 0.30]),
        mean_panel_metric_inside_fields=np.array([0.61, 0.55, 0.40]),
        panel_metric_supported_field_fraction=np.array([0.98, 0.90, 0.80]),
        split_half_correlations=np.array([0.98, 0.70, 0.50]),
        episode_rate_map_correlations=np.array([0.10, 0.20, 0.30]),
        field_counts=np.array([9, 3, 1]),
        world_overlay=None,
        mean_spatial_information_bits=0.45,
        shared_color_scale=False,
        colormap_mode="turbo",
        panel_metric_name="quantile_thresholded_reliability",
        panel_metric_fill_sigma_bins=0.0,
        title_prefix="spatial firing summary",
        show_captions=True,
    )
    try:
        assert rate_axes, "expected at least one unit row"
        for axis in rate_axes:
            assert not any("SI" in text for text in _heatmap_overlay_texts(axis)), (
                "metric values must not be painted on top of the rate map"
            )
        assert any("SI" in axis.get_xlabel() for axis in rate_axes), (
            "per-unit metrics should appear as a caption below the rate map"
        )
    finally:
        plt.close(figure)


def _panel_rate_axis_cmaps(rate_axes: list[plt.Axes]) -> list[str]:
    return [axis.images[0].get_cmap().name for axis in rate_axes if axis.images]


def _build_panel(rate_maps: np.ndarray):
    num_units = rate_maps.shape[0]
    return _build_summary_panel_figure(
        source_name="encoder.place_codes",
        split_name="test",
        bounds=((-7.0, 7.0), (-8.0, 8.0)),
        rate_maps=rate_maps,
        panel_metric_maps=np.clip(rate_maps / (np.abs(rate_maps).max() + 1e-9), 0.0, 1.0),
        ranked_indices=np.arange(num_units),
        field_masks=(rate_maps > rate_maps.max() * 0.5),
        spatial_information_bits=np.linspace(0.6, 0.2, num_units),
        mean_panel_metric_inside_fields=np.linspace(0.6, 0.3, num_units),
        panel_metric_supported_field_fraction=np.full(num_units, 0.9),
        split_half_correlations=np.full(num_units, 0.8),
        episode_rate_map_correlations=np.full(num_units, 0.2),
        field_counts=np.full(num_units, 2),
        world_overlay=None,
        mean_spatial_information_bits=0.4,
        shared_color_scale=False,
        colormap_mode="inferno",
        panel_metric_name="quantile_thresholded_reliability",
        panel_metric_fill_sigma_bins=0.0,
        title_prefix="spatial firing summary",
        show_captions=True,
    )


def test_support_map_renders_as_standalone_file(tmp_path: Path) -> None:
    out = tmp_path / "support_map.png"
    _render_support_map(
        out,
        bounds=_BOUNDS,
        support_counts=np.full((14, 12), 3.0, dtype=np.float32),
        world_overlay=None,
        title="Support Map\nColor = visiting episodes",
        render_dpi=80,
    )
    assert out.exists()


def test_panel_heatmaps_are_despined_with_world_axis_labels() -> None:
    figure, rate_axes = _build_panel(_fake_rate_maps(num_units=2))
    try:
        rate_axis = rate_axes[0]
        assert not rate_axis.spines["top"].get_visible(), "top spine should be removed"
        assert not rate_axis.spines["right"].get_visible(), "right spine should be removed"
        all_labels = [a.get_xlabel() for a in figure.axes] + [a.get_ylabel() for a in figure.axes]
        assert any("World" in label for label in all_labels), (
            "position axes labelled World x/World y"
        )
        assert not any("(a.u.)" in label for label in all_labels), (
            "no arbitrary-units jargon on axes"
        )
    finally:
        plt.close(figure)


def test_panel_clean_variant_labels_both_axes_world(tmp_path: Path, monkeypatch) -> None:
    rate_maps = _fake_rate_maps(num_units=2)
    num_units = rate_maps.shape[0]
    captured: dict[str, list[tuple[str, str]]] = {}
    real_savefig = matplotlib.figure.Figure.savefig

    def _spy_savefig(self, fname, *args, **kwargs):
        captured[str(fname)] = [(ax.get_xlabel(), ax.get_ylabel()) for ax in self.axes]
        return real_savefig(self, fname, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", _spy_savefig)

    clean = tmp_path / "panel_clean.png"
    _render_summary_panel(
        tmp_path / "panel.png",
        source_name="encoder.place_codes",
        split_name="test",
        bounds=_BOUNDS,
        rate_maps=rate_maps,
        panel_metric_maps=np.clip(rate_maps / (np.abs(rate_maps).max() + 1e-9), 0.0, 1.0),
        ranked_indices=np.arange(num_units),
        field_masks=(rate_maps > rate_maps.max() * 0.5),
        spatial_information_bits=np.linspace(0.6, 0.2, num_units),
        mean_panel_metric_inside_fields=np.linspace(0.6, 0.3, num_units),
        panel_metric_supported_field_fraction=np.full(num_units, 0.9),
        split_half_correlations=np.full(num_units, 0.8),
        episode_rate_map_correlations=np.full(num_units, 0.2),
        field_counts=np.full(num_units, 2),
        spike_positions_by_unit={},
        world_overlay=None,
        use_absolute_activations=False,
        mean_spatial_information_bits=0.45,
        shared_color_scale=False,
        colormap_mode="turbo",
        panel_metric_name="quantile_thresholded_reliability",
        panel_metric_fill_sigma_bins=0.0,
        show_spike_positions=False,
        title_prefix="spatial firing summary",
        render_dpi=80,
        clean_path=clean,
    )

    assert str(clean) in captured, "clean variant should have been saved"
    xlabels = [x for x, _ in captured[str(clean)]]
    ylabels = [y for _, y in captured[str(clean)]]
    assert "World x" in xlabels, f"clean x-axis must be World x, got {xlabels}"
    assert "World y" in ylabels, f"clean y-axis must be World y (the original bug), got {ylabels}"
    assert not any("a.u." in label for label in xlabels + ylabels), "no a.u. jargon in clean panel"


def test_panel_omits_support_column() -> None:
    num_units = 2
    figure, _ = _build_panel(_fake_rate_maps(num_units=num_units))
    try:
        titles = [axis.get_title() for axis in figure.axes]
        assert not any("Support" in title for title in titles), "support column should be removed"
        map_axes = [axis for axis in figure.axes if axis.images]
        assert len(map_axes) == 3 * num_units, (
            f"expected 3 map columns per row, got {len(map_axes) / num_units}"
        )
    finally:
        plt.close(figure)


def test_mostly_positive_population_uses_one_sequential_colormap() -> None:
    rate_maps = _fake_rate_maps(num_units=4)
    rate_maps[1, 0, 0] = -0.001
    figure, rate_axes = _build_panel(rate_maps)
    try:
        cmaps = _panel_rate_axis_cmaps(rate_axes)
        assert cmaps, "expected rate-map axes"
        assert len(set(cmaps)) == 1, f"all cells must share one colormap, got {set(cmaps)}"
        assert cmaps[0] == "inferno", f"non-negative population should use inferno, got {cmaps[0]}"
    finally:
        plt.close(figure)


def test_signed_population_uses_one_diverging_colormap() -> None:
    rate_maps = _fake_rate_maps(num_units=4)
    rate_maps[2] = rate_maps[2] - 0.06
    figure, rate_axes = _build_panel(rate_maps)
    try:
        cmaps = _panel_rate_axis_cmaps(rate_axes)
        assert len(set(cmaps)) == 1, f"signed population must be consistent, got {set(cmaps)}"
        assert cmaps[0] == "RdBu_r", (
            f"signed population should use the diverging map, got {cmaps[0]}"
        )
    finally:
        plt.close(figure)


def test_panel_world_axis_ticks_are_round_numbers() -> None:
    rate_maps = _fake_rate_maps()
    panel_metric_maps = np.clip(rate_maps / (rate_maps.max() + 1e-9), 0.0, 1.0)
    figure, rate_axes = _build_summary_panel_figure(
        source_name="encoder.place_codes",
        split_name="test",
        bounds=((-7.0, 7.0), (-8.0, 8.0)),
        rate_maps=rate_maps,
        panel_metric_maps=panel_metric_maps,
        ranked_indices=np.array([0, 1, 2]),
        field_masks=(rate_maps > rate_maps.max() * 0.5),
        spatial_information_bits=np.array([0.61, 0.42, 0.30]),
        mean_panel_metric_inside_fields=np.array([0.61, 0.55, 0.40]),
        panel_metric_supported_field_fraction=np.array([0.98, 0.90, 0.80]),
        split_half_correlations=np.array([0.98, 0.70, 0.50]),
        episode_rate_map_correlations=np.array([0.10, 0.20, 0.30]),
        field_counts=np.array([9, 3, 1]),
        world_overlay=None,
        mean_spatial_information_bits=0.45,
        shared_color_scale=False,
        colormap_mode="turbo",
        panel_metric_name="quantile_thresholded_reliability",
        panel_metric_fill_sigma_bins=0.0,
        title_prefix="spatial firing summary",
        show_captions=True,
    )
    try:
        axis = rate_axes[0]
        lower, upper = sorted(axis.get_ylim())
        visible_ticks = [tick for tick in axis.get_yticks() if lower - 1e-6 <= tick <= upper + 1e-6]
        assert visible_ticks, "expected visible y ticks within the view"
        for tick in visible_ticks:
            assert abs(tick - round(tick)) < 1e-6, f"axis tick {tick} is not a round number"
    finally:
        plt.close(figure)
