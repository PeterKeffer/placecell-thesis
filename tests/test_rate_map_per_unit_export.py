from __future__ import annotations

from pathlib import Path

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.rate_maps import _RateMapModuleBase


def _synthetic_place_activity(
    positions: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    width: float = 0.18,
) -> np.ndarray:
    squared_distance = (positions[..., 0] - center_x) ** 2 + (positions[..., 1] - center_y) ** 2
    return np.exp(-squared_distance / max(width ** 2, 1e-6)).astype(np.float32)


def _wallgap_analysis_input() -> AnalysisInput:
    episodes = 5
    steps = 48
    position_sequences = []
    representation_sequences = []
    for _episode_index in range(episodes):
        x_positions = np.linspace(-7.0, 7.0, steps, dtype=np.float32)
        y_positions = 8.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32))
        positions = np.stack([x_positions, y_positions], axis=-1)
        position_sequences.append(positions)
        unit_0 = _synthetic_place_activity(positions, center_x=-3.5, center_y=-1.5, width=1.8)
        unit_1 = _synthetic_place_activity(positions, center_x=4.5, center_y=2.5, width=1.8)
        unit_2 = np.linspace(0.05, 0.15, steps, dtype=np.float32)
        representation_sequences.append(np.stack([unit_0, unit_1, unit_2], axis=-1))
    return AnalysisInput(
        representation=np.stack(representation_sequences, axis=0),
        position_xy=np.stack(position_sequences, axis=0),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((episodes, steps), dtype=bool),
        source_name="encoder.place_codes",
        label="encoder_place_cells",
        split_name="validation",
        metadata={"env_id": "MiniWorld-WallGapAsym-v0"},
    )


_BASE_CONFIG = {
    "num_bins_x": 24,
    "num_bins_y": 24,
    "smoothing_sigma": 0.8,
    "min_occupancy": 1e-6,
    "reliability_threshold_fraction": 0.3,
    "place_field_threshold_fraction": 0.35,
    "rate_map_panel_top_k": 2,
    "rate_map_grid_top_k": 2,
    "rate_map_panel_show_all_units": False,
}


def test_per_unit_export_writes_pngs_and_npz(tmp_path: Path) -> None:
    result = _RateMapModuleBase().run(
        _wallgap_analysis_input(),
        tmp_path,
        {**_BASE_CONFIG, "rate_map_export_per_unit": True},
    )

    units_dir = tmp_path / "rate_map_bundle" / "rate_map_units"
    npz_path = (
        tmp_path / "rate_map_bundle" / "rate_map_units__encoder.place_codes__validation.npz"
    )
    assert result.tables["rate_map_units"] == npz_path
    assert npz_path.exists()

    with np.load(npz_path) as bundle:
        unit_ids = bundle["unit_ids"]
        rate_maps = bundle["rate_maps"]
        occupancy = bundle["occupancy"]
        skaggs_bits = bundle["skaggs_bits"]

    assert unit_ids.shape == (2,)
    assert np.issubdtype(unit_ids.dtype, np.integer)
    assert rate_maps.shape == (2, 24, 24)
    assert rate_maps.dtype == np.float32
    assert np.isnan(rate_maps).any()
    assert occupancy.shape == (24, 24)
    assert skaggs_bits.shape == (2,)
    np.testing.assert_allclose(
        skaggs_bits,
        result.per_unit_metrics["spatial_information_bits"][unit_ids].astype(np.float32),
        rtol=1e-6,
    )

    for unit_id in unit_ids.tolist():
        png_path = units_dir / f"unit_{unit_id:04d}__encoder.place_codes__validation.png"
        assert png_path.exists()
        assert result.figures[f"rate_map_unit_{unit_id:04d}"] == png_path
    assert len(list(units_dir.glob("*.png"))) == 2
    assert result.metadata["rate_map_export_per_unit"] is True


def test_per_unit_export_covers_all_units_with_show_all_units(tmp_path: Path) -> None:
    result = _RateMapModuleBase().run(
        _wallgap_analysis_input(),
        tmp_path,
        {
            **_BASE_CONFIG,
            "rate_map_export_per_unit": True,
            "rate_map_panel_show_all_units": True,
        },
    )

    units_dir = tmp_path / "rate_map_bundle" / "rate_map_units"
    assert len(list(units_dir.glob("*.png"))) == 3
    with np.load(result.tables["rate_map_units"]) as bundle:
        assert bundle["unit_ids"].tolist() == [0, 1, 2]
        assert bundle["rate_maps"].shape == (3, 24, 24)


def test_per_unit_export_is_off_by_default(tmp_path: Path) -> None:
    result = _RateMapModuleBase().run(
        _wallgap_analysis_input(),
        tmp_path,
        dict(_BASE_CONFIG),
    )

    assert not (tmp_path / "rate_map_bundle" / "rate_map_units").exists()
    assert "rate_map_units" not in result.tables
    assert not any(key.startswith("rate_map_unit_") for key in result.figures)
    assert result.metadata["rate_map_export_per_unit"] is False
