from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from placecell_research.analysis.base import AnalysisInput
from placecell_research.analysis.place_field_detection import PlaceFieldDetectionModule
from placecell_research.numerics.rate_map_kernels import compute_place_field_mask


def _analysis_input_stub() -> AnalysisInput:
    return AnalysisInput(
        representation=np.zeros((1, 1, 1), dtype=np.float32),
        position_xy=np.zeros((1, 1, 2), dtype=np.float32),
        heading=None,
        kinematics=None,
        actions=None,
        valid_mask=np.ones((1, 1), dtype=bool),
        source_name="encoder.place_codes",
        label="stub",
        split_name="test",
    )


def test_compute_place_field_mask_is_nan_safe_for_empty_and_nonempty_maps() -> None:
    mask, field_count, field_area = compute_place_field_mask(
        np.asarray([[np.nan, 0.5], [np.nan, 1.0]], dtype=np.float32),
        threshold_fraction=0.3,
    )

    assert field_count == 1
    assert field_area == 2.0
    assert mask.dtype == np.bool_

    empty_mask, empty_count, empty_area = compute_place_field_mask(
        np.full((2, 2), np.nan, dtype=np.float32),
        threshold_fraction=0.3,
    )

    assert empty_count == 0
    assert empty_area == 0.0
    assert not empty_mask.any()


def test_place_field_detection_uses_nan_safe_masks_and_current_smoothing_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_smoothing_sigma: list[float] = []

    def fake_get_or_compute_rate_maps(*args, **kwargs):
        captured_smoothing_sigma.append(float(kwargs["smoothing_sigma"]))
        return SimpleNamespace(
            rate_maps=np.asarray(
                [
                    [[np.nan, 0.5], [np.nan, 1.0]],
                    [[np.nan, np.nan], [np.nan, np.nan]],
                ],
                dtype=np.float32,
            )
        )

    monkeypatch.setattr(
        "placecell_research.analysis.place_field_detection.get_or_compute_rate_maps",
        fake_get_or_compute_rate_maps,
    )

    result = PlaceFieldDetectionModule().run(_analysis_input_stub(), tmp_path, {})

    assert captured_smoothing_sigma == [0.3]
    np.testing.assert_allclose(
        result.per_unit_metrics["field_count"], np.asarray([1.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        result.per_unit_metrics["field_area"], np.asarray([2.0, 0.0], dtype=np.float32)
    )
    assert result.metrics["mean_field_count"] == 0.5
    assert result.metrics["mean_field_area"] == 1.0
