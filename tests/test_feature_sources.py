"""Unit tests for downstream place-code feature sources."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from placecell_research.config.downstream_feature_sources import (
    PLACE_CODE_FEATURE_SOURCES,
    PLACE_CODE_SOURCES,
    STATS_REQUIRED_PLACE_CODE_SOURCES,
    PlaceCodeSourceSpec,
    missing_place_code_stats_sources,
)
from placecell_research.downstream.feature_sources import (
    PlaceCodeFeatureSource,
    StepContext,
    apply_place_code_source,
    transform_place_code_batch,
)
from placecell_research.downstream.place_code_stats import PlaceCodeStats


class _FakePlaceRuntime:
    """Minimal stand-in for PlaceCodeFeatureRuntime (no model loading)."""

    def __init__(
        self,
        code: np.ndarray,
        head_row_norms: np.ndarray,
        place_code_stats: PlaceCodeStats | None = None,
    ) -> None:
        self._code = np.asarray(code, dtype=np.float32)
        self.extractor = SimpleNamespace(
            head_row_norms=np.asarray(head_row_norms, dtype=np.float32)
        )
        self.feature_dim = int(self._code.shape[-1])
        self.place_code_stats = place_code_stats

    def reset(self) -> None:
        return None

    def extract_raw(self, context: StepContext, previous_action: int | None) -> np.ndarray:
        del context, previous_action
        return self._code.copy()


def _context() -> StepContext:
    return StepContext(
        rgb=np.zeros((3, 4, 4), dtype=np.float32),
        position_xy=np.zeros(2, dtype=np.float32),
        heading=0.0,
        kinematics=None,
        goal_position_xy=None,
    )


def test_place_code_source_specs_hold_transform_prescale_and_stats_metadata() -> None:
    assert set(PLACE_CODE_FEATURE_SOURCES) == set(PLACE_CODE_SOURCES)
    assert PLACE_CODE_SOURCES["place_codes_headnorm_l2"] == PlaceCodeSourceSpec(
        transform="l2",
        pre_scale="head_row_norm",
    )
    assert PLACE_CODE_SOURCES["place_codes_rms_l2"] == PlaceCodeSourceSpec(
        transform="active_rms_l2",
        requires_stats=True,
    )
    assert STATS_REQUIRED_PLACE_CODE_SOURCES == frozenset(
        name for name, spec in PLACE_CODE_SOURCES.items() if spec.requires_stats
    )
    assert missing_place_code_stats_sources(
        ["place_codes", "place_codes_zscore"],
        stats_available=False,
    ) == ["place_codes_zscore"]


def test_apply_place_code_source_owns_prescale_and_transform() -> None:
    codes = np.array([[0.0, 6.0, 0.0, 2.0]], dtype=np.float32)
    head_row_norms = np.array([1.0, 3.0, 1.0, 2.0], dtype=np.float32)

    result = apply_place_code_source(
        "place_codes_headnorm_l2",
        codes,
        head_row_norms=head_row_norms,
        stats=None,
    )

    expected = transform_place_code_batch(codes / head_row_norms, mode="l2")[0]
    np.testing.assert_allclose(result, expected.reshape(1, -1), rtol=1e-6)


def test_headnorm_l2_prescales_by_head_row_norm_then_l2() -> None:
    code = np.array([0.0, 6.0, 0.0, 2.0], dtype=np.float32)
    head_row_norms = np.array([1.0, 3.0, 1.0, 2.0], dtype=np.float32)
    runtime = _FakePlaceRuntime(code=code, head_row_norms=head_row_norms)

    source = PlaceCodeFeatureSource(runtime=runtime, source_name="place_codes_headnorm_l2")
    result = source.extract(_context(), previous_action=None)

    expected = transform_place_code_batch((code / head_row_norms)[None], mode="l2")[0]
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    plain_l2 = transform_place_code_batch(code[None], mode="l2")[0]
    assert not np.allclose(result, plain_l2)


def test_active_rms_uses_fixed_stats_from_shared_runtime() -> None:
    code = np.array([0.0, 4.0, -6.0], dtype=np.float32)
    stats = PlaceCodeStats(
        mean=np.zeros((3,), dtype=np.float32),
        std=np.ones((3,), dtype=np.float32),
        active_rms=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        sample_count=10,
        active_count=np.array([0, 5, 5], dtype=np.int64),
    )
    runtime = _FakePlaceRuntime(
        code=code,
        head_row_norms=np.ones((3,), dtype=np.float32),
        place_code_stats=stats,
    )

    source = PlaceCodeFeatureSource(runtime=runtime, source_name="place_codes_active_rms")
    result = source.extract(_context(), previous_action=None)

    np.testing.assert_allclose(result, [0.0, 2.0, -2.0], rtol=1e-6)


def test_rms_l2_is_active_rms_then_l2() -> None:
    code = np.array([0.0, 6.0, 0.0, 2.0], dtype=np.float32)
    stats = PlaceCodeStats(
        mean=np.zeros((4,), dtype=np.float32),
        std=np.ones((4,), dtype=np.float32),
        active_rms=np.array([1.0, 3.0, 1.0, 2.0], dtype=np.float32),
        sample_count=10,
        active_count=np.array([0, 5, 0, 5], dtype=np.int64),
    )
    runtime = _FakePlaceRuntime(
        code=code,
        head_row_norms=np.ones((4,), dtype=np.float32),
        place_code_stats=stats,
    )

    source = PlaceCodeFeatureSource(runtime=runtime, source_name="place_codes_rms_l2")
    result = source.extract(_context(), previous_action=None)

    scaled = transform_place_code_batch(code[None], mode="active_rms", stats=stats)
    expected = transform_place_code_batch(scaled, mode="l2")[0]
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, rtol=1e-6)
