from __future__ import annotations

import pytest
import torch

from placecell_research.spatial_model.types import ModuleOutputs, RepresentationBundle


def _make_composite_bundle() -> RepresentationBundle:
    return RepresentationBundle(
        modules={
            "encoder": ModuleOutputs(place_codes=torch.randn(2, 5, 16)),
            "predictor": ModuleOutputs(place_codes=torch.randn(2, 5, 16)),
            "teacher": ModuleOutputs(place_codes=torch.randn(2, 5, 16)),
        },
    )


def _make_custom_bundle() -> RepresentationBundle:
    return RepresentationBundle(
        modules={
            "stream_a": ModuleOutputs(place_codes=torch.randn(2, 5, 16)),
            "stream_b": ModuleOutputs(
                place_codes=torch.randn(2, 5, 16),
                hidden_state=torch.randn(2, 5, 32),
            ),
        },
    )


def test_get_representation_with_standard_modules() -> None:
    bundle = _make_composite_bundle()
    codes = bundle.get_representation("encoder.place_codes")
    assert codes.shape == (2, 5, 16)


def test_get_representation_with_pre_sparsifier_field() -> None:
    pre_sparsifier = torch.randn(2, 5, 16)
    bundle = RepresentationBundle(
        modules={"encoder": ModuleOutputs(pre_sparsifier=pre_sparsifier)},
    )

    assert torch.equal(bundle.get_representation("encoder.pre_sparsifier"), pre_sparsifier)
    assert "encoder.pre_sparsifier" in bundle.available_representations()


def test_get_representation_with_custom_modules() -> None:
    bundle = _make_custom_bundle()
    codes = bundle.get_representation("stream_a.place_codes")
    assert codes.shape == (2, 5, 16)
    hidden = bundle.get_representation("stream_b.hidden_state")
    assert hidden.shape == (2, 5, 32)


def test_get_representation_preserves_float64_values() -> None:
    bundle = RepresentationBundle(
        modules={"encoder": ModuleOutputs(place_codes=torch.randn(2, 5, 16, dtype=torch.float64))}
    )

    assert bundle.get_representation("encoder.place_codes").dtype == torch.float64


def test_available_representations_lists_all_modules() -> None:
    bundle = _make_custom_bundle()
    available = bundle.available_representations()
    assert "stream_a.place_codes" in available
    assert "stream_b.place_codes" in available
    assert "stream_b.hidden_state" in available


def test_get_representation_raises_on_missing_module() -> None:
    bundle = _make_custom_bundle()
    with pytest.raises(KeyError, match="not_a_module"):
        bundle.get_representation("not_a_module.place_codes")


def test_get_representation_raises_on_missing_field() -> None:
    bundle = _make_custom_bundle()
    with pytest.raises(KeyError, match="not_a_field"):
        bundle.get_representation("stream_a.not_a_field")


def test_infer_device() -> None:
    bundle = _make_custom_bundle()
    assert bundle.infer_device() == torch.device("cpu")


def test_infer_device_empty_bundle() -> None:
    bundle = RepresentationBundle(modules={})
    assert bundle.infer_device() == torch.device("cpu")


def test_get_auxiliary_searches_all_modules() -> None:
    outputs = ModuleOutputs(
        place_codes=torch.randn(2, 5, 16),
        auxiliary={"projected": torch.randn(2, 5, 8)},
    )
    bundle = RepresentationBundle(modules={"custom": outputs})
    result = bundle.get_auxiliary("projected")
    assert result.shape == (2, 5, 8)


def test_get_auxiliary_falls_back_to_auxiliary_outputs() -> None:
    bundle = RepresentationBundle(
        modules={"enc": ModuleOutputs(place_codes=torch.randn(2, 5, 16))},
        auxiliary_outputs={"global_thing": torch.randn(2, 5, 4)},
    )
    result = bundle.get_auxiliary("global_thing")
    assert result.shape == (2, 5, 4)


def test_get_auxiliary_raises_on_missing() -> None:
    bundle = RepresentationBundle(
        modules={"enc": ModuleOutputs(place_codes=torch.randn(2, 5, 16))},
    )
    with pytest.raises(KeyError, match="nope"):
        bundle.get_auxiliary("nope")
