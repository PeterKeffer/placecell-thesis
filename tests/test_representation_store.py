"""A stored representation set must hand back exactly the arrays that were written."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import zarr

from placecell_research.evaluation.representation_store import (
    RepresentationRequest,
    read_representation_set,
    resolve_representations,
    stored_split_names,
    write_representation_batches,
    write_representation_manifest,
)

EPISODES, STEPS, UNITS = 3, 64, 16


def _request() -> RepresentationRequest:
    return RepresentationRequest(
        place_model_artifact_id="model-a",
        dataset_artifact_id="data-a",
        dataset_artifact_type="encoded_dataset",
        split_artifact_id="split-a",
        checkpoint_selection="last",
        device="cpu",
        batch_size=2,
        allow_tf32=False,
        torch_version="test",
        episode_ids=[2, 4, 8],
    )


def _write_request(directory: Path, request: RepresentationRequest) -> None:
    manifest = asdict(request)
    manifest["episode_ids"] = {"test": request.episode_ids}
    write_representation_manifest(directory, manifest)


def _write(
    directory: Path,
    split_name: str,
    representations: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
) -> None:
    write_representation_batches(
        directory,
        split_name=split_name,
        episode_count=EPISODES,
        batches=[(representations, metadata)],
    )


def test_streamed_batches_match_the_original_arrays(tmp_path):
    representations, metadata = _arrays(9)

    def batches():
        for start in (0, 2):
            yield (
                {name: value[start : start + 2] for name, value in representations.items()},
                {name: value[start : start + 2] for name, value in metadata.items()},
            )

    write_representation_batches(tmp_path, split_name="test", episode_count=3, batches=batches())
    actual, actual_metadata = read_representation_set(
        tmp_path,
        split_name="test",
        source_names=list(representations),
    )
    for name, value in representations.items():
        np.testing.assert_array_equal(actual[name], value)
    for name, value in metadata.items():
        np.testing.assert_array_equal(actual_metadata[name], value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("place_model_artifact_id", "other"),
        ("dataset_artifact_id", "other"),
        ("dataset_artifact_type", "raw_dataset"),
        ("split_artifact_id", "other"),
        ("checkpoint_selection", "best_primary"),
        ("device", "cuda"),
        ("batch_size", 1),
        ("allow_tf32", True),
        ("torch_version", "other"),
        ("episode_ids", [1, 2, 3]),
    ],
)
def test_cached_representations_reject_mismatched_requests(tmp_path, field, value):
    request = _request()
    _write_request(tmp_path, request)
    with pytest.raises(ValueError, match="Representation"):
        resolve_representations(
            artifact_directory=tmp_path,
            split_name="test",
            source_names=["encoder.place_codes"],
            request=replace(request, **{field: value}),
            collect=lambda: pytest.fail("An incompatible cache must not silently run inference"),
        )


def test_cached_prefix_preserves_inference_batch_boundaries(tmp_path):
    request = _request()
    _write_request(tmp_path, request)
    representations, metadata = _arrays(4)
    _write(tmp_path, "test", representations, metadata)
    actual, actual_metadata = read_representation_set(
        tmp_path,
        split_name="test",
        source_names=list(representations),
        request=replace(request, episode_ids=[2, 4]),
    )
    for name in representations:
        np.testing.assert_array_equal(actual[name], representations[name][:2])
    np.testing.assert_array_equal(actual_metadata["valid_steps"], metadata["valid_steps"][:2])
    with pytest.raises(ValueError, match="final inference batch"):
        read_representation_set(
            tmp_path,
            split_name="test",
            source_names=list(representations),
            request=replace(request, episode_ids=[2]),
        )


@pytest.mark.parametrize("required_keys", [[], ["latent"]])
def test_resolver_reads_only_standard_and_requested_metadata(tmp_path, monkeypatch, required_keys):
    request = _request()
    _write_request(tmp_path, request)
    representations, metadata = _arrays(4)
    metadata["latent"] = np.ones((EPISODES, STEPS, 8), dtype=np.float32)
    metadata["rgb"] = np.ones((EPISODES, STEPS, 8, 8, 3), dtype=np.uint8)
    _write(tmp_path, "test", representations, metadata)
    reads = []
    original_getitem = zarr.Array.__getitem__

    def tracked_getitem(array, selection):
        reads.append(array.path)
        return original_getitem(array, selection)

    monkeypatch.setattr(zarr.Array, "__getitem__", tracked_getitem)
    actual, actual_metadata = resolve_representations(
        artifact_directory=tmp_path,
        split_name="test",
        source_names=list(representations),
        request=request,
        require_metadata_keys=required_keys,
        collect=lambda: pytest.fail("A valid cache must not run inference"),
    )
    expected_keys = (set(metadata) - {"rgb", "latent"}) | set(required_keys)
    assert set(actual_metadata) == expected_keys
    for name in expected_keys:
        np.testing.assert_array_equal(actual_metadata[name], metadata[name])
    for name in representations:
        np.testing.assert_array_equal(actual[name], representations[name])
    assert not any(path.endswith("/rgb") for path in reads)
    assert any(path.endswith("/latent") for path in reads) == bool(required_keys)


def test_streamed_write_rejects_incomplete_split(tmp_path):
    representations, metadata = _arrays(0)
    with pytest.raises(ValueError, match="Expected 4.*received 3"):
        write_representation_batches(
            tmp_path, split_name="test", episode_count=4, batches=[(representations, metadata)]
        )


def _arrays(seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    representations = {
        "encoder.place_codes": rng.standard_normal((EPISODES, STEPS, UNITS), dtype=np.float32),
        "predictor.place_codes": rng.standard_normal((EPISODES, STEPS, UNITS), dtype=np.float32),
    }
    metadata = {
        "valid_steps": rng.integers(0, 2, (EPISODES, STEPS)).astype(bool),
        "position_xy": rng.standard_normal((EPISODES, STEPS, 2), dtype=np.float32),
        "heading": rng.standard_normal((EPISODES, STEPS), dtype=np.float32),
    }
    return representations, metadata


def test_round_trip_is_byte_identical(tmp_path: Path) -> None:
    representations, metadata = _arrays(0)
    _write(tmp_path, "test", representations, metadata)
    read_representations, read_metadata = read_representation_set(
        tmp_path, split_name="test", source_names=sorted(representations)
    )
    for name, array in representations.items():
        assert np.array_equal(read_representations[name], array)
        assert read_representations[name].dtype == array.dtype
    for name, array in metadata.items():
        assert np.array_equal(read_metadata[name], array)
        assert read_metadata[name].dtype == array.dtype


def test_splits_stay_separate(tmp_path: Path) -> None:
    validation, validation_metadata = _arrays(1)
    test, test_metadata = _arrays(2)
    _write(tmp_path, "validation", validation, validation_metadata)
    _write(tmp_path, "test", test, test_metadata)
    assert stored_split_names(tmp_path) == ["test", "validation"]
    read_test, _ = read_representation_set(
        tmp_path, split_name="test", source_names=["encoder.place_codes"]
    )
    assert np.array_equal(read_test["encoder.place_codes"], test["encoder.place_codes"])


def test_missing_source_raises_rather_than_falling_back(tmp_path: Path) -> None:
    representations, metadata = _arrays(0)
    _write(tmp_path, "test", representations, metadata)
    with pytest.raises(KeyError, match="no source"):
        read_representation_set(tmp_path, split_name="test", source_names=["encoder.hidden_state"])


def test_missing_split_raises(tmp_path: Path) -> None:
    representations, metadata = _arrays(0)
    _write(tmp_path, "test", representations, metadata)
    with pytest.raises(KeyError, match="not in this representation set"):
        read_representation_set(
            tmp_path, split_name="validation", source_names=["encoder.place_codes"]
        )


def test_source_union_covers_evaluation_and_analysis() -> None:
    """The stage must collect every source a later stage will ask for, not just evaluation's."""
    import placecell_research.config.schema  # noqa: F401
    from placecell_research.config import load_experiment_config
    from placecell_research.stages.collect_representations import resolve_source_names

    config = load_experiment_config("configs/experiment/wallgap.yaml")
    names = resolve_source_names(config)
    assert names[: len(config.evaluation.sources)] == list(config.evaluation.sources)
    assert len(names) == len(set(names)), "duplicates would collect the same pass twice"
    for target in config.analysis.targets.values():
        if getattr(target, "enabled", True) and getattr(target, "source", None):
            assert target.source in names


@pytest.mark.parametrize("observation_key", ["rgb", "latent"])
def test_resolver_allows_an_unavailable_optional_observation(tmp_path, observation_key):
    request = _request()
    _write_request(tmp_path, request)
    representations, metadata = _arrays(4)
    metadata[observation_key] = np.ones((EPISODES, STEPS, 8), dtype=np.float32)
    _write(tmp_path, "test", representations, metadata)
    _, actual = resolve_representations(
        artifact_directory=tmp_path,
        split_name="test",
        source_names=list(representations),
        request=request,
        optional_metadata_keys=["rgb", "latent"],
        collect=lambda: pytest.fail("The cached path must not run inference"),
    )
    assert set(actual) == set(metadata)
    np.testing.assert_array_equal(actual[observation_key], metadata[observation_key])
    missing_key = "rgb" if observation_key == "latent" else "latent"
    with pytest.raises(KeyError, match=missing_key):
        read_representation_set(
            tmp_path,
            split_name="test",
            source_names=list(representations),
            require_metadata_keys=[missing_key],
        )
