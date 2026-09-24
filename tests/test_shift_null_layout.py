"""Null layout changes preserve the original bin reductions and bounded buffer lifetime."""

import weakref

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from placecell_research.analysis import shift_nulls
from placecell_research.numerics.rate_map_kernels import (
    compute_spatial_bin_assignments,
    skaggs_spatial_information,
)


def _reference_null(activity, positions, valid, selected, sigma):
    episodes = [
        episode[mask][:, selected]
        for episode, mask in zip(activity, valid, strict=False)
        if mask.any()
    ]
    lengths = np.array([len(episode) for episode in episodes])
    bins, *_ = compute_spatial_bin_assignments(
        positions[valid], num_bins_x=4, num_bins_y=3, bounds=((0, 1), (0, 1))
    )
    occupancy = np.bincount(bins, minlength=12).reshape(3, 4).astype(np.float32)
    if sigma:
        occupancy = gaussian_filter(occupancy, sigma=sigma)
    safe = np.where(occupancy >= 1e-6, occupancy, np.nan)
    order = np.argsort(bins, kind="stable")
    occupied, starts = np.unique(bins[order], return_index=True)
    rng = np.random.default_rng(7)
    scores = np.full((7, activity.shape[-1]), np.nan, dtype=np.float32)
    for shuffle in range(7):
        offsets = shift_nulls.draw_circular_shift_offsets(lengths, rng, 0.05)
        rolled = np.concatenate(
            [
                np.roll(episode, offset, axis=0)
                for episode, offset in zip(episodes, offsets, strict=False)
            ]
        )[order]
        sums = np.zeros((12, selected.sum()), dtype=np.float32)
        sums[occupied] = np.add.reduceat(rolled, starts, axis=0)
        maps = sums.T.reshape(-1, 3, 4)
        if sigma:
            maps = gaussian_filter(maps, sigma=(0, sigma, sigma))
        scores[shuffle, selected] = skaggs_spatial_information(maps / safe[None], occupancy)
    return scores


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("sigma", [0, 0.3])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("budget", [240, 1 << 25])
@pytest.mark.parametrize("density", [0.03, 1.0])
def test_null_matches_original_reduction(monkeypatch, workers, sigma, signed, budget, density):
    rng = np.random.default_rng(41)
    activity = rng.normal(size=(5, 257, 9)).astype(np.float32)
    if not signed:
        activity = np.maximum(activity, 0)
    activity[rng.random(activity.shape) >= density] = 0
    activity[..., 1] = 0
    positions = rng.uniform(size=(5, 257, 2)).astype(np.float32)
    valid = rng.random((5, 257)) > 0.1
    valid[0] = False
    valid[1] = False
    valid[1, 0] = True
    selected = np.arange(9) % 3 != 0
    expected = _reference_null(activity, positions, valid, selected, sigma)
    monkeypatch.setenv("PLACECELL_ANALYSIS_WORKERS", str(workers))
    monkeypatch.setattr(shift_nulls, "_CIRCULAR_SHIFT_GATHER_ELEMENT_BUDGET", budget)
    actual = shift_nulls.circular_shift_spatial_information_null(
        np.asfortranarray(activity),
        positions,
        valid,
        num_bins_x=4,
        num_bins_y=3,
        smoothing_sigma=sigma,
        min_occupancy=1e-6,
        bounds=((0, 1), (0, 1)),
        num_shuffles=7,
        rng_seed=7,
        unit_mask=selected,
    )
    np.testing.assert_array_equal(actual, expected)


def test_gather_budget_and_lifetime_include_both_buffers(monkeypatch):
    monkeypatch.setenv("PLACECELL_ANALYSIS_WORKERS", "1")
    monkeypatch.setattr(shift_nulls, "_CIRCULAR_SHIFT_GATHER_ELEMENT_BUDGET", 64)
    rng = np.random.default_rng(5)
    activity = rng.random((2, 12, 4), dtype=np.float32)
    positions = np.tile(np.array([[0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5]]), (6, 1))
    buffers = []
    sizes = []
    contiguous = np.ascontiguousarray

    def record_buffers(values):
        assert all(reference() is None for reference in buffers)
        result = contiguous(values)
        sizes.append(values.size + result.size)
        buffers.extend((weakref.ref(values.base), weakref.ref(result)))
        return result

    monkeypatch.setattr(shift_nulls.np, "ascontiguousarray", record_buffers)
    shift_nulls.circular_shift_spatial_information_null(
        activity,
        positions.reshape(2, 12, 2),
        None,
        num_bins_x=4,
        num_bins_y=1,
        smoothing_sigma=0.3,
        min_occupancy=1e-6,
        bounds=((0, 1), (0, 1)),
        num_shuffles=2,
    )
    assert len(sizes) == 8
    assert max(sizes) <= 64
    assert all(reference() is None for reference in buffers)


def test_working_set_log_counts_both_buffers(monkeypatch, capsys):
    monkeypatch.setenv("PLACECELL_ANALYSIS_WORKERS", "2")
    shift_nulls._log_circular_shift_working_set(
        total_valid_steps=32768,
        num_selected_units=512,
        activity_bytes=64 << 20,
        max_block_rows=32768,
        num_blocks=1,
        num_shuffles=2,
        gather_bytes_per_row=2 * 512 * 4,
    )
    message = capsys.readouterr().err
    assert "worker_gib=0.12" in message
    assert "peak_gib=0.31" in message


def test_sparse_gathers_respect_budget_and_release_dense_output(monkeypatch):
    monkeypatch.setenv("PLACECELL_ANALYSIS_WORKERS", "1")
    monkeypatch.setattr(shift_nulls, "_CIRCULAR_SHIFT_GATHER_ELEMENT_BUDGET", 1280)
    activity = np.zeros((96, 32), dtype=np.float32)
    activity[np.arange(96), np.arange(96) % 32] = 1
    activity = activity.reshape(2, 48, 32)
    centers = np.array([[(x + 0.5) / 4, (y + 0.5) / 3] for y in range(3) for x in range(4)])
    positions = np.tile(centers, (8, 1)).reshape(2, 48, 2)
    valid = np.ones((2, 48), dtype=bool)
    selected = np.ones(32, dtype=bool)
    expected = _reference_null(activity, positions, valid, selected, 0.3)
    outputs = []
    toarray = shift_nulls.csr_matrix.toarray

    def checked_toarray(matrix, *args, **kwargs):
        assert all(reference() is None for reference in outputs)
        row_bytes = matrix.shape[1] * (4 + 2 * (4 + matrix.indices.itemsize))
        assert matrix.shape[0] * row_bytes <= 1280 * 4
        result = toarray(matrix, *args, **kwargs)
        outputs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(shift_nulls.csr_matrix, "toarray", checked_toarray)
    actual = shift_nulls.circular_shift_spatial_information_null(
        activity,
        positions,
        valid,
        num_bins_x=4,
        num_bins_y=3,
        smoothing_sigma=0.3,
        min_occupancy=1e-6,
        bounds=((0, 1), (0, 1)),
        num_shuffles=7,
        rng_seed=7,
        unit_mask=selected,
    )
    np.testing.assert_array_equal(actual, expected)
    assert len(outputs) == 12 * 7
    assert all(reference() is None for reference in outputs)
