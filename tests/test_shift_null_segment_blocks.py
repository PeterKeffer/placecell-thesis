from __future__ import annotations

import numpy as np

from placecell_research.analysis.shift_nulls import _segment_blocks


def _block_rows(segment_bounds: np.ndarray, blocks: list[tuple[int, int]]) -> list[int]:
    return [
        int(segment_bounds[last_segment] - segment_bounds[first_segment])
        for first_segment, last_segment in blocks
    ]


def test_segment_blocks_close_before_the_overflowing_segment() -> None:
    segment_bounds = np.array([0, 60, 120])
    blocks = _segment_blocks(segment_bounds, 100)
    assert blocks == [(0, 1), (1, 2)]
    assert max(_block_rows(segment_bounds, blocks)) <= 100


def test_segment_blocks_fill_up_to_the_budget() -> None:
    segment_bounds = np.array([0, 50, 100, 150])
    blocks = _segment_blocks(segment_bounds, 100)
    assert blocks == [(0, 2), (2, 3)]
    assert _block_rows(segment_bounds, blocks) == [100, 50]


def test_segment_blocks_keep_an_oversized_segment_alone() -> None:
    segment_bounds = np.array([0, 150, 190])
    blocks = _segment_blocks(segment_bounds, 100)
    assert blocks == [(0, 1), (1, 2)]
    assert _block_rows(segment_bounds, blocks) == [150, 40]


def test_segment_blocks_cover_every_segment_once() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        segment_sizes = rng.integers(1, 40, size=int(rng.integers(1, 12)))
        segment_bounds = np.concatenate(([0], np.cumsum(segment_sizes)))
        budget = int(rng.integers(1, 120))
        blocks = _segment_blocks(segment_bounds, budget)
        assert [first for first, _ in blocks] == [0] + [last for _, last in blocks[:-1]]
        assert blocks[-1][1] == segment_sizes.size
        oversized = [
            rows for rows in _block_rows(segment_bounds, blocks) if rows > budget
        ]
        assert all(
            end - start == 1
            for rows, (start, end) in zip(_block_rows(segment_bounds, blocks), blocks, strict=False)
            if rows > budget
        ), oversized
