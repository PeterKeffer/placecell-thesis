from collections import Counter

import numpy as np
import pytest
import zarr

from placecell_research.datasets.batch_iterator import _EpisodeChunkReader


class CountingStore(zarr.storage.MemoryStore):
    def __init__(self):
        super().__init__()
        self.reads = Counter()

    def __getitem__(self, key):
        if not key.startswith("."):
            self.reads[key] += 1
        return super().__getitem__(key)


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.bool_])
@pytest.mark.parametrize("order", ["C", "F"])
def test_streaming_batches_reuse_chunks_without_changing_values(dtype, order):
    values = np.arange(17 * 3).reshape(17, 3).astype(dtype)
    store = CountingStore()
    array = zarr.array(values, chunks=(8, 3), store=store, order=order)
    store.reads.clear()
    reader = _EpisodeChunkReader(array)
    for selection in [[6, 0, 2, 2], [7, 3], [9, 15], [8], [16, -1]]:
        result = reader.read(selection)
        np.testing.assert_array_equal(result, values[selection])
        assert result.dtype == values.dtype
        assert result.flags.c_contiguous if order == "C" else result.flags.f_contiguous
    assert sum(store.reads.values()) == 3


def test_streaming_cache_does_not_alias_returned_batches():
    values = np.arange(24).reshape(8, 3)
    reader = _EpisodeChunkReader(zarr.array(values, chunks=(4, 3)))
    first = reader.read([1, 2])
    first[:] = -1
    np.testing.assert_array_equal(reader.read([2, 1]), values[[2, 1]])
    reader.read([5])
    assert reader._cached_chunk[1].nbytes == values[:4].nbytes
    np.testing.assert_array_equal(reader.read([1]), values[[1]])


@pytest.mark.parametrize("chunks", [(1, 8), (4, 8)])
def test_single_episode_and_oversized_chunks_use_direct_reads(monkeypatch, chunks):
    import placecell_research.datasets.batch_iterator as batch_iterator

    monkeypatch.setattr(batch_iterator, "_TARGET_CHUNK_RAW_BYTES", 64)
    values = np.arange(64).reshape(8, 8)
    reader = _EpisodeChunkReader(zarr.array(values, chunks=chunks))
    np.testing.assert_array_equal(reader.read([7, 2, 2, -1]), values[[7, 2, 2, -1]])
    assert reader._cached_chunk is None


@pytest.mark.parametrize("selection", [[8], [-9]])
def test_streaming_reader_preserves_out_of_bounds_errors(selection):
    reader = _EpisodeChunkReader(zarr.array(np.arange(24).reshape(8, 3), chunks=(4, 3)))
    with pytest.raises(IndexError):
        reader.read(selection)


@pytest.mark.parametrize("selection", [[], [1.5], ["1"], [[1, 2]], [True, False] * 4])
def test_streaming_reader_preserves_native_non_integer_indexing(selection):
    array = zarr.array(np.arange(24).reshape(8, 3), chunks=(4, 3))
    reader = _EpisodeChunkReader(array)
    try:
        expected = array[selection]
    except (IndexError, TypeError) as error:
        with pytest.raises(type(error)):
            reader.read(selection)
    else:
        np.testing.assert_array_equal(reader.read(selection), expected)
