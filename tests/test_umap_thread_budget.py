from types import SimpleNamespace

import numpy as np
import pytest
import torch

from placecell_research.analysis import topology_umap
from placecell_research.launch.submit import _threading_exports


@pytest.mark.parametrize("failure_stage", [None, "load", "construct", "fit"])
def test_umap_preserves_pytorch_thread_budget(monkeypatch, failure_stage):
    state = {"threads": 4}
    monkeypatch.setattr(torch, "get_num_threads", lambda: state["threads"])
    monkeypatch.setattr(torch, "set_num_threads", lambda value: state.update(threads=value))

    def change_threads(stage):
        state["threads"] = 16
        if stage == failure_stage:
            raise RuntimeError("UMAP failed")

    class Estimator:
        def __init__(self, **kwargs):
            change_threads("construct")

        def fit_transform(self, features):
            change_threads("fit")
            return features[:, :2]

    def load_estimator():
        change_threads("load")
        return Estimator

    monkeypatch.setattr(topology_umap, "_load_umap_estimator", load_estimator)
    features = np.arange(24, dtype=np.float32).reshape(8, 3)
    if failure_stage is None:
        result = topology_umap._fit_umap_embedding(features, {})
        np.testing.assert_array_equal(result, features[:, :2])
    else:
        with pytest.raises(RuntimeError, match="UMAP failed"):
            topology_umap._fit_umap_embedding(features, {})
    assert state["threads"] == 4


@pytest.mark.parametrize("threads", [1, 4])
def test_slurm_numba_budget_matches_openmp(threads):
    config = SimpleNamespace(
        launcher=SimpleNamespace(
            threading_safety=SimpleNamespace(
                omp_num_threads=threads,
                mkl_num_threads=threads,
                openblas_num_threads=threads,
                torch_num_threads=1,
            )
        )
    )
    assert f"export NUMBA_NUM_THREADS={threads}" in _threading_exports(config.launcher)
