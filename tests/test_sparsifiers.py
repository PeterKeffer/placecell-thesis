"""Native alpha-entmax sparsifier: correctness, sparsity, gradients."""

from __future__ import annotations

import pytest
import torch

from placecell_research.spatial_model.components.sparsifiers import (
    EntmaxSparsifier,
    entmax_bisect,
    sparsemax,
)


def test_entmax_bisect_sums_to_one():
    z = torch.randn(4, 16)
    p = entmax_bisect(z, alpha=1.5)
    assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-4)


def test_entmax_bisect_alpha2_matches_sparsemax():
    z = torch.randn(8, 32)
    assert torch.allclose(entmax_bisect(z, alpha=2.0), sparsemax(z, dim=-1), atol=1e-4)


def test_entmax_bisect_matches_reference_entmax15():
    entmax_pkg = pytest.importorskip("entmax")
    z = torch.randn(8, 64)
    ref = entmax_pkg.entmax_bisect(z, alpha=1.5, dim=-1)
    got = entmax_bisect(z, alpha=1.5)
    assert torch.allclose(got, ref, atol=1e-4)


def test_entmax_bisect_is_sparse_for_alpha_1p5():
    z = torch.linspace(-3.0, 3.0, 32).unsqueeze(0)
    p = entmax_bisect(z, alpha=1.5)
    assert bool((p == 0).any())


def test_entmax_bisect_gradient_flows():
    z = torch.randn(4, 16, requires_grad=True)
    entmax_bisect(z, alpha=1.5).sum().backward()
    assert z.grad is not None
    assert bool(torch.isfinite(z.grad).all())


def test_entmax_bisect_backward_matches_reference():
    entmax_pkg = pytest.importorskip("entmax")
    weights = torch.randn(4, 32, dtype=torch.float64)
    z_native = torch.randn(4, 32, dtype=torch.float64, requires_grad=True)
    z_ref = z_native.detach().clone().requires_grad_(True)
    (entmax_bisect(z_native, alpha=1.5) * weights).sum().backward()
    (entmax_pkg.entmax_bisect(z_ref, alpha=1.5, dim=-1) * weights).sum().backward()
    assert torch.allclose(z_native.grad, z_ref.grad, atol=1e-5)


def test_entmax_sparsifier_is_not_dense_softmax():
    z = torch.linspace(-4.0, 4.0, 64).unsqueeze(0)
    out = EntmaxSparsifier(alpha=1.5)(z)
    assert bool((out == 0).any())
    assert not torch.allclose(out, torch.softmax(z, dim=-1), atol=1e-3)
