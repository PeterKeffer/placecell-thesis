"""VICReg covariance term must use canonical per-dimension (1/d) normalization."""

from __future__ import annotations

import torch

from placecell_research.config.schema import ObjectiveConfig
from placecell_research.objectives.vicreg import VICRegObjective
from placecell_research.spatial_model.types import ModuleOutputs, RepresentationBundle


def test_vicreg_covariance_uses_canonical_per_dimension_normalization():
    torch.manual_seed(0)
    batch_size, time_steps, code_dim = 4, 5, 6
    codes = torch.randn(batch_size, time_steps, code_dim)
    codes[..., 1] = codes[..., 0]

    bundle = RepresentationBundle(
        modules={"encoder": ModuleOutputs(place_codes=codes)},
        masks={"valid_steps": torch.ones(batch_size, time_steps, dtype=torch.bool)},
    )
    config = ObjectiveConfig(
        type="vicreg",
        targets=["encoder.place_codes"],
        variance_weight=0.0,
        covariance_weight=1.0,
        minimum_std=0.0,
    )

    loss = VICRegObjective(name="vicreg_test", config=config).compute(bundle, {}).loss

    flat = codes.reshape(-1, code_dim)
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    canonical = off_diagonal.pow(2).sum() / code_dim
    legacy = off_diagonal.pow(2).mean()

    assert torch.allclose(loss, canonical, atol=1e-6)
    assert not torch.allclose(loss, legacy, atol=1e-6)


def test_vicreg_covariance_normalization_is_selectable():
    """covariance_normalization switches between canonical (1/d) and legacy (1/d^2)."""
    torch.manual_seed(0)
    batch_size, time_steps, code_dim = 4, 5, 6
    codes = torch.randn(batch_size, time_steps, code_dim)
    codes[..., 1] = codes[..., 0]

    bundle = RepresentationBundle(
        modules={"encoder": ModuleOutputs(place_codes=codes)},
        masks={"valid_steps": torch.ones(batch_size, time_steps, dtype=torch.bool)},
    )

    def covariance_loss_for(mode: str) -> torch.Tensor:
        config = ObjectiveConfig(
            type="vicreg",
            targets=["encoder.place_codes"],
            variance_weight=0.0,
            covariance_weight=1.0,
            minimum_std=0.0,
            covariance_normalization=mode,
        )
        return VICRegObjective(name="vicreg_test", config=config).compute(bundle, {}).loss

    flat = codes.reshape(-1, code_dim)
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    canonical = off_diagonal.pow(2).sum() / code_dim
    legacy = off_diagonal.pow(2).mean()

    assert torch.allclose(covariance_loss_for("feature_count"), canonical, atol=1e-6)
    assert torch.allclose(covariance_loss_for("feature_count_squared"), legacy, atol=1e-6)
    assert torch.allclose(
        covariance_loss_for("feature_count"),
        covariance_loss_for("feature_count_squared") * code_dim,
        atol=1e-6,
    )


def test_vicreg_covariance_normalization_defaults_to_canonical():
    """Default must preserve current (canonical 1/d) behavior so existing runs are unchanged."""
    assert ObjectiveConfig(type="vicreg").covariance_normalization == "feature_count"


def test_vicreg_reports_variance_covariance_and_dead_dim_metrics():
    codes = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 1.0]],
            [[1.0, 3.0, 0.0, 0.0], [1.0, 4.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    valid_steps = torch.ones(2, 2, dtype=torch.bool)
    bundle = RepresentationBundle(
        modules={"encoder": ModuleOutputs(place_codes=codes)},
        masks={"valid_steps": valid_steps},
    )
    config = ObjectiveConfig(
        type="vicreg",
        targets=["encoder.place_codes"],
        variance_weight=5.0,
        covariance_weight=2.0,
        minimum_std=0.5,
    )

    result = VICRegObjective(name="vicreg_test", config=config).compute(bundle, {})

    flat = codes.reshape(-1, codes.shape[-1])
    centered = flat - flat.mean(dim=0, keepdim=True)
    std_per_dim = torch.sqrt(centered.var(dim=0) + 1e-4)
    variance_loss = torch.relu(torch.tensor(0.5) - std_per_dim).mean()
    covariance = centered.T @ centered / (centered.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = off_diagonal.pow(2).sum() / centered.shape[1]
    dead_dim_fraction = (std_per_dim <= 1.05e-2).float().mean()

    assert torch.allclose(result.metrics["mean_std"], std_per_dim.mean())
    assert torch.allclose(result.metrics["variance_loss"], variance_loss)
    assert torch.allclose(result.metrics["covariance_loss"], covariance_loss)
    assert torch.allclose(result.metrics["dead_dim_fraction"], dead_dim_fraction)
