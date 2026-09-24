from __future__ import annotations

import torch

from placecell_research.config.schema import ObjectiveConfig
from placecell_research.objectives.temporal_stability import TemporalStabilityObjective
from placecell_research.spatial_model.types import ModuleOutputs, RepresentationBundle

_VARIANCE_EPS = 1.0e-6


def _objective(lag: int, decorrelation_weight: float = 0.0) -> TemporalStabilityObjective:
    return TemporalStabilityObjective(
        name="stability",
        config=ObjectiveConfig(
            type="temporal_stability",
            targets=["encoder.place_codes"],
            horizon=lag,
            decorrelation_weight=decorrelation_weight,
        ),
    )


def _bundle(target: torch.Tensor, valid_mask: torch.Tensor) -> RepresentationBundle:
    return RepresentationBundle(
        modules={"encoder": ModuleOutputs(place_codes=target)},
        masks={"valid_steps": valid_mask},
    )


def test_constant_signal_is_dead_not_rewarded() -> None:
    target = torch.full((2, 32, 4), 3.5)
    valid_mask = torch.ones(2, 32, dtype=torch.bool)

    result = _objective(lag=16).compute(_bundle(target, valid_mask), {})

    assert torch.allclose(result.loss, torch.zeros(()))
    assert torch.allclose(result.metrics["mean_variance"], torch.zeros(()))
    assert torch.allclose(result.metrics["dead_unit_fraction"], torch.ones(()))
    assert torch.allclose(result.metrics["mean_code_rms"], torch.full((), 3.5))


def test_fast_noise_scores_higher_than_slow_signal() -> None:
    generator = torch.Generator().manual_seed(0)
    fast = torch.randn(2, 64, 8, generator=generator)
    slow = torch.linspace(0.0, 1.0, 64).reshape(1, 64, 1).expand(2, 64, 8).contiguous()
    valid_mask = torch.ones(2, 64, dtype=torch.bool)
    objective = _objective(lag=4)

    fast_loss = objective.compute(_bundle(fast, valid_mask), {}).loss
    slow_loss = objective.compute(_bundle(slow, valid_mask), {}).loss

    assert fast_loss > 1.5
    assert slow_loss < 0.1
    assert fast_loss > 10.0 * slow_loss


def test_variance_normalization_makes_loss_scale_invariant() -> None:
    generator = torch.Generator().manual_seed(1)
    target = torch.randn(2, 48, 6, generator=generator)
    valid_mask = torch.ones(2, 48, dtype=torch.bool)
    objective = _objective(lag=8)

    base_loss = objective.compute(_bundle(target, valid_mask), {}).loss
    scaled_loss = objective.compute(_bundle(target * 10.0, valid_mask), {}).loss

    assert torch.allclose(base_loss, scaled_loss, rtol=1e-3)


def test_episode_latch_units_are_excluded_not_rewarded() -> None:
    generator = torch.Generator().manual_seed(2)
    num_episodes, num_steps = 4, 64
    noise = torch.randn(num_episodes, num_steps, 4, generator=generator)
    latch = (
        torch.arange(1.0, num_episodes + 1.0)
        .reshape(num_episodes, 1, 1)
        .expand(num_episodes, num_steps, 4)
    )
    combined = torch.cat([latch, noise], dim=-1)
    valid_mask = torch.ones(num_episodes, num_steps, dtype=torch.bool)
    objective = _objective(lag=4)

    combined_result = objective.compute(_bundle(combined, valid_mask), {})
    noise_only_loss = objective.compute(_bundle(noise, valid_mask), {}).loss

    assert combined_result.loss > 1.5
    assert torch.allclose(combined_result.loss, noise_only_loss, rtol=1e-4)
    assert torch.allclose(combined_result.metrics["dead_unit_fraction"], torch.full((), 0.5))


def test_dead_units_are_excluded_and_counted() -> None:
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(2, 64, 3, generator=generator)
    dead = torch.full((2, 64, 1), 0.25)
    combined = torch.cat([noise, dead], dim=-1)
    valid_mask = torch.ones(2, 64, dtype=torch.bool)
    objective = _objective(lag=4)

    combined_result = objective.compute(_bundle(combined, valid_mask), {})
    noise_only_loss = objective.compute(_bundle(noise, valid_mask), {}).loss

    assert combined_result.metrics["dead_unit_fraction"] > 0.0
    assert torch.allclose(combined_result.loss, noise_only_loss, rtol=1e-4)


def test_pairs_spanning_internal_invalid_steps_are_excluded() -> None:
    target = torch.tensor([[[0.0], [1.0], [99.0], [4.0], [9.0], [25.0]]])
    valid_mask = torch.tensor([[True, True, False, True, True, True]])
    lag = 2

    result = _objective(lag=lag).compute(_bundle(target, valid_mask), {})

    valid_values = target[valid_mask][:, 0]
    variance = valid_values.var(unbiased=False)
    expected = torch.tensor((25.0 - 4.0) ** 2) / (variance + _VARIANCE_EPS)

    assert torch.allclose(result.loss, expected)


def test_masked_lag_pairs_match_manual_computation() -> None:
    target = torch.tensor(
        [
            [[0.0], [1.0], [4.0], [9.0], [99.0]],
            [[2.0], [3.0], [5.0], [-99.0], [-99.0]],
        ]
    )
    valid_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    lag = 2

    result = _objective(lag=lag).compute(_bundle(target, valid_mask), {})

    episode_values = [target[0, valid_mask[0], 0], target[1, valid_mask[1], 0]]
    centered_energy = sum(((values - values.mean()) ** 2).sum() for values in episode_values)
    total_count = valid_mask.sum()
    variance = centered_energy / total_count
    squared_changes = torch.tensor([(4.0 - 0.0) ** 2, (9.0 - 1.0) ** 2, (5.0 - 2.0) ** 2])
    expected = squared_changes.mean() / (variance + _VARIANCE_EPS)

    assert torch.allclose(result.loss, expected)
    assert torch.allclose(result.metrics["temporal_stability"], expected)
    assert torch.allclose(result.metrics["mean_variance"], variance)


def test_decorrelation_term_penalizes_duplicated_units() -> None:
    generator = torch.Generator().manual_seed(4)
    base = torch.randn(2, 64, 1, generator=generator)
    duplicated = torch.cat([base, base], dim=-1)
    valid_mask = torch.ones(2, 64, dtype=torch.bool)

    plain_result = _objective(lag=4).compute(_bundle(duplicated, valid_mask), {})
    decorrelated_result = _objective(lag=4, decorrelation_weight=0.25).compute(
        _bundle(duplicated, valid_mask), {}
    )

    assert torch.allclose(decorrelated_result.metrics["decorrelation"], torch.ones(()), rtol=1e-3)
    assert torch.allclose(
        decorrelated_result.loss - plain_result.loss,
        0.25 * decorrelated_result.metrics["decorrelation"],
        rtol=1e-4,
    )


def test_decorrelation_is_not_diluted_by_dead_units() -> None:
    generator = torch.Generator().manual_seed(4)
    base = torch.randn(2, 64, 1, generator=generator)
    duplicated = torch.cat([base, base], dim=-1)
    padded = torch.cat([duplicated, torch.zeros(2, 64, 510)], dim=-1)
    valid_mask = torch.ones(2, 64, dtype=torch.bool)
    objective = _objective(lag=4, decorrelation_weight=0.25)

    duplicated_result = objective.compute(_bundle(duplicated, valid_mask), {})
    padded_result = objective.compute(_bundle(padded, valid_mask), {})

    assert torch.allclose(padded_result.metrics["dead_unit_fraction"], torch.full((), 510 / 512))
    assert torch.allclose(
        padded_result.metrics["decorrelation"],
        duplicated_result.metrics["decorrelation"],
        rtol=1e-4,
    )
    assert torch.allclose(padded_result.metrics["decorrelation"], torch.ones(()), rtol=1e-3)
