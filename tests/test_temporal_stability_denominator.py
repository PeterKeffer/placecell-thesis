"""The Psi denominator decides whether silence pays."""

from __future__ import annotations

import torch

from placecell_research.config.schema import ObjectiveConfig
from placecell_research.objectives.temporal_stability import TemporalStabilityObjective


class _Bundle:
    def __init__(self, codes: torch.Tensor) -> None:
        self._codes = codes
        self.masks: dict[str, torch.Tensor] = {}
        self.inputs: dict[str, torch.Tensor] = {}

    def get_representation(self, name: str) -> torch.Tensor:
        return self._codes


def _scale_gradient(*, detach: bool) -> tuple[float, float]:
    """Loss and d(loss)/d(scale) for a code multiplied by a single learnable scale."""
    torch.manual_seed(0)
    codes = torch.randn(4, 64, 32)
    scale = torch.tensor(1.0, requires_grad=True)
    objective = TemporalStabilityObjective(
        name="stability",
        config=ObjectiveConfig(
            type="temporal_stability",
            targets=["encoder.place_codes"],
            weight=1.0,
            horizon=4,
            decorrelation_weight=5.0,
            detach_variance_denominator=detach,
        ),
    )
    valid = torch.ones(4, 64, dtype=torch.bool)
    result = objective.compute(_Bundle(codes * scale), {"valid_steps": valid})
    result.loss.backward()
    assert scale.grad is not None
    return float(result.loss), float(scale.grad)


def test_detached_denominator_rewards_shrinking_the_code() -> None:
    """The historical default: reducing the code scale reduces the loss."""
    _, gradient = _scale_gradient(detach=True)
    assert gradient > 1.0


def test_live_denominator_makes_the_ratio_scale_invariant() -> None:
    """The Wyss form: shrinking the code buys essentially nothing."""
    _, gradient = _scale_gradient(detach=False)
    assert abs(gradient) < 1e-3


def test_both_forms_agree_on_the_loss_value() -> None:
    """Only the gradient differs; the reported Psi is the same number."""
    detached_loss, _ = _scale_gradient(detach=True)
    live_loss, _ = _scale_gradient(detach=False)
    assert detached_loss == live_loss


def test_default_preserves_the_historical_behaviour() -> None:
    """Existing runs keep their meaning unless a config opts out explicitly."""
    config = ObjectiveConfig(type="temporal_stability", targets=["encoder.place_codes"])
    assert config.detach_variance_denominator is True
