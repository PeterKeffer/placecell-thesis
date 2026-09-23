from __future__ import annotations

import torch

from placecell_research.config.schema import (
    RegularizationConfig,
    RegularizationSiteConfig,
)
from placecell_research.spatial_model.regularization import DropoutAndNoise, SequenceRegularizer


def test_sequence_regularizer_reports_active_predictor_sites() -> None:
    inactive = SequenceRegularizer(RegularizationConfig(sites=[]))
    assert inactive.has_active_site("predictor.") is False

    active = SequenceRegularizer(
        RegularizationConfig(
            sites=[RegularizationSiteConfig(target="predictor.hidden", dropout=0.1)]
        )
    )
    assert active.has_active_site("predictor.") is True
    assert active.has_active_site("encoder.") is False


def test_dropout_and_noise_are_independent_when_combined() -> None:
    regularizer = DropoutAndNoise(dropout_probability=1.0, noise_scale=0.25).train()
    inputs = torch.ones(4, 5, 6)

    torch.manual_seed(17)
    expected_noise = 0.25 * torch.randn_like(inputs)
    torch.manual_seed(17)

    outputs = regularizer(inputs)

    torch.testing.assert_close(outputs, expected_noise)


def test_combined_dropout_does_not_gate_or_amplify_noise() -> None:
    regularizer = DropoutAndNoise(dropout_probability=0.5, noise_scale=0.25).train()
    inputs = torch.ones(4, 5, 6)

    torch.manual_seed(23)
    keep_mask = torch.empty_like(inputs).bernoulli_(0.5) / 0.5
    independent_noise = 0.25 * torch.randn_like(inputs)
    expected = inputs * keep_mask + independent_noise
    torch.manual_seed(23)

    outputs = regularizer(inputs)

    torch.testing.assert_close(outputs, expected)


def test_predictor_regularizer_defaults_unknown_active_sites_to_stepwise() -> None:
    sequence_safe = SequenceRegularizer(
        RegularizationConfig(
            sites=[
                RegularizationSiteConfig(
                    target="predictor.pre_sparsifier",
                    dropout=0.1,
                )
            ]
        )
    )
    recurrent_interleaved = SequenceRegularizer(
        RegularizationConfig(
            sites=[
                RegularizationSiteConfig(
                    target="predictor.recurrent_state",
                    dropout=0.1,
                )
            ]
        )
    )

    assert sequence_safe.predictor_requires_stepwise() is False
    assert recurrent_interleaved.predictor_requires_stepwise() is True
