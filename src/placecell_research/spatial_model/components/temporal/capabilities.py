"""What each temporal backend family can do, declared once."""

from __future__ import annotations

from dataclasses import dataclass

STEPWISE_CAPABLE_FLA_VARIANTS = frozenset({"gated_deltanet"})


@dataclass(frozen=True, slots=True)
class TemporalBackendCapabilities:
    """One backend family's declared contract."""

    family: str
    supports_stepwise: bool
    supports_fused: bool
    needs_cuda: bool
    state_layout: str
    fla_variant: str | None = None

    def stepwise_unsupported_message(self) -> str:
        """The one reason both the validator and the builder give for the same rejection."""
        return (
            f"spatial_model.predictor.family={self.family!r} with "
            f"fla_variant={self.fla_variant!r} has no stepwise core: FLATemporal.forward_step "
            "threads a recurrent state only for fla_variant in "
            f"{sorted(STEPWISE_CAPABLE_FLA_VARIANTS)}. Use one of those, or a gru/lstm predictor."
        )


def _recurrent(family: str, *, fused: bool) -> TemporalBackendCapabilities:
    if family.startswith("lstm"):
        layout = "one (hidden, cell) pair per layer"
    elif family == "clockwork":
        layout = "one hidden tensor plus one per-row tick phase per layer"
    else:
        layout = "one hidden tensor per layer"
    return TemporalBackendCapabilities(
        family=family,
        supports_stepwise=True,
        supports_fused=fused,
        needs_cuda=False,
        state_layout=layout,
    )


TEMPORAL_BACKEND_CAPABILITIES: dict[str, TemporalBackendCapabilities] = {
    **{
        family: TemporalBackendCapabilities(
            family=family, supports_stepwise=True, supports_fused=False,
            needs_cuda=False, state_layout="one leaky trace per layer",
        )
        for family in ("wyss", "leaky_hierarchy")
    },
    "mlp": TemporalBackendCapabilities(
        family="mlp",
        supports_stepwise=True,
        supports_fused=False,
        needs_cuda=False,
        state_layout="none",
    ),
    **{family: _recurrent(family, fused=True) for family in ("rnn", "gru", "lstm")},
    **{
        family: _recurrent(family, fused=False)
        for family in (
            "gru_softplus",
            "gru_relu",
            "lstm_softplus",
            "lstm_relu",
            "clockwork",
            "mtrnn",
        )
    },
    "transformer": TemporalBackendCapabilities(
        family="transformer",
        supports_stepwise=True,
        supports_fused=True,
        needs_cuda=False,
        state_layout="a causal token stack, not a recurrent state",
    ),
    "ssm": TemporalBackendCapabilities(
        family="ssm",
        supports_stepwise=True,
        supports_fused=True,
        needs_cuda=False,
        state_layout="one complex S5 state per layer",
    ),
    "ema_ssm": TemporalBackendCapabilities(
        family="ema_ssm",
        supports_stepwise=True,
        supports_fused=True,
        needs_cuda=False,
        state_layout="one hidden tensor per layer",
    ),
    "mamba": TemporalBackendCapabilities(
        family="mamba",
        supports_stepwise=True,
        supports_fused=True,
        needs_cuda=False,
        state_layout="MambaLayerState(conv_state, ssm_state, hidden_state) per layer",
    ),
    "xlstm": TemporalBackendCapabilities(
        family="xlstm",
        supports_stepwise=True,
        supports_fused=True,
        needs_cuda=False,
        state_layout="one mLSTM/sLSTM cell state per layer",
    ),
    "fla": TemporalBackendCapabilities(
        family="fla",
        supports_stepwise=True,
        supports_fused=True,
        needs_cuda=True,
        state_layout=(
            "FLAGatedDeltaNetLayerState(conv windows, fast weights) per layer for "
            "gated_deltanet; none for gla and mamba2"
        ),
        fla_variant="gated_deltanet",
    ),
}


def temporal_backend_capabilities(family: str, fla_variant: str) -> TemporalBackendCapabilities:
    """The declared contract for one configured backend."""
    capabilities = TEMPORAL_BACKEND_CAPABILITIES.get(family)
    if capabilities is None:
        raise ValueError(f"Unsupported temporal family: {family}")
    if family != "fla":
        return capabilities
    return TemporalBackendCapabilities(
        family=capabilities.family,
        supports_stepwise=fla_variant in STEPWISE_CAPABLE_FLA_VARIANTS,
        supports_fused=capabilities.supports_fused,
        needs_cuda=capabilities.needs_cuda,
        state_layout=capabilities.state_layout,
        fla_variant=fla_variant,
    )
