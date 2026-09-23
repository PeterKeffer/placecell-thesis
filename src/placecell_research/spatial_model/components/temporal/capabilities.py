"""What each temporal backend family can do, declared once."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TemporalBackendCapabilities:
    """One backend family's declared contract."""

    family: str
    supports_fused: bool
    state_layout: str


def _recurrent(family: str, *, fused: bool) -> TemporalBackendCapabilities:
    if family.startswith("lstm"):
        layout = "one (hidden, cell) pair per layer"
    elif family == "clockwork":
        layout = "one hidden tensor plus one per-row tick phase per layer"
    else:
        layout = "one hidden tensor per layer"
    return TemporalBackendCapabilities(family=family, supports_fused=fused, state_layout=layout)


TEMPORAL_BACKEND_CAPABILITIES: dict[str, TemporalBackendCapabilities] = {
    "mlp": TemporalBackendCapabilities(family="mlp", supports_fused=False, state_layout="none"),
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
}


def temporal_backend_capabilities(family: str) -> TemporalBackendCapabilities:
    """The declared contract for one configured backend."""
    capabilities = TEMPORAL_BACKEND_CAPABILITIES.get(family)
    if capabilities is None:
        raise ValueError(f"Unsupported temporal family: {family}")
    return capabilities
