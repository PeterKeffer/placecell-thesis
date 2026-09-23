"""Model-contract helpers."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from .composite import CompositePlaceModel


def _parameter_count(module: nn.Module, *, trainable_only: bool) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


def build_model_contract(
    model: CompositePlaceModel,
    active_objectives: list[str],
) -> dict[str, object]:
    """Assemble the final machine-readable contract for a trained model artifact."""
    contract = dict(model.model_contract())
    contract["active_objectives"] = list(active_objectives)
    if "active_auxiliary_heads" not in contract:
        contract["active_auxiliary_heads"] = sorted(model.auxiliary_heads.keys())
    contract["total_parameters"] = _parameter_count(model, trainable_only=False)
    contract["trainable_parameters"] = _parameter_count(model, trainable_only=True)
    return contract


def write_parameter_shapes_csv(path: Path, model: CompositePlaceModel) -> Path:
    """Write deterministic parameter inventory for transparency outputs."""
    rows: list[list[object]] = []
    for name, parameter in model.named_parameters():
        rows.append(
            [
                name,
                "x".join(str(dimension) for dimension in parameter.shape),
                int(parameter.numel()),
                bool(parameter.requires_grad),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("name,shape,count,trainable\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    return path
