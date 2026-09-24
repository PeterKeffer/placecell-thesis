"""Optimizer creation."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

from torch import Tensor, nn
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from placecell_research.config.schema import SpatialTrainingConfig
from placecell_research.spatial_model.protocol import PlaceModel
from placecell_research.utils.metrics import materialize_metric_values

NO_DECAY_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LocalResponseNorm,
    nn.Embedding,
)

NO_DECAY_MODULE_TYPE_NAMES = frozenset({"S5SSM", "SoftWTASparsifier", "PostHeadRMSNorm"})

NO_DECAY_PARAMETER_NAME_SUFFIXES = (
    "_token",
    ".mixer.A_log",
    ".mixer.D",
    ".gate_logits",
    ".gate_blackout_shift",
    ".gate_innovation_shift",
)


def clip_gradients_by_component(
    parameters: list[nn.Parameter],
    isolated_parameters: list[nn.Parameter],
    causal_groups: dict[str, list[nn.Parameter]],
    max_norm: float,
) -> tuple[Tensor, dict[str, float]]:
    """Clip the encoder-side and predictor-side groups INDEPENDENTLY, at the same threshold."""
    isolated_ids = {id(parameter) for parameter in isolated_parameters}
    assigned_ids: set[int] = set(isolated_ids)
    diagnostics: dict[str, float] = {}
    encoder_pre_clip: Tensor | None = None

    ordered_groups: list[tuple[str, list[nn.Parameter]]] = []
    for name in ("encoder", "predictor"):
        group = [
            parameter
            for parameter in causal_groups.get(name, [])
            if id(parameter) not in assigned_ids
        ]
        assigned_ids.update(id(parameter) for parameter in group)
        ordered_groups.append((name, group))
    ordered_groups.append(("other", [p for p in parameters if id(p) not in assigned_ids]))
    if isolated_parameters:
        ordered_groups.append(("isolated", isolated_parameters))

    pre_clip_norms: dict[str, Tensor] = {}
    for name, group in ordered_groups:
        if not group:
            diagnostics[f"{name}_grad_norm_pre_clip"] = 0.0
            diagnostics[f"{name}_grad_norm_post_clip"] = 0.0
            diagnostics[f"{name}_clip_scale"] = 1.0
            continue
        pre_clip = nn.utils.clip_grad_norm_(group, max_norm=max_norm)
        pre_clip_norms[name] = pre_clip
        if name == "encoder":
            encoder_pre_clip = pre_clip

    for name, pre_clip_value in materialize_metric_values(pre_clip_norms).items():
        post_clip_value = min(pre_clip_value, max_norm) if max_norm > 0.0 else pre_clip_value
        diagnostics[f"{name}_grad_norm_pre_clip"] = pre_clip_value
        diagnostics[f"{name}_grad_norm_post_clip"] = post_clip_value
        diagnostics[f"{name}_clip_scale"] = (
            min(1.0, max_norm / pre_clip_value) if pre_clip_value > 0.0 else 1.0
        )

    if encoder_pre_clip is None:
        for _name, group in ordered_groups:
            if group:
                encoder_pre_clip = nn.utils.clip_grad_norm_(group, max_norm=float("inf"))
                break
    if encoder_pre_clip is None:
        encoder_pre_clip = nn.utils.clip_grad_norm_([], max_norm=max_norm)
    return encoder_pre_clip, diagnostics


def clip_gradients(
    parameters: list[nn.Parameter],
    isolated_parameters: list[nn.Parameter],
    max_norm: float,
) -> Tensor:
    """Clip the main model and a gradient-isolated submodule SEPARATELY."""
    isolated_ids = {id(parameter) for parameter in isolated_parameters}
    main_parameters = [parameter for parameter in parameters if id(parameter) not in isolated_ids]
    main_norm = nn.utils.clip_grad_norm_(main_parameters, max_norm=max_norm)
    isolated_norm = (
        nn.utils.clip_grad_norm_(isolated_parameters, max_norm=max_norm)
        if isolated_parameters
        else None
    )
    if not main_parameters and isolated_norm is not None:
        return isolated_norm
    return main_norm


def _named_parameter_lookup(modules: Iterable[nn.Module]) -> tuple[dict[int, str], set[int]]:
    parameter_names: dict[int, str] = {}
    no_decay_module_parameter_ids: set[int] = set()
    for module in modules:
        for name, parameter in module.named_parameters():
            parameter_names.setdefault(id(parameter), name)
        for submodule in module.modules():
            is_excluded_module = isinstance(submodule, NO_DECAY_MODULE_TYPES) or (
                type(submodule).__name__ in NO_DECAY_MODULE_TYPE_NAMES
            )
            if is_excluded_module:
                for parameter in submodule.parameters(recurse=False):
                    no_decay_module_parameter_ids.add(id(parameter))
    return parameter_names, no_decay_module_parameter_ids


def _uses_no_weight_decay(
    parameter: nn.Parameter,
    *,
    parameter_name: str,
    no_decay_module_parameter_ids: set[int],
) -> bool:
    if id(parameter) in no_decay_module_parameter_ids:
        return True
    if parameter_name == "bias" or parameter_name.endswith(".bias"):
        return True
    return parameter_name.endswith(NO_DECAY_PARAMETER_NAME_SUFFIXES)


def build_optimizer(
    model: PlaceModel,
    auxiliary_heads: nn.ModuleDict,
    config: SpatialTrainingConfig,
) -> tuple[Optimizer, dict[str, list[nn.Parameter]]]:
    if config.optimizer == "adam" and config.weight_decay > 0.0:
        warnings.warn(
            "optimizer=adam with weight_decay>0 uses coupled Adam L2 decay. "
            "For decoupled weight decay use spatial_model.training.optimizer=adamw, "
            "or set spatial_model.training.weight_decay=0.0 for no decay.",
            RuntimeWarning,
            stacklevel=2,
        )
    parameter_groups = model.parameters_by_group()
    if "auxiliary" not in parameter_groups:
        parameter_groups["auxiliary"] = list(auxiliary_heads.parameters())
    unknown_lr_groups = sorted(set(config.group_learning_rates) - set(parameter_groups))
    if unknown_lr_groups:
        raise ValueError(
            f"training.group_learning_rates names unknown optimizer group(s) {unknown_lr_groups}; "
            f"valid groups for this model are {sorted(parameter_groups)}."
        )
    parameter_names, no_decay_module_parameter_ids = _named_parameter_lookup(
        [model, auxiliary_heads]
    )
    optimizer_groups = []
    for group_name, parameters in parameter_groups.items():
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            continue
        group_weight_decay = config.weight_decay
        group_learning_rate = config.group_learning_rates.get(group_name, config.learning_rate)
        if not config.exclude_bias_and_norm_from_weight_decay or config.weight_decay <= 0.0:
            optimizer_groups.append(
                {
                    "name": group_name,
                    "params": trainable,
                    "lr": group_learning_rate,
                    "weight_decay": group_weight_decay,
                }
            )
            continue

        decay_parameters: list[nn.Parameter] = []
        no_decay_parameters: list[nn.Parameter] = []
        for parameter in trainable:
            parameter_name = parameter_names.get(id(parameter), "")
            if _uses_no_weight_decay(
                parameter,
                parameter_name=parameter_name,
                no_decay_module_parameter_ids=no_decay_module_parameter_ids,
            ):
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        if decay_parameters:
            optimizer_groups.append(
                {
                    "name": group_name,
                    "params": decay_parameters,
                    "lr": group_learning_rate,
                    "weight_decay": group_weight_decay,
                }
            )
        if no_decay_parameters:
            optimizer_groups.append(
                {
                    "name": f"{group_name}_no_decay",
                    "params": no_decay_parameters,
                    "lr": group_learning_rate,
                    "weight_decay": 0.0,
                }
            )
    optimizer_class = {"adamw": AdamW, "adam": Adam, "rmsprop": RMSprop, "sgd": SGD}[
        config.optimizer
    ]
    options = {}
    if config.optimizer == "rmsprop":
        options = {
            "alpha": config.rmsprop_alpha,
            "momentum": config.rmsprop_momentum,
            "eps": config.rmsprop_eps,
        }
    optimizer = optimizer_class(
        optimizer_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        **options,
    )
    return optimizer, parameter_groups
