"""Config validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import sqrt

from placecell_research.spatial_model.regularization import REGULARIZATION_TARGETS
from placecell_research.spatial_model.representation_contract import (
    base_representation_names,
    base_representation_shapes,
    equal_code_block_bounds,
    representation_view_specs,
)

from .loader import revalidate_config
from .reuse import summarize_reuse
from .schema import ExperimentConfig, SpatialModelConfig, StudyConfig

HARD_K_SPARSIFIER_TYPES = frozenset({"kwinners", "grouped_kwinners", "lateral_inhibition"})

_SUPPORTED_ENVIRONMENT_KINDS = {"miniworld", "jaxenstein"}
_SIMPLEX_SPARSIFIER_TYPES = {"sparsemax", "entmax"}
_VICREG_VARIANCE_EPSILON = 1e-4


def _validate_input_corruption(config: ExperimentConfig) -> None:
    corruption = config.spatial_model.inputs.input_corruption
    if corruption.enabled and config.spatial_model.inputs.observation_source != "latent":
        raise ValueError("Input corruption requires latent observation_source.")
    if corruption.enabled and not config.spatial_model.teacher_student.teacher_sees_unmasked:
        raise ValueError(
            "Input corruption requires clean teacher targets. Set "
            "spatial_model.teacher_student.teacher_sees_unmasked=true."
        )


def _validate_regularization(config: ExperimentConfig) -> None:
    seen_targets: set[str] = set()
    for site in config.spatial_model.regularization.sites:
        if isinstance(site, Mapping):
            target = str(site.get("target", ""))
        else:
            target = site.target
        if target not in REGULARIZATION_TARGETS:
            raise ValueError(
                "spatial_model.regularization.sites[*].target must be one of "
                f"{REGULARIZATION_TARGETS!r}, got {target!r}."
            )
        if target in seen_targets:
            raise ValueError(
                "spatial_model.regularization.sites must not repeat targets. "
                f"Duplicate target: {target!r}."
            )
        seen_targets.add(target)


def _validate_representation_views(config: ExperimentConfig) -> None:
    code_blocks = sorted(config.spatial_model.code_blocks, key=lambda block: block.start)
    expected_start = 0
    seen_block_names: set[str] = set()
    for block in code_blocks:
        if not block.name or "." in block.name:
            raise ValueError(
                "spatial_model.code_blocks names must be non-empty and contain no dots."
            )
        if block.name in seen_block_names:
            raise ValueError(f"spatial_model.code_blocks contains duplicate name {block.name!r}.")
        if block.start != expected_start or block.end <= block.start:
            raise ValueError(
                "spatial_model.code_blocks must tile [0, code_dim) without gaps or overlaps; "
                f"got block {block.name!r} with start={block.start}, end={block.end}."
            )
        expected_start = block.end
        seen_block_names.add(block.name)
    if code_blocks and expected_start != config.spatial_model.training.code_dim:
        raise ValueError(
            "spatial_model.code_blocks must tile [0, code_dim); "
            f"got end={expected_start}, code_dim={config.spatial_model.training.code_dim}."
        )

    view_specs = representation_view_specs(config.spatial_model)
    explicit_names = {view.name for view in config.spatial_model.representation_views}
    auto_names = {
        f"{module_name}.{block.name}"
        for block in config.spatial_model.code_blocks
        for module_name in ("encoder", "predictor")
    }
    collisions = sorted(explicit_names & auto_names)
    if collisions:
        raise ValueError(
            f"spatial_model.representation_views must not redefine code block views: {collisions}."
        )
    base_config = replace(config.spatial_model, representation_views=[], code_blocks=[])
    base_shapes = base_representation_shapes(base_config)
    base_names = set(base_shapes)
    for name, (source, start, end) in view_specs.items():
        if name in base_names:
            raise ValueError(
                f"representation view '{name}' collides with a base representation name."
            )
        if source not in base_names:
            raise ValueError(
                f"representation view '{name}' source '{source}' is not a base representation. "
                f"Available: {sorted(base_names)}"
            )
        if not 0 <= start < end:
            raise ValueError(
                f"representation view '{name}' requires 0 <= start < end, "
                f"got start={start}, end={end}."
            )
        source_dim = base_shapes[source][-1]
        if isinstance(source_dim, int) and end > source_dim:
            raise ValueError(
                f"representation view '{name}' end={end} exceeds source "
                f"'{source}' size {source_dim}."
            )


def _validate_code_block_sparsifiers(config: ExperimentConfig) -> None:
    model = config.spatial_model
    code_dim = int(model.training.code_dim)
    if int(model.num_code_blocks) > code_dim:
        raise ValueError(
            "spatial_model.num_code_blocks cannot exceed training.code_dim "
            f"({model.num_code_blocks} > {code_dim}); every automatic code block must "
            "contain at least one dimension."
        )

    for block in config.spatial_model.code_blocks:
        sparsifier = block.sparsifier
        width = int(block.end) - int(block.start)
        if sparsifier.type in HARD_K_SPARSIFIER_TYPES:
            if sparsifier.k_fraction * width < 1:
                raise ValueError(
                    f"code block {block.name!r} {sparsifier.type} k_fraction must keep "
                    "at least one active code."
                )


def _simplex_block_widths(
    model: SpatialModelConfig,
    module_name: str,
) -> list[tuple[str, int]]:
    code_dim = int(model.training.code_dim)
    if module_name == "predictor":
        if model.predictor_sparsifier.type in _SIMPLEX_SPARSIFIER_TYPES:
            return [("predictor", code_dim)]
        return []

    if model.code_blocks:
        return [
            (block.name, int(block.end) - int(block.start))
            for block in model.code_blocks
            if block.sparsifier.type in _SIMPLEX_SPARSIFIER_TYPES
        ]
    if model.sparsifier.type not in _SIMPLEX_SPARSIFIER_TYPES:
        return []
    if model.num_code_blocks > 1:
        return [
            (f"block_{index}", end - start)
            for index, (start, end) in enumerate(
                equal_code_block_bounds(code_dim, model.num_code_blocks)
            )
        ]
    return [("encoder", code_dim)]


def _place_code_l1_is_constant(model: SpatialModelConfig, module_name: str) -> bool:
    if module_name == "predictor":
        return model.predictor_sparsifier.type in _SIMPLEX_SPARSIFIER_TYPES
    if model.code_blocks:
        return all(
            block.sparsifier.type in _SIMPLEX_SPARSIFIER_TYPES for block in model.code_blocks
        )
    return model.sparsifier.type in _SIMPLEX_SPARSIFIER_TYPES


def _place_code_targets(
    model: SpatialModelConfig,
) -> dict[str, tuple[SpatialModelConfig, str]]:
    return {
        "encoder.place_codes": (model, "encoder"),
        "teacher.place_codes": (model, "encoder"),
        "predictor.place_codes": (model, "predictor"),
    }


def _validate_simplex_objectives(config: ExperimentConfig) -> None:
    model = config.spatial_model
    place_code_targets = _place_code_targets(model)
    for objective_name, objective in model.objectives.items():
        if objective.weight <= 0.0:
            continue
        for target in objective.targets:
            target_contract = place_code_targets.get(target)
            if target_contract is None:
                continue
            target_model, module_name = target_contract
            if objective.type == "vicreg" and objective.variance_weight > 0.0:
                for block_name, block_width in _simplex_block_widths(target_model, module_name):
                    maximum_common_std = sqrt(1.0 / block_width + _VICREG_VARIANCE_EPSILON)
                    if objective.minimum_std > maximum_common_std:
                        raise ValueError(
                            f"objective {objective_name!r} targets simplex place code {target!r} "
                            f"block {block_name!r} (width={block_width}), but VICReg minimum_std="
                            f"{objective.minimum_std:g} exceeds the mathematical common-dimension "
                            f"upper bound {maximum_common_std:.4f}. Lower minimum_std, target the "
                            "dense pre-sparsifier representation, or use a non-simplex sparsifier."
                        )
            elif objective.type in {"l1_sparsity", "l1_capacity"} and _place_code_l1_is_constant(
                target_model, module_name
            ):
                dense_target = target.removesuffix(".place_codes") + ".pre_sparsifier"
                raise ValueError(
                    f"L1 objective {objective_name!r} is constant on simplex target {target!r} "
                    f"and therefore has zero gradient. Target {dense_target!r} instead or remove "
                    "the objective."
                )


def _validate_context_channels(channels: list[str], field_name: str) -> None:
    duplicate_channels = [
        channel for index, channel in enumerate(channels) if channel in channels[:index]
    ]
    if duplicate_channels:
        raise ValueError(
            f"spatial_model.inputs.{field_name} must not contain duplicates. "
            f"Got duplicates {duplicate_channels!r}."
        )


def _validate_collection_action_probabilities(config: ExperimentConfig) -> None:
    if config.collection.policy == "continuous_random" and config.environment.kind != "miniworld":
        raise ValueError(
            "collection.policy='continuous_random' requires environment.kind='miniworld'."
        )
    supported_policies = {"ou_smoothed_random", "independent_random"}
    if (
        config.collection.policy not in supported_policies
        and config.collection.action_probabilities
    ):
        raise ValueError(
            "collection.action_probabilities is only supported when "
            "collection.policy is 'ou_smoothed_random' or 'independent_random'."
        )


def _validate_collection_vectorized_contract(config: ExperimentConfig) -> None:
    if not config.collection.vectorized:
        return
    environment_kind = str(config.environment.kind).strip().lower()
    if environment_kind != "jaxenstein":
        raise ValueError(
            "collection.vectorized is only supported when environment.kind='jaxenstein'."
        )
    if config.collection.policy != "ou_smoothed_random":
        raise ValueError(
            "collection.vectorized for environment.kind='jaxenstein' requires "
            "collection.policy='ou_smoothed_random' because the JAX rollout uses JaxOUPolicy."
        )
    if config.collection.action_probabilities:
        raise ValueError(
            "collection.action_probabilities is ignored by collection.vectorized for "
            "environment.kind='jaxenstein'; remove it or disable vectorized collection."
        )
    if not config.collection.save_rgb:
        raise ValueError(
            "collection.vectorized for environment.kind='jaxenstein' requires "
            "collection.save_rgb=true."
        )
    if config.collection.save_topdown:
        raise ValueError(
            "collection.vectorized for environment.kind='jaxenstein' requires "
            "collection.save_topdown=false."
        )
    if config.collection.spawn_regions:
        raise ValueError(
            "collection.spawn_regions is ignored by collection.vectorized for "
            "environment.kind='jaxenstein'."
        )
    if (
        bool(config.environment.env_kwargs.get("uniform_spawn", False))
        and config.environment.randomize_agent_start is False
    ):
        raise ValueError(
            "environment.env_kwargs.uniform_spawn=true conflicts with "
            "environment.randomize_agent_start=false during vectorized JAXenstein collection."
        )
    if int(config.collection.safety.num_workers) != 1:
        raise ValueError(
            "collection.vectorized for environment.kind='jaxenstein' requires "
            "collection.safety.num_workers=1."
        )
    if config.collection.safety.isolate_cuda_miniworld:
        raise ValueError(
            "collection.vectorized for environment.kind='jaxenstein' requires "
            "collection.safety.isolate_cuda_miniworld=false."
        )


def _validate_environment_backend_config(config: ExperimentConfig) -> None:
    environment_kind = str(config.environment.kind).strip().lower()
    if environment_kind not in _SUPPORTED_ENVIRONMENT_KINDS:
        raise ValueError(
            f"environment.kind must be one of {sorted(_SUPPORTED_ENVIRONMENT_KINDS)!r}, "
            f"got {config.environment.kind!r}."
        )


def validate_experiment_config(config: ExperimentConfig) -> list[str]:
    """Raise on invalid config, return warnings otherwise."""
    config = revalidate_config(config)
    warnings: list[str] = []
    reuse_summary = summarize_reuse(config)
    split = config.splits
    if split.strategy != "manual_ids":
        total = split.train_fraction + split.validation_fraction + split.test_fraction
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split fractions must sum to 1.0.")
    _validate_input_corruption(config)
    _validate_regularization(config)
    _validate_representation_views(config)
    _validate_code_block_sparsifiers(config)
    _validate_simplex_objectives(config)
    _validate_environment_backend_config(config)
    if config.spatial_model.sparsifier.type in HARD_K_SPARSIFIER_TYPES:
        if config.spatial_model.sparsifier.k_fraction * config.spatial_model.training.code_dim < 1:
            raise ValueError(
                f"{config.spatial_model.sparsifier.type} k_fraction must keep at least one "
                "active code."
            )
    if config.spatial_model.predictor_sparsifier.type in HARD_K_SPARSIFIER_TYPES:
        if (
            config.spatial_model.predictor_sparsifier.k_fraction
            * config.spatial_model.training.code_dim
            < 1
        ):
            raise ValueError(
                f"predictor {config.spatial_model.predictor_sparsifier.type} k_fraction must "
                "keep at least one active code."
            )
    available_representations = set(base_representation_names(config.spatial_model))
    if config.evaluation.online_decode_source not in available_representations:
        raise ValueError(
            "evaluation.online_decode_source must be one of the model's "
            f"available runtime representations. Got "
            f"{config.evaluation.online_decode_source!r}; "
            f"available: {sorted(available_representations)}"
        )
    _validate_context_channels(
        config.spatial_model.inputs.encoder_context_channels,
        "encoder_context_channels",
    )
    _validate_context_channels(
        config.spatial_model.inputs.predictor_context_channels,
        "predictor_context_channels",
    )
    _validate_collection_action_probabilities(config)
    _validate_collection_vectorized_contract(config)
    if config.dataset.artifact_id == "":
        warnings.append(
            "dataset.artifact_id is empty. Collection or pipeline injection must fill it "
            "explicitly."
        )
    training = config.spatial_model.training
    if training.optimizer == "adam" and training.weight_decay > 0.0:
        warnings.append(
            "spatial_model.training.optimizer=adam with weight_decay>0 uses coupled Adam L2 decay. "
            "Use spatial_model.training.optimizer=adamw for decoupled weight decay, or set "
            "spatial_model.training.weight_decay=0.0 for no decay."
        )
    if (
        config.policies.training_resume != "fresh"
        and not config.reuse.vision_encoder_artifact_id
        and not config.reuse.place_model_artifact_id
        and not config.reuse.place_model_checkpoint_path
    ):
        warnings.append(
            "training_resume is set, but no artifact id or place-model checkpoint path was "
            "configured. Training will still start fresh."
        )
    if config.reuse.place_model_checkpoint_path:
        if config.policies.training_resume == "fresh":
            raise ValueError(
                "reuse.place_model_checkpoint_path requires policies.training_resume to be "
                "weights_only or weights_and_optimizer."
            )
        if config.reuse.place_model_artifact_id:
            raise ValueError(
                "Configure only one place-model resume source: "
                "reuse.place_model_artifact_id or reuse.place_model_checkpoint_path."
            )
    if reuse_summary["vision_encoder"]["stage_behavior"] == "reuse_existing_artifact":
        warnings.append(
            "reuse.vision_encoder_artifact_id is set. `train_vision_encoder` will reuse that "
            "artifact instead of training a new encoder."
        )
    if reuse_summary["place_model"]["stage_behavior"] == "reuse_existing_artifact":
        warnings.append(
            "reuse.place_model_artifact_id is set. `train_place_model` will reuse that artifact "
            "instead of training a new model."
        )
    if any(
        objective.type == "prediction_alignment" and objective.target_offset == 0
        for objective in config.spatial_model.objectives.values()
    ) and (
        config.spatial_model.training.bptt_window > 0
        or config.spatial_model.prediction_bootstrap.enabled
    ):
        raise ValueError("Same-step prediction requires full sequences without bootstrapping.")
    if config.spatial_model.teacher_student.mode == "none" and any(
        objective.type == "prediction_alignment"
        for objective in config.spatial_model.objectives.values()
    ):
        warnings.append(
            "A prediction-style alignment objective is configured without an EMA teacher. "
            "It will fall back to detached online encoder targets."
        )
    for name, objective in config.spatial_model.objectives.items():
        if not objective.type:
            raise ValueError(f"Objective '{name}' is missing a type.")
    return warnings


def validate_study_config(config: StudyConfig) -> list[str]:
    """Raise on invalid study config."""
    config = revalidate_config(config)
    warnings: list[str] = []
    if config.sweep is None and config.curriculum is None:
        raise ValueError("Study config must define either `sweep` or `curriculum`.")
    if config.sweep and config.curriculum:
        raise ValueError("Study config may define only one of `sweep` or `curriculum`.")
    if config.sweep and config.sweep.method == "paired":
        parameter_lengths = {
            key: len(values)
            for key, values in config.sweep.parameters.items()
            if isinstance(values, list)
        }
        non_list_parameters = sorted(
            key for key, values in config.sweep.parameters.items() if not isinstance(values, list)
        )
        if non_list_parameters:
            raise ValueError(
                "Paired sweep parameters must all be lists; got scalar values for "
                f"{non_list_parameters}."
            )
        if not parameter_lengths:
            raise ValueError("Paired sweeps must define at least one parameter list.")
        if len(set(parameter_lengths.values())) != 1:
            raise ValueError(
                f"Paired sweep parameter lists must have equal lengths; got {parameter_lengths}."
            )
        if next(iter(parameter_lengths.values())) == 0:
            raise ValueError("Paired sweep parameter lists must not be empty.")
    if config.curriculum:
        unsupported_encoding_keys = sorted(set(config.curriculum.encoding) - {"encode_each"})
        if unsupported_encoding_keys:
            raise ValueError(
                f"Curriculum encoding supports only `encode_each`; got {unsupported_encoding_keys}."
            )
        for source_name, source in config.curriculum.sources.items():
            if not source.raw_dataset and not source.environment and not source.collection:
                raise ValueError(
                    f"Curriculum source '{source_name}' must set raw_dataset or provide "
                    "environment/collection overrides for collection."
                )
        for phase in config.curriculum.phases:
            if phase.resume_from and phase.resume_policy == "fresh":
                warnings.append(
                    f"Curriculum phase '{phase.name}' sets resume_from but resume_policy=fresh."
                )
    return warnings
