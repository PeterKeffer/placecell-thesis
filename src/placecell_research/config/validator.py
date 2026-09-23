"""Config validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from math import isfinite, sqrt

from pydantic import TypeAdapter

from placecell_research.spatial_model.dag import (
    available_dag_representations,
    resolve_node_config,
    topological_node_order,
)
from placecell_research.spatial_model.regularization import REGULARIZATION_TARGETS
from placecell_research.spatial_model.representation_contract import (
    base_representation_names,
    base_representation_shapes,
    equal_code_block_bounds,
    representation_view_specs,
)

from .downstream_feature_sources import (
    PLACE_CODE_FEATURE_SOURCES,
    PLACE_CODE_SOURCES,
    missing_place_code_stats_sources,
    resolved_place_code_stats_path,
)
from .downstream_schema import DownstreamRunConfig
from .reuse import summarize_reuse
from .schema import ExperimentConfig, SpatialModelConfig, StudyConfig

HARD_K_SPARSIFIER_TYPES = frozenset({"kwinners", "grouped_kwinners", "lateral_inhibition"})

_SUPPORTED_ENVIRONMENT_KINDS = {"miniworld", "jaxenstein"}
_SIMPLEX_SPARSIFIER_TYPES = {"sparsemax", "entmax"}
_VICREG_VARIANCE_EPSILON = 1e-4
_COORDINATE_GOAL_SOURCES = {
    "goal_xy",
    "goal_xy_scaled",
    "goal_xy_map01",
    "goal_delta_xy",
    "goal_rbf_code",
}
_ALL_GOAL_SOURCES = _COORDINATE_GOAL_SOURCES | {
    "goal_place_code",
    "goal_grid_code",
}


def _validate_campaign_tracking(config: ExperimentConfig) -> None:
    tracking = config.tracking
    required_values = {
        "campaign_name": tracking.campaign_name,
        "experiment_id": tracking.experiment_id,
        "experiment_arm": tracking.experiment_arm,
    }
    configured = {name for name, value in required_values.items() if str(value).strip()}
    if configured and len(configured) != len(required_values):
        missing = sorted(set(required_values) - configured)
        raise ValueError(
            "Campaign indexing requires tracking.campaign_name, tracking.experiment_id, and "
            f"tracking.experiment_arm together. Missing: {missing}."
        )
    if tracking.campaign_group and not configured:
        raise ValueError(
            "tracking.campaign_group requires the campaign name, experiment id, and arm."
        )


def _revalidate_config(config):
    adapter = TypeAdapter(type(config))
    payload = asdict(config) if is_dataclass(config) else adapter.dump_python(config, mode="python")
    return adapter.validate_python(payload)


def _available_runtime_representations(config: ExperimentConfig) -> set[str]:
    if config.spatial_model.dag_nodes:
        available = available_dag_representations(config.spatial_model)
        if config.spatial_model.dag_expert_combiner in {"mixture", "attention"}:
            available |= {"experts.place_codes", "experts.gate_responsibilities"}
        elif config.spatial_model.dag_expert_combiner == "competence":
            available |= {
                "experts.place_codes",
                "experts_teacher.place_codes",
                "experts.routing_weights",
                "experts.routing_onehot",
            }
        elif config.spatial_model.dag_expert_combiner == "thalamic":
            available |= {
                "experts.place_codes",
                "experts_teacher.place_codes",
                "experts.routing_weights",
                "experts.routing_onehot",
                "thalamus.gate_logits",
                "thalamus.competence_target",
            }
        return available
    return set(base_representation_names(config.spatial_model))


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
    models = [("spatial_model", config.spatial_model)]
    models.extend(
        (f"spatial_model.dag_nodes[{node.name!r}].model", node.model)
        for node in config.spatial_model.dag_nodes
    )
    for label, model in models:
        code_dim = int(model.training.code_dim)
        if int(model.num_code_blocks) > code_dim:
            raise ValueError(
                f"{label}.num_code_blocks cannot exceed training.code_dim "
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
    targets = {
        "encoder.place_codes": (model, "encoder"),
        "teacher.place_codes": (model, "encoder"),
        "predictor.place_codes": (model, "predictor"),
    }
    for node in model.dag_nodes:
        targets.update(
            {
                f"{node.name}:encoder.place_codes": (node.model, "encoder"),
                f"{node.name}:teacher.place_codes": (node.model, "encoder"),
                f"{node.name}:predictor.place_codes": (node.model, "predictor"),
            }
        )
    return targets


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


def _validate_objective_batch_requirements(config: ExperimentConfig) -> None:
    active_cpc_objectives = sorted(
        name
        for name, objective in config.spatial_model.objectives.items()
        if objective.type == "cpc_multi_horizon" and objective.weight > 0.0
    )
    batch_size = int(config.spatial_model.training.batch_size)
    if active_cpc_objectives and batch_size < 2:
        raise ValueError(
            "cpc_multi_horizon requires training.batch_size >= 2 for cross-episode negatives; "
            f"got {batch_size} for objectives {active_cpc_objectives}."
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


def _node_root_divergence_warnings(config: ExperimentConfig) -> list[str]:
    """Warn where a DAG node equals a schema default that differs from the root."""
    warnings: list[str] = []
    if not config.spatial_model.dag_nodes:
        return warnings
    default_model = SpatialModelConfig()
    watched = ("normalize_codes", "head_activation", "head_weight_sparsity", "dropout")
    for node in config.spatial_model.dag_nodes:
        for module in ("encoder", "predictor"):
            root_module = getattr(config.spatial_model, module)
            node_module = getattr(node.model, module)
            schema_module = getattr(default_model, module)
            for field_name in watched:
                schema_value = getattr(schema_module, field_name, None)
                root_value = getattr(root_module, field_name, None)
                node_value = getattr(node_module, field_name, None)
                if node_value == schema_value and root_value != schema_value:
                    warnings.append(
                        f"spatial_model.dag_nodes[{node.name!r}].model.{module}.{field_name}="
                        f"{node_value!r} is the SCHEMA DEFAULT, but the root sets {root_value!r}. "
                        "A node is built from its own model block and never inherits the root, so "
                        "verify that this difference is intentional."
                    )
    return warnings


def _temporal_family_contract_warnings(config: ExperimentConfig) -> list[str]:
    warnings: list[str] = []
    predictor_family = config.spatial_model.predictor.family
    self_fed_rollout_enabled = any(
        objective.type == "multistep_rollout" and objective.weight > 0.0
        for objective in config.spatial_model.objectives.values()
    )
    if predictor_family == "transformer":
        warnings.append(
            "spatial_model.predictor.family='transformer' uses full-sequence causal training and "
            "exact bounded-context recomputation for stepwise rollouts. Self-fed rollout cost "
            "therefore grows with the configured context and rollout horizon."
        )
    if (
        predictor_family in {"transformer", "ssm", "ema_ssm", "mamba", "xlstm"}
        and not self_fed_rollout_enabled
    ):
        warnings.append(
            f"spatial_model.predictor.family={predictor_family!r} has no active "
            "multistep_rollout objective. Its ordinary predictor pass is teacher-forced from the "
            "previous encoder code, so prediction success alone does not establish persistent "
            "path integration."
        )
    return warnings


def _has_masked_prediction_source(config: ExperimentConfig) -> bool:
    masking = config.spatial_model.inputs.visual_masking
    corruption = config.spatial_model.inputs.input_corruption
    return (
        masking.blackout_probability > 0.0
        or masking.stride > 1
        or (
            corruption.enabled
            and (corruption.blackout_num_blocks > 0 or corruption.noise_num_blocks > 0)
        )
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


def _validate_dag_nodes(config: ExperimentConfig) -> None:
    dag_nodes = config.spatial_model.dag_nodes
    lateral_consensus = config.spatial_model.lateral_consensus
    if lateral_consensus.enabled:
        if not dag_nodes:
            raise ValueError("spatial_model.lateral_consensus.enabled=true requires dag_nodes.")
        if config.spatial_model.dag_expert_combiner not in {"competence", "thalamic"}:
            raise ValueError(
                "spatial_model.lateral_consensus is applied only by the competence or thalamic "
                "DAG router; set dag_expert_combiner to 'competence' or 'thalamic'."
            )
    if not dag_nodes:
        if config.spatial_model.dag_expert_combiner != "none":
            raise ValueError(
                "spatial_model.dag_expert_combiner must be 'none' when dag_nodes is empty."
            )
        return
    topological_node_order(config.spatial_model, list(dag_nodes))
    expert_combiner_active = config.spatial_model.dag_expert_combiner != "none"
    root_code_dim = config.spatial_model.training.code_dim
    for node in dag_nodes:
        if node.model.dag_nodes:
            raise ValueError(
                f"dag node {node.name!r} model.dag_nodes must be empty (no nested graphs)."
            )
        resolved = resolve_node_config(node)
        if expert_combiner_active and resolved.training.code_dim != root_code_dim:
            raise ValueError(
                f"dag_expert_combiner={config.spatial_model.dag_expert_combiner!r} combines every "
                f"expert code, so node {node.name!r} code_dim={resolved.training.code_dim} must "
                f"equal the root code_dim={root_code_dim}."
            )
        for label, sparsifier in (
            ("sparsifier", resolved.sparsifier),
            ("predictor_sparsifier", resolved.predictor_sparsifier),
        ):
            if sparsifier.type in HARD_K_SPARSIFIER_TYPES:
                if sparsifier.k_fraction * resolved.training.code_dim < 1:
                    raise ValueError(
                        f"dag node {node.name!r} {label} {sparsifier.type} k_fraction must keep at "
                        "least one active code (k_fraction*code_dim >= 1)."
                    )
    if config.spatial_model.dag_expert_combiner == "thalamic":
        routed_candidate_count = len(dag_nodes) + int(
            config.spatial_model.thalamic_router.include_root
        )
        if routed_candidate_count < 2:
            raise ValueError(
                "dag_expert_combiner='thalamic' requires at least two routed candidates; "
                f"got {routed_candidate_count}."
            )
        if config.spatial_model.thalamic_router.top_k > routed_candidate_count:
            raise ValueError(
                "spatial_model.thalamic_router.top_k cannot exceed the number of routed "
                f"candidates ({config.spatial_model.thalamic_router.top_k} > "
                f"{routed_candidate_count})."
            )
        distillation_objectives = [
            (name, objective)
            for name, objective in config.spatial_model.objectives.items()
            if objective.type == "router_distillation" and objective.weight > 0.0
        ]
        if not distillation_objectives:
            raise ValueError(
                "dag_expert_combiner='thalamic' requires a positive-weight router_distillation "
                "objective on ['thalamus.gate_logits', 'thalamus.competence_target']; otherwise "
                "the detached gate has no learning signal."
            )
        expected_targets = ["thalamus.gate_logits", "thalamus.competence_target"]
        for objective_name, objective in distillation_objectives:
            if objective.targets != expected_targets:
                raise ValueError(
                    f"router_distillation objective {objective_name!r} requires targets "
                    f"{expected_targets}, got {objective.targets}."
                )
        phases = config.spatial_model.training.phases
        if phases and not any("experts" in phase.train for phase in phases):
            raise ValueError(
                "dag_expert_combiner='thalamic' with an explicit training.phases schedule must "
                "train the 'experts' selector in at least one phase; otherwise the thalamic gate "
                "is frozen for the entire run."
            )


def _validate_top_down_context(config: ExperimentConfig) -> None:
    model = config.spatial_model
    source = model.top_down_source
    if not isfinite(model.top_down_context_limit) or model.top_down_context_limit < 0:
        raise ValueError("spatial_model.top_down_context_limit must be finite and nonnegative.")
    if not source and model.top_down_context_limit != 0:
        raise ValueError("top_down_context_limit requires top_down_source.")
    if not source:
        if model.top_down_context_dim != 0:
            raise ValueError(
                "spatial_model.top_down_context_dim must be 0 when top_down_source is empty."
            )
        return
    if not model.dag_nodes:
        raise ValueError(
            "spatial_model.top_down_source requires dag_nodes because only DagPlaceModel "
            "implements open-loop top-down context."
        )
    available = available_dag_representations(model)
    if source not in available:
        raise ValueError(
            f"spatial_model.top_down_source={source!r} is not produced by the configured DAG. "
            f"Available representations: {sorted(available)}"
        )

    producer_config = model
    local_source = source
    if ":" in source:
        node_name, local_source = source.split(":", 1)
        producer_config = next(node.model for node in model.dag_nodes if node.name == node_name)
    if local_source not in {
        "encoder.place_codes",
        "predictor.place_codes",
        "teacher.place_codes",
    }:
        raise ValueError(
            "spatial_model.top_down_source currently supports place-code representations only, "
            f"got {source!r}."
        )
    expected_dim = producer_config.training.code_dim
    if model.top_down_context_dim != expected_dim:
        raise ValueError(
            f"spatial_model.top_down_context_dim={model.top_down_context_dim} must equal "
            f"top_down_source {source!r} code_dim={expected_dim}."
        )
    noncausal_reason = _top_down_noncausal_reason(model, source)
    if noncausal_reason is not None:
        raise ValueError(
            "spatial_model.top_down_source must be prefix-causal before its one-step delay; "
            f"{source!r} is non-causal because {noncausal_reason}."
        )


def _top_down_noncausal_reason(
    root_config: SpatialModelConfig,
    source: str,
) -> str | None:
    producer_config = root_config
    local_source = source
    producer_node = None
    if ":" in source:
        node_name, local_source = source.split(":", 1)
        producer_node = next(node for node in root_config.dag_nodes if node.name == node_name)
        producer_config = producer_node.model

    module_name = local_source.split(".", 1)[0]
    temporal_configs = []
    if module_name in {"encoder", "teacher"}:
        temporal_configs.append(("encoder", producer_config.encoder))
    elif module_name == "predictor":
        temporal_configs.extend(
            [
                ("encoder", producer_config.encoder),
                ("predictor", producer_config.predictor),
            ]
        )
    for temporal_name, temporal_config in temporal_configs:
        if temporal_config.family == "transformer" and not temporal_config.causal:
            owner = producer_node.name if producer_node is not None else "root"
            return f"{owner}.{temporal_name}.family='transformer' has causal=false"

    environment_code = producer_config.environment_code
    if temporal_configs and environment_code.enabled and environment_code.inference == "episode":
        owner = producer_node.name if producer_node is not None else "root"
        return f"{owner}.environment_code.inference='episode' pools future steps"

    if producer_node is not None:
        return _top_down_noncausal_reason(root_config, producer_node.input_source)
    return None


def _validate_stored_replay_config(config: ExperimentConfig) -> None:
    replay = config.spatial_model.training.replay
    if not replay.enabled:
        return
    if config.spatial_model.inputs.observation_source != "latent":
        raise ValueError(
            "spatial_model.training.replay.enabled requires "
            "spatial_model.inputs.observation_source='latent' because stored replay batches "
            "contain latent observations."
        )
    if not replay.source_encoded_dataset_artifact_id or not replay.source_split_artifact_id:
        raise ValueError(
            "spatial_model.training.replay.enabled requires both "
            "source_encoded_dataset_artifact_id and source_split_artifact_id."
        )


def _removed_feature_settings(config: ExperimentConfig) -> list[str]:
    settings = {
        "evaluation.expert_probe.enabled": config.evaluation.expert_probe.enabled,
        "evaluation.history_ablation.enabled": config.evaluation.history_ablation.enabled,
    }
    return sorted(
        config.spatial_model.removed_feature_settings()
        + [name for name, is_set in settings.items() if is_set]
    )


def _reject_removed_features(config: ExperimentConfig) -> None:
    removed = _removed_feature_settings(config)
    if removed:
        raise ValueError(f"These settings are not supported in this repository: {removed}.")


def validate_experiment_config(config: ExperimentConfig) -> list[str]:
    """Raise on invalid config, return warnings otherwise."""
    config = _revalidate_config(config)
    _reject_removed_features(config)
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
    _validate_objective_batch_requirements(config)
    _validate_dag_nodes(config)
    _validate_top_down_context(config)
    _validate_stored_replay_config(config)
    _validate_campaign_tracking(config)
    _validate_environment_backend_config(config)
    warnings.extend(_node_root_divergence_warnings(config))
    warnings.extend(_temporal_family_contract_warnings(config))
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
    available_representations = _available_runtime_representations(config)
    if config.evaluation.online_decode_source not in available_representations:
        raise ValueError(
            "evaluation.online_decode_source must be one of the model's "
            f"available runtime representations. Got "
            f"{config.evaluation.online_decode_source!r}; "
            f"available: {sorted(available_representations)}"
        )
    history_ablation = config.evaluation.history_ablation
    if history_ablation.enabled:
        if config.spatial_model.inputs.observation_source == "action":
            raise ValueError("evaluation.history_ablation requires latent or RGB observations.")
        if config.spatial_model.dag_nodes:
            raise ValueError(
                "evaluation.history_ablation currently supports non-DAG place models only."
            )
        required_history_sources = {
            "encoder.state_readout",
            "encoder.place_codes",
        }
        uncollected_history_sources = sorted(
            required_history_sources - set(config.evaluation.sources)
        )
        if uncollected_history_sources:
            raise ValueError(
                "evaluation.history_ablation requires its clean sources in evaluation.sources. "
                f"Missing: {uncollected_history_sources}."
            )
        missing_history_sources = sorted(required_history_sources - available_representations)
        if missing_history_sources:
            raise ValueError(
                "evaluation.history_ablation requires exported encoder state and code sources. "
                f"Missing: {missing_history_sources}; available: "
                f"{sorted(available_representations)}"
            )
    expert_probe = config.evaluation.expert_probe
    if expert_probe.enabled:
        probe_sources = [expert_probe.routing_source, *expert_probe.sources]
        unknown_probe_sources = sorted(set(probe_sources) - available_representations)
        if unknown_probe_sources:
            raise ValueError(
                "evaluation.expert_probe sources must be available runtime representations. "
                f"Unknown: {unknown_probe_sources}; available: {sorted(available_representations)}"
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
        objective.type in {"prediction_alignment", "masked_prediction_alignment"}
        for objective in config.spatial_model.objectives.values()
    ):
        warnings.append(
            "A prediction-style alignment objective is configured without an EMA teacher. "
            "It will fall back to detached online encoder targets."
        )
    if any(
        objective.type == "masked_prediction_alignment"
        for objective in config.spatial_model.objectives.values()
    ):
        if not config.spatial_model.masked_predictor.enabled:
            raise ValueError(
                "masked_prediction_alignment requires spatial_model.masked_predictor.enabled=true."
            )
        if not _has_masked_prediction_source(config):
            warnings.append(
                "masked_prediction_alignment is configured, but no visual masking or input "
                "corruption source is active, so the masked objective will see no targets."
            )
        if not config.spatial_model.teacher_student.teacher_sees_unmasked:
            raise ValueError(
                "masked_prediction_alignment requires clean teacher targets. Set "
                "spatial_model.teacher_student.teacher_sees_unmasked=true."
            )
    for name, objective in config.spatial_model.objectives.items():
        if not objective.type:
            raise ValueError(f"Objective '{name}' is missing a type.")
    return warnings


def _validate_online_pcdt_training_config(
    config: DownstreamRunConfig,
    *,
    algorithm: str,
    her_goal_representation: str,
) -> None:
    from .downstream_validation import validate_route_stitching_position_source

    online_pcdt = config.training.online_pcdt
    if algorithm == "online_pcdt":
        if online_pcdt is None:
            raise ValueError(
                "training.algorithm='online_pcdt' requires a training.online_pcdt block."
            )
        if config.observation.mode != "feature_vector":
            raise ValueError(
                "training.algorithm='online_pcdt' currently requires feature_vector observations."
            )
        if her_goal_representation not in {"goal_xy", "goal_grid_code"}:
            raise ValueError(
                "training.algorithm='online_pcdt' supports "
                "training.her_goal_representation in {'goal_xy', 'goal_grid_code'}."
            )
        if her_goal_representation == "goal_grid_code":
            if "synthetic_grid_cells" not in config.observation.feature_sources:
                raise ValueError(
                    "training.her_goal_representation='goal_grid_code' requires "
                    "'synthetic_grid_cells' in observation.feature_sources so the current-position "
                    "state code and the goal code share one grid bank."
                )
            xy_only_mechanisms = {
                "spatially_balanced_replay": online_pcdt.spatially_balanced_replay,
                "route_stitching": online_pcdt.route_stitching,
                "practice_goal_relabeling": online_pcdt.practice_goal_relabeling,
                "goal_chain_on_success": online_pcdt.goal_chain_on_success,
            }
            enabled_xy_only = sorted(
                name for name, value in xy_only_mechanisms.items() if bool(value)
            )
            if enabled_xy_only:
                raise ValueError(
                    "training.her_goal_representation='goal_grid_code' operates in grid-code goal "
                    f"space; disable XY-only mechanisms {enabled_xy_only} on training.online_pcdt."
                )
        validate_route_stitching_position_source(config)
        embedded_goal_sources = {
            "goal_xy",
            "goal_xy_scaled",
            "goal_xy_map01",
            "goal_delta_xy",
            "goal_rbf_code",
            "goal_place_code",
            "goal_grid_code",
        }.intersection(config.observation.feature_sources)
        if embedded_goal_sources:
            raise ValueError(
                "training.algorithm='online_pcdt' requires a goal-independent base observation; "
                f"remove {sorted(embedded_goal_sources)!r} from observation.feature_sources."
            )
        if int(config.training.n_envs) != 1:
            raise ValueError(
                "training.algorithm='online_pcdt' currently requires training.n_envs=1."
            )
        if str(config.training.train_freq_unit) != "step":
            raise ValueError(
                "training.algorithm='online_pcdt' requires training.train_freq_unit='step'."
            )
        if isinstance(config.training.gradient_steps, str):
            raise ValueError(
                "training.algorithm='online_pcdt' requires an explicit integer gradient_steps."
            )
        if int(config.training.buffer_size) < int(config.training.batch_size):
            raise ValueError("training.algorithm='online_pcdt' requires buffer_size >= batch_size.")
        first_predictive_update = (
            (int(config.training.batch_size) + int(config.training.train_freq) - 1)
            // int(config.training.train_freq)
        ) * int(config.training.train_freq)
        if int(config.training.learning_starts) <= first_predictive_update:
            raise ValueError(
                "training.algorithm='online_pcdt' requires learning_starts to exceed the first "
                f"predictive update at step {first_predictive_update}, so the encoder receives "
                "a representation-only burn-in."
            )
        if int(config.training.total_timesteps) <= int(config.training.learning_starts):
            raise ValueError(
                "training.algorithm='online_pcdt' requires total_timesteps > learning_starts."
            )
        if int(online_pcdt.hidden_dim) % int(online_pcdt.attention_heads) != 0:
            raise ValueError(
                "training.online_pcdt.hidden_dim must be divisible by attention_heads."
            )
        if int(online_pcdt.hindsight_control_segment_length) > 1:
            if float(online_pcdt.hindsight_ratio) != 1.0:
                raise ValueError(
                    "training.online_pcdt.hindsight_control_segment_length > 1 "
                    "requires hindsight_ratio=1.0."
                )
            if float(online_pcdt.hindsight_near_fraction) > 0.0 and int(
                online_pcdt.hindsight_control_segment_length
            ) > int(online_pcdt.hindsight_near_horizon):
                raise ValueError(
                    "training.online_pcdt.hindsight_control_segment_length cannot "
                    "exceed hindsight_near_horizon when near hindsight is enabled."
                )
            if int(online_pcdt.hindsight_control_segment_length) > int(config.training.batch_size):
                raise ValueError(
                    "training.online_pcdt.hindsight_control_segment_length cannot "
                    "exceed training.batch_size."
                )
        if float(online_pcdt.segment_goal_contrast_weight) > 0.0:
            segment_length = int(online_pcdt.hindsight_control_segment_length)
            batch_size = int(config.training.batch_size)
            if online_pcdt.control_objective != "lean_gcsl":
                raise ValueError(
                    "training.online_pcdt.segment_goal_contrast_weight > 0 "
                    "requires control_objective='lean_gcsl'."
                )
            if segment_length <= 1:
                raise ValueError(
                    "training.online_pcdt.segment_goal_contrast_weight > 0 "
                    "requires hindsight_control_segment_length > 1."
                )
            if batch_size % segment_length != 0:
                raise ValueError(
                    "training.batch_size must be divisible by "
                    "hindsight_control_segment_length when segment goal contrast is enabled."
                )
            if batch_size // segment_length < 2:
                raise ValueError(
                    "segment goal contrast requires at least two segments per actor batch."
                )
        if online_pcdt.mask_stalled_forward or online_pcdt.exclude_stalled_forward_control_anchors:
            if config.environment.kind != "jaxenstein":
                raise ValueError(
                    "stalled-forward handling currently requires environment.kind='jaxenstein'."
                )
            if (
                not config.observation.feature_sources
                or config.observation.feature_sources[0] != "current_position_xy"
            ):
                raise ValueError(
                    "stalled-forward handling requires "
                    "observation.feature_sources to start with 'current_position_xy'."
                )
        if online_pcdt.mask_stalled_forward:
            if int(online_pcdt.stalled_forward_turn_steps) >= int(online_pcdt.context_length):
                raise ValueError(
                    "training.online_pcdt.stalled_forward_turn_steps must be below "
                    "context_length so the causal history retains the collision transition."
                )
        if online_pcdt.control_objective == "lean_gcsl":
            if float(online_pcdt.hindsight_ratio) != 1.0:
                raise ValueError(
                    "training.online_pcdt.control_objective='lean_gcsl' requires "
                    "hindsight_ratio=1.0 so reward-bearing commanded transitions are never "
                    "used for policy updates."
                )
            if online_pcdt.encoder_hindsight_near_fraction != 0.0:
                raise ValueError(
                    "training.online_pcdt.control_objective='lean_gcsl' requires "
                    "encoder_hindsight_near_fraction=0.0 for full-range encoder goals."
                )
            if (
                float(online_pcdt.hindsight_min_goal_distance) <= 0.0
                and not online_pcdt.stationary_hindsight_ablation
            ):
                raise ValueError(
                    "training.online_pcdt.control_objective='lean_gcsl' requires a "
                    "positive hindsight_min_goal_distance so stationary discrete-action "
                    "transitions cannot become ambiguous XY-goal labels, unless the explicit "
                    "stationary_hindsight_ablation is enabled."
                )
            if (
                online_pcdt.stationary_hindsight_ablation
                and float(online_pcdt.hindsight_min_goal_distance) != 0.0
            ):
                raise ValueError(
                    "stationary_hindsight_ablation requires hindsight_min_goal_distance=0.0."
                )
            if not online_pcdt.direct_goal_conditioning:
                raise ValueError(
                    "training.online_pcdt.control_objective='lean_gcsl' requires "
                    "direct_goal_conditioning=true."
                )
            if not online_pcdt.goal_chain_on_success:
                raise ValueError(
                    "training.online_pcdt.control_objective='lean_gcsl' requires "
                    "goal_chain_on_success=true."
                )
            if not online_pcdt.use_evaluation_actor:
                raise ValueError(
                    "training.online_pcdt.control_objective='lean_gcsl' requires "
                    "use_evaluation_actor=true."
                )
        if online_pcdt.goal_chain_on_success:
            if config.goal_task.schedule not in {"cycle", "random"}:
                raise ValueError(
                    "training.online_pcdt.goal_chain_on_success requires "
                    "goal_task.schedule='cycle' or 'random'."
                )
            if len(config.goal_task.candidate_positions_xy) < 2:
                raise ValueError(
                    "training.online_pcdt.goal_chain_on_success requires at least two "
                    "candidate goal positions."
                )
        if config.goal_teacher is not None:
            raise ValueError("training.algorithm='online_pcdt' does not yet support goal_teacher.")
        if config.curriculum is not None:
            raise ValueError("training.algorithm='online_pcdt' does not yet support curriculum.")
        if config.preview.enabled:
            raise ValueError(
                "training.algorithm='online_pcdt' does not yet support training previews; "
                "set preview.enabled=false."
            )
        if config.training.final_eval_diagnostics:
            raise ValueError(
                "training.algorithm='online_pcdt' does not yet support final_eval_diagnostics."
            )
    elif online_pcdt is not None:
        raise ValueError("training.online_pcdt is set but training.algorithm is not 'online_pcdt'.")


def _validate_goal_teacher_config(
    config: DownstreamRunConfig,
    *,
    algorithm: str,
    her_algorithms: set[str],
    her_goal_representation: str,
) -> None:
    teacher = config.goal_teacher
    if teacher is None:
        return
    levels = [float(value) for value in teacher.max_spawn_distance_levels]
    if algorithm not in her_algorithms:
        raise ValueError("goal_teacher requires a HER training algorithm.")
    if her_goal_representation != "goal_place_code":
        raise ValueError("goal_teacher currently requires HER goal_place_code training.")
    if "synthetic_place_cells" not in config.observation.feature_sources:
        raise ValueError("goal_teacher requires synthetic_place_cells in the student observation.")
    if str(config.environment.kind).strip().lower() != "jaxenstein":
        raise ValueError("goal_teacher currently supports only environment.kind='jaxenstein'.")
    if int(config.training.n_envs) != 1:
        raise ValueError("goal_teacher currently requires training.n_envs=1.")
    if not bool(config.environment.env_kwargs.get("uniform_spawn", False)):
        raise ValueError("goal_teacher requires environment.env_kwargs.uniform_spawn=true.")
    if config.curriculum is not None:
        raise ValueError("goal_teacher may not be combined with the static curriculum config.")
    if str(config.goal_task.schedule).strip().lower() == "codebook":
        raise ValueError(
            "goal_teacher requires physical candidate goals and may not use "
            "goal_task.schedule='codebook'."
        )
    if not config.eval_candidate_positions_xy:
        raise ValueError(
            "goal_teacher requires eval_candidate_positions_xy for teacher-free "
            "balanced evaluation."
        )
    if not levels or any(value <= 0.0 for value in levels):
        raise ValueError("goal_teacher.max_spawn_distance_levels must contain positive values.")
    if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
        raise ValueError("goal_teacher.max_spawn_distance_levels must be strictly increasing.")
    if int(teacher.initial_level) >= len(levels):
        raise ValueError("goal_teacher.initial_level must index max_spawn_distance_levels.")
    if int(teacher.minimum_challenge_steps) >= int(config.environment.episode_length):
        raise ValueError("goal_teacher.minimum_challenge_steps must be below the episode horizon.")
    if int(teacher.maximum_challenge_steps) < int(teacher.minimum_challenge_steps):
        raise ValueError(
            "goal_teacher.maximum_challenge_steps must not be below minimum_challenge_steps."
        )
    if int(teacher.maximum_challenge_steps) > int(config.environment.episode_length):
        raise ValueError(
            "goal_teacher.maximum_challenge_steps "
            f"({int(teacher.maximum_challenge_steps)}) must not exceed "
            "environment.episode_length "
            f"({int(config.environment.episode_length)})."
        )
    if float(teacher.promote_too_easy_rate) + float(teacher.demote_too_hard_rate) <= 1.0:
        raise ValueError("goal_teacher promote and demote rates must sum to more than 1.")


def _validate_downstream_training_contract(
    config: DownstreamRunConfig,
    *,
    algorithm: str,
    sb3_her_algorithms: set[str],
    her_algorithms: set[str],
) -> None:
    uses_online_pcdt_code = "online_pcdt_code_xy_heading" in config.observation.feature_sources
    if uses_online_pcdt_code:
        if not str(config.models.online_pcdt_checkpoint_path).strip():
            raise ValueError(
                "feature source 'online_pcdt_code_xy_heading' requires "
                "models.online_pcdt_checkpoint_path."
            )
        if config.environment.kind not in {"jaxenstein", "miniworld"}:
            raise ValueError(
                "feature source 'online_pcdt_code_xy_heading' currently supports only "
                "JAXenstein and MiniWorld navigation environments."
            )
        if algorithm == "online_pcdt":
            raise ValueError(
                "training.algorithm='online_pcdt' cannot consume a frozen online PCDT code."
            )
        if algorithm in sb3_her_algorithms:
            raise ValueError(
                "HER cannot relabel goal-conditioned online PCDT codes that were computed for "
                "the original goal. Use PPO or non-HER DQN for the transfer baseline."
            )
        if int(config.training.n_envs) != 1:
            raise ValueError(
                "feature source 'online_pcdt_code_xy_heading' currently requires training.n_envs=1 "
                "because it maintains one causal history."
            )
    gradient_steps = config.training.gradient_steps
    if isinstance(gradient_steps, str):
        if gradient_steps.strip().lower() != "auto_default_utd":
            raise ValueError(
                "training.gradient_steps must be a positive integer or 'auto_default_utd'."
            )
    elif int(gradient_steps) < 1:
        raise ValueError(
            "training.gradient_steps must be a positive integer or 'auto_default_utd'."
        )
    raw_pixel_aux_sources = {
        "goal_xy",
        "goal_xy_scaled",
        "goal_xy_map01",
        "heading_sin_cos",
    }
    if config.observation.mode == "raw_pixels":
        unsupported_raw_sources = sorted(
            set(config.observation.feature_sources) - raw_pixel_aux_sources
        )
        if unsupported_raw_sources:
            raise ValueError(
                "raw_pixels mode may only define small auxiliary feature sources "
                f"{sorted(raw_pixel_aux_sources)!r}; got {unsupported_raw_sources!r}."
            )
    if config.observation.mode == "feature_vector" and not config.observation.feature_sources:
        raise ValueError("feature_vector mode requires at least one feature source.")
    if algorithm == "dqn_bootstrap" and config.observation.mode != "feature_vector":
        raise ValueError("training.algorithm='dqn_bootstrap' requires feature_vector observations.")
    if algorithm == "dqn_bootstrap" and config.training.replay_buffer_type != "nstep":
        raise ValueError("training.algorithm='dqn_bootstrap' requires replay_buffer_type='nstep'.")
    if algorithm == "dqn_success_replay" and config.observation.mode != "feature_vector":
        raise ValueError(
            "training.algorithm='dqn_success_replay' requires feature_vector observations."
        )
    if algorithm == "dqn_success_replay" and config.training.replay_buffer_type != "nstep":
        raise ValueError(
            "training.algorithm='dqn_success_replay' requires replay_buffer_type='nstep'."
        )
    if algorithm in her_algorithms and int(config.training.n_envs) > 1:
        raise ValueError(
            f"training.algorithm={config.training.algorithm!r} currently requires "
            "training.n_envs=1. Keep HER runs single-env until downstream grows an explicitly "
            "tested multi-env HER path."
        )


def _validate_downstream_place_code_sources(
    config: DownstreamRunConfig,
    *,
    algorithm: str,
    her_algorithms: set[str],
    her_goal_representation: str,
    her_uses_synthetic_place_codes: bool,
) -> list[str]:
    warnings: list[str] = []
    missing_place_code_sources = sorted(
        PLACE_CODE_FEATURE_SOURCES.intersection(config.observation.feature_sources)
        if not config.models.place_model_artifact_id
        else []
    )
    if missing_place_code_sources:
        raise ValueError(
            f"feature source {missing_place_code_sources[0]!r} "
            "requires models.place_model_artifact_id."
        )
    missing_stats_sources = missing_place_code_stats_sources(
        config.observation.feature_sources,
        stats_available=resolved_place_code_stats_path(config.models) is not None,
    )
    if missing_stats_sources:
        raise ValueError(
            f"feature source {missing_stats_sources[0]!r} requires models.place_code_stats_path."
        )
    head_norm_sources = sorted(
        source
        for source in config.observation.feature_sources
        if source in PLACE_CODE_SOURCES and PLACE_CODE_SOURCES[source].pre_scale == "head_row_norm"
    )
    if head_norm_sources and config.models.place_representation_source != "encoder.place_codes":
        raise ValueError(
            f"feature source {head_norm_sources[0]!r} uses head_row_norm pre-scaling, which reads "
            "the encoder code-head weights and requires "
            "models.place_representation_source='encoder.place_codes'; got "
            f"{config.models.place_representation_source!r}."
        )
    if (
        "goal_place_code" in config.observation.feature_sources
        and not config.models.place_model_artifact_id
    ):
        raise ValueError(
            "feature source 'goal_place_code' requires models.place_model_artifact_id."
        )
    if "goal_place_code" in config.observation.feature_sources and int(config.training.n_envs) > 1:
        raise ValueError("goal_place_code feature source currently requires training.n_envs=1.")
    if any(
        source in PLACE_CODE_FEATURE_SOURCES | {"ae_latent", "goal_place_code"}
        for source in config.observation.feature_sources
    ) or (
        algorithm in her_algorithms
        and her_goal_representation == "goal_place_code"
        and not her_uses_synthetic_place_codes
    ):
        normalized_vision_reference = (
            str(config.models.vision_encoder_artifact_id or "").strip().lower()
        )
        if not normalized_vision_reference and not config.models.place_model_artifact_id:
            warnings.append(
                "A learned visual feature source is configured without "
                "models.vision_encoder_artifact_id. This is valid only if the place model "
                "artifact already embeds raw-RGB support."
            )
        elif not normalized_vision_reference and config.models.place_model_artifact_id:
            warnings.append(
                "models.vision_encoder_artifact_id is empty. For latent-input place models, "
                "downstream will fail fast unless you set an exact vision-encoder artifact id, "
                "a tag reference, or 'auto' to infer the matching encoder from the place-model "
                "lineage."
            )
        elif normalized_vision_reference == "auto" and not config.models.place_model_artifact_id:
            warnings.append(
                "models.vision_encoder_artifact_id=auto requires "
                "models.place_model_artifact_id so downstream can infer the matching vision "
                "encoder from the place-model lineage."
            )
    return warnings


def _validate_downstream_goal_contract(
    config: DownstreamRunConfig,
    *,
    algorithm: str,
    sb3_her_algorithms: set[str],
    her_algorithms: set[str],
    her_goal_representation: str,
    her_uses_synthetic_place_codes: bool,
    uses_uniform_random_goals: bool,
) -> list[str]:
    warnings: list[str] = []
    fixed_candidate_goal_sources = _ALL_GOAL_SOURCES.intersection(
        config.observation.feature_sources
    )
    if (
        fixed_candidate_goal_sources
        and not config.goal_task.candidate_positions_xy
        and not uses_uniform_random_goals
    ):
        raise ValueError("Goal feature sources require goal_task.candidate_positions_xy.")
    if (
        "goal_rbf_code" in config.observation.feature_sources
        and not config.goal_task.candidate_positions_xy
    ):
        raise ValueError(
            "observation.feature_sources=['goal_rbf_code'] requires "
            "goal_task.candidate_positions_xy."
        )
    if algorithm in her_algorithms:
        if her_goal_representation not in {"goal_xy", "goal_place_code", "goal_grid_code"}:
            raise ValueError(
                f"training.her_goal_representation="
                f"{config.training.her_goal_representation!r} is unsupported."
            )
        requested_goal_sources = sorted(
            _ALL_GOAL_SOURCES.intersection(config.observation.feature_sources)
        )
        if requested_goal_sources:
            raise ValueError(
                f"training.algorithm={config.training.algorithm!r} may not combine HER with "
                f"goal-conditioned observation.feature_sources {requested_goal_sources!r}. "
                "HER relabels desired_goal and reward, but it does not rewrite goal features "
                "embedded in the base observation. Keep the base observation goal-independent "
                "and let the GoalEnv desired_goal/achieved_goal channels carry goal information."
            )
    uses_codebook_goals = str(config.goal_task.schedule).strip().lower() == "codebook"
    if uses_codebook_goals and not (
        algorithm in her_algorithms and her_goal_representation == "goal_place_code"
    ):
        raise ValueError(
            "goal_task.schedule='codebook' is only supported with a HER algorithm and "
            "training.her_goal_representation='goal_place_code'."
        )
    if (
        algorithm in her_algorithms
        and not config.goal_task.candidate_positions_xy
        and not uses_codebook_goals
    ):
        raise ValueError(
            f"training.algorithm={config.training.algorithm!r} requires "
            "goal_task.candidate_positions_xy."
        )
    _validate_goal_teacher_config(
        config,
        algorithm=algorithm,
        her_algorithms=her_algorithms,
        her_goal_representation=her_goal_representation,
    )
    uses_place_code_codebook = (
        algorithm in her_algorithms
        and her_goal_representation == "goal_place_code"
        and uses_codebook_goals
    )
    if uses_place_code_codebook:
        if not str(config.models.goal_codebook_path).strip():
            raise ValueError(
                "goal_task.schedule='codebook' with "
                "training.her_goal_representation='goal_place_code' requires "
                "models.goal_codebook_path."
            )
        if "synthetic_place_cells" not in config.observation.feature_sources:
            raise ValueError(
                "goal_task.schedule='codebook' requires observation.feature_sources to include "
                "'synthetic_place_cells' so the current-position code shares the codebook's basis."
            )
    if (
        algorithm in her_algorithms
        and her_goal_representation == "goal_place_code"
        and not config.models.place_model_artifact_id
        and not her_uses_synthetic_place_codes
    ):
        raise ValueError(
            f"training.algorithm={config.training.algorithm!r} with "
            "training.her_goal_representation='goal_place_code' requires "
            "models.place_model_artifact_id."
        )
    if (
        algorithm in her_algorithms
        and her_goal_representation == "goal_xy"
        and not bool(config.environment.env_kwargs.get("reward_on_goal", False))
    ):
        raise ValueError(
            f"training.algorithm={config.training.algorithm!r} with "
            "training.her_goal_representation='goal_xy' requires "
            "environment.env_kwargs.reward_on_goal=true so HER can reconstruct the task reward."
        )
    if algorithm in sb3_her_algorithms and int(config.training.learning_starts) <= int(
        config.environment.episode_length
    ):
        raise ValueError(
            f"training.algorithm={config.training.algorithm!r} requires training.learning_starts "
            f"({int(config.training.learning_starts)}) to be greater than "
            f"environment.episode_length ({int(config.environment.episode_length)}). SB3 HER "
            "cannot sample until at least one full episode has finished."
        )
    if (
        algorithm in her_algorithms
        and her_goal_representation == "goal_place_code"
        and bool(config.environment.env_kwargs.get("reward_on_goal", False))
    ):
        warnings.append(
            f"training.algorithm={config.training.algorithm!r} uses place-code HER rewards. "
            "environment.env_kwargs.reward_on_goal will be disabled at runtime inside the HER "
            "wrapper."
        )
    if (
        algorithm in her_algorithms
        and her_goal_representation == "goal_place_code"
        and bool(config.environment.env_kwargs.get("terminate_on_goal", False))
    ):
        warnings.append(
            f"training.algorithm={config.training.algorithm!r} uses place-code HER success "
            "checks. environment.env_kwargs.terminate_on_goal will be disabled at runtime inside "
            "the HER wrapper."
        )
    goal_metric = str(config.goal_code.distance_metric).strip().lower()
    if goal_metric not in {"l2", "cosine"}:
        raise ValueError(
            f"goal_code.distance_metric={config.goal_code.distance_metric!r} is unsupported."
        )
    if (
        config.preview.save_ae_reconstruction_gif
        and not str(config.models.vision_encoder_artifact_id or "").strip()
    ):
        warnings.append(
            "preview.save_ae_reconstruction_gif is enabled, but "
            "models.vision_encoder_artifact_id is empty. Downstream previews will include RGB "
            "and trajectory outputs only."
        )
    return warnings


def _validate_downstream_environment_contract(
    config: DownstreamRunConfig,
    *,
    algorithm: str,
    her_algorithms: set[str],
    her_goal_representation: str,
    uses_uniform_random_goals: bool,
) -> None:
    environment_kind = str(config.environment.kind).strip().lower()
    if uses_uniform_random_goals and environment_kind not in {"miniworld", "jaxenstein"}:
        raise ValueError(
            "goal_task.schedule='uniform_random' supports environment.kind 'miniworld' or "
            "'jaxenstein'."
        )


def _validate_downstream_curriculum(config: DownstreamRunConfig) -> list[str]:
    if config.curriculum is None:
        return []
    environment_kind = str(config.environment.kind).strip().lower()
    if environment_kind != "miniworld":
        raise ValueError(
            "downstream curriculum currently supports only environment.kind='miniworld'."
        )
    warnings: list[str] = []
    candidate_goal_count = len(config.goal_task.candidate_positions_xy)
    for phase in config.curriculum.spawn_schedule:
        if phase.spawn_region_xz is not None and len(phase.spawn_region_xz) != 4:
            raise ValueError(
                f"Curriculum phase '{phase.name}' spawn_region_xz must contain exactly 4 floats."
            )
        if phase.goal_index is not None and phase.goal_schedule == "uniform_random":
            raise ValueError(
                f"Curriculum phase '{phase.name}' may not combine goal_index with "
                "goal_schedule='uniform_random'."
            )
        if phase.goal_index is not None and candidate_goal_count == 0:
            raise ValueError(
                f"Curriculum phase '{phase.name}' sets goal_index, but "
                "goal_task.candidate_positions_xy is empty."
            )
        if phase.goal_index is not None and phase.goal_index >= candidate_goal_count:
            raise ValueError(
                f"Curriculum phase '{phase.name}' goal_index={phase.goal_index} exceeds the "
                f"configured candidate goal count {candidate_goal_count}."
            )
        if phase.goal_schedule in {"fixed", "cycle", "random"} and candidate_goal_count == 0:
            raise ValueError(
                f"Curriculum phase '{phase.name}' overrides goal scheduling, but "
                "goal_task.candidate_positions_xy is empty."
            )
        if (
            phase.goal_schedule == "uniform_random"
            and "goal_rbf_code" in config.observation.feature_sources
        ):
            warnings.append(
                f"Curriculum phase '{phase.name}' uses goal_schedule='uniform_random' while "
                "observation.feature_sources includes 'goal_rbf_code'. The RBF code will still "
                "be computed, but it will encode distances from arbitrary runtime goals to the "
                "fixed candidate anchor set."
            )
        if (
            phase.goal_schedule == "uniform_random"
            and "goal_place_code" in config.observation.feature_sources
        ):
            warnings.append(
                f"Curriculum phase '{phase.name}' uses goal_schedule='uniform_random' while "
                "observation.feature_sources includes 'goal_place_code'. Downstream will snapshot "
                "and encode those runtime goals on demand instead of using the cached candidate "
                "goal codebook."
            )
    return warnings


def validate_study_config(config: StudyConfig) -> list[str]:
    """Raise on invalid study config."""
    config = _revalidate_config(config)
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


def validate_downstream_run_config(config: DownstreamRunConfig) -> list[str]:
    """Validate downstream RL single-run config."""
    config = _revalidate_config(config)
    warnings: list[str] = []
    algorithm = str(config.training.algorithm).strip().lower()
    sb3_her_algorithms = {"dqn_her", "qrdqn_her"}
    her_algorithms = sb3_her_algorithms | {"online_pcdt"}
    her_goal_representation = str(config.training.her_goal_representation).strip().lower()
    _validate_online_pcdt_training_config(
        config,
        algorithm=algorithm,
        her_goal_representation=her_goal_representation,
    )
    _validate_downstream_training_contract(
        config,
        algorithm=algorithm,
        sb3_her_algorithms=sb3_her_algorithms,
        her_algorithms=her_algorithms,
    )
    her_uses_synthetic_place_codes = (
        algorithm in her_algorithms
        and her_goal_representation == "goal_place_code"
        and "synthetic_place_cells" in config.observation.feature_sources
    )
    warnings.extend(
        _validate_downstream_place_code_sources(
            config,
            algorithm=algorithm,
            her_algorithms=her_algorithms,
            her_goal_representation=her_goal_representation,
            her_uses_synthetic_place_codes=her_uses_synthetic_place_codes,
        )
    )
    uses_uniform_random_goals = str(config.goal_task.schedule).strip().lower() == "uniform_random"
    warnings.extend(
        _validate_downstream_goal_contract(
            config,
            algorithm=algorithm,
            sb3_her_algorithms=sb3_her_algorithms,
            her_algorithms=her_algorithms,
            her_goal_representation=her_goal_representation,
            her_uses_synthetic_place_codes=her_uses_synthetic_place_codes,
            uses_uniform_random_goals=uses_uniform_random_goals,
        )
    )
    _validate_downstream_environment_contract(
        config,
        algorithm=algorithm,
        her_algorithms=her_algorithms,
        her_goal_representation=her_goal_representation,
        uses_uniform_random_goals=uses_uniform_random_goals,
    )
    warnings.extend(_validate_downstream_curriculum(config))
    return warnings
