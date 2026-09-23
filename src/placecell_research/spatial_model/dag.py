"""Pure config/topology helpers for the DAG-of-place-models (spatial_model.dag_nodes)."""

from __future__ import annotations

import copy

from placecell_research.config.schema import DagNodeConfig, SpatialModelConfig

from .representation_contract import base_representation_names

RESERVED_NODE_NAMES: frozenset[str] = frozenset(
    {
        "encoder",
        "predictor",
        "teacher",
        "grid",
        "jepa_predictor",
        "predictor_rollout",
        "top_down",
    }
)

VISION_LATENT_SOURCE = "vision_latent"

NODE_SEP = ":"


def prefix_node(name: str, dotted: str) -> str:
    """Namespace a dotted ref: 'encoder.place_codes' -> 'name:encoder.place_codes'."""
    module, dot, field = dotted.partition(".")
    return f"{name}{NODE_SEP}{module}{dot}{field}"


def resolve_node_config(node: DagNodeConfig) -> SpatialModelConfig:
    resolved = copy.deepcopy(node.model)
    resolved.dag_nodes = []
    resolved.inputs.observation_source = "latent"
    resolved.inputs.input_corruption.enabled = False
    resolved.objectives = {}
    return resolved


def namespaced_node_reps(node: DagNodeConfig) -> set[str]:
    """Namespaced reps a node produces: {'name:encoder.place_codes', ...}."""
    resolved = resolve_node_config(node)
    return {prefix_node(node.name, rep) for rep in base_representation_names(resolved)}


def root_producible(root_config: SpatialModelConfig) -> set[str]:
    """Root (base) un-namespaced reps, plus the vision-latent input source."""
    return set(base_representation_names(root_config)) | {VISION_LATENT_SOURCE}


def available_dag_representations(root_config: SpatialModelConfig) -> set[str]:
    available = set(base_representation_names(root_config))
    for node in root_config.dag_nodes:
        available |= namespaced_node_reps(node)
    return available


def topological_node_order(
    root_config: SpatialModelConfig, dag_nodes: list[DagNodeConfig]
) -> list[DagNodeConfig]:
    """Order nodes so every node's input_source is produced before it runs."""
    if not dag_nodes:
        return []
    seen: set[str] = set()
    for node in dag_nodes:
        if NODE_SEP in node.name or "." in node.name:
            raise ValueError(f"dag node name {node.name!r} must not contain '{NODE_SEP}' or '.'.")
        if node.name in RESERVED_NODE_NAMES:
            raise ValueError(f"dag node name {node.name!r} collides with a base module name.")
        if node.name in seen:
            raise ValueError(f"duplicate dag node name {node.name!r}.")
        seen.add(node.name)

    producible = root_producible(root_config)
    remaining = list(dag_nodes)
    ordered: list[DagNodeConfig] = []
    while remaining:
        ready = [node for node in remaining if node.input_source in producible]
        if not ready:
            unsatisfiable = {node.name: node.input_source for node in remaining}
            raise ValueError(
                "dag_nodes have unsatisfiable input_source references (cycle or dangling ref): "
                f"{unsatisfiable}. Producible sources: {sorted(producible)}"
            )
        for node in ready:
            ordered.append(node)
            producible |= namespaced_node_reps(node)
            remaining.remove(node)
    return ordered


def source_output_dim(
    input_source: str,
    *,
    vision_latent_dim: int,
    root_contract: dict,
    node_contracts: dict[str, dict],
) -> int:
    if input_source == VISION_LATENT_SOURCE:
        return int(vision_latent_dim)
    if NODE_SEP in input_source:
        node_name, _, local = input_source.partition(NODE_SEP)
        contract = node_contracts[node_name]
        key = local
    else:
        contract = root_contract
        key = input_source
    return int(contract["tensor_shapes"][key][-1])
