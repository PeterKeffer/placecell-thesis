"""Resolve objective targets to their owning model and forward-time bundle."""

from __future__ import annotations

from collections.abc import Sequence

from placecell_research.config.schema import SpatialModelConfig
from placecell_research.spatial_model.types import RepresentationBundle


def representation_parts(name: str) -> tuple[str, str, str]:
    """Return (DAG namespace, module, field) for a representation name."""

    module_name, separator, field_name = name.partition(".")
    if not separator:
        return "", module_name, ""
    namespace, namespace_separator, local_module = module_name.rpartition(":")
    if not namespace_separator:
        return "", module_name, field_name
    return namespace, local_module, field_name


def require_shared_namespace(
    representation_names: Sequence[str],
    *,
    objective_name: str,
) -> str:
    """Require all targets to belong to the same root or DAG-node model."""

    namespaces = {representation_parts(name)[0] for name in representation_names}
    if len(namespaces) != 1:
        raise ValueError(
            f"{objective_name} targets must belong to the same model namespace, got "
            f"{list(representation_names)!r}."
        )
    return namespaces.pop()


def model_config_for_representation(
    model_config: SpatialModelConfig,
    representation_name: str,
) -> SpatialModelConfig:
    """Resolve the model configuration that produces a representation."""

    namespace, _module_name, _field_name = representation_parts(representation_name)
    if not namespace:
        return model_config
    target_node = next(
        (node for node in model_config.dag_nodes if node.name == namespace),
        None,
    )
    if target_node is None:
        raise ValueError(
            f"Representation {representation_name!r} refers to unknown DAG node {namespace!r}."
        )
    return target_node.model


def runtime_representation(
    bundle: RepresentationBundle,
    representation_name: str,
    *,
    objective_name: str,
) -> tuple[RepresentationBundle, str]:
    """Resolve a target and its masks on the producer's native timeline."""

    namespace, module_name, field_name = representation_parts(representation_name)
    if not namespace:
        return bundle, representation_name
    node_bundles = bundle.metadata.get("dag_node_bundles")
    if not isinstance(node_bundles, dict) or namespace not in node_bundles:
        raise RuntimeError(
            f"{objective_name} could not find runtime bundle for DAG node {namespace!r}."
        )
    local_name = f"{module_name}.{field_name}" if field_name else module_name
    return node_bundles[namespace], local_name
