"""Typed config system."""

from .downstream_schema import DownstreamRunConfig
from .loader import (
    load_downstream_run_config,
    load_experiment_config,
    materialize_dataclass,
)
from .reuse import (
    artifact_match_fingerprint,
    resolve_artifact_reference_id,
    resolve_matching_artifact,
    resolve_reuse_target,
    summarize_reuse,
)
from .schema import ExperimentConfig
from .validator import (
    validate_downstream_run_config,
    validate_experiment_config,
)

__all__ = [
    "DownstreamRunConfig",
    "ExperimentConfig",
    "load_downstream_run_config",
    "load_experiment_config",
    "materialize_dataclass",
    "artifact_match_fingerprint",
    "resolve_artifact_reference_id",
    "resolve_matching_artifact",
    "resolve_reuse_target",
    "summarize_reuse",
    "validate_downstream_run_config",
    "validate_experiment_config",
]
