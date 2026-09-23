"""Write config snapshots into published artifact directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_artifact_config_snapshots(
    output_dir: Path,
    raw_config: dict[str, Any],
    active_config: dict[str, Any],
    *,
    stage_name: str,
    section_names: list[str],
    extra_payload: dict[str, Any] | None = None,
) -> None:
    """Write the configured payload and the active values of the stage's sections."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(raw_config, sort_keys=False))

    used_hyperparameters: dict[str, Any] = {
        "stage_name": stage_name,
        "selected_config_sections": {
            section_name: active_config[section_name]
            for section_name in section_names
            if section_name in active_config
        },
    }
    if extra_payload:
        used_hyperparameters["stage_context"] = extra_payload
    (output_dir / "used_hyperparameters.yaml").write_text(
        yaml.safe_dump(used_hyperparameters, sort_keys=False)
    )
