"""Sweep execution."""

from __future__ import annotations

import gc
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from placecell_research.config.schema import SweepConfig
from placecell_research.tracking import sweep_tags


@dataclass(slots=True)
class SweepRunResult:
    """Sweep execution summary."""

    rows: list[dict[str, object]]


def _cleanup_after_trial() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _expand_grid(parameters: dict[str, object]) -> list[dict[str, object]]:
    keys = list(parameters)
    value_lists = [value if isinstance(value, list) else [value] for value in parameters.values()]
    return [dict(zip(keys, values, strict=False)) for values in product(*value_lists)]


def _expand_paired(parameters: dict[str, object]) -> list[dict[str, object]]:
    """Zip aligned parameter lists into explicitly paired sweep trials."""
    keys = list(parameters)
    value_lists = [value for value in parameters.values() if isinstance(value, list)]
    if len(value_lists) != len(keys):
        raise ValueError("Paired sweep parameters must all be lists.")
    lengths = {len(values) for values in value_lists}
    if len(lengths) != 1:
        raise ValueError("Paired sweep parameter lists must have equal lengths.")
    if not lengths or next(iter(lengths)) == 0:
        raise ValueError("Paired sweep parameter lists must not be empty.")
    return [dict(zip(keys, values, strict=False)) for values in zip(*value_lists, strict=False)]


def _serialize_override_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value)
    return str(value)


def seed_overrides(seed: int) -> list[str]:
    """Return all experiment seed overrides needed for one sweep trial."""
    return [
        f"seed.global_seed={seed}",
        f"seed.collection_seed={seed}",
        f"seed.split_seed={seed}",
        f"seed.training_seed={seed}",
        f"splits.seed={seed}",
        f"spatial_model.training.online.seed={seed}",
        f"analysis.example_episode_random_seed={seed}",
        f"analysis.umap_random_seed={seed}",
        f"analysis.probing_shuffle_seed={seed}",
        f"analysis.remapping_shuffle_seed={seed}",
    ]


def _format_sweep_values(row: dict[str, object], *, exclude_keys: set[str]) -> str:
    parts = [
        f"{key}={value}"
        for key, value in row.items()
        if key not in exclude_keys
    ]
    return " ".join(parts)


def run_sweep(
    sweep_config: SweepConfig,
    experiment_path: Path,
    runner: Callable[[Path, list[str], dict[str, object]], dict[str, object]],
    *,
    base_tracking_tags: list[str] | None = None,
    study_name: str | None = None,
    on_row: Callable[[dict[str, object]], None] | None = None,
) -> SweepRunResult:
    """Execute a study sweep through a caller-provided runner."""
    effective_study_name = study_name or sweep_config.base_experiment
    if sweep_config.method in {"grid", "paired"}:
        combinations = (
            _expand_grid(sweep_config.parameters)
            if sweep_config.method == "grid"
            else _expand_paired(sweep_config.parameters)
        )
        rows: list[dict[str, object]] = []
        trial_index = 0
        total_trials = len(sweep_config.seeds) * len(combinations)
        for seed in sweep_config.seeds:
            for combination in combinations:
                tags = sweep_tags(
                    effective_study_name,
                    trial_index,
                    seed,
                    base_tags=base_tracking_tags or [],
                )
                overrides = [
                    f"tracking.tags={json.dumps(tags)}",
                    *seed_overrides(int(seed)),
                ] + [
                    f"{key}={_serialize_override_value(value)}"
                    for key, value in combination.items()
                ]
                row = dict(combination)
                row["seed"] = seed
                row["trial_index"] = trial_index
                trial_number = trial_index + 1
                started_at = time.perf_counter()
                print(
                    "[sweep] starting "
                    f"{trial_number}/{total_trials}: trial_index={trial_index} "
                    f"seed={int(seed)} "
                    f"{_format_sweep_values(row, exclude_keys={'seed', 'trial_index'})}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    row.update(runner(experiment_path, overrides, row))
                finally:
                    _cleanup_after_trial()
                elapsed = time.perf_counter() - started_at
                objective_value = row.get(sweep_config.objective_metric)
                objective_text = (
                    ""
                    if objective_value is None
                    else f" {sweep_config.objective_metric}={objective_value}"
                )
                print(
                    "[sweep] finished "
                    f"{trial_number}/{total_trials}: trial_index={trial_index} "
                    f"seed={int(seed)}{objective_text} elapsed={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(row)
                if on_row is not None:
                    on_row(row)
                trial_index += 1
        return SweepRunResult(rows=rows)

    raise ValueError(f"Unsupported sweep method: {sweep_config.method}")
