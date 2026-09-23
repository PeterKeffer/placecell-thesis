"""Curriculum execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from placecell_research.config.schema import CurriculumConfig
from placecell_research.tracking import curriculum_tags, merge_tags, study_tags
from placecell_research.utils.repo_paths import find_repo_root


@dataclass(slots=True)
class CurriculumRunResult:
    """Curriculum execution summary."""

    rows: list[dict[str, object]]


@dataclass(slots=True)
class CurriculumStageRunners:
    """Stage functions used by the curriculum runner."""

    train_model: Callable[[Path, list[str]], dict[str, object]]
    analyze_model: Callable[[Path, list[str]], dict[str, object]]
    create_split: Callable[[Path, list[str]], dict[str, object]]
    collect_dataset: Callable[[Path, list[str]], dict[str, object]] | None = None
    train_vision_encoder: Callable[[Path, list[str]], dict[str, object]] | None = None
    encode_dataset: Callable[[Path, list[str]], dict[str, object]] | None = None


def _format_override_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def _flatten_overrides(prefix: str, payload: dict[str, object]) -> list[str]:
    overrides: list[str] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            overrides.extend(_flatten_overrides(path, value))
        else:
            overrides.append(f"{path}={_format_override_value(value)}")
    return overrides


def _resolve_dataset_alias(dataset_reference: str, dataset_aliases: dict[str, str]) -> str:
    return dataset_aliases.get(dataset_reference, dataset_reference)


def _resolve_source_alias(source_reference: str, source_aliases: dict[str, str]) -> str:
    return source_aliases.get(source_reference, source_reference)


def _tracking_tags_override(tags: list[str]) -> str:
    return f"tracking.tags={json.dumps(tags)}"


def _ensure_split_artifact(
    dataset_reference: str,
    dataset_aliases: dict[str, str],
    split_aliases: dict[str, str],
    experiment_path: Path,
    runners: CurriculumStageRunners,
) -> str:
    dataset_artifact_id = _resolve_dataset_alias(dataset_reference, dataset_aliases)
    if dataset_reference in split_aliases:
        return split_aliases[dataset_reference]
    split_result = runners.create_split(
        experiment_path,
        [
            f"dataset.artifact_id={dataset_artifact_id}",
            "dataset.artifact_type=encoded_dataset",
        ],
    )
    split_artifact_id = str(split_result["splits.artifact_id"])
    split_aliases[dataset_reference] = split_artifact_id
    split_aliases[dataset_artifact_id] = split_artifact_id
    return split_artifact_id


def _resolve_vision_payload(
    curriculum_config: CurriculumConfig, experiment_path: Path
) -> dict[str, object]:
    if not curriculum_config.vision_encoder:
        return {}
    payload = dict(curriculum_config.vision_encoder)
    config_name = str(payload.pop("config", "") or "")
    if config_name:
        repo_root = find_repo_root(experiment_path)
        config_path = repo_root / "configs" / "vision" / f"{config_name}.yaml"
        preset_payload = yaml.safe_load(config_path.read_text()) or {}
        if isinstance(preset_payload, dict) and set(preset_payload) == {"vision"}:
            preset_payload = preset_payload["vision"]
        if not isinstance(preset_payload, dict):
            raise TypeError(f"Vision preset must resolve to a mapping: {config_path}")
        payload = {**preset_payload, **payload}
    if "train_on" in payload:
        payload["datasets"] = [
            {"artifact_id": str(entry["dataset"])} for entry in payload.pop("train_on")
        ]
    payload.pop("freeze_after", None)
    return payload


def _collect_curriculum_sources(
    *,
    curriculum_config: CurriculumConfig,
    experiment_path: Path,
    runners: CurriculumStageRunners,
    study_tracking_tags: list[str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    source_aliases: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    if not curriculum_config.sources:
        return source_aliases, rows
    if runners.collect_dataset is None:
        raise ValueError(
            "Curriculum requested source collection but no collect_dataset runner was provided."
        )
    for source_name, source in curriculum_config.sources.items():
        if source.raw_dataset:
            artifact_id = str(source.raw_dataset)
            rows.append(
                {
                    "row_type": "raw_dataset",
                    "source": source_name,
                    "dataset_artifact_id": artifact_id,
                    "source_mode": "reuse_existing_artifact",
                }
            )
        else:
            collect_result = runners.collect_dataset(
                experiment_path,
                [
                    _tracking_tags_override(
                        merge_tags(
                            study_tracking_tags, [f"source:{source_name}", "role:raw_dataset"]
                        )
                    ),
                    *_flatten_overrides("environment", source.environment),
                    *_flatten_overrides("collection", source.collection),
                    "policies.artifact_reuse=reuse_if_config_match",
                ],
            )
            artifact_id = str(collect_result["dataset.artifact_id"])
            rows.append(
                {
                    "row_type": "raw_dataset",
                    "source": source_name,
                    "dataset_artifact_id": artifact_id,
                    **collect_result,
                }
            )
        source_aliases[source_name] = artifact_id
        source_aliases[artifact_id] = artifact_id
    return source_aliases, rows


def _run_phase_analyses(
    *,
    study_name: str,
    phase_index: int,
    phase_name: str,
    phase_model_artifact_id: str,
    datasets: list[str],
    modules: list[str],
    comparative_modules: list[str],
    dataset_aliases: dict[str, str],
    split_aliases: dict[str, str],
    experiment_path: Path,
    runners: CurriculumStageRunners,
    base_tracking_tags: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_reference in datasets:
        dataset_artifact_id = _resolve_dataset_alias(dataset_reference, dataset_aliases)
        split_artifact_id = _ensure_split_artifact(
            dataset_reference,
            dataset_aliases,
            split_aliases,
            experiment_path,
            runners,
        )
        phase_tags = curriculum_tags(
            study_name,
            phase_name,
            phase_index,
            dataset_reference,
            base_tags=base_tracking_tags,
        )
        analysis_result = runners.analyze_model(
            experiment_path,
            [
                _tracking_tags_override(merge_tags(phase_tags, ["role:analysis"])),
                f"analysis.model_artifact_id={phase_model_artifact_id}",
                f"analysis.dataset_artifact_id={dataset_artifact_id}",
                "analysis.dataset_artifact_type=encoded_dataset",
                f"analysis.split_artifact_id={split_artifact_id}",
                "analysis.split_name=test",
                "analysis.targets.curriculum_target.source=encoder.place_codes",
                f"analysis.targets.curriculum_target.modules={json.dumps(modules)}",
                "analysis.targets.curriculum_target.enabled=true",
                "analysis.comparative={}",
            ],
        )
        rows.append(
            {
                "row_type": "phase_analysis",
                "phase": phase_name,
                "dataset": dataset_reference,
                **analysis_result,
            }
        )

    if comparative_modules and len(datasets) >= 2:
        injected_inputs = []
        for dataset_reference in datasets:
            dataset_artifact_id = _resolve_dataset_alias(dataset_reference, dataset_aliases)
            split_artifact_id = _ensure_split_artifact(
                dataset_reference,
                dataset_aliases,
                split_aliases,
                experiment_path,
                runners,
            )
            injected_inputs.append(
                {
                    "label": str(dataset_reference),
                    "model_artifact_id": phase_model_artifact_id,
                    "dataset_artifact_id": dataset_artifact_id,
                    "dataset_artifact_type": "encoded_dataset",
                    "split_artifact_id": split_artifact_id,
                    "split_name": "test",
                    "source": "encoder.place_codes",
                }
            )

        for comparative_module in comparative_modules:
            first_dataset_artifact_id = _resolve_dataset_alias(datasets[0], dataset_aliases)
            first_split_artifact_id = _ensure_split_artifact(
                datasets[0], dataset_aliases, split_aliases, experiment_path, runners
            )
            analysis_result = runners.analyze_model(
                experiment_path,
                [
                    _tracking_tags_override(
                        merge_tags(
                            phase_tags,
                            [f"module:{comparative_module}", "role:comparative_analysis"],
                        )
                    ),
                    f"analysis.model_artifact_id={phase_model_artifact_id}",
                    f"analysis.dataset_artifact_id={first_dataset_artifact_id}",
                    "analysis.dataset_artifact_type=encoded_dataset",
                    f"analysis.split_artifact_id={first_split_artifact_id}",
                    "analysis.split_name=test",
                    "analysis.targets={}",
                    f"analysis.comparative.{comparative_module}.module={comparative_module}",
                    "analysis.comparative.{module}.enabled=true".replace(
                        "{module}", comparative_module
                    ),
                    "analysis.comparative.{module}.source=encoder.place_codes".replace(
                        "{module}", comparative_module
                    ),
                    f"analysis.comparative.{comparative_module}.inputs={json.dumps(injected_inputs)}",
                ],
            )
            rows.append(
                {
                    "row_type": "phase_comparative_analysis",
                    "phase": phase_name,
                    "module": comparative_module,
                    **analysis_result,
                }
            )
    return rows


def _run_final_analyses(
    *,
    curriculum_config: CurriculumConfig,
    phase_model_artifact_ids: list[str],
    dataset_aliases: dict[str, str],
    split_aliases: dict[str, str],
    experiment_path: Path,
    runners: CurriculumStageRunners,
    base_tracking_tags: list[str],
) -> list[dict[str, object]]:
    if not curriculum_config.final_analysis.modules:
        return []

    checkpoint_spec = curriculum_config.final_analysis.checkpoints
    if checkpoint_spec == "all_phase_checkpoints" or checkpoint_spec is None:
        checkpoint_ids = phase_model_artifact_ids
    elif isinstance(checkpoint_spec, list):
        checkpoint_ids = [str(value) for value in checkpoint_spec]
    else:
        checkpoint_ids = [str(checkpoint_spec)]

    comparative_inputs = []
    for checkpoint_id in checkpoint_ids:
        for dataset_reference in curriculum_config.final_analysis.datasets:
            dataset_artifact_id = _resolve_dataset_alias(dataset_reference, dataset_aliases)
            split_artifact_id = _ensure_split_artifact(
                dataset_reference,
                dataset_aliases,
                split_aliases,
                experiment_path,
                runners,
            )
            comparative_inputs.append(
                {
                    "label": f"{checkpoint_id}__{dataset_reference}",
                    "model_artifact_id": checkpoint_id,
                    "dataset_artifact_id": dataset_artifact_id,
                    "dataset_artifact_type": "encoded_dataset",
                    "split_artifact_id": split_artifact_id,
                    "split_name": "test",
                    "source": "encoder.place_codes",
                }
            )

    rows: list[dict[str, object]] = []
    for module_name in curriculum_config.final_analysis.modules:
        final_tags = curriculum_tags(
            curriculum_config.name,
            "final",
            len(phase_model_artifact_ids),
            curriculum_config.final_analysis.datasets[0],
            base_tags=base_tracking_tags,
        )
        final_dataset_reference = curriculum_config.final_analysis.datasets[0]
        final_dataset_artifact_id = _resolve_dataset_alias(final_dataset_reference, dataset_aliases)
        final_split_artifact_id = _ensure_split_artifact(
            final_dataset_reference, dataset_aliases, split_aliases, experiment_path, runners
        )
        analysis_result = runners.analyze_model(
            experiment_path,
            [
                _tracking_tags_override(
                    merge_tags(final_tags, [f"module:{module_name}", "role:final_analysis"])
                ),
                f"analysis.model_artifact_id={checkpoint_ids[0]}",
                f"analysis.dataset_artifact_id={final_dataset_artifact_id}",
                "analysis.dataset_artifact_type=encoded_dataset",
                f"analysis.split_artifact_id={final_split_artifact_id}",
                "analysis.split_name=test",
                "analysis.targets={}",
                f"analysis.comparative.{module_name}.module={module_name}",
                f"analysis.comparative.{module_name}.enabled=true",
                "analysis.comparative.{module}.source=encoder.place_codes".replace(
                    "{module}", module_name
                ),
                f"analysis.comparative.{module_name}.inputs={json.dumps(comparative_inputs)}",
            ],
        )
        rows.append(
            {
                "row_type": "final_comparative_analysis",
                "module": module_name,
                **analysis_result,
            }
        )
    return rows


def run_curriculum(
    curriculum_config: CurriculumConfig,
    experiment_path: Path,
    runners: CurriculumStageRunners,
    *,
    base_tracking_tags: list[str] | None = None,
) -> CurriculumRunResult:
    """Run curriculum phases sequentially through explicit stage runners."""
    rows: list[dict[str, object]] = []
    source_aliases: dict[str, str] = {}
    dataset_aliases: dict[str, str] = {}
    split_aliases: dict[str, str] = {}
    previous_model_artifact_id: str | None = None
    phase_model_artifact_ids: list[str] = []
    study_tracking_tags = study_tags(
        curriculum_config.name,
        mode="curriculum",
        base_tags=base_tracking_tags or [],
    )

    source_aliases, source_rows = _collect_curriculum_sources(
        curriculum_config=curriculum_config,
        experiment_path=experiment_path,
        runners=runners,
        study_tracking_tags=study_tracking_tags,
    )
    rows.extend(source_rows)

    if curriculum_config.vision_encoder:
        if runners.train_vision_encoder is None:
            raise ValueError(
                "Curriculum requested vision_encoder stage but no runner was provided."
            )
        vision_payload = _resolve_vision_payload(curriculum_config, experiment_path)
        if "datasets" in vision_payload:
            vision_payload["datasets"] = [
                {
                    **recipe,
                    "artifact_id": _resolve_source_alias(
                        str(recipe["artifact_id"]), source_aliases
                    ),
                }
                for recipe in vision_payload["datasets"]
            ]
        vision_result = runners.train_vision_encoder(
            experiment_path,
            [
                _tracking_tags_override(merge_tags(study_tracking_tags, ["role:vision_encoder"])),
                *_flatten_overrides("vision", vision_payload),
            ],
        )
        vision_artifact_id = str(vision_result["vision.artifact_id"])
        rows.append(
            {
                "row_type": "vision_encoder",
                "vision_artifact_id": vision_artifact_id,
                **vision_result,
            }
        )

        encode_each = list(curriculum_config.encoding.get("encode_each", []))
        if encode_each:
            if runners.encode_dataset is None:
                raise ValueError(
                    "Curriculum requested encoding but no encode_dataset runner was provided."
                )
            for entry in encode_each:
                alias = str(entry["alias"])
                source_artifact_id = _resolve_source_alias(str(entry["source"]), source_aliases)
                encode_result = runners.encode_dataset(
                    experiment_path,
                    [
                        _tracking_tags_override(
                            merge_tags(
                                study_tracking_tags, [f"dataset:{alias}", "role:encoded_dataset"]
                            )
                        ),
                        "policies.artifact_reuse=reuse_if_config_match",
                        f"dataset.artifact_id={source_artifact_id}",
                        "dataset.artifact_type=raw_dataset",
                        f"reuse.vision_encoder_artifact_id={vision_artifact_id}",
                    ],
                )
                encoded_artifact_id = str(encode_result["dataset.artifact_id"])
                dataset_aliases[alias] = encoded_artifact_id
                dataset_aliases[encoded_artifact_id] = encoded_artifact_id
                split_artifact_id = _ensure_split_artifact(
                    alias, dataset_aliases, split_aliases, experiment_path, runners
                )
                rows.append(
                    {
                        "row_type": "encoded_dataset",
                        "dataset_alias": alias,
                        "dataset_artifact_id": encoded_artifact_id,
                        "split_artifact_id": split_artifact_id,
                        **encode_result,
                    }
                )

    for phase_index, phase in enumerate(curriculum_config.phases):
        dataset_artifact_id = _resolve_dataset_alias(phase.dataset, dataset_aliases)
        split_artifact_id = _ensure_split_artifact(
            phase.dataset,
            dataset_aliases,
            split_aliases,
            experiment_path,
            runners,
        )
        phase_tags = curriculum_tags(
            curriculum_config.name,
            phase.name,
            phase_index,
            phase.dataset,
            base_tags=base_tracking_tags or [],
        )
        overrides = [
            _tracking_tags_override(merge_tags(phase_tags, ["role:train_model"])),
            f"dataset.artifact_id={dataset_artifact_id}",
            "dataset.artifact_type=encoded_dataset",
            f"splits.artifact_id={split_artifact_id}",
            f"spatial_model.training.epochs={phase.epochs}",
            f"policies.training_resume={phase.resume_policy}",
        ]
        if phase.resume_from == "previous":
            if previous_model_artifact_id is None:
                raise ValueError(
                    f"Curriculum phase '{phase.name}' requested resume_from=previous but no "
                    "previous model artifact exists."
                )
            overrides.append(f"reuse.place_model_artifact_id={previous_model_artifact_id}")

        result = runners.train_model(experiment_path, overrides)
        phase_model_artifact_id = str(
            result.get("place_model_artifact_id") or previous_model_artifact_id or ""
        )
        rows.append(
            {"row_type": "phase_train", "phase": phase.name, "dataset": phase.dataset, **result}
        )
        if not phase_model_artifact_id:
            raise ValueError(
                f"Curriculum phase '{phase.name}' did not produce a place_model_artifact_id."
            )
        previous_model_artifact_id = phase_model_artifact_id
        phase_model_artifact_ids.append(phase_model_artifact_id)

        rows.extend(
            _run_phase_analyses(
                study_name=curriculum_config.name,
                phase_index=phase_index,
                phase_name=phase.name,
                phase_model_artifact_id=phase_model_artifact_id,
                datasets=phase.analyze_after.datasets,
                modules=phase.analyze_after.modules,
                comparative_modules=phase.analyze_after.comparative_modules,
                dataset_aliases=dataset_aliases,
                split_aliases=split_aliases,
                experiment_path=experiment_path,
                runners=runners,
                base_tracking_tags=base_tracking_tags or [],
            )
        )

    rows.extend(
        _run_final_analyses(
            curriculum_config=curriculum_config,
            phase_model_artifact_ids=phase_model_artifact_ids,
            dataset_aliases=dataset_aliases,
            split_aliases=split_aliases,
            experiment_path=experiment_path,
            runners=runners,
            base_tracking_tags=base_tracking_tags or [],
        )
    )
    return CurriculumRunResult(rows=rows)
