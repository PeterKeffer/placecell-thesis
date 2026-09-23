"""Study stage."""

from __future__ import annotations

import json
from pathlib import Path

from placecell_research.artifacts.ids import config_fingerprint, generate_artifact_id
from placecell_research.artifacts.manifests import ArtifactManifest, CreatedBy
from placecell_research.artifacts.registry import ArtifactRegistry
from placecell_research.config import (
    load_experiment_config,
    load_study_config,
    validate_study_config,
)
from placecell_research.config.diff import build_comparison_card
from placecell_research.stages import (
    analyze_model,
    collect_dataset,
    create_split,
    encode_dataset,
    train_place_model,
    train_vision_encoder,
)
from placecell_research.stages.pipeline import _inject_automatic_reuse_overrides
from placecell_research.studies.curriculum import CurriculumStageRunners, run_curriculum
from placecell_research.studies.summary import write_study_summary, write_sweep_comparison_links
from placecell_research.studies.sweep import run_sweep
from placecell_research.tracking import (
    WandbLogger,
    build_wandb_run_config,
    emit_metrics_block,
    merge_tags,
    study_tags,
)
from placecell_research.tracking.naming import (
    RunIdentity,
    capture_git_state,
    generate_variant_slug,
    make_run_id,
)
from placecell_research.tracking.run_directory import RunDirectory
from placecell_research.utils.environment_info import capture_environment_info
from placecell_research.utils.repo_paths import find_repo_root, resolve_experiment_config_path
from placecell_research.utils.timing import utc_now_iso


def _study_tracking_overrides(study_config) -> list[str]:
    return [
        f"tracking.use_wandb={'true' if study_config.tracking.use_wandb else 'false'}",
        f"tracking.wandb_mode={study_config.tracking.wandb_mode}",
        f"tracking.wandb_failure_mode={study_config.tracking.wandb_failure_mode}",
        f"tracking.wandb_project={study_config.tracking.wandb_project}",
        f"tracking.study_name={study_config.name}",
        f"tracking.run_root={study_config.tracking.run_root}",
        f"tracking.artifact_root={study_config.tracking.artifact_root}",
    ]


def run(config_path: Path, overrides: list[str]) -> dict[str, object]:
    study_config = load_study_config(config_path, overrides)
    validate_study_config(study_config)

    repo_root = find_repo_root(config_path)
    variant_slug = generate_variant_slug(
        study_config.to_dict(),
        fallback_name=study_config.name,
    )
    run_identity = RunIdentity(
        run_id=make_run_id(repo_root, descriptor=variant_slug),
        study_name=study_config.tracking.study_name,
        variant_name=study_config.name,
        variant_slug=variant_slug,
        signature=f"study__{study_config.name}",
    )
    run_directory = RunDirectory(repo_root / study_config.tracking.run_root, run_identity)
    run_directory.create()
    git_state = capture_git_state(repo_root)
    run_directory.write_run_manifest(
        {
            "stage_name": "run_study",
            "run_id": run_identity.run_id,
            "created_at": utc_now_iso(),
            "variant_name": run_identity.variant_name,
            "variant_slug": run_identity.variant_slug,
            "status": "initialized",
            "git_state": git_state,
            "environment_info": capture_environment_info(),
        }
    )
    run_directory.write_comparison_card(
        {
            "variant_name": run_identity.variant_name,
            "variant_slug": run_identity.variant_slug,
            "signature": run_identity.signature,
            **build_comparison_card(study_config.to_dict()),
        }
    )

    logger = WandbLogger(
        enabled=bool(
            study_config.tracking.use_wandb
            and study_config.tracking.wandb_mode != "disabled"
        ),
        project=study_config.tracking.wandb_project,
        name=f"study__{study_config.name}__{run_identity.run_id}",
        group=f"study:{study_config.name}",
        job_type="run_study",
        tags=study_tags(
            study_config.name,
            mode="study",
            base_tags=merge_tags(study_config.tracking.tags, ["stage:run_study"]),
        ),
        config=build_wandb_run_config(
            study_config.to_dict(),
            identity=run_identity,
            stage_name="run_study",
            git_state=git_state,
        ),
        mode=study_config.tracking.wandb_mode,
        directory=run_directory.path / "wandb",
        failure_mode=study_config.tracking.wandb_failure_mode,
    )
    logger.start()
    stage_log_path = run_directory.logs_dir / "stage_run_study.log"

    registry = ArtifactRegistry(repo_root / study_config.tracking.artifact_root)
    comparison_link_root = (
        repo_root / study_config.tracking.run_root / "studies" / study_config.name
    )
    objective_metric = "status"
    objective_mode = "max"
    rows: list[dict[str, object]]
    try:
        logger.upload_files([run_directory.manifests_dir / "run_manifest.json"])
        logger.update_config(
            {
                "study": {
                    "name": study_config.name,
                    "mode": "sweep" if study_config.sweep is not None else "curriculum",
                }
            }
        )
        propagated_tracking_overrides = _study_tracking_overrides(study_config)
        if study_config.sweep is not None:
            experiment_path = resolve_experiment_config_path(
                config_path,
                study_config.sweep.base_experiment,
            )
            experiment_base_tags = load_experiment_config(experiment_path).tracking.tags
            comparison_parameter_keys = list(study_config.sweep.parameters)

            def run_train_trial(
                path: Path,
                stage_overrides: list[str],
                row: dict[str, object],
            ) -> dict[str, object]:
                def link_runtime(runtime) -> None:
                    write_sweep_comparison_links(
                        comparison_link_root,
                        [
                            {
                                **row,
                                "stage.run_id": runtime.run_directory.identity.run_id,
                                "stage.run_path": str(runtime.run_directory.path),
                            }
                        ],
                        comparison_parameter_keys,
                    )

                trial_overrides = [*propagated_tracking_overrides, *stage_overrides]
                _inject_automatic_reuse_overrides(
                    config_path=path,
                    active_overrides=trial_overrides,
                    registry=registry,
                )
                return train_place_model.run(
                    path,
                    trial_overrides,
                    on_runtime_created=link_runtime,
                )

            sweep_result = run_sweep(
                study_config.sweep,
                experiment_path,
                run_train_trial,
                base_tracking_tags=merge_tags(
                    experiment_base_tags,
                    study_config.tracking.tags,
                ),
                study_name=study_config.name,
                on_row=lambda row: write_sweep_comparison_links(
                    comparison_link_root,
                    [row],
                    comparison_parameter_keys,
                ),
            )
            rows = sweep_result.rows
            objective_metric = study_config.sweep.objective_metric
            objective_mode = "min" if study_config.sweep.direction == "minimize" else "max"
        elif study_config.curriculum is not None:
            experiment_path = resolve_experiment_config_path(
                config_path,
                study_config.curriculum.base_experiment,
            )
            experiment_base_tags = load_experiment_config(experiment_path).tracking.tags
            curriculum_result = run_curriculum(
                study_config.curriculum,
                experiment_path,
                CurriculumStageRunners(
                    collect_dataset=lambda path, stage_overrides: collect_dataset.run(
                        path,
                        [*propagated_tracking_overrides, *stage_overrides],
                    ),
                    train_model=lambda path, stage_overrides: train_place_model.run(
                        path,
                        [*propagated_tracking_overrides, *stage_overrides],
                    ),
                    analyze_model=lambda path, stage_overrides: analyze_model.run(
                        path,
                        [*propagated_tracking_overrides, *stage_overrides],
                    ),
                    create_split=lambda path, stage_overrides: create_split.run(
                        path,
                        [*propagated_tracking_overrides, *stage_overrides],
                    ),
                    train_vision_encoder=lambda path, stage_overrides: (
                        train_vision_encoder.run(
                            path,
                            [*propagated_tracking_overrides, *stage_overrides],
                        )
                    ),
                    encode_dataset=lambda path, stage_overrides: encode_dataset.run(
                        path,
                        [*propagated_tracking_overrides, *stage_overrides],
                    ),
                ),
                base_tracking_tags=merge_tags(
                    experiment_base_tags,
                    study_config.tracking.tags,
                ),
            )
            rows = curriculum_result.rows
        else:
            raise ValueError("Study config must define either sweep or curriculum.")

        artifact_id = generate_artifact_id("study_report", study_config.name, run_identity.run_id)
        report_path = registry.artifact_path("study_report", artifact_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.mkdir(parents=True, exist_ok=False)
        written_paths = write_study_summary(report_path, rows, objective_metric, objective_mode)
        comparison_parameter_keys = (
            list(study_config.sweep.parameters) if study_config.sweep is not None else []
        )
        comparison_links = write_sweep_comparison_links(
            comparison_link_root,
            rows,
            comparison_parameter_keys,
        )
        comparison_links_path = str(comparison_link_root) if comparison_links else ""
        if comparison_links:
            written_paths["comparison_links"] = comparison_link_root
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type="study_report",
            created_by=CreatedBy(run_id=run_identity.run_id, stage_name="run_study"),
            input_artifact_ids=sorted(
                {
                    str(value)
                    for row in rows
                    for key, value in row.items()
                    if key.endswith("_artifact_id") and value
                }
            ),
            config_fingerprint=config_fingerprint(str(study_config.to_dict())),
            git_commit=str(capture_git_state(repo_root).get("commit", "")),
            summary={
                "study_name": study_config.name,
                "row_count": len(rows),
                "objective_metric": objective_metric,
                "objective_mode": objective_mode,
            },
        )
        manifest.write(report_path / "manifest.json")
        run_directory.update_run_manifest(
            {
                "status": "completed",
                "produced_artifact_ids": [artifact_id],
                "summary": {
                    "study_report_id": artifact_id,
                    "study_report_path": str(report_path),
                    "row_count": len(rows),
                    "objective_metric": objective_metric,
                    "objective_mode": objective_mode,
                    "comparison_links_path": comparison_links_path,
                },
            }
        )
        run_directory.write_symlink("results/study_report", report_path)
        logger.set_summary(
            {
                "status": "completed",
                "study_report_id": artifact_id,
                "study_report_path": str(report_path),
                "row_count": len(rows),
                "objective_metric": objective_metric,
                "objective_mode": objective_mode,
                "comparison_links_path": comparison_links_path,
            }
        )
        emit_metrics_block(
            "run_study",
            {
                "study/row_count": len(rows),
            },
            metadata={
                "study_name": study_config.name,
                "objective_metric": objective_metric,
                "objective_mode": objective_mode,
                "report_id": artifact_id,
            },
            log_path=stage_log_path,
        )
        logger.upload_files(
            [
                report_path / "README.md",
                report_path / "study_index.csv",
                report_path / "summary_table.csv",
                report_path / "launched_runs.csv",
                report_path / "best_runs_by_metric.json",
            ]
        )
        if rows:
            logger.log({"study/row_count": len(rows)})
            best_row = json.loads((report_path / "best_runs_by_metric.json").read_text())
            numeric_best = {
                f"best/{key}": float(value)
                for key, value in best_row.items()
                if isinstance(value, (int, float))
            }
            if numeric_best:
                logger.log(numeric_best)
        return {
            "study_report_id": artifact_id,
            "study_report_path": str(report_path),
            **{key: str(value) for key, value in written_paths.items()},
        }
    except Exception as exc:
        logger.set_summary({"status": "failed", "error": str(exc)})
        raise
    finally:
        logger.finish()
