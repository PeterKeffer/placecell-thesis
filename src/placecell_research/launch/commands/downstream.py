"""Downstream RL commands for the pc CLI."""

from __future__ import annotations

import typer
import yaml

from . import ConfigOption, OverrideOption


def register(app: typer.Typer) -> None:
    @app.command("downstream-rollout")
    def downstream_rollout_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Roll out forward or random actions in a navigation environment and report success."""
        from placecell_research.config import (
            load_downstream_run_config,
            validate_downstream_run_config,
        )
        from placecell_research.downstream.rollout import run_downstream_rollout
        from placecell_research.downstream.session import initialize_downstream_session

        downstream_config = load_downstream_run_config(
            config, override or []
        )
        validate_downstream_run_config(downstream_config)
        session = initialize_downstream_session(
            config_path=config,
            config_name=downstream_config.name,
            tracking=downstream_config.tracking,
            stage_name="downstream_rollout",
            overrides=override or [],
        )
        summary = run_downstream_rollout(
            repo_root=session.repo_root,
            config=downstream_config,
            output_dir=session.run_directory.results_dir / "downstream_rollout",
        )
        session.run_directory.update_run_manifest(
            {
                "status": "completed",
                "summary": {
                    "episodes": summary.episodes,
                    "mean_return": summary.mean_return,
                    "success_rate": summary.success_rate,
                    "mean_steps": summary.mean_steps,
                },
            }
        )
        typer.echo(
            yaml.safe_dump(
                {
                    "episodes": summary.episodes,
                    "mean_return": summary.mean_return,
                    "success_rate": summary.success_rate,
                    "mean_steps": summary.mean_steps,
                    "run_directory": str(session.run_directory.path),
                },
                sort_keys=False,
            )
        )

    @app.command("downstream-train")
    def downstream_train_command(
        config: ConfigOption,
        override: OverrideOption = None,
    ) -> None:
        """Train a navigation policy on the configured input."""
        from placecell_research.config import (
            load_downstream_run_config,
            validate_downstream_run_config,
        )
        from placecell_research.downstream.session import initialize_downstream_session
        from placecell_research.downstream.train import train_downstream_agent
        from placecell_research.tracking import managed_stage_run, stage_tags

        downstream_config = load_downstream_run_config(
            config, override or []
        )
        validate_downstream_run_config(downstream_config)
        session = initialize_downstream_session(
            config_path=config,
            config_name=downstream_config.name,
            tracking=downstream_config.tracking,
            stage_name="downstream_train",
            overrides=override or [],
        )
        output_dir = session.run_directory.results_dir / "downstream_train"
        try:
            with managed_stage_run(
                config=downstream_config,
                run_directory=session.run_directory,
                stage_name="downstream_train",
                run_name=f"{downstream_config.name}__{session.run_directory.identity.run_id}",
                tags=stage_tags(
                    "downstream_train",
                    downstream_config.environment.env_id,
                    base_tags=downstream_config.tracking.tags,
                    study_name=downstream_config.tracking.study_name,
                    variant_name=downstream_config.name,
                    variant_slug=session.run_directory.identity.variant_slug,
                    seed=downstream_config.seed,
                ),
            ) as stage_run:
                stage_run.define_metric("trainer/step")
                stage_run.define_metric("eval/*", step_metric="trainer/step")
                stage_run.define_metric("eval_stochastic/*", step_metric="trainer/step")
                stage_run.define_metric("eval_curriculum/*", step_metric="trainer/step")
                stage_run.define_metric("curriculum/phase_index", step_metric="trainer/step")
                stage_run.define_metric("train_episode/completed_episodes")
                stage_run.define_metric(
                    "train_episode/*", step_metric="train_episode/completed_episodes"
                )
                result = train_downstream_agent(
                    repo_root=session.repo_root,
                    config=downstream_config,
                    output_dir=output_dir,
                    metric_logger=lambda payload, step: stage_run.log(payload, step=step),
                )
                manifest_summary = {
                    "mean_return": result.final_metrics["mean_return"],
                    "success_rate": result.final_metrics["success_rate"],
                    "mean_time_to_goal": result.final_metrics["mean_time_to_goal"],
                    "metrics_path": str(result.metrics_path),
                }
                session.run_directory.update_run_manifest(
                    {
                        "status": result.status,
                        "summary": manifest_summary,
                    }
                )
                stage_run.finalize(
                    status=result.status,
                    summary=manifest_summary,
                    upload_files=[
                        result.metrics_path,
                        output_dir / "used_hyperparameters.yaml",
                        *([] if result.final_model_path is None else [result.final_model_path]),
                        *([] if result.best_model_path is None else [result.best_model_path]),
                        *(
                            []
                            if result.interrupted_model_path is None
                            else [result.interrupted_model_path]
                        ),
                    ],
                )
        except Exception as exc:
            session.run_directory.update_run_manifest(
                {
                    "status": "failed",
                    "summary": {"error": str(exc)},
                }
            )
            raise
        typer.echo(
            yaml.safe_dump(
                {
                    "status": result.status,
                    "mean_return": result.final_metrics["mean_return"],
                    "success_rate": result.final_metrics["success_rate"],
                    "mean_time_to_goal": result.final_metrics["mean_time_to_goal"],
                    "metrics_path": str(result.metrics_path),
                    "run_directory": str(session.run_directory.path),
                },
                sort_keys=False,
            )
        )
