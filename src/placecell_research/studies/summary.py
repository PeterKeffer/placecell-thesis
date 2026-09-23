"""Study summary helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from placecell_research.analysis.helpers import write_csv


def _first_non_empty(row: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in {"", None}:
            return str(value)
    return ""


def _study_row_label(row: dict[str, object]) -> str:
    row_type = str(row.get("row_type", "unknown"))
    if row_type == "raw_dataset":
        return f"Collect raw dataset for source {_first_non_empty(row, 'source', 'dataset')}"
    if row_type == "vision_encoder":
        return "Train shared vision encoder"
    if row_type == "encoded_dataset":
        source = _first_non_empty(row, "dataset_alias", "source")
        return f"Encode source {source} with shared vision encoder"
    if row_type == "phase_train":
        return (
            f"Train place model in phase {_first_non_empty(row, 'phase')} "
            f"on {_first_non_empty(row, 'dataset', 'dataset_alias')}"
        )
    if row_type == "phase_analysis":
        return (
            f"Analyze phase {_first_non_empty(row, 'phase')} model "
            f"on {_first_non_empty(row, 'dataset', 'dataset_alias')}"
        )
    if row_type == "phase_comparative_analysis":
        return (
            f"Run comparative module {_first_non_empty(row, 'module')} "
            f"for phase {_first_non_empty(row, 'phase')}"
        )
    if row_type == "final_comparative_analysis":
        return f"Run final comparative module {_first_non_empty(row, 'module')} across checkpoints"
    return row_type.replace("_", " ")


def _study_index_row(row: dict[str, object]) -> dict[str, object]:
    analysis_report_path = _first_non_empty(row, "analysis_report_path")
    return {
        "row_label": _study_row_label(row),
        "row_type": _first_non_empty(row, "row_type"),
        "source": _first_non_empty(row, "source"),
        "phase": _first_non_empty(row, "phase"),
        "dataset_alias": _first_non_empty(row, "dataset", "dataset_alias"),
        "module": _first_non_empty(row, "module"),
        "dataset_artifact_id": _first_non_empty(row, "dataset_artifact_id", "dataset.artifact_id"),
        "split_artifact_id": _first_non_empty(row, "split_artifact_id"),
        "vision_artifact_id": _first_non_empty(row, "vision_artifact_id", "vision.artifact_id"),
        "place_model_artifact_id": _first_non_empty(
            row,
            "place_model_artifact_id",
            "train_place_model.model_artifact_id",
            "analysis.model_artifact_id",
            "evaluation.model_artifact_id",
        ),
        "analysis_report_id": _first_non_empty(row, "analysis_report_id"),
        "analysis_report_path": analysis_report_path,
        "analysis_figures_path": (
            str(Path(analysis_report_path) / "figures") if analysis_report_path else ""
        ),
        "analysis_tables_path": (
            str(Path(analysis_report_path) / "tables") if analysis_report_path else ""
        ),
        "checkpoint_path": _first_non_empty(row, "checkpoint_path"),
        "stage_run_id": _first_non_empty(row, "stage.run_id"),
        "stage_run_path": _first_non_empty(row, "stage.run_path"),
    }


def _write_study_readme(output_dir: Path) -> Path:
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Study Report",
                "",
                "- `study_index.csv`: human-readable map of what each row/report represents.",
                "- `summary_table.csv`: full machine-readable table with every metric "
                "and artifact field.",
                "- `best_runs_by_metric.json`: best row under the configured study objective.",
                "- `launched_runs.csv`: raw row dump for all launched study steps.",
                "",
                "Common row types:",
                "",
                "- `raw_dataset`: collected source dataset",
                "- `vision_encoder`: shared autoencoder training result",
                "- `encoded_dataset`: encoded dataset produced from one source",
                "- `phase_train`: one place-model training phase",
                "- `phase_analysis`: analysis of one phase checkpoint on one dataset",
                "- `phase_comparative_analysis`: comparative module for one phase",
                "- `final_comparative_analysis`: comparative module across phase checkpoints",
                "",
                "For figure-heavy rows, start with `by_source/`, `by_phase/`, "
                "`shared/`, and `final/`.",
                "Use `study_index.csv` when you need the exact artifact ids and "
                "report paths behind those grouped links.",
                "",
            ]
        )
        + "\n"
    )
    return readme_path


def _slug(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "_" for character in value.strip())
    return normalized.strip("_") or "unnamed"


def _write_relative_symlink(link_path: Path, target_path: Path) -> None:
    if not target_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    relative_target = Path(os.path.relpath(target_path.resolve(), start=link_path.parent.resolve()))
    link_path.symlink_to(relative_target, target_is_directory=target_path.is_dir())


def _replace_relative_symlink(link_path: Path, target_path: Path) -> None:
    if not target_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Cannot replace non-symlink comparison path: {link_path}")
    relative_target = Path(os.path.relpath(target_path.resolve(), start=link_path.parent.resolve()))
    link_path.symlink_to(relative_target, target_is_directory=target_path.is_dir())


def _parameter_folder_name(parameter_key: str, value: object) -> str:
    key_parts = parameter_key.split(".")
    if key_parts[:1] == ["spatial_model"]:
        key_parts = key_parts[1:]
    key_token = _slug("_".join(reversed(key_parts)))
    if isinstance(value, list):
        value_token = "_".join(_slug(str(item)) for item in value)
    else:
        value_token = _slug(str(value))
    return f"{key_token}-{value_token}"


def _sweep_comparison_link(
    row: dict[str, object],
    output_dir: Path,
    parameter_keys: list[str],
) -> Path | None:
    run_path_text = _first_non_empty(row, "stage.run_path")
    if not run_path_text:
        return None

    seed = _slug(_first_non_empty(row, "seed") or "unknown")
    path = output_dir
    for key in parameter_keys:
        if key not in row:
            continue
        path /= _parameter_folder_name(key, row[key])
    return path / f"seed-{seed}"


def write_sweep_comparison_links(
    output_dir: Path,
    rows: list[dict[str, object]],
    parameter_keys: list[str],
) -> set[Path]:
    """Write human-readable sweep aliases to canonical run directories."""
    written: set[Path] = set()
    for row in rows:
        link_path = _sweep_comparison_link(row, output_dir, parameter_keys)
        if link_path is None:
            continue
        target_path = Path(_first_non_empty(row, "stage.run_path"))
        _replace_relative_symlink(link_path, target_path)
        if link_path.is_symlink():
            written.add(link_path)
    return written


def _link_target_from_row(row: dict[str, object]) -> Path | None:
    analysis_report_path = _first_non_empty(row, "analysis_report_path")
    if analysis_report_path:
        return Path(analysis_report_path)
    checkpoint_path = _first_non_empty(row, "checkpoint_path")
    if checkpoint_path:
        return Path(checkpoint_path)
    stage_run_path = _first_non_empty(row, "stage.run_path")
    if stage_run_path:
        return Path(stage_run_path)
    return None


def _write_grouped_links(output_dir: Path, rows: list[dict[str, object]]) -> set[Path]:
    written_roots: set[Path] = set()
    for row in rows:
        row_type = _first_non_empty(row, "row_type")
        target_path = _link_target_from_row(row)
        if target_path is None:
            continue
        if row_type == "raw_dataset":
            source = _slug(_first_non_empty(row, "source"))
            artifact_id = _slug(
                _first_non_empty(row, "dataset_artifact_id", "dataset.artifact_id")
            )
            _write_relative_symlink(
                output_dir / "by_source" / f"{source}__raw_dataset__{artifact_id}",
                target_path,
            )
            written_roots.add(output_dir / "by_source")
        elif row_type == "encoded_dataset":
            dataset_alias = _slug(_first_non_empty(row, "dataset_alias", "dataset"))
            artifact_id = _slug(_first_non_empty(row, "dataset_artifact_id", "dataset.artifact_id"))
            _write_relative_symlink(
                output_dir / "by_source" / f"{dataset_alias}__encoded_dataset__{artifact_id}",
                target_path,
            )
            written_roots.add(output_dir / "by_source")
        elif row_type == "vision_encoder":
            artifact_id = _slug(_first_non_empty(row, "vision_artifact_id", "vision.artifact_id"))
            _write_relative_symlink(
                output_dir / "shared" / f"vision_encoder__{artifact_id}",
                target_path,
            )
            written_roots.add(output_dir / "shared")
        elif row_type == "phase_train":
            phase = _slug(_first_non_empty(row, "phase"))
            dataset_alias = _slug(_first_non_empty(row, "dataset", "dataset_alias"))
            _write_relative_symlink(
                output_dir / "by_phase" / f"{phase}__train_on__{dataset_alias}",
                target_path,
            )
            written_roots.add(output_dir / "by_phase")
        elif row_type == "phase_analysis":
            phase = _slug(_first_non_empty(row, "phase"))
            dataset_alias = _slug(_first_non_empty(row, "dataset", "dataset_alias"))
            report_id = _slug(_first_non_empty(row, "analysis_report_id"))
            _write_relative_symlink(
                output_dir / "by_phase" / f"{phase}__analysis_on__{dataset_alias}__{report_id}",
                target_path,
            )
            written_roots.add(output_dir / "by_phase")
        elif row_type == "phase_comparative_analysis":
            phase = _slug(_first_non_empty(row, "phase"))
            module = _slug(_first_non_empty(row, "module"))
            report_id = _slug(_first_non_empty(row, "analysis_report_id"))
            _write_relative_symlink(
                output_dir / "by_phase" / f"{phase}__comparative__{module}__{report_id}",
                target_path,
            )
            written_roots.add(output_dir / "by_phase")
        elif row_type == "final_comparative_analysis":
            module = _slug(_first_non_empty(row, "module"))
            report_id = _slug(_first_non_empty(row, "analysis_report_id"))
            _write_relative_symlink(output_dir / "final" / f"{module}__{report_id}", target_path)
            written_roots.add(output_dir / "final")
    return written_roots


def write_study_summary(
    output_dir: Path,
    rows: list[dict[str, object]],
    objective_metric: str,
    objective_mode: str,
) -> dict[str, Path]:
    """Write canonical study summary artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    header = sorted({key for row in rows for key in row}) if rows else ["status"]
    if rows:
        summary_table = write_csv(
            output_dir / "summary_table.csv",
            header,
            [[row.get(column, "") for column in header] for row in rows],
        )
        reverse = objective_mode == "max"
        numeric_rows = []
        for row in rows:
            try:
                numeric_rows.append((float(row[objective_metric]), row))
            except (KeyError, TypeError, ValueError):
                continue
        best_row = (
            sorted(numeric_rows, key=lambda item: item[0], reverse=reverse)[0][1]
            if numeric_rows else {}
        )
        best_json = output_dir / "best_runs_by_metric.json"
        best_json.write_text(json.dumps(best_row, indent=2, sort_keys=True) + "\n")
    else:
        summary_table = write_csv(output_dir / "summary_table.csv", ["status"], [["no_runs"]])
        best_json = output_dir / "best_runs_by_metric.json"
        best_json.write_text("{}\n")
    index_header = [
        "row_label",
        "row_type",
        "source",
        "phase",
        "dataset_alias",
        "module",
        "dataset_artifact_id",
        "split_artifact_id",
        "vision_artifact_id",
        "place_model_artifact_id",
        "analysis_report_id",
        "analysis_report_path",
        "analysis_figures_path",
        "analysis_tables_path",
        "checkpoint_path",
        "stage_run_id",
        "stage_run_path",
    ]
    study_index = write_csv(
        output_dir / "study_index.csv",
        index_header,
        [[_study_index_row(row).get(column, "") for column in index_header] for row in rows]
        if rows
        else [["no_rows", *[""] * (len(index_header) - 1)]],
    )
    launched_runs = write_csv(
        output_dir / "launched_runs.csv",
        header,
        [[row.get(column, "") for column in header] for row in rows] if rows else [["no_runs"]],
    )
    readme = _write_study_readme(output_dir)
    grouped_roots = _write_grouped_links(output_dir, rows)
    written_paths: dict[str, Path] = {
        "summary_table": summary_table,
        "study_index": study_index,
        "best_runs": best_json,
        "launched_runs": launched_runs,
        "readme": readme,
    }
    for key, path in (
        ("by_source", output_dir / "by_source"),
        ("by_phase", output_dir / "by_phase"),
        ("shared", output_dir / "shared"),
        ("final", output_dir / "final"),
    ):
        if path in grouped_roots:
            written_paths[key] = path
    return written_paths
