from __future__ import annotations

import json
from pathlib import Path

from placecell_research.analysis.base import AnalysisResult
from placecell_research.stages.analyze_model import (
    _analysis_output_group_name,
    _analysis_work_item_count,
    _analysis_work_item_count_with_coverage_splits,
    _comparative_input_payloads,
    _copy_analysis_outputs,
    _enabled_comparative_items,
    _grouped_output_file_name,
    _preserve_unfinished_analysis_workspace,
    _refresh_analysis_run_shortcuts,
    _set_live_analysis_staging_shortcut,
    _write_partial_analysis_snapshot,
)
from placecell_research.tracking.naming import RunIdentity
from placecell_research.tracking.run_directory import RunDirectory


def test_analysis_outputs_group_by_target_with_shortened_names(tmp_path: Path) -> None:
    source_figure = (
        tmp_path
        / "workspace"
        / "dataset_coverage"
        / "dataset_coverage__encoder.place_codes__validation.png"
    )
    source_figure.parent.mkdir(parents=True, exist_ok=True)
    source_figure.write_bytes(b"figure")

    grouped_results = {
        "encoder_place_cells.dataset_coverage": AnalysisResult(
            metrics={"occupied_bins_fraction": 0.5},
            per_unit_metrics={},
            figures={"dataset_coverage": source_figure},
            tables={},
        )
    }

    output_dir = tmp_path / "report"
    _copy_analysis_outputs(output_dir, grouped_results)

    assert (
        output_dir / "figures" / "encoder_place_cells" / "dataset_coverage__validation.png"
    ).exists()
    assert not (
        output_dir / "figures" / "dataset_coverage__encoder.place_codes__validation.png"
    ).exists()


def test_declared_figure_destinations_are_honored_verbatim(tmp_path: Path) -> None:
    per_unit_figure = (
        tmp_path
        / "workspace"
        / "rate_map_bundle"
        / "gallery"
        / "unit_0007__encoder.place_codes__validation.png"
    )
    plain_figure = (
        tmp_path
        / "workspace"
        / "rate_map_bundle"
        / "rate_map_panel__encoder.place_codes__validation.png"
    )
    for source_figure in (per_unit_figure, plain_figure):
        source_figure.parent.mkdir(parents=True, exist_ok=True)
        source_figure.write_bytes(b"figure")

    grouped_results = {
        "encoder_place_cells.rate_map_bundle": AnalysisResult(
            metrics={},
            per_unit_metrics={},
            figures={"rate_map_unit_0007": per_unit_figure, "rate_map_panel": plain_figure},
            tables={},
            figure_destinations={"rate_map_unit_0007": Path("gallery") / per_unit_figure.name},
        )
    }

    output_dir = tmp_path / "report"
    _copy_analysis_outputs(output_dir, grouped_results)

    group_dir = output_dir / "figures" / "encoder_place_cells"
    assert (group_dir / "gallery" / per_unit_figure.name).exists()
    assert (group_dir / "rate_map_panel__validation.png").exists()


def test_partial_analysis_snapshot_keeps_completed_outputs(tmp_path: Path) -> None:
    source_figure = (
        tmp_path
        / "workspace"
        / "rate_map_panel"
        / "rate_map_panel__encoder.place_codes__validation.png"
    )
    source_figure.parent.mkdir(parents=True, exist_ok=True)
    source_figure.write_bytes(b"figure")

    result = AnalysisResult(
        metrics={"mean_rate": 1.25},
        per_unit_metrics={},
        figures={"rate_map_panel": source_figure},
        tables={},
    )
    single_results = {"encoder_place_cells.rate_map_panel": result}
    output_dir = tmp_path / "partial_analysis"

    _write_partial_analysis_snapshot(
        output_dir,
        single_results=single_results,
        comparative_results={},
        completed_results=single_results,
    )

    assert (
        output_dir / "figures" / "encoder_place_cells" / "rate_map_panel__validation.png"
    ).exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "metrics.csv").exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["encoder_place_cells.rate_map_panel.mean_rate"] == 1.25


def test_unfinished_analysis_workspace_is_preserved_with_marker(tmp_path: Path) -> None:
    analysis_workspace = tmp_path / ".staging" / "analysis_stage_abc"
    staged_figure = analysis_workspace / "encoder.place_codes" / "rate_map.png"
    staged_figure.parent.mkdir(parents=True)
    staged_figure.write_bytes(b"partial figure")
    output_dir = tmp_path / "partial_analysis"

    _preserve_unfinished_analysis_workspace(
        output_dir,
        analysis_workspace,
        reason="KeyboardInterrupt",
    )

    preserved_root = output_dir / "unfinished_workspace"
    assert (
        preserved_root / "encoder.place_codes" / "rate_map.png"
    ).read_bytes() == b"partial figure"
    marker_text = (preserved_root / "UNFINISHED.md").read_text()
    assert "KeyboardInterrupt" in marker_text
    assert str(analysis_workspace) in marker_text


def test_analysis_run_shortcuts_expose_partial_figures_and_staging_workspace(
    tmp_path: Path,
) -> None:
    run_directory = RunDirectory(
        tmp_path / "runs",
        RunIdentity(
            run_id="analysis_run",
            study_name="study",
            variant_name="variant",
            variant_slug="env-wallgap__seed-7",
            signature="signature",
        ),
    )
    run_directory.create()
    partial_analysis_dir = run_directory.results_dir / "partial_analysis"
    figures_dir = partial_analysis_dir / "figures"
    staging_workspace = partial_analysis_dir / "unfinished_workspace"
    (figures_dir / "encoder_place_cells").mkdir(parents=True)
    (figures_dir / "encoder_place_cells" / "rate_map_panel.png").write_bytes(b"figure")
    staging_workspace.mkdir(parents=True)
    (staging_workspace / "raw_staged_panel.png").write_bytes(b"staged figure")

    _refresh_analysis_run_shortcuts(run_directory, partial_analysis_dir)

    assert (run_directory.path / "figures").is_symlink()
    assert (run_directory.path / "figures").resolve() == figures_dir.resolve()
    assert (run_directory.path / "staging_workspace").is_symlink()
    assert (run_directory.path / "staging_workspace").resolve() == staging_workspace.resolve()


def test_analysis_run_shortcuts_expose_live_staging_workspace(tmp_path: Path) -> None:
    run_directory = RunDirectory(
        tmp_path / "runs",
        RunIdentity(
            run_id="analysis_run",
            study_name="study",
            variant_name="variant",
            variant_slug="env-wallgap__seed-7",
            signature="signature",
        ),
    )
    run_directory.create()
    live_staging_workspace = tmp_path / "artifacts" / ".staging" / "analysis_stage_live"
    live_staging_workspace.mkdir(parents=True)
    (live_staging_workspace / "encoder.place_codes").mkdir()

    _set_live_analysis_staging_shortcut(run_directory, live_staging_workspace)

    assert (run_directory.path / "staging_workspace").is_symlink()
    assert (run_directory.path / "staging_workspace").resolve() == live_staging_workspace.resolve()


def test_analysis_run_shortcuts_remove_stale_live_staging_link_after_completion(
    tmp_path: Path,
) -> None:
    run_directory = RunDirectory(
        tmp_path / "runs",
        RunIdentity(
            run_id="analysis_run",
            study_name="study",
            variant_name="variant",
            variant_slug="env-wallgap__seed-7",
            signature="signature",
        ),
    )
    run_directory.create()
    live_staging_workspace = tmp_path / "artifacts" / ".staging" / "analysis_stage_live"
    final_report = tmp_path / "artifacts" / "reports" / "analysis" / "final_report"
    live_staging_workspace.mkdir(parents=True)
    (final_report / "figures").mkdir(parents=True)
    _set_live_analysis_staging_shortcut(run_directory, live_staging_workspace)

    _refresh_analysis_run_shortcuts(run_directory, final_report)

    assert not (run_directory.path / "staging_workspace").is_symlink()
    assert (run_directory.path / "figures").resolve() == (final_report / "figures").resolve()


def test_analysis_output_group_name_prefixes_comparative_results() -> None:
    assert (
        _analysis_output_group_name("encoder_place_cells.rate_map_panel") == "encoder_place_cells"
    )
    assert _analysis_output_group_name("remapping") == "comparative_remapping"


def test_grouped_output_file_name_preserves_non_source_segments() -> None:
    renamed = _grouped_output_file_name(
        Path("representation_drift__encoder.place_codes__phase_a__validation.png")
    )
    assert renamed == "representation_drift__phase_a__validation.png"
    unchanged = _grouped_output_file_name(Path("summary.json"))
    assert unchanged == "summary.json"


def test_comparative_input_payloads_requires_inputs() -> None:
    try:
        _comparative_input_payloads(
            "remapping",
            {
                "enabled": True,
                "inputs": [],
            },
        )
    except ValueError as exc:
        assert "no explicit inputs were provided" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected empty comparative inputs to be rejected")

    inputs = _comparative_input_payloads(
        "remapping",
        {
            "enabled": True,
            "inputs": [{"label": "phase_a"}],
        },
    )
    assert inputs == [{"label": "phase_a"}]


def test_disabled_comparative_items_are_filtered_before_input_validation() -> None:
    analysis_config = {
        "targets": {},
        "comparative": {
            "remapping": {
                "enabled": False,
                "inputs": [],
            },
            "active_comparison": {
                "enabled": True,
                "inputs": [{"label": "phase_a"}],
            },
        },
    }

    assert _enabled_comparative_items(analysis_config) == [
        ("active_comparison", {"enabled": True, "inputs": [{"label": "phase_a"}]})
    ]
    assert _analysis_work_item_count(analysis_config) == 1


def test_analysis_work_item_count_includes_extra_dataset_coverage_splits() -> None:
    analysis_config = {
        "split_name": "validation",
        "dataset_coverage_extra_splits": ["train", "test"],
        "targets": {
            "encoder_place_cells": {
                "enabled": True,
                "modules": ["dataset_coverage", "sparsity"],
            }
        },
        "comparative": {},
    }

    assert (
        _analysis_work_item_count_with_coverage_splits(
            analysis_config,
            default_split_name="validation",
            available_splits=["train", "validation", "test"],
        )
        == 7
    )
