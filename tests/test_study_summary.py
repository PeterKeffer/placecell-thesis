from __future__ import annotations

import csv
import json
from pathlib import Path

from placecell_research.studies.summary import write_study_summary, write_sweep_comparison_links


def test_write_study_summary_adds_human_index_and_grouped_links(tmp_path: Path) -> None:
    analysis_report_path = tmp_path / "external_analysis_report"
    analysis_report_path.mkdir()
    (analysis_report_path / "figures").mkdir()
    (analysis_report_path / "tables").mkdir()
    checkpoint_path = tmp_path / "weights_best_primary.pt"
    checkpoint_path.write_text("checkpoint")

    written = write_study_summary(
        tmp_path / "study_report",
        rows=[
            {
                "row_type": "phase_train",
                "phase": "train_on_env_a",
                "dataset": "encoded_env_a",
                "checkpoint_path": str(checkpoint_path),
                "place_model_artifact_id": "place_model_a",
                "stage.run_id": "run_train",
                "stage.run_path": "/tmp/run_train",
                "status": 1.0,
            },
            {
                "row_type": "phase_analysis",
                "phase": "train_on_env_a",
                "dataset": "encoded_env_b",
                "analysis_report_id": "analysis_report_b",
                "analysis_report_path": str(analysis_report_path),
                "stage.run_id": "run_analysis",
                "stage.run_path": "/tmp/run_analysis",
                "status": 0.5,
            },
        ],
        objective_metric="status",
        objective_mode="max",
    )

    study_index_rows = list(
        csv.DictReader((tmp_path / "study_report" / "study_index.csv").read_text().splitlines())
    )
    assert (
        study_index_rows[0]["row_label"]
        == "Train place model in phase train_on_env_a on encoded_env_a"
    )
    assert study_index_rows[1]["analysis_figures_path"].endswith("/figures")
    assert study_index_rows[1]["analysis_tables_path"].endswith("/tables")

    grouped_phase_dir = tmp_path / "study_report" / "by_phase"
    assert any(
        path.is_symlink() and path.name.startswith("train_on_env_a__analysis_on__encoded_env_b__")
        for path in grouped_phase_dir.iterdir()
    )

    best_row = json.loads((tmp_path / "study_report" / "best_runs_by_metric.json").read_text())
    assert best_row["row_type"] == "phase_train"
    assert written["by_phase"] == tmp_path / "study_report" / "by_phase"
    assert (tmp_path / "study_report" / "README.md").exists()


def test_write_sweep_comparison_links_groups_exclusive_sparsity_runs(tmp_path: Path) -> None:
    l1_run_path = tmp_path / "runs" / "by_id" / "run_l1"
    kwinners_run_path = tmp_path / "runs" / "by_id" / "run_kwinners"
    l1_run_path.mkdir(parents=True)
    kwinners_run_path.mkdir(parents=True)

    written = write_sweep_comparison_links(
        tmp_path / "runs" / "studies" / "ablation_sparsity",
        [
            {
                "stage.run_path": str(l1_run_path),
                "seed": 0,
                "spatial_model.sparsifier.type": "none",
                "spatial_model.objectives.l1_encoder.weight": 0.01,
            },
            {
                "stage.run_path": str(kwinners_run_path),
                "seed": 1,
                "spatial_model.sparsifier.type": "kwinners",
                "spatial_model.sparsifier.k_fraction": 0.06,
                "spatial_model.objectives.l1_encoder.weight": 0.0,
            },
        ],
        [
            "spatial_model.sparsifier.type",
            "spatial_model.sparsifier.k_fraction",
            "spatial_model.objectives.l1_encoder.weight",
        ],
    )

    l1_link = (
        tmp_path
        / "runs"
        / "studies"
        / "ablation_sparsity"
        / "type_sparsifier-none"
        / "weight_l1_encoder_objectives-0_01"
        / "seed-0"
    )
    kwinners_link = (
        tmp_path
        / "runs"
        / "studies"
        / "ablation_sparsity"
        / "type_sparsifier-kwinners"
        / "k_fraction_sparsifier-0_06"
        / "weight_l1_encoder_objectives-0_0"
        / "seed-1"
    )
    assert l1_link.is_symlink()
    assert l1_link.resolve() == l1_run_path
    assert kwinners_link.is_symlink()
    assert kwinners_link.resolve() == kwinners_run_path
    assert written == {l1_link, kwinners_link}


def test_write_sweep_comparison_links_preserves_parameter_order(tmp_path: Path) -> None:
    run_path = tmp_path / "runs" / "by_id" / "run_generic"
    run_path.mkdir(parents=True)

    write_sweep_comparison_links(
        tmp_path / "runs" / "studies" / "generic",
        [
            {
                "stage.run_path": str(run_path),
                "seed": 3,
                "optimizer.learning_rate": 0.0003,
                "model.dropout": 0.1,
            }
        ],
        ["optimizer.learning_rate", "model.dropout"],
    )

    assert (
        tmp_path
        / "runs"
        / "studies"
        / "generic"
        / "learning_rate_optimizer-0_0003"
        / "dropout_model-0_1"
        / "seed-3"
    ).is_symlink()
