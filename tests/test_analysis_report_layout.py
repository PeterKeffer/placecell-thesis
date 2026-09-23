from __future__ import annotations

from pathlib import Path

from placecell_research.stages.analyze_model import (
    _write_analysis_browser_links,
    _write_analysis_report_readme,
)


def test_analysis_browser_links_group_reports_by_model_dataset_and_run(tmp_path: Path) -> None:
    registry_root = tmp_path / "artifacts"
    report_path = registry_root / "reports" / "analysis" / "analysis_report_example"
    report_path.mkdir(parents=True)

    _write_analysis_browser_links(
        registry_root=registry_root,
        report_path=report_path,
        report_id="analysis_report_example",
        run_id="run_123",
        model_artifact_id="place_model_abc",
        dataset_artifact_id="encoded_dataset_xyz",
        split_name="test",
    )

    assert (
        registry_root
        / "reports"
        / "analysis"
        / "by_model"
        / "place_model_abc__test__encoded_dataset_xyz__analysis_report_example"
    ).is_symlink()
    assert (
        registry_root
        / "reports"
        / "analysis"
        / "by_dataset"
        / "encoded_dataset_xyz__test__place_model_abc__analysis_report_example"
    ).is_symlink()
    assert (
        registry_root / "reports" / "analysis" / "by_run" / "run_123__analysis_report_example"
    ).is_symlink()


def test_analysis_browser_links_shorten_long_run_and_report_names(tmp_path: Path) -> None:
    registry_root = tmp_path / "artifacts"
    report_id = f"analysis_report_{'very_long_variant_' * 12}123456"
    report_path = registry_root / "reports" / "analysis" / report_id
    report_path.mkdir(parents=True)
    run_id = f"slurm_9254302__{'very_long_variant_' * 12}20260531_1742_abc123_a1b2c3"

    _write_analysis_browser_links(
        registry_root=registry_root,
        report_path=report_path,
        report_id=report_id,
        run_id=run_id,
        model_artifact_id="place_model_abc",
        dataset_artifact_id="encoded_dataset_xyz",
        split_name="test",
    )

    by_run_links = list((registry_root / "reports" / "analysis" / "by_run").iterdir())
    assert len(by_run_links) == 1
    assert by_run_links[0].is_symlink()
    assert len(by_run_links[0].name) <= 255
    assert by_run_links[0].name.startswith("slurm_9254302__very_long_variant")


def test_analysis_report_readme_lists_targets_and_comparative_modules(tmp_path: Path) -> None:
    _write_analysis_report_readme(
        output_dir=tmp_path,
        model_artifact_id="place_model_abc",
        dataset_artifact_id="encoded_dataset_xyz",
        split_name="validation",
        target_names=["encoder_place_cells", "predictor_place_cells"],
        comparative_names=["remapping_comparison"],
    )

    readme_text = (tmp_path / "README.md").read_text()
    assert "model: `place_model_abc`" in readme_text
    assert "dataset: `encoded_dataset_xyz`" in readme_text
    assert "- `encoder_place_cells`" in readme_text
    assert "- `comparative_remapping_comparison`" in readme_text
