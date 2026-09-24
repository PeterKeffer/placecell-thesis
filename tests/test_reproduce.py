from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from placecell_research.config import (
    load_downstream_run_config,
    load_experiment_config,
    validate_downstream_run_config,
    validate_experiment_config,
)
from placecell_research.reproduce import execute
from placecell_research.reproduce.execute import run_locally, submit_to_slurm
from placecell_research.reproduce.plan import Step, build_plan

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def full_plan() -> dict[str, Step]:
    return {step.name: step for step in build_plan(CONFIG_ROOT)}


def _overrides(step: Step) -> list[str]:
    arguments = list(step.arguments)
    return [arguments[index + 1] for index, value in enumerate(arguments) if value == "-o"]


def test_full_plan_covers_every_condition_seed_and_navigation_run(full_plan) -> None:
    counts = Counter(step.kind for step in full_plan.values())
    assert counts == {
        "data": 3,
        "train": 83,
        "representations": 83,
        "measures": 84,
        "navigation": 85,
        "summary": 1,
    }
    assert {"train.baseline.s42", "train.baseline.s1", "train.baseline.s2"} <= set(full_plan)
    assert "train.winners_1.s42" in full_plan and "train.winners_1.s1" not in full_plan
    assert {"data.baseline", "data.museum", "data.objects_removed"} <= set(full_plan)
    assert full_plan["train.museum_untrained.s2"].dependencies == ("data.museum",)


def test_dependent_conditions_wait_for_their_parent_model(full_plan) -> None:
    retrofitted = full_plan["train.retrofitted_competition.s1"]
    assert "train.no_competition.s1" in retrofitted.dependencies
    assert "reuse.place_model_artifact_id=tag:no_competition_seed1" in _overrides(retrofitted)
    assert "retrofitted_competition_seed1" in retrofitted.arguments
    added = full_plan["measures.competition_added_after_training.s42"]
    assert added.dependencies == ("representations.no_competition.s42",)
    assert "train.competition_added_after_training.s42" not in full_plan
    assert "reuse.place_model_artifact_id=tag:no_competition" in _overrides(added)
    default_decay = full_plan["train.weight_decay_1e-5.s42"]
    assert default_decay.arguments[-2:] == ("--dataset", "auto")
    navigation = full_plan["navigation.dqn_pixels_south.s9"]
    assert navigation.dependencies == ("train.baseline.s42",)
    assert "seed=9" in _overrides(navigation)


def test_stage_decoding_and_input_decoding_follow_the_thesis_protocol(full_plan) -> None:
    baseline = _overrides(full_plan["representations.baseline.s42"])
    assert any("encoder.pre_sparsifier" in item for item in baseline)
    assert not any(
        item.startswith("representation_collection.sources")
        for item in _overrides(full_plan["representations.baseline.s1"])
    )
    feedforward = _overrides(full_plan["representations.feedforward_encoder.s42"])
    assert any("encoder.backbone_output" in item for item in feedforward)
    assert full_plan["measures.baseline.s42"].arguments[-1] == "--inputs"
    assert "--inputs" not in full_plan["measures.baseline.s1"].arguments


def test_only_adds_the_upstream_runs_a_selection_needs() -> None:
    names = [
        step.name for step in build_plan(CONFIG_ROOT, only=["retrofitted_competition"], seeds=[1])
    ]
    assert names.index("train.no_competition.s1") < names.index("train.retrofitted_competition.s1")
    assert not any(name.startswith("navigation") for name in names)
    navigation = [step.name for step in build_plan(CONFIG_ROOT, only=["ppo_place_code_north"])]
    assert "train.baseline.s42" in navigation
    assert [name for name in navigation if name.startswith("navigation.ppo")] == [
        "navigation.ppo_place_code_north.s7",
        "navigation.ppo_place_code_north.s8",
        "navigation.ppo_place_code_north.s9",
    ]
    with pytest.raises(ValueError, match="Unknown names"):
        build_plan(CONFIG_ROOT, only=["not_a_condition"])


def test_smoke_plan_is_small_keeps_conditions_distinct_and_writes_to_smoke() -> None:
    steps = {
        step.name: step
        for step in build_plan(
            CONFIG_ROOT,
            only=["baseline", "encoder_256_predictor_256", "no_competition"],
            smoke=True,
        )
    }
    assert not any(name.endswith((".s1", ".s2")) for name in steps)
    assert [name for name in steps if name.startswith("navigation.")] == [
        "navigation.ppo_place_code_north.s7",
        "navigation.measures",
    ]
    baseline = _overrides(steps["train.baseline.s42"])
    narrow = _overrides(steps["train.encoder_256_predictor_256.s42"])
    assert "spatial_model.encoder.layer_sizes=[32,32,32]" in baseline
    assert "spatial_model.encoder.layer_sizes=[8,8,8]" in narrow
    assert "tracking.artifact_root=smoke/artifacts" in baseline
    assert steps["summary"].arguments[1] == "smoke/measures"


def test_every_smoke_job_config_resolves_and_validates() -> None:
    for step in build_plan(CONFIG_ROOT, smoke=True):
        arguments = list(step.arguments)
        if "--config" not in arguments:
            continue
        config_path = Path(arguments[arguments.index("--config") + 1])
        overrides = _overrides(step)
        if step.downstream:
            validate_downstream_run_config(load_downstream_run_config(config_path, overrides))
        else:
            validate_experiment_config(load_experiment_config(config_path, overrides))


def test_smoke_trainings_of_different_conditions_never_share_a_model() -> None:
    trained: dict[tuple, str] = {}
    for step in build_plan(CONFIG_ROOT, smoke=True):
        if step.kind != "train":
            continue
        config = load_experiment_config(step.config, _overrides(step))
        if config.policies.artifact_reuse == "force_recompute":
            continue
        model = (step.dependencies, str(config.to_dict()["spatial_model"]), str(config.seed))
        assert model not in trained, f"{step.name} would reuse the model of {trained[model]}"
        trained[model] = step.name


def _dry_run(steps, tmp_path, monkeypatch, profile="hpc3") -> list[str]:
    def refuse(_path):
        raise AssertionError("a dry run must not call sbatch")

    monkeypatch.setattr(execute, "run_sbatch", refuse)
    lines: list[str] = []
    submit_to_slurm(
        steps,
        profile=profile,
        repo_root=tmp_path,
        state_dir=tmp_path / "runs" / "reproduce",
        launcher_overrides=[],
        dry_run=True,
        echo=lines.append,
    )
    return lines


def test_slurm_dry_run_caps_concurrency_with_lanes_and_chains_dependencies(
    tmp_path, monkeypatch
) -> None:
    steps = build_plan(CONFIG_ROOT, only=["baseline", "no_competition", "retrofitted_competition"])
    lines = _dry_run(steps, tmp_path, monkeypatch)
    lanes = set(re.findall(r"lane=(gpu\d+|cpu\d+)", "\n".join(lines)))
    assert {lane for lane in lanes if lane.startswith("gpu")} <= {"gpu0", "gpu1"}
    scripts = {path.stem: path.read_text() for path in (tmp_path / "runs/reproduce").rglob("*.sh")}
    assert len(scripts) == len(steps)
    retrofitted = scripts["train.retrofitted_competition.s42"]
    assert re.search(r"#SBATCH --dependency=afterok:\d+:\d+", retrofitted)
    assert "#SBATCH --gres=gpu:H100.80gb:1" in retrofitted
    measures = scripts["measures.baseline.s42"]
    assert "#SBATCH --partition=klab-cpu" in measures and "--gres" not in measures
    assert "touch" in measures and "measures.baseline.s42" in measures
    summary_dependency = re.search(r"--dependency=afterany:([\d:]+)", scripts["summary"])
    assert summary_dependency is not None
    assert len(summary_dependency.group(1).split(":")) <= 8


def test_slurm_dry_run_skips_finished_steps(tmp_path, monkeypatch) -> None:
    steps = build_plan(CONFIG_ROOT, only=["baseline"], include_navigation=False)
    done = tmp_path / "runs" / "reproduce" / "done"
    done.mkdir(parents=True)
    for name in ("data.baseline", "train.baseline.s42"):
        (done / name).touch()
    lines = _dry_run(steps, tmp_path, monkeypatch, profile="slurm")
    assert not any("job" in line and "train.baseline.s42" in line for line in lines)
    scripts = tmp_path / "runs" / "reproduce" / "dry_run_scripts"
    assert "--dependency" not in (scripts / "representations.baseline.s42.sh").read_text()


def test_local_run_resumes_from_markers_and_stops_at_a_failure(tmp_path) -> None:
    state_dir = tmp_path / "state"
    ok = Step(name="help", kind="summary", config=CONFIG_ROOT, arguments=("--help",))
    broken = Step(name="broken", kind="summary", config=CONFIG_ROOT, arguments=("no-such-command",))
    lines: list[str] = []
    assert (
        run_locally([ok], repo_root=tmp_path, state_dir=state_dir, dry_run=False, echo=lines.append)
        == 0
    )
    assert (state_dir / "done" / "help").is_file()
    status = run_locally(
        [ok, broken], repo_root=tmp_path, state_dir=state_dir, dry_run=False, echo=lines.append
    )
    assert status != 0
    assert any("help: done, skipped" in line for line in lines)
    assert not (state_dir / "done" / "broken").exists()
