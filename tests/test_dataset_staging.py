from __future__ import annotations

import atexit
from pathlib import Path

import pytest

from placecell_research.datasets import staging
from placecell_research.datasets.staging import (
    STAGE_DIR_ENV,
    create_dataset_archive,
    dataset_archive_digest_path,
    dataset_archive_path,
    stage_dataset_dir,
    try_create_dataset_archive,
)

_CHUNK_NAMES = ("0.0", "1.0", "2.0")


@pytest.fixture(autouse=True)
def isolated_staging_state():
    """Keep the per-process stage cache and owned copies from leaking between tests."""
    staging._staged_directories.clear()
    staging._owned_paths.clear()
    yield
    staging._staged_directories.clear()
    staging._owned_paths.clear()
    atexit.unregister(staging.remove_staged_copies)


def _write_artifact(root: Path, artifact_id: str = "encoded_demo_f0788a") -> Path:
    artifact_dir = root / artifact_id
    store_dir = artifact_dir / "dataset.zarr" / "observations" / "latent"
    store_dir.mkdir(parents=True)
    (artifact_dir / "dataset.zarr" / ".zgroup").write_text('{"zarr_format": 2}')
    for index, name in enumerate(_CHUNK_NAMES):
        (store_dir / name).write_bytes(bytes([index]) * 512)
    (artifact_dir / "manifest_summary.json").write_text('{"num_actions": 4}')
    return artifact_dir


def _assert_matches_artifact(staged_dir: Path, artifact_dir: Path) -> None:
    assert (staged_dir / "manifest_summary.json").read_bytes() == (
        artifact_dir / "manifest_summary.json"
    ).read_bytes()
    for name in _CHUNK_NAMES:
        relative = Path("dataset.zarr") / "observations" / "latent" / name
        assert (staged_dir / relative).read_bytes() == (artifact_dir / relative).read_bytes()


def test_stage_dir_unset_returns_artifact_dir(tmp_path, monkeypatch):
    monkeypatch.delenv(STAGE_DIR_ENV, raising=False)
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)

    assert stage_dataset_dir(artifact_dir) == artifact_dir


def test_missing_archive_falls_back_with_one_warning(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_artifact(tmp_path)
    monkeypatch.setenv(STAGE_DIR_ENV, str(tmp_path / "stage"))

    assert stage_dataset_dir(artifact_dir) == artifact_dir
    output = capsys.readouterr().out
    assert output.count("WARNING") == 1
    assert "no sidecar archive" in output


def test_stage_in_produces_a_byte_identical_local_copy(tmp_path, monkeypatch):
    artifact_dir = _write_artifact(tmp_path)
    archive_path = create_dataset_archive(artifact_dir)
    assert archive_path == dataset_archive_path(artifact_dir)
    stage_root = tmp_path / "stage"
    monkeypatch.setenv(STAGE_DIR_ENV, str(stage_root))
    monkeypatch.setenv("SLURM_JOB_ID", "9999001")

    staged_dir = stage_dataset_dir(artifact_dir)

    assert staged_dir != artifact_dir
    assert staged_dir.parent == stage_root / "pc_dataset_9999001_encoded_demo_f0788a"
    assert (staged_dir.parent / ".stage_complete").is_file()
    _assert_matches_artifact(staged_dir, artifact_dir)


def test_sentinel_lets_a_second_process_reuse_the_staged_copy(tmp_path, monkeypatch):
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)
    monkeypatch.setenv(STAGE_DIR_ENV, str(tmp_path / "stage"))
    monkeypatch.setenv("SLURM_JOB_ID", "9999002")
    staged_dir = stage_dataset_dir(artifact_dir)

    def _fail_if_extracted(*_args, **_kwargs):
        raise AssertionError("the published copy was extracted a second time")

    staging._staged_directories.clear()
    monkeypatch.setattr(staging, "_extract_archive", _fail_if_extracted)
    assert stage_dataset_dir(artifact_dir) == staged_dir


def test_unreadable_archive_falls_back_and_leaves_no_debris(tmp_path, monkeypatch):
    artifact_dir = _write_artifact(tmp_path)
    archive_path = create_dataset_archive(artifact_dir)
    archive_path.write_bytes(b"not a tar file at all")
    stage_root = tmp_path / "stage"
    monkeypatch.setenv(STAGE_DIR_ENV, str(stage_root))

    assert stage_dataset_dir(artifact_dir) == artifact_dir
    assert list(stage_root.iterdir()) == []


def test_corrupted_archive_content_fails_the_identity_check(tmp_path, monkeypatch):
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)
    (artifact_dir / "dataset.zarr" / ".zgroup").write_text('{"zarr_format": 3}')
    stage_root = tmp_path / "stage"
    monkeypatch.setenv(STAGE_DIR_ENV, str(stage_root))

    assert stage_dataset_dir(artifact_dir) == artifact_dir
    assert list(stage_root.iterdir()) == []


def test_insufficient_free_space_falls_back(tmp_path, monkeypatch):
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)
    monkeypatch.setenv(STAGE_DIR_ENV, str(tmp_path / "stage"))
    monkeypatch.setattr(staging, "_free_bytes", lambda path: 512)

    assert stage_dataset_dir(artifact_dir) == artifact_dir


def test_cleanup_removes_the_staged_copy(tmp_path, monkeypatch):
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)
    stage_root = tmp_path / "stage"
    monkeypatch.setenv(STAGE_DIR_ENV, str(stage_root))
    staged_dir = stage_dataset_dir(artifact_dir)
    assert staged_dir.is_dir()

    staging.remove_staged_copies()

    assert not staged_dir.parent.exists()
    assert artifact_dir.is_dir()


def test_cgroup_memory_files_follow_the_slurm_job_cgroup():
    v1_pairs = list(
        staging._cgroup_memory_files("9:memory:/slurm/uid_1234/job_9999/step_0\n8:cpuset:/\n")
    )
    v2_pairs = list(staging._cgroup_memory_files("0::/system.slice/slurmstepd.scope/job_9999\n"))

    assert v1_pairs == [
        (
            Path("/sys/fs/cgroup/memory/slurm/uid_1234/job_9999/step_0/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/slurm/uid_1234/job_9999/step_0/memory.usage_in_bytes"),
        )
    ]
    assert v2_pairs == [
        (
            Path("/sys/fs/cgroup/system.slice/slurmstepd.scope/job_9999/memory.max"),
            Path("/sys/fs/cgroup/system.slice/slurmstepd.scope/job_9999/memory.current"),
        )
    ]


def test_create_dataset_archive_refuses_to_overwrite(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)

    with pytest.raises(FileExistsError):
        create_dataset_archive(artifact_dir)
    create_dataset_archive(artifact_dir, overwrite=True)


def test_publishing_an_encoded_dataset_writes_the_sidecar_archive(tmp_path):
    artifact_dir = _write_artifact(tmp_path)

    archive_path = try_create_dataset_archive(artifact_dir, "encoded_dataset")

    assert archive_path == dataset_archive_path(artifact_dir)
    assert archive_path.is_file()


@pytest.mark.parametrize("artifact_type", ["raw_dataset", "place_model"])
def test_other_artifact_types_get_no_archive(tmp_path, artifact_type):
    artifact_dir = _write_artifact(tmp_path)

    assert try_create_dataset_archive(artifact_dir, artifact_type) is None
    assert not dataset_archive_path(artifact_dir).exists()


def test_existing_archive_is_left_alone(tmp_path):
    artifact_dir = _write_artifact(tmp_path)
    archive_path = dataset_archive_path(artifact_dir)
    archive_path.write_bytes(b"placeholder")

    assert try_create_dataset_archive(artifact_dir, "encoded_dataset") is None
    assert archive_path.read_bytes() == b"placeholder"


def test_archive_failure_warns_instead_of_raising(tmp_path, monkeypatch, capsys):
    """A cache must not break a pipeline: the stage keeps going without a tar."""
    artifact_dir = _write_artifact(tmp_path)

    def _explode(_artifact_dir):
        raise OSError("storage target went away")

    monkeypatch.setattr(staging, "_write_archive_in_niced_child", _explode)

    assert try_create_dataset_archive(artifact_dir, "encoded_dataset") is None
    assert not dataset_archive_path(artifact_dir).exists()
    assert "storage target went away" in capsys.readouterr().out


def test_corrupted_unsampled_member_fails_verification(tmp_path, monkeypatch):
    """Every archived chunk is covered, not only the ones a sparse sample happened to hit."""
    artifact_dir = _write_artifact(tmp_path)
    archive_path = create_dataset_archive(artifact_dir)
    original_content = bytes([2]) * 512
    archive_bytes = archive_path.read_bytes()
    assert archive_bytes.count(original_content) == 1
    archive_path.write_bytes(
        archive_bytes.replace(original_content, bytes([9]) + original_content[1:])
    )
    stage_root = tmp_path / "stage"
    monkeypatch.setenv(STAGE_DIR_ENV, str(stage_root))

    assert stage_dataset_dir(artifact_dir) == artifact_dir
    assert list(stage_root.iterdir()) == []


def test_modified_staged_member_is_not_reused_through_the_sentinel(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_artifact(tmp_path)
    create_dataset_archive(artifact_dir)
    monkeypatch.setenv(STAGE_DIR_ENV, str(tmp_path / "stage"))
    monkeypatch.setenv("SLURM_JOB_ID", "9999004")
    staged_dir = stage_dataset_dir(artifact_dir)
    chunk_path = staged_dir / "dataset.zarr" / "observations" / "latent" / "2.0"
    chunk_path.write_bytes(bytes([9]) * 512)

    staging._staged_directories.clear()
    assert stage_dataset_dir(artifact_dir) == artifact_dir
    assert "content digest" in capsys.readouterr().out


def test_archive_child_survives_a_refused_niceness(tmp_path, monkeypatch, capsys):
    artifact_dir = _write_artifact(tmp_path)

    def _refuse(_increment):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(staging.os, "nice", _refuse)

    staging._archive_child_entry_point(str(artifact_dir))

    assert dataset_archive_path(artifact_dir).is_file()
    assert "niceness" in capsys.readouterr().out


def test_killed_child_leaves_no_temp_tar_or_digest(tmp_path, monkeypatch):
    """A SIGKILLed child skips its own cleanup, so the parent clears both temp files by pid."""
    artifact_dir = _write_artifact(tmp_path)
    archive_path = dataset_archive_path(artifact_dir)
    digest_path = dataset_archive_digest_path(artifact_dir)
    child_pid = 4242
    temp_paths = [
        archive_path.with_name(f"{archive_path.name}.tmp.{child_pid}"),
        digest_path.with_name(f".{digest_path.name}.tmp.{child_pid}"),
    ]

    class _KilledChild:
        pid = child_pid
        exitcode = 0

        def __init__(self, **kwargs):
            pass

        def start(self):
            for temp_path in temp_paths:
                temp_path.write_text("partial")

        def join(self):
            pass

    class _Context:
        Process = _KilledChild

    monkeypatch.setattr(staging.multiprocessing, "get_context", lambda _method: _Context())

    staging._write_archive_in_niced_child(artifact_dir)

    assert [temp_path for temp_path in temp_paths if temp_path.exists()] == []
