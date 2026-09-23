"""Per-job code snapshots: selection, budgets, reuse, rendering, retention."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import pytest

from placecell_research.launch import code_snapshot, remote_run
from placecell_research.utils.repo_paths import find_repo_root

REMOTE_REPO_ROOT = "/remote/placecell-thesis"


def _script(timestamp: str = "20260826T101112Z", cutoff: str = "20260812T101112Z") -> str:
    return code_snapshot.build_code_snapshot_script(
        remote_repo_root=REMOTE_REPO_ROOT,
        timestamp=timestamp,
        cutoff=cutoff,
    )


def test_snapshot_covers_only_importable_and_resolvable_directories() -> None:
    assert code_snapshot.SNAPSHOT_SOURCE_DIRS == (
        "src",
        "configs",
        "scripts/slurm",
    )
    for forbidden in ("artifacts", "runs", "results", "writing", ".venv", ".git", "scripts"):
        assert forbidden not in code_snapshot.SNAPSHOT_SOURCE_DIRS
    assert "scripts/slurm" in " ".join(code_snapshot.SNAPSHOT_SOURCE_DIRS)


def test_snapshot_never_carries_pyproject_because_it_would_move_the_repo_root() -> None:
    script = _script()

    assert "pyproject.toml" not in code_snapshot.SNAPSHOT_SOURCE_DIRS
    assert 'if [[ -e "${snapshot_dir}/pyproject.toml" ]]; then' in script


def test_snapshot_selection_excludes_caches_and_bytecode_in_both_renderings() -> None:
    script = _script()

    for excluded_dir in code_snapshot.SNAPSHOT_EXCLUDED_DIRS:
        assert f"! -path '*/{excluded_dir}/*'" in script
        assert f"--exclude {excluded_dir}/" in script
    assert "! -name '*.pyc'" in script
    assert "--exclude '*.pyc'" in script


def test_snapshot_script_refuses_a_set_that_exceeds_the_budgets() -> None:
    script = _script()

    assert f'if [[ "${{file_count}}" -gt {code_snapshot.MAX_SNAPSHOT_FILES} ]]; then' in script
    assert f'if [[ "${{total_bytes}}" -gt {code_snapshot.MAX_SNAPSHOT_BYTES} ]]; then' in script
    assert script.count("exit 1") >= 3
    assert code_snapshot.MAX_SNAPSHOT_BYTES == 50 * 1024 * 1024
    assert code_snapshot.MAX_SNAPSHOT_FILES == 20_000


def test_snapshot_script_reuses_an_existing_directory_for_identical_content() -> None:
    script = _script()

    assert 'existing=$(ls -1d "${snapshot_root}"/*_"${content_hash}"' in script
    assert 'if [[ -n "${existing}" && -f "${existing}/.snapshot_hash" ]]; then' in script
    assert "  reused=1" in script
    assert 'echo "[code-snapshot] reused=${reused}"' in script


def test_snapshot_script_hardlinks_against_the_newest_previous_snapshot() -> None:
    script = _script()

    assert 'link_dest=$(ls -1d "${snapshot_root}"/*/' in script
    assert '${link_dest:+--link-dest="${link_dest%/}"}' in script


def test_snapshot_script_prunes_only_old_generated_directories() -> None:
    script = _script(cutoff="20260812T101112Z")

    assert f'[[ "${{name}}" =~ {code_snapshot.SNAPSHOT_DIR_PATTERN} ]] || continue' in script
    assert f'[[ "${{kept}}" -gt {code_snapshot.SNAPSHOT_RETENTION_KEEP} ]] || continue' in script
    assert '[[ "${name%%_*}" < "20260812T101112Z" ]] || continue' in script
    assert 'rm -rf -- "${snapshot_root}/${name}"' in script


def test_retention_cutoff_trails_the_moment_by_the_retention_window() -> None:
    moment = datetime(2026, 8, 26, 10, 11, 12, tzinfo=UTC)

    assert code_snapshot.snapshot_timestamp(moment) == "20260826T101112Z"
    assert code_snapshot.retention_cutoff(moment) == "20260812T101112Z"
    assert code_snapshot.SNAPSHOT_RETENTION_DAYS >= 7


def test_parse_code_snapshot_output_reads_fields_from_a_noisy_remote_shell() -> None:
    output = "\n".join(
        [
            "_spack_view ()",
            "{",
            "    echo hi",
            "}",
            f"[code-snapshot] path={REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34",
            "[code-snapshot] hash=ab12cd34",
            "[code-snapshot] files=865",
            "[code-snapshot] bytes=6236829",
            "[code-snapshot] reused=0",
            "[code-snapshot] pruned=20260101T000000Z_00000000",
        ]
    )

    snapshot = code_snapshot.parse_code_snapshot_output(output)

    assert snapshot.path == f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34"
    assert snapshot.content_hash == "ab12cd34"
    assert snapshot.file_count == 865
    assert snapshot.total_bytes == 6236829
    assert snapshot.reused is False
    assert snapshot.python_path.endswith("20260826T101112Z_ab12cd34/src")


def test_parse_code_snapshot_output_fails_loudly_on_a_truncated_report() -> None:
    with pytest.raises(RuntimeError, match="hash, files, bytes, reused"):
        code_snapshot.parse_code_snapshot_output("[code-snapshot] path=/somewhere")


def test_rewrite_config_path_for_snapshot_maps_into_the_frozen_copy() -> None:
    rewritten = code_snapshot.rewrite_config_path_for_snapshot(
        remote_config_path=PurePosixPath(
            f"{REMOTE_REPO_ROOT}/configs/experiment/smoke_miniworld.yaml"
        ),
        remote_repo_root=PurePosixPath(REMOTE_REPO_ROOT),
        snapshot_path=f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34",
    )

    assert str(rewritten) == (
        f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34"
        "/configs/experiment/smoke_miniworld.yaml"
    )


def test_rewrite_config_path_for_snapshot_rejects_a_path_outside_configs() -> None:
    with pytest.raises(ValueError, match="only cover configs/"):
        code_snapshot.rewrite_config_path_for_snapshot(
            remote_config_path=PurePosixPath(f"{REMOTE_REPO_ROOT}/scripts/run.yaml"),
            remote_repo_root=PurePosixPath(REMOTE_REPO_ROOT),
            snapshot_path=f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34",
        )


def test_repo_root_resolves_past_a_snapshot_to_the_shared_checkout(tmp_path: Path) -> None:
    """The audit that keeps artifacts/ and runs/ on the shared tree."""
    repo_root = tmp_path / "placecell_research"
    (repo_root / "src" / "placecell_research").mkdir(parents=True)
    (repo_root / "configs" / "experiment").mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='placecell-research'\n")
    snapshot = repo_root / code_snapshot.SNAPSHOT_DIR_NAME / "20260826T101112Z_ab12cd34"
    (snapshot / "src" / "placecell_research").mkdir(parents=True)
    (snapshot / "configs" / "experiment").mkdir(parents=True)
    snapshot_config = snapshot / "configs" / "experiment" / "smoke_miniworld.yaml"
    snapshot_config.write_text("name: smoke\n")

    assert find_repo_root(snapshot_config) == repo_root


def test_config_fingerprint_survives_the_move_into_a_snapshot(tmp_path: Path) -> None:
    """Artifact reuse keys on the resolved config's text, which must not carry its own path."""
    import shutil

    import yaml

    from placecell_research.artifacts.ids import config_fingerprint
    from placecell_research.config import load_experiment_config

    repo_root = Path(__file__).resolve().parents[1]
    shutil.copytree(
        repo_root / "configs",
        tmp_path / "configs",
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    def fingerprint(config_path: Path) -> str:
        config = load_experiment_config(config_path, ["launcher=hpc3"])
        return config_fingerprint(yaml.safe_dump(config.to_dict(), sort_keys=False))

    relative_config = Path("configs") / "experiment" / "smoke_miniworld.yaml"

    assert fingerprint(repo_root / relative_config) == fingerprint(tmp_path / relative_config)


def test_import_fails_loudly_when_the_snapshot_is_not_on_the_path(tmp_path: Path) -> None:
    """Precedence bugs must fail at t=0, not hours into collection."""
    completed = subprocess.run(
        [sys.executable, "-c", "import placecell_research"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            code_snapshot.SNAPSHOT_ENV_VAR: str(tmp_path / "not_the_snapshot"),
        },
    )

    assert completed.returncode != 0
    assert "PLACECELL_CODE_SNAPSHOT" in completed.stderr
    assert "does not pin" in completed.stderr


def test_import_succeeds_when_the_package_comes_from_the_declared_snapshot() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-c", "import placecell_research"],
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(source_root),
            code_snapshot.SNAPSHOT_ENV_VAR: str(source_root.parent),
        },
    )

    assert completed.returncode == 0, completed.stderr


SETTINGS = remote_run.RemoteSettings(
    host="cluster-login", repo_root=REMOTE_REPO_ROOT, setup="", python="python"
)


def test_submit_command_points_the_job_at_the_snapshot() -> None:
    snapshot = code_snapshot.CodeSnapshot(
        path=f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34",
        content_hash="ab12cd34",
        file_count=865,
        total_bytes=6236829,
        reused=False,
    )
    settings = remote_run.RemoteSettings(
        host="cluster-login",
        repo_root=REMOTE_REPO_ROOT,
        setup="conda activate placecell",
        python="python",
    )

    script = remote_run.format_remote_cli_command(
        settings,
        ["submit", "--config", f"{snapshot.path}/configs/experiment/smoke_miniworld.yaml"],
        snapshot,
    )

    assert f"export PYTHONPATH={snapshot.path}/src" in script
    assert "${PYTHONPATH:+:${PYTHONPATH}}" in script
    assert f"export PLACECELL_CODE_SNAPSHOT={snapshot.path}" in script
    assert f"--config {snapshot.path}/configs/experiment/smoke_miniworld.yaml" in script
    assert f"cd {REMOTE_REPO_ROOT}" in script
    assert script.index("conda activate placecell") < script.index("export PYTHONPATH")


def test_submit_command_without_a_snapshot_keeps_the_live_checkout_on_the_path() -> None:
    script = remote_run.format_remote_cli_command(SETTINGS, ["submit"], None)

    assert "export PYTHONPATH=src" in script
    assert "PLACECELL_CODE_SNAPSHOT" not in script


def _fake_submit_transport(observed_commands: list[list[str]], snapshot_path: str):
    def fake_run_subprocess(command: list[str]):
        observed_commands.append(command)
        if "[code-snapshot]" in command[-1] or "snapshot_root=" in command[-1]:
            stdout = "\n".join(
                [
                    f"[code-snapshot] path={snapshot_path}",
                    "[code-snapshot] hash=ab12cd34",
                    "[code-snapshot] files=865",
                    "[code-snapshot] bytes=6236829",
                    "[code-snapshot] reused=0",
                ]
            )
            return type("Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "Submitted batch job 77777\n", "stderr": ""},
        )()

    return fake_run_subprocess


def test_submit_remote_slurm_job_snapshots_by_default(monkeypatch) -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    snapshot_path = f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34"
    observed_commands: list[list[str]] = []
    monkeypatch.setattr(
        remote_run,
        "_run_subprocess",
        _fake_submit_transport(observed_commands, snapshot_path),
    )

    result = remote_run.submit_remote_slurm_job(config_path, [], SETTINGS, sync_repo=False)

    assert result.job_id == "77777"
    assert result.code_snapshot_path == snapshot_path
    assert result.code_snapshot_hash == "ab12cd34"
    assert len(observed_commands) == 2
    submit_script = observed_commands[1][-1]
    assert f"--config {snapshot_path}/configs/experiment/smoke_miniworld.yaml" in submit_script
    assert f"export PYTHONPATH={snapshot_path}/src" in submit_script
    assert (
        result.slurm_log_path
        == f"{REMOTE_REPO_ROOT}/smoke/runs/slurm_logs/placecell_research_77777.out"
    )


def test_submit_remote_slurm_job_escape_hatch_skips_the_snapshot(monkeypatch) -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    observed_commands: list[list[str]] = []
    monkeypatch.setattr(
        remote_run,
        "_run_subprocess",
        _fake_submit_transport(observed_commands, "unused"),
    )

    result = remote_run.submit_remote_slurm_job(
        config_path, [], SETTINGS, sync_repo=False, snapshot_code=False
    )

    assert result.code_snapshot_path == ""
    assert len(observed_commands) == 1
    assert (
        f"--config {REMOTE_REPO_ROOT}/configs/experiment/smoke_miniworld.yaml"
        in observed_commands[0][-1]
    )


def test_default_rsync_excludes_protect_remote_snapshots() -> None:
    assert f"/{code_snapshot.SNAPSHOT_DIR_NAME}/" in remote_run.DEFAULT_RSYNC_EXCLUDES


def test_run_record_provenance_records_the_snapshot_the_job_imported(monkeypatch) -> None:
    """The git stamp cannot describe uncommitted work; the snapshot hash can."""
    from placecell_research.tracking import naming

    snapshot_path = f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_ab12cd34"
    monkeypatch.setattr(naming, "_git", lambda args, cwd: "deadbeef")
    monkeypatch.setenv(code_snapshot.SNAPSHOT_ENV_VAR, snapshot_path)

    git_state = naming.capture_git_state(Path("/nowhere"))

    assert git_state["commit"] == "deadbeef"
    assert git_state["code_snapshot"] == snapshot_path
    assert git_state["code_snapshot_hash"] == "ab12cd34"


def test_run_record_provenance_stays_quiet_for_local_runs(monkeypatch) -> None:
    from placecell_research.tracking import naming

    monkeypatch.setattr(naming, "_git", lambda args, cwd: "deadbeef")
    monkeypatch.delenv(code_snapshot.SNAPSHOT_ENV_VAR, raising=False)

    git_state = naming.capture_git_state(Path("/nowhere"))

    assert "code_snapshot" not in git_state


def _fake_chained_transport(created_snapshot_paths: list[str], submitted_scripts: list[str]):
    """Emulate the remote: identical content resolves to the one snapshot directory."""
    snapshot_paths_by_hash: dict[str, str] = {}
    next_job_id = iter(range(12345, 12400))

    def fake_run_subprocess(command: list[str]):
        script = command[-1]
        if "snapshot_root=" in script:
            content_hash = "ab12cd34"
            reused = content_hash in snapshot_paths_by_hash
            if not reused:
                path = f"{REMOTE_REPO_ROOT}/.code_snapshots/20260826T101112Z_{content_hash}"
                snapshot_paths_by_hash[content_hash] = path
                created_snapshot_paths.append(path)
            stdout = "\n".join(
                [
                    f"[code-snapshot] path={snapshot_paths_by_hash[content_hash]}",
                    f"[code-snapshot] hash={content_hash}",
                    "[code-snapshot] files=865",
                    "[code-snapshot] bytes=6236829",
                    f"[code-snapshot] reused={int(reused)}",
                ]
            )
            return type("Completed", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()
        submitted_scripts.append(script)
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": f"Submitted batch job {next(next_job_id)}\n", "stderr": ""},
        )()

    return fake_run_subprocess


def test_a_chain_of_detached_submissions_shares_one_snapshot(monkeypatch) -> None:
    """Ten chained jobs off an unchanged tree get one frozen copy, each pinned to it."""
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke_miniworld.yaml"
    )
    created_snapshot_paths: list[str] = []
    submitted_scripts: list[str] = []

    monkeypatch.setattr(
        remote_run,
        "_run_subprocess",
        _fake_chained_transport(created_snapshot_paths, submitted_scripts),
    )
    monkeypatch.setattr(
        remote_run,
        "stream_remote_job",
        lambda *args, **kwargs: pytest.fail("a chained submission must not block on the stream"),
    )

    results = []
    previous_job_id = None
    for _ in range(3):
        results.append(
            remote_run.run_remote_slurm_job(
                config_path,
                [],
                SETTINGS,
                sync_repo=False,
                stream_logs=False,
                slurm_dependency=None if previous_job_id is None else f"afterany:{previous_job_id}",
            )
        )
        previous_job_id = results[-1].job_id

    assert [result.job_id for result in results] == ["12345", "12346", "12347"]
    assert len(created_snapshot_paths) == 1
    assert {result.code_snapshot_path for result in results} == set(created_snapshot_paths)
    assert {result.code_snapshot_hash for result in results} == {"ab12cd34"}
    for script in submitted_scripts:
        assert f"export PYTHONPATH={created_snapshot_paths[0]}/src" in script
        assert f"export {code_snapshot.SNAPSHOT_ENV_VAR}={created_snapshot_paths[0]}" in script
        assert (
            f"--config {created_snapshot_paths[0]}/configs/experiment/smoke_miniworld.yaml"
            in script
        )
    assert "--slurm-dependency" not in submitted_scripts[0]
    assert "--slurm-dependency afterany:12345" in submitted_scripts[1]
    assert "--slurm-dependency afterany:12346" in submitted_scripts[2]


_REQUIRED_TOOLS = ("bash", "rsync", "sha256sum")


def _run_snapshot_script(root: Path, *, timestamp: str, cutoff: str) -> dict[str, str]:
    for tool in _REQUIRED_TOOLS:
        if shutil.which(tool) is None:
            pytest.skip(f"the snapshot script needs {tool}")
    script = code_snapshot.build_code_snapshot_script(
        remote_repo_root=str(root), timestamp=timestamp, cutoff=cutoff
    )
    completed = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    fields: dict[str, str] = {"pruned": ""}
    pruned: list[str] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("[code-snapshot] "):
            continue
        key, _, value = line[len("[code-snapshot] ") :].partition("=")
        if key == "pruned":
            pruned.append(value)
        else:
            fields[key] = value
    fields["pruned"] = ",".join(pruned)
    return fields


def _make_remote_checkout(root: Path) -> Path:
    module = root / "src" / "placecell_research" / "loop.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    (root / "configs").mkdir()
    (root / "configs" / "smoke_miniworld.yaml").write_text("epochs: 1\n")
    (root / "scripts" / "slurm").mkdir(parents=True)
    (root / "scripts" / "slurm" / "env_common.sh").write_text("# env\n")
    return module


def test_a_byte_change_that_preserves_size_and_mtime_changes_the_snapshot(tmp_path: Path) -> None:
    module = _make_remote_checkout(tmp_path)
    first = _run_snapshot_script(tmp_path, timestamp="20260826T101112Z", cutoff="20260812T101112Z")

    before = module.stat()
    module.write_text("VALUE = 2\n")
    os.utime(module, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert module.stat().st_size == before.st_size
    assert module.stat().st_mtime_ns == before.st_mtime_ns

    second = _run_snapshot_script(tmp_path, timestamp="20260826T101113Z", cutoff="20260812T101112Z")

    assert second["hash"] != first["hash"]
    assert second["reused"] == "0"
    snapshot_module = Path(second["path"]) / "src" / "placecell_research" / "loop.py"
    assert snapshot_module.read_text() == "VALUE = 2\n"


def test_a_snapshot_copies_slurm_env_scripts_at_their_relative_path(tmp_path: Path) -> None:
    """The generated batch script sources ${PLACECELL_CODE_SNAPSHOT}/scripts/slurm/<script>."""
    _make_remote_checkout(tmp_path)
    (tmp_path / "scripts" / "slurm" / "env_miniworld.sh").write_text("# env2\n")
    stray = tmp_path / "scripts" / "install"
    stray.mkdir()
    (stray / "helper.sh").write_text("# not part of the snapshot\n")

    fields = _run_snapshot_script(tmp_path, timestamp="20260826T101112Z", cutoff="20260812T101112Z")

    env_script = Path(fields["path"]) / "scripts" / "slurm" / "env_miniworld.sh"
    assert env_script.read_text() == "# env2\n"
    assert not (Path(fields["path"]) / "slurm").exists()
    assert not (Path(fields["path"]) / "scripts" / "install").exists()
    assert fields["files"] == "4"


def test_identical_content_still_reuses_the_same_snapshot(tmp_path: Path) -> None:
    _make_remote_checkout(tmp_path)
    first = _run_snapshot_script(tmp_path, timestamp="20260826T101112Z", cutoff="20260812T101112Z")
    second = _run_snapshot_script(tmp_path, timestamp="20260826T101113Z", cutoff="20260812T101112Z")

    assert second["reused"] == "1"
    assert second["path"] == first["path"]
    assert second["files"] == first["files"] == "3"
    assert int(second["bytes"]) == len("VALUE = 1\nepochs: 1\n# env\n")


def _fill_the_retention_window(snapshot_root: Path) -> None:
    for index in range(code_snapshot.SNAPSHOT_RETENTION_KEEP + 1):
        filler = snapshot_root / f"2026083{index // 10}T00{index:02d}00Z_deadbeef"
        filler.mkdir()
        (filler / ".snapshot_hash").write_text("deadbeef\n")


def test_retention_never_deletes_the_snapshot_this_submit_selected(tmp_path: Path) -> None:
    _make_remote_checkout(tmp_path)
    stale = "20260101T000000Z"
    cutoff = "20260812T101112Z"
    first = _run_snapshot_script(tmp_path, timestamp=stale, cutoff=cutoff)
    snapshot_root = tmp_path / code_snapshot.SNAPSHOT_DIR_NAME
    _fill_the_retention_window(snapshot_root)
    doomed = snapshot_root / "20260102T000000Z_0badcafe"
    doomed.mkdir()

    second = _run_snapshot_script(tmp_path, timestamp="20260826T101113Z", cutoff=cutoff)

    assert second["reused"] == "1"
    assert second["path"] == first["path"]
    assert Path(second["path"]).is_dir(), "the job is about to import this directory"
    assert second["pruned"] == doomed.name


def test_a_snapshot_that_does_not_match_its_source_is_refused(tmp_path: Path) -> None:
    """The post-copy re-hash is the guard that a hardlinked or partial copy cannot slip through."""
    _make_remote_checkout(tmp_path)
    script = code_snapshot.build_code_snapshot_script(
        remote_repo_root=str(tmp_path), timestamp="20260826T101112Z", cutoff="20260812T101112Z"
    )
    assert 'copied_manifest=$(cd "${snapshot_dir}"' in script
    assert 'if [[ "${copied_manifest}" != "${manifest}" ]]; then' in script

    sabotaged = script.replace(
        '  if [[ -e "${snapshot_dir}/pyproject.toml" ]]; then',
        "  printf 'tampered\\n' > \"${snapshot_dir}/src/placecell_research/loop.py\"\n"
        '  if [[ -e "${snapshot_dir}/pyproject.toml" ]]; then',
    )
    assert sabotaged != script
    completed = subprocess.run(["bash", "-c", sabotaged], capture_output=True, text=True)
    assert completed.returncode == 1
    assert "does not match the source it was copied from" in completed.stderr
