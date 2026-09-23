"""Node-local stage-in for dataset artifacts."""

from __future__ import annotations

import atexit
import hashlib
import multiprocessing
import os
import shutil
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

STAGE_DIR_ENV = "PLACECELL_DATASET_STAGE_DIR"

ARCHIVED_ENTRY_NAMES = ("dataset.zarr", "manifest_summary.json")

_LOG_PREFIX = "[dataset-staging]"
_SENTINEL_NAME = ".stage_complete"
_PAYLOAD_NAME = "data"
_DIGEST_SUFFIX = ".digest"
_READ_BLOCK_BYTES = 1 << 20
_ZARR_ROOT_METADATA_NAMES = (".zgroup", "zarr.json")
_ARCHIVE_NICENESS = 10

_staged_directories: dict[Path, Path] = {}
_owned_paths: set[Path] = set()
_owner_pid = os.getpid()


def dataset_archive_path(artifact_dir: Path) -> Path:
    """Sidecar archive for a dataset artifact, named after the artifact id."""
    artifact_dir = Path(artifact_dir)
    return artifact_dir.with_name(f"{artifact_dir.name}.tar")


def dataset_archive_digest_path(artifact_dir: Path) -> Path:
    """Content digest of everything the sidecar archive holds."""
    archive_path = dataset_archive_path(artifact_dir)
    return archive_path.with_name(f"{archive_path.name}{_DIGEST_SUFFIX}")


def create_dataset_archive(artifact_dir: Path, *, overwrite: bool = False) -> Path:
    """Write the sidecar tar next to a dataset artifact."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise NotADirectoryError(f"{artifact_dir} is not a dataset artifact directory.")
    entry_names = [name for name in ARCHIVED_ENTRY_NAMES if (artifact_dir / name).exists()]
    if "dataset.zarr" not in entry_names:
        raise FileNotFoundError(f"{artifact_dir} contains no dataset.zarr to archive.")
    archive_path = dataset_archive_path(artifact_dir)
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"{archive_path} already exists; pass overwrite=True to replace it.")
    temp_path = archive_path.with_name(f"{archive_path.name}.tmp.{os.getpid()}")
    try:
        entries: list[tuple[str, int, str]] = []
        with tarfile.open(temp_path, "w") as archive:
            for source_path, member_name in _archive_sources(artifact_dir, entry_names):
                entries.append(_add_archive_member(archive, source_path, member_name))
        _write_atomically(dataset_archive_digest_path(artifact_dir), _content_digest(entries))
        os.replace(temp_path, archive_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return archive_path


def backfill_dataset_archive_digest(artifact_dir: Path, *, overwrite: bool = False) -> Path:
    """Write the digest sidecar for a tar that already exists, by reading it back."""
    artifact_dir = Path(artifact_dir)
    archive_path = dataset_archive_path(artifact_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(f"no archive at {archive_path}; use create_dataset_archive.")
    digest_path = dataset_archive_digest_path(artifact_dir)
    if digest_path.exists() and not overwrite:
        raise FileExistsError(f"{digest_path} already exists; pass overwrite=True to replace it.")
    entries: list[tuple[str, int, str]] = []
    with tarfile.open(archive_path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            hasher = hashlib.sha256()
            for block in _read_blocks(source):
                hasher.update(block)
            entries.append((member.name, member.size, hasher.hexdigest()))
    _write_atomically(digest_path, _content_digest(entries))
    return digest_path


def _archive_sources(artifact_dir: Path, entry_names: list[str]) -> Iterator[tuple[Path, str]]:
    """(source file, member name) for every regular file the archive holds."""
    for name in entry_names:
        source = artifact_dir / name
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in candidates:
            if path.is_dir():
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{path} is not a regular file and cannot be archived.")
            yield path, str(path.relative_to(artifact_dir))


def _add_archive_member(
    archive: tarfile.TarFile,
    source_path: Path,
    member_name: str,
) -> tuple[str, int, str]:
    member = archive.gettarinfo(str(source_path), arcname=member_name)
    hasher = hashlib.sha256()
    with source_path.open("rb") as source_file:
        archive.addfile(member, _HashingReader(source_file, hasher))
    return member_name, member.size, hasher.hexdigest()


class _HashingReader:
    """File wrapper that hashes what it hands out, so writing the tar also digests it."""

    def __init__(self, source_file, hasher) -> None:
        self._source_file = source_file
        self._hasher = hasher

    def read(self, size: int = -1) -> bytes:
        block = self._source_file.read(size)
        self._hasher.update(block)
        return block


def _content_digest(entries: list[tuple[str, int, str]]) -> str:
    """One digest over member paths, sizes, and contents, independent of traversal order."""
    total = hashlib.sha256()
    for member_name, size, member_digest in sorted(entries):
        total.update(f"{member_name}\0{size}\0{member_digest}\n".encode())
    return total.hexdigest()


def _write_atomically(path: Path, text: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(f"{text}\n")
    os.replace(temp_path, path)


def try_create_dataset_archive(artifact_dir: Path, artifact_type: str) -> Path | None:
    """Write the stage-in sidecar for a freshly published encoded dataset, or give up quietly."""
    artifact_dir = Path(artifact_dir)
    archive_path = dataset_archive_path(artifact_dir)
    if artifact_type != "encoded_dataset":
        return None
    if archive_path.exists() or not (artifact_dir / "dataset.zarr").is_dir():
        return None
    try:
        _write_archive_in_niced_child(artifact_dir)
        archive_gigabytes = archive_path.stat().st_size / 1024**3
    except Exception as error:
        print(
            f"{_LOG_PREFIX} WARNING could not write {archive_path.name} ({error}); "
            "jobs will read this dataset from the shared filesystem",
            flush=True,
        )
        return None
    print(
        f"{_LOG_PREFIX} wrote {archive_path.name} "
        f"({archive_gigabytes:.2f} GB) for node-local stage-in",
        flush=True,
    )
    return archive_path


def _write_archive_in_niced_child(artifact_dir: Path) -> None:
    """Run the archive write in a spawned child so its niceness cannot leak into the pipeline."""
    process = multiprocessing.get_context("spawn").Process(
        target=_archive_child_entry_point,
        args=(str(artifact_dir),),
    )
    process.start()
    child_pid = process.pid
    process.join()
    archive_path = dataset_archive_path(artifact_dir)
    digest_path = dataset_archive_digest_path(artifact_dir)
    archive_path.with_name(f"{archive_path.name}.tmp.{child_pid}").unlink(missing_ok=True)
    digest_path.with_name(f".{digest_path.name}.tmp.{child_pid}").unlink(missing_ok=True)
    if process.exitcode != 0:
        raise RuntimeError(f"archive subprocess exited with status {process.exitcode}")


def _archive_child_entry_point(artifact_dir: str) -> None:
    try:
        os.nice(_ARCHIVE_NICENESS)
    except OSError as error:
        print(f"{_LOG_PREFIX} could not lower archive niceness ({error})", flush=True)
    create_dataset_archive(Path(artifact_dir))


def stage_dataset_dir(artifact_dir: Path) -> Path:
    """Return a node-local copy of a dataset artifact, or the original path on any miss."""
    artifact_dir = Path(artifact_dir)
    stage_root = os.environ.get(STAGE_DIR_ENV, "").strip()
    if not stage_root:
        return artifact_dir
    cache_key = artifact_dir.resolve()
    cached_dir = _staged_directories.get(cache_key)
    if cached_dir is not None:
        return cached_dir
    try:
        staged_dir = _stage_artifact(cache_key, Path(stage_root))
        print(f"{_LOG_PREFIX} reading {artifact_dir.name} from {staged_dir}", flush=True)
    except Exception as error:
        print(
            f"{_LOG_PREFIX} WARNING could not stage {artifact_dir.name} into {stage_root} "
            f"({error}); reading from {artifact_dir}",
            flush=True,
        )
        staged_dir = artifact_dir
    _staged_directories[cache_key] = staged_dir
    return staged_dir


def _stage_artifact(artifact_dir: Path, stage_root: Path) -> Path:
    archive_path = dataset_archive_path(artifact_dir)
    if not archive_path.is_file():
        raise FileNotFoundError(f"no sidecar archive at {archive_path}")
    stage_root.mkdir(parents=True, exist_ok=True)
    target_dir = stage_root / f"pc_dataset_{_job_key()}_{artifact_dir.name}"
    expected_digest = _expected_archive_digest(artifact_dir)
    if (target_dir / _SENTINEL_NAME).is_file():
        _verify_published_copy(artifact_dir, target_dir, expected_digest)
        return target_dir / _PAYLOAD_NAME
    _raise_if_stage_root_too_small(stage_root, archive_path.stat().st_size)

    temp_dir = stage_root / f"{target_dir.name}.tmp.{os.getpid()}"
    shutil.rmtree(temp_dir, ignore_errors=True)
    _take_ownership(temp_dir)
    try:
        staged_digest = _extract_archive(archive_path, temp_dir / _PAYLOAD_NAME)
        _raise_on_digest_mismatch(staged_digest, expected_digest, temp_dir / _PAYLOAD_NAME)
        _verify_staged_copy(artifact_dir, temp_dir / _PAYLOAD_NAME)
        (temp_dir / _SENTINEL_NAME).write_text(f"{artifact_dir}\n{staged_digest}\n")
        os.rename(temp_dir, target_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        _owned_paths.discard(temp_dir)
        if (target_dir / _SENTINEL_NAME).is_file():
            return target_dir / _PAYLOAD_NAME
        raise
    _owned_paths.discard(temp_dir)
    _take_ownership(target_dir)
    return target_dir / _PAYLOAD_NAME


def _job_key() -> str:
    job_id = (os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "").strip()
    key = job_id or f"pid{os.getpid()}"
    return "".join(character if character.isalnum() else "_" for character in key)


def _take_ownership(path: Path) -> None:
    if not _owned_paths:
        atexit.register(remove_staged_copies)
    _owned_paths.add(path)


def remove_staged_copies() -> None:
    """Delete every node-local copy this process staged."""
    if os.getpid() != _owner_pid:
        return
    while _owned_paths:
        shutil.rmtree(_owned_paths.pop(), ignore_errors=True)


def _free_bytes(path: Path) -> int:
    filesystem = os.statvfs(str(path))
    free_bytes = int(filesystem.f_bavail) * int(filesystem.f_frsize)
    if _is_memory_backed(path):
        free_bytes = min(free_bytes, _cgroup_memory_headroom_bytes())
    return free_bytes


def _is_memory_backed(path: Path) -> bool:
    """True when path lives on tmpfs or ramfs, per the process mount table."""
    resolved = str(path.resolve())
    best_mount_point = ""
    best_filesystem_type = ""
    try:
        mount_table = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return False
    for line in mount_table:
        head, _, tail = line.partition(" - ")
        fields = head.split()
        separator_fields = tail.split()
        if len(fields) < 5 or not separator_fields:
            continue
        mount_point = fields[4]
        if resolved == mount_point or resolved.startswith(mount_point.rstrip("/") + "/"):
            if len(mount_point) >= len(best_mount_point):
                best_mount_point = mount_point
                best_filesystem_type = separator_fields[0]
    return best_filesystem_type in {"tmpfs", "ramfs"}


def _cgroup_memory_headroom_bytes() -> int:
    """Bytes still available under this job's cgroup memory limit, or "unlimited"."""
    try:
        cgroup_text = Path("/proc/self/cgroup").read_text()
    except OSError:
        return sys.maxsize
    for limit_path, usage_path in _cgroup_memory_files(cgroup_text):
        try:
            limit_text = limit_path.read_text().strip()
            usage_text = usage_path.read_text().strip()
        except OSError:
            continue
        if limit_text == "max":
            return sys.maxsize
        return max(0, int(limit_text) - int(usage_text))
    return sys.maxsize


def _cgroup_memory_files(cgroup_text: str) -> Iterator[tuple[Path, Path]]:
    """(limit, usage) file pairs for the memory cgroup this process belongs to."""
    for entry in cgroup_text.splitlines():
        fields = entry.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, relative_path = fields[1], fields[2].lstrip("/")
        if not controllers:
            root = Path("/sys/fs/cgroup") / relative_path
            yield root / "memory.max", root / "memory.current"
        elif "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory") / relative_path
            yield root / "memory.limit_in_bytes", root / "memory.usage_in_bytes"


def _raise_if_stage_root_too_small(stage_root: Path, archive_bytes: int) -> None:
    free_bytes = _free_bytes(stage_root)
    required_bytes = int(archive_bytes * 1.1)
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"{stage_root} has {free_bytes / 1024**3:.1f} GB free but the staged copy needs "
            f"about {required_bytes / 1024**3:.1f} GB"
        )


def _extract_archive(archive_path: Path, destination: Path) -> str:
    """Unpack every member and return the content digest of what was written."""
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, int, str]] = []
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = member.name.removeprefix("./")
            target_path = _member_target_path(destination, member_name)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            hasher = hashlib.sha256()
            source_file = archive.extractfile(member)
            with target_path.open("wb") as target_file:
                for block in _read_blocks(source_file):
                    hasher.update(block)
                    target_file.write(block)
            entries.append((member_name, target_path.stat().st_size, hasher.hexdigest()))
    return _content_digest(entries)


def _member_target_path(destination: Path, member_name: str) -> Path:
    """Where one member lands, refusing any name that would escape the staging directory."""
    target_path = (destination / member_name).resolve()
    if destination.resolve() not in target_path.parents:
        raise ValueError(f"archive member {member_name!r} escapes {destination}")
    return target_path


def _expected_archive_digest(artifact_dir: Path) -> str:
    digest_path = dataset_archive_digest_path(artifact_dir)
    if not digest_path.is_file():
        raise FileNotFoundError(f"no archive content digest at {digest_path}")
    return digest_path.read_text().strip()


def _raise_on_digest_mismatch(actual_digest: str, expected_digest: str, staged_dir: Path) -> None:
    if actual_digest != expected_digest:
        raise ValueError(
            f"content digest of {staged_dir} is {actual_digest[:12]} but the archive records "
            f"{expected_digest[:12]}"
        )


def _verify_published_copy(artifact_dir: Path, target_dir: Path, expected_digest: str) -> None:
    """Re-check a copy this job already published: the sentinel alone proves nothing about it."""
    staged_dir = target_dir / _PAYLOAD_NAME
    sentinel_lines = (target_dir / _SENTINEL_NAME).read_text().splitlines()
    recorded_digest = sentinel_lines[1].strip() if len(sentinel_lines) > 1 else ""
    _raise_on_digest_mismatch(recorded_digest, expected_digest, target_dir / _SENTINEL_NAME)
    _raise_on_digest_mismatch(_directory_digest(staged_dir), expected_digest, staged_dir)
    _verify_staged_copy(artifact_dir, staged_dir)


def _directory_digest(staged_dir: Path) -> str:
    entries = [
        (
            str(path.relative_to(staged_dir)),
            path.stat().st_size,
            _file_digest(path),
        )
        for path in sorted(staged_dir.rglob("*"))
        if path.is_file()
    ]
    return _content_digest(entries)


def _verify_staged_copy(artifact_dir: Path, staged_dir: Path) -> None:
    """Byte-compare the Zarr root metadata against the artifact the archive was made from."""
    relative_paths = [
        f"dataset.zarr/{name}"
        for name in _ZARR_ROOT_METADATA_NAMES
        if (artifact_dir / "dataset.zarr" / name).is_file()
    ]
    if not relative_paths:
        raise FileNotFoundError(f"{artifact_dir / 'dataset.zarr'} has no Zarr root metadata")
    for relative_path in relative_paths:
        if _file_digest(artifact_dir / relative_path) != _file_digest(staged_dir / relative_path):
            raise ValueError(f"staged copy differs from the artifact at {relative_path}")


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source_file:
        for block in _read_blocks(source_file):
            hasher.update(block)
    return hasher.hexdigest()


def _read_blocks(source_file) -> Iterator[bytes]:
    while True:
        block = source_file.read(_READ_BLOCK_BYTES)
        if not block:
            return
        yield block
