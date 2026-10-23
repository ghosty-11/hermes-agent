"""Opaque, create-only rollback commitments in the installed Git repository.

The capsule supplies bytes, never the expected digest. Only the explicitly selected
installed checkout supplies the local ref namespace; old uncommitted capsules refuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

from hermes_cli.update_local import local_git

_ID = re.compile(r"[0-9a-f]{32}\Z")


def _ref(manifest: Path, phase: str) -> str:
    if not _ID.fullmatch(manifest.parent.name) or manifest.name != "manifest.json":
        raise ValueError("invalid rollback attestation identity")
    return f"refs/hermes/local-update-attestations/{manifest.parent.name}/{phase}"


def _payload(data: dict, phase: str) -> bytes:
    if phase == "capture":
        data = {key: value for key, value in data.items() if key not in {
            "state", "new_selections", "post_head", "post_branch", "post_branch_object",
        }}
        data = {**data, "roots": [
            {key: value for key, value in record.items() if key not in {"post", "post_metadata"}}
            for record in data["roots"]
        ]}
    else:
        data = {**data, "state": "sealed"}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("utf-8")).hexdigest().encode("ascii")


def _git_directory(install: Path) -> Path:
    directory = Path(local_git(install, "rev-parse", "--path-format=absolute",
                               "--git-common-dir").stdout.strip())
    return directory.resolve(strict=True)


def _sync(path: Path) -> None:
    if os.name == "nt" and path.is_dir():
        raise RuntimeError("rollback attestation requires durable directory publication")
    fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_capsule_tree(capsule_fd: int) -> None:
    """Flush only the held, non-following capsule tree before committing its ref."""
    def sync_directory(directory_fd: int, *, root: bool = False) -> None:
        for name in os.listdir(directory_fd):
            if root and name.isdecimal():
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory_fd)
                except OSError as exc:
                    raise ValueError("rollback capsule contains an unsafe entry") from exc
                try:
                    if not stat.S_ISDIR(os.fstat(fd).st_mode):
                        raise ValueError("rollback capsule contains an unsafe entry")
                    sync_directory(fd)
                finally:
                    os.close(fd)
            elif not root or name in {"code.bundle", "manifest.json"}:
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory_fd)
                except OSError as exc:
                    raise ValueError("rollback capsule contains an unsafe entry") from exc
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise ValueError("rollback capsule contains an unsafe entry")
                    os.fsync(fd)
                finally:
                    os.close(fd)
            else:
                raise ValueError("rollback capsule contains an unsafe entry")
        os.fsync(directory_fd)

    sync_directory(capsule_fd, root=True)


def publish(install: Path, manifest: Path, data: dict, phase: str) -> None:
    ref = _ref(manifest, phase)
    payload = _payload(data, phase)
    oid = local_git(install, "hash-object", "-w", "--stdin", binary=True,
                    input=payload).stdout.strip().decode("ascii")
    git_dir = _git_directory(install)
    loose = git_dir / "objects" / oid[:2] / oid[2:]
    _sync(loose)
    _sync(loose.parent)
    _sync(loose.parent.parent)
    local_git(install, "update-ref", ref, oid, "0" * len(oid))
    ref_file = git_dir / ref
    if not ref_file.is_file():
        raise RuntimeError("rollback attestation ref was not durably published")
    _sync(ref_file)
    directory = ref_file.parent
    while directory != git_dir:
        _sync(directory)
        directory = directory.parent
    _sync(git_dir)


def sync_restored_git(install: Path, branch: str | None) -> None:
    """Harden the final ref and worktree HEAD before publishing completion."""
    common = _git_directory(install)
    worktree = Path(local_git(install, "rev-parse", "--path-format=absolute",
                              "--git-dir").stdout.strip()).resolve(strict=True)

    def sync_entry(entry: Path, root: Path) -> None:
        if not entry.is_file() or entry.is_symlink() or not entry.is_relative_to(root):
            raise RuntimeError(f"restored Git reference is not a regular file: {entry}")
        _sync(entry)
        directory = entry.parent
        while directory != root:
            _sync(directory)
            directory = directory.parent
        _sync(root)

    if branch:
        if not branch.startswith("refs/heads/") or ".." in Path(branch).parts:
            raise ValueError("invalid original Git branch")
        loose = common / branch
        # An unchanged branch can still live only in packed-refs.
        sync_entry(loose if loose.is_file() else common / "packed-refs", common)
    sync_entry(worktree / "HEAD", worktree)
    if worktree.is_relative_to(common) and worktree != common:
        directory = worktree.parent
        while directory != common:
            _sync(directory)
            directory = directory.parent
        _sync(common)


def verify(install: Path, manifest: Path, data: dict, phase: str) -> None:
    ref = _ref(manifest, phase)
    result = local_git(install, "show-ref", "--verify", "--hash", ref, check=False)
    if result.returncode or not result.stdout.strip():
        raise ValueError(f"rollback {phase} attestation is missing")
    actual = local_git(install, "cat-file", "blob", result.stdout.strip(), binary=True).stdout
    if actual != _payload(data, phase):
        raise ValueError(f"rollback {phase} attestation does not match capsule")
