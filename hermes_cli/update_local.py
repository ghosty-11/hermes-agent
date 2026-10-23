"""Exact, offline Git transfer for the native source update transaction."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess


_SHA = re.compile(r"[0-9a-f]{40}\Z")

def local_git(root: Path, *args: str, check: bool = True,
              binary: bool = False, input: str | bytes | None = None) -> subprocess.CompletedProcess:
    """Use only local objects and file transport; never run repository hooks."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_ALLOW_PROTOCOL="file")
    if binary and isinstance(input, str) or not binary and isinstance(input, bytes):
        raise TypeError("Git input must match its output mode")
    from pm.writer_claim import child_spawn

    with child_spawn(env) as spawn:
        proc = subprocess.run(
            ["git", "-c", "protocol.allow=never", "-c", "protocol.file.allow=always",
             "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
             "-c", "credential.helper=", "-c", "submodule.recurse=false",
             # Auto gc/maintenance detaches and outlives the owned writer.
             "-c", "maintenance.auto=false", "-c", "gc.auto=0",
             "-C", str(root), *args],
            env=env, input=input, stdin=subprocess.DEVNULL if input is None else None,
            capture_output=True, text=not binary, **spawn)
    if check and proc.returncode:
        stderr = proc.stderr.decode("utf-8", "replace") if binary else proc.stderr
        stdout = proc.stdout.decode("utf-8", "replace") if binary else proc.stdout
        raise RuntimeError(f"local Git {args[0]} failed: {stderr.strip() or stdout.strip()}")
    return proc


def _identity(root: Path) -> str:
    if not root.is_dir():
        raise ValueError(f"not a Git checkout: {root}")
    top = local_git(root, "rev-parse", "--show-toplevel").stdout.strip()
    if Path(top).resolve() != root.resolve():
        raise ValueError(f"not the root of a Git checkout: {root}")
    if local_git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout:
        raise RuntimeError(f"dirty Git checkout: {root}")
    return local_git(root, "rev-parse", "HEAD^{commit}").stdout.strip()


def verify_closure(root: Path, target: str) -> None:
    """Enumerate reachable commits, trees and blobs without promisor acquisition."""
    result = local_git(root, "rev-list", "--objects", "--missing=print", target)
    missing = [line for line in result.stdout.splitlines() if line.startswith("?")]
    if missing:
        raise RuntimeError(f"incomplete local object closure for {target}: {missing[0]}")


def prepare_local_source(install_root: Path, source_repo: Path, target: str) -> str:
    """Validate both roots and exact descendant before transferring any object."""
    if not _SHA.fullmatch(target):
        raise ValueError("--target requires a full lowercase 40-hex commit ID")
    install_root, source_repo = install_root.resolve(), source_repo.resolve()
    before = _identity(install_root)
    _identity(source_repo)
    if local_git(source_repo, "rev-parse", "--verify", f"{target}^{{commit}}", check=False).stdout.strip() != target:
        raise RuntimeError("target commit is not present in reviewed source")
    if local_git(source_repo, "merge-base", "--is-ancestor", before, target, check=False).returncode != 0:
        raise RuntimeError("target is not a descendant of installed HEAD")
    verify_closure(source_repo, target)
    return before


def transfer_local_source(install_root: Path, source_repo: Path, target: str) -> None:
    """Refetch the complete closure without moving tracking refs or negotiating thin packs."""
    local_git(install_root, "fetch", "--refetch", "--no-filter", "--no-tags",
              "--no-recurse-submodules", str(source_repo.resolve()), target)
    verify_closure(install_root, target)


def refuse_active_backups(home: Path) -> None:
    """Refuse contended profile backup locks without creating lockfiles."""
    from hermes_cli.local_update_rollback import _homes
    from pm.filesystem import lock_fd

    for profile in _homes(home):
        lock = profile / ".backup.lock"
        if lock.is_symlink():
            raise RuntimeError(f"backup lock is a symlink: {lock}")
        if not lock.exists():
            continue
        with lock.open("r+b") as handle:
            if not lock_fd(handle.fileno(), wait=False):
                raise RuntimeError(f"backup writer still owns {lock}")


def apply_local_source(install_root: Path, source_repo: Path, target: str) -> str:
    """Advance the current branch only by fast-forward; no stash, reset or branch switch."""
    before = prepare_local_source(install_root, source_repo, target)
    source_head = _identity(source_repo)
    transfer_local_source(install_root, source_repo, target)
    if _identity(install_root) != before or _identity(source_repo) != source_head:
        raise RuntimeError("checkout changed during local transfer")
    if before != target:
        local_git(install_root, "merge", "--ff-only", target)
    if local_git(install_root, "rev-parse", "HEAD").stdout.strip() != target:
        raise RuntimeError("local checkout did not reach exact target")
    return before
