"""Import-light, retained kernel claims for installed code and dependency writers.

An environment value names a handed-off descriptor, never grants ownership. A child
joins only after proving the descriptor owns the expected, independently contended file.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import threading
from contextlib import contextmanager


_CLAIMS = "HERMES_RETAINED_WRITER_CLAIMS"
_local = threading.local()
_outer_lock = threading.Lock()


def domains(install: Path | None = None) -> tuple[Path, Path, Path]:
    from pm.environments import dependency_home_root
    from pm.paths import install_root, writable_store_root

    root = Path(install or install_root()).resolve()
    if (root / ".git").is_symlink():
        raise RuntimeError("installation Git metadata must not be a symlink")
    from hermes_cli._early_recovery import _git_dir
    git = (root / ".git").exists()
    checkout = _git_dir(root) if git else dependency_home_root().resolve() / "install-claims"
    identity = hashlib.sha256(os.fsencode(root)).hexdigest()
    installation = checkout.resolve() / ("hermes-native-update.lock" if git else f"{identity}.lock")
    home = dependency_home_root().resolve()
    store = writable_store_root(root).resolve()
    return (installation, home / ".hermes-dependency-writer.lock",
            store.parent / "logs" / ".hermes-pm-transaction.lock")


def _open(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    identity = os.fstat(fd)
    if (path.is_symlink() or not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1
            or stat.S_IMODE(identity.st_mode) != 0o600
            or (identity.st_dev, identity.st_ino) != (path.stat().st_dev, path.stat().st_ino)):
        os.close(fd)
        raise RuntimeError(f"writer claim is not a standalone regular file: {path}")
    return fd



def _claim(path: Path) -> int:
    fd = _open(path)
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _join(path: Path, fd: int) -> None:
    owned = os.fstat(fd)
    if (not stat.S_ISREG(owned.st_mode) or owned.st_nlink != 1
            or stat.S_IMODE(owned.st_mode) != 0o600):
        raise RuntimeError("inherited writer claim is not a standalone file")
    if path.is_symlink() or (owned.st_dev, owned.st_ino) != (path.stat().st_dev, path.stat().st_ino):
        raise RuntimeError("inherited writer claim names a different inode")
    import fcntl
    probe = _open(path)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("inherited writer claim is not held")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another owner holds the writer claim") from exc
    finally:
        os.close(probe)


def _inherited(expected: tuple[Path, ...]) -> tuple[int, ...] | None:
    value = os.environ.get(_CLAIMS)
    if not value:
        return None
    data = json.loads(value)
    if data["paths"] != [str(p) for p in expected] or len(data["fds"]) != len(expected):
        raise RuntimeError("writer claim domains differ from inherited ownership")
    fds = []
    try:
        for fd in data["fds"]:
            fds.append(os.dup(int(fd)))
        for path, fd in zip(expected, fds):
            _join(path, fd)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise
    return tuple(fds)


def claim_domains() -> tuple[Path, Path, Path]:
    """The current thread's claim paths, including an explicitly selected install."""
    active = getattr(_local, "active", None)
    return active[0] if active is not None else domains()


@contextmanager
def writer_claim(*, install: Path | None = None, inherited_only: bool = False):
    if os.name != "posix":
        yield
        return
    active = getattr(_local, "active", None)
    if active is not None:
        # A private staging store never retargets the original dependency claim.
        expected = active[0] if install is None else domains(install)
    else:
        # The worker's checkout can differ from the install selected by its
        # parent. Recompute every domain from that install before joining.
        if install is None and inherited_only and os.environ.get(_CLAIMS):
            install = Path(json.loads(os.environ[_CLAIMS])["install"])
        if install is None:
            from pm.paths import install_root
            install = install_root()
        install = Path(install).resolve()
        expected = domains(install)
    if active is not None:
        if active[0] != expected:
            raise RuntimeError("writer claim cannot change domains within an operation")
        active[2] += 1
        try:
            yield
        finally:
            active[2] -= 1
        return
    _outer_lock.acquire()
    try:
        with _outer_claim(expected, inherited_only, install):
            yield
    finally:
        _outer_lock.release()


@contextmanager
def _outer_claim(expected: tuple[Path, ...], inherited_only: bool, install: Path):
    inherited = _inherited(expected)
    if inherited is None and inherited_only:
        raise RuntimeError("mutating child received no retained writer claim")
    fds = []
    if inherited is None:
        try:
            for path in expected:
                fds.append(_claim(path))
        except BaseException:
            for fd in reversed(fds):
                os.close(fd)
            raise
    else:
        fds = list(inherited)
    previous = os.environ.get(_CLAIMS)
    os.environ[_CLAIMS] = json.dumps({
        "install": str(install),
        "paths": [str(p) for p in expected],
        "fds": fds,
    })
    _local.active = [expected, tuple(fds), 1, inherited is not None, install]
    try:
        yield
    finally:
        _local.active = None
        if previous is None:
            os.environ.pop(_CLAIMS, None)
        else:
            os.environ[_CLAIMS] = previous
        # Joined handles belong to this process as well; close, never unlock an OFD.
        for fd in reversed(fds):
            os.close(fd)


def joined_claim() -> bool:
    if os.name != "posix":
        return False
    active = getattr(_local, "active", None)
    return active is not None and active[3]


def has_writer_claim() -> bool:
    if os.name != "posix":
        return False
    return getattr(_local, "active", None) is not None


LIVENESS_ENV = "HERMES_OWNED_LIVENESS_FDS"


def inherited_liveness_fds() -> list[int]:
    """Open liveness pipe write ends this process inherited from owned ancestors."""
    if os.name != "posix":
        return []
    fds = []
    for item in os.environ.get(LIVENESS_ENV, "").split(","):
        if not item.isdecimal():
            continue
        try:
            if stat.S_ISFIFO(os.fstat(int(item)).st_mode):
                fds.append(int(item))
        except OSError:
            continue  # an ancestor-advertised descriptor this process does not hold
    return fds


def exec_inheriting_claims(executable: str, argv: list[str]) -> None:
    """An in-place interpreter switch retains its currently held claims."""
    if os.name != "posix":
        raise RuntimeError("writer claim exec requires POSIX")
    active = getattr(_local, "active", None)
    if active is not None:
        for fd in active[1]:
            os.set_inheritable(fd, True)
    for fd in inherited_liveness_fds():
        os.set_inheritable(fd, True)
    os.execv(executable, argv)


@contextmanager
def child_spawn(environment: dict[str, str]):
    """Pass only acquired claims to one bounded child and close transfer copies.

    Inherited liveness sentinels travel with the claims so an owned ancestor can
    still observe this child's tree after it leaves the ancestor's group.
    """
    if os.name != "posix":
        yield {}
        return
    active = getattr(_local, "active", None)
    liveness = inherited_liveness_fds()
    if liveness:
        environment[LIVENESS_ENV] = ",".join(map(str, liveness))
    if active is None:
        yield {"pass_fds": tuple(liveness)} if liveness else {}
        return
    fds = active[1]
    yield {"pass_fds": tuple(dict.fromkeys((*fds, *liveness)))}


def child_kwargs() -> dict:
    """POSIX spawn kwargs for retained claims."""
    if os.name != "posix":
        return {}
    active = getattr(_local, "active", None)
    if active is None:
        return {}
    return {"pass_fds": tuple(dict.fromkeys((*active[1], *inherited_liveness_fds())))}


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if len(sys.argv) != 3 or sys.argv[1] != "--describe-domains":
        raise SystemExit(2)
    print(json.dumps([str(path) for path in domains(Path(sys.argv[2]))]))
