"""Installation-scoped, kernel-held exclusion for cooperating update writers.

The home marker is a status hint for older UI readers, never lock authority.
Older installed clients do not participate; first activation needs an external
writer hold until every client has been replaced. Non-POSIX hosts keep
upstream's marker lock unchanged (``hermes_cli._update_lock_marker``).
"""

from __future__ import annotations

import logging
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MARKER_NAME = ".hermes-update-in-progress"

# Retained for the old handoff writer's import surface; never lock authority.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Exit code meaning "another updater/instance owns this install right now" — the same
# contract as the Windows shim / venv-holder guards in _cmd_update_impl, matched by the
# Tauri updater (UPDATE_EXIT_CONCURRENT in update.rs) to show "Hermes is still running".
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Uses the *process* Hermes home (never the context-local profile override): the Rust
    updater resolves ``$HERMES_HOME`` or the platform default and the desktop pins that same
    value into the updater's env, so a profile-scoped path would be one the other owners never look at.
    """
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home() / MARKER_NAME


def _pid_alive(pid: int) -> bool:
    """Use the dependency-free, Windows-safe probe before PM is available."""
    if pid <= 0:
        return False
    try:
        from hermes_cli._early_recovery import _pid_is_running
        return _pid_is_running(pid)
    except Exception as exc:
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return False



@dataclass(frozen=True)
class UpdateHolder:
    """Status hint from the update marker, not evidence of kernel ownership."""

    pid: int
    age_seconds: float


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Read the display marker without modifying it or deciding ownership."""
    if os.name != "posix":
        from hermes_cli._update_lock_marker import read_live_update as marker_read_live_update
        return marker_read_live_update(path=path)
    marker = path or update_marker_path()
    try:
        lines = marker.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    try:
        pid = int(lines[0].strip())
        started_at = float(lines[1].strip())
    except (IndexError, ValueError):
        return None
    if not math.isfinite(started_at) or not _pid_alive(pid):
        return None
    return UpdateHolder(pid=pid, age_seconds=max(0, time.time() - started_at))


def describe_holder(holder: UpdateHolder | None) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    minutes, seconds = divmod(int(max(0 if holder is None else holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    who = f", process {holder.pid}" if holder else ""
    return (
        f"✗ Another Hermes update is already running (started {elapsed} ago{who}).\n"
        "\n"
        "  Running two at once would corrupt the install. Wait for it to finish\n"
        "  (watch `hermes logs`), or close the Desktop/dashboard window that\n"
        "  started it, then run `hermes update` again."
    )


class UpdateLock:
    """Hold all three writer domains until completion and the terminal receipt."""

    def __init__(self, *, path: Path | None = None, install_root: Path | None = None) -> None:
        from pm.paths import install_root as current_install

        self.path = path or update_marker_path()
        self.install_root = Path(install_root or current_install()).resolve()
        self.acquired = False
        self.holder: UpdateHolder | None = None
        self._claim = None
        self._published = False

    def acquire(self) -> bool:
        from pm.writer_claim import joined_claim, writer_claim

        if self.acquired:
            raise RuntimeError("update lock is already held")
        claim = writer_claim(install=self.install_root)
        try:
            claim.__enter__()
        except BlockingIOError:
            self.holder = read_live_update(path=self.path)
            return False
        self._claim = claim
        try:
            if joined_claim():
                # Inherited writer authority is not provenance for the completion tail.
                # The marker is only a display hint, including when it has disappeared.
                self.acquired = True
                return True
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_symlink():
                raise RuntimeError("update status path is a symlink")
            status_fd = os.open(self.path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                identity = os.fstat(status_fd)
                if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                    raise RuntimeError("update status is not a standalone regular file")
                os.ftruncate(status_fd, 0)
                os.write(status_fd, f"{os.getpid()}\n{int(time.time())}\n".encode("ascii"))
                os.fsync(status_fd)
            finally:
                os.close(status_fd)
            self._published = True
            self.acquired = True
            return True
        except BaseException:
            claim.__exit__(None, None, None)
            self._claim = None
            raise

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self._published:
                try:
                    owner = int(self.path.read_text(encoding="utf-8-sig").splitlines()[0])
                    if owner == os.getpid():
                        self.path.unlink(missing_ok=True)
                except (OSError, IndexError, ValueError):
                    pass
        finally:
            self._claim.__exit__(None, None, None)
            self._claim = None
            self._published = False
            self.acquired = False

    def __enter__(self) -> "UpdateLock":
        if not self.acquire():
            raise RuntimeError(describe_holder(self.holder))
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


if os.name != "posix":
    from hermes_cli._update_lock_marker import UpdateLock  # noqa: F811 - upstream marker lock
