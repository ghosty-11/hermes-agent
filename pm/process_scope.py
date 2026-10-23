"""Owned completion sessions and conservative cleanup of retained staging."""
from __future__ import annotations
from contextlib import contextmanager

import os
from pathlib import Path
import shutil
import tempfile
import signal
import subprocess
import time



def retain_exception(exc: BaseException, paths) -> None:
    """Keep the original cancellation while marking its inputs unsafe to remove."""
    retained = tuple(dict.fromkeys((*getattr(exc, "retained_paths", ()),
                                   *(Path(path) for path in paths))))
    exc.retained_paths = retained
    exc.add_note(f"Retained update inputs: {', '.join(map(str, retained))}")



class ProcessStillRunning(Exception):
    """A child tree has not been proved quiescent; these paths must remain intact."""

    def __init__(self, paths, reason: str):
        self.retained_paths = tuple(Path(path) for path in paths)
        super().__init__(f"{reason}; retained: {', '.join(map(str, self.retained_paths))}")


@contextmanager
def retained_directory(*, prefix: str):
    """A temporary input that cannot be removed while a descendant may write."""
    directory = Path(tempfile.mkdtemp(prefix=prefix))
    safe = True
    try:
        yield str(directory)
    except BaseException as exc:
        if isinstance(exc, ProcessStillRunning):
            safe = False
            if directory not in exc.retained_paths:
                exc.retained_paths += (directory,)
                exc.args = (f"{exc.args[0]}; retained: {directory}",)
        elif getattr(exc, "retained_paths", None) is not None or not isinstance(exc, Exception):
            safe = False
            retain_exception(exc, [directory])
        raise
    finally:
        if safe:
            shutil.rmtree(directory)



def require_child_quiescence(status: Path, retained) -> None:
    """A child may report an escaped writer after its original group exits."""
    if not status.exists():
        return
    try:
        import json
        paths = json.loads(status.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ProcessStillRunning(retained, "child quiescence attestation unreadable") from exc
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ProcessStillRunning(retained, "invalid child quiescence attestation")
    raise ProcessStillRunning([*retained, *paths], "child retained a writer")


def owned_spawn_kwargs() -> dict:
    """Only a child started with these options may be signalled as a group."""
    return {"start_new_session": True}






class LivenessSentinel:
    """Read end of a pipe whose write end every owned descendant inherits.

    EOF arrives only after every process holding the write end has exited or
    closed it, including descendants that left the owned process group, so it
    needs neither group membership nor ``/proc`` access. Residual: a descendant
    that closes inherited descriptors yet keeps writing is invisible here; the
    retained writer claim, not this sentinel, remains the exclusion boundary.
    """

    def __init__(self, read_fd: int):
        self.read_fd = read_fd

    def drained(self, timeout: float) -> bool:
        import select

        if self.read_fd < 0:
            return False  # already consumed; a second probe proves nothing
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            ready, _, _ = select.select([self.read_fd], [], [], remaining)
            if ready and os.read(self.read_fd, 4096) == b"":
                return True

    def close(self) -> None:
        if self.read_fd >= 0:
            os.close(self.read_fd)
            self.read_fd = -1


class OwnedScope:
    """Own one child from spawn through tree quiescence.

    A child created before Popen returns cannot be published to this scope.
    If its inherited liveness sentinel remains held, inputs are retained
    rather than treating that publication window as a successful release.
    """

    def __init__(self, spawn: dict, environment: dict, retained=()):
        if os.name != "posix":
            raise RuntimeError("owned process scope requires POSIX")
        self.retained = list(dict.fromkeys(Path(path) for path in retained))
        self.proc: subprocess.Popen | None = None
        self.released = False
        self.write_fd = None
        from pm.writer_claim import LIVENESS_ENV, inherited_liveness_fds

        read_fd, write_fd = os.pipe()
        self.liveness = LivenessSentinel(read_fd)
        self.write_fd = write_fd
        carried = [*inherited_liveness_fds(), write_fd]
        environment[LIVENESS_ENV] = ",".join(map(str, carried))
        self.kwargs = {**spawn, **owned_spawn_kwargs(),
                       "pass_fds": tuple(dict.fromkeys((*spawn.get("pass_fds", ()), *carried)))}

    def __enter__(self) -> "OwnedScope":
        return self

    def spawn(self, args, **popen_kwargs) -> subprocess.Popen:
        if self.proc is not None or self.released:
            raise RuntimeError("owned scope accepts one child")
        self.proc = subprocess.Popen(args, **popen_kwargs, **self.kwargs)
        return self.proc

    def retain(self, paths) -> None:
        self.retained.extend(path for path in (Path(p) for p in paths)
                             if path not in self.retained)

    def _close_writer(self) -> None:
        if self.write_fd is not None:
            fd, self.write_fd = self.write_fd, None
            os.close(fd)

    def _close_pipes(self) -> None:
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass  # Tree proof, not pipe flushing, decides retained inputs.


    def finish(self, *, stop: bool = False) -> None:
        if self.proc is None or self.released:
            raise RuntimeError("owned scope has no unfinished child")
        self._close_writer()
        _quiescent(self.proc, self.retained, stop=stop, liveness=self.liveness)
        self.released = True

    def __exit__(self, exc_type, exc, _tb) -> bool:
        if self.released:
            return False
        if self.proc is None:
            errors = []
            try:
                self._close_writer()
            except BaseException as error:
                errors.append(error)
            proven = False
            if not errors:
                try:
                    proven = self.liveness.drained(10.5)
                except BaseException as error:
                    errors.append(error)
            try:
                self.liveness.close()
            except BaseException as error:
                errors.append(error)
            if not proven:
                retained = tuple(dict.fromkeys((*getattr(exc, "retained_paths", ()),
                                                *self.retained)))
                cancellation = next(
                    (error for error in (exc, *errors) if error is not None and
                     (not isinstance(error, Exception) or
                      type(error).__name__ == "CancelledError")),
                    None,
                )
                if cancellation is not None:
                    retain_exception(cancellation, retained)
                    for error in (exc, *errors):
                        if error is not None and error is not cancellation:
                            cancellation.add_note(f"Unpublished child cleanup unproved: {error!r}")
                    if cancellation is exc:
                        if errors and exc.__cause__ is None:
                            exc.__cause__ = errors[0]
                        return False
                    raise cancellation from exc
                raise ProcessStillRunning(
                    retained, "a child created before publication still holds the liveness sentinel"
                ) from (exc if exc is not None else errors[0] if errors else None)
            if exc is not None:
                for error in errors:
                    exc.add_note(f"Unpublished child cleanup error: {error!r}")
                return False
            if errors:
                raise errors[0]
            raise RuntimeError("owned child not finished")

        errors = []
        try:
            self._close_writer()
        except BaseException as error:
            errors.append(error)
        proven = False
        try:
            _quiescent(self.proc, self.retained, stop=True, liveness=self.liveness)
        except BaseException as error:
            errors.append(error)
        else:
            proven = True
        try:
            self._close_pipes()
        except BaseException as error:
            errors.append(error)
        try:
            self.liveness.close()
        except BaseException as error:
            errors.append(error)

        cancellation = next(
            (error for error in (exc, *errors) if error is not None and
             (not isinstance(error, Exception) or type(error).__name__ == "CancelledError")),
            None,
        )
        if not proven:
            retained = tuple(dict.fromkeys((*getattr(exc, "retained_paths", ()),
                                            *self.retained)))
            if cancellation is not None:
                retain_exception(cancellation, retained)
                for error in (exc, *errors):
                    if error is not None and error is not cancellation:
                        cancellation.add_note(f"Owned child cleanup unproved: {error!r}")
                if cancellation is exc:
                    if errors and exc.__cause__ is None:
                        exc.__cause__ = errors[0]
                    return False
                raise cancellation from exc
            raise ProcessStillRunning(
                retained, f"owned child cleanup unproved: {'; '.join(map(str, errors))}"
            ) from (exc if exc is not None else errors[0])

        self.released = True
        if cancellation is not None:
            if cancellation is exc:
                return False
            for error in errors:
                if error is not cancellation:
                    cancellation.add_note(f"Owned child cleanup error: {error!r}")
            raise cancellation from exc
        if exc is not None:
            for error in errors:
                exc.add_note(f"Owned child cleanup error: {error!r}")
            return False
        if errors:
            raise errors[0]
        raise RuntimeError("owned child not finished")


def owned_spawn(spawn: dict, environment: dict, retained=()) -> OwnedScope:
    """Create one scoped child; environment is the child's POSIX env mapping."""
    return OwnedScope(spawn, environment, retained)

def _quiescent(proc: subprocess.Popen, paths, *, stop: bool = False,
              liveness: "LivenessSentinel | None" = None) -> None:
    """Wait for the owned session, refusing cleanup without positive tree quiescence.

    A stopped leader is never used to authorize signalling a reused group ID.
    An empty POSIX group is insufficient if a descendant escaped it; the inherited
    liveness sentinel must also reach EOF.
    """
    # Popen's unreaped leader pins its PID. Never signal via a PID after wait
    # or poll has reaped it: a new, unrelated group may reuse that number.
    if stop and proc.returncode is None:
        try:
            if os.getpgid(proc.pid) == proc.pid:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise ProcessStillRunning(paths, f"owned group signal was inconclusive: {exc}") from exc
    try:
        proc.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProcessStillRunning(paths, f"owned process did not exit: {exc}") from exc
    deadline = time.monotonic() + 10
    while True:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        except OSError as exc:
            raise ProcessStillRunning(paths, f"owned group probe was inconclusive: {exc}") from exc
        if time.monotonic() >= deadline:
            raise ProcessStillRunning(paths, "owned process group remains live")
        time.sleep(0.05)
    if liveness is None:
        raise ProcessStillRunning(paths, "owned child has no liveness sentinel")
    if not liveness.drained(max(0.0, deadline - time.monotonic()) + 0.5):
        raise ProcessStillRunning(
            paths, "a descendant outside the owned group still holds the liveness sentinel")
    liveness.close()
