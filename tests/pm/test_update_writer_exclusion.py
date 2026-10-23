"""Shared dependency writers cannot interleave with a held native transaction."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from pm.operations import pm_writer_claim

ROOT = Path(__file__).resolve().parents[2]


def test_other_pm_writer_refuses_while_update_owns_shared_state(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    child = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from pm.operations import pm_writer_claim; "
        "\nwith pm_writer_claim(): pass\n"
    )
    environment = {key: value for key, value in os.environ.items()
                   if key != "HERMES_RETAINED_WRITER_CLAIMS"}
    with pm_writer_claim(transaction=True):
        refused = subprocess.run([sys.executable, "-I", "-S", "-c", child],
                                 env=environment, capture_output=True, text=True, timeout=15)
        assert refused.returncode != 0
    admitted = subprocess.run([sys.executable, "-I", "-S", "-c", child],
                              env=environment, capture_output=True, text=True, timeout=15)
    assert admitted.returncode == 0, admitted.stderr


@pytest.mark.platforms("posix")
def test_unpublished_child_interrupt_retains_inputs(tmp_path, monkeypatch):
    import pm.process_scope as process_scope

    retained = tmp_path / "staged-input"
    created = []
    real_popen = subprocess.Popen
    interruption = KeyboardInterrupt()

    def interrupt_before_publication(*args, **kwargs):
        created.append(real_popen(*args, **kwargs))
        raise interruption

    monkeypatch.setattr(process_scope.subprocess, "Popen", interrupt_before_publication)
    env = dict(os.environ)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with process_scope.owned_spawn({}, env, retained=[retained]) as scope:
                scope.spawn(
                    [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                    env=env)
        assert raised.value is interruption
        assert retained in raised.value.retained_paths
        assert len(created) == 1
        assert created[0].poll() is None
    finally:
        for proc in created:
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_unpublished_child_failure_raises_retention_error(tmp_path, monkeypatch):
    import pm.process_scope as process_scope

    retained = tmp_path / "staged-input"
    created = []
    real_popen = subprocess.Popen
    failure = RuntimeError("popen failed after child creation")

    def fail_before_publication(*args, **kwargs):
        created.append(real_popen(*args, **kwargs))
        raise failure

    monkeypatch.setattr(process_scope.subprocess, "Popen", fail_before_publication)
    env = dict(os.environ)
    try:
        with pytest.raises(process_scope.ProcessStillRunning) as raised:
            with process_scope.owned_spawn({}, env, retained=[retained]) as scope:
                scope.spawn(
                    [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                    env=env)
        assert raised.value.__cause__ is failure
        assert raised.value.retained_paths == (retained,)
        assert len(created) == 1
        assert created[0].poll() is None
    finally:
        for proc in created:
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_spawn_failure_without_child_preserves_original_error(tmp_path):
    from pm.process_scope import owned_spawn

    retained = tmp_path / "staged-input"
    missing = tmp_path / "missing-executable"
    env = dict(os.environ)
    with pytest.raises(FileNotFoundError) as raised:
        with owned_spawn({}, env, retained=[retained]) as scope:
            scope.spawn([str(missing)], env=env)
    assert raised.value.filename == str(missing)
    assert raised.value.__cause__ is None
    assert not hasattr(raised.value, "retained_paths")



def _fail_after_spawn(monkeypatch, process_scope, created, error):
    """Raise ``error`` once the owned child exists and the scope holds it."""
    original = process_scope.OwnedScope.spawn

    def spawn_then_fail(self, *args, **kwargs):
        proc = original(self, *args, **kwargs)
        created.append(proc)
        raise error

    monkeypatch.setattr(process_scope.OwnedScope, "spawn", spawn_then_fail)


def test_failure_after_spawn_reaps_child(monkeypatch):
    import pm.process_scope as process_scope

    created = []
    _fail_after_spawn(monkeypatch, process_scope, created, KeyboardInterrupt)
    env = dict(os.environ)
    try:
        with pytest.raises(KeyboardInterrupt):
            with process_scope.owned_spawn({}, env) as scope:
                scope.spawn([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                            env=env, stdout=subprocess.PIPE)
        assert created[0].returncode is not None
        assert created[0].stdout.closed
    finally:
        for proc in created:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


def test_interrupt_after_child_creation_reaps_before_propagating(monkeypatch):
    import pm.process_scope as process_scope

    created = []
    _fail_after_spawn(monkeypatch, process_scope, created, KeyboardInterrupt)
    env = dict(os.environ)
    prior_sigint = signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(KeyboardInterrupt):
            with process_scope.owned_spawn({}, env) as scope:
                scope.spawn([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                            env=env)
        assert len(created) == 1
        assert created[0].returncode is not None
        if os.name == "posix":
            with pytest.raises(ProcessLookupError):
                os.kill(created[0].pid, 0)
        assert signal.getsignal(signal.SIGINT) is prior_sigint
    finally:
        for proc in created:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


@pytest.mark.parametrize("failed_step", ["terminate", "wait"])
def test_unproved_spawn_cleanup_retains_child(monkeypatch, failed_step):
    import pm.process_scope as process_scope

    created = []
    original_kill = subprocess.Popen.kill
    original_wait = subprocess.Popen.wait
    original_killpg = getattr(os, "killpg", None)

    def refuse_cleanup(*_args, **_kwargs):
        raise OSError("termination failed")

    def timeout_wait(proc, *args, **kwargs):
        if proc in created:
            raise subprocess.TimeoutExpired(proc.args, kwargs.get("timeout"))
        return original_wait(proc, *args, **kwargs)

    _fail_after_spawn(monkeypatch, process_scope, created, RuntimeError("post-spawn failure"))
    if failed_step == "terminate":
        monkeypatch.setattr(subprocess.Popen, "kill", refuse_cleanup)
        if original_killpg is not None:
            monkeypatch.setattr(process_scope.os, "killpg", refuse_cleanup)
    else:
        monkeypatch.setattr(subprocess.Popen, "wait", timeout_wait)
    env = dict(os.environ)
    try:
        with pytest.raises(process_scope.ProcessStillRunning) as raised:
            with process_scope.owned_spawn({}, env) as scope:
                scope.spawn([sys.executable, "-I", "-S", "-c",
                             "import time; time.sleep(60)"], env=env)
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert raised.value.retained_paths == ()
    finally:
        for proc in created:
            if proc.poll() is None:
                if os.name == "posix":
                    original_killpg(proc.pid, signal.SIGKILL)
                else:
                    original_kill(proc)
            original_wait(proc, timeout=10)


def test_unproved_spawn_cleanup_preserves_cancellation(monkeypatch):
    import pm.process_scope as process_scope

    created = []
    cancellation = KeyboardInterrupt()
    original_kill = subprocess.Popen.kill
    original_wait = subprocess.Popen.wait
    original_killpg = getattr(os, "killpg", None)

    def refuse_cleanup(*_args, **_kwargs):
        raise OSError("termination failed")

    _fail_after_spawn(monkeypatch, process_scope, created, cancellation)
    monkeypatch.setattr(subprocess.Popen, "kill", refuse_cleanup)
    if original_killpg is not None:
        monkeypatch.setattr(process_scope.os, "killpg", refuse_cleanup)
    env = dict(os.environ)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with process_scope.owned_spawn({}, env) as scope:
                scope.spawn([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                            env=env)
        assert raised.value is cancellation
        assert cancellation.retained_paths == ()
    finally:
        for proc in created:
            if proc.poll() is None:
                if os.name == "posix":
                    original_killpg(proc.pid, signal.SIGKILL)
                else:
                    original_kill(proc)
            original_wait(proc, timeout=10)


@pytest.mark.platforms("posix")
# Teardown kills the test's own escaped grandchild, which is reparented outside
# the test's process subtree once its parent is stopped.
@pytest.mark.live_system_guard_bypass
def test_exception_retains_escaped_descendant_holding_liveness(tmp_path):
    from pm.process_scope import ProcessStillRunning, owned_spawn
    import time

    marker = tmp_path / "escaped.pid"
    child = (
        "import os, subprocess, sys, time; from pathlib import Path; "
        f"marker = Path({str(marker)!r}); "
        "grandchild = subprocess.Popen([sys.executable, '-I', '-S', '-c', "
        "'import time; time.sleep(60)'], start_new_session=True, close_fds=False); "
        "marker.write_text(str(grandchild.pid)); "
        "time.sleep(60)"
    )
    env = dict(os.environ)
    pid = None
    try:
        with pytest.raises(ProcessStillRunning) as raised:
            with owned_spawn({}, env, retained=[marker]) as scope:
                scope.spawn([sys.executable, "-I", "-S", "-c", child], env=env)
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert marker.exists()
                pid = int(marker.read_text())
                raise ValueError("caller failed")
        assert isinstance(raised.value.__cause__, ValueError)
        assert marker in raised.value.retained_paths
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.platforms("posix")
def test_writer_close_interrupt_still_stops_or_retains_child(tmp_path, monkeypatch):
    from pm.process_scope import owned_spawn

    retained = tmp_path / "input"
    env = dict(os.environ)
    scope = owned_spawn({}, env, retained=[retained])
    proc = None
    interruption = KeyboardInterrupt()

    def interrupt_close():
        raise interruption

    monkeypatch.setattr(scope, "_close_writer", interrupt_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with scope:
                proc = scope.spawn(
                    [sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                    env=env)
                raise ValueError("caller failed")
        assert raised.value is interruption
        assert proc.returncode is not None
        assert retained in interruption.retained_paths
    finally:
        if proc is not None:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        if scope.write_fd is not None:
            os.close(scope.write_fd)
        scope.liveness.close()


def _spawn_escaped_liveness_holder(scope, env, marker):
    import time

    child = (
        "import os, subprocess, sys, time; from pathlib import Path; "
        f"marker = Path({str(marker)!r}); "
        "grandchild = subprocess.Popen([sys.executable, '-I', '-S', '-c', "
        "'import time; time.sleep(60)'], start_new_session=True, close_fds=False); "
        "marker.write_text(str(grandchild.pid)); "
        "time.sleep(60)"
    )
    scope.spawn([sys.executable, "-I", "-S", "-c", child], env=env)
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert marker.exists()
    return int(marker.read_text())


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_pipe_close_interrupt_after_failed_proof_retains_child(tmp_path, monkeypatch):
    from pm.process_scope import owned_spawn

    marker = tmp_path / "escaped.pid"
    env = dict(os.environ)
    scope = owned_spawn({}, env, retained=[marker])
    interruption = KeyboardInterrupt()
    pid = None

    def interrupt_close():
        raise interruption

    monkeypatch.setattr(scope, "_close_pipes", interrupt_close)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with scope:
                pid = _spawn_escaped_liveness_holder(scope, env, marker)
                raise ValueError("caller failed")
        assert raised.value is interruption
        assert marker in interruption.retained_paths
        assert scope.proc.returncode is not None
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass
def test_nested_unproved_scopes_union_retained_inputs(tmp_path):
    from pm.process_scope import ProcessStillRunning, owned_spawn

    outer_marker = tmp_path / "outer.pid"
    inner_marker = tmp_path / "inner.pid"
    env = dict(os.environ)
    pids = []
    try:
        with pytest.raises(ProcessStillRunning) as raised:
            with owned_spawn({}, env, retained=[outer_marker]) as outer:
                pids.append(_spawn_escaped_liveness_holder(outer, env, outer_marker))
                with owned_spawn({}, env, retained=[inner_marker]) as inner:
                    pids.append(_spawn_escaped_liveness_holder(inner, env, inner_marker))
                    raise ValueError("caller failed")
        assert raised.value.retained_paths == (inner_marker, outer_marker)
        assert isinstance(raised.value.__cause__, ProcessStillRunning)
        assert outer.proc.returncode is not None
        assert inner.proc.returncode is not None
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_owned_scope_does_not_change_signal_handlers():
    import asyncio
    from pm.process_scope import owned_spawn

    watched = (signal.SIGINT, signal.SIGTERM, *(
        (signal.SIGBREAK,) if hasattr(signal, "SIGBREAK") else ()))
    env = dict(os.environ)
    with asyncio.Runner() as runner:
        async def exercise():
            before = tuple(signal.getsignal(sig) for sig in watched)
            with owned_spawn({}, env) as scope:
                assert tuple(signal.getsignal(sig) for sig in watched) == before
                proc = scope.spawn([sys.executable, "-I", "-S", "-c", "pass"], env=env)
                assert tuple(signal.getsignal(sig) for sig in watched) == before
                proc.wait(timeout=10)
                scope.finish()
                assert tuple(signal.getsignal(sig) for sig in watched) == before
            assert tuple(signal.getsignal(sig) for sig in watched) == before
        runner.run(exercise())


def test_owned_scope_requires_finish():
    from pm.process_scope import owned_spawn

    env = dict(os.environ)
    with pytest.raises(RuntimeError, match="owned child not finished"):
        with owned_spawn({}, env) as scope:
            proc = scope.spawn([sys.executable, "-I", "-S", "-c", "import time; time.sleep(60)"],
                               env=env)
    assert proc.returncode is not None


def test_completion_retains_request_when_spawn_cleanup_unproved(tmp_path, monkeypatch):
    import hermes_cli.update_completion as completion
    import pm.process_scope as process_scope

    directory = tmp_path / "request"
    directory.mkdir()
    monkeypatch.setattr(completion.tempfile, "mkdtemp", lambda **_kwargs: str(directory))
    created = []
    original_kill = subprocess.Popen.kill
    original_wait = subprocess.Popen.wait
    original_killpg = getattr(os, "killpg", None)

    def refuse_cleanup(*_args, **_kwargs):
        raise OSError("termination failed")

    def timeout_wait(proc, *args, **kwargs):
        if proc in created:
            raise subprocess.TimeoutExpired(proc.args, kwargs.get("timeout"))
        return original_wait(proc, *args, **kwargs)

    _fail_after_spawn(monkeypatch, process_scope, created, RuntimeError("post-spawn failure"))
    monkeypatch.setattr(subprocess.Popen, "kill", refuse_cleanup)
    monkeypatch.setattr(subprocess.Popen, "wait", timeout_wait)
    if original_killpg is not None:
        monkeypatch.setattr(process_scope.os, "killpg", refuse_cleanup)
    monkeypatch.setattr(completion, "_seal_post_quiescence",
                        lambda *_args: pytest.fail("unproved cleanup cannot seal"))
    request = {"source": str(tmp_path), "home": str(tmp_path),
               "receipt": {"update_id": "test-update"}}
    try:
        with pytest.raises(process_scope.ProcessStillRunning) as raised:
            completion.run_completion(request)
        assert (directory / "request.json").exists()
        assert directory in raised.value.retained_paths
        assert not (directory / "result.json").exists()
    finally:
        for proc in created:
            if proc.poll() is None:
                if os.name == "posix":
                    original_killpg(proc.pid, signal.SIGKILL)
                else:
                    original_kill(proc)
            original_wait(proc, timeout=10)
            if proc.stdout is not None:
                proc.stdout.close()




