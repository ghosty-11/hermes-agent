"""Kernel ownership must cover dependency selections, runtime bootstrap and descendants."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from threading import Event, Thread

import pytest

from pm.operations import pm_writer_claim
from tests.pm._fixtures import isolated_python as isolated_python

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.platforms("posix")
def test_isolated_manager_worker_reaches_real_request_dispatch(tmp_path, monkeypatch, isolated_python):
    """The manager has its dependencies, but -I does not put the checkout on sys.path."""
    from pm.writer_claim import child_spawn, writer_claim

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    worker = ROOT / "pm" / "worker.py"
    request = {
        "id": "isolated-dispatch", "operation": "venv_is_current",
        "arguments": {}, "callbacks": [], "packages": [],
        "context": {"repo": str(tmp_path), "lockfile": str(tmp_path / "lock.json")},
    }
    # Calibration executes the identical file with explicit root insertion, not a
    # replacement pm module or a .pth in the manager environment.
    commands = (
        [str(isolated_python), "-I", "-B", "-c",
         "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
         "runpy.run_path(sys.argv[2],run_name='__main__')", str(ROOT), str(worker)],
        [str(isolated_python), "-I", "-B", str(worker)],
    )
    with writer_claim():
        for command in commands:
            env = dict(os.environ)
            with child_spawn(env) as spawn:
                result = subprocess.run(command, input=json.dumps(request) + "\n",
                                        env=env, capture_output=True, text=True,
                                        timeout=40, **spawn)
            assert result.returncode == 0, result.stderr
            replies = [json.loads(line) for line in result.stdout.splitlines()]
            assert replies[-1]["id"] == request["id"]
            assert replies[-1]["type"] == "result"
            assert replies[-1]["result"] is False


@pytest.mark.platforms("posix")
def test_independent_threads_cannot_adopt_advertised_claim(tmp_path, monkeypatch):
    from pm.writer_claim import writer_claim

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    entered, release, finished = Event(), Event(), Event()
    outcomes = []

    def owner():
        with writer_claim():
            with writer_claim():  # Same-thread nesting is a valid reentry.
                entered.set()
                assert release.wait(10)

    def contender():
        assert entered.wait(10)
        try:
            with writer_claim():
                outcomes.append("admitted-after-release" if release.is_set() else "borrowed")
        except (OSError, RuntimeError):
            outcomes.append("refused")
        finally:
            finished.set()

    first = Thread(target=owner)
    second = Thread(target=contender)
    first.start()
    second.start()
    try:
        finished.wait(2)
        assert outcomes != ["borrowed"], "another thread adopted the live kernel claim"
    finally:
        release.set()
        first.join(timeout=10)
        second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert outcomes in (["refused"], ["admitted-after-release"])
    assert "HERMES_RETAINED_WRITER_CLAIMS" not in os.environ
    with writer_claim():
        pass


@pytest.mark.platforms("posix")
def test_extras_marker_fallback_cannot_publish_manager_during_native_transaction(tmp_path, monkeypatch):
    from pm import paths
    from pm.environments import install_state_dir

    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    selection = install_state_dir(project) / "pm-runtime" / "selected.json"
    child = f"""
import sys, venv
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from pm import paths, extras, runtime
paths.repo_root = lambda: Path({str(project)!r})
import pm._uv, pm.runtime_stage
pm._uv._toolchain = lambda **kw: (Path(sys.executable), Path(sys.executable))
def build(_uv, _python, out, **kw):
    venv.EnvBuilder(with_pip=False).create(out)
    return runtime._python(out)
pm.runtime_stage.stage_runtime = build
assert extras._evaluate_in_runtime("sys_platform == 'linux'", {{"sys_platform": "linux"}})
"""
    env = {key: value for key, value in os.environ.items()
           if key != "HERMES_RETAINED_WRITER_CLAIMS"}
    with pm_writer_claim(transaction=True):
        refused = _child(child, env, timeout=40)
        assert refused.returncode == 0, refused.stderr
        assert not selection.exists(), "marker fallback published a manager generation under another writer"
    admitted = _child(child, env, timeout=40)
    assert admitted.returncode == 0, admitted.stderr
    fact = json.loads(selection.read_text())
    assert (selection.parent / fact["generation"] / "pm-runtime.json").is_file()


@pytest.mark.platforms("posix")
def test_sealed_install_claim_lives_in_writable_state_and_keeps_install_identity(tmp_path, monkeypatch):
    from pm.writer_claim import domains, writer_claim

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    roots = (tmp_path / "sealed-a", tmp_path / "sealed-b")
    for root in roots:
        root.mkdir()
        root.chmod(0o555)
    try:
        first, second = (domains(root)[0] for root in roots)
        assert first != second
        assert all(not claim.is_relative_to(root) for claim, root in zip((first, second), roots))
        assert first.is_relative_to(home) and second.is_relative_to(home)
        with writer_claim(install=roots[0]):
            assert first.is_file()
    finally:
        for root in roots:
            root.chmod(0o755)

def _child(code: str, environment: dict[str, str], *, timeout: int = 20):
    return subprocess.run([sys.executable, "-I", "-S", "-c", code], env=environment,
                          capture_output=True, text=True, timeout=timeout)


def _runtime_bootstrap_child(project: Path, *, cli: bool = False) -> str:
    operation = ("runtime.run_cli(['status'])" if cli else
                 f"client._request('prune_cache', {{'cache': {str(project / 'cache')!r}, 'offline': True}})")
    return f"""
import sys, venv
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from pm import client, paths, runtime
from pm.environments import install_state_dir
paths.repo_root = lambda: Path({str(project)!r})
import pm._uv
pm._uv._toolchain = lambda **kw: (Path(sys.executable), Path(sys.executable))
import pm.runtime_stage
# Only the dependency builder is controlled; client/CLI bootstrap and publication stay real.
def build(_uv, _python, out, **kw):
    venv.EnvBuilder(with_pip=False).create(out)
    return runtime._python(out)
pm.runtime_stage.stage_runtime = build
selected = install_state_dir(Path({str(project)!r})) / 'pm-runtime/selected.json'
try:
    {operation}
except Exception:
    if not selected.exists():
        raise
if not selected.exists():
    raise RuntimeError('no runtime selection was published')
"""


@pytest.mark.platforms("posix")
def test_runtime_selection_claim_covers_bootstrap_across_store_overrides(tmp_path, monkeypatch):
    from pm import paths
    from pm.environments import install_state_dir

    home = tmp_path / "home"
    home.mkdir()
    project, other = tmp_path / "project", tmp_path / "other"
    project.mkdir()
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store-a" / "tools"))
    monkeypatch.delenv("HERMES_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(paths, "repo_root", lambda: project)
    selected = install_state_dir(project) / "pm-runtime" / "selected.json"
    other_home = tmp_path / "other-home"
    other_home.mkdir()
    with monkeypatch.context() as scoped:
        scoped.setenv("HERMES_HOME", str(other_home))
        other_selected = install_state_dir(other) / "pm-runtime" / "selected.json"
    # The controlled builder publishes a real selection without manager dependencies.
    env = {key: value for key, value in os.environ.items()
           if key != "HERMES_RETAINED_WRITER_CLAIMS"}
    with pm_writer_claim(transaction=True):
        for store in ("store-a", "store-b"):
            contender_env = {**env, "HERMES_RUNTIME_DIR": str(tmp_path / store / "tools")}
            refused = _child(_runtime_bootstrap_child(project), contender_env, timeout=40)
            assert refused.returncode != 0
            assert not selected.exists()
        cli_refused = _child(_runtime_bootstrap_child(project, cli=True), env, timeout=40)
        assert cli_refused.returncode != 0
        assert not selected.exists()
        # A distinct installation with a distinct dependency home may proceed.
        independent = _child(
            _runtime_bootstrap_child(other),
            {**env, "HERMES_HOME": str(other_home),
             "HERMES_RUNTIME_DIR": str(tmp_path / "store-b" / "tools")}, timeout=40)
        assert independent.returncode == 0, independent.stderr
        assert other_selected.is_file()
    admitted = _child(_runtime_bootstrap_child(project),
                      {**env, "HERMES_RUNTIME_DIR": str(tmp_path / "store-b" / "tools")}, timeout=40)
    assert admitted.returncode == 0, admitted.stderr
    fact = json.loads(selected.read_text())
    assert (selected.parent / fact["generation"] / "pm-runtime.json").is_file()
    assert (selected.parent / fact["generation"] / "pyvenv.cfg").is_file()


@pytest.mark.platforms("linux")
def test_owner_death_does_not_release_exclusion_while_child_is_mutating(tmp_path):
    import ctypes

    home = tmp_path / "home"
    home.mkdir()
    store = tmp_path / "tools"
    gate = tmp_path / "child-gate"
    os.mkfifo(gate)
    env = {key: value for key, value in os.environ.items()
           if key != "HERMES_RETAINED_WRITER_CLAIMS"}
    env.update(HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(store))
    contender = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from pm.operations import pm_writer_claim; "
        "\nwith pm_writer_claim(): pass\n"
    )
    grandchild = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from pm.writer_claim import writer_claim; "
        "\nwith writer_claim(inherited_only=True):\n"
        " print('ready', flush=True)\n"
        " with Path(sys.argv[1]).open('rb') as gate: gate.read(1)\n"
    )
    descendant = f"""
import os, subprocess, sys
sys.path.insert(0, {str(ROOT)!r})
from pm.operations import pm_writer_claim
from pm.writer_claim import child_kwargs
with pm_writer_claim():
    child = subprocess.Popen([sys.executable, '-I', '-S', '-c', {grandchild!r}, sys.argv[1]],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             env=dict(os.environ), **child_kwargs())
    assert child.stdout.readline().strip() == 'ready'
    print(child.pid, flush=True)
"""
    owner = f"""
import os, subprocess, sys
sys.path.insert(0, {str(ROOT)!r})
from pm.operations import pm_writer_claim
from pm.writer_claim import child_kwargs
with pm_writer_claim(transaction=True):
    child = subprocess.Popen([sys.executable, '-I', '-S', '-c', {descendant!r}, {str(gate)!r}],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                             env=dict(os.environ), **child_kwargs())
    grandchild_pid = int(child.stdout.readline().strip())
    child.wait(timeout=10)
    assert child.returncode == 0
    print(grandchild_pid, flush=True)
    sys.stdin.read(1)
"""
    # Reap the orphaned fixture child ourselves; never leave an adopted zombie
    # for the test runner or init. The Linux-only test restores its original flag.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    process = None
    descendant_pid = None
    try:
        process = subprocess.Popen([sys.executable, "-I", "-S", "-c", owner],
                                   env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        ready, _, _ = select.select([process.stdout], [], [], 20)
        assert ready, "fixture owner never published its child-ready handshake"
        descendant_pid = int(process.stdout.readline().strip())
        refused = _child(contender, env)
        assert refused.returncode != 0
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
        # The child is still inside the claim, blocked on its own fixture gate.
        os.kill(descendant_pid, 0)
        refused = _child(contender, env)
        assert refused.returncode != 0
    finally:
        try:
            if process is not None:
                # Only the session/process group created by this fixture is signalled.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate(timeout=15)
            if descendant_pid is not None:
                os.waitpid(descendant_pid, 0)
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0
    admitted = _child(contender, env)
    assert admitted.returncode == 0, admitted.stderr
