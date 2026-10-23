"""Native local admission must fail closed before touching the installed checkout."""
from __future__ import annotations

import os
import json
import shutil
from pathlib import Path
import subprocess
import sys
import select
import shlex
import signal


import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.platforms("posix")
def test_source_script_child_adopts_retained_kernel_claim(tmp_path, monkeypatch):
    from hermes_cli.source_build import run_source_script
    from pm.writer_claim import child_spawn, writer_claim

    root, home = tmp_path / "source", tmp_path / "home"
    root.mkdir()
    home.mkdir()
    script = root / "scripts/build/probe.py"
    script.parent.mkdir(parents=True)
    proof = tmp_path / "child-adopted"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim, joined_claim\n"
        "with writer_claim(inherited_only=True):\n"
        f"    assert joined_claim()\n    Path({str(proof)!r}).write_text('adopted')\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "node").symlink_to(sys.executable)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    with writer_claim(install=root):
        env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.defpath}
        with child_spawn(env) as spawn:
            calibration = subprocess.run([str(bin_dir / "node"), str(script)],
                                         cwd=root, env=env, capture_output=True,
                                         text=True, timeout=15, **spawn)
        assert calibration.returncode == 0, calibration.stderr
        proof.unlink()
        run_source_script(root, "scripts/build/probe.py", env=env, label="Fixture child")
        assert proof.read_text() == "adopted"

@pytest.mark.platforms("linux")
@pytest.mark.parametrize("caller", ["source", "git", "completion"])
def test_mutating_caller_retains_claim_after_parent_loss(tmp_path, caller):
    """The caller's real spawn must leave a kernel owner after its parent dies."""
    import ctypes


    root, home = tmp_path / "installed", tmp_path / "home"
    root.mkdir()
    home.mkdir()
    gate = tmp_path / "release"
    os.mkfifo(gate)
    ready = tmp_path / "adopted"
    os.mkfifo(ready)
    child = tmp_path / "mutation.py"
    child.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim, joined_claim\n"
        "with writer_claim(inherited_only=True):\n"
        "    assert joined_claim()\n"
        f"    with os.fdopen(os.open({str(gate)!r}, os.O_RDWR), 'rb') as stream:\n"
        f"        with Path({str(ready)!r}).open('w') as status: status.write(str(os.getpid()))\n"
        "        stream.read(1)\n"
    )
    child.chmod(0o700)
    fixture_bin = tmp_path / "bin"
    fixture_bin.mkdir()
    if caller == "source":
        script = root / "scripts/build/mutation.py"
        script.parent.mkdir(parents=True)
        script.write_bytes(child.read_bytes())
        (fixture_bin / "node").symlink_to(sys.executable)
        invocation = (
            "from hermes_cli.source_build import run_source_script\n"
            f"run_source_script(Path({str(root)!r}), 'scripts/build/mutation.py', "
            "env=dict(os.environ), label='Owned source child')\n"
        )
    elif caller == "git":
        child.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "from pathlib import Path\n"
            "content = sys.stdin.buffer.read()\n"
            f"with os.fdopen(os.open({str(gate)!r}, os.O_RDWR), 'rb') as stream:\n"
            f"    with Path({str(ready)!r}).open('w') as status:\n"
            "        status.write(f'{os.getpid()}:{os.getppid()}')\n"
            "    stream.read(1)\n"
            "sys.stdout.buffer.write(content)\n"
        )
        _git(root, "init", "-b", "main")
        (root / ".gitattributes").write_text("tracked filter=hold\n")
        (root / "tracked").write_text("owned Git mutation\n")
        _git(root, "config", "filter.hold.smudge", shlex.quote(str(child)))
        _git(root, "config", "filter.hold.clean", "cat")
        _git(root, "config", "filter.hold.required", "true")
        _git(root, "add", ".")
        _git(root, "commit", "-m", "fixture")
        invocation = (
            "from hermes_cli.update_local import local_git\n"
            f"local_git(Path({str(root)!r}), 'checkout', '--force', 'HEAD')\n"
        )
    else:
        completion = root / "hermes_cli/source_completion.py"
        completion.parent.mkdir()
        completion.write_bytes(child.read_bytes())
        invocation = (
            "from hermes_cli.venv_sync import _finish_source_update\n"
            f"_finish_source_update(Path({str(root)!r}), current=True, "
            f"pending=Path({str(tmp_path / 'pending')!r}))\n"
        )
    parent = (
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pm.writer_claim import writer_claim, child_spawn\n"
        f"root = Path({str(root)!r})\n"
        "with writer_claim(install=root):\n"
        "    if sys.argv[1] == 'control':\n"
        "        env = dict(os.environ)\n"
        "        with child_spawn(env) as spawn:\n"
        + (f"            subprocess.run([{str(shutil.which('git'))!r}, '-C', str(root), "
           "'checkout', '--force', 'HEAD'], env=env, check=True, **spawn)\n"
           if caller == "git" else
           f"            subprocess.run([sys.executable, {str(child)!r}], env=env, check=True, **spawn)\n")
        + "    else:\n"
        + "".join("        " + row + "\n" for row in invocation.splitlines())
    )
    contender = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim\n"
        f"with writer_claim(install=Path({str(root)!r})): pass\n"
    )
    env = {key: value for key, value in os.environ.items()
           if key != "HERMES_RETAINED_WRITER_CLAIMS"}
    env.update(HERMES_HOME=str(home), HERMES_RUNTIME_DIR=str(tmp_path / "tools"),
               HERMES_INSTALL_ROOT=str(root),
               PATH=str(fixture_bin) + os.pathsep + os.environ.get("PATH", os.defpath))
    unheld = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                            env=env, capture_output=True, text=True, timeout=15)
    assert unheld.returncode == 0, unheld.stderr
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        for mode in ("control", "caller"):
            if caller == "git":
                (root / "tracked").unlink()
            owner = subprocess.Popen([sys.executable, "-I", "-S", "-c", parent, mode],
                                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                     start_new_session=True)
            descendant_pid = None
            git_pid = None
            fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
            try:
                readable, _, _ = select.select([fd], [], [], 20)
                if readable:
                    raw = os.read(fd, 64)
                    if raw:
                        parts = raw.decode().split(":")
                        descendant_pid = int(parts[0])
                        if len(parts) == 2:
                            git_pid = int(parts[1])
                assert descendant_pid is not None, (
                    owner.stderr.read().decode() if owner.poll() is not None else
                    f"{caller}: no adoption handshake")
                os.kill(owner.pid, signal.SIGKILL)
                owner.wait(timeout=10)
                refused = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                                         env=env, capture_output=True, text=True, timeout=15)
                assert refused.returncode != 0, f"{caller} child lost kernel ownership"
                with gate.open("wb") as release:
                    release.write(b"x")
                if git_pid is not None:
                    git_fd = os.pidfd_open(git_pid)
                    try:
                        done, _, _ = select.select([git_fd], [], [], 20)
                        assert done, "owned Git mutation did not finish after its caller exited"
                    finally:
                        os.close(git_fd)
                    os.waitpid(git_pid, 0)
                    git_pid = None
                    assert (root / "tracked").read_text() == "owned Git mutation\n"
            finally:
                os.close(fd)
                try:
                    os.killpg(owner.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                owner.communicate(timeout=10)
                if git_pid is not None:
                    os.waitpid(git_pid, 0)
                if descendant_pid is not None:
                    try:
                        os.waitpid(descendant_pid, 0)
                    except ChildProcessError:
                        if caller != "git":
                            raise
        admitted = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                                  env=env, capture_output=True, text=True, timeout=15)
        assert admitted.returncode == 0, admitted.stderr
    finally:
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


@pytest.mark.platforms("linux")
def test_cancelled_offline_completion_keeps_request_while_selected_descendant_writes(
    tmp_path, monkeypatch,
):
    """The real parent transport must not erase inputs held by a selected descendant."""
    import ctypes
    from hermes_cli.update_completion import run_completion
    from pm.writer_claim import writer_claim

    root, home = tmp_path / "installed", tmp_path / "home"
    (root / "hermes_cli").mkdir(parents=True)
    home.mkdir()
    gate = tmp_path / "writer-release"
    os.mkfifo(gate)
    proof = tmp_path / "request-location"
    selected = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim, joined_claim\n"
        "with writer_claim(install=Path(sys.argv[3]), inherited_only=True):\n"
        "    assert joined_claim()\n"
        "    Path(sys.argv[2]).write_text(json.dumps({'request': sys.argv[1], 'pid': os.getpid()}))\n"
        "    print('selected-ready', flush=True)\n"
        "    with Path(sys.argv[4]).open('rb') as stream: stream.read(1)\n"
    )
    preparation = (
        "import os, subprocess, sys\n"
        # The completion transport already starts this preparation as a session
        # leader; the selected descendant below escapes that owned group itself.
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim, child_spawn\n"
        f"with writer_claim(install=Path({str(root)!r}), inherited_only=True):\n"
        "    env = dict(os.environ)\n"
        "    with child_spawn(env) as spawn:\n"
        f"        proc = subprocess.Popen([sys.executable, '-I', '-S', '-c', {selected!r}, "
        f"sys.argv[1], {str(proof)!r}, {str(root)!r}, {str(gate)!r}], "
        "env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, "
        "start_new_session=True, **spawn)\n"
        "    assert proc.stdout.readline().strip() == 'selected-ready'\n"
        "    print('prepared', flush=True)\n"
    )
    (root / "hermes_cli/update_completion.py").write_text(preparation)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    original_stdout = sys.stdout

    class InterruptOnPreparation:
        def isatty(self):
            return False

        def write(self, text):
            if "prepared" in text:
                raise KeyboardInterrupt("owned cancellation")
            return original_stdout.write(text)

        def flush(self):
            original_stdout.flush()

    request = {"schema": 1, "source": str(root), "home": str(home),
               "offline": True, "receipt": {"update_id": "owned-cancellation"},
               "staged_tools": str(tmp_path / "staged-tools")}
    (tmp_path / "staged-tools").mkdir()
    contender = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from pm.writer_claim import writer_claim\n"
        f"with writer_claim(install=Path({str(root)!r})): pass\n"
    )
    env = {key: value for key, value in os.environ.items()
           if key != "HERMES_RETAINED_WRITER_CLAIMS"}
    unheld = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                            env=env, capture_output=True, text=True, timeout=15)
    assert unheld.returncode == 0, unheld.stderr
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        with writer_claim(install=root):
            with monkeypatch.context() as scoped:
                scoped.setattr(sys, "stdout", InterruptOnPreparation())
                with pytest.raises(KeyboardInterrupt, match="owned cancellation"):
                    run_completion(request)
        live = json.loads(proof.read_text())
        request_path = Path(live["request"])
        os.kill(live["pid"], 0)
        refused = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                                 env=env, capture_output=True, text=True, timeout=15)
        assert refused.returncode != 0, "selected descendant no longer owns the writer domains"
        retained = json.loads(request_path.read_text())
        assert retained["staged_tools"] == str(tmp_path / "staged-tools")
    finally:
        if proof.exists():
            owned_pid = json.loads(proof.read_text())["pid"]
            os.kill(owned_pid, signal.SIGKILL)
            os.waitpid(owned_pid, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0
    admitted = subprocess.run([sys.executable, "-I", "-S", "-c", contender],
                              env=env, capture_output=True, text=True, timeout=15)
    assert admitted.returncode == 0, admitted.stderr


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("failure", ["empty_attestation", "retained_cancellation"])
def test_selected_completion_retention_survives_process_transport(tmp_path, monkeypatch, failure):
    """An empty retention list still protects the parent request and forbids sealing."""
    from hermes_cli.update_completion import run_completion
    from pm.process_scope import ProcessStillRunning
    from pm.writer_claim import writer_claim

    root, home = tmp_path / "installed", tmp_path / "home"
    (root / "hermes_cli").mkdir(parents=True)
    home.mkdir()
    proof = tmp_path / "selected-request"
    script = (
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from hermes_cli import update_completion\n"
        "from pm.process_scope import ProcessStillRunning\n"
        "from pm.writer_claim import child_spawn, writer_claim\n"
        "request_path, result_path = map(Path, sys.argv[1:3])\n"
        "request = json.loads(request_path.read_text())\n"
        "with writer_claim(install=Path(request['source']), inherited_only=True):\n"
        "    if '--prepared' in sys.argv:\n"
        f"        Path({str(proof)!r}).write_text(str(request_path))\n"
        "        def fail_selected(_request):\n"
        + (
            "            raise ProcessStillRunning((), 'selected cleanup unproved')\n"
            if failure == "empty_attestation" else
            "            exc = KeyboardInterrupt('selected cancellation')\n"
            "            exc.retained_paths = ()\n"
            "            raise exc\n"
        )
        + "        update_completion._complete_selected = fail_selected\n"
        "        raise SystemExit(update_completion._finish(request, result_path))\n"
        "    env = dict(os.environ)\n"
        "    with child_spawn(env) as handed:\n"
        "        raise SystemExit(subprocess.call([request['selected_python'], '-I', '-S', '-u',\n"
        "            __file__, str(request_path), str(result_path), '--prepared'],\n"
        "            cwd=request['source'], env=env, **handed))\n"
    )
    (root / "hermes_cli/update_completion.py").write_text(script)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    selected_python = tmp_path / "selected-python"
    selected_python.symlink_to(sys.executable)
    request = {
        "schema": 1, "source": str(root), "home": str(home),
        "selected_python": str(selected_python),
        "receipt": {"update_id": "selected-retention", "steps": []},
        "gateway_mode": False, "windows_resume": None,
    }
    if failure == "retained_cancellation":
        request["local_rollback_manifest"] = str(tmp_path / "rollback.json")
        Path(request["local_rollback_manifest"]).write_text("{}")

    try:
        with writer_claim(install=root):
            with pytest.raises(ProcessStillRunning) as caught:
                run_completion(request)

        request_path = Path(proof.read_text())
        result = json.loads((request_path.parent / "result.json").read_text())
        assert request_path.is_file()
        assert request_path.parent in caught.value.retained_paths
        assert result["not_quiescent"] == []
        assert "seal_pending" not in result
        if failure == "retained_cancellation":
            assert result["cancelled"] is True
    finally:
        if proof.exists():
            shutil.rmtree(Path(proof.read_text()).parent, ignore_errors=True)


@pytest.mark.platforms("linux")
def test_cancelled_local_entry_does_not_seal_while_escaped_completion_owns_inputs(
    tmp_path, monkeypatch,
):
    """The local entry must not seal a capsule while completion cannot prove quiescence."""
    import ctypes
    from types import SimpleNamespace

    from hermes_cli import main, local_update_preflight, update_cmd, update_inventory, update_receipt
    from hermes_cli._early_recovery import local_transaction_marker
    from hermes_cli.local_update_snapshot import freeze_target
    from pm.process_scope import retained_directory
    from pm.writer_claim import writer_claim

    root, source, home = tmp_path / "installed", tmp_path / "reviewed", tmp_path / "home"
    (root / "hermes_cli").mkdir(parents=True)
    home.mkdir()
    gate = tmp_path / "writer-release"
    os.mkfifo(gate)
    proof = tmp_path / "request-location"
    selected = (
        "import json, os, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim, joined_claim, inherited_liveness_fds\n"
        "with writer_claim(install=Path(sys.argv[3]), inherited_only=True):\n"
        "    assert joined_claim() and inherited_liveness_fds() and os.getsid(0) == os.getpid()\n"
        "    Path(sys.argv[2]).write_text(json.dumps({'request': sys.argv[1], 'pid': os.getpid()}))\n"
        "    print('selected-ready', flush=True)\n"
        "    with Path(sys.argv[4]).open('rb') as stream: stream.read(1)\n"
    )
    preparation = (
        "import os, subprocess, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from pm.writer_claim import writer_claim, child_spawn\n"
        f"with writer_claim(install=Path({str(root)!r}), inherited_only=True):\n"
        "    env = dict(os.environ)\n"
        "    with child_spawn(env) as spawn:\n"
        f"        proc = subprocess.Popen([sys.executable, '-I', '-S', '-c', {selected!r}, "
        f"sys.argv[1], {str(proof)!r}, {str(root)!r}, {str(gate)!r}], "
        "env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, "
        "start_new_session=True, **spawn)\n"
        "    assert proc.stdout.readline().strip() == 'selected-ready'\n"
        "    print('prepared', flush=True)\n"
    )
    (root / "hermes_cli/update_completion.py").write_text(preparation)
    (root / "version.txt").write_text("before\n")
    _git(root, "init", "-b", "main")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "before")
    _git(tmp_path, "clone", "--no-local", str(root), str(source))
    (source / "version.txt").write_text("after\n")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "target")
    target = _git(source, "rev-parse", "HEAD")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    # An installed host has its runtime and gateway lock directories; capture
    # anchors the runtime selection and restart obligation there (396/398).
    (tmp_path / "runtime").mkdir()
    (tmp_path / "gateway-locks").mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    monkeypatch.setenv("HERMES_LOCAL_UPDATE_CONTEXT", "1")
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    monkeypatch.setattr(update_inventory, "collect_runtime_inventory",
                        lambda: update_inventory.UpdatePlan())
    monkeypatch.setattr(update_cmd, "_resolve_pre_update_backup_mode", lambda _: "quick")
    monkeypatch.setattr(
        update_cmd, "_resolve_update_options",
        lambda *_: SimpleNamespace(assume_yes=True, pre_update_version="fixture"),
    )
    quick_snapshot = home / "state-snapshots" / "fixture"

    def capture_quick_snapshot(_):
        quick_snapshot.mkdir(parents=True)
        (quick_snapshot / "state").write_text("before\n")
        return "fixture"

    def stage_inputs(_source, _root, staging, _extras):
        tools = staging / "tools"
        tools.mkdir()
        (tools / "input").write_text("selected tool\n")
        return []

    monkeypatch.setattr(main, "_run_pre_update_backup", capture_quick_snapshot)
    monkeypatch.setattr(local_update_preflight, "stage_local_inputs", stage_inputs)
    interrupt = KeyboardInterrupt("owned cancellation")
    original_stdout = sys.stdout

    class InterruptOnPreparation:
        def isatty(self):
            return False

        def write(self, text):
            if "prepared" in text:
                raise interrupt
            return original_stdout.write(text)

        def flush(self):
            original_stdout.flush()

    contender = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from pm.writer_claim import writer_claim\n"
        f"with writer_claim(install=Path({str(root)!r})): pass\n"
    )
    env = {key: value for key, value in os.environ.items()
           if key not in ("HERMES_RETAINED_WRITER_CLAIMS", "HERMES_OWNED_LIVENESS_FDS")}
    unheld = subprocess.run(
        [sys.executable, "-I", "-S", "-c", contender], env=env,
        capture_output=True, text=True, timeout=15,
    )
    assert unheld.returncode == 0, unheld.stderr
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    original_receipt = update_receipt._current.get()
    snapshot_stage = None
    retained = ()
    try:
        with pytest.raises(KeyboardInterrupt, match="owned cancellation") as caught:
            with writer_claim(install=root):
                with retained_directory(prefix="hermes-target-snapshot-") as temporary:
                    snapshot_stage = Path(temporary)
                    snapshot = freeze_target(source, target, snapshot_stage / "snapshot")
                    with monkeypatch.context() as scoped:
                        scoped.setattr(sys, "stdout", InterruptOnPreparation())
                        update_cmd._cmd_local_update(
                            SimpleNamespace(source_repo=str(snapshot), target=target, required_extra=[]),
                            gateway_mode=False,
                        )
        assert caught.value is interrupt
        retained = interrupt.retained_paths
        live = json.loads(proof.read_text())
        request_path = Path(live["request"])
        request = json.loads(request_path.read_text())
        stage = Path(request["staged_tools"]).parent
        capsule = Path(request["local_rollback_manifest"])
        assert {request_path.parent, stage, capsule, snapshot_stage} <= set(retained)
        assert request_path.is_file()
        assert request["offline"] is True
        assert request["expected_sha"] == target
        assert request["source"] == str(root)
        assert request["snapshot_id"] == "fixture"
        assert request["staged_tools"] == str(stage / "tools")
        assert (stage / "tools" / "input").read_text() == "selected tool\n"
        assert (snapshot_stage / "snapshot" / "version.txt").read_text() == "after\n"
        assert (quick_snapshot / "state").read_text() == "before\n"
        assert json.loads(capsule.read_text())["state"] == "captured"
        assert json.loads(local_transaction_marker(root).read_text())["capsule"] == str(capsule)
        os.kill(live["pid"], 0)
        refused = subprocess.run(
            [sys.executable, "-I", "-S", "-c", contender], env=env,
            capture_output=True, text=True, timeout=15,
        )
        assert refused.returncode != 0, "escaped descendant lost the writer claim"
    finally:
        if proof.exists():
            owned_pid = json.loads(proof.read_text())["pid"]
            os.kill(owned_pid, signal.SIGKILL)
            os.waitpid(owned_pid, 0)
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0
        current_receipt = update_receipt._current.get()
        if current_receipt is not original_receipt:
            update_receipt._current.reset(current_receipt.current_token)
        for path in retained:
            if path.name.startswith(("hermes-target-snapshot-", "hermes-local-preflight-",
                                     "hermes-completion-")):
                shutil.rmtree(path)
    admitted = subprocess.run(
        [sys.executable, "-I", "-S", "-c", contender], env=env,
        capture_output=True, text=True, timeout=15,
    )
    assert admitted.returncode == 0, admitted.stderr


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        env={**{key: value for key, value in os.environ.items()
                if not key.startswith("GIT_CONFIG_") and key not in
                ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")},
             "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
             "GIT_AUTHOR_NAME": "Fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
             "GIT_COMMITTER_NAME": "Fixture", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"},
        capture_output=True, text=True, check=True, timeout=15,
    )
    return result.stdout.strip()



@pytest.mark.platforms("posix")
def test_frozen_entry_imports_snapshot_bootstrap_under_hostile_git_env(tmp_path, monkeypatch):
    from hermes_cli.update_lock import UpdateLock
    from pm.writer_claim import child_spawn
    from pm.environments import install_state_dir, runtime_facts_path, site_packages

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("hermes_bootstrap.py", "hermes_constants.py", "hermes_startup_watchdog.py"):
        shutil.copy2(ROOT / name, snapshot / name)
    for name in ("pm", "hermes_cli", "hermes_platform"):
        shutil.copytree(ROOT / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__"))
    _git(snapshot, "init", "-b", "main")
    _git(snapshot, "add", ".")
    _git(snapshot, "commit", "-m", "frozen")
    target = _git(snapshot, "rev-parse", "HEAD")
    installed, home = tmp_path / "installed", tmp_path / "home"
    installed.mkdir()
    home.mkdir()
    poison = tmp_path / "installed-bootstrap-imported"
    (installed / "hermes_bootstrap.py").write_text(
        f"from pathlib import Path\nPath({str(poison)!r}).touch()\n"
        "raise RuntimeError('installed bootstrap was imported')\n")
    hook_ran = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "fsmonitor"
    hook.write_text(f"#!/bin/sh\nprintf executed > '{hook_ran}'\nexit 1\n")
    hook.chmod(0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    generation = install_state_dir(installed) / "environments" / "selected" / "venv"
    site_packages(generation).mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("home = fixture\n")
    facts = runtime_facts_path(installed)
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text(json.dumps({"schema": 1, "packages": {
        "venv": {"environment": str(generation)}}}))
    code = (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        "script, installed, home, snapshot, target = sys.argv[1:]\n"
        "sys.argv = [script, installed, home, snapshot, target]\n"
        "try:\n"
        "    runpy.run_path(script, run_name='__main__')\n"
        "except BaseException:\n"
        "    pass\n"
        "module = sys.modules.get('hermes_bootstrap')\n"
        "assert module is not None and Path(module.__file__).resolve() == Path(snapshot) / 'hermes_bootstrap.py'\n"
        "print('frozen bootstrap reached')\n"
    )
    control_env = {**{key: value for key, value in os.environ.items()
                      if not key.startswith("GIT_")},
                   "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.fsmonitor",
                   "GIT_CONFIG_VALUE_0": str(hook)}
    subprocess.run(["git", "-C", str(snapshot), "status", "--porcelain=v1"],
                   env=control_env, capture_output=True, text=True, timeout=15)
    assert hook_ran.exists()
    hook_ran.unlink()
    with UpdateLock(install_root=installed):
        environment = {**os.environ, "GIT_CONFIG_COUNT": "1",
                       "GIT_CONFIG_KEY_0": "core.fsmonitor",
                       "GIT_CONFIG_VALUE_0": str(hook), "GIT_DIR": str(installed)}
        with child_spawn(environment) as spawn:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code, str(snapshot / "hermes_cli/local_update_entry.py"),
                 str(installed), str(home), str(snapshot), target],
                env=environment, capture_output=True, text=True, timeout=45, **spawn)
    assert result.returncode == 0, result.stderr
    assert "frozen bootstrap reached" in result.stdout
    assert not poison.exists()
    assert not hook_ran.exists()
    assert _git(snapshot, "rev-parse", "HEAD") == target
    assert _git(snapshot, "status", "--porcelain=v1", "--untracked-files=all") == ""


@pytest.mark.platforms("posix")
def test_local_git_rejects_inherited_fsmonitor_and_repository_override(tmp_path, monkeypatch):
    from hermes_cli.update_local import local_git

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-b", "main")
    (checkout / "tracked").write_text("unchanged", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    executed = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor"
    hook.write_text(f"#!/bin/sh\nprintf executed > '{executed}'\n", encoding="utf-8")
    hook.chmod(0o700)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.fsmonitor")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(hook))
    monkeypatch.setenv("GIT_DIR", str(tmp_path))
    assert local_git(checkout, "status", "--porcelain=v1").stdout == ""
    assert not executed.exists()


@pytest.mark.platforms("posix")
def test_held_installation_cannot_be_borrowed_through_free_other_store(tmp_path, monkeypatch):
    from hermes_cli.update_lock import UpdateLock

    install = tmp_path / "checkout"
    home = tmp_path / "home"
    install.mkdir()
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store-a" / "tools"))
    child_code = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from hermes_cli.update_lock import UpdateLock; "
        f"lock=UpdateLock(install_root=Path({str(install)!r})); "
        "won=lock.acquire(); lock.release(); raise SystemExit(0 if won else 1)"
    )
    with UpdateLock(install_root=install):
        environment = {**os.environ, "HERMES_RUNTIME_DIR": str(tmp_path / "store-b" / "tools")}
        result = subprocess.run([sys.executable, "-I", "-S", "-c", child_code],
                                env=environment, capture_output=True, text=True, timeout=15)
        assert result.returncode != 0, result.stderr
        assert (home / ".hermes-update-in-progress").is_file()
    result = subprocess.run([sys.executable, "-I", "-S", "-c", child_code],
                            env={**os.environ, "HERMES_RUNTIME_DIR": str(tmp_path / "store-b" / "tools")},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr

@pytest.mark.platforms("posix")
@pytest.mark.parametrize("foreign_inode", [False, True])
def test_inherited_claim_requires_held_expected_inode(tmp_path, monkeypatch, foreign_inode):
    from pm.writer_claim import domains

    installed, home = tmp_path / "installed", tmp_path / "home"
    installed.mkdir()
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    expected = domains(installed)
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
    other = tmp_path / "unrelated-claim"
    other.touch(mode=0o600)
    actual = (other, *expected[1:]) if foreign_inode else expected
    fds = [os.open(path, os.O_RDWR) for path in actual]
    try:
        env = {**os.environ, "HERMES_RETAINED_WRITER_CLAIMS":
               json.dumps({"paths": list(map(str, expected)), "fds": fds})}
        code = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            "from pm.writer_claim import writer_claim; "
            f"\nwith writer_claim(install=Path({str(installed)!r}), inherited_only=True):\n"
            " print('admitted')\n"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                                env=env, pass_fds=fds, capture_output=True, text=True, timeout=15)
        assert result.returncode != 0
        assert "admitted" not in result.stdout
    finally:
        for fd in fds:
            os.close(fd)

@pytest.mark.platforms("posix")
def test_native_intent_skips_pending_ordinary_completion_before_marker(tmp_path, monkeypatch):
    """Both explicit local commands reach the real bootstrap without repairing a stale tail."""
    from hermes_cli.venv_sync import completion_pending_path

    root = tmp_path / "installed"
    root.mkdir()
    for name in ("hermes_bootstrap.py", "hermes_constants.py", "hermes_startup_watchdog.py"):
        shutil.copy2(ROOT / name, root / name)
    for name in ("pm", "hermes_cli", "hermes_platform"):
        shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "pyproject.toml").write_text("[project]\nname='hermes-agent'\nversion='1.0'\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    pending = completion_pending_path(root)
    pending.parent.mkdir(parents=True)
    pending.write_text("ordinary completion still owed\n")
    env = {**os.environ, "HERMES_RUNTIME_DIR": str(tmp_path / "tools")}
    env.pop("HERMES_LOCAL_UPDATE_CONTEXT", None)
    for argv in (("--source-repo", str(tmp_path / "reviewed"), "--target", "a" * 40),
                 ("--rollback-local-update", str(tmp_path / "capsule"))):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                "sys.argv=['hermes', 'update', *sys.argv[2:]]; "
                "import hermes_bootstrap; print('bootstrap reached')")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(root), *argv],
                                cwd=root, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "bootstrap reached" in result.stdout
        assert pending.read_text() == "ordinary completion still owed\n"
        assert not (root / ".update-incomplete").exists()

