"""Managed-update review regressions (independent-astra-reviews/managed-update), RED first.

These are the test-first corrections for the coordinated managed-update review;
the product fixes do not exist yet, so the tests below fail against the current
framework. Each test drives a real consumer surface (the custody-composed build
runner, the offline local-admission preflight, the offline update-completion
build) with real files and real child processes — no mocks on the paths under
test, no fixture-policy escapes.

Owned defects (review findings, 2026-10-25):
* P1 combined checkout+writer custody: ``source_build.run_in_custody`` forwards
  ``contained_command()``'s checkout-lock ``pass_fds`` into
  ``pm.progress.run_contained``, which also forwards the writer claim's
  ``pass_fds`` — duplicate keyword, ``TypeError`` before any builder spawns
  (``pm/progress.py:160-161``). After a merge, the ``_REAP_TREE`` custody
  launcher must hand the UNION through to the real command
  (``hermes_cli/update_custody.py:707``).
* P1 offline preflight: ``stage_local_inputs`` validates with ``engine.check()``
  without the resolved workspace lock, so the lock's intentional
  ``[tool.uv] override-dependencies`` resolution (browser-harness's Pillow) is
  rejected as unexpected (``hermes_cli/local_update_preflight.py:140``).
* P1 removed-core migration: offline ``build_update_products`` runs
  ``migrate_left_core()`` without an offline policy — it contacts the live
  plugin catalog (``plugin_catalog.fetch_live_catalog`` → ``httpx.get``) and
  swallows the missing-plugin failure, completing with the feature unavailable
  (``hermes_cli/source_build.py:309-316``).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests.pm._fixtures import _wheel

REPO = Path(__file__).resolve().parents[2]

# pm.writer_claim's inheritance channel: the JSON the parent writes so bounded
# children learn which claim files they must have inherited (paths + parent fds).
CLAIMS_ENV = "HERMES_RETAINED_WRITER_CLAIMS"


# ── Combined checkout + writer custody child execution ────────────────────────

# The stand-in build command: a real child that records which kernel claim
# descriptors it inherited at its own runtime. The launcher stands between this
# process and the command, exactly as the Node build does.
_CUSTODY_CHILD = textwrap.dedent("""
    import json, os
    inherited = set()
    for entry in os.listdir("/proc/self/fd"):
        if entry.isdigit():
            try:
                inherited.add(os.path.realpath(os.readlink(f"/proc/self/fd/{entry}")))
            except OSError:
                pass
    claims = json.loads(os.environ["HERMES_RETAINED_WRITER_CLAIMS"])
    proof = {
        "pid": os.getpid(),
        "marker": "ran",
        "checkout_fd": os.path.realpath(os.environ["HERMES_TEST_LOCK"]) in inherited,
        "claim_fds": [os.path.realpath(p) in inherited for p in claims["paths"]],
    }
    with open(os.environ["HERMES_TEST_PROOF"], "w") as fh:
        json.dump(proof, fh)
""")


@pytest.mark.platforms("posix")
def test_locked_source_build_child_runs_holding_every_retained_claim(tmp_path, monkeypatch):
    """The real combined custody path executes the build child and both real
    claims outlive it: under a held ``UpdateLock`` (checkout lock + its writer
    claim), ``run_in_custody`` must run the command — with the checkout lock
    descriptor AND every writer-claim descriptor inherited — and must still
    hold both claims after the child finished."""
    from hermes_cli import update_lock as ul
    from hermes_cli.source_build import run_in_custody

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    repo = tmp_path / "installed"
    repo.mkdir()

    proof = tmp_path / "child-proof.json"
    command = [sys.executable, "-c", _CUSTODY_CHILD]

    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=repo)
    assert lock.acquire(), lock.holder
    # The update's own writer claim is live from here on; its inheritance JSON
    # names the three claim files every owned child must receive.
    claim_paths = json.loads(os.environ[CLAIMS_ENV])["paths"]
    # Built after the acquire: the writer claim's inheritance JSON must reach
    # the build child, and only the held update publishes it.
    env = {**os.environ,
           "HERMES_TEST_LOCK": str(ul.checkout_lock_path(repo)),
           "HERMES_TEST_PROOF": str(proof)}
    try:
        # The build under both locks: today this raises TypeError (duplicate
        # pass_fds) before the launcher ever spawns the command.
        run_in_custody(repo, command, "probe build", env=env, cwd=repo)

        assert proof.is_file(), "the custody launcher never ran the build command"
        child = json.loads(proof.read_text())
        assert child["marker"] == "ran"
        assert child["pid"] != os.getpid()
        assert child["checkout_fd"], "the build command ran without the checkout lock descriptor"
        assert all(child["claim_fds"]), \
            f"the custody launcher dropped writer-claim descriptors for the command: {child}"

        # Both claims are still this process's until the update releases them.
        assert ul.holds_checkout_lock(repo)
        for claim_path in claim_paths:
            probe = os.open(claim_path, os.O_RDWR)
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                os.close(probe)
    finally:
        lock.release()

    # The hold was real: after release a contender takes the installation claim.
    if claim_paths:
        probe = os.open(claim_paths[0], os.O_RDWR)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(probe, fcntl.LOCK_UN)
        finally:
            os.close(probe)
    assert not ul.holds_checkout_lock(repo)


# ── Offline preflight: lock-aware dependency validation ───────────────────────

def _stage_tool_store(source: Path, store: Path, target: str, host_python: str) -> None:
    """Publish the lock's python+uv tools into *store* the way an installed
    preparation looks: real store entries (a shim interpreter, the real uv) and
    recorded facts bound to the framework lock's own pins."""
    from pm.lock import Facts, Lockfile
    from pm.registry import get_package
    from pm.store import tree_digest

    real = Lockfile(REPO / "pm" / "lock.json")
    lock = Lockfile(source / "pm" / "lock.json")
    for name in ("python", "uv"):
        version, artifacts = real.version(name), real.artifacts(name, target)
        assert version and artifacts, f"the framework lock pins {name} for {target}"
        lock.set_pin(name, version, {target: artifacts})
    lock.save()

    facts = Facts(store / "facts.json")
    for name in ("python", "uv"):
        package = get_package(name)
        version = lock.version(name)
        entry = store / package.store_entry(version, target)
        binary = package.binary(entry, target)
        binary.parent.mkdir(parents=True, exist_ok=True)
        if name == "python":
            binary.write_text(f"#!/bin/sh\nexec {shlex.quote(host_python)} \"$@\"\n", encoding="utf-8")
        else:
            shutil.copy2(shutil.which("uv"), binary)
        binary.chmod(0o755)
        facts.record(name, version, entry.name, package.env(entry, target), store,
                     target=target, artifacts=[row["sha256"] for row in lock.artifacts(name, target)],
                     digest=tree_digest(entry))


def _admit_override_preflight(tmp_path, monkeypatch):
    """``stage_local_inputs`` must validate the target venv against the lock it
    was synced from: a candidate whose ``uv.lock`` resolves an intentional
    ``override-dependencies`` entry (the browser-harness/Pillow shape) is
    admitted, with the overridden version actually installed."""
    from hermes_cli import main_install_repair
    from hermes_cli.local_update_preflight import stage_local_inputs
    from pm.lock import Lockfile
    from pm.store import current_target

    uv = shutil.which("uv")
    assert uv, "the preflight contract requires real uv"
    host_python = str(Path(sys._base_executable).resolve())
    target = current_target()

    home = tmp_path / "home"
    home.mkdir()
    store = tmp_path / "store"
    store.mkdir()
    installed = tmp_path / "installed"
    installed.mkdir()
    source = tmp_path / "candidate"
    (source / "pm").mkdir(parents=True)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    _wheel(wheels, "provider", "1.0", requirements=("fixturedep>=2.0",))
    _wheel(wheels, "fixturedep", "1.0")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    # The extras channel is not this defect: the existing suite substitutes it
    # the same way (test_local_source_update.py).
    monkeypatch.setattr(main_install_repair, "configured_update_extras", lambda *_a, **_k: [])
    # Resolution must survive the harness's ambient UV_NO_CONFIG=1 (which makes
    # uv skip the candidate manifest's [tool.uv]): UV_FIND_LINKS and UV_NO_INDEX
    # are index/transport knobs pm forwards into EVERY engine uv command
    # (pm.index_config.FORWARDED_UV_SETTINGS), so the staged admission resolves
    # the wheels the same way a real air-gapped preflight does — never a registry.
    monkeypatch.setenv("UV_FIND_LINKS", str(wheels))
    monkeypatch.setenv("UV_NO_INDEX", "1")

    # The candidate resolves one dependency against its manifest's explicit
    # override — exactly the lock shape the reviewer's browser-harness case has.
    (source / "pyproject.toml").write_text(
        "[project]\n"
        'name = "preflight-proof"\n'
        'version = "1"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = ["provider==1.0"]\n'
        "[project.optional-dependencies]\n"
        "all = []\n"
        "[tool.uv]\n"
        "package = false\n"
        "no-index = true\n"
        "default-groups = []\n"
        f"find-links = [{json.dumps(wheels.as_posix())}]\n"
        'override-dependencies = ["fixturedep==1.0"]\n',
        encoding="utf-8",
    )
    # The lock step mirrors the engine's own environment policy: ambient
    # UV_NO_CONFIG is dropped — pm.environment._base_environment strips every
    # non-forwarded UV_* for the engine's uv commands too — so the candidate
    # manifest's [tool.uv] (its intentional override-dependencies) applies,
    # with the wheels directory pinned by explicit flags as well.
    locked = subprocess.run(
        [uv, "lock", "--offline", "--no-index", "--find-links", str(wheels),
         "--python", host_python], cwd=source,
        env={**{key: value for key, value in os.environ.items() if key != "UV_NO_CONFIG"},
             "UV_OFFLINE": "1", "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
             "UV_PYTHON_DOWNLOADS": "never"},
        capture_output=True, text=True, timeout=300, check=False)
    assert locked.returncode == 0, locked.stdout + locked.stderr

    _stage_tool_store(source, store, target, host_python)

    stage_local_inputs(source, installed, tmp_path / "staging", [])

    # Admission consulted THE RESOLVED LOCK: the workspace lock records the
    # intentional override, and the override's version is what got installed.
    import tomllib

    resolved = tomllib.loads((tmp_path / "staging" / "workspace" / "uv.lock").read_text())
    overrides = resolved.get("manifest", {}).get("overrides", [])
    assert any(entry.get("name") == "fixturedep" for entry in overrides), \
        "the resolved workspace lock lost the intentional override"
    venv_python = tmp_path / "staging" / "venv" / "bin" / "python"
    installed_version = subprocess.run(
        [str(venv_python), "-I", "-c",
         "import importlib.metadata as m; print(m.version('fixturedep'))"],
        capture_output=True, text=True, timeout=60, check=False)
    assert installed_version.returncode == 0, installed_version.stderr
    assert installed_version.stdout.strip() == "1.0"
    return home, source, installed


@pytest.mark.platforms("posix")
def test_offline_preflight_admits_the_locks_intentional_override(tmp_path, monkeypatch):
    _admit_override_preflight(tmp_path, monkeypatch)


@pytest.mark.platforms("posix")
def test_offline_preflight_refuses_a_missing_configured_plugin_before_mutation(
        tmp_path, monkeypatch):
    from hermes_cli.local_update_preflight import stage_local_inputs

    home, source, installed = _admit_override_preflight(tmp_path, monkeypatch)
    _home_using_removed_core_feature(home)
    config_before = (home / "config.yaml").read_bytes()
    installed_state = installed / "application-state"
    installed_state.write_text("unchanged\n")
    with pytest.raises(RuntimeError, match="Configured plugin 'spotify' is unavailable locally"):
        stage_local_inputs(source, installed, tmp_path / "missing-plugin-admission", [])
    assert (home / "config.yaml").read_bytes() == config_before
    assert installed_state.read_text() == "unchanged\n"


# ── Offline completion: removed-core migration must stay offline, fail loud ───

class _RecordingCatalog:
    """A loopback host that records every request and answers the live-catalog
    URL with a catalog whose 'spotify' entry clones back to this host."""

    def __init__(self):
        self.requests: list[str] = []
        self._lock = threading.Lock()
        # Bound (and therefore port-numbered) at construction, served by
        # running(): the world helper points LIVE_CATALOG_URL here before the
        # context is entered.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def serve_entry(self, name: str) -> dict:
        return {
            "name": name,
            # The same pin label as any catalog row; the repo is this host, so
            # an install attempt can only ever reach the loopback recorder.
            "repo": self.url.replace("http://", "https://", 1) + f"/{name}.git",
            "sha": "b" * 40,
            "description": "offline-contact probe entry",
            "maintainer": "test",
            "category": "tools",
        }

    def _handler(self):
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with recorder._lock:
                    recorder.requests.append(self.path)
                if self.path.endswith("plugin-catalog.json"):
                    payload = json.dumps({
                        "entries": [recorder.serve_entry("spotify")],
                        "removed": [],
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, *_args):
                pass

        return Handler

    @contextlib.contextmanager
    def running(self):
        thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self._server.shutdown()
            thread.join(timeout=5)
            self._server.server_close()


def _home_using_removed_core_feature(home: Path) -> None:
    """A profile that still selects Spotify in core, predating the split, with
    no replacement plugin installed."""
    (home / "config.yaml").write_text(
        "_config_version: 1\ntoolsets: [spotify]\n", encoding="utf-8")


def _offline_update_world(tmp_path, monkeypatch):
    """The offline completion inputs: one Spotify-using home, a minimal real
    update source, and the live catalog pointed at a recording loopback host."""
    from hermes_cli import main_install_repair, plugin_catalog

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("HERMES_UPDATE_OFFLINE", "1")
    _home_using_removed_core_feature(home)
    # The extras channel is not the defect under test; without this the
    # feature-dependency step fails on the synthetic root BEFORE the migration
    # (the first RED run's false green came from exactly that unrelated raise).
    monkeypatch.setattr(main_install_repair, "configured_update_extras", lambda *_a, **_k: [])
    # A real update source carries a manifest.
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        "[project]\n"
        'name = "offline-update-proof"\n'
        'version = "1"\n'
        'requires-python = ">=3.11"\n'
        "dependencies = []\n"
        "[project.optional-dependencies]\n"
        "all = []\n",
        encoding="utf-8")
    # A fresh catalog cache and no remembered failure, so the first migration
    # really makes the live request the review flagged.
    monkeypatch.setattr(plugin_catalog, "_live_fetch_failed_until", 0)
    recorder = _RecordingCatalog()
    monkeypatch.setattr(plugin_catalog, "LIVE_CATALOG_URL",
                        recorder.url + "/docs/api/plugin-catalog.json")
    return project, recorder


@pytest.mark.platforms("posix")
def test_offline_update_never_contacts_the_plugin_catalog(tmp_path, monkeypatch):
    """Offline completion performs no remote acquisition: building the update
    products with the offline flags set must not send the live-catalog request
    (nor any install traffic) for a home still using a removed-core feature."""
    from hermes_cli.source_build import build_update_products

    project, recorder = _offline_update_world(tmp_path, monkeypatch)
    with recorder.running():
        with contextlib.suppress(Exception):  # the failure mode itself is pinned below
            build_update_products(project, desktop=False, offline=True)
    assert recorder.requests == [], \
        f"offline completion contacted the catalog host anyway: {recorder.requests}"


@pytest.mark.platforms("posix")
def test_offline_update_does_not_swallow_a_missing_left_core_plugin(tmp_path, monkeypatch):
    """A home using a removed-core feature without its replacement plugin must
    fail the offline completion loudly: it may not return successfully while the
    configured feature stays unavailable."""
    from hermes_cli.source_build import build_update_products

    project, recorder = _offline_update_world(tmp_path, monkeypatch)
    with recorder.running():
        with pytest.raises(Exception, match="(?i)spotify|plugin|migrat|offline|catalog"):
            build_update_products(project, desktop=False, offline=True)


