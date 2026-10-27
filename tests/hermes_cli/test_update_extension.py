"""Consumer witness and trust contracts for the generic Root-pinned
update-extension seam of native local updates.

The native local update takes a caller-owned context
(``--update-extension-context`` / request key ``update_extension_context``).
The descriptor-pinned entrypoint's ``validate_update`` runs after generic
descriptor/context validation but before backup, capsule capture or any
native mutation; its ``complete_update`` runs inside the native completion,
only after ``_complete_selected`` succeeds and before any capsule seal, and
a plain failure keeps actual effects and follows the strict-local settlement.

Extension imports remain inside fixtures and tests, so collection does not
execute entrypoints or adapt the production Root trust boundary.

Only the transport and the product/launcher tail run substituted: the
selected-interpreter transport becomes the real two phases in this one
process (``_finish`` then ``_seal``, the same seam
``isolated_source_completion`` sanctions), and ``complete_source_checkout``
stands in for the dependency/product tail. Everything the tests assert is
real: both Git repositories and their fast-forward, the capsule
capture/seal/restore, the receipt journal, the native completion itself and
the pinned entrypoint bytes executing with relative sibling imports through
the normal meta_path machinery.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.platforms("posix")

DESCRIPTOR_NAME = "update-extension.json"
CONTEXT_NAME = "update-extension-context.json"
EFFECT_RELATIVE = Path("update-extension") / "effect.json"

#: Opaque JSON object the guard captured into the context envelope.
WITNESS_PAYLOAD = {
    "witness": "native-finish-consumer",
    "cohort": ["update-extension/effect.json"],
}

# The pinned entrypoint: fixed callbacks, a relative import that must resolve
# only to descriptor-pinned siblings, and one real file effect inside the
# capsule-covered home.
_ENTRYPOINT = '''\
"""Witness callbacks for the generic update-extension contract."""
import json
from pathlib import Path

from . import witness_sidecar


def validate_update(payload, *, install_root, home, target, frozen):
    """Read-only admission; return the captured JSON payload."""
    return {"payload": payload, "sidecar": witness_sidecar.NAME,
            "install_root": str(install_root), "home": str(home),
            "target": target, "frozen": str(frozen)}


def complete_update(payload, *, install_root, home, target, frozen):
    """Apply the real cohort: one new file inside the capsule-covered home."""
    effect = Path(home) / "update-extension" / "effect.json"
    effect.parent.mkdir(parents=True, exist_ok=True)
    effect.write_text(json.dumps({
        "payload": payload, "sidecar": witness_sidecar.NAME,
        "install_root": str(install_root), "home": str(home),
        "target": target, "frozen": str(frozen)}), encoding="utf-8")
    return {"effect": str(effect), "sidecar": witness_sidecar.NAME}
'''

# A pinned entrypoint whose completion phase fails after a real effect: the
# native settlement must keep the effect and still seal the capsule.
_FAILING_ENTRY = '''\
"""Witness callbacks whose completion phase fails after real effects."""


def validate_update(payload, *, install_root, home, target, frozen):
    return dict(payload)


def complete_update(payload, *, install_root, home, target, frozen):
    from pathlib import Path

    effect = Path(home) / "update-extension" / "effect.json"
    effect.parent.mkdir(parents=True, exist_ok=True)
    effect.write_text("applied before the refusal\\n", encoding="utf-8")
    raise RuntimeError("witness extension completion refused")
'''

_SIDECAR = 'NAME = "witness-sidecar"\n'

# A pinned subpackage whose initializer exports the admission rules used by
# the entrypoint's ``from .helpers import validate_payload``.
_HELPERS_RULES = 'MODULE_NAME = "pinned-payload-rules"\n'

_HELPERS_INIT = '''\
"""Initializer-exported admission rules; importing this package runs me."""

from . import payload_rules

INITIALIZED = [payload_rules.MODULE_NAME]


def validate_payload(payload):
    return {"ok": payload.get("witness") == "native-finish-consumer",
            "init": list(INITIALIZED), "rule": payload_rules.MODULE_NAME}
'''

_SUBPACKAGE_ENTRY = '''\
"""Callbacks resolved through a pinned subpackage initializer."""

import json
from pathlib import Path

from .helpers import validate_payload


def validate_update(payload, *, install_root, home, target, frozen):
    return {"payload": payload, "admission": validate_payload(payload),
            "install_root": str(install_root), "home": str(home),
            "target": target, "frozen": str(frozen)}


def complete_update(payload, *, install_root, home, target, frozen):
    effect = Path(home) / "update-extension" / "effect.json"
    effect.parent.mkdir(parents=True, exist_ok=True)
    effect.write_text(json.dumps({
        "admission": validate_payload(payload["payload"]),
        "install_root": str(install_root), "home": str(home),
        "target": target, "frozen": str(frozen)}), encoding="utf-8")
    return {"effect": str(effect), "init": validate_payload(payload["payload"])["init"]}
'''

# A pinned root initializer: names it exports must be reachable through the
# unique package namespace itself (``from . import ROOT_INIT``).
_ROOT_INIT = 'ROOT_INIT = "root-init-ran"\n'

_ROOT_INIT_ENTRY = '''\
"""Callbacks using a name the pinned root initializer exports."""

import json
from pathlib import Path

from . import ROOT_INIT


def validate_update(payload, *, install_root, home, target, frozen):
    return {"payload": payload, "root_init": ROOT_INIT,
            "install_root": str(install_root), "home": str(home),
            "target": target, "frozen": str(frozen)}


def complete_update(payload, *, install_root, home, target, frozen):
    effect = Path(home) / "update-extension" / "effect.json"
    effect.parent.mkdir(parents=True, exist_ok=True)
    effect.write_text(json.dumps({"root_init": ROOT_INIT, "target": target}),
                      encoding="utf-8")
    return {"effect": str(effect), "root_init": ROOT_INIT}
'''

# A pinned initializer used AS the descriptor entrypoint: both callbacks live
# in the ``__init__.py`` itself and import a pinned relative sibling.
_INITIALIZER_SIDECAR = 'NAME = "initializer-sidecar"\n'

_INITIALIZER_ENTRY = '''\
"""Callbacks defined in the pinned initializer itself."""

import json
from pathlib import Path

from . import initializer_sidecar


def validate_update(payload, *, install_root, home, target, frozen):
    return {"payload": payload, "sidecar": initializer_sidecar.NAME,
            "install_root": str(install_root), "home": str(home),
            "target": target, "frozen": str(frozen)}


def complete_update(payload, *, install_root, home, target, frozen):
    effect = Path(home) / "update-extension" / "effect.json"
    effect.parent.mkdir(parents=True, exist_ok=True)
    effect.write_text(json.dumps({
        "payload": payload, "sidecar": initializer_sidecar.NAME,
        "install_root": str(install_root), "home": str(home),
        "target": target, "frozen": str(frozen)}), encoding="utf-8")
    return {"effect": str(effect), "sidecar": initializer_sidecar.NAME}
'''

# Two pinned members that map to the same import identity: the root module
# ``hook.py`` and the package initializer ``hook/__init__.py``. Each defines
# independent callbacks and records its own marker when executed.
_COLLIDING_ENTRY = '''\
"""Root-module callbacks; the bundle also pins hook/__init__.py."""
import os
from pathlib import Path


def _mark(name):
    marks = Path(os.environ["COLLISION_MARKS"])
    marks.mkdir(parents=True, exist_ok=True)
    (marks / name).write_text("ran", encoding="utf-8")


_mark("entry-toplevel.json")


def validate_update(payload, *, install_root, home, target, frozen):
    _mark("entry-validate.json")
    return {"source": "entry", "payload": payload}


def complete_update(payload, *, install_root, home, target, frozen):
    _mark("entry-complete.json")
    return {"source": "entry"}
'''

_COLLIDING_INIT = '''\
"""Package-initializer callbacks; the bundle also pins hook.py."""
import os
from pathlib import Path


def _mark(name):
    marks = Path(os.environ["COLLISION_MARKS"])
    marks.mkdir(parents=True, exist_ok=True)
    (marks / name).write_text("ran", encoding="utf-8")


_mark("init-toplevel.json")


def validate_update(payload, *, install_root, home, target, frozen):
    _mark("init-validate.json")
    return {"source": "init", "payload": payload}


def complete_update(payload, *, install_root, home, target, frozen):
    _mark("init-complete.json")
    return {"source": "init"}
'''

# A pinned entrypoint whose completion phase applies one real effect and then
# exits instead of returning: the native settlement must keep the effect and
# finish nonzero whatever exit shape the callback used.
_EXIT_AFTER_EFFECT_ENTRY = '''\
"""Callbacks whose completion phase exits after one real effect."""


def validate_update(payload, *, install_root, home, target, frozen):
    return dict(payload)


def complete_update(payload, *, install_root, home, target, frozen):
    import sys
    from pathlib import Path

    effect = Path(home) / "update-extension" / "effect.json"
    effect.parent.mkdir(parents=True, exist_ok=True)
    effect.write_text("applied before the exit\\n", encoding="utf-8")
    sys.exit(__EXIT_ARGUMENT__)
'''


def _exiting_entry(argument) -> str:
    return _EXIT_AFTER_EFFECT_ENTRY.replace("__EXIT_ARGUMENT__", repr(argument))


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


def _pin(path: Path) -> dict:
    info = path.stat()
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": info.st_size, "mode": stat.S_IMODE(info.st_mode)}


def _write_extension_bundle(base: Path, *, install_root: str, home: str, target: str,
                            entry: str = _ENTRYPOINT, sidecar: str = _SIDECAR,
                            members: dict[str, str] | None = None,
                            entrypoint: str = "ext.py"):
    """Write one complete Root-prepared bundle: descriptor, pinned members and
    the caller-owned correlated context file.

    ``members`` overrides the whole pinned file set (names may reach into
    subdirectories, e.g. ``helpers/__init__.py``); ``entrypoint`` names the
    descriptor's entrypoint pin; the defaults keep the flat witness layout."""
    bundle = base / "update-extension"
    bundle.mkdir(mode=0o755)
    if members is None:
        members = {"ext.py": entry, "witness_sidecar.py": sidecar,
                   "payload.json": json.dumps(WITNESS_PAYLOAD) + "\n"}
    for name, text in members.items():
        member = bundle / name
        member.parent.mkdir(parents=True, exist_ok=True)
        parent = member.parent
        while parent != bundle.parent:
            parent.chmod(0o755)
            parent = parent.parent
        member.write_text(text, encoding="utf-8")
        member.chmod(0o640)
    descriptor = {
        "version": 1,
        "entrypoint": entrypoint,
        "files": {name: _pin(bundle / name) for name in members},
        "transaction": {"install_root": install_root, "home": home, "target": target},
    }
    descriptor_bytes = json.dumps(descriptor).encode("utf-8")
    (bundle / DESCRIPTOR_NAME).write_bytes(descriptor_bytes)
    (bundle / DESCRIPTOR_NAME).chmod(0o600)
    context = {"descriptor_path": str(bundle / DESCRIPTOR_NAME),
               "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
               "payload": WITNESS_PAYLOAD}
    context_file = base / CONTEXT_NAME
    context_file.write_text(json.dumps(context), encoding="utf-8")
    context_file.chmod(0o600)
    return SimpleNamespace(bundle=bundle, context=context_file,
                           descriptor=bundle / DESCRIPTOR_NAME, members=members)


def _rewrite_descriptor(built, descriptor: dict) -> None:
    """Rewrite the descriptor and its digest pin inside the context file."""
    data = json.dumps(descriptor).encode("utf-8")
    built.descriptor.write_bytes(data)
    built.descriptor.chmod(0o600)
    context = json.loads(built.context.read_text())
    context["descriptor_sha256"] = hashlib.sha256(data).hexdigest()
    built.context.write_text(json.dumps(context), encoding="utf-8")
    built.context.chmod(0o600)


def _adapt_root_pinned_trust(monkeypatch) -> None:
    """Fixture-only UID/ancestry adaptation for the generic trust anchor.

    Production anchors the descriptor tree in Root-0 ownership through ``/``;
    pytest's temporary tree has no such anchor, so this fixture adapts UID and
    ancestry inside its own namespace, exactly as the retired paired trust
    fixtures did. The generic helpers are imported and called qualified at
    call time, so patching the module attributes is the whole adaptation.
    Everything else — bundle pins, digest re-verification, entrypoint
    execution — stays real. While ``hermes_cli.update_extension`` is absent
    the adaptation is inert: the native witness then proves by assertion that
    the future request key is ignored.
    """
    try:
        module = importlib.import_module("hermes_cli.update_extension")
    except ModuleNotFoundError:
        return
    fixture_uid = os.geteuid()
    monkeypatch.setattr(module, "ROOT_UID", fixture_uid)

    def fixture_ancestry(directory, label, *, owner=None):
        expected = fixture_uid if owner is None else owner
        info = os.stat(directory)
        if info.st_uid != expected:
            raise ValueError(f"{label} is not owned by the fixture trust anchor: {directory}")
        if stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError(f"{label} is group or world writable: {directory}")

    monkeypatch.setattr(module, "require_protected_ancestry", fixture_ancestry)


class _RootPreparedBundleOS:
    """Fixture-only ``os`` facade for one bundle namespace, installed into
    ``hermes_cli.update_extension`` alone.

    Every path inside the bundle directory — and every lexical ancestor
    through ``/`` — stats exactly as Root prepared it on a real host: uid 0,
    with ancestors presented as plain 0755 directories. Every other path, the
    caller-owned context file included, keeps its real ownership, and modes,
    link counts, sizes and device/inode identity inside the bundle stay real.
    Unlike ``_adapt_root_pinned_trust`` this never patches ``ROOT_UID``: the
    production descriptor/bundle checks keep demanding uid 0 through the real
    helpers, so a context file owned by anyone but the actual caller is
    caught by the real ownership check."""

    def __init__(self, real_os, bundle, *, foreign=(), context=None, context_uid=None):
        self._os = real_os
        self._bundle = os.fspath(bundle)
        self._foreign = {os.fspath(path) for path in foreign}
        self._context = os.fspath(context) if context is not None else None
        self._context_uid = context_uid

    def _adapts(self, path) -> bool:
        candidate = os.fspath(path)
        return (candidate == self._bundle
                or candidate.startswith(self._bundle + os.sep)
                or candidate == os.sep
                or self._bundle.startswith(candidate + os.sep))

    @staticmethod
    def _presented(info, *, uid, permissions=None):
        mode = info.st_mode if permissions is None else (info.st_mode & ~0o777) | permissions
        return os.stat_result((mode, info.st_ino, info.st_dev, info.st_nlink, uid,
                               info.st_gid, info.st_size, info.st_atime, info.st_mtime,
                               info.st_ctime))

    def _adapted(self, info, path):
        candidate = os.fspath(path)
        if candidate in self._foreign:
            return info
        if self._context is not None and candidate == self._context:
            if self._context_uid is None:
                return info
            return self._presented(info, uid=self._context_uid)
        if not self._adapts(candidate):
            return info
        if candidate == os.sep or self._bundle.startswith(candidate + os.sep):  # a lexical ancestor
            return self._presented(info, uid=0, permissions=0o755)
        return self._presented(info, uid=0)

    def lstat(self, path):
        return self._adapted(self._os.lstat(path), path)

    def scandir(self, path):
        return _AdaptedScanDir(self._os.scandir(path), self)

    def __getattr__(self, name):
        return getattr(self._os, name)


class _AdaptedScanDir:
    def __init__(self, entries, adapter):
        self._entries = entries
        self._adapter = adapter

    def __enter__(self):
        self._entries.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._entries.__exit__(*exc_info)

    def __iter__(self):
        for entry in self._entries:
            yield _AdaptedDirEntry(entry, self._adapter)


class _AdaptedDirEntry:
    def __init__(self, entry, adapter):
        self._entry = entry
        self._adapter = adapter

    @property
    def name(self):
        return self._entry.name

    @property
    def path(self):
        return self._entry.path

    def stat(self, *, follow_symlinks=True):
        return self._adapter._adapted(self._entry.stat(follow_symlinks=follow_symlinks),
                                      self._entry.path)

    def __getattr__(self, name):
        return getattr(self._entry, name)


def _pin_bundle_namespace_as_root_prepared(monkeypatch, built, *, foreign=(),
                                           context_uid=None):
    """Serve ``hermes_cli.update_extension`` its bundle namespace exactly as
    Root prepared it, leaving the caller-owned context file's real ownership
    in place (``context_uid`` overrides only the context's presented owner,
    ``foreign`` lists bundle paths that keep their real owner)."""
    module = importlib.import_module("hermes_cli.update_extension")
    monkeypatch.setattr(module, "os", _RootPreparedBundleOS(
        os, built.bundle, foreign=foreign, context=built.context,
        context_uid=context_uid))


def _in_process_completion(request: dict) -> dict:
    """The real selected-interpreter transport's two phases, in this process.

    Only the transport is replaced: ``_finish`` (the native completion, where
    the update extension must run before any seal) and ``_seal`` (the real
    capsule seal) are the real functions, driven exactly as ``run_completion``
    drives them across the process seam — including the ordinary nonzero
    settlement a failing extension phase must follow.
    """
    from hermes_cli import update_completion

    with tempfile.TemporaryDirectory(prefix="witness-completion-") as directory:
        result_path = Path(directory) / "result.json"
        code = update_completion._finish(request, result_path)
        intermediate = json.loads(result_path.read_text(encoding="utf-8-sig"))
        assert "not_quiescent" not in intermediate, intermediate
        if intermediate.get("seal_pending"):
            assert intermediate.get("update_id") == request["receipt"]["update_id"]
            request["receipt"] = intermediate["receipt_data"]
            code = update_completion._seal(request, result_path)
        final = json.loads(result_path.read_text(encoding="utf-8-sig"))
    assert final["schema"] == 1 and final["update_id"] == request["receipt"]["update_id"]
    assert final["exit_code"] == code, final
    return final


@pytest.fixture
def native_environment(tmp_path, monkeypatch):
    """One real local-update transaction on an installed host, with the
    native completion seam running in-process."""
    from hermes_cli import (
        local_update_preflight, main, source_completion, update_cmd, update_inventory,
        update_receipt,
    )
    from hermes_constants import get_hermes_home

    root, source, home = tmp_path / "installed", tmp_path / "reviewed", tmp_path / "home"
    root.mkdir()
    home.mkdir()
    _git(root, "init", "-b", "main")
    (root / "version.txt").write_text("before\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "before")
    before = _git(root, "rev-parse", "HEAD")
    _git(tmp_path, "clone", "--no-local", str(root), str(source))
    (source / "version.txt").write_text("after\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "target")
    target = _git(source, "rev-parse", "HEAD")
    (home / "config.yaml").write_bytes(b"old config\n")

    # The installed host's environment: capture anchors its runtime selection
    # and restart obligation in these directories (as in the native-boundary
    # cancellation regression).
    (tmp_path / "runtime").mkdir()
    (tmp_path / "gateway-locks").mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "gateway-locks"))
    monkeypatch.setenv("HERMES_LOCAL_UPDATE_CONTEXT", "1")
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    monkeypatch.setattr(update_inventory, "collect_runtime_inventory",
                        lambda: update_inventory.UpdatePlan())
    monkeypatch.setattr(update_cmd, "_resolve_pre_update_backup_mode", lambda _args: "quick")
    monkeypatch.setattr(
        update_cmd, "_resolve_update_options",
        lambda *_args: SimpleNamespace(assume_yes=True, pre_update_version="fixture"))
    quick_snapshot = home / "state-snapshots" / "fixture"

    def capture_quick_snapshot(_args):
        quick_snapshot.mkdir(parents=True)
        (quick_snapshot / "state").write_text("before\n", encoding="utf-8")
        return "fixture"

    monkeypatch.setattr(main, "_run_pre_update_backup", capture_quick_snapshot)

    def stage_inputs(_source, _root, staging, _extras):
        tools = staging / "tools"
        tools.mkdir()
        (tools / "input").write_text("selected tool\n", encoding="utf-8")
        return []

    monkeypatch.setattr(local_update_preflight, "stage_local_inputs", stage_inputs)
    # The dependency/product/launcher tail is outside this witness's boundary.
    monkeypatch.setattr(source_completion, "complete_source_checkout",
                        lambda *_args, **_kwargs: True)
    # The transport is replaced by its real two phases in this one process.
    monkeypatch.setattr(update_cmd, "run_completion", _in_process_completion)

    def capsule_manifest() -> Path:
        found = [path for path in tmp_path.rglob("manifest.json")
                 if path.parent.parent.name == "local-update-capsules"]
        assert len(found) == 1, f"expected exactly one rollback capsule: {found}"
        return found[0]

    yield SimpleNamespace(
        base=tmp_path, root=root, source=source, home=home, before=before, target=target,
        effect=home / EFFECT_RELATIVE, home_value=str(get_hermes_home()),
        install_value=str(root.resolve()), capsule_manifest=capsule_manifest)
    update_receipt._current.set(None)


@pytest.fixture
def native_update(native_environment, monkeypatch):
    """The native transaction plus the default Root-prepared witness bundle."""
    built = _write_extension_bundle(
        native_environment.base, install_root=native_environment.install_value,
        home=native_environment.home_value, target=native_environment.target)
    # Root-0 trust cannot exist under pytest's temporary tree; adapt UID and
    # ancestry inside this fixture only.
    _adapt_root_pinned_trust(monkeypatch)
    native_environment.context = built.context
    native_environment.payload = WITNESS_PAYLOAD
    return native_environment


def test_native_finish_seals_extension_effect_and_rollback_restores_preimage(native_update, monkeypatch):
    """The extension's real effect must land before the seal and restore away after it."""
    from hermes_cli import update_cmd, update_receipt
    from hermes_cli._early_recovery import local_transaction_marker
    from hermes_cli.local_update_rollback import restore_local_update_capsule

    monkeypatch.setattr(update_receipt, "_collection_enabled_now", lambda: None)

    update_cmd._cmd_local_update(
        SimpleNamespace(source_repo=str(native_update.source),
                        target=native_update.target,
                        required_extra=[],
                        update_extension_context=str(native_update.context)),
        gateway_mode=False)

    # The native update itself is real: the checkout fast-forwarded to target.
    assert _git(native_update.root, "rev-parse", "HEAD") == native_update.target
    assert not local_transaction_marker(native_update.root).exists()

    # The extension's private effect is part of the sealed capsule postimage.
    assert native_update.effect.is_file(), (
        "native completion sealed the capsule without the extension's file effect: "
        "update_extension_context was ignored")
    landed = json.loads(native_update.effect.read_text(encoding="utf-8"))
    assert landed["payload"]["payload"] == native_update.payload
    assert landed["payload"]["sidecar"] == "witness-sidecar"
    assert landed["install_root"] == native_update.install_value
    assert landed["home"] == native_update.home_value
    assert landed["target"] == native_update.target
    assert Path(landed["frozen"]) == Path(update_cmd.__file__).resolve().parents[1]

    manifest = native_update.capsule_manifest()
    assert json.loads(manifest.read_text())["state"] == "sealed"

    # The receipt finalizer parks this run's metric receipt after the seal (consent is
    # unknowable in-process); rollback must classify and remove it, not refuse drift.
    parked = native_update.home / "telemetry" / "shared_metrics" / "pending_updates"
    assert sorted(path.name for path in parked.glob("*.json")), (
        "native completion parked no pending metric receipt after the seal")
    # Explicit acknowledged rollback restores the captured preimage, effect included.
    outcome = restore_local_update_capsule(
        manifest, expected_install_root=native_update.root,
        expected_hermes_home=native_update.home, acknowledge_data_rollback=True)
    assert outcome["before"] == native_update.before
    assert _git(native_update.root, "rev-parse", "HEAD") == native_update.before
    assert not native_update.effect.exists()
    assert (native_update.home / "config.yaml").read_bytes() == b"old config\n"
    assert not parked.exists()
    assert json.loads(manifest.read_text())["state"] == "restored"


def test_extension_phase_failure_keeps_actual_effects_and_seals_the_capsule(
        native_environment, monkeypatch):
    """A failing complete_update follows the ordinary nonzero settlement: the
    sealed postimage keeps the partial effect for explicit recovery."""
    from hermes_cli import update_cmd

    built = _write_extension_bundle(
        native_environment.base, install_root=native_environment.install_value,
        home=native_environment.home_value, target=native_environment.target,
        entry=_FAILING_ENTRY)
    _adapt_root_pinned_trust(monkeypatch)
    with pytest.raises(SystemExit) as caught:
        update_cmd._cmd_local_update(
            SimpleNamespace(source_repo=str(native_environment.source),
                            target=native_environment.target,
                            required_extra=[],
                            update_extension_context=str(built.context)),
            gateway_mode=False)
    assert caught.value.code == 1
    assert native_environment.effect.read_text(encoding="utf-8") == "applied before the refusal\n"
    manifest = native_environment.capsule_manifest()
    assert json.loads(manifest.read_text())["state"] == "sealed"
    assert _git(native_environment.root, "rev-parse", "HEAD") == native_environment.target


def test_load_and_complete_run_the_pinned_entrypoint_in_one_unique_namespace(
        tmp_path, monkeypatch):
    """Each phase re-verifies the bundle and executes the pinned bytes; the
    captured context is exactly the contract shape and admission stays
    read-only."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "3" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(tmp_path, install_root=str(install), home=str(home),
                                    target=target)
    _adapt_root_pinned_trust(monkeypatch)
    captured = update_extension.load_update_extension_context(
        built.context, install_root=install, home=home, target=target, frozen=frozen)
    assert set(captured) == {"descriptor_path", "descriptor_sha256", "payload", "frozen"}
    assert captured["descriptor_path"] == str(built.descriptor)
    assert captured["payload"] == {
        "payload": WITNESS_PAYLOAD, "sidecar": "witness-sidecar",
        "install_root": str(install), "home": str(home),
        "target": target, "frozen": str(frozen)}
    assert not (home / EFFECT_RELATIVE).exists()  # admission stays read-only

    facts = update_extension.complete_update_extension(
        captured, install_root=install, home=home, target=target)
    effect = home / EFFECT_RELATIVE
    assert facts["effect"] == str(effect)
    assert facts["sidecar"] == "witness-sidecar"
    landed = json.loads(effect.read_text(encoding="utf-8"))
    assert landed["payload"] == captured["payload"]
    # The unique execution namespace is removed; nothing leaked into sys.modules.
    assert not [name for name in sys.modules if name.startswith("hermes_update_extension_")]


@pytest.mark.parametrize("case", [
    "context_symlink", "context_mode", "context_extra_key", "payload_not_object",
    "target_mismatch", "home_mismatch", "pin_tamper", "extra_bundle_file",
    "entrypoint_not_pinned",
])
def test_load_update_extension_context_refuses_unsafe_or_mismatched_inputs(
        tmp_path, monkeypatch, case):
    """Every refusal lands before any effect: no file is written anywhere."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "1" * 40
    built = _write_extension_bundle(tmp_path, install_root=str(install), home=str(home),
                                    target=target)
    loader_target, loader_home = target, str(home)
    if case == "context_symlink":
        real = tmp_path / "real-context.json"
        real.write_bytes(built.context.read_bytes())
        real.chmod(0o600)
        built.context.unlink()
        built.context.symlink_to(real)
    elif case == "context_mode":
        built.context.chmod(0o644)
    elif case == "context_extra_key":
        context = json.loads(built.context.read_text())
        context["extra"] = True
        built.context.write_text(json.dumps(context))
        built.context.chmod(0o600)
    elif case == "payload_not_object":
        context = json.loads(built.context.read_text())
        context["payload"] = ["not", "an", "object"]
        built.context.write_text(json.dumps(context))
        built.context.chmod(0o600)
    elif case == "target_mismatch":
        loader_target = "2" * 40
    elif case == "home_mismatch":
        loader_home = str(tmp_path / "other-home")
    elif case == "pin_tamper":
        (built.bundle / "witness_sidecar.py").write_text(_SIDECAR + "# tampered\n")
        (built.bundle / "witness_sidecar.py").chmod(0o640)
    elif case == "extra_bundle_file":
        (built.bundle / "smuggled.py").write_text("VALUE = 1\n")
        (built.bundle / "smuggled.py").chmod(0o640)
    elif case == "entrypoint_not_pinned":
        descriptor = json.loads(built.descriptor.read_text())
        descriptor["entrypoint"] = "missing.py"
        _rewrite_descriptor(built, descriptor)

    _adapt_root_pinned_trust(monkeypatch)
    with pytest.raises(ValueError):
        update_extension.load_update_extension_context(
            built.context, install_root=install, home=loader_home, target=loader_target,
            frozen=Path(__file__).resolve().parents[2])
    assert not (home / EFFECT_RELATIVE).exists()


@pytest.mark.parametrize("suffix", ["", "usr"])
def test_untrusted_alias_to_root_owned_ancestry_is_rejected(tmp_path, suffix):
    """Protected release inputs cannot gain trust through path aliases."""
    from hermes_cli import update_extension

    alias = tmp_path / "root-alias"
    alias.symlink_to("/", target_is_directory=True)
    candidate = alias / suffix if suffix else alias
    with pytest.raises(ValueError):
        update_extension.require_protected_ancestry(candidate, "protected fixture input")


def test_writable_bundle_directory_cannot_pass_tree_inventory(tmp_path, monkeypatch):
    from hermes_cli import update_extension

    root = tmp_path / "bundle"
    nested = root / "nested"
    nested.mkdir(parents=True, mode=0o700)
    (nested / "payload.py").write_text("VALUE = 'reviewed'\n")
    monkeypatch.setattr(update_extension, "ROOT_UID", os.geteuid())
    # This case exercises descendant metadata. Trust-anchor ancestry has its
    # separate real-path regression above; the fixture needs no Root identity.
    monkeypatch.setattr(update_extension, "require_protected_ancestry", lambda *_a, **_k: None)
    assert update_extension.walk_protected_tree(root, "bundle fixture") == (
        {"nested/payload.py"}, {"nested"}
    )
    nested.chmod(0o777)
    with pytest.raises(ValueError):
        update_extension.walk_protected_tree(root, "bundle fixture")


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="at uid 0 the calling user and the Root bundle owner are one identity")
def test_caller_owned_context_is_admitted_while_the_bundle_stays_root_pinned(
        tmp_path, monkeypatch):
    """The loader admits a context file owned by the actual calling user while
    the descriptor bundle keeps its Root-0 custody contract: the guard writes
    the context as the invoking user, never as Root."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "6" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(tmp_path, install_root=str(install),
                                    home=str(home), target=target)
    _pin_bundle_namespace_as_root_prepared(monkeypatch, built)

    captured = update_extension.load_update_extension_context(
        built.context, install_root=install, home=home, target=target, frozen=frozen)

    assert set(captured) == {"descriptor_path", "descriptor_sha256", "payload", "frozen"}
    assert captured["payload"]["sidecar"] == "witness-sidecar"
    assert captured["payload"]["payload"] == WITNESS_PAYLOAD
    assert not (home / EFFECT_RELATIVE).exists()  # admission stays read-only


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="at uid 0 a Root-owned context is also caller-owned")
def test_context_owned_by_root_is_refused_for_a_non_root_caller(tmp_path, monkeypatch):
    """Caller-owned means the invoking uid exactly: a context Root authored is
    refused for a non-root run."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "6" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(tmp_path, install_root=str(install),
                                    home=str(home), target=target)
    _pin_bundle_namespace_as_root_prepared(monkeypatch, built,
                                           context_uid=update_extension.ROOT_UID)

    with pytest.raises(ValueError):
        update_extension.load_update_extension_context(
            built.context, install_root=install, home=home, target=target, frozen=frozen)
    assert not (home / EFFECT_RELATIVE).exists()


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="at uid 0 the fixture's real ownership cannot differ from Root")
def test_faithful_fixture_still_refuses_a_foreign_owned_bundle_member(
        tmp_path, monkeypatch):
    """A bundle member owned by anyone but Root is refused by name, without
    weakening the caller-owned context's admission."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "6" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(tmp_path, install_root=str(install),
                                    home=str(home), target=target)
    _pin_bundle_namespace_as_root_prepared(
        monkeypatch, built, foreign=(built.bundle / "witness_sidecar.py",))

    with pytest.raises(ValueError, match=r"witness_sidecar\.py"):
        update_extension.load_update_extension_context(
            built.context, install_root=install, home=home, target=target, frozen=frozen)
    assert not (home / EFFECT_RELATIVE).exists()


@pytest.mark.parametrize("exit_argument,expected_code", [
    ("postcheck failed", None),
    (0, None),
    (3, 3),
], ids=["string_status", "zero_status", "nonzero_status"])
def test_extension_completion_exit_settles_nonzero_and_never_claims_success(
        native_environment, monkeypatch, exit_argument, expected_code):
    """A complete_update that exits instead of returning follows the ordinary
    strict-local settlement whatever exit shape it used: the real effect
    stays, the capsule still seals, the result is nonzero, the receipt claims
    no success and no update-extension fact is recorded; a nonzero status
    keeps its exact value."""
    from hermes_cli import update_cmd

    built = _write_extension_bundle(
        native_environment.base, install_root=native_environment.install_value,
        home=native_environment.home_value, target=native_environment.target,
        entry=_exiting_entry(exit_argument))
    _adapt_root_pinned_trust(monkeypatch)

    transported = {}

    def capturing_transport(request):
        transported["final"] = _in_process_completion(request)
        return transported["final"]

    monkeypatch.setattr(update_cmd, "run_completion", capturing_transport)

    raised = None
    try:
        update_cmd._cmd_local_update(
            SimpleNamespace(source_repo=str(native_environment.source),
                            target=native_environment.target,
                            required_extra=[],
                            update_extension_context=str(built.context)),
            gateway_mode=False)
    except SystemExit as exit_exc:
        raised = exit_exc

    assert "final" in transported, "the local update never reached its completion transport"
    final = transported["final"]
    assert final["exit_code"] != 0, (
        f"complete_update exiting with {exit_argument!r} settled as a successful "
        f"completion (exit code {final['exit_code']})")
    if expected_code is not None:
        assert final["exit_code"] == expected_code
    if raised is not None:
        assert raised.code == final["exit_code"]
    receipt = final["receipt"]
    assert receipt is not None, "a settled completion must publish its terminal receipt"
    if expected_code is not None:
        assert receipt["outcome"] == "failed"
    else:
        assert receipt["outcome"] != "success"
    assert not [key for key in receipt if key.startswith("update_extension_")], (
        "an exiting complete_update must not leave recorded extension facts")
    # The actual partial effect is preserved and the capsule still seals for
    # explicit recovery: the settlement keeps strict-local semantics.
    assert native_environment.effect.read_text(encoding="utf-8") == "applied before the exit\n"
    assert json.loads(native_environment.capsule_manifest().read_text())["state"] == "sealed"
    assert _git(native_environment.root, "rev-parse", "HEAD") == native_environment.target


def test_pinned_subpackage_initializer_runs_for_relative_sibling_imports(
        tmp_path, monkeypatch):
    """``from .helpers import validate_payload`` executes the pinned
    ``helpers/__init__.py`` — initializer-exported functions and that
    initializer's own nested relative import — rather than a synthesized
    empty package."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "7" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(
        tmp_path, install_root=str(install), home=str(home), target=target,
        members={"ext.py": _SUBPACKAGE_ENTRY,
                 "helpers/__init__.py": _HELPERS_INIT,
                 "helpers/payload_rules.py": _HELPERS_RULES,
                 "payload.json": json.dumps(WITNESS_PAYLOAD) + "\n"})
    _adapt_root_pinned_trust(monkeypatch)

    captured = update_extension.load_update_extension_context(
        built.context, install_root=install, home=home, target=target, frozen=frozen)
    admission = {"ok": True, "init": ["pinned-payload-rules"],
                 "rule": "pinned-payload-rules"}
    assert captured["payload"]["admission"] == admission
    assert not (home / EFFECT_RELATIVE).exists()  # admission stays read-only

    facts = update_extension.complete_update_extension(
        captured, install_root=install, home=home, target=target)
    assert facts["init"] == ["pinned-payload-rules"]
    landed = json.loads((home / EFFECT_RELATIVE).read_text(encoding="utf-8"))
    assert landed["admission"] == admission


def test_pinned_root_initializer_runs_for_relative_imports(tmp_path, monkeypatch):
    """A pinned root ``__init__.py`` is the unique package's initializer: the
    names it exports are reachable from the entrypoint through
    ``from . import ...``."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "8" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(
        tmp_path, install_root=str(install), home=str(home), target=target,
        members={"ext.py": _ROOT_INIT_ENTRY, "__init__.py": _ROOT_INIT,
                 "payload.json": json.dumps(WITNESS_PAYLOAD) + "\n"})
    _adapt_root_pinned_trust(monkeypatch)

    captured = update_extension.load_update_extension_context(
        built.context, install_root=install, home=home, target=target, frozen=frozen)
    assert captured["payload"]["root_init"] == "root-init-ran"
    assert not (home / EFFECT_RELATIVE).exists()  # admission stays read-only

    facts = update_extension.complete_update_extension(
        captured, install_root=install, home=home, target=target)
    assert facts["root_init"] == "root-init-ran"
    landed = json.loads((home / EFFECT_RELATIVE).read_text(encoding="utf-8"))
    assert landed["root_init"] == "root-init-ran"


@pytest.mark.parametrize("entrypoint", ["__init__.py", "hooks/__init__.py"],
                         ids=["root_initializer", "nested_initializer"])
def test_initializer_entrypoints_run_both_callbacks(tmp_path, monkeypatch, entrypoint):
    """A descriptor whose entrypoint is a pinned ``__init__.py`` — the bundle
    root's or a nested package's — runs both callbacks defined in that
    initializer and resolves its relative sidecar import through the same
    package."""
    from hermes_cli import update_extension

    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "9" * 40
    frozen = Path(__file__).resolve().parents[2]
    prefix = "" if entrypoint == "__init__.py" else "hooks/"
    built = _write_extension_bundle(
        tmp_path, install_root=str(install), home=str(home), target=target,
        members={entrypoint: _INITIALIZER_ENTRY,
                 f"{prefix}initializer_sidecar.py": _INITIALIZER_SIDECAR,
                 "payload.json": json.dumps(WITNESS_PAYLOAD) + "\n"},
        entrypoint=entrypoint)
    _adapt_root_pinned_trust(monkeypatch)

    captured = update_extension.load_update_extension_context(
        built.context, install_root=install, home=home, target=target, frozen=frozen)
    assert captured["payload"]["sidecar"] == "initializer-sidecar"
    assert captured["payload"]["payload"] == WITNESS_PAYLOAD
    assert not (home / EFFECT_RELATIVE).exists()  # admission stays read-only

    facts = update_extension.complete_update_extension(
        captured, install_root=install, home=home, target=target)
    assert facts["sidecar"] == "initializer-sidecar"
    landed = json.loads((home / EFFECT_RELATIVE).read_text(encoding="utf-8"))
    assert landed["sidecar"] == "initializer-sidecar"
    assert landed["payload"] == captured["payload"]


@pytest.mark.parametrize("extension_requested", [True, False],
                         ids=["extension_requested", "no_extension"])
def test_selected_tail_zero_exit_never_seals_success_for_a_requested_extension(
        native_environment, monkeypatch, extension_requested):
    """A selected-tail SystemExit(0) keeps the ordinary zero-exit success only
    when no extension is requested: with an update extension requested, the
    completion settles nonzero and claims no success, because the extension
    callback never ran or returned. The native tail's real effect stays in the
    sealed capsule either way."""
    from hermes_cli import source_completion, update_cmd

    built = None
    if extension_requested:
        built = _write_extension_bundle(
            native_environment.base, install_root=native_environment.install_value,
            home=native_environment.home_value, target=native_environment.target)
        _adapt_root_pinned_trust(monkeypatch)

    native_effect = native_environment.home / "native-tail" / "effect.txt"

    def exiting_tail(*_args, **_kwargs):
        native_effect.parent.mkdir(parents=True, exist_ok=True)
        native_effect.write_text("native tail applied before the zero exit\n",
                                 encoding="utf-8")
        raise SystemExit(0)

    monkeypatch.setattr(source_completion, "complete_source_checkout", exiting_tail)

    transported = {}

    def capturing_transport(request):
        transported["final"] = _in_process_completion(request)
        return transported["final"]

    monkeypatch.setattr(update_cmd, "run_completion", capturing_transport)

    raised = None
    try:
        update_cmd._cmd_local_update(
            SimpleNamespace(source_repo=str(native_environment.source),
                            target=native_environment.target,
                            required_extra=[],
                            update_extension_context=str(built.context) if built else None),
            gateway_mode=False)
    except SystemExit as exit_exc:
        raised = exit_exc

    assert "final" in transported, "the local update never reached its completion transport"
    final = transported["final"]
    receipt = final["receipt"]
    assert receipt is not None, "a settled completion must publish its terminal receipt"
    if extension_requested:
        assert final["exit_code"] != 0, (
            "a selected-tail zero exit settled a requested extension's completion as "
            f"success (exit code {final['exit_code']}) without the callback ever running")
        if raised is not None:
            assert raised.code == final["exit_code"]
        assert receipt["outcome"] != "success"
        assert not [key for key in receipt if key.startswith("update_extension_")], (
            "a completion whose extension callback never ran records no extension facts")
        assert not native_environment.effect.exists(), (
            "the extension's complete_update must not have run")
    else:
        assert final["exit_code"] == 0
        assert receipt["outcome"] == "success"
    # The native tail's real effect is retained in the sealed capsule postimage.
    assert native_effect.read_text(encoding="utf-8") == "native tail applied before the zero exit\n"
    assert json.loads(native_environment.capsule_manifest().read_text())["state"] == "sealed"
    assert _git(native_environment.root, "rev-parse", "HEAD") == native_environment.target


@pytest.mark.parametrize("members", [
    {"hook.py": _COLLIDING_ENTRY, "hook/__init__.py": _COLLIDING_INIT,
     "payload.json": json.dumps(WITNESS_PAYLOAD) + "\n"},
    {"hook/__init__.py": _COLLIDING_INIT, "hook.py": _COLLIDING_ENTRY,
     "payload.json": json.dumps(WITNESS_PAYLOAD) + "\n"},
], ids=["entry_first", "init_first"])
def test_duplicate_import_identity_is_refused_before_any_bundle_code_runs(
        tmp_path, monkeypatch, members):
    """A bundle pinning both ``hook.py`` and ``hook/__init__.py`` maps both to
    import identity ``hook``: the loader refuses the collision by name before
    executing any bundle bytes, whichever order the descriptor records (the
    two orders serialize to different descriptor bytes). All file pins stay
    valid, so the collision is the only refusal reason."""
    from hermes_cli import update_extension

    monkeypatch.setenv("COLLISION_MARKS", str(tmp_path / "marks"))
    install, home = tmp_path / "installed", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    target = "a" * 40
    frozen = Path(__file__).resolve().parents[2]
    built = _write_extension_bundle(
        tmp_path, install_root=str(install), home=str(home), target=target,
        members=members, entrypoint="hook.py")
    _adapt_root_pinned_trust(monkeypatch)

    with pytest.raises(ValueError, match=r"hook"):
        update_extension.load_update_extension_context(
            built.context, install_root=install, home=home, target=target, frozen=frozen)

    # No top-level and no callback effect: rejection precedes any import.
    marks = tmp_path / "marks"
    assert not marks.exists() or not any(marks.iterdir()), (
        "top-level or callback bundle bytes executed before the collision refusal")
    assert not (home / EFFECT_RELATIVE).exists()
    assert not [name for name in sys.modules
                if name.startswith("hermes_update_extension_")]
