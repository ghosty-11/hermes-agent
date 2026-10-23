"""Exact local source selection refuses drift and never needs a remote."""
from pathlib import Path
import os
import subprocess

import pytest

from hermes_cli.update_local import prepare_local_source, apply_local_source, refuse_active_backups


def git(root, *args, check=True):
    proc = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True,
                          env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"})
    if check:
        assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip() if check else proc


@pytest.fixture
def repositories(tmp_path):
    installed = tmp_path / "installed"
    installed.mkdir()
    git(installed, "init", "-b", "main")
    git(installed, "config", "user.name", "Fixture")
    git(installed, "config", "user.email", "fixture@example.invalid")
    (installed / "version.txt").write_text("before")
    git(installed, "add", ".")
    git(installed, "commit", "-m", "base")
    source = tmp_path / "reviewed"
    git(installed, "worktree", "add", "-b", "reviewed", str(source))
    (source / "version.txt").write_text("after")
    git(source, "add", ".")
    git(source, "commit", "-m", "reviewed target")
    target = git(source, "rev-parse", "HEAD")
    assert (source / ".git").is_file()
    return installed, source, target


def test_linked_worktree_exact_commit_without_tracking_ref_change(repositories):
    installed, source, target = repositories
    git(installed, "remote", "add", "origin", "https://invalid.example/forbidden.git")
    before = git(installed, "rev-parse", "HEAD")
    prepare_local_source(installed, source, target)
    assert git(installed, "rev-parse", "HEAD") == before
    apply_local_source(installed, source, target)
    assert git(installed, "rev-parse", "HEAD") == target
    assert (installed / "version.txt").read_text() == "after"
    assert git(installed, "for-each-ref", "--format=%(refname)", "refs/remotes") == ""


@pytest.mark.parametrize("damage", ["dirty_source", "dirty_install", "unknown_target"])
def test_refuses_invalid_inputs_before_checkout(repositories, damage):
    installed, source, target = repositories
    before = git(installed, "rev-parse", "HEAD")
    if damage == "dirty_source":
        (source / "untracked").write_text("not reviewed")
    elif damage == "dirty_install":
        (installed / "version.txt").write_text("local edits")
    else:
        target = "a" * 40
    with pytest.raises((RuntimeError, ValueError)):
        prepare_local_source(installed, source, target)
    assert git(installed, "rev-parse", "HEAD") == before


@pytest.mark.platforms("posix")
def test_backup_writer_refused_before_checkout(tmp_path):
    import fcntl

    home = tmp_path / "home"
    home.mkdir()
    lock = home / ".backup.lock"
    with lock.open("w+b") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="backup writer"):
            refuse_active_backups(home)


@pytest.mark.parametrize("ledger", [None, "store", "runtime"])
def test_local_admission_keeps_a_ledgerless_legacy_venv_selection(tmp_path, monkeypatch, ledger):
    """An installation without runtime facts is migrating a main-era venv;
    local admission keeps the [all] plus installed opt-in selection ordinary
    migration keeps, unioned with any shared store record. Only this
    installation's own runtime ledger replaces the legacy selection."""
    import json
    from hermes_cli import local_update_preflight, main_install_repair
    from pm.environments import runtime_facts_path
    from pm.paths import store_root

    installed, source = tmp_path / "installed", tmp_path / "source"
    source.mkdir()
    site = installed / "venv" / "lib" / "python3.11" / "site-packages"
    (site / "fal_client").mkdir(parents=True)
    monkeypatch.setattr(main_install_repair, "configured_update_extras",
                        lambda *_a, **_k: ["mcp"])
    record = json.dumps({"schema": 1, "packages": {"venv": {"extras": ["telegram"]}}})
    facts = {"runtime": runtime_facts_path(installed),
             "store": store_root(installed) / "facts.json"}.get(ledger)
    if facts is not None:
        facts.parent.mkdir(parents=True, exist_ok=True)
        facts.write_text(record)

    extras, _ = local_update_preflight.admitted_extras(source, installed, ["pdf"])

    if ledger is None:
        assert {"all", "fal", "mcp", "pdf"} <= set(extras)
    elif ledger == "store":
        assert {"all", "fal", "telegram", "mcp", "pdf"} <= set(extras)
    else:
        assert set(extras) == {"telegram", "mcp", "pdf"}


def test_local_git_never_starts_repository_maintenance(repositories, tmp_path):
    """A fetch must not start gc/maintenance: detached, it outlives the owned
    writer, holds inherited descriptors and keeps writing the repository."""
    from hermes_cli.update_local import local_git

    installed, source, target = repositories
    receiver = tmp_path / "receiver"
    git(tmp_path, "init", "-q", str(receiver))
    (receiver / "seed.txt").write_text("seed")
    git(receiver, "add", ".")
    git(receiver, "-c", "user.name=F", "-c", "user.email=f@example.invalid", "commit", "-qm", "seed")
    git(receiver, "repack", "-adq")
    # Foreground auto-gc (no detach) makes a started maintenance observable:
    # it would consolidate the fetched pack with the existing one.
    for key, value in (("gc.autoPackLimit", "1"), ("gc.autoDetach", "false"),
                       ("maintenance.autoDetach", "false"), ("fetch.unpackLimit", "1"),
                       ("transfer.unpackLimit", "1")):
        git(receiver, "config", key, value)
    local_git(receiver, "fetch", "--no-tags", "-q", source.as_uri(), target)
    # Foreground maintenance leaves a commit-graph and a cruft pack's .mtimes.
    objects = receiver / ".git" / "objects"
    assert not (objects / "info" / "commit-graph").exists()
    assert not list((objects / "pack").glob("*.mtimes"))


def test_local_preflight_reaches_its_first_decision(tmp_path):
    """Every module the offline preflight imports must exist: the first full
    rehearsal crashed here, past every narrower test."""
    from hermes_cli.local_update_preflight import stage_local_inputs

    with pytest.raises(ValueError, match="distinct from the installed tree"):
        stage_local_inputs(tmp_path, tmp_path, tmp_path / "staging", [])


@pytest.mark.parametrize("untracked", ["product", "source"])
def test_workspace_link_closure_leaves_product_subtrees_to_their_image(tmp_path, untracked):
    """npm links a workspace from the root node_modules; the linked workspace's
    own node_modules and build output are products imaged separately, while
    its untracked source files still have no rollback owner."""
    from hermes_cli.local_update_runtime import PRODUCTS, dependency_closure

    install, home, state = tmp_path / "install", tmp_path / "home", tmp_path / "state"
    for path in (install, home, state, tmp_path / "store"):
        path.mkdir()
    git(install, "init", "-q")
    (install / "ui-tui").mkdir()
    (install / "ui-tui" / "package.json").write_text("{}")
    git(install, "add", ".")
    git(install, "-c", "user.name=F", "-c", "user.email=f@example.invalid", "commit", "-qm", "ws")
    (install / "node_modules" / "@hermes").mkdir(parents=True)
    (install / "node_modules" / "@hermes" / "ui-tui").symlink_to("../../ui-tui")
    (install / "ui-tui" / "node_modules" / "undici").mkdir(parents=True)
    (install / "ui-tui" / "node_modules" / "undici" / "README.md").write_text("dep")
    if untracked == "source":
        (install / "ui-tui" / "src").mkdir()
        (install / "ui-tui" / "src" / "scratch.ts").write_text("untracked")
    products = tuple(install / name for name in PRODUCTS)
    if untracked == "product":
        dependency_closure([], products, install, [home], tmp_path / "store",
                           tmp_path / "store", state)
    else:
        with pytest.raises(ValueError, match="untracked workspace dependency.*scratch.ts"):
            dependency_closure([], products, install, [home], tmp_path / "store",
                               tmp_path / "store", state)


def test_dependency_closure_skips_bytecode_the_image_excludes(tmp_path):
    """A generation reached through a link may hold __pycache__; resource images
    exclude bytecode, so the closure must not demand an owner for it (404)."""
    from hermes_cli.local_update_runtime import PRODUCTS, dependency_closure

    install, home, state = tmp_path / "install", tmp_path / "home", tmp_path / "state"
    for path in (install, home, state, tmp_path / "store"):
        path.mkdir()
    git(install, "init", "-q")
    site = state / "pm-runtime" / "generations" / "g1" / "lib" / "site-packages"
    (site / "__pycache__").mkdir(parents=True)
    (site / "mod.py").write_text("VALUE = 1\n")
    (site / "__pycache__" / "mod.cpython-314.pyc").write_bytes(b"\x00" * 16)
    (install / "node_modules").mkdir()
    (install / "node_modules" / "runtime").symlink_to(site)
    retained, _ = dependency_closure([], tuple(install / name for name in PRODUCTS), install,
                                     [home], tmp_path / "store", tmp_path / "store", state)
    assert state / "pm-runtime" / "generations" / "g1" in retained
