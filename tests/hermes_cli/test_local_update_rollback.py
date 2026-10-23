"""Behavior contracts for explicit, offline local-update rollback."""

from contextlib import closing, contextmanager
import json
import os
import shutil
import signal
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli._launchers import mint_launcher
from hermes_cli.local_update_rollback import (
    capture_local_update_capsule,
    restore_local_update_capsule,
    seal_local_update_capsule,
    verify_local_update_preimage,
)
from hermes_cli.runtime_state import collect_generations, lease_directory
from hermes_cli.source_stamp import write_source_stamp
from hermes_cli.update_lock import UpdateLock
from pm.environments import install_state_dir, project_python, runtime_facts_path, selected_venv
from pm.cli import _gc_store
from pm.lock import Facts
from pm.runtime import collect_runtime_generations
from pm.store import Store


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def transaction(tmp_path, monkeypatch):
    install = tmp_path / "install"
    install.mkdir()
    git(install, "init")
    git(install, "config", "user.name", "Capsule Test")
    git(install, "config", "user.email", "capsule@example.invalid")
    (install / "app.py").write_bytes(b"old code\n")
    git(install, "add", ".")
    git(install, "commit", "-m", "old")
    before = git(install, "rev-parse", "HEAD")
    source = tmp_path / "source"
    git(tmp_path, "clone", "--no-local", str(install), str(source))
    git(source, "config", "user.name", "Capsule Test")
    git(source, "config", "user.email", "capsule@example.invalid")
    (source / "app.py").write_bytes(b"target code\n")
    git(source, "add", ".")
    git(source, "commit", "-m", "target")
    target = git(source, "rev-parse", "HEAD")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_bytes(b"old config\n")
    sibling = home / "profiles" / "work"
    sibling.mkdir(parents=True)
    (sibling / "config.yaml").write_bytes(b"old sibling\n")
    (sibling / "config.yaml").chmod(0o640)
    (sibling / "skills").mkdir()
    (sibling / "skills" / "sample.md").write_bytes(b"old skill\n")
    db = sibling / "state.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE events (value TEXT)")
        conn.execute("INSERT INTO events VALUES ('before')")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.update_inventory.collect_runtime_inventory", lambda: SimpleNamespace(runtimes=[]))
    return install, source, home, sibling, before, target


def advance(install, source, target):
    git(install, "fetch", "--no-tags", str(source), target)
    git(install, "checkout", "--detach", target)

@contextmanager
def substituted_fifo(path):
    original = path.read_bytes()
    path.unlink()
    os.mkfifo(path, 0o600)

    def timeout(_signum, _frame):
        pytest.fail(f"rollback blocked on FIFO input: {path}")

    previous = signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(5)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        path.unlink()
        path.write_bytes(original)


def test_roundtrip_multiple_homes_and_wal(transaction):
    install, source, home, sibling, before, target = transaction
    large = sibling / "skills" / "large.bin"
    with large.open("wb") as stream:
        stream.seek((1 << 30) + 17)
        stream.write(b"last byte")
    large.chmod(0o640)
    (sibling / "link").symlink_to("skills/sample.md")
    home_mtime = home.stat().st_mtime_ns
    sibling_mtime = sibling.stat().st_mtime_ns
    manifest = capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"new config\n")
    (sibling / "config.yaml").write_bytes(b"new sibling\n")
    (sibling / "skills" / "sample.md").write_bytes(b"new skill\n")
    (sibling / "config.yaml").chmod(0o600)
    (sibling / "cron").mkdir()
    (sibling / "cron" / "new-job.json").write_bytes(b"created by migration\n")
    with closing(sqlite3.connect(sibling / "state.db")) as conn, conn:
        conn.execute("INSERT INTO events VALUES ('after')")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    with pytest.raises(ValueError, match="acknowledge"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (install / "app.py").read_bytes() == b"old code\n"
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert (sibling / "config.yaml").read_bytes() == b"old sibling\n"
    assert (sibling / "config.yaml").stat().st_mode & 0o777 == 0o640
    assert not (sibling / "cron").exists()
    assert (sibling / "skills" / "sample.md").read_bytes() == b"old skill\n"
    assert large.stat().st_size == (1 << 30) + 26
    assert large.stat().st_mode & 0o777 == 0o640
    assert os.readlink(sibling / "link") == "skills/sample.md"
    assert home.stat().st_mtime_ns == home_mtime
    assert sibling.stat().st_mtime_ns == sibling_mtime
    with closing(sqlite3.connect(sibling / "state.db")) as conn:
        assert conn.execute("SELECT value FROM events").fetchall() == [("before",)]
    assert git(install, "cat-file", "-t", target) == "commit"


def test_linked_worktree_source_preserves_install_history(transaction):
    install, source, home, sibling, before, target = transaction
    linked = source.parent / "linked"
    git(source, "worktree", "add", "--detach", str(linked), target)
    manifest = capture_local_update_capsule(install, home, linked, target)
    advance(install, linked, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (install / "app.py").read_bytes() == b"old code\n"
    assert git(install, "cat-file", "-p", target).startswith("tree ")


def test_postimage_drift_refuses_before_any_mutation(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"postimage\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    (sibling / "config.yaml").write_bytes(b"foreign write\n")
    with pytest.raises((ValueError, RuntimeError), match="drift|changed|postimage"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert (home / "config.yaml").read_bytes() == b"postimage\n"
    assert (sibling / "config.yaml").read_bytes() == b"foreign write\n"


def test_volatile_home_state_survives_rollback_without_postimage_drift(transaction):
    install, source, home, sibling, before, target = transaction
    for owner in (home, sibling):
        (owner / "logs").mkdir()
        (owner / "logs" / "agent.log").write_bytes(b"before\n")
        (owner / "auth.lock").touch()
        (owner / "cache" / "images").mkdir(parents=True)
        (owner / "cache" / "tool_discovery_cache.json").write_bytes(b"before cache\n")
        (owner / "cache" / "images" / "saved.png").write_bytes(b"original image\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    (sibling / "skills" / "sample.md").write_bytes(b"target skill\n")
    for owner in (home, sibling):
        (owner / "cache" / "images" / "saved.png").write_bytes(b"target image\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    for owner in (home, sibling):
        with (owner / "logs" / "agent.log").open("ab") as stream:
            stream.write(b"after seal\n")
        (owner / "auth.lock").touch()
        os.utime(owner / "auth.lock", ns=(1_710_000_000_000_000_000,) * 2)
        (owner / "cache" / "tool_discovery_cache.json").write_bytes(b"after seal cache\n")
        (owner / "logs").chmod(0o750)
        (owner / "cache").chmod(0o750)
        (owner / "logs" / "later.log").write_bytes(b"new operational log\n")
        (owner / "cache" / "new-entry.json").write_bytes(b"new operational cache\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert (sibling / "skills" / "sample.md").read_bytes() == b"old skill\n"
    for owner in (home, sibling):
        assert (owner / "logs" / "agent.log").read_bytes() == b"before\nafter seal\n"
        assert (owner / "auth.lock").is_file()
        assert (owner / "auth.lock").stat().st_mtime_ns == 1_710_000_000_000_000_000
        assert (owner / "cache" / "tool_discovery_cache.json").read_bytes() == b"after seal cache\n"
        assert (owner / "cache" / "images" / "saved.png").read_bytes() == b"original image\n"
        assert (owner / "logs" / "later.log").read_bytes() == b"new operational log\n"
        assert (owner / "cache" / "new-entry.json").read_bytes() == b"new operational cache\n"
        assert (owner / "logs").stat().st_mode & 0o777 == 0o750
        assert (owner / "cache").stat().st_mode & 0o777 == 0o750


@pytest.mark.parametrize("reason", ["good", "pre-migrate"])
def test_automatic_known_good_config_copies_do_not_block_rollback(transaction, reason):
    """The config loader writes backups/config/config.yaml.good.<ts> on each changed successful
    parse (rehearsal 425: every seat's first post-update load). Those copies derive from
    config.yaml, which rollback restores; any other config backup stays protected."""
    install, source, home, sibling, before, target = transaction
    copies = sibling / "backups" / "config"
    copies.mkdir(parents=True)
    (copies / "config.yaml.good.20260101-000000").write_bytes(b"old sibling\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    (copies / f"config.yaml.{reason}.20261021-170707").write_bytes(b"target sibling\n")
    if reason == "good":
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
        assert git(install, "rev-parse", "HEAD") == before
        assert (sibling / "config.yaml").read_bytes() == b"old sibling\n"
        assert (copies / "config.yaml.good.20260101-000000").read_bytes() == b"old sibling\n"
        assert not (copies / "config.yaml.good.20261021-170707").exists()
    else:
        with pytest.raises(ValueError, match="postimage drift"):
            restore_local_update_capsule(
                manifest, expected_install_root=install, expected_hermes_home=home,
                acknowledge_data_rollback=True)


def test_relaunch_bootstrap_records_do_not_block_rollback(transaction):
    """Every launch rewrites installs/<id>/bootstrap/<profile>.json (rehearsal 430). It names
    the running code; restored old code re-bootstraps on the identity mismatch."""
    install, source, home, sibling, before, target = transaction
    records = install_state_dir(install) / "bootstrap"
    records.mkdir(parents=True)
    (records / "default.json").write_text('{"identity": "old"}')
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (records / "default.json").write_text('{"identity": "target", "at": 1}')
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    (records / "default.json").write_text('{"identity": "target", "at": 2}')
    (records / "work.json").write_text('{"identity": "target", "at": 2}')
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (records / "default.json").read_text() == '{"identity": "old"}'
    assert not (records / "work.json").exists()


@pytest.mark.parametrize("after_seal", ["discharged", "rewritten"])
def test_discharged_host_obligation_does_not_block_rollback(transaction, tmp_path, monkeypatch, after_seal):
    """The update arms the host restart obligation; a gateway on the target (or a host with
    none) discharges it. Discharged is the preimage, so restore overwrites nothing."""
    from hermes_cli.update_host_obligation import (
        clear_host_obligation, host_obligation_path, write_host_obligation)
    locks = tmp_path / "gateway-locks"
    locks.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
    manifest = capture_local_update_capsule(install := transaction[0], transaction[2],
                                            transaction[1], transaction[5])
    advance(install, transaction[1], transaction[5])
    assert write_host_obligation(expected_sha=transaction[5])
    seal_local_update_capsule(manifest, expected_install_root=install,
                              expected_hermes_home=transaction[2])
    if after_seal == "discharged":
        clear_host_obligation()
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=transaction[2],
            acknowledge_data_rollback=True)
        assert git(install, "rev-parse", "HEAD") == transaction[4]
        assert not host_obligation_path().exists()
    else:
        assert write_host_obligation(expected_sha="0" * 40)
        with pytest.raises(ValueError, match="postimage drift"):
            restore_local_update_capsule(
                manifest, expected_install_root=install, expected_hermes_home=transaction[2],
                acknowledge_data_rollback=True)


def test_good_copy_created_with_backup_directory_and_after_seal_is_removed(transaction):
    install, source, home, sibling, before, target = transaction
    copies = home / "backups" / "config"
    assert not copies.exists()
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"updated\n")
    copies.mkdir(parents=True)
    (copies / "config.yaml.good.20261021-170707").write_bytes(b"updated\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    (copies / "config.yaml.good.20261021-170708").write_bytes(b"later\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert not copies.exists()


def test_manually_named_good_backup_is_protected_from_later_history(transaction):
    install, source, home, sibling, before, target = transaction
    copy = sibling / "backups" / "config" / "config.yaml.good.operator-copy"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"operator original\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    copy.write_bytes(b"operator later\n")
    with pytest.raises(ValueError, match="postimage drift"):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert copy.read_bytes() == b"operator later\n"


def test_planted_generated_good_backup_is_removed_and_pre_copy_restored(transaction):
    install, source, home, sibling, before, target = transaction
    copies = sibling / "backups" / "config"
    copies.mkdir(parents=True)
    original = copies / "config.yaml.good.20260101-000000"
    original.write_bytes(b"pre-good\n")
    original.chmod(0o640)
    original_mtime = original.stat().st_mtime_ns
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    original.write_bytes(b"updated-good\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    original.write_bytes(b"later-good\n")
    planted = copies / "config.yaml.good.29991231-235959"
    planted.write_bytes(b"planted\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert not planted.exists()
    assert original.read_bytes() == b"pre-good\n"
    assert original.stat().st_mode & 0o777 == 0o640
    assert original.stat().st_mtime_ns == original_mtime
    assert git(install, "rev-parse", "HEAD") == before

def test_generated_backup_in_directory_created_after_seal_is_removed(transaction):
    install, source, home, sibling, before, target = transaction
    copies = home / "backups" / "config"
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    copies.mkdir(parents=True)
    (copies / "config.yaml.good.20261021-170709").write_bytes(b"later\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert not (home / "backups").exists()
    assert git(install, "rev-parse", "HEAD") == before


def test_generated_backup_name_as_link_is_protected(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    copies = sibling / "backups" / "config"
    copies.mkdir(parents=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    victim = tmp_path / "outside"
    victim.write_bytes(b"outside\n")
    link = copies / "config.yaml.good.20261021-170709"
    link.symlink_to(victim)
    with pytest.raises(ValueError, match="postimage drift"):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert link.is_symlink() and victim.read_bytes() == b"outside\n"

def test_preexisting_link_with_generated_name_is_restored_after_regular_update(
        transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    copies = sibling / "backups" / "config"
    copies.mkdir(parents=True)
    victim = tmp_path / "outside"
    victim.write_bytes(b"outside\n")
    copy = copies / "config.yaml.good.20260101-000000"
    copy.symlink_to(victim)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    copy.unlink()
    copy.write_bytes(b"generated\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    copy.write_bytes(b"later generated\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert copy.is_symlink() and copy.resolve() == victim
    assert victim.read_bytes() == b"outside\n"
    assert git(install, "rev-parse", "HEAD") == before


def test_bootstrap_tmp_link_refuses_but_rewritten_record_restores(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    records = install_state_dir(install) / "bootstrap"
    records.mkdir(parents=True)
    record = records / "default.json"
    record.write_bytes(b'{"identity": "old"}')
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    record.write_bytes(b'{"identity": "target"}')
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    victim = tmp_path / "victim"
    victim.write_bytes(b"untouched\n")
    tmp = records / "default.json.tmp"
    tmp.symlink_to(victim)
    with pytest.raises(ValueError, match="postimage drift"):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert victim.read_bytes() == b"untouched\n"
    tmp.unlink()
    record.write_bytes(b'{"identity": "later"}')
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert record.read_bytes() == b'{"identity": "old"}'
    assert git(install, "rev-parse", "HEAD") == before


def test_external_selection_returned_to_file_preimage_skips_type_restore(
        transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    locks = tmp_path / "gateway-locks"
    locks.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
    from hermes_cli.update_host_obligation import host_obligation_path
    choice = host_obligation_path()
    choice.write_bytes(b"pre\n")
    pre_stat = choice.stat()
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    choice.unlink()
    choice.mkdir()
    (choice / "during-update").write_bytes(b"post\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    (choice / "during-update").unlink()
    choice.rmdir()
    choice.write_bytes(b"pre\n")
    choice.chmod(pre_stat.st_mode & 0o777)
    os.utime(choice, ns=(pre_stat.st_mtime_ns, pre_stat.st_mtime_ns))
    restored = restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert restored["state"] == "restored"
    assert choice.read_bytes() == b"pre\n"
    assert choice.stat().st_mtime_ns == pre_stat.st_mtime_ns
    assert git(install, "rev-parse", "HEAD") == before


@pytest.mark.parametrize("owner", ["root", "profile"])
@pytest.mark.parametrize("after_seal", ["discharged", "rewritten"])
def test_in_home_host_obligation_is_independent_selection(
        transaction, monkeypatch, owner, after_seal):
    from hermes_cli.update_host_obligation import (
        clear_host_obligation, host_obligation_path, write_host_obligation)
    install, source, home, sibling, before, target = transaction
    locks = (home if owner == "root" else sibling) / "gateway-locks"
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(locks))
    manifest = capture_local_update_capsule(install, home, source, target)
    choice = host_obligation_path()
    rows = json.loads(manifest.read_text())["roots"]
    assert any(row["kind"] == "selection" and row["include"] == [str(choice)] for row in rows)
    root_record = next(row for row in rows if row["kind"] == "home" and row["path"] == str(home))
    assert any(row["path"] == locks.relative_to(home).as_posix()
               for row in root_record["pre"])
    advance(install, source, target)
    assert write_host_obligation(expected_sha=target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    if after_seal == "discharged":
        clear_host_obligation()
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
        assert not choice.exists()
        assert git(install, "rev-parse", "HEAD") == before
    else:
        assert write_host_obligation(expected_sha="0" * 40)
        with pytest.raises(ValueError, match="postimage drift"):
            restore_local_update_capsule(
                manifest, expected_install_root=install, expected_hermes_home=home,
                acknowledge_data_rollback=True)
        assert git(install, "rev-parse", "HEAD") == target


@pytest.mark.parametrize("changed", [
    "config.yaml", "cache/images/saved.png", "directory.lock/owner.dat",
])
def test_protected_home_data_still_refuses_later_history(transaction, changed):
    install, source, home, sibling, before, target = transaction
    protected = sibling / changed
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_bytes(b"before\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    protected.write_bytes(b"sealed\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    protected.write_bytes(b"later\n")
    with pytest.raises(ValueError, match="sealed postimage drift"):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert protected.read_bytes() == b"later\n"


@pytest.mark.parametrize("location", ["override", "xdg"])
def test_capture_creates_host_obligation_owner_before_anchoring(
        transaction, tmp_path, monkeypatch, location):
    install, source, home, sibling, before, target = transaction
    account = tmp_path / "account"
    account.mkdir()
    monkeypatch.setattr(Path, "home", lambda: account)
    if location == "override":
        owner = tmp_path / "fresh-state" / "gateway-locks"
        monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(owner))
    else:
        owner = tmp_path / "fresh-state" / "hermes" / "gateway-locks"
        monkeypatch.delenv("HERMES_GATEWAY_LOCK_DIR", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "fresh-state"))
    assert not owner.exists()
    manifest = capture_local_update_capsule(install, home, source, target)
    assert owner.is_dir()
    assert owner.stat().st_mode & 0o777 == 0o700
    records = json.loads(manifest.read_text())["roots"]
    assert any(row["kind"] == "selection" and row["path"] == str(owner)
               and row["include"] == [str(owner / "host-update-restart.json")] for row in records)


def test_capture_refuses_symlink_in_host_obligation_owner(transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(linked / "gateway-locks"))
    with pytest.raises((OSError, ValueError), match="symlink|real|outside"):
        capture_local_update_capsule(install, home, source, target)
    assert not (real / "gateway-locks").exists()


def test_capsule_corruption_and_later_history_refuse(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    payload = manifest.read_bytes()
    manifest.write_bytes(b"not json")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    manifest.write_bytes(payload)
    (install / "app.py").write_bytes(b"later commit\n")
    git(install, "add", ".")
    git(install, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "later")
    later = git(install, "rev-parse", "HEAD")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == later


def test_artifact_and_archive_path_tampering_refuse_without_data_loss(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    payload = manifest.read_text()
    capsule = json.loads(payload)
    capsule["roots"][0]["pre"][0]["path"] = "../outside"
    manifest.write_text(json.dumps(capsule))
    with pytest.raises(ValueError):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    manifest.write_text(payload)
    # A damaged bundle must be refused before even checking mutable postimages.
    bundle = manifest.parent / "code.bundle"
    with bundle.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target


def test_symlink_parent_and_missing_source_objects_refuse(transaction):
    install, source, home, sibling, before, target = transaction
    outside = home.parent / "outside"
    outside.mkdir()
    (sibling / "external").symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        capture_local_update_capsule(install, home, source, "f" * 40)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    (sibling / "skills").rename(outside / "skills")
    (sibling / "skills").symlink_to(outside / "skills", target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target


def test_metadata_roundtrip_and_attribute_drift(transaction):
    install, source, home, sibling, before, target = transaction
    file = sibling / "skills" / "sample.md"
    directory = sibling / "skills"
    owner = (file.stat().st_uid, file.stat().st_gid)
    directory_owner = (directory.stat().st_uid, directory.stat().st_gid)
    try:
        os.setxattr(file, "user.rollback", b"private preimage")
        os.setxattr(directory, "user.rollback", b"directory preimage")
    except (AttributeError, OSError) as exc:
        pytest.skip(f"user extended attributes unavailable: {exc}")
    manifest = capture_local_update_capsule(install, home, source, target)
    payload = manifest.read_bytes()
    assert b"private preimage" not in payload
    assert b"directory preimage" not in payload
    advance(install, source, target)
    os.setxattr(file, "user.rollback", b"postimage")
    os.removexattr(directory, "user.rollback")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    os.setxattr(file, "user.rollback", b"later history")
    with pytest.raises(ValueError, match="drift"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert os.getxattr(file, "user.rollback") == b"later history"
    os.setxattr(file, "user.rollback", b"postimage")
    alternate = ((owner[0] + 1, owner[1]) if os.geteuid() == 0 else
                 next(((owner[0], group) for group in os.getgroups() if group != owner[1]), None))
    if alternate is not None:
        os.chown(file, *alternate)
        with pytest.raises(ValueError, match="drift"):
            restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
        assert git(install, "rev-parse", "HEAD") == target
        os.chown(file, *owner)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert os.getxattr(file, "user.rollback") == b"private preimage"
    assert os.getxattr(directory, "user.rollback") == b"directory preimage"
    assert (file.stat().st_uid, file.stat().st_gid) == owner
    assert (directory.stat().st_uid, directory.stat().st_gid) == directory_owner


def test_pre_target_failure_seals_actual_head_and_restores_data(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    (sibling / "config.yaml").write_bytes(b"migration failed before checkout\n")
    sealed = seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    assert sealed["target"] == target
    assert sealed["post_head"] == before
    post_mtime = (sibling / "config.yaml").stat().st_mtime_ns
    (sibling / "config.yaml").write_bytes(b"later history\n")
    with pytest.raises(ValueError, match="drift"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert (sibling / "config.yaml").read_bytes() == b"later history\n"
    (sibling / "config.yaml").write_bytes(b"migration failed before checkout\n")
    os.utime(sibling / "config.yaml", ns=(post_mtime, post_mtime))
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (sibling / "config.yaml").read_bytes() == b"old sibling\n"


def test_abbreviated_target_refused_before_admission(transaction):
    install, source, home, sibling, before, target = transaction
    with pytest.raises(ValueError, match="exact"):
        capture_local_update_capsule(install, home, source, target[:12])
    assert git(install, "rev-parse", "HEAD") == before


def test_named_access_and_default_acl_roundtrip(transaction):
    if not shutil.which("setfacl") or not hasattr(os, "getxattr"):
        pytest.skip("named POSIX access/default ACL fixtures require setfacl and xattr support")
    install, source, home, sibling, before, target = transaction
    file = sibling / "skills" / "sample.md"
    directory = sibling / "skills"
    named_user = os.getuid() + 1
    for path, rule in ((file, f"u:{named_user}:r--"), (directory, f"d:u:{named_user}:r--")):
        result = subprocess.run(["setfacl", "-m", rule, str(path)], capture_output=True, text=True)
        if result.returncode:
            pytest.skip(f"named POSIX ACL fixture unavailable: {result.stderr.strip()}")
    access = os.getxattr(file, "system.posix_acl_access")
    default = os.getxattr(directory, "system.posix_acl_default")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    subprocess.run(["setfacl", "-b", str(file)], check=True)
    subprocess.run(["setfacl", "-k", str(directory)], check=True)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    post_mode = file.stat().st_mode & 0o7777
    subprocess.run(["setfacl", "-m", f"u:{named_user}:rw-", str(file)], check=True)
    with pytest.raises(ValueError, match="drift"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    subprocess.run(["setfacl", "-b", str(file)], check=True)
    file.chmod(post_mode)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert os.getxattr(file, "system.posix_acl_access") == access
    assert os.getxattr(directory, "system.posix_acl_default") == default
    assert git(install, "rev-parse", "HEAD") == before


def test_seal_refuses_unrelated_head(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    (install / "app.py").write_bytes(b"unrelated code\n")
    git(install, "add", ".")
    git(install, "commit", "-m", "unrelated")
    later = git(install, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="neither"):
        seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    assert git(install, "rev-parse", "HEAD") == later
    assert json.loads(manifest.read_text())["state"] == "captured"


def test_updater_marker_and_generation_leases_are_not_sealed_data(transaction):
    install, source, home, sibling, before, target = transaction
    generation = install_state_dir(install) / "environments" / "old"
    generation.mkdir(parents=True)
    (generation / ".lease-managed").touch()
    marker = home / ".hermes-update-in-progress"
    with UpdateLock(path=marker) as update:
        assert update.acquired
        release = lease_directory(generation)
        try:
            manifest = capture_local_update_capsule(install, home, source, target)
            advance(install, source, target)
            (home / "config.yaml").write_bytes(b"post-update config\n")
            seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
        finally:
            release()
    assert not marker.exists()
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert not marker.exists()
    assert (generation / ".lease-managed").is_file()
    assert not list((generation / ".leases").glob("*"))


def test_generated_products_launchers_and_source_identity_restore(transaction):
    install, source, home, sibling, before, target = transaction
    outputs = {
        "node_modules/.hermes-node-deps": b"old dependency receipt\n",
        "ui-tui/dist/main.js": b"old terminal bundle\n",
        "hermes_cli/web_dist/index.html": b"old web bundle\n",
    }
    for name, content in outputs.items():
        file = install / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content)
    receipt = install / ".hermes-bootstrap-complete"
    receipt.write_text(json.dumps({"schemaVersion": 1, "pinnedCommit": before,
                                   "pinnedBranch": git(install, "symbolic-ref", "--short", "HEAD")}))
    stamp_before = write_source_stamp(install)
    assert stamp_before["commit"] == before
    launcher_dir = install / ".hermes" / "bin"
    launcher_dir.mkdir(parents=True)
    old_python = home / "tools" / "python-old" / "bin" / "python3"
    old_python.parent.mkdir(parents=True)
    old_python.write_bytes(b"old pinned interpreter\n")
    launchers = [mint_launcher(name, install, launcher_dir, old_python, None)
                 for name in ("hermes", "hermes-acp")]
    assert all(launcher is not None for launcher in launchers)
    old_launchers = [launcher.read_bytes() for launcher in launchers]
    old_stamp = (install / "install-stamp.json").read_bytes()
    old_receipt = receipt.read_bytes()
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    for name in outputs:
        (install / name).write_bytes(b"target product\n")
    new_python = home / "tools" / "python-new" / "bin" / "python3"
    new_python.parent.mkdir(parents=True)
    new_python.write_bytes(b"new pinned interpreter\n")
    for name, launcher in zip(("hermes", "hermes-acp"), launchers):
        assert mint_launcher(name, install, launcher_dir, new_python, None) is not None
        assert new_python.as_posix().encode() in launcher.read_bytes()
    assert write_source_stamp(install)["commit"] == target
    assert json.loads(receipt.read_text())["pinnedCommit"] == target
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    for name, content in outputs.items():
        assert (install / name).read_bytes() == content
    for launcher, old_bytes in zip(launchers, old_launchers):
        assert launcher.read_bytes() == old_bytes
        assert old_python.as_posix().encode() in launcher.read_bytes()
    assert (install / "install-stamp.json").read_bytes() == old_stamp
    assert receipt.read_bytes() == old_receipt


def test_external_runtime_selection_and_retained_generation_restore(transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    store = tmp_path / "external-runtime"
    store.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    state = install_state_dir(install)
    old_generation = state / "environments" / "old"
    old_venv = old_generation / "venv"
    old_venv.mkdir(parents=True)
    (old_venv / "pyvenv.cfg").write_text("home = old\n")
    old_workspace = old_generation / "workspace" / "uv.lock"
    old_workspace.parent.mkdir()
    old_workspace.write_bytes(b"old resolved dependency graph\n")
    Facts(runtime_facts_path(install)).record_state(
        "venv", "old-stamp", ["pdf"], environment=old_venv, resolved_lock=old_workspace)
    pm_root = state / "pm-runtime"
    old_pm = pm_root / "generations" / "old"
    old_pm.mkdir(parents=True)
    (old_pm / ".lease-managed").touch()
    (old_pm / "pm-runtime.json").write_text(json.dumps({"inputs": "old-inputs"}))
    (pm_root / "selected.json").write_text(json.dumps({
        "inputs": "old-inputs", "generation": "generations/old"}))
    old_tool = store / "python-old" / "bin" / "python3"
    old_tool.parent.mkdir(parents=True)
    old_tool.write_bytes(b"old pinned python\n")
    Facts(store / "facts.json").record("python", "old-version", "python-old", {},
                                      store, target="linux-x64", artifacts=["old-sha"])
    alias = tmp_path / "shared-python"
    os.link(old_tool, alias)
    alias_stat = alias.stat()
    assert selected_venv(install) == old_venv
    assert project_python(install).is_relative_to(old_venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new_generation = state / "environments" / "new"
    new_venv = new_generation / "venv"
    new_venv.mkdir(parents=True)
    (new_venv / "pyvenv.cfg").write_text("home = new\n")
    new_workspace = new_generation / "workspace" / "uv.lock"
    new_workspace.parent.mkdir()
    new_workspace.write_bytes(b"new resolved dependency graph\n")
    Facts(runtime_facts_path(install)).record_state(
        "venv", "new-stamp", [], environment=new_venv, resolved_lock=new_workspace)
    new_pm = pm_root / "generations" / "new"
    new_pm.mkdir(parents=True)
    (new_pm / ".lease-managed").touch()
    (new_pm / "pm-runtime.json").write_text(json.dumps({"inputs": "new-inputs"}))
    (pm_root / "selected.json").write_text(json.dumps({
        "inputs": "new-inputs", "generation": "generations/new"}))
    new_tool = store / "python-new" / "bin" / "python3"
    new_tool.parent.mkdir(parents=True)
    new_tool.write_bytes(b"new pinned python\n")
    Facts(store / "facts.json").record("python", "new-version", "python-new", {},
                                      store, target="linux-x64", artifacts=["new-sha"])
    assert selected_venv(install) == new_venv
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    collect_runtime_generations(pm_root)
    _gc_store(Store(store), Facts(store / "facts.json", strict=True))
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert selected_venv(install) == old_venv
    assert project_python(install).is_relative_to(old_venv)
    assert old_workspace.read_bytes() == b"old resolved dependency graph\n"
    assert json.loads((pm_root / "selected.json").read_text()) == {
        "inputs": "old-inputs", "generation": "generations/old"}
    assert (old_pm / "pm-runtime.json").is_file()
    assert Facts(store / "facts.json", strict=True).get("python")["entry"] == "python-old"
    assert old_tool.read_bytes() == b"old pinned python\n"
    assert alias.read_bytes() == b"old pinned python\n"
    assert (alias.stat().st_ino, alias.stat().st_mode, alias.stat().st_mtime_ns) == (
        alias_stat.st_ino, alias_stat.st_mode, alias_stat.st_mtime_ns)
    assert old_tool.stat().st_ino != alias.stat().st_ino


def test_computed_external_state_owner_restores_without_parent_authority(
        transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    owner = tmp_path / "other-pm-home"
    owner.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(owner))
    state = install_state_dir(install)
    old = state / "environments" / "old"
    (old / "venv").mkdir(parents=True)
    (old / "venv" / "pyvenv.cfg").write_text("home = old\n")
    facts_path = runtime_facts_path(install)
    Facts(facts_path).record_state("venv", "old", [], environment=old / "venv")
    unrelated = owner / "unrelated-user-file"
    unrelated.write_bytes(b"outside selection\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new = state / "environments" / "new"
    (new / "venv").mkdir(parents=True)
    (new / "venv" / "pyvenv.cfg").write_text("home = new\n")
    Facts(facts_path).record_state("venv", "new", [], environment=new / "venv")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    unrelated.write_bytes(b"later external data\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert Facts(facts_path, strict=True).get("venv")["environment"] == str(old / "venv")
    assert unrelated.read_bytes() == b"later external data\n"


def test_external_store_facts_later_history_refuses_restore(transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    store = tmp_path / "external-runtime"
    store.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    facts_path = store / "facts.json"
    Facts(facts_path).record("python", "before", "python-before", {}, store)
    old_tool = store / "python-before" / "bin" / "python3"
    old_tool.parent.mkdir(parents=True)
    old_tool.write_bytes(b"before interpreter\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    Facts(facts_path).record("python", "target", "python-target", {}, store)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    Facts(facts_path).record("python", "later", "python-later", {}, store)
    with pytest.raises((ValueError, RuntimeError), match="drift|changed|postimage"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert Facts(facts_path, strict=True).get("python")["entry"] == "python-later"


def test_old_dependency_generation_survives_collection_until_restore(transaction):
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    old = state / "environments" / "old"
    venv = old / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = old\n")
    (old / ".lease-managed").touch()
    workspace = old / "workspace" / "uv.lock"
    workspace.parent.mkdir()
    workspace.write_bytes(b"old resolved lock\n")
    Facts(runtime_facts_path(install)).record_state(
        "venv", "old-stamp", [], environment=venv, resolved_lock=workspace)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new = state / "environments" / "new"
    new_venv = new / "venv"
    new_venv.mkdir(parents=True)
    (new_venv / "pyvenv.cfg").write_text("home = new\n")
    Facts(runtime_facts_path(install)).record_state("venv", "new-stamp", [], environment=new_venv)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    collect_generations(install, min_age_seconds=0)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert selected_venv(install) == venv
    assert workspace.read_bytes() == b"old resolved lock\n"


def test_unselected_home_generation_referenced_by_node_survives_collection_or_refuses(
        transaction):
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    old = state / "environments" / "old-extra"
    package = old / "package"
    package.mkdir(parents=True)
    (package / "entry.js").write_bytes(b"old package bytes\n")
    (old / ".lease-managed").touch()
    selected = state / "environments" / "selected"
    venv = selected / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = selected\n")
    Facts(runtime_facts_path(install)).record_state("venv", "selected", [], environment=venv)
    link = install / "node_modules" / "package"
    link.parent.mkdir()
    link.symlink_to(package, target_is_directory=True)
    try:
        manifest = capture_local_update_capsule(install, home, source, target)
    except ValueError:
        assert git(install, "rev-parse", "HEAD") == before
        assert (link / "entry.js").read_bytes() == b"old package bytes\n"
        return
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    collect_generations(install, min_age_seconds=0)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert selected_venv(install) == venv
    assert (link / "entry.js").read_bytes() == b"old package bytes\n"


def test_skills_referent_excluded_from_home_needs_effective_owner_or_capture_refusal(
        transaction):
    install, source, home, sibling, before, target = transaction
    referent = home / "tools" / "custom-skills"
    referent.mkdir(parents=True)
    skill = referent / "SKILL.md"
    skill.write_bytes(b"original skill\n")
    (home / "skills").symlink_to(referent, target_is_directory=True)
    try:
        manifest = capture_local_update_capsule(install, home, source, target)
    except ValueError:
        assert git(install, "rev-parse", "HEAD") == before
        assert skill.read_bytes() == b"original skill\n"
        return
    advance(install, source, target)
    skill.write_bytes(b"updated skill\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (home / "skills" / "SKILL.md").read_bytes() == b"original skill\n"


def test_unsupported_external_skills_root_refuses_capture(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "external-skills"
    external.mkdir()
    (external / "bundled").mkdir()
    skill = external / "bundled" / "SKILL.md"
    skill.write_bytes(b"old external skill\n")
    (home / "skills").symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError), match="skill|symlink|external"):
        capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    assert skill.read_bytes() == b"old external skill\n"


@pytest.mark.parametrize("collision", ["file", "directory"])
def test_ignored_later_code_collision_refuses_without_overwriting(transaction, collision):
    install, source, home, sibling, _, _ = transaction
    tracked = "app.py" if collision == "file" else "owned/old.txt"
    if collision == "directory":
        (install / "owned").mkdir()
        (install / tracked).write_bytes(b"old tracked directory content\n")
        git(install, "add", tracked)
        git(install, "commit", "-m", "add tracked directory")
    before = git(install, "rev-parse", "HEAD")
    git(source, "fetch", "--no-tags", str(install), before)
    git(source, "checkout", "--detach", before)
    git(source, "rm", "-r", "app.py" if collision == "file" else "owned")
    ignore = "app.py\n" if collision == "file" else "owned/\n"
    (source / ".gitignore").write_text(ignore)
    git(source, "add", ".gitignore")
    git(source, "commit", "-m", "drop and ignore previously tracked path")
    target = git(source, "rev-parse", "HEAD")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    later = install / ("app.py" if collision == "file" else "owned/new.txt")
    later.parent.mkdir(parents=True, exist_ok=True)
    later.write_bytes(b"user-owned ignored content\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    with pytest.raises((ValueError, RuntimeError), match="ignor|overwrite|replac|collision"):
        restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert later.read_bytes() == b"user-owned ignored content\n"


def test_restore_recovers_original_branch_attachment(transaction):
    install, source, home, sibling, before, target = transaction
    branch = git(install, "symbolic-ref", "--short", "HEAD")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home, acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert git(install, "symbolic-ref", "--short", "HEAD") == branch
    assert git(install, "rev-parse", branch) == before


def test_scope_mismatch_and_captured_input_drift_refuse_before_checkout(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    other = tmp_path / "different-install"
    other.mkdir()
    with pytest.raises(ValueError, match="scope"):
        verify_local_update_preimage(manifest, expected_install_root=other, expected_hermes_home=home)
    with pytest.raises(ValueError, match="scope"):
        restore_local_update_capsule(
            manifest, expected_install_root=other, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    (sibling / "config.yaml").write_bytes(b"later edit\n")
    with pytest.raises(ValueError, match="preimage"):
        verify_local_update_preimage(
            manifest, expected_install_root=install, expected_hermes_home=home)
    assert git(install, "rev-parse", "HEAD") == before
    assert (sibling / "config.yaml").read_bytes() == b"later edit\n"


def test_unmanaged_interpreter_drift_refuses_before_restoring_data(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "unmanaged-python"
    external.write_bytes(b"old interpreter\n")
    generation = install_state_dir(install) / "environments" / "old"
    venv = generation / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = unmanaged\n")
    (venv / "bin" / "python3").symlink_to(external)
    Facts(runtime_facts_path(install)).record_state(
        "venv", "old", [], environment=venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    external.write_bytes(b"later interpreter\n")
    with pytest.raises(ValueError, match="interpreter"):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert (home / "config.yaml").read_bytes() == b"target config\n"


def test_external_executable_dependency_refuses_before_capture(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "unmanaged-python"
    external.write_bytes(b"#!/bin/sh\nexit 0\n")
    external.chmod(0o755)
    generation = install_state_dir(install) / "environments" / "old"
    venv = generation / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = unmanaged\n")
    (venv / "bin" / "python3").symlink_to(external)
    Facts(runtime_facts_path(install)).record_state("venv", "old", [], environment=venv)

    with pytest.raises(ValueError, match="external executable"):
        capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    assert external.read_bytes() == b"#!/bin/sh\nexit 0\n"


def test_owned_new_exposure_removed_without_touching_foreign_command(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    user_bin = tmp_path / ".local" / "bin"
    user_bin.mkdir(parents=True)
    foreign = user_bin / "hermes-agent"
    foreign.write_bytes(b"#!/bin/sh\nexec other-install \"$@\"\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    exposed = user_bin / "hermes"
    exposed.write_text(f'#!/bin/sh\nexec {install}/.hermes/bin/hermes \"$@\"\n')
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert not exposed.exists()
    assert foreign.read_bytes() == b"#!/bin/sh\nexec other-install \"$@\"\n"
    assert git(install, "rev-parse", "HEAD") == before


def test_native_fast_forward_restores_branch_and_preserves_target(transaction):
    install, source, home, sibling, before, target = transaction
    branch = git(install, "symbolic-ref", "HEAD")
    manifest = capture_local_update_capsule(install, home, source, target)
    git(install, "fetch", "--no-tags", str(source), target)
    git(install, "merge", "--ff-only", target)
    assert git(install, "rev-parse", branch) == target
    (home / "config.yaml").write_bytes(b"updated data\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "symbolic-ref", "HEAD") == branch
    assert git(install, "rev-parse", branch) == before
    assert (install / "app.py").read_bytes() == b"old code\n"
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert git(install, "rev-parse", f"refs/hermes/local-update-target/{manifest.parent.name}") == target


def test_detached_restore_refuses_later_original_branch_movement(transaction):
    install, source, home, sibling, before, target = transaction
    branch = git(install, "symbolic-ref", "HEAD")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"updated data\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    git(install, "update-ref", branch, target, before)
    refs_before = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "rev-parse", branch) == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert (home / "config.yaml").read_bytes() == b"updated data\n"


@pytest.mark.parametrize("kind", ["home", "state", "selection", "resource", "convenience"])
def test_self_consistent_foreign_root_has_no_authority(transaction, tmp_path, kind):
    from hermes_cli.local_update_rollback import _filters, _image, _metadata, _record_path

    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"updated data\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    foreign = tmp_path / "foreign-data"
    foreign.mkdir()
    foreign_file = foreign / "config.yaml"
    foreign_file.write_bytes(b"old foreign data\n")
    if kind == "home":
        data["homes"].append(str(foreign))
    _record_path(
        foreign, kind, data["roots"], manifest.parent,
        [Path(path) for path in data["homes"]], install, selected=(foreign_file,),
    )
    forged = data["roots"][-1]
    if kind != "resource":
        foreign_file.write_bytes(b"updated foreign data\n")
    forged["post"] = _image(
        foreign, set(map(Path, data["homes"])), install, **_filters(forged),
    )
    forged["post_metadata"] = {
        **_metadata(foreign), "kind": "dir", "mode": foreign.stat().st_mode & 0o7777,
    }
    foreign_bytes = foreign_file.read_bytes()
    manifest.write_text(json.dumps(data))
    refs_before = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert (home / "config.yaml").read_bytes() == b"updated data\n"
    assert foreign_file.read_bytes() == foreign_bytes


@pytest.mark.parametrize("owner,phase", [
    ("convenience", "pre"), ("convenience", "post"), ("selection", "pre"),
])
def test_filtered_owner_rejects_foreign_member_before_mutation(
        transaction, tmp_path, owner, phase):
    from hermes_cli.local_update_rollback import _image

    install, source, home, sibling, before, target = transaction
    if owner == "convenience":
        foreign = tmp_path / ".ssh" / "authorized_keys"
        selected = tmp_path / ".local" / "bin" / "hermes"
    else:
        state = install_state_dir(install)
        foreign = state / "environments" / "old-extra" / "venv" / "pyvenv.cfg"
        selected = runtime_facts_path(install)
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"private before\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    record = next(row for row in data["roots"]
                  if row["kind"] == owner and row["include"] == [str(selected)])
    assert Path(record["path"]) == (tmp_path if owner == "convenience" else state)
    assert foreign != selected and foreign.is_relative_to(Path(record["path"]))
    artifacts = manifest.parent / record["id"]
    staged = tmp_path / "foreign-member-artifacts"
    forged = _image(Path(record["path"]), set(map(Path, data["homes"])), install, staged,
                    include=(foreign,))
    member = next(item for item in forged if item["path"] == foreign.relative_to(
        Path(record["path"])).as_posix())
    index = len(record[phase])
    (artifacts / f"x-{index}").write_bytes((staged / member["xattrs_object"]).read_bytes())
    (artifacts / str(index)).write_bytes((staged / member["object"]).read_bytes())
    record[phase].append({**member, "object": str(index), "xattrs_object": f"x-{index}"})
    if phase == "post":
        foreign.write_bytes(b"private at target\n")
        # Postimage members must describe the actual sealed bytes, not a stale preimage.
        forged_post = _image(Path(record["path"]), set(map(Path, data["homes"])), install,
                             include=(foreign,))
        record["post"][-1].update({
            key: forged_post[-1][key] for key in ("sha256", "size", "mtime_ns")
        })
    else:
        foreign.write_bytes(b"private at target\n")
    foreign_bytes = foreign.read_bytes()
    manifest.write_text(json.dumps(data))
    refs_before = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert (home / "config.yaml").read_bytes() == b"target config\n"
    assert foreign.read_bytes() == foreign_bytes


def test_recorded_symlink_ancestor_cannot_write_outside_home(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    (sibling / "internal-link").symlink_to("skills", target_is_directory=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    external = tmp_path / "external"
    external.mkdir()
    data = json.loads(manifest.read_text())
    record = next(row for row in data["roots"] if row["kind"] == "home" and row["path"] == str(home))
    original_link = next(row for row in record["pre"] if row["path"] == "profiles/work/internal-link")
    original_file = next(row for row in record["pre"] if row["path"] == "profiles/work/skills/sample.md")
    artifacts = manifest.parent / record["id"]
    link_index = len(record["pre"])
    file_index = link_index + 1
    artifacts.joinpath(f"x-{link_index}").write_bytes(artifacts.joinpath(original_link["xattrs_object"]).read_bytes())
    artifacts.joinpath(f"x-{file_index}").write_bytes(artifacts.joinpath(original_file["xattrs_object"]).read_bytes())
    artifacts.joinpath(str(file_index)).write_bytes(artifacts.joinpath(original_file["object"]).read_bytes())
    record["pre"].extend([
        {**original_link, "path": "profiles/work/a", "target": str(external),
         "xattrs_object": f"x-{link_index}"},
        {**original_file, "path": "profiles/work/a/file", "object": str(file_index),
         "xattrs_object": f"x-{file_index}"},
    ])
    manifest.write_text(json.dumps(data))
    refs_before = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert not (external / "file").exists()
    assert not (sibling / "a").exists()


def test_internal_directory_ancestor_restores_new_file_once(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (sibling / "skills" / "new.md").write_bytes(b"new skill\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (sibling / "skills" / "sample.md").read_bytes() == b"old skill\n"
    assert not (sibling / "skills" / "new.md").exists()


def test_hardlinked_sqlite_destination_refuses_before_restore(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    db = sibling / "state.db"
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"updated data\n")
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("INSERT INTO events VALUES ('after')")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    external = tmp_path / "outside-state.db"
    os.link(db, external)
    external_bytes = external.read_bytes()
    refs_before = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    assert (home / "config.yaml").read_bytes() == b"updated data\n"
    assert external.read_bytes() == external_bytes
    with closing(sqlite3.connect(external)) as conn:
        assert conn.execute("SELECT value FROM events").fetchall() == [("before",), ("after",)]


def test_pm_uv_cache_hardlinks_are_outside_rollback_image_and_coverage(
        transaction, tmp_path):
    from hermes_cli.local_update_coverage import dependency_coverage
    from hermes_cli.local_update_runtime import inventory

    install, source, home, sibling, before, target = transaction
    cache = home / "cache" / "uv"
    archived = cache / "archive-v0" / "x" / "file"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b"cached wheel bytes\n")
    alias = archived.with_name("alias")
    os.link(archived, alias)

    manifest = capture_local_update_capsule(install, home, source, target)
    data = json.loads(manifest.read_text())
    assert not any(
        (Path(root["path"]) / member["path"]).is_relative_to(cache)
        for root in data["roots"] for member in root["pre"]
    )
    coverage = dependency_coverage(
        [home, sibling], install, (), inventory(install, [home, sibling]), set())
    assert coverage.byte_owners(archived) == []
    assert coverage.byte_owners(alias) == []

    advance(install, source, target)
    archived.write_bytes(b"later cached bytes\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    sealed = json.loads(manifest.read_text())
    assert not any(
        (Path(root["path"]) / member["path"]).is_relative_to(cache)
        for root in sealed["roots"] for member in root.get("post", ())
    )
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert archived.read_bytes() == alias.read_bytes() == b"later cached bytes\n"
    assert archived.stat().st_ino == alias.stat().st_ino

    mutable = home / "user-data.bin"
    mutable.write_bytes(b"durable data\n")
    os.link(mutable, tmp_path / "external-user-data.bin")
    with pytest.raises(ValueError, match="multiply linked mutable file cannot be rolled back"):
        capture_local_update_capsule(install, home, source, target)


def test_capture_refuses_sqlite_with_foreign_hardlink(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    alias = tmp_path / "external-state.db"
    os.link(sibling / "state.db", alias)
    with pytest.raises((ValueError, RuntimeError)):
        capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    with closing(sqlite3.connect(alias)) as conn:
        assert conn.execute("SELECT value FROM events").fetchall() == [("before",)]


def test_nested_external_skill_category_refuses_capture(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "outside-skills"
    external.mkdir()
    skill = external / "SKILL.md"
    skill.write_bytes(b"user skill\n")
    skills = home / "skills"
    skills.mkdir()
    (skills / "category").symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        capture_local_update_capsule(install, home, source, target)
    assert skills.is_dir() and not skills.is_symlink()
    assert git(install, "rev-parse", "HEAD") == before
    assert skill.read_bytes() == b"user skill\n"


@pytest.mark.parametrize("cycle", [False, True])
def test_internal_skill_alias_chain_refuses_external_descendant_at_capture(
        transaction, tmp_path, cycle):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "foreign-category"
    external.mkdir()
    skill = external / "SKILL.md"
    skill.write_bytes(b"outside skill\n")
    actual = home / "actual-skills"
    alias = home / "alias-skills"
    actual.mkdir()
    alias.mkdir()
    (home / "skills").symlink_to(actual, target_is_directory=True)
    (actual / "internal").symlink_to(alias, target_is_directory=True)
    (alias / "category").symlink_to(external, target_is_directory=True)
    if cycle:
        (alias / "back").symlink_to(actual, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    assert skill.read_bytes() == b"outside skill\n"


def test_nested_internal_skill_category_restores_with_home(transaction):
    install, source, home, sibling, before, target = transaction
    actual = home / "internal-category"
    actual.mkdir()
    (actual / "existing.md").write_bytes(b"old internal skill\n")
    skills = home / "skills"
    skills.mkdir()
    (skills / "category").symlink_to(actual, target_is_directory=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (actual / "new.md").write_bytes(b"updated internal skill\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (skills / "category").is_symlink()
    assert (actual / "existing.md").read_bytes() == b"old internal skill\n"
    assert not (actual / "new.md").exists()


@pytest.mark.platforms("posix")
def test_protected_git_never_runs_configured_fsmonitor(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    marker = tmp_path / "fsmonitor-executed"
    monitor = tmp_path / "fsmonitor"
    monitor.write_text(f"#!/bin/sh\nprintf executed > '{marker}'\n")
    monitor.chmod(0o700)
    git(install, "config", "core.fsmonitor", str(monitor))
    manifest = capture_local_update_capsule(install, home, source, target)
    verify_local_update_preimage(
        manifest, expected_install_root=install, expected_hermes_home=home)
    assert not marker.exists()
    git(install, "-c", "core.fsmonitor=false", "fetch", "--no-tags", str(source), target)
    git(install, "-c", "core.fsmonitor=false", "checkout", "--detach", target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert not marker.exists()


def test_protected_git_ignores_object_directory_redirection(transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    bogus_objects = tmp_path / "bogus-objects"
    bogus_objects.mkdir()
    with monkeypatch.context() as env:
        env.setenv("GIT_OBJECT_DIRECTORY", str(bogus_objects))
        env.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(bogus_objects))
        manifest = capture_local_update_capsule(install, home, source, target)
        verify_local_update_preimage(
            manifest, expected_install_root=install, expected_hermes_home=home)
    advance(install, source, target)
    with monkeypatch.context() as env:
        env.setenv("GIT_OBJECT_DIRECTORY", str(bogus_objects))
        env.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(bogus_objects))
        seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert git(install, "cat-file", "-t", target) == "commit"
    assert not list(bogus_objects.iterdir())


@pytest.mark.platforms("posix")
def test_non_utf8_tracked_path_roundtrips_through_binary_git_inventory(transaction):
    install, source, home, sibling, _, _ = transaction
    name = os.fsdecode(b"non-utf8-\xff\npath")
    original = install / name
    original.write_bytes(b"old bytes\n")
    git(install, "add", "--all")
    git(install, "commit", "-m", "add binary pathname")
    before = git(install, "rev-parse", "HEAD")
    git(source, "fetch", "--no-tags", str(install), before)
    git(source, "checkout", "--detach", before)
    (source / "app.py").write_bytes(b"target code again\n")
    git(source, "add", "app.py")
    git(source, "commit", "-m", "target after binary pathname")
    target = git(source, "rev-parse", "HEAD")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert original.read_bytes() == b"old bytes\n"


def test_failed_artifact_fsync_never_publishes_capsule(transaction, monkeypatch):
    install, source, home, sibling, before, target = transaction

    def fail_flush(_fd):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_flush)
    with pytest.raises(OSError, match="injected fsync failure"):
        capture_local_update_capsule(install, home, source, target)
    assert not list((home / "local-update-capsules").glob("*/manifest.json"))
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert git(install, "rev-parse", "HEAD") == before


def test_failed_capture_resource_flush_never_publishes_capsule(transaction, monkeypatch):
    install, source, home, sibling, before, target = transaction
    from pm import environments

    def fail_flush():
        raise OSError("injected resource flush failure")

    monkeypatch.setattr(environments, "flush_before_selecting", fail_flush)
    with pytest.raises(OSError, match="injected resource flush failure"):
        capture_local_update_capsule(install, home, source, target)
    assert not list((home / "local-update-capsules").glob("*/manifest.json"))
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert git(install, "rev-parse", "HEAD") == before


def test_failed_resource_flush_does_not_publish_restored_selection(transaction, monkeypatch):
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    old = state / "environments" / "old"
    (old / "venv").mkdir(parents=True)
    (old / "venv" / "pyvenv.cfg").write_text("home = old\n")
    facts_path = runtime_facts_path(install)
    Facts(facts_path).record_state("venv", "old-stamp", [], environment=old / "venv")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new = state / "environments" / "new"
    (new / "venv").mkdir(parents=True)
    (new / "venv" / "pyvenv.cfg").write_text("home = new\n")
    Facts(facts_path).record_state("venv", "new-stamp", [], environment=new / "venv")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    from pm import environments

    def fail_flush():
        raise OSError("injected resource flush failure")

    monkeypatch.setattr(environments, "flush_before_selecting", fail_flush)
    with pytest.raises(OSError, match="injected resource flush failure"):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert Facts(facts_path, strict=True).get("venv")["environment"] == str(new / "venv")
    assert json.loads(manifest.read_text())["state"] == "sealed"


def test_internal_symlinked_skills_have_one_restore_owner(transaction):
    install, source, home, sibling, before, target = transaction
    real_skills = home / "real-skills"
    real_skills.mkdir()
    (real_skills / "existing.md").write_bytes(b"existing skill\n")
    (home / "skills").symlink_to(real_skills, target_is_directory=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (real_skills / "new.md").write_bytes(b"new skill\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert (real_skills / "existing.md").read_bytes() == b"existing skill\n"
    assert not (real_skills / "new.md").exists()
    assert (home / "skills").is_symlink()
    assert json.loads(manifest.read_text())["state"] == "restored"


def test_new_side_environment_selection_removed_but_generation_retained(transaction):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    side = home / "environments" / "new-tool"
    generation = side / ("gen-" + "a" * 32)
    (generation / "venv").mkdir(parents=True)
    (generation / "venv" / "pyvenv.cfg").write_text("home = retained\n")
    (side / "active.json").write_text(json.dumps({"generation": generation.name}))
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert not (side / "active.json").exists()
    assert (generation / "venv" / "pyvenv.cfg").read_text() == "home = retained\n"
    assert json.loads(manifest.read_text())["state"] == "restored"


def test_owner_created_side_environment_keeps_lock_and_generation_on_rollback(
        transaction, tmp_path, monkeypatch):
    from pm import _uv, operations

    install, source, home, sibling, before, target = transaction
    base = tmp_path / "base" / "python"
    base.parent.mkdir()
    base.touch()
    monkeypatch.setattr(_uv, "_toolchain", lambda **kwargs: (None, base))
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    side = home / "environments" / "new-tool"

    def build(generation, base_python):
        venv = generation / "venv"
        python = operations._python(venv)
        python.parent.mkdir(parents=True)
        python.write_bytes(b"retained placeholder interpreter\n")
        (venv / "pyvenv.cfg").write_text(f"home = {base_python.parent}\n")
        return python

    python = operations._ensure_generation(
        "new-tool", side, {}, build, record={}, explicit=True)
    generation = python.parent.parent.parent
    assert generation.parent == side
    assert (side / ".install.lock").is_file()
    assert (side / "active.json").is_file()
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert side.is_dir() and (side / ".install.lock").is_file()
    assert not (side / "active.json").exists()
    assert python.read_bytes() == b"retained placeholder interpreter\n"
    assert (generation / "venv" / "pyvenv.cfg").is_file()


@pytest.mark.parametrize("name", [
    "backups", "state-snapshots", "browser-profile", "browser-profiles",
])
def test_durable_profile_data_refuses_later_history(transaction, name):
    install, source, home, sibling, before, target = transaction
    file = sibling / name / "user-data.bin"
    file.parent.mkdir()
    file.write_bytes(b"private old data\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    file.write_bytes(b"updated data\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    file.write_bytes(b"later user data\n")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert file.read_bytes() == b"later user data\n"


def test_user_logs_remain_live_without_rewinding_update_audit(transaction):
    install, source, home, sibling, before, target = transaction
    logs = home / "logs"
    receipt = logs / "update_receipts" / "latest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"update": "before"}')
    user_log = logs / "user-notes.log"
    user_log.write_bytes(b"old user notes\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    receipt.write_text('{"update": "target"}')
    user_log.write_bytes(b"target user notes\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert user_log.read_bytes() == b"target user notes\n"
    assert receipt.read_text() == '{"update": "target"}'


def test_audit_only_logs_parent_retained_when_ordinary_log_arrives(transaction):
    install, source, home, sibling, before, target = transaction
    logs = home / "logs"
    receipt = logs / "update_receipts" / "latest.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(b"old audit\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    receipt.write_bytes(b"target audit\n")
    user_log = logs / "user-notes.log"
    user_log.write_bytes(b"target ordinary log\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert logs.is_dir() and receipt.read_bytes() == b"target audit\n"
    assert user_log.read_bytes() == b"target ordinary log\n"


def test_durable_profile_trees_roundtrip_without_omitting_user_data(transaction):
    install, source, home, sibling, before, target = transaction
    files = [sibling / name / "user-data.bin" for name in (
        "backups", "state-snapshots", "browser-profile", "browser-profiles")]
    for file in files:
        file.parent.mkdir()
        file.write_bytes(b"private old data\n")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    for file in files:
        file.write_bytes(b"updated data\n")
        (file.parent / "new-data.bin").write_bytes(b"new data\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    for file in files:
        assert file.read_bytes() == b"private old data\n"
        assert not (file.parent / "new-data.bin").exists()


@pytest.mark.parametrize("consumer", ["application", "copied-python", "pm-runtime", "node-package"])
def test_selected_dependency_closure_survives_real_store_collection_or_refuses_capture(
        transaction, tmp_path, monkeypatch, consumer):
    install, source, home, sibling, before, target = transaction
    store = tmp_path / "external-store"
    store.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    old = store / ("node-old" if consumer == "node-package" else "python-old")
    dependency = old / ("package" if consumer == "node-package" else "lib/stdlib.txt")
    dependency.parent.mkdir(parents=True)
    dependency.write_bytes(b"old dependency bytes\n")
    state = install_state_dir(install)
    if consumer in {"application", "copied-python"}:
        venv = state / "environments" / "old" / "venv"
        (venv / "bin").mkdir(parents=True)
        (old / "bin").mkdir()
        executable = old / "bin" / "python3"
        executable.write_bytes(b"old python bytes\n")
        (venv / "pyvenv.cfg").write_text(f"home = {old / 'bin'}\n")
        if consumer == "application":
            (venv / "bin" / "python3").symlink_to(executable)
        else:
            (venv / "bin" / "python3").write_bytes(b"copied python bytes\n")
        Facts(runtime_facts_path(install)).record_state(
            "venv", "old", [], environment=venv)
    elif consumer == "pm-runtime":
        pm_root = state / "pm-runtime"
        runtime = pm_root / "generations" / "old"
        (runtime / "venv" / "bin").mkdir(parents=True)
        (old / "bin").mkdir()
        (old / "bin" / "python3").write_bytes(b"old python bytes\n")
        (runtime / "venv" / "pyvenv.cfg").write_text(f"home = {old / 'bin'}\n")
        (runtime / "venv" / "bin" / "python3").symlink_to(old / "bin" / "python3")
        (runtime / "pm-runtime.json").write_text(json.dumps({"inputs": "old"}))
        (pm_root / "selected.json").write_text(json.dumps({
            "inputs": "old", "generation": "generations/old"}))
    else:
        package = install / "node_modules" / "package"
        package.parent.mkdir()
        package.symlink_to(old)
    # The old entry is deliberately absent from current store facts: GC will remove it.
    Facts(store / "facts.json").record("python", "new", "python-new", {}, store)
    (store / "python-new").mkdir()
    try:
        manifest = capture_local_update_capsule(install, home, source, target)
    except ValueError:
        assert git(install, "rev-parse", "HEAD") == before
        assert dependency.read_bytes() == b"old dependency bytes\n"
        return
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    _gc_store(Store(store), Facts(store / "facts.json", strict=True))
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert dependency.read_bytes() == b"old dependency bytes\n"
    if consumer == "application":
        assert (venv / "bin" / "python3").resolve().read_bytes() == b"old python bytes\n"
    elif consumer == "copied-python":
        assert (venv / "bin" / "python3").read_bytes() == b"copied python bytes\n"
        assert (old / "bin" / "python3").read_bytes() == b"old python bytes\n"
    elif consumer == "pm-runtime":
        assert (runtime / "venv" / "bin" / "python3").resolve().read_bytes() == b"old python bytes\n"
    else:
        assert (install / "node_modules" / "package" / "package").read_bytes() == b"old dependency bytes\n"


def test_all_admitted_directory_mtimes_restore_after_selection_publication(
        transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    store = tmp_path / "external-runtime"
    store.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    state = install_state_dir(install)
    old = state / "environments" / "old"
    (old / "venv").mkdir(parents=True)
    (old / "venv" / "pyvenv.cfg").write_text("home = old\n")
    Facts(runtime_facts_path(install)).record_state("venv", "old", [], environment=old / "venv")
    Facts(store / "facts.json").record("python", "old", "python-old", {}, store)
    (store / "python-old").mkdir()
    (store / "python-old" / "tool").write_bytes(b"old tool\n")
    side = home / "environments" / "tool"
    generation = side / ("gen-" + "b" * 32)
    (generation / "venv").mkdir(parents=True)
    (generation / "venv" / "pyvenv.cfg").write_text("home = old\n")
    (side / "active.json").write_text(json.dumps({"generation": generation.name}))
    product = install / "ui-tui" / "dist"
    product.mkdir(parents=True)
    (product / "main.js").write_bytes(b"old product\n")
    # Capture creates a missing host-obligation owner (a designed, persistent
    # side effect); start from a host where the gateway already created it.
    from hermes_cli.update_host_obligation import host_obligation_path
    host_obligation_path().parent.mkdir(parents=True, exist_ok=True)
    expected = {directory: directory.stat().st_mtime_ns for directory in (
        tmp_path, install, home, sibling, state, state / "environments",
        old, old / "venv", store, store / "python-old", side,
        generation, generation / "venv", product)}
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (product / "main.js").write_bytes(b"new product\n")
    (home / "config.yaml").write_bytes(b"new config\n")
    Facts(runtime_facts_path(install)).record_state("venv", "new", [], environment=old / "venv")
    Facts(store / "facts.json").record("python", "new", "python-old", {}, store)
    (side / "active.json").write_text(json.dumps({"generation": generation.name, "updated": True}))
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    for directory, mtime_ns in expected.items():
        assert directory.stat().st_mtime_ns == mtime_ns, directory
    assert (product / "main.js").read_bytes() == b"old product\n"
    assert (side / "active.json").read_text() == json.dumps({"generation": generation.name})


def test_collected_application_and_pm_generation_container_mtimes_restore(transaction):
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    applications = state / "environments"
    old_app = applications / "old"
    (old_app / "venv").mkdir(parents=True)
    (old_app / "venv" / "pyvenv.cfg").write_text("home = old\n")
    (old_app / ".lease-managed").touch()
    Facts(runtime_facts_path(install)).record_state(
        "venv", "old", [], environment=old_app / "venv")
    pm_root = state / "pm-runtime"
    generations = pm_root / "generations"
    old_pm = generations / "old"
    old_pm.mkdir(parents=True)
    (old_pm / ".lease-managed").touch()
    (old_pm / "pm-runtime.json").write_text(json.dumps({"inputs": "old"}))
    (pm_root / "selected.json").write_text(json.dumps({
        "inputs": "old", "generation": "generations/old"}))
    original_mtimes = {directory: directory.stat().st_mtime_ns
                       for directory in (applications, generations)}
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new_app = applications / "new"
    (new_app / "venv").mkdir(parents=True)
    (new_app / "venv" / "pyvenv.cfg").write_text("home = new\n")
    Facts(runtime_facts_path(install)).record_state(
        "venv", "new", [], environment=new_app / "venv")
    new_pm = generations / "new"
    new_pm.mkdir()
    (new_pm / "pm-runtime.json").write_text(json.dumps({"inputs": "new"}))
    (pm_root / "selected.json").write_text(json.dumps({
        "inputs": "new", "generation": "generations/new"}))
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    assert old_app in collect_generations(install, min_age_seconds=0)
    assert old_pm in collect_runtime_generations(pm_root)
    assert not old_app.exists() and not old_pm.exists()
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert selected_venv(install) == old_app / "venv"
    assert json.loads((pm_root / "selected.json").read_text())["generation"] == "generations/old"
    assert (old_app / "venv" / "pyvenv.cfg").read_text() == "home = old\n"
    assert (old_pm / "pm-runtime.json").is_file()
    for directory, mtime_ns in original_mtimes.items():
        assert directory.stat().st_mtime_ns == mtime_ns, directory


def omit_capsule_owners(manifest, data, *omitted):
    """Keep capsule artifact identities consistent after removing forged-away owners."""
    for record in omitted:
        shutil.rmtree(manifest.parent / record["id"])
    remaining = [record for record in data["roots"] if record not in omitted]
    for index, record in enumerate(remaining):
        old = manifest.parent / record["id"]
        new = manifest.parent / str(index)
        if old != new:
            old.rename(new)
            record["id"] = str(index)
    data["roots"] = remaining


def test_unreferenced_self_authorizing_store_resource_refuses_before_checkout(
        transaction, tmp_path, monkeypatch):
    from hermes_cli.local_update_rollback import _record_path

    install, source, home, sibling, before, target = transaction
    store = tmp_path / "external-store"
    store.mkdir()
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)

    rogue = store / "rogue"
    rogue.mkdir()
    (rogue / "payload").write_bytes(b"forged retained payload\n")
    (rogue / "back").symlink_to(rogue, target_is_directory=True)
    data = json.loads(manifest.read_text())
    _record_path(rogue, "resource", data["roots"], manifest.parent,
                 [Path(path) for path in data["homes"]], install)
    assert {row["path"] for row in data["roots"] if row["kind"] == "resource"} == {str(rogue)}
    shutil.rmtree(rogue)
    manifest.write_text(json.dumps(data))

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert (home / "config.yaml").read_bytes() == b"target config\n"
    assert not rogue.exists()


def test_widened_absent_launcher_anchor_cannot_change_account_home(transaction, tmp_path):
    from hermes_cli.local_update_rollback import _attributes, _metadata

    install, source, home, sibling, before, target = transaction
    user_bin = tmp_path / ".local" / "bin"
    user_bin.mkdir(parents=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    launcher = user_bin / "hermes"
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    record = next(row for row in data["roots"] if row.get("command_path") == str(launcher))
    assert record["path"] == str(user_bin) and record["pre"] == record["post"] == []
    account_mode = tmp_path.stat().st_mode & 0o7777
    record["path"] = str(tmp_path)
    metadata = _metadata(tmp_path)
    (manifest.parent / record["id"] / "x-root").write_bytes(_attributes(tmp_path))
    record["pre_metadata"] = {**metadata, "kind": "dir", "xattrs_object": "x-root",
                              "mode": 0o700 if account_mode != 0o700 else 0o755}
    record["post_metadata"] = {**metadata, "mode": account_mode}
    manifest.write_text(json.dumps(data))

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert (tmp_path.stat().st_mode & 0o7777) == account_mode
    assert not launcher.exists()
    assert (home / "config.yaml").read_bytes() == b"target config\n"


def test_omitted_created_launcher_owner_refuses_before_checkout(transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    user_bin = tmp_path / ".local" / "bin"
    user_bin.mkdir(parents=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    launcher = user_bin / "hermes"
    launcher.write_text(f'#!/bin/sh\nexec {install}/.hermes/bin/hermes "$@"\n')
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    omitted = next(row for row in data["roots"] if row.get("command_path") == str(launcher))
    omit_capsule_owners(manifest, data, omitted)
    manifest.write_text(json.dumps(data))

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    launcher_bytes = launcher.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert launcher.read_bytes() == launcher_bytes
    assert (home / "config.yaml").read_bytes() == b"target config\n"


def test_omitted_excluded_mutation_referent_refuses_before_checkout(transaction, monkeypatch):
    install, source, home, sibling, before, target = transaction
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(home / "tools"))
    referent = home / "tools" / "custom-skills"
    referent.mkdir(parents=True)
    skill = referent / "SKILL.md"
    skill.write_bytes(b"original skill\n")
    (home / "skills").symlink_to(referent, target_is_directory=True)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    skill.write_bytes(b"target skill\n")
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    omitted = next(row for row in data["roots"]
                   if row["kind"] == "convenience" and row["include"] == [str(referent)])
    omit_capsule_owners(manifest, data, omitted)
    manifest.write_text(json.dumps(data))

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert skill.read_bytes() == b"target skill\n"
    assert (home / "skills").is_symlink()
    assert (home / "config.yaml").read_bytes() == b"target config\n"


def test_existing_side_selection_cannot_be_relabelled_as_new(transaction):
    install, source, home, sibling, before, target = transaction
    side = home / "environments" / "tool"
    old = side / ("gen-" + "a" * 32)
    (old / "venv").mkdir(parents=True)
    (old / "venv" / "pyvenv.cfg").write_text("home = old\n")
    pointer = side / "active.json"
    before_pointer = json.dumps({"generation": old.name})
    pointer.write_text(before_pointer)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new = side / ("gen-" + "b" * 32)
    (new / "venv").mkdir(parents=True)
    (new / "venv" / "pyvenv.cfg").write_text("home = new\n")
    after_pointer = json.dumps({"generation": new.name})
    pointer.write_text(after_pointer)
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)

    data = json.loads(manifest.read_text())
    selection = next(row for row in data["roots"]
                     if row["kind"] == "selection" and row["include"] == [str(pointer)])
    data["new_selections"].append({"path": str(pointer), "post": selection["post"]})
    old_resource = next(row for row in data["roots"]
                        if row["kind"] == "resource" and row["path"] == str(old))
    omit_capsule_owners(manifest, data, selection, old_resource)
    manifest.write_text(json.dumps(data))

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert pointer.read_text() == after_pointer
    assert (old / "venv" / "pyvenv.cfg").read_text() == "home = old\n"
    assert (new / "venv" / "pyvenv.cfg").read_text() == "home = new\n"
    assert (home / "config.yaml").read_bytes() == b"target config\n"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("owner", ["home", "state"])
def test_owned_directory_dependency_checks_external_executable_descendants(
        transaction, tmp_path, owner):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "unmanaged-helper"
    external.write_bytes(b"#!/bin/sh\nexit 0\n")
    external.chmod(0o755)
    package = (home / "package" if owner == "home"
               else install_state_dir(install) / "shared-package")
    package.mkdir(parents=True)
    (package / "helper").symlink_to(external)
    link = install / "node_modules" / "package"
    link.parent.mkdir()
    link.symlink_to(package, target_is_directory=True)

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert (link / "helper").resolve() == external
    assert external.read_bytes() == b"#!/bin/sh\nexit 0\n"
    assert (home / "config.yaml").read_bytes() == b"old config\n"

@pytest.mark.platforms("posix")
def test_product_pycache_directory_link_to_external_tree_refuses_capture(
        transaction, tmp_path):
    install, source, home, sibling, before, target = transaction
    external = tmp_path / "external-tree"
    external.mkdir()
    (external / "module.py").write_bytes(b"outside\n")
    package = install / "node_modules" / "pkg"
    package.mkdir(parents=True)
    (package / "__pycache__").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="external dependency|no unique rollback byte owner"):
        capture_local_update_capsule(install, home, source, target)
    assert git(install, "rev-parse", "HEAD") == before
    assert (external / "module.py").read_bytes() == b"outside\n"


@pytest.mark.platforms("posix")
def test_preexisting_empty_side_parent_keeps_metadata_and_new_generation(
        transaction, monkeypatch):
    from pm import _uv, operations
    import sys

    install, source, home, sibling, before, target = transaction
    side = home / "environments" / "tool"
    side.mkdir(parents=True)
    side.chmod(0o750)
    os.utime(side, ns=(1_700_000_000_000_000_000,) * 2)
    original = side.stat()
    base = Path(sys.executable).resolve()
    monkeypatch.setattr(_uv, "_toolchain", lambda **kwargs: (None, base))
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)

    def build(generation, base_python):
        venv = generation / "venv"
        python = operations._python(venv)
        python.parent.mkdir(parents=True)
        python.symlink_to(base_python)
        (venv / "pyvenv.cfg").write_text(f"home = {base_python.parent}\n")
        return python

    python = operations._ensure_generation(
        "tool", side, {}, build, record={}, explicit=True)
    generation = python.parent.parent.parent
    assert generation.parent == side and (side / ".install.lock").is_file()
    assert (side / "active.json").is_file()
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert side.is_dir() and (side / ".install.lock").is_file()
    assert not (side / "active.json").exists()
    assert python.is_symlink() and python.resolve() == base
    assert (generation / "venv" / "pyvenv.cfg").is_file()
    assert (side.stat().st_mode & 0o7777) == (original.st_mode & 0o7777)
    assert side.stat().st_mtime_ns == original.st_mtime_ns
    assert side.stat().st_uid == original.st_uid and side.stat().st_gid == original.st_gid


def test_receipt_metadata_restores_without_rewinding_live_logs(transaction):
    install, source, home, sibling, before, target = transaction
    logs = home / "logs"
    receipts = logs / "update_receipts"
    receipts.mkdir(parents=True)
    receipt = receipts / "latest.json"
    receipt.write_bytes(b"old audit\n")
    logs.chmod(0o750)
    receipts.chmod(0o700)
    for directory in (logs, receipts):
        os.utime(directory, ns=(1_700_000_000_000_000_000,) * 2)
    original = {directory: directory.stat() for directory in (logs, receipts)}
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    receipt.write_bytes(b"target audit\n")
    log = logs / "user.log"
    log.write_bytes(b"target user log\n")
    logs.chmod(0o755)
    receipts.chmod(0o755)
    os.utime(receipts, ns=(1_710_000_000_000_000_000,) * 2)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == before
    assert log.read_bytes() == b"target user log\n"
    assert receipt.read_bytes() == b"target audit\n"
    assert (logs.stat().st_mode & 0o7777) == 0o755
    before_stat = original[receipts]
    after_stat = receipts.stat()
    assert (after_stat.st_mode & 0o7777) == (before_stat.st_mode & 0o7777)
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert (after_stat.st_uid, after_stat.st_gid) == (before_stat.st_uid, before_stat.st_gid)


def test_forged_retained_lease_descendants_refuse_before_checkout(
        transaction, tmp_path):
    from hermes_cli.local_update_rollback import _image

    install, source, home, sibling, before, target = transaction
    staging = tmp_path / "lease-stage"
    forged = staging / ".leases" / "forged"
    forged.parent.mkdir(parents=True)
    forged.write_bytes(b"unauthorized lease payload\n")
    staged_objects = tmp_path / "lease-objects"
    members = _image(staging, set(), install, staged_objects)
    old = install_state_dir(install) / "environments" / "old"
    venv = old / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = old\n")
    (old / ".lease-managed").touch()
    Facts(runtime_facts_path(install)).record_state("venv", "old", [], environment=venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    new = old.parent / "new"
    (new / "venv").mkdir(parents=True)
    (new / "venv" / "pyvenv.cfg").write_text("home = new\n")
    Facts(runtime_facts_path(install)).record_state("venv", "new", [], environment=new / "venv")
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)

    data = json.loads(manifest.read_text())
    resource = next(row for row in data["roots"]
                    if row["kind"] == "resource" and row["path"] == str(old))
    assert any(item["path"] == ".lease-managed" for item in resource["pre"])
    artifacts = manifest.parent / resource["id"]
    for member in members:
        index = len(resource["pre"])
        item = {**member, "xattrs_object": f"x-{index}"}
        (artifacts / item["xattrs_object"]).write_bytes(
            (staged_objects / member["xattrs_object"]).read_bytes())
        if item["kind"] == "file":
            item["object"] = str(index)
            (artifacts / item["object"]).write_bytes(
                (staged_objects / member["object"]).read_bytes())
        resource["pre"].append(item)
    manifest.write_text(json.dumps(data))
    assert old in collect_generations(install, min_age_seconds=0)
    assert not old.exists()

    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    with pytest.raises((ValueError, RuntimeError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert git(install, "rev-parse", "HEAD") == target
    assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
    assert not old.exists() and not (old / ".leases" / "forged").exists()
    assert (new / "venv" / "pyvenv.cfg").read_text() == "home = new\n"
    assert (home / "config.yaml").read_bytes() == b"target config\n"


@pytest.mark.parametrize("artifact_kind", ["root-xattrs", "file", "database", "bundle"])
def test_capsule_write_after_artifact_check_never_restores_unverified_input(
        transaction, monkeypatch, artifact_kind):
    from hermes_cli import local_update_rollback as rollback

    install, source, home, sibling, before, target = transaction
    branch = git(install, "symbolic-ref", "--short", "HEAD")
    if artifact_kind == "root-xattrs":
        try:
            os.setxattr(home, "user.rollback", b"captured home attribute")
        except (AttributeError, OSError) as exc:
            pytest.skip(f"user extended attributes unavailable: {exc}")
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    if artifact_kind == "root-xattrs":
        os.setxattr(home, "user.rollback", b"target home attribute")
    if artifact_kind == "database":
        with closing(sqlite3.connect(sibling / "state.db")) as conn, conn:
            conn.execute("INSERT INTO events VALUES ('after')")
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    if artifact_kind == "bundle":
        artifact = manifest.parent / "code.bundle"
    else:
        root = next(row for row in data["roots"] if row["kind"] == "home"
                    and row["path"] == str(home))
        if artifact_kind == "root-xattrs":
            artifact = manifest.parent / root["id"] / "x-root"
        else:
            member_path = ("config.yaml" if artifact_kind == "file"
                           else "profiles/work/state.db")
            member = next(item for item in root["pre"] if item["path"] == member_path)
            artifact = manifest.parent / root["id"] / member["object"]
    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")
    checked = rollback._checked_artifacts
    injected = False

    def write_after_real_check(path, capsule_data):
        nonlocal injected
        checked(path, capsule_data)
        if injected:
            return
        injected = True
        if artifact_kind == "database":
            with closing(sqlite3.connect(artifact)) as conn, conn:
                conn.execute("UPDATE events SET value = 'untrusted'")
        else:
            artifact.write_bytes({
                "root-xattrs": b"{}",
                "file": b"untrusted config\n",
                "bundle": b"not a Git bundle\n",
            }[artifact_kind])

    monkeypatch.setattr(rollback, "_checked_artifacts", write_after_real_check)
    try:
        result = restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    except (ValueError, RuntimeError, OSError):
        assert injected
        assert json.loads(manifest.read_text())["state"] == "sealed"
        assert git(install, "rev-parse", "HEAD") == target
        assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
        assert git(install, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
        assert (install / "app.py").read_bytes() == b"target code\n"
        assert (home / "config.yaml").read_bytes() == b"target config\n"
        if artifact_kind == "root-xattrs":
            assert os.getxattr(home, "user.rollback") == b"target home attribute"
        if artifact_kind == "database":
            with closing(sqlite3.connect(sibling / "state.db")) as conn:
                assert conn.execute("SELECT value FROM events").fetchall() == [
                    ("before",), ("after",)]
    else:
        assert injected
        assert result["state"] == json.loads(manifest.read_text())["state"] == "restored"
        assert git(install, "rev-parse", "HEAD") == before
        assert git(install, "symbolic-ref", "--short", "HEAD") == branch
        assert git(install, "rev-parse", branch) == before
        assert (install / "app.py").read_bytes() == b"old code\n"
        assert (home / "config.yaml").read_bytes() == b"old config\n"
        if artifact_kind == "root-xattrs":
            assert os.getxattr(home, "user.rollback") == b"captured home attribute"
        if artifact_kind == "database":
            with closing(sqlite3.connect(sibling / "state.db")) as conn:
                assert conn.execute("SELECT value FROM events").fetchall() == [("before",)]


@pytest.mark.parametrize("location", ["capsule-storage", "generation-leases", "covered-home"])
def test_node_directory_reference_requires_effective_image_coverage(
        transaction, location):
    install, source, home, sibling, before, target = transaction
    branch = git(install, "symbolic-ref", "HEAD")
    if location == "capsule-storage":
        capsule_dir = home / "local-update-capsules"
        capsule_dir.mkdir(mode=0o700)
        package = capsule_dir / "package"
    elif location == "generation-leases":
        old = install_state_dir(install) / "environments" / "old"
        venv = old / "venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = old\n")
        (old / ".lease-managed").touch()
        Facts(runtime_facts_path(install)).record_state(
            "venv", "old", [], environment=venv)
        package = old / ".leases" / "package"
    else:
        package = home / "dependencies" / "package"
    package.mkdir(parents=True)
    entry = package / "entry.js"
    entry.write_bytes(b"captured package bytes\n")
    link = install / "node_modules" / "package"
    link.parent.mkdir()
    link.symlink_to(package, target_is_directory=True)
    refs = git(install, "for-each-ref", "--format=%(refname) %(objectname)")

    if location != "covered-home":
        with pytest.raises((ValueError, RuntimeError)):
            capture_local_update_capsule(install, home, source, target)
        assert git(install, "rev-parse", "HEAD") == before
        assert git(install, "symbolic-ref", "HEAD") == branch
        assert git(install, "for-each-ref", "--format=%(refname) %(objectname)") == refs
        assert (install / "app.py").read_bytes() == b"old code\n"
        assert (home / "config.yaml").read_bytes() == b"old config\n"
        assert (sibling / "config.yaml").read_bytes() == b"old sibling\n"
        assert entry.read_bytes() == b"captured package bytes\n"
        assert link.is_symlink() and (link / "entry.js").read_bytes() == entry.read_bytes()
        assert not list((home / "local-update-capsules").glob("*/manifest.json"))
        return

    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    entry.write_bytes(b"target package bytes\n")
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    restored = restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert restored["state"] == json.loads(manifest.read_text())["state"] == "restored"
    assert git(install, "rev-parse", "HEAD") == before
    assert git(install, "symbolic-ref", "HEAD") == branch
    assert (install / "app.py").read_bytes() == b"old code\n"
    assert link.is_symlink()
    assert entry.read_bytes() == (link / "entry.js").read_bytes() == b"captured package bytes\n"


@pytest.mark.parametrize("artifact_kind", ["xattrs", "file"])
def test_capture_artifact_destination_cannot_write_capsule_writer_victim(
        transaction, tmp_path, monkeypatch, artifact_kind):
    from hermes_cli import local_update_rollback as rollback

    install, source, home, sibling, before, target = transaction
    original = home / "config.yaml"
    if artifact_kind == "xattrs":
        try:
            os.setxattr(original, "user.rollback", b"captured attribute")
        except (AttributeError, OSError) as exc:
            pytest.skip(f"user extended attributes unavailable: {exc}")
    positive = capture_local_update_capsule(install, home, source, target)
    baseline = json.loads(positive.read_text())
    root = next(row for row in baseline["roots"] if row["path"] == str(home))
    item = next(row for row in root["pre"] if row["path"] == "config.yaml")
    object_name = item["xattrs_object"] if artifact_kind == "xattrs" else item["object"]
    victim = tmp_path / "unrelated-writable-file"
    victim.write_bytes(b"unrelated ordinary user data\n")
    victim.chmod(0o640)
    saved = (victim.read_bytes(), victim.stat().st_mode & 0o777)
    injected = False

    def public_artifact():
        capsules = home / "local-update-capsules"
        other = [entry for entry in capsules.iterdir()
                 if entry.is_dir() and entry != positive.parent
                 and len(entry.name) == 32
                 and all(char in "0123456789abcdef" for char in entry.name)]
        assert len(other) == 1
        return other[0] / root["id"] / object_name

    def substitute_path(destination):
        nonlocal injected
        if not injected and Path(destination) == public_artifact():
            assert not destination.exists() and not destination.is_symlink()
            destination.symlink_to(victim)
            injected = True

    original_metadata = rollback._metadata
    original_copy = rollback.shutil.copyfile
    original_open = os.open

    def at_attribute_write(path, artifact=None):
        if artifact_kind == "xattrs" and path == original and artifact is not None:
            substitute_path(artifact)
        return original_metadata(path, artifact)

    def at_file_copy(src, dst, *args, **kwargs):
        if artifact_kind == "file" and Path(src) == original:
            substitute_path(Path(dst))
        return original_copy(src, dst, *args, **kwargs)

    def at_publication(name, flags, *args, **kwargs):
        nonlocal injected
        parent_fd = kwargs.get("dir_fd")
        if (not injected and parent_fd is not None and str(name) == object_name
                and flags & os.O_CREAT):
            destination = public_artifact()
            if destination.parent.is_dir():
                named, held = destination.parent.stat(), os.fstat(parent_fd)
                if (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino):
                    os.symlink(victim, object_name, dir_fd=parent_fd)
                    injected = True
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(rollback, "_metadata", at_attribute_write)
    monkeypatch.setattr(rollback.shutil, "copyfile", at_file_copy)
    monkeypatch.setattr(os, "open", at_publication)
    try:
        attacked = capture_local_update_capsule(install, home, source, target)
    except (ValueError, RuntimeError, OSError):
        pass
    else:
        data = json.loads(attacked.read_text())
        assert data["state"] == "captured"
        assert not (attacked.parent / root["id"] / object_name).is_symlink()
    assert injected
    assert (victim.read_bytes(), victim.stat().st_mode & 0o777) == saved
    assert json.loads(positive.read_text())["state"] == "captured"
    assert original.read_bytes() == b"old config\n"
    assert git(install, "rev-parse", "HEAD") == before




def test_manifest_stage_name_swap_cannot_chmod_unrelated_file(
        transaction, tmp_path, monkeypatch):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    victim = tmp_path / "ordinary-writable-manifest-victim"
    victim.write_bytes(b"outside capsule\n")
    victim.chmod(0o640)
    saved = (victim.read_bytes(), victim.stat().st_mode & 0o777)
    original_open = Path.open
    original_rename = os.rename
    capsule_identity = manifest.parent.stat()
    substituted = False

    def replace_opened_stage(path, mode="r", *args, **kwargs):
        nonlocal substituted
        stream = original_open(path, mode, *args, **kwargs)
        if path.parent == manifest.parent and path.name.startswith(".manifest-") and mode == "x":
            path.unlink()
            path.symlink_to(victim)
            substituted = True
        return stream

    def replace_destination_before_publish(src, dst, *args, **kwargs):
        nonlocal substituted
        parent_fd = kwargs.get("dst_dir_fd")
        if parent_fd is not None and str(dst) == "manifest.json":
            held = os.fstat(parent_fd)
            if (held.st_dev, held.st_ino) == (
                    capsule_identity.st_dev, capsule_identity.st_ino):
                os.unlink("manifest.json", dir_fd=parent_fd)
                os.symlink(victim, "manifest.json", dir_fd=parent_fd)
                substituted = True
        return original_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_opened_stage)
    monkeypatch.setattr(os, "rename", replace_destination_before_publish)
    try:
        seal_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home)
    except (ValueError, RuntimeError, OSError):
        if not manifest.is_symlink():
            assert json.loads(manifest.read_text())["state"] == "captured"
    else:
        assert not manifest.is_symlink()
        assert json.loads(manifest.read_text())["state"] == "sealed"
    assert substituted
    assert (victim.read_bytes(), victim.stat().st_mode & 0o777) == saved
    assert git(install, "rev-parse", "HEAD") == target


def test_restored_manifest_publication_cannot_follow_replaced_capsule_directory(
        transaction, tmp_path, monkeypatch):
    from hermes_cli import local_update_rollback as rollback

    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"updated config\n")
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    outside = tmp_path / "unrelated-directory"
    outside.mkdir()
    victim = outside / "manifest.json"
    victim.write_bytes(b"unrelated manifest data\n")
    victim.chmod(0o640)
    saved = (victim.read_bytes(), victim.stat().st_mode & 0o777)
    parked = tmp_path / "original-capsule"
    snapshot = rollback._private_artifacts
    substituted = False

    def replace_after_snapshot(authority, data, private):
        nonlocal substituted
        snapshot(authority, data, private)
        authority.path.parent.rename(parked)
        authority.path.parent.symlink_to(outside, target_is_directory=True)
        substituted = True

    monkeypatch.setattr(rollback, "_private_artifacts", replace_after_snapshot)
    try:
        result = restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    except (ValueError, RuntimeError, OSError):
        assert json.loads((parked / "manifest.json").read_text())["state"] == "sealed"
    else:
        assert result["state"] == "restored"
        assert json.loads((parked / "manifest.json").read_text())["state"] == "restored"
    assert substituted
    assert (victim.read_bytes(), victim.stat().st_mode & 0o777) == saved
    assert manifest.parent.is_symlink()


def test_retained_resource_capsule_named_subtree_survives_collection_and_executes(
        transaction):
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    old = state / "environments" / "old"
    package = old / "local-update-capsules" / "package"
    package.mkdir(parents=True)
    entry = package / "entry.js"
    entry.write_bytes(b"#!/bin/sh\nprintf 'original package runs\\n'\n")
    entry.chmod(0o700)
    (old / ".lease-managed").touch()
    selected = state / "environments" / "selected"
    venv = selected / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = selected\n")
    Facts(runtime_facts_path(install)).record_state(
        "venv", "selected", [], environment=venv)
    link = install / "node_modules" / "package"
    link.parent.mkdir()
    link.symlink_to(package, target_is_directory=True)
    assert subprocess.check_output([str(link / "entry.js")]) == b"original package runs\n"
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    collect_generations(install, min_age_seconds=0)
    assert not old.exists()
    restored = restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert restored["state"] == json.loads(manifest.read_text())["state"] == "restored"
    assert git(install, "rev-parse", "HEAD") == before
    assert selected_venv(install) == venv
    assert entry.read_bytes() == b"#!/bin/sh\nprintf 'original package runs\\n'\n"
    assert subprocess.check_output([str(link / "entry.js")]) == b"original package runs\n"


@pytest.mark.parametrize("change", ["bytecode", "file"])
def test_bytecode_written_into_a_selected_resource_does_not_block_seal(transaction, change):
    """Resource bytecode may be regenerated during the update; other changes refuse."""
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    venv = state / "environments" / "selected" / "venv"
    module = venv / "lib" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    (venv / "pyvenv.cfg").write_text("home = selected\n")
    Facts(runtime_facts_path(install)).record_state("venv", "selected", [], environment=venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    if change == "bytecode":
        (module.parent / "__pycache__").mkdir()
        (module.parent / "__pycache__" / "mod.cpython-314.pyc").write_bytes(b"\x00" * 16)
        seal_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home)
        assert json.loads(manifest.read_text())["state"] == "sealed"
    else:
        (module.parent / "added.py").write_text("VALUE = 2\n")
        with pytest.raises(ValueError, match="selected resource changed during update"):
            seal_local_update_capsule(
                manifest, expected_install_root=install, expected_hermes_home=home)


def test_planted_valid_resource_bytecode_is_purged_on_restore(transaction):
    import importlib.util
    import py_compile
    import sys

    install, source, home, sibling, before, target = transaction
    venv = install_state_dir(install) / "environments" / "selected" / "venv"
    stdlib = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib.mkdir(parents=True)
    module = stdlib / "rollback_probe.py"
    module.write_bytes(b"VALUE = 1\n")
    (venv / "pyvenv.cfg").write_text("home = selected\n")
    Facts(runtime_facts_path(install)).record_state("venv", "selected", [], environment=venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    cache = Path(importlib.util.cache_from_source(str(module)))
    cache.parent.mkdir()
    py_compile.compile(str(module), cfile=str(cache), doraise=True)
    assert cache.read_bytes().startswith(importlib.util.MAGIC_NUMBER)
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert not cache.exists()
    assert module.read_bytes() == b"VALUE = 1\n"
    assert git(install, "rev-parse", "HEAD") == before

@pytest.mark.platforms("posix")
def test_resource_bytecode_purge_failure_refuses_restore_before_code_or_data(transaction):
    if os.geteuid() == 0:
        pytest.skip("root can unlink entries in a read-only directory")

    install, source, home, sibling, before, target = transaction
    venv = install_state_dir(install) / "environments" / "selected" / "venv"
    module = venv / "lib" / "module.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"VALUE = 1\n")
    (venv / "pyvenv.cfg").write_text("home = selected\n")
    Facts(runtime_facts_path(install)).record_state("venv", "selected", [], environment=venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"post config\n")
    cache = module.parent / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"regenerable bytecode\n")
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    code = install / "app.py"
    code_stat = code.stat()
    os.utime(code, ns=(code_stat.st_atime_ns, code_stat.st_mtime_ns + 5_000_000_000))
    index = install / ".git" / "index"
    objects = install / ".git" / "objects"
    index_before = index.read_bytes()
    objects_before = {p.relative_to(objects) for p in objects.rglob("*") if p.is_file()}
    cache.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            restore_local_update_capsule(
                manifest, expected_install_root=install, expected_hermes_home=home,
                acknowledge_data_rollback=True)
        assert git(install, "rev-parse", "HEAD") == target
        assert (home / "config.yaml").read_bytes() == b"post config\n"
        assert json.loads(manifest.read_text())["state"] == "sealed"
        assert index.read_bytes() == index_before
        assert {p.relative_to(objects) for p in objects.rglob("*") if p.is_file()} == objects_before
    finally:
        cache.chmod(0o755)

    restored = restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert restored["state"] == "restored"
    assert git(install, "rev-parse", "HEAD") == before
    assert (home / "config.yaml").read_bytes() == b"old config\n"
    assert not (cache / "module.pyc").exists()


@pytest.mark.parametrize("cache_type", ["symlink", "file"])
def test_non_directory_resource_pycache_is_unlinked_without_following(
        transaction, tmp_path, cache_type):
    install, source, home, sibling, before, target = transaction
    venv = install_state_dir(install) / "environments" / "selected" / "venv"
    module = venv / "lib" / "module.py"
    module.parent.mkdir(parents=True)
    module.write_bytes(b"VALUE = 1\n")
    (venv / "pyvenv.cfg").write_text("home = selected\n")
    Facts(runtime_facts_path(install)).record_state("venv", "selected", [], environment=venv)
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    seal_local_update_capsule(manifest, expected_install_root=install, expected_hermes_home=home)
    cache = module.parent / "__pycache__"
    victim = tmp_path / "external-cache"
    victim.mkdir()
    (victim / "keep").write_bytes(b"external\n")
    if cache_type == "symlink":
        cache.symlink_to(victim, target_is_directory=True)
    else:
        cache.write_bytes(b"not a directory\n")
    restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert not cache.exists() and not cache.is_symlink()
    assert (victim / "keep").read_bytes() == b"external\n"
    assert git(install, "rev-parse", "HEAD") == before




def test_oversized_artifact_is_refused_before_initial_validation_hash(
        transaction, monkeypatch):
    from hermes_cli import local_update_rollback as rollback

    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    root = next(row for row in data["roots"] if row["path"] == str(home))
    item = next(row for row in root["pre"] if row["path"] == "config.yaml")
    artifact = manifest.parent / root["id"] / item["object"]
    with artifact.open("wb") as stream:
        stream.truncate(64 << 20)
    original_digest = rollback._digest

    def bounded_digest(path):
        if Path(path) == artifact and path.stat().st_size > item["size"]:
            pytest.fail("oversized artifact reached the hash reader")
        return original_digest(path)

    monkeypatch.setattr(rollback, "_digest", bounded_digest)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert json.loads(manifest.read_text())["state"] == "sealed"
    assert git(install, "rev-parse", "HEAD") == target
    assert (home / "config.yaml").read_bytes() == b"target config\n"


@pytest.mark.parametrize("change", ["after-check", "after-fstat"])
def test_snapshot_never_reads_beyond_attested_file_length(
        transaction, monkeypatch, change):
    from contextlib import contextmanager
    from hermes_cli import local_update_rollback as rollback

    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    advance(install, source, target)
    (home / "config.yaml").write_bytes(b"target config\n")
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    data = json.loads(manifest.read_text())
    root = next(row for row in data["roots"] if row["path"] == str(home))
    item = next(row for row in root["pre"] if row["path"] == "config.yaml")
    artifact = manifest.parent / root["id"] / item["object"]
    original_check = rollback._checked_artifacts
    original_fstat = os.fstat
    original_fdopen = os.fdopen
    armed = False
    grew = False
    reads = 0
    inode = None

    def after_validation(path, capsule_data):
        nonlocal armed, inode
        original_check(path, capsule_data)
        if not armed:
            if change == "after-check":
                replacement = artifact.with_name("oversized-replacement")
                with replacement.open("wb") as stream:
                    stream.truncate(64 << 20)
                replacement.replace(artifact)
            inode = artifact.stat().st_ino
            armed = True

    def grow_after_stat(fd):
        nonlocal grew
        result = original_fstat(fd)
        if armed and not grew and result.st_ino == inode and change == "after-fstat":
            with artifact.open("r+b") as writer:
                writer.truncate(64 << 20)
            grew = True
        return result

    def bounded_fdopen(fd, *args, **kwargs):
        nonlocal reads
        if not armed or original_fstat(fd).st_ino != inode:
            return original_fdopen(fd, *args, **kwargs)

        @contextmanager
        def counted():
            nonlocal reads
            with original_fdopen(fd, *args, **kwargs) as stream:
                def read(amount):
                    nonlocal reads
                    block = stream.read(amount)
                    reads += len(block)
                    if reads > item["size"] + 1:
                        pytest.fail("snapshot consumed beyond the attested length")
                    return block

                yield SimpleNamespace(read=read)

        return counted()

    monkeypatch.setattr(rollback, "_checked_artifacts", after_validation)
    monkeypatch.setattr(os, "fstat", grow_after_stat)
    monkeypatch.setattr(os, "fdopen", bounded_fdopen)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        restore_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home,
            acknowledge_data_rollback=True)
    assert armed
    if change == "after-fstat":
        assert grew
    assert reads <= item["size"] + 1
    assert json.loads(manifest.read_text())["state"] == "sealed"
    assert git(install, "rev-parse", "HEAD") == target
    assert (home / "config.yaml").read_bytes() == b"target config\n"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("phase", ["verify", "seal", "restore"])
@pytest.mark.parametrize("artifact_kind", ["bundle", "file", "database", "xattrs", "manifest"])
def test_fifo_capsule_input_refused_without_blocking_or_publishing(
        transaction, phase, artifact_kind):
    install, source, home, sibling, before, target = transaction
    manifest = capture_local_update_capsule(install, home, source, target)
    if phase != "verify":
        advance(install, source, target)
        (home / "config.yaml").write_bytes(b"target config\n")
    if phase == "restore":
        seal_local_update_capsule(
            manifest, expected_install_root=install, expected_hermes_home=home)

    data = json.loads(manifest.read_text())
    root = next(row for row in data["roots"] if row["kind"] == "home"
                and row["path"] == str(home))
    if artifact_kind == "bundle":
        artifact = manifest.parent / "code.bundle"
    elif artifact_kind == "manifest":
        artifact = manifest
    elif artifact_kind == "xattrs":
        artifact = manifest.parent / root["id"] / root["pre_metadata"]["xattrs_object"]
    else:
        name = ("config.yaml" if artifact_kind == "file"
                else "profiles/work/state.db")
        member = next(row for row in root["pre"] if row["path"] == name)
        artifact = manifest.parent / root["id"] / member["object"]

    original_manifest = manifest.read_bytes()
    with substituted_fifo(artifact):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            if phase == "verify":
                verify_local_update_preimage(
                    manifest, expected_install_root=install, expected_hermes_home=home)
            elif phase == "seal":
                seal_local_update_capsule(
                    manifest, expected_install_root=install, expected_hermes_home=home)
            else:
                restore_local_update_capsule(
                    manifest, expected_install_root=install, expected_hermes_home=home,
                    acknowledge_data_rollback=True)
        assert artifact.is_fifo()

    assert manifest.read_bytes() == original_manifest
    assert json.loads(manifest.read_text())["state"] == (
        "sealed" if phase == "restore" else "captured")
    assert git(install, "rev-parse", "HEAD") == (before if phase == "verify" else target)
    assert (install / "app.py").read_bytes() == (
        b"old code\n" if phase == "verify" else b"target code\n")
    assert (home / "config.yaml").read_bytes() == (
        b"old config\n" if phase == "verify" else b"target config\n")
    with closing(sqlite3.connect(sibling / "state.db")) as conn:
        assert conn.execute("SELECT value FROM events").fetchall() == [("before",)]


def test_sqlite_sidecar_product_dependency_survives_collection_or_refuses_capture(
        transaction):
    install, source, home, sibling, before, target = transaction
    state = install_state_dir(install)
    old = state / "environments" / "old"
    package = old / "package"
    package.mkdir(parents=True)
    db = package / "cache.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE entries (value TEXT)")
        conn.execute("INSERT INTO entries VALUES ('captured')")
    sidecar = package / "cache.db-journal"
    sidecar_bytes = b"\0" * 512 + b"non-hot dependency bytes\n"
    sidecar.write_bytes(sidecar_bytes)
    venv = old / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = old\n")
    (old / ".lease-managed").touch()
    Facts(runtime_facts_path(install)).record_state(
        "venv", "old", [], environment=venv)
    link = install / "node_modules" / "cache-journal"
    link.parent.mkdir()
    link.symlink_to(sidecar)
    assert link.read_bytes() == sidecar_bytes

    try:
        manifest = capture_local_update_capsule(install, home, source, target)
    except ValueError as exc:
        assert "dependency" in str(exc) or "sidecar" in str(exc)
        assert git(install, "rev-parse", "HEAD") == before
        assert link.is_symlink() and link.read_bytes() == sidecar_bytes
        assert not list((home / "local-update-capsules").glob("*/manifest.json"))
        return

    assert sidecar.read_bytes() == sidecar_bytes
    advance(install, source, target)
    new = state / "environments" / "new"
    (new / "venv").mkdir(parents=True)
    (new / "venv" / "pyvenv.cfg").write_text("home = new\n")
    Facts(runtime_facts_path(install)).record_state(
        "venv", "new", [], environment=new / "venv")
    seal_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home)
    assert old in collect_generations(install, min_age_seconds=0)
    assert not old.exists()
    assert link.is_symlink() and not link.exists()
    restored = restore_local_update_capsule(
        manifest, expected_install_root=install, expected_hermes_home=home,
        acknowledge_data_rollback=True)
    assert restored["state"] == json.loads(manifest.read_text())["state"] == "restored"
    assert git(install, "rev-parse", "HEAD") == before
    assert link.is_symlink() and link.read_bytes() == sidecar_bytes
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT value FROM entries").fetchall() == [("captured",)]
