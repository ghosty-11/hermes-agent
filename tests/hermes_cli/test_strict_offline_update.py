"""Offline completion must refuse incomplete migration, recovery, or profile dependencies.

The fixtures own their homes; failure injection is confined to a migration step or
an actual filesystem write/read boundary. Online updates retain best-effort behavior.
"""

import json
import os
import sqlite3
from pathlib import Path

import hermes_yaml as yaml
import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    sibling = root / "profiles" / "work"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root, sibling


def _config(home, **fields):
    from hermes_cli.config import DEFAULT_CONFIG

    path = home / "config.yaml"
    path.write_text(yaml.safe_dump({"_config_version": DEFAULT_CONFIG["_config_version"], **fields}),
                    encoding="utf-8")
    return path


def _jobs(home, jobs):
    path = home / "cron" / "jobs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
    return path


def _snapshot(home):
    from hermes_cli.backup import create_quick_snapshot

    return create_quick_snapshot(label="pre-update", hermes_home=home, keep=1)

def _copy_platform_manifests(project):
    source = Path(__file__).resolve().parents[2] / "plugins" / "platforms"
    for directory in source.iterdir():
        if directory.is_dir() and (directory / "plugin.yaml").is_file():
            target = project / "plugins" / "platforms" / directory.name
            target.mkdir(parents=True)
            (target / "plugin.yaml").write_bytes((directory / "plugin.yaml").read_bytes())


def _selected_site(project):
    from pm.environments import site_packages

    selected = project / "venv"
    selected.mkdir()
    (selected / "pyvenv.cfg").write_text(
        "version = 3.11.0\ninclude-system-site-packages = false\n", encoding="utf-8",
    )
    site = site_packages(selected)
    site.mkdir(parents=True)
    return site


def _dependency_project(tmp_path, extras=(
    "discord", "feishu", "google", "google-chat", "mcp",
)):
    project = tmp_path / "checkout"
    project.mkdir(exist_ok=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0"\n'
        '[project.optional-dependencies]\n'
        + "".join(f'"{extra}" = []\n' for extra in extras),
        encoding="utf-8",
    )
    _copy_platform_manifests(project)
    return project


@pytest.mark.parametrize("target", ["active", "sibling"])
def test_offline_migration_refuses_failed_registered_step_even_when_version_stamped(
    homes, monkeypatch, target,
):
    from hermes_cli import config_migrations, update_cmd
    from hermes_cli.config import DEFAULT_CONFIG

    root, sibling = homes
    version = DEFAULT_CONFIG["_config_version"]
    step_version, real_step = config_migrations.MIGRATIONS[-1]
    _config(root)
    _config(sibling)
    affected = root if target == "active" else sibling

    def failing_step(results, quiet):
        real_step(results, quiet)
        raise OSError("migration step could not finish")

    monkeypatch.setattr(config_migrations, "MIGRATIONS",
                        (*config_migrations.MIGRATIONS[:-1], (step_version, failing_step)))
    config_path = affected / "config.yaml"
    previous = yaml.safe_dump({"_config_version": version - 1})
    config_path.write_text(previous, encoding="utf-8")

    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert yaml.safe_load(config_path.read_text(encoding="utf-8"))["_config_version"] < step_version


@pytest.mark.parametrize("kind", ["cron", "model"])
def test_offline_recovery_refuses_failed_active_restore(homes, monkeypatch, kind):
    from hermes_cli import backup, config as config_module, update_cmd

    root, _ = homes
    config_path = _config(root, model={"provider": "custom", "default": "original-model"})
    if kind == "cron":
        live_path = _jobs(root, [{"id": "saved", "prompt": "important"}])
    else:
        live_path = config_path
    snapshot_id = _snapshot(root)
    assert snapshot_id

    if kind == "cron":
        damaged = json.dumps({"jobs": []})
        real_copy = backup.shutil.copy2

        def deny_restore(source, destination, *args, **kwargs):
            if Path(destination) == live_path:
                raise OSError("cron snapshot cannot be restored")
            return real_copy(source, destination, *args, **kwargs)

        monkeypatch.setattr(backup.shutil, "copy2", deny_restore)
    else:
        damaged = yaml.safe_dump({"_config_version": yaml.safe_load(config_path.read_text())["_config_version"],
                                  "model": {"provider": "other", "default": "replacement-model"}})
        real_write = config_module.atomic_config_write

        def deny_restore(path, values, *args, **kwargs):
            if Path(path) == live_path:
                raise OSError("model snapshot cannot be restored")
            return real_write(path, values, *args, **kwargs)

        monkeypatch.setattr(config_module, "atomic_config_write", deny_restore)

    live_path.write_text(damaged, encoding="utf-8")
    update_cmd._check_and_apply_config_migration(pre_update_snapshot_id=snapshot_id)
    assert live_path.read_text(encoding="utf-8") == damaged

    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(pre_update_snapshot_id=snapshot_id, offline=True)
    assert live_path.read_text(encoding="utf-8") == damaged


def test_offline_recovery_refuses_failed_sibling_restore(homes, monkeypatch):
    from hermes_cli import backup, update_cmd, update_cmd_config

    root, sibling = homes
    _config(root)
    _config(sibling)
    jobs_path = _jobs(sibling, [{"id": "saved", "prompt": "important"}])
    snapshot_id = _snapshot(sibling)
    assert snapshot_id
    jobs_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    monkeypatch.setattr(update_cmd_config, "_LAST_SIBLING_SNAPSHOTS", {"work": snapshot_id})
    real_copy = backup.shutil.copy2

    def deny_sibling_copy(source, destination, *args, **kwargs):
        if Path(destination) == jobs_path:
            raise OSError("sibling cron snapshot cannot be restored")
        return real_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(backup.shutil, "copy2", deny_sibling_copy)
    update_cmd._check_and_apply_config_migration()
    assert json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"] == []
    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"] == []


def test_offline_recovery_refuses_skipped_sibling_probe(homes, monkeypatch):
    from hermes_cli import backup, update_cmd, update_cmd_config

    root, sibling = homes
    _config(root)
    _config(sibling)
    jobs_path = _jobs(sibling, [{"id": "saved"}])
    snapshot_id = _snapshot(sibling)
    assert snapshot_id
    jobs_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    monkeypatch.setattr(update_cmd_config, "_LAST_SIBLING_SNAPSHOTS", {"work": snapshot_id})
    real_count = backup._count_cron_jobs

    def unreadable_jobs(path):
        if path == jobs_path:
            raise OSError("sibling cron check cannot read live jobs")
        return real_count(path)

    monkeypatch.setattr(backup, "_count_cron_jobs", unreadable_jobs)
    update_cmd._check_and_apply_config_migration()
    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"] == []


def test_offline_recovery_refuses_incomplete_sibling_enumeration(homes, monkeypatch):
    from hermes_cli import update_cmd

    root, sibling = homes
    _config(root)
    _config(sibling)
    profiles = root / "profiles"
    real_iterdir = Path.iterdir

    def unreadable_profiles(path):
        if path == profiles:
            raise OSError("profile directory cannot be enumerated")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", unreadable_profiles)
    update_cmd._check_and_apply_config_migration()
    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(offline=True)


def test_offline_configured_dependencies_include_sibling_only_mcp(homes, monkeypatch, tmp_path):
    import pm
    from hermes_cli import main_install_repair
    from pm.plugins_state import dependency_homes

    root, sibling = homes
    _config(root)
    _config(sibling, mcp_servers={"private": {"command": "never-run"}})
    assert set(dependency_homes()) == {root, sibling}
    project = _dependency_project(tmp_path, ("mcp",))
    _selected_site(project)
    requested = main_install_repair.configured_update_extras(project, strict=True)
    assert "mcp" in requested
    syncs = []

    def record_sync(extras, **kwargs):
        syncs.append(extras)
        raise OSError("prepared cache lacks the sibling MCP extra")

    monkeypatch.setattr(pm, "sync_venv", record_sync)
    # The selected Python lacks the sibling's MCP; offline completion must not
    # attempt a late dependency publication after interpreter selection.
    main_install_repair._install_configured_features_missing_deps(project)
    with pytest.raises(RuntimeError, match="mcp"):
        main_install_repair._install_configured_features_missing_deps(project, offline=True)
    assert not syncs


def test_configured_extras_use_each_home_secret_scope(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    sibling_env = sibling / ".env"
    sibling_env.write_text("FEISHU_APP_ID=cli_x\nFEISHU_APP_SECRET=y\n", encoding="utf-8")
    project = _dependency_project(tmp_path, ("feishu",))

    assert "feishu" in configured_update_extras(project, strict=True)
    sibling_env.write_text("", encoding="utf-8")
    assert "feishu" not in configured_update_extras(project, strict=True)


@pytest.mark.parametrize("inspection_fails", [False, True])
def test_dependency_inspection_preserves_invoking_authorization_and_secret_scope(
    homes, monkeypatch, tmp_path, inspection_fails,
):
    from agent.secret_scope import (
        build_profile_secret_scope, current_secret_scope, current_secret_scope_home,
        reset_secret_scope, set_secret_scope,
    )
    from gateway import config_loader
    from hermes_cli.main_install_repair import configured_update_extras
    from hermes_constants import get_hermes_home
    from pm import extras as pm_extras

    root, sibling = homes
    _config(root, gateway={"allow_all_users": False})
    _config(sibling, gateway={"allow_all_users": True},
            platforms={"feishu": {"enabled": True}})
    (root / ".env").write_text("FEISHU_APP_SECRET=invoking-secret\n", encoding="utf-8")
    (sibling / ".env").write_text(
        "FEISHU_APP_ID=cli_sibling\nFEISHU_APP_SECRET=sibling-secret\n", encoding="utf-8",
    )
    project = _dependency_project(tmp_path, ("feishu",))
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setenv("FEISHU_APP_SECRET", "invoking-process-secret")
    monkeypatch.setattr(config_loader, "_BRIDGED_ALLOW_ALL_USERS", False)
    before = {key: os.environ.get(key) for key in
              ("GATEWAY_ALLOW_ALL_USERS", "FEISHU_APP_SECRET", "HERMES_HOME")}
    token = set_secret_scope(build_profile_secret_scope(root), profile_home=str(root))
    invoking_scope = current_secret_scope()
    try:
        if inspection_fails:
            real_supported = pm_extras.extra_supported

            def fail_sibling_dependency_check(name, **kwargs):
                if get_hermes_home() == sibling:
                    raise OSError("sibling dependency check unavailable")
                return real_supported(name, **kwargs)

            monkeypatch.setattr(pm_extras, "extra_supported", fail_sibling_dependency_check)
            with pytest.raises(RuntimeError):
                configured_update_extras(project, strict=True)
        else:
            assert "feishu" in configured_update_extras(project, strict=True)
        assert {key: os.environ.get(key) for key in before} == before
        assert get_hermes_home() == root
        assert current_secret_scope() is invoking_scope
        assert current_secret_scope_home() == str(root)
        assert invoking_scope["FEISHU_APP_SECRET"] == "invoking-secret"
    finally:
        reset_secret_scope(token)
        if before["GATEWAY_ALLOW_ALL_USERS"] is None:
            os.environ.pop("GATEWAY_ALLOW_ALL_USERS", None)
        else:
            os.environ["GATEWAY_ALLOW_ALL_USERS"] = before["GATEWAY_ALLOW_ALL_USERS"]


@pytest.mark.parametrize("listing_fails", [False, True])
def test_offline_plugin_grandfathering_distinguishes_empty_inventory_from_failed_scan(
    homes, monkeypatch, listing_fails,
):
    from hermes_cli import update_cmd
    from hermes_cli.config import DEFAULT_CONFIG

    root, sibling = homes
    config_path = _config(root, _config_version=20)
    _config(sibling)
    plugins = root / "plugins"
    plugins.mkdir()
    if listing_fails:
        existing = plugins / "already-installed"
        existing.mkdir()
        (existing / "plugin.yaml").write_text("name: already-installed\n", encoding="utf-8")
        real_iterdir = Path.iterdir

        def fail_plugin_listing(path):
            if path == plugins:
                raise OSError("installed plugins cannot be enumerated")
            return real_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", fail_plugin_listing)
        with pytest.raises(RuntimeError):
            update_cmd._check_and_apply_config_migration(offline=True)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["_config_version"] == 20
        assert "enabled" not in config.get("plugins", {})
    else:
        update_cmd._check_and_apply_config_migration(offline=True)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["plugins"]["enabled"] == []
        assert config["_config_version"] == DEFAULT_CONFIG["_config_version"]


def test_offline_migration_refuses_failed_profile_soul_inspection(homes, monkeypatch):
    from hermes_cli import update_cmd
    from tools.bot_mode_probe import _PROTOCOL_HEADING

    root, sibling = homes
    config_path = _config(root, _config_version=40)
    _config(sibling)
    soul = sibling / "SOUL.md"
    soul.write_text(f"{_PROTOCOL_HEADING}\nlegacy protocol\n", encoding="utf-8")
    original = config_path.read_bytes()
    original_soul = soul.read_bytes()
    real_read = Path.read_text

    def fail_soul_read(path, *args, **kwargs):
        if path == soul:
            raise OSError("profile soul cannot be read")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_soul_read)
    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert config_path.read_bytes() == original
    assert soul.read_bytes() == original_soul


@pytest.mark.parametrize("failure", ["copy", "held"])
def test_offline_maintenance_refuses_unrestored_corrupt_state(
    homes, monkeypatch, failure,
):
    from hermes_cli import update_cmd, update_cmd_maint
    from hermes_cli.backup import verify_sqlite_integrity

    root, sibling = homes
    _config(root)
    _config(sibling)
    state = root / "state.db"
    snapshot = root / "state-snapshots" / "pre-update" / "state.db"
    snapshot.parent.mkdir(parents=True)
    with sqlite3.connect(snapshot) as conn:
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO sessions (name) VALUES ('saved')")
    assert verify_sqlite_integrity(snapshot, check_header=True, run_pragma=True)["valid"]
    state.write_bytes(b"\x00" * 4096)
    assert not verify_sqlite_integrity(state, check_header=True, run_pragma=True)["valid"]
    monkeypatch.setattr("hermes_cli.backup_restore._foreign_db_holder_pids", lambda path: [])
    monkeypatch.setattr("hermes_cli.model_catalog.seed_cache_from_checkout", lambda path: False)
    monkeypatch.setattr("hermes_cli.update_receipt.record_skip", lambda *args: None)
    if failure == "copy":
        real_copy = update_cmd_maint.shutil.copy2

        def fail_state_copy(source, destination, *args, **kwargs):
            if Path(source) == snapshot and Path(destination) == state:
                raise OSError("snapshot copy failed")
            return real_copy(source, destination, *args, **kwargs)

        monkeypatch.setattr(update_cmd_maint.shutil, "copy2", fail_state_copy)
    else:
        from hermes_cli.sqlite_safe_read import connect_tracked

        held = connect_tracked(state, connect_fn=sqlite3.connect, check_same_thread=False)
    # These independent maintenance tasks cannot affect database recovery.
    monkeypatch.setattr(update_cmd_maint, "_invalidate_live_plugin_catalog_caches", lambda: None)
    monkeypatch.setattr(update_cmd_maint, "_print_bundled_skills_sync_report", lambda: None)
    monkeypatch.setattr(update_cmd_maint, "_sync_profiles_after_update", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_check_and_apply_config_migration", lambda **kw: None)
    monkeypatch.setattr(update_cmd_maint, "_print_post_update_notices_and_self_heals", lambda **kw: None)
    monkeypatch.setattr(update_cmd, "_post_update_sqlite_runtime_status", lambda: (True, object()))
    args = dict(assume_yes=True, gateway_mode=False, pre_update_snapshot_id=None,
                had_desktop_app_before_update=False, pre_update_version=None,
                completion_message="✓ Complete")
    try:
        assert update_cmd._run_post_update_maintenance(**args, offline=False) is True
        with pytest.raises(RuntimeError):
            update_cmd._run_post_update_maintenance(**args, offline=True)
        assert not verify_sqlite_integrity(state, check_header=True, run_pragma=True)["valid"]
        with sqlite3.connect(snapshot) as conn:
            assert conn.execute("SELECT name FROM sessions").fetchone() == ("saved",)
    finally:
        if failure == "held":
            held.close()


@pytest.mark.parametrize("failure", ["malformed", "unreadable"])
def test_strict_dependency_union_refuses_failed_sibling_config(
    homes, tmp_path, monkeypatch, failure,
):
    import builtins

    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    broken = _config(sibling, mcp_servers={"sibling": {"command": "never-run"}})
    project = _dependency_project(tmp_path, ("mcp",))
    assert "mcp" in configured_update_extras(project, strict=True)
    if failure == "malformed":
        broken.write_text("mcp_servers: [unterminated\n", encoding="utf-8")
    else:
        real_open = builtins.open

        def refuse_config_read(path, *args, **kwargs):
            if Path(path) == broken:
                raise OSError("sibling config unreadable")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", refuse_config_read)
    before = broken.read_bytes()
    with pytest.raises(RuntimeError):
        configured_update_extras(project, strict=True)
    assert broken.read_bytes() == before


@pytest.mark.parametrize("failure", ["malformed", "unreadable"])
def test_offline_selected_interpreter_refuses_fallback_config(
    homes, monkeypatch, failure,
):
    import builtins

    from hermes_cli.main_install_repair import _configured_features_missing_deps

    root, sibling = homes
    config_path = _config(root, mcp_servers={"active": {"command": "never-run"}})
    _config(sibling)
    if failure == "malformed":
        config_path.write_text("mcp_servers: [unterminated\n", encoding="utf-8")
    else:
        real_open = builtins.open

        def refuse_config_read(path, *args, **kwargs):
            if Path(path) == config_path:
                raise OSError("active config unreadable")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", refuse_config_read)
    before = config_path.read_bytes()
    with pytest.raises(RuntimeError):
        _configured_features_missing_deps(offline=True)
    assert config_path.read_bytes() == before


def test_strict_dependency_union_accepts_absent_sibling_config(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    (sibling / ".env").write_text("", encoding="utf-8")
    project = _dependency_project(tmp_path, ("mcp",))
    assert configured_update_extras(project, strict=True) == []
    assert not (sibling / "config.yaml").exists()


@pytest.mark.parametrize("broken_path", ["directory", "metadata"])
def test_offline_migration_refuses_invalid_sibling_config_path(
    homes, monkeypatch, broken_path,
):
    from hermes_cli import update_cmd

    root, sibling = homes
    _config(root)
    config_path = sibling / "config.yaml"
    if broken_path == "directory":
        config_path.mkdir()
        original = None
    else:
        _config(sibling)
        original = config_path.read_bytes()
        real_stat = Path.stat

        def fail_config_metadata(path, *args, **kwargs):
            if path == config_path:
                raise OSError("sibling config metadata unavailable")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fail_config_metadata)
    with pytest.raises(Exception):
        update_cmd._check_and_apply_config_migration(offline=True)
    if broken_path == "directory":
        assert config_path.is_dir()
    else:
        assert config_path.read_bytes() == original


def test_offline_migration_refuses_failed_sibling_directory_metadata(
    homes, monkeypatch,
):
    from hermes_cli import update_cmd

    root, sibling = homes
    _config(root)
    config_path = _config(sibling)
    original = config_path.read_bytes()
    real_stat = Path.stat

    def fail_sibling_metadata(path, *args, **kwargs):
        if path == sibling:
            raise OSError("sibling directory metadata unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_sibling_metadata)
    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert config_path.read_bytes() == original


def test_offline_migration_skips_genuinely_absent_sibling_config(homes):
    from hermes_cli import update_cmd

    root, sibling = homes
    _config(root)
    marker = sibling / ".env"
    marker.write_text("FEISHU_APP_SECRET=owned\n", encoding="utf-8")
    update_cmd._check_and_apply_config_migration(offline=True)
    assert marker.read_text(encoding="utf-8") == "FEISHU_APP_SECRET=owned\n"
    assert not (sibling / "config.yaml").exists()


def test_dependency_home_identity_metadata_failure_is_not_an_empty_roster(
    homes, monkeypatch,
):
    from pm.plugins_state import dependency_homes

    root, sibling = homes
    _config(root)
    config_path = _config(sibling)
    assert sibling in dependency_homes()
    real_stat = Path.stat

    def fail_identity_metadata(path, *args, **kwargs):
        if path == config_path:
            raise OSError("profile identity metadata unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fail_identity_metadata)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        dependency_homes()


@pytest.mark.parametrize("snapshot_yaml", ["{}\n", "# no model settings\n"])
def test_offline_recovery_accepts_valid_empty_model_snapshot(
    homes, snapshot_yaml,
):
    from hermes_cli import update_cmd

    root, sibling = homes
    live = _config(root, model={"provider": "custom", "default": "kept-model"})
    _config(sibling)
    snapshot_id = _snapshot(root)
    assert snapshot_id
    snapshot_path = root / "state-snapshots" / snapshot_id / "config.yaml"
    snapshot_path.write_text(snapshot_yaml, encoding="utf-8")
    before = live.read_bytes()
    update_cmd._check_and_apply_config_migration(
        pre_update_snapshot_id=snapshot_id, offline=True,
    )
    assert live.read_bytes() == before
    assert yaml.safe_load(live.read_text(encoding="utf-8"))["model"]["default"] == "kept-model"
    assert snapshot_path.read_text(encoding="utf-8") == snapshot_yaml


def test_offline_recovery_refuses_malformed_model_snapshot(homes):
    from hermes_cli import update_cmd

    root, sibling = homes
    live = _config(root, model={"provider": "custom", "default": "kept-model"})
    _config(sibling)
    snapshot_id = _snapshot(root)
    assert snapshot_id
    snapshot_path = root / "state-snapshots" / snapshot_id / "config.yaml"
    snapshot_path.write_text("model: [unterminated\n", encoding="utf-8")
    before = live.read_bytes()
    with pytest.raises(RuntimeError):
        update_cmd._check_and_apply_config_migration(
            pre_update_snapshot_id=snapshot_id, offline=True,
        )
    assert live.read_bytes() == before
    assert snapshot_path.read_text(encoding="utf-8") == "model: [unterminated\n"


def _private_tree_state(root):
    """Content and metadata, including directory entries and dangling links."""
    entries = [root, *root.rglob("*")]
    result = {}
    for path in entries:
        stat = path.lstat()
        result[str(path.relative_to(root))] = (
            stat.st_mode, stat.st_mtime_ns,
            path.read_bytes() if path.is_file() and not path.is_symlink() else
            os.readlink(path) if path.is_symlink() else None,
        )
    return result


def test_cold_strict_inspection_preserves_all_homes_and_process_state(homes, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root, secrets={"command": {
        "enabled": True, "command": f"printf invoked > {tmp_path / 'secret-source-ran'}",
    }})
    _config(sibling, mcp_servers={"private": {"command": "never-run"}})
    (sibling / ".env").write_text(
        "FEISHU_APP_ID=cli_private\nFEISHU_APP_SECRET=private-test-value\n",
        encoding="utf-8",
    )
    soul = sibling / "SOUL.md"
    soul.write_text("private, owned soul\n", encoding="utf-8")
    project = _dependency_project(tmp_path, ("feishu", "mcp"))
    before_tree = _private_tree_state(root)
    before_env = dict(os.environ)

    for invoking_home in (root, sibling, root):
        token = set_hermes_home_override(invoking_home)
        try:
            with pytest.raises(RuntimeError):
                configured_update_extras(project, strict=True)
        finally:
            reset_hermes_home_override(token)
        assert _private_tree_state(root) == before_tree
        assert dict(os.environ) == before_env
        assert soul.read_text(encoding="utf-8") == "private, owned soul\n"
        assert not (tmp_path / "secret-source-ran").exists()


@pytest.mark.parametrize(
    ("platform", "extra_config", "expected"),
    [
        ("google_chat", {"http_events_url": "https://chat.example.invalid/events"},
         {"google", "google-chat"}),
        ("wecom_callback", {"corp_id": "corp-test", "corp_secret": "private-test"},
         {"wecom"}),
    ],
)
def test_sibling_connector_requires_complete_declared_extras(
    homes, tmp_path, platform, extra_config, expected,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={platform: {"enabled": True, "extra": extra_config}})
    project = _dependency_project(tmp_path, ("google", "google-chat", "wecom"))
    assert expected <= set(configured_update_extras(project, strict=True))


def test_offline_selected_python_refuses_missing_google_sibling_extra(
    homes, tmp_path, monkeypatch,
):
    import pm
    from hermes_cli.main_install_repair import _install_configured_features_missing_deps
    from pm import extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"google_chat": {
        "enabled": True, "extra": {"http_events_url": "https://chat.example.invalid/events"},
    }})
    project = _dependency_project(tmp_path, ("google", "google-chat"))
    selected_site = _selected_site(project)
    (selected_site / "google" / "cloud").mkdir(parents=True)
    (selected_site / "google" / "cloud" / "pubsub_v1.py").write_text("", encoding="utf-8")
    assert extras.passive_available("google-chat", search_paths=(selected_site,))
    assert not extras.passive_available("google", search_paths=(selected_site,))

    def no_late_install(*args, **kwargs):
        pytest.fail("offline selected Python must not publish dependencies")

    monkeypatch.setattr(pm, "sync_venv", no_late_install)
    with pytest.raises(RuntimeError):
        _install_configured_features_missing_deps(project, offline=True)


@pytest.mark.parametrize("failure", ["read", "dangling"])
def test_strict_inspection_refuses_unavailable_sibling_env(
    homes, tmp_path, monkeypatch, failure,
):
    import builtins
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    env = sibling / ".env"
    env.write_text("FEISHU_APP_ID=cli_private\nFEISHU_APP_SECRET=private-test\n",
                   encoding="utf-8")
    project = _dependency_project(tmp_path, ("feishu",))
    assert "feishu" in configured_update_extras(project, strict=True)
    original = env.read_bytes()
    if failure == "dangling":
        env.unlink()
        env.symlink_to(sibling / "missing-private-secrets")
    else:
        real_open = builtins.open
        real_path_open = Path.open

        def deny_env_read(path, *args, **kwargs):
            if Path(path) == env:
                raise PermissionError("private env unreadable")
            return real_open(path, *args, **kwargs)

        def deny_env_path_read(path, *args, **kwargs):
            if path == env:
                raise PermissionError("private env unreadable")
            return real_path_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", deny_env_read)
        monkeypatch.setattr(Path, "open", deny_env_path_read)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)
    if failure == "dangling":
        assert env.is_symlink() and not env.exists()
    else:
        with real_path_open(env, "rb") as handle:
            assert handle.read() == original


def test_strict_inspection_accepts_genuinely_absent_sibling_env(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    project = _dependency_project(tmp_path, ("feishu",))
    assert "feishu" not in configured_update_extras(project, strict=True)
    assert not (sibling / ".env").exists()


@pytest.mark.parametrize("failure", ["read", "malformed"])
def test_strict_inspection_refuses_invalid_configured_platform_manifest(
    homes, tmp_path, monkeypatch, failure,
):
    import builtins
    import io
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    configured = _config(sibling, platforms={"google_chat": {
        "enabled": True, "extra": {"http_events_url": "https://chat.example.invalid/events"},
    }})
    # The actual bundled declaration, not a fake discovery result. The catalog
    # must fail if it cannot account for an explicitly configured connector.
    project = _dependency_project(tmp_path)
    manifest = project / "plugins" / "platforms" / "google_chat" / "plugin.yaml"
    assert manifest.is_file()
    original = configured.read_bytes()
    real_open = builtins.open
    real_path_open = Path.open

    def manifest_content(mode):
        if failure == "read":
            raise PermissionError("configured platform declaration unreadable")
        return io.BytesIO(b"platform_dependencies: [unterminated\n") if "b" in mode else io.StringIO(
            "platform_dependencies: [unterminated\n"
        )

    def deny_manifest_read(path, *args, **kwargs):
        if Path(path) == manifest:
            return manifest_content(args[0] if args else kwargs.get("mode", "r"))
        return real_open(path, *args, **kwargs)

    def deny_manifest_path_read(path, *args, **kwargs):
        if path == manifest:
            return manifest_content(args[0] if args else kwargs.get("mode", "r"))
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_manifest_read)
    monkeypatch.setattr(Path, "open", deny_manifest_path_read)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)
    assert configured.read_bytes() == original


def test_strict_inspection_refuses_invalid_plugin_selection(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    configured = _config(sibling, plugins={"enabled": "google_chat-platform"},
                         platforms={"google_chat": {
                             "enabled": True,
                             "extra": {"http_events_url": "https://chat.example.invalid/events"},
                         }})
    before = configured.read_bytes()
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(_dependency_project(tmp_path), strict=True)
    assert configured.read_bytes() == before


@pytest.mark.parametrize("filename", ["config.yaml", "gateway.json"])
def test_strict_inspection_refuses_dangling_sibling_configuration(
    homes, tmp_path, filename,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    if filename == "gateway.json":
        _config(sibling)
    broken = sibling / filename
    broken.symlink_to(sibling / f"missing-{filename}")
    project = _dependency_project(tmp_path, ("mcp",))
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)
    assert broken.is_symlink() and not broken.exists()


def test_offline_migration_refuses_dangling_sibling_config(homes):
    from hermes_cli import update_cmd

    root, sibling = homes
    _config(root)
    broken = sibling / "config.yaml"
    broken.symlink_to(sibling / "missing-config")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert broken.is_symlink() and not broken.exists()


def test_strict_recovery_refuses_dangling_live_and_snapshot_jobs(homes):
    from hermes_cli.backup import restore_cron_jobs_if_emptied

    root, _ = homes
    live = _jobs(root, [])
    snapshot_id = "private-snapshot"
    saved = root / "state-snapshots" / snapshot_id / "cron" / "jobs.json"
    saved.parent.mkdir(parents=True)
    saved.symlink_to(saved.parent / "unavailable-jobs")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        restore_cron_jobs_if_emptied(snapshot_id, hermes_home=root, strict=True)
    assert live.read_bytes() == b'{"jobs": []}'
    saved.unlink()
    live.unlink()
    live.symlink_to(root / "cron" / "unavailable-live-jobs")
    with pytest.raises((OSError, RuntimeError, ValueError)):
        restore_cron_jobs_if_emptied(snapshot_id, hermes_home=root, strict=True)
    assert live.is_symlink() and not live.exists()
    assert not saved.exists()


def test_strict_recovery_accepts_absent_snapshot_jobs(homes):
    from hermes_cli.backup import restore_cron_jobs_if_emptied

    root, _ = homes
    live = _jobs(root, [])
    assert restore_cron_jobs_if_emptied(
        "absent-snapshot", hermes_home=root, strict=True,
    ) is None
    assert live.read_bytes() == b'{"jobs": []}'


@pytest.mark.parametrize("document", ["[]", "false", "0", '""'])
def test_strict_inspection_refuses_falsey_legacy_gateway_root(
    homes, tmp_path, document,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    legacy = sibling / "gateway.json"
    legacy.write_text(document, encoding="utf-8")
    project = _dependency_project(tmp_path, ("mcp",))
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)
    assert legacy.read_text(encoding="utf-8") == document


def test_strict_inspection_accepts_empty_legacy_gateway_object(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    legacy = sibling / "gateway.json"
    legacy.write_text("{}", encoding="utf-8")
    project = _dependency_project(tmp_path, ("mcp",))
    assert configured_update_extras(project, strict=True) == []
    assert legacy.read_text(encoding="utf-8") == "{}"


def test_cold_dotted_sdk_anchor_does_not_import_its_parent(homes, tmp_path, monkeypatch):
    import sys
    from hermes_cli.main_install_repair import configured_update_extras
    from pm import extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"google_chat": {
        "enabled": True, "extra": {"http_events_url": "https://chat.example.invalid/events"},
    }})
    project = _dependency_project(tmp_path)
    selected_site = _selected_site(project)
    package = selected_site / "strict_inventory_probe_sdk"
    package.mkdir()
    marker = tmp_path / "sdk-parent-executed"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    (package / "flow.py").write_text("VALUE = 1\n", encoding="utf-8")
    (selected_site / "pre_anchor.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "strict_inventory_probe_sdk", raising=False)
    monkeypatch.delitem(sys.modules, "strict_inventory_probe_sdk.flow", raising=False)
    monkeypatch.setitem(extras.ANCHORS, "google", (
        "pre_anchor", "strict_inventory_probe_sdk.flow",
    ))

    try:
        assert {"google", "google-chat"} <= set(configured_update_extras(project, strict=True))
        assert not marker.exists()
        assert "strict_inventory_probe_sdk" not in sys.modules
    finally:
        sys.modules.pop("strict_inventory_probe_sdk.flow", None)
        sys.modules.pop("strict_inventory_probe_sdk", None)


@pytest.mark.parametrize("source", ["user", "project"])
def test_enabled_custom_platform_winner_refuses_without_loading_plugin(
    homes, tmp_path, monkeypatch, source,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    project = _dependency_project(tmp_path)
    if source == "project":
        monkeypatch.chdir(project)
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
        plugin = project / ".hermes" / "plugins" / "platforms" / "google_chat"
    else:
        plugin = sibling / "plugins" / "platforms" / "google_chat"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: google_chat-platform\nkind: platform\nversion: 1.0.0\n"
        "platform_dependencies:\n  google_chat:\n    extras: [private-chat]\n    env: [PRIVATE_CHAT_TOKEN]\n",
        encoding="utf-8",
    )
    marker = tmp_path / "plugin-loaded"
    (plugin / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    _config(sibling, plugins={"enabled": ["platforms/google_chat"]},
            platforms={"google_chat": {"enabled": True}})

    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)
    assert not marker.exists()


def test_enabled_unbundled_platform_refuses_without_importing_it(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    plugin = sibling / "plugins" / "platforms" / "private_chat"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: private-chat-platform\nkind: platform\nversion: 1.0.0\n"
        "platform_dependencies:\n  private_chat:\n"
        "    extras: [private-chat]\n    env: [PRIVATE_CHAT_TOKEN]\n",
        encoding="utf-8",
    )
    marker = tmp_path / "private-plugin-loaded"
    (plugin / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    _config(sibling, plugins={"enabled": ["platforms/private_chat"]},
            platforms={"private_chat": {"enabled": True}})
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(_dependency_project(tmp_path), strict=True)
    assert not marker.exists()


@pytest.mark.parametrize("placement", ["platforms", "top"])
def test_disabled_platform_is_dormant_in_both_config_locations(homes, tmp_path, placement):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    section = ({"platforms": {"google_chat": {"enabled": True}}}
               if placement == "platforms" else {"google_chat": {"enabled": True}})
    _config(sibling, plugins={"disabled": ["platforms/google_chat"]}, **section)
    (sibling / ".env").write_text(
        "GOOGLE_CHAT_HTTP_EVENTS_URL=https://chat.example.invalid/events\n",
        encoding="utf-8",
    )
    assert configured_update_extras(_dependency_project(tmp_path), strict=True) == []


def test_legacy_discord_disable_yields_to_token_but_yaml_disable_wins(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / "gateway.json").write_text(
        '{"platforms": {"discord": {"enabled": false}}}', encoding="utf-8",
    )
    (sibling / ".env").write_text("DISCORD_BOT_TOKEN=fixture-token\n", encoding="utf-8")
    project = _dependency_project(tmp_path)

    assert "discord" in configured_update_extras(project, strict=True)
    _config(sibling, platforms={"discord": {"enabled": False}})
    assert "discord" not in configured_update_extras(project, strict=True)


@pytest.mark.parametrize("platform", ["api_server", "signal", "google_chat-platform"])
def test_core_platforms_and_manifest_alias_are_recognized(homes, tmp_path, platform):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={platform: {
        "enabled": True,
        "extra": {"http_events_url": "https://chat.example.invalid/events"},
    }})
    if platform == "google_chat-platform":
        assert {"google", "google-chat"} <= set(
            configured_update_extras(_dependency_project(tmp_path), strict=True))
    else:
        assert configured_update_extras(_dependency_project(tmp_path), strict=True) == []


@pytest.mark.parametrize(
    ("credentials", "active"),
    [
        ("GOOGLE_APPLICATION_CREDENTIALS=/private/fixture.json\n"
         "GOOGLE_CLOUD_PROJECT=unrelated-project\n", False),
        ("GOOGLE_CHAT_PROJECT_ID=chat-project\n", False),
        ("GOOGLE_CHAT_SUBSCRIPTION_NAME=subscriptions/test\n", False),
        ("GOOGLE_CHAT_HTTP_EVENTS_URL=https://chat.example.invalid/events\n", True),
        ("GOOGLE_CLOUD_PROJECT=chat-project\n"
         "GOOGLE_CHAT_SUBSCRIPTION=subscriptions/test\n", True),
    ],
)
def test_google_chat_requires_inbound_credential_groups(homes, tmp_path, credentials, active):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(credentials, encoding="utf-8")
    found = set(configured_update_extras(_dependency_project(tmp_path), strict=True))
    assert ({"google", "google-chat"} <= found) is active
    if not active:
        assert not found


@pytest.mark.parametrize("credentials", [
    "FEISHU_APP_ID=fixture-app\n",
    "FEISHU_APP_SECRET=fixture-secret\n",
    "FEISHU_APP_ID=fixture-app\nFEISHU_APP_SECRET=fixture-secret\n",
])
def test_feishu_credential_group_needs_both_values(homes, tmp_path, credentials):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(credentials, encoding="utf-8")
    assert ("feishu" in configured_update_extras(_dependency_project(tmp_path), strict=True)) is (
        "FEISHU_APP_ID=" in credentials and "FEISHU_APP_SECRET=" in credentials)


@pytest.mark.parametrize(
    ("publication", "expected"),
    [
        ("cold", "refuse"),
        ("failed", "refuse"),
        ("empty", "empty"),
        ("feishu", "feishu"),
        ("changed", "refuse"),
    ],
)
def test_external_secret_snapshot_authority_is_per_declaration(
    homes, tmp_path, publication, expected,
):
    from hermes_cli import env_loader
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    marker = tmp_path / "source-executions"
    supplied = (
        "FEISHU_APP_ID=fixture-app\\nFEISHU_APP_SECRET=fixture-secret\\n"
        if publication in {"feishu", "changed"} else ""
    )
    result = f"printf '{supplied}'" if publication != "failed" else "exit 9"
    command = f"printf run >> '{marker}'; {result}"
    _config(sibling, secrets={"command": {"enabled": True, "command": command}})
    project = _dependency_project(tmp_path)

    try:
        if publication != "cold":
            env_loader.hydrate_profile_secret_sources(sibling)
            assert marker.read_text(encoding="utf-8") == "run"
        if publication == "changed":
            _config(sibling, secrets={"command": {
                "enabled": True, "command": "printf 'FEISHU_APP_ID=rotated\\n'",
            }})
        if expected == "refuse":
            with pytest.raises((OSError, RuntimeError, ValueError)):
                configured_update_extras(project, strict=True)
        else:
            found = configured_update_extras(project, strict=True)
            assert ("feishu" in found) is (expected == "feishu")
            if expected == "empty":
                assert found == []
        assert (marker.read_text(encoding="utf-8") if marker.exists() else "") == (
            "" if publication == "cold" else "run"
        )
    finally:
        env_loader.reset_secret_source_cache(sibling)


@pytest.mark.parametrize("root_kind", ["managed-link", "managed-metadata", "profiles-link",
                                       "profile-link", "profiles-metadata"])
def test_strict_inventory_refuses_unavailable_directory_roots(
    homes, tmp_path, monkeypatch, root_kind,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    project = _dependency_project(tmp_path)
    profiles = root / "profiles"
    managed = tmp_path / "managed"
    if root_kind.startswith("managed"):
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        if root_kind == "managed-link":
            managed.symlink_to(tmp_path / "missing-managed")
        else:
            managed.mkdir()
            (managed / ".env").write_text(
                "FEISHU_APP_ID=fixture-app\nFEISHU_APP_SECRET=fixture-secret\n",
                encoding="utf-8",
            )
    elif root_kind == "profile-link":
        # A dangling named entry must not silently disappear from the shared roster.
        (sibling / "config.yaml").unlink()
        sibling.rmdir()
        sibling.symlink_to(profiles / "missing-work")
    elif root_kind == "profiles-link":
        (sibling / "config.yaml").unlink()
        sibling.rmdir()
        profiles.rmdir()
        profiles.symlink_to(root / "missing-profiles")

    if root_kind.endswith("metadata"):
        denied = managed if root_kind.startswith("managed") else profiles
        real_stat = Path.stat

        def inaccessible(path, *args, **kwargs):
            if path == denied:
                raise PermissionError("directory metadata unavailable")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", inaccessible)

    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)


def test_offline_migration_refuses_dangling_profiles_root(homes):
    from hermes_cli import update_cmd

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / "config.yaml").unlink()
    sibling.rmdir()
    profiles = root / "profiles"
    profiles.rmdir()
    profiles.symlink_to(root / "missing-profiles")

    with pytest.raises((OSError, RuntimeError, ValueError)):
        update_cmd._check_and_apply_config_migration(offline=True)
    assert profiles.is_symlink()


def test_absent_managed_and_profiles_roots_are_normal_first_run(
    homes, tmp_path, monkeypatch,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    sibling.rmdir()
    (root / "profiles").rmdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "not-created"))
    assert configured_update_extras(_dependency_project(tmp_path), strict=True) == []


@pytest.mark.parametrize("failure", ["missing-manifest", "missing-kind", "directory-metadata"])
def test_env_only_connector_cannot_vanish_from_incomplete_bundled_inventory(
    homes, tmp_path, monkeypatch, failure,
):
    import builtins
    import io
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(
        "FEISHU_APP_ID=fixture-app\nFEISHU_APP_SECRET=fixture-secret\n",
        encoding="utf-8",
    )
    project = _dependency_project(tmp_path)
    plugin = project / "plugins" / "platforms" / "feishu"
    manifest = plugin / "plugin.yaml"
    if failure == "missing-kind":
        replacement = manifest.read_text(encoding="utf-8").replace("kind: platform\n", "")
        real_open = builtins.open
        real_path_open = Path.open

        def altered(path, *args, **kwargs):
            if Path(path) == manifest:
                mode = args[0] if args else kwargs.get("mode", "r")
                return io.BytesIO(replacement.encode()) if "b" in mode else io.StringIO(replacement)
            return real_open(path, *args, **kwargs)

        def altered_path(path, *args, **kwargs):
            if path == manifest:
                mode = args[0] if args else kwargs.get("mode", "r")
                return io.BytesIO(replacement.encode()) if "b" in mode else io.StringIO(replacement)
            return real_path_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", altered)
        monkeypatch.setattr(Path, "open", altered_path)
    else:
        denied = manifest if failure == "missing-manifest" else plugin
        real_stat = Path.stat

        def missing_or_unreadable(path, *args, **kwargs):
            if path == denied:
                if failure == "missing-manifest":
                    raise FileNotFoundError(str(path))
                raise PermissionError("bundled directory metadata unavailable")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", missing_or_unreadable)
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)


def test_requested_runtime_installer_ignores_unrelated_broken_manifest(homes, monkeypatch):
    import builtins
    import io
    import pm
    from pm.platform_requirements import ensure_platform_dependencies

    root, sibling = homes
    _config(root)
    _config(sibling)

    unrelated = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "google_chat" / "plugin.yaml"
    real_open = builtins.open

    def unrelated_malformed(path, *args, **kwargs):
        if Path(path) == unrelated:
            return io.StringIO("kind: [unterminated\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unrelated_malformed)
    # The actual package-manager transaction is outside this test; the installer
    # must still resolve Discord and retain its success contract.
    monkeypatch.setattr(pm, "ensure_import", lambda extra: None)
    manifest = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "discord" / "plugin.yaml"
    assert ensure_platform_dependencies("discord", manifest) is True


def test_disabled_installed_entrypoint_never_invokes_import_finder(homes, tmp_path, monkeypatch):
    import importlib.metadata
    import sys
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root, plugins={"disabled": ["passive_finder_control"]})
    _config(sibling)
    project = _dependency_project(tmp_path)
    selected_site = _selected_site(project)
    dist = selected_site / "passive_finder_control-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: passive-finder-control\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\npassive_finder_control = passive_finder_target:register\n",
        encoding="utf-8",
    )
    assert any(ep.name == "passive_finder_control" for package in
               importlib.metadata.distributions(path=[str(selected_site)])
               for ep in package.entry_points)
    monkeypatch.syspath_prepend(str(selected_site))
    calls = []

    class HostileFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "passive_finder_target":
                calls.append(fullname)
                raise RuntimeError("disabled entry point ran a finder")
            return None

    monkeypatch.setattr(sys, "meta_path", [HostileFinder(), *sys.meta_path])
    assert configured_update_extras(project, strict=True) == []
    assert calls == []


@pytest.mark.parametrize("site_exists", [False, True])
def test_offline_selected_sdk_excludes_caller_and_foreign_development_paths(
    homes, tmp_path, monkeypatch, site_exists,
):
    import sys
    from hermes_cli.main_install_repair import _configured_features_missing_deps
    from pm import extras

    root, sibling = homes
    _config(root, mcp_servers={"private": {"command": "never-run"}})
    _config(sibling)
    project = _dependency_project(tmp_path)
    selected = project / "venv"
    selected.mkdir()
    (selected / "pyvenv.cfg").write_text(
        "version = 3.11.0\ninclude-system-site-packages = false\n", encoding="utf-8",
    )
    from pm.environments import site_packages
    site = site_packages(selected)
    if site_exists:
        site.mkdir(parents=True)
    foreign = tmp_path / "development-path"
    foreign.mkdir()
    (foreign / "passive_foreign_sdk.py").write_text("AVAILABLE = True\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(foreign))
    assert str(foreign) in sys.path
    monkeypatch.setitem(extras.ANCHORS, "mcp", "passive_foreign_sdk")
    if site_exists:
        missing = _configured_features_missing_deps(offline=True, project_root=project)
        assert [name for name, _hint, _extra in missing] == ["mcp"]
    else:
        with pytest.raises((OSError, RuntimeError, ValueError)):
            _configured_features_missing_deps(offline=True, project_root=project)
    if site_exists:
        (site / "passive_foreign_sdk.py").write_text("AVAILABLE = True\n", encoding="utf-8")
        assert _configured_features_missing_deps(offline=True, project_root=project) == []


@pytest.mark.parametrize("shadow", ["regular-package", "module"])
def test_passive_sdk_resolver_respects_first_import_winner(tmp_path, shadow):
    from pm.extras import passive_importable

    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    package = second / "passive_shadow_sdk"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "child.py").write_text("", encoding="utf-8")
    if shadow == "module":
        (first / "passive_shadow_sdk.py").write_text("", encoding="utf-8")
    else:
        earlier = first / "passive_shadow_sdk"
        earlier.mkdir()
        (earlier / "__init__.py").write_text("", encoding="utf-8")
    assert passive_importable("passive_shadow_sdk.child", search_paths=(second,))
    assert not passive_importable("passive_shadow_sdk.child", search_paths=(first, second))


def test_passive_sdk_resolver_combines_genuine_namespace_portions(tmp_path):
    from pm.extras import passive_importable

    first, second = tmp_path / "first", tmp_path / "second"
    (first / "passive_namespace_sdk").mkdir(parents=True)
    (second / "passive_namespace_sdk").mkdir(parents=True)
    (second / "passive_namespace_sdk" / "child.py").write_text("", encoding="utf-8")
    assert not passive_importable("passive_namespace_sdk.child", search_paths=(first,))
    assert passive_importable("passive_namespace_sdk.child", search_paths=(first, second))


@pytest.mark.parametrize("policy", [
    {"preserve_existing": ["FEISHU_APP_ID"]},
    {"profile_alias": False},
    {"preserve_existing": ["FEISHU_APP_ID"], "profile_alias": False},
])
@pytest.mark.parametrize("published", [False, True])
def test_secret_source_policy_is_not_mistaken_for_a_source(
    homes, tmp_path, policy, published,
):
    from hermes_cli import env_loader
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    config = dict(policy)
    if published:
        config["command"] = {
            "enabled": True,
            "command": "printf 'FEISHU_APP_ID=fixture-app\\nFEISHU_APP_SECRET=fixture-secret\\n'",
        }
    _config(sibling, secrets=config)
    project = _dependency_project(tmp_path)
    try:
        if published:
            env_loader.hydrate_profile_secret_sources(sibling)
            assert "feishu" in configured_update_extras(project, strict=True)
            _config(sibling, secrets={**config, "profile_alias": not policy.get("profile_alias", True)})
            with pytest.raises((OSError, RuntimeError, ValueError)):
                configured_update_extras(project, strict=True)
        else:
            assert configured_update_extras(project, strict=True) == []
    finally:
        env_loader.reset_secret_source_cache(sibling)


def _relocated_platform_manifests(tmp_path):
    bundled = tmp_path / "relocated-bundle"
    original = Path(__file__).resolve().parents[2] / "plugins" / "platforms"
    for source in original.iterdir():
        if source.is_dir() and (source / "plugin.yaml").is_file():
            target = bundled / "platforms" / source.name
            target.mkdir(parents=True)
            (target / "plugin.yaml").write_bytes((source / "plugin.yaml").read_bytes())
    return bundled


def test_runtime_bundled_override_owns_strict_platform_declarations(homes, tmp_path, monkeypatch):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(
        "FEISHU_APP_ID=fixture-app\nFEISHU_APP_SECRET=fixture-secret\n", encoding="utf-8",
    )
    project = _dependency_project(tmp_path)
    bundled = _relocated_platform_manifests(tmp_path)
    manifest = bundled / "platforms" / "feishu" / "plugin.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["platform_dependencies"]["feishu"]["extras"] = ["mcp"]
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    assert "mcp" in configured_update_extras(project, strict=True)
    assert "feishu" not in configured_update_extras(project, strict=True)


def test_frozen_target_owns_extra_platform_gates(homes, tmp_path, monkeypatch):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(
        "FEISHU_APP_ID=fixture-app\nFEISHU_APP_SECRET=fixture-secret\n", encoding="utf-8",
    )
    project = _dependency_project(tmp_path)
    with (project / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write(
            'fixture_gate = []\n[tool.hermes.extras-platforms]\n'
            'fixture_gate = "sys_platform == \'never-a-host\'"\n'
        )
    bundled = _relocated_platform_manifests(tmp_path)
    manifest = bundled / "platforms" / "feishu" / "plugin.yaml"
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["platform_dependencies"]["feishu"]["extras"] = ["fixture_gate"]
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    with pytest.raises(RuntimeError):
        configured_update_extras(project, strict=True)


def test_frozen_source_uses_installed_selection_for_gated_sdk(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(
        "FEISHU_APP_ID=fixture-app\nFEISHU_APP_SECRET=fixture-secret\n", encoding="utf-8",
    )
    source = _dependency_project(tmp_path, ("feishu",))
    with (source / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write('[tool.hermes.extras-platforms]\nfeishu = "sys_platform == \'never-a-host\'"\n')
    installed = tmp_path / "installed"
    installed.mkdir()
    site = _selected_site(installed)
    (site / "lark_oapi.py").write_text("SDK_ANCHOR = True\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        configured_update_extras(source, strict=True)
    assert configured_update_extras(
        source, strict=True, environment_root=installed,
    ) == ["feishu"]
    (site / "lark_oapi.py").unlink()
    with pytest.raises(RuntimeError):
        configured_update_extras(source, strict=True, environment_root=installed)


@pytest.mark.parametrize("placement", ["platforms", "top"])
def test_disabled_custom_platform_identity_is_dormant(homes, tmp_path, placement):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    plugin = sibling / "plugins" / "platforms" / "private_chat"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: private-chat-platform\nkind: platform\nversion: 1.0.0\n"
        "platform_dependencies:\n  private_chat:\n"
        "    extras: [mcp]\n    env: [PRIVATE_CHAT_TOKEN]\n",
        encoding="utf-8",
    )
    section = ({"platforms": {"private_chat": {"enabled": True}}}
               if placement == "platforms" else {"private_chat": {"enabled": True}})
    _config(sibling, plugins={"disabled": ["platforms/private_chat"]}, **section)
    (sibling / ".env").write_text("PRIVATE_CHAT_TOKEN=fixture-token\n", encoding="utf-8")
    assert configured_update_extras(_dependency_project(tmp_path), strict=True) == []


@pytest.mark.parametrize(("name", "expected"), [
    (" DISCORD ", {"discord"}),
    (" Google_Chat-platform ", {"google", "google-chat"}),
])
def test_platform_alias_uses_runtime_strip_and_case_normalization(
    homes, tmp_path, name, expected,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={name: {
        "enabled": True, "extra": {"http_events_url": "https://chat.example.invalid/events"},
    }})
    (sibling / ".env").write_text("DISCORD_BOT_TOKEN=fixture-token\n", encoding="utf-8")
    assert expected <= set(configured_update_extras(_dependency_project(tmp_path), strict=True))


@pytest.mark.parametrize("failure", ["malformed", "unreadable"])
def test_explicitly_disabled_photon_does_not_open_invalid_auth(homes, tmp_path, monkeypatch, failure):
    from gateway.platform_configuration import plugin_configured
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"photon": {"enabled": False}})
    auth = sibling / "auth.json"
    auth.write_text("{broken-json\n", encoding="utf-8")
    project = _dependency_project(tmp_path)
    with monkeypatch.context() as patch:
        if failure == "unreadable":
            real_open = Path.open

            def unreadable(path, *args, **kwargs):
                if path == auth:
                    raise PermissionError("synthetic auth file is inaccessible")
                return real_open(path, *args, **kwargs)

            patch.setattr(Path, "open", unreadable)
        assert configured_update_extras(project, strict=True) == []
    auth.write_text(
        '{"credential_pool":{"photon_project":[{"project_id":"fixture",'
        '"project_secret":"fixture-secret"}]}}', encoding="utf-8",
    )
    assert plugin_configured("photon", {}, lambda _name: None, sibling)


def test_launch_home_multiplex_buzz_does_not_use_ambient_credentials(homes, tmp_path):
    from agent.secret_scope import set_multiplex_active
    from pm.platform_requirements import configured_platform_requirements

    root, sibling = homes
    ambient = tmp_path / ".config" / "buzz"
    ambient.mkdir(parents=True)
    (ambient / "fixture-credentials.json").write_text(
        '{"nsec":"synthetic-private-key"}', encoding="utf-8",
    )
    explicit = tmp_path / "private-buzz.json"
    explicit.write_text('{"nsec":"synthetic-private-key"}', encoding="utf-8")
    config = {"platforms": {"buzz": {"extra": {"relay_url": "https://buzz.example.invalid"}}}}
    project = _dependency_project(tmp_path)
    set_multiplex_active(True)
    try:
        assert "buzz" not in configured_platform_requirements(
            config, lambda _name: None, home=root, routed=False,
            source_root=project, selected_paths=(),
        )
        assert "buzz" in configured_platform_requirements(
            config, lambda name: str(explicit) if name == "BUZZ_CREDENTIALS_FILE" else None,
            home=root, routed=False, source_root=project, selected_paths=(),
        )
    finally:
        set_multiplex_active(False)


def test_google_chat_partial_yaml_does_not_join_incomplete_env_seed(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"google_chat": {"enabled": True, "extra": {
        "project_id": "fixture-project",
    }}})
    (sibling / ".env").write_text(
        "GOOGLE_CHAT_SUBSCRIPTION_NAME=subscriptions/fixture\n", encoding="utf-8",
    )
    project = _dependency_project(tmp_path)
    assert not ({"google", "google-chat"} & set(configured_update_extras(project, strict=True)))
    _config(sibling, platforms={"google_chat": {"enabled": True, "extra": {
        "project_id": "fixture-project", "subscription_name": "subscriptions/fixture",
    }}})
    assert {"google", "google-chat"} <= set(configured_update_extras(project, strict=True))


def test_root_discord_typed_token_does_not_become_runtime_credential(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, discord={"enabled": True, "token": "fixture-root-token"})
    project = _dependency_project(tmp_path)
    assert "discord" not in configured_update_extras(project, strict=True)
    (sibling / ".env").write_text("DISCORD_BOT_TOKEN=fixture-env-token\n", encoding="utf-8")
    assert "discord" in configured_update_extras(project, strict=True)


def test_whitespace_relay_does_not_suppress_direct_discord(homes, tmp_path, monkeypatch):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text("DISCORD_BOT_TOKEN=fixture-token\n", encoding="utf-8")
    monkeypatch.setenv("GATEWAY_RELAY_URL", "   ")
    assert "discord" in configured_update_extras(_dependency_project(tmp_path), strict=True)


@pytest.mark.parametrize("failure", ["missing", "malformed"])
def test_matrix_installer_contains_requested_declaration_failure(
    homes, tmp_path, monkeypatch, failure,
):
    import builtins
    import io
    from hermes_cli.main_install_repair import configured_update_extras
    from pm.platform_requirements import required_platform_extras
    from plugins.platforms.matrix.adapter import ensure_matrix_deps

    root, sibling = homes
    _config(root)
    _config(sibling)
    (sibling / ".env").write_text(
        "MATRIX_HOMESERVER=https://matrix.example.invalid\nMATRIX_ACCESS_TOKEN=fixture-token\n",
        encoding="utf-8",
    )
    runtime_manifest = Path(__file__).resolve().parents[2] / "plugins" / "platforms" / "matrix" / "plugin.yaml"
    project = _dependency_project(
        tmp_path, ("discord", "feishu", "google", "google-chat", "mcp", "matrix"),
    )
    manifest = project / "plugins" / "platforms" / "matrix" / "plugin.yaml"
    assert "matrix" in required_platform_extras("matrix", manifest)
    failing_manifests = {manifest, runtime_manifest}
    real_stat = Path.stat
    real_open = builtins.open
    if failure == "missing":
        def broken_stat(path, *args, **kwargs):
            if path in failing_manifests:
                raise FileNotFoundError(str(path))
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", broken_stat)
    else:
        def broken_open(path, *args, **kwargs):
            if Path(path) in failing_manifests:
                return io.StringIO("platform_dependencies: [unterminated\n")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", broken_open)
    with pytest.raises((OSError, ValueError, yaml.YAMLError)):
        required_platform_extras("matrix", manifest)
    assert ensure_matrix_deps() is False
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(project, strict=True)


def test_overlapping_source_fetches_never_certify_a_later_failed_publication(
    homes, tmp_path, monkeypatch,
):
    import threading
    from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, secrets={"passive_race": {"enabled": True}})
    monkeypatch.setenv("FEISHU_APP_ID", "")
    monkeypatch.setenv("FEISHU_APP_SECRET", "")
    first_entered, second_entered = threading.Event(), threading.Event()
    finish_first, finish_second = threading.Event(), threading.Event()
    counter_lock = threading.Lock()

    class ControlledSource(SecretSource):
        name = "passive_race"
        shape = "bulk"

        def __init__(self):
            self.calls = 0

        def fetch(self, cfg, home_path):
            with counter_lock:
                self.calls += 1
                number = self.calls
            if number == 1:
                first_entered.set()
                if not finish_first.wait(3):
                    return FetchResult().fail("first release missing", ErrorKind.TIMEOUT)
                return FetchResult(secrets={
                    "FEISHU_APP_ID": "fixture-app", "FEISHU_APP_SECRET": "fixture-secret",
                })
            second_entered.set()
            if not finish_second.wait(3):
                return FetchResult().fail("second release missing", ErrorKind.TIMEOUT)
            return FetchResult().fail("synthetic fetch failure", ErrorKind.INTERNAL)

    source = ControlledSource()
    prior = snapshot_registration(source.name, scope=str(sibling))
    assert register_source(source, scope=str(sibling))
    project = _dependency_project(tmp_path)
    failures = []

    def fetch():
        try:
            env_loader._apply_external_secret_sources(sibling)
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=fetch)
    second = threading.Thread(target=fetch)
    try:
        first.start()
        assert first_entered.wait(3)
        second.start()
        overlapped = second_entered.wait(1)
        finish_first.set()
        first.join(4)
        assert not first.is_alive()
        assert not failures
        assert "feishu" in configured_update_extras(project, strict=True)
        finish_second.set()
        second.join(4)
        assert not second.is_alive()
        assert not failures
        if overlapped or second_entered.is_set():
            with pytest.raises((OSError, RuntimeError, ValueError)):
                configured_update_extras(project, strict=True)
        else:
            # A serialized second caller can skip a home already successfully applied.
            assert "feishu" in configured_update_extras(project, strict=True)
    finally:
        finish_first.set()
        finish_second.set()
        if first.ident is not None:
            first.join(4)
        if second.ident is not None:
            second.join(4)
        env_loader.reset_secret_source_cache(sibling)
        assert restore_registration(source.name, source, prior, scope=str(sibling))


def test_reset_during_source_fetch_cannot_restore_old_authority(
    homes, tmp_path, monkeypatch,
):
    import threading
    from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, secrets={"passive_reset": {"enabled": True}})
    monkeypatch.setenv("FEISHU_APP_ID", "")
    monkeypatch.setenv("FEISHU_APP_SECRET", "")
    entered, finish, reset_started, reset_done = (
        threading.Event(), threading.Event(), threading.Event(), threading.Event()
    )

    class ControlledSource(SecretSource):
        name = "passive_reset"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            entered.set()
            if not finish.wait(3):
                return FetchResult().fail("release missing", ErrorKind.TIMEOUT)
            return FetchResult(secrets={
                "FEISHU_APP_ID": "fixture-app", "FEISHU_APP_SECRET": "fixture-secret",
            })

    source = ControlledSource()
    prior = snapshot_registration(source.name, scope=str(sibling))
    assert register_source(source, scope=str(sibling))
    failures = []

    def fetch():
        try:
            env_loader._apply_external_secret_sources(sibling)
        except BaseException as exc:
            failures.append(exc)

    def reset():
        reset_started.set()
        try:
            env_loader.reset_secret_source_cache(sibling)
        except BaseException as exc:
            failures.append(exc)
        finally:
            reset_done.set()

    worker = threading.Thread(target=fetch)
    resetter = threading.Thread(target=reset)
    try:
        worker.start()
        assert entered.wait(3)
        resetter.start()
        assert reset_started.wait(3)
        # A lock-serialized reset waits for the fetch; a generation-based reset may finish first.
        reset_done.wait(1)
        finish.set()
        worker.join(4)
        resetter.join(4)
        assert not worker.is_alive() and not resetter.is_alive()
        assert not failures
        with pytest.raises((OSError, RuntimeError, ValueError)):
            configured_update_extras(_dependency_project(tmp_path), strict=True)
    finally:
        finish.set()
        if worker.ident is not None:
            worker.join(4)
        if resetter.ident is not None:
            resetter.join(4)
        env_loader.reset_secret_source_cache(sibling)
        assert restore_registration(source.name, source, prior, scope=str(sibling))

def test_enabled_google_chat_commits_complete_env_seed_over_blank_yaml(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    config = {"platforms": {"google_chat": {"enabled": True, "extra": {
        "project_id": "", "subscription_name": "",
    }}}}
    _config(sibling, **config)
    (sibling / ".env").write_text(
        "GOOGLE_CHAT_PROJECT_ID=fixture-project\n"
        "GOOGLE_CHAT_SUBSCRIPTION_NAME=subscriptions/fixture\n", encoding="utf-8",
    )
    project = _dependency_project(tmp_path)
    assert {"google", "google-chat"} <= set(configured_update_extras(project, strict=True))
    (sibling / ".env").write_text(
        "GOOGLE_CHAT_PROJECT_ID=fixture-project\n", encoding="utf-8",
    )
    assert not ({"google", "google-chat"} & set(configured_update_extras(project, strict=True)))


@pytest.mark.parametrize(
    ("platform", "required", "optional", "field"),
    [
        ("signal", ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
         "SIGNAL_IGNORE_STORIES", "ignore_stories"),
        ("bluebubbles", ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
         "BLUEBUBBLES_SEND_READ_RECEIPTS", "send_read_receipts"),
    ],
)
def test_runtime_credential_defaults_match_passive_seed(
    homes, monkeypatch, platform, required, optional, field,
):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from gateway.config import GatewayConfig, Platform
    from gateway.config_env import _Cred, _ENV_STEPS
    from gateway.platform_configuration import credential_seed

    step = next(s for s in _ENV_STEPS if isinstance(s, _Cred) and s.platform.value == platform)
    for name in (*required, optional):
        monkeypatch.delenv(name, raising=False)
    values = {name: "fixture-value" for name in required}
    for raw, expected in ((None, True), ("", False)):
        scoped = dict(values)
        if raw is not None:
            scoped[optional] = raw
        token = set_secret_scope(scoped, profile_home=str(homes[1]))
        try:
            config = GatewayConfig()
            step(config)
            actual = config.platforms[Platform(platform)].extra[field]
            passive = credential_seed(step, lambda name: scoped.get(name))
            assert actual is expected
            assert passive[field] is actual
        finally:
            reset_secret_scope(token)


@pytest.mark.parametrize("mode", ["environment", "legacy"])
def test_cold_launch_home_buzz_uses_effective_multiplex_mode(
    homes, tmp_path, monkeypatch, mode,
):
    from gateway.config import GatewayConfig
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root, platforms={"buzz": {"extra": {
        "relay_url": "https://buzz.example.invalid",
    }}})
    _config(sibling)
    if mode == "environment":
        monkeypatch.setenv("GATEWAY_MULTIPLEX_PROFILES", "true")
    else:
        monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
        (root / "gateway.json").write_text('{"multiplex_profiles": true}', encoding="utf-8")
    assert GatewayConfig.from_dict(
        {"multiplex_profiles": True} if mode == "legacy" else {}
    ).multiplex_profiles is True
    ambient = tmp_path / ".config" / "buzz"
    ambient.mkdir(parents=True)
    (ambient / "fixture-credentials.json").write_text(
        '{"nsec":"fixture-private-key"}', encoding="utf-8",
    )
    project = _dependency_project(tmp_path, ("buzz",))
    # Buzz ships without extras; this source fixture makes selection visible at the updater boundary.
    manifest = project / "plugins" / "platforms" / "buzz" / "plugin.yaml"
    declaration = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    declaration["platform_dependencies"]["buzz"]["extras"] = ["buzz"]
    manifest.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    assert "buzz" not in configured_update_extras(project, strict=True)
    explicit = tmp_path / "private-buzz.json"
    explicit.write_text('{"nsec":"fixture-private-key"}', encoding="utf-8")
    (root / ".env").write_text(f"BUZZ_CREDENTIALS_FILE={explicit}\n", encoding="utf-8")
    assert "buzz" in configured_update_extras(project, strict=True)


def test_dormant_discord_typed_token_does_not_enable_plugin(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    project = _dependency_project(tmp_path)
    _config(sibling, platforms={"discord": {"token": "fixture-token"}})
    assert "discord" not in configured_update_extras(project, strict=True)
    _config(sibling, platforms={"discord": {"enabled": True, "token": "fixture-token"}})
    assert "discord" in configured_update_extras(project, strict=True)
    _config(sibling, platforms={"discord": {"token": "fixture-token"}})
    (sibling / ".env").write_text("DISCORD_BOT_TOKEN=fixture-env-token\n", encoding="utf-8")
    assert "discord" in configured_update_extras(project, strict=True)


@pytest.mark.parametrize(
    ("yaml_extra", "stored", "connected"),
    [
        ({"project_id": "fixture-project"}, {"project_secret": "fixture-secret"}, False),
        ({"project_id": "fixture-project", "project_secret": "fixture-secret"}, {}, True),
        ({}, {"project_id": "fixture-project", "project_secret": "fixture-secret"}, True),
    ],
)
def test_photon_fallback_requires_a_complete_credential_group(
    homes, tmp_path, monkeypatch, yaml_extra, stored, connected,
):
    from agent.secret_scope import get_secret, reset_secret_scope, set_secret_scope
    from gateway.config import PlatformConfig
    from gateway.config_loader import read_yaml_layers
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from pm.platform_requirements import configured_platform_requirements
    from plugins.platforms.photon.adapter import validate_config

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"photon": {"enabled": True, "extra": yaml_extra}})
    (sibling / "auth.json").write_text(
        json.dumps({"credential_pool": {"photon_project": [stored]}}), encoding="utf-8",
    )
    monkeypatch.delenv("PHOTON_PROJECT_ID", raising=False)
    monkeypatch.delenv("PHOTON_PROJECT_SECRET", raising=False)
    token = set_secret_scope({}, profile_home=str(sibling))
    home_token = set_hermes_home_override(sibling)
    try:
        assert validate_config(PlatformConfig(enabled=True, extra=yaml_extra)) is connected
        project = _dependency_project(tmp_path, ("photon",))
        found = configured_platform_requirements(
            read_yaml_layers(sibling, strict=True), get_secret, home=sibling,
            source_root=project, selected_paths=(), routed=True,
        )
        assert ("photon" in found) is connected
    finally:
        reset_hermes_home_override(home_token)
        reset_secret_scope(token)

@pytest.mark.parametrize("metadata_state", ["absent", "unreadable", "dangling", "directory"])
def test_selected_distribution_metadata_failure_is_not_an_empty_entrypoint_inventory(
    homes, tmp_path, monkeypatch, metadata_state,
):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root, plugins={"disabled": ["fixture-installed-plugin"]})
    _config(sibling)
    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    dist = site / "fixture_installed_plugin-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Name: fixture-installed-plugin\nVersion: 1.0\n", encoding="utf-8",
    )
    entry = dist / "entry_points.txt"
    if metadata_state == "unreadable":
        entry.write_text(
            "[hermes_agent.plugins]\nfixture-installed-plugin = fixture_pkg:register\n",
            encoding="utf-8",
        )
        real_open = Path.open

        def inaccessible(path, *args, **kwargs):
            if path == entry:
                raise PermissionError("selected metadata read denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", inaccessible)
    elif metadata_state == "dangling":
        entry.symlink_to(dist / "unavailable-entry-points.txt")
    elif metadata_state == "directory":
        entry.mkdir()

    if metadata_state == "absent":
        assert configured_update_extras(project, strict=True) == []
    else:
        with pytest.raises((OSError, RuntimeError, ValueError)):
            configured_update_extras(project, strict=True)


@pytest.mark.platforms("linux")
def test_selected_python_extension_shadow_blocks_later_package_child(tmp_path):
    import re
    import sys
    import sysconfig
    from pm.environments import site_packages
    from pm.extras import passive_importable, selected_import_paths

    project = tmp_path / "checkout"
    project.mkdir()
    selected = project / "venv"
    selected.mkdir()
    minor = sys.version_info.minor + 1
    (selected / "pyvenv.cfg").write_text(
        f"version = {sys.version_info.major}.{minor}.0\n"
        "include-system-site-packages = false\n", encoding="utf-8",
    )
    site = site_packages(selected)
    site.mkdir(parents=True)
    extension = sysconfig.get_config_var("EXT_SUFFIX")
    assert extension and re.search(r"cpython-\d+", extension)
    target_suffix = re.sub(
        r"cpython-\d+", f"cpython-{sys.version_info.major}{minor}", extension, count=1,
    )
    (project / f"fixture_shadow_sdk{target_suffix}").write_bytes(b"fixture extension")
    later = site / "fixture_shadow_sdk"
    later.mkdir()
    (later / "__init__.py").write_text("", encoding="utf-8")
    (later / "child.py").write_text("", encoding="utf-8")
    paths = selected_import_paths(project)
    assert passive_importable("fixture_shadow_sdk.child", search_paths=(site,))
    assert not passive_importable("fixture_shadow_sdk.child", search_paths=paths)
    (project / "fixture_shadow_sdk.py").write_text("", encoding="utf-8")
    assert not passive_importable("fixture_shadow_sdk.child", search_paths=paths)


def test_finder_backed_editable_provider_remains_visible_to_ordinary_lookup(
    homes, tmp_path, monkeypatch,
):
    import importlib.metadata
    import importlib.util
    import sys
    from hermes_cli.main_install_repair import configured_update_extras
    from plugins.memory import find_provider_dir

    root, sibling = homes
    _config(root, plugins={"disabled": ["fixture_editable_provider"]})
    _config(sibling)
    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    dist = site / "fixture_editable_provider-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Name: fixture-editable-provider\nVersion: 1.0\n", encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.memory_providers]\n"
        "fixture_editable_provider = fixture_editable_pkg:register\n"
        "[hermes_agent.plugins]\n"
        "fixture_editable_provider = fixture_editable_pkg:register\n",
        encoding="utf-8",
    )
    package = tmp_path / "editable-source" / "fixture_editable_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def register_memory_provider():\n    pass\n", encoding="utf-8",
    )
    (package / "config_schema.py").write_text("CONFIG_SCHEMA = {}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    assert any(ep.name == "fixture_editable_provider" for dist_entry in
               importlib.metadata.distributions(path=[str(site)])
               for ep in dist_entry.entry_points)
    calls = []

    class EditableFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "fixture_editable_pkg":
                return None
            calls.append(fullname)
            return importlib.util.spec_from_file_location(
                fullname, package / "__init__.py", submodule_search_locations=[str(package)],
            )

    monkeypatch.setattr(sys, "meta_path", [EditableFinder(), *sys.meta_path])
    assert configured_update_extras(project, strict=True) == []
    assert calls == []
    assert find_provider_dir("fixture_editable_provider") == package
    assert calls == ["fixture_editable_pkg"]
    assert "fixture_editable_pkg" not in sys.modules


def test_fresh_strict_reader_leaves_private_home_and_environment_unchanged(tmp_path):
    import subprocess
    import sys

    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "config.yaml").write_text("plugins:\n  enabled: []\n", encoding="utf-8")
    (root / ".env").write_text("# private, empty fixture\n", encoding="utf-8")
    project = _dependency_project(tmp_path)
    before = _private_tree_state(root)
    env = dict(os.environ)
    env.update(HOME=str(tmp_path), HERMES_HOME=str(root))
    child = (
        "import json, os, sys\n"
        "before = dict(os.environ)\n"
        "calls = []\n"
        "class SDKGuard:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'lark_oapi' or fullname.startswith('lark_oapi.'):\n"
        "            calls.append(fullname)\n"
        "            raise AssertionError('optional SDK activated')\n"
        "        return None\n"
        "sys.meta_path.insert(0, SDKGuard())\n"
        "from hermes_cli.main_install_repair import configured_update_extras\n"
        "extras = configured_update_extras(__import__('pathlib').Path(sys.argv[1]), strict=True)\n"
        "print(json.dumps({'extras': extras, 'env_unchanged': dict(os.environ) == before,"
        " 'sdk_calls': calls}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child, str(project)],
        cwd=Path(__file__).resolve().parents[2], env=env, capture_output=True,
        text=True, timeout=25, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output == {"extras": [], "env_unchanged": True, "sdk_calls": []}
    assert _private_tree_state(root) == before

@pytest.mark.parametrize("failure", ["config", "apply", "empty-report"])
@pytest.mark.parametrize("routed", [False, True])
def test_later_prepublication_failure_invalidates_completed_source_publication(
    homes, monkeypatch, routed, failure,
):
    import threading
    from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
    from agent.secret_sources import registry
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[1] if routed else homes[0]
    config_path = _config(home, secrets={"passive_config_race": {"enabled": True}})
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    first_fetching, finish_first, second_reading, finish_second = (
        threading.Event() for _ in range(4)
    )

    class ControlledSource(SecretSource):
        name = "passive_config_race"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            first_fetching.set()
            if not finish_first.wait(3):
                return FetchResult().fail("first attempt not released", ErrorKind.TIMEOUT)
            return FetchResult(secrets={"FEISHU_APP_ID": "fixture-app"})

    source = ControlledSource()
    prior = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    real_read = env_loader._load_secrets_config

    def blocked_read(path):
        if threading.current_thread().name == "passive-second-reader":
            second_reading.set()
            if not finish_second.wait(3):
                raise OSError("second reader not released")
        return real_read(path)

    monkeypatch.setattr(env_loader, "_load_secrets_config", blocked_read)
    real_apply = registry.apply_all

    def second_apply(*args, **kwargs):
        if threading.current_thread().name == "passive-second-reader":
            if failure == "apply":
                raise OSError("source orchestration failed")
            if failure == "empty-report":
                return registry.ApplyReport()
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(registry, "apply_all", second_apply)
    fetch = (env_loader.hydrate_profile_secret_sources if routed else
             env_loader._apply_external_secret_sources)
    failures = []

    def invoke():
        try:
            fetch(home)
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=invoke, name="passive-first-reader")
    second = threading.Thread(target=invoke, name="passive-second-reader")
    original = config_path.read_bytes()
    try:
        first.start()
        assert first_fetching.wait(3)
        second.start()
        assert second_reading.wait(3)
        finish_first.set()
        first.join(4)
        assert not first.is_alive() and not failures
        assert env_loader.get_authoritative_secret_source_values(home)["FEISHU_APP_ID"] == (
            "fixture-app"
        )
        if failure == "config":
            config_path.write_text("secrets: [unclosed\n", encoding="utf-8")
        finish_second.set()
        second.join(4)
        assert not second.is_alive() and not failures
        config_path.write_bytes(original)
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
    finally:
        finish_first.set()
        finish_second.set()
        if first.ident is not None:
            first.join(4)
        if second.ident is not None:
            second.join(4)
        config_path.write_bytes(original)
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, prior, scope=str(home))


@pytest.mark.parametrize("routed", [False, True])
def test_failed_replacement_preserves_route_specific_retry_policy(homes, monkeypatch, routed):
    import threading
    from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[1] if routed else homes[0]
    _config(home, secrets={"passive_retry_race": {"enabled": True}})
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    lock = threading.Lock()

    class ControlledSource(SecretSource):
        name = "passive_retry_race"
        shape = "bulk"

        def __init__(self):
            self.calls = 0

        def fetch(self, cfg, home_path):
            with lock:
                index = self.calls
                self.calls += 1
            if index < 2:
                entered[index].set()
                if not release[index].wait(3):
                    return FetchResult().fail("attempt not released", ErrorKind.TIMEOUT)
            if index == 1:
                return FetchResult().fail("replacement failed", ErrorKind.INTERNAL)
            return FetchResult(secrets={"FEISHU_APP_ID": f"fixture-{index}"})

    source = ControlledSource()
    prior = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    failures = []
    fetch = (env_loader.hydrate_profile_secret_sources if routed else
             env_loader._apply_external_secret_sources)

    def invoke():
        try:
            fetch(home)
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    try:
        first.start()
        assert entered[0].wait(3)
        second.start()
        assert entered[1].wait(3)
        release[0].set()
        first.join(4)
        assert not first.is_alive() and not failures
        assert env_loader.get_authoritative_secret_source_values(home)["FEISHU_APP_ID"] == (
            "fixture-0"
        )
        release[1].set()
        second.join(4)
        assert not second.is_alive() and not failures
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
        if routed:
            assert fetch(home)["FEISHU_APP_ID"] == "fixture-2"
            assert env_loader.get_authoritative_secret_source_values(home)["FEISHU_APP_ID"] == (
                "fixture-2"
            )
            assert source.calls == 3
        else:
            fetch(home)
            assert source.calls == 2
            with pytest.raises((OSError, RuntimeError, ValueError)):
                env_loader.get_authoritative_secret_source_values(home)
    finally:
        for event in release:
            event.set()
        if first.ident is not None:
            first.join(4)
        if second.ident is not None:
            second.join(4)
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, prior, scope=str(home))


def test_disabled_dotted_entrypoint_discovery_does_not_import_parent(homes, tmp_path, monkeypatch):
    import importlib.metadata
    import sys
    from hermes_cli.main_install_repair import configured_update_extras
    from hermes_cli.plugins import PluginManager
    from plugins.memory import find_provider_dir

    root, _ = homes
    _config(root, plugins={"disabled": ["fixture_dotted_provider"]})
    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    dist = site / "fixture_dotted_provider-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Name: fixture-dotted-provider\nVersion: 1.0\n", encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\n"
        "fixture_dotted_provider = disabled_package.adapter:register\n"
        "[hermes_agent.memory_providers]\n"
        "fixture_dotted_provider = disabled_package.adapter:register\n", encoding="utf-8",
    )
    package = site / "disabled_package"
    package.mkdir()
    marker = tmp_path / "parent-imported"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    (package / "adapter.py").write_text(
        "def register_memory_provider():\n    pass\n", encoding="utf-8",
    )
    assert any(ep.name == "fixture_dotted_provider" for installed in
               importlib.metadata.distributions(path=[str(site)])
               for ep in installed.entry_points)
    monkeypatch.syspath_prepend(str(site))
    class HostileFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.startswith("disabled_package"):
                raise AssertionError("strict inspection called an import finder")
            return None

    with monkeypatch.context() as guard:
        guard.setattr(sys, "meta_path", [HostileFinder(), *sys.meta_path])
        assert configured_update_extras(project, strict=True) == []
    assert not marker.exists()
    try:
        PluginManager().discover_and_load()
        assert not marker.exists()
        assert "disabled_package" not in sys.modules
        # Ordinary finder-backed lookup still resolves the dotted module itself.
        from hermes_cli.plugins import resolve_module_origin
        assert resolve_module_origin("disabled_package.adapter") == str(package / "adapter.py")
        assert not marker.exists()
    finally:
        sys.modules.pop("disabled_package.adapter", None)
        sys.modules.pop("disabled_package", None)


@pytest.mark.parametrize("failure", ["config", "apply", "empty-report"])
@pytest.mark.parametrize("routed", [False, True])
def test_older_fetch_failure_completing_after_success_replaces_authority(
    homes, monkeypatch, failure, routed,
):
    import threading
    from agent.secret_sources.base import FetchResult, SecretSource
    from agent.secret_sources import registry
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[1] if routed else homes[0]
    _config(home, secrets={"completion_order": {"enabled": True}})
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    reading, release = threading.Event(), threading.Event()
    real_read = env_loader._load_secrets_config

    def delayed_first_read(path):
        if threading.current_thread().name == "older-secret-fetch":
            reading.set()
            if not release.wait(3):
                raise OSError("older fetch not released")
            if failure == "config":
                raise OSError("older config read failed")
        return real_read(path)

    class Source(SecretSource):
        name = "completion_order"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={"FEISHU_APP_ID": "fixture-new"})

    source = Source()
    previous = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    monkeypatch.setattr(env_loader, "_load_secrets_config", delayed_first_read)
    real_apply = registry.apply_all

    def older_apply(*args, **kwargs):
        if threading.current_thread().name == "older-secret-fetch":
            if failure == "apply":
                raise OSError("older apply failed")
            if failure == "empty-report":
                return registry.ApplyReport()
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(registry, "apply_all", older_apply)
    fetch = (env_loader.hydrate_profile_secret_sources if routed else
             env_loader._apply_external_secret_sources)
    errors = []

    def run_older():
        try:
            fetch(home)
        except BaseException as exc:
            errors.append(exc)

    older = threading.Thread(target=run_older, name="older-secret-fetch")
    try:
        older.start()
        assert reading.wait(3)
        fetch(home)
        assert env_loader.get_authoritative_secret_source_values(home)["FEISHU_APP_ID"] == "fixture-new"
        release.set()
        older.join(4)
        assert not older.is_alive() and not errors
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
    finally:
        release.set()
        if older.ident is not None:
            older.join(4)
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, previous, scope=str(home))


def test_reset_fences_old_fetch_from_process_credentials_after_replacement(
    homes, monkeypatch,
):
    import threading
    from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[0]
    _config(home, secrets={"rotating_fixture": {"enabled": True, "override_existing": True}})
    monkeypatch.delenv("FEISHU_APP_ID", raising=False)
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()

    class RotatingSource(SecretSource):
        name = "rotating_fixture"
        shape = "bulk"

        def __init__(self):
            self.calls = 0

        def fetch(self, cfg, home_path):
            with lock:
                self.calls += 1
                call = self.calls
            if call == 1:
                entered.set()
                if not release.wait(3):
                    return FetchResult().fail("old fetch not released", ErrorKind.TIMEOUT)
            return FetchResult(secrets={"FEISHU_APP_ID": "fixture-old" if call == 1 else "fixture-new"})

    source = RotatingSource()
    previous = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    errors = []

    def run_old():
        try:
            env_loader._apply_external_secret_sources(home)
        except BaseException as exc:
            errors.append(exc)

    old = threading.Thread(target=run_old)
    try:
        old.start()
        assert entered.wait(3)
        env_loader.reset_secret_source_cache(home)
        env_loader._apply_external_secret_sources(home)
        assert env_loader.get_authoritative_secret_source_values(home)["FEISHU_APP_ID"] == "fixture-new"
        assert os.environ["FEISHU_APP_ID"] == "fixture-new"
        release.set()
        old.join(4)
        assert not old.is_alive() and not errors
        assert env_loader.get_authoritative_secret_source_values(home)["FEISHU_APP_ID"] == "fixture-new"
        assert os.environ["FEISHU_APP_ID"] == "fixture-new"
        env_loader.reset_secret_source_cache(home)
        source.fetch = lambda cfg, home_path: FetchResult().fail("offline", ErrorKind.NETWORK)
        env_loader._apply_external_secret_sources(home)
        assert os.environ["FEISHU_APP_ID"] == "fixture-new"
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
    finally:
        release.set()
        if old.ident is not None:
            old.join(4)
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, previous, scope=str(home))


@pytest.mark.parametrize(("override", "preserve", "expected_source"), [
    (False, False, False), (True, True, False), (True, False, True),
])
def test_dotenv_reload_during_source_fetch_owns_commit_precedence(
    homes, monkeypatch, override, preserve, expected_source,
):
    import threading
    from agent.secret_sources.base import FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[0]
    key = "FIXTURE_RELOAD_TOKEN"
    cfg = {"enabled": True, "override_existing": override}
    secrets = {"reload_fixture": cfg}
    if preserve:
        secrets["preserve_existing"] = [key]
    _config(home, secrets=secrets)
    monkeypatch.delenv(key, raising=False)
    entered, release = threading.Event(), threading.Event()

    class Source(SecretSource):
        name = "reload_fixture"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("fixture fetch was not released")
            return FetchResult(secrets={key: "fixture-source"})

    source = Source()
    prior = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    errors = []

    def fetch():
        try:
            env_loader._apply_external_secret_sources(home)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=fetch)
    try:
        worker.start()
        assert entered.wait(3)
        dotenv = home / ".env"
        dotenv.write_text(f"{key}=fixture-user\n", encoding="utf-8")
        env_loader._load_dotenv_with_fallback(dotenv, override=True)
        release.set()
        worker.join(4)
        assert not worker.is_alive() and not errors
        assert os.environ[key] == ("fixture-source" if expected_source else "fixture-user")
        assert env_loader.get_secret_source(key) == (source.name if expected_source else None)
        assert env_loader.get_authoritative_secret_source_values(home)[key] == os.environ[key]
        assert (key in env_loader._SECRET_SOURCE_RESTORE_BY_HOME.get(str(home.resolve()), {})) is (
            expected_source
        )
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(4)
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, prior, scope=str(home))


def test_invalid_secret_value_rejects_whole_commit_and_keeps_startup_fail_open(
    homes, monkeypatch,
):
    from agent.secret_sources.base import FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[0]
    _config(home, secrets={"invalid_fixture": {"enabled": True}})
    good, bad = "FIXTURE_GOOD_TOKEN", "FIXTURE_BAD_TOKEN"
    monkeypatch.delenv(good, raising=False)
    monkeypatch.delenv(bad, raising=False)

    class Source(SecretSource):
        name = "invalid_fixture"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={good: "fixture-good", bad: "fixture\x00bad"})

    source = Source()
    prior = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    try:
        env_loader.load_hermes_dotenv(hermes_home=home)
        assert good not in os.environ and bad not in os.environ
        assert env_loader.get_secret_source(good) is None
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
        assert str(home.resolve()) not in env_loader._APPLIED_HOMES
    finally:
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, prior, scope=str(home))


def test_source_provenance_is_removed_when_new_dotenv_value_wins(homes, monkeypatch):
    from agent.secret_sources.base import FetchResult, SecretSource
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = homes[0]
    key = "FIXTURE_RELOAD_OWNERSHIP_TOKEN"
    _config(home, secrets={"ownership_fixture": {"enabled": True}})
    monkeypatch.delenv(key, raising=False)

    class Source(SecretSource):
        name = "ownership_fixture"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={key: "fixture-source"})

    source = Source()
    prior = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    try:
        env_loader._apply_external_secret_sources(home)
        assert os.environ[key] == "fixture-source"
        assert env_loader.get_secret_source(key) == source.name
        env_loader.reset_secret_source_cache(home)
        dotenv = home / ".env"
        dotenv.write_text(f"{key}=fixture-user\n", encoding="utf-8")
        env_loader._load_dotenv_with_fallback(dotenv, override=True)
        env_loader._apply_external_secret_sources(home)
        assert os.environ[key] == "fixture-user"
        assert env_loader.get_secret_source(key) is None
        assert env_loader.get_authoritative_secret_source_values(home)[key] == "fixture-user"
    finally:
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, prior, scope=str(home))


@pytest.mark.parametrize("consumer", ["distributions", "pth"])
def test_selected_site_listing_failure_refuses_instead_of_empty_inventory(
    homes, tmp_path, monkeypatch, consumer,
):
    from hermes_cli.main_install_repair import configured_update_extras
    from pm.extras import selected_import_paths

    root, sibling = homes
    _config(root)
    _config(sibling)
    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    assert site.is_dir()
    if consumer == "distributions":
        dist = site / "fixture_installed-1.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Name: fixture-installed\nVersion: 1.0\n", encoding="utf-8")
        (dist / "entry_points.txt").write_text(
            "[hermes_agent.plugins]\nfixture_installed = fixture_pkg:register\n", encoding="utf-8",
        )
        from hermes_cli.plugins_discovery import discover_entrypoint_manifests
        assert any(m.name == "fixture_installed" for m in
                   discover_entrypoint_manifests(strict=True, selected_paths=(site,)))
    else:
        extra_path = site / "editable-fixture"
        extra_path.mkdir()
        (site / "fixture.pth").write_text("editable-fixture\n", encoding="utf-8")
        assert extra_path in selected_import_paths(project)
    if consumer == "pth":
        real_listdir = os.listdir

        def listing(path):
            if Path(path) == site:
                raise PermissionError("selected .pth listing denied")
            return real_listdir(path)

        monkeypatch.setattr(os, "listdir", listing)
        with pytest.raises(PermissionError, match="selected .pth listing denied"):
            selected_import_paths(project)
    else:
        assert configured_update_extras(project, strict=True) == []
        real_scandir = os.scandir

        def listing(path):
            if Path(path) == site:
                raise PermissionError("selected distributions listing denied")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", listing)
        # Strict inspection wraps per-home failures; the listing error must be the cause.
        with pytest.raises(RuntimeError) as refused:
            configured_update_extras(project, strict=True)
        cause = refused.value
        while cause is not None and not isinstance(cause, PermissionError):
            cause = cause.__cause__
        assert cause is not None and "selected distributions listing denied" in str(cause)

def _standard_editable_pth(site, project, *, mapping_value=None, namespace_value=None,
                           mapping_key="fixture_editable_pkg",
                           placeholder="'__editable__.fixture-0.0.0.finder' + '.__path_hook__'",
                           extra_statement=""):
    module = "__editable___fixture_0_0_0_finder"
    workspace = project / "workspace"
    finder = site / f"{module}.py"
    finder.write_text(
        "from __future__ import annotations\n"
        "import sys\n"
        "from importlib.machinery import ModuleSpec, PathFinder\n"
        "from importlib.machinery import all_suffixes as module_suffixes\n"
        "from importlib.util import spec_from_file_location\n"
        "from itertools import chain\n"
        "from pathlib import Path\n\n"
        f"MAPPING: dict[str, str] = {{{mapping_key!r}: "
        f"{mapping_value if mapping_value is not None else repr(str(workspace / mapping_key))}}}\n"
        f"NAMESPACES: dict[str, list[str]] = "
        f"{{'fixture_namespace': [{namespace_value if namespace_value is not None else repr(str(workspace / 'fixture_namespace'))}]}}\n"
        f"PATH_PLACEHOLDER = {placeholder}\n\n"
        "class _EditableFinder:\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname: str, path=None, target=None) -> ModuleSpec | None:\n"
        "        if fullname in MAPPING:\n"
        "            pkg_path = MAPPING[fullname]\n"
        "            return cls._find_spec(fullname, Path(pkg_path))\n"
        "        parent, _, child = fullname.rpartition('.')\n"
        "        if parent and parent in MAPPING:\n"
        "            return PathFinder.find_spec(fullname, path=[MAPPING[parent]])\n"
        "        return None\n\n"
        "    @classmethod\n"
        "    def _find_spec(cls, fullname: str, candidate_path: Path) -> ModuleSpec | None:\n"
        "        init = candidate_path / '__init__.py'\n"
        "        candidates = (candidate_path.with_suffix(x) for x in module_suffixes())\n"
        "        for candidate in chain([init], candidates):\n"
        "            if candidate.exists():\n"
        "                return spec_from_file_location(fullname, candidate)\n"
        "        return None\n\n"
        "class _EditableNamespaceFinder:\n"
        "    @classmethod\n"
        "    def _path_hook(cls, path) -> type[_EditableNamespaceFinder]:\n"
        "        if path == PATH_PLACEHOLDER:\n"
        "            return cls\n"
        "        raise ImportError\n\n"
        "    @classmethod\n"
        "    def _paths(cls, fullname: str) -> list[str]:\n"
        "        paths = NAMESPACES[fullname]\n"
        "        if not paths and fullname in MAPPING:\n"
        "            paths = [MAPPING[fullname]]\n"
        "        return [*paths, PATH_PLACEHOLDER]\n\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname: str, target=None) -> ModuleSpec | None:\n"
        "        if fullname in NAMESPACES:\n"
        "            spec = ModuleSpec(fullname, None, is_package=True)\n"
        "            spec.submodule_search_locations = cls._paths(fullname)\n"
        "            return spec\n"
        "        return None\n\n"
        "    @classmethod\n"
        "    def find_module(cls, _fullname) -> None:\n"
        "        return None\n\n"
        "def install():\n"
        "    if not any(finder == _EditableFinder for finder in sys.meta_path):\n"
        "        sys.meta_path.append(_EditableFinder)\n"
        "    if not NAMESPACES:\n"
        "        return\n"
        "    if not any(hook == _EditableNamespaceFinder._path_hook for hook in sys.path_hooks):\n"
        "        sys.path_hooks.append(_EditableNamespaceFinder._path_hook)\n"
        "    if PATH_PLACEHOLDER not in sys.path:\n"
        "        sys.path.append(PATH_PLACEHOLDER)\n"
        f"{extra_statement}",
        encoding="utf-8",
    )
    pth = site / "__editable__.fixture-0.0.0.pth"
    pth.write_text(f"import {module}; {module}.install()\n", encoding="utf-8")
    return finder


def test_selected_site_accepts_standard_virtualenv_and_editable_finder(tmp_path, monkeypatch):
    import hashlib
    from pm import extras

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    root_package = project / "fixture_editable_pkg"
    root_package.mkdir()
    (root_package / "__init__.py").write_text("", encoding="utf-8")
    (root_package / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    root_namespace = project / "fixture_namespace"
    (root_namespace / "nested").mkdir(parents=True)
    (root_namespace / "nested" / "shared.py").write_text("VALUE = 1\n", encoding="utf-8")
    workspace = project / "workspace"
    package = workspace / "fixture_editable_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "shared.py").write_text("VALUE = 2\n", encoding="utf-8")
    namespace = workspace / "fixture_namespace"
    (namespace / "nested").mkdir(parents=True)
    (namespace / "nested" / "shared.py").write_text("VALUE = 2\n", encoding="utf-8")
    shim_bytes = b"# distutils shim\n"
    (site / "_virtualenv.py").write_bytes(shim_bytes)
    (site / "_virtualenv.pth").write_text("# uv shim\nimport _virtualenv\n", encoding="utf-8")
    monkeypatch.setattr(extras, "_CERTIFIED_VIRTUALENV_HASHES",
                        {hashlib.sha256(shim_bytes).hexdigest()})
    _standard_editable_pth(site, project)

    paths = extras.selected_import_paths(project)
    assert extras.passive_module_origin("fixture_editable_pkg.shared",
                                       search_paths=paths) == str(root_package / "shared.py")
    assert extras.passive_module_origin("fixture_namespace.nested.shared",
                                       search_paths=paths) == str(root_namespace / "nested" / "shared.py")
    assert paths == (project.resolve(), site)


def test_selected_site_does_not_expose_unmapped_workspace_module(tmp_path):
    from pm.extras import passive_module_origin, selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    package = project / "fixture_editable_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    mapped = workspace / "fixture_editable_pkg"
    mapped.mkdir(parents=True)
    (mapped / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "fixture_namespace").mkdir()
    (workspace / "unmapped_editable_only.py").write_text("", encoding="utf-8")
    _standard_editable_pth(site, project)

    assert passive_module_origin("unmapped_editable_only",
                                 search_paths=selected_import_paths(project)) is None


def test_selected_site_refuses_project_namespace_for_mapped_regular_package(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (project / "fixture_editable_pkg").mkdir()
    (project / "fixture_namespace").mkdir()
    package = project / "workspace" / "fixture_editable_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "workspace" / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project)

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_refuses_mapped_package_with_missing_root_child(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    package = project / "fixture_editable_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    mapped = workspace / "fixture_editable_pkg"
    mapped.mkdir(parents=True)
    (mapped / "__init__.py").write_text("", encoding="utf-8")
    (mapped / "workspace_only.py").write_text("", encoding="utf-8")
    (workspace / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project)

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_refuses_namespace_with_missing_root_descendant(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    package = project / "fixture_editable_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace" / "nested").mkdir(parents=True)
    workspace = project / "workspace"
    mapped = workspace / "fixture_editable_pkg"
    mapped.mkdir(parents=True)
    (mapped / "__init__.py").write_text("", encoding="utf-8")
    nested = workspace / "fixture_namespace" / "nested"
    nested.mkdir(parents=True)
    (nested / "workspace_only.py").write_text("", encoding="utf-8")
    _standard_editable_pth(site, project)

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def _editable_mirror_layout(tmp_path):
    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    package = project / "fixture_editable_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    mapped = workspace / "fixture_editable_pkg"
    mapped.mkdir(parents=True)
    (mapped / "__init__.py").write_text("", encoding="utf-8")
    namespace = workspace / "fixture_namespace"
    namespace.mkdir()
    _standard_editable_pth(site, project)
    return project, site, package, mapped, namespace


@pytest.mark.parametrize("root_child", ["directory", "dangling-symlink"])
def test_selected_site_refuses_mapped_module_with_incompatible_root_child(
    tmp_path, root_child,
):
    from pm.extras import selected_import_paths

    project, _, package, mapped, _ = _editable_mirror_layout(tmp_path)
    (mapped / "child.py").write_text("", encoding="utf-8")
    child = package / "child.py"
    if root_child == "directory":
        child.mkdir()
    else:
        child.symlink_to(package / "missing.py")

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


@pytest.mark.parametrize("location", ["namespace", "mapped-package"])
def test_selected_site_refuses_importable_workspace_cache_without_root_mirror(
    tmp_path, location,
):
    from pm.extras import selected_import_paths

    project, _, _, mapped, namespace = _editable_mirror_layout(tmp_path)
    cache = (namespace if location == "namespace" else mapped) / "__pycache__"
    cache.mkdir()
    (cache / "__init__.py").write_text("", encoding="utf-8")
    if location == "namespace":
        (cache / "payload.py").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_accepts_only_generated_workspace_bytecode(tmp_path):
    import sys
    from pm.extras import selected_import_paths

    project, site, _, mapped, _ = _editable_mirror_layout(tmp_path)
    cache = mapped / "__pycache__"
    cache.mkdir()
    (cache / f"mod.{sys.implementation.cache_tag}.pyc").write_bytes(b"bytecode")

    assert selected_import_paths(project) == (project.resolve(), site)

def test_selected_site_refuses_placeholder_not_derived_from_pth(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    package = project / "fixture_editable_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    mapped = workspace / "fixture_editable_pkg"
    mapped.mkdir(parents=True)
    (mapped / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project,
                           placeholder='"__editable__.other-1.0.finder.__path_hook__"')

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


@pytest.mark.parametrize("location", ["project", "site", "environment"])
def test_selected_site_refuses_placeholder_directory_in_selected_paths(tmp_path, location):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    collision = {"project": project, "site": site, "environment": project / "venv"}[location]
    (collision / "__editable__.fixture-0.0.0.finder.__path_hook__").mkdir()
    package = project / "fixture_editable_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    mapped = workspace / "fixture_editable_pkg"
    mapped.mkdir(parents=True)
    (mapped / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project)

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_accepts_editable_single_file_stem(tmp_path):
    from pm.extras import passive_module_origin, selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    root_module = project / "fixture_editable_mod.py"
    root_module.write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    workspace.mkdir()
    (workspace / "fixture_editable_mod.py").write_text("", encoding="utf-8")
    (workspace / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project, mapping_key="fixture_editable_mod")

    assert passive_module_origin("fixture_editable_mod",
                                 search_paths=selected_import_paths(project)) == str(root_module)


def test_selected_site_refuses_executable_editable_placeholder(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (project / "fixture_editable_pkg").mkdir()
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    (workspace / "fixture_editable_pkg").mkdir(parents=True)
    (workspace / "fixture_namespace").mkdir()
    _standard_editable_pth(
        site, project,
        placeholder='sys.path.insert(0, "/outside") or "__editable__.fixture.finder.__path_hook__"',
    )

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_accepts_virtualenv_shim_without_search_path(tmp_path, monkeypatch):
    import hashlib
    from pm import extras

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    shim_bytes = b"# distutils shim\n"
    (site / "_virtualenv.py").write_bytes(shim_bytes)
    (site / "_virtualenv.pth").write_text("import _virtualenv\n", encoding="utf-8")
    monkeypatch.setattr(extras, "_CERTIFIED_VIRTUALENV_HASHES",
                        {hashlib.sha256(shim_bytes).hexdigest()})

    assert extras.selected_import_paths(project) == (project.resolve(), site)


def test_selected_site_refuses_modified_virtualenv_module(tmp_path, monkeypatch):
    import hashlib
    from pm import extras

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    trusted = b"# distutils shim\n"
    modified = trusted + b"import os\n"
    (site / "_virtualenv.py").write_bytes(modified)
    (site / "_virtualenv.pth").write_text("import _virtualenv\n", encoding="utf-8")
    monkeypatch.setattr(extras, "_CERTIFIED_VIRTUALENV_HASHES",
                        {hashlib.sha256(trusted).hexdigest()})

    with pytest.raises(RuntimeError, match=hashlib.sha256(modified).hexdigest()):
        extras.selected_import_paths(project)


@pytest.mark.parametrize("change", ["statement", "mapping", "namespace"])
def test_selected_site_refuses_uncertified_editable_finder(tmp_path, change):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (project / "fixture_editable_pkg").mkdir()
    (project / "fixture_namespace").mkdir()
    workspace = project / "workspace"
    (workspace / "fixture_editable_pkg").mkdir(parents=True)
    (workspace / "fixture_namespace").mkdir()
    outside = tmp_path / "outside"
    (outside / "fixture_editable_pkg").mkdir(parents=True)
    (outside / "fixture_namespace").mkdir()
    options = {
        "statement": {"extra_statement": "sys.meta_path.insert(0, object())\n"},
        "mapping": {"mapping_value": repr(str(outside / "fixture_editable_pkg"))},
        "namespace": {"namespace_value": repr(str(outside / "fixture_namespace"))},
    }
    _standard_editable_pth(site, project, **options[change])

    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_refuses_editable_finder_missing_project_mapping(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (project / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project)
    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_refuses_editable_finder_missing_project_namespace(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (project / "fixture_editable_pkg").mkdir()
    (project / "fixture_editable_pkg" / "__init__.py").write_text("", encoding="utf-8")
    _standard_editable_pth(site, project)
    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


@pytest.mark.parametrize("line", [
    "import _virtualenv\nimport os\n",
    "import _virtualenv; import os\n",
    "import os\n",
])
def test_selected_site_refuses_extra_executable_statement(tmp_path, line):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (site / "_virtualenv.py").write_text("", encoding="utf-8")
    (site / "_virtualenv.pth").write_text(line, encoding="utf-8")
    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_refuses_virtualenv_shim_without_module(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (site / "_virtualenv.pth").write_text("import _virtualenv\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


def test_selected_site_refuses_nonliteral_editable_mapping(tmp_path):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    (project / "fixture_editable_pkg").mkdir()
    (project / "fixture_editable_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (project / "fixture_namespace").mkdir()
    _standard_editable_pth(site, project, mapping_value="str('dynamic')")
    with pytest.raises(RuntimeError, match="uninspectable import path declaration"):
        selected_import_paths(project)


@pytest.mark.platforms("linux")
def test_selected_site_refuses_fifo_pth_without_opening_it(tmp_path, monkeypatch):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    fifo = site / "blocked.pth"
    os.mkfifo(fifo)
    real_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == fifo:
            raise AssertionError("FIFO was opened")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(RuntimeError, match="uninspectable.*\\.pth"):
        selected_import_paths(project)


def test_selected_site_propagates_pth_read_failure(tmp_path, monkeypatch):
    from pm.extras import selected_import_paths

    project = _dependency_project(tmp_path)
    site = _selected_site(project)
    pth = site / "fixture.pth"
    pth.write_text("editable-fixture\n", encoding="utf-8")
    real_read = Path.read_text

    def read(path, *args, **kwargs):
        if path == pth:
            raise PermissionError("selected .pth read denied")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(PermissionError, match="selected .pth read denied"):
        selected_import_paths(project)


@pytest.mark.parametrize("alias", [" DISCORD ", " Google_Chat-platform "])
def test_nested_gateway_alias_uses_runtime_platform_identity(homes, tmp_path, alias):
    from gateway.config import Platform
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    if "DISCORD" in alias:
        _config(sibling, gateway={alias: {"enabled": True, "token": "fixture-token"}})
        expected = {"discord"}
        assert Platform(alias) is Platform.DISCORD
    else:
        _config(sibling, gateway={alias: {
            "enabled": True, "http_events_url": "https://chat.example.invalid/events",
        }})
        expected = {"google", "google-chat"}
        assert Platform(alias) is Platform("google_chat")
    assert expected <= set(configured_update_extras(_dependency_project(tmp_path), strict=True))


@pytest.mark.parametrize(("flag", "active"), [("true", True), ("false", False),
                                                ("${ENABLE_DISCORD}", True)])
def test_strict_enabled_string_agrees_with_runtime_and_disable_provenance(
    homes, tmp_path, monkeypatch, flag, active,
):
    from gateway.config import PlatformConfig
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"discord": {"enabled": flag, "token": "fixture-token"}})
    # A routed profile resolves ${VAR} through its own secret scope, as runtime does,
    # never through the launching process's environment.
    (sibling / ".env").write_text("ENABLE_DISCORD=true\n", encoding="utf-8")
    assert PlatformConfig.from_dict({"enabled": "true" if flag.startswith("${") else flag}).enabled is active
    extras = set(configured_update_extras(_dependency_project(tmp_path), strict=True))
    assert ("discord" in extras) is active


def test_strict_enabled_string_refuses_unknown_runtime_token(homes, tmp_path):
    from hermes_cli.main_install_repair import configured_update_extras

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"discord": {"enabled": "sometimes", "token": "fixture-token"}})
    with pytest.raises((OSError, RuntimeError, ValueError)):
        configured_update_extras(_dependency_project(tmp_path), strict=True)


def test_matrix_password_and_a2a_port_match_runtime_normalization(homes, tmp_path, monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.config_loader import read_yaml_layers
    from hermes_cli.main_install_repair import configured_update_extras
    from pm.extras import selected_import_paths
    from pm.platform_requirements import configured_platform_requirements
    from plugins.platforms.a2a import is_connected as a2a_connected
    from plugins.platforms.matrix.adapter import _is_connected as matrix_connected

    root, sibling = homes
    _config(root)
    _config(sibling, platforms={"matrix": {"enabled": True}})
    (sibling / ".env").write_text(
        "MATRIX_HOMESERVER=https://matrix.example.invalid\n"
        "MATRIX_PASSWORD=   \nA2A_PORT=   \n", encoding="utf-8",
    )
    project = _dependency_project(tmp_path, ("matrix", "a2a"))
    paths = selected_import_paths(project)

    def inspect():
        return configured_platform_requirements(
            read_yaml_layers(sibling, strict=True), os.environ.get,
            home=sibling, source_root=project, selected_paths=paths,
        )

    # Runtime and passive consumers see the same fixture credentials, not the operator's shell.
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.invalid")
    monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("MATRIX_PASSWORD", "   ")
    monkeypatch.setenv("A2A_PORT", "   ")
    assert not matrix_connected(PlatformConfig(enabled=True))
    assert not a2a_connected(PlatformConfig())
    assert not ({"matrix", "a2a"} & inspect().keys())
    assert "matrix" not in configured_update_extras(project, strict=True)

    (sibling / ".env").write_text(
        "MATRIX_HOMESERVER=https://matrix.example.invalid\n"
        "MATRIX_PASSWORD=fixture-password\nA2A_PORT=9900\n", encoding="utf-8",
    )
    monkeypatch.setenv("MATRIX_PASSWORD", "fixture-password")
    monkeypatch.setenv("A2A_PORT", "9900")
    assert matrix_connected(PlatformConfig(enabled=True))
    assert a2a_connected(PlatformConfig())
    assert {"matrix", "a2a"} <= inspect().keys()
    assert "matrix" in configured_update_extras(project, strict=True)
