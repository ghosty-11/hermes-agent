"""Update dependency diagnostics, historical markers and Windows launcher recovery."""

import logging
import os
import shutil
import sys
import time as _time

from pathlib import Path
from typing import NoReturn
from hermes_cli import _early_recovery as _early_recovery_mod

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")


def _pyproject_project(debug_fmt: str | None = None) -> dict | None:
    """``[project]`` table of pyproject.toml, or ``None`` when absent/unreadable."""
    from hermes_cli.main import PROJECT_ROOT
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except Exception as exc:
        if debug_fmt:
            logger.debug(debug_fmt, exc)
        return None
    return project if isinstance(project, dict) else None


# Install-scoped breadcrumbs live next to the venv (not under $HERMES_HOME)
# because the venv is shared across profiles.
#   ``.update-incomplete``       — generic core ``.[all]`` install was interrupted;
#     cleared only after a confirmed full dependency reinstall/recovery.
#   ``.lazy-refresh-incomplete`` — lazy-backend refresh may have corrupted packages;
#     cleared only after import-probe repair confirms healthy (never on indeterminate).
# Narrow lazy probes must NEVER clear the generic core marker.
# See #58004.
def _update_marker_path() -> Path:
    from hermes_cli.main import PROJECT_ROOT
    return PROJECT_ROOT / ".update-incomplete"


def _lazy_refresh_marker_path() -> Path:
    from hermes_cli.main import PROJECT_ROOT
    return PROJECT_ROOT / ".lazy-refresh-incomplete"


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True under pytest when ``root`` is this checkout: unsandboxed update/recovery tests must
    neither litter the live repo root with breadcrumbs (false-arming the developer's next launch)
    nor run a real reinstall against the executing venv (cf. ``managed_scope._under_pytest``)."""
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def _clear_marker_file(path: Path, *, label: str) -> None:
    """Remove an update-recovery breadcrumb. Never raises."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Could not clear %s marker: %s", label, exc)


def _clear_update_incomplete_marker() -> None:
    """Remove the interrupted core-install breadcrumb. Never raises."""
    _clear_marker_file(_update_marker_path(), label="update-incomplete")


def _clear_lazy_refresh_incomplete_marker() -> None:
    """Remove the interrupted lazy-refresh breadcrumb. Never raises."""
    _clear_marker_file(_lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


# Frozen old-updater import; current updates never detach dependency work.
_UPDATE_REEXEC_ENV = "HERMES_UPDATE_REEXEC"


def _reexec_dependency_sync_off_windows_shim() -> NoReturn:
    """Stop a mixed old-code/new-files updater at the retired sync boundary."""
    from hermes_cli._old_updater import stop_for_relaunch
    stop_for_relaunch()


def _is_windows() -> bool:
    return sys.platform == "win32"


def _venv_scripts_dir() -> Path | None:
    """Return the venv Scripts directory if we're running inside the project venv."""
    from hermes_cli.main import PROJECT_ROOT
    from hermes_constants import project_venv_dir
    from pm.environments import venv_bin_dir
    venv_dir = project_venv_dir(PROJECT_ROOT)
    if venv_dir is None:
        return None
    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


def _hermes_exe_shims(scripts_dir: Path) -> list[Path]:
    """Entry-point shims uv may rewrite during ``pip install -e .`` — Windows .exe launchers
    only; POSIX shims are plain scripts replaced atomically."""
    if not _is_windows():
        return []
    names = set(_load_console_script_names()) or {"hermes", "hermes-agent", "hermes-acp"}
    # Not a [project.scripts] entry point, but older update/install paths still
    # rewrite and quarantine it.
    names.add("hermes-gateway")
    return [scripts_dir / f"{name}.exe" for name in sorted(names)]


_PENDING_RENAME_KEY = r"SYSTEM\CurrentControlSet\Control\Session Manager"
_PENDING_RENAME_VALUE = "PendingFileRenameOperations"


def _filter_pending_shim_renames(entries: list[str], shims: list[Path]) -> tuple[list[str], int]:
    """Drop our ``<shim>`` -> ``<shim>.old.<stamp>`` pairs from a PendingFileRenameOperations
    value (a flat REG_MULTI_SZ of (source, target) pairs shared with other installers).
    Returns the entries to keep and how many pairs were dropped."""
    import ntpath

    def _norm(value: str) -> str:
        path = str(value).lstrip("!")
        if path.startswith("\\??\\"):
            path = path[4:]
        return ntpath.normcase(ntpath.normpath(path))

    shim_paths = {_norm(str(shim)) for shim in shims}
    kept: list[str] = []
    removed = 0
    for index in range(0, len(entries) - 1, 2):
        source, target = entries[index], entries[index + 1]
        source_norm = _norm(source)
        if source_norm in shim_paths and _norm(target).startswith(f"{source_norm}.old."):
            removed += 1
        else:
            kept.extend((source, target))
    if len(entries) % 2:
        kept.append(entries[-1])
    return kept, removed


def _cleanup_pending_shim_renames(scripts_dir: Path) -> int:
    """Drop reboot renames older Hermes versions queued for our shims: ``MOVEFILE_DELAY_UNTIL_REBOOT``
    fallbacks outlive the update that queued them and move away whatever sits at the shim path
    at next boot — even a shim a later repair just wrote. Needs elevation; a no-op otherwise."""
    if not _is_windows():
        return 0
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, _PENDING_RENAME_KEY, 0,
            winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE,
        ) as key:
            entries, value_type = winreg.QueryValueEx(key, _PENDING_RENAME_VALUE)
            if value_type != winreg.REG_MULTI_SZ or not isinstance(entries, list):
                return 0
            kept, removed = _filter_pending_shim_renames(entries, _hermes_exe_shims(scripts_dir))
            if not removed:
                return 0
            if kept:
                winreg.SetValueEx(key, _PENDING_RENAME_VALUE, 0, winreg.REG_MULTI_SZ, kept)
            else:
                winreg.DeleteValue(key, _PENDING_RENAME_VALUE)
            return removed
    except (OSError, ValueError):
        return 0


# Fresh orphan files can belong to an updater still running.
_QUARANTINE_GRACE_SECONDS = 15 * 60


def _quarantine_stamp_ms(stale: Path) -> int | None:
    """The ``.old.<unix-ms>`` stamp in a quarantine filename; ``None`` (not ours — neither rescued
    nor deleted) otherwise. Parsed from the NAME, not ``st_mtime``: ``rename`` preserves the
    shim's mtime (when uv wrote it), not when it was quarantined."""
    try:
        return int(stale.name.rsplit(".old.", 1)[1])
    except (IndexError, ValueError):
        return None


def _cleanup_quarantined_exes(scripts_dir: Path | None = None) -> None:
    """Sweep — and where necessary RESCUE — ``hermes.exe.old.*`` from updates.

    Called early on every invocation. Two cases an unconditional ``unlink()`` gets wrong:
    (1) orphan rescue — ``hermes.exe`` missing while ``hermes.exe.old.*`` exists means the .old
    file is the ONLY surviving copy (update died between rename and uv's write); put it back via
    the same retry-and-report helper the update-time restore uses. (2) concurrency — a fresh
    quarantine file may belong to an update in flight in another process; leave anything inside
    the grace window alone. Silent no-op on non-Windows, nothing to do, or locked/permission errors.

    Deleting it converts a one-rename recovery into a full reinstall. See #75584.
    """
    if not _is_windows():
        return
    scripts_dir = scripts_dir if scripts_dir is not None else _venv_scripts_dir()
    if scripts_dir is None:
        return
    _cleanup_pending_shim_renames(scripts_dir)
    now = _time.time()
    try:
        candidates = [
            (stamp, stale) for stale in scripts_dir.glob("*.exe.old.*")
            if (stamp := _quarantine_stamp_ms(stale)) is not None]
    except OSError:
        return
    # Newest first by PARSED stamp: lexicographic order breaks when a stray ``.old.999`` exists.
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    for stamp, stale in candidates:
        try:
            original = stale.with_name(stale.name.rsplit(".old.", 1)[0])
            if not original.exists():
                # Orphan rescue: last copy of the shim — retry ladder + recovery message.
                _early_recovery_mod.restore_quarantined_shims([(original, stale)])
                continue
            if now - stamp / 1000.0 < _QUARANTINE_GRACE_SECONDS:
                continue  # may be a live quarantine from a concurrent update
            stale.unlink()
        except OSError:
            pass  # still locked or in use — try again next run


def configured_update_extras(project_root: Path, *, strict: bool = False,
                             environment_root: Path | None = None) -> list[str]:
    """Read source declarations and the chosen install's dependency metadata without activation."""
    from agent.secret_scope import (
        build_profile_secret_scope, get_secret, reset_secret_scope, set_secret_scope,
        reset_multiplex_context, set_multiplex_context,
    )
    from gateway.config_loader import read_yaml_layers, load_legacy_gateway_json
    from hermes_constants import (
        get_routing_process_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )
    from gateway.platform_configuration import multiplex_for_passive_inspection
    from pm.extras import extra_supported, passive_importable, selected_import_paths
    from pm.features import declared_extras
    from pm.platform_requirements import configured_platform_requirements
    from pm.plugins_state import dependency_homes

    import functools

    selected_paths = selected_import_paths(
        project_root if environment_root is None else environment_root) if strict else ()
    probe = functools.partial(passive_importable, search_paths=selected_paths) if strict else None
    declared = set(declared_extras(project_root))
    required: set[str] = set()
    for home in dependency_homes():
        home_token = set_hermes_home_override(home)
        secret_token = None
        multiplex_token = None
        try:
            secrets = build_profile_secret_scope(
                home, strict=strict, include_external=True, authoritative_external=strict)
            secret_token = set_secret_scope(secrets, profile_home=str(home))
            if home.resolve() != get_routing_process_hermes_home().resolve():
                multiplex_token = set_multiplex_context(True)
            config = read_yaml_layers(home, strict=strict)
            legacy = load_legacy_gateway_json(home, strict=strict)
            routed = home.resolve() != get_routing_process_hermes_home().resolve()
            mode = multiplex_for_passive_inspection(config, legacy, get_secret)
            platforms = configured_platform_requirements(
                config, get_secret, legacy=legacy, home=home, source_root=project_root,
                selected_paths=selected_paths if strict else None,
                multiplex_active=mode if mode is not None else strict,
                routed=routed)
            for platform, requirement in platforms.items():
                for name in requirement.extras:
                    if name not in declared:
                        raise ValueError(f"Configured platform {platform} needs undeclared extra {name}")
                    if not extra_supported(name, **({"strict": True, "importable": probe,
                                                    "source_root": project_root} if strict else {})):
                        raise RuntimeError(f"Configured extra {name} is unsupported on this host")
                    required.add(name)
            if config.get("mcp_servers") or legacy.get("mcp_servers"):
                if "mcp" not in declared:
                    raise ValueError("Configured MCP servers require the mcp extra")
                required.add("mcp")
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Cannot inspect configured dependencies for {home}") from exc
            logger.debug("Configured dependencies for %s skipped: %s", home, exc)
        finally:
            if multiplex_token is not None:
                reset_multiplex_context(multiplex_token)
            if secret_token is not None:
                reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    return sorted(required)




def _configured_features_missing_deps(*, offline: bool = False,
                                     project_root: Path | None = None) -> list[tuple[str, str, str]]:
    """Check configured platforms/MCP in the fresh, selected update-build child.

    PM preserves recorded extras atomically, but configuration can reference an
    unselected SDK. Never call this in the updater's stale import graph (#10651).
    Rows are ``(label, hint, candidate extra)``: a platform's SDK is the extra named after it.
    """
    if offline:
        from pm.extras import install_hint, passive_available, selected_import_paths
        from pm.paths import repo_root

        root = project_root or repo_root()
        paths = selected_import_paths(root)
        return [(name, ", ".join(install_hint(name)), name)
                for name in configured_update_extras(root, strict=True)
                if not passive_available(name, search_paths=paths, source_root=root)]
    missing: list[tuple[str, str, str]] = []
    try:
        from gateway.config import load_gateway_config
        from gateway.platform_registry import platform_registry

        for platform in load_gateway_config().get_connected_platforms():
            entry = platform_registry.get(platform.value)
            if entry is not None and not entry.check_fn():
                missing.append((entry.label, entry.install_hint or "Run `hermes setup` to install support.",
                                entry.name))
    except Exception as exc:
        logger.debug("configured-platform dependency check skipped: %s", exc)
    try:
        import importlib.util
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        if config.get("mcp_servers") and importlib.util.find_spec("mcp") is None:
            missing.append(("MCP servers", "Run `hermes pm install` to install MCP support.", "mcp"))
    except Exception as exc:
        logger.debug("configured-MCP dependency check skipped: %s", exc)
    return missing


def _install_configured_features_missing_deps(project_root: Path, *, offline: bool = False) -> None:
    """Add the extras configured features need to the dependency environment (#124228).

    Recorded extras never follow config, so an enabled platform whose SDK is not
    selected would come up without its adapter after the gateway restart.
    """
    if offline:
        missing = _configured_features_missing_deps(offline=True, project_root=project_root)
        if missing:
            raise RuntimeError(
                "Offline completion lacks configured feature dependencies in the selected Python: "
                + ", ".join(label for label, _hint, _name in missing))
        return
    import pm
    from pm.extras import extra_supported
    from pm.features import declared_extras

    missing = _configured_features_missing_deps()
    if not missing:
        return
    declared = set(declared_extras(project_root))
    extras = sorted({extra for *_, name in missing
                     if (extra := name.replace("_", "-")) in declared and extra_supported(extra)})
    if extras:
        try:
            pm.sync_venv(extras, explicit=True, project_root=project_root,
                         evict_incompatible_plugins=True)
        except Exception as exc:
            print(f"  ⚠ Could not install {', '.join(extras)} for configured features: {exc}")
        else:
            missing = [row for row in missing if row[2].replace("_", "-") not in extras]
    if missing:
        print("  ⚠ Configured features whose dependencies are still missing — the gateway will fail to load them on restart:")
        for feature, hint, _extra in missing:
            print(f"    - {feature}: {hint}")


def _load_console_script_names() -> list[str]:
    """Return ``[project.scripts]`` entry-point names from pyproject.toml."""
    project = _pyproject_project("console script verification: failed to read pyproject.toml: %s")
    scripts = (project or {}).get("scripts", {}) or {}
    return [str(name) for name in scripts if name]


def _is_termux_env(env: dict[str, str] | None = None) -> bool:
    from hermes_cli.main import _is_termux_startup_environment
    return _is_termux_startup_environment(env)


def _is_windows_npm_path(npm_path: str) -> bool:
    """True if ``npm_path`` points at a Windows npm shim (WSL drive interop, ``.cmd``/``.exe``, UNC).

    Callers use this only on a POSIX host — on native Windows ``npm.cmd`` is correct.
    """
    low = npm_path.lower()
    mount = low.split("/", 3)[2] if low.startswith("/mnt/") else ""
    return (
        low.endswith((".exe", ".cmd", ".bat"))
        or (len(mount) == 1 and mount.isalpha())
        or "\\" in npm_path
    )


def _resolve_node_runtime_npm() -> str | None:
    """PM's npm, refused on a POSIX host when it is a Windows shim (EISDIR over WSL UNC paths, #30271).

    Never re-scans the user's PATH for another npm: Hermes runs only its PM-managed toolchain.
    """
    from hermes_constants import find_node_executable
    npm = find_node_executable("npm")
    if _is_windows() or not npm:
        return npm
    return None if _is_windows_npm_path(npm) else npm


def _resolve_update_branch(args) -> str:
    """Normalize ``args.branch`` to a non-empty name (default ``main``; blank/whitespace = default)."""
    return (getattr(args, "branch", None) or "main").strip() or "main"
