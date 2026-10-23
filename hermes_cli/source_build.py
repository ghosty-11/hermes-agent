"""Source launch/update composition over the shared JavaScript builders."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def source_product_current(project_root: Path, product: str, out: Path) -> bool:
    """Read the compiler's receipt without acquiring tools or dependencies."""
    from pm import env_for, installed_package

    installed = installed_package("node")
    if installed is None or installed.binary is None:
        return False  # Only PM's Node may run the receipt reader; never the user's PATH copy.
    node, env = str(installed.binary), env_for("node")
    try:
        result = subprocess.run(
            [node, str(project_root / "scripts/build/freshness.mjs"),
             "--source", str(project_root), "--product", product, "--out", str(out)],
            cwd=project_root, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
        return result.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def _normalized_source_build_env(base_env: dict | None = None, *, offline: bool = False) -> dict[str, str]:
    """Apply receipt-significant source-build settings without selecting or acquiring tools."""
    from pm.environments import project_python, running_from_selected_environment
    from pm.paths import repo_root
    from hermes_constants import get_hermes_home

    # Icon generation starts an isolated child, which needs the selected venv
    # executable when the update runs on store Python.
    root = repo_root()
    python = str(project_python(root)) if running_from_selected_environment(root) else sys.executable
    env = {**os.environ, **(base_env or {}), "CI": "1", "HERMES_PYTHON": python,
           "PYTHON": python}
    env.pop("ESBUILD_BINARY_PATH", None)
    npmrc = get_hermes_home() / "npmrc"
    if npmrc.is_file():
        env.setdefault("NPM_CONFIG_USERCONFIG", str(npmrc))
    if offline:
        from pm.offline import offline_environment
        env = offline_environment(env)
    return env


def source_build_env(base_env: dict | None = None, *, explicit: bool = False,
                     offline: bool = False) -> dict[str, str]:
    from pm import ensure

    env = _normalized_source_build_env(base_env, offline=offline)
    return ensure("npm", base_env=env, explicit=explicit,
                  **({"offline": True} if offline else {})).env


def run_source_script(project_root: Path, script: str, *args: str, env: dict, label: str) -> None:
    from pm.progress import run_contained

    # npm's deprecation warnings are the loudest lines and never actionable
    # here; they still land in the failure tail.
    run_contained(
        [shutil.which("node", path=env["PATH"]), str(project_root / script), *args],
        label, hide=lambda line: line.lower().startswith("npm warn"), indent="  ",
        cwd=project_root, env=env,
    )


def prepare_source_dependencies(project_root: Path, workspaces: tuple[str, ...], *, env: dict,
                                explicit: bool = False, offline: bool = False) -> None:
    from pm import lazy_installs_allowed

    run_source_script(
        project_root, "scripts/build/node-deps.mjs", "--source", str(project_root), "--reuse",
        *(() if not offline and (explicit or lazy_installs_allowed()) else ("--no-install",)),
        *(arg for workspace in workspaces for arg in ("--workspace", workspace)), env=env,
        label="Preparing Node dependencies",
    )


def prepare_launch_dependencies(project_root: Path, *, env: dict) -> None:
    """A launch rebuild must not prune another installed source frontend."""
    from hermes_cli.main_desktop import _desktop_dist_exists, _desktop_packaged_executable

    desktop_dir = project_root / "apps/desktop"
    desktop = _desktop_dist_exists(desktop_dir) or _desktop_packaged_executable(desktop_dir) is not None
    workspaces = ("ui-tui", "web") + (("apps/desktop",) if desktop else ())
    prepare_source_dependencies(project_root, workspaces, env=env)


def build_source_tui(project_root: Path, *, env: dict) -> None:
    run_source_script(project_root, "scripts/build/tui.mjs", env=env, label="Building the TUI")


def build_source_web(project_root: Path, *, env: dict, icons: Path | None = None) -> None:
    # Default-brand icons are committed; installs never render them.
    icons = icons or project_root
    run_source_script(project_root, "scripts/build/web.mjs", "--source", str(project_root),
                      "--icons", str(icons), "--out", str(project_root / "hermes_cli/web_dist"), env=env,
                      label="Building the web UI")


def source_frontends(project_root: Path) -> tuple[str, ...]:
    """The frontend workspaces this checkout carries. A source slice without them
    (python-only installs, the installer's acceptance fixture) has no products
    to build; it still publishes commands and runs the maintenance tail."""
    return tuple(name for name in ("ui-tui", "web") if (project_root / name / "package.json").is_file())


def validate_offline_products(source_root: Path, installed_root: Path, *,
                              env: dict[str, str] | None = None) -> dict:
    """Read-only admission against the installed preparation, not the new checkout's outputs.

    The caller stages selected tools privately when needed. This function never
    ensures tools or publishes a dependency/application selection.
    """
    # Node admission covers the source TUI and web workspaces only.

    source_root, installed_root = Path(source_root).resolve(), Path(installed_root).resolve()
    frontends = source_frontends(source_root)
    workspaces = frontends
    if source_root == installed_root:
        raise ValueError("Offline admission needs distinct source and installed trees")
    if not workspaces:
        return {"source_root": str(source_root), "installed_root": str(installed_root),
                "workspaces": (), "products": (), "mutation_roots": []}
    if env is None:
        from pm import env_for, installed_package
        selected = installed_package("node")
        npm_selected = installed_package("npm")
        if selected is None or selected.binary is None or npm_selected is None or npm_selected.binary is None:
            raise RuntimeError("Offline products require selected Node and npm or a privately staged tool environment")
        env = env_for("npm", base_env=dict(os.environ))
        node = str(selected.binary)
    else:
        node = shutil.which("node", path=env.get("PATH", env.get("Path", "")))
    tool_path = env.get("PATH", env.get("Path", ""))
    if not node or not shutil.which("npm", path=tool_path):
        raise RuntimeError("Offline products require prepared Node and npm tools")
    child_env = _normalized_source_build_env(env, offline=True)
    result = {
        "source_root": str(source_root), "installed_root": str(installed_root),
        "workspaces": workspaces, "products": tuple(frontends),
        "mutation_roots": [str(installed_root / "node_modules"),
                           *(str(installed_root / item) for item in (
                               ("ui-tui/dist",) if "ui-tui" in frontends else ()) + (
                               ("hermes_cli/web_dist",) if "web" in frontends else ()))],
    }
    if workspaces:
        prepared = subprocess.run(
            [node, str(source_root / "scripts/build/node-deps.mjs"),
             "--source", str(source_root), "--prepared-source", str(installed_root),
             "--check-prepared", *(arg for workspace in workspaces for arg in ("--workspace", workspace))],
            cwd=source_root, env=child_env, capture_output=True, text=True, check=True,
        )
        import json
        result["node"] = json.loads(prepared.stdout.splitlines()[-1])
    return result


def build_update_products(project_root: Path, *, desktop: bool, offline: bool = False) -> None:
    """Prepare the selected union once; a failed product aborts the update."""
    # Both current updates and historical takeover reach this in a fresh target
    # interpreter, never in the updater's pre-sync import graph.
    from hermes_cli.main_install_repair import _install_configured_features_missing_deps
    from hermes_cli.update_stage import publish_stage

    _install_configured_features_missing_deps(project_root, **({"offline": True} if offline else {}))
    frontends = source_frontends(project_root)
    if not frontends:
        if offline:
            from hermes_cli.memory_provider_migration import migrate_all_homes
            migrate_all_homes(offline=True)
        return
    env = source_build_env(explicit=True, **({"offline": True} if offline else {}))
    workspaces = frontends + (("apps/desktop",) if desktop else ())
    publish_stage("Updating Node dependencies")
    prepare_source_dependencies(project_root, workspaces, env=env, explicit=True,
                                **({"offline": True} if offline else {}))
    if "ui-tui" in frontends:
        publish_stage("Building the TUI")
        build_source_tui(project_root, env=env)
    if "web" in frontends:
        publish_stage("Building the web UI")
        build_source_web(project_root, env=env)
    if desktop:
        from hermes_cli.main_desktop import _refresh_installed_desktop_apps, build_prepared_desktop

        publish_stage("Building the desktop app")
        # The desktop build mutates checkout-scoped node_modules and
        # apps/desktop/release; serialize it against a concurrent manual
        # `hermes desktop` (#93940). The update path waits rather than exits:
        # the in-flight build it queues behind produces the same fresh tree
        # this update needs.
        from hermes_cli.desktop_build_lock import DesktopBuildLock

        build_lock = DesktopBuildLock(project_root)
        build_lock.acquire(wait=True)
        try:
            build_prepared_desktop(
                project_root / "apps/desktop", source_mode=False,
                npm=shutil.which("npm", path=env["PATH"]), env=env, icons=project_root,
            )
        finally:
            build_lock.release()
        # A current release/ can still sit beside a stale installed copy (an earlier
        # update rebuilt but never installed); healing must not wait for the next build.
        _refresh_installed_desktop_apps(project_root / "apps/desktop")
    # A configured memory provider must remain available after the core cutover.
    from hermes_cli.memory_provider_migration import migrate_all_homes

    if offline:
        migrate_all_homes(offline=True)
    else:
        try:
            migrate_all_homes()
        except Exception as exc:
            print(f"  ⚠ Memory provider migration skipped: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build source-install frontends")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    build_update_products(args.source.resolve(), desktop=args.desktop)
