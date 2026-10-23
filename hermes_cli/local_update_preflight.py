"""Admit frozen-source inputs without publishing live selections or dependency facts."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import os
from pathlib import Path

@contextmanager
def _private_store(path: Path):
    previous = os.environ.get("HERMES_RUNTIME_DIR")
    os.environ["HERMES_RUNTIME_DIR"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HERMES_RUNTIME_DIR", None)
        else:
            os.environ["HERMES_RUNTIME_DIR"] = previous



def admitted_extras(source_root: Path, installed_root: Path,
                    required_extras: list[str]) -> tuple[list[str], dict]:
    """The extras the target graph must keep, and the recorded venv facts.

    A main-era venv has neither ledger; ordinary migration keeps its ``all``
    plus installed opt-in selection, so local admission keeps it too.
    """
    from hermes_cli.main_install_repair import configured_update_extras
    from pm.environments import runtime_facts_path
    from pm.extras import legacy_selection
    from pm.lock import Facts
    from pm.paths import store_root

    recorded = (Facts(runtime_facts_path(installed_root), strict=True).get("venv")
                or Facts(store_root(installed_root) / "facts.json", strict=True).get("venv") or {})
    if not isinstance(recorded.get("extras", []), list):
        raise ValueError("recorded extras are invalid")
    # As ordinary migration decides: only this installation's runtime ledger
    # replaces the legacy selection; a shared store record is unioned with it.
    legacy = [] if runtime_facts_path(installed_root).is_file() else legacy_selection(installed_root)
    extras = sorted(set(recorded.get("extras", [])) | set(legacy)
                    | set(configured_update_extras(
                        source_root, strict=True, environment_root=installed_root))
                    | set(required_extras))
    return extras, recorded


def stage_local_inputs(source_root: Path, installed_root: Path, staging: Path,
                       required_extras: list[str]) -> list[str]:
    """Prove the target graph with real offline builders before changing installed HEAD."""
    from hermes_cli.source_build import validate_offline_products
    from pm.plugin_declarations import manifest_version_error, read_python_declaration
    from pm.install import _install
    from pm.environment import managed_environment
    from pm.lock import Facts, Lockfile
    from pm.offline import offline_environment
    from pm.paths import writable_store_root, store_root
    from pm.registry import get_package, tool_roots
    from pm.store import Store, current_target
    from pm.build_operations import stage_tools, verified_tools
    from pm.workspace import enabled_plugin_dirs, lock_and_sync

    source_root, installed_root = Path(source_root).resolve(), Path(installed_root).resolve()
    if source_root == installed_root:
        raise ValueError("local preflight requires a frozen source distinct from the installed tree")
    extras, recorded = admitted_extras(source_root, installed_root, required_extras)
    tools = staging / "tools"
    original = writable_store_root(installed_root).resolve()
    published = tuple(dict.fromkeys((store_root(installed_root).resolve(), original)))
    store = Store(tools)
    lock = Lockfile(source_root / "pm" / "lock.json")
    target = current_target()
    # The private runtime has no inherited uv facts. Admit its locked engine
    # alongside public tools before the offline environment is created.
    names = list(dict.fromkeys((*tool_roots(lock.names()), "uv")))
    from pm.package import InstallError
    import shutil
    source_facts = {root: Facts(root / "facts.json", strict=True) for root in published}
    missing = []
    for name in names:
        package = get_package(name)
        if getattr(package, "pin_only", False):
            continue
        for root in published:
            fact = source_facts[root].get(name)
            if fact and fact.get("entry") == package.store_entry(lock.version(name), target):
                try:
                    verified_tools([name], source_store=root, target=target, lock=lock)
                except InstallError:
                    continue
                stage_tools([name], source_store=root, out=tools, target=target, lock=lock)
                break
        else:
            missing.append(package)
    # Only verified local archives for missing target pins enter the private store.
    for package in missing:
        for artifact in lock.artifacts(package.name, target):
            for root in published:
                try:
                    with (nullcontext() if (root.parent / "manifest.json").is_file()
                          else Store(root).install_lock()):
                        archive = Store(root).fetch_many([artifact], staging, offline=True)[0]
                        dest = tools / f"fetch-{artifact['sha256']}"
                        dest.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(archive, dest / archive.name)
                    break
                except InstallError:
                    continue
            else:
                raise InstallError(package.name, f"offline archive {artifact['sha256']} is missing")
    with _private_store(tools):
        with store.install_lock():
            facts = Facts(tools / "facts.json", strict=True)
            for package in missing:
                _install(package, lock, facts, store, target, _lock_held=True, offline=True)
        selection = verified_tools(names, source_store=tools, target=target, lock=lock)
        python = selection.entries["python"].binary
        if python is None:
            raise RuntimeError("frozen Python pin has no executable")
        environment = offline_environment(selection.environment(dict(os.environ)))
        members = []
        for path in enabled_plugin_dirs(skip_invalid_secondary=False):
            declaration = read_python_declaration(path)
            reason = manifest_version_error(declaration.manifest, path.name)
            if reason:
                raise RuntimeError(f"configured plugin is incompatible with target: {reason}")
            if declaration.is_member:
                members.append(path)
        workspace = staging / "workspace"
        candidate = staging / "venv"
        engine = managed_environment(candidate, python=python, env=environment,
                                     offline=True, explicit=True)
        engine.create()
        seed = recorded.get("resolved_lock") if members else source_root / "uv.lock"
        if seed and not Path(seed).is_file():
            raise RuntimeError("recorded plugin resolution is missing")
        lock_and_sync(members, extras, root=workspace, source=source_root,
                      seed_lock=Path(seed) if seed else None,
                      frozen=not members, environment=engine)
        engine.check()
        validate_offline_products(source_root, installed_root, env=environment)
    return extras
