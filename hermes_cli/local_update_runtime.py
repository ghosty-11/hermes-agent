"""Explicit rollback ownership inventory; no resolver publishes or acquires resources."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from pm.environments import base_venv, install_state_dir, runtime_facts_path, store_root
from hermes_cli.local_update_coverage import dependency_coverage, retained_bookkeeping


PRODUCTS = (
    "node_modules", "ui-tui/node_modules", "web/node_modules",
    "ui-tui/dist", "ui-tui/packages/hermes-ink/dist",
    "hermes_cli/web_dist", ".hermes/bin", "install-stamp.json",
    ".hermes-bootstrap-complete",
)


def _record(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        if path.exists() or path.is_symlink():
            raise ValueError(f"invalid rollback selection: {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid rollback selection: {path}")
    return data


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _owned(path: Path, base: Path) -> Path:
    if not _within(path.resolve(), base.resolve()):
        raise ValueError(f"rollback resource escapes its owner: {path}")
    return path


def inventory(install: Path, homes: list[Path]) -> dict:
    """Resolve the committed closure, including externally located state and stores."""
    state = install_state_dir(install)
    store = store_root(install)
    # writable_store_root is defined for the executing PM checkout; a capsule
    # may name another installed tree with a different stamped runtime store.
    from hermes_constants import get_default_hermes_root
    writable = (get_default_hermes_root() / "tools"
                if (store.parent / "manifest.json").is_file() else store)
    selections = [runtime_facts_path(install), state / "pm-runtime" / "selected.json",
                  store / "facts.json"]
    if writable != store:
        selections.append(writable / "facts.json")
    resources: list[Path] = []
    application = _record(selections[0]).get("packages", {}).get("venv", {})
    if application.get("environment"):
        environment = Path(application["environment"])
        generation = _owned(environment.parent, state / "environments")
        if environment != generation / "venv" or not (environment / "pyvenv.cfg").is_file():
            raise ValueError(f"invalid selected application environment: {environment}")
        resolved = application.get("resolved_lock")
        if resolved and (not _within(Path(resolved), generation / "workspace")
                         or not Path(resolved).is_file()):
            raise ValueError("selected resolved lock is missing or escapes the generation")
        resources.append(generation)
    else:
        legacy = base_venv(install)
        if (legacy / "pyvenv.cfg").is_file():
            if not any(_within(legacy.resolve(), scope) for scope in (install, *homes)):
                raise ValueError(f"selected legacy environment escapes captured scopes: {legacy}")
            resources.append(legacy)
    pm = _record(selections[1])
    if pm:
        relative = Path(pm["generation"])
        if len(relative.parts) != 2 or relative.parts[0] != "generations" or relative.parts[1] in {".", ".."}:
            raise ValueError("invalid selected PM runtime")
        selected_runtime = _owned(state / "pm-runtime" / relative, state / "pm-runtime" / "generations")
        if not (selected_runtime / "pm-runtime.json").is_file():
            raise ValueError("selected PM runtime is incomplete")
        resources.append(selected_runtime)
    for location in (store, writable) if writable != store else (store,):
        facts = _record(location / "facts.json")
        packages = facts.get("packages", {})
        if not isinstance(packages, dict):
            raise ValueError("invalid tool facts")
        for fact in packages.values():
            if "entry" not in fact:
                continue
            entry = fact["entry"]
            if not isinstance(entry, str) or not entry or Path(entry).name != entry or entry in {".", ".."}:
                raise ValueError("invalid tool entry")
            resources.append(_owned(location / entry, location))
    for home in homes:
        envs = home / "environments"
        if not envs.is_dir():
            continue
        if envs.is_symlink():
            raise ValueError(f"external environment mutation root: {envs}")
        for entry in envs.iterdir():
            if not entry.is_dir() or entry.is_symlink():
                continue
            selected = entry / "active.json"
            selections.append(selected)
            active = _record(selected)
            if active:
                name = active["generation"]
                if not isinstance(name, str) or not name.startswith("gen-") or len(name) != 36 or any(c not in "0123456789abcdef" for c in name[4:]):
                    raise ValueError(f"invalid selected side environment: {selected}")
                side = _owned(entry / name, entry)
                if not (side / "venv" / "pyvenv.cfg").is_file():
                    raise ValueError(f"selected side environment is incomplete: {side}")
                resources.append(side)
    # Never snapshot a source inode through a hardlink; the capsule copy is independent.
    return {"selections": sorted(set(selections)), "resources": sorted(set(resources)),
            "state": state, "store": store, "writable": writable}


def dependency_closure(resources: list[Path], products: tuple[Path, ...],
                       install: Path, homes: list[Path], store: Path,
                       writable: Path, state: Path) -> tuple[list[Path], list[Path]]:
    """Retain owned entries named by venv bases and package links, even after facts change."""
    stores = (store, writable)
    external: set[Path] = set()
    retained = set(resources)
    pending = [(resource, False) for resource in resources]
    visited: set[tuple[Path, bool]] = set()
    checked_source: set[Path] = set()
    visiting_source: set[Path] = set()
    runtime = {"state": state, "store": store, "writable": writable}
    coverage = dependency_coverage(homes, install, products, runtime, retained)

    def require_image(target: Path, resolved: Path) -> None:
        owners = coverage.byte_owners(resolved)
        if len(owners) != 1:
            raise ValueError(f"excluded dependency has no unique rollback byte owner: {target}")


    def reference(target: Path) -> None:
        nonlocal coverage
        resolved = target.resolve(strict=True)
        # Generation directories are excluded from the state/home images even
        # when the state itself lives inside a profile home.
        for container in (state / "environments", state / "pm-runtime" / "generations"):
            if resolved.is_relative_to(container) and resolved != container:
                entry = container / resolved.relative_to(container).parts[0]
                if not entry.is_dir() or entry.is_symlink():
                    raise ValueError(f"unsupported generation dependency: {target}")
                if entry not in retained:
                    retained.add(entry)
                    coverage = dependency_coverage(homes, install, products, runtime, retained)
                    pending.append((entry, False))
                require_image(target, resolved)
                if resolved.is_dir():
                    pending.append((resolved, True))
                return
        for home in homes:
            container = home / "environments"
            if resolved.is_relative_to(container):
                parts = resolved.relative_to(container).parts
                if len(parts) >= 2 and parts[1].startswith("gen-"):
                    entry = container / parts[0] / parts[1]
                    if not entry.is_dir() or entry.is_symlink():
                        raise ValueError(f"unsupported side generation dependency: {target}")
                    if entry not in retained:
                        retained.add(entry)
                        coverage = dependency_coverage(homes, install, products, runtime, retained)
                        pending.append((entry, False))
                    require_image(target, resolved)
                    if resolved.is_dir():
                        pending.append((resolved, True))
                    return
                raise ValueError(f"unsupported side environment dependency: {target}")
        for owner in stores:
            if resolved == owner:
                raise ValueError(f"unbounded managed dependency: {target}")
            if resolved.is_relative_to(owner):
                entry = owner / resolved.relative_to(owner).parts[0]
                if not entry.is_dir() or entry.is_symlink():
                    raise ValueError(f"unsupported managed dependency: {target}")
                if entry not in retained:
                    retained.add(entry)
                    coverage = dependency_coverage(homes, install, products, runtime, retained)
                    pending.append((entry, False))
                require_image(target, resolved)
                if resolved.is_dir():
                    pending.append((resolved, True))
                return
        if resolved.is_relative_to(install):
            if any(_within(resolved, product) for product in products):
                require_image(target, resolved)
                if resolved.is_dir():
                    pending.append((resolved, True))
                return
            if resolved in visiting_source or resolved == install:
                raise ValueError(f"cyclic or unbounded workspace dependency: {target}")
            if resolved in checked_source:
                return
            visiting_source.add(resolved)
            from hermes_cli.update_local import local_git

            relative = resolved.relative_to(install)
            tracked = set(local_git(install, "ls-files", "-z", "--", os.fspath(relative),
                                    binary=True).stdout.split(b"\0"))
            if not tracked:
                raise ValueError(f"unsupported untracked workspace dependency: {target}")

            def source_paths(top: Path):
                # Product subtrees inside a linked workspace (its node_modules,
                # build output) are imaged as products, never as tracked source.
                for directory, dirs, files in os.walk(top, followlinks=False):
                    base = Path(directory)
                    dirs[:] = [name for name in dirs
                               if not any(_within(base / name, product) for product in products)]
                    yield from (base / name for name in (*dirs, *files))

            paths = [resolved] if resolved.is_file() else source_paths(resolved)
            for path in paths:
                if path.is_symlink():
                    reference(path)
                elif path.is_file() and os.fsencode(path.relative_to(install)) not in tracked:
                    raise ValueError(f"unsupported untracked workspace dependency: {path}")
            visiting_source.remove(resolved)
            checked_source.add(resolved)
            return
        if (resolved.is_relative_to(state)
                or any(_within(resolved, scope) for scope in retained)
                or any(_within(resolved, home) for home in homes)):
            require_image(target, resolved)
            if resolved.is_dir():
                pending.append((resolved, True))
            return
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(f"unsupported external dependency: {target}")
        if resolved.stat().st_mode & 0o111 or resolved.suffix.lower() in {".exe", ".cmd", ".bat"}:
            raise ValueError(f"unsupported external executable closure: {target}")
        external.add(resolved)

    for product in products:
        if product.is_symlink():
            reference(product)
        elif product.is_dir():
            pending.append((product, False))

    from hermes_cli._launchers import _owns_launcher

    for name in ("hermes", "hermes-acp"):
        launcher = install / ".hermes" / "bin" / name
        if not launcher.is_file() or not _owns_launcher(launcher, install):
            continue
        body = launcher.read_text(encoding="utf-8")
        if not body.startswith("#!/bin/sh\n"):
            raise ValueError(f"unsupported launcher interpreter binding: {launcher}")
        tokens = shlex.split(body.split("\n", 1)[1])
        if (len(tokens) < 4 or tokens[0] != "exec" or tokens[-1] != "$@"
                or not Path(tokens[1]).is_absolute()):
            raise ValueError(f"unsupported launcher interpreter binding: {launcher}")
        reference(Path(tokens[1]))

    while pending:
        root, dependency = pending.pop()
        if (root, dependency) in visited or not root.is_dir():
            continue
        visited.add((root, dependency))
        for directory, dirs, files in os.walk(root, followlinks=False):
            # Prune only the caches excluded from retained resource images.
            base = Path(directory)
            cache = base / "__pycache__"
            if "__pycache__" in dirs and any(
                    retained_bookkeeping(cache, resource_root=resource)
                    for resource in retained if cache.is_relative_to(resource)):
                dirs.remove("__pycache__")
            for name in (*dirs, *files):
                path = base / name
                if dependency:
                    require_image(path, path)
                if path.is_symlink():
                    reference(path)
                if name == "pyvenv.cfg" and path.is_file():
                    settings = {}
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if "=" in line:
                            key, value = line.split("=", 1)
                            settings[key.strip().lower()] = value.strip()
                    base = Path(settings.get("home", ""))
                    if base.is_absolute():
                        if not base.is_dir():
                            raise ValueError(f"unsupported missing Python base: {base}")
                        if not any(base.resolve().is_relative_to(owner) for owner in (*stores, *resources)):
                            raise ValueError(f"unsupported external Python base: {base}")
                        for name in ("python3", "python", "python.exe"):
                            executable = base / name
                            if executable.exists():
                                if not any(executable.resolve().is_relative_to(owner)
                                           for owner in (*stores, *resources)):
                                    raise ValueError(f"unsupported external Python executable: {executable}")
                                reference(executable)
                                break
                        else:
                            raise ValueError(f"unsupported Python base: {base}")
    return sorted(retained), sorted(external)


def mutation_roots(homes: list[Path], install: Path) -> list[Path]:
    """Inspect the real directory graph reached by maintenance, including link targets."""
    scopes = [install, *homes]
    referents: set[Path] = set()
    visited: set[Path] = set()

    def inspect(candidate: Path) -> None:
        if candidate.is_symlink():
            referent = candidate.resolve(strict=True)
            if not any(_within(referent, scope) for scope in scopes) or referent == install:
                raise ValueError(f"external symlinked mutation root: {candidate}")
            if referent not in homes:
                referents.add(referent)
        else:
            referent = candidate.resolve(strict=True)
        if not referent.is_dir() or referent in visited:
            return
        visited.add(referent)
        for child in referent.iterdir():
            if child.is_dir():
                inspect(child)
            elif child.is_symlink():
                target = child.resolve(strict=True)
                if target.is_dir():
                    inspect(child)

    for home in homes:
        for name in ("skills", "config.yaml", ".env", "cron", "plugins", "environments"):
            dest = home / name
            if name == "skills" and dest.is_dir():
                inspect(dest)
            elif dest.is_symlink():
                inspect(dest)
    return sorted(referents)


def owned_conveniences(install: Path) -> list[Path]:
    """Capture existing owned commands and absent default exposure destinations."""
    from hermes_cli._launchers import _owns_launcher, WINDOWS_BIN_LAUNCHERS
    from hermes_constants import get_default_hermes_root

    default = Path.home() / ".local" / "bin"
    candidates = {default, get_default_hermes_root() / "bin", Path("/usr/local/bin")}
    result = []
    for directory in candidates:
        if directory != default and not _owns_launcher(directory / "hermes", install):
            continue
        for name in (*WINDOWS_BIN_LAUNCHERS, "hermes-agent"):
            path = directory / name
            if not (path.exists() or path.is_symlink()) or _owns_launcher(path, install):
                result.append(path)
    return sorted(set(result))


