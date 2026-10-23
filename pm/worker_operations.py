"""Import-light routing and prerequisite policy for the private PM worker.

PM's own runtime is deliberately absent: its public staging entry point must
run directly, before the worker's dependencies exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Literal


@dataclass(frozen=True)
class Operation:
    module: str
    packages: tuple[str, ...] | None  # None selects the request's named package.
    bootstrap: Literal["always", "policy", "state", "never"]
    shared_writer: bool = True  # False for reads and requests whose destinations must be classified.

    def resolve(self, name: str):
        return getattr(import_module(self.module), name)


OPERATIONS = {
    "ensure": Operation("pm.install", None, "state"),
    "stage_only": Operation("pm.install", None, "always"),
    "stage_tools": Operation("pm.build_operations", None, "never"),
    "prepare_tools": Operation("pm.build_operations", None, "always"),
    "sync_venv": Operation("pm.install", ("venv",), "policy"),
    "venv_is_current": Operation("pm.install", ("venv",), "never", shared_writer=False),
    "build_environment": Operation("pm.operations", ("uv",), "policy"),
    "lock_project": Operation("pm.operations", ("uv",), "policy"),
    "ensure_environment": Operation("pm.operations", ("uv",), "policy"),
    "ensure_project_environment": Operation("pm.operations", ("uv",), "policy"),
    "ensure_python_tool": Operation("pm.operations", ("uv",), "policy"),
    "check_project_lock": Operation("pm.build_operations", ("uv",), "policy", shared_writer=False),
    "export_requirements": Operation("pm.build_operations", ("uv",), "policy", shared_writer=False),
    "build_requirements_environment": Operation("pm.build_operations", ("uv",), "policy", shared_writer=False),
    "prune_cache": Operation("pm.build_operations", ("uv",), "always"),
}

def _writer_domain_roots():
    from hermes_constants import get_default_hermes_root
    from pm.environments import dependency_home_root
    from pm.paths import install_root, store_root, writable_store_root

    return tuple(root.resolve() for root in (
        install_root(), dependency_home_root(), get_default_hermes_root(),
        store_root(), writable_store_root(), get_default_hermes_root() / "cache" / "uv",
    ))


def private_write_destination(value, *, roots=None) -> bool:
    """Resolve missing descendants through existing parents; reject final links."""
    if value is None:
        return False
    from pathlib import Path

    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        return False
    if path.is_symlink():
        return False
    ancestor = path
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    # Concurrent same-UID ancestor swaps remain outside this cooperative protocol.
    resolved = ancestor.resolve() / path.relative_to(ancestor)
    domains = _writer_domain_roots() if roots is None else roots
    return all(not resolved.is_relative_to(root) for root in domains)


def writes_shared_state(operation: str, arguments: dict) -> bool:
    """Exempt only explicit destinations and caches outside every writer domain."""
    if OPERATIONS[operation].shared_writer:
        return True
    if operation == "venv_is_current":
        return False
    roots = _writer_domain_roots()
    destinations = ("cache",) if operation == "check_project_lock" else ("out", "cache")
    return not all(private_write_destination(arguments.get(name), roots=roots) for name in destinations)
