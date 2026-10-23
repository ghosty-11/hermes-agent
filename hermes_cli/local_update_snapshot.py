"""Freeze the exact reviewed commit before importing any of its application code."""
from __future__ import annotations

from pathlib import Path


def freeze_target(source: Path, target: str, destination: Path) -> Path:
    from hermes_cli.update_local import local_git, verify_closure

    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    local_git(destination, "init", "-q")
    # file:// forces pack transport, not a local clone's hardlinks to mutable objects.
    local_git(destination, "fetch", "--no-tags", "--no-filter", "--no-recurse-submodules",
              source.as_uri(), target)
    verify_closure(destination, target)
    local_git(destination, "checkout", "--detach", target)
    if (local_git(destination, "rev-parse", "HEAD").stdout.strip() != target
            or local_git(destination, "status", "--porcelain=v1", "--untracked-files=all").stdout):
        raise RuntimeError("frozen target does not match the requested commit")
    return destination
