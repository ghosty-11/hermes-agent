"""Rollback byte and directory-metadata ownership, independent of capsule members."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.backup import _KEPT_CACHE_SUBDIRS


_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})


_GOOD_COPY = re.compile(r"config\.yaml\.good\.[0-9]{8}-[0-9]{6}\Z")


def regenerated_member(path: Path, *, kind: str, root: Path,
                       homes: tuple[Path, ...] | set[Path], entry_kind: str) -> bool:
    """Only producer-shaped regular records may change after seal."""
    if entry_kind != "file":
        return False
    if kind == "home":
        return any(path.parent == home / "backups" / "config"
                   and _GOOD_COPY.fullmatch(path.name) for home in homes)
    if kind == "state" and path.parent == root / "bootstrap" and path.suffix == ".json":
        from hermes_constants import PROFILE_ID_RE

        return path.stem == "default" or bool(PROFILE_ID_RE.fullmatch(path.stem))
    return False


def sqlite_file(path: Path) -> bool:
    if path.suffix.lower() in _SQLITE_SUFFIXES:
        return True
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"SQLite candidate is not a regular file: {path}")
        return os.read(fd, 16) == b"SQLite format 3\0"
    finally:
        os.close(fd)


def _sqlite_sidecar(path: Path) -> bool:
    if not path.name.endswith(("-wal", "-shm", "-journal")):
        return False
    if path.is_symlink() or not path.is_file():
        return False
    base = path.with_name(path.name.rsplit("-", 1)[0])
    return not base.is_symlink() and base.is_file() and sqlite_file(base)


@dataclass(frozen=True)
class CoverageOwner:
    kind: str
    root: Path
    include: tuple[Path, ...] | None
    excluded: tuple[Path, ...]
    homes: tuple[Path, ...] = ()

    def covers(self, path: Path, effect: str) -> bool:
        if self.kind == "metadata":
            return effect == "metadata" and path == self.root
        if path == self.root:
            return effect == "metadata"
        if not path.is_relative_to(self.root):
            return False
        if excluded_member(path, kind=self.kind, root=self.root, homes=self.homes,
                           excluded=self.excluded):
            return effect == "metadata" and self.kind == "home" and any(
                path == home / "logs" / "update_receipts" for home in self.homes)
        if self.include is not None and not any(
            path == chosen or path.is_relative_to(chosen) for chosen in self.include
        ):
            return False
        return True


def excluded_member(path: Path, *, kind: str, root: Path, homes: tuple[Path, ...] | set[Path],
                    excluded: tuple[Path, ...]) -> bool:
    """Apply the same byte exclusions to traversal and dependency ownership."""
    if (any(path == block or path.is_relative_to(block) for block in excluded)
            or retained_bookkeeping(path, resource_root=root if kind == "resource" else None)
            or _sqlite_sidecar(path)):
        return True
    if kind != "home":
        return False
    if path.is_relative_to(root / "local-update-capsules") or retained_path(path, homes):
        return True
    for home in homes:
        if path == home / "logs" or path.is_relative_to(home / "logs"):
            return True
        if (path.parent == home and path.name.endswith(".lock")
                and not path.is_symlink() and (not path.exists() or path.is_file())):
            return True
        # Backups are protected data, including producer-shaped preimages.
        cache = home / "cache"
        if path == cache:
            return True  # Traversed for kept children, but its metadata is operational.
        if path.is_relative_to(cache) and path.relative_to(cache).parts[0] not in _KEPT_CACHE_SUBDIRS:
            return True
    return False


def retained_bookkeeping(path: Path, *, resource_root: Path | None = None) -> bool:
    # Interpreters regenerate bytecode beside retained stdlib. Restore purges it;
    # excluded caches are never trusted as part of the resource image.
    return (resource_root is not None and (
                path.is_relative_to(resource_root / ".leases")
                or path.is_relative_to(resource_root)
                and "__pycache__" in path.relative_to(resource_root).parts)
            or path.name == ".prepare.lock" and path.parent.name == "pm-runtime"
            or path.name == ".install.lock" and path.parent.parent.name == "installs")

def _generation(name: str) -> bool:
    return name.startswith("gen-") and len(name) == 36 and all(
        digit in "0123456789abcdef" for digit in name[4:]
    )


def retained_path(path: Path, homes: tuple[Path, ...] | set[Path]) -> bool:
    """Keep only the defined audit and generation bookkeeping outside home images."""
    for home in homes:
        if path == home / ".hermes-update-in-progress":
            return True
        if path.is_relative_to(home / "logs" / "update_receipts"):
            return True
        if path.is_relative_to(home / "environments"):
            parts = path.relative_to(home / "environments").parts
            if len(parts) >= 2 and (
                parts[1] in {"active.json", ".install.lock"} or _generation(parts[1])
            ):
                return True
    return False


def retained_parent(path: Path, homes: tuple[Path, ...] | set[Path]) -> bool:
    """A parent with excluded bookkeeping survives even with an empty image."""
    for home in homes:
        if path == home / "environments":
            return any(child.is_dir() and not child.is_symlink()
                       and retained_parent(child, homes) for child in path.iterdir())
        if path.parent == home / "environments" and path.is_dir() and not path.is_symlink():
            return any(child.name in {"active.json", ".install.lock"} or _generation(child.name)
                       for child in path.iterdir())
    return False


def selection_exclusions(runtime: dict, install: Path,
                         homes: tuple[Path, ...] | list[Path]) -> tuple[Path, ...]:
    from hermes_constants import get_default_hermes_root
    from hermes_cli.update_host_obligation import host_obligation_path

    # Match pm.packages.uv_cache_dir without its cache-seeding side effect.
    uv_cache = get_default_hermes_root() / "cache" / "uv"
    obligation = host_obligation_path()
    inside = (obligation,) if obligation is not None and any(
        obligation.is_relative_to(home) for home in homes) else ()
    return tuple(dict.fromkeys(
        (install, runtime["state"], runtime["store"], runtime["writable"], uv_cache, *inside)))


def state_exclusions(runtime: dict, install: Path) -> tuple[Path, ...]:
    from pm.environments import runtime_facts_path

    state = runtime["state"]
    return (state / "environments", state / "pm-runtime" / "generations",
            runtime_facts_path(install), state / "pm-runtime" / "selected.json",
            runtime["store"] / "facts.json", runtime["writable"] / "facts.json")


def dependency_coverage(homes: list[Path], install: Path, products: tuple[Path, ...],
                        runtime: dict, resources: set[Path]) -> "CoverageInventory":
    """Use the same image filters as capture, including transitive exclusions."""
    roots = [home for home in homes if not any(
        home != other and home.is_relative_to(other) for other in homes)]
    owners = [CoverageOwner("home", home, None, selection_exclusions(runtime, install, homes),
                            tuple(homes)) for home in roots]
    owners.append(CoverageOwner("state", runtime["state"], (runtime["state"],),
                                state_exclusions(runtime, install), tuple(homes)))
    owners.append(CoverageOwner("product", install, products, (), tuple(homes)))
    owners.extend(CoverageOwner("resource", resource, (resource,), (), tuple(homes))
                  for resource in resources)
    return CoverageInventory(owners)


class CoverageInventory:
    def __init__(self, owners: list[CoverageOwner]):
        self.owners = owners

    def owner(self, path: Path, effect: str, phase: str) -> CoverageOwner:
        if effect not in {"bytes", "metadata"} or phase not in {"pre", "post"}:
            raise ValueError("invalid rollback coverage effect or phase")
        matched = [owner for owner in self.owners if owner.covers(path, effect)]
        # Selection and convenience creation anchors do not displace an image
        # owner for directory metadata. Identical anchors use a stable winner.
        if effect == "metadata" and matched:
            image = [owner for owner in matched if owner.kind not in {"selection", "convenience"}]
            if image:
                matched = image
            deepest = max(len(owner.root.parts) for owner in matched)
            matched = [owner for owner in matched if len(owner.root.parts) == deepest]
            if len(matched) > 1 and len({owner.root for owner in matched}) == 1:
                matched = [min(matched, key=lambda owner: (owner.kind, owner.include or ()))]
        if len(matched) != 1:
            raise ValueError(f"rollback {effect} has {len(matched)} owners: {path}")
        return matched[0]

    def byte_owners(self, path: Path) -> list[CoverageOwner]:
        return [owner for owner in self.owners if owner.kind != "metadata" and (
            owner.covers(path, "bytes")
            or (owner.root == path and owner.include is not None
                and path in owner.include and path.is_dir())
        )]

    def validate_overlaps(self) -> None:
        for owner in self.owners:
            if owner.kind == "metadata":
                continue
            for selected in owner.include or (owner.root,):
                for path in (selected, selected / "__rollback_owner_probe__"):
                    if len(self.byte_owners(path)) > 1:
                        raise ValueError(f"overlapping rollback byte owners: {path}")

    def member(self, owner: CoverageOwner, path: Path, kind: str, phase: str) -> None:
        effect = "metadata" if kind == "dir" else "bytes"
        if not owner.covers(path, effect) or self.owner(path, effect, phase) != owner:
            raise ValueError(f"rollback archive member is outside admitted coverage: {path}")
