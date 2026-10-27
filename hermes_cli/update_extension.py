"""Generic Root-pinned update extensions for native local updates.

A Root-prepared update-extension bundle extends one native local update with
private completion work the framework never interprets: the operator's guard
captures an opaque JSON payload into a caller-owned context file, and this
module only verifies structure, custody and pins, then executes the
descriptor-pinned entrypoint's fixed callbacks.

Trust is narrow, structural and POSIX-only: UID 0 owns the descriptor, every
bundle file and directory, and every lexical ancestor through ``/``; nothing
writable, foreign, linked or special takes part, and no path alias resolves
into trust. The entry bytes execute as one unique temporary package
namespace so relative imports resolve only descriptor-pinned siblings;
bytecode writes stay off and only that namespace is removed afterwards.

``load_update_extension_context`` runs after generic validation but before
backup, capsule capture or any native mutation; ``complete_update_extension``
runs only after ``_complete_selected`` succeeds and before any capsule seal,
so callback failures keep every actual effect and follow the native
strict-local settlement unchanged. This module contains no host policy and
adds no retry, restart, rollback or agent loop of its own.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

#: The only identity whose bundle bytes this framework will execute.
ROOT_UID = 0

DESCRIPTOR_NAME = "update-extension.json"
DESCRIPTOR_VERSION = 1
DESCRIPTOR_KEYS = ("version", "entrypoint", "files", "transaction")
TRANSACTION_KEYS = ("install_root", "home", "target")
CONTEXT_KEYS = ("descriptor_path", "descriptor_sha256", "payload")
CAPTURED_KEYS = (*CONTEXT_KEYS, "frozen")
PIN_FIELDS = ("sha256", "size", "mode")

# Bounded reads everywhere: a descriptor names its bytes by digest, so
# anything past a sane bound is a malformed pin, not evidence to grind.
DESCRIPTOR_MAX_BYTES = 1 << 18
CONTEXT_MAX_BYTES = 1 << 22
FILE_MAX_BYTES = 1 << 24
TOTAL_FILE_MAX_BYTES = 1 << 26
FILE_COUNT_MAX = 1024
DEPTH_MAX = 128

CONTEXT_MODE = 0o600

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_RE = re.compile(r"[0-9a-f]{40}\Z")


# --- shared file safety -------------------------------------------------------


def read_owned_file(path, label, *, owner: int, limit: int) -> tuple[bytes, os.stat_result]:
    """Read one link-free, single-link, owner-only, bounded regular file,
    fd-anchored so the bytes hashed are the bytes validated."""
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if info.st_nlink != 1:
        raise ValueError(f"{label} must not be hard-linked: {path}")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise ValueError(f"{label} must not be group or world writable: {path}")
    if info.st_uid != owner:
        raise ValueError(f"{label} is owned by {info.st_uid}, not {owner}: {path}")
    if info.st_size > limit:
        raise ValueError(f"{label} exceeds its size bound ({info.st_size} bytes): {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise ValueError(f"{label} changed while reading: {path}")
        if stat.S_IMODE(opened.st_mode) & 0o022:
            raise ValueError(f"{label} became group or world writable: {path}")
        chunks = []
        while sum(len(chunk) for chunk in chunks) <= limit:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise ValueError(f"{label} exceeds its size bound: {path}")
    return data, info


def require_protected_ancestry(directory, label, *, owner=None) -> None:
    """Every directory from ``/`` down to ``directory`` is real, protected-uid
    owned and not group/world writable. A writable or foreign component above
    the bundle could swap it between validation and use."""
    anchor = ROOT_UID if owner is None else owner
    current = Path(directory)
    if not current.is_absolute() or ".." in current.parts:
        raise ValueError(f"{label} ancestry must be an absolute normalized path")
    for component in reversed([current, *current.parents]):
        try:
            info = os.lstat(component)
        except OSError as exc:
            raise ValueError(f"{label} ancestry is unreadable: {component}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"{label} ancestry is not a real directory: {component}")
        if info.st_uid != anchor:
            raise ValueError(f"{label} ancestry is not owned by the protected uid: {component}")
        if stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError(f"{label} ancestry is group or world writable: {component}")


def walk_protected_tree(root, label, *, owner=None) -> tuple[set[str], set[str]]:
    """Every file and directory under ``root`` as relative POSIX paths,
    refusing links, special files and untrusted directories."""
    anchor = ROOT_UID if owner is None else owner
    files: set[str] = set()
    dirs: set[str] = set()
    require_protected_ancestry(root, label, owner=owner)
    pending = [(Path(root), "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = f"{prefix}{entry.name}"
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError(f"{label} contains a symlink: {relative}")
                    if stat.S_ISDIR(info.st_mode):
                        if info.st_uid != anchor or stat.S_IMODE(info.st_mode) & 0o022:
                            raise ValueError(f"{label} contains an untrusted directory: {relative}")
                        dirs.add(relative)
                        pending.append((Path(entry.path), f"{relative}/"))
                    elif stat.S_ISREG(info.st_mode):
                        files.add(relative)
                    else:
                        raise ValueError(f"{label} contains a non-regular file: {relative}")
        except OSError as exc:
            raise ValueError(f"{label} is unreadable: {directory}") from exc
    return files, dirs


# --- pinned-bundle validation -------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relative(value, label) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} escapes its root: {value}")
    if any(part in {"", "."} for part in path.parts) or path.as_posix() != value:
        raise ValueError(f"{label} must be normalized: {value}")
    return path


def _require_pin(value, label) -> None:
    if not isinstance(value, dict) or set(value) != set(PIN_FIELDS):
        raise ValueError(f"{label} must record exactly sha256, size and mode")
    if not isinstance(value["sha256"], str) or not _SHA256_RE.fullmatch(value["sha256"]):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    for field in ("size", "mode"):
        if type(value[field]) is not int or value[field] < 0:
            raise ValueError(f"{label}.{field} must be a nonnegative integer")
    if value["mode"] > 0o777 or value["mode"] & (0o7000 | 0o022):
        raise ValueError(f"{label}.mode must be a plain non-writable permission mask")


def _require_parent_closure(pins: dict, actual_dirs: set[str], label: str) -> None:
    needed: set[str] = set()
    for key in pins:
        for parent in PurePosixPath(key).parents:
            if parent.as_posix() != ".":
                needed.add(parent.as_posix())
    if actual_dirs != needed:
        extra = sorted(actual_dirs - needed)
        missing = sorted(needed - actual_dirs)
        raise ValueError(f"{label} directory set does not match the pinned file set "
                         f"(extra: {extra}, missing: {missing})")


def _verify_pinned_file(path: Path, pin: dict, label: str, *, owner: int, limit: int) -> bytes:
    data, info = read_owned_file(path, label, owner=owner, limit=limit)
    if _sha256(data) != pin["sha256"]:
        raise ValueError(f"{label} bytes do not match the descriptor pin: {path}")
    if info.st_size != pin["size"]:
        raise ValueError(f"{label} size does not match the descriptor pin: {path}")
    if stat.S_IMODE(info.st_mode) != pin["mode"]:
        raise ValueError(f"{label} mode does not match the descriptor pin: {path}")
    return data


def _normalized_absolute(value, label) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be an absolute path string")
    normalized = os.path.normpath(value)
    if (not os.path.isabs(normalized) or ".." in Path(normalized).parts
            or normalized != value):
        raise ValueError(f"{label} must be a normalized absolute path: {value}")
    return normalized


def _load_descriptor(descriptor_path, *, install_root: str, home: str, target: str,
                     owner: int) -> tuple[dict, bytes, dict[str, str]]:
    """Fully validate one Root-pinned bundle: ancestry, the exact file set,
    every pin, the fixed transaction and the descriptor's own bytes.

    Returns the parsed descriptor, its bytes (for the caller's digest pin)
    and the verified UTF-8 source of every pinned ``.py`` member."""
    if not isinstance(descriptor_path, str) or not descriptor_path:
        raise ValueError("update-extension context must name the descriptor path")
    path = Path(descriptor_path)
    if path.name != DESCRIPTOR_NAME:
        raise ValueError(f"update-extension descriptor must be named {DESCRIPTOR_NAME}")
    directory = path.parent
    require_protected_ancestry(directory, "update-extension bundle", owner=owner)
    data, _info = read_owned_file(path, "update-extension descriptor", owner=owner,
                                  limit=DESCRIPTOR_MAX_BYTES)
    try:
        descriptor = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("update-extension descriptor is not valid JSON") from exc
    if not isinstance(descriptor, dict) or set(descriptor) != set(DESCRIPTOR_KEYS):
        raise ValueError("update-extension descriptor must record exactly "
                         "version, entrypoint, files and transaction")
    if type(descriptor["version"]) is not int or descriptor["version"] != DESCRIPTOR_VERSION:
        raise ValueError("update-extension descriptor version is not supported")
    entrypoint = _safe_relative(descriptor["entrypoint"], "update-extension entrypoint")
    if entrypoint.suffix != ".py":
        raise ValueError("update-extension entrypoint must be a relative .py path")
    module_parts = (*entrypoint.parts[:-1], entrypoint.stem)
    if any(not part.isidentifier() for part in module_parts):
        raise ValueError("update-extension entrypoint must be a plain module path")
    files = descriptor["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("update-extension descriptor must pin a non-empty file set")
    if len(files) > FILE_COUNT_MAX:
        raise ValueError("update-extension descriptor exceeds its file count bound")
    total = 0
    for name, pin in files.items():
        relative = _safe_relative(name, "update-extension file pin")
        if relative.as_posix() == DESCRIPTOR_NAME:
            raise ValueError("update-extension descriptor must not pin itself")
        if len(relative.parts) > DEPTH_MAX:
            raise ValueError(f"update-extension file pin exceeds its depth bound: {name}")
        _require_pin(pin, f"update-extension file pin {name}")
        total += pin["size"]
    if total > TOTAL_FILE_MAX_BYTES:
        raise ValueError("update-extension descriptor exceeds its total size bound")
    if entrypoint.as_posix() not in files:
        raise ValueError("update-extension entrypoint must occur in the pinned files")
    transaction = descriptor["transaction"]
    if not isinstance(transaction, dict) or set(transaction) != set(TRANSACTION_KEYS):
        raise ValueError("update-extension transaction must record exactly "
                         "install_root, home and target")
    if (_normalized_absolute(transaction["install_root"],
                             "update-extension transaction install_root") != install_root):
        raise ValueError("update-extension transaction install_root does not match this update")
    if _normalized_absolute(transaction["home"], "update-extension transaction home") != home:
        raise ValueError("update-extension transaction home does not match this update")
    if (not isinstance(transaction["target"], str)
            or not _TARGET_RE.fullmatch(transaction["target"])):
        raise ValueError("update-extension transaction target must be full lowercase 40-hex")
    if transaction["target"] != target:
        raise ValueError("update-extension transaction target does not match this update")
    actual_files, actual_dirs = walk_protected_tree(directory, "update-extension bundle",
                                                    owner=owner)
    if actual_files - {DESCRIPTOR_NAME} != set(files):
        raise ValueError("update-extension bundle does not match the pinned file set exactly")
    _require_parent_closure(files, actual_dirs, "update-extension bundle")
    sources: dict[str, str] = {}
    identities: dict[str, str] = {}
    for name, pin in files.items():
        member = directory / PurePosixPath(name)
        payload = _verify_pinned_file(member, pin, f"update-extension file {name}",
                                      owner=owner, limit=FILE_MAX_BYTES)
        if member.suffix == ".py":
            try:
                sources[name] = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"update-extension file {name} is not UTF-8 Python source") from exc
            # Two pinned files mapping to one import identity would silently
            # execute only one file's bytes; refuse the collision before any
            # bundle code runs.
            identity = ".".join(_module_parts(name))
            first = identities.setdefault(identity, name)
            if first != name:
                raise ValueError(f"update-extension files {first} and {name} both map "
                                 f"to import identity {identity!r}")
    if descriptor["entrypoint"] not in sources:
        raise ValueError("update-extension entrypoint is not a pinned Python file")
    return descriptor, data, sources


# --- pinned entrypoint execution ----------------------------------------------


def _module_parts(relative) -> tuple[str, ...]:
    """The module path a pinned member maps to: an initializer is its own
    package (the bundle root's maps to the namespace itself)."""
    path = PurePosixPath(relative)
    if path.name == "__init__.py":
        return path.parts[:-1]
    return (*path.parts[:-1], path.stem)


class _PinnedBundleFinder:
    """meta_path finder serving only descriptor-pinned Python members.

    Relative imports inside the entry resolve through the normal import
    machinery, but every ``<package>...`` name is answered from the verified
    bundle bytes alone: nothing else exists in the namespace. A pinned
    ``__init__.py`` is its own package — the bundle root's initializer
    included — so importing a package executes its verified source."""

    def __init__(self, package: str, sources: dict[str, str]):
        self._package = package
        self._origin = f"<{package}>"
        members: dict[str, str] = {}
        member_paths: dict[str, str] = {}
        packages: set[str] = set()
        for name, source in sources.items():
            path = PurePosixPath(name)
            member = ".".join(_module_parts(name))
            members[member] = source
            member_paths[member] = name
            for depth in range(1, len(path.parts)):
                packages.add(".".join(path.parts[:depth]))
        self._members = members
        self._member_paths = member_paths
        self._packages = packages

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._package:
            spec = importlib.util.spec_from_loader(fullname, self, is_package=True)
            spec.submodule_search_locations = [self._origin]
            return spec
        if not fullname.startswith(self._package + "."):
            return None
        member = fullname[len(self._package) + 1:]
        if member not in self._members:
            if member not in self._packages:
                return None
            spec = importlib.util.spec_from_loader(fullname, self, is_package=True)
            spec.submodule_search_locations = [self._origin]
            return spec
        is_package = member in self._packages
        spec = importlib.util.spec_from_loader(fullname, self, is_package=is_package)
        if is_package:
            spec.submodule_search_locations = [self._origin]
        return spec

    def create_module(self, spec):
        return None  # default module creation

    def exec_module(self, module):
        member = module.__name__[len(self._package) + 1:]
        source = self._members.get(member)
        if source is None:
            return  # a directory package without a pinned initializer
        module.__file__ = f"{self._origin}/{self._member_paths[member]}"
        code = compile(source, module.__file__, "exec")
        exec(code, module.__dict__)


@contextmanager
def _pinned_entry(sources: dict[str, str], entrypoint: str):
    """Execute the verified entry bytes as one unique temporary package
    namespace; restore interpreter state and remove only that namespace."""
    package = f"hermes_update_extension_{uuid.uuid4().hex}"
    finder = _PinnedBundleFinder(package, sources)
    module_name = ".".join((package, *_module_parts(entrypoint)))
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.dont_write_bytecode = previous_bytecode
        try:
            sys.meta_path.remove(finder)
        except ValueError:  # pragma: no cover — the finder is ours to remove
            pass
        for name in [name for name in sys.modules
                     if name == package or name.startswith(package + ".")]:
            del sys.modules[name]


def _require_callback(module, name: str):
    callback = getattr(module, name, None)
    if not callable(callback):
        raise ValueError(f"update-extension entrypoint does not define callable {name}")
    return callback


def _json_object(value, label) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must return a JSON object")
    try:
        snapshotted = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} returned a value that is not JSON") from exc
    if snapshotted != value:
        raise ValueError(f"{label} returned a value that is not JSON-stable")
    return snapshotted


def _record_fact(key: str, value) -> None:
    from hermes_cli.update_receipt import record_fact

    record_fact(key, value)


# --- the two native phases ----------------------------------------------------


def load_update_extension_context(path, *, install_root, home, target, frozen) -> dict:
    """Validate the caller-owned correlated context and the whole Root-pinned
    bundle, run the entrypoint's read-only admission, and return the captured
    JSON context (exact input keys plus ``frozen``) for the native request.

    Any refusal fails the update before backup, capsule capture or any other
    native mutation; the context file's own path is never read again."""
    root_value = _normalized_absolute(str(install_root), "update-extension install root")
    home_value = _normalized_absolute(str(home), "update-extension home")
    if not isinstance(target, str) or not _TARGET_RE.fullmatch(target):
        raise ValueError("update-extension context requires a full lowercase 40-hex target")
    data, info = read_owned_file(path, "update-extension context", owner=os.geteuid(),
                                 limit=CONTEXT_MAX_BYTES)
    if stat.S_IMODE(info.st_mode) != CONTEXT_MODE:
        raise ValueError("update-extension context must be owner-only (0600)")
    try:
        context = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("update-extension context is not valid JSON") from exc
    if not isinstance(context, dict) or set(context) != set(CONTEXT_KEYS):
        raise ValueError("update-extension context must record exactly "
                         "descriptor_path, descriptor_sha256 and payload")
    if not isinstance(context["descriptor_path"], str) or not context["descriptor_path"]:
        raise ValueError("update-extension context must name the descriptor path")
    if (not isinstance(context["descriptor_sha256"], str)
            or not _SHA256_RE.fullmatch(context["descriptor_sha256"])):
        raise ValueError("update-extension context must pin the descriptor by SHA-256")
    if not isinstance(context["payload"], dict):
        raise ValueError("update-extension context payload must be a JSON object")
    descriptor, descriptor_bytes, sources = _load_descriptor(
        context["descriptor_path"], install_root=root_value, home=home_value,
        target=target, owner=ROOT_UID)
    if _sha256(descriptor_bytes) != context["descriptor_sha256"]:
        raise ValueError("update-extension descriptor bytes do not match the context pin")
    try:
        with _pinned_entry(sources, descriptor["entrypoint"]) as module:
            admitted = _require_callback(module, "validate_update")(
                context["payload"], install_root=Path(root_value), home=Path(home_value),
                target=target, frozen=Path(frozen))
    except SystemExit as exc:
        raise ValueError("update-extension validate_update exited instead of "
                         f"returning: {exc.code!r}") from exc
    captured = _json_object(admitted, "update-extension validate_update")
    return {"descriptor_path": context["descriptor_path"],
            "descriptor_sha256": context["descriptor_sha256"],
            "payload": captured, "frozen": str(frozen)}


def complete_update_extension(captured, *, install_root, home, target) -> dict:
    """Re-verify the captured context and the whole Root-pinned bundle, then
    run the entrypoint's ``complete_update`` and record its actual facts.

    Runs only after ``_complete_selected`` succeeds and before any capsule
    seal. Callback exceptions keep every actual effect and follow the native
    strict-local settlement unchanged."""
    if not isinstance(captured, dict) or set(captured) != set(CAPTURED_KEYS):
        raise ValueError("captured update-extension context must record exactly "
                         "descriptor_path, descriptor_sha256, payload and frozen")
    if not isinstance(captured["descriptor_path"], str) or not captured["descriptor_path"]:
        raise ValueError("captured update-extension context must name the descriptor path")
    if (not isinstance(captured["descriptor_sha256"], str)
            or not _SHA256_RE.fullmatch(captured["descriptor_sha256"])):
        raise ValueError("captured update-extension context must pin the descriptor by SHA-256")
    if not isinstance(captured["payload"], dict):
        raise ValueError("captured update-extension payload must be a JSON object")
    frozen = Path(captured["frozen"])
    if not frozen.is_dir():
        raise ValueError("update-extension context names a frozen root that is gone")
    root_value = _normalized_absolute(str(install_root), "update-extension install root")
    home_value = _normalized_absolute(str(home), "update-extension home")
    if not isinstance(target, str) or not _TARGET_RE.fullmatch(target):
        raise ValueError("update-extension completion requires a full lowercase 40-hex target")
    descriptor, descriptor_bytes, sources = _load_descriptor(
        captured["descriptor_path"], install_root=root_value, home=home_value,
        target=target, owner=ROOT_UID)
    if _sha256(descriptor_bytes) != captured["descriptor_sha256"]:
        raise ValueError("update-extension descriptor bytes do not match the captured pin")
    try:
        with _pinned_entry(sources, descriptor["entrypoint"]) as module:
            applied = _require_callback(module, "complete_update")(
                captured["payload"], install_root=Path(root_value), home=Path(home_value),
                target=target, frozen=frozen)
    except SystemExit as exc:
        if isinstance(exc.code, int) and exc.code:
            raise  # a nonzero status keeps the native SystemExit settlement
        raise ValueError("update-extension complete_update exited instead of "
                         f"returning: {exc.code!r}") from exc
    facts = _json_object(applied, "update-extension complete_update")
    _record_fact("update_extension_descriptor", captured["descriptor_path"])
    _record_fact("update_extension_descriptor_sha256", captured["descriptor_sha256"])
    _record_fact("update_extension_facts", facts)
    return facts
