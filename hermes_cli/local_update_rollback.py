"""Explicit, offline code-and-profile rollback for a local-source update.

The capsule is private to this host. Installed Git attests its captured inventory
and sealed postimages; neither substitutes for stopping writers at both boundaries.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from contextlib import ExitStack, contextmanager

from hermes_cli.backup_sqlite import _safe_copy_db
from hermes_cli.local_update_attestation import (
    publish as publish_attestation, sync_capsule_tree, sync_restored_git,
    verify as verify_attestation,
)
from hermes_cli.local_update_coverage import (
    CoverageInventory, CoverageOwner, excluded_member, regenerated_member, retained_parent,
    sqlite_file, selection_exclusions as _selection_exclusions,
    state_exclusions as _state_exclusions,
)
from hermes_cli.update_host_obligation import host_obligation_path
from hermes_cli.local_update_runtime import (
    PRODUCTS, dependency_closure, inventory, mutation_roots, owned_conveniences,
)
from hermes_cli.update_local import local_git

_CAPSULE_DIR = "local-update-capsules"

# Directory mtimes are restored after all selections; regular-file mtimes
# remain capture and drift sentinels.


def _attributes(path: Path) -> bytes:
    if not all(hasattr(os, name) for name in ("listxattr", "getxattr", "setxattr", "removexattr")):
        raise ValueError("non-following extended attribute support is required for complete rollback")
    return json.dumps({
        name: base64.b64encode(os.getxattr(path, name, follow_symlinks=False)).decode("ascii")
        for name in sorted(os.listxattr(path, follow_symlinks=False))
    }, sort_keys=True).encode("utf-8")


def _metadata(path: Path, artifact: Path | None = None) -> dict:
    st = path.lstat()
    raw = _attributes(path)
    result = {"uid": st.st_uid, "gid": st.st_gid, "xattrs_sha256": hashlib.sha256(raw).hexdigest(),
              "xattrs_size": len(raw)}
    if artifact is not None:
        with artifact.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
        result["xattrs_object"] = artifact.name
    return result


def _restore_metadata(path: Path, item: dict, artifact: Path) -> None:
    attrs = json.loads(artifact.read_bytes())
    current = set(os.listxattr(path, follow_symlinks=False))
    for name in current - attrs.keys():
        os.removexattr(path, name, follow_symlinks=False)
    if (path.lstat().st_uid, path.lstat().st_gid) != (item["uid"], item["gid"]):
        os.chown(path, item["uid"], item["gid"], follow_symlinks=False)
    if item["kind"] != "link":
        path.chmod(item["mode"])
    for name, value in attrs.items():
        os.setxattr(path, name, base64.b64decode(value), follow_symlinks=False)


def _restorable_identity(path: Path, item: dict) -> None:
    if not hasattr(os, "geteuid") or not hasattr(os, "chown"):
        raise ValueError("owner restoration is unavailable on this platform")
    if item["uid"] != os.geteuid() and os.geteuid() != 0:
        raise ValueError(f"cannot restore owner of {path}")
    if item["gid"] not in {os.getegid(), *os.getgroups()} and os.geteuid() != 0:
        raise ValueError(f"cannot restore group of {path}")


def _preflight_metadata(path: Path, item: dict, artifact: Path) -> None:
    """Prove owner/group and filesystem attribute writes before admitting checkout."""
    _restorable_identity(path, item)
    parent = path.parent.stat()
    with ExitStack() as stack:
        if parent.st_uid == os.geteuid():
            stack.callback(os.utime, path.parent, ns=(parent.st_atime_ns, parent.st_mtime_ns),
                           follow_symlinks=False)
        probe_dir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=".rollback-probe-", dir=path.parent))
        probe = Path(probe_dir) / "entry"
        if item["kind"] == "dir":
            probe.mkdir()
        elif item["kind"] == "link":
            probe.symlink_to(item["target"])
        else:
            probe.touch()
        try:
            _restore_metadata(probe, item, artifact)
        except OSError as exc:
            raise ValueError(f"cannot restore attributes of {path}: {exc}") from exc
        if (probe.lstat().st_uid, probe.lstat().st_gid) != (item["uid"], item["gid"]):
            raise ValueError(f"cannot restore owner/group of {path}")
        if item["kind"] != "link" and stat.S_IMODE(probe.lstat().st_mode) != item["mode"]:
            raise ValueError(f"cannot restore mode and attributes of {path}")
        if hashlib.sha256(_attributes(probe)).hexdigest() != item["xattrs_sha256"]:
            raise ValueError(f"cannot restore exact attributes of {path}")


def _git(repo: Path, *args: str, input: str | None = None) -> str:
    return local_git(repo, *args, input=input).stdout.strip()


def _git_bytes(repo: Path, *args: str, input: bytes | None = None) -> bytes:
    return local_git(repo, *args, binary=True, input=input).stdout


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _integrity(path: Path) -> None:
    # The ordinary backup policy substitutes a header/schema probe above 2 GiB.
    from hermes_cli.backup import verify_sqlite_integrity
    result = verify_sqlite_integrity(path, max_bytes=0)
    if not result["valid"]:
        raise ValueError(f"SQLite image is not intact: {path}: {result['message']}")


def _homes(invoking: Path) -> list[Path]:
    from hermes_cli.profiles import _PROFILE_ID_RE

    if invoking.is_symlink():
        raise ValueError("invoking profile must be a real directory")
    invoking = invoking.resolve(strict=True)
    if not invoking.is_dir():
        raise ValueError("invoking profile must be a real directory")
    root = invoking.parent.parent if invoking.parent.name == "profiles" else invoking
    homes = [root]
    profiles = root / "profiles"
    if profiles.is_symlink():
        raise ValueError("profile directory is a symlink")
    if profiles.is_dir():
        for entry in sorted(profiles.iterdir()):
            if entry.name != "default" and _PROFILE_ID_RE.fullmatch(entry.name) and entry.is_dir():
                if entry.is_symlink():
                    raise ValueError(f"profile home is a symlink: {entry}")
                homes.append(entry)
    if invoking not in homes:
        homes.append(invoking)
    return [p.resolve(strict=True) for p in homes]


def _roots(homes: list[Path]) -> list[Path]:
    return [home for home in homes if not any(home != other and home.is_relative_to(other) for other in homes)]


def _quiescent(homes: list[Path]) -> None:
    from hermes_cli.update_inventory import collect_runtime_inventory
    from hermes_cli.data_cleanup_holders import _refuse_backend_writers, _refuse_cron_writers
    from hermes_cli.active_sessions import _prune_dead, _read_entries
    from gateway.control_socket import identify_gateway
    from gateway.status import get_running_pid_identity_strict

    plan = collect_runtime_inventory()
    if plan.runtimes:
        raise RuntimeError("local rollback requires every affected runtime to be stopped")
    for home in homes:
        _refuse_backend_writers(home)
        _refuse_cron_writers(home)
        if identify_gateway(home) is not None or get_running_pid_identity_strict(home / "gateway.pid") is not None:
            raise RuntimeError(f"gateway still owns {home}")
        sessions = home / "runtime" / "active_sessions.json"
        if sessions.exists() and _prune_dead(_read_entries(sessions, strict=True), strict=True):
            raise RuntimeError(f"active sessions still own {home}")


def _walk(root: Path, homes: set[Path], install: Path, *, include: tuple[Path, ...] | None = None,
          excluded: tuple[Path, ...] = (), retained_dirs: frozenset[Path] | None = None,
          resource: bool = False):
    """Yield non-following entries inside explicitly owned mutable paths."""
    obligation = host_obligation_path() if include is None and excluded else None
    def visit(directory: Path):
        for child in sorted(directory.iterdir()):
            if excluded_member(child, kind="resource" if resource else "home" if include is None else "other",
                               root=root, homes=homes, excluded=excluded):
                if include is None:
                    for home in homes:
                        if child == home / "logs":
                            receipt = child / "update_receipts"
                            if (child.is_dir() and not child.is_symlink()
                                    and receipt.is_dir() and not receipt.is_symlink()
                                    and (retained_dirs is None or receipt in retained_dirs)):
                                yield receipt.relative_to(root), receipt, receipt.lstat().st_mode
                            break
                        if child == home / "cache":
                            if child.is_dir() and not child.is_symlink():
                                yield from visit(child)
                            break
                continue
            if include is None and child == install:
                continue
            if include is not None and not any(child == chosen or child.is_relative_to(chosen)
                                                or chosen.is_relative_to(child) for chosen in include):
                continue
            rel = child.relative_to(root)
            mode = child.lstat().st_mode
            if include is None and stat.S_ISDIR(mode) and retained_parent(child, homes):
                # Bookkeeping parents survive even if their image is otherwise empty.
                # Newly created parents remain live without becoming deletion candidates.
                if retained_dirs is None or child in retained_dirs:
                    yield rel, child, mode
                yield from visit(child)
                continue
            if stat.S_ISDIR(mode):
                if ((include is None or any(child == chosen or child.is_relative_to(chosen) for chosen in include))
                        and not any(blocked.is_relative_to(child) and blocked != obligation
                                    for blocked in excluded)):
                    yield rel, child, mode
                yield from visit(child)
            elif stat.S_ISREG(mode) or stat.S_ISLNK(mode):
                yield rel, child, mode
            else:
                raise ValueError(f"unsupported mutable data entry: {child}")
    yield from visit(root)


def _image(root: Path, homes: set[Path], install: Path, objects: Path | None = None,
           *, include: tuple[Path, ...] | None = None, excluded: tuple[Path, ...] = (),
           retained_dirs: frozenset[Path] | None = None,
           immutable: bool = False) -> list[dict]:
    entries = []
    if objects is not None:
        objects.mkdir(mode=0o700)
    for rel, path, mode in _walk(root, homes, install, include=include, excluded=excluded,
                                 retained_dirs=retained_dirs, resource=immutable):
        record = {"path": rel.as_posix(), "mode": stat.S_IMODE(mode),
                  "mtime_ns": path.lstat().st_mtime_ns}
        if stat.S_ISDIR(mode):
            record["kind"] = "dir"
            record.update(_metadata(path, objects / f"x-{len(entries)}" if objects is not None else None))
        elif stat.S_ISLNK(mode):
            record.update(kind="link", target=os.readlink(path))
            record.update(_metadata(path, objects / f"x-{len(entries)}" if objects is not None else None))
        else:
            record["kind"] = "db" if sqlite_file(path) else "file"
            record.update(_metadata(path, objects / f"x-{len(entries)}" if objects is not None else None))
            before = path.stat(follow_symlinks=False)
            if before.st_nlink != 1 and (not immutable or record["kind"] == "db"):
                raise ValueError(f"multiply linked mutable file cannot be rolled back: {path}")
            if objects is not None:
                image = objects / str(len(entries))
                if record["kind"] == "db":
                    if not _safe_copy_db(path, image):
                        raise ValueError(f"cannot snapshot SQLite database: {path}")
                    _integrity(image)
                else:
                    src_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    try:
                        opened = os.fstat(src_fd)
                        if (opened.st_dev, opened.st_ino, opened.st_size) != (
                                before.st_dev, before.st_ino, before.st_size):
                            raise ValueError(f"mutable data changed during capture: {path}")
                        dst_fd = os.open(image, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                        with os.fdopen(src_fd, "rb", closefd=False) as source_stream, os.fdopen(dst_fd, "wb") as output:
                            os.fchmod(output.fileno(), 0o600)
                            remaining = before.st_size
                            while remaining:
                                block = source_stream.read(min(1 << 20, remaining))
                                if not block:
                                    raise ValueError(f"mutable data truncated during capture: {path}")
                                output.write(block)
                                remaining -= len(block)
                            if source_stream.read(1):
                                raise ValueError(f"mutable data grew during capture: {path}")
                    finally:
                        os.close(src_fd)
                record["object"] = image.name
                record["sha256"] = _digest(image)
                record["size"] = image.stat().st_size
            else:
                if record["kind"] == "db":
                    with tempfile.TemporaryDirectory() as temp:
                        snap = Path(temp) / "db"
                        if not _safe_copy_db(path, snap):
                            raise ValueError(f"cannot read SQLite database: {path}")
                        _integrity(snap)
                        record["sha256"] = _digest(snap)
                        record["size"] = snap.stat().st_size
                else:
                    record["sha256"] = _digest(path)
                    record["size"] = before.st_size
            after = path.stat(follow_symlinks=False)
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise ValueError(f"mutable data changed during capture: {path}")
            if _metadata(path)["xattrs_sha256"] != record["xattrs_sha256"]:
                raise ValueError(f"mutable metadata changed during capture: {path}")
        entries.append(record)
    return entries


@contextmanager
def _directory_fd(name, parent=None):
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                 **({"dir_fd": parent} if parent is not None else {}))
    try:
        yield fd
    finally:
        os.close(fd)


def _ensure_host_obligation_parent(path: Path) -> None:
    """Create the external selection anchor without following existing components."""
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"host obligation path has no absolute creation anchor: {path}")
    with ExitStack() as stack:
        parent = stack.enter_context(_directory_fd(Path("/")))
        for component in path.parent.parts[1:]:
            try:
                os.mkdir(component, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            try:
                parent = stack.enter_context(_directory_fd(component, parent))
            except OSError as exc:
                raise ValueError(
                    f"host obligation owner has a symlink or non-directory component: {path.parent}"
                ) from exc


class _CapsuleAuthority:
    """Hold the selected capsule and every non-following ancestor through publication."""

    def __init__(self, manifest: Path, stack: ExitStack, *, create: bool = False):
        if (not manifest.is_absolute() or manifest.name != "manifest.json"
                or len(manifest.parent.name) != 32
                or any(c not in "0123456789abcdef" for c in manifest.parent.name)):
            raise ValueError("invalid rollback manifest path")
        self.path = manifest
        self.storage = manifest.parent.parent
        self.chain = []
        parent = stack.enter_context(_directory_fd(Path("/")))
        for index, component in enumerate(self.storage.parts[1:], 1):
            if create and index == len(self.storage.parts) - 1:
                try:
                    os.mkdir(component, 0o700, dir_fd=parent)
                except FileExistsError:
                    pass
            fd = stack.enter_context(_directory_fd(component, parent))
            self.chain.append((parent, component, fd))
            parent = fd
        self.storage_fd = parent
        if create:
            os.mkdir(manifest.parent.name, 0o700, dir_fd=parent)
        self.capsule_fd = stack.enter_context(_directory_fd(manifest.parent.name, parent))
        self._check()

    def _check(self) -> None:
        for parent, name, fd in (*self.chain,
                                 (self.storage_fd, self.path.parent.name, self.capsule_fd)):
            named = os.stat(name, dir_fd=parent, follow_symlinks=False)
            held = os.fstat(fd)
            if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
                raise ValueError("rollback capsule namespace changed")
        for fd in (self.storage_fd, self.capsule_fd):
            st = os.fstat(fd)
            if st.st_uid != os.geteuid() or st.st_mode & 0o077:
                raise ValueError("rollback capsule storage must be owner-only")


@contextmanager
def _capsule_authority(manifest: Path, *, create: bool = False):
    with ExitStack() as stack:
        yield _CapsuleAuthority(manifest, stack, create=create)


@contextmanager
def _producer_staging(authority: _CapsuleAuthority):
    authority._check()
    name = ".rollback-producer-" + uuid.uuid4().hex
    os.mkdir(name, 0o700, dir_fd=authority.storage_fd)
    with _directory_fd(name, authority.storage_fd) as fd:
        owned = os.fstat(fd)
        try:
            yield authority.storage / name
            authority._check()
        finally:
            try:
                authority._check()
                named = os.stat(name, dir_fd=authority.storage_fd, follow_symlinks=False)
            except (OSError, ValueError):
                pass  # Unknown namespace: leave it untouched.
            else:
                if (named.st_dev, named.st_ino) == (owned.st_dev, owned.st_ino):
                    shutil.rmtree(authority.storage / name)


def _write_manifest(authority: _CapsuleAuthority, data: dict) -> None:
    authority._check()
    name = ".manifest-" + uuid.uuid4().hex
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600,
                 dir_fd=authority.storage_fd)
    staged = os.fstat(fd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(data, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        authority._check()
        named = os.stat(name, dir_fd=authority.storage_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (staged.st_dev, staged.st_ino):
            raise ValueError("rollback manifest stage changed")
        os.rename(name, "manifest.json", src_dir_fd=authority.storage_fd,
                  dst_dir_fd=authority.capsule_fd)
        published = os.stat("manifest.json", dir_fd=authority.capsule_fd,
                            follow_symlinks=False)
        if (published.st_dev, published.st_ino) != (staged.st_dev, staged.st_ino):
            raise ValueError("rollback manifest publication changed")
        os.fsync(authority.capsule_fd)
        for _, _, directory in reversed(authority.chain):
            os.fsync(directory)
        authority._check()
    except Exception:
        try:
            authority._check()
            named = os.stat(name, dir_fd=authority.storage_fd, follow_symlinks=False)
        except (OSError, ValueError):
            pass  # Keep unknown or already-renamed staging names.
        else:
            if (named.st_dev, named.st_ino) == (staged.st_dev, staged.st_ino):
                os.unlink(name, dir_fd=authority.storage_fd)
        raise


def _read_manifest(authority: _CapsuleAuthority) -> dict:
    authority._check()
    try:
        fd = os.open("manifest.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=authority.capsule_fd)
    except OSError as exc:
        raise ValueError("invalid rollback manifest") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("invalid rollback manifest")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as stream:
            data = json.load(stream)
        if data["version"] != 4 or data["state"] not in {"captured", "sealed", "restored"}:
            raise ValueError("unknown rollback capsule format or state")
        if Path(data["capsule"]) != authority.path.parent:
            raise ValueError("rollback manifest path mismatch")
        if not isinstance(data["roots"], list) or not isinstance(data["homes"], list):
            raise ValueError("invalid rollback roots")
        authority._check()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("malformed rollback capsule") from exc
    finally:
        os.close(fd)
    return data


def _admitted_resource(path: Path, runtime: dict, homes: list[Path], install: Path) -> bool:
    from pm.environments import base_venv

    state = runtime["state"]
    if path.parent in {state / "environments", state / "pm-runtime" / "generations"}:
        return path.name not in {".", "..", "facts.json"}
    for store in (runtime["store"], runtime["writable"]):
        if path.parent == store and path.name not in {".", "..", "facts.json"}:
            return True
    for home in homes:
        if (path.parent.parent == home / "environments"
                and path.name.startswith("gen-") and len(path.name) == 36
                and all(c in "0123456789abcdef" for c in path.name[4:])):
            return True
    return path == base_venv(install) and path.is_relative_to(install)


def _record_root(path: Path, kind: str) -> Path:
    """The existing creation anchor for one independently owned path."""
    root = path
    if kind in {"state", "selection", "convenience"}:
        while not root.is_dir():
            if root.is_symlink() or root == root.parent:
                raise ValueError(f"rollback path has no real creation anchor: {path}")
            root = root.parent
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError(f"rollback root must be a real directory: {root}")
    return root


def _admitted_command_root(path: Path, root: Path) -> bool:
    from hermes_constants import get_default_hermes_root

    default = get_default_hermes_root()
    boundary = (Path.home() if path.is_relative_to(Path.home()) else
                default if path.is_relative_to(default) else path.parent)
    return root.is_relative_to(boundary) and path.is_relative_to(root)


def _coverage(data: dict) -> CoverageInventory:
    profile_homes = tuple(Path(path) for path in data["homes"])
    owners = [
        CoverageOwner(record["kind"], Path(record["path"]),
                      _filters(record)["include"], _filters(record)["excluded"], profile_homes)
        for record in data["roots"]
    ]
    owners.extend(CoverageOwner("new_selections", Path(item["path"]).parent,
                                (Path(item["path"]),), (), profile_homes)
                  for item in data.get("new_selections", ()))
    return CoverageInventory(owners)


def _check_members(data: dict) -> None:
    coverage = _coverage(data)
    coverage.validate_overlaps()
    for record, owner in zip(data["roots"], coverage.owners):
        for phase in ("pre", "post"):
            if phase in record:
                _validate_members(record[phase])
                for item in record[phase]:
                    coverage.member(owner, owner.root / item["path"], item["kind"], phase)
    for item, owner in zip(data.get("new_selections", ()),
                           coverage.owners[len(data["roots"]):]):
        _validate_members(item["post"])
        for member in item["post"]:
            coverage.member(owner, owner.root / member["path"], member["kind"], "post")


def _admit_records(data: dict, install: Path, invoking: Path, *,
                   expected_resources: set[Path]) -> list[Path]:
    """A capsule describes bytes, not authority over paths or inventory filters."""
    from pm.environments import dependency_home_root, runtime_facts_path
    from hermes_constants import get_default_hermes_root

    homes = _homes(invoking)
    if data["homes"] != [str(home) for home in homes]:
        raise ValueError("rollback home ownership changed")
    runtime = inventory(install, homes)
    home_roots = _roots(homes)
    products = tuple(install / name for name in PRODUCTS)
    state = runtime["state"]
    state_owner = dependency_home_root()
    default_owner = get_default_hermes_root()
    if state != state_owner / "installs" / state.name or not state_owner.is_absolute():
        raise ValueError("PM state is outside its computed install owner")
    state_exclusions = _state_exclusions(runtime, install)
    home_exclusions = _selection_exclusions(runtime, install, homes)
    selections = {runtime_facts_path(install), state / "pm-runtime" / "selected.json",
                  runtime["store"] / "facts.json", runtime["writable"] / "facts.json"}
    for home in homes:
        envs = home / "environments"
        if envs.is_dir() and not envs.is_symlink():
            selections.update(entry / "active.json" for entry in envs.iterdir()
                              if entry.is_dir() and not entry.is_symlink())
    obligation = host_obligation_path()
    if obligation is not None:
        selections.add(obligation)
    obligation_owners = (Path.home(), *homes, *(
        Path(value).resolve() for key in ("XDG_STATE_HOME", "HERMES_GATEWAY_LOCK_DIR")
        if (value := os.environ.get(key))))
    commands = set(owned_conveniences(install))
    referents = set(mutation_roots(homes, install))
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for record in data["roots"]:
        kind = record["kind"]
        root = Path(record["path"])
        include = record.get("include")
        excluded = record.get("excluded", [])
        if not root.is_absolute() or str(root) != record["path"]:
            raise ValueError("invalid rollback root path")
        if kind == "home":
            valid = root in home_roots and include is None and excluded == [str(p) for p in home_exclusions]
        elif kind == "state":
            valid = (root.is_relative_to(state_owner) and state.is_relative_to(root)
                     and include == [str(state)]
                     and excluded == [str(p) for p in state_exclusions])
        elif kind == "product":
            valid = root == install and include == [str(p) for p in products] and not excluded
        elif kind == "metadata":
            valid = (root in {state / "environments", state / "pm-runtime" / "generations"}
                     and include == [str(root)] and not excluded)
        else:
            valid = not excluded and isinstance(include, list) and len(include) == 1
            if valid:
                chosen = Path(include[0])
                valid = (chosen.is_absolute() and str(chosen) == include[0]
                         and (chosen == root or chosen.is_relative_to(root)))
                if valid and kind == "resource":
                    valid = _admitted_resource(chosen, runtime, homes, install) and root == chosen
                elif valid and kind == "selection":
                    valid = chosen in selections and (
                        (chosen.is_relative_to(state) and root.is_relative_to(state_owner))
                        or (chosen.is_relative_to(runtime["store"])
                            and (root.is_relative_to(runtime["store"])
                                 or (runtime["store"].is_relative_to(default_owner)
                                     and root.is_relative_to(default_owner))))
                        or (chosen.is_relative_to(runtime["writable"])
                            and (root.is_relative_to(runtime["writable"])
                                 or (runtime["writable"].is_relative_to(default_owner)
                                     and root.is_relative_to(default_owner))))
                        or any(chosen.is_relative_to(home / "environments")
                               and root.is_relative_to(home) for home in homes)
                        or (chosen == obligation and any(
                            chosen.is_relative_to(owner) and root.is_relative_to(owner)
                            for owner in obligation_owners)))
                elif valid and kind == "convenience":
                    valid = (
                        (chosen in commands and _admitted_command_root(chosen, root))
                        or (chosen in referents and root == _record_root(chosen, "convenience")
                            and any(chosen.is_relative_to(owner)
                                    for owner in (install, *homes))))
                else:
                    valid = False
                if valid and kind in {"selection", "convenience"}:
                    valid = root.is_dir() and not root.is_symlink()
                if valid and record.get("command_path") is not None:
                    valid = (kind == "convenience" and record["command_path"] == include[0]
                             and chosen in commands)
        if not valid:
            raise ValueError(f"rollback root or filter is outside admitted ownership: {root}")
        key = (kind, str(root) if kind == "home" else "", tuple(include or ()))
        if key in seen:
            raise ValueError("duplicate rollback owner")
        seen.add(key)
    if {Path(row["path"]) for row in data["roots"] if row["kind"] == "home"} != set(home_roots):
        raise ValueError("incomplete home inventory")
    if sum(row["kind"] == "state" for row in data["roots"]) != 1 or sum(
            row["kind"] == "product" for row in data["roots"]) != 1:
        raise ValueError("incomplete state or product inventory")
    recorded_selections = {Path(row["include"][0]) for row in data["roots"]
                           if row["kind"] == "selection"}
    expected_selections = set(runtime["selections"])
    if obligation is not None:
        expected_selections.add(obligation)
    if recorded_selections != expected_selections or data["new_selections"]:
        raise ValueError("incomplete rollback selection inventory")
    required_commands = {path for path in commands if not path.is_relative_to(install)
                         and not any(path.is_relative_to(root) for root in home_roots)}
    image_owners = CoverageInventory([
        owner for owner in _coverage(data).owners if owner.kind not in {"convenience", "selection"}
    ])
    required_referents = {path for path in referents if not image_owners.byte_owners(path)}
    recorded_conveniences = {Path(row["include"][0]) for row in data["roots"]
                             if row["kind"] == "convenience"}
    if recorded_conveniences != required_commands | required_referents:
        raise ValueError("incomplete rollback convenience inventory")
    for record in data["roots"]:
        if record["kind"] in {"selection", "convenience", "state"}:
            chosen = Path(record["include"][0])
            if Path(record["path"]) != _record_root(chosen, record["kind"]):
                raise ValueError("rollback creation anchor changed")
    if any(row["kind"] == "convenience" and
           (row.get("command_path") == str(Path(row["include"][0]))) !=
           (Path(row["include"][0]) in required_commands)
           for row in data["roots"]):
        raise ValueError("rollback command classification changed")
    if {Path(row["path"]) for row in data["roots"] if row["kind"] == "resource"
            } != expected_resources:
        raise ValueError("incomplete rollback resource inventory")
    containers = {state / "environments", state / "pm-runtime" / "generations"}
    captured_containers = {Path(row["path"]) for row in data["roots"] if row["kind"] == "metadata"}
    required = {path for path in containers if path.is_dir() and not path.is_symlink()}
    if captured_containers != required:
        raise ValueError("incomplete resource-container metadata inventory")
    _check_members(data)
    return homes


def _validate_members(entries: list[dict]) -> None:
    import unicodedata

    members: dict[str, str] = {}
    for item in entries:
        raw = item["path"]
        if not isinstance(raw, str):
            raise ValueError("invalid rollback archive member")
        rel = Path(raw)
        if (not rel.parts or rel.is_absolute() or rel.as_posix() != raw
                or any(part in {".", ".."} for part in rel.parts)
                or item["kind"] not in {"dir", "file", "db", "link"}):
            raise ValueError("rollback archive path escape")
        normalized = os.path.normcase(unicodedata.normalize("NFC", raw))
        if normalized in members:
            raise ValueError("colliding rollback archive members")
        members[normalized] = item["kind"]
    for name in members:
        for parent in Path(name).parents:
            if str(parent) == ".":
                break
            ancestor = parent.as_posix()
            if ancestor in members and members[ancestor] != "dir":
                raise ValueError("rollback archive member crosses a recorded non-directory")


def _artifact_specs(data: dict):
    yield "", "code.bundle", data["bundle_sha256"], data["bundle_size"], False
    for index, root in enumerate(data["roots"]):
        if root["id"] != str(index) or root["kind"] not in {
                "home", "state", "product", "selection", "resource", "convenience", "metadata"}:
            raise ValueError("invalid rollback root")
        if root["kind"] == "home" and Path(root["path"]) not in {Path(p) for p in data["homes"]}:
            raise ValueError("rollback root is not a recorded profile")
        entries = root["pre"]
        if not isinstance(entries, list):
            raise ValueError("invalid rollback members")
        _validate_members(entries)
        if "post" in root:
            _validate_members(root["post"])
        if root["pre_metadata"]["xattrs_object"] != "x-root":
            raise ValueError("invalid rollback root metadata object")
        yield root["id"], "x-root", root["pre_metadata"]["xattrs_sha256"], root["pre_metadata"]["xattrs_size"], False
        for index, item in enumerate(entries):
            if item["xattrs_object"] != f"x-{index}":
                raise ValueError("invalid rollback attribute object")
            yield root["id"], item["xattrs_object"], item["xattrs_sha256"], item["xattrs_size"], False
            if item["kind"] in {"file", "db"}:
                name = item["object"]
                if not isinstance(name, str) or name != str(index):
                    raise ValueError("invalid rollback object")
                yield root["id"], name, item["sha256"], item["size"], item["kind"] == "db"


def _verified_artifact(parent_fd: int, name: str, digest: str, size: int,
                       output_fd: int | None = None) -> None:
    if type(size) is not int or size < 0 or not isinstance(digest, str):
        raise ValueError(f"invalid rollback artifact length or hash: {name}")
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"invalid rollback artifact: {name}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise ValueError(f"rollback artifact size changed: {name}")
        hashed = hashlib.sha256()
        remaining = size
        with os.fdopen(fd, "rb", closefd=False) as stream:
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError(f"rollback artifact truncated: {name}")
                if output_fd is not None:
                    view = memoryview(block)
                    while view:
                        written = os.write(output_fd, view)
                        if not written:
                            raise OSError("rollback artifact output stopped")
                        view = view[written:]
                hashed.update(block)
                remaining -= len(block)
            excess = stream.read(1)
        after = os.fstat(fd)
        if (excess or hashed.hexdigest() != digest
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                    before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                            after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError(f"rollback artifact changed: {name}")
    finally:
        os.close(fd)


def _checked_artifacts(authority: _CapsuleAuthority, data: dict) -> None:
    authority._check()
    install = Path(data["install"])
    if not install.is_dir() or install.resolve(strict=True) != install:
        raise ValueError("installed code root moved since capture")
    _check_members(data)
    with ExitStack() as stack:
        directories = {"": authority.capsule_fd}
        for directory, name, digest, size, _ in _artifact_specs(data):
            if directory not in directories:
                directories[directory] = stack.enter_context(
                    _directory_fd(directory, authority.capsule_fd))
            _verified_artifact(directories[directory], name, digest, size)
            if directory:
                root = data["roots"][int(directory)]
                path = Path(root["path"])
                if (not path.is_absolute() or path.is_symlink()
                        or path.parent.resolve() != path.parent
                        or (not path.is_dir() and root["kind"] != "resource")
                        or (path.is_dir() and path.resolve(strict=True) != path)):
                    raise ValueError("rollback root moved since capture")
        for directory, fd in directories.items():
            if directory:
                named = os.stat(directory, dir_fd=authority.capsule_fd, follow_symlinks=False)
                held = os.fstat(fd)
                if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
                    raise ValueError("rollback artifact directory changed")
    authority._check()


def _private_artifacts(authority: _CapsuleAuthority, data: dict, private: Path) -> None:
    """Copy only attested lengths into private inodes before restore-side mutation."""
    _checked_artifacts(authority, data)
    if private.resolve().is_relative_to(authority.path.parent):
        raise ValueError("rollback snapshot must be outside capsule authority")
    with ExitStack() as stack:
        directories = {"": authority.capsule_fd}
        for directory, name, digest, size, db in _artifact_specs(data):
            if directory not in directories:
                directories[directory] = stack.enter_context(
                    _directory_fd(directory, authority.capsule_fd))
            destination_dir = private / directory
            destination_dir.mkdir(mode=0o700, exist_ok=True)
            destination = destination_dir / name
            fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.fchmod(fd, 0o600)
                _verified_artifact(directories[directory], name, digest, size, fd)
            finally:
                os.close(fd)
            if db:
                _integrity(destination)
        for directory, fd in directories.items():
            if directory:
                named = os.stat(directory, dir_fd=authority.capsule_fd, follow_symlinks=False)
                held = os.fstat(fd)
                if (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
                    raise ValueError("rollback artifact directory changed during snapshot")
    authority._check()


def _publish_artifacts(authority: _CapsuleAuthority, data: dict, private: Path) -> None:
    """Stream private producer bytes into exclusive capsule entries, never into aliases."""
    with ExitStack() as stack:
        source = stack.enter_context(_directory_fd(private))
        sources = {"": source}
        targets = {"": authority.capsule_fd}
        for directory, name, digest, size, _ in _artifact_specs(data):
            authority._check()
            if directory not in sources:
                sources[directory] = stack.enter_context(_directory_fd(directory, source))
                os.mkdir(directory, 0o700, dir_fd=authority.capsule_fd)
                targets[directory] = stack.enter_context(
                    _directory_fd(directory, authority.capsule_fd))
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=targets[directory])
            try:
                os.fchmod(fd, 0o600)
                _verified_artifact(sources[directory], name, digest, size, fd)
                os.fsync(fd)
            finally:
                os.close(fd)
        for directory, fd in targets.items():
            os.fsync(fd)
    authority._check()


def _same(actual: list[dict], expected: list[dict]) -> bool:
    keys = ("path", "kind", "mode", "uid", "gid", "xattrs_sha256", "xattrs_size",
            "target", "sha256", "size")
    for current, saved in zip(actual, expected):
        if any(current.get(key) != saved.get(key) for key in keys):
            return False
        if current["kind"] != "dir" and current["mtime_ns"] != saved["mtime_ns"]:
            return False
    return len(actual) == len(expected)

def _regenerated_parts(actual: list[dict], expected: list[dict], record: dict,
                       homes: list[Path]) -> tuple[set[str], set[str]]:
    kind = record["kind"]
    if kind not in {"home", "state"}:
        return set(), set()
    root = Path(record["path"])
    state_root = Path(record["include"][0]) if kind == "state" else root
    expected_dirs = {item["path"] for item in expected if item["kind"] == "dir"}
    entries: set[str] = set()
    extra_dirs: set[str] = set()
    for item in actual:
        path = root / item["path"]
        if regenerated_member(path, kind=kind, root=state_root, homes=homes,
                              entry_kind=item["kind"]):
            entries.add(item["path"])
            parent = path.parent
            while parent != root:
                name = parent.relative_to(root).as_posix()
                if name in expected_dirs:
                    break
                extra_dirs.add(name)
                parent = parent.parent
    return entries, extra_dirs


def _regenerated_image_matches(actual: list[dict], expected: list[dict], record: dict,
                               homes: list[Path]) -> bool:
    current_entries, extra_dirs = _regenerated_parts(actual, expected, record, homes)
    sealed_entries, _ = _regenerated_parts(expected, expected, record, homes)
    current = [item for item in actual if item["path"] not in current_entries
               and not (item["kind"] == "dir" and item["path"] in extra_dirs)]
    sealed = [item for item in expected if item["path"] not in sealed_entries]
    return _same(current, sealed)


def _filters(record: dict) -> dict:
    return {"include": tuple(Path(p) for p in record["include"]) if "include" in record else None,
            "excluded": tuple(Path(p) for p in record.get("excluded", []))}


def _record_path(path: Path, kind: str, roots: list[dict], capsule: Path, homes: list[Path],
                 install: Path, *, selected: tuple[Path, ...] | None = None,
                 excluded: tuple[Path, ...] = ()) -> None:
    root = _record_root(path, kind)
    index = str(len(roots))
    objects = capsule / index
    if kind == "metadata":
        objects.mkdir(mode=0o700)
    options = {"include": tuple(selected or (path,))} if kind != "home" else {}
    if kind in {"home", "state"}:
        options["excluded"] = excluded
    pre = ([] if kind == "metadata" else
           _image(root, set(homes), install, objects, immutable=kind == "resource", **options))
    meta = _metadata(root, objects / "x-root")
    meta.update(kind="dir", mode=stat.S_IMODE(root.stat().st_mode))
    roots.append({"id": index, "kind": kind, "path": str(root), "pre_metadata": meta,
                  "pre_mtime_ns": root.stat().st_mtime_ns, "pre": pre,
                  **({"include": [str(p) for p in options["include"]]} if "include" in options else {}),
                  **({"excluded": [str(p) for p in excluded]} if excluded else {})})


def _preflight_record(record: dict, capsule: Path) -> None:
    root = Path(record["path"])
    artifacts = capsule / record["id"]
    _preflight_metadata(root, record["pre_metadata"], artifacts / "x-root")
    for item in record["pre"]:
        _preflight_metadata(root / item["path"], item, artifacts / item["xattrs_object"])


def _matches(record: dict, homes: list[Path], install: Path, field: str,
             *, regenerated: bool = False) -> bool:
    root = Path(record["path"])
    actual = ([] if record["kind"] == "metadata" else
              _image(root, set(homes), install, immutable=record["kind"] == "resource",
                     retained_dirs=frozenset(root / item["path"] for item in record["pre"]
                                             if item["kind"] == "dir"),
                     **_filters(record)))
    return (_regenerated_image_matches(actual, record[field], record, homes) if regenerated
            else _same(actual, record[field]))


def _resource_matches(record: dict, homes: list[Path], install: Path) -> bool:
    root = Path(record["path"])
    return (root.is_dir() and not root.is_symlink()
            and {"mode": stat.S_IMODE(root.stat().st_mode), **_metadata(root)}
            == {key: record["pre_metadata"][key] for key in
                ("mode", "uid", "gid", "xattrs_sha256", "xattrs_size")}
            and _matches(record, homes, install, "pre"))


def capture_local_update_capsule(install_root, hermes_home, source_repo, target) -> Path:
    """Capture and verify a complete local preimage; never change installed HEAD."""
    install = Path(install_root).resolve(strict=True)
    source = Path(source_repo).resolve(strict=True)
    homes = _homes(Path(hermes_home))
    roots = _roots(homes)
    if any(root == install for root in roots):
        raise ValueError("code checkout cannot be a profile home")
    runtime = inventory(install, homes)
    referents = mutation_roots(homes, install)
    _quiescent(homes)
    before = _git(install, "rev-parse", "--verify", "HEAD^{commit}")
    target_sha = _git(source, "rev-parse", "--verify", f"{target}^{{commit}}")
    if len(before) != 40 or len(target_sha) != 40 or target != target_sha:
        raise ValueError("target is not an exact local commit")
    host_obligation = host_obligation_path()
    if host_obligation is not None:
        _ensure_host_obligation_parent(host_obligation)
        if host_obligation.is_symlink():
            raise ValueError(f"host obligation mutation root is a symlink: {host_obligation}")
    root_mtimes = {root: root.stat().st_mtime_ns for root in roots}
    manifest = roots[0] / _CAPSULE_DIR / uuid.uuid4().hex / "manifest.json"
    with _capsule_authority(manifest, create=True) as authority:
        # The storage parent belongs to the home but the selected UUID does not
        # control this sibling producer directory.
        with _producer_staging(authority) as private:
            bare = private / "git"
            _git(install, "init", "--bare", str(bare))
            for repo, sha, name in ((install, before, "before"), (source, target_sha, "target")):
                _git(bare, "fetch", "--no-tags", "--no-write-fetch-head",
                     "--no-recurse-submodules", str(repo), sha)
                _git(bare, "update-ref", f"refs/capsule/{name}", sha)
            _git(bare, "merge-base", "--is-ancestor", before, target_sha)
            _git(bare, "fsck", "--full", "--strict", "--no-reflogs")
            _git(bare, "bundle", "create", str(private / "code.bundle"), "--all")
            bundle_fd = os.open(private / "code.bundle", os.O_RDONLY | os.O_NOFOLLOW)
            try:
                if not stat.S_ISREG(os.fstat(bundle_fd).st_mode):
                    raise ValueError("Git bundle is not a regular file")
                os.fchmod(bundle_fd, 0o600)
            finally:
                os.close(bundle_fd)
            _git(install, "bundle", "verify", str(private / "code.bundle"))
            records = []
            for root in roots:
                _record_path(root, "home", records, private, homes, install,
                             excluded=_selection_exclusions(runtime, install, homes))
                records[-1]["pre_mtime_ns"] = root_mtimes[root]
            _record_path(runtime["state"], "state", records, private, homes, install,
                         excluded=_state_exclusions(runtime, install))
            products = tuple(install / name for name in PRODUCTS)
            resources, external = dependency_closure(
                runtime["resources"], products, install, homes, runtime["store"],
                runtime["writable"], runtime["state"])
            _record_path(install, "product", records, private, homes, install, selected=products)
            for container in (runtime["state"] / "environments",
                              runtime["state"] / "pm-runtime" / "generations"):
                if container.is_dir() and not container.is_symlink():
                    _record_path(container, "metadata", records, private, homes, install)
            for resource in resources:
                if not resource.is_dir() or resource.is_symlink():
                    raise ValueError(f"selected rollback resource is missing: {resource}")
                _record_path(resource, "resource", records, private, homes, install)
            for referent in referents:
                owners = _coverage({"roots": records, "homes": [str(home) for home in homes]}).byte_owners(referent)
                if len(owners) > 1:
                    raise ValueError(f"overlapping rollback mutation owners: {referent}")
                if owners and owners[0].kind == "resource":
                    raise ValueError(f"mutation targets immutable rollback resource: {referent}")
                if not owners:
                    if referent.is_relative_to(install):
                        raise ValueError(f"unsupported code-tree mutation referent: {referent}")
                    _record_path(referent, "convenience", records, private, homes, install)
            for selection in runtime["selections"]:
                _record_path(selection, "selection", records, private, homes, install)
            if host_obligation is not None:
                _record_path(host_obligation, "selection", records, private, homes, install)
            for command in owned_conveniences(install):
                if not command.is_relative_to(install) and not any(command.is_relative_to(root) for root in roots):
                    _record_path(command, "convenience", records, private, homes, install)
                    records[-1]["command_path"] = str(command)
            external_interpreters = [
                {"path": str(target_path), "sha256": _digest(target_path),
                 "mode": stat.S_IMODE(target_path.stat().st_mode)}
                for target_path in external
            ]
            bundle = private / "code.bundle"
            data = {"version": 4, "state": "captured", "capsule": str(manifest.parent),
                    "install": str(install), "source": str(source), "before": before, "target": target_sha,
                    "branch": ("refs/heads/" + name if (name := _git(install, "branch", "--show-current")) else None),
                    "invoking": str(Path(hermes_home).resolve(strict=True)),
                    "homes": [str(home) for home in homes], "roots": records,
                    "new_selections": [], "external_interpreters": external_interpreters,
                    "bundle_sha256": _digest(bundle), "bundle_size": bundle.stat().st_size}
            _admit_records(data, install, Path(hermes_home), expected_resources=set(resources))
            for record in records:
                _preflight_record(record, private)
            for record in records:
                root = Path(record["path"])
                if (not _matches(record, homes, install, "pre")
                        or {"mode": stat.S_IMODE(root.stat().st_mode), **_metadata(root)}
                        != {key: record["pre_metadata"][key] for key in
                            ("mode", "uid", "gid", "xattrs_sha256", "xattrs_size")}):
                    raise ValueError("mutable data changed during rollback capture")
            if _git(install, "rev-parse", "HEAD") != before:
                raise ValueError("installed code changed during rollback capture")
            _quiescent(homes)
            from pm.environments import flush_before_selecting
            flush_before_selecting()
            _publish_artifacts(authority, data, private)
            _write_manifest(authority, data)
            _checked_artifacts(authority, data)
            sync_capsule_tree(authority.capsule_fd)
            authority._check()
            publish_attestation(install, manifest, data, "capture")
            return manifest


def seal_local_update_capsule(manifest, *, expected_install_root, expected_hermes_home) -> dict:
    """Seal the actual before/target HEAD and mutable postimages of an attempted update."""
    with _capsule_authority(Path(manifest)) as authority:
        data = _read_manifest(authority)
        return _seal_verified(authority, data, expected_install_root, expected_hermes_home)


def _seal_verified(authority: _CapsuleAuthority, data: dict, install_root, hermes_home) -> dict:
    path = authority.path
    if data["state"] != "captured":
        raise ValueError("rollback capsule is not awaiting seal")
    install, homes = _scope(data, install_root, hermes_home, sealing=True)
    _checked_artifacts(authority, data)
    _quiescent(homes)
    post_head = _git(install, "rev-parse", "HEAD")
    if post_head not in {data["before"], data["target"]}:
        raise ValueError("current code is neither the recorded before nor target commit")
    post_branch = _git(install, "branch", "--show-current") or None
    attached = "refs/heads/" + post_branch if post_branch else None
    if attached not in {None, data["branch"]}:
        raise ValueError("original branch changed during local update")
    branch_post = None
    if data["branch"]:
        branch_post = _git(install, "rev-parse", "--verify", data["branch"])
        if branch_post != (post_head if attached == data["branch"] else data["before"]):
            raise ValueError("original branch moved independently during local update")
    if _git(Path(data["install"]), "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tracked code changed during local update")
    for record in data["roots"]:
        if record.get("kind") == "resource":
            root = Path(record["path"])
            if root.exists() and not _resource_matches(record, homes, Path(data["install"])):
                raise ValueError(f"selected resource changed during update: {root}")
            continue  # An old generation may be collected after sealing.
        root = Path(record["path"])
        record["post_metadata"] = {"mode": stat.S_IMODE(root.stat().st_mode), **_metadata(root)}
        record["post"] = ([] if record["kind"] == "metadata" else
                          _image(root, set(homes), install,
                                 retained_dirs=frozenset(root / item["path"] for item in record["pre"]
                                                         if item["kind"] == "dir"),
                                 **_filters(record)))
        if record.get("command_path"):
            from hermes_cli._launchers import _owns_launcher
            command = Path(record["command_path"])
            if (command.exists() or command.is_symlink()) and not _owns_launcher(command, Path(data["install"])):
                raise ValueError(f"foreign launcher replaced rollback-owned destination: {command}")
        _restorable_identity(root, record["post_metadata"])
        for item in record["post"]:
            _restorable_identity(root / item["path"], item)
    if any(not _matches(record, homes, install, "post")
           or {"mode": stat.S_IMODE(Path(record["path"]).stat().st_mode), **_metadata(Path(record["path"]))}
           != record["post_metadata"] for record in data["roots"] if record.get("kind") != "resource"):
        raise ValueError("data changed while sealing rollback capsule")
    if any(not _same(_image(Path(item["path"]).parent, set(homes), install,
                            include=(Path(item["path"]),)), item["post"])
           for item in data["new_selections"]):
        raise ValueError("new selection changed while sealing rollback capsule")
    if _git(Path(data["install"]), "rev-parse", "HEAD") != post_head:
        raise ValueError("code changed while sealing rollback capsule")
    data["post_head"] = post_head
    data["post_branch"] = post_branch
    data["post_branch_object"] = branch_post
    data["state"] = "sealed"
    _write_manifest(authority, data)
    publish_attestation(install, path, data, "seal")
    return data


def _scope(data: dict, install_root, hermes_home, *, sealing: bool = False) -> tuple[Path, list[Path]]:
    install = Path(install_root).resolve(strict=True)
    home = Path(hermes_home).resolve(strict=True)
    if install != Path(data["install"]) or home != Path(data["invoking"]):
        raise ValueError("rollback capsule scope does not match admitted installation and home")
    homes = [Path(name) for name in data["homes"]]
    roots = _roots(homes)
    if (not roots or Path(data["capsule"]).parent != roots[0] / _CAPSULE_DIR
            or home not in homes):
        raise ValueError("rollback capsule is outside the expected home")
    manifest = Path(data["capsule"]) / "manifest.json"
    verify_attestation(install, manifest, data, "capture")
    if _homes(home) != homes:
        raise ValueError("rollback home ownership changed since capture")
    if sealing:
        if data["state"] != "captured" or data["new_selections"] or any(
                "post" in row or "post_metadata" in row for row in data["roots"]):
            raise ValueError("rollback capsule already has postimages before seal")
        current = homes
        selected = {Path(row["include"][0]) for row in data["roots"] if row["kind"] == "selection"}
        for choice in inventory(install, current)["selections"]:
            if choice in selected or not any(
                    choice.parent.parent == owned_home / "environments" for owned_home in homes):
                continue
            post = _image(choice.parent, set(homes), install, include=(choice,))
            if post:
                data["new_selections"].append({"path": str(choice), "post": post})
        for item in data["new_selections"]:
            if (len(item["post"]) != 1 or item["post"][0]["path"] != "active.json"
                    or item["post"][0]["kind"] != "file"):
                raise ValueError("invalid new selection image")
    else:
        if data["state"] in {"sealed", "restored"}:
            verify_attestation(install, manifest, data, "seal")
    _check_members(data)
    return install, homes


def _verify_unmanaged_interpreters(data: dict) -> None:
    for item in data["external_interpreters"]:
        interpreter = Path(item["path"])
        if (not interpreter.is_file() or interpreter.is_symlink()
                or stat.S_IMODE(interpreter.stat().st_mode) != item["mode"]
                or _digest(interpreter) != item["sha256"]):
            raise ValueError(f"unmanaged interpreter changed: {interpreter}")


def verify_local_update_preimage(manifest, *, expected_install_root, expected_hermes_home) -> dict:
    """Bind the admitted roots and verify all captured inputs before checkout."""
    with _capsule_authority(Path(manifest)) as authority:
        data = _read_manifest(authority)
        return _verify_preimage(authority, data, expected_install_root, expected_hermes_home)


def _verify_preimage(authority: _CapsuleAuthority, data: dict, install_root, hermes_home) -> dict:
    install, homes = _scope(data, install_root, hermes_home)
    if data["state"] != "captured":
        raise ValueError("rollback capsule no longer has a captured preimage")
    _checked_artifacts(authority, data)
    _verify_unmanaged_interpreters(data)
    _quiescent(homes)
    if (_git(install, "rev-parse", "HEAD") != data["before"]
            or ("refs/heads/" + name if (name := _git(install, "branch", "--show-current")) else None) != data["branch"]
            or _git(install, "status", "--porcelain", "--untracked-files=no")):
        raise ValueError("installed code changed since rollback capture")
    for record in data["roots"]:
        root = Path(record["path"])
        if (not _matches(record, homes, install, "pre")
                or {"mode": stat.S_IMODE(root.stat().st_mode), **_metadata(root)}
                != {key: record["pre_metadata"][key] for key in
                    ("mode", "uid", "gid", "xattrs_sha256", "xattrs_size")}):
            raise ValueError(f"captured rollback preimage changed: {root}")
    return data


def _restore_record(record: dict, capsule: Path, homes: list[Path], install: Path) -> None:
    root = Path(record["path"])
    resource = record.get("kind") == "resource"
    if resource and root.exists():
        return  # The retained image was checked before checkout. Never replace a live generation.
    if resource:
        root.mkdir(parents=True)
    pre = {item["path"]: item for item in record["pre"]}
    post = {item["path"]: item for item in record.get("post", [])}
    if record["kind"] in {"home", "state"}:
        actual = _image(root, set(homes), install, **_filters(record))
        generated, extra_dirs = _regenerated_parts(actual, record["post"], record, homes)
        sealed_generated, _ = _regenerated_parts(record["post"], record["post"], record, homes)
        for name in generated:
            (root / name).unlink()
        for name in sorted(extra_dirs, key=lambda s: s.count("/"), reverse=True):
            (root / name).rmdir()
        post = {name: item for name, item in post.items()
                if name not in sealed_generated}
    for name in sorted(post.keys() - pre.keys(), key=lambda s: s.count("/"), reverse=True):
        dest = root / name
        if post[name]["kind"] == "dir":
            if not retained_parent(dest, set(homes)):
                dest.rmdir()
        else:
            dest.unlink()
    for name, item in sorted(pre.items(), key=lambda pair: pair[0].count("/")):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name in post and post[name]["kind"] != item["kind"]:
            if post[name]["kind"] == "dir":
                dest.rmdir()
            else:
                dest.unlink()
        if item["kind"] == "dir":
            dest.mkdir(exist_ok=True)
        elif item["kind"] == "link":
            dest.unlink(missing_ok=True)
            dest.symlink_to(item["target"])
        else:
            src = capsule / record["id"] / item["object"]
            if item["kind"] == "db":
                from hermes_cli.backup_restore import _safe_restore_db
                if not _safe_restore_db(src, dest):
                    raise RuntimeError(f"SQLite restore refused: {dest}")
            else:
                staged = dest.with_name(".rollback-" + uuid.uuid4().hex)
                try:
                    shutil.copyfile(src, staged)
                    staged.chmod(item["mode"])
                    staged.replace(dest)
                finally:
                    staged.unlink(missing_ok=True)
        _restore_metadata(dest, item, capsule / record["id"] / item["xattrs_object"])
        if item["kind"] != "dir":
            os.utime(dest, ns=(item["mtime_ns"], item["mtime_ns"]), follow_symlinks=False)
    if not _matches(record, homes, install, "pre"):
        raise RuntimeError(f"restored data verification failed: {root}")


def _restore_directory_metadata(records: list[dict], capsule: Path, homes: list[Path],
                                existing_resources: set[Path],
                                skipped_selections: set[str]) -> None:
    coverage = _coverage({"homes": [str(home) for home in homes], "roots": records,
                          "new_selections": []})
    directories: dict[Path, tuple[dict, Path, int]] = {}
    for record, owner in zip(records, coverage.owners):
        if record["id"] in skipped_selections:
            continue
        root = Path(record["path"])
        if record["kind"] == "resource" and root in existing_resources:
            continue
        artifacts = capsule / record["id"]
        if coverage.owner(root, "metadata", "pre") == owner:
            directories[root] = (record["pre_metadata"], artifacts / "x-root",
                                 record["pre_mtime_ns"])
        for item in record["pre"]:
            if item["kind"] == "dir":
                directory = root / item["path"]
                if coverage.owner(directory, "metadata", "pre") == owner:
                    directories[directory] = (item, artifacts / item["xattrs_object"],
                                              item["mtime_ns"])
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        item, artifact, mtime = directories[directory]
        _restore_metadata(directory, item, artifact)
        os.utime(directory, ns=(mtime, mtime))


def _purge_resource_bytecode(root: Path) -> None:
    """Clear regenerable caches without following directory links."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        if "__pycache__" not in dirs and "__pycache__" not in files:
            continue
        cache = Path(directory) / "__pycache__"
        if cache.is_dir() and not cache.is_symlink():
            for entry in cache.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        else:
            cache.unlink()
        if "__pycache__" in dirs:
            dirs.remove("__pycache__")


def restore_local_update_capsule(manifest, *, expected_install_root, expected_hermes_home,
                                 acknowledge_data_rollback=False) -> dict:
    """Refuse later history before restoring resources, then publish selections last."""
    if not acknowledge_data_rollback:
        raise ValueError("acknowledge data rollback explicitly")
    with _capsule_authority(Path(manifest)) as authority:
        data = _read_manifest(authority)
        install, homes = _scope(data, expected_install_root, expected_hermes_home)
        if data["state"] != "sealed":
            raise ValueError("rollback capsule has no sealed postimage")
        with tempfile.TemporaryDirectory(prefix="hermes-rollback-verified-") as directory:
            private = Path(directory)
            _private_artifacts(authority, data, private)
            authority._check()
            return _restore_verified(authority, data, install, homes, private)


def _restore_verified(authority: _CapsuleAuthority, data: dict, install: Path, homes: list[Path],
                      private: Path) -> dict:
    path = authority.path
    authority._check()
    _verify_unmanaged_interpreters(data)
    _quiescent(homes)
    if (_git(install, "--no-optional-locks", "rev-parse", "HEAD") != data["post_head"]
            or (_git(install, "--no-optional-locks", "branch", "--show-current") or None) != data["post_branch"]
            or _git(install, "--no-optional-locks", "status", "--porcelain", "--untracked-files=no")):
        raise ValueError("current code changed since the local update")
    branch = data["branch"]
    if branch and _git(install, "--no-optional-locks", "rev-parse", "--verify", branch) != data["post_branch_object"]:
        raise ValueError("original branch changed since seal")
    skipped_selections: set[str] = set()
    for record in data["roots"]:
        root = Path(record["path"])
        resource = record.get("kind") == "resource"
        if resource:
            if (root.exists() or root.is_symlink()) and not _resource_matches(record, homes, install):
                raise ValueError(f"retained rollback resource changed: {root}")
        elif not root.is_dir() or root.is_symlink():
            raise ValueError(f"sealed postimage drift in {root}; no data restored")
        else:
            metadata = {"mode": stat.S_IMODE(root.stat().st_mode), **_metadata(root)}
            if (record["kind"] == "selection"
                    and metadata == {key: record["pre_metadata"][key] for key in metadata}
                    and _matches(record, homes, install, "pre")):
                skipped_selections.add(record["id"])
            elif (metadata != record["post_metadata"]
                  or not _matches(record, homes, install, "post", regenerated=True)):
                raise ValueError(f"sealed postimage drift in {root}; no data restored")
        for item in record["pre"]:
            dest = root / item["path"]
            if any(parent.is_symlink() for parent in dest.parents if parent == root or root in parent.parents):
                raise ValueError(f"rollback archive path crosses a symlink: {dest}")
    for item in data["new_selections"]:
        choice = Path(item["path"])
        if not _same(_image(choice.parent, set(homes), install, include=(choice,)), item["post"]):
            raise ValueError(f"new selection changed since seal: {choice}")
    # Git's ordinary checkout overwrites ignored paths. Inspect both path directions,
    # including ignored directories, before importing or touching user data.
    tracked = [os.fsdecode(name) for name in _git_bytes(install, "--no-optional-locks", "ls-tree", "-rz", "--name-only", data["before"]).split(b"\0") if name]
    current = [os.fsdecode(name) for name in _git_bytes(install, "--no-optional-locks", "ls-tree", "-rz", "--name-only", "HEAD").split(b"\0") if name]
    others = [os.fsdecode(name) for name in _git_bytes(install, "--no-optional-locks", "ls-files", "-z", "--others", "--exclude-standard").split(b"\0") if name]
    others += [os.fsdecode(name) for name in _git_bytes(install, "--no-optional-locks", "ls-files", "-z", "--others", "--ignored", "--exclude-standard").split(b"\0") if name]
    before_dirs = {str(parent) for name in tracked for parent in Path(name).parents if str(parent) != "."}
    current_dirs = {str(parent) for name in current for parent in Path(name).parents if str(parent) != "."}
    newly_untracked_dirs = before_dirs - current_dirs
    if (any(a == b or a.startswith(b + "/") or b.startswith(a + "/") for a in tracked for b in others)
            or any(path.startswith(directory + "/") or path == directory
                   for directory in newly_untracked_dirs for path in others)):
        raise ValueError("rollback would replace ignored or untracked code")
    existing_resources = {Path(row["path"]) for row in data["roots"]
                          if row["kind"] == "resource" and Path(row["path"]).is_dir()}
    # Removed bytecode is regenerable; purge before changing code, data, or selections.
    for record in data["roots"]:
        if record["kind"] == "resource":
            _purge_resource_bytecode(Path(record["path"]))
    _git(install, "fetch", "--no-tags", "--no-write-fetch-head", str(private / "code.bundle"),
         "refs/capsule/before", "refs/capsule/target")
    if _git(install, "rev-parse", "--verify", f"{data['before']}^{{commit}}") != data["before"]:
        raise ValueError("rollback code object unavailable")
    recovery = f"refs/hermes/local-update-target/{path.parent.name}"
    try:
        _git(install, "update-ref", recovery, data["target"], "0" * 40)
    except RuntimeError:
        if _git(install, "rev-parse", "--verify", recovery) != data["target"]:
            raise
    _git(install, "checkout", "--no-overwrite-ignore", "--detach", data["before"])
    for kind in ("home", "state", "product", "convenience", "resource"):
        for record in data["roots"]:
            if record["kind"] == kind:
                _restore_record(record, private, homes, install)
    from pm.environments import flush_before_selecting
    flush_before_selecting()
    for record in data["roots"]:
        if record["kind"] == "selection" and record["id"] not in skipped_selections:
            _restore_record(record, private, homes, install)
    for item in data["new_selections"]:
        Path(item["path"]).unlink()
    _restore_directory_metadata(data["roots"], private, homes, existing_resources,
                                skipped_selections)
    flush_before_selecting()
    if branch:
        if data["post_branch_object"] != data["before"]:
            _git(install, "update-ref", branch, data["before"], data["post_branch_object"])
        elif _git(install, "rev-parse", "--verify", branch) != data["before"]:
            raise RuntimeError("original branch moved during rollback")
        _git(install, "symbolic-ref", "HEAD", branch)
    restored_branch = _git(install, "branch", "--show-current")
    if (_git(install, "rev-parse", "HEAD") != data["before"]
            or ("refs/heads/" + restored_branch if restored_branch else None) != branch
            or _git(install, "status", "--porcelain", "--untracked-files=no")):
        raise RuntimeError("restored code verification failed")
    sync_restored_git(install, branch)
    data["state"] = "restored"
    _write_manifest(authority, data)
    return data
