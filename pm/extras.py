"""Extras: the runtime features of the python venv.

Feature names are pyproject extra names. Runtime availability uses import
anchors; strict inventory resolves those anchors in the selected environment
without importing their parents. sync_venv([extra]) selects an extra through PM.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import importlib.util
import os
import re
import sys
from pathlib import Path
from typing import Callable


# extra name -> module that proves it is installed
ANCHORS: dict[str, str | tuple[str, ...]] = {
    "anthropic": "anthropic",
    "bedrock": "boto3",
    "vertex": "google.auth",
    "azure-identity": "azure.identity",
    "exa": "exa_py",
    "firecrawl": "firecrawl",
    "parallel-web": "parallel",
    "ddgs": "ddgs",
    "otlp": "opentelemetry.sdk",
    "langfuse": "langfuse",
    "mistral": "mistralai",
    "edge-tts": "edge_tts",
    "neutts": "neutts",
    "kittentts": ("kittentts", "soundfile"),
    "piper": "piper",
    "tts-premium": "elevenlabs",
    "voice": "faster_whisper",
    "stt-whisper": "faster_whisper",
    "audio-io": ("sounddevice", "numpy"),
    "silk": "pilk",
    "wake": "pyopen_wakeword",
    "wake-openwakeword": "pyopen_wakeword",
    "wake-sherpa": "sherpa_onnx",
    "wake-porcupine": "pvporcupine",
    "fal": "fal_client",
    "honcho": "honcho",
    "supermemory": "supermemory",
    "mem0": "mem0",
    "messaging": "telegram",
    "telegram": "telegram",
    "discord": "discord",
    "slack": "slack_bolt",
    "matrix": ("mautrix", "asyncpg", "aiosqlite", "markdown", "aiohttp_socks"),
    "dingtalk": "dingtalk_stream",
    "feishu": "lark_oapi",
    "wecom": "defusedxml",
    "teams": "microsoft_teams.apps",
    "modal": "modal",
    "daytona": "daytona",
    "vercel": "vercel",
    "google": ("googleapiclient", "google.auth", "google_auth_oauthlib.flow", "google_auth_httplib2"),
    "google-chat": "google.cloud.pubsub_v1",
    "google-meet": ("playwright.sync_api", "websockets"),
    "youtube": "youtube_transcript_api",
    "acp": "acp",
    "web": "fastapi",
    "doc-extract": "anydoc",
    # Productivity PDF skill (opt-in `pdf` extra): all three anchors must
    # import for availability checks and legacy-selection carry-over.
    "pdf": ("pypdf", "reportlab", "pdfplumber"),
    "computer-use": "mcp",
    "trace-upload": "huggingface_hub",
    # Pillow resize recovery for vision tools (the `vision` extra is a no-op
    # back-compat alias — Pillow is core — but ensure_import("vision") must
    # still resolve an anchor so availability checks work).
    "vision": "PIL",
}


def _anchors(extra: str) -> tuple[str, ...]:
    got = ANCHORS.get(extra, extra.replace("-", "_"))
    return got if isinstance(got, tuple) else (got,)


def _importable(anchor: str) -> bool:
    """An anchor already present in sys.modules counts even without a
    findable spec — tests fake SDKs by inserting modules there, and the
    caller's import right after this check resolves the same way."""
    import sys

    if anchor in sys.modules:
        return True
    try:
        return importlib.util.find_spec(anchor) is not None
    except (ImportError, ValueError):
        return False

_CERTIFIED_VIRTUALENV_HASHES = frozenset({
    "cfb3db86aaa53bb62b5ff764970bec2d71c9228590a0ebec57f6ec926cc0bf1a",
})
_CERTIFIED_EDITABLE_FINGERPRINTS = frozenset({
    "afde38c19f3bc9cf1e9eb341498e85213ad60e6d494fc040d87d47fb414c8267",
})


def _workspace_path(value: str, workspace: Path) -> Path | None:
    path = Path(value)
    if not path.is_absolute():
        return None
    try:
        relative = path.relative_to(workspace)
    except ValueError:
        return None
    if not relative.parts or ".." in relative.parts:
        return None
    if any(component.is_symlink() for component in (path, *path.parents)):
        return None
    return path


def _generated_bytecode(path: Path) -> bool:
    return (path.parent.name == "__pycache__" and path.name.endswith(".pyc")
            and f".{sys.implementation.cache_tag}" in path.name
            and path.is_file() and not path.is_symlink())


def _workspace_tree_mirrors_root(workspace_dir: Path, root_dir: Path) -> bool:
    """Check the recursive entry types without following symlinks."""
    pending = [(workspace_dir, root_dir)]
    while pending:
        source, target = pending.pop()
        if (source.is_symlink() or target.is_symlink()
                or not source.is_dir() or not target.is_dir()):
            return False
        for name in os.listdir(source):
            child, counterpart = source / name, target / name
            if child.is_symlink() or counterpart.is_symlink():
                return False
            if _generated_bytecode(child):
                continue
            if (name == "__pycache__" and child.is_dir()
                    and all(_generated_bytecode(child / entry) for entry in os.listdir(child))):
                if counterpart.exists() and not counterpart.is_dir():
                    return False
                continue
            if child.is_dir():
                if not counterpart.is_dir():
                    return False
                pending.append((child, counterpart))
            elif not child.is_file() or not counterpart.is_file():
                return False
    return True


def _certified_executable_pth(pth: Path, site: Path, root: Path,
                              environment: Path, workspace: Path, line: str) -> tuple[Path, ...] | None:
    """The project root, selected site and environment are checked for the
    placeholder. A same-named filesystem import source (directory or archive,
    such as a ZIP) in an arbitrary runtime cwd is outside this inventory.
    """
    if pth.name == "_virtualenv.pth" and line == "import _virtualenv":
        shim = site / "_virtualenv.py"
        if not shim.is_file() or shim.is_symlink():
            return None
        digest = hashlib.sha256(shim.read_bytes()).hexdigest()
        if digest not in _CERTIFIED_VIRTUALENV_HASHES:
            raise RuntimeError(f"uninspectable virtualenv shim {shim}: sha256 {digest}")
        return ()
    if not re.fullmatch(r"__editable__\..+\.pth", pth.name):
        return None
    match = re.fullmatch(r"import (__editable___\w+_finder); \1\.install\(\)", line)
    if match is None or not match[1].isidentifier():
        return None
    finder = site / f"{match[1]}.py"
    if not finder.is_file() or finder.is_symlink():
        return None
    try:
        tree = ast.parse(finder.read_text(encoding="utf-8-sig"), filename=str(finder))
    except (SyntaxError, ValueError):
        return None
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not set(names) & {"MAPPING", "NAMESPACES", "PATH_PLACEHOLDER"}:
            continue
        if len(targets) != 1 or len(names) != 1 or names[0] in values or node.value is None:
            return None
        name = names[0]
        if name == "PATH_PLACEHOLDER":
            expression = node.value
            if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                placeholder = expression.value
            elif (isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add)
                  and isinstance(expression.left, ast.Constant)
                  and isinstance(expression.left.value, str)
                  and isinstance(expression.right, ast.Constant)
                  and isinstance(expression.right.value, str)):
                placeholder = expression.left.value + expression.right.value
            else:
                return None
            if (placeholder != f"{pth.name[:-4]}.finder.__path_hook__"
                    or os.sep in placeholder or (os.altsep and os.altsep in placeholder)
                    or ".." in Path(placeholder).parts
                    or any((directory / placeholder).exists()
                           or (directory / placeholder).is_symlink()
                           for directory in (root, site, environment))):
                return None
            values[name] = placeholder
        else:
            try:
                values[name] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                return None
        node.value = ast.Constant(None)
    fingerprint = hashlib.sha256(ast.dump(tree, include_attributes=False).encode("utf-8")).hexdigest()
    if fingerprint not in _CERTIFIED_EDITABLE_FINGERPRINTS:
        return None
    mapping = values.get("MAPPING")
    namespaces = values.get("NAMESPACES")
    if not isinstance(mapping, dict) or not isinstance(namespaces, dict):
        return None
    if not all(isinstance(key, str) and key.isidentifier() and isinstance(value, str)
               for key, value in mapping.items()):
        return None
    if not all(isinstance(key, str) and isinstance(value, list)
               and all(isinstance(path, str) for path in value)
               for key, value in namespaces.items()):
        return None
    for key, value in mapping.items():
        target = _workspace_path(value, workspace)
        if target != workspace / key:
            return None
        root_entry = root / key
        root_package = (root_entry.is_dir() and not root_entry.is_symlink()
                        and (root_entry / "__init__.py").is_file()
                        and not (root_entry / "__init__.py").is_symlink())
        if target.is_dir():
            if not root_package or not _workspace_tree_mirrors_root(target, root_entry):
                return None
        elif (root_package or target.is_symlink() or not any(
                (candidate := target.with_suffix(suffix)).is_file()
                and not candidate.is_symlink()
                and (root_candidate := root_entry.with_suffix(suffix)).is_file()
                and not root_candidate.is_symlink()
                for suffix in importlib.machinery.all_suffixes())):
            return None
    for key, paths in namespaces.items():
        parts = key.split(".")
        if (any(not part or part in {".", ".."} or os.sep in part
                or (os.altsep and os.altsep in part) for part in parts)
                or any(root.joinpath(*parts[:index]).is_symlink()
                       for index in range(1, len(parts) + 1))):
            return None
        root_dir = root.joinpath(*parts)
        if not root_dir.is_dir():
            return None
        for value in paths:
            source = _workspace_path(value, workspace)
            if source is None or not _workspace_tree_mirrors_root(source, root_dir):
                return None
    # Root-first resolution wins mapped modules and regular packages. The
    # recursive mirror ensures every name the finder's mapped-child fallback
    # or namespace hook can resolve also resolves with the same type in the
    # preceding project root. An exempt-only __pycache__ may still import as
    # an empty namespace, but cannot supply a module.
    return ()


def selected_import_paths(project_root: Path) -> tuple[Path, ...]:
    """Search roots activated for this install, never the inspecting interpreter's paths."""
    import zipfile
    from pm.environments import base_venv, committed_venv, site_packages, venv_python_version

    root = Path(project_root).resolve()
    selected = committed_venv(root)
    environment = selected if selected is not None else base_venv(root)
    if selected is None:
        for candidate in (environment, root / "venv", root / ".venv"):
            try:
                candidate.lstat()
            except FileNotFoundError:
                continue
            break
        else:
            return ()
    config = environment / "pyvenv.cfg"
    if not config.is_file():
        raise RuntimeError(f"selected dependency environment metadata unavailable: {environment}")
    config.read_text(encoding="utf-8-sig")  # unreadable metadata is not an absent environment
    if venv_python_version(environment) is None:
        raise RuntimeError(f"selected dependency environment version unavailable: {environment}")
    site = site_packages(environment)
    if not site.is_dir():
        raise RuntimeError(f"selected dependency environment has no site-packages: {site}")
    paths = [root, site]
    # The editable finder follows the project root and exposes only declared keys.
    # site.addsitedir processes these in filename order. Only the two standard
    # executable shims can be certified without running their code.
    # Enumerate with an error-propagating listing: glob() swallows OSError and
    # would turn an unreadable selected site into an empty, certified inventory.
    for pth in sorted(site / name for name in os.listdir(site) if name.endswith(".pth")):
        # Reading a FIFO would wait for a writer; metadata must be a regular file.
        if not pth.is_file() or pth.is_symlink():
            raise RuntimeError(f"uninspectable import path declaration: {pth}")
        lines = [line for line in pth.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        for raw in lines:
            line = raw.strip()
            if line.startswith(("import ", "import\t")):
                certified = (_certified_executable_pth(
                    pth, site, root, environment, environment.parent / "workspace", raw,
                ) if len(lines) == 1 else None)
                if certified is None:
                    raise RuntimeError(f"uninspectable import path declaration: {pth}")
                continue
            candidate = (site / line).resolve()
            if candidate.is_dir() or (candidate.is_file() and zipfile.is_zipfile(candidate)):
                paths.append(candidate)
            elif candidate.exists():
                raise RuntimeError(f"uninspectable import path declaration: {pth}")
    return tuple(dict.fromkeys(paths))


class UnsupportedSelectedExtension(RuntimeError):
    """Selected ABI has an extension suffix this interpreter cannot order."""


def passive_module_origin(anchor: str, *, search_paths: tuple[Path, ...]) -> str | None:
    """Ordered filesystem import resolution; a regular winner stops namespace collection."""
    import importlib.machinery
    import zipfile

    parts = anchor.split(".")
    if not parts or any(not part.isidentifier() for part in parts):
        raise ValueError(f"invalid module anchor: {anchor}")
    suffixes = (*importlib.machinery.EXTENSION_SUFFIXES, ".py", ".pyc")
    extension_suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    paths = tuple(Path(p) for p in search_paths)
    origin = None
    for index, part in enumerate(parts):
        namespaces: list[Path] = []
        winner = None
        children: tuple[Path, ...] = ()
        for base in paths:
            if base.is_file():
                if not zipfile.is_zipfile(base):
                    raise RuntimeError(f"uninspectable import path: {base}")
                with zipfile.ZipFile(base) as archive:
                    members = set(archive.namelist())
                prefix = "/".join(parts[:index + 1])
                package = next((f"{prefix}/__init__{suffix}" for suffix in (".pyc", ".py")
                                if f"{prefix}/__init__{suffix}" in members), None)
                module = next((f"{prefix}{suffix}" for suffix in (".pyc", ".py")
                               if f"{prefix}{suffix}" in members), None)
                if package is not None:
                    winner, children = f"{base}:{package}", (base,)
                    break
                if module is not None:
                    winner = f"{base}:{module}"
                    break
                if any(name.startswith(f"{prefix}/") for name in members):
                    raise RuntimeError(f"zip namespace layout cannot be certified: {base}:{prefix}")
                continue
            if not base.is_dir():
                if any(parent.suffix.lower() in {".zip", ".egg", ".whl"} and parent.is_file()
                       for parent in (base, *base.parents)):
                    raise RuntimeError(f"uninspectable archive import path: {base}")
                continue
            target = base / part
            # FileFinder checks a regular package before a sibling module.
            package = next((target / f"__init__{suffix}" for suffix in suffixes
                            if (target / f"__init__{suffix}").is_file()), None)
            module = next((base / f"{part}{suffix}" for suffix in suffixes
                           if (base / f"{part}{suffix}").is_file()), None)
            # A selected interpreter's ABI-tagged extension can shadow a later
            # pure-Python package. Do not certify that package using this process's
            # incompatible suffix list.
            for parent, stem in ((target, "__init__"), (base, part)):
                if parent.is_dir():
                    for candidate in parent.glob(f"{stem}.*"):
                        if candidate.suffix.lower() in {".so", ".pyd"} and not any(
                            candidate.name == stem + suffix for suffix in extension_suffixes
                        ):
                            raise UnsupportedSelectedExtension(
                                f"unsupported selected interpreter extension: {candidate}")
            if package is not None:
                winner, children = package, (target,)
                break
            if module is not None:
                winner = module
                break
            if target.is_dir():
                namespaces.append(target)
        if winner is not None:
            origin = str(winner)
            if index < len(parts) - 1 and not children:
                return None
            paths = children
        elif namespaces:
            origin = None
            paths = tuple(namespaces)
        else:
            return None
    return origin if origin is not None else ("" if paths else None)


def passive_importable(anchor: str, *, search_paths: tuple[Path, ...] | None = None) -> bool:
    """Probe the selected filesystem paths without importing a parent or calling import finders."""
    import sys

    try:
        return passive_module_origin(anchor, search_paths=(
            search_paths if search_paths is not None else tuple(
                Path(p) if p else Path.cwd() for p in sys.path)
        )) is not None
    except UnsupportedSelectedExtension:
        return False

def passive_available(extra: str, *, search_paths: tuple[Path, ...],
                      source_root: Path | None = None) -> bool:
    """Verify every anchor in the selected interpreter without loading an SDK."""
    probe = lambda anchor: passive_importable(anchor, search_paths=search_paths)
    return extra_supported(extra, importable=probe, strict=True, source_root=source_root) and all(
        probe(anchor) for anchor in _anchors(extra))



def available(extra: str) -> bool:
    """Runtime availability check for an extra's anchors.

    Dotted find_spec calls may import parent packages. Passive inspection uses
    passive_available instead. A platform-gated extra without installed anchors
    reads as unavailable on excluded hosts."""
    if not extra_supported(extra):
        return False
    return all(_importable(a) for a in _anchors(extra))


def _platform_gates(*, strict: bool = False, source_root: Path | None = None) -> dict[str, str]:
    """Read the selected source's [tool.hermes.extras-platforms] marker table."""
    global _PLATFORM_GATES
    if source_root is None and _PLATFORM_GATES is not None and not strict:
        return _PLATFORM_GATES
    import tomllib

    from pm.paths import repo_root

    gates: dict[str, str] = {}
    try:
        with ((source_root or repo_root()) / "pyproject.toml").open("rb") as f:
            data = tomllib.load(f)
        table = data.get("tool", {}).get("hermes", {}).get("extras-platforms", {})
        if not isinstance(table, dict):
            raise ValueError("extras platform gates must be a mapping")
        if strict and any(not isinstance(k, str) or not isinstance(v, str) for k, v in table.items()):
            raise ValueError("extras platform gates must contain string markers")
        gates = {str(k): str(v) for k, v in table.items()}
    except (OSError, ValueError):
        if strict:
            raise
    if source_root is None and not strict:
        _PLATFORM_GATES = gates
    return gates


_PLATFORM_GATES: dict[str, str] | None = None


def extra_supported(extra: str, *, environment: dict[str, str] | None = None,
                    importable: Callable[[str], bool] | None = None,
                    strict: bool = False, source_root: Path | None = None) -> bool:
    """Is this extra installable on THIS platform? True when the extra
    carries no gate, or its marker matches the running platform. An
    extra that IS present on this machine (anchors importable) is always
    supported — an installed override beats the table (dev machines,
    hand-synced venvs)."""
    if not strict and all((importable or _importable)(a) for a in _anchors(extra)):
        return True
    marker = _platform_gates(strict=strict, source_root=source_root).get(extra)
    if marker is None:
        return True
    import os
    import platform
    import sys

    if environment is None:
        environment = {
            "sys_platform": sys.platform,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "os_name": os.name,
        }
    try:
        from packaging.markers import Marker
    except ImportError:
        if strict:
            raise RuntimeError("Strict platform inspection requires its local marker parser") from None
        # The historical update takeover runs PM in the OLD venv's
        # interpreter, which need not ship `packaging`. PM's runtime does.
        return _evaluate_in_runtime(marker, environment)
    try:
        supported = bool(Marker(marker).evaluate(environment=environment))
        return supported or (strict and all((importable or passive_importable)(a)
                                            for a in _anchors(extra)))
    except Exception:
        if strict:
            raise
        # A malformed marker must never brick availability — treat as
        # ungated and let the resolver be the authority.
        return True


def _evaluate_in_runtime(marker: str, environment: dict[str, str]) -> bool:
    import json
    import subprocess

    from pm.runtime import runtime_command, runtime_environment
    from pm.writer_claim import child_spawn, writer_claim

    try:
        with writer_claim():
            command = runtime_command(Path(__file__).with_name("_marker_eval.py"),
                                      [marker, json.dumps(environment)])
            child_env = runtime_environment()
            with child_spawn(child_env) as spawn:
                result = subprocess.run(command, env=child_env, capture_output=True,
                                        text=True, timeout=60, **spawn)
    except Exception:
        return True  # same fallback as a malformed marker: the resolver decides
    if result.returncode != 0:
        return True
    return result.stdout.strip() == "1"


def install_hint(extra: str) -> str:
    """The one command users are told to run for a missing extra."""
    return f"hermes pm install --extra {extra}"


def ensure_import(extra: str) -> None:
    """Make an extra available: no-op when the anchor imports, otherwise
    sync the venv with the extra enabled. Raises InstallError on failure
    — including when a platform gate excludes this machine."""
    if available(extra):
        return
    if not extra_supported(extra):
        from pm.package import InstallError

        marker = _platform_gates().get(extra, "")
        raise InstallError(
            "venv",
            f"extra {extra!r} is not supported on this platform "
            f"(gate: {marker!r}); the adapter degrades without it",
        )
    import sys
    from pm.package import InstallError

    # prompt_toolkit already owns stdin during a CLI turn; never read from it.
    app_running = False
    if "prompt_toolkit.application.current" in sys.modules:
        from prompt_toolkit.application.current import get_app_or_none

        app_running = bool(getattr(get_app_or_none(), "is_running", False))
    if not app_running and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            answer = input(f"\nThis needs Hermes' optional {extra!r} feature, which isn't installed yet.\n"
                           "Install it now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise InstallError("venv", f"installation of extra {extra!r} declined")
    from pm.client import sync_venv

    sync_venv([extra])
    # The sync published a new generation. Swap this process onto it when nothing
    # already imported would change underneath it (adopt_selected); otherwise only
    # a restart can load it.
    from pm.environments import selected_venv, site_packages
    from pm.environments_adopt import adopt_selected, restart_needed
    from pm.paths import repo_root, runtime_facts_path
    import sys
    from pathlib import Path

    if runtime_facts_path().is_file():
        root = repo_root()
        adopt_selected(root)
        selected = site_packages(selected_venv(root)).resolve()
        if selected not in {Path(entry).resolve() for entry in sys.path}:
            reason = restart_needed(root) or "this process does not run from the install's dependency environment"
            raise InstallError("venv", f"{extra} installed; restart Hermes to activate it ({reason})")




def missing(extra: str) -> tuple[str, ...]:
    return tuple(a for a in _anchors(extra) if not _importable(a))


def _installed_in(site_packages: Path, anchor: str) -> bool:
    """Is ``anchor`` installed under a site-packages we must not import from?"""
    *parents, leaf = anchor.split(".")
    parent = site_packages.joinpath(*parents)
    if (parent / leaf).is_dir():
        return True
    # A top-level module can be a plain file or a compiled extension.
    return any(entry.name.rsplit(".", 1)[-1] in {"py", "so", "pyd"}
               for entry in parent.glob(f"{leaf}.*"))


def legacy_selection(project_root: Path) -> list[str]:
    """The extras PM's first generation selects when it replaces a main-era venv.

    Main-era installers selected ``[all]`` and then lazily installed opt-in
    extras (FAL, messaging SDKs, ...) into the checkout's own venv, with no
    ledger. Selecting only ``[all]`` drops those, and the first launch after
    the migration asks to reinstall a feature that already worked. The old
    venv's site-packages is read, never imported: the migrating process may
    not run from it.
    """
    root = Path(project_root)
    trees = [tree for venv in (root / "venv", root / ".venv")
             for tree in (*venv.glob("lib/python*/site-packages"), venv / "Lib" / "site-packages")
             if tree.is_dir()]
    carried = sorted(
        # An umbrella extra shares its anchor with one member; carrying it
        # would install every sibling the user never chose.
        extra for extra in ANCHORS if extra not in {"messaging", "voice", "wake"}
        # PM refuses a gated extra outside its platform even if a hand-synced venv carried it.
        # Installed first: judging a gate may cost a PM-runtime subprocess.
        if any(all(_installed_in(tree, anchor) for anchor in _anchors(extra)) for tree in trees)
        and extra_supported(extra, importable=lambda _anchor: False)
    )
    return ["all", *carried]
