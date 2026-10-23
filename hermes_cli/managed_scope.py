"""Managed scope — IT-pushed, user-immutable config & env layer.

DISTINCT from ``hermes_cli.config.is_managed()`` / ``HERMES_MANAGED`` (a coarse package-manager
write-lock that blocks all mutation); this layer injects specific immutable values. The two are
independent and may coexist. v1 enforcement is filesystem permissions only (see
``docs/design/managed-scope.md`` §7); ``get_managed_dir()`` is the single seam for adding
macOS / Windows native locations later.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional


# Stale-module bridge: this module binds ``utils.file_signature`` at import time, so a fresh
# import in a post-pull updater process (pre-handoff purge keeps root modules cached) dies
# unless the stale ``utils`` is dropped first. See hermes_cli.stale_modules.
from hermes_cli.stale_modules import drop_stale_root_modules

drop_stale_root_modules()

from utils import fast_safe_load, file_signature

logger = logging.getLogger(__name__)

# POSIX default. Other-platform locations belong ONLY inside get_managed_dir().
_DEFAULT_MANAGED_DIR = Path("/etc/hermes")

_CACHE_LOCK = threading.Lock()
# path_key -> (*file_signature, parsed)
_CONFIG_CACHE: Dict[str, tuple] = {}
_ENV_CACHE: Dict[str, tuple] = {}


def _under_pytest() -> bool:
    """True inside the test suite: ignore the system ``/etc/hermes`` so a real managed scope on a
    dev/CI box can't leak policy into the suite. An explicit ``HERMES_MANAGED_DIR`` still wins."""
    return "PYTEST_CURRENT_TEST" in os.environ


def get_managed_dir(*, strict: bool = False) -> Optional[Path]:
    """Resolve the administrator scope; strict reads distinguish absence from bad metadata."""
    override = os.environ.get("HERMES_MANAGED_DIR", "").strip()
    if override:
        p = Path(override)
    elif _under_pytest():
        return None
    else:
        p = _DEFAULT_MANAGED_DIR
    if strict:
        from hermes_cli.config_paths import strict_directory_present
        return p if strict_directory_present(p) else None
    return p if p.is_dir() else None


def invalidate_managed_cache() -> None:
    """Drop cached managed config/env. For tests and post-edit reloads."""
    with _CACHE_LOCK:
        _CONFIG_CACHE.clear()
        _ENV_CACHE.clear()


def _cached_read(path: Path, cache: Dict[str, tuple], parse, *, strict: bool = False):
    """Stat-keyed read; strict readers refuse unreadable or invalid input."""
    if strict:
        from hermes_cli.config_paths import strict_file_present
        return parse(path) if strict_file_present(path) else None
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        if strict:
            raise
        return None
    key = file_signature(st)
    path_key = str(path)
    with _CACHE_LOCK:
        hit = cache.get(path_key)
        if hit is not None and hit[:len(key)] == key:
            return copy.deepcopy(hit[len(key)])
    try:
        parsed = parse(path)
    except Exception as exc:
        if strict:
            raise
        logger.warning(
            "managed scope: failed to parse %s: %s — IGNORING this managed file. "
            "Admin policy from this file is NOT being applied. Fix and restart.",
            path, exc)
        return None
    with _CACHE_LOCK:
        cache[path_key] = (*key, copy.deepcopy(parsed))
    return parsed


def _load_managed_file(name: str, cache: Dict[str, tuple], parse, *, strict: bool = False) -> dict:
    managed_dir = get_managed_dir(strict=True) if strict else get_managed_dir()
    if managed_dir is None:
        return {}
    path = managed_dir / name
    parsed = _cached_read(path, cache, parse, strict=strict)
    if strict and parsed is not None and not isinstance(parsed, dict):
        raise ValueError(f"Managed configuration is not a mapping: {path}")
    return parsed if isinstance(parsed, dict) else {}


def load_managed_config(*, strict: bool = False) -> dict:
    """Parsed administrator config; strict reads refuse failed inspection."""
    return _load_managed_file(
        "config.yaml", _CONFIG_CACHE, lambda p: fast_safe_load(p.read_text(encoding="utf-8-sig")),
        strict=strict,
    )


def load_managed_env(*, strict: bool = False) -> Dict[str, str]:
    """Parsed managed environment; strict reads refuse failed inspection."""
    return _load_managed_file(".env", _ENV_CACHE, _parse_managed_env, strict=strict)


def _parse_managed_env(path: Path) -> Dict[str, str]:
    from agent.secret_scope import _parse_env_text

    return _parse_env_text(path.read_text(encoding="utf-8-sig"))


def apply_managed_overlay(config: dict, *, strict: bool = False) -> dict:
    """Overlay administrator-pinned config values on top of an already-built dict.

    ``${VAR}`` refs in the managed config expand against the PROCESS env only, so a user cannot
    shadow a managed literal via a ref they control; a bare root ``model: x/y`` string is promoted
    to ``model.default`` so it can't clobber the dict shape callers expect; managed values
    deep-merge ON TOP per leaf while sibling keys stay user-controlled. Ordinary readers
    fail open; strict inspection raises instead of silently dropping administrator policy.
    """
    try:
        managed = load_managed_config(strict=True) if strict else load_managed_config()
        if not managed:
            return config
        from hermes_cli.config_primitives import _deep_merge, _expand_env_vars, _normalize_root_model_keys
        managed_expanded = _normalize_root_model_keys(_expand_env_vars(managed))
        # _normalize_root_model_keys only promotes the string when root provider/base_url
        # keys exist to migrate; handle the bare case here (matches cli.py) so _deep_merge
        # never replaces the caller's ``model`` dict with a string.
        if isinstance(managed_expanded.get("model"), str):
            managed_expanded = dict(managed_expanded)
            managed_expanded["model"] = {"default": managed_expanded["model"]}
        return _deep_merge(config, managed_expanded)
    except Exception:  # noqa: BLE001 — ordinary runtime must remain available
        if strict:
            raise
        logger.warning("managed scope: failed to apply config overlay", exc_info=True)
        return config


def _flatten_keys(d: dict, prefix: str = "") -> set:
    keys: set = set()
    for k, v in d.items():
        dotted = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and v:
            keys |= _flatten_keys(v, dotted)
        else:
            keys.add(dotted)
    return keys


def managed_config_keys() -> set:
    """Dotted leaf keys pinned by the managed config (e.g. {'model.default'})."""
    return _flatten_keys(load_managed_config())


def is_key_managed(dotted_key: str) -> bool:
    """True if the exact dotted config key is pinned by the managed layer."""
    return dotted_key in managed_config_keys()


def is_env_managed(name: str) -> bool:
    """True if the env var name is pinned by the managed .env layer."""
    return name in load_managed_env()
