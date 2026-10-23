"""Read-only configuration primitives shared by startup and passive inspection."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger("hermes_cli.config")
_ENV_REF_RE = re.compile(r"\${([^}]+)}")


def cfg_get(cfg: dict | None, *keys: str, default: Any = None) -> Any:
    """Traverse dict keys; only missing keys select the default."""
    if not isinstance(cfg, dict):
        return default
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override over base without replacing dict sections with YAML null."""
    result = base.copy()
    for key, value in override.items():
        over_dict = isinstance(result.get(key), dict)
        if over_dict and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        elif not (over_dict and value is None):
            result[key] = value
    return result


def _env_ref_lookup(name: str) -> str | None:
    """Honor the active profile's secret scope, not another profile's process env."""
    try:
        from agent.secret_scope import current_secret_scope, get_secret
    except Exception:
        return os.environ.get(name)
    return os.environ.get(name) if current_secret_scope() is None else get_secret(name)


def _is_non_env_secret_ref(ref: str) -> bool:
    return ":" in ref and re.match(r"^[a-z][a-z0-9_-]*:", ref) is not None


def _env_ref_var_name(ref: str) -> str | None:
    ref = ref.strip()
    if ref.startswith("env:"):
        return ref[len("env:"):].strip() or None
    if _is_non_env_secret_ref(ref):
        return None
    return ref


def _env_expand_match(match: re.Match) -> str:
    raw = match.group(0)
    inner = match.group(1).strip()
    name = _env_ref_var_name(inner)
    if name is None:
        if not inner.startswith("env:") and _is_non_env_secret_ref(inner):
            logger.warning(
                "Config ref %r uses source %r which is not resolvable in "
                "config.yaml — external secret sources inject env vars at "
                "startup, so reference the variable as ${env:NAME} instead",
                raw, inner.split(":", 1)[0])
        return raw
    value = _env_ref_lookup(name)
    if value is not None:
        return value
    if inner.startswith("env:"):
        logger.warning(
            "Config ref %r: %s is not set (check ~/.hermes/.env); "
            "keeping the literal placeholder", raw, name)
    return raw


def _expand_env_vars(obj):
    """Expand env references in values, leaving unresolved references verbatim."""
    if isinstance(obj, str):
        return _ENV_REF_RE.sub(_env_expand_match, obj)
    if isinstance(obj, dict):
        return {key: _expand_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def _normalize_root_model_keys(config: dict) -> dict:
    """Normalize legacy root model keys and model aliases without provider activation."""
    model_in = config.get("model")
    model_provider = model_in.get("provider") if isinstance(model_in, dict) else None
    needs_model_work = (model_provider is not None and not isinstance(model_provider, str)) or (
        isinstance(model_in, dict) and (
            model_in.get("api_base") or model_in.get("model") or model_in.get("name")
            or any(isinstance(model_in.get(k), dict) for k in ("default", "model", "name"))))
    has_root = any(config.get(k) for k in ("provider", "base_url", "context_length", "api_base"))
    if not has_root and not needs_model_work:
        return config

    config = dict(config)
    model = config.get("model")
    model = dict(model) if isinstance(model, dict) else {"default": model} if model else {}
    config["model"] = model
    for key in ("default", "model", "name"):
        val = model.get(key)
        if isinstance(val, dict):
            nested_model = val.get("model") or val.get("default")
            nested_provider = str(val.get("provider") or "").strip()
            model[key] = str(nested_model or "").strip()
            if nested_provider and (not str(model.get("provider") or "").strip()
                                    or str(model.get("provider") or "").strip() == "auto"):
                model["provider"] = nested_provider
    for key in ("provider", "base_url", "context_length"):
        root_val = config.get(key)
        if root_val and not model.get(key):
            model[key] = root_val
        config.pop(key, None)
    if model.get("provider") is not None:
        model["provider"] = str(model["provider"]).strip()
    for alias_val in (config.get("api_base"), model.get("api_base")):
        if alias_val and not model.get("base_url"):
            model["base_url"] = alias_val
    config.pop("api_base", None)
    model.pop("api_base", None)
    alias = model.get("model") or model.get("name")
    if not model.get("default") and alias:
        model["default"] = alias
    if model.get("default"):
        model.pop("model", None)
        model.pop("name", None)
    return config
