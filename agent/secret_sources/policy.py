"""Dependency-light parsing for the orchestrator's secret-source policies."""

from __future__ import annotations


_POLICY_KEYS = frozenset({"sources", "preserve_existing", "profile_alias"})


def source_policy(secrets_cfg: dict, *, strict: bool = False) -> tuple[frozenset[str], bool]:
    """Return the effective preserve set and alias switch used by the orchestrator.

    Strict inspection rejects malformed policy instead of treating it as an empty policy.
    Runtime remains fail-open for existing configurations.
    """
    preserve_raw = secrets_cfg.get("preserve_existing")
    if strict and "preserve_existing" in secrets_cfg and (
        not isinstance(preserve_raw, list) or any(not isinstance(name, str) for name in preserve_raw)
    ):
        raise ValueError("secret source preserve_existing must be a list of names")
    preserve = (frozenset(name.strip() for name in preserve_raw
                          if isinstance(name, str) and name.strip())
                if isinstance(preserve_raw, list) else frozenset())
    alias = secrets_cfg.get("profile_alias", True)
    if strict and not isinstance(alias, bool):
        raise ValueError("secret source profile_alias must be a boolean")
    return preserve, bool(alias)


def normalized_declarations(config: dict) -> tuple[dict, set[str]]:
    """Validate source sections and retain effective policies in the revision input."""
    if not isinstance(config, dict):
        raise ValueError("secret source declarations must be a mapping")
    sources = config.get("sources", [])
    if not isinstance(sources, list) or any(not isinstance(name, str) for name in sources):
        raise ValueError("secret source order must be a list of names")
    preserve, alias = source_policy(config, strict=True)
    sections = {name: value for name, value in config.items() if name not in _POLICY_KEYS}
    if any(not isinstance(value, dict) or not isinstance(value.get("enabled", False), bool)
           for value in sections.values()):
        raise ValueError("secret source declaration must contain a boolean enabled flag")
    normalized = dict(sections)
    normalized["sources"] = sources
    normalized["preserve_existing"] = sorted(preserve)
    normalized["profile_alias"] = alias
    return normalized, {name for name, value in sections.items() if value.get("enabled")}
