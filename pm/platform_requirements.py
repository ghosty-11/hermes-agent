"""Declarative connector dependencies without importing or activating connectors."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformRequirements:
    extras: tuple[str, ...]
    env: tuple[str, ...]


def _mapping(value, label: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _names(value, label: str, *, optional: bool = False) -> set[str] | None:
    if optional and value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"{label} must be a list of names")
    return set(value)


def _platform_inventory(config: dict, *, source_root: Path | None = None,
                        selected_paths: tuple[Path, ...] | None = None
                        ) -> tuple[dict[str, PlatformRequirements], set[str], dict[str, str], set[str]]:
    """Resolve directory/entry-point winners before reading selected dependencies."""
    import stat

    from gateway.config import _BUILTIN_PLATFORM_VALUES
    from hermes_cli.config_paths import strict_directory_present, strict_file_present
    from hermes_cli.plugins_discovery import (
        bundled_plugins_dir, collect_directory_manifests, discover_entrypoint_manifests,
        gate_manifest, resolve_manifest_winners,
    )
    from hermes_cli.plugins_manifest import manifest_key
    from hermes_yaml import safe_load

    if source_root is None:
        from pm.paths import repo_root
        source_root = repo_root()
    plugins = _mapping(config.get("plugins"), "plugins")
    disabled = _names(plugins.get("disabled", []), "plugins.disabled")
    enabled = _names(plugins.get("enabled"), "plugins.enabled", optional=True)
    root = bundled_plugins_dir(source_root) / "platforms"
    bundled_keys: set[str] = set()
    declared_ids: set[str] = set()
    bundled: dict[str, dict] = {}
    aliases: dict[str, str] = {}
    if not strict_directory_present(root):
        raise ValueError(f"bundled platform directory missing: {root}")
    for directory in sorted(root.iterdir()):
        try:
            is_directory = stat.S_ISDIR(directory.stat().st_mode)
        except FileNotFoundError as exc:
            raise ValueError(f"listed platform directory disappeared or has no target: {directory}") from exc
        if not is_directory:
            continue
        path = next((candidate for candidate in (directory / "plugin.yaml", directory / "plugin.yml")
                     if strict_file_present(candidate)), None)
        if path is None:
            raise ValueError(f"platform manifest missing: {directory}")
        with open(path, encoding="utf-8-sig") as stream:
            data = _mapping(safe_load(stream), str(path))
        if data.get("kind") != "platform" or not isinstance(data.get("name"), str) or not data["name"]:
            raise ValueError(f"invalid platform manifest kind or name: {path}")
        declarations = data.get("platform_dependencies")
        if not isinstance(declarations, dict) or not declarations or any(
            not isinstance(name, str) or not name for name in declarations
        ):
            raise ValueError(f"platform dependency declaration is missing or invalid: {path}")
        declared_ids.update(declarations)
        bundled_keys.add("platforms/" + directory.name)
        bundled[str(directory)] = data
        aliases[data["name"].lower()] = directory.name
    manifests = collect_directory_manifests(strict=True, source_root=source_root)
    directory_keys = {manifest_key(manifest) for manifest in manifests}
    manifests.extend(manifest for manifest in discover_entrypoint_manifests(
        strict=True, selected_paths=selected_paths) if manifest_key(manifest) not in directory_keys)
    winners = resolve_manifest_winners(manifests)
    known: set[str] = set(_BUILTIN_PLATFORM_VALUES) | declared_ids
    catalog: dict[str, PlatformRequirements] = {}
    for key, manifest in winners.items():
        if manifest.source == "bundled" and key not in bundled_keys:
            continue
        verdict = gate_manifest(manifest, disabled, enabled, strict=True)
        if manifest.source != "bundled":
            if verdict.action in {"load", "defer", "load_now"}:
                raise ValueError(f"selected custom plugin may register a platform: {key}")
            if manifest.kind == "platform" and manifest.path and manifest.source != "entrypoint":
                # Disabled identities remain known without admitting their requirements.
                path = Path(manifest.path)
                manifest_file = next((candidate for candidate in (path / "plugin.yaml", path / "plugin.yml")
                                      if strict_file_present(candidate)), None)
                if manifest_file is not None:
                    with manifest_file.open(encoding="utf-8-sig") as stream:
                        data = _mapping(safe_load(stream), str(manifest_file))
                    declarations = data.get("platform_dependencies") or {}
                    if not isinstance(declarations, dict) or any(
                        not isinstance(name, str) or not name for name in declarations
                    ):
                        raise ValueError(f"invalid platform identities: {manifest_file}")
                    known.update(declarations)
            continue
        declarations = bundled[manifest.path].get("platform_dependencies")
        if not isinstance(declarations, dict) or not declarations:
            raise ValueError(f"platform dependency declaration is missing: {manifest.path}")
        for platform, declaration in declarations.items():
            if not isinstance(platform, str) or not platform or platform in catalog:
                raise ValueError(f"invalid or duplicate platform declaration: {manifest.path}")
            declaration = _mapping(declaration, platform)
            extras, env = declaration.get("extras"), declaration.get("env")
            _names(extras, platform + ".extras")
            _names(env, platform + ".env")
            known.add(platform)
            if verdict.action == "defer":
                catalog[platform] = PlatformRequirements(tuple(extras), tuple(env))
    for platform in _BUILTIN_PLATFORM_VALUES - declared_ids:
        catalog[platform] = PlatformRequirements((), ())
    return catalog, known, aliases, declared_ids


def platform_dependency_catalog(config: dict | None = None) -> dict[str, PlatformRequirements]:
    """Complete strict catalog of winning, admitted platform declarations."""
    return _platform_inventory(config or {})[0]


def required_platform_extras(platform: str, manifest_path: Path) -> tuple[str, ...]:
    """Read only the requested platform's explicit manifest, never the catalog."""
    from hermes_cli.config_paths import strict_file_present
    from hermes_yaml import safe_load

    path = Path(manifest_path)
    if not strict_file_present(path):
        raise ValueError(f"platform manifest is missing: {path}")
    with open(path, encoding="utf-8-sig") as stream:
        data = _mapping(safe_load(stream), str(path))
    if data.get("kind") != "platform":
        raise ValueError(f"platform manifest is not a platform: {path}")
    declarations = _mapping(data.get("platform_dependencies"), str(path))
    if platform not in declarations:
        raise ValueError(f"platform has no dependency declaration: {platform}")
    declaration = _mapping(declarations[platform], platform)
    extras = declaration.get("extras")
    _names(extras, platform + ".extras")
    _names(declaration.get("env"), platform + ".env")
    return tuple(extras)


def ensure_platform_dependencies(platform: str, manifest_path: Path) -> bool:
    """Attempt every declared extra; preserve the first package-manager refusal."""
    from pm import ensure_import
    from pm.package import InstallError

    first_error = None
    for extra in required_platform_extras(platform, manifest_path):
        try:
            ensure_import(extra)
        except InstallError as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error
    return True


def configured_platform_requirements(config: dict, lookup: Callable[[str], str | None],
                                     *, legacy: dict | None = None, home: Path | None = None,
                                     routed: bool = False, multiplex_active: bool = False,
                                     source_root: Path | None = None,
                                     selected_paths: tuple[Path, ...] | None
                                     ) -> dict[str, PlatformRequirements]:
    """Inspect runtime platform intent and credential gates, without plugin activation."""
    import copy
    from agent.secret_scope import is_multiplex_active

    from gateway.config import (
        Platform, PlatformConfig, SHARED_LISTENER_MIRROR_PLATFORMS,
        _BUILTIN_PLATFORM_VALUES, _bool_token, normalize_platform_name,
    )
    from gateway.config_env import _Cred, _ENV_STEPS
    from gateway.config_loader import merge_platform_sections, bridge_platform_shared_keys
    from gateway.platform_configuration import (
        credential_groups_present, credential_seed, plugin_configured, plugin_probe_and_commit,
        relay_configuration,
    )
    from utils import is_truthy_value

    catalog, known, aliases, declared_ids = _platform_inventory(
        config, source_root=source_root, selected_paths=selected_paths)
    data = copy.deepcopy(legacy or {})
    gateway = _mapping(config.get("gateway"), "gateway")
    sections = merge_platform_sections(config, gateway, data, known=known | set(aliases),
                                       aliases=aliases)
    from types import SimpleNamespace
    targets = []
    for name in catalog:
        if name in _BUILTIN_PLATFORM_VALUES:
            targets.append(Platform(name))
        else:
            targets.append(SimpleNamespace(value=name))
    bridge_platform_shared_keys(config, gateway.get("platforms"), data, sections, targets)
    blocks: dict[str, PlatformConfig] = {}
    for name, raw in sections.items():
        if not isinstance(raw, dict):
            raise ValueError(f"platform configuration is not a mapping: {name}")
        if "enabled" in raw:
            flag = raw["enabled"]
            if not isinstance(flag, bool) and not (
                isinstance(flag, str) and _bool_token(flag) is not None
            ):
                raise ValueError(f"platform enabled flag must be boolean: {name}")
        canonical = normalize_platform_name(name, aliases) if isinstance(name, str) else name
        if canonical not in known:
            if "enabled" not in raw or PlatformConfig.from_dict(raw).enabled:
                raise ValueError(f"unknown configured platform: {name}")
            continue
        if canonical not in catalog:
            continue  # selected plugin was disabled, not unknown
        blocks[canonical] = PlatformConfig.from_dict(
            {key: value for key, value in raw.items() if key != "home_channel"}
            if "home_channel" in raw else raw)
    steps = {step.platform.value: step for step in _ENV_STEPS if isinstance(step, _Cred)}
    relay_block = blocks.get("relay")
    relay_explicitly_off = bool(relay_block and not relay_block.enabled
                                and relay_block.extra.get("_enabled_explicit"))
    relay_url, relay_exclusive = relay_configuration(relay_block.extra if relay_block else {}, lookup)
    relay_exclusive = relay_exclusive and not relay_explicitly_off
    multiplex_active = multiplex_active or is_multiplex_active() or routed
    required = {}
    for name, declaration in catalog.items():
        block = blocks.get(name, PlatformConfig())
        explicit_off = not block.enabled and block.extra.get("_enabled_explicit")
        if explicit_off:
            continue
        if name == "whatsapp" and str(lookup("WHATSAPP_ENABLED") or "").lower() in {"false", "0", "no"}:
            continue
        step = steps.get(name)
        env_gate = credential_groups_present(step.creds, lookup) if step else False
        if name == "whatsapp":
            env_gate = is_truthy_value(lookup("WHATSAPP_ENABLED"))
        if name == "api_server":
            from gateway.config import _has_usable_api_server_key
            env_gate = _has_usable_api_server_key(lookup("API_SERVER_KEY"))
        if name in {"webhook", "msgraph_webhook"}:
            env_gate = is_truthy_value(lookup(name.upper() + "_ENABLED"))
        if name == "relay":
            env_gate = bool(relay_url)
        if routed and name in SHARED_LISTENER_MIRROR_PLATFORMS:
            env_gate = False
        if env_gate and step:
            block.extra.update(credential_seed(step, lookup))
            if step.token and lookup(step.token):
                block.token = lookup(step.token)
        if name == "api_server" and env_gate:
            block.extra["key"] = lookup("API_SERVER_KEY")
        if name == "msgraph_webhook" and lookup("MSGRAPH_WEBHOOK_CLIENT_STATE"):
            block.extra["client_state"] = lookup("MSGRAPH_WEBHOOK_CLIENT_STATE")
        if name == "relay" and env_gate:
            block.extra["relay_url"] = relay_url
        enabled = block.enabled or env_gate
        if name in declared_ids:
            probe_admitted = plugin_probe_and_commit(
                name, block.extra, lookup, home or Path.cwd(), already_enabled=block.enabled,
                multiplex_active=multiplex_active)
            configured = plugin_configured(
                name, block.extra, lookup, home or Path.cwd(),
                multiplex_active=multiplex_active, token=block.token, enabled=block.enabled)
            enabled = enabled or probe_admitted
        else:
            configured = False
        if not enabled:
            continue
        if relay_exclusive and name not in {"relay", "local", "api_server", "webhook"}:
            continue
        if name in declared_ids:
            if not (block.token or block.api_key or configured):
                continue
        else:
            from gateway.config import _PLATFORM_CONNECTED_CHECKERS
            checker = _PLATFORM_CONNECTED_CHECKERS.get(Platform(name))
            if checker is not None and not (block.token or block.api_key or checker(block)):
                continue
        required[name] = declaration
    return required
