"""Dependency-light configuration predicates for passive inventory and gateway adapters.

An enabled block and an installed SDK are separate facts. Callers provide the profile's
credential lookup; this module never discovers plugins, installs packages or writes env.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

Lookup = Callable[[str], str | None]


def multiplex_for_passive_inspection(config: dict, legacy: dict, lookup: Lookup) -> bool | None:
    """Resolve explicit multiplex intent without the startup migration preflight.

    True is established; None means the effective runtime mode depends on the
    preflight. A saved or environment false is retired, not an opt-out. The
    caller must forbid ambient credential discovery when the verdict is None.
    """
    raw = lookup("GATEWAY_MULTIPLEX_PROFILES")
    token = str(raw or "").strip().lower()
    if token in {"true", "1", "yes", "on"}:
        return True
    if token in {"false", "0", "no", "off"}:
        return None
    for layer in (config, legacy):
        value = layer.get("multiplex_profiles")
        if value is None:
            gateway = layer.get("gateway")
            value = gateway.get("multiplex_profiles") if isinstance(gateway, dict) else None
        if value is not None:
            return True if value is True or str(value).strip().lower() in {
                "true", "1", "yes", "on",
            } else None
    return None


def credential_groups_present(groups: tuple, lookup: Lookup) -> bool:
    return all(any(lookup(name) for name in (group if isinstance(group, tuple) else (group,)))
               for group in groups)


def google_chat_inbound(extra: dict, lookup: Lookup) -> bool:
    return bool(extra.get("http_events_url") or lookup("GOOGLE_CHAT_HTTP_EVENTS_URL") or (
        (extra.get("project_id") or lookup("GOOGLE_CHAT_PROJECT_ID") or lookup("GOOGLE_CLOUD_PROJECT"))
        and (extra.get("subscription_name") or lookup("GOOGLE_CHAT_SUBSCRIPTION_NAME")
             or lookup("GOOGLE_CHAT_SUBSCRIPTION"))))


_GOOGLE_CHAT_OPTIONAL_SEED = {
    "GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE": "http_events_audience",
    "GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL": "http_events_service_account_email",
    "GOOGLE_CHAT_MAX_MESSAGES": "max_messages",
    "GOOGLE_CHAT_MAX_BYTES": "max_bytes",
    "GOOGLE_CHAT_BOOTSTRAP_SPACES": "bootstrap_spaces",
    "GOOGLE_CHAT_DEBUG_RAW": "debug_raw",
}


def google_chat_seed(lookup: Lookup) -> dict:
    """Only a complete env inbound mode seeds the runtime plugin probe."""
    project = lookup("GOOGLE_CHAT_PROJECT_ID") or lookup("GOOGLE_CLOUD_PROJECT")
    subscription = lookup("GOOGLE_CHAT_SUBSCRIPTION_NAME") or lookup("GOOGLE_CHAT_SUBSCRIPTION")
    url = lookup("GOOGLE_CHAT_HTTP_EVENTS_URL")
    if not (url or (project and subscription)):
        return {}
    seed = {key: value for key, value in (
        ("project_id", project), ("subscription_name", subscription), ("http_events_url", url),
        ("service_account_json", lookup("GOOGLE_CHAT_SERVICE_ACCOUNT_JSON")
         or lookup("GOOGLE_APPLICATION_CREDENTIALS")),
    ) if value}
    for env, key in _GOOGLE_CHAT_OPTIONAL_SEED.items():
        raw = str(lookup(env) or "").strip()
        if raw:
            seed[key] = raw
    home = str(lookup("GOOGLE_CHAT_HOME_CHANNEL") or "").strip()
    if home:
        name = lookup("GOOGLE_CHAT_HOME_CHANNEL_NAME")
        seed["home_channel"] = {"chat_id": home, "name": "Home" if name is None else name}
    return seed


def relay_configuration(extra: dict, lookup: Lookup) -> tuple[str, bool]:
    """Normalized relay URL and direct-platform exclusion from the env stamp."""
    from utils import is_truthy_value

    env_url = str(lookup("GATEWAY_RELAY_URL") or "").strip()
    url = (env_url or str(extra.get("relay_url") or "").strip()).rstrip("/")
    return url, bool(env_url and not is_truthy_value(lookup("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS")))


def _pair(extra: dict, lookup: Lookup, *fields: tuple[str, str]) -> bool:
    return all(extra.get(field) or lookup(name) for field, name in fields)


def photon_credential_groups(extra: dict, lookup: Lookup, stored: tuple[str | None, str | None]) -> bool:
    """A complete YAML/env pair or a complete stored pair; never join the two groups."""
    return bool(_pair(extra, lookup, ("project_id", "PHOTON_PROJECT_ID"),
                      ("project_secret", "PHOTON_PROJECT_SECRET"))
                or (stored[0] and stored[1]))


def _photon_credentials(extra: dict, lookup: Lookup, home: Path) -> bool:
    if _pair(extra, lookup, ("project_id", "PHOTON_PROJECT_ID"),
             ("project_secret", "PHOTON_PROJECT_SECRET")):
        return True
    from hermes_cli.config_paths import strict_file_present
    path = home / "auth.json"
    if not strict_file_present(path):
        return False
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    pool = data.get("credential_pool", {}).get("photon_project") or []
    record = pool[0] if isinstance(pool, list) and pool else {}
    return photon_credential_groups(extra, lookup, (
        record.get("spectrum_project_id") or record.get("project_id"),
        record.get("project_secret"),
    ))


def buzz_credentials_candidates(extra: dict, lookup: Lookup, *, multiplex_active: bool,
                                strict: bool = False, default_dir: Path | None = None) -> tuple[Path, ...]:
    """Explicit credential file wins; ambient discovery is forbidden under multiplex."""
    configured = str(lookup("BUZZ_CREDENTIALS_FILE") or "").strip() or str(
        extra.get("credentials_file") or "").strip()
    if configured:
        return (Path(configured).expanduser(),)
    if multiplex_active:
        return ()
    root = default_dir if default_dir is not None else Path("~/.config/buzz").expanduser()
    if strict:
        from hermes_cli.config_paths import strict_directory_present
        return tuple(sorted(root.glob("*credentials*.json"))) if strict_directory_present(root) else ()
    try:
        return tuple(sorted(root.glob("*credentials*.json")))
    except OSError:
        return ()


def _buzz_key(extra: dict, lookup: Lookup, *, multiplex_active: bool) -> bool:
    if str(lookup("BUZZ_PRIVATE_KEY") or "").strip():
        return True
    from hermes_cli.config_paths import strict_file_present
    for path in buzz_credentials_candidates(extra, lookup, multiplex_active=multiplex_active, strict=True):
        if not strict_file_present(path):
            continue
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        if isinstance(data, dict) and any(isinstance(data.get(key), str) and data[key].strip()
                                          for key in ("nsec", "private_key_hex", "private_key")):
            return True
    return False


def credential_seed(step, lookup: Lookup) -> dict:
    """Fields written by a credential step; runtime and inspection use the same projection."""
    from gateway.config_env import _INT

    extra = {}
    for key, env, *rest in step.fixed:
        default = rest[0] if rest else ""
        if isinstance(env, tuple):
            value = next((v for v in map(lookup, env) if v), "") or default
        else:
            value = lookup(env)
            # Runtime's getenv(name, default) uses the default only for absence.
            value = default if value is None else value
        extra[key] = rest[1](value) if len(rest) > 1 else value
    for specs, strip in ((step.optional, False), (step.optional_stripped, True)):
        for key, env, *converter in specs:
            value = lookup(env) or ""
            if strip:
                value = value.strip()
            if not value:
                continue
            if converter and converter[0] is _INT:
                try:
                    extra[key] = int(value)
                except ValueError:
                    continue
            else:
                extra[key] = converter[0](value) if converter else value
    return extra


def plugin_probe_and_commit(name: str, extra: dict, lookup: Lookup, home: Path, *,
                            already_enabled: bool, multiplex_active: bool) -> bool:
    """Runtime enable pass: probe seeded extras, then commit the complete env seed."""
    seed = google_chat_seed(lookup) if name == "google_chat" else {}
    probe = dict(extra)
    for key, value in seed.items():
        if key != "home_channel":
            probe.setdefault(key, value)
    admitted = already_enabled or plugin_configured(
        name, probe, lookup, home, multiplex_active=multiplex_active, enabled=False)
    if admitted:
        extra.update({key: value for key, value in seed.items() if key != "home_channel"})
    return admitted


def matrix_connected(extra: dict, lookup: Lookup, *, token: str | None = None) -> bool:
    """Matrix requires a nonblank homeserver and a nonblank token or password."""
    homeserver = extra.get("homeserver") or lookup("MATRIX_HOMESERVER")
    credential = token or lookup("MATRIX_ACCESS_TOKEN") or lookup("MATRIX_PASSWORD")
    return bool(str(homeserver or "").strip() and str(credential or "").strip())


def plugin_configured(name: str, extra: dict, lookup: Lookup, home: Path | None = None, *,
                      multiplex_active: bool = False, token: str | None = None,
                      enabled: bool = False) -> bool:
    """Bundled platform's configuration gate, independent of SDK availability."""
    if name == "google_chat":
        return google_chat_inbound(extra, lambda _name: None)
    if name == "photon":
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return _photon_credentials(extra, lookup, home)
    if name == "buzz":
        return bool(str(extra.get("relay_url") or lookup("BUZZ_RELAY_URL") or "").strip()
                    and _buzz_key(extra, lookup, multiplex_active=multiplex_active))
    if name == "raft":
        return bool(extra.get("enabled") or extra.get("bridge_token") or lookup("RAFT_PROFILE"))
    if name == "a2a":
        return bool(extra.get("enabled") or str(lookup("A2A_PORT") or "").strip())
    if name == "irc":
        return _pair(extra, lookup, ("server", "IRC_SERVER"), ("channel", "IRC_CHANNEL"))
    if name == "line":
        return _pair(extra, lookup, ("channel_access_token", "LINE_CHANNEL_ACCESS_TOKEN"),
                     ("channel_secret", "LINE_CHANNEL_SECRET"))
    if name == "teams":
        return _pair(extra, lookup, ("client_id", "TEAMS_CLIENT_ID"),
                     ("client_secret", "TEAMS_CLIENT_SECRET"), ("tenant_id", "TEAMS_TENANT_ID"))
    if name == "dingtalk":
        return _pair(extra, lookup, ("client_id", "DINGTALK_CLIENT_ID"),
                     ("client_secret", "DINGTALK_CLIENT_SECRET"))
    if name == "ntfy":
        return bool(extra.get("topic") or lookup("NTFY_TOPIC"))
    if name == "simplex":
        return bool(extra.get("ws_url") or lookup("SIMPLEX_WS_URL"))
    if name == "mattermost":
        return bool(lookup("MATTERMOST_URL") and lookup("MATTERMOST_TOKEN"))
    if name == "homeassistant":
        return bool(lookup("HASS_TOKEN"))
    if name == "sms":
        return bool(lookup("TWILIO_ACCOUNT_SID"))
    if name == "email":
        return bool(extra.get("address") or (lookup("EMAIL_ADDRESS") or "").strip())
    if name == "feishu":
        return bool(extra.get("app_id"))
    if name == "wecom":
        return bool(extra.get("bot_id"))
    if name == "wecom_callback":
        return bool(extra.get("corp_id") or extra.get("apps"))
    if name == "matrix":
        return matrix_connected(extra, lookup, token=token)
    if name in {"telegram", "discord", "slack"}:
        return bool(token or lookup(name.upper() + "_BOT_TOKEN"))
    if name == "whatsapp":
        return bool((enabled and extra) or str(lookup("WHATSAPP_ENABLED") or "").strip().lower()
                    in {"true", "1", "yes"})
    raise ValueError(f"unsupported passive platform configuration: {name}")
