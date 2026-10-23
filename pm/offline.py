"""Per-operation child environment for cache-only package and product builds."""

from collections.abc import Mapping


def offline_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Retain the caller's selected tool paths while disabling child downloads."""
    return {**env, "HERMES_UPDATE_OFFLINE": "1",
            "UV_OFFLINE": "1", "PIP_NO_INDEX": "1",
            "npm_config_offline": "true", "npm_config_audit": "false",
            "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
