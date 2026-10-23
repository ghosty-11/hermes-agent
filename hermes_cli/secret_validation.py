"""Placeholder-aware secret validation without provider or config initialization."""
from __future__ import annotations

from typing import Any

_PLACEHOLDER_SECRET_VALUES = {
    "*", "**", "***", "changeme", "your_api_key", "your_api_key_here", "your-api-key",
    "placeholder", "example", "dummy", "null", "none",
}
_PLACEHOLDER_KEY_PREFIXES = ("sk-", "ghp_", "hf_")


def _is_placeholder_shape(value: str) -> bool:
    lowered = value.lower()
    if lowered.startswith("your_") and lowered.endswith("_here"):
        return True
    for prefix in _PLACEHOLDER_KEY_PREFIXES:
        if lowered.startswith(prefix):
            tail = lowered[len(prefix):]
            if tail and all(c == "x" for c in tail):
                return True
    stripped = lowered.replace(" ", "").replace("-", "").replace("_", "")
    return bool(stripped) and all(c == "x" for c in stripped)


def has_usable_secret(value: Any, *, min_length: int = 4) -> bool:
    """Reject empty, short, and documented placeholder secrets."""
    if not isinstance(value, str):
        return False
    cleaned = value.strip()
    return (len(cleaned) >= min_length
            and cleaned.lower() not in _PLACEHOLDER_SECRET_VALUES
            and not _is_placeholder_shape(cleaned))
