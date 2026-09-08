"""Single source of truth for agent execution and context-file directories.

`TERMINAL_CWD` is the runtime carrier for the configured working directory (`terminal.cwd`
is bridged to it once at gateway/cron startup; the local CLI leaves it unset and relies on
the launch dir). Reading it in one place keeps the system prompt, tool surfaces, and
context-file discovery agreeing on where the agent executes. Multi-session gateways can pin a
logical execution cwd via `_SESSION_CWD`; multiplex gateways independently pin
`_CONTEXT_FILE_CWD` to the routed profile home.
"""

import logging
import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_SESSION_CWD: ContextVar = ContextVar("HERMES_SESSION_CWD", default=_UNSET)
_CONTEXT_FILE_CWD: ContextVar = ContextVar("HERMES_CONTEXT_FILE_CWD", default=_UNSET)

# The package/source root (<root>/agent/runtime_cwd.py). A backend launched from or
# self-spawned into this tree (desktop default) must never let an os.getcwd() fallback
# inject this repo's contributor AGENTS.md as project context.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _is_install_tree(p: Path) -> bool:
    """True only when ``p`` IS the package root or sits inside it — ancestors
    (a home dir containing the checkout) are legitimate workspaces."""
    try:
        p = p.resolve()
    except Exception:
        return False
    return p == _PACKAGE_ROOT or _PACKAGE_ROOT in p.parents


def set_session_cwd(cwd: str | None) -> Token:
    """Pin the logical cwd for the current context."""
    return _SESSION_CWD.set((cwd or "").strip())


def clear_session_cwd() -> None:
    _SESSION_CWD.set("")


def set_context_file_cwd(cwd: str | None) -> Token:
    """Pin context-file discovery without changing the execution cwd."""
    return _CONTEXT_FILE_CWD.set((cwd or "").strip())


def reset_context_file_cwd(token: Token) -> None:
    """Restore the context-file discovery override for a nested scope."""
    _CONTEXT_FILE_CWD.reset(token)


def is_context_file_cwd_scoped() -> bool:
    """Return whether instruction discovery has an independent profile scope."""
    value = _CONTEXT_FILE_CWD.get()
    return value is not _UNSET and bool(str(value).strip())


def scope_terminal_cwd() -> str:
    """Scope-aware TERMINAL_CWD value (may be empty) — every cwd consumer reads through this.

    Under gateway multiplexing the per-turn terminal scope carries the active profile's cwd;
    the process-global env var may hold another profile's. Only an ImportError falls back: an
    active refusal scope must raise, not silently resolve the launch profile's cwd.
    """
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def _existing_dir(raw: str, label: str) -> Path | None:
    p = Path(raw).expanduser()
    if p.is_dir():
        return p
    logger.warning("%s does not exist: %s", label, raw)
    return None


def _resolve_configured_cwd(*, override_is_final: bool) -> Path | None:
    """Session override, then TERMINAL_CWD; each validated as a real directory.

    ``override_is_final``: a set-but-missing session override yields None
    instead of falling through to TERMINAL_CWD.
    """
    override = _SESSION_CWD.get()
    override = "" if override is _UNSET else str(override).strip()
    if override:
        p = _existing_dir(override, "configured working directory")
        if p is not None or override_is_final:
            return p
    raw = scope_terminal_cwd().strip()
    return _existing_dir(raw, "TERMINAL_CWD") if raw else None


def resolve_agent_cwd() -> Path:
    """Configured cwd, else the launch dir (os.getcwd()'s OSError on a deleted cwd deliberately propagates)."""
    return _resolve_configured_cwd(override_is_final=False) or Path(os.getcwd())


def resolve_context_cwd() -> Path | None:
    """Resolve the authoritative context-file directory for this context."""
    context_override = _CONTEXT_FILE_CWD.get()
    if context_override is not _UNSET:
        raw = str(context_override).strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_dir():
            logger.error(
                "configured context-file directory does not exist: %s — "
                "instruction discovery will find nothing (not falling back to "
                "the execution cwd)",
                raw,
            )
        return path
    return _resolve_configured_cwd(override_is_final=True)
