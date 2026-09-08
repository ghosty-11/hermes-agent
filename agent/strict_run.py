"""Per-request enforcement for ``hermes.strict_run.v1`` agent turns."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_loop")


def active(agent: Any) -> bool:
    return getattr(agent, "_hermes_strict_run", False) is True


def terminal_failure(
    agent: Any,
    *,
    messages: list,
    conversation_history: Any,
    api_call_count: int,
    api_error: Optional[BaseException] = None,
    reason: str = "provider_error",
    summary: Optional[str] = None,
) -> dict:
    """End a strict run without exposing provider bodies or request content."""
    status_code = getattr(api_error, "status_code", None) if api_error is not None else None
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    detail = f" (HTTP {status_code})" if status_code else ""
    if summary is None:
        summary = f"Strict run provider request failed{detail}."
    logger.warning(
        "%sStrict run terminal failure%s reason=%s error_type=%s",
        agent.log_prefix,
        detail,
        reason,
        type(api_error).__name__ if api_error is not None else "-",
    )
    try:
        agent._persist_session(messages, conversation_history)
    except Exception:
        logger.debug("strict run: session persist after terminal failure failed", exc_info=True)
    return {
        "final_response": summary,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": summary,
        "failure_reason": f"strict_run_{reason}",
        "failure_retryable": False,
    }


def apply_transport_controls(agent: Any) -> None:
    """Disable fallbacks and SDK retries on this strict agent instance."""
    if not active(agent):
        return
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._fallback_index = 0
    client_kwargs = getattr(agent, "_client_kwargs", None)
    if isinstance(client_kwargs, dict):
        client_kwargs["max_retries"] = 0
    client = getattr(agent, "client", None)
    if client is None or getattr(client, "max_retries", None) == 0:
        return
    try:
        agent.client = client.with_options(max_retries=0)
    except Exception:
        logger.debug("strict run: could not apply private zero-retry client copy", exc_info=True)
