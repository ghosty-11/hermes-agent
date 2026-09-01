"""Stream diagnostics — per-attempt counters, exception chains, retry logging.

When a streaming chat-completions request dies mid-response, we want to
know why: which Cloudflare edge served the request, which OpenRouter
downstream provider answered, how many bytes/chunks we got before the
drop, the HTTP status, the underlying httpx error class.  These helpers
collect that info and emit it both to ``agent.log`` (full detail) and to
the user-facing status line (compact).

All helpers are extracted from :class:`AIAgent` for cleanliness.
``run_agent`` keeps thin forwarder methods so existing call sites and
tests that patch ``run_agent.<helper>`` keep working.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Per-attempt stream diagnostic headers.  Lowercased; httpx returns
# CIMultiDict so case-insensitive lookups already work, but we read .get()
# on the dict from agent.log for free-form post-hoc analysis.
STREAM_DIAG_HEADERS = (
    "cf-ray",
    "cf-cache-status",
    "x-openrouter-provider",
    "x-openrouter-model",
    "x-openrouter-id",
    "x-request-id",
    "x-vercel-id",
    "via",
    "server",
    "x-forwarded-for",
)


def stream_diag_init() -> Dict[str, Any]:
    """Return a fresh per-attempt diagnostic dict.

    Mutated in-place by the streaming functions and read from the retry
    block when a stream dies.  Lives on ``request_client_holder`` so it
    survives across the closure boundary.

    ``last_chunk_at`` / ``max_gap`` track the largest pause *between* two
    provider chunks, which ``first_chunk_at`` alone cannot express.  The
    stale detector keys on one threshold for both "nothing ever arrived"
    and "the stream went quiet mid-flight", so a record that cannot
    separate them leaves the failure undiagnosable, which is exactly how a
    run of ``Stream stale for <N>s`` WARNINGs resists diagnosis.
    """
    return {
        "started_at": time.time(),
        "first_chunk_at": None,
        "last_chunk_at": None,
        "max_gap": 0.0,
        "chunks": 0,
        "bytes": 0,
        "headers": {},
        "http_status": None,
        "usage": None,
    }


def usage_summary(usage: Any) -> Optional[Dict[str, Any]]:
    """Flatten a provider usage object to plain ints, or ``None``.

    Providers disagree on whether usage is a pydantic model, a dataclass or
    a bare dict, and several omit fields entirely, so every field is read
    defensively and absent ones stay ``None`` rather than becoming zero --
    "the provider did not say" and "the provider said nothing was used" are
    different facts on a stalled stream.
    """
    if usage is None:
        return None

    def _field(name: str) -> Optional[int]:
        value = (
            usage.get(name)
            if isinstance(usage, dict)
            else getattr(usage, name, None)
        )
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "prompt_tokens": _field("prompt_tokens"),
        "completion_tokens": _field("completion_tokens"),
        "total_tokens": _field("total_tokens"),
    }


def stream_diag_note_chunk(diag: Dict[str, Any], now: float) -> None:
    """Record one provider chunk's arrival on *diag*.

    Counts EVERY provider chunk, including empty-``choices`` usage and
    keep-alive chunks and tool-call chunks.  That is deliberate: it is the
    same population the stale detector's ``last_chunk_time`` tracks, so the
    record explains the detector rather than measuring something adjacent.
    The text-delta observer hook cannot substitute -- it never fires while a
    tool call is accumulating, so a tool-call-only stream would read as a
    first-token stall.

    Best-effort and allocation-free on the hot path: two dict reads, a
    subtract and a compare per chunk.
    """
    try:
        previous = diag.get("last_chunk_at")
        if previous is None:
            diag["first_chunk_at"] = now
        else:
            gap = now - previous
            if gap > diag.get("max_gap", 0.0):
                diag["max_gap"] = gap
        diag["last_chunk_at"] = now
        diag["chunks"] = int(diag.get("chunks", 0)) + 1
    except Exception:
        pass


def stream_diag_snapshot(diag: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a flat, JSON-safe completion record for *diag*, or ``None``.

    Shipped on the ``on_stream_end`` observer payload so a telemetry plugin
    can persist one row per streaming attempt without reaching into agent
    internals.  ``ttfb`` is ``None`` when no chunk ever arrived -- that is
    precisely the first-token-stall signal, and it is why the field is
    nullable rather than zero.
    """
    if not isinstance(diag, dict):
        return None
    try:
        started = float(diag.get("started_at") or 0.0)
        first = diag.get("first_chunk_at")
        last = diag.get("last_chunk_at")
        now = time.time()
        return {
            "started_at": started or None,
            "elapsed": max(0.0, now - started) if started else None,
            "ttfb": max(0.0, float(first) - started) if first and started else None,
            "max_gap": float(diag.get("max_gap") or 0.0),
            "idle_at_end": max(0.0, now - float(last)) if last else None,
            "chunks": int(diag.get("chunks") or 0),
            "bytes": int(diag.get("bytes") or 0),
            "http_status": diag.get("http_status"),
            "usage": diag.get("usage"),
        }
    except Exception:
        return None


def stream_diag_capture_response(agent: Any, diag: Dict[str, Any], http_response: Any) -> None:
    """Snapshot interesting headers + HTTP status from the live stream.

    Called once at stream open (before iterating chunks) so the metadata
    survives even if the stream dies before any chunk arrives.  Failures
    are swallowed — diag is best-effort.
    """
    if http_response is None or not isinstance(diag, dict):
        return
    try:
        diag["http_status"] = getattr(http_response, "status_code", None)
    except Exception:
        pass
    try:
        headers = getattr(http_response, "headers", None) or {}
        captured: Dict[str, str] = {}
        # Allow per-agent override of the headers list (back-compat).
        target_headers = getattr(agent, "_STREAM_DIAG_HEADERS", STREAM_DIAG_HEADERS)
        for name in target_headers:
            try:
                val = headers.get(name)
                if val:
                    # Truncate single-value to keep log lines bounded.
                    captured[name] = str(val)[:120]
            except Exception:
                continue
        diag["headers"] = captured
    except Exception:
        pass


def flatten_exception_chain(error: BaseException) -> str:
    """Return a compact ``Outer(msg) <- Inner(msg) <- ...`` rendering.

    OpenAI SDK wraps httpx errors as ``APIConnectionError`` /
    ``APIError`` and only the wrapper's class is visible at the catch
    site — but the underlying ``RemoteProtocolError`` /
    ``ConnectError`` / ``ReadError`` is what tells us WHY the stream
    died.  Walks ``__cause__`` then ``__context__`` (deduped, max 4
    deep) to surface the chain in one line.
    """
    seen: List[BaseException] = []
    link: Optional[BaseException] = error
    while link is not None and len(seen) < 4:
        if link in seen:
            break
        seen.append(link)
        nxt = getattr(link, "__cause__", None) or getattr(
            link, "__context__", None
        )
        if nxt is None or nxt is link:
            break
        link = nxt
    parts: List[str] = []
    for e in seen:
        msg = str(e).strip().replace("\n", " ")
        if len(msg) > 140:
            msg = msg[:140] + "…"
        parts.append(f"{type(e).__name__}({msg})" if msg else type(e).__name__)
    return " <- ".join(parts) if parts else type(error).__name__


def log_stream_retry(
    agent: Any,
    *,
    kind: str,
    error: BaseException,
    attempt: int,
    max_attempts: int,
    mid_tool_call: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a transient stream-drop and retry to ``agent.log``.

    Always logs a structured WARNING so users have a breadcrumb regardless
    of UI verbosity.  Subagents in particular benefit because their
    retries no longer spam the parent's terminal — but the file log keeps
    full detail (provider, error class, attempt, base_url, subagent_id).

    When *diag* is provided (the per-attempt stream-diagnostic dict from
    :func:`stream_diag_init`), the WARNING also captures upstream headers
    (cf-ray, x-openrouter-provider, x-openrouter-id), HTTP status, bytes
    streamed before the drop, and elapsed time on the dying attempt.
    These are the breadcrumbs needed to answer "is one CF edge / one
    downstream provider responsible, or is it random across runs?"
    """
    try:
        try:
            _summary = agent._summarize_api_error(error)
        except Exception:
            _summary = str(error)
        if _summary and len(_summary) > 240:
            _summary = _summary[:240] + "…"

        # Inner-cause chain (httpx errors hide under openai.APIError).
        try:
            _chain = flatten_exception_chain(error)
        except Exception:
            _chain = type(error).__name__

        # Per-attempt counters and upstream headers.
        _now = time.time()
        _bytes = 0
        _chunks = 0
        _elapsed = 0.0
        _ttfb = None
        _headers_repr = "-"
        _http_status = "-"
        if isinstance(diag, dict):
            try:
                _bytes = int(diag.get("bytes") or 0)
                _chunks = int(diag.get("chunks") or 0)
                _started = float(diag.get("started_at") or _now)
                _elapsed = max(0.0, _now - _started)
                _first = diag.get("first_chunk_at")
                if _first is not None:
                    _ttfb = max(0.0, float(_first) - _started)
                headers = diag.get("headers") or {}
                if isinstance(headers, dict) and headers:
                    _headers_repr = " ".join(
                        f"{k}={v}" for k, v in headers.items()
                    )
                if diag.get("http_status") is not None:
                    _http_status = str(diag.get("http_status"))
            except Exception:
                pass

        logger.warning(
            "Stream %s on attempt %s/%s — retrying. "
            "subagent_id=%s depth=%s provider=%s base_url=%s "
            "error_type=%s error=%s "
            "chain=%s "
            "http_status=%s bytes=%d chunks=%d elapsed=%.2fs ttfb=%s "
            "upstream=[%s]",
            kind,
            attempt,
            max_attempts,
            getattr(agent, "_subagent_id", None) or "-",
            getattr(agent, "_delegate_depth", 0),
            agent.provider or "-",
            agent.base_url or "-",
            type(error).__name__,
            _summary,
            _chain,
            _http_status,
            _bytes,
            _chunks,
            _elapsed,
            f"{_ttfb:.2f}s" if _ttfb is not None else "-",
            _headers_repr,
            extra={"mid_tool_call": mid_tool_call},
        )
    except Exception:
        logger.debug("stream-retry log emit failed", exc_info=True)


def emit_stream_drop(
    agent: Any,
    *,
    error: BaseException,
    attempt: int,
    max_attempts: int,
    mid_tool_call: bool,
    diag: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a single user-visible line for a stream drop+retry.

    Both top-level agents and subagents announce drops in the UI — the
    parent prefixes subagent lines with ``[subagent-N]`` via ``log_prefix``
    so they're easy to attribute.  All cases also write a structured
    WARNING to ``agent.log`` via :func:`log_stream_retry` with the full
    diagnostic detail (subagent_id, provider, base_url, error_type,
    cf-ray, x-openrouter-provider, bytes/chunks, elapsed) for post-hoc
    analysis.

    The user-visible status line is intentionally compact: provider,
    error class, attempt N/M, plus ``after Xs`` when the stream dropped
    mid-flight.  Full diagnostic detail goes to ``agent.log`` only —
    ``hermes logs --level WARNING | grep "Stream drop"`` to inspect.
    """
    kind = "drop mid tool-call" if mid_tool_call else "drop"
    log_stream_retry(
        agent,
        kind=kind,
        error=error,
        attempt=attempt,
        max_attempts=max_attempts,
        mid_tool_call=mid_tool_call,
        diag=diag,
    )
    provider = agent.provider or "provider"
    # Compose a brief "after Xs" suffix when we have timing data — helps
    # the user distinguish "couldn't connect" (0s) from "died after 30s
    # of streaming" (likely upstream idle-kill or proxy timeout).
    _suffix = ""
    if isinstance(diag, dict):
        try:
            started = diag.get("started_at")
            if started is not None:
                _suffix = f" after {max(0.0, time.time() - float(started)):.1f}s"
        except Exception:
            pass
    try:
        agent._buffer_status(
            f"⚠️ {provider} stream {kind} ({type(error).__name__}){_suffix} "
            f"— reconnecting, retry {attempt}/{max_attempts}"
        )
        agent._touch_activity(
            f"stream retry {attempt}/{max_attempts} "
            f"after {type(error).__name__}"
        )
    except Exception:
        pass


__all__ = [
    "STREAM_DIAG_HEADERS",
    "stream_diag_init",
    "stream_diag_note_chunk",
    "usage_summary",
    "stream_diag_snapshot",
    "stream_diag_capture_response",
    "flatten_exception_chain",
    "log_stream_retry",
    "emit_stream_drop",
]
