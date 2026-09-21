"""Raw-chunk stream diagnostics on the ``on_stream_end`` observer payload.

A run of provider-side stream stalls (several ``Stream stale for <N>s``
WARNINGs) could not be diagnosed because nothing
recorded whether the stall happened before the first chunk or between two
chunks. The stale detector keys on ``last_chunk_time``, which every provider
chunk refreshes, so one threshold covers both failure modes and the log line
"no chunks received" is static wording rather than a measurement.

The text-delta observer (``on_stream_delta``) cannot answer the question
either: it fires only for text, and ``chat_completion_helpers`` suppresses it
entirely once a tool call is accumulating. A tool-call-only stream therefore
produces zero deltas, which is indistinguishable from a first-token stall.

These tests pin the contract that ``on_stream_end`` carries a ``stream_diag``
snapshot counting *provider chunks*, so the two failure modes are separable.
"""

import time

from types import SimpleNamespace
from unittest.mock import patch


def _agent():
    from run_agent import AIAgent

    return AIAgent(
        api_key="test-key",
        base_url="https://provider.example.com/api/v1",
        provider="custom",
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def _text_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, reasoning_content=None, reasoning=None, tool_calls=None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        model="Qwen/Qwen3.6-35B-A3B-FP8",
    )


def _tool_chunk(name="", arguments="", index=0, call_id="call_1"):
    tc = SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
        type="function",
    )
    delta = SimpleNamespace(
        content=None, reasoning_content=None, reasoning=None, tool_calls=[tc]
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=None)],
        model="Qwen/Qwen3.6-35B-A3B-FP8",
    )


def _usage_chunk(prompt_tokens=67022, completion_tokens=140):
    return SimpleNamespace(
        choices=[],
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _collect(monkeypatch, calls):
    monkeypatch.setattr(
        "hermes_cli.plugins.iter_hook_callbacks",
        lambda name: tuple(
            {
                "on_stream_delta": [
                    lambda **kw: calls.append(("on_stream_delta", kw))
                ],
                "on_stream_end": [lambda **kw: calls.append(("on_stream_end", kw))],
            }.get(name, ())
        ),
    )


def _end_payload(calls):
    return next(c[1] for c in calls if c[0] == "on_stream_end")


def _run_stream(chunks_factory):
    agent = _agent()
    agent.api_mode = "chat_completions"
    mock_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kw: chunks_factory())
        )
    )
    return agent, mock_client


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_stream_end_reports_provider_chunk_count_and_ttfb(
    _mock_close, mock_create, monkeypatch
):
    """A healthy stream reports every provider chunk, not just text deltas."""
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    _collect(monkeypatch, calls)

    agent, mock_client = _run_stream(
        lambda: iter(
            [
                _text_chunk(content="hello "),
                _text_chunk(content="world"),
                _text_chunk(finish_reason="stop"),
                _usage_chunk(),
            ]
        )
    )
    mock_create.return_value = mock_client
    agent._interruptible_streaming_api_call({})

    _wait_for(lambda: any(c[0] == "on_stream_end" for c in calls))
    shutdown_plugin_stream_hook_dispatcher()

    diag = _end_payload(calls)["stream_diag"]

    # Four provider chunks arrived but only two text deltas fired. A counter
    # built on the delta hook would report 2 and miss the finish and usage
    # chunks entirely -- both of which refresh the stale detector's clock.
    delta_count = sum(1 for c in calls if c[0] == "on_stream_delta")
    assert delta_count == 2
    assert diag["chunks"] == 4

    assert diag["ttfb"] is not None and diag["ttfb"] >= 0.0
    assert diag["max_gap"] is not None and diag["max_gap"] >= 0.0
    assert diag["elapsed"] >= diag["ttfb"]
    assert diag["usage"]["prompt_tokens"] == 67022
    assert diag["usage"]["completion_tokens"] == 140


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_tool_call_only_stream_is_not_mistaken_for_a_first_token_stall(
    _mock_close, mock_create, monkeypatch
):
    """Zero text deltas must still report chunks>0 and a real TTFB.

    This is the discriminator requirement 2 exists for: without it a stream
    that returned only a tool call looks exactly like a stream that returned
    nothing at all.
    """
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    shutdown_plugin_stream_hook_dispatcher()
    calls = []
    _collect(monkeypatch, calls)

    agent, mock_client = _run_stream(
        lambda: iter(
            [
                _tool_chunk(name="terminal"),
                _tool_chunk(arguments='{"command":'),
                _tool_chunk(arguments='"ls"}'),
                _text_chunk(finish_reason="tool_calls"),
            ]
        )
    )
    mock_create.return_value = mock_client
    agent._interruptible_streaming_api_call({})

    _wait_for(lambda: any(c[0] == "on_stream_end" for c in calls))
    shutdown_plugin_stream_hook_dispatcher()

    diag = _end_payload(calls)["stream_diag"]

    assert sum(1 for c in calls if c[0] == "on_stream_delta") == 0
    assert diag["chunks"] == 4
    assert diag["ttfb"] is not None


@patch("run_agent.AIAgent._create_request_openai_client")
@patch("run_agent.AIAgent._close_request_openai_client")
def test_first_token_stall_is_distinguishable_from_mid_stream_gap(
    _mock_close, mock_create, monkeypatch
):
    """chunks==0 means the stall was before the first token; chunks>0 with a
    large max_gap means it was mid-stream. The 2026-08-27 log lines could not
    tell these apart."""
    from agent.plugin_stream_hooks import shutdown_plugin_stream_hook_dispatcher

    # --- first-token stall: provider yields nothing, then dies -------------
    shutdown_plugin_stream_hook_dispatcher()
    first_token_calls = []
    _collect(monkeypatch, first_token_calls)

    def _die_before_any_chunk():
        raise ConnectionError("peer closed connection")
        yield  # pragma: no cover - generator marker

    agent, mock_client = _run_stream(_die_before_any_chunk)
    mock_create.return_value = mock_client
    try:
        agent._interruptible_streaming_api_call({})
    except Exception:
        pass

    _wait_for(lambda: any(c[0] == "on_stream_end" for c in first_token_calls))
    shutdown_plugin_stream_hook_dispatcher()
    stall = _end_payload(first_token_calls)["stream_diag"]

    assert stall["chunks"] == 0
    assert stall["ttfb"] is None

    # --- mid-stream gap: chunks flow, then a long pause -------------------
    shutdown_plugin_stream_hook_dispatcher()
    gap_calls = []
    _collect(monkeypatch, gap_calls)

    def _gap_stream():
        yield _text_chunk(content="thinking")
        time.sleep(0.35)
        yield _text_chunk(content=" done")
        yield _text_chunk(finish_reason="stop")

    agent2, mock_client2 = _run_stream(_gap_stream)
    mock_create.return_value = mock_client2
    agent2._interruptible_streaming_api_call({})

    _wait_for(lambda: any(c[0] == "on_stream_end" for c in gap_calls))
    shutdown_plugin_stream_hook_dispatcher()
    gap = _end_payload(gap_calls)["stream_diag"]

    assert gap["chunks"] == 3
    assert gap["ttfb"] is not None
    assert gap["max_gap"] >= 0.3

    # The two failure modes are separable on the record alone.
    assert stall["chunks"] == 0 and gap["chunks"] > 0
