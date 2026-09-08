"""Behavioral policy guards for strict runs (hermes.strict_run.v1).

These exercise the REAL agent loop and executors with per-request strict
attributes — not mock-echo assertions. Each test demonstrates a policy the
contract forbids violating:

1. A configured fallback provider is never called for a strict run, and the
   actual OpenAI SDK client has transport retries zeroed (SDK defaults still
   permit ~3 provider HTTP requests even with a framework retry count of 0).
2. Fabricated tool calls execute zero handlers when tool_policy is none.
3. An image-rejection 4xx yields exactly one provider call — no
   strip-and-retry, no aux-vision detour, no session vision demotion.
4. A fabricated tool call on a no-tools run ends the turn: no second model
   round, nothing a second round might say is ever returned.
5. The success-path recovery loops (invalid/empty-choices response,
   empty-content retry) are terminal for strict runs; legacy runs keep
   their retries (control test).
6. No tool definitions reach the wire for a no-tools run, even after the
   shared tool-snapshot rebuild would have repopulated agent.tools.

The strict attributes used here (``_hermes_strict_run``, ``_strict_no_tools``,
``_strict_force_native_images``) are the per-request controls the API server
sets when admitting a ``hermes.strict_run.v1`` run.

Fixture notes (RED-wave corrections):
* Fabricated tool calls use the real OpenAI attribute shape
  (``tool_call.function.name`` / ``.arguments``), not dicts.
* The patched OpenAI client returns a real response shape so streaming
  warm-ups cannot TypeError on MagicMock reasoning parts.
* The aux-vision description seam is stubbed so a text-mode detour cannot
  reach real auxiliary providers; the tests observe the strict policy, not
  auxiliary availability.
"""

from __future__ import annotations

import copy
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import (
    execute_tool_calls_concurrent,
    execute_tool_calls_sequential,
)


class _FakeApiError(Exception):
    """Stand-in for an openai APIError with status_code + body."""

    def __init__(self, status_code: int, message: str, body: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {"error": {"message": message}}
        self.response = None


def _mock_response(content: str, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="bb-avatar", usage=None)


def _shaped_mock_client():
    """OpenAI-client stand-in whose create() returns a real response shape."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response("warm-up ok")
    return client


def _make_agent(**agent_kwargs):
    """Build a minimal AIAgent, mirroring the idiom in
    tests/run_agent/test_69078_image_corrupt_recovery.py."""
    from run_agent import AIAgent

    client = _shaped_mock_client()
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("hermes_logging.setup_logging"),
    ):
        agent = AIAgent(
            api_key="fx",  # unused — the OpenAI client is mocked below
            base_url="http://127.0.0.1:4000/v1",
            provider="custom:gateway",
            model="bb-avatar",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            **agent_kwargs,
        )
        agent.client = client
        # Established fixture pattern: force the non-streaming request path
        # so the injected failure reaches the real request seam instead of a
        # streaming warm-up answering first.
        agent._disable_streaming = True
        agent._use_prompt_caching = False
        return agent


def _image_history():
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what is currently visible."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEA"
                            "AAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQA"
                            "AAAABJRU5ErkJggg=="
                        )
                    },
                },
            ],
        },
    ]


def _isolate_aux_vision(agent):
    """Stub the aux-vision description seam for this agent.

    The non-strict text-mode detour calls ``vision_analyze`` through this
    method; stubbing keeps auxiliary providers out of the policy tests.
    """
    return patch.object(
        type(agent),
        "_describe_image_for_anthropic_fallback",
        return_value="[An image was attached to this message.]",
    )


# ─── 1. Configured fallback is never called ─────────────────────────────────


def test_strict_run_never_activates_configured_fallback():
    """A strict run with a configured fallback provider must fail terminally
    on the primary route. No fallback model/provider switch, no second
    provider call, and the actual OpenAI SDK client must have its transport
    retries zeroed — SDK defaults ignore framework retry counts.
    """
    calls = []

    def fake_api_call(api_kwargs):
        calls.append(copy.deepcopy(api_kwargs.get("messages")))
        raise _FakeApiError(status_code=429, message="rate limited")

    agent = _make_agent(
        fallback_model={"provider": "fallback-prov", "model": "fallback-model"}
    )
    assert agent._fallback_chain, "fixture must configure a fallback provider"
    agent._api_max_retries = 3
    agent._hermes_strict_run = True

    # Real SDK client with provider-default retries — Main's LiteLLM probe
    # showed num_retries:0 at the router still allows ~3 provider HTTP
    # requests via OpenAI SDK defaults. Strict mode must zero the client.
    from openai import OpenAI as _OpenAI

    agent.client = _OpenAI(
        api_key="sk-strict-test", base_url="http://127.0.0.1:4000/v1", max_retries=2
    )
    assert agent.client.max_retries == 2

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        _isolate_aux_vision(agent),
        patch("agent.process_bootstrap.OpenAI", return_value=_shaped_mock_client()),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation("look at my desktop")

    # Terminal failure on the locked route.
    assert result["completed"] is False
    assert result["failed"] is True

    # Exactly one provider call — no retry cycle into the fallback chain.
    assert len(calls) == 1

    # The runtime identity never left the locked route.
    assert agent.provider == "custom:gateway"
    assert agent.model == "bb-avatar"
    assert agent._fallback_activated is False
    assert agent._fallback_index == 0

    # The actual OpenAI SDK client must not retry at the transport layer.
    assert agent.client.max_retries == 0

    # The safe error must not echo provider bodies or internals.
    assert "rate limited" not in str(result.get("error", "")).lower()


# ─── 2. Fabricated tool calls execute zero handlers ─────────────────────────


def _fabricated_tool_calls(count: int):
    """Real OpenAI ChatCompletionMessageToolCall attribute shape."""
    return [
        SimpleNamespace(
            id=f"call_{i}",
            type="function",
            function=SimpleNamespace(
                name="terminal",
                arguments=json.dumps({"command": "id"}),
            ),
        )
        for i in range(count)
    ]


def test_strict_no_tools_fabricated_call_runs_zero_handlers_sequential():
    """tool_policy:none denies dispatch even when the provider fabricates a
    tool call. The executor must append a refusal result and reach zero
    handlers."""
    agent = _make_agent()
    agent._strict_no_tools = True

    handler = MagicMock(return_value="handler output that must never exist")
    assistant_message = SimpleNamespace(tool_calls=_fabricated_tool_calls(1))
    messages: list = []

    with patch("model_tools.handle_function_call", handler):
        execute_tool_calls_sequential(
            agent, assistant_message, messages, "task-strict"
        )

    handler.assert_not_called()

    # The history stays provider-consistent: every fabricated call gets a
    # refusal tool result, never silence and never handler output.
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "not permitted" in str(tool_msgs[0].get("content", ""))
    assert "handler output" not in str(tool_msgs[0].get("content", ""))
    assert tool_msgs[0].get("tool_call_id") == "call_0"


def test_strict_no_tools_fabricated_calls_run_zero_handlers_concurrent():
    """Same denial on the concurrent executor — the other dispatch entry."""
    agent = _make_agent()
    agent._strict_no_tools = True

    handler = MagicMock(return_value="handler output that must never exist")
    assistant_message = SimpleNamespace(tool_calls=_fabricated_tool_calls(3))
    messages: list = []

    with patch("model_tools.handle_function_call", handler):
        execute_tool_calls_concurrent(
            agent, assistant_message, messages, "task-strict"
        )

    handler.assert_not_called()

    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    assert [m.get("tool_call_id") for m in tool_msgs] == [
        "call_0",
        "call_1",
        "call_2",
    ]
    for msg in tool_msgs:
        assert "not permitted" in str(msg.get("content", ""))
        assert "handler output" not in str(msg.get("content", ""))


def test_non_strict_fabricated_call_still_reaches_handlers():
    """Control: without the strict flag the executor dispatches normally, so
    the two refusals above fail for the right reason (the guard, not the
    fixture)."""
    agent = _make_agent()
    handler = MagicMock(return_value="handler output")
    assistant_message = SimpleNamespace(tool_calls=_fabricated_tool_calls(1))
    messages: list = []

    with patch("model_tools.handle_function_call", handler):
        execute_tool_calls_sequential(
            agent, assistant_message, messages, "task-ordinary"
        )

    assert handler.called
    assert any(
        m.get("role") == "tool" and "handler output" in str(m.get("content"))
        for m in messages
    )


# ─── 3. Image 4xx: one provider call, no strip-and-retry ────────────────────


def test_strict_image_rejection_4xx_is_terminal_single_call():
    """A required-image strict run whose provider rejects the image content
    with a 4xx must fail visibly. No text-only strip retry, no aux-vision
    detour, no vision_supported demotion, no second provider call, and the
    one failing request still carried the image part."""
    calls = []

    def fake_api_call(api_kwargs):
        calls.append(copy.deepcopy(api_kwargs.get("messages")))
        raise _FakeApiError(
            status_code=400,
            message="Only 'text' content type is supported.",
        )

    agent = _make_agent()
    agent._api_max_retries = 3
    agent._hermes_strict_run = True
    agent._strict_force_native_images = True

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        _isolate_aux_vision(agent),
        patch("agent.process_bootstrap.OpenAI", return_value=_shaped_mock_client()),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ):
        result = agent.run_conversation(
            "Describe what is currently visible.",
            conversation_history=_image_history(),
        )

    assert result["completed"] is False
    assert result["failed"] is True
    assert len(calls) == 1

    # The single failing request still carried the image part — no strip.
    only_call = calls[0]
    assert any(
        isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
        for m in only_call
    ), f"image part was stripped from the failing strict request: {only_call!r}"

    # The session was not demoted to text-only vision.
    assert getattr(agent, "_vision_supported", True) is True

    # The safe error must not echo the data URL or provider wording.
    error_text = str(result.get("error", ""))
    assert "data:image" not in error_text
    assert "Only 'text' content type" not in error_text


# ─── 4. Fabricated tool call on a no-tools run ends the turn ─────────────────


def _run_with(agent, responses, user_message="look at my desktop"):
    """Drive run_conversation with a scripted provider; returns (result, calls).

    Backoff is zeroed (not just sleep) so the legacy control test's retry
    path stays fast without busy-spinning through the interrupt-aware
    sleep loops.
    """
    calls = []
    script = list(responses)

    def fake_api_call(api_kwargs):
        calls.append(copy.deepcopy(api_kwargs))
        if not script:
            raise AssertionError("provider called more often than scripted")
        nxt = script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    with ExitStack() as stack:
        stack.enter_context(patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call))
        stack.enter_context(patch.object(agent, "_persist_session"))
        stack.enter_context(patch.object(agent, "_save_trajectory"))
        stack.enter_context(patch.object(agent, "_cleanup_task_resources"))
        stack.enter_context(_isolate_aux_vision(agent))
        stack.enter_context(patch("agent.process_bootstrap.OpenAI", return_value=_shaped_mock_client()))
        stack.enter_context(patch("agent.agent_runtime_helpers.time.sleep"))
        stack.enter_context(patch("agent.retry_utils.jittered_backoff", return_value=0.0))
        stack.enter_context(patch("agent.model_metadata.get_model_context_length", return_value=200000))
        result = agent.run_conversation(user_message)
    return result, calls


def test_strict_no_tools_fabricated_call_ends_turn_without_second_model_round():
    """On a tool_policy:none run a fabricated tool call executes nothing AND
    the turn ends there: the screenshot is not re-sent for a second model
    round, and whatever a second round would have said is never spoken."""
    agent = _make_agent()
    agent._hermes_strict_run = True
    agent._strict_no_tools = True
    agent._api_max_retries = 3
    handler = MagicMock(return_value="handler output that must never exist")

    fabricated = _mock_response("", tool_calls=_fabricated_tool_calls(1))
    fabricated.choices[0].finish_reason = "tool_calls"
    second_round = _mock_response("SECOND ROUND ANSWER")

    with patch("model_tools.handle_function_call", handler):
        result, calls = _run_with(agent, [fabricated, second_round])

    handler.assert_not_called()
    assert len(calls) == 1, "a fabricated tool call must not buy a second model round"
    assert result["completed"] is False
    assert result["failed"] is True
    assert "SECOND ROUND ANSWER" not in str(result.get("final_response", ""))
    assert "SECOND ROUND ANSWER" not in str(result.get("error", ""))


# ─── 5. Success-path recovery loops are gated too ───────────────────────────


def test_strict_invalid_response_is_terminal_without_retry():
    """A 200 with no choices (a common gateway upstream-failure shape) must
    fail the strict run after ONE provider call — no backoff/retry loop
    re-uploading the screenshot."""
    agent = _make_agent()
    agent._hermes_strict_run = True
    agent._api_max_retries = 2

    empty_choices = SimpleNamespace(choices=[], model="bb-avatar", usage=None)
    result, calls = _run_with(agent, [empty_choices, empty_choices, empty_choices])

    assert len(calls) == 1
    assert result["completed"] is False
    assert result["failed"] is True


def test_strict_empty_content_response_is_terminal_without_retry():
    """An empty assistant message (no content, no reasoning) must not enter
    the empty-content retry loop on a strict run: one call, visible failure,
    never a spoken "(empty)"."""
    agent = _make_agent()
    agent._hermes_strict_run = True
    agent._api_max_retries = 2

    empty = _mock_response("")
    result, calls = _run_with(agent, [empty, empty, empty, empty])

    assert len(calls) == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result.get("final_response") != "(empty)"


def test_legacy_invalid_response_still_retries():
    """Control: without the strict flag the invalid-response loop keeps its
    retry behavior, so the gate above fails for the right reason."""
    agent = _make_agent()
    agent._api_max_retries = 2

    empty_choices = SimpleNamespace(choices=[], model="bb-avatar", usage=None)
    result, calls = _run_with(agent, [empty_choices, _mock_response("recovered")])

    assert len(calls) == 2
    assert result.get("final_response") == "recovered"


# ─── 6. No tool definitions reach the wire after a snapshot rebuild ─────────


def _tool_def(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def test_strict_no_tools_request_carries_no_tool_definitions_after_rebuild():
    """Even if some in-turn path repopulates agent.tools (MCP prologue,
    compaction rebuild), the request built for a tool_policy:none run must
    carry no tool definitions."""
    agent = _make_agent()
    agent._hermes_strict_run = True
    agent._strict_no_tools = True
    agent.tools = [_tool_def("terminal"), _tool_def("read_file")]
    agent.valid_tool_names = {"terminal", "read_file"}

    api_kwargs = agent._build_api_kwargs([{"role": "user", "content": "look"}])

    assert not api_kwargs.get("tools")
    assert "tool_choice" not in api_kwargs


def test_strict_no_tools_snapshot_rebuild_keeps_tool_surface_empty(monkeypatch):
    """The shared tool-snapshot rebuild (between-turns MCP refresh and the
    compaction commit both route through it) must not re-grant tools to a
    tool_policy:none agent."""
    import model_tools
    from tools import mcp_tool_agent

    agent = _make_agent()
    agent._hermes_strict_run = True
    agent._strict_no_tools = True
    agent.tools = []
    agent.valid_tool_names = set()

    registry_defs = [_tool_def("terminal"), _tool_def("mcp_late_tool")]
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: registry_defs)

    added = mcp_tool_agent.refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True)
    assert added == set()
    assert agent.tools == []
    assert agent.valid_tool_names == set()

    added = mcp_tool_agent.refresh_agent_mcp_tools(agent, content_aware=True)
    assert added == set()
    assert agent.tools == []
    assert agent.valid_tool_names == set()


@pytest.mark.parametrize("strict_no_tools", [False, True])
def test_saved_tool_prefix_respects_strict_no_tools(monkeypatch, strict_no_tools):
    from tools import mcp_tool_agent
    from tools.registry import registry

    agent = _make_agent()
    agent._hermes_strict_run = strict_no_tools
    agent._strict_no_tools = strict_no_tools
    agent.tools = []
    agent.valid_tool_names = set()
    entry = SimpleNamespace(name="terminal", schema=_tool_def("terminal")["function"])
    monkeypatch.setattr(registry, "get_entry", lambda name: entry)
    monkeypatch.setattr(registry, "get_all_entries", lambda: [entry])

    restored = mcp_tool_agent.restore_agent_tool_prefix(agent, ["terminal"])

    assert restored is (not strict_no_tools)
    expected_names = set() if strict_no_tools else {"terminal"}
    assert agent.valid_tool_names == expected_names
    assert {tool["function"]["name"] for tool in agent.tools} == expected_names


def test_strict_scratchpad_and_reasoning_only_cannot_continue_the_model():
    for content, reasoning in (
        ("<REASONING_SCRATCHPAD>unfinished", None),
        ("", "Internal reasoning without an answer"),
    ):
        agent = _make_agent()
        agent._hermes_strict_run = True
        agent._strict_no_tools = True
        response = _mock_response(content)
        response.choices[0].message.reasoning_content = reasoning
        result, calls = _run_with(agent, [response, _mock_response("UNEXPECTED FOLLOWUP")])
        assert len(calls) == 1
        assert result["failed"] is True
        assert "UNEXPECTED FOLLOWUP" not in str(result.get("final_response"))




def test_strict_request_debug_dump_never_archives_or_prints_pixels(tmp_path, monkeypatch, capsys):
    agent = _make_agent()
    agent._hermes_strict_run = True
    agent.logs_dir = tmp_path
    agent.session_id = "strict-request-dump"
    monkeypatch.setenv("HERMES_DUMP_REQUEST_STDOUT", "1")
    messages = _image_history()
    image_url = messages[0]["content"][1]["image_url"]["url"]
    result = agent._dump_api_request_debug(
        {"model": "bb-avatar", "messages": messages},
        reason="preflight",
        error=RuntimeError(image_url),
    )
    assert result is None
    assert not list(tmp_path.glob("request_dump_*.json"))
    assert image_url not in capsys.readouterr().out
