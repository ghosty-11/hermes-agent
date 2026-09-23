"""The ``ephemeral_context`` seam: private per-turn context rides the WIRE copy of the
current user row only (companion context contract).

Contracts under test, against the real prologue + request builder + a real SessionDB:

- ``pre_llm_call`` results may carry ``ephemeral_context``; it is collected ONCE per turn
  into ``TurnContext.ephemeral_user_context`` and never into ``plugin_user_context``.
- ``build_api_messages`` appends the same bytes to the wire copy of the current user row on
  EVERY pass (retries, tool iterations), at the prologue's ``current_turn_user_idx`` —
  never after later tool results, never into the live ``messages`` list.
- Private-only input leaves the persisted row clean: ``content`` unchanged, ``api_content``
  NULL. Replay-safe ``context`` still stamps ``api_content``; ephemeral bytes never do.
- A missing current-turn anchor withholds the ephemeral bytes entirely and records an
  explicit reason on the agent; no row is guessed and nothing is persisted.
- The hook payload exposes the server-created ``turn_origin``; a malformed origin is
  dropped (fail-closed to the default required policy), never partially trusted.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
import time
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest

import agent.turn_context as turn_context_module
from agent.turn_context import TurnContext, build_api_messages, build_turn_context
from hermes_state import SessionDB

EPHEMERAL = '<ambient_context version="1" origin="direct_reply">PRIVATE-SENTINEL</ambient_context>'


class _FakeGuardrails:
    def reset_for_turn(self):
        pass


class _FakeAgent:
    """Minimal stand-in covering what the prologue touches (mirrors test_turn_context)."""

    # build_api_messages reads this stock attribute on every agent.
    ephemeral_system_prompt = None

    def __init__(self):
        # Unique per instance: a fixed id made pytest warn about concurrent sessions
        # sharing one DB filename across tests.
        self.session_id = f"sess-eph-{uuid.uuid4().hex[:8]}"
        self.model = "test/model"
        self.provider = "openrouter"
        self.api_mode = "chat_completions"
        self.platform = "discord"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2,
            should_compress=lambda tokens=None: False,
            should_compress_info=lambda tokens=None: (False, None),
        )
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._emit_warning = MagicMock()
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        self._pending_cli_user_message = None
        self._session_persist_lock = threading.RLock()
        self._restore_primary_runtime = lambda: None
        self._session_db = None
        self._turn_origin = "__unset__"
        self._ephemeral_context_withheld = "__unset__"

    def _ensure_db_session(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, messages, _history=None):
        """Mirror the real flush enough to prove DB cleanliness: append the live rows
        (content + api_content sidecar) to the actual SessionDB."""
        for m in messages:
            if isinstance(m, dict) and m.get("role"):
                self._session_db.append_message(
                    self.session_id, m["role"], m.get("content"),
                    api_content=m.get("api_content"), timestamp=time.time(),
                )

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        yield


@contextmanager
def _hook_results(results):
    """Patch the pre_llm_call hook at its lazy import site; capture the payload."""
    captured = {}

    def _fake_invoke_hook(hook_name, **kwargs):
        captured["hook"] = hook_name
        captured["payload"] = kwargs
        return list(results)

    with patch("hermes_cli.lifecycle.invoke_hook", _fake_invoke_hook):
        yield captured


@pytest.fixture
def agent_db(tmp_path):
    agent = _FakeAgent()
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(agent.session_id, source="cli")
    agent._session_db = db
    return agent, db


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s if isinstance(s, str) else "",
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def _wire(agent, ctx, idx=None):
    return build_api_messages(
        agent, ctx.messages,
        current_turn_user_idx=ctx.current_turn_user_idx if idx is None else idx,
        ext_prefetch_cache=ctx.ext_prefetch_cache,
        plugin_user_context=ctx.plugin_user_context,
        ephemeral_user_context=ctx.ephemeral_user_context,
        moa_config=None, active_system_prompt=ctx.active_system_prompt,
    )[0]


def test_private_only_hook_context_leaves_database_clean(agent_db):
    agent, db = agent_db
    with _hook_results([{"ephemeral_context": EPHEMERAL}]) as captured:
        ctx = _build(agent)

    # The detection signal companion plugins key on.
    assert getattr(turn_context_module, "SUPPORTS_EPHEMERAL_CONTEXT", False) is True
    assert ctx.ephemeral_user_context == EPHEMERAL
    assert ctx.plugin_user_context == ""
    # The hook ran exactly once, in the prologue.
    assert captured["hook"] == "pre_llm_call"

    rows = db.get_messages_as_conversation(agent.session_id)
    user_row = rows[-1]
    assert user_row["role"] == "user"
    assert EPHEMERAL not in (user_row.get("content") or "")
    assert user_row.get("content") == "hello"
    assert user_row.get("api_content") is None, "private-only input must not stamp a sidecar"

    wire = _wire(agent, ctx)
    assert wire[-1]["content"] == "hello\n\n" + EPHEMERAL
    assert sum(EPHEMERAL in (m.get("content") or "") for m in wire) == 1


def test_ephemeral_bytes_never_reach_api_content_sidecar(agent_db):
    agent, db = agent_db
    with _hook_results([{"context": "SAFE-NOTES", "ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)

    rows = db.get_messages_as_conversation(agent.session_id)
    user_row = rows[-1]
    # Replay-safe context stamps the sidecar; the private bytes must NOT join it.
    assert user_row.get("api_content") == "hello\n\nSAFE-NOTES"
    assert EPHEMERAL not in (user_row.get("api_content") or "")

    wire = _wire(agent, ctx)
    # Wire copy = prologue sidecar (replay-safe bytes) + the ephemeral tail.
    assert wire[-1]["content"] == "hello\n\nSAFE-NOTES\n\n" + EPHEMERAL
    # The live dict never grew the private bytes.
    assert EPHEMERAL not in (ctx.messages[ctx.current_turn_user_idx].get("content") or "")


def test_identical_wire_bytes_across_retries_and_tool_iterations(agent_db):
    agent, _db = agent_db
    with _hook_results([{"ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)

    first = _wire(agent, ctx)
    user_idx_in_wire = len(first) - 1

    # A tool continuation appends assistant tool-call + tool rows, then a retry pass runs.
    ctx.messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]})
    ctx.messages.append({"role": "tool", "tool_call_id": "c1", "content": "result text"})

    second = _wire(agent, ctx)

    assert second[user_idx_in_wire]["content"] == first[user_idx_in_wire]["content"] == (
        "hello\n\n" + EPHEMERAL
    ), "identical ephemeral bytes at the same originating user row on every pass"
    assert second[-1]["role"] == "tool" and second[-1]["content"] == "result text"
    # Exactly one injection in the whole request — no duplication after tool results.
    assert sum(EPHEMERAL in (m.get("content") or "") for m in second) == 1


def test_multimodal_turn_appends_cloned_text_part_without_mutating_shared_list(agent_db):
    agent, _db = agent_db
    media_part = {"type": "image_url", "image_url": {"url": "file:///tmp/x.png"}}
    with _hook_results([{"ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent, user_message=[{"type": "text", "text": "look"}, media_part])

    live_content = ctx.messages[ctx.current_turn_user_idx]["content"]
    assert isinstance(live_content, list) and len(live_content) == 2
    frozen_live = [dict(p) for p in live_content]

    wire = _wire(agent, ctx)
    wire_content = wire[-1]["content"]
    assert isinstance(wire_content, list)
    assert wire_content[0] == {"type": "text", "text": "look"}
    assert wire_content[1] is not None and wire_content[1].get("type") == "image_url"
    assert wire_content[-1] == {"type": "text", "text": EPHEMERAL}
    # The live multimodal list is untouched — same length, same elements, no mutation.
    assert live_content == frozen_live and len(live_content) == 2
    assert wire_content[:-1] != live_content or wire_content is not live_content

    wire2 = _wire(agent, ctx)
    assert wire2[-1]["content"] == wire_content, "byte-identical wire projection every pass"


def test_missing_anchor_withholds_ephemeral_and_records_reason(agent_db):
    agent, _db = agent_db
    with _hook_results([{"ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)

    # Simulate compaction losing the current user row: the anchor no longer resolves.
    request = build_api_messages(
        agent, ctx.messages, current_turn_user_idx=None,
        ext_prefetch_cache="", plugin_user_context="",
        ephemeral_user_context=EPHEMERAL, moa_config=None, active_system_prompt="",
    )[0]
    assert all(EPHEMERAL not in (m.get("content") or "") for m in request), (
        "a missing anchor must withhold private context entirely"
    )
    reason = getattr(agent, "_ephemeral_context_withheld", None)
    assert isinstance(reason, str) and reason, "withholding must leave an explicit local reason"


def test_pre_llm_call_payload_carries_server_created_turn_origin(agent_db):
    agent, _db = agent_db
    origin = {"event_id": "evt-7", "kind": "ambient", "response_policy": "discretionary"}
    with _hook_results([]) as captured:
        _build(agent, turn_origin=origin)

    assert captured["payload"]["turn_origin"] == origin
    assert agent._turn_origin == origin


def test_malformed_turn_origin_is_dropped_fail_closed(agent_db):
    agent, _db = agent_db
    with _hook_results([]):
        ctx = _build(agent, turn_origin={"event_id": "evt-8", "kind": "ambient",
                                         "response_policy": "whenever-i-feel-like-it"})
    assert agent._turn_origin is None, "an unverifiable policy must never reach plugins"
    assert ctx.ephemeral_user_context == ""


def test_turn_context_field_defaults():
    """A turn with no hooks and no origin still yields a well-formed context."""
    ctx = TurnContext(
        user_message="u", original_user_message="u", messages=[], conversation_history=None,
        active_system_prompt="s", effective_task_id="t", turn_id="turn-1",
        current_turn_user_idx=0,
    )
    assert ctx.ephemeral_user_context == ""


def test_moa_turns_withhold_ephemeral_context(agent_db):
    """Review finding: MoA fans the wire copy out to reference models + the aggregator —
    extra recipients the private-context contract never authorized. Fail closed: the
    private bytes are withheld for the whole request, with the explicit reason recorded."""
    agent, _db = agent_db
    with _hook_results([{"ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)
    assert ctx.ephemeral_user_context == EPHEMERAL
    request = build_api_messages(
        agent, ctx.messages, current_turn_user_idx=ctx.current_turn_user_idx,
        ext_prefetch_cache="", plugin_user_context="",
        ephemeral_user_context=EPHEMERAL, moa_config={"presets": {}}, active_system_prompt="",
    )[0]
    assert all(EPHEMERAL not in (m.get("content") or "") for m in request)
    assert getattr(agent, "_ephemeral_context_withheld", None) == (
        "moa_reference_models_would_receive_private_context")


def test_virtual_moa_provider_withholds_ephemeral_without_inline_config(agent_db):
    agent, db = agent_db
    agent.provider = "moa"
    with _hook_results([{"ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)
    request = build_api_messages(
        agent, ctx.messages, current_turn_user_idx=ctx.current_turn_user_idx,
        ext_prefetch_cache=ctx.ext_prefetch_cache,
        plugin_user_context=ctx.plugin_user_context,
        ephemeral_user_context=ctx.ephemeral_user_context,
        moa_config=None, active_system_prompt=ctx.active_system_prompt,
    )[0]
    assert all(EPHEMERAL not in str(row.get("content") or "") for row in request)
    assert db.get_messages_as_conversation(agent.session_id)[-1].get("api_content") is None
    assert getattr(agent, "_ephemeral_context_withheld", None) == (
        "moa_reference_models_would_receive_private_context")


def test_companion_withholds_replayable_hook_and_native_prefetch_before_store(agent_db):
    agent, db = agent_db
    manager = types.SimpleNamespace(
        on_turn_start=MagicMock(),
        prefetch_all=MagicMock(return_value="NATIVE-PREFETCH-SENTINEL"),
        describe_recall=MagicMock(return_value="recall ready"),
    )
    agent._memory_manager = manager
    with patch("hermes_cli.config.load_config",
               return_value={"optmem": {"speaker_scoped": True}}), \
         patch("agent.turn_context.is_trivial_prompt", return_value=False), \
         _hook_results([
             {"context": "PERSISTENT-HOOK-SENTINEL",
              "ephemeral_context": EPHEMERAL},
             "BARE-HOOK-SENTINEL",
         ]):
        ctx = _build(agent, user_message="tell me about our prior conversations")

    manager.on_turn_start.assert_not_called()
    manager.prefetch_all.assert_not_called()
    assert ctx.ext_prefetch_cache == ""
    assert ctx.plugin_user_context == ""
    assert ctx.ephemeral_user_context == EPHEMERAL
    user = db.get_messages_as_conversation(agent.session_id)[-1]
    assert user["content"] == "tell me about our prior conversations"
    assert user.get("api_content") is None
    wire = _wire(agent, ctx)
    assert wire[-1]["content"] == user["content"] + "\n\n" + EPHEMERAL
    assert all("PERSISTENT-HOOK" not in str(row) and "BARE-HOOK" not in str(row)
               and "NATIVE-PREFETCH" not in str(row) for row in wire)


def test_companion_skips_untrusted_context_before_stringification_or_spill(agent_db):
    agent, db = agent_db

    class UntrustedContext:
        def __str__(self):
            raise AssertionError("blocked context must never stringify")
        def __bool__(self):
            raise AssertionError("blocked context must never be truth-tested")

    with patch("hermes_cli.config.load_config",
               return_value={"optmem": {"speaker_scoped": True}}), \
         patch("tools.hook_output_spill.spill_if_oversized") as spill, \
         _hook_results([{"context": UntrustedContext(),
                         "ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)
    spill.assert_not_called()
    assert ctx.ephemeral_user_context == EPHEMERAL
    assert db.get_messages_as_conversation(agent.session_id)[-1].get("api_content") is None


def test_discord_config_failure_withholds_replayable_context(agent_db):
    agent, db = agent_db
    with patch("hermes_cli.config.load_config",
               side_effect=RuntimeError("profile unreadable")), \
         _hook_results([{"context": "PERSISTENT-HOOK-SENTINEL",
                         "ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)
    assert ctx.ephemeral_user_context == EPHEMERAL
    assert ctx.plugin_user_context == ""
    assert db.get_messages_as_conversation(agent.session_id)[-1].get("api_content") is None


def test_companion_keeps_server_owned_one_shot_note_but_not_plugin_context(agent_db):
    agent, db = agent_db
    agent._gateway_turn_context_notes = "SERVER-OWNED-NOTICE"
    with patch("hermes_cli.config.load_config",
               return_value={"optmem": {"speaker_scoped": True}}), \
         _hook_results([{"context": "PERSISTENT-HOOK-SENTINEL",
                         "ephemeral_context": EPHEMERAL}]):
        ctx = _build(agent)
    assert ctx.plugin_user_context == "SERVER-OWNED-NOTICE"
    user = db.get_messages_as_conversation(agent.session_id)[-1]
    assert user.get("api_content") == "hello\n\nSERVER-OWNED-NOTICE"
    assert EPHEMERAL not in user["api_content"]
    assert "PERSISTENT-HOOK" not in user["api_content"]
