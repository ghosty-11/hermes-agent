"""Coordinated-merge turn regressions: the composed consumer, not the mixins in isolation.

The framework merge reinstated older ``_hmwa`` bodies inside ``gateway.run_turn``'s
``GatewayTurnMixin`` and ``gateway.run_turn_runner``'s ``TurnRunner``, shadowing the incoming
prepare/persistence/progress mixins those same classes inherit. Every test here drives the
composed classes — ``GatewayRunner`` (whose MRO carries ``GatewayTurnMixin``) and ``TurnRunner`` —
so the shadowing bodies are the code under test, and asserts consumer-visible outcomes only:
replies, transcript rows, admission results, viewer/API event streams, queue placement, prompt
bytes and the live approval bypass.

The authoritative incoming behavior (``run_turn_prepare``, ``run_turn_persistence``,
``run_turn_progress``) is the contract; these assertions are expected RED on the merged tree
until it is restored. Control cases that must stay green on both (launch seeding on an
untouched route, messaging reasoning projection, the non-finite footer) prove each fixture is
alive rather than broken.
"""

import asyncio
import os
from types import SimpleNamespace

import pytest
import pytest_asyncio

from agent.i18n import t
from agent.turn_failure_copy import (
    FAILED_TURN_DISPLAY_KIND,
    PARTIAL_FAILED_TURN_NOTICE,
    launch_credentials_copy,
)
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.session_transcript import TranscriptReadError


# ── shared fixtures ─────────────────────────────────────────────────────────────────────


class _Store:
    """Async routing-store stand-in: scripted transcript reads, recorded writes."""

    def __init__(self, history=None, error=None):
        self.history = [] if history is None else history
        self.error = error
        self.appended = []
        self.reset_calls = []
        self.tail = "assistant"
        self.has_owner = True
        self.has_sessions = False
        # ``GatewayRunner.async_session_store`` reuses a cached facade only when its
        # ``_store`` is the runner's ``session_store``; a self-reference satisfies that check.
        self._store = self

    async def load_transcript(self, _sid):
        if self.error is not None:
            raise self.error
        return self.history

    async def has_any_sessions(self):
        return self.has_sessions

    async def has_input_owner(self, *_args, **_kwargs):
        return self.has_owner

    async def transcript_tail_role(self, _sid):
        return self.tail

    async def append_to_transcript(self, _sid, row):
        self.appended.append(row)

    async def reset_session(self, session_key, *_args, **_kwargs):
        self.reset_calls.append(session_key)
        return SimpleNamespace(session_id="fresh", session_key=session_key)


async def _anone(*_args, **_kwargs):
    return None


def _msg(text, platform, *, internal=False, inherit=False, chat_type="dm"):
    return MessageEvent(
        text=text,
        source=SessionSource(platform=platform, chat_id="chat", user_id="u", chat_type=chat_type),
        message_id="m1",
        internal=internal,
        inherit_channel_pin=inherit,
    )


def _prepare_runner(monkeypatch, store, recorder):
    """A ``GatewayRunner`` whose composed ``_hmwa_prepare_turn`` runs with hermetic
    collaborators; the decision structure under test (yolo restore, prompt policy, refusal
    passthrough, failure capture) is the real composed method."""
    import gateway.run as gateway_run
    import gateway.run_turn_prepare as gateway_prepare

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(gateway_prepare, "build_session_context", lambda *a, **k: "CTX")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={})
    runner.adapters = {}
    runner.session_store = store
    runner._async_session_store = store  # canonical backing field of the read-only property

    async def _open(*_args, **_kwargs):
        return (False, False)

    async def _hygiene(*_args, **_kwargs):
        return []

    runner._hmwa_open_session = _open
    runner._hmwa_acquire_turn_lease = _anone
    runner._mark_durable_active_turn = _anone
    runner._hmwa_run_session_hygiene = _hygiene
    runner._clear_session_env = lambda tokens: None
    runner._set_session_env = lambda context: []
    runner._voice_channel_sidecar_note = lambda *a, **k: None
    runner._hmwa_stop_typing_for_turn = _anone
    runner._delivery_adapter_for = lambda source: None
    runner._bind_adapter_run_generation = lambda *a, **k: None
    runner._set_pending_turn_sidecar_notes = lambda key, notes: recorder["staged"].extend(notes)
    runner._hmwa_deliver_auto_reset_notice = _anone

    async def _rehydrate(session_key, expected_session_id):
        recorder["rehydrations"].append((session_key, expected_session_id))

    runner._rehydrate_prompt_pins = _rehydrate

    def _render(context, redact_pii, session_key, *, internal=False):
        recorder["renders"].append(bool(internal))
        return "PINNED" if internal else "FRESH"

    runner._pinned_session_context_prompt = _render

    async def _inbound_text(*, event, source, history, session_key):
        recorder["inbound_calls"].append(event)
        return recorder["inbound_result"]

    runner._prepare_profile_scoped_inbound_message_text = _inbound_text
    return runner


def _recorder():
    return {"staged": [], "rehydrations": [], "renders": [], "inbound_calls": [],
            "inbound_result": "hello", "notices": []}


def _entry(yolo=None):
    return SimpleNamespace(yolo=yolo, session_id="sid", session_key="key", resume_pending=False)


# ── persisted approval-bypass revocation (run_turn.py:914-915) ──────────────────────────


@pytest.mark.asyncio
async def test_persisted_yolo_revocation_is_restored_before_launch_seeding(monkeypatch):
    """A persisted yolo=False is the authoritative routing value: preparation must register it
    before TurnRunner's frozen yolo=True launch seeding (run_turn_runner.py applies
    ``apply_launch_yolo``), so an explicitly revoked bypass cannot be re-enabled."""
    from tools.approval import clear_session, enable_session_yolo, is_session_yolo_enabled
    from tools.approval_yolo import apply_launch_yolo

    key = "coordinated-yolo-revoked"
    try:
        clear_session(key)
        # The stale live bypass a fresh in-process owner can inherit.
        enable_session_yolo(key)
        recorder = _recorder()
        runner = _prepare_runner(monkeypatch, _Store(), recorder)

        await runner._hmwa_prepare_turn(_msg("hi", Platform.TELEGRAM), _msg("hi", Platform.TELEGRAM).source,
                                        _entry(yolo=False), key, key, 1)
        assert not is_session_yolo_enabled(key), "persisted yolo=False must revoke the live bypass"

        # The exact seeding TurnRunner performs for a frozen yolo=True launch policy.
        apply_launch_yolo(key)
        assert not is_session_yolo_enabled(key), "launch seeding must not re-arm a revoked bypass"
    finally:
        clear_session(key)


@pytest.mark.asyncio
async def test_untouched_route_still_receives_the_launch_seed(monkeypatch):
    """Control: yolo=None means nobody toggled the route, so the launch policy seeds normally —
    proves the seeding mechanism is live in this fixture (green on both trees)."""
    from tools.approval import clear_session, is_session_yolo_enabled
    from tools.approval_yolo import apply_launch_yolo

    key = "coordinated-yolo-untouched"
    try:
        clear_session(key)
        recorder = _recorder()
        runner = _prepare_runner(monkeypatch, _Store(), recorder)

        await runner._hmwa_prepare_turn(_msg("hi", Platform.TELEGRAM), _msg("hi", Platform.TELEGRAM).source,
                                        _entry(yolo=None), key, key, 1)
        apply_launch_yolo(key)
        assert is_session_yolo_enabled(key) is True
    finally:
        clear_session(key)


# ── local-session context prompt + synthetic pin reuse (run_turn.py:926-930) ────────────


@pytest.mark.asyncio
async def test_local_session_prepare_carries_no_messaging_context_prompt(monkeypatch):
    """A LOCAL session (CLI one-shot, ACP, TUI attach) must not acquire messaging-only
    system-prompt content: its context prompt stays empty instead of the pinned render."""
    recorder = _recorder()
    runner = _prepare_runner(monkeypatch, _Store(), recorder)

    prepared, _tokens = await runner._hmwa_prepare_turn(
        _msg("hi", Platform.LOCAL), _msg("hi", Platform.LOCAL).source, _entry(), "key", "key", 1)

    assert prepared.context_prompt == "", "local handoffs must keep the classic in-process prompt"
    assert recorder["renders"] == [], "the messaging pin renderer must not run for a local session"


@pytest.mark.asyncio
async def test_inherit_channel_pin_reuses_the_channel_pin(monkeypatch):
    """A synthetic goal/heartbeat/resume turn (inherit_channel_pin, not internal) rehydrates the
    durable pin and renders through the pin-reuse path, so it reuses the originating channel's
    prompt instead of re-keying it."""
    recorder = _recorder()
    runner = _prepare_runner(monkeypatch, _Store(), recorder)
    event = _msg("[goal] continue", Platform.TELEGRAM, inherit=True)

    prepared, _tokens = await runner._hmwa_prepare_turn(
        event, event.source, _entry(), "key", "key", 1)

    assert recorder["rehydrations"] == [("key", "sid")], \
        "synthetic continuations must adopt the durable pin snapshot"
    assert prepared.context_prompt == "PINNED", \
        "synthetic continuations must render through the pin-reuse path, not a fresh render"


# ── internal first turns and human onboarding (run_turn.py:965) ─────────────────────────


@pytest.mark.asyncio
async def test_internal_first_turn_skips_human_onboarding(tmp_path, monkeypatch):
    """An internal first turn (plugin/system) makes no human first contact: no onboarding
    sidecar note rides its message and no platform notice is sent."""
    from unittest.mock import AsyncMock

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "home")
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")

    recorder = _recorder()
    runner = _prepare_runner(monkeypatch, _Store(), recorder)
    notices = AsyncMock()
    runner._deliver_platform_notice = notices

    event = _msg("wake", Platform.TELEGRAM, internal=True)
    await runner._hmwa_prepare_turn(event, event.source, _entry(), "key", "key", 1)

    assert recorder["staged"] == [], "an internal first turn must not carry onboarding notes"
    notices.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_first_turn_still_gets_the_onboarding_note(tmp_path, monkeypatch):
    """Control: a human first DM does carry the first-contact note (green on both trees),
    proving the onboarding fixture above is live."""
    from unittest.mock import AsyncMock

    from agent.onboarding import profile_build_directive

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "home")
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")

    recorder = _recorder()
    runner = _prepare_runner(monkeypatch, _Store(), recorder)
    runner._deliver_platform_notice = AsyncMock()

    event = _msg("hello there", Platform.TELEGRAM)
    await runner._hmwa_prepare_turn(event, event.source, _entry(), "key", "key", 1)

    assert recorder["staged"] == [profile_build_directive().strip()]


# ── refused local context references (run_turn.py:977-978) ──────────────────────────────


@pytest.mark.asyncio
async def test_refused_local_context_reference_is_delivered_as_the_reply(monkeypatch):
    """A blocked local ``@`` reference answers with its refusal instead of aborting silently:
    the refusal stored on the event is the turn's reply."""
    refusal = "Those @file references were refused: the turn did not run."
    recorder = _recorder()
    recorder["inbound_result"] = None
    runner = _prepare_runner(monkeypatch, _Store(), recorder)

    event = _msg("compare @file:a.txt @file:b.txt", Platform.LOCAL)
    event._gateway_turn_refusal = refusal
    reply, _tokens = await runner._hmwa_prepare_turn(event, event.source, _entry(), "key", "key", 1)

    assert reply == refusal, "the refused turn's reply must be the refusal explanation itself"


# ── pre-execution failures keep a failed admission (run_turn.py:855-857, 963) ───────────


@pytest.mark.asyncio
async def test_unreadable_history_records_a_failed_admission(monkeypatch):
    """An unreadable transcript stops the turn with the history-unavailable apology AND commits
    a failed result for the executing admission, so no surface settles it completed."""
    from gateway.session_results import execution_result

    recorder = _recorder()
    store = _Store(error=TranscriptReadError("sid"))
    runner = _prepare_runner(monkeypatch, store, recorder)

    captured = {}
    token = execution_result.set(captured)
    try:
        event = _msg("hi", Platform.TELEGRAM)
        reply, _tokens = await runner._hmwa_prepare_turn(
            event, event.source, _entry(), "key", "key", 1)
    finally:
        execution_result.reset(token)

    assert reply == t("gateway.errors.history_unavailable")
    result = captured.get("result")
    assert result is not None, "the admission must not fall through as an ordinary reply"
    assert result["failed"] is True and result["completed"] is False
    assert result["final_response"] == ""


@pytest.mark.asyncio
async def test_agent_exception_records_a_failed_admission_with_safe_reply(monkeypatch):
    """When agent construction fails before a TurnRunner result exists, the apology keeps its
    chat-safe wording while the executing admission is committed failed."""
    from gateway.session_results import execution_result

    recorder = _recorder()
    store = _Store()
    store.tail = "user"
    runner = _prepare_runner(monkeypatch, store, recorder)

    event = _msg("do the thing", Platform.TELEGRAM)
    prepared = SimpleNamespace(history=[], message_text="do the thing", persist_user_message=None,
                               persist_user_timestamp=None, persist_user_display_kind=None,
                               persistence_session_id="sid", persistence_owner="owner")

    captured = {}
    token = execution_result.set(captured)
    try:
        reply = await runner._hmwa_agent_error_reply(
            RuntimeError("provider exploded"), event, event.source, _entry(), "key", prepared)
    finally:
        execution_result.reset(token)

    assert PARTIAL_FAILED_TURN_NOTICE in reply
    result = captured.get("result")
    assert result is not None, "the failed admission must be recorded, not just answered"
    assert result["failed"] is True and result["completed"] is False
    assert result["error"]


@pytest.mark.asyncio
async def test_launch_credentials_failure_records_reason_and_names_new(monkeypatch):
    """A route whose launch credentials are gone is non-retryable: the admission is stamped with
    the ``launch_credentials_unavailable`` reason and the reply is the /new copy, not the
    generic retry guidance."""
    from gateway.session_results import execution_result

    class _CredentialsGone(RuntimeError):
        reason = "launch_credentials_unavailable"

    recorder = _recorder()
    runner = _prepare_runner(monkeypatch, _Store(), recorder)

    err = _CredentialsGone("route key rotated")
    event = _msg("run", Platform.TELEGRAM)
    prepared = SimpleNamespace(history=[], message_text="run", persist_user_message=None,
                               persist_user_timestamp=None, persist_user_display_kind=None,
                               persistence_session_id="sid", persistence_owner="owner")

    captured = {}
    token = execution_result.set(captured)
    try:
        reply = await runner._hmwa_agent_error_reply(err, event, event.source, _entry(), "key", prepared)
    finally:
        execution_result.reset(token)

    assert reply == launch_credentials_copy(err)
    assert PARTIAL_FAILED_TURN_NOTICE not in reply, "a dead route must not invite a resend"
    result = captured.get("result")
    assert result is not None
    assert result["failure_reason"] == "launch_credentials_unavailable"
    assert result["failure_retryable"] is False


# ── failed-turn tail closure (run_turn.py:454-455) ──────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["user", "tool"])
async def test_failed_turn_closes_open_user_and_tool_tails(tail):
    """A failed turn whose durable tail is the unanswered user row OR a completed tool row gets
    its assistant boundary with the actions-may-have-run warning, so later replay cannot treat
    the unfinished tool-bearing turn as an ordinary continuation."""
    harness = GatewayTurnMixin()
    store = _Store()
    store.tail = tail
    harness.async_session_store = store

    await harness._hmwa_close_failed_turn("sid", PARTIAL_FAILED_TURN_NOTICE)

    assert len(store.appended) == 1
    boundary = store.appended[0]
    assert boundary["role"] == "assistant"
    assert boundary["content"] == PARTIAL_FAILED_TURN_NOTICE
    assert boundary["display_kind"] == FAILED_TURN_DISPLAY_KIND


@pytest.mark.asyncio
async def test_failed_turn_does_not_stack_a_second_boundary():
    """Control: an assistant tail is already closed (green on both trees)."""
    harness = GatewayTurnMixin()
    store = _Store()
    store.tail = "assistant"
    harness.async_session_store = store

    await harness._hmwa_close_failed_turn("sid", PARTIAL_FAILED_TURN_NOTICE)
    assert store.appended == []


# ── compression-exhaustion reset (run_turn.py:513-518) ─────────────────────────────────


@pytest.mark.asyncio
async def test_api_session_keeps_its_identity_after_compression_exhaustion(tmp_path, monkeypatch):
    """An API conversation stays addressable after exhaustion: the caller addresses it by id and
    every door re-validates that binding, so the route must not rotate; the typed result is
    returned unchanged."""
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_api_turn import admit_api_turn
    from gateway.session_authority import SessionAuthority, initialize_session_authority

    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    # ``_VOICE_MODE_PATH`` is frozen at import time from the collection-time home, which the
    # HomeIOGuard protects; point it at the test's temp home (a missing file loads no modes).
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    runner = GatewayRunner(GatewayConfig())
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.API_SERVER: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    adapter._session_db = authority.db
    try:
        admit_api_turn(adapter, session_id="dm1", user_message="disk status?", conversation_history=[])
        source = authority.sessions["dm1"].source
        route = runner.session_store._generate_session_key(source)
        exhausted = {"failed": True, "compression_exhausted": True,
                     "failure_reason": "context_overflow", "messages": []}

        response, entry = await runner._hmwa_compression_exhaustion_reset(
            exhausted, "copy", runner.session_store.lookup_by_session_key(route), route, source)

        assert entry.session_id == "dm1", "the API route must not rotate out from its caller"
        assert response == "copy", "the API caller gets the typed exhaustion result, not a reset note"
        admit_api_turn(adapter, session_id="dm1", user_message="next", conversation_history=[])
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        authority.db.close()


@pytest_asyncio.fixture
async def native(tmp_path, monkeypatch):
    """Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; only the
    agent turn is a stand-in (the pattern of test_native_fifo_context_exhaustion)."""
    import asyncio

    from gateway.platforms.base import SendResult
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from hermes_state_runtime import list_session_admissions
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="fixture-reply")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    native = SimpleNamespace(runner=runner, adapter=adapter, authority=authority, source=source,
                             list_session_admissions=list_session_admissions)
    try:
        yield native
    finally:
        from gateway.session_runtime_workers import mutation_tasks
        await asyncio.gather(*mutation_tasks(authority))
        authority.db.close()


async def _admit(native, text):
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home

    event = MessageEvent(text=text, source=native.source, message_id=text)
    with native_callback(native.runner, event, get_hermes_home()):
        return event, await native.authority.admit_native(event)


@pytest.mark.asyncio
async def test_exhaustion_reset_rehomes_queued_messages_with_their_identity(native, monkeypatch):
    """Messages queued behind an exhausted turn still run, in order, on the fresh session the
    auto-reset creates, keeping their admission identity — not paused forever on the retired
    session while their input is cleared."""
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result

    runner, authority, db = native.runner, native.authority, native.authority.db
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    old = admitted[0][1].ref
    executed, fresh = [], {}

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        captured = execution_result.get()
        if event.text == "A":
            result = {"failed": True, "compression_exhausted": True, "messages": []}
            captured["result"] = result
            _, new_entry = await runner._hmwa_compression_exhaustion_reset(
                result, "context exhausted", entry, session_key, source)
            fresh["sid"] = new_entry.session_id
        else:
            captured["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(old)

    new = fresh["sid"]
    assert new != old.session_id, "the exhausted route must rotate to a fresh session"
    if new in authority.sessions:
        await authority._drain(SessionRef(authority.profile_id, new))

    assert [(text, sid) for text, sid in executed if text in ("B", "C")] == [
        ("B", new), ("C", new)], "queued messages must run on the fresh session"

    new_rows = native.list_session_admissions(db, session_id=new, pending_only=False)
    assert [(r["request_id"], r["status"], r["outcome"]) for r in new_rows if r["request_id"] in ("B", "C")] \
        == [("B", "terminal", "completed"), ("C", "terminal", "completed")]
    assert [r["admission_id"] for r in new_rows if r["request_id"] in ("B", "C")] \
        == [admitted[1][1].admission_id, admitted[2][1].admission_id], \
        "the re-home keeps each message's admission identity"


@pytest.mark.asyncio
async def test_local_active_admission_defers_its_exhaustion_reset(monkeypatch):
    """A LOCAL session's own reset refuses while its admission is executing: the reset is handed
    to the authority drain (deferred), not applied against the live route."""
    from gateway.session_ingress import executing_admission

    harness = GatewayTurnMixin()
    store = _Store()
    harness.async_session_store = store
    harness._evict_cached_agent = lambda *a, **k: None
    harness._clear_conversation_scope = lambda *a, **k: None
    harness._sync_telegram_topic_binding = lambda *a, **k: None
    deferred = {}
    harness.session_authorities = SimpleNamespace(active=lambda: SimpleNamespace(deferred_resets=deferred))
    source = SimpleNamespace(platform=Platform.LOCAL)
    entry = SimpleNamespace(session_id="old", session_key="key")

    token = executing_admission.set(object())
    try:
        response, returned = await harness._hmwa_compression_exhaustion_reset(
            {"failed": True, "compression_exhausted": True, "messages": []},
            "context exhausted", entry, "key", source)
    finally:
        executing_admission.reset(token)

    assert store.reset_calls == [], "the reset must not run under the active admission"
    assert "key" in deferred, "the reset must be queued on the authority drain"
    assert returned is entry and returned.session_id == "old"
    assert response != "context exhausted", "the caller still learns the session was exhausted"


# ── canonical tool/viewer events (run_turn_runner.py:985-991 and siblings) ──────────────


def _turn(platform, published, api_calls, *, live=None):
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(
        source=SessionSource(platform=platform, chat_id="chat", user_id="u"),
        mute_notification_reply=False, scheduled_heartbeat=False,
        streaming_tts_consumer_holder=[None], stream_consumer_holder=[None],
        interim_assistant_messages_enabled=False, user_config={},
        resolve_display_setting=lambda *a: None, _run_still_current=lambda: True,
        _voice_ack_guild=[None], _native_slack_task_cards=False, progress_mode="all",
        session_id="s", progress_queue=None, _status_adapter=SimpleNamespace(),
        _status_chat_id="chat", _status_thread_metadata=None,
        _hooks_ref=SimpleNamespace(loaded_hooks=False, emit=_anone), log_queue=None,
    )
    turn._runner = SimpleNamespace(config=None, _delivery_adapter_for=lambda source: None)
    live = live or SimpleNamespace(tool_progress_mode=None)
    authority = SimpleNamespace(sessions={"s": live})
    turn._approval_owner = (authority, "s", 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    turn._publish_api_tool = lambda *a, **k: api_calls.append(a)
    turn._schedule = lambda coro, _message: coro.close()
    return turn


def test_tool_start_publishes_the_canonical_event_for_every_route():
    """Every route's viewers and API observers receive the ID-correlated ``tool.start`` frame,
    paired with the completions the inherited callback already publishes."""
    for platform in (Platform.LOCAL, Platform.TELEGRAM):
        published, api_calls = [], []
        turn = _turn(platform, published, api_calls)

        turn.combined_tool_start_callback("c1", "read_file", {"path": "x"})

        starts = [payload for kind, payload in published if kind == "tool.start"]
        assert starts, f"{platform}: the canonical tool.start frame must be published"
        assert starts[0]["tool_id"] == "c1" and starts[0]["name"] == "read_file"
        assert any(call[:3] == ("tool.start", "c1", "read_file") for call in api_calls), \
            f"{platform}: this admission's API observers must see the tool start"


def test_local_viewer_tool_lifecycle_keeps_start_state_for_durations():
    """A local viewer's tool lifecycle carries the start-note state the completion reads: the
    start callback records it, so the completion's payload has a real duration."""
    published, api_calls = [], []
    turn = _turn(Platform.LOCAL, published, api_calls)

    turn.combined_tool_start_callback("w1", "write_file", {"path": "x", "content": "hi"})
    turn.combined_tool_complete_callback("w1", "write_file", {"path": "x", "content": "hi"},
                                         '{"bytes_written": 2}')

    complete = next(payload for kind, payload in published if kind == "tool.complete")
    assert isinstance(complete["duration_s"], float) and complete["duration_s"] >= 0, \
        "the completion must correlate with the recorded start"


def test_step_and_status_updates_reach_local_viewers(monkeypatch):
    """Local viewers read their live state from the event stream: iteration steps publish
    ``agent.step`` and status lines publish ``status.update`` instead of a platform send."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_prepare_gateway_status_message",
                        lambda platform, event_type, message: "Working…")
    published, api_calls = [], []
    turn = _turn(Platform.LOCAL, published, api_calls)

    turn._step_callback_sync(1, [])
    turn._status_callback_sync("lifecycle", "working hard")

    assert ("agent.step", {"iteration": 1}) in published
    assert ("status.update", {"kind": "lifecycle", "text": "Working…"}) in published


def test_messaging_route_keeps_status_off_the_event_stream(monkeypatch):
    """Control: a messaging route's status still goes to its platform adapter, never the
    viewer stream (green on both trees)."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_prepare_gateway_status_message",
                        lambda platform, event_type, message: "Working…")
    published, api_calls = [], []
    turn = _turn(Platform.TELEGRAM, published, api_calls)

    turn._status_callback_sync("lifecycle", "working hard")
    assert [kind for kind, _ in published if kind == "status.update"] == []


# ── local final answers: reasoning and footer (run_turn.py:335-379) ────────────────────


def test_local_final_never_duplicates_reasoning(monkeypatch):
    """Local viewers receive reasoning through their event stream: the final answer must not
    carry the projected reasoning block as well."""
    import gateway.run as gateway_run
    import gateway.session_setting_commands as session_setting_commands

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": True}})
    monkeypatch.setattr(session_setting_commands, "local_policy",
                        lambda runner, src: SimpleNamespace(source="cli"))
    harness = GatewayTurnMixin()
    harness._show_reasoning = True
    result = {"last_reasoning": "private chain"}

    local = harness._hmwa_prepend_reasoning(result, "final answer",
                                            SimpleNamespace(platform=Platform.LOCAL), False)
    assert local == "final answer", "the local viewer already saw reasoning as events"


def test_messaging_final_keeps_the_reasoning_projection(monkeypatch):
    """Control: a messaging platform still gets the projected block (green on both trees)."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_load_gateway_config",
                        lambda: {"display": {"show_reasoning": True}})
    harness = GatewayTurnMixin()
    harness._show_reasoning = True

    reply = harness._hmwa_prepend_reasoning({"last_reasoning": "private chain"}, "final answer",
                                            SimpleNamespace(platform=Platform.TELEGRAM), False)
    assert "private chain" in reply and reply.endswith("final answer")


def test_finite_turn_final_answer_carries_no_runtime_footer(monkeypatch):
    """A finite (``-z`` / ``chat -q``) turn prints the completion text as the answer: the
    runtime-metadata footer must not become part of it."""
    from gateway import runtime_footer
    from gateway.session_finite import finite_turn_scope

    monkeypatch.setattr(runtime_footer, "build_footer_line", lambda **kwargs: "FOOTER-MERGE")
    harness = GatewayTurnMixin()

    with finite_turn_scope(True):
        assert harness._hmwa_runtime_footer_line({"model": "m"}, SimpleNamespace(platform=Platform.LOCAL), 1.0) == ""
    assert harness._hmwa_runtime_footer_line({"model": "m"}, SimpleNamespace(platform=Platform.LOCAL), 1.0) \
        == "FOOTER-MERGE"

# ── queued first-response delivery addressing (run_turn_runner.py, run_turn.py:222) ─────
#
# The durable delivery record on the final assistant row needs three things from the
# composed turn: the producer-exported ``turn_id`` carried through the runner's envelope,
# the post-run EFFECTIVE session (a compression split repoints the runner result while the
# turn context keeps the parent), and exactly one speech synthesis when the runner's voice
# reply owns a completed streamed handoff. Every fixture feeds REAL turn identity through
# the code under test — no hand-stamped ``turn_id`` downstream of the producer.


@pytest.fixture
def _queued_ledger(tmp_path, monkeypatch):
    """Hermetic delivery ledger for the queued first-response bracket: obligations land in
    this test's own SQLite, never the operator's state.db (pattern: test_egress_decline_terminal)."""
    from gateway import delivery_ledger as dl

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    return home


class _SendTransport(BasePlatformAdapter):
    """Base-contract adapter whose only transport is a recorded successful send."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content))
        return SendResult(success=True, message_id=f"m{len(self.sent)}")


class _RoutingStore:
    """Routing-store stand-in: the entry ``_sync_session_after_run`` repoints after a split,
    with the writes recorded. The durable surface under test is the SessionDB row, not this."""

    def __init__(self, session_key, session_id):
        self._entries = {session_key: SimpleNamespace(session_id=session_id, session_key=session_key)}
        self.saves = []
        self.peers = []

    def _save(self):
        self.saves.append(True)

    def _record_gateway_session_peer(self, session_id, session_key, source):
        self.peers.append((session_id, session_key))


def _run_turn_envelope(monkeypatch, db, *, session_key, session_id, turn_id,
                       question, answer, split_to=None):
    """Run one real ``TurnRunner._run_sync_scoped`` turn; returns ``(runner, ctx, envelope)``.

    The agent double honours the producer contract: it flushes this turn's rows into the real
    SessionDB mid-run — stamping the DB's ``_row_id`` onto the live dicts, exactly what
    conversation_loop's flush does — and the result is stamped by the REAL
    ``export_current_turn_boundary`` from ``agent._current_turn_id``. ``split_to`` rotates the
    session mid-run like a context-compression split, leaving the turn context on the parent."""
    import gateway.run as gateway_run
    from agent.turn_context import export_current_turn_boundary
    from gateway import session_policy
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(session_policy, "policy_for_source", lambda runner, source: None)
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 8)
    ctx = TurnContext(
        source=SessionSource(platform=Platform.SLACK, chat_id="chat", user_id="u", chat_type="dm"),
        session_key=session_key, session_id=session_id, run_generation=1, message=question,
        event_message_id="in-1", inbound_message_id="in-1", user_config={},
        _status_thread_metadata={}, _run_still_current=lambda: True,
    )
    agent = SimpleNamespace(session_id=session_id, _current_turn_id=turn_id, model="m")

    def run_conversation():
        effective = session_id
        if split_to is not None:
            db.end_session(session_id, "compression")
            effective = db.create_session(split_to, "test", parent_session_id=session_id)
            agent.session_id = effective
        user_row = db.append_message(effective, "user", question)
        answer_row = db.append_message(effective, "assistant", answer)
        raw = {
            "final_response": answer, "completed": True, "api_calls": 1,
            "messages": [
                {"role": "user", "content": question, "_row_id": user_row},
                {"role": "assistant", "content": answer, "_row_id": answer_row},
            ],
        }
        return export_current_turn_boundary(agent, raw, question)

    runner = object.__new__(GatewayRunner)
    runner._provider_routing = {}
    runner._resolve_session_agent_runtime = lambda **kw: ("m", {"provider": "custom"})
    runner._resolve_session_reasoning_config = lambda **kw: None
    runner._resolve_session_service_tier = lambda **kw: None
    runner._resolve_turn_agent_config = lambda message, model, runtime: {"model": model}
    runner._sync_session_model_from_agent = lambda sid, _agent: None
    runner._sync_telegram_topic_binding = lambda source, entry, reason=None: None
    runner._delivery_adapter_for = lambda source: None
    runner._pop_post_delivery_callback = lambda *_args: None
    runner._session_db = db
    runner.session_store = _RoutingStore(session_key, session_id)

    turn = TurnRunner.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = ctx
    turn._combined_ephemeral_prompt = lambda: None
    turn._setup_stream_consumer = lambda platform_key: (None, None, None, False)

    def _resolve_turn_agent(route, platform_key, ephemeral, max_iterations, reasoning, pr):
        ctx.agent_holder[0] = agent  # interrupt support, as the real resolver wires it
        return agent, False

    turn._resolve_turn_agent = _resolve_turn_agent
    turn._wire_turn_agent_callbacks = lambda *a, **k: None
    turn._load_turn_history = lambda agent, reused: ([], None, [])
    turn._prepare_turn_message = lambda history: (None, None)
    turn._run_conversation_with_approval = (
        lambda agent, history, group, persist_msg, persist_ts: run_conversation())
    return runner, ctx, turn._run_sync_scoped()


@pytest.mark.asyncio
async def test_turn_runner_envelope_keeps_the_producer_turn_id_for_row_addressed_delivery(
        tmp_path, monkeypatch, _queued_ledger):
    """The runner's envelope keeps the turn producer's ``turn_id``
    (``export_current_turn_boundary``): the queued first-response lane addresses the
    assistant row with it, and replay reads the recorded delivery, never UNKNOWN."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "turn-id.db")
    try:
        sid = db.create_session("turn-id", "test")
        runner, ctx, envelope = _run_turn_envelope(
            monkeypatch, db, session_key="agent:main:slack:dm:chat", session_id=sid,
            turn_id="turn-1", question="what is the answer", answer="forty-two")

        # The exported pair travels together through the reconstruction.
        assert envelope["final_response"] == "forty-two"
        assert envelope["session_id"] == sid
        assert envelope["current_turn_user_idx"] == 0
        # turn_id pairs with current_turn_user_idx on the same exported boundary.
        assert envelope.get("turn_id") == "turn-1", \
            "the runner's result envelope must keep the turn producer's turn_id"

        adapter = _SendTransport()
        runner._delivery_adapter_for = lambda source: adapter
        adapter.gateway_runner = runner
        await runner._run_agent_deliver_first_response(ctx, adapter, envelope, envelope, None)

        assert [content for _chat, content in adapter.sent] == ["forty-two"], \
            "control: the queued lane did deliver the first response over the wire"
        row = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]
        delivery = (row.get("display_metadata") or {}).get("delivery")
        # The row-addressed record is the durable outcome of the queued delivery.
        assert delivery, "the delivered first response must leave a durable record on its row"
        assert delivery["turn_id"] == "turn-1"
        assert delivery["disposition"] == "text"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_queued_first_response_delivery_addresses_the_compression_child_row(
        tmp_path, monkeypatch, _queued_ledger):
    """A mid-run compression split moves the turn's rows into the CHILD session; the runner
    result reports that child while the turn context keeps the parent. The delivery address
    targets the result's effective session — the parent's verified resume lineage never
    contains the child's rows."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "split.db")
    try:
        parent = db.create_session("split-parent", "test")
        key = "agent:main:slack:dm:chat"
        runner, ctx, envelope = _run_turn_envelope(
            monkeypatch, db, session_key=key, session_id=parent, split_to="split-child",
            turn_id="turn-2", question="and after the split?", answer="the child holds the row")

        # The runner exports the post-run CHILD session and the routing entry is repointed,
        # while the turn context keeps the pre-split parent.
        assert envelope["session_id"] == "split-child"
        assert runner.session_store._entries[key].session_id == "split-child"
        assert ctx.session_id == parent, "the turn context itself keeps the pre-split parent"

        # _hmwa_turn_delivery_address's session preference (run_turn.py): the runner
        # result's session wins over the still-parent session entry.
        answer_row = db.get_messages_as_conversation("split-child", include_row_ids=True)[-1]["_row_id"]
        address = runner._hmwa_turn_delivery_address(
            {"session_id": "split-child", "turn_id": "turn-2",
             "messages": [{"role": "assistant", "content": "the child holds the row",
                           "_row_id": answer_row}]},
            session_entry=SimpleNamespace(session_id=parent), disposition_hint="text")
        assert address is not None, "control: every address leg is present"
        assert address["session_id"] == "split-child", \
            "the delivery address must prefer the runner result's effective session"

        adapter = _SendTransport()
        runner._delivery_adapter_for = lambda source: adapter
        adapter.gateway_runner = runner
        await runner._run_agent_deliver_first_response(ctx, adapter, envelope, envelope, None)
        row = db.get_messages_as_conversation("split-child", include_row_ids=True)[-1]
        delivery = (row.get("display_metadata") or {}).get("delivery")
        # The record lands on the row that holds it: the child's assistant row.
        assert delivery, "the split turn's delivery must be recorded on the child row"
        assert delivery["disposition"] == "text"
        assert delivery["turn_id"] == "turn-2"

        # Durable discrimination: the parent's verified resume lineage cannot address the
        # child's rows at all.
        probe_row = db.append_message("split-child", "assistant", "probe")
        assert db.record_message_delivery(
            parent, probe_row, "probe-turn", {"disposition": "text"}) is False
        assert db.record_message_delivery(
            "split-child", probe_row, "probe-turn", {"disposition": "text"}) is True
    finally:
        db.close()


@pytest.mark.asyncio
async def test_completed_handoff_voice_reply_marks_the_turn_as_speech_owned(
        tmp_path, monkeypatch, _queued_ledger):
    """A completed streamed handoff on a voice-input chat makes the runner's voice reply the
    turn's ONLY speech: it must claim the per-turn streaming-TTS marker (the same one the
    streaming path sets, run_turn.py's _run_agent_finalize_streaming_tts) so the base
    adapter's auto-TTS lane does not synthesize the same text a second time."""
    from gateway.delivery_handoff import delivery_handoff_from_receipt

    chat = "C-voice"
    key = "agent:main:slack:dm:C-voice"
    text = "the streamed answer, already on the wire"
    adapter = _SendTransport()
    adapter._auto_tts_enabled_chats.add(chat)  # /voice tts for this chat: base auto-TTS is armed
    runner = object.__new__(GatewayRunner)
    runner._voice_mode = {f"slack:{chat}": "all"}
    runner._adapter_profile_for_source = lambda source: None
    runner._delivery_adapter_for = lambda source: adapter
    voice_synths = []

    async def _one_voice_reply(event, reply_text):
        voice_synths.append(reply_text)
        return [SendResult(success=True, message_id="voice-1")]

    runner._send_voice_reply = _one_voice_reply
    agent_result = {
        "already_sent": True,
        "_stream_delivery_handoff": delivery_handoff_from_receipt(
            text, SendResult(success=True, message_id="stream-final")),
    }
    event = MessageEvent(
        text="question", message_type=MessageType.VOICE,
        source=SessionSource(platform=Platform.SLACK, chat_id=chat, user_id="u", chat_type="dm"),
        message_id="in-voice-1",
    )
    assert adapter._streaming_tts_turn_completed(key, 1) is False, \
        "control: streaming TTS delivered nothing this turn (whole-file lanes stay armed)"

    delivered = await runner._hmwa_deliver_turn_response(
        event, event.source, SimpleNamespace(session_id="sid-voice"), key, 1,
        agent_result, [], text, None, False)

    assert delivered == text, "control: the accounting text is retained (nonempty return)"
    assert voice_synths == [text], "control: the voice lane synthesized this turn's speech once"
    # The marker is what suppresses base's whole-file auto-TTS for this turn.
    assert adapter._streaming_tts_turn_completed(key, 1) is True, \
        "the voice reply must claim the turn's speech-ownership marker"
    interrupt_event = asyncio.Event()
    interrupt_event._hermes_run_generation = 1  # what _bind_adapter_run_generation stamps
    assert adapter._wants_auto_tts(event, key, interrupt_event, text, []) is False, \
        "base auto-TTS must not synthesize a second copy of the delivered speech"

    # The connected-voice-channel branch returns NO receipts (the audio played in VC,
    # run_voice.py's _deliver_voice_reply): ownership is claimed at the gate, never
    # conditional on receipts — or exactly the voice-input case re-synthesizes.
    voice_synths.clear()

    async def _vc_voice_reply(current, reply_text):
        voice_synths.append(reply_text)
        return []  # played in the connected voice channel: no send receipts

    runner._send_voice_reply = _vc_voice_reply
    delivered = await runner._hmwa_deliver_turn_response(
        event, event.source, SimpleNamespace(session_id="sid-voice"), key, 2,
        agent_result, [], text, None, False)
    assert delivered == text, "control: receipt-less playback still returns the accounting text"
    assert voice_synths == [text], "control: the voice lane still synthesized once"
    assert adapter._streaming_tts_turn_completed(key, 2) is True, \
        "audio played in the voice channel leaves no receipts; the marker must not depend on them"


@pytest.mark.asyncio
async def test_composed_voice_turn_synthesizes_speech_exactly_once(
        tmp_path, monkeypatch, _queued_ledger):
    """End to end through the adapter's delivery bracket: a voice-input turn whose stream
    completed (auto-TTS on, streaming TTS off) must produce exactly ONE speech synthesis —
    the runner's voice reply — and keep the handoff text as the owed-body identity."""
    from gateway.delivery_handoff import delivery_handoff_from_receipt
    from hermes_state import SessionDB

    chat = "C-voice"
    key = "agent:main:slack:dm:C-voice"
    text = "one speech, delivered once"
    db = SessionDB(db_path=tmp_path / "voice-turn.db")
    try:
        sid = db.create_session("voice-turn", "test")
        db.append_message(sid, "user", "question")
        answer_row = db.append_message(sid, "assistant", text)
        adapter = _SendTransport()
        adapter._auto_tts_enabled_chats.add(chat)
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._voice_mode = {f"slack:{chat}": "all"}
        runner._adapter_profile_for_source = lambda source: None
        runner._delivery_adapter_for = lambda source: adapter
        runner._pop_post_delivery_callback = lambda *_args: None
        voice_synths = []

        async def _one_voice_reply(event, reply_text):
            voice_synths.append(reply_text)
            return [SendResult(success=True, message_id="voice-1")]

        runner._send_voice_reply = _one_voice_reply
        adapter.gateway_runner = runner
        base_synths = []

        async def _count_base_synth(speech_text):
            base_synths.append(speech_text)
            return [], None

        adapter._synthesize_auto_tts = _count_base_synth
        agent_result = {
            "final_response": text, "turn_id": "voice-turn-1", "already_sent": True,
            "messages": [{"role": "assistant", "content": text, "_row_id": answer_row}],
            "_stream_delivery_handoff": delivery_handoff_from_receipt(
                text, SendResult(success=True, message_id="stream-final")),
        }
        event = MessageEvent(
            text="question", message_type=MessageType.VOICE,
            source=SessionSource(platform=Platform.SLACK, chat_id=chat, user_id="u", chat_type="dm"),
            message_id="in-voice-1",
        )
        delivered = {}

        async def completion(current):
            delivered["value"] = await runner._hmwa_deliver_turn_response(
                current, current.source, SimpleNamespace(session_id=sid), key, 1,
                agent_result, [], text, None, False)
            return delivered["value"]

        adapter._message_handler = completion
        interrupt_event = asyncio.Event()
        interrupt_event._hermes_run_generation = 1  # what _bind_adapter_run_generation stamps
        adapter._active_sessions[key] = interrupt_event
        await adapter._process_message_background(event, key)

        assert delivered["value"] == text, "the completion keeps the handoff text (nonempty return)"
        assert voice_synths == [text], "exactly one speech synthesis: the runner's voice reply"
        # Base's whole-file auto-TTS reads the same marker (base.py:_wants_auto_tts).
        assert base_synths == [], "the base adapter must not synthesize a second copy"
        assert getattr(event, "_streamed_final_response", None) == text, \
            "the completed stream's model text stays the obligation identity"
    finally:
        db.close()
