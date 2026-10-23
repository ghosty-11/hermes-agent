"""A connector's declined destination is terminal for the turn, including recovery."""

from __future__ import annotations

import asyncio
import os
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.delivery_handoff import delivery_handoff_from_receipt
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.egress import EGRESS_DECLINE_CODE
from gateway.session import SessionSource, SessionStore


CHAT = "C1"
SESSION_KEY = "agent:main:slack:channel:C1"
TEXT = "Preview and final answer"
DECLINE = SendResult(success=False, error="declined", raw_response={
    "success": False, "code": EGRESS_DECLINE_CODE,
})


@pytest.fixture(autouse=True)
def _ledger_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Media root checks read import-time roots; the test's media lives under tmp_path.
    from gateway.platforms import base as platform_base
    monkeypatch.setattr(platform_base, "_HERMES_ROOT", home)
    monkeypatch.setattr(platform_base, "_media_delivery_allowed_roots", lambda: [tmp_path])
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)


class _Transport(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.calls = []
        self.edit_receipt = DECLINE
        self.send_receipt = SendResult(success=True, message_id="delivered")

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("send", chat_id))
        return self.send_receipt

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.calls.append(("edit", chat_id))
        return self.edit_receipt

    async def send_multiple_images(self, chat_id, images, **kwargs):
        self.calls.append(("image", chat_id))
        return SendResult(success=True, message_id="image")

    async def send_document(self, chat_id, file_path, **kwargs):
        self.calls.append(("document", chat_id))
        return SendResult(success=True, message_id="document")

    async def send_voice(self, chat_id, audio_path, **kwargs):
        self.calls.append(("audio", chat_id))
        return SendResult(success=True, message_id="audio")

    async def send_video(self, chat_id, video_path, **kwargs):
        self.calls.append(("video", chat_id))
        return SendResult(success=True, message_id="video")


def _event(text="question"):
    return MessageEvent(
        text=text, message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id=CHAT, chat_type="channel"),
        message_id="inbound-1",
    )


def _ledger_rows():
    with dl._connect() as conn:
        return conn.execute("SELECT state, content FROM delivery_obligations").fetchall()


@pytest.mark.asyncio
async def test_declined_destination_queued_edit_cannot_reenter_completion(tmp_path):
    from gateway.run import GatewayRunner
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from hermes_state import SessionDB

    adapter = _Transport()
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    preview = "Preview"
    await consumer._first_send(preview, finalize=False)
    assert adapter.calls == [("send", CHAT)]
    db = SessionDB(db_path=tmp_path / "session.db")
    try:
        sid = db.create_session("declined-queued", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", TEXT)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        result = {"final_response": TEXT, "turn_id": "first-turn", "messages": [
            {"role": "assistant", "content": TEXT, "_row_id": row_id}]}
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._delivery_adapter_for = lambda source: adapter
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._should_send_voice_reply = lambda *_args, **_kwargs: False
        adapter.gateway_runner = runner
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key=SESSION_KEY, session_id=sid,
            source=_event().source, stream_consumer_holder=[consumer],
            _status_thread_metadata={}, event_message_id="inbound-1",
            inbound_message_id="inbound-1", run_generation=1,
        )
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        event = _event()

        async def completion(current):
            return await runner._hmwa_deliver_turn_response(
                current, current.source, SimpleNamespace(session_id=sid),
                SESSION_KEY, 1, result, [], TEXT, None, False)

        adapter._message_handler = completion
        adapter._active_sessions[SESSION_KEY] = asyncio.Event()
        await adapter._process_message_background(event, SESSION_KEY)

        assert [call for call in adapter.calls if call[1] == CHAT] == [
            ("send", CHAT), ("edit", CHAT)]
        rows = _ledger_rows()
        assert rows and all(state not in ("pending", "attempting", "failed") for state, _ in rows)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_declined_destination_ledger_never_replays_at_runtime_or_startup(tmp_path):
    from gateway.run import GatewayRunner
    from hermes_state import SessionDB

    adapter = _Transport()
    adapter.send_receipt = DECLINE
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    store._db = SessionDB(db_path=dl._db_path())
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    adapter.gateway_runner = runner
    runner._schedule_flood_redelivery = lambda *_args, **_kwargs: None
    event = _event()
    entry = store.get_or_create_session(event.source)
    # Leave the active marker behind, as after a crash before marker cleanup.
    store.mark_turn_active(entry.session_key)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "question"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": TEXT})
    try:
        receipt, _ = await adapter.send_final_ledgered(event, entry.session_key, TEXT, {}, reply_to=None)
        assert not receipt.success
        adapter.send_receipt = SendResult(success=True, message_id="replay")
        with dl._connect() as conn:
            failed_at = conn.execute(
                "SELECT updated_at FROM delivery_obligations LIMIT 1").fetchone()[0]
        claimed = dl.sweep_failed_for_runtime("slack", now=failed_at + 60)
        await runner._redeliver_claimed_obligations(claimed)
        await runner._recover_unclean_sessions()
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
                "WHERE state IN ('pending', 'attempting', 'failed')")
        await runner._redeliver_pending_obligations()

        assert [call for call in adapter.calls if call[1] == CHAT] == [("send", CHAT)]
        rows = _ledger_rows()
        assert rows and all(state not in ("pending", "attempting", "failed") for state, _ in rows)
    finally:
        store._db.close()


@pytest.mark.asyncio
async def test_declined_destination_text_only_edit_has_no_fallback_send():
    from gateway.run_notifications import GatewayNotificationsMixin

    adapter = _Transport()
    adapter.edit_receipt = SendResult(
        success=False,
        error="discord egress declined: target is not an approved destination for this connection",
    )
    await GatewayNotificationsMixin._deliver_queued_first_response(
        object.__new__(GatewayNotificationsMixin), TEXT, _event().source, adapter,
        stream_consumer=SimpleNamespace(message_id="preview-id", _turn_split_delivery=False),
        deliver_media=False,
    )

    assert [call for call in adapter.calls if call[1] == CHAT] == [("edit", CHAT)]


@pytest.mark.asyncio
async def test_declined_destination_final_text_stops_media_and_voice(tmp_path):
    from gateway.run import GatewayRunner

    adapter = _Transport()
    adapter.send_receipt = DECLINE
    media = [tmp_path / name for name in ("report.pdf", "speech.mp3", "clip.mp4")]
    for path in media:
        path.write_bytes(b"media")
    response = "Answer ![image](https://example.test/picture.png)\n" + "\n".join(
        f"MEDIA:{path}" for path in media)
    adapter._message_handler = AsyncMock(return_value=response)
    event = _event()
    extracted = await adapter._extract_response_content(
        response, event, SESSION_KEY, is_ephemeral_response=False)
    assert extracted.images
    assert {path.rsplit(".", 1)[-1] for path, _ in extracted.media_files} == {
        "pdf", "mp3", "mp4"}
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    await adapter._process_message_background(event, SESSION_KEY)

    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._should_send_voice_reply = lambda *_args, **_kwargs: True
    async def voice_reply(current, response):
        await adapter.send_voice(current.source.chat_id, "generated.ogg")

    runner._send_voice_reply = AsyncMock(side_effect=voice_reply)
    incoming = {"final_response": TEXT, "_stream_delivery_handoff": delivery_handoff_from_receipt(TEXT, DECLINE)}
    await runner._hmwa_deliver_turn_response(
        _event(), _event().source, SimpleNamespace(session_id="sid"),
        SESSION_KEY, 1, incoming, [], TEXT, None, False)

    assert [call for call in adapter.calls if call[1] == CHAT] == [("send", CHAT)]
    runner._send_voice_reply.assert_not_awaited()
    rows = _ledger_rows()
    assert rows and all(state not in ("pending", "attempting", "failed") for state, _ in rows)


@pytest.mark.asyncio
async def test_ownerless_recovery_decline_is_terminal_across_sweeps():
    from gateway.run import GatewayRunner

    adapter = _Transport()
    adapter.send_receipt = DECLINE
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    runner._schedule_flood_redelivery = lambda *_args, **_kwargs: None
    obligation_id = dl.compute_obligation_id(SESSION_KEY, "ownerless", TEXT)
    dl.record_obligation(
        obligation_id=obligation_id, session_key=SESSION_KEY, platform="slack",
        chat_id=CHAT, thread_id=None, content=TEXT)
    dl.mark_failed(obligation_id, "send_path_degraded")
    with dl._connect() as conn:
        failed_at = conn.execute(
            "SELECT updated_at FROM delivery_obligations WHERE obligation_id=?", (obligation_id,)
        ).fetchone()[0]
    claimed = dl.sweep_failed_for_runtime("slack", now=failed_at + 60)
    assert len(claimed) == 1 and claimed[0]["turn_receipt"] is None

    assert await runner._redeliver_claimed_obligations(claimed) == 0
    with dl._connect() as conn:
        state, receipt = conn.execute(
            "SELECT state, turn_receipt FROM delivery_obligations WHERE obligation_id=?",
            (obligation_id,),
        ).fetchone()
    assert state == "declined"
    assert receipt is not None
    assert not dl.sweep_failed_for_runtime("slack", now=failed_at + 3600)
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
            "WHERE obligation_id=?", (obligation_id,))
    assert not dl.sweep_recoverable(now=failed_at + 3600)
    assert adapter.calls == [("send", CHAT)]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["pending", "attempting"])
async def test_crash_left_declined_owner_suppresses_different_reply(tmp_path, state):
    from gateway.run import GatewayRunner
    from hermes_state import SessionDB

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    store._db = SessionDB(db_path=dl._db_path())
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    entry = store.get_or_create_session(_event().source)
    store.mark_turn_active(entry.session_key)
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "question"})
    different_reply = "Persisted answer after refusal"
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": different_reply})
    owner = delivery_handoff_from_receipt(TEXT, DECLINE)
    obligation_id = dl.compute_obligation_id(entry.session_key, "refused", TEXT)
    dl.record_obligation(
        obligation_id=obligation_id, session_key=entry.session_key, platform="slack",
        chat_id=CHAT, thread_id=None, content=owner.owed,
        original_content=TEXT, turn_receipt=owner.to_record())
    if state == "attempting":
        dl.mark_attempting(obligation_id)
    try:
        assert await runner._ledger_crash_left_replies(3600) == 0
        with dl._connect() as conn:
            rows = conn.execute(
                "SELECT obligation_id, state FROM delivery_obligations WHERE session_key=?",
                (entry.session_key,),
            ).fetchall()
        assert len(rows) == 1 and rows[0][0] == obligation_id
        assert store.get_or_create_session(_event().source).active_turn_token is None
    finally:
        store._db.close()


def test_successful_receipt_with_decline_error_is_not_a_refusal():
    receipt = SendResult(
        success=True, message_id="accepted",
        error="slack egress declined: target is not an approved destination for this connection",
    )
    owner = delivery_handoff_from_receipt(TEXT, receipt, attempted_content=TEXT)
    assert owner.complete
    assert owner.to_record()["egress_declined"] is False


@pytest.mark.parametrize("ambiguity_key", ["delivery_ambiguous", "ambiguous"])
def test_ambiguous_decline_marker_is_not_terminal(ambiguity_key):
    receipt = SendResult(
        success=False, error="declined",
        raw_response={"success": False, "code": EGRESS_DECLINE_CODE,
                      ambiguity_key: True},
    )
    owner = delivery_handoff_from_receipt(TEXT, receipt)
    assert owner.ambiguous
    assert owner.to_record()["egress_declined"] is False


def test_quoted_decline_marker_inside_failure_is_not_a_refusal():
    receipt = SendResult(
        success=False,
        error="upstream reported 'slack egress declined: target is not an approved destination for this connection'",
    )
    owner = delivery_handoff_from_receipt(TEXT, receipt)
    assert owner.to_record()["egress_declined"] is False


@pytest.mark.parametrize("receipt", [
    SendResult(
        success=False,
        error="slack egress declined: target is not an approved destination for this connection",
    ),
    DECLINE,
])
def test_exact_uniform_and_structured_declines_remain_terminal(receipt):
    owner = delivery_handoff_from_receipt(TEXT, receipt)
    assert owner.to_record()["egress_declined"] is True


@pytest.mark.asyncio
async def test_narrowed_ownerless_recovery_decline_stays_terminal_across_runtimes():
    from gateway.run import GatewayRunner

    adapter = _Transport()
    adapter.send_receipt = DECLINE
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    runner._schedule_flood_redelivery = lambda *_args, **_kwargs: None
    original, remaining = "head\ntail", "tail"
    obligation_id = dl.compute_obligation_id(SESSION_KEY, "partial", original)
    dl.record_obligation(
        obligation_id=obligation_id, session_key=SESSION_KEY, platform="slack",
        chat_id=CHAT, thread_id=None, content=original)
    dl.mark_failed(obligation_id, "send_path_degraded", remaining_content=remaining)
    with dl._connect() as conn:
        failed_at = conn.execute(
            "SELECT updated_at FROM delivery_obligations WHERE obligation_id=?", (obligation_id,)
        ).fetchone()[0]
    claimed = dl.sweep_failed_for_runtime("slack", now=failed_at + 60)
    assert len(claimed) == 1
    assert claimed[0]["content"] == remaining
    assert claimed[0]["original_content"] == original

    assert await runner._redeliver_claimed_obligations(claimed) == 0
    with dl._connect() as conn:
        state, content, identity, receipt = conn.execute(
            "SELECT state, content, original_content, turn_receipt FROM delivery_obligations "
            "WHERE obligation_id=?", (obligation_id,)
        ).fetchone()
    assert (state, content, identity) == ("declined", remaining, original)
    assert json.loads(receipt)["source_content"] == original
    assert not dl.sweep_failed_for_runtime("slack", now=failed_at + 3600)
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
            "WHERE obligation_id=?", (obligation_id,))
    assert not dl.sweep_recoverable(now=failed_at + 3600)
    assert adapter.calls == [("send", CHAT)]


def test_crash_left_adoption_excludes_decline_under_immediate_write_lock(monkeypatch):
    since = time.time() - 60
    declined = delivery_handoff_from_receipt(TEXT, DECLINE).to_record()
    with dl._connect() as conn:
        conn.execute("SELECT 1 FROM delivery_obligations")
    other = sqlite3.connect(dl._db_path(), timeout=0)

    def insert_decline(session_key):
        other.execute(
            """INSERT INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, content, original_content,
                state, created_at, updated_at, turn_receipt)
               VALUES (?, ?, 'slack', ?, ?, ?, 'declined', ?, ?, ?)""",
            (f"decline:{session_key}", session_key, CHAT, TEXT, TEXT,
             time.time(), time.time(), json.dumps(declined)))

    try:
        committed_key = SESSION_KEY + ":committed"
        with other:
            insert_decline(committed_key)
        dl.record_crash_left_reply(
            obligation_id="crash:committed", session_key=committed_key, platform="slack",
            chat_id=CHAT, thread_id=None, content="different reply", since=since)
        with dl._connect() as conn:
            assert conn.execute(
                "SELECT obligation_id FROM delivery_obligations WHERE session_key=?",
                (committed_key,),
            ).fetchall() == [(f"decline:{committed_key}",)]

        contested_key = SESSION_KEY + ":contested"
        check = dl._declined_since

        def probe(conn, session_key, cutoff):
            found = check(conn, session_key, cutoff)
            assert not found
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                insert_decline(session_key)
            other.rollback()
            return found

        monkeypatch.setattr(dl, "_declined_since", probe)
        dl.record_crash_left_reply(
            obligation_id="crash:contested", session_key=contested_key, platform="slack",
            chat_id=CHAT, thread_id=None, content="different reply", since=since)
        monkeypatch.setattr(dl, "_declined_since", check)
        with dl._connect() as conn:
            assert conn.execute(
                "SELECT obligation_id FROM delivery_obligations WHERE session_key=?",
                (contested_key,),
            ).fetchall() == [("crash:contested",)]
    finally:
        other.close()
