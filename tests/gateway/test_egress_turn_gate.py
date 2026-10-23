"""A connector refusal closes every chat-addressed lane for its owning turn."""

import asyncio
import os
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.egress import EGRESS_DECLINE_CODE, declined_send
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_refusals import begin_turn, refused


CHAT = "C1"
DECLINE = SendResult(success=False, error="declined", raw_response={
    "success": False, "code": EGRESS_DECLINE_CODE,
})


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway.platforms import base
    monkeypatch.setattr(base, "_HERMES_ROOT", home)
    monkeypatch.setattr(base, "_media_delivery_allowed_roots", lambda: [tmp_path])
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=False), Platform.SLACK)
        self.calls = []
        self.results = {}

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    def receipt(self, name, chat_id):
        self.calls.append((name, chat_id))
        return self.results.get(name, SendResult(success=True, message_id="preview"))

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return self.receipt("send", chat_id)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        return self.receipt("edit", chat_id)

    async def send_document(self, chat_id, file_path, **kwargs):
        return self.receipt("document", chat_id)

    async def send_native_task_card_progress(self, chat_id, tasks, **kwargs):
        return self.receipt("card", chat_id)

    async def stop_native_task_card_progress(self, chat_id, **kwargs):
        return self.receipt("stop", chat_id)


def event():
    return MessageEvent(text="question", message_type=MessageType.TEXT,
                        source=SessionSource(platform=Platform.SLACK, chat_id=CHAT, chat_type="channel"),
                        message_id="inbound-1")


def progress(adapter):
    ctx = SimpleNamespace(source=event().source, _progress_reply_to=None,
                          _progress_metadata={}, _cleanup_progress=False,
                          stream_consumer_holder=[None], progress_queue=queue.Queue(),
                          session_key="agent:main:slack:channel:C1", _run_still_current=lambda: True,
                          progress_grouping="accumulate")
    return TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx), ctx


@pytest.mark.asyncio
async def test_gate_chat_identity_and_turn_lifetime():
    adapter = Adapter()
    adapter.results["edit"] = DECLINE
    with begin_turn():
        assert await adapter.edit_message(CHAT, "preview", "update") is DECLINE
        assert refused(Platform.SLACK, CHAT) is DECLINE
        blocked = await adapter.send(CHAT, "secret")
        assert declined_send(blocked)
        assert blocked.raw_response["code"] == EGRESS_DECLINE_CODE
        assert adapter.calls == [("edit", CHAT)]
        assert (await adapter.send("C2", "safe")).success
        async def inherited():
            return await adapter.send(CHAT, "from task")
        assert declined_send(await asyncio.create_task(inherited()))
        assert adapter.calls == [("edit", CHAT), ("send", "C2")]
        with begin_turn():
            assert refused(Platform.SLACK, CHAT) is None
            assert (await adapter.send(CHAT, "nested turn")).success
        assert refused(Platform.SLACK, CHAT) is DECLINE
    assert refused(Platform.SLACK, CHAT) is None
    assert (await adapter.send(CHAT, "outside turn")).success
    with begin_turn():
        assert refused(Platform.SLACK, CHAT) is None
        assert (await adapter.send(CHAT, "next turn")).success


@pytest.mark.asyncio
async def test_relay_card_frame_and_cosmetic_decline_share_logical_chat():
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    class Connector:
        def __init__(self):
            self.frames = []

        async def send_outbound(self, action, *, platform=None):
            self.frames.append((action["op"], action["chat_id"], platform))
            return {"success": False, "code": EGRESS_DECLINE_CODE}

    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="slack", label="Relay",
        max_message_length=4096, supports_draft_streaming=False,
        supports_edit=True, supports_threads=True, markdown_dialect="slack",
        len_unit="chars", supported_ops=("send", "task_card", "task_card_stop"))
    connector = Connector()
    adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport=connector)
    with begin_turn():
        first = await adapter.send_native_task_card_progress(
            CHAT, [{"id": "one", "title": "work", "status": "in_progress"}],
            reply_to="parent")
        assert declined_send(first)
        assert refused(Platform.SLACK, CHAT) is not None
        assert declined_send(await adapter.stop_native_task_card_progress(CHAT, reply_to="parent"))
        assert declined_send(await adapter.send(CHAT, "secret"))
        await adapter.send_typing(CHAT)
        assert connector.frames == [("task_card", CHAT, None)]


@pytest.mark.asyncio
async def test_heartbeat_declined_edit_stops_fallback_and_future_ticks(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway import run_turn
    adapter = Adapter()
    adapter.results["edit"] = DECLINE
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._should_emit_long_running_notification = lambda *args: True
    ctx = SimpleNamespace(source=event().source, session_key="session", agent_holder=[None],
                          _status_thread_metadata={}, _cleanup_progress=False)
    disp = SimpleNamespace(_display_surface_mode=lambda *args, **kwargs: "detailed",
                           resolve_display_setting=lambda *args: False,
                           user_config={}, platform_key="slack",
                           _generic_status_phrase=lambda *args: "working")
    runner._agent_activity_summary = lambda agent: None
    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "1")
    original_sleep = asyncio.sleep
    ticks = 0
    async def tick(delay):
        nonlocal ticks
        ticks += 1
        if ticks > 3:
            raise asyncio.CancelledError
        await original_sleep(0)
    monkeypatch.setattr(run_turn.asyncio, "sleep", tick)
    with begin_turn():
        await runner._run_agent_notify_long_running(disp, ctx, [None])
    assert adapter.calls == [("send", CHAT), ("edit", CHAT)]
    assert ticks == 2


@pytest.mark.asyncio
async def test_card_decline_blocks_cleanup_and_stop_decline_blocks_final():
    adapter = Adapter()
    runner, ctx = progress(adapter)
    ctx.progress_queue.put({"type": "tool.started", "tool_call_id": "one", "tool_name": "terminal"})
    adapter.results["card"] = DECLINE
    with begin_turn():
        await runner._send_native_task_card_progress(adapter)
    assert adapter.calls == [("card", CHAT)]

    ctx._progress_decline_receipt = None
    adapter.calls.clear()
    adapter.results.pop("card")
    adapter.results["stop"] = DECLINE
    ctx.progress_queue.put({"type": "tool.started", "tool_call_id": "two", "tool_name": "browser"})
    async def finish_card():
        task = asyncio.create_task(runner._send_native_task_card_progress(adapter))
        while ("card", CHAT) not in adapter.calls:
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    with begin_turn():
        await asyncio.wait_for(finish_card(), timeout=2)
        assert refused(Platform.SLACK, CHAT) is DECLINE
        assert declined_send(await adapter.send(CHAT, "final"))
    assert adapter.calls == [("card", CHAT), ("stop", CHAT)]


@pytest.mark.asyncio
async def test_media_failure_notice_decline_blocks_next_attachment(tmp_path):
    adapter = Adapter()
    adapter.results["document"] = SendResult(success=False, error="upload failed")
    adapter.results["send"] = DECLINE
    adapter.warning_notifications_enabled = lambda *args, **kwargs: True
    paths = [tmp_path / name for name in ("one.pdf", "two.pdf")]
    for path in paths:
        path.write_bytes(b"content")
    recorded = []
    with begin_turn():
        await adapter._deliver_media_attachments(event(), [(str(path), False) for path in paths], [],
            force_document_attachments=False, human_delay=0, metadata={},
            record_delivery=recorded.append,
            delivery_declined=lambda: refused(Platform.SLACK, CHAT) is not None)
    assert adapter.calls == [("document", CHAT), ("send", CHAT)]
    assert any(declined_send(receipt) for receipt in recorded)


@pytest.mark.asyncio
async def test_progress_and_stream_share_decline_in_both_directions():
    adapter = Adapter()
    runner, ctx = progress(adapter)
    state = runner._progress_edit_state(adapter)
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    with begin_turn():
        adapter.results["edit"] = DECLINE
        state.progress_msg_id = "preview"
        state.progress_lines = ["tool"]
        await runner._progress_send_or_edit(state, "tool")
        await consumer._first_send("stream", finalize=False)
        assert refused(Platform.SLACK, CHAT) is not None
        assert adapter.calls == [("edit", CHAT)]
    adapter.calls.clear()
    adapter.results.pop("edit")
    ctx._progress_decline_receipt = None
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    adapter.results["send"] = DECLINE
    with begin_turn():
        await consumer._first_send("stream", finalize=False)
        await runner._progress_send_or_edit(state, "later")
        assert adapter.calls == [("send", CHAT)]


@pytest.mark.asyncio
async def test_declined_owner_settles_ledger_without_transport_and_cannot_be_adopted(tmp_path):
    from gateway.run import GatewayRunner
    from gateway.session import SessionStore
    from gateway.config import GatewayConfig
    from hermes_state import SessionDB
    from gateway.delivery_handoff import delivery_handoff_from_receipt
    adapter = Adapter()
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    if store._db is not None:
        store._db.close()
    store._db = SessionDB(db_path=dl._db_path())
    gateway = object.__new__(GatewayRunner)
    gateway.session_store = store
    gateway.adapters = {Platform.SLACK: adapter}
    gateway._profile_adapters = {}
    gateway._active_profile_name = lambda: "default"
    gateway._schedule_flood_redelivery = lambda *args, **kwargs: None
    gateway._should_send_voice_reply = lambda *args, **kwargs: False
    gateway._delivery_adapter_for = lambda source: adapter
    adapter.gateway_runner = gateway
    current = event()
    entry = store.get_or_create_session(current.source)
    token = store.mark_turn_active(entry.session_key)
    started_at = entry.active_turn_started_at
    store.append_to_transcript(entry.session_id, {"role": "user", "content": "question"})
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "answer"})
    async def handler(ev):
        ev._stream_delivery_handoff = delivery_handoff_from_receipt("answer", DECLINE)
        return "answer"
    adapter.set_message_handler(handler)
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    try:
        await adapter._process_message_background(current, entry.session_key)
        with dl._connect() as conn:
            states = [row[0] for row in conn.execute("SELECT state FROM delivery_obligations")]
        assert states == ["declined"]
        assert adapter.calls == []
        # Restore the original marker timestamp, as if the process died before clearing it.
        with store._lock:
            store._set_turn_marker_locked(entry.session_key, entry, token, started_at)
        await gateway._recover_unclean_sessions()
        with dl._connect() as conn:
            assert [row[0] for row in conn.execute("SELECT state FROM delivery_obligations")] == ["declined"]
        assert adapter.calls == []
    finally:
        store._db.close()


@pytest.mark.asyncio
async def test_mixin_inherited_media_method_is_gated():
    class MediaMixin:
        async def send_document(self, chat_id, file_path, **kwargs):
            return self.receipt("mixin_document", chat_id)

    class MixedAdapter(MediaMixin, Adapter):
        pass

    adapter = MixedAdapter()
    adapter.results["send"] = DECLINE
    with begin_turn():
        assert await adapter.send(CHAT, "first") is DECLINE
        assert declined_send(await adapter.send_document(CHAT, "/tmp/document.txt"))
        assert adapter.calls == [("send", CHAT)]


@pytest.mark.asyncio
async def test_native_stream_frame_is_gated_without_sending():
    class StreamingMixin:
        async def send_stream_frame(self, text, *, finalize=False, chat_id=None, **kwargs):
            self.calls.append(("frame", chat_id))
            return True

    class StreamingAdapter(StreamingMixin, Adapter):
        pass

    adapter = StreamingAdapter()
    adapter.results["send"] = DECLINE
    with begin_turn():
        assert await adapter.send(CHAT, "first") is DECLINE
        assert await adapter.send_stream_frame("secret", finalize=True, chat_id=CHAT) is False
        assert adapter.calls == [("send", CHAT)]


@pytest.mark.asyncio
async def test_relay_seal_retry_stops_when_concurrent_publisher_refuses():
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
    from gateway.turn_refusals import record

    class Connector:
        def __init__(self):
            self.frames = []

        async def send_outbound(self, action, *, platform=None):
            self.frames.append(action)
            if action["op"] == "draft" and not action["final"]:
                return {"success": True}
            if len(self.frames) == 2:
                record(Platform.SLACK, CHAT, DECLINE)
                return {"success": False, "ambiguous": True}
            return {"success": True, "message_id": "unexpected"}

    connector = Connector()
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="slack", label="Relay",
        max_message_length=4096, supports_draft_streaming=True,
        supports_edit=True, supports_threads=True, markdown_dialect="slack",
        len_unit="chars", supported_ops=("send", "draft"))
    adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport=connector)
    with begin_turn():
        assert (await adapter.send_draft(CHAT, 7, "partial")).success
        receipt = await adapter.send(CHAT, "final")
        assert declined_send(receipt)
        assert [(frame["op"], frame["final"]) for frame in connector.frames] == [
            ("draft", False), ("draft", True)]


@pytest.mark.asyncio
async def test_detached_publisher_outlives_turn_refusal():
    from gateway.turn_refusals import record
    adapter = Adapter()
    adapter.results["send"] = DECLINE
    publish = asyncio.Event()

    async def detached():
        await publish.wait()
        record(Platform.SLACK, CHAT, DECLINE)
        assert refused(Platform.SLACK, CHAT) is None
        return await adapter.send(CHAT, "after turn")

    with begin_turn():
        assert await adapter.send(CHAT, "first") is DECLINE
        assert declined_send(await asyncio.create_task(adapter.send(CHAT, "during turn")))
        task = asyncio.create_task(detached())
        assert adapter.calls == [("send", CHAT)]
    publish.set()
    adapter.results.pop("send")
    assert (await task).success
    assert adapter.calls == [("send", CHAT), ("send", CHAT)]
