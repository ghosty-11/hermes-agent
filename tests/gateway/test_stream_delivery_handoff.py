"""A known stream remainder crosses into the ordinary final delivery bracket once."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_state import AsyncSessionDB, SessionDB
from gateway.delivery_handoff import StreamDeliveryHandoff, delivery_handoff_from_receipt


class _Adapter(BasePlatformAdapter):
    def __init__(self, receipts):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.receipts = iter(receipts)
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return next(self.receipts)


@pytest.fixture(autouse=True)
def _ledger_home(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")


def _event():
    return MessageEvent(text="question", message_type=MessageType.TEXT,
                        source=SessionSource(platform=Platform.SLACK, chat_id="C1", chat_type="channel"),
                        message_id="inbound-1")


async def _finish(adapter, text="Hello", *, fallback=False):
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(buffer_only=fallback))
    if fallback:
        consumer._edit_supported = False
        consumer._fallback_final_send = True
        consumer._message_id = "__no_edit__"
    consumer.on_delta(text)
    consumer.finish(text)
    await consumer.run()
    return consumer

def test_record_round_trip_preserves_address_and_editable_carriage_order():
    original = delivery_handoff_from_receipt(
        "Old prose", SendResult(success=False, message_id="prose", delivered_text="Old prose",
                                 error="media refused", raw_response={
                                     "partial_overflow": True, "message_ids": ["prose"],
                                     "remaining_content": "https://example.test/a.gif",
                                     "model_remaining_content": "",
                                     "attachment_remaining_content": "https://example.test/a.gif"}))
    original = delivery_handoff_from_receipt(
        "Old prose", SendResult(success=True, message_id="gif",
                                delivered_text="https://example.test/a.gif"),
        previous=original, attempted_content="https://example.test/a.gif")
    address = {"session_id": "sid", "row_id": 3, "turn_id": "turn",
               "disposition_hint": "text", "streamed_text": "not transport metadata"}
    restored = StreamDeliveryHandoff.from_record(original.with_address(address).to_record())
    assert restored.address == {key: address[key] for key in
                                ("session_id", "row_id", "turn_id", "disposition_hint")}
    edited = delivery_handoff_from_receipt(
        "Replacement prose", SendResult(success=True, message_id="prose",
                                         delivered_text="Replacement prose"),
        previous=restored, edited_message_id="prose", attempted_content="Replacement prose")
    assert edited.to_record()["acknowledged_carriages"] == [
        {"message_id": "prose", "text": "Replacement prose"},
        {"message_id": "gif", "text": "https://example.test/a.gif"}]
    assert edited.address == restored.address
    assert edited.without_deleted(["prose"]).to_record()["acknowledged_carriages"] == [
        {"message_id": "gif", "text": "https://example.test/a.gif"}]


@pytest.mark.asyncio
async def test_stream_failure_hands_attachment_and_original_identity_to_final_bracket():
    url = "https://example.test/attached.gif"
    adapter = _Adapter([
        SendResult(success=False, error="attachment unavailable", message_id="prose-1",
                   delivered_text="Hello", raw_response={"remaining_content": url,
                   "message_ids": ["prose-1"], "partial_overflow": True}),
        SendResult(success=True, message_id="gif-2", delivered_text=url),
    ])
    consumer = await _finish(adapter, fallback=True)
    handoff = consumer.delivery_handoff_for_final("Hello")
    assert handoff.result.raw_response["remaining_content"] == url
    assert not consumer.final_content_delivered
    event = _event()
    event._stream_delivery_handoff = handoff
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:slack:channel:C1", "Hello", {}, reply_to=None)
    assert adapter.sent == ["Hello", url]
    assert result.success and result.message_id == "gif-2"
    with dl._connect() as conn:
        row = conn.execute("SELECT obligation_id, content, original_content, state FROM delivery_obligations").fetchone()
    assert tuple(row) == (dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", "Hello"),
                          url, "Hello", "delivered")
    outcome = BasePlatformAdapter._turn_delivery_outcome([result], hint="text", streamed_text=None)
    assert outcome["disposition"] == "text"
    assert outcome["text"] == "Hello\n" + url
    assert outcome["message_ids"] == ["prose-1", "gif-2"]
    assert outcome["resolved"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("recovered", (False, True))
async def test_actual_final_lane_records_combined_acks_and_unresolved_tail_on_turn_row(tmp_path, recovered):
    url = "https://example.test/attached.gif"
    final_receipt = (SendResult(success=True, message_id="gif-2", delivered_text=url)
                     if recovered else SendResult(
                         success=False, error="flood_control:120", retry_after=120,
                         raw_response={"remaining_content": url, "delivery_error": "flood_control:120"}))
    adapter = _Adapter([
        SendResult(success=False, error="gif unavailable", message_id="prose-1",
                   delivered_text="Hello", raw_response={
                       "partial_overflow": True, "remaining_content": url,
                       "message_ids": ["prose-1"]}),
        final_receipt,
    ])
    consumer = await _finish(adapter, fallback=True)
    db = SessionDB(db_path=tmp_path / "session.db")
    try:
        session_id = db.create_session("stream-turn", "test")
        db.append_message(session_id, "user", "question")
        db.append_message(session_id, "assistant", "Hello")
        row_id = db.get_messages_as_conversation(session_id, include_row_ids=True)[-1]["_row_id"]
        event = _event()
        event._stream_delivery_handoff = consumer.delivery_handoff_for_final("Hello")
        event.turn_delivery_address = {
            "session_id": session_id, "row_id": row_id, "turn_id": "t", "disposition_hint": "text"}
        runner = GatewayTurnMixin()
        runner._session_db = db
        adapter.gateway_runner = runner
        adapter._message_handler = AsyncMock(return_value="Hello")
        session_key = "agent:main:slack:channel:C1"
        adapter._active_sessions[session_key] = asyncio.Event()
        await adapter._process_message_background(event, session_key)

        delivery = db.get_messages_as_conversation(
            session_id, include_row_ids=True)[-1]["display_metadata"]["delivery"]
        assert delivery["disposition"] == "text"
        assert delivery["text"] == ("Hello\n" + url if recovered else "Hello")
        assert delivery["message_ids"] == (["prose-1", "gif-2"] if recovered else ["prose-1"])
        assert delivery.get("error") == (None if recovered else "flood_control:120")
        assert db.get_messages_as_conversation(session_id)[-1]["content"] == "Hello"
    finally:
        db.close()
    assert adapter.sent == ["Hello", url]
    with dl._connect() as conn:
        row = conn.execute("SELECT content, original_content, state FROM delivery_obligations").fetchone()
    assert tuple(row) == (url, "Hello", "delivered" if recovered else "failed")


@pytest.mark.asyncio
async def test_empty_final_still_delivers_known_remainder_through_normal_bracket():
    url = "https://example.test/attached.gif"
    adapter = _Adapter([
        SendResult(success=False, error="offline", raw_response={"remaining_content": url}),
        SendResult(success=True, message_id="gif", delivered_text=url),
    ])
    consumer = await _finish(adapter)
    event = _event()
    event._stream_delivery_handoff = consumer.delivery_handoff_for_final("")
    adapter._message_handler = AsyncMock(return_value="")
    session_key = "agent:main:slack:channel:C1"
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)
    assert adapter.sent == ["Hello", url]
    assert event._stream_delivery_handoff.complete
    with dl._connect() as conn:
        row = conn.execute("SELECT content, original_content, state FROM delivery_obligations").fetchone()
    assert tuple(row) == (url, "", "delivered")


@pytest.mark.asyncio
async def test_failed_fallback_keeps_later_chunks_and_final_suffix_without_another_stream_send():
    url = "https://example.test/attached.gif"
    adapter = _Adapter([
        SendResult(success=True, message_id="one", delivered_text="A"),
        SendResult(success=False, error="flood", retry_after=120, message_id="two", delivered_text="B",
                   raw_response={"remaining_content": url, "partial_overflow": True,
                                 "message_ids": ["two"], "delivery_error": "flood_control:120"}),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig())
    consumer._fallback_final_send = True
    consumer._edit_supported = False
    consumer._message_id = "__no_edit__"
    consumer._split_text_chunks = lambda text, *args, **kwargs: ["A", "B", "C"]
    consumer.on_delta("ABC")
    consumer.finish("ABC")
    await consumer.run()
    handoff = consumer.delivery_handoff_for_final("ABC plus")
    assert handoff.result.raw_response["remaining_content"] == "C plus\n" + url
    assert handoff.result.delivered_text == "A\nB"
    assert handoff.result.raw_response["message_ids"] == ["one", "two"]
    assert adapter.sent == ["A", "B"]
    assert consumer.delivery_handoff_for_final("Revised").result.raw_response["remaining_content"] == (
        "Revised\n" + url)
    assert not consumer.final_content_delivered


@pytest.mark.asyncio
async def test_empty_fallback_failure_keeps_preview_ack_and_adapter_added_remainder():
    url = "https://example.test/attached.gif"
    adapter = _Adapter([SendResult(
        success=False, error="1 carriage failed", retry_after=120, raw_response={
            "remaining_content": url, "delivery_error": "flood_control:120"})])
    adapter.RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(buffer_only=True))
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Hello"
    consumer._already_sent = True
    consumer._retain_delivery_remainder(
        "Hello", SendResult(success=True, message_id="preview-1", delivered_text="Hello"))
    consumer._edit_supported = False
    consumer._fallback_final_send = True
    consumer.on_delta("Hello")
    consumer.finish("Hello")
    await consumer.run()
    handoff = consumer.delivery_handoff_for_final("Hello")
    assert adapter.sent == ["Hello"]
    assert handoff.result.raw_response["remaining_content"] == url
    assert handoff.result.delivered_text == "Hello"
    assert handoff.result.raw_response["message_ids"] == ["preview-1"]
    assert not consumer.final_content_delivered


@pytest.mark.asyncio
async def test_underlying_flood_classification_bounded_retry_keeps_attached_payload(monkeypatch):
    url = "https://example.test/attached.gif"
    adapter = _Adapter([
        SendResult(success=False, error="2 carriages failed", retry_after=0.1,
                   raw_response={"delivery_error": "flood_control:0.1",
                                 "remaining_content": "Hello\n" + url}),
        SendResult(success=True, message_id="recovered", delivered_text="Hello\n" + url),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1")
    monkeypatch.setattr("gateway.stream_consumer_fallback.asyncio.sleep", AsyncMock())
    result = await consumer._send_with_flood_retry(content="Hello", retry_log="retry %.1f")
    assert result.success
    assert adapter.sent == ["Hello", "Hello\n" + url]


@pytest.mark.asyncio
async def test_first_send_known_remainder_is_not_retried_by_stream():
    url = "https://example.test/attached.gif"
    adapter = _Adapter([SendResult(success=False, error="offline", raw_response={
        "remaining_content": "Hello\n" + url, "delivery_error": "send_path_degraded"})])
    consumer = await _finish(adapter)
    assert adapter.sent == ["Hello"]
    assert consumer.delivery_handoff_for_final("Hello").result.raw_response["remaining_content"] == (
        "Hello\n" + url)
    assert not consumer.final_response_sent


@pytest.mark.asyncio
async def test_interleaved_streams_keep_their_own_remainder():
    adapters = [
        _Adapter([SendResult(success=False, error="offline",
                             raw_response={"remaining_content": f"Hello {i}\nhttps://example.test/{i}.gif"})])
        for i in (1, 2)
    ]
    consumers = [GatewayStreamConsumer(adapter, "C1") for adapter in adapters]
    for i, consumer in enumerate(consumers, start=1):
        consumer.on_delta(f"Hello {i}")
        consumer.finish(f"Hello {i}")
    await asyncio.gather(*(consumer.run() for consumer in consumers))
    assert [consumer.delivery_handoff_for_final(f"Hello {i}").result.raw_response["remaining_content"]
            for i, consumer in enumerate(consumers, start=1)] == [
                "Hello 1\nhttps://example.test/1.gif",
                "Hello 2\nhttps://example.test/2.gif",
            ]
    assert [adapter.sent for adapter in adapters] == [["Hello 1"], ["Hello 2"]]


@pytest.mark.asyncio
async def test_segment_reset_preserves_owed_payload_and_buffers_later_final_text():
    url = "https://example.test/attached.gif"
    adapter = _Adapter([SendResult(
        success=False, error="offline", raw_response={"remaining_content": "Hello\n" + url})])
    consumer = GatewayStreamConsumer(adapter, "C1")
    consumer.on_delta("Hello")
    consumer.on_segment_break()
    consumer.on_delta(" later")
    consumer.finish("Hello later")
    await consumer.run()
    assert adapter.sent == ["Hello"]
    assert consumer.delivery_handoff_for_final("Hello later").result.raw_response["remaining_content"] == (
        "Hello later\n" + url)
    assert not consumer.final_response_sent


def test_handoff_reconciles_replacement_without_dropping_attached_url():
    adapter = _Adapter([])
    consumer = GatewayStreamConsumer(adapter, "C1")
    consumer._retain_delivery_remainder("Hello", SendResult(
        success=False, error="offline", raw_response={
            "remaining_content": "Hello\nhttps://example.test/attached.gif"}))
    handoff = consumer.delivery_handoff_for_final("Revised Hello")
    assert handoff.result.raw_response["remaining_content"] == (
        "Revised Hello\nhttps://example.test/attached.gif")
    assert consumer.delivery_handoff_for_final("Revised Hello\nhttps://example.test/attached.gif").result.raw_response[
        "remaining_content"] == "Revised Hello\nhttps://example.test/attached.gif"


@pytest.mark.asyncio
async def test_runner_transfers_only_current_turn_handoff_without_claiming_already_sent():
    class _Runner(GatewayTurnMixin):
        def _delivery_adapter_for(self, source):
            return None

        def _should_send_voice_reply(self, *args, **kwargs):
            return False

    runner = _Runner()
    adapter = _Adapter([SendResult(success=False, error="offline", raw_response={
        "remaining_content": "Hello\nhttps://example.test/attached.gif"})])
    consumer = await _finish(adapter)
    response = {"final_response": "Hello", "already_sent": True}
    ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=_event().source, session_key="s")
    await runner._run_agent_mark_streamed_delivery(response, ctx)
    assert not response.get("already_sent")
    assert response["_stream_delivery_handoff"].result.raw_response["remaining_content"].endswith(".gif")
    event = _event()
    delivered = await runner._hmwa_deliver_turn_response(
        event, event.source, SimpleNamespace(session_id="s"), "s", 1, response, [], "Hello", None, False)
    assert delivered == "Hello"
    assert event._stream_delivery_handoff.source_content == "Hello"
    assert event._stream_delivery_handoff.result.raw_response["remaining_content"].endswith(".gif")
    other = _event()
    assert not hasattr(other, "_stream_delivery_handoff")
    normalized = _event()
    await runner._hmwa_deliver_turn_response(
        normalized, normalized.source, SimpleNamespace(session_id="s"), "s", 2,
        response, [], "Updated Hello", None, False)
    assert normalized._stream_delivery_handoff.result.raw_response["remaining_content"] == (
        "Updated Hello\nhttps://example.test/attached.gif")


@pytest.mark.asyncio
@pytest.mark.parametrize("queued_success, queued_partial", (
    (False, False), (True, False), (False, True)))
async def test_queued_receipt_reaches_real_turn_record_without_replaying_prior_ack(
    tmp_path, queued_success, queued_partial,
):
    from gateway.run import GatewayRunner

    url = "https://example.test/attached.gif"
    initial_owed = url + "\nA\nB" if queued_partial else url
    queued = (
        SendResult(success=True, message_id="gif", delivered_text=url)
        if queued_success else SendResult(
            success=False, error="flood_control:120", retry_after=120,
            delivered_text=url + "\nA" if queued_partial else None,
            raw_response=(
                {"partial_overflow": True, "remaining_content": "B", "message_ids": ["gif-and-a"]}
                if queued_partial else {"remaining_content": url, "delivery_error": "flood_control:120"})))
    receipts = [SendResult(
        success=False, error="offline", delivered_text="Hello",
        raw_response={"partial_overflow": True, "remaining_content": initial_owed,
                      "message_ids": ["head"]}), queued]
    adapter = _Adapter(receipts)
    consumer = await _finish(adapter, fallback=True)
    db = SessionDB(db_path=tmp_path / "turn.db")
    try:
        session_id = db.create_session("queued", "test")
        db.append_message(session_id, "user", "question")
        db.append_message(session_id, "assistant", "Hello")
        row_id = db.get_messages_as_conversation(session_id, include_row_ids=True)[-1]["_row_id"]
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._delivery_adapter_for = lambda source: adapter
        runner._should_send_voice_reply = lambda *args, **kwargs: False
        adapter.gateway_runner = runner
        result = {"final_response": "Hello", "response_previewed": True,
                  "turn_id": "queued", "messages": [
                      {"role": "assistant", "content": "Hello", "_row_id": row_id}]}
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key="agent:main:slack:channel:C1",
            session_id=session_id, stream_consumer_holder=[consumer], source=_event().source,
            _status_thread_metadata={}, event_message_id="inbound-1",
            inbound_message_id="inbound-1", run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        queued_delivery = db.get_messages_as_conversation(
            session_id, include_row_ids=True)[-1]["display_metadata"]["delivery"]
        assert queued_delivery["text"] == (
            "Hello\n" + url if queued_success else
            "Hello\n" + url + "\nA" if queued_partial else "Hello")
        assert queued_delivery["message_ids"] == (
            ["head", "gif"] if queued_success else
            ["head", "gif-and-a"] if queued_partial else ["head"])
        assert queued_delivery.get("error") == (None if queued_success else "flood_control:120")
        event = _event()

        async def handler(current):
            return await runner._hmwa_deliver_turn_response(
                current, current.source, SimpleNamespace(session_id=session_id),
                ctx.session_key, 1, result, [], "Hello", None, False)

        adapter._message_handler = handler
        adapter._active_sessions[ctx.session_key] = asyncio.Event()
        await adapter._process_message_background(event, ctx.session_key)
        delivery = db.get_messages_as_conversation(
            session_id, include_row_ids=True)[-1]["display_metadata"]["delivery"]
        assert delivery["text"] == queued_delivery["text"]
        assert delivery["message_ids"] == queued_delivery["message_ids"]
        assert delivery.get("error") == queued_delivery.get("error")
        assert adapter.sent == ["Hello", initial_owed]
        with dl._connect() as conn:
            rows = conn.execute(
                "SELECT content,original_content,state FROM delivery_obligations").fetchall()
        assert rows == [(
            ("B" if queued_partial else url), "Hello",
            "delivered" if queued_success else "failed")]
    finally:
        db.close()

@pytest.mark.asyncio
async def test_concurrent_turn_receipts_and_reused_event_keep_distinct_obligations(tmp_path):
    from gateway.run import GatewayRunner

    first_adapter = _Adapter([
        SendResult(success=False, error="ack unknown", message_id="first-head",
                   delivered_text="first", raw_response={
                       "partial_overflow": True, "message_ids": ["first-head"],
                       "delivery_ambiguous": True, "remaining_content": " tail"}),
        SendResult(success=True, message_id="third-id", delivered_text="new answer"),
    ])
    second_adapter = _Adapter([
        SendResult(success=True, message_id="second-id", delivered_text="second"),
    ])
    first_consumer, second_consumer = await asyncio.gather(
        _finish(first_adapter, "first", fallback=True),
        _finish(second_adapter, "second", fallback=True))
    db = SessionDB(db_path=tmp_path / "isolated-turns.db")
    try:
        sid = db.create_session("isolated-turns", "test")
        db.append_message(sid, "user", "question")
        for text in ("first", "second"):
            db.append_message(sid, "assistant", text)
        rows = db.get_messages_as_conversation(sid, include_row_ids=True)
        first_row, second_row = (row["_row_id"] for row in rows[-2:])
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._delivery_adapter_for = lambda source: (
            first_adapter if source.chat_id == "C1" else second_adapter)
        runner._should_send_voice_reply = lambda *args, **kwargs: False
        first, second = _event(), _event()
        second.source = SessionSource(platform=Platform.SLACK, chat_id="C2", chat_type="channel")
        second.message_id = "inbound-2"
        key = "agent:main:slack:channel:C1"
        other_key = "agent:main:slack:channel:C2"
        first_result = {
            "final_response": "first", "turn_id": "first",
            "_stream_delivery_handoff": first_consumer.delivery_handoff_for_final("first"),
            "messages": [{"role": "assistant", "_row_id": first_row}]}
        second_result = {
            "final_response": "second", "turn_id": "second",
            "_stream_delivery_handoff": second_consumer.delivery_handoff_for_final("second"),
            "messages": [{"role": "assistant", "_row_id": second_row}]}

        async def first_handler(event):
            return await runner._hmwa_deliver_turn_response(
                event, event.source, SimpleNamespace(session_id=sid), key, 1,
                first_result, [], first_result["final_response"], None, False)

        async def second_handler(event):
            return await runner._hmwa_deliver_turn_response(
                event, event.source, SimpleNamespace(session_id=sid), other_key, 1,
                second_result, [], "second", None, False)

        first_adapter.gateway_runner = second_adapter.gateway_runner = runner
        first_adapter._message_handler = first_handler
        second_adapter._message_handler = second_handler
        first_adapter._active_sessions[key] = asyncio.Event()
        second_adapter._active_sessions[other_key] = asyncio.Event()
        await asyncio.gather(
            first_adapter._process_message_background(first, key),
            second_adapter._process_message_background(second, other_key))

        db.append_message(sid, "assistant", "new answer")
        third_row = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        first.message_id = "inbound-3"
        first_result = {
            "final_response": "new answer", "turn_id": "third",
            "messages": [{"role": "assistant", "_row_id": third_row}]}
        await first_adapter._process_message_background(first, key)

        deliveries = {
            row["_row_id"]: row["display_metadata"]["delivery"]
            for row in db.get_messages_as_conversation(sid, include_row_ids=True)
            if row["role"] == "assistant"}
        assert first_adapter.sent == ["first", "new answer"]
        assert second_adapter.sent == ["second"]
        assert deliveries[first_row]["text"] == "first"
        assert deliveries[first_row]["message_ids"] == ["first-head"]
        assert deliveries[first_row]["partial"] is True
        assert deliveries[first_row]["error"]
        assert deliveries[second_row]["text"] == "second"
        assert deliveries[second_row]["message_ids"] == ["second-id"]
        assert not deliveries[second_row].get("error")
        assert deliveries[third_row]["text"] == "new answer"
        assert deliveries[third_row]["message_ids"] == ["third-id"]
        assert not deliveries[third_row].get("error")
        with dl._connect() as conn:
            obligations = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchall()
        assert {tuple(row) for row in obligations} == {
            (dl.compute_obligation_id(key, "inbound-1", "first"),
             " tail", "first", "failed"),
            (dl.compute_obligation_id(other_key, "inbound-2", "second"),
             "", "second", "delivered"),
            (dl.compute_obligation_id(key, "inbound-3", "new answer"),
             "new answer", "new answer", "delivered"),
        }
    finally:
        db.close()

@pytest.mark.asyncio
async def test_queued_failure_without_new_remainder_keeps_exact_prior_owed_payload():
    from gateway.run import GatewayRunner

    url = "https://example.test/owed.gif"
    adapter = _Adapter([
        SendResult(success=False, error="offline", delivered_text="Hello",
                   raw_response={"partial_overflow": True, "remaining_content": url,
                                 "message_ids": ["head"]}),
        SendResult(success=False, error="flood_control:120", retry_after=120),
    ])
    consumer = await _finish(adapter, fallback=True)
    runner = object.__new__(GatewayRunner)
    await runner._deliver_queued_first_response(
        "Hello", _event().source, adapter, stream_consumer=consumer,
        session_key="agent:main:slack:channel:C1", inbound_message_id="inbound-1",
        deliver_media=False,
        record_delivery=lambda receipt: consumer.accept_final_delivery_result("Hello", receipt))
    handoff = consumer.delivery_handoff_for_final("Hello")
    assert handoff.result.raw_response["remaining_content"] == url
    assert handoff.result.raw_response["stream_initial_remaining_content"] == url
    assert handoff.result.raw_response["stream_transport_remainder"] == url
    assert handoff.result.raw_response["message_ids"] == ["head"]
    assert handoff.result.error == "flood_control:120"
    assert adapter.sent == ["Hello", url]


@pytest.mark.asyncio
async def test_plain_queued_refusal_stays_owed_on_normal_reentry(tmp_path):
    from gateway.run import GatewayRunner
    adapter = _Adapter([
        SendResult(success=False, error="flood_control:120", retry_after=120),
    ])
    db = SessionDB(db_path=tmp_path / "resolved.db")
    try:
        session_id = db.create_session("resolved", "test")
        db.append_message(session_id, "user", "question")
        db.append_message(session_id, "assistant", "Hello")
        row_id = db.get_messages_as_conversation(session_id, include_row_ids=True)[-1]["_row_id"]
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._delivery_adapter_for = lambda source: adapter
        runner._should_send_voice_reply = lambda *args, **kwargs: False
        adapter.gateway_runner = runner
        result = {"final_response": "Hello", "turn_id": "resolved", "messages": [
            {"role": "assistant", "content": "Hello", "_row_id": row_id}]}
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key="agent:main:slack:channel:C1",
            session_id=session_id, stream_consumer_holder=[None], source=_event().source,
            _status_thread_metadata={}, event_message_id="inbound-1",
            inbound_message_id="inbound-1", run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        prior = db.get_messages_as_conversation(session_id)[-1]["display_metadata"]["delivery"]
        assert prior["disposition"] == "failure"
        assert prior["error"] == "flood_control:120"
        with dl._connect() as conn:
            before = conn.execute(
                "SELECT obligation_id,content,state,attempts,created_at,updated_at,last_error "
                "FROM delivery_obligations").fetchone()
        assert before[2] == "failed" and before[6] == "flood_control:120"

        event = _event()
        async def handler(current):
            return await runner._hmwa_deliver_turn_response(
                current, current.source, SimpleNamespace(session_id=session_id),
                ctx.session_key, 1, result, [], "Hello", None, False)

        adapter._message_handler = handler
        adapter._active_sessions[ctx.session_key] = asyncio.Event()
        await adapter._process_message_background(event, ctx.session_key)
        delivery = db.get_messages_as_conversation(session_id)[-1]["display_metadata"]["delivery"]
        assert delivery["disposition"] == prior["disposition"]
        assert delivery["message_ids"] == prior["message_ids"]
        assert delivery.get("error") == prior.get("error")
        assert adapter.sent == ["Hello"]
        with dl._connect() as conn:
            after = conn.execute(
                "SELECT obligation_id,content,state,attempts,created_at,updated_at,last_error "
                "FROM delivery_obligations").fetchone()
        assert after == before
        assert dl.sweep_failed_for_runtime("slack", now=before[5] + 1, profile="default") == []
    finally:
        db.close()

@pytest.mark.asyncio
@pytest.mark.parametrize("stream_present,prose_landed", (
    (True, True), (False, True), (False, False),
))
@pytest.mark.parametrize("early_return", ("refusal", "stale_goal", "recursion_cap"))
async def test_first_queued_failure_keeps_one_bracket_on_early_return_and_reentry(
    tmp_path, stream_present, prose_landed, early_return,
):
    from gateway.run import GatewayRunner

    url = "https://example.test/queued.gif"
    first = SendResult(
        success=False, error="gif refused", retry_after=120,
        message_id="prose" if prose_landed else None,
        delivered_text="Hello" if prose_landed else None,
        raw_response={"remaining_content": url if prose_landed else "Hello\n" + url,
                      "partial_overflow": prose_landed,
                      "message_ids": ["prose"] if prose_landed else []})
    adapter = _Adapter([first])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig()) if stream_present else None
    db = SessionDB(db_path=tmp_path / "queued-first.db")
    try:
        sid = db.create_session("first-queued", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", "Hello")
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._delivery_adapter_for = lambda source: adapter
        runner._should_send_voice_reply = lambda *args, **kwargs: False
        adapter.gateway_runner = runner
        result = {"final_response": "Hello", "turn_id": "first-queued",
                  "messages": [{"role": "assistant", "content": "Hello", "_row_id": row_id}]}
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key="agent:main:slack:channel:C1",
            session_id=sid, stream_consumer_holder=[consumer], source=_event().source,
            _status_thread_metadata={}, event_message_id="inbound-1",
            inbound_message_id="inbound-1", run_generation=1, _interrupt_depth=0,
            history=[], result_holder=[result])
        runner._MAX_INTERRUPT_DEPTH = 0 if early_return == "recursion_cap" else 8
        runner._is_goal_continuation_event = lambda *_args: early_return == "stale_goal"
        runner._goal_still_active_for_session = lambda *_args: False
        runner._session_key_for_source = lambda *_args: ctx.session_key
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value=None)
        pending = SimpleNamespace(
            source=ctx.source, message_id="next", reply_expected=None,
            message_type=None, internal=False, metadata={})
        returned = await runner._run_agent_queued_followup(
            ctx, adapter, "next", pending, result, result, None)
        assert returned is result
        with dl._connect() as conn:
            before = conn.execute(
                "SELECT obligation_id,content,state,attempts,created_at,updated_at,last_error "
                "FROM delivery_obligations").fetchone()
        event = _event()

        async def handler(current):
            return await runner._hmwa_deliver_turn_response(
                current, current.source, SimpleNamespace(session_id=sid),
                ctx.session_key, 1, result, [], "Hello", None, False)

        adapter._message_handler = handler
        adapter._active_sessions[ctx.session_key] = asyncio.Event()
        await adapter._process_message_background(event, ctx.session_key)
        assert adapter.sent == ["Hello"]
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        assert delivery.get("text") == ("Hello" if prose_landed else None)
        assert delivery["message_ids"] == (["prose"] if prose_landed else [])
        assert delivery["error"]
        with dl._connect() as conn:
            after = conn.execute(
                "SELECT obligation_id,content,state,attempts,created_at,updated_at,last_error "
                "FROM delivery_obligations").fetchone()
        assert after == before
        assert after[1:3] == (url if prose_landed else "Hello\n" + url, "failed")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_final_word_continuation_never_extends_an_owed_attachment_url():
    url = "https://example.test/reaction.gif"
    adapter = _Adapter([
        SendResult(success=False, error="gif refused", message_id="prose",
                   delivered_text="Good morn", raw_response={
                       "remaining_content": url, "partial_overflow": True,
                       "message_ids": ["prose"]}),
        SendResult(success=True, message_id="tail", delivered_text="ing\n" + url),
    ])
    consumer = await _finish(adapter, "Good morn", fallback=True)
    event = _event()
    event._stream_delivery_handoff = consumer.delivery_handoff_for_final("Good morning")
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:slack:channel:C1", "Good morning", {}, reply_to=None)
    assert adapter.sent == ["Good morn", "ing\n" + url]
    assert result.success
    assert result.delivered_text == "Good morn\ning\n" + url
    assert result.raw_response["message_ids"] == ["prose", "tail"]
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == ("ing\n" + url, "Good morning", "delivered")


@pytest.mark.asyncio
async def test_unsent_model_remainder_and_word_continuation_precede_attached_url():
    url = "https://example.test/reaction.gif"
    adapter = _Adapter([
        SendResult(success=False, error="gif refused", message_id="prose",
                   delivered_text="Good ", raw_response={
                       "remaining_content": "morn\n" + url, "partial_overflow": True,
                       "message_ids": ["prose"]}),
        SendResult(success=True, message_id="tail", delivered_text="morning\n" + url),
    ])
    consumer = await _finish(adapter, "Good morn", fallback=True)
    event = _event()
    event._stream_delivery_handoff = consumer.delivery_handoff_for_final("Good morning")
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:slack:channel:C1", "Good morning", {}, reply_to=None)
    assert adapter.sent == ["Good morn", "morning\n" + url]
    assert result.delivered_text == "Good \nmorning\n" + url
    assert result.raw_response["message_ids"] == ["prose", "tail"]
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == ("morning\n" + url, "Good morning", "delivered")


@pytest.mark.asyncio
async def test_failing_split_tail_preserves_sealed_heads_on_final_assistant_row(tmp_path):
    url = "https://example.test/tail.gif"
    head_text = "A" * 650
    tail_text = "B" * 130
    original = head_text + "\n" + tail_text
    # The real stream budget is max(500, adapter limit - 100), not a consumer setting.
    head_chunk, tail_chunk = BasePlatformAdapter.truncate_message(original, 700)
    assert head_chunk.startswith(head_text)
    assert tail_chunk.startswith(tail_text)
    adapter = _Adapter([
        SendResult(success=True, message_id="head-id", delivered_text=head_chunk),
        SendResult(success=False, error="gif refused", message_id="tail-id",
                   delivered_text=tail_text, raw_response={
                       "partial_overflow": True, "remaining_content": url,
                       "message_ids": ["tail-id"]}),
        SendResult(success=True, message_id="gif-id", delivered_text=url),
    ])
    adapter.MAX_MESSAGE_LENGTH = 800
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    consumer.on_delta(original)
    consumer.finish(original)
    await consumer.run()
    db = SessionDB(db_path=tmp_path / "split-tail.db")
    try:
        sid = db.create_session("split-tail", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", original)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        event = _event()
        event._stream_delivery_handoff = consumer.delivery_handoff_for_final(original)
        event.turn_delivery_address = {
            "session_id": sid, "row_id": row_id, "turn_id": "split-tail",
            "disposition_hint": "text"}
        runner = GatewayTurnMixin()
        runner._session_db = db
        adapter.gateway_runner = runner
        adapter._message_handler = AsyncMock(return_value=original)
        key = "agent:main:slack:channel:C1"
        adapter._active_sessions[key] = asyncio.Event()
        await adapter._process_message_background(event, key)
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        assert adapter.sent == [head_chunk, tail_text, url]
        assert delivery["text"] == head_chunk + "\n" + tail_text + "\n" + url
        assert delivery["message_ids"] == ["head-id", "tail-id", "gif-id"]
        with dl._connect() as conn:
            row = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations").fetchone()
        assert tuple(row) == (
            dl.compute_obligation_id(key, "inbound-1", original), url, original, "delivered")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_repeated_model_suffix_is_preserved_before_owed_attachment():
    url = "https://example.test/repeated.gif"
    adapter = _Adapter([
        SendResult(success=False, error="attachment refused", message_id="head",
                   delivered_text="abc", raw_response={
                       "remaining_content": "abc\n" + url,
                       "model_remaining_content": "abc",
                       "attachment_remaining_content": "\n" + url,
                       "partial_overflow": True, "message_ids": ["head"]}),
        SendResult(success=True, message_id="tail", delivered_text="abcabc\n" + url),
    ])
    consumer = await _finish(adapter, "abcabc", fallback=True)
    event = _event()
    event._stream_delivery_handoff = consumer.delivery_handoff_for_final("abcabcabc")
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:slack:channel:C1", "abcabcabc", {}, reply_to=None)
    assert adapter.sent == ["abcabc", "abcabc\n" + url]
    assert result.success
    assert result.raw_response["message_ids"] == ["head", "tail"]
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == ("abcabc\n" + url, "abcabcabc", "delivered")


async def _complete_addressed_handoff(tmp_path, adapter, consumer, final, *, async_db=False):
    """Exercise the normal final bracket and the addressed assistant-row writer."""
    db = SessionDB(db_path=tmp_path / "addressed.db")
    sid = db.create_session("addressed", "test")
    db.append_message(sid, "user", "question")
    db.append_message(sid, "assistant", final)
    row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
    event = _event()
    event._stream_delivery_handoff = consumer.delivery_handoff_for_final(final)
    event.turn_delivery_address = {
        "session_id": sid, "row_id": row_id, "turn_id": "addressed",
        "disposition_hint": "text"}
    runner = GatewayTurnMixin()
    runner._session_db = AsyncSessionDB(db) if async_db else db
    adapter.gateway_runner = runner
    adapter._message_handler = AsyncMock(return_value=final)
    key = "agent:main:slack:channel:C1"
    adapter._active_sessions[key] = asyncio.Event()
    try:
        await adapter._process_message_background(event, key)
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        with dl._connect() as conn:
            obligation = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchone()
        return delivery, tuple(obligation)
    finally:
        db.close()


@pytest.mark.asyncio
async def test_rejected_cursor_preview_only_owes_model_continuation_and_attachment(tmp_path):
    url = "https://example.test/cursor.gif"
    adapter = _Adapter([
        SendResult(success=False, error="preview refused", raw_response={
            "remaining_content": "Good morn ▉\n" + url,
            "model_remaining_content": "Good morn ▉",
            "attachment_remaining_content": "\n" + url}),
        SendResult(success=True, message_id="final-id",
                   delivered_text="Good morning\n" + url),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=" ▉"))
    consumer._stream_ledger = "Good morn"
    await consumer._first_send("Good morn ▉", finalize=False)
    delivery, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, "Good morning")
    assert adapter.sent == ["Good morn ▉", "Good morning\n" + url]
    assert "▉" not in adapter.sent[-1]
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", "Good morning"),
        "Good morning\n" + url, "Good morning", "delivered")
    assert delivery["text"] == "Good morning\n" + url
    assert delivery["message_ids"] == ["final-id"]
    assert "error" not in delivery


@pytest.mark.asyncio
async def test_interim_split_head_survives_later_tail_failure_on_assistant_row(tmp_path):
    head = "A" * 650
    tail = "B" * 130
    original = head + "\n" + tail
    head_chunk, _ = BasePlatformAdapter.truncate_message(original, 700)
    adapter = _Adapter([
        SendResult(success=True, message_id="sealed", delivered_text=head_chunk),
        SendResult(success=False, error="tail refused", raw_response={
            "remaining_content": tail, "delivery_error": "tail refused"}),
        SendResult(success=True, message_id="recovered", delivered_text=tail),
    ])
    adapter.MAX_MESSAGE_LENGTH = 800
    tail_attempted = asyncio.Event()
    send = adapter.send

    async def observed_send(*args, **kwargs):
        receipt = await send(*args, **kwargs)
        if len(adapter.sent) == 2:
            tail_attempted.set()
        return receipt

    adapter.send = observed_send
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    consumer.on_delta(original)
    task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(tail_attempted.wait(), 2)
    consumer.finish(original)
    await task
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, original)
    assert adapter.sent == [head_chunk, tail, tail]
    assert delivery["text"] == head_chunk + "\n" + tail
    assert delivery["message_ids"] == ["sealed", "recovered"]
    assert "error" not in delivery
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", original),
        tail, original, "delivered")


@pytest.mark.asyncio
async def test_earlier_tick_sealed_head_survives_later_tail_failure(tmp_path):
    head = "A" * 650
    tail = "B" * 130
    suffix = "C" * 60
    original = head + "\n" + tail + suffix
    adapter = _Adapter([
        SendResult(success=True, message_id="head", delivered_text=head),
        SendResult(success=True, message_id="tail-id", delivered_text=tail),
        SendResult(success=True, message_id="recovered", delivered_text=suffix),
    ])
    adapter.MAX_MESSAGE_LENGTH = 800
    first_sent = asyncio.Event()
    tail_sent = asyncio.Event()
    edit_failed = asyncio.Event()
    send = adapter.send

    async def observed_send(*args, **kwargs):
        receipt = await send(*args, **kwargs)
        if len(adapter.sent) == 1:
            first_sent.set()
        elif len(adapter.sent) == 2:
            tail_sent.set()
        return receipt

    async def failed_tail_edit(**kwargs):
        edit_failed.set()
        return SendResult(success=False, error="tail edit refused",
                          message_id="tail-id", delivered_text=tail,
                          raw_response={"partial_overflow": True,
                                        "message_ids": ["tail-id"],
                                        "remaining_content": suffix,
                                        "model_remaining_content": suffix,
                                        "attachment_remaining_content": ""})

    adapter.send = observed_send
    adapter.edit_message = failed_tail_edit
    consumer = GatewayStreamConsumer(
        adapter, "C1", StreamConsumerConfig(cursor="", buffer_threshold=1))
    consumer.on_delta(head)
    task = asyncio.create_task(consumer.run())
    await asyncio.wait_for(first_sent.wait(), 2)
    consumer.on_delta("\n" + tail)
    await asyncio.wait_for(tail_sent.wait(), 2)
    await asyncio.sleep(0)
    consumer.on_delta(suffix)
    await asyncio.wait_for(edit_failed.wait(), 2)
    consumer.finish(original)
    await task
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, original)
    assert adapter.sent == [head, tail, suffix]
    assert delivery["text"] == head + "\n" + tail + "\n" + suffix
    assert delivery["message_ids"] == ["head", "tail-id", "recovered"]
    assert not delivery.get("error")
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", original),
        suffix, original, "delivered")


@pytest.mark.asyncio
async def test_acknowledged_split_head_and_zero_ack_timeout_remain_partial(tmp_path):
    head = "A" * 650
    tail = "B" * 130
    original = head + "\n" + tail
    head_chunk, _ = BasePlatformAdapter.truncate_message(original, 700)
    adapter = _Adapter([
        SendResult(success=True, message_id="head", delivered_text=head_chunk),
        SendResult(success=False, error="timeout", raw_response={
            "delivery_ambiguous": True, "remaining_content": tail}),
    ])
    adapter.MAX_MESSAGE_LENGTH = 800
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    consumer.on_delta(original)
    consumer.finish(original)
    await consumer.run()
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, original)
    assert adapter.sent == [head_chunk, tail]
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", original),
        tail, original, "failed")
    assert delivery["disposition"] == "text"
    assert delivery["text"] == head_chunk
    assert delivery["message_ids"] == ["head"]
    assert delivery["partial"] is True
    assert delivery["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_outcome", ("success", "partial", "ambiguous"))
async def test_queued_reconciliation_edit_accounts_for_carriage_before_fallback(
    tmp_path, edit_outcome,
):
    from gateway.run import GatewayRunner

    head, tail = "Hello", " and more"
    final = head + tail
    adapter = _Adapter([
        SendResult(success=True, message_id="preview", delivered_text=head),
        SendResult(success=True, message_id="tail", delivered_text=tail),
    ])
    if edit_outcome == "success":
        edit = SendResult(success=True, message_id="preview", delivered_text=final)
    else:
        edit = SendResult(
            success=False, error="edit timeout" if edit_outcome == "ambiguous" else "edit refused",
            message_id="preview", delivered_text=head,
            raw_response={"partial_overflow": True, "message_ids": ["preview"],
                          "remaining_content": tail,
                          **({"delivery_ambiguous": True} if edit_outcome == "ambiguous" else {})})
    adapter.edit_message = AsyncMock(return_value=edit)
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._first_send(head, finalize=False)
    db = SessionDB(db_path=tmp_path / "queued-edit.db")
    try:
        sid = db.create_session("queued-edit", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", final)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._delivery_adapter_for = lambda source: adapter
        adapter.gateway_runner = runner
        result = {"final_response": final, "turn_id": "queued-edit",
                  "messages": [{"role": "assistant", "content": final, "_row_id": row_id}]}
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key="agent:main:slack:channel:C1",
            session_id=sid, stream_consumer_holder=[consumer], source=_event().source,
            _status_thread_metadata={}, event_message_id="inbound-1",
            inbound_message_id="inbound-1", run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        with dl._connect() as conn:
            obligations = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchall()
        if edit_outcome == "success":
            assert adapter.sent == [head]
            assert [tuple(row) for row in obligations] == [(
                dl.compute_obligation_id(ctx.session_key, "inbound-1", final),
                "", final, "delivered")]
            assert delivery["text"] == final
            assert delivery["message_ids"] == ["preview"]
            assert "error" not in delivery
        else:
            assert adapter.sent == ([head] if edit_outcome == "ambiguous" else [head, tail])
            assert [tuple(row) for row in obligations] == [(
                dl.compute_obligation_id(ctx.session_key, "inbound-1", final),
                tail, final, "failed" if edit_outcome == "ambiguous" else "delivered")]
            assert delivery["text"] == (head if edit_outcome == "ambiguous" else head + "\n" + tail)
            assert delivery["message_ids"] == (
                ["preview"] if edit_outcome == "ambiguous" else ["preview", "tail"])
            assert bool(delivery.get("error")) == (edit_outcome == "ambiguous")
            assert delivery.get("partial", False) == (edit_outcome == "ambiguous")
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transformed", (False, True))
@pytest.mark.parametrize("edit_outcome", ("partial", "ambiguous"))
async def test_normal_reconciliation_edit_retains_partial_or_uncertain_carriage(
    tmp_path, transformed, edit_outcome,
):
    from gateway.run import GatewayRunner

    head, tail = "Hello", " and more"
    final = head + tail
    adapter = _Adapter([
        SendResult(success=True, message_id="preview", delivered_text=head),
        SendResult(success=True, message_id="tail", delivered_text=tail),
    ])
    adapter.edit_message = AsyncMock(return_value=SendResult(
        success=False, error="edit timeout" if edit_outcome == "ambiguous" else "edit refused",
        message_id="preview", delivered_text=head,
        raw_response={"partial_overflow": True, "remaining_content": tail,
                      "message_ids": ["preview"],
                      **({"delivery_ambiguous": True} if edit_outcome == "ambiguous" else {})}))
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._first_send(head, finalize=False)
    if not transformed:
        consumer._final_content_delivered = True
        consumer._delivered_final_text = head
    db = SessionDB(db_path=tmp_path / "normal-edit.db")
    try:
        sid = db.create_session("normal-edit", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", final)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        runner = object.__new__(GatewayRunner)
        runner._session_db = db
        runner._delivery_adapter_for = lambda source: adapter
        runner._should_send_voice_reply = lambda *args, **kwargs: False
        adapter.gateway_runner = runner
        result = {"final_response": final, "response_transformed": transformed,
                  "turn_id": "normal-edit",
                  "messages": [{"role": "assistant", "content": final, "_row_id": row_id}]}
        ctx = SimpleNamespace(stream_consumer_holder=[consumer], source=_event().source,
                              session_key="agent:main:slack:channel:C1")
        await runner._run_agent_mark_streamed_delivery(result, ctx)
        event = _event()

        async def handler(current):
            return await runner._hmwa_deliver_turn_response(
                current, current.source, SimpleNamespace(session_id=sid),
                ctx.session_key, 1, result, [], final, None, False)

        adapter._message_handler = handler
        adapter._active_sessions[ctx.session_key] = asyncio.Event()
        await adapter._process_message_background(event, ctx.session_key)
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        with dl._connect() as conn:
            obligations = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchall()
        assert adapter.sent == ([head] if edit_outcome == "ambiguous" else [head, tail])
        assert [tuple(row) for row in obligations] == [(
            dl.compute_obligation_id(ctx.session_key, "inbound-1", final),
            tail, final, "failed" if edit_outcome == "ambiguous" else "delivered")]
        assert delivery["text"] == (head if edit_outcome == "ambiguous" else head + "\n" + tail)
        assert delivery["message_ids"] == (
            ["preview"] if edit_outcome == "ambiguous" else ["preview", "tail"])
        assert bool(delivery.get("error")) == (edit_outcome == "ambiguous")
        assert delivery.get("partial", False) == (edit_outcome == "ambiguous")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_ambiguous_fresh_final_is_not_erased_by_edit_fallback(tmp_path):
    head, final = "Hello", "Hello plus"
    adapter = _Adapter([
        SendResult(success=True, message_id="preview", delivered_text=head),
        SendResult(success=False, error="fresh ack unknown", raw_response={
            "delivery_ambiguous": True, "remaining_content": final}),
        SendResult(success=True, message_id="duplicate", delivered_text=final),
    ])
    adapter.prefers_fresh_final_streaming = lambda *_args, **_kwargs: True
    adapter.edit_message = AsyncMock(return_value=SendResult(
        success=True, message_id="preview", delivered_text=final))
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._first_send(head, finalize=False)
    consumer._stream_ledger = final
    await consumer._edit_existing(final, finalize=True, is_turn_final=True)
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, final)
    assert adapter.sent == [head, final]
    assert adapter.edit_message.await_count == 0
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final),
        final, final, "failed")
    assert delivery["text"] == head
    assert delivery["message_ids"] == ["preview"]
    assert delivery["partial"] is True
    assert delivery["error"]


@pytest.mark.asyncio
async def test_ambiguous_segment_tail_remains_owned_after_segment_flush(tmp_path):
    head, tail = "Hello", " plus"
    final = head + tail
    adapter = _Adapter([
        SendResult(success=True, message_id="preview", delivered_text=head),
        SendResult(success=False, error="segment ack unknown", raw_response={
            "delivery_ambiguous": True, "remaining_content": tail}),
        SendResult(success=True, message_id="duplicate", delivered_text=final),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._first_send(head, finalize=False)
    consumer._fallback_final_send = True
    consumer._fallback_prefix = head
    consumer._accumulated = consumer._stream_ledger = final
    await consumer._flush_segment_tail_on_edit_failure()
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, final)
    assert adapter.sent == [head, tail]
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final),
        tail, final, "failed")
    assert delivery["text"] == head
    assert delivery["message_ids"] == ["preview"]
    assert delivery["partial"] is True
    assert delivery["error"]

@pytest.mark.asyncio
async def test_ambiguous_tail_after_segment_reset_keeps_head_and_new_model_suffix(tmp_path):
    head, tail, suffix = "Hello", " plus", " more"
    final = head + tail + suffix
    adapter = _Adapter([
        SendResult(success=True, message_id="preview", delivered_text=head),
        SendResult(success=False, error="segment ack unknown", raw_response={
            "delivery_ambiguous": True, "remaining_content": tail}),
        SendResult(success=True, message_id="duplicate", delivered_text=final),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._first_send(head, finalize=False)
    consumer._reset_segment_state()
    consumer._append_accumulated(tail)
    await consumer._flush_segment_tail_on_edit_failure()
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, final)
    assert adapter.sent == [head, tail]
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final),
        tail + suffix, final, "failed")
    assert delivery["text"] == head
    assert delivery["message_ids"] == ["preview"]
    assert delivery["partial"] is True
    assert delivery["error"]


@pytest.mark.asyncio
async def test_empty_fallback_timeout_carries_uncertainty_to_final_bracket(tmp_path):
    final = "Hello"
    adapter = _Adapter([
        SendResult(success=True, message_id="duplicate", delivered_text=final),
    ])
    send = adapter.send
    attempted = []

    async def transport(*args, **kwargs):
        attempted.append(kwargs.get("content", args[1] if len(args) > 1 else None))
        if len(attempted) == 1:
            raise asyncio.TimeoutError("empty fallback ack unknown")
        return await send(*args, **kwargs)

    adapter.send = transport
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._send_empty_fallback_final(final)
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, final)
    assert attempted == [final]
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final),
        final, final, "failed")
    assert delivery["disposition"] == "failure"
    assert delivery["error"]


@pytest.mark.asyncio
async def test_retry_exhaustion_notice_does_not_acknowledge_original_reply(tmp_path, monkeypatch):
    from gateway.platforms import base

    real_sleep = asyncio.sleep

    async def cooperative_retry_wait(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(base.asyncio, "sleep", cooperative_retry_wait)
    final = "The requested answer"
    adapter = _Adapter([
        SendResult(success=False, error="network unavailable", retryable=True),
        SendResult(success=False, error="network unavailable", retryable=True),
        SendResult(success=False, error="network unavailable", retryable=True),
        SendResult(success=True, message_id="notice-id", delivered_text="notice"),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, final)
    assert adapter.sent[:3] == [final, final, final]
    assert len(adapter.sent) == 4
    assert adapter.sent[3] != final
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final),
        final, final, "failed")
    assert delivery["disposition"] == "failure"
    assert delivery["message_ids"] == []
    assert delivery["error"]


@pytest.mark.asyncio
async def test_matching_interim_commentary_timeout_is_not_replayed_as_final(tmp_path):
    final = "Answer"
    adapter = _Adapter([
        SendResult(success=False, error="commentary ack unknown", raw_response={
            "delivery_ambiguous": True, "remaining_content": final}),
        SendResult(success=True, message_id="duplicate", delivered_text=final),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._send_commentary(final)
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, final)
    assert adapter.sent == [final]
    assert obligation == (
        dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final),
        final, final, "failed")
    assert delivery["disposition"] == "failure"
    assert delivery["message_ids"] == []
    assert delivery["error"]

@pytest.mark.asyncio
async def test_returned_telegram_timeout_is_not_replayed_by_stream_segment_or_fresh_final(tmp_path):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    telegram = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    timeout = telegram._classify_send_exception(TimeoutError("Timed out"), ((), (), TimeoutError))
    assert not timeout.success and not timeout.retryable
    assert timeout.error
    for lane in ("ordinary", "segment", "fresh"):
        adapter = _Adapter([SendResult(success=True, message_id="preview", delivered_text="Hello"),
                            timeout, SendResult(success=True, message_id="duplicate", delivered_text="Hello plus")])
        adapter.edit_message = AsyncMock(return_value=SendResult(
            success=True, message_id="preview", delivered_text="Hello plus"))
        consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
        await consumer._first_send("Hello", finalize=False)
        if lane == "fresh":
            adapter.prefers_fresh_final_streaming = lambda *_a, **_kw: True
            consumer._stream_ledger = "Hello plus"
            await consumer._edit_existing("Hello plus", finalize=True, is_turn_final=True)
        elif lane == "segment":
            consumer._fallback_final_send = True
            consumer._fallback_prefix = "Hello"
            consumer._accumulated = consumer._stream_ledger = "Hello plus"
            await consumer._flush_segment_tail_on_edit_failure()
        else:
            consumer._edit_supported = False
            await consumer._send_fallback_final("Hello plus")
        delivery, obligation = await _complete_addressed_handoff(
            tmp_path, adapter, consumer, "Hello plus")
        assert len(adapter.sent) == 2
        assert obligation[3] == "failed"
        assert delivery["message_ids"] == ["preview"]
        assert delivery["error"]


@pytest.mark.asyncio
async def test_ambiguous_receipt_survives_reopen_without_automatic_timer_or_reconnect_replay(
    tmp_path, monkeypatch,
):
    from gateway.run import GatewayRunner

    adapter = _Adapter([SendResult(success=False, error="Timed out", raw_response={
        "delivery_ambiguous": True, "remaining_content": "Answer"})])
    consumer = await _finish(adapter, "Answer", fallback=True)
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, "Answer")
    assert delivery["error"] and obligation[3] == "failed"
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT obligation_id,content,state,last_error FROM delivery_obligations").fetchone()
    assert row[1:3] == ("Answer", "failed")

    # A new runner opens the persisted ledger. Its timer and reconnect path must
    # both decline the uncertain receipt even after an ordinary retry becomes due.
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(dl, "FLOOD_RETRY_SLACK_SECONDS", 0.01)
    fresh = object.__new__(GatewayRunner)
    fresh._running = True
    fresh._authorization_adapter = lambda *_args: adapter
    fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(side_effect=lambda rows, **_kw: rows)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="duplicate"))
    try:
        fresh._schedule_flood_redelivery(Platform.SLACK)
        await asyncio.wait_for(fresh._flood_redelivery_tasks[("slack", "default")], timeout=2)
        await fresh._redeliver_failed_obligations_for_platform(Platform.SLACK)
        adapter.send.assert_not_awaited()
        with dl._connect() as conn:
            current = conn.execute(
                "SELECT obligation_id,content,state,last_error FROM delivery_obligations").fetchone()
        assert current[0] == row[0] and current[1:3] == row[1:3]
    finally:
        fresh._running = False
        for task in getattr(fresh, "_flood_redelivery_tasks", {}).values():
            task.cancel()

@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ("fresh", "empty", "full_resend"))
@pytest.mark.parametrize("deleted", (True, False))
async def test_replacement_receipt_tracks_only_still_visible_messages(tmp_path, lane, deleted):
    class VisibleAdapter(_Adapter):
        def __init__(self):
            super().__init__([])
            self.visible = {}
            self.deleted = []

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            message_id = str(len(self.sent) + 1)
            self.sent.append(content)
            self.visible[message_id] = content
            return SendResult(success=True, message_id=message_id, delivered_text=content)

        async def delete_message(self, chat_id, message_id):
            self.deleted.append(message_id)
            if deleted:
                self.visible.pop(str(message_id))
            return deleted

    adapter = VisibleAdapter()
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    await consumer._first_send("Old", finalize=False)
    if lane == "fresh":
        assert await consumer._try_fresh_final("New")
    elif lane == "empty":
        assert await consumer._send_empty_fallback_final("New") == "delivered"
    else:
        consumer._fallback_prefix = ""
        await consumer._send_fallback_final("New")
    delivery, obligation = await _complete_addressed_handoff(tmp_path, adapter, consumer, "New")
    assert adapter.deleted == (["1", "1"] if lane == "empty" and not deleted else ["1"])
    assert adapter.visible == ({"2": "New"} if deleted else {"1": "Old", "2": "New"})
    assert delivery["message_ids"] == list(adapter.visible)
    assert delivery["text"] == "\n".join(adapter.visible.values())
    assert obligation[3] == "delivered"

@pytest.mark.asyncio
async def test_explicit_durable_recovery_completes_original_addressed_partial_row(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    gif = "https://example.test/recovery.gif"
    adapter = _Adapter([
        SendResult(success=False, error="media refused", message_id="prose",
                   delivered_text="Hello", raw_response={
                       "partial_overflow": True, "message_ids": ["prose"],
                       "remaining_content": gif}),
        SendResult(success=False, error="offline", raw_response={"remaining_content": gif}),
        SendResult(success=False, error="offline", raw_response={"remaining_content": gif}),
    ])
    consumer = await _finish(adapter, "Hello", fallback=True)
    initial, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, "Hello", async_db=True)
    assert initial["text"] == "Hello"
    assert initial["message_ids"] == ["prose"]
    assert initial["partial"] and initial["error"]
    assert obligation[1:4] == (gif, "Hello", "failed")
    with dl._connect() as conn:
        stored = conn.execute(
            "SELECT obligation_id,session_key,original_content,content,state "
            "FROM delivery_obligations").fetchone()
    assert stored[0] == dl.compute_obligation_id(
        "agent:main:slack:channel:C1", "inbound-1", "Hello")
    assert stored[1:4] == ("agent:main:slack:channel:C1", "Hello", gif)

    # The recovery header is transport context, not repeated model prose.
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    reopened = SessionDB(db_path=tmp_path / "addressed.db")
    try:
        fresh = object.__new__(GatewayRunner)
        fresh._session_db = AsyncSessionDB(reopened)
        fresh._authorization_adapter = lambda *_args: adapter
        fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
            side_effect=lambda rows, **_kw: rows)
        adapter.gateway_runner = fresh
        sent = []

        async def recover(chat_id, content, reply_to=None, metadata=None):
            sent.append(content)
            return SendResult(success=True, message_id="gif-recovered", delivered_text=gif)

        adapter.send = recover
        count = await fresh._redeliver_failed_obligations_for_platform(Platform.SLACK)
        assert count == 1
        messages = reopened.get_messages_as_conversation("addressed")
        assistant = [message for message in messages if message["role"] == "assistant"]
        assert len(assistant) == 1 and assistant[0]["content"] == "Hello"
        delivery = assistant[0]["display_metadata"]["delivery"]
        assert delivery["message_ids"] == ["prose", "gif-recovered"]
        assert delivery["text"] == "Hello\n" + gif
        assert "partial" not in delivery and "error" not in delivery
        assert sent[0].endswith(gif) and sent[0] != gif
        with dl._connect() as conn:
            after = conn.execute(
                "SELECT obligation_id,session_key,original_content,content,state "
                "FROM delivery_obligations").fetchone()
        assert after[:4] == stored[:4] and after[4] == "delivered"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_raised_timeout_after_accepted_tail_finalizes_original_bracket_and_address(
    tmp_path, monkeypatch,
):
    from gateway.run import GatewayRunner

    head, final = "Visible head", "Visible head and accepted tail"
    tail = final[len(head):]

    class AcceptedThenUnacknowledged(_Adapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.sent.append(content)
            if content == head:
                return SendResult(success=True, message_id="head-id", delivered_text=head)
            assert content == tail
            raise TimeoutError()

    adapter = AcceptedThenUnacknowledged([])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    assert await consumer._first_send(head, finalize=False)
    delivery, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, final, async_db=True)
    expected_id = dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", final)
    assert obligation == (expected_id, tail, final, "failed")
    assert delivery["text"] == head
    assert delivery["message_ids"] == ["head-id"]
    assert delivery["partial"] is True and delivery["error"]
    with dl._connect() as conn:
        before = conn.execute(
            "SELECT obligation_id,attempts,updated_at,state,last_error "
            "FROM delivery_obligations").fetchone()
    assert before[0] == expected_id and before[3] == "failed" and before[4]
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    fresh = object.__new__(GatewayRunner)
    fresh._authorization_adapter = lambda *_args: adapter
    fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
        side_effect=lambda rows, **_kw: rows)
    fresh.adapters = {Platform.SLACK: adapter}
    fresh._running = True
    try:
        fresh._schedule_flood_redelivery(Platform.SLACK)
        await asyncio.wait_for(fresh._flood_redelivery_tasks[("slack", "default")], timeout=2)
        assert await fresh._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        assert await fresh._redeliver_pending_obligations() == 0
        assert adapter.sent == [head, tail]
        with dl._connect() as conn:
            after = conn.execute(
                "SELECT obligation_id,attempts,updated_at,state,last_error "
                "FROM delivery_obligations").fetchone()
        assert after == before
    finally:
        fresh._running = False
        for task in getattr(fresh, "_flood_redelivery_tasks", {}).values():
            task.cancel()


@pytest.mark.asyncio
async def test_boot_recovery_writes_only_routed_profile_row_with_colliding_address(
    tmp_path, monkeypatch,
):
    import threading
    from pathlib import Path
    from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED, _profile_runtime_scope

    default = tmp_path / ".hermes"
    secondary = default / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default))
    import hermes_state
    # The hermetic conftest pins DEFAULT_DB_PATH; restore the production per-scope
    # resolution so the unpinned runner opens each profile's own state.db.
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    source = SessionSource(platform=Platform.SLACK, chat_id="C1",
                           chat_type="channel", profile="secondary")
    key = "agent:secondary:slack:channel:C1"
    final, head = "Hello, secondary", "Hello"
    tail = final[len(head):]
    adapter = _Adapter([
        SendResult(success=True, message_id="head-id", delivered_text=head),
        SendResult(success=False, error="offline", raw_response={"remaining_content": tail}),
    ])
    adapter._owner_profile = "default"  # transport owner differs from routed row owner
    default_db = SessionDB(db_path=default / "state.db")
    secondary_db = SessionDB(db_path=secondary / "state.db")
    try:
        for db, text in ((default_db, "sentinel"), (secondary_db, final)):
            sid = db.create_session("colliding-session", "test")
            assert sid == "colliding-session"
            db.append_message(sid, "user", "question")
            db.append_message(sid, "assistant", text)
        primary_row = default_db.get_messages_as_conversation(
            "colliding-session", include_row_ids=True)[-1]["_row_id"]
        row_id = secondary_db.get_messages_as_conversation(
            "colliding-session", include_row_ids=True)[-1]["_row_id"]
        assert primary_row == row_id
        assert await AsyncSessionDB(default_db).record_message_delivery(
            "colliding-session", primary_row, "profile-turn",
            {"disposition": "text", "text": "sentinel",
             "message_ids": ["sentinel-id"]})
        sentinel = default_db.get_messages_as_conversation("colliding-session")[-1]
        consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
        assert await consumer._first_send(head, finalize=False)
        event = MessageEvent(text="question", source=source, message_id="inbound-1")
        event._stream_delivery_handoff = consumer.delivery_handoff_for_final(final)
        event.turn_delivery_address = {
            "session_id": "colliding-session", "row_id": row_id,
            "turn_id": "profile-turn", "disposition_hint": "text"}
        first = GatewayTurnMixin()
        first._session_db = AsyncSessionDB(secondary_db)
        adapter.gateway_runner = first
        adapter._message_handler = AsyncMock(return_value=final)
        adapter._active_sessions[key] = asyncio.Event()
        with _profile_runtime_scope(secondary, {}):
            await adapter._process_message_background(event, key)
        before = secondary_db.get_messages_as_conversation("colliding-session")[-1]
        assert before["display_metadata"]["delivery"]["message_ids"] == ["head-id"]
        assert before["display_metadata"]["delivery"]["error"]
        with dl._connect() as conn:
            obligation = conn.execute(
                "SELECT obligation_id,session_key,content,original_content,state,adapter_profile "
                "FROM delivery_obligations").fetchone()
        assert tuple(obligation) == (
            dl.compute_obligation_id(key, "inbound-1", final),
            key, tail, final, "failed", "default")
        monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
        # Boot recovery may claim only a previous process's obligation.
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
                "WHERE obligation_id=?", (obligation[0],))
        fresh = object.__new__(GatewayRunner)
        fresh._session_db_pinned = _SESSION_DB_UNPINNED
        fresh._session_db_handles = {}
        fresh._session_db_handles_lock = threading.Lock()
        fresh.adapters = {Platform.SLACK: adapter}
        fresh._profile_adapters = {}
        fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
            side_effect=lambda rows, **_kw: rows)
        adapter.gateway_runner = fresh
        sent = []

        async def recover(chat_id, content, reply_to=None, metadata=None):
            sent.append(content)
            return SendResult(success=True, message_id="tail-id", delivered_text=content)

        adapter.send = recover
        try:
            assert fresh._resolve_profile_home_for_source(source) == secondary
            with _profile_runtime_scope(default, {}):
                assert isinstance(fresh._session_db, AsyncSessionDB)
                assert await fresh._redeliver_pending_obligations() == 1
            resolved = secondary_db.get_messages_as_conversation("colliding-session")[-1]
            assert resolved["display_metadata"]["delivery"]["message_ids"] == [
                "head-id", "tail-id"]
            assert resolved["display_metadata"]["delivery"]["text"] == head + "\n" + tail
            assert "error" not in resolved["display_metadata"]["delivery"]
            assert default_db.get_messages_as_conversation("colliding-session")[-1] == sentinel
            with dl._connect() as conn:
                after = conn.execute(
                    "SELECT obligation_id,session_key,content,original_content,state,adapter_profile "
                    "FROM delivery_obligations").fetchone()
            assert tuple(after[:4]) == tuple(obligation[:4])
            assert after[4:] == ("delivered", "default")
            assert len(sent) == 1 and sent[0].endswith(tail) and final not in sent[0]
        finally:
            fresh.close_all_session_db_handles()
            for task in getattr(fresh, "_flood_redelivery_tasks", {}).values():
                task.cancel()
    finally:
        secondary_db.close()
        default_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ("empty_fallback", "fallback_entry"))
async def test_successful_cursor_cleanup_edit_is_visible_on_identical_final_row(
    tmp_path, cleanup,
):
    final, cursor = "Visible prose", " ▉"
    adapter = _Adapter([SendResult(
        success=True, message_id="preview", delivered_text=final + cursor)])
    adapter.edit_message = AsyncMock(return_value=SendResult(
        success=True, message_id="preview", delivered_text=final))
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=cursor))
    consumer._stream_ledger = final
    assert await consumer._first_send(final + cursor, finalize=False)
    if cleanup == "empty_fallback":
        assert await consumer._fallback_when_nothing_unseen(final) is None
    else:
        await consumer._try_strip_cursor()
        assert await consumer._fallback_when_nothing_unseen(final) is None
    delivery, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, final, async_db=True)
    assert adapter.edit_message.await_count == 1
    assert adapter.sent == [final + cursor]
    assert delivery["message_ids"] == ["preview"]
    assert delivery["text"] == final
    assert obligation[3] == "delivered"


@pytest.mark.asyncio
async def test_model_authored_cursor_is_not_removed_as_consumer_suffix(tmp_path):
    model = "Text▉"
    adapter = _Adapter([SendResult(
        success=True, message_id="authored", delivered_text=model)])
    adapter.edit_message = AsyncMock(return_value=SendResult(
        success=True, message_id="authored", delivered_text="Text"))
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor="▉"))
    consumer._stream_ledger = model
    assert await consumer._first_send(model, finalize=False)
    await consumer._try_strip_cursor()
    delivery, _ = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, model, async_db=True)
    assert delivery["text"] == model
    adapter.edit_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("model,cursor", (
    ("Text▉", "▉"), ("Text▉▉", "▉"), ("Text ▉", " ▉"),
))
async def test_rejected_preview_keeps_authored_suffix_but_not_appended_cursor(
    tmp_path, model, cursor,
):
    attempted = model + cursor
    adapter = _Adapter([
        SendResult(success=False, error="preview refused",
                   raw_response={"remaining_content": attempted}),
        SendResult(success=True, message_id="final-id", delivered_text=model),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=cursor))
    consumer._stream_ledger = model
    await consumer._first_send(attempted, finalize=False)
    delivery, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, model, async_db=True)
    assert adapter.sent == [attempted, model]
    assert delivery["text"] == model
    assert delivery["message_ids"] == ["final-id"]
    assert "error" not in delivery
    assert obligation[1:4] == (model, model, "delivered")


@pytest.mark.parametrize("preview,model,tail", (
    ("Text", "Text▉", "▉"),
    ("Text▉", "Text▉ plus", " plus"),
))
@pytest.mark.asyncio
async def test_returned_edit_refusal_preserves_authored_cursor_but_not_appended_cursor(
    tmp_path, preview, model, tail,
):
    cursor = "▉"
    adapter = _Adapter([
        SendResult(success=True, message_id="preview", delivered_text=preview),
        SendResult(success=True, message_id="tail", delivered_text=tail),
    ])
    attempted = []

    async def edit(chat_id, message_id, content, *, finalize=False):
        attempted.append(content)
        return SendResult(success=False, error="edit refused", raw_response={
            "remaining_content": content, "model_remaining_content": content,
            "attachment_remaining_content": ""})

    adapter.edit_message = edit
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=cursor))
    consumer._stream_ledger = preview
    assert await consumer._first_send(preview, finalize=False)
    consumer._stream_ledger = model
    assert not await consumer._edit_existing(model + cursor, finalize=False, is_turn_final=False)
    delivery, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, model, async_db=True)
    assert attempted == [model + cursor]
    assert adapter.sent == [preview, tail]
    assert delivery["text"] == preview + "\n" + tail
    assert delivery["message_ids"] == ["preview", "tail"]
    assert obligation[1:4] == (tail, model, "delivered")


@pytest.mark.asyncio
@pytest.mark.parametrize("image_success", (True, False))
async def test_text_recovery_preserves_independent_image_delivery_outcome(
    tmp_path, monkeypatch, image_success,
):
    from gateway.run import GatewayRunner

    image_url = "https://example.test/photo.png"
    response = f"Hello ![photo]({image_url})"
    final, head = "Hello", "Hel"
    tail = final[len(head):]
    adapter = _Adapter([
        SendResult(success=True, message_id="head-id", delivered_text=head),
        SendResult(success=False, error="text tail refused",
                   raw_response={"remaining_content": tail}),
    ])
    adapter.send_multiple_images = AsyncMock(return_value=SendResult(
        success=image_success, message_id="image-id" if image_success else None,
        error=None if image_success else "image refused"))
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    assert await consumer._first_send(head, finalize=False)
    db = SessionDB(db_path=tmp_path / "independent-media.db")
    try:
        sid = db.create_session("independent-media", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", response)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        runner = GatewayTurnMixin()
        runner._session_db = AsyncSessionDB(db)
        adapter.gateway_runner = runner
        adapter._message_handler = AsyncMock(return_value=response)
        event = _event()
        event._stream_delivery_handoff = consumer.delivery_handoff_for_final(final)
        event.turn_delivery_address = {
            "session_id": sid, "row_id": row_id, "turn_id": "independent-media",
            "disposition_hint": "text"}
        key = "agent:main:slack:channel:C1"
        adapter._active_sessions[key] = asyncio.Event()
        await adapter._process_message_background(event, key)
        adapter.send_multiple_images.assert_awaited_once()
        initial = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        assert initial["message_ids"] == (
            ["head-id", "image-id"] if image_success else ["head-id"])
        assert initial["partial"] and initial["error"]
        assert initial["lanes"]["text"]["message_ids"] == ["head-id"]
        if image_success:
            assert initial["lanes"]["media"]["message_ids"] == ["image-id"]
        else:
            assert initial["lanes"]["media"]["error"] == "image refused"
        with dl._connect() as conn:
            before = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchone()
        assert tuple(before) == (
            dl.compute_obligation_id(key, "inbound-1", final),
            tail, final, "failed")
        monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
        reopened = SessionDB(db_path=tmp_path / "independent-media.db")
        try:
            fresh = object.__new__(GatewayRunner)
            fresh._session_db = AsyncSessionDB(reopened)
            fresh._authorization_adapter = lambda *_args: adapter
            fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
                side_effect=lambda rows, **_kw: rows)
            adapter.gateway_runner = fresh
            sent = []

            async def recover(chat_id, content, reply_to=None, metadata=None):
                sent.append(content)
                return SendResult(success=True, message_id="tail-id", delivered_text=tail)

            adapter.send = recover
            assert await fresh._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
            resolved = reopened.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
            assert resolved["message_ids"] == (
                ["head-id", "image-id", "tail-id"] if image_success
                else ["head-id", "tail-id"])
            assert resolved["lanes"]["text"]["message_ids"] == ["head-id", "tail-id"]
            assert resolved["lanes"]["media"]["message_ids"] == (
                ["image-id"] if image_success else [])
            assert resolved["text"] == head + "\n" + tail
            if image_success:
                assert "error" not in resolved and "partial" not in resolved
            else:
                assert resolved["partial"] and "image refused" in resolved["error"]
            assert len(sent) == 1 and sent[0].endswith(tail) and final not in sent[0]
            adapter.send_multiple_images.assert_awaited_once()
            with dl._connect() as conn:
                after = conn.execute(
                    "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
                ).fetchone()
            assert after[:3] == before[:3] and after[3] == "delivered"
        finally:
            reopened.close()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_recovery_of_rejected_cursor_preview_sends_authored_suffix_once(
    tmp_path, monkeypatch,
):
    from gateway.run import GatewayRunner

    model, cursor = "Text▉▉", "▉"
    adapter = _Adapter([
        SendResult(success=False, error="preview refused",
                   raw_response={"remaining_content": model + cursor}),
        SendResult(success=False, error="offline",
                   raw_response={"remaining_content": model}),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=cursor))
    consumer._stream_ledger = model
    await consumer._first_send(model + cursor, finalize=False)
    failed, obligation = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, model, async_db=True)
    assert failed["error"] and failed["message_ids"] == []
    assert obligation[1:4] == (model, model, "failed")
    assert adapter.sent == [model + cursor, model]
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    db = SessionDB(db_path=tmp_path / "addressed.db")
    try:
        fresh = object.__new__(GatewayRunner)
        fresh._session_db = AsyncSessionDB(db)
        fresh._authorization_adapter = lambda *_args: adapter
        fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
            side_effect=lambda rows, **_kw: rows)
        adapter.gateway_runner = fresh
        requested = []

        async def recover(chat_id, content, reply_to=None, metadata=None):
            requested.append(content)
            return SendResult(success=True, message_id="recovered", delivered_text=content)

        adapter.send = recover
        assert await fresh._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        assert len(requested) == 1 and requested[0].endswith(model)
        assert requested[0].count(model) == 1 and not requested[0].endswith(model + cursor)
        delivery = db.get_messages_as_conversation("addressed")[-1]["display_metadata"]["delivery"]
        assert delivery["message_ids"] == ["recovered"]
        assert delivery["text"] == model
        assert "error" not in delivery
        with dl._connect() as conn:
            after = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchone()
        assert tuple(after) == (
            dl.compute_obligation_id("agent:main:slack:channel:C1", "inbound-1", model),
            model, model, "delivered")
    finally:
        db.close()


@pytest.mark.asyncio
async def test_recovery_row_refusal_does_not_settle_obligation_as_delivered(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    final, head = "Acknowledged answer", "Acknowledged"
    tail = final[len(head):]
    adapter = _Adapter([
        SendResult(success=True, message_id="head", delivered_text=head),
        SendResult(success=False, error="offline", raw_response={"remaining_content": tail}),
    ])
    consumer = GatewayStreamConsumer(adapter, "C1", StreamConsumerConfig(cursor=""))
    assert await consumer._first_send(head, finalize=False)
    initial, _ = await _complete_addressed_handoff(
        tmp_path, adapter, consumer, final, async_db=True)
    assert initial["message_ids"] == ["head"] and initial["error"]
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    wrong_db = SessionDB(db_path=tmp_path / "missing-address.db")
    try:
        fresh = object.__new__(GatewayRunner)
        fresh._session_db = AsyncSessionDB(wrong_db)
        fresh._authorization_adapter = lambda *_args: adapter
        fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
            side_effect=lambda rows, **_kw: rows)
        adapter.gateway_runner = fresh
        delivered = []

        async def recover(chat_id, content, reply_to=None, metadata=None):
            delivered.append(content)
            return SendResult(success=True, message_id="tail", delivered_text=tail)

        adapter.send = recover
        assert await fresh._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        assert len(delivered) == 1
        with dl._connect() as conn:
            row = conn.execute(
                "SELECT state,turn_receipt FROM delivery_obligations").fetchone()
        assert row[0] == "failed"
        assert StreamDeliveryHandoff.from_record(json.loads(row[1])).ambiguous
    finally:
        wrong_db.close()
