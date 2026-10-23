"""A queued-lane final is ledger-bracketed like the normal final send.

When a follow-up is queued behind a turn (a message typed while the agent worked, a
subagent batch reporting back), the first response is delivered by the queued lane in
``run_turn.py`` before the follow-up runs. That lane called ``adapter.send`` bare and
discarded the result: no delivery-ledger row was recorded, so a final refused there
(flood control, a transport that had just died) was lost for good. Neither the boot
sweep nor the runtime redelivery could see it, and the follow-up ran as if the answer
had landed. On 5 Sep 2026 two long replies were lost this way within an hour, both
refused by Telegram flood control while a delegation batch arrived the same second.

Now the queued lane runs the same bracket as the normal lane: the obligation is
recorded before the send, the send goes through the adapter's retrying transport, and
the result finalizes the row. The obligation id is keyed on the raw inbound message id
exactly as the normal lane keys it; the reply anchor is only the reply target and is
None wherever replies are not used (Telegram forum topics), so it cannot identify the
turn. Reconciliation edits are reduced into the same turn receipt before deciding
whether any safe remainder can be sent. Adapters without the base contract and sends
without a session key keep the plain send.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult

SESSION_KEY = "agent:main:telegram:dm:5230977008"
TOPIC_SESSION_KEY = "agent:main:telegram:group:-1001:topic:7"
CHAT = "5230977008"
INBOUND_ID = "5301"
TEXT = "**Yes.** I would make room for one island trip and one Balkan trip."


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    yield


_COLUMNS = ("obligation_id", "session_key", "state", "attempts", "last_error", "content", "chat_id",
            "platform")


def _rows():
    with dl._connect() as conn:
        cur = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM delivery_obligations")
        return [dict(zip(_COLUMNS, r)) for r in cur.fetchall()]


def _source(*, chat_id=CHAT, thread_id=None, chat_type="dm"):
    # parent_chat_id is a real SessionSource field; the queued-followup path reads it to pin the
    # channel inputs, so the stub carries it like every real source does.
    return SimpleNamespace(platform=Platform.TELEGRAM, chat_id=chat_id, thread_id=thread_id,
                           chat_type=chat_type, parent_chat_id=None)


def _telegram_adapter(send_result=None):
    """A real Telegram adapter (the base contract) whose transport send is a mock."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    runner = MagicMock()
    runner._delivery_adapter_for = MagicMock(return_value=adapter)
    runner._schedule_flood_redelivery = MagicMock(return_value=187.0)
    adapter.gateway_runner = runner
    adapter.send = AsyncMock(return_value=send_result or SendResult(success=True, message_id="900"))
    adapter.edit_message = AsyncMock(
        return_value=SendResult(success=False, error="message to edit not found"))
    return adapter


def _plain_adapter():
    """A relay-style double without the base contract: only ``send`` and the media helpers."""
    return SimpleNamespace(
        name="relay", extract_media=BasePlatformAdapter.extract_media,
        send=AsyncMock(return_value=SendResult(success=True, message_id="p1")))


def _runner():
    from gateway.run import GatewayRunner

    return object.__new__(GatewayRunner)


async def _deliver(adapter, *, session_key=SESSION_KEY, stream_consumer=None, metadata=None,
                   text=TEXT, source=None, anchor=INBOUND_ID, inbound_id=INBOUND_ID):
    from gateway.run import GatewayRunner

    await GatewayRunner._deliver_queued_first_response(
        _runner(), text, source=source or _source(), adapter=adapter, metadata=metadata,
        event_message_id=anchor, text_already_delivered=False, deliver_media=False,
        stream_consumer=stream_consumer, session_key=session_key, inbound_message_id=inbound_id)


# ---------------------------------------------------------------------------
# The bracket: record before the send, finalize from the result.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_delivered_queued_final_is_recorded_and_marked_delivered():
    adapter = _telegram_adapter()

    await _deliver(adapter)

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "delivered"
    assert rows[0]["session_key"] == SESSION_KEY
    assert rows[0]["content"] == TEXT
    assert (rows[0]["platform"], rows[0]["chat_id"]) == ("telegram", CHAT)
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_forum_topic_turns_are_identified_by_their_inbound_id_not_the_reply_anchor():
    """Telegram forum topics route by topic metadata and never reply, so the anchor is None for
    every message in the topic. Two turns answering with the same text must still get two rows,
    or the second would overwrite the first's outstanding obligation and retry state."""
    adapter = _telegram_adapter()
    topic = _source(chat_id="-1001", thread_id="7", chat_type="supergroup")

    await _deliver(adapter, session_key=TOPIC_SESSION_KEY, source=topic, anchor=None,
                   inbound_id="5301")
    await _deliver(adapter, session_key=TOPIC_SESSION_KEY, source=topic, anchor=None,
                   inbound_id="5302")

    ids = sorted(r["obligation_id"] for r in _rows())
    assert ids == sorted([dl.compute_obligation_id(TOPIC_SESSION_KEY, "5301", TEXT),
                          dl.compute_obligation_id(TOPIC_SESSION_KEY, "5302", TEXT)])
    assert [c.kwargs["reply_to"] for c in adapter.send.await_args_list] == [None, None]


@pytest.mark.asyncio
async def test_a_flood_refused_queued_final_stays_in_the_ledger_as_failed(caplog):
    """The row survives the refusal, so the sweeps (and a flood timer, where present) can redeliver
    it."""
    adapter = _telegram_adapter(SendResult(success=False, error="flood_control:185.0"))

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await _deliver(adapter)

    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["state"] == "failed"
    assert rows[0]["last_error"] == "flood_control:185.0"


@pytest.mark.asyncio
async def test_the_queued_final_is_sent_like_the_normal_final():
    """Reply anchor on the inbound message and a notify-worthy copy of the thread metadata."""
    adapter = _telegram_adapter()
    metadata = {"thread_id": "topic-7"}

    await _deliver(adapter, metadata=metadata)

    kwargs = adapter.send.await_args.kwargs
    assert kwargs["chat_id"] == CHAT
    assert kwargs["content"] == TEXT
    assert kwargs["reply_to"] == INBOUND_ID
    assert kwargs["metadata"]["notify"] is True
    assert kwargs["metadata"]["thread_id"] == "topic-7"
    assert metadata == {"thread_id": "topic-7"}  # the caller's dict is cloned, not mutated


# ---------------------------------------------------------------------------
# What is unchanged: reconcile-by-edit, plain adapters, no session key.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_adapter_without_the_base_contract_keeps_the_plain_send():
    adapter = _plain_adapter()

    await _deliver(adapter, metadata={"thread_id": "t"})

    adapter.send.assert_awaited_once_with(CHAT, TEXT, metadata={"thread_id": "t"})
    assert _rows() == []


# ---------------------------------------------------------------------------
# The call site hands the session key and the raw inbound id over.
# ---------------------------------------------------------------------------

def _chain_runner_and_ctx(followup_return):
    """A runner whose recursive ``_run_agent`` returns ``followup_return``, plus a topic turn."""
    from gateway.run import GatewayRunner

    runner = _runner()
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value=followup_return)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=TOPIC_SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="the follow-up")
    runner._reply_anchor_for_event = MagicMock(return_value=None)
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    topic = _source(chat_id="-1001", thread_id="7", chat_type="supergroup")
    turn_ctx = SimpleNamespace(
        source=topic, session_id="sid", session_key=TOPIC_SESSION_KEY, run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata={"thread_id": "7"},
        context_prompt=None, result_holder=[None])
    pending_event = SimpleNamespace(
        source=topic, message_id="6002", channel_prompt=None, message_type=None,
        internal=False, metadata={}, reply_expected=None)
    return GatewayRunner, runner, turn_ctx, pending_event


@pytest.mark.asyncio
async def test_a_chained_queued_turn_carries_its_own_inbound_id():
    """A queued follow-up that itself queues another follow-up (recursive _run_agent) must pass the
    inbound id on, or the chained turn's own queued final would key on None in a forum topic and
    collide with a sibling. The recursive call carries pending_event's raw message id."""
    from gateway.run import GatewayRunner

    runner = _runner()
    runner._MAX_INTERRUPT_DEPTH = 8
    runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": []})
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._is_goal_continuation_event = MagicMock(return_value=False)
    runner._session_key_for_source = MagicMock(return_value=TOPIC_SESSION_KEY)
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="the follow-up")
    runner._reply_anchor_for_event = MagicMock(return_value=None)   # forum topic: no anchor
    runner._delivery_adapter_for = MagicMock(return_value=None)
    runner._refresh_agent_cache_message_count = AsyncMock()
    topic = _source(chat_id="-1001", thread_id="7", chat_type="supergroup")
    turn_ctx = SimpleNamespace(
        source=topic, session_id="sid", session_key=TOPIC_SESSION_KEY, run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata={"thread_id": "7"},
        context_prompt=None, result_holder=[None])
    pending_event = SimpleNamespace(
        source=topic, message_id="6002", channel_prompt=None, message_type=None,
        internal=False, metadata={}, reply_expected=None)

    await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hi again", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None)

    runner._run_agent.assert_awaited_once()
    assert runner._run_agent.await_args.kwargs["inbound_message_id"] == "6002"
    assert runner._run_agent.await_args.kwargs["event_message_id"] is None


# ---------------------------------------------------------------------------
# A chain's terminal reply must not overwrite an earlier turn's row.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_terminal_turn_of_a_chain_is_ledgered_under_its_own_inbound_id():
    """Two turns of one queued chain answering with the SAME text must keep two ledger rows.

    The outer final send is bracketed by the adapter against the event that OPENED the chain. Keyed
    on that event's id, a terminal reply carrying the same text as an earlier refused reply computes
    the earlier row's obligation id, replaces its outstanding row and marks it delivered, so the
    refused reply is never redelivered. ``MessageEvent.ledger_message_id`` carries the terminal
    turn's own inbound id so the rows stay distinct. ``_send_with_retry`` is stubbed here because
    the retry ladder is covered elsewhere and this pins the ledger identity only.
    """
    from gateway.platforms.base import MessageEvent

    same_text = "Done."
    adapter = _telegram_adapter()
    # Turn A (inbound 101): its queued final is refused by flood control and stays outstanding.
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=False, error="flood_control:30.0"))
    await _deliver(adapter, text=same_text, anchor="101", inbound_id="101")

    first = _rows()
    assert len(first) == 1
    assert first[0]["state"] == "failed"
    turn_a_id = first[0]["obligation_id"]

    # Turn B is the chain's TERMINAL turn (inbound 102). The adapter still holds turn A's event, so
    # only the ledger override distinguishes the row.
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="901"))
    event = MessageEvent(text="hi again", source=_source(), message_id="101",
                         ledger_message_id="102")
    await adapter._send_final_text(event, SESSION_KEY, same_text, {}, False, 0, lambda _r: None)

    rows = {r["obligation_id"]: r for r in _rows()}
    assert len(rows) == 2, "the terminal reply reused the earlier turn's obligation id"
    assert rows[turn_a_id]["state"] == "failed", \
        "turn A's refused reply was overwritten and would never be redelivered"
    assert rows[dl.compute_obligation_id(SESSION_KEY, "102", same_text)]["state"] == "delivered"


@pytest.mark.asyncio
async def test_a_deeper_chain_keeps_the_innermost_inbound_id():
    """Chained follow-ups nest, and the LAST message answered owns the ledger identity, so an id
    already set by a deeper recursion must not be overwritten on the way out."""
    GatewayRunner, runner, turn_ctx, pending_event = _chain_runner_and_ctx(
        {"final_response": "done", "messages": [], "queued_terminal_inbound_id": "6003"})

    merged = await GatewayRunner._run_agent_queued_followup(
        runner, turn_ctx, adapter=None, pending="hi again", pending_event=pending_event,
        response="resp", result={"interrupted": True, "messages": []}, stream_task=None)

    assert merged["queued_terminal_inbound_id"] == "6003"


# ---------------------------------------------------------------------------
# The adapter that sent the final owns its message id: the ephemeral delete goes there.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ephemeral_delete_targets_the_adapter_that_sent_the_final(tmp_path, monkeypatch):
    """A reconnect between the send and the delete swaps the runner's live adapter; the delete
    must still go to the transport that produced ``result.message_id``."""
    from gateway.platforms.event import MessageEvent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sender = _telegram_adapter()
    replacement = _telegram_adapter()
    sender._schedule_ephemeral_delete = MagicMock()
    replacement._schedule_ephemeral_delete = MagicMock()
    live = {"adapter": sender}
    sender.gateway_runner._delivery_adapter_for = MagicMock(side_effect=lambda _s: live["adapter"])

    async def swap_then_send(*args, **kwargs):
        live["adapter"] = replacement  # reconnect lands while the send is in flight
        return SendResult(success=True, message_id="900")

    sender.send = AsyncMock(side_effect=swap_then_send)
    event = MessageEvent(text="hi", source=_source(), message_id=INBOUND_ID)
    await sender._send_final_text(event, SESSION_KEY, TEXT, {}, False, 30, lambda _r: None)

    sender._schedule_ephemeral_delete.assert_called_once_with(CHAT, "900", 30)
    replacement._schedule_ephemeral_delete.assert_not_called()


@pytest.mark.asyncio
async def test_queued_stream_handoff_preserves_gif_and_original_identity():
    from contextlib import closing
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _telegram_adapter()
    gif = "https://example.com/owed.gif"
    visible = []
    first = True

    async def transport(chat_id, content, **kwargs):
        nonlocal first
        if first:
            first = False
            visible.append("Hello")
            return SendResult(
                success=False, message_id="landed-prose", delivered_text="Hello",
                error="send_path_degraded", retryable=True,
                raw_response={"partial_overflow": True, "remaining_content": gif,
                              "message_ids": ["landed-prose"]})
        visible.append(content)
        return SendResult(success=True, message_id="landed-tail", delivered_text=content)

    adapter.send = transport
    consumer = GatewayStreamConsumer(
        adapter, CHAT, StreamConsumerConfig(cursor=""), initial_reply_to_id=INBOUND_ID)
    await consumer._send_fallback_final("Hello")
    await _deliver(adapter, text="Hello", stream_consumer=consumer)
    assert visible == ["Hello", gif]
    with closing(dl._connect()) as conn:
        rows = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchall()
    assert rows == [(gif, "Hello", "delivered")]


@pytest.mark.asyncio
async def test_queued_partial_rebases_owed_tail_without_ordinary_replay():
    from contextlib import closing
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    adapter = _telegram_adapter()
    gif = "https://example.com/owed.gif"
    final = "Hello"
    visible = []

    async def transport(chat_id, content, **kwargs):
        visible.append(content)
        if len(visible) == 1:
            return SendResult(success=False, error="offline", delivered_text=final,
                              raw_response={"partial_overflow": True, "remaining_content": gif + "\nA\nB",
                                            "message_ids": ["head"]})
        if len(visible) == 2:
            return SendResult(success=False, error="flood_control:120", retry_after=120,
                              delivered_text=gif + "\nA", raw_response={
                                  "partial_overflow": True, "remaining_content": "B",
                                  "message_ids": ["gif-and-a"]})
        return SendResult(success=True, message_id="tail", delivered_text=content)

    adapter.send = transport
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    await consumer._send_fallback_final(final)
    consumer._message_id = "head"  # the acknowledged head is still editable
    consumer._turn_split_delivery = False
    from gateway.run import GatewayRunner
    queued_results = []

    def record(receipt):
        queued_results.append(receipt)
        consumer.accept_final_delivery_result(final, receipt)

    delivered = await GatewayRunner._deliver_queued_first_response(
        _runner(), final, _source(), adapter, stream_consumer=consumer,
        session_key=SESSION_KEY, inbound_message_id=INBOUND_ID, deliver_media=False,
        record_delivery=record)
    assert not delivered
    assert len(queued_results) == 1
    assert queued_results[0].delivered_text == "Hello\n" + gif + "\nA"
    assert consumer.delivery_handoff_for_final(final).result.raw_response["remaining_content"] == "B"
    assert consumer.delivery_handoff_for_final("Hello plus").result.raw_response["remaining_content"] == " plus\nB"
    assert adapter.edit_message.await_count == 0
    assert visible == [final, gif + "\nA\nB"]
    with closing(dl._connect()) as conn:
        rows = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchall()
    assert rows == [("B", final, "failed")]


@pytest.mark.asyncio
async def test_queued_handoff_on_plain_adapter_sends_only_owed_content():
    from gateway.stream_consumer import GatewayStreamConsumer

    adapter = _plain_adapter()
    consumer = GatewayStreamConsumer(adapter, CHAT)
    consumer._retain_delivery_remainder("Hello", SendResult(
        success=False, error="offline", raw_response={"remaining_content": "owed"}))
    from gateway.run import GatewayRunner
    await GatewayRunner._deliver_queued_first_response(
        _runner(), "Hello", _source(), adapter, stream_consumer=consumer,
        deliver_media=False, record_delivery=lambda receipt:
        consumer.accept_final_delivery_result("Hello", receipt))
    adapter.send.assert_awaited_once_with(CHAT, "owed", metadata=None)
    owner = consumer.delivery_handoff_for_final("Hello")
    assert owner.complete
    assert owner.result.raw_response["remaining_content"] == ""
    assert owner.result.raw_response["message_ids"] == ["p1"]
    assert owner.result.delivered_text == "owed"


@pytest.mark.asyncio
async def test_ambiguous_first_reply_is_durable_before_successful_recursive_followup(tmp_path):
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from hermes_state import SessionDB

    first, second = "First answer", "Second answer"
    url = "https://example.test/owed.gif"
    wire = []
    adapter = _telegram_adapter()

    async def transport(chat_id, content, **kwargs):
        wire.append(content)
        if len(wire) == 1:
            return SendResult(success=False, error="unknown ack", raw_response={
                "delivery_ambiguous": True, "remaining_content": first + "\n" + url})
        return SendResult(success=True, message_id="second-id", delivered_text=content)

    adapter.send = transport
    adapter.send_typing = AsyncMock()
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    consumer._stream_ledger = first
    await consumer._first_send(first, finalize=False)
    db = SessionDB(db_path=tmp_path / "chain.db")
    try:
        sid = db.create_session("chain", "test")
        db.append_message(sid, "user", "first question")
        db.append_message(sid, "assistant", first)
        first_row = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        first_result = {"final_response": first, "turn_id": "first-turn",
                        "messages": [{"role": "assistant", "content": first, "_row_id": first_row}]}
        runner = _runner()
        runner._session_db = db
        runner._MAX_INTERRUPT_DEPTH = 8
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._delivery_adapter_for = lambda source: adapter
        runner._refresh_agent_cache_message_count = AsyncMock()
        adapter.gateway_runner = runner
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key=SESSION_KEY, session_id=sid,
            source=_source(), stream_consumer_holder=[consumer],
            _status_thread_metadata={}, event_message_id=INBOUND_ID,
            inbound_message_id=INBOUND_ID, run_generation=1, _interrupt_depth=0,
            history=[], context_prompt=None, channel_prompt=None, result_holder=[first_result])

        async def successful_followup(**kwargs):
            # The first turn cannot rely on a handoff attached to the return value:
            # recursion replaces that value with the second turn's result.
            first_delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
            assert first_delivery["disposition"] == "failure"
            assert first_delivery["error"]
            assert first_delivery["message_ids"] == []
            with dl._connect() as conn:
                first_obligation = conn.execute(
                    "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
                ).fetchone()
            assert tuple(first_obligation) == (
                dl.compute_obligation_id(SESSION_KEY, INBOUND_ID, first),
                first + "\n" + url, first, "failed")
            assert wire == [first]

            db.append_message(sid, "user", "follow-up")
            db.append_message(sid, "assistant", second)
            second_row = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
            event = MessageEvent(text="follow-up", source=_source(),
                                 message_id="next-inbound", ledger_message_id="next-inbound")
            event.turn_delivery_address = {
                "session_id": sid, "row_id": second_row, "turn_id": "second-turn",
                "disposition_hint": "text"}
            receipt, _ = await adapter.send_final_ledgered(
                event, SESSION_KEY, second, {}, reply_to=None)
            await adapter._finalize_turn_delivery_record(event, [receipt])
            return {"final_response": second, "turn_id": "second-turn",
                    "messages": [{"role": "assistant", "content": second, "_row_id": second_row}],
                    "queued_terminal_inbound_id": "next-inbound"}

        runner._run_agent = successful_followup
        merged = await runner._run_agent_queued_followup(
            ctx, adapter, "follow-up", None, first_result, first_result, None)
        assert merged["final_response"] == second
        assert wire == [first, second]
        deliveries = [
            message["display_metadata"]["delivery"]
            for message in db.get_messages_as_conversation(sid) if message["role"] == "assistant"]
        assert deliveries[0]["disposition"] == "failure"
        assert deliveries[0]["error"]
        assert deliveries[0]["message_ids"] == []
        assert deliveries[1]["text"] == second
        assert deliveries[1]["message_ids"] == ["second-id"]
        with dl._connect() as conn:
            rows = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchall()
        assert {tuple(row) for row in rows} == {
            (dl.compute_obligation_id(SESSION_KEY, INBOUND_ID, first),
             first + "\n" + url, first, "failed"),
            (dl.compute_obligation_id(SESSION_KEY, "next-inbound", second),
             second, second, "delivered"),
        }
    finally:
        db.close()


@pytest.mark.asyncio
async def test_queued_raised_timeout_keeps_head_and_one_original_bracket_before_recursion(
    tmp_path,
):
    from gateway.run import GatewayRunner
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from hermes_state import AsyncSessionDB, SessionDB

    head, final = "Acknowledged head", "Acknowledged head with tail"
    tail = final[len(head):]
    wire = []
    adapter = _telegram_adapter()

    async def transport(chat_id, content, **kwargs):
        wire.append(content)
        if content == head:
            return SendResult(success=True, message_id="head-id", delivered_text=head)
        assert content == tail
        # The server accepted the tail, but no acknowledgement reached this process.
        raise TimeoutError()

    adapter.send = transport
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    assert await consumer._first_send(head, finalize=False)
    db = SessionDB(db_path=tmp_path / "queued-raised.db")
    try:
        sid = db.create_session("queued-raised", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", final)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        result = {"final_response": final, "turn_id": "queued-raised",
                  "messages": [{"role": "assistant", "content": final, "_row_id": row_id}]}
        runner = _runner()
        runner._session_db = AsyncSessionDB(db)
        runner._pop_post_delivery_callback = lambda *_args: None
        runner._delivery_adapter_for = lambda source: adapter
        adapter.gateway_runner = runner
        ctx = SimpleNamespace(
            mute_notification_reply=False, session_key=SESSION_KEY, session_id=sid,
            source=_source(), stream_consumer_holder=[consumer],
            _status_thread_metadata={}, event_message_id=INBOUND_ID,
            inbound_message_id=INBOUND_ID, run_generation=1)
        await runner._run_agent_deliver_first_response(ctx, adapter, result, result, None)
        assert wire == [head, tail]
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        assert delivery["text"] == head
        assert delivery["message_ids"] == ["head-id"]
        assert delivery["partial"] and delivery["error"]
        with dl._connect() as conn:
            original = conn.execute(
                "SELECT obligation_id,content,original_content,state,attempts,last_error "
                "FROM delivery_obligations").fetchall()
        # ``attempts`` is the redelivery budget spent by sweeps; the first in-process
        # bracket never spends it, uncertain or not.
        assert [tuple(row[:5]) for row in original] == [(
            dl.compute_obligation_id(SESSION_KEY, INBOUND_ID, final),
            tail, final, "failed", 0)]
        assert original[0][5]
        # Recursive dispatch is allowed to replace the first result, not its durable row.
        async def followup(**kwargs):
            first = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
            assert first["message_ids"] == ["head-id"] and first["error"]
            return {"final_response": "second turn"}

        runner._run_agent = followup
        runner._MAX_INTERRUPT_DEPTH = 8
        runner._refresh_agent_cache_message_count = AsyncMock()
        ctx._interrupt_depth = 0
        ctx.history = []
        ctx.context_prompt = None
        ctx.channel_prompt = None
        ctx.result_holder = [result]
        merged = await runner._run_agent_queued_followup(
            ctx, adapter, "follow-up", None, result, result, None)
        assert merged["final_response"] == "second turn"
        assert wire == [head, tail]
        with dl._connect() as conn:
            after = conn.execute(
                "SELECT obligation_id,content,original_content,state,attempts,last_error "
                "FROM delivery_obligations").fetchall()
        assert [tuple(row) for row in after] == [tuple(row) for row in original]
    finally:
        db.close()
