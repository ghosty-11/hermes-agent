"""Split Telegram replies under flood control: no duplicated head, no interleaving, no hammering.

A reply past ``MAX_MESSAGE_LENGTH`` goes out as several ``sendMessage`` calls. When chunk 2 is refused
(``RetryAfter`` past the inline cap) the adapter must report the partial delivery via the existing
``partial_overflow`` contract and ``_send_with_retry`` must resume from the undelivered remainder — never
re-send chunk 1 (the reporter saw a duplicated head), never drop the tail.
"""
import asyncio
import re
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import RetryAfter

from gateway import delivery_ledger as dl
from gateway.config import PlatformConfig
from gateway.delivery_handoff import delivery_handoff_from_receipt
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


class _FloodError(Exception):
    def __init__(self, seconds: float):
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


def _adapter(send_message: AsyncMock) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = send_message
    return adapter


def _three_chunk_text() -> str:
    return "\n".join(" ".join(f"w{i * 20 + j}" for j in range(20)) for i in range(80))  # ~9.5k chars, 3 chunks


@pytest.mark.asyncio
async def test_split_send_resumes_from_undelivered_tail_without_resending_head(monkeypatch):
    """Chunk 2 refused with a 7s RetryAfter (> the 5s adapter cap, < the 60s base cap): the retry
    delivers chunks 2..3 only — every chunk exactly once, in order."""
    sent: list = []
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _FloodError(7.0)
        sent.append(text)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    async def _penalty_elapses(_delay):  # the base retry sleep is mocked; model the window having passed
        adapter._telegram_send_cooldown_until.clear()

    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", _penalty_elapses)
    content = _three_chunk_text()
    expected = adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH)
    assert len(expected) >= 3

    result = await adapter._send_with_retry(chat_id="4242", content=content)

    assert result.success is True
    assert [t.split()[0] for t in sent] == [c.split()[0] for c in expected]  # 1, 2, 3 — no duplicate head
    assert result.raw_response["message_ids"] == ["1001"] + [str(1002 + i) for i in range(1, len(expected))]


@pytest.mark.asyncio
async def test_over_cap_flood_returns_partial_overflow_and_arms_cooldown(monkeypatch):
    """A refusal past the base inline cap is returned typed (ledger owns the wait) but now carries the
    partial_overflow contract; the next send to that chat fails closed locally, another chat is unaffected."""
    calls = {"n": 0}

    async def fake_send_message(text: str, **_kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _FloodError(120.0)
        return MagicMock(message_id=1000 + calls["n"])

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", AsyncMock())

    content = _three_chunk_text()
    total = len(adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH))
    result = await adapter._send_with_retry(chat_id="4242", content=content)

    assert result.success is False and result.error == "flood_control:120.0"
    raw = result.raw_response
    assert raw["partial_overflow"] is True and raw["delivered_chunks"] == 1 and raw["total_chunks"] == total >= 3
    assert raw["last_message_id"] == "1001" and len(raw["undelivered_chunks"]) == total - 1
    assert calls["n"] == 2
    # Facet 3: a follow-up send inside the penalty window makes no API call; a different chat still sends.
    again = await adapter.send("4242", "hello again")
    assert again.success is False and again.error.startswith("flood_control:") and calls["n"] == 2
    other = await adapter.send("999", "hello other")
    assert other.success is True and calls["n"] == 3


@pytest.mark.asyncio
async def test_flood_refused_split_keeps_only_unacknowledged_bytes_in_durable_recovery(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    sent = []
    refused = True

    async def transport(*, text, **kwargs):
        nonlocal refused
        if refused and len(sent) == 1:
            refused = False
            raise RetryAfter(120)
        sent.append(text)
        return MagicMock(message_id=1000 + len(sent))

    adapter = _adapter(AsyncMock(side_effect=transport))
    content = _three_chunk_text() + "\nhttps://example.test/attachment.gif"
    chunks = adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH)
    chunks = [re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk) for chunk in chunks]
    assert len(chunks) >= 3
    oid = dl.compute_obligation_id("agent:main:telegram:chat:4242", "inbound", content)
    initial = delivery_handoff_from_receipt(
        content, SendResult(success=False, raw_response={
            "model_remaining_content": content, "attachment_remaining_content": "",
            "remaining_content": content}))
    dl.begin_delivery(
        obligation_id=oid, session_key="agent:main:telegram:chat:4242",
        platform="telegram", chat_id="4242", thread_id=None, content=content,
        original_content=content, turn_receipt=initial.to_record())
    dl.mark_attempting(oid)

    receipt = await adapter.send("4242", content, metadata={"notify": True})
    assert not receipt.success
    assert sent == [chunks[0]]
    expected_tail = "\n".join(chunks[1:])
    owner = delivery_handoff_from_receipt(
        content, receipt, previous=initial, attempted_content=content, resolve_attempt=True)
    assert [(c["message_id"], c["text"]) for c in owner.to_record()["acknowledged_carriages"]] == [
        ("1001", chunks[0])]
    assert owner.owed == expected_tail
    assert owner.to_record()["model_remaining_content"] == expected_tail
    assert owner.to_record()["attachment_remaining_content"] == ""
    assert "attachment" in owner.owed
    assert dl.finalize_delivery(
        oid, success=owner.complete, error=receipt.error or "",
        remaining_content=owner.owed, turn_receipt=owner.to_record())
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state, content FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone() == ("failed", expected_tail)
    assert dl.sweep_failed_for_runtime("telegram", now=time.time() + 60) == []
    adapter._telegram_send_cooldown_until.clear()
    claimed = dl.sweep_failed_for_runtime("telegram", now=time.time() + 122)
    assert len(claimed) == 1 and claimed[0]["content"] == expected_tail
    recovered = await adapter.send("4242", claimed[0]["content"], metadata={"notify": True})
    assert recovered.success
    assert sent[1:] and all(chunk != chunks[0] for chunk in sent[1:])
    assert "attachment" in "".join(sent[1:])
    recovered_owner = delivery_handoff_from_receipt(
        content, recovered, previous=owner, attempted_content=expected_tail, resolve_attempt=True)
    assert recovered_owner.complete
    assert dl.finalize_delivery(
        oid, success=recovered_owner.complete, turn_receipt=recovered_owner.to_record())
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone() == ("delivered",)



@pytest.mark.asyncio
async def test_formatted_split_tail_recovery_keeps_markdown_transport_bytes_once(
    tmp_path, monkeypatch,
):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session import SessionSource
    from hermes_state import AsyncSessionDB, SessionDB

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    sent = []
    refused = False

    async def transport(*, text, **_kw):
        nonlocal refused
        if len(sent) == 1 and not refused:
            refused = True
            raise RetryAfter(120)
        sent.append(text)
        return MagicMock(message_id=1000 + len(sent))

    adapter = _adapter(AsyncMock(side_effect=transport))
    content = ("*Bold* [link](https://example.test/a?q=1&b=2)!_ "
               "punctuation (literal) [brackets].\n") * 130
    db = SessionDB(db_path=tmp_path / "session.db")
    try:
        sid = db.create_session("telegram-markdown", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", content)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        event = MessageEvent(
            text="question", source=SessionSource(
                platform=Platform.TELEGRAM, chat_id="4242", chat_type="chat"),
            message_id="inbound")
        event.turn_delivery_address = {
            "session_id": sid, "row_id": row_id, "turn_id": "markdown",
            "disposition_hint": "text"}
        runner = GatewayTurnMixin()
        runner._session_db = db
        adapter.gateway_runner = runner
        key = "agent:main:telegram:chat:4242"
        result, _ = await adapter.send_final_ledgered(
            event, key, content, {"notify": True}, reply_to=None)
        await adapter._finalize_turn_delivery_record(event, [result])
        owner = event._stream_delivery_handoff
        tail = owner.result.raw_response["undelivered_chunks"]
        assert len(tail) >= 1 and "\\" in tail[0] and "[" in tail[0]
        assert sent == [owner.to_record()["acknowledged_carriages"][0]["text"]]
        with dl._connect() as conn:
            row = conn.execute(
                "SELECT content,original_content,state FROM delivery_obligations").fetchone()
        assert tuple(row) == ("\n".join(tail), content, "failed")
        adapter._telegram_send_cooldown_until.clear()
        fresh = object.__new__(GatewayRunner)
        fresh._session_db = AsyncSessionDB(db)
        fresh._authorization_adapter = lambda *_args: adapter
        fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
            side_effect=lambda rows, **_kw: rows)
        adapter.gateway_runner = fresh
        claimed = dl.sweep_failed_for_runtime(
            "telegram", now=time.time() + 122, profile="default")
        assert len(claimed) == 1
        assert await fresh._redeliver_claimed_obligations(claimed) == 1
        assert sent[1:] == [adapter.format_message(dl.FLOOD_MARKER), *tail]
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        assert delivery["text"] == "\n".join([sent[0], *tail])
        assert len(delivery["message_ids"]) == len(tail) + 1
        with dl._connect() as conn:
            row = conn.execute(
                "SELECT content,original_content,state FROM delivery_obligations").fetchone()
        assert tuple(row) == ("\n".join(tail), content, "delivered")
    finally:
        db.close()

@pytest.mark.asyncio
async def test_formatted_stream_remainder_crosses_final_bracket_without_reformatting(
    tmp_path, monkeypatch,
):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    sent = []
    attempts = []
    refused = False

    async def transport(*, text, **_kwargs):
        nonlocal refused
        attempts.append(text)
        if len(sent) == 1 and not refused:
            refused = True
            raise RetryAfter(120)
        sent.append(text)
        return MagicMock(message_id=1000 + len(sent))

    adapter = _adapter(AsyncMock(side_effect=transport))
    content = ("*Bold* [link](https://example.test/a?q=1&b=2)!_ "
               "punctuation (literal) [brackets].\n") * 130
    consumer = GatewayStreamConsumer(
        adapter, "4242", StreamConsumerConfig(cursor=""))
    consumer._stream_ledger = content
    assert not await consumer._first_send(content, finalize=False)
    handoff = consumer.delivery_handoff_for_final(content)
    frames = [
        re.sub(r" \((\d+)/(\d+)\)$", r" \\(\1/\2\\)", chunk)
        for chunk in adapter.truncate_message(
            adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH)]
    assert len(frames) >= 2 and "\\" in frames[1] and "[" in frames[1]
    assert sent == [frames[0]] and attempts == frames[:2]
    adapter._telegram_send_cooldown_until.clear()

    event = MessageEvent(
        text="question", source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="4242", chat_type="chat"),
        message_id="inbound")
    event._stream_delivery_handoff = handoff
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:telegram:chat:4242", content,
        {"notify": True}, reply_to=None)
    assert result.success
    assert sent == frames and attempts == frames[:2] + frames[1:]
    assert event._stream_delivery_handoff.owed == ""
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == ("\n".join(frames[1:]), content, "delivered")



@pytest.mark.asyncio
async def test_proven_unsent_telegram_timeouts_stay_runtime_recoverable(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    for error_text, recoverable in (
        ("connect timed out", True), ("pool timeout", True), ("timed out", False),
    ):
        adapter = _adapter(AsyncMock(side_effect=TimeoutError(error_text)))
        content = "answer " + error_text
        key = "agent:main:telegram:chat:4242"
        oid = dl.compute_obligation_id(key, content, content)
        initial = delivery_handoff_from_receipt(
            content, SendResult(success=False, raw_response={
                "remaining_content": content, "model_remaining_content": content,
                "attachment_remaining_content": ""}))
        dl.begin_delivery(
            obligation_id=oid, session_key=key, platform="telegram", chat_id="4242",
            thread_id=None, content=content, original_content=content,
            turn_receipt=initial.to_record())
        dl.mark_attempting(oid)
        receipt = await adapter._send_with_retry(
            chat_id="4242", content=content, max_retries=0)
        owner = delivery_handoff_from_receipt(
            content, receipt, previous=initial, attempted_content=content,
            resolve_attempt=True)
        assert not receipt.success and receipt.retryable == recoverable
        assert owner.ambiguous is not recoverable
        assert dl.finalize_delivery(
            oid, success=False, error=receipt.error or "", remaining_content=owner.owed,
            turn_receipt=owner.to_record())
        claimed = dl.sweep_failed_for_runtime("telegram", now=time.time() + 60)
        assert [row["obligation_id"] for row in claimed] == ([oid] if recoverable else [])
    # Retryable is not a claim that a timeout happened before the send.
    unknown = delivery_handoff_from_receipt(
        "uncertain answer",
        SendResult(success=False, error="timed out", retryable=True))
    assert unknown.ambiguous
    assert not unknown.complete


@pytest.mark.asyncio
async def test_concurrent_split_sends_to_one_chat_do_not_interleave():
    """Two 3-chunk sends racing on one chat arrive as A A A B B B, not A B A B A B."""
    order: list = []

    async def fake_send_message(text: str, **_kw):
        await asyncio.sleep(0)  # yield like a real round-trip so an unlocked loop interleaves
        order.append(text.split()[0])
        return MagicMock(message_id=len(order))

    adapter = _adapter(AsyncMock(side_effect=fake_send_message))
    long = lambda tag: "\n".join(" ".join([tag] * 30) for _ in range(90))  # noqa: E731

    await asyncio.gather(adapter.send("1", long("REPORT")), adapter.send("1", long("ALERT")))

    assert len(order) >= 4
    switches = sum(1 for a, b in zip(order, order[1:]) if a != b)
    assert switches == 1, order
