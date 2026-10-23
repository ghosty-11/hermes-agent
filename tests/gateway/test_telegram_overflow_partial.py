"""Regression coverage for partial Telegram overflow delivery."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import telegram  # noqa: F401 - require the SDK; the adapter's optional fallback is not a producer witness

from gateway import delivery_ledger as dl
from gateway.config import PlatformConfig
from gateway.delivery_handoff import delivery_handoff_from_receipt
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


def _message(message_id: int | str) -> SimpleNamespace:
    return SimpleNamespace(message_id=message_id)


@pytest.fixture
def telegram_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = MagicMock()
    object.__setattr__(adapter, "MAX_MESSAGE_LENGTH", 160)
    return adapter


@pytest.mark.asyncio
async def test_edit_overflow_split_reports_later_partial_failure_after_some_continuations_land(telegram_adapter):
    """Partial metadata tracks the last delivered continuation before failure."""
    content = "word " * 120
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[
            _message(202),
            RuntimeError("telegram send failed"),
            RuntimeError("telegram send failed"),
        ]
    )

    result = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=False, metadata={"thread_id": "77"}
    )

    assert result.success is False
    assert result.message_id == "202"
    assert result.raw_response["partial_overflow"] is True
    assert result.raw_response["delivered_chunks"] == 2
    assert result.raw_response["last_message_id"] == "202"
    assert result.continuation_message_ids == ("202",)


@pytest.mark.asyncio
async def test_first_overflow_edit_timeout_is_uncertain_without_plain_retry(telegram_adapter):
    content = "word " * 120
    chunks = telegram_adapter.truncate_message(content, telegram_adapter.MAX_MESSAGE_LENGTH)
    telegram_adapter._bot.edit_message_text = AsyncMock(side_effect=TimeoutError())
    telegram_adapter._bot.send_message = AsyncMock()

    receipt = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=True)

    assert not receipt.success
    assert receipt.raw_response["delivery_ambiguous"] is True
    assert receipt.raw_response["model_remaining_content"] == "\n".join(chunks)
    assert receipt.raw_response["attachment_remaining_content"] == ""
    telegram_adapter._bot.edit_message_text.assert_awaited_once()
    telegram_adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_accepted_overflow_continuation_timeout_blocks_plain_fallback_and_replay(
    telegram_adapter, tmp_path, monkeypatch,
):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    content = "word " * 120
    chunks = telegram_adapter.truncate_message(content, telegram_adapter.MAX_MESSAGE_LENGTH)
    assert len(chunks) >= 3
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    accepted = []

    async def transport(*, text, **kwargs):
        accepted.append(text)
        if len(accepted) == 1:
            return _message(202)
        raise TimeoutError()

    telegram_adapter._bot.send_message = AsyncMock(side_effect=transport)
    initial = delivery_handoff_from_receipt(
        content, SendResult(success=False, raw_response={
            "model_remaining_content": content, "attachment_remaining_content": "",
            "remaining_content": content}))
    oid = dl.compute_obligation_id("agent:main:telegram:chat:12345", "inbound", content)
    dl.begin_delivery(
        obligation_id=oid, session_key="agent:main:telegram:chat:12345",
        platform="telegram", chat_id="12345", thread_id=None, content=content,
        original_content=content, turn_receipt=initial.to_record())
    dl.mark_attempting(oid)

    receipt = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=True, metadata={"thread_id": "77"})
    owner = delivery_handoff_from_receipt(
        content, receipt, previous=initial, attempted_content=content,
        edited_message_id="201", resolve_attempt=True)
    expected_tail = "\n".join(chunks[2:])
    assert len(accepted) == 2  # no second, plain-text attempt after an accepted-but-unacknowledged send
    assert [(c["message_id"], c["text"]) for c in owner.to_record()["acknowledged_carriages"]] == [
        ("201", chunks[0]), ("202", chunks[1])]
    assert owner.ambiguous and not owner.complete
    assert owner.owed == expected_tail
    assert owner.to_record()["model_remaining_content"] == expected_tail
    assert owner.to_record()["attachment_remaining_content"] == ""
    assert dl.finalize_delivery(
        oid, success=owner.complete, error=receipt.error or "",
        remaining_content=owner.owed, turn_receipt=owner.to_record())
    assert dl.sweep_failed_for_runtime("telegram", now=time.time() + 10_000) == []
    assert dl.pending_retries(now=time.time() + 10_000) == []
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
            "WHERE obligation_id=?", (oid,))
    claimed = dl.sweep_recoverable(now=time.time() + 10_000)
    assert len(claimed) == 1 and claimed[0]["adopted"] and claimed[0]["uncertain"]
    assert len(accepted) == 2


@pytest.mark.asyncio
async def test_successful_overflow_edit_acks_every_continuation(telegram_adapter):
    content = "word " * 120
    chunks = telegram_adapter.truncate_message(content, telegram_adapter.MAX_MESSAGE_LENGTH)
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    telegram_adapter._bot.send_message = AsyncMock(
        side_effect=[_message(202 + i) for i in range(len(chunks) - 1)])

    receipt = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=True, metadata={"thread_id": "77"})
    owner = delivery_handoff_from_receipt(content, receipt, attempted_content=content)

    assert receipt.success and owner.complete
    assert owner.owed == ""
    assert [item["message_id"] for item in owner.to_record()["acknowledged_carriages"]] == [
        str(201 + i) for i in range(len(chunks))]


@pytest.mark.asyncio
async def test_refused_overflow_source_tail_recovers_with_telegram_formatting_once(
    telegram_adapter, tmp_path, monkeypatch,
):
    from telegram.error import RetryAfter
    from gateway.run import GatewayRunner

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    object.__setattr__(telegram_adapter, "MAX_MESSAGE_LENGTH", 360)
    telegram_adapter._rich_send_disabled = True
    content = "*Bold* [link](https://example.test/a?q=1&b=2)!_ " * 8
    chunks = telegram_adapter.truncate_message(
        content, telegram_adapter.MAX_MESSAGE_LENGTH)
    assert len(chunks) == 2
    telegram_adapter._bot.edit_message_text = AsyncMock(return_value=True)
    accepted = []
    attempts = []
    refused = False

    async def transport(*, text, **_kwargs):
        nonlocal refused
        attempts.append(text)
        if not refused:
            refused = True
            raise RetryAfter(120)
        accepted.append(text)
        return _message(202 + len(accepted))

    telegram_adapter._bot.send_message = AsyncMock(side_effect=transport)
    receipt = await telegram_adapter._edit_overflow_split(
        "12345", "201", content, finalize=True)
    owner = delivery_handoff_from_receipt(
        content, receipt, attempted_content=content, edited_message_id="201")
    owed = "\n".join(chunks[1:])
    assert owner.owed == owed
    assert attempts == [telegram_adapter.format_message(chunks[1])]
    key = "agent:main:telegram:chat:12345"
    oid = dl.compute_obligation_id(key, "inbound", content)
    dl.begin_delivery(
        obligation_id=oid, session_key=key, platform="telegram", chat_id="12345",
        thread_id=None, content=owed, original_content=content,
        turn_receipt=owner.to_record())
    dl.mark_attempting(oid)
    assert dl.finalize_delivery(
        oid, success=False, error=receipt.error or "",
        remaining_content=owner.owed, turn_receipt=owner.to_record())
    # The overflow refusal may not open a flood window; clear one only if present.
    telegram_adapter.__dict__.pop("_telegram_send_cooldown_until", None)
    fresh = object.__new__(GatewayRunner)
    fresh._authorization_adapter = lambda *_args: telegram_adapter
    fresh._clear_resume_pending_for_claimed_obligations = AsyncMock(
        side_effect=lambda rows, **_kw: rows)
    claimed = dl.sweep_failed_for_runtime(
        "telegram", now=time.time() + 122, profile="default")
    assert len(claimed) == 1
    assert claimed[0]["needs_marker"]
    assert await fresh._redeliver_claimed_obligations(claimed) == 1
    expected = telegram_adapter.format_message(claimed[0]["marker"] + owed)
    assert accepted == [expected]
    assert attempts == [telegram_adapter.format_message(chunks[1]), expected]
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == (owed, content, "delivered")
