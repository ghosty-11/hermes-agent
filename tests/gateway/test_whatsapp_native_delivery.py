import time
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import PlatformConfig
from gateway.delivery_handoff import delivery_handoff_from_receipt
from gateway.platforms.base import SendResult
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


class TestWhatsAppNativeFormatting:

    def test_invisible_unicode_prefixes_are_sanitized(self):
        adapter = _make_adapter()

        assert adapter.format_message("\u2060\u202ftext") == " text"


@pytest.mark.asyncio
async def test_send_location_posts_to_bridge_location_endpoint():
    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "loc-msg"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = await adapter.send_location(
        "15551234567",
        41.015,
        28.979,
        name="HQ",
        address="Example Street",
    )

    assert result.success
    assert result.message_id == "loc-msg"
    call = adapter._http_session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/send-location"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "latitude": 41.015,
        "longitude": 28.979,
        "name": "HQ",
        "address": "Example Street",
    }




@pytest.mark.asyncio
async def test_accepted_message_less_bridge_timeout_keeps_head_and_blocks_replay(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    adapter = _make_adapter()
    adapter._outgoing_chunk_limit = lambda: 160
    content = "safe word " * 55
    chunks = adapter.truncate_message(adapter.format_message(content), 160)
    assert len(chunks) >= 3
    accepted = []
    ack = MagicMock(status=200)
    ack.json = AsyncMock(return_value={"success": True, "messageId": "head-1"})

    class AcceptedTimeout:
        async def __aenter__(self):
            raise TimeoutError()

        async def __aexit__(self, *exc):
            return False

    def post(url, *, json, **kwargs):
        accepted.append(json["message"])
        return _AsyncCM(ack) if len(accepted) == 1 else AcceptedTimeout()

    adapter._http_session.post = MagicMock(side_effect=post)
    initial = delivery_handoff_from_receipt(
        content, SendResult(success=False, raw_response={
            "model_remaining_content": content, "attachment_remaining_content": "",
            "remaining_content": content}))
    oid = dl.compute_obligation_id("agent:main:whatsapp:chat:15551234567", "inbound", content)
    dl.begin_delivery(
        obligation_id=oid, session_key="agent:main:whatsapp:chat:15551234567",
        platform="whatsapp", chat_id="15551234567", thread_id=None, content=content,
        original_content=content, turn_receipt=initial.to_record())
    dl.mark_attempting(oid)

    receipt = await adapter.send("15551234567", content)
    owner = delivery_handoff_from_receipt(
        content, receipt, previous=initial, attempted_content=content, resolve_attempt=True)
    expected_tail = "\n".join(chunks[1:])
    assert accepted == chunks[:2]
    assert owner.to_record()["acknowledged_carriages"] == [
        {"message_id": "head-1", "text": chunks[0]}]
    assert owner.ambiguous and not owner.complete
    assert owner.owed == expected_tail
    assert owner.to_record()["model_remaining_content"] == expected_tail
    assert owner.to_record()["attachment_remaining_content"] == ""
    assert dl.finalize_delivery(
        oid, success=owner.complete, error=receipt.error or "",
        remaining_content=owner.owed, turn_receipt=owner.to_record())
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state, content FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone() == ("failed", expected_tail)
    assert dl.sweep_failed_for_runtime("whatsapp", now=time.time() + 10_000) == []
    assert dl.pending_retries(now=time.time() + 10_000) == []
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
            "WHERE obligation_id=?", (oid,))
    claimed = dl.sweep_recoverable(now=time.time() + 10_000)
    assert len(claimed) == 1 and claimed[0]["adopted"] and claimed[0]["uncertain"]
    assert accepted == chunks[:2]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refused", "uncertain"])
async def test_bridge_partial_failure_preserves_underlying_delivery_decision(
    failure, tmp_path, monkeypatch,
):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    adapter = _make_adapter()
    adapter._outgoing_chunk_limit = lambda: 160
    content = "safe word " * 55
    chunks = adapter.truncate_message(adapter.format_message(content), 160)
    accepted = []
    ack = MagicMock(status=200)
    ack.json = AsyncMock(return_value={"success": True, "messageId": "head-1"})
    denied = MagicMock(status=503)
    denied.text = AsyncMock(return_value="bridge refused")

    def post(url, *, json, **kwargs):
        accepted.append(json["message"])
        return _AsyncCM(ack if len(accepted) == 1 else denied)

    adapter._http_session.post = MagicMock(side_effect=post)
    if failure == "uncertain":
        real_post = adapter._post_bridge_message

        async def underlying(path, payload, *, timeout):
            result = await real_post(path, payload, timeout=timeout)
            if not result.success:
                return replace(result, raw_response={"delivery_ambiguous": True})
            return result

        adapter._post_bridge_message = underlying

    initial = delivery_handoff_from_receipt(
        content, SendResult(success=False, raw_response={
            "model_remaining_content": content, "attachment_remaining_content": "",
            "remaining_content": content}))
    oid = dl.compute_obligation_id("agent:main:whatsapp:chat:15551234567", "inbound", content)
    dl.begin_delivery(
        obligation_id=oid, session_key="agent:main:whatsapp:chat:15551234567",
        platform="whatsapp", chat_id="15551234567", thread_id=None, content=content,
        original_content=content, turn_receipt=initial.to_record())
    dl.mark_attempting(oid)
    receipt = await adapter.send("15551234567", content)
    owner = delivery_handoff_from_receipt(
        content, receipt, previous=initial, attempted_content=content, resolve_attempt=True)
    assert accepted == chunks[:2]
    assert owner.to_record()["acknowledged_carriages"] == [
        {"message_id": "head-1", "text": chunks[0]}]
    assert owner.owed == "\n".join(chunks[1:])
    assert owner.to_record()["model_remaining_content"] == owner.owed
    assert owner.to_record()["attachment_remaining_content"] == ""
    assert owner.ambiguous is (failure == "uncertain")
    assert dl.finalize_delivery(
        oid, success=False, error=receipt.error or "",
        remaining_content=owner.owed, turn_receipt=owner.to_record())
    if failure == "refused":
        claimed = dl.sweep_failed_for_runtime("whatsapp", now=time.time() + 31)
        assert len(claimed) == 1 and claimed[0]["content"] == owner.owed
    else:
        assert dl.sweep_failed_for_runtime("whatsapp", now=time.time() + 10_000) == []
        assert dl.pending_retries(now=time.time() + 10_000) == []
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
                "WHERE obligation_id=?", (oid,))
        claimed = dl.sweep_recoverable(now=time.time() + 10_000)
        assert len(claimed) == 1 and claimed[0]["adopted"] and claimed[0]["uncertain"]
        assert accepted == chunks[:2]


@pytest.mark.asyncio
async def test_bridge_success_acknowledges_all_split_chunks():
    adapter = _make_adapter()
    adapter._outgoing_chunk_limit = lambda: 160
    content = "safe word " * 55
    chunks = adapter.truncate_message(adapter.format_message(content), 160)
    posted = []

    def post(url, *, json, **kwargs):
        posted.append(json["message"])
        response = MagicMock(status=200)
        response.json = AsyncMock(return_value={
            "success": True, "messageId": f"chunk-{len(posted)}"})
        return _AsyncCM(response)

    adapter._http_session.post = MagicMock(side_effect=post)
    receipt = await adapter.send("15551234567", content)
    owner = delivery_handoff_from_receipt(content, receipt, attempted_content=content)
    assert posted == chunks
    assert owner.complete
    assert owner.to_record()["acknowledged_carriages"] == [
        {"message_id": f"chunk-{i + 1}", "text": chunk}
        for i, chunk in enumerate(chunks)]
