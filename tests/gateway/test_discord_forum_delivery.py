"""Forum starter ACKs do not authorize replay of failed follow-ups to the parent."""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from gateway import delivery_ledger as dl
from gateway.config import PlatformConfig
from gateway.delivery_handoff import delivery_handoff_from_receipt
from gateway.platforms.base import SendResult
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_followup", [False, True])
async def test_forum_followup_retains_warning_contract_without_parent_replay(
    reject_followup, tmp_path, monkeypatch,
):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.MAX_MESSAGE_LENGTH = 160
    content = "forum word " * 55
    chunks = adapter._cap_split_chunks(
        adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH))
    assert len(chunks) >= 3
    thread_sends = []

    async def send_followup(*, content):
        thread_sends.append(content)
        if reject_followup and len(thread_sends) == 1:
            raise RuntimeError("forum follow-up refused")
        return SimpleNamespace(id=8000 + len(thread_sends))

    thread = SimpleNamespace(id=777, send=AsyncMock(side_effect=send_followup))
    posts = []

    class FixtureForum(discord.ForumChannel):
        async def create_thread(self, *, name, content):
            posts.append(content)
            return SimpleNamespace(thread=thread, message=SimpleNamespace(id=8000))

    parent = object.__new__(FixtureForum)
    parent.id = 666
    adapter._client = SimpleNamespace(
        get_channel=lambda channel_id: parent if channel_id == 666 else None,
        fetch_channel=AsyncMock())
    initial = delivery_handoff_from_receipt(
        content, SendResult(success=False, raw_response={
            "model_remaining_content": content, "attachment_remaining_content": "",
            "remaining_content": content}))
    session = "agent:main:discord:channel:666"
    oid = dl.compute_obligation_id(session, "inbound", content)
    dl.begin_delivery(
        obligation_id=oid, session_key=session, platform="discord",
        chat_id="666", thread_id=None, content=content,
        original_content=content, turn_receipt=initial.to_record())
    dl.mark_attempting(oid)

    receipt = await adapter.send("666", content)
    owner = delivery_handoff_from_receipt(
        content, receipt, previous=initial, attempted_content=content, resolve_attempt=True)
    accepted = [{"message_id": "8000", "text": chunks[0]}]
    accepted.extend(
        {"message_id": str(8000 + i), "text": chunk}
        for i, chunk in enumerate(chunks[1:], start=1)
        if not reject_followup or i > 1)
    # Failed follow-ups remain warnings on the successful starter post; no
    # unacknowledged chunk is projected as a carriage or retried to the parent.
    assert thread_sends == chunks[1:]
    assert posts == [chunks[0]]
    assert receipt.success and owner.complete
    assert owner.to_record()["acknowledged_carriages"] == accepted
    assert owner.owed == ""
    assert dl.finalize_delivery(
        oid, success=owner.complete, turn_receipt=owner.to_record())
    with dl._connect() as conn:
        assert conn.execute(
            "SELECT state, chat_id, thread_id FROM delivery_obligations WHERE obligation_id=?",
            (oid,)).fetchone() == ("delivered", "666", None)
    assert dl.sweep_recoverable(now=time.time() + 60) == []
    assert dl.sweep_failed_for_runtime("discord", now=time.time() + 60) == []
    assert posts == [chunks[0]]
