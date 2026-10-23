"""Regression tests for the Discord split-delivery cap (issue #86581).

A degenerate turn can produce tens of thousands of characters.  Without a
ceiling, the adapter posts every 2000-char chunk back-to-back and floods the
channel — the #86581 incident delivered 60,698 chars as 31 messages.  The
cap keeps the first ``MAX_SPLIT_MESSAGES`` chunks and replaces the remainder
with a short notice.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


MAX = DiscordAdapter.MAX_MESSAGE_LENGTH
CAP = DiscordAdapter.MAX_SPLIT_MESSAGES


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _huge_content(chars: int = 60_000) -> str:
    # Distinct filler — this test is about SIZE, not repetition.
    return " ".join(f"word-{i}-" + "x" * 12 for i in range(chars // 20))


class TestCapSplitChunks:
    def test_below_cap_unchanged(self):
        adapter = _make_adapter()
        chunks = ["a", "b", "c"]
        assert adapter._cap_split_chunks(chunks) == chunks

    def test_over_cap_keeps_n_minus_1_plus_notice(self):
        adapter = _make_adapter()
        chunks = [f"chunk-{i}-" + "z" * 100 for i in range(40)]
        capped = adapter._cap_split_chunks(chunks)
        assert len(capped) == CAP
        assert capped[0] == chunks[0]
        assert capped[-1] not in chunks  # last slot is the truncation notice
        # The notice itself must stay under Discord's per-message cap.
        assert len(capped[-1]) <= MAX


class TestSendCap:
    @pytest.mark.asyncio
    async def test_send_caps_split_flood(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        sends = []

        async def fake_send(*, content, reference=None):
            sends.append(content)
            return SimpleNamespace(id=9000 + len(sends))

        channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
        adapter._client = SimpleNamespace(
            get_channel=lambda _cid: channel,
            fetch_channel=AsyncMock(),
        )

        result = await adapter.send("555", _huge_content())

        assert result.success is True
        assert len(sends) == CAP
        assert "Response truncated" in sends[-1]


class TestForumCap:
    @pytest.mark.asyncio
    async def test_send_to_forum_caps_followup_chunks(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        thread_sends = []

        async def fake_thread_send(*, content):
            thread_sends.append(content)
            return SimpleNamespace(id=8000 + len(thread_sends))

        thread_channel = SimpleNamespace(
            id=777, send=AsyncMock(side_effect=fake_thread_send)
        )
        forum_channel = SimpleNamespace(
            id=666,
            type=SimpleNamespace(value=15),
            create_thread=AsyncMock(return_value=SimpleNamespace(
                id=777,
                thread=thread_channel,
                message=SimpleNamespace(id=8000),
            )),
        )

        result = await adapter._send_to_forum(forum_channel, _huge_content())

        assert result.success is True
        # 1 starter message + at most (CAP - 1) follow-up chunks.
        assert len(thread_sends) <= CAP - 1
        assert "Response truncated" in thread_sends[-1]


class TestEditOverflowCap:
    @pytest.mark.asyncio
    async def test_edit_overflow_split_capped(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        edits = []
        sends = []

        async def fake_edit(*, content):
            edits.append(content)

        async def fake_send(*, content, reference=None):
            sends.append(content)
            return SimpleNamespace(id=9000 + len(sends))

        msg = SimpleNamespace(id=42, edit=AsyncMock(side_effect=fake_edit))
        channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))

        result = await adapter._edit_overflow_split(channel, msg, "42", _huge_content())

        assert result.success is True
        # 1 in-place edit + at most (CAP - 1) continuation sends.
        assert len(edits) == 1
        assert len(sends) <= CAP - 1
        assert "Response truncated" in (sends[-1] if sends else edits[-1])

    @pytest.mark.asyncio
    async def test_failed_continuation_owes_exact_tail_and_reports_only_acknowledged_chunks(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        content = _huge_content(5_500)
        chunks = adapter._cap_split_chunks(
            adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH))
        msg = SimpleNamespace(id=42, edit=AsyncMock())
        sends = []

        async def fail_second_chunk(*, content, reference=None):
            sends.append(content)
            raise RuntimeError("transport unavailable")

        channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fail_second_chunk))
        result = await adapter._edit_overflow_split(channel, msg, "42", content)

        assert result.success is False
        assert result.delivered_text == chunks[0]
        assert result.raw_response["message_ids"] == ("42",)
        assert result.raw_response["remaining_content"] == "\n".join(chunks[1:])
        assert result.raw_response["partial_overflow"] is True
        assert len(sends) == 2  # anchored attempt and one bare-reference attempt

    @pytest.mark.asyncio
    async def test_timed_out_continuation_does_not_claim_the_unknown_chunk_unsent(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        msg = SimpleNamespace(id=42, edit=AsyncMock())
        content = _huge_content(5_500)
        chunks = adapter._cap_split_chunks(
            adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH))
        channel = SimpleNamespace(
            id=555, send=AsyncMock(side_effect=TimeoutError("read timed out")))
        result = await adapter._edit_overflow_split(
            channel, msg, "42", content)
        assert not result.success
        assert result.raw_response["message_ids"] == ("42",)
        # Timed-out continuation retains its exact uncertain remainder, not proof of non-delivery.
        assert result.raw_response["delivery_ambiguous"] is True
        assert result.raw_response["remaining_content"] == "\n".join(chunks[1:])
        assert channel.send.await_count == 1

    @pytest.mark.asyncio
    async def test_real_consumer_hands_off_failed_final_overflow_without_duplicate_edit(
        self, monkeypatch, tmp_path,
    ):
        from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        adapter = _make_adapter()
        content = _huge_content(5_500)
        chunks = adapter._cap_split_chunks(
            adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH))
        msg = SimpleNamespace(id=42, edit=AsyncMock())
        channel = SimpleNamespace(
            id=555, get_partial_message=lambda _id: msg,
            send=AsyncMock(side_effect=ConnectionError("transport unavailable")))
        adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
        adapter._resolve_channel = AsyncMock(return_value=channel)
        consumer = GatewayStreamConsumer(adapter, "555", StreamConsumerConfig(buffer_only=True))
        consumer._resolve_length_budget = lambda: (len, 20_000)
        consumer._message_id = "42"
        consumer._last_sent_text = "previous preview"
        consumer._already_sent = True
        consumer.on_delta(content)
        consumer.finish(content)
        await consumer.run()

        handoff = consumer.delivery_handoff_for_final(content)
        assert handoff.result.delivered_text == chunks[0]
        assert handoff.result.raw_response["remaining_content"] == "\n".join(chunks[1:])
        assert handoff.result.raw_response["message_ids"] == ["42"]
        assert not consumer.final_response_sent
        assert msg.edit.await_count == 1
        assert channel.send.await_count == 2
