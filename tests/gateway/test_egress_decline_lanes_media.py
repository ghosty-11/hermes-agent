"""Declined media and voice destinations stop later sends in the same turn."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.delivery_handoff import delivery_handoff_from_receipt
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.egress import EGRESS_DECLINE_CODE, declined_send
from gateway.session import SessionSource


CHAT = "C1"
SESSION_KEY = "agent:main:slack:channel:C1"
TEXT = "The answer"
DECLINE = SendResult(success=False, error="declined", raw_response={
    "success": False, "code": EGRESS_DECLINE_CODE,
})


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway.platforms import base
    monkeypatch.setattr(base, "_HERMES_ROOT", home)


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.calls = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("text", chat_id, content))
        return SendResult(success=True, message_id="text")

    async def send_image(self, chat_id, image_url, caption=None, reply_to=None, metadata=None):
        self.calls.append(("image", chat_id, image_url))
        return DECLINE

    async def send_voice(self, chat_id, audio_path, **kwargs):
        self.calls.append(("voice", chat_id, audio_path))
        return DECLINE


def _event():
    return MessageEvent(
        text="question", message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.SLACK, chat_id=CHAT, chat_type="channel"),
        message_id="inbound-1",
    )


@pytest.mark.asyncio
async def test_image_batch_stops_at_decline_and_retains_receipt():
    adapter = _Adapter()
    result = await adapter.send_multiple_images(CHAT, [
        ("https://example.test/first.png", "first"),
        ("https://example.test/second.png", "second"),
    ])

    assert adapter.calls == [("image", CHAT, "https://example.test/first.png")]
    assert result is DECLINE
    assert declined_send(result)


@pytest.mark.asyncio
async def test_voice_decline_merges_into_incomplete_owner_without_final_text(tmp_path):
    adapter = _Adapter()
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"voice")
    previous = delivery_handoff_from_receipt(TEXT, SendResult(
        success=False, error="stream incomplete", raw_response={"remaining_content": TEXT,
            "model_remaining_content": TEXT, "attachment_remaining_content": ""},
    ))

    async def handler(event):
        event._stream_delivery_handoff = previous
        return TEXT

    adapter.set_message_handler(handler)
    adapter._wants_auto_tts = lambda *_args: True
    adapter._synthesize_auto_tts = AsyncMock(return_value=([str(audio)], str(audio)))

    async def play(event, text, path, first, metadata, record_delivery):
        record_delivery(await adapter.send_voice(event.source.chat_id, path))
        return False

    adapter._play_tts_file = play
    event = _event()
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    await adapter._process_message_background(event, SESSION_KEY)

    assert adapter.calls == [("voice", CHAT, str(audio))]
    assert declined_send(event._stream_delivery_handoff.result)
    assert event._stream_delivery_handoff.owed == TEXT
    assert not audio.exists()


@pytest.mark.asyncio
async def test_runner_voice_decline_stops_audio_and_normal_completion(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway import run_voice
    from tools import tts_tool

    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    runner._get_guild_id = lambda event: None
    runner._reply_anchor_for_event = lambda event: None
    runner._thread_metadata_for_source = lambda source, anchor: {}
    runner._should_send_voice_reply = lambda *_args, **_kwargs: True
    event = _event()
    paths = [tmp_path / "first.ogg", tmp_path / "second.ogg"]
    monkeypatch.setattr(run_voice, "build_auto_tts_output_path", lambda platform: str(paths[0]))

    def fake_tts(*, text, output_path):
        for path in paths:
            path.write_bytes(b"voice")
        return json.dumps({"success": True, "file_paths": [str(path) for path in paths]})

    monkeypatch.setattr(tts_tool, "text_to_speech_tool", fake_tts)
    receipts = await runner._send_voice_reply(event, TEXT)
    assert adapter.calls == [("voice", CHAT, str(paths[0]))]
    assert receipts == [DECLINE]
    assert all(not path.exists() for path in paths)

    adapter.calls.clear()

    async def handler(current):
        return await runner._hmwa_deliver_turn_response(
            current, current.source, SimpleNamespace(session_id="sid"), SESSION_KEY, 1,
            {"final_response": TEXT}, [], TEXT, None, False)

    adapter.set_message_handler(handler)
    adapter._active_sessions[SESSION_KEY] = asyncio.Event()
    await adapter._process_message_background(event, SESSION_KEY)
    assert adapter.calls == [("voice", CHAT, str(paths[0]))]
    assert declined_send(event._stream_delivery_handoff.result)
    assert all(not path.exists() for path in paths)
