"""Progress egress refusal must terminate publication and final delivery for its turn."""

from __future__ import annotations

from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource


CHAT = "refused-chat"
DECLINE = SendResult(success=False, error="refused", raw_response={
    "success": False, "code": "egress_declined",
})


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway.platforms import base as platform_base
    monkeypatch.setattr(platform_base, "_HERMES_ROOT", home)


class _Adapter(BasePlatformAdapter):
    def __init__(self, *, edit_result=DECLINE, card_result=None, send_result=None):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.calls = []
        self.edit_result = edit_result
        self.card_result = card_result
        self.send_result = send_result or SendResult(success=True, message_id="bubble-1")

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.calls.append(("send", chat_id, content))
        return self.send_result

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.calls.append(("edit", chat_id, content))
        return self.edit_result

    async def send_native_task_card_progress(self, **kwargs):
        self.calls.append(("card", kwargs["chat_id"], kwargs["fallback_text"]))
        return self.card_result



def _turn(adapter):
    ctx = SimpleNamespace(
        source=SessionSource(platform=Platform.SLACK, chat_id=CHAT, chat_type="channel"),
        _progress_reply_to=None, _progress_metadata={}, _cleanup_progress=False,
        stream_consumer_holder=[None], session_key="agent:main:slack:channel:refused-chat",
        _run_still_current=lambda: True, progress_grouping="accumulate",
    )
    runner = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda source: adapter), ctx)
    return runner, ctx


@pytest.mark.asyncio
async def test_declined_progress_edit_does_not_fall_through_or_publish_later():
    adapter = _Adapter()
    runner, _ = _turn(adapter)
    state = runner._progress_edit_state(adapter)
    state.progress_msg_id = "bubble-1"
    state.progress_lines = ["first tool", "second tool"]

    await runner._progress_send_or_edit(state, "second tool")
    await runner._progress_send_or_edit(state, "third tool")

    assert adapter.calls == [("edit", CHAT, "first tool\nsecond tool")]


@pytest.mark.asyncio
async def test_declined_native_card_fallback_send_blocks_later_events():
    adapter = _Adapter(card_result=SendResult(success=False, error="card unavailable"), send_result=DECLINE)
    runner, _ = _turn(adapter)
    state = runner._TaskCardState(adapter)
    state.apply_event({"type": "tool.started", "tool_call_id": "one", "tool_name": "terminal"})
    await runner._task_card_publish(state)
    state.apply_event({"type": "tool.started", "tool_call_id": "two", "tool_name": "browser"})
    await runner._task_card_publish(state)

    assert [op for op, chat, _ in adapter.calls] == ["card", "send"]
    assert all(chat == CHAT for _, chat, _ in adapter.calls)


@pytest.mark.asyncio
async def test_progress_decline_survives_final_handoff_without_final_text_send():
    from gateway.delivery_handoff import delivery_handoff_from_receipt
    from gateway.run import GatewayRunner

    adapter = _Adapter()
    progress, ctx = _turn(adapter)
    ctx.progress_queue = queue.Queue()
    state = progress._progress_edit_state(adapter)
    state.progress_lines.append("tool started")  # the producer accumulates before sending
    await progress._progress_send_or_edit(state, "tool started")
    state.progress_lines.append("tool finished")
    await progress._drain_progress_on_cancel(state)
    expected = [("send", CHAT, "tool started"), ("edit", CHAT, "tool started\ntool finished")]
    assert adapter.calls == expected

    # An unfinished stream owner offered after progress cleanup cannot replace the refusal.
    unfinished = delivery_handoff_from_receipt(
        "private final answer", SendResult(success=False, error="stream unavailable"))
    ctx.stream_consumer_holder[0] = SimpleNamespace(
        delivery_handoff_for_final=lambda text: unfinished)
    response = {"final_response": "private final answer", "messages": []}
    gateway = object.__new__(GatewayRunner)
    await gateway._run_agent_mark_streamed_delivery(response, ctx)
    gateway._delivery_adapter_for = lambda source: adapter
    adapter.gateway_runner = gateway
    gateway._should_send_voice_reply = lambda *args, **kwargs: False
    event = MessageEvent(
        text="question", message_type=MessageType.TEXT, source=ctx.source, message_id="inbound-1")
    await gateway._hmwa_deliver_turn_response(
        event, ctx.source, SimpleNamespace(session_id="sid"), ctx.session_key, 1,
        response, [], response["final_response"], None, False,
    )
    await adapter.send_final_ledgered(
        event, ctx.session_key, response["final_response"], {}, reply_to=None)

    assert adapter.calls == expected
