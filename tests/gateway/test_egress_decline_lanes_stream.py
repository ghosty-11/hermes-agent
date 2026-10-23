"""Connector refusals retain their receipt from relay send through stream completion."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.egress import declined_send
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


CHAT = "declined-chat"
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


class _Connector:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def send_outbound(self, action, *, platform=None):
        self.calls.append((action["op"], action["chat_id"]))
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguity_key", [None, "ambiguous", "delivery_ambiguous"])
async def test_relay_send_preserves_code_only_decline_and_ambiguity(ambiguity_key):
    body = {"success": False, "code": "egress_declined", "error": "refused"}
    if ambiguity_key:
        body[ambiguity_key] = True
    connector = _Connector(body)
    descriptor = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION, platform="discord", label="Relay",
        max_message_length=4096, supports_draft_streaming=False,
        supports_edit=True, supports_threads=False, markdown_dialect="plain",
        len_unit="chars",
    )
    adapter = RelayAdapter(PlatformConfig(enabled=True, extra={}), descriptor, transport=connector)

    receipt = await adapter.send(CHAT, "answer")

    assert connector.calls == [("send", CHAT)]
    assert receipt.raw_response == body
    assert declined_send(receipt) is (ambiguity_key is None)


class _Adapter(BasePlatformAdapter):
    def __init__(self, declined_op):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.declined_op = declined_op
        self.calls = []
        self.attempted = asyncio.Event()
        self.sent = asyncio.Event()

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return None

    def supports_draft_streaming(self, **kwargs):
        return True

    def prefers_fresh_final_streaming(self, text, **kwargs):
        return self.declined_op == "fresh_final"

    def _receipt(self, op, chat_id):
        self.calls.append((op, chat_id))
        if self.declined_op == "fresh_final" and op == "send" and len(self.calls) == 2:
            self.attempted.set()
            return DECLINE
        if op == self.declined_op:
            self.attempted.set()
            return DECLINE
        if op == "send":
            self.sent.set()
        return SendResult(success=True, message_id="preview-1")

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return self._receipt("send", chat_id)

    async def send_draft(self, chat_id, draft_id, content, metadata=None):
        return self._receipt("draft", chat_id)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        return self._receipt("edit", chat_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("declined_op, expected_calls", [
    ("send", [("send", CHAT)]),
    ("draft", [("draft", CHAT)]),
    ("edit", [("send", CHAT), ("edit", CHAT)]),
])
async def test_stream_decline_retains_final_owner_without_readdressing_chat(
    declined_op, expected_calls,
):
    adapter = _Adapter(declined_op)
    config = StreamConsumerConfig(
        transport="draft" if declined_op == "draft" else "edit",
        chat_type="dm", cursor="", edit_interval=0, buffer_threshold=1,
    )
    consumer = GatewayStreamConsumer(adapter, CHAT, config)
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Preview")
    if declined_op == "edit":
        # Let the first successful send establish an editable carriage.
        await asyncio.wait_for(adapter.sent.wait(), timeout=2)
        consumer.on_delta(" and final answer")
    await asyncio.wait_for(adapter.attempted.wait(), timeout=2)
    final = "Preview and final answer" if declined_op == "edit" else "Preview"
    consumer.finish(final)
    await asyncio.wait_for(task, timeout=2)

    owner = consumer.delivery_handoff_for_final(final)
    assert owner is not None
    assert owner.to_record()["egress_declined"] is True
    assert owner.source_content == final
    assert adapter.calls == expected_calls


@pytest.mark.asyncio
async def test_fresh_final_decline_does_not_edit_after_refused_send():
    adapter = _Adapter("fresh_final")
    consumer = GatewayStreamConsumer(
        adapter, CHAT, StreamConsumerConfig(cursor="", edit_interval=0, buffer_threshold=1))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("Preview")
    await asyncio.wait_for(adapter.sent.wait(), timeout=2)
    consumer.finish("Preview and final answer")
    await asyncio.wait_for(adapter.attempted.wait(), timeout=2)
    await asyncio.wait_for(task, timeout=2)

    owner = consumer.delivery_handoff_for_final("Preview and final answer")
    assert owner is not None and owner.to_record()["egress_declined"] is True
    assert adapter.calls == [("send", CHAT), ("send", CHAT)]


@pytest.mark.asyncio
async def test_fallback_decline_is_not_retried_as_flood_control():
    class _RateDecline(_Adapter):
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            self.calls.append(("send", chat_id))
            return SendResult(
                success=False, error="rate limit", retry_after=0.01,
                raw_response={"success": False, "code": "egress_declined"})

    adapter = _RateDecline("send")
    consumer = GatewayStreamConsumer(adapter, CHAT, StreamConsumerConfig(cursor=""))
    await consumer._send_fallback_final("final answer")

    owner = consumer.delivery_handoff_for_final("final answer")
    assert owner is not None and owner.to_record()["egress_declined"] is True
    assert adapter.calls == [("send", CHAT)]
