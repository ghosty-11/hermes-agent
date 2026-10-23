"""Regression tests for Discord oversized edit_message split-and-deliver.

Issue #27881 surfaced as silent truncation: ``edit_message`` clipped any
formatted payload over Discord's 2,000-char cap to ``[:1997] + "..."`` and
returned ``success=True``, so the stream consumer believed the full reply
landed and the user lost everything past the boundary.

The fix mirrors the proven Telegram contract (and its #48648 lesson):

* **Mid-stream** (``finalize=False``) — never split.  A mid-stream split moves
  the edit target to a continuation; the next accumulated-token tick re-edits
  the full text into it and re-splits, looping forever.  We truncate a
  one-message preview in place instead.
* **Final** (``finalize=True``) — split-and-deliver: edit chunk 1 in place,
  send chunks 2..N as reply-threaded continuations, return the LAST visible id
  in ``message_id`` plus every continuation in ``continuation_message_ids``.
"""

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


MAX = DiscordAdapter.MAX_MESSAGE_LENGTH  # 2000


def _make_adapter():
    return DiscordAdapter(PlatformConfig(enabled=True, token="***"))


def _wire_channel(adapter, *, original_msg, send_side_effect=None):
    """Wire a fake client whose channel returns ``original_msg`` on fetch and
    records every ``channel.send`` call."""
    sends = []

    async def fake_send(*, content, reference=None):
        sends.append({"content": content, "reference": reference})
        if send_side_effect is not None:
            res = send_side_effect(len(sends), content, reference)
            if res is not None:
                return res
        return SimpleNamespace(id=9000 + len(sends))

    channel = SimpleNamespace(
        id=555,
        get_partial_message=MagicMock(return_value=original_msg),
        send=AsyncMock(side_effect=fake_send),
    )
    adapter._client = SimpleNamespace(
        get_channel=lambda _cid: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    return channel, sends


# --------------------------------------------------------------------------- #
# Happy path — short edits unchanged
# --------------------------------------------------------------------------- #


class TestEditMessageHappyPath:
    @pytest.mark.asyncio
    async def test_short_edit_in_place(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        result = await adapter.edit_message("555", "42", "short reply")

        assert result.success is True
        assert result.message_id == "42"
        assert result.continuation_message_ids == ()
        assert edits == ["short reply"]
        assert sends == []  # no continuations for a short edit


# --------------------------------------------------------------------------- #
# Mid-stream overflow — TRUNCATE, never split (the #48648 lesson)
# --------------------------------------------------------------------------- #


class TestMidStreamOverflowTruncates:
    @pytest.mark.asyncio
    async def test_oversized_streaming_edit_truncates_in_place(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        big = "p" * 6000
        result = await adapter.edit_message("555", "42", big, finalize=False)

        # No split: the original message stays the target, no continuations.
        assert result.success is True
        assert result.message_id == "42"
        assert result.continuation_message_ids == ()
        assert sends == [], "mid-stream overflow must NOT create continuations"
        # Exactly one in-place edit, clipped to a single chunk under the cap.
        assert len(edits) == 1
        assert len(edits[0]) <= MAX
        # No literal "..." truncation marker leaks in (fence-aware truncation).
        assert not edits[0].endswith("...")


# --------------------------------------------------------------------------- #
# Saturated-preview dedup — stop flood-control edit storms (mirrors the
# Telegram #58563 fix)
# --------------------------------------------------------------------------- #


class TestSaturatedPreviewDedup:
    @pytest.mark.asyncio
    async def test_saturated_preview_dedups_repeat_oversized_edits(self):
        """Once a mid-stream preview saturates at the truncation cap, further
        oversized edits truncate to the SAME text — re-sending them is a
        visual no-op that still counts against Discord's edit rate limit
        (the exact "Telegram #48648 lesson" this file's own docstring
        already references). The adapter must skip identical saturated
        previews without an API call."""
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        # First oversized edit: delivers the truncated preview (1 API call).
        r1 = await adapter.edit_message("555", "42", "x" * 2500, finalize=False)
        assert r1.success is True
        assert len(edits) == 1

        # Stream keeps growing within the same chunk count (2500-3500 chars
        # all truncate to the same "...(1/2)" chunk-1 preview) — no API calls.
        for grow in (3000, 3500):
            r = await adapter.edit_message("555", "42", "x" * grow, finalize=False)
            assert r.success is True
            assert r.message_id == "42"
        assert len(edits) == 1, "identical saturated previews must not be re-sent"

        # Chunk-count boundary: 4000+ chars cross into "(1/3)" — a real
        # change that SHOULD be delivered.
        await adapter.edit_message("555", "42", "x" * 4000, finalize=False)
        assert len(edits) == 2
        # ...and saturates again at the new marker.
        await adapter.edit_message("555", "42", "x" * 4500, finalize=False)
        assert len(edits) == 2

        # Finalize always delivers real content, even if identical to the
        # last saturated preview (full split-and-deliver, not a dedup skip).
        result = await adapter.edit_message("555", "42", "x" * 4500, finalize=True)
        assert result.success is True
        assert len(edits) == 3


# --------------------------------------------------------------------------- #
# Final overflow — SPLIT and deliver every chunk
# --------------------------------------------------------------------------- #


class TestFinalOverflowSplits:
    @pytest.mark.asyncio
    async def test_oversized_final_edit_splits_and_delivers(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=SimpleNamespace(kind="ref")),
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        big = "q" * 6000  # ~3-4 chunks at 2000 cap
        result = await adapter.edit_message("555", "42", big, finalize=True)

        assert result.success is True
        # message_id points at the LAST visible continuation, not the original.
        assert result.continuation_message_ids, "expected continuations"
        assert result.message_id == result.continuation_message_ids[-1]
        # chunk 1 edited in place; chunks 2..N sent as new messages.
        assert len(edits) == 1
        assert len(sends) == len(result.continuation_message_ids)
        # Every delivered chunk is under the cap.
        for c in edits + [s["content"] for s in sends]:
            assert len(c) <= MAX
        # No "..." truncation marker anywhere.
        for c in edits + [s["content"] for s in sends]:
            assert not c.endswith("...")

    @pytest.mark.asyncio
    async def test_byte_coverage_preserved(self):
        adapter = _make_adapter()
        edits = []
        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=object()),
            edit=AsyncMock(side_effect=lambda *, content: edits.append(content)),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        # Distinctive marker at the very end must survive end-to-end.
        body = "a" * 5000 + "END_MARKER_XYZ"
        result = await adapter.edit_message("555", "42", body, finalize=True)

        assert result.success is True
        delivered = "".join(edits + [s["content"] for s in sends])
        assert "END_MARKER_XYZ" in delivered


# --------------------------------------------------------------------------- #
# Reactive overflow — Discord 50035 mid-edit triggers the same branch logic
# --------------------------------------------------------------------------- #


class TestReactiveOverflowDetection:
    @pytest.mark.asyncio
    async def test_50035_on_final_edit_triggers_split(self):
        adapter = _make_adapter()
        edit_calls = []

        # format_message leaves content under the cap, but the first edit
        # raises 50035 (server-side rejection); the split path then runs.
        def edit_effect(*, content):
            edit_calls.append(content)
            if len(edit_calls) == 1:
                raise RuntimeError(
                    "400 Bad Request (error code: 50035): Invalid Form Body\n"
                    "In content: Must be 2000 or fewer in length."
                )

        msg = SimpleNamespace(
            id=42,
            to_reference=MagicMock(return_value=object()),
            edit=AsyncMock(side_effect=edit_effect),
        )
        channel, sends = _wire_channel(adapter, original_msg=msg)

        # Content is UNDER the cap so pre-flight passes; the 50035 on edit
        # forces the reactive split.
        result = await adapter.edit_message("555", "42", "u" * 1500, finalize=True)

        assert result.success is True
        # Reactive split re-edited chunk 1 and may add continuations.
        assert len(edit_calls) >= 1


# --------------------------------------------------------------------------- #
# Overflow detector helper
# --------------------------------------------------------------------------- #


class TestLengthOverflowDetector:

    def test_ignores_non_length_50035(self):
        err = RuntimeError("error code: 50035: Cannot reply to a system message")
        assert DiscordAdapter._is_length_overflow_error(err) is False



class TestPartialMessageContinuationReferences:
    """When the edit target is a PartialMessage (no to_reference — the
    no-fetch edit path), overflow continuations must still thread: the
    adapter builds the reference from ids instead of silently dropping it."""

    @pytest.mark.asyncio
    async def test_continuations_threaded_with_ids_built_reference(self):
        adapter = _make_adapter()
        partial = SimpleNamespace(id=42, edit=AsyncMock())  # no to_reference
        channel, sends = _wire_channel(adapter, original_msg=partial)

        long_text = "chunk alpha " * 600  # > MAX_MESSAGE_LENGTH
        result = await adapter.edit_message("555", "42", long_text, finalize=True)

        assert result.success is True
        assert len(sends) >= 1, "overflow should send continuations"
        for call in sends:
            assert call["reference"] is not None, (
                "continuation lost its reply reference — the ids-built "
                "fallback for PartialMessage regressed")


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_timeout", (False, True))
async def test_timeout_during_real_overflow_edit_retains_uncertain_tail_in_final_bracket(
    tmp_path, monkeypatch, retry_timeout,
):
    import asyncio

    from gateway import delivery_ledger as dl
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session import SessionSource
    from hermes_state import SessionDB

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "ledger.db")
    adapter = _make_adapter()
    edited = []
    message = SimpleNamespace(
        id=42, edit=AsyncMock(side_effect=lambda *, content: edited.append(content)),
        to_reference=lambda **_kwargs: object(),
    )

    def continuation(attempt, content, reference):
        if not retry_timeout and attempt == 2:
            raise asyncio.TimeoutError("continuation ack unknown")
        if retry_timeout and attempt == 2:
            raise RuntimeError("reply anchor unavailable")
        if retry_timeout and attempt == 3:
            raise asyncio.TimeoutError("retry ack unknown")
        return None

    channel, sends = _wire_channel(adapter, original_msg=message, send_side_effect=continuation)
    final = "x" * 6100
    chunks = adapter._cap_split_chunks(adapter.truncate_message(adapter.format_message(final), MAX))
    assert len(chunks) >= 4  # one accepted continuation, one uncertain, one never attempted
    # The first continuation lands, and the later one is unknown; in the retry case
    # the same continuation is attempted once with and once without its reply anchor.
    receipt = await adapter.edit_message("555", "42", final, finalize=True)
    assert edited == [chunks[0]]
    assert [s["content"] for s in sends] == (
        [chunks[1], chunks[2], chunks[2]] if retry_timeout else [chunks[1], chunks[2]])
    assert not receipt.success
    assert receipt.delivered_text == "\n".join(chunks[:2])
    assert set(receipt.raw_response["message_ids"]) == {"42", "9001"}
    assert "9002" not in receipt.raw_response["message_ids"]
    assert receipt.raw_response["delivery_ambiguous"] is True
    owed = "\n".join(chunks[2:])
    assert receipt.raw_response["remaining_content"] == owed

    from gateway.delivery_handoff import delivery_handoff_from_receipt

    handoff = delivery_handoff_from_receipt(final, receipt)
    db = SessionDB(db_path=tmp_path / "session.db")
    try:
        sid = db.create_session("overflow", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", final)
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        runner = GatewayTurnMixin()
        runner._session_db = db
        adapter.gateway_runner = runner
        source = SessionSource(platform=Platform.DISCORD, chat_id="555", chat_type="channel")
        event = MessageEvent(text="question", source=source, message_id="inbound")
        event._stream_delivery_handoff = handoff
        event.turn_delivery_address = {
            "session_id": sid, "row_id": row_id, "turn_id": "overflow",
            "disposition_hint": "text"}
        key = "agent:main:discord:channel:555"
        result, _ = await adapter.send_final_ledgered(event, key, final, {}, reply_to=None)
        await adapter._finalize_turn_delivery_record(event, [result])
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        with dl._connect() as conn:
            obligation = conn.execute(
                "SELECT obligation_id,content,original_content,state FROM delivery_obligations"
            ).fetchone()
        assert len(sends) == (3 if retry_timeout else 2)
        assert tuple(obligation) == (
            dl.compute_obligation_id(key, "inbound", final), owed, final, "failed")
        assert delivery["text"] == "\n".join(chunks[:2])
        assert delivery["message_ids"] == ["42", "9001"]
        assert delivery["partial"] is True
        assert delivery["error"]
    finally:
        db.close()

@pytest.mark.asyncio
@pytest.mark.parametrize("model,preview,final,attempt,finalize,tail", (
    ("Hello", "Hello▉", "Hello plus", "Hello plus▉", False, " plus"),
    ("Hello", "Hello▉", "Hello", "Hello", True, ""),
    ("Hello▉", "Hello▉▉", "Hello▉ plus", "Hello▉ plus▉", False, " plus"),
    ("Hello▉", "Hello▉", "Hello▉ plus", "Hello▉ plus▉", False, " plus"),
))
async def test_refused_edit_after_cursor_preview_owes_only_model_tail(
    tmp_path, monkeypatch, model, preview, final, attempt, finalize, tail,
):
    from gateway import delivery_ledger as dl
    from gateway.delivery_handoff import StreamDeliveryHandoff
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "ledger.db")
    adapter = _make_adapter()
    edits = []

    async def refuse(*, content):
        edits.append(content)
        raise RuntimeError("edit refused")

    message = SimpleNamespace(id=9001, content=preview, edit=refuse)
    _, sends = _wire_channel(adapter, original_msg=message)
    adapter._reply_to_mode = "off"
    consumer = GatewayStreamConsumer(adapter, "555", StreamConsumerConfig(cursor="▉"))
    consumer._stream_ledger = model
    assert await consumer._first_send(preview, finalize=False)
    consumer._stream_ledger = final
    assert not await consumer._edit_existing(
        attempt, finalize=finalize, is_turn_final=finalize)
    handoff = consumer.delivery_handoff_for_final(final)
    assert handoff.owed == tail
    assert handoff.result.delivered_text == preview
    restored = StreamDeliveryHandoff.from_record(handoff.to_record())
    assert restored.owed == tail and restored.result.delivered_text == preview

    event = MessageEvent(
        text="question", source=SessionSource(
            platform=Platform.DISCORD, chat_id="555", chat_type="channel"),
        message_id="inbound")
    event._stream_delivery_handoff = handoff
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:discord:channel:555", final, {}, reply_to=None)
    assert edits[0] == attempt and all(edit == model for edit in edits[1:])
    assert [item["content"] for item in sends] == (
        [preview, tail] if tail else [preview])
    assert message.content == preview
    assert event._stream_delivery_handoff.result.delivered_text == (
        preview + "\n" + tail if tail else preview)
    assert result.success
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == (tail, final, "delivered")

@pytest.mark.asyncio
async def test_refused_edit_after_sealed_head_cursor_tail_owes_only_model_suffix(
    tmp_path, monkeypatch,
):
    from gateway import delivery_ledger as dl
    from gateway.config import Platform
    from gateway.delivery_handoff import StreamDeliveryHandoff
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "ledger.db")
    adapter = _make_adapter()
    head_text = "H" * 500
    edits = []

    async def edit_head(*, content):
        edits.append(("9001", content))
        head.content = content

    async def refuse_tail(*, content):
        edits.append(("9002", content))
        raise RuntimeError("edit refused")

    head = SimpleNamespace(id=9001, content=head_text + "▉", edit=edit_head)
    tail = SimpleNamespace(id=9002, content="Tail▉", edit=refuse_tail)
    channel, sends = _wire_channel(adapter, original_msg=head)
    channel.get_partial_message = lambda mid: {9001: head, 9002: tail}[mid]
    adapter._reply_to_mode = "off"
    consumer = GatewayStreamConsumer(adapter, "555", StreamConsumerConfig(cursor="▉"))
    consumer._len_fn = len
    consumer._safe_limit = len(head_text)
    consumer._append_accumulated(head_text)
    assert await consumer._first_send(head_text + "▉", finalize=False)

    consumer._append_accumulated("Tail")
    await consumer._seal_overflow_heads()
    assert consumer._turn_split_delivery and consumer._accumulated == "Tail"
    tick = _Tick()
    await consumer._push_update(tick)
    assert tick.update_visible
    assert sends[:2] == [
        {"content": head_text + "▉", "reference": None},
        {"content": "Tail▉", "reference": None},
    ]
    assert consumer._delivery_handoff.to_record()["cursor_carriage"] == {
        "message_id": "9002", "text": "Tail▉"}

    consumer._append_accumulated(" plus")
    final = head_text + "Tail plus"
    assert not await consumer._edit_existing(
        "Tail plus▉", finalize=False, is_turn_final=False)
    handoff = consumer.delivery_handoff_for_final(final)
    assert handoff.owed == " plus"
    assert handoff.to_record()["acknowledged_carriages"] == [
        {"message_id": "9001", "text": head_text},
        {"message_id": "9002", "text": "Tail▉"},
    ]
    restored = StreamDeliveryHandoff.from_record(handoff.to_record())
    assert restored.owed == " plus"
    assert restored.result.delivered_text == head_text + "\nTail▉"

    event = MessageEvent(
        text="question", source=SessionSource(
            platform=Platform.DISCORD, chat_id="555", chat_type="channel"),
        message_id="inbound")
    event._stream_delivery_handoff = handoff
    result, _ = await adapter.send_final_ledgered(
        event, "agent:main:discord:channel:555", final, {}, reply_to=None)
    assert edits[0] == ("9001", head_text)
    assert edits[1] == ("9002", "Tail plus▉")
    assert all(edit == ("9002", "Tail") for edit in edits[2:])
    assert [item["content"] for item in sends] == [
        head_text + "▉", "Tail▉", " plus"]
    assert head.content == head_text and tail.content == "Tail▉"
    assert event._stream_delivery_handoff.result.delivered_text == (
        head_text + "\nTail▉\n plus")
    assert result.success
    with dl._connect() as conn:
        row = conn.execute(
            "SELECT content,original_content,state FROM delivery_obligations").fetchone()
    assert tuple(row) == (" plus", final, "delivered")

@pytest.mark.asyncio
async def test_definitely_refused_edit_sends_only_tail_beside_visible_preview(
    tmp_path, monkeypatch,
):
    from gateway import delivery_ledger as dl
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run_turn import GatewayTurnMixin
    from gateway.session import SessionSource
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from hermes_state import SessionDB

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "ledger.db")
    adapter = _make_adapter()
    edits = []

    async def refuse(*, content):
        edits.append(content)
        raise RuntimeError("edit refused")

    message = SimpleNamespace(id=9001, edit=refuse)
    _, sends = _wire_channel(adapter, original_msg=message)
    adapter._reply_to_mode = "off"
    consumer = GatewayStreamConsumer(adapter, "555", StreamConsumerConfig(cursor=""))
    assert await consumer._first_send("Hello", finalize=False)
    consumer._stream_ledger = "Hello plus"
    assert not await consumer._edit_existing("Hello plus", finalize=True, is_turn_final=True)
    handoff = consumer.delivery_handoff_for_final("Hello plus")
    db = SessionDB(db_path=tmp_path / "session.db")
    try:
        sid = db.create_session("refused-edit", "test")
        db.append_message(sid, "user", "question")
        db.append_message(sid, "assistant", "Hello plus")
        row_id = db.get_messages_as_conversation(sid, include_row_ids=True)[-1]["_row_id"]
        runner = GatewayTurnMixin()
        runner._session_db = db
        adapter.gateway_runner = runner
        event = MessageEvent(
            text="question", source=SessionSource(
                platform=Platform.DISCORD, chat_id="555", chat_type="channel"),
            message_id="inbound")
        event._stream_delivery_handoff = handoff
        event.turn_delivery_address = {
            "session_id": sid, "row_id": row_id, "turn_id": "refused-edit",
            "disposition_hint": "text"}
        result, _ = await adapter.send_final_ledgered(
            event, "agent:main:discord:channel:555", "Hello plus", {}, reply_to=None)
        await adapter._finalize_turn_delivery_record(event, [result])
        delivery = db.get_messages_as_conversation(sid)[-1]["display_metadata"]["delivery"]
        with dl._connect() as conn:
            row = conn.execute(
                "SELECT content,original_content,state FROM delivery_obligations").fetchone()
        assert edits == ["Hello plus"]
        assert [s["content"] for s in sends] == ["Hello", " plus"]
        assert delivery["text"] == "Hello\n plus"
        assert delivery["message_ids"] == ["9001", "9002"]
        assert tuple(row) == (" plus", "Hello plus", "delivered")
    finally:
        db.close()
