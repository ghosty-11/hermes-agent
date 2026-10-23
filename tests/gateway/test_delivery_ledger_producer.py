"""Producer-hook tests: _process_message_background records delivery
obligations around the final send (gateway/platforms/base.py).

Contract: obligation recorded (pending→attempting) BEFORE the send await,
delivered/failed by SendResult afterward; slash commands, ephemeral
replies, and empty responses are never recorded; ledger failures never
block the send.
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


class _Adapter(BasePlatformAdapter):  # type: ignore[misc]
    """Minimal concrete adapter driving the real base-class pipeline."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.SLACK)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover
        return True

    async def disconnect(self):  # pragma: no cover - unused
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - unused
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="m1")


def _event(text="hello agent"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK, chat_id="C1", chat_type="channel"
        ),
        message_id="msg-42",
    )


def _rows():
    with dl._connect() as conn:
        return conn.execute(
            """SELECT obligation_id, state, content, adapter_profile
               FROM delivery_obligations"""
        ).fetchall()


def _blocking_probe():
    """Return a blocking ledger call and an event-loop progress witness."""
    ledger_started = threading.Event()
    event_loop_progressed = threading.Event()
    blocked_event_loop = []

    def _slow_ledger_call(*args, **kwargs):
        ledger_started.set()
        # Generous timeout: a genuinely blocked loop can never set the event
        # (the witness coroutine cannot run), so a longer wait only guards
        # against loaded-CI scheduling flake, not against missing the bug.
        if not event_loop_progressed.wait(timeout=5.0):
            blocked_event_loop.append(True)

    async def _event_loop_witness():
        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


async def _run(adapter, event, response="final answer"):
    adapter._message_handler = AsyncMock(return_value=response)
    session_key = "agent:main:slack:channel:C1"
    adapter._active_sessions[session_key] = asyncio.Event()
    await adapter._process_message_background(event, session_key)


class TestProducerHook:
    @pytest.mark.asyncio
    async def test_normal_turn_records_and_delivers(self):
        adapter = _Adapter()
        await _run(adapter, _event())

        assert adapter.sent == ["final answer"]
        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "delivered"
        assert rows[0][2] == "final answer"

    @pytest.mark.asyncio
    async def test_send_failure_leaves_failed_row(self):
        adapter = _Adapter()
        adapter.send = AsyncMock(
            return_value=SendResult(success=False, error="chat_not_found")
        )
        await _run(adapter, _event())

        rows = _rows()
        assert len(rows) == 1
        assert rows[0][1] == "failed"

    @pytest.mark.asyncio
    async def test_partial_final_and_partial_replay_send_only_unlanded_content(self):
        from gateway.run_startup import GatewayStartupMixin

        adapter = _Adapter()
        runner = MagicMock()
        runner._delivery_adapter_for.return_value = adapter
        runner._obligation_adapter = AsyncMock(return_value=adapter)
        runner._arm_flood_timers_for_waiting_rows = AsyncMock()
        adapter.gateway_runner = runner
        sent = []
        outcomes = iter([
            SendResult(success=False, error="partially sent", retryable=True,
                       raw_response={"partial_overflow": True, "remaining_content": "link A\nlink B",
                                     "delivery_error": "send_path_degraded"}),
            SendResult(success=False, error="partially sent", retryable=True,
                       raw_response={"partial_overflow": True, "remaining_content": "link B",
                                     "delivery_error": "send_path_degraded"}),
            SendResult(success=True, message_id="delivered"),
        ])

        async def transport(chat_id, content, reply_to=None, metadata=None):
            sent.append(content)
            return next(outcomes)

        adapter.send = transport
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result, _ = await adapter.send_final_ledgered(
                _event(), "agent:main:slack:channel:C1", "landed prose\nlink A\nlink B", {},
                reply_to=None)
        assert result.error == "partially sent"
        oid, state, content, profile = _rows()[0]
        assert (state, content, profile) == ("failed", "link A\nlink B", "default")
        assert dl.pending_retries() == []

        for expected in ("link A\nlink B", "link B"):
            claimed = dl.sweep_failed_for_runtime("slack")
            assert len(claimed) == 1
            assert claimed[0]["obligation_id"] == oid
            assert claimed[0]["content"] == expected
            assert await GatewayStartupMixin._redeliver_claimed_obligations(runner, claimed) == (
                0 if expected == "link A\nlink B" else 1)
            if expected == "link A\nlink B":
                assert _rows() == [(oid, "failed", "link B", "default")]
                assert dl.pending_retries() == []

        assert len(sent) == 3
        assert sent[0] == "landed prose\nlink A\nlink B"
        assert sent[1].endswith("link A\nlink B") and "landed prose" not in sent[1]
        assert sent[2].endswith("link B") and "link A" not in sent[2]
        assert _rows() == [(oid, "delivered", "link B", "default")]

    @pytest.mark.asyncio
    async def test_failed_turn_marker_cleanup_does_not_restore_narrowed_reply_on_startup(self, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.run import GatewayRunner
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        if store._db is not None:
            store._db.close()
        store._db = SessionDB(db_path=dl._db_path())
        runner = object.__new__(GatewayRunner)
        runner.session_store = store
        adapter = _Adapter()
        adapter.gateway_runner = runner
        source_event = _event()
        entry = store.get_or_create_session(source_event.source)
        token = store.mark_turn_active(entry.session_key)
        original = "already visible prose\nhttps://example.com/owed.gif"
        remainder = "https://example.com/owed.gif"
        store.append_to_transcript(entry.session_id, {"role": "user", "content": "show the GIF"})
        store.append_to_transcript(entry.session_id, {"role": "assistant", "content": original})
        source_event._turn_marker_handoff = True
        source_event._gateway_active_turn_session_key = entry.session_key
        source_event._gateway_active_turn_token = token
        adapter.send = AsyncMock(return_value=SendResult(
            success=False, error="partially sent", retry_after=120,
            raw_response={"partial_overflow": True, "remaining_content": remainder,
                          "delivery_error": "flood_control:120"}))
        try:
            with (
                patch.object(runner.async_session_store, "clear_turn_active",
                             new_callable=AsyncMock, side_effect=OSError("state store unavailable")) as clear,
                patch.object(dl, "_owner_stamp", return_value=(999999999, 1)),
            ):
                result, _ = await adapter.send_final_ledgered(
                    source_event, entry.session_key, original, {}, reply_to=None)
            assert not result.success
            assert clear.await_count == 3
            assert store.get_or_create_session(source_event.source).active_turn_token == token
            assert [(state, content) for _, state, content, _ in _rows()] == [
                ("failed", remainder)]

            assert await runner._recover_unclean_sessions() == (0, 1)
            assert store.get_or_create_session(source_event.source).active_turn_token is None
            assert [(state, content) for _, state, content, _ in _rows()] == [
                ("failed", remainder)]
            claimed = dl.sweep_recoverable(now=time.time() + 121,
                                           deliverable_platforms={"slack"})
            assert len(claimed) == 1
            assert claimed[0]["content"] == remainder
            assert claimed[0]["needs_marker"] is True
        finally:
            store._db.close()

    @pytest.mark.asyncio
    async def test_total_failure_with_attached_remainder_survives_exhaustion_and_startup(self):
        from gateway.run_startup import GatewayStartupMixin

        original = "visible only if delivered"
        owed = original + "\nhttps://example.com/attached.gif"
        adapter = _Adapter()
        runner = MagicMock()
        runner._obligation_adapter = AsyncMock(return_value=adapter)
        runner._arm_flood_timers_for_waiting_rows = AsyncMock()
        adapter.gateway_runner = runner
        sent = []
        failed = SendResult(
            success=False, error="transport failed", retryable=True,
            raw_response={"partial_overflow": False, "remaining_content": owed,
                          "delivery_error": "flood_control:2"})
        outcomes = iter([failed, failed, failed, SendResult(success=True, message_id="landed")])

        async def transport(chat_id, content, reply_to=None, metadata=None):
            sent.append(content)
            return next(outcomes)

        adapter.send = transport
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result, _ = await adapter.send_final_ledgered(
                _event(), "agent:main:slack:channel:C1", original, {}, reply_to=None)
        assert not result.success
        assert sent[:3] == [original, owed, owed]
        oid, state, content, _ = _rows()[0]
        assert (state, content) == ("failed", owed)

        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
                "WHERE obligation_id=?", (oid,))
        claimed = dl.sweep_recoverable(now=time.time() + 3)
        assert len(claimed) == 1
        assert claimed[0]["content"] == owed
        assert await GatewayStartupMixin._redeliver_claimed_obligations(runner, claimed) == 1
        assert sent[3] == dl.FLOOD_MARKER + owed
        assert _rows() == [(oid, "delivered", owed, "default")]

    @pytest.mark.asyncio
    async def test_whole_flood_failure_keeps_full_content_and_wait(self):
        adapter = _Adapter()
        adapter.send = AsyncMock(return_value=SendResult(
            success=False, error="delivery failed", retry_after=120.0,
            raw_response={"delivery_error": "flood_control:120"}))
        result, _ = await adapter.send_final_ledgered(
            _event(), "agent:main:slack:channel:C1", "full reply", {}, reply_to=None)

        assert result.error == "delivery failed"
        oid, state, content, _ = _rows()[0]
        assert state == "failed" and content == "full reply"
        with dl._connect() as conn:
            error, failed_at = conn.execute(
                "SELECT last_error, updated_at FROM delivery_obligations WHERE obligation_id=?",
                (oid,)).fetchone()
        assert error == "flood_control:120"
        assert dl.sweep_failed_for_runtime("slack", now=failed_at + 119) == []
        assert [r["content"] for r in dl.sweep_failed_for_runtime(
            "slack", now=failed_at + 123)] == ["full reply"]

    @pytest.mark.asyncio
    async def test_late_transient_failure_signals_reconnected_runner(self):
        """A replacement installed mid-send must trigger another ledger sweep."""
        adapter = _Adapter()
        adapter._owner_profile = "reviewer"
        replacement = _Adapter()
        replacement._owner_profile = "reviewer"
        runner = MagicMock()
        runner._delivery_adapter_for.side_effect = [adapter, replacement]
        runner._redeliver_failed_obligations_for_platform = AsyncMock(return_value=1)
        adapter.gateway_runner = runner
        adapter.send = AsyncMock(
            return_value=SendResult(
                success=False,
                error="send_path_degraded",
                retryable=True,
            )
        )

        await _run(adapter, _event())

        assert _rows()[0][1] == "failed"
        assert _rows()[0][3] == "reviewer"
        runner._redeliver_failed_obligations_for_platform.assert_awaited_once_with(
            Platform.SLACK, profile="reviewer"
        )


    @pytest.mark.asyncio
    async def test_slow_ledger_record_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_record, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch(
            "gateway.delivery_ledger.record_obligation",
            side_effect=slow_record,
        ), patch("gateway.delivery_ledger.mark_attempting"):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_slow_ledger_update_does_not_block_event_loop(self):
        adapter = _Adapter()
        slow_delivered, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch("gateway.delivery_ledger.record_obligation"), patch(
            "gateway.delivery_ledger.mark_attempting"
        ), patch(
            "gateway.delivery_ledger.finalize_delivery",
            side_effect=slow_delivered,
        ):
            await asyncio.gather(_run(adapter, _event()), event_loop_witness())

        assert blocked_event_loop == []
        assert adapter.sent == ["final answer"]

    @pytest.mark.asyncio
    async def test_crash_between_attempting_and_ack_is_recoverable(self):
        """The core scenario (#58818): process dies mid-send. The row must
        be claimable by a later process and carry the ambiguity marker."""
        adapter = _Adapter()

        async def _dies_mid_send(chat_id, content, reply_to=None, metadata=None):
            raise ConnectionError("gateway killed mid-await")

        adapter.send = _dies_mid_send
        # _send_with_retry raising propagates; the background task catches
        # broadly — drive only through the send block by tolerating the error.
        try:
            await _run(adapter, _event())
        except Exception:
            pass

        rows = _rows()
        assert len(rows) == 1
        # Row is stuck in 'attempting' (or failed if retry wrapper caught it):
        # either way it is non-delivered and recoverable.
        assert rows[0][1] in ("attempting", "failed")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1"
            )
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is True
