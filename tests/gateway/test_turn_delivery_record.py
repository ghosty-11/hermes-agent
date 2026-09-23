"""Turn-delivery accounting across the gateway/adapter boundary (companion
response contract).

Contracts:

- ``GatewayTurnMixin._hmwa_turn_delivery_address`` binds the actual-delivery record to the
  final assistant ROW id + the exported turn id + session — never to "the newest row that
  happens to match some text". Missing any leg yields None (no record, replay shows an
  unlabelled — unknown — delivery, never success).
- ``BasePlatformAdapter._turn_delivery_outcome`` classifies from the ACTUAL receipts:
  successful text send -> text with the post-screening projection; withheld send
  (success, empty ``delivered_text``, no message id) -> no carriage, reason surfaced;
  media-only -> media; refused send -> failure with error; streamed turn -> text with the
- The adapter finalizer hands the outcome to the gateway's ``record_turn_delivery``, which
  writes it through ``SessionDB.record_message_delivery`` on the event's stamped address.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn import GatewayTurnMixin


class _Adapter(BasePlatformAdapter):
    """Concrete stub so the mixin's delivery helpers can be exercised directly."""

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="m", delivered_text=content)

    async def get_chat_info(self, chat_id):
        return {"name": chat_id, "type": "dm"}


class _Runner(GatewayTurnMixin):
    def __init__(self, session_db=None):
        self._session_db = session_db


class TestTurnDeliveryAddress:
    def _result(self, *, turn_id="turn-9", row_id=77, extra=None):
        messages = [
            {"role": "user", "content": "q", "_row_id": 76},
            {"role": "assistant", "content": "a", "_row_id": row_id},
        ]
        result = {"turn_id": turn_id, "messages": messages}
        if extra:
            result.update(extra)
        return result

    def test_binds_session_row_and_turn(self):
        runner = _Runner()
        addr = runner._hmwa_turn_delivery_address(
            self._result(), session_entry=SimpleNamespace(session_id="sess-1"),
            disposition_hint="text")
        assert addr == {"session_id": "sess-1", "row_id": 77, "turn_id": "turn-9",
                        "disposition_hint": "text", "streamed_text": None}

    def test_picks_the_last_assistant_row_not_the_newest_any_row(self):
        runner = _Runner()
        result = self._result()
        result["messages"].append({"role": "user", "content": "follow-up", "_row_id": 90})
        addr = runner._hmwa_turn_delivery_address(
            result, session_entry=SimpleNamespace(session_id="sess-1"),
            disposition_hint="text")
        assert addr["row_id"] == 77

    def test_missing_turn_id_row_id_or_session_yields_none(self):
        runner = _Runner()
        entry = SimpleNamespace(session_id="sess-1")
        assert runner._hmwa_turn_delivery_address(
            self._result(turn_id=""), session_entry=entry, disposition_hint="text") is None
        assert runner._hmwa_turn_delivery_address(
            self._result(extra={"turn_id": "turn-9", "messages": [
                {"role": "user", "content": "q", "_row_id": 76}]}),
            session_entry=entry, disposition_hint="text") is None
        assert runner._hmwa_turn_delivery_address(
            self._result(), session_entry=SimpleNamespace(session_id=""),
            disposition_hint="text") is None


class TestTurnDeliveryOutcome:
    def test_successful_text_send_records_post_screening_projection(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [SendResult(success=True, message_id="m-1", delivered_text="the actual text")],
            hint="text", streamed_text=None)
        assert outcome["disposition"] == "text"
        assert outcome["text"] == "the actual text"
        assert outcome["message_ids"] == ["m-1"]

    def test_withheld_send_is_no_carriage_with_reason_surfaced(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [SendResult(success=True, message_id=None, delivered_text="",
                        error="screened_context_echo: private payload withheld")],
            hint="text", streamed_text=None)
        assert outcome["disposition"] == "unknown"
        assert outcome.get("text") in (None, "")
        assert outcome["reason"] == "screened_context_echo: private payload withheld"

    def test_media_only_send_records_media(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [SendResult(success=True, message_id="m-2")],  # media lane: no text projection
            hint="text", streamed_text=None)
        assert outcome["disposition"] == "media"
        assert outcome["message_ids"] == ["m-2"]

    def test_failed_send_records_failure_with_error(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [SendResult(success=False, error="HTTP 429", error_kind="rate_limited")],
            hint="text", streamed_text=None)
        assert outcome["disposition"] == "failure"
        assert outcome["error"] == "HTTP 429"

    def test_intentional_silence_hint_with_no_receipts(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome([], hint="silence", streamed_text=None)
        assert outcome["disposition"] == "silence"

    def test_streamed_turn_records_the_streamed_projection(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [], hint="streamed", streamed_text="streamed body")
        assert outcome["disposition"] == "text"
        assert outcome["text"] == "streamed body"

    def test_no_receipt_and_no_hint_is_unknown_never_success(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome([], hint="text", streamed_text=None)
        assert outcome["disposition"] == "unknown"

    def test_successful_media_does_not_mask_a_failed_text_send(self):
        """Review finding: a delivered attachment must not make a dropped final text read
        as a clean success — the record carries partial=True and the failed lane's error."""
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [SendResult(success=False, error="HTTP 429: flood control"),
             SendResult(success=True, message_id="m-2")],
            hint="text", streamed_text=None)
        assert outcome["disposition"] == "media"
        assert outcome.get("partial") is True
        assert outcome["error"] == "HTTP 429: flood control"

    def test_partial_failure_flag_survives_text_success(self):
        outcome = BasePlatformAdapter._turn_delivery_outcome(
            [SendResult(success=True, message_id="m-1", delivered_text="sent anyway"),
             SendResult(success=False, error="edit overflow lost the tail")],
            hint="text", streamed_text=None)
        assert outcome["disposition"] == "text"
        assert outcome.get("partial") is True
        assert outcome["error"] == "edit overflow lost the tail"

class TestGatewayRecorder:
    def test_record_turn_delivery_writes_through_session_db(self):
        recorded = []

        class _DB:
            def record_message_delivery(self, *args):
                recorded.append(args)
                return True

        runner = _Runner(session_db=_DB())
        addr = {"session_id": "s", "row_id": 5, "turn_id": "t", "disposition_hint": "text"}
        outcome = {"disposition": "text", "text": "x", "message_ids": ["m"]}
        assert runner.record_turn_delivery(addr, outcome) is True
        assert recorded == [("s", 5, "t", outcome)]

    def test_record_turn_delivery_without_db_is_a_silent_noop(self):
        runner = _Runner(session_db=None)
        assert runner.record_turn_delivery(
            {"session_id": "s", "row_id": 5, "turn_id": "t", "disposition_hint": "text"},
            {"disposition": "unknown"}) is False


class TestAdapterFinalizer:
    @pytest.mark.asyncio
    async def test_finalizer_forwards_outcome_on_the_stamped_address(self):
        adapter = _Adapter.__new__(_Adapter)
        forwarded = []
        adapter.gateway_runner = SimpleNamespace(
            record_turn_delivery=lambda addr, outcome: forwarded.append((addr, outcome)) or True)
        event = SimpleNamespace(
            turn_delivery_address={"session_id": "s", "row_id": 5, "turn_id": "t",
                                   "disposition_hint": "text"})
        await adapter._finalize_turn_delivery_record(
            event, [SendResult(success=True, message_id="m-1", delivered_text="sent text")])
        addr, outcome = forwarded[0]
        assert addr["row_id"] == 5
        assert outcome["disposition"] == "text" and outcome["text"] == "sent text"

    @pytest.mark.asyncio
    async def test_finalizer_without_address_or_runner_does_nothing(self):
        adapter = _Adapter.__new__(_Adapter)
        # No gateway_runner, no address: must not raise, must not invent a record.
        await adapter._finalize_turn_delivery_record(
            SimpleNamespace(), [SendResult(success=True, message_id="m")])
        adapter.gateway_runner = SimpleNamespace()
        await adapter._finalize_turn_delivery_record(
            SimpleNamespace(turn_delivery_address={"session_id": "s", "row_id": 1,
                                                   "turn_id": "t", "disposition_hint": "text"}),
            [])


class TestProcessMessageBackgroundRuntimeLane:
    """Drive the REAL ``_process_message_background`` (the lane base.py actually dispatches)
    through its send/record path. Regression for the shadowed-duplicate defect: a second
    legacy def of this method in the class body replaces the instrumented one, silently
    disabling delivery recording — with that shape present, these assertions fail because
    no record is ever forwarded."""

    ADDRESS = {"session_id": "s", "row_id": 5, "turn_id": "t",
               "disposition_hint": "text", "streamed_text": None}

    def _adapter(self, response_text, sent_results):
        adapter = _Adapter.__new__(_Adapter)
        forwarded = []
        # ``name`` is a read-only property backed by platform.value — set the backing.
        adapter.platform = SimpleNamespace(value="stub")
        adapter.gateway_runner = SimpleNamespace(
            record_turn_delivery=lambda addr, outcome: forwarded.append((addr, outcome)) or True)
        adapter._active_sessions = {}
        adapter._pending_messages = {}
        adapter._expected_cancelled_tasks = set()
        adapter._streaming_tts_completed_turns = set()
        adapter._streaming_tts_turn_key = lambda *a, **k: ""
        adapter._start_typing_refresh = lambda *a, **k: None

        async def _noop(*_a, **_k):
            return None

        adapter._stop_typing_refresh = _noop
        adapter._run_processing_hook = _noop
        adapter._flush_text_debounce_now = _noop
        adapter._fire_post_delivery_callback = _noop
        adapter._finish_session_task = lambda *a, **k: None
        adapter._media_delivery_scope = lambda _src: contextlib.nullcontext()

        async def _handler(_event):
            return response_text

        adapter._message_handler = _handler
        adapter.extract_media = lambda r: ([], r)
        adapter.extract_images = lambda r: ([], r)
        adapter.extract_local_files = lambda t: ([], t)
        adapter.filter_media_delivery_paths = lambda files, session_key=None: files
        adapter.filter_local_delivery_paths = lambda files, session_key=None: files
        adapter._bounded_history_media_paths_for_session = _noop
        adapter._wants_auto_tts = lambda *a, **k: False
        adapter.pause_typing_for_chat = lambda *a, **k: None
        adapter._get_human_delay = lambda: 0.0

        async def _send_final_text(_event, _sk, text_content, _meta, _is_eph, _ttl, record_delivery):
            for result in sent_results:
                record_delivery(result)

        adapter._send_final_text = _send_final_text
        event = SimpleNamespace(
            source=SimpleNamespace(platform=SimpleNamespace(value="stub"), chat_id="c1"),
            turn_delivery_address=dict(self.ADDRESS), metadata=None, message_type=None)
        return adapter, event, forwarded

    @pytest.mark.asyncio
    async def test_runtime_lane_records_actual_text_delivery(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.base._thread_metadata_for_event", lambda _e: None)
        monkeypatch.setattr("gateway.platforms.base.diagnostic_wake_muted", lambda _e: False)
        adapter, event, forwarded = self._adapter(
            "final answer", [SendResult(success=True, message_id="m-1", delivered_text="final answer")])
        await adapter._process_message_background(event, "sk:1")
        assert len(forwarded) == 1, "the dispatched method must record the turn's delivery"
        addr, outcome = forwarded[0]
        assert addr["row_id"] == 5
        assert outcome["disposition"] == "text" and outcome["text"] == "final answer"

    @pytest.mark.asyncio
    async def test_runtime_lane_records_silence_when_nothing_sent(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.base._thread_metadata_for_event", lambda _e: None)
        monkeypatch.setattr("gateway.platforms.base.diagnostic_wake_muted", lambda _e: False)
        adapter, event, forwarded = self._adapter("", [])
        event.turn_delivery_address = {**self.ADDRESS, "disposition_hint": "silence"}
        await adapter._process_message_background(event, "sk:2")
        assert len(forwarded) == 1
        assert forwarded[0][1]["disposition"] == "silence"

    @pytest.mark.asyncio
    async def test_runtime_lane_records_partial_failure(self, monkeypatch):
        monkeypatch.setattr("gateway.platforms.base._thread_metadata_for_event", lambda _e: None)
        monkeypatch.setattr("gateway.platforms.base.diagnostic_wake_muted", lambda _e: False)
        adapter, event, forwarded = self._adapter(
            "answer + image",
            [SendResult(success=False, error="HTTP 429: flood control"),
             SendResult(success=True, message_id="m-2")])
        await adapter._process_message_background(event, "sk:3")
        assert len(forwarded) == 1
        outcome = forwarded[0][1]
        assert outcome["disposition"] == "media" and outcome.get("partial") is True
        assert outcome["error"] == "HTTP 429: flood control"


class TestNotifyTurnErrorSanitized:
    @pytest.mark.asyncio
    async def test_notice_carries_exception_class_only(self, monkeypatch):
        """The delivery lane's last-resort error notice must not forward raw exception
        text to the chat (SDK errors can embed URLs with tokens, paths); detail stays in
        the log, matching the agent-turn error path's policy."""
        import gateway.platforms.base as base_module
        monkeypatch.setattr(base_module, "_thread_metadata_for_event", lambda _e: None)
        monkeypatch.setattr(base_module, "diagnostic_wake_muted", lambda _e: False)
        adapter = _Adapter.__new__(_Adapter)
        adapter.platform = SimpleNamespace(value="stub")
        sent = []

        async def _send(chat_id, content, metadata=None):
            sent.append(content)
            return SendResult(success=True, message_id="m")

        adapter.send = _send
        adapter.warning_text = lambda text, fallback, **_k: text
        event = SimpleNamespace(
            source=SimpleNamespace(platform=SimpleNamespace(value="stub"), chat_id="c1"))
        err = ConnectionError(
            "GET https://api.example.com/v1/chat?token=sk-secret-123 failed: 502 Bad Gateway")
        await adapter._notify_turn_error(event, err)
        assert sent and "ConnectionError" in sent[0]
        assert "sk-secret-123" not in sent[0]
        assert "api.example.com" not in sent[0]
