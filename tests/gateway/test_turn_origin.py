"""Server-created ``TurnOrigin`` and response policy for companion turns.

Contracts:

- ``gateway.turn_origin.TurnOrigin`` validates a trusted candidate; junk kinds/policies are
  dropped — public text is never consulted, and an unverifiable policy fails CLOSED to
  ``required``.
- ``turn_origin_allows_silence`` permits intentional silence ONLY for an explicit
  ``discretionary`` policy (machinery display kinds keep their existing lane).
- Gateway resolution: an adapter MAY supply the origin via ``turn_origin_for_event``; no
  adapter (or a junk return) falls back to the gateway default built from trusted state
  only — ``required`` policy, kind scheduled/resume/direct from flags, never from text.
- Shaping (``_hmwa_shape_agent_response``): the blanket "silence marker rejected on a user
  turn" rule is replaced by the policy rule. ``required`` turns keep the existing fallback
  reply (work surfaces unchanged); ``discretionary`` companion turns honor silence, including
  the queued-terminal turn that owns the chain's verdict.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.run_turn import GatewayTurnMixin, _UNEXPECTED_SILENCE_REPLY
from gateway.turn_origin import (
    RESPONSE_POLICY_DISCRETIONARY,
    RESPONSE_POLICY_REQUIRED,
    TURN_ORIGIN_KINDS,
    TurnOrigin,
    normalize_turn_origin,
    turn_origin_allows_silence,
)


class TestNormalizeTurnOrigin:
    def test_valid_origin_passes_through(self):
        origin = normalize_turn_origin(
            {"event_id": "evt-1", "kind": "ambient", "response_policy": "discretionary"}
        )
        assert isinstance(origin, TurnOrigin)
        assert (origin.event_id, origin.kind, origin.response_policy) == (
            "evt-1", "ambient", RESPONSE_POLICY_DISCRETIONARY
        )

    def test_turn_origin_instance_passes_through(self):
        origin = TurnOrigin(event_id="evt-2", kind="direct", response_policy=RESPONSE_POLICY_REQUIRED)
        assert normalize_turn_origin(origin) is origin

    def test_unknown_kind_or_policy_is_dropped(self):
        assert normalize_turn_origin({"event_id": "e", "kind": "side_channel"}) is None
        assert normalize_turn_origin(
            {"event_id": "e", "kind": "direct", "response_policy": "whenever"}) is None
        assert normalize_turn_origin({"kind": "direct", "response_policy": "required"}) is None

    def test_extra_fields_from_untrusted_sources_are_ignored(self):
        origin = normalize_turn_origin(
            {"event_id": "e", "kind": "direct", "response_policy": "required",
             "response_policy_from_text": "discretionary", "text": "[SILENT] please"}
        )
        assert origin is not None and origin.response_policy == RESPONSE_POLICY_REQUIRED

    def test_policy_never_upgrades_from_a_non_string(self):
        assert normalize_turn_origin(
            {"event_id": "e", "kind": "direct", "response_policy": True}) is None


class TestSilencePermission:
    def test_required_and_missing_origins_forbid_silence(self):
        assert turn_origin_allows_silence(None) is False
        assert turn_origin_allows_silence(
            {"event_id": "e", "kind": "direct", "response_policy": "required"}) is False
        assert turn_origin_allows_silence({"kind": "ambient"}) is False

    def test_discretionary_policy_permits_silence(self):
        assert turn_origin_allows_silence(
            {"event_id": "e", "kind": "ambient", "response_policy": "discretionary"}) is True

    def test_kind_vocabulary_is_the_approved_set(self):
        assert TURN_ORIGIN_KINDS == frozenset(
            {"direct", "ambient", "catch_up", "scheduled", "resume"})


class _OriginAdapter:
    def __init__(self, origin):
        self._origin = origin

    def turn_origin_for_event(self, event):
        return self._origin


class _NoOriginAdapter:
    pass


class _Runner(GatewayTurnMixin):
    def __init__(self, adapter=None, session_entry=None):
        self._adapter = adapter
        self._session_entry = session_entry or SimpleNamespace(resume_pending=False)
        self.async_session_store = SimpleNamespace(clear_resume_pending=self._noop)

    def _delivery_adapter_for(self, _source):
        return self._adapter


    async def _noop(self, *_a, **_k):
        return None

    async def _clear_restart_failure_count(self, *_a, **_k):
        return None


def _source():
    return SimpleNamespace(chat_id="c1", platform=SimpleNamespace(value="discord"))


class TestGatewayOriginResolution:
    def test_adapter_supplied_discretionary_origin_is_kept(self):
        runner = _Runner(adapter=_OriginAdapter(
            {"event_id": "evt-9", "kind": "direct", "response_policy": "discretionary"}))
        origin = runner._resolve_turn_origin(SimpleNamespace(_heartbeat_session_id=None), _source())
        assert origin == {"event_id": "evt-9", "kind": "direct",
                          "response_policy": "discretionary"}

    def test_adapter_junk_falls_back_to_required_default(self):
        runner = _Runner(adapter=_OriginAdapter({"event_id": "evt-9", "kind": "ambient",
                                                 "response_policy": "sure"}))
        origin = runner._resolve_turn_origin(SimpleNamespace(_heartbeat_session_id=None), _source())
        assert origin["response_policy"] == RESPONSE_POLICY_REQUIRED
        assert origin["kind"] == "direct"

    def test_no_adapter_defaults_to_required_direct(self):
        runner = _Runner(adapter=_NoOriginAdapter())
        origin = runner._resolve_turn_origin(SimpleNamespace(_heartbeat_session_id=None), _source())
        assert origin == {"event_id": "", "kind": "direct", "response_policy": "required"}

    def test_heartbeat_event_defaults_to_scheduled_required(self):
        runner = _Runner(adapter=_NoOriginAdapter())
        origin = runner._resolve_turn_origin(
            SimpleNamespace(_heartbeat_session_id="sess-hb"), _source())
        assert origin["kind"] == "scheduled" and origin["response_policy"] == "required"

    def test_resume_pending_session_defaults_to_resume_required(self):
        runner = _Runner(adapter=_NoOriginAdapter())
        origin = runner._resolve_turn_origin(
            SimpleNamespace(_heartbeat_session_id=None), _source(),
            session_entry=SimpleNamespace(resume_pending=True))
        assert origin["kind"] == "resume" and origin["response_policy"] == "required"


class TestShapingPolicy:
    def _shape(self, runner, agent_result, **kwargs):
        return runner._hmwa_shape_agent_response(
            agent_result, _source(), history=[], session_entry=SimpleNamespace(session_id="s"),
            session_key=None, _quick_key=None, run_generation=0, _run_start_session_id="s",
            _platform_name="discord", _msg_start_time=0.0, **kwargs,
        )

    @pytest.mark.asyncio
    async def test_required_policy_keeps_the_rejection_fallback(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "[SILENT]", "messages": [], "api_calls": 1},
            persist_user_display_kind=None,
            turn_origin={"event_id": "e", "kind": "direct", "response_policy": "required"},
        )
        assert silent is False
        assert response == _UNEXPECTED_SILENCE_REPLY

    @pytest.mark.asyncio
    async def test_absent_origin_keeps_the_rejection_fallback(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner, {"final_response": "NO_REPLY", "messages": [], "api_calls": 1},
            persist_user_display_kind=None,
        )
        assert silent is False and response == _UNEXPECTED_SILENCE_REPLY

    @pytest.mark.asyncio
    async def test_discretionary_origin_honors_intentional_silence(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "[SILENT]", "messages": [], "api_calls": 1},
            persist_user_display_kind=None,
            turn_origin={"event_id": "e", "kind": "ambient", "response_policy": "discretionary"},
        )
        assert silent is True
        assert response in ("", "[SILENT]") and not response.strip("[ ]SILENT")

    @pytest.mark.asyncio
    async def test_discretionary_silence_sends_no_plumbing_sentence(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "SILENT", "messages": [], "api_calls": 1},
            persist_user_display_kind=None,
            turn_origin={"event_id": "e", "kind": "direct", "response_policy": "discretionary"},
        )
        assert silent is True
        assert response == ""

    @pytest.mark.asyncio
    async def test_machinery_display_kind_lane_is_unchanged(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "[SILENT]", "messages": [], "api_calls": 1},
            persist_user_display_kind="internal_notification",
        )
        assert silent is True

    @pytest.mark.asyncio
    async def test_queued_terminal_origin_owns_the_silence_verdict(self):
        runner = _Runner()
        # The event that OPENED the chain was discretionary, but the queued terminal turn is
        # a required work turn: silence must be rejected.
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "[SILENT]", "messages": [], "api_calls": 1,
             "queued_terminal_turn_origin": {"event_id": "e2", "kind": "direct",
                                             "response_policy": "required"}},
            persist_user_display_kind=None,
            turn_origin={"event_id": "e1", "kind": "ambient", "response_policy": "discretionary"},
        )
        assert silent is False and response == _UNEXPECTED_SILENCE_REPLY

    @pytest.mark.asyncio
    async def test_queued_terminal_discretionary_honors_silence(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "[SILENT]", "messages": [], "api_calls": 1,
             "queued_terminal_turn_origin": {"event_id": "e2", "kind": "ambient",
                                             "response_policy": "discretionary"}},
            persist_user_display_kind=None,
            turn_origin={"event_id": "e1", "kind": "direct", "response_policy": "required"},
        )
        assert silent is True and response == ""

    @pytest.mark.asyncio
    async def test_discretionary_policy_does_not_suppress_real_text(self):
        runner = _Runner()
        response, silent, _ = await self._shape(
            runner,
            {"final_response": "an actual answer", "messages": [], "api_calls": 1},
            persist_user_display_kind=None,
            turn_origin={"event_id": "e", "kind": "ambient", "response_policy": "discretionary"},
        )
        assert silent is False and response == "an actual answer"


class TestQueuedFirstResponseSilencePolicy:
    """The queued-first-response lane must use the completion path's two-lane silence
    gate: machinery display kind OR discretionary origin. A companion surface must not
    receive the silence-fallback sentence mid-chain (review finding)."""

    def _turn_ctx(self, origin):
        return SimpleNamespace(
            mute_notification_reply=False, session_key="sk:queued", stream_consumer_holder=[None],
            persist_user_display_kind=None, turn_origin=origin, source=_source(),
            _status_thread_metadata=None, event_message_id=None, inbound_message_id=None,
            run_generation=0)

    def _runner(self, sent):
        runner = _Runner()

        async def _deliver(first_response, **_kwargs):
            sent.append(first_response)
            return True

        async def _await_stream(_task):
            return None

        runner._deliver_queued_first_response = _deliver
        runner._run_agent_stream_confirmed_final_delivery = lambda *a, **k: False
        runner._pop_post_delivery_callback = lambda *a, **k: None
        return runner

    @pytest.mark.asyncio
    async def test_discretionary_origin_sends_nothing(self):
        sent = []
        runner = self._runner(sent)
        await runner._run_agent_deliver_first_response(
            self._turn_ctx({"event_id": "e", "kind": "ambient", "response_policy": "discretionary"}),
            None, None, {"final_response": "[SILENT]", "messages": []}, None,
        )
        assert sent == [], "honored silence must send nothing — no fallback sentence mid-chain"

    @pytest.mark.asyncio
    async def test_required_origin_still_gets_the_fallback_text(self):
        sent = []
        runner = self._runner(sent)
        await runner._run_agent_deliver_first_response(
            self._turn_ctx({"event_id": "e", "kind": "direct", "response_policy": "required"}),
            None, None, {"final_response": "[SILENT]", "messages": []}, None,
        )
        assert sent == [_UNEXPECTED_SILENCE_REPLY]
