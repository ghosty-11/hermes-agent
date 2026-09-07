"""Behavioral regressions for the strict native avatar /v1/runs protocol."""

import asyncio
import base64
import io
import json
import struct
import zlib

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from gateway.platforms import api_server_strict_runs as strict_mod
from gateway.platforms.api_server import _openai_error

from .test_api_server_runs import _create_runs_app, _make_adapter


def _encode_png(size=(3, 2)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (12, 200, 90)).save(buf, format="PNG")
    return buf.getvalue()


STRICT_TEXT = "Describe what is currently visible on my desktop."
TINY_PNG = _encode_png()
IMAGE_DATA = "data:image/png;base64," + base64.b64encode(TINY_PNG).decode("ascii")
GATEWAY_URL = "http://127.0.0.1:4000/v1"
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def _strict_input(*, content=None, tool_policy="inherit", require_image_input=False):
    return {
        "type": "hermes.strict_run.v1",
        "content": content if content is not None else [{"type": "text", "text": STRICT_TEXT}],
    }, {
        "model": "bb-avatar",
        "provider": "custom:gateway",
        "require_model_lock": True,
        "tool_policy": tool_policy,
        "require_image_input": require_image_input,
    }


def _agent(*, model="bb-avatar", provider="custom:gateway", base_url=GATEWAY_URL, vision=True):
    """Constructed-runtime stand-in: the attributes a real AIAgent exposes
    after _create_agent (model/provider/base_url and the dialed client URL)."""
    agent = MagicMock()
    agent.model = model
    agent.provider = provider
    agent.base_url = base_url
    # httpx-style client URL carries a trailing slash.
    agent.client.base_url = f"{base_url}/" if base_url else base_url
    agent._fallback_activated = False
    agent._model_supports_vision.return_value = vision
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    return agent


def _seed_session(adapter, session_id, *, session_key=None, lock=None):
    """Create a real session row (and optionally a confirmed lock) in the
    per-test state DB, the way an earlier turn or POST /api/sessions would."""
    db = adapter._ensure_session_db()
    assert db is not None
    db.create_session(session_id, "api_server", session_key=session_key)
    if lock:
        db.update_session_runtime_lock(
            session_id,
            model=lock["model"],
            provider=lock["provider"],
            route_source="raw_request",
            confirmed=True,
        )
    return db


def _persisted_lock(adapter, session_id):
    row = adapter._ensure_session_db().get_session(session_id)
    return strict_mod.parse_persisted_browser_lock(row.get("model_config")) if row else None


def _gateway_route(adapter):
    adapter._model_routes = {
        "bb-avatar": {
            "model": "bb-avatar",
            "provider": "custom:gateway",
            "base_url": GATEWAY_URL,
        },
    }


async def _wait_for_settlement(cli, run_id):
    for _ in range(80):
        response = await cli.get(f"/v1/runs/{run_id}")
        status = await response.json()
        if status.get("status") in TERMINAL:
            return status
        await asyncio.sleep(0.01)
    pytest.fail("strict run did not settle")


async def _events(cli, run_id):
    events_resp = await cli.get(f"/v1/runs/{run_id}/events")
    assert events_resp.status == 200
    sse_body = await events_resp.text()
    return [
        json.loads(line[len("data: "):])
        for line in sse_body.splitlines()
        if line.startswith("data: ")
    ]


async def _submit_and_settle(adapter, agent, *, body, headers=None):
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent) as create:
            response = await cli.post("/v1/runs", json=body, headers=headers or {})
            payload = await response.json()
            assert response.status == 202, payload
            status = await _wait_for_settlement(cli, payload["run_id"])
            events = await _events(cli, payload["run_id"])
    return status, events, create


# ─── Admission and runtime confirmation ─────────────────────────────────────


@pytest.mark.asyncio
async def test_strict_run_accepts_discriminated_input_and_locks_requested_route():
    """The new object input is accepted without falling back to legacy strings."""
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    captured = {}
    agent = _agent()

    def run(**kwargs):
        captured.update(kwargs)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run
    status, _events_, create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )

    kwargs = create.call_args.kwargs
    assert kwargs["requested_model"] == "bb-avatar"
    assert kwargs["requested_provider"] == "custom:gateway"
    assert kwargs["session_id"] == "avatar-session"
    assert captured["user_message"] == [{"type": "text", "text": STRICT_TEXT}]
    assert status["status"] == "completed"


@pytest.mark.asyncio
async def test_strict_run_streams_constructed_runtime_before_any_output_event():
    """The runtime descriptor is queued as run.runtime and merged into status
    BEFORE any provider output, and it describes the CONSTRUCTED agent: the
    configured route's base URL and route_source, the agent's own
    model/provider, and a lock that was really confirmed on the session."""
    adapter = _make_adapter()
    _gateway_route(adapter)
    _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    expected_runtime = {
        "strict_run_version": 1,
        "requested_provider": "custom:gateway",
        "provider": "custom:gateway",
        "model": "bb-avatar",
        "base_url": GATEWAY_URL,
        "route_source": "model_routes",
        "model_lock": True,
        "tool_policy": "inherit",
        "image_input_mode": "none",
        "require_image_input": False,
    }
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=_agent()):
            response = await cli.post(
                "/v1/runs",
                json={"input": strict_input, **options, "session_id": "avatar-session"},
            )
            body = await response.json()
            assert response.status == 202, body
            run_id = body["run_id"]
            status = await _wait_for_settlement(cli, run_id)
            events = await _events(cli, run_id)

    names = [e.get("event") for e in events]
    assert "run.completed" in names, names
    assert "run.runtime" in names, names
    assert names.index("run.runtime") < names.index("run.completed"), names
    runtime_event = events[names.index("run.runtime")]
    assert runtime_event["run_id"] == run_id
    assert runtime_event["runtime"] == expected_runtime
    assert status["runtime"] == expected_runtime
    # The 202 acknowledgement carries no runtime: nothing is confirmed
    # before the agent is constructed.
    assert "runtime" not in body


@pytest.mark.asyncio
async def test_strict_runtime_mismatch_refuses_before_inference():
    """A constructed agent whose model/provider differ from the requested
    pair never reaches run_conversation and never attests a runtime."""
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    agent = _agent(model="glm-4.5-air", provider="openrouter",
                   base_url="https://openrouter.ai/api/v1")

    status, events, _create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )

    agent.run_conversation.assert_not_called()
    assert status["status"] == "failed"
    assert status.get("error_code") == "runtime_mismatch"
    assert "runtime" not in status
    assert "run.runtime" not in [e.get("event") for e in events]
    # Escalation carries the exact resolved identity, not a relabel.
    assert "openrouter" in status["error"]
    assert "glm-4.5-air" in status["error"]


@pytest.mark.asyncio
async def test_strict_bare_custom_identity_not_owned_by_config_is_escalated():
    """The named-provider resolver canonicalizes ``custom:<name>`` to the
    billing class ``custom``. With no configured entry owning the dialed URL
    the identity cannot be confirmed as the requested one: the run refuses
    with the exact returned identity instead of relabelling it."""
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    agent = _agent(provider="custom")

    status, _events_, _create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )

    agent.run_conversation.assert_not_called()
    assert status["status"] == "failed"
    assert status.get("error_code") == "runtime_mismatch"
    assert "'custom'" in status["error"] or " custom" in status["error"]
    assert "runtime" not in status


@pytest.mark.asyncio
async def test_strict_identity_cannot_fall_back_to_the_global_custom_provider():
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    agent = _agent(provider="custom", base_url="http://127.0.0.1:7999/v1")
    configured_url = "http://127.0.0.1:4000/v1"
    with (
        patch("hermes_cli.runtime_provider.find_custom_provider_identity",
              side_effect=lambda url: "custom:gateway" if str(url).rstrip("/") == configured_url else None),
        patch("hermes_cli.runtime_provider._get_model_config",
              return_value={"provider": "custom:gateway"}),
        patch("hermes_cli.runtime_provider._get_named_custom_provider",
              return_value={"base_url": configured_url}),
    ):
        status, events, _create = await _submit_and_settle(
            adapter, agent,
            body={"input": strict_input, **options, "session_id": "avatar-session"},
        )
    agent.run_conversation.assert_not_called()
    assert status["status"] == "failed"
    assert status["error_code"] == "runtime_mismatch"
    assert "run.runtime" not in [event.get("event") for event in events]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:sekrit-pw@127.0.0.1:4000/v1",
        "http://127.0.0.1:4000/v1?api_key=sekrit-pw",
        "http://127.0.0.1:4000/v1#sekrit-pw",
        "",
    ],
)
async def test_strict_unconfirmable_base_url_refuses_instead_of_sanitizing(base_url):
    """Credential-, query- or fragment-bearing and unknown client URLs are
    refused before inference — never sanitized into an apparently approved
    endpoint — and the secret never reaches status or events."""
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    agent = _agent(base_url=base_url)

    status, events, _create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )

    agent.run_conversation.assert_not_called()
    assert status["status"] == "failed"
    assert status.get("error_code") == "runtime_unconfirmed"
    assert "runtime" not in status
    assert "sekrit-pw" not in json.dumps(status)
    assert "sekrit-pw" not in json.dumps(events)


@pytest.mark.asyncio
async def test_strict_required_image_refuses_without_native_vision_capability():
    """require_image_input needs the RESOLVED model's native vision
    capability. Absent/unknown capability fails closed before any provider
    request instead of shipping the screenshot and hoping."""
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    content = [
        {"type": "text", "text": STRICT_TEXT},
        {"type": "image_url", "image_url": {"url": IMAGE_DATA}},
    ]
    strict_input, options = _strict_input(content=content, tool_policy="none", require_image_input=True)
    agent = _agent(vision=False)

    status, _events_, _create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )

    agent.run_conversation.assert_not_called()
    assert status["status"] == "failed"
    assert status.get("error_code") == "image_capability_unavailable"
    assert "runtime" not in status
    assert "data:image" not in json.dumps(status)

@pytest.mark.asyncio
@pytest.mark.parametrize("declared_vision", [True, False, None])
async def test_strict_image_uses_only_its_configured_model_capability(monkeypatch, declared_vision):
    from run_agent import AIAgent

    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    config = {
        "model": {"provider": "unrelated-default", "supports_vision": declared_vision is not True},
        "providers": {"gateway": {"models": {
            "bb-avatar": {} if declared_vision is None else {"supports_vision": declared_vision}
        }}},
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    content = [
        {"type": "text", "text": STRICT_TEXT},
        {"type": "image_url", "image_url": {"url": IMAGE_DATA}},
    ]
    strict_input, options = _strict_input(content=content, tool_policy="none", require_image_input=True)
    agent = _agent()
    agent._model_supports_vision = lambda **kwargs: AIAgent._model_supports_vision(agent, **kwargs)
    status, _, _ = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )
    if declared_vision is True:
        assert status["status"] == "completed"
        agent.run_conversation.assert_called_once()
    else:
        agent.run_conversation.assert_not_called()
        assert status.get("error_code") == "image_capability_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_body",
    [
        {"input": {"type": "hermes.strict_run.v1", "content": [{"type": "text", "text": "x"}]}},
        {"input": {"type": "hermes.strict_run.v1", "content": [{"type": "text", "text": "x"}]}, "model": "bb-avatar", "provider": "custom:gateway", "require_model_lock": False, "tool_policy": "inherit", "require_image_input": False},
        {"input": {"type": "hermes.strict_run.v1", "content": [{"type": "text", "text": "x"}]}, "model": "bb-avatar", "provider": "custom:gateway", "require_model_lock": True, "tool_policy": "bogus", "require_image_input": False},
        {"input": {"type": "hermes.strict_run.v1", "content": [{"type": "text", "text": "x"}], "extra": True}, "model": "bb-avatar", "provider": "custom:gateway", "require_model_lock": True, "tool_policy": "inherit", "require_image_input": False},
    ],
)
async def test_strict_run_malformed_schema_is_rejected_before_reservation(request_body):
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post("/v1/runs", json=request_body)
            body = await response.json()
    assert response.status == 400, body
    assert adapter._run_streams == {}
    assert adapter._run_statuses == {}
    create.assert_not_called()


@pytest.mark.asyncio
async def test_strict_route_pinned_to_other_provider_is_rejected_before_work():
    """A model_routes alias pinned to a different provider than the strict
    request names is an ambiguous runtime: refuse before agent creation
    rather than dialing one provider's URL with another's credentials."""
    adapter = _make_adapter()
    adapter._model_routes = {
        "bb-avatar": {"model": "bb-avatar", "provider": "openrouter"},
    }
    strict_input, options = _strict_input()
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post("/v1/runs", json={"input": strict_input, **options})
            body = await response.json()
    assert response.status == 400, body
    create.assert_not_called()
    assert adapter._run_streams == {}


@pytest.mark.asyncio
async def test_strict_image_run_preserves_image_and_forces_no_tools():
    adapter = _make_adapter()
    _seed_session(adapter, "avatar-session")
    content = [
        {"type": "text", "text": STRICT_TEXT},
        {"type": "image_url", "image_url": {"url": IMAGE_DATA}},
    ]
    strict_input, options = _strict_input(content=content, tool_policy="none", require_image_input=True)
    captured = {}
    agent = _agent()

    def run(**kwargs):
        captured.update(kwargs)
        return {"final_response": "seen"}

    agent.run_conversation.side_effect = run
    status, _events_, create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )
    assert status["status"] == "completed"
    assert status["runtime"]["image_input_mode"] == "native"
    assert status["runtime"]["tool_policy"] == "none"
    assert status["runtime"]["require_image_input"] is True
    assert status["runtime"]["model_lock"] is True

    assert create.call_count == 1
    assert captured["user_message"] == content


@pytest.mark.asyncio
async def test_strict_required_image_rejects_missing_image_without_starting_work():
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    strict_input, options = _strict_input(tool_policy="none", require_image_input=True)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post("/v1/runs", json={"input": strict_input, **options})
            body = await response.json()
    assert response.status == 400, body
    assert adapter._run_streams == {}
    create.assert_not_called()


@pytest.mark.asyncio
async def test_strict_invalid_image_does_not_retry_or_echo_image_canary():
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    canary = "strict-image-canary-must-not-escape"
    bad_content = [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": f"data:image/svg+xml,{canary}"}},
    ]
    strict_input, options = _strict_input(content=bad_content, tool_policy="none", require_image_input=True)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post("/v1/runs", json={"input": strict_input, **options})
            text = await response.text()
    assert response.status == 400
    assert canary not in text
    create.assert_not_called()
    assert adapter._run_streams == {}


# ─── Effective-session lock semantics ───────────────────────────────────────


@pytest.mark.asyncio
async def test_strict_lock_conflict_on_header_declared_session_is_refused():
    """With no body session_id the run executes against the conversation the
    ``X-Hermes-Session-Key`` header declares. A confirmed lock on THAT row
    for another model/provider must refuse the strict request before any
    agent work — exactly like a body session_id would."""
    # X-Hermes-Session-Key is honored only on authenticated requests.
    adapter = _make_adapter(api_key="sk-secret")
    _seed_session(
        adapter, "declared-session", session_key="avatar-key",
        lock={"model": "anthropic/claude-sonnet", "provider": "openrouter"},
    )
    strict_input, options = _strict_input()
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            response = await cli.post(
                "/v1/runs",
                json={"input": strict_input, **options},
                headers={
                    "Authorization": "Bearer sk-secret",
                    "X-Hermes-Session-Key": "avatar-key",
                },
            )
            body = await response.json()
    assert response.status == 409, body
    assert body["error"]["code"] == "model_lock_conflict"
    create.assert_not_called()
    assert adapter._run_streams == {}
    # The foreign lock is untouched.
    assert _persisted_lock(adapter, "declared-session")["model"] == "anthropic/claude-sonnet"


@pytest.mark.asyncio
async def test_strict_concurrency_rejection_persists_no_lock():
    """A 429 at concurrency admission must leave the session row unlocked:
    a request that never ran cannot pin a shared session to bb-avatar."""
    adapter = _make_adapter()
    _seed_session(adapter, "shared-session")
    strict_input, options = _strict_input()
    app = _create_runs_app(adapter)
    limited = web.json_response(_openai_error("busy", code="concurrency_limited"), status=429)
    async with TestClient(TestServer(app)) as cli:
        with (
            patch.object(adapter, "_concurrency_limited_response", return_value=limited),
            patch.object(adapter, "_create_agent") as create,
        ):
            response = await cli.post(
                "/v1/runs",
                json={"input": strict_input, **options, "session_id": "shared-session"},
            )
    assert response.status == 429
    create.assert_not_called()
    assert _persisted_lock(adapter, "shared-session") is None


@pytest.mark.asyncio
async def test_strict_idempotency_conflict_persists_no_lock():
    """A 409 idempotency-key conflict is a rejected request: no durable lock."""
    adapter = _make_adapter()
    _seed_session(adapter, "shared-session")
    strict_input, options = _strict_input()
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=_agent()):
            first = await cli.post(
                "/v1/runs",
                json={"input": strict_input, **options, "session_id": "other-session"},
                headers={"Idempotency-Key": "reused-key"},
            )
            assert first.status == 202, await first.text()
            await _wait_for_settlement(cli, (await first.json())["run_id"])
        with patch.object(adapter, "_create_agent") as create:
            second = await cli.post(
                "/v1/runs",
                json={"input": strict_input, **options, "session_id": "shared-session"},
                headers={"Idempotency-Key": "reused-key"},
            )
    assert second.status == 409, await second.text()
    create.assert_not_called()
    assert _persisted_lock(adapter, "shared-session") is None


@pytest.mark.asyncio
async def test_strict_lock_is_confirmed_on_effective_session_before_inference():
    """After admission and runtime confirmation the lock is persisted on the
    effective session and is already durable when run_conversation starts;
    model_lock:true in the descriptor reports that confirmed lock."""
    adapter = _make_adapter()
    db = _seed_session(adapter, "avatar-session")
    strict_input, options = _strict_input()
    agent = _agent()
    seen = {}

    def run(**kwargs):
        seen["lock_at_inference"] = _persisted_lock(adapter, "avatar-session")
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run
    status, _events_, _create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "avatar-session"},
    )

    assert status["status"] == "completed"
    assert status["runtime"]["model_lock"] is True
    lock = seen["lock_at_inference"]
    assert lock is not None and lock["confirmed"] is True
    assert lock["model"] == "bb-avatar"
    assert lock["provider"] == "custom:gateway"
    assert _persisted_lock(adapter, "avatar-session")["model"] == "bb-avatar"
    assert db.get_session("avatar-session") is not None


@pytest.mark.asyncio
async def test_strict_run_without_confirmable_lock_refuses_before_inference():
    """When no effective session row exists to carry the lock (and the
    constructed agent does not create one), the lock cannot be confirmed:
    the run refuses before inference rather than attesting model_lock:true."""
    adapter = _make_adapter()
    strict_input, options = _strict_input()
    agent = _agent()

    status, _events_, _create = await _submit_and_settle(
        adapter, agent,
        body={"input": strict_input, **options, "session_id": "never-created"},
    )

    agent.run_conversation.assert_not_called()
    assert status["status"] == "failed"
    assert status.get("error_code") == "model_lock_unconfirmed"
    assert "runtime" not in status


# ─── Strict agent construction ──────────────────────────────────────────────


def _strict_create_kwargs():
    return {
        "session_id": "avatar-session",
        "requested_model": "bb-avatar",
        "requested_provider": "custom:gateway",
        "route": {"model": "bb-avatar", "provider": "custom:gateway"},
        "strict_controls": {
            "tool_policy": "none",
            "require_image_input": False,
            "has_image": False,
        },
    }


class _FakeAgent:
    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = dict(kwargs)
        self.model = kwargs.get("model")
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.tools = [{"type": "function", "function": {"name": "terminal", "parameters": {}}}]
        self.valid_tool_names = {"terminal"}


def test_strict_agent_resolves_only_requested_provider_without_global_seed(monkeypatch):
    """A strict agent is built from the explicitly requested provider alone.
    A broken/fallen-through global provider resolution must not abort or
    seed it, and the legacy per-provider fallback resolver is never used."""
    from .test_api_server import _patch_create_agent_runtime

    _patch_create_agent_runtime(monkeypatch, {}, _FakeAgent)

    def _global_broken():
        raise RuntimeError("global provider auth failed")

    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", _global_broken)
    legacy = MagicMock(return_value={"provider": "custom", "base_url": "http://legacy/v1", "api_key": "k"})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", legacy)
    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        lambda provider, target_model=None: {
            "provider": "custom",
            "api_key": "sk-gateway",
            "base_url": GATEWAY_URL,
            "api_mode": "chat_completions",
        },
    )
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    agent = adapter._create_agent(**_strict_create_kwargs())

    legacy.assert_not_called()
    assert _FakeAgent.captured["base_url"] == GATEWAY_URL
    assert _FakeAgent.captured["api_key"] == "sk-gateway"
    assert _FakeAgent.captured["model"] == "bb-avatar"
    assert _FakeAgent.captured["fallback_model"] is None
    # tool_policy:none clears the exposed surface and the dispatch names.
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_strict_agent_provider_resolution_failure_is_typed_without_legacy_fallback(monkeypatch):
    """When the requested provider cannot be resolved, the strict path
    raises the typed provider-auth failure instead of initializing through
    the legacy fallback resolver."""
    from gateway.platforms.api_server import _ProviderAuthResolutionError
    from .test_api_server import _patch_create_agent_runtime

    _patch_create_agent_runtime(monkeypatch, {}, _FakeAgent)
    legacy = MagicMock(return_value={"provider": "custom", "base_url": "http://legacy/v1", "api_key": "k"})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", legacy)

    def _requested_broken(provider, target_model=None):
        raise RuntimeError("custom:gateway has no usable credential")

    monkeypatch.setattr(
        "gateway.platforms.api_server._resolve_request_runtime_agent_kwargs",
        _requested_broken,
    )
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)

    with pytest.raises(_ProviderAuthResolutionError):
        adapter._create_agent(**_strict_create_kwargs())
    legacy.assert_not_called()


# ─── Image validation ───────────────────────────────────────────────────────


def _png_chunk(cid: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + cid + data + struct.pack(">I", zlib.crc32(cid + data) & 0xFFFFFFFF)


def _padded_png(total_size: int) -> bytes:
    """A structurally valid PNG of exactly *total_size* bytes: the tiny 3x2
    frame with one private ancillary chunk inserted after IHDR."""
    ihdr_end = 8 + 12 + 13
    pad_len = total_size - len(TINY_PNG) - 12
    assert pad_len >= 0
    return TINY_PNG[:ihdr_end] + _png_chunk(b"prVt", b"\0" * pad_len) + TINY_PNG[ihdr_end:]


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def test_strict_image_bound_admits_full_three_mib_frame_and_refuses_one_byte_more():
    """The advertised max_image_bytes is 3 MiB DECODED: a valid frame of
    exactly that size is admitted; one byte more is image_too_large."""
    exact = _padded_png(strict_mod.MAX_STRICT_IMAGE_BYTES)
    assert len(exact) == strict_mod.MAX_STRICT_IMAGE_BYTES
    meta = strict_mod.validate_strict_image_data_url(_data_url(exact))
    assert meta["decoded_bytes"] == strict_mod.MAX_STRICT_IMAGE_BYTES
    assert (meta["width"], meta["height"]) == (3, 2)

    over = _padded_png(strict_mod.MAX_STRICT_IMAGE_BYTES + 1)
    with pytest.raises(strict_mod.StrictRunError) as exc:
        strict_mod.validate_strict_image_data_url(_data_url(over))
    assert exc.value.code == "image_too_large"


def test_strict_image_with_valid_header_but_corrupt_body_is_invalid():
    """A readable IHDR over corrupt compressed data is not a valid frame."""
    idat_at = TINY_PNG.index(b"IDAT")
    corrupt = bytearray(TINY_PNG)
    for offset in range(idat_at + 4, idat_at + 12):
        corrupt[offset] ^= 0xFF
    with pytest.raises(strict_mod.StrictRunError) as exc:
        strict_mod.validate_strict_image_data_url(_data_url(bytes(corrupt)))
    assert exc.value.code == "invalid_image"


def test_strict_image_truncated_png_is_invalid():
    """A PNG cut off inside IDAT still parses its header; it must be refused."""
    idat_at = TINY_PNG.index(b"IDAT")
    truncated = TINY_PNG[: idat_at + 8]
    with pytest.raises(strict_mod.StrictRunError) as exc:
        strict_mod.validate_strict_image_data_url(_data_url(truncated))
    assert exc.value.code == "invalid_image"


def test_strict_image_truncated_jpeg_is_invalid():
    """A JPEG with an intact SOF header but a truncated scan is refused."""
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 30, 30)).save(buf, format="JPEG")
    jpeg = buf.getvalue()
    full = strict_mod.validate_strict_image_data_url(_data_url(jpeg, "image/jpeg"))
    assert (full["width"], full["height"]) == (16, 16)

    scan_at = jpeg.index(b"\xff\xda")
    truncated = jpeg[: scan_at + 20]
    with pytest.raises(strict_mod.StrictRunError) as exc:
        strict_mod.validate_strict_image_data_url(_data_url(truncated, "image/jpeg"))
    assert exc.value.code == "invalid_image"
