"""Conversation identity reaches a configured gateway, not a stale route."""
from unittest.mock import patch

from agent import auxiliary_client as aux
from agent.chat_completion_helpers import build_api_kwargs
from run_agent import AIAgent

GATEWAY = {
    "name": "gateway", "provider_key": "gateway",
    "base_url": "http://127.0.0.1:4000/v1",
    "session_affinity_header": "x-litellm-session-id",
}
MESSAGES = [{"role": "user", "content": "Synthetic protocol check"}]


def test_main_and_auxiliary_share_the_configured_conversation_scope():
    agent = AIAgent(api_key="synthetic-key", provider="custom",
                    requested_provider="custom:gateway", model="bb-chat",
                    base_url=GATEWAY["base_url"], quiet_mode=True,
                    skip_context_files=True, skip_memory=True,
                    session_id="conversation-a")
    token = aux.set_runtime_main("custom", "bb-chat", requested_provider="custom:gateway",
                                 base_url=GATEWAY["base_url"], session_id="conversation-a")
    try:
        with patch("hermes_cli.config.get_compatible_custom_providers", return_value=[GATEWAY]):
            main = build_api_kwargs(agent, MESSAGES)
            other = aux._build_call_kwargs("custom", "bb-chat", MESSAGES,
                                           base_url=GATEWAY["base_url"])
        assert main.get("extra_headers", {}).get("x-litellm-session-id") == "conversation-a"
        assert other.get("extra_headers", {}).get("x-litellm-session-id") == "conversation-a"
    finally:
        aux._RUNTIME_MAIN_CONTEXT.reset(token)


def test_provider_name_cannot_send_identity_to_a_different_destination():
    from agent.opencode_affinity import merge_session_affinity_headers
    with patch("hermes_cli.config.get_compatible_custom_providers", return_value=[GATEWAY]):
        request = merge_session_affinity_headers({}, "gateway", "https://unrelated.example/v1", "conversation-a")
    assert "x-litellm-session-id" not in request.get("extra_headers", {})


def test_missing_route_identity_does_not_select_the_first_configured_provider():
    from hermes_cli.config import get_custom_provider_session_affinity_header
    assert get_custom_provider_session_affinity_header(custom_providers=[GATEWAY]) is None
