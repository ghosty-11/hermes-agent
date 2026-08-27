"""Regression test for #17929: AIAgent.__init__ should try fallback_model
when primary provider credentials are exhausted."""
import pytest
from unittest.mock import patch, MagicMock
from run_agent import AIAgent


def _make_tool_defs():
    return [{"type": "function", "function": {"name": "web_search",
             "description": "search", "parameters": {"type": "object", "properties": {}}}}]


def _mock_client(api_key="fb-key-1234567890", base_url="https://fb.example.com/v1"):
    c = MagicMock()
    c.api_key = api_key
    c.base_url = base_url
    c._default_headers = None
    return c


def test_init_tries_fallback_when_primary_returns_none():
    """When resolve_provider_client returns None for primary but succeeds for
    a fallback entry, __init__ should NOT raise RuntimeError."""
    fb = _mock_client()

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "tencent-token-plan":
            return fb, "kimi2.5"
        return None, None  # primary exhausted

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="alibaba-coding-plan",
            model="qwen3.6-plus",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[{"provider": "tencent-token-plan", "model": "kimi2.5"}],
        )
        assert agent.provider == "tencent-token-plan"
        assert agent.model == "kimi2.5"
        assert agent._fallback_activated is True


def test_init_raises_when_no_fallback_configured():
    """When primary returns None and no fallback is set, should raise."""
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match="no API key was found"):
            AIAgent(
                provider="alibaba-coding-plan",
                model="qwen3.6-plus",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=None,
            )


def test_init_tries_fallback_when_openrouter_pool_exhausted():
    """Regression: an exhausted ``openrouter`` pool (single credential entry
    in 429-cooldown) must still reach fallback_providers at init.

    Before the fix, the init-time fallback block was nested inside the
    ``_explicit not in {auto, openrouter, custom}`` guard, so the default
    openrouter setup skipped the chain entirely and raised the misleading
    "No LLM provider configured" RuntimeError (2026-08-23 outage).
    """
    fb = _mock_client(api_key="local-key-1234567890",
                      base_url="http://127.0.0.1:11434/v1")

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if provider == "custom" and explicit_base_url:
            return fb, "qwen3.5:4b"
        return None, None  # openrouter pool exhausted

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider="openrouter",
            model="z-ai/glm-5.2:free",
            api_key=None,
            base_url=None,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[
                {"provider": "openrouter", "model": "poolside/laguna-s-2.1:free"},
                {"provider": "custom", "model": "qwen3.5:4b",
                 "base_url": "http://127.0.0.1:11434/v1"},
            ],
        )
        assert agent.provider == "custom"
        assert agent.model == "qwen3.5:4b"
        assert agent._fallback_activated is True


def test_init_openrouter_exhausted_without_chain_keeps_generic_error():
    """openrouter exhausted + no usable fallback keeps the generic
    'No LLM provider configured' error (not the named-provider one)."""
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            AIAgent(
                provider="openrouter",
                model="z-ai/glm-5.2:free",
                api_key=None,
                base_url=None,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[{"provider": "openrouter",
                                 "model": "poolside/laguna-s-2.1:free"}],
            )


@pytest.mark.parametrize("primary_provider", ["auto", "openrouter", "custom"])
def test_init_tries_fallback_for_every_exempt_primary(primary_provider):
    """The three provider names exempted by the ``_explicit not in
    {auto, openrouter, custom}`` guard must ALL reach fallback_providers at
    init when the primary resolves no credentials.

    The guard exists to choose an error MESSAGE (named provider vs generic),
    so nesting the recovery chain inside it made recovery depend on which
    name the seat happened to use. ``custom`` is the arm that matters here:
    this fleet's seats are configured as ``custom:<endpoint>``, which
    resolves to bare ``custom``, so before the fix a seat whose primary
    endpoint was unreachable died in __init__ with "No LLM provider
    configured" and never attempted a single declared rung.
    """
    fb = _mock_client(api_key="rung-key-1234567890",
                      base_url="https://rung.example.com/api/v1")

    def fake_resolve(provider, model=None, raw_codex=False,
                     explicit_base_url=None, explicit_api_key=None):
        if explicit_base_url == "https://rung.example.com/api/v1":
            return fb, "rung-model"
        return None, None  # primary resolves nothing

    with patch("agent.auxiliary_client.resolve_provider_client", side_effect=fake_resolve), \
         patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()), \
         patch("run_agent.check_toolset_requirements", return_value={}), \
         patch("run_agent.OpenAI", return_value=MagicMock()):

        agent = AIAgent(
            provider=primary_provider,
            model="primary-model",
            api_key=None,
            base_url="https://primary.example.com/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=[
                {"provider": "custom", "model": "rung-model",
                 "base_url": "https://rung.example.com/api/v1"},
            ],
        )
        assert agent._fallback_activated is True
        assert agent.provider == "custom"
        assert agent.model == "rung-model"
