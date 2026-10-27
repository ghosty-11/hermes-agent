"""Behavior contracts for the coordinated-update runtime/config merge seams (RED first).

Four findings from the independent runtime-config review
(``coordinated-20261025/independent-astra-reviews/runtime-config/result.json``);
Main runs this file RED on the unchanged framework before any product edit:

1. ``hermes_cli/middleware.py:67-70`` — ``_apply_request_chain`` imports
   ``get_plugin_manager`` but still calls ``invoke_middleware``, so every registered
   ``llm_request``/``tool_request`` middleware raises ``NameError`` and the callers
   (``agent/turn_api_request.py:140-157``, the tool pipeline) fail open with the
   unmodified payload. The registered transform must actually apply and chain.
2. ``pm/platform_requirements.py:74-78`` — the strict bundled inventory requires
   ``platform_dependencies`` in every platform manifest; the six incoming platforms
   (bluebubbles, qqbot, signal, weixin, whatsapp_cloud, yuanbao) lack it, so strict
   inspection ValueError-refuses even an unrelated Discord configuration. Both the
   declarations AND the passive configuration predicates
   (``gateway/platform_configuration.py:plugin_configured`` has no branch for the new
   platforms) must exist for a configured new platform to be demanded.
3. ``plugins/memory/holographic/__init__.py:18-19`` — the bundled package imports
   ``.store``/``.retrieval`` it does not ship, and ``plugins.memory.find_provider_dir``
   prefers the bundled directory over a complete user installation, so a configured
   Holographic provider resolves to an unimportable package.
4. ``tools/session_search_tool.py:785-792`` + ``743`` — ``session_search`` opens the
   foreign profile DB before ``_dispatch`` applies the companion restriction, and
   ``_dispatch`` then references ``owned_dbs``, which exists only in ``session_search``'s
   local scope, so every valid unrestricted cross-profile search raises ``NameError``.

The restricted-surface tests use a fail-on-foreign-open sentinel (not the permissive
recorder of ``tests/tools/test_session_search_companion_scope.py``) so a pre-gate open
is observable as a leaked lookup instead of being masked by a valid handle. All
storage is fixture-local; no provider, network, or live-home access.
"""

import json
from contextvars import ContextVar
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from hermes_cli import plugins
from hermes_cli.middleware import apply_llm_request_middleware, apply_tool_request_middleware
from hermes_cli.plugins import PluginManager
from pm.platform_requirements import PlatformRequirements, configured_platform_requirements
from plugins.memory import find_provider_dir, load_memory_provider
from tools import session_search_tool as sst
from tools.session_search_tool import session_search

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]


# =========================================================================
# 1. Registered request middleware must transform, not fail open
# =========================================================================

_MIDDLEWARE_PLUGIN = """
SEEN = {}

def register(ctx):
    def llm_first(**kw):
        return {"request": {**kw["request"], "steps": ["first"]}, "source": "first"}

    def llm_second(**kw):
        SEEN["llm"] = kw["request"]
        return {"request": {**kw["request"],
                            "steps": kw["request"]["steps"] + ["second"]},
                "source": "second"}

    def tool_jail(**kw):
        SEEN["tool"] = (kw["tool_name"], kw["args"])
        return {"args": {**kw["args"], "root": "/fixture/jail"}, "source": "fixture-jail"}

    ctx.register_middleware("llm_request", llm_first)
    ctx.register_middleware("llm_request", llm_second)
    ctx.register_middleware("tool_request", tool_jail)
"""


class TestRegisteredRequestMiddleware:
    @pytest.fixture
    def seen(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        plugin = home / "plugins" / "request_chain"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: request_chain\nversion: 0.1.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(_MIDDLEWARE_PLUGIN, encoding="utf-8")
        (home / "config.yaml").write_text("plugins:\n  enabled: [request_chain]\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        manager = PluginManager()
        manager.discover_and_load()
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
        return manager._plugins["request_chain"].module.SEEN

    def test_registered_llm_request_middleware_chain_transforms_provider_request(self, seen):
        request = {"model": "base-model", "messages": []}

        result = apply_llm_request_middleware(request)

        # The second registered callback sees the first one's rewrite: the chain feeds
        # the effective payload forward, and the result is what replaces the provider
        # kwargs instead of the request failing open unmodified.
        assert seen["llm"] == {"model": "base-model", "messages": [], "steps": ["first"]}
        assert result.payload == {"model": "base-model", "messages": [], "steps": ["first", "second"]}
        assert [entry["source"] for entry in result.trace] == ["first", "second"]
        assert result.changed is True
        assert result.original_payload == {"model": "base-model", "messages": []}

    def test_registered_tool_request_middleware_transforms_tool_arguments(self, seen):
        result = apply_tool_request_middleware("read_file", {"path": "README.md"})

        assert seen["tool"] == ("read_file", {"path": "README.md"})
        assert result.payload == {"path": "README.md", "root": "/fixture/jail"}
        assert [entry["source"] for entry in result.trace] == ["fixture-jail"]
        assert result.changed is True
        assert result.original_payload == {"path": "README.md"}


# =========================================================================
# 2. Strict platform inventory over the bundled tree
# =========================================================================

class TestStrictPlatformInventory:
    def test_strict_inventory_admits_unrelated_discord_configuration(self, tmp_path):
        """A Discord-only home must pass strict dependency inspection on the bundled
        tree as shipped. Today the inventory refuses on the first incoming platform
        manifest that lacks ``platform_dependencies`` (bluebubbles, sorted first)."""
        required = configured_platform_requirements(
            {"platforms": {"discord": {"enabled": True}}},
            {"DISCORD_BOT_TOKEN": "fixture-bot-token"}.get,
            home=tmp_path, source_root=FRAMEWORK_ROOT, selected_paths=(),
        )

        assert required["discord"] == PlatformRequirements(
            extras=("discord",), env=("DISCORD_BOT_TOKEN",))

    def test_declared_new_platform_requirements_match_manifest(self, tmp_path):
        """An enabled, credentialed incoming platform must be demanded with exactly the
        requirements its own manifest declares. This stays RED after a declaration-only
        repair: ``plugin_configured`` still refuses the new platforms with
        ``unsupported passive platform configuration: signal`` until their passive
        predicates exist."""
        manifest = yaml.safe_load(
            (FRAMEWORK_ROOT / "plugins" / "platforms" / "signal" / "plugin.yaml").read_text(encoding="utf-8"))
        declarations = manifest.get("platform_dependencies")
        assert isinstance(declarations, dict) and "signal" in declarations, (
            "signal must declare platform_dependencies keyed by its Platform value")
        declaration = declarations["signal"]

        required = configured_platform_requirements(
            {"platforms": {"signal": {"enabled": True}}},
            {"SIGNAL_HTTP_URL": "http://127.0.0.1:8080", "SIGNAL_ACCOUNT": "+15551234567"}.get,
            home=tmp_path, source_root=FRAMEWORK_ROOT, selected_paths=(),
        )

        assert required["signal"] == PlatformRequirements(
            extras=tuple(declaration["extras"]), env=tuple(declaration["env"]))


# =========================================================================
# 3. A complete user Holographic installation must not be shadowed
# =========================================================================

_USER_HOLOGRAPHIC = """
from agent.memory_provider import MemoryProvider


class HolographicUserProvider(MemoryProvider):
    name = "holographic"

    def is_available(self):
        return True

    def initialize(self, *a, **kw):
        self.initialized = True

    def get_tool_schemas(self):
        return []


def register(ctx):
    ctx.register_memory_provider(HolographicUserProvider())
"""


class TestHolographicProviderResolution:
    def test_configured_holographic_resolves_complete_user_install(self, tmp_path, monkeypatch):
        """``memory.provider: holographic`` with a complete user-installed provider must
        resolve to that installation. The bundled ``plugins/memory/holographic/`` ships
        only ``__init__.py`` while importing ``.store``/``.retrieval``, and bundled
        resolution wins over the user directory — so the configured provider resolves to
        an unimportable package and loads None."""
        home = tmp_path / "home"
        provider_dir = home / "plugins" / "holographic"
        provider_dir.mkdir(parents=True)
        (provider_dir / "plugin.yaml").write_text(
            "name: holographic-manifest\nkind: exclusive\n", encoding="utf-8")
        (provider_dir / "__init__.py").write_text(_USER_HOLOGRAPHIC, encoding="utf-8")
        (home / "config.yaml").write_text(
            "memory:\n  provider: holographic\nplugins:\n  disabled: []\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))

        token = set_hermes_home_override(home)
        try:
            assert find_provider_dir("holographic") == provider_dir
            provider = load_memory_provider("holographic")
        finally:
            reset_hermes_home_override(token)

        assert provider is not None
        assert provider.name == "holographic"
        assert provider.is_available()


# =========================================================================
# 4. Cross-profile session search: privacy gate and unrestricted route
# =========================================================================

def _seed(db):
    """s_root (compression ancestor) <- s_current; s_foreign is another conversation."""
    db.create_session("s_root", source="cli")
    db.append_message("s_root", role="user", content="wombat kumquat harvest planning")
    db.append_message("s_root", role="assistant", content="wombat kumquat rows mapped")
    db.end_session("s_root", "compression")
    db.create_session("s_current", source="cli", parent_session_id="s_root")
    db.append_message("s_current", role="user", content="wombat lantern fish question")
    db.create_session("s_foreign", source="cli")
    db.append_message("s_foreign", role="user", content="wombat zeppelin ticket refund please")


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


@pytest.fixture
def companion(monkeypatch):
    """Bind the ambient identity attestation + discord session vars for one turn."""
    from gateway import session_context

    identity = ContextVar("fixture_contract_ambient_identity", default=None)
    monkeypatch.setattr(session_context, "_ambient_turn_identity", identity, raising=False)
    monkeypatch.setattr(sst, "_profile_restricted", lambda: True)
    tokens = []

    def bind(message_id="m-verified", user_id="111", attested=True):
        values = {
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_MESSAGE_ID": message_id,
            "HERMES_SESSION_USER_ID": user_id,
        }
        for name, value in values.items():
            tokens.append(session_context._VAR_MAP[name].set(value))
        tokens.append(identity.set((message_id, user_id) if attested else None))

    yield bind
    while tokens:
        token = tokens.pop()
        token.var.reset(token)


class TestCompanionCrossProfileBoundary:
    def test_restricted_surface_refuses_profile_param_before_foreign_open(
            self, db, companion, monkeypatch):
        _seed(db)
        companion()
        attempts = []

        def refuse_foreign_open(profile):
            attempts.append(profile)
            raise AssertionError(
                f"foreign profile '{profile}' state.db opened before the companion refusal")

        monkeypatch.setattr(sst, "_resolve_profile_db", refuse_foreign_open)

        result = json.loads(session_search(
            db=db, current_session_id="s_current", profile="operator", session_id="s_other"))

        assert result["success"] is False
        assert "cross-profile" in result["error"], result["error"]
        assert attempts == [], f"foreign profile lookup/open leaked before refusal: {attempts}"

    def test_restricted_surface_refuses_profile_link_before_foreign_open(
            self, db, companion, monkeypatch):
        _seed(db)
        companion()
        attempts = []

        def refuse_foreign_open(profile):
            attempts.append(profile)
            raise AssertionError(
                f"foreign profile '{profile}' state.db opened before the companion refusal")

        monkeypatch.setattr(sst, "_resolve_profile_db", refuse_foreign_open)

        result = json.loads(session_search(
            db=db, current_session_id="s_current", session_id="@session:operator/s_other"))

        assert result["success"] is False
        assert "cross-profile" in result["error"], result["error"]
        assert attempts == [], f"foreign profile lookup/open leaked before refusal: {attempts}"

    def test_unrestricted_cross_profile_discovery_returns_known_message_and_closes(
            self, db, monkeypatch, tmp_path):
        """Off the restricted surface a cross-profile discovery must return the foreign
        conversation's known message, resolve the named profile through exactly one
        path, and leave no foreign handle open behind the call."""
        monkeypatch.setattr(sst, "_profile_restricted", lambda: False)
        other = SessionDB(tmp_path / "other-state.db")
        other.create_session("s_other", source="cli")
        other.append_message("s_other", role="user", content="wombat zeppelin ticket refund confirmed")
        opens = []

        def record_foreign_open(profile):
            opens.append(profile)
            return other

        monkeypatch.setattr(sst, "_resolve_profile_db", record_foreign_open)
        try:
            result = json.loads(session_search(db=db, query="zeppelin", profile="companion"))
            assert result["success"] is True
            assert result["count"] >= 1
            assert [entry["session_id"] for entry in result["results"]] == ["s_other"]
            assert "zeppelin" in json.dumps(result["results"]).lower()
            assert opens == ["companion"]
            assert other._conn is None, "session_search must close the foreign DB it opened"
        finally:
            other.close()  # idempotent; the sweep owns any leak this test exposes
