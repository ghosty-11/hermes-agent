"""Tests for plugin secret-source first-process re-pull (#64177)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION,
    FetchResult,
    SecretSource,
)
from hermes_cli.plugins import PluginManager


class _StubSource(SecretSource):
    """Minimal spec-compliant plugin source for tests."""

    api_version = SECRET_SOURCE_API_VERSION
    shape = "bulk"

    def __init__(self, name: str = "myvault", scheme: str | None = None):
        self.name = name
        self.scheme = scheme

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        return FetchResult(secrets={})


class _CustomActivationSource(_StubSource):
    """Ignores ``enabled`` and activates when a custom key is present."""

    def is_enabled(self, cfg: dict) -> bool:
        return bool(isinstance(cfg, dict) and cfg.get("vault_id"))


def test_refresh_secret_sources_noop_without_plugin_sources(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_plugin_sources", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


def test_refresh_secret_sources_noop_when_only_builtins(monkeypatch):
    """Bundled sources must never trigger a re-pull."""
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    reg._reset_registry_for_tests()
    assert reg.list_plugin_sources() == []
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"bitwarden": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


def test_refresh_secret_sources_repulls_when_plugin_enabled(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_plugin_sources", lambda: [_StubSource()])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 1, "load": 1}


def test_refresh_respects_custom_is_enabled(monkeypatch):
    """A source with custom activation (no ``enabled`` key) is re-pulled."""
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(
        reg, "list_plugin_sources", lambda: [_CustomActivationSource()]
    )
    # No `enabled` key at all — only the source's custom contract decides.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"vault_id": "abc123"}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 1, "load": 1}


def test_refresh_skips_custom_source_when_not_activated(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(
        reg, "list_plugin_sources", lambda: [_CustomActivationSource()]
    )
    # `enabled: true` but the custom contract ignores it and requires vault_id.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}


def test_refresh_skips_source_whose_is_enabled_raises(monkeypatch):
    mgr = PluginManager()
    called = {"reset": 0, "load": 0}

    class _Boom(_StubSource):
        def is_enabled(self, cfg: dict) -> bool:
            raise RuntimeError("boom")

    import agent.secret_sources.registry as reg

    monkeypatch.setattr(reg, "list_plugin_sources", lambda: [_Boom()])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"secrets": {"myvault": {"enabled": True}}},
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.reset_secret_source_cache",
        lambda *a, **kw: called.__setitem__("reset", called["reset"] + 1),
    )
    monkeypatch.setattr(
        "hermes_cli.env_loader.load_hermes_dotenv",
        lambda **kw: called.__setitem__("load", called["load"] + 1),
    )

    mgr._refresh_secret_sources_after_discovery()
    assert called == {"reset": 0, "load": 0}




def test_real_plugin_source_discovery_applies_dotenv(monkeypatch, tmp_path):
    """A cold process discovers a real plugin and applies its credential."""
    import agent.secret_sources.registry as reg
    from hermes_cli import env_loader

    reg._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    home = tmp_path / ".hermes"
    plugin_dir = home / "plugins" / "fixture-secret-source"
    plugin_dir.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "plugins:\n"
        "  enabled: [fixture-secret-source]\n"
        "secrets:\n"
        "  fixturevault:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: fixture-secret-source\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from agent.secret_sources.base import (\n"
        "    SECRET_SOURCE_API_VERSION, FetchResult, SecretSource,\n"
        ")\n\n"
        "class FixtureVault(SecretSource):\n"
        "    name = 'fixturevault'\n"
        "    label = 'Fixture vault'\n"
        "    api_version = SECRET_SOURCE_API_VERSION\n"
        "    shape = 'bulk'\n\n"
        "    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:\n"
        "        return FetchResult(secrets={\n"
        "            'HERMES_TEST_PLUGIN_BOOTSTRAP': 'from-plugin',\n"
        "        })\n\n"
        "def register(ctx):\n"
        "    ctx.register_secret_source(FixtureVault())\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TEST_PLUGIN_BOOTSTRAP", raising=False)

    try:
        PluginManager().discover_and_load()

        assert os.environ["HERMES_TEST_PLUGIN_BOOTSTRAP"] == "from-plugin"
        assert [source.name for source in reg.list_plugin_sources()] == [
            "fixturevault"
        ]
    finally:
        os.environ.pop("HERMES_TEST_PLUGIN_BOOTSTRAP", None)
        reg._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


_FIXTURE_PLUGIN = (
    "from pathlib import Path\n"
    "from agent.secret_sources.base import (\n"
    "    SECRET_SOURCE_API_VERSION, FetchResult, SecretSource,\n"
    ")\n\n"
    "class FixtureVault(SecretSource):\n"
    "    name = 'fixturevault'\n"
    "    label = 'Fixture vault'\n"
    "    api_version = SECRET_SOURCE_API_VERSION\n"
    "    shape = 'bulk'\n\n"
    "    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:\n"
    "        return FetchResult(secrets={'REVIEW_PLUGIN_SECRET': 'from-plugin'})\n\n"
    "def register(ctx):\n"
    "    ctx.register_secret_source(FixtureVault())\n"
)


def _write_plugin_home(home: Path, *, plugin_enabled: bool) -> None:
    plugin_dir = home / "plugins" / "fixture-secret-source"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text("name: fixture-secret-source\nversion: 0.1.0\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_FIXTURE_PLUGIN, encoding="utf-8")
    enabled = "[fixture-secret-source]" if plugin_enabled else "[]"
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: {enabled}\nsecrets:\n  fixturevault:\n    enabled: true\n", encoding="utf-8")


@pytest.mark.parametrize("replaced_by_other_owner", [False, True], ids=["revoked", "other-owner-kept"])
def test_removing_the_last_plugin_source_revokes_its_process_env_value(
        monkeypatch, tmp_path, replaced_by_other_owner):
    """Disabling/removing a plugin secret source takes its value back out of ``os.environ`` (#126982
    review): the per-home snapshot was already rebuilt, but the process env kept the injected value and
    single-profile ``get_secret()`` served it through its ``os.environ`` fallback until restart. A value
    another owner wrote over the plugin's since is theirs and stays. A sibling home (A -> B -> A) never
    has its own view disturbed by the revocation."""
    import agent.secret_sources.registry as reg
    from agent.secret_scope import get_secret
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    reg._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()
    home_a = tmp_path / "a" / ".hermes"
    home_b = tmp_path / "b" / ".hermes"
    _write_plugin_home(home_a, plugin_enabled=True)
    home_b.mkdir(parents=True)
    (home_b / ".env").write_text("SIBLING_ONLY=sibling\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.delenv("REVIEW_PLUGIN_SECRET", raising=False)

    try:
        manager = PluginManager()
        manager.discover_and_load()
        assert os.environ["REVIEW_PLUGIN_SECRET"] == "from-plugin"
        assert get_secret("REVIEW_PLUGIN_SECRET") == "from-plugin"

        # B: a sibling profile's activity in between does not touch A's record or value.
        token = set_hermes_home_override(str(home_b))
        try:
            env_loader.hydrate_profile_secret_sources(home_b)
        finally:
            reset_hermes_home_override(token)
        assert os.environ["REVIEW_PLUGIN_SECRET"] == "from-plugin"

        if replaced_by_other_owner:
            os.environ["REVIEW_PLUGIN_SECRET"] = "from-operator"

        # A: the plugin is disabled; the next discovery unloads its registration and reconciles.
        _write_plugin_home(home_a, plugin_enabled=False)
        manager.discover_and_load(force=True)
        assert reg.list_plugin_sources() == []

        if replaced_by_other_owner:
            assert os.environ["REVIEW_PLUGIN_SECRET"] == "from-operator"
            assert get_secret("REVIEW_PLUGIN_SECRET") == "from-operator"
        else:
            assert "REVIEW_PLUGIN_SECRET" not in os.environ
            assert get_secret("REVIEW_PLUGIN_SECRET") is None
            assert env_loader.get_secret_source("REVIEW_PLUGIN_SECRET") is None
    finally:
        os.environ.pop("REVIEW_PLUGIN_SECRET", None)
        reg._reset_registry_for_tests()
        env_loader.reset_secret_source_cache()


def test_custom_activation_cold_authority_refuses_and_successful_route_stays_cached(
    monkeypatch, tmp_path,
):
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  myvault:\n    vault_id: fixture-vault\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("FIXTURE_CUSTOM_VAULT_TOKEN", raising=False)

    class Source(_CustomActivationSource):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def fetch(self, cfg, home_path):
            self.calls += 1
            assert cfg["vault_id"] == "fixture-vault"
            return FetchResult(secrets={"FIXTURE_CUSTOM_VAULT_TOKEN": "fixture-secret"})

    source = Source()
    previous = snapshot_registration(source.name, scope=str(home))
    assert register_source(source, scope=str(home))
    try:
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
        assert source.calls == 0
        assert env_loader.hydrate_profile_secret_sources(home) == {
            "FIXTURE_CUSTOM_VAULT_TOKEN": "fixture-secret",
        }
        assert env_loader.get_authoritative_secret_source_values(home) == {
            "FIXTURE_CUSTOM_VAULT_TOKEN": "fixture-secret",
        }
        assert env_loader.hydrate_profile_secret_sources(home) == {
            "FIXTURE_CUSTOM_VAULT_TOKEN": "fixture-secret",
        }
        assert source.calls == 1
        assert os.environ.get("FIXTURE_CUSTOM_VAULT_TOKEN") is None
    finally:
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(source.name, source, previous, scope=str(home))

@pytest.mark.parametrize("mode", ["activation-error", "unregistered"])
def test_mixed_custom_activation_cannot_certify_incomplete_evaluation(
    monkeypatch, tmp_path, mode,
):
    from agent.secret_sources.registry import register_source, restore_registration, snapshot_registration
    from hermes_cli import env_loader

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  myvault:\n    vault_id: fixture-vault\n"
        "  fixture_other:\n    enabled: true\n", encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    class Broken(_CustomActivationSource):
        def __init__(self):
            super().__init__()
            self.fails = True

        def is_enabled(self, cfg):
            if self.fails:
                raise RuntimeError("fixture activation failure")
            return super().is_enabled(cfg)

    class Other(_StubSource):
        def __init__(self):
            super().__init__("fixture_other")

        def fetch(self, cfg, home_path):
            return FetchResult(secrets={"FIXTURE_OTHER_TOKEN": "fixture-value"})

    other = Other()
    broken = Broken()
    prior_other = snapshot_registration(other.name, scope=str(home))
    prior_broken = snapshot_registration(broken.name, scope=str(home))
    assert register_source(other, scope=str(home))
    if mode == "activation-error":
        assert register_source(broken, scope=str(home))
    try:
        assert env_loader.hydrate_profile_secret_sources(home) == {
            "FIXTURE_OTHER_TOKEN": "fixture-value",
        }
        with pytest.raises((OSError, RuntimeError, ValueError)):
            env_loader.get_authoritative_secret_source_values(home)
        # An incomplete routed evaluation is retryable, even though another source succeeded.
        assert str(home.resolve()) not in env_loader._APPLIED_HOMES
        broken.fails = False
        if mode == "unregistered":
            assert register_source(broken, scope=str(home))
        assert env_loader.hydrate_profile_secret_sources(home) == {
            "FIXTURE_OTHER_TOKEN": "fixture-value",
        }
        assert env_loader.get_authoritative_secret_source_values(home) == {
            "FIXTURE_OTHER_TOKEN": "fixture-value",
        }
    finally:
        env_loader.reset_secret_source_cache(home)
        assert restore_registration(broken.name, broken, prior_broken, scope=str(home))
        assert restore_registration(other.name, other, prior_other, scope=str(home))
