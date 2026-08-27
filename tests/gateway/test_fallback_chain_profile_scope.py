"""The gateway must honour the per-profile HERMES_HOME override when it
re-reads ``fallback_providers``.

Observed on BarkBox 2026-08-27: a Bastet Discord session was offered ROOT's
fallback rungs (`custom:orcarouter`, `opencode-go`, then metered
`openrouter/google/gemini-3.6-flash`) while resolving providers against
Bastet's own config, which declares none of them. The free rungs failed
"provider not configured" and the turn landed on the metered rung.

Cause: ``_load_gateway_config()`` resolves its path through
``_gateway_config_home()``, which consults ``get_hermes_home_override()`` --
but ``_refresh_fallback_model()`` reads ``_hermes_home / "config.yaml"``
directly. Since that refresh runs on every agent create/reuse (#60955), it
overwrites the chain with root's on every turn of every profile.

The sibling test in ``test_fallback_chain_reload.py`` monkeypatches
``gateway.run._hermes_home``, so it never exercises the override path and
passes either way.
"""

from __future__ import annotations

from types import SimpleNamespace


def _write_chain(path, provider, model):
    path.write_text(
        "fallback_providers:\n"
        f"  - provider: {provider}\n"
        f"    model: {model}\n"
    )


def test_refresh_fallback_model_honours_profile_override(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    root_home = tmp_path / "root"
    profile_home = tmp_path / "profiles" / "bastet"
    root_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)

    # Root names a metered rung; the profile names a free one.
    _write_chain(root_home / "config.yaml", "openrouter", "google/gemini-3.6-flash")
    _write_chain(profile_home / "config.yaml", "nous", "poolside/laguna-s-2.1:free")

    monkeypatch.setattr("gateway.run._hermes_home", root_home)

    runner = SimpleNamespace(_fallback_model=None)
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)

    # Sanity: with no override active, root's chain is correct.
    assert bound() == [
        {"provider": "openrouter", "model": "google/gemini-3.6-flash"}
    ]

    # Inside a profile scope -- exactly what _profile_runtime_scope installs for
    # a multiplexed inbound message -- the PROFILE's chain must win.
    token = set_hermes_home_override(str(profile_home))
    try:
        scoped = bound()
    finally:
        reset_hermes_home_override(token)

    assert scoped == [
        {"provider": "nous", "model": "poolside/laguna-s-2.1:free"}
    ], (
        "profile session inherited root's fallback chain; a stalled primary "
        "will skip the profile's free rungs and reach root's metered rung"
    )
    assert runner._fallback_model == scoped


def test_refresh_fallback_model_scoped_home_without_config(tmp_path, monkeypatch):
    """An override pointing at a home with no config.yaml must not silently
    read root's file."""
    from gateway.run import GatewayRunner
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    root_home = tmp_path / "root"
    empty_home = tmp_path / "profiles" / "empty"
    root_home.mkdir(parents=True)
    empty_home.mkdir(parents=True)
    _write_chain(root_home / "config.yaml", "openrouter", "google/gemini-3.6-flash")

    monkeypatch.setattr("gateway.run._hermes_home", root_home)

    runner = SimpleNamespace(_fallback_model=None)
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)

    token = set_hermes_home_override(str(empty_home))
    try:
        scoped = bound()
    finally:
        reset_hermes_home_override(token)

    # No config.yaml in that home: the documented behaviour for a successful
    # read that genuinely lacks the key is a cleared chain, NOT root's chain.
    assert scoped is None
