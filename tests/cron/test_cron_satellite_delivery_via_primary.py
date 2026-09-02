"""Downstream knob: satellite cron delivery fronted by the primary gateway.

Upstream a07370abd0 ("fix(cron): reserve shared adapters for the default
profile only") assumes every multiplexed profile owns its own bot. On a
shared-bot topology that assumption is false: the satellite profiles are
deliberately credential-less, because issuing them the default bot's token is
a ``duplicate_credential`` fatal, and the default bot has always fronted their
cron output. ``_preflight_check_delivery`` loads the gateway config of the
job's OWN home, where such a platform correctly reads as unconnected, so from
a07370abd0 onwards every satellite cron job was refused ``blocked_config``
before any LLM call.

``gateway.satellite_cron_delivery_via_primary`` (bool, default false — upstream
behaviour is unchanged when unset) declares that topology. Read from the
PRIMARY home's config.yaml, exactly like ``profile_routes`` is in
``_delivery_platform_routed_from_primary_gateway``, and independently of it:
the routes mechanism rescues one platform routed to one profile, the knob
covers a deployment where the primary's adapters front every satellite.
"""

from unittest.mock import MagicMock, patch

import pytest
import yaml

from cron.scheduler import (
    _preflight_check_delivery,
    _satellite_delivery_via_primary_enabled,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _gateway_config(connected_values):
    config = MagicMock()
    config.get_connected_platforms.return_value = [
        MagicMock(value=v) for v in connected_values
    ]
    return config


def _build_homes(tmp_path, monkeypatch, primary_yaml):
    """A primary root (optionally carrying ``primary_yaml``) plus a satellite
    home under it, with ``get_default_hermes_root`` pointed at the root."""
    root = tmp_path / "root"
    satellite_home = root / "profiles" / "thoth"
    satellite_home.mkdir(parents=True)
    if primary_yaml is not None:
        (root / "config.yaml").write_text(
            yaml.safe_dump(primary_yaml), encoding="utf-8"
        )
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    return root, satellite_home


@pytest.fixture
def satellite_home(tmp_path, monkeypatch):
    """Factory: build the homes for a given primary config and serve the
    satellite home, exactly as the multiplex ticker does per profile."""
    tokens = []

    def _make(primary_yaml, *, serve_root=False):
        root, satellite = _build_homes(tmp_path, monkeypatch, primary_yaml)
        tokens.append(set_hermes_home_override(str(root if serve_root else satellite)))
        return root, satellite

    yield _make
    for token in reversed(tokens):
        reset_hermes_home_override(token)


_NESTED_ON = {"gateway": {"multiplex_profiles": True,
                          "satellite_cron_delivery_via_primary": True}}


class TestSatelliteDeliveryViaPrimaryPreflight:
    def test_knob_on_lets_unconnected_satellite_platform_through(self, satellite_home):
        """The live case: thoth has no discord credentials of its own, the knob
        is on in the primary's config — preflight must not block."""
        satellite_home(_NESTED_ON)
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config(set())):
            assert _preflight_check_delivery({"deliver": "discord:12345"}) is None

    def test_knob_absent_preserves_the_block(self, satellite_home):
        """Upstream default: the satellite's own unconnected reading blocks."""
        satellite_home({"gateway": {"multiplex_profiles": True}})
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config(set())):
            reason = _preflight_check_delivery({"deliver": "discord:12345"})
            assert reason is not None
            assert "discord" in reason

    def test_knob_explicitly_false_preserves_the_block(self, satellite_home):
        satellite_home(
            {"gateway": {"satellite_cron_delivery_via_primary": False}}
        )
        with patch("gateway.config.load_gateway_config",
                   return_value=_gateway_config(set())):
            reason = _preflight_check_delivery({"deliver": "discord:12345"})
            assert reason is not None
            assert "discord" in reason

    def test_unknown_platform_still_blocks_under_the_knob(self, satellite_home):
        """The knob speaks to credentials, never to a bogus deliver target."""
        satellite_home(_NESTED_ON)
        with patch("cron.scheduler._is_known_delivery_platform",
                   return_value=False):
            reason = _preflight_check_delivery({"deliver": "nonexistent-platform"})
            assert reason is not None
            assert "not a known" in reason


class TestSatelliteDeliveryViaPrimaryEnabled:
    def test_nested_gateway_form(self, satellite_home):
        satellite_home(_NESTED_ON)
        assert _satellite_delivery_via_primary_enabled() is True

    def test_top_level_form(self, satellite_home):
        """Same key written at the top level of config.yaml is honored too."""
        satellite_home({"satellite_cron_delivery_via_primary": True})
        assert _satellite_delivery_via_primary_enabled() is True

    def test_primary_home_itself_is_never_rescued(self, satellite_home):
        """Serving the primary home: it owns the adapters, so there is nothing
        to borrow and a missing credential there is a real block."""
        satellite_home(_NESTED_ON, serve_root=True)
        assert _satellite_delivery_via_primary_enabled() is False

    def test_missing_primary_config_fails_closed(self, satellite_home):
        satellite_home(None)
        assert _satellite_delivery_via_primary_enabled() is False

    def test_unparseable_primary_config_fails_closed(self, tmp_path, monkeypatch):
        root = tmp_path / "root"
        satellite = root / "profiles" / "thoth"
        satellite.mkdir(parents=True)
        (root / "config.yaml").write_text("gateway: [not, a, mapping\n",
                                          encoding="utf-8")
        monkeypatch.setattr("hermes_constants.get_default_hermes_root",
                            lambda: root)
        token = set_hermes_home_override(str(satellite))
        try:
            assert _satellite_delivery_via_primary_enabled() is False
        finally:
            reset_hermes_home_override(token)

    def test_non_bool_truthy_value_enables(self, satellite_home):
        """``hermes config set`` coerces types — a stringy 'true' must not read
        as unset. Anything YAML resolves to a bool is honored as written."""
        satellite_home({"gateway": {"satellite_cron_delivery_via_primary": "yes"}})
        assert _satellite_delivery_via_primary_enabled() is True
