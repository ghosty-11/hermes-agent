"""Offline admission must not preserve a stale enabled renderer against a new backend."""
from pathlib import Path

import pytest

from hermes_cli.source_build import validate_offline_products

pytestmark = pytest.mark.platforms("posix")


def _renderer_case(tmp_path, monkeypatch, *, enabled):
    from pm import plugins_state

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        "webapp:\n  enabled: " + ("true" if enabled else "false") + "\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_state, "dependency_homes", lambda: [home])
    source = tmp_path / "source"
    installed = tmp_path / "installed"
    source.mkdir()
    output = installed / "apps/desktop/dist-webapp"
    output.mkdir(parents=True)
    renderer = output / "index.html"
    renderer.write_text("<main>previous renderer</main>\n")
    return source, installed, renderer


def test_enabled_webapp_without_target_receipt_is_refused_read_only(tmp_path, monkeypatch):
    source, installed, renderer = _renderer_case(tmp_path, monkeypatch, enabled=True)
    before = renderer.read_bytes()
    with pytest.raises(RuntimeError, match="(?i)webapp"):
        validate_offline_products(source, installed)
    assert renderer.read_bytes() == before
    assert not (renderer.parent / ".hermes-stale").exists()


def test_disabled_webapp_does_not_require_node_preparation(tmp_path, monkeypatch):
    source, installed, renderer = _renderer_case(tmp_path, monkeypatch, enabled=False)
    admitted = validate_offline_products(source, installed)
    assert admitted["workspaces"] == ()
    assert admitted["products"] == ()
    assert renderer.read_text() == "<main>previous renderer</main>\n"
