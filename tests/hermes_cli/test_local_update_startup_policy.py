"""A failed local transaction never becomes an automatic online source update."""
from hermes_cli import _early_recovery
from hermes_cli._early_recovery import (
    interrupted_pull_marker, local_transaction_marker, recover_if_needed, restore_interrupted_pull,
)
from hermes_cli.venv_sync import prepare_launch


def test_pending_local_transaction_leaves_retry_and_interrupted_pull_for_explicit_recovery(
    tmp_path, monkeypatch,
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    root = tmp_path / "installed"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='hermes-agent'\nversion='1.0'\n")
    (root / ".git").mkdir()
    local = local_transaction_marker(root)
    local.write_text('{"state":"failed"}')
    interrupted = interrupted_pull_marker(root)
    # Recovery checks for the local marker before parsing the ordinary pull marker.
    interrupted.write_text("pid=0\npre=unknown\ntarget=unknown\n")
    retry = root / ".update-incomplete"
    retry.write_text("pid=0\n")
    monkeypatch.setattr(_early_recovery, "_project_root", lambda: root)
    assert _early_recovery._should_skip_external_secret_sources()

    assert prepare_launch(root, ["chat"]) is None
    assert restore_interrupted_pull(root) is False
    assert recover_if_needed(root, argv=["chat"]) is False
    assert local.read_text() == '{"state":"failed"}'
    assert interrupted.read_text() == "pid=0\npre=unknown\ntarget=unknown\n"
    assert retry.read_text() == "pid=0\n"
