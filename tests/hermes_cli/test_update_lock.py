"""Cooperating writers retain the installation claim through completion."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli.update_lock import UpdateLock

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.platforms("posix")
def test_one_checkout_excludes_other_profile_even_after_marker_age(tmp_path, monkeypatch):
    install = tmp_path / "checkout"
    install.mkdir()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(a))
    first = UpdateLock(install_root=install)
    assert first.acquire()
    try:
        marker = b / ".hermes-update-in-progress"
        # A different profile has no marker; its real installation claim still collides.
        monkeypatch.setenv("HERMES_HOME", str(b))
        child = (
            "import sys; from pathlib import Path; "
            f"sys.path.insert(0, {str(REPO)!r}); "
            "from hermes_cli.update_lock import UpdateLock; "
            f"lock=UpdateLock(install_root=Path({str(install)!r})); "
            "\ntry: won=lock.acquire()\nexcept RuntimeError: won=False\n"
            "raise SystemExit(1 if won else 0)"
        )
        result = subprocess.run([sys.executable, "-I", "-S", "-c", child],
                                env={**os.environ, "HERMES_HOME": str(b)}, capture_output=True,
                                text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        assert not marker.exists()
        # The PID marker is only status: aging it never releases the kernel claim.
        (a / ".hermes-update-in-progress").write_text(f"{os.getpid()}\n1\n")
        assert subprocess.run([sys.executable, "-I", "-S", "-c", child],
                              env={**os.environ, "HERMES_HOME": str(b)},
                              capture_output=True, text=True, timeout=15).returncode == 0
    finally:
        first.release()
    with UpdateLock(install_root=install) as acquired:
        assert acquired.acquired


@pytest.mark.platforms("posix")
def test_inherited_descriptor_keeps_parent_status_and_exclusion(tmp_path, monkeypatch):
    from pm.writer_claim import child_spawn

    install, home = tmp_path / "checkout", tmp_path / "home"
    install.mkdir()
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    child = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0, {str(REPO)!r}); "
        "from hermes_cli.update_lock import UpdateLock; "
        f"lock=UpdateLock(install_root=Path({str(install)!r})); "
        "assert lock.acquire(); lock.release()"
    )
    with UpdateLock(install_root=install):
        environment = dict(os.environ)
        with child_spawn(environment) as spawn:
            result = subprocess.run([sys.executable, "-I", "-S", "-c", child],
                                    env=environment, capture_output=True, text=True,
                                    timeout=15, **spawn)
        assert result.returncode == 0, result.stderr
        assert (home / ".hermes-update-in-progress").read_text().splitlines()[0] == str(os.getpid())
    assert not (home / ".hermes-update-in-progress").exists()


def test_unwritable_status_refuses_even_without_other_writer(tmp_path):
    install = tmp_path / "checkout"
    install.mkdir()
    marker = tmp_path / "occupied"
    marker.mkdir()
    lock = UpdateLock(path=marker, install_root=install)
    with pytest.raises(OSError):
        lock.acquire()
    assert not lock.acquired


@pytest.mark.platforms("posix")
def test_symlinked_claim_refuses_before_status_publication(tmp_path):
    from pm.writer_claim import domains

    install = tmp_path / "checkout"
    install.mkdir()
    (tmp_path / "other").write_text("not a writer domain")
    domains(install)[0].parent.mkdir(parents=True, exist_ok=True)
    domains(install)[0].symlink_to(tmp_path / "other")
    lock = UpdateLock(path=tmp_path / "status", install_root=install)
    with pytest.raises(OSError):
        lock.acquire()
    assert not (tmp_path / "status").exists()
