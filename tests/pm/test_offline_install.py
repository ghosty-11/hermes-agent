"""Cache-only PM acquisition exercises the real archive publisher."""

import shutil

import pytest

from pm.lock import Lockfile
from pm.package import InstallError
from pm.store import Store, current_target
from tests.pm._fixtures import served as served  # noqa: F401
from tests.pm.test_pm_core import pm_env as pm_env  # noqa: F401


def test_offline_tool_refuses_cold_cache_before_downloader_and_installs_cached_archive(pm_env, monkeypatch):
    from pm.downloader import Download
    from pm.install import ensure, is_installed
    from pm.paths import store_root

    lock_path, _runtime, docroot, _base_url = pm_env
    artifact = Lockfile(lock_path).artifacts("faketool", current_target())[0]
    requests = []

    def no_requests(self, *args, **kwargs):
        requests.append(self)
        raise AssertionError("offline acquisition entered the downloader")

    monkeypatch.setattr(Download, "run", no_requests)
    with pytest.raises(InstallError, match="offline|cache"):
        ensure("faketool", explicit=True, offline=True, base_env={})
    assert not requests
    assert not is_installed("faketool")

    entry = Store(store_root()).entry(f"fetch-{artifact['sha256']}")
    entry.mkdir(parents=True)
    shutil.copy2(docroot / "faketool-1.0.tar.gz", entry / "faketool-1.0.tar.gz")
    ensure("faketool", explicit=True, offline=True, base_env={})
    assert is_installed("faketool")
    from pm.install import installed_package
    assert (installed_package("faketool").path / "bin" / "faketool").read_bytes() == b"#!x"
    assert not requests
