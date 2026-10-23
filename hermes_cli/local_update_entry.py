"""Run native local update from the guard's frozen, verified commit snapshot.

Usage: managed-python local_update_entry.py INSTALL_ROOT HERMES_HOME SOURCE_SNAPSHOT SHA [EXTRA...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit("expected installed root, Hermes home, frozen source and target")
    installed, home, snapshot = (Path(value).resolve() for value in sys.argv[1:4])
    target = sys.argv[4]
    extras = sys.argv[5:]
    rollback = None
    if extras[:1] == ["--rollback-local-update"]:
        if len(extras) != 3 or extras[2] != "--acknowledge-data-rollback":
            raise SystemExit("recovery requires a manifest and explicit data-rollback acknowledgment")
        rollback, extras = extras[1], []
    if snapshot != Path(__file__).resolve().parents[1]:
        raise RuntimeError("entry is not running from the selected snapshot")
    # Set frozen imports and use the same protected Git policy as the apply path.
    # Imports before the clean-tree probe must not leave untracked bytecode in the snapshot.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(snapshot))
    from hermes_cli.update_local import local_git, verify_closure

    if not installed.is_dir() or not home.is_dir():
        raise RuntimeError("installed root or Hermes home is unavailable")
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_INSTALL_ROOT"] = str(installed)
    os.environ["HERMES_LOCAL_UPDATE_CONTEXT"] = "1"
    from pm.writer_claim import writer_claim
    with writer_claim(install=installed, inherited_only=True):
        if (local_git(snapshot, "rev-parse", "HEAD").stdout.strip() != target
                or local_git(snapshot, "status", "--porcelain=v1", "--untracked-files=all").stdout):
            raise RuntimeError("frozen snapshot changed after staging")
        verify_closure(snapshot, target)
        from pm.environments import activate_dependencies
        activate_dependencies(installed)
        sys.path.insert(0, str(snapshot))
        import hermes_bootstrap
        from hermes_cli import main as native
        native.PROJECT_ROOT = installed  # must precede retarget_to_owning_install
        try:
            native.cmd_update(argparse.Namespace(
                source_repo=None if rollback else str(snapshot), target=None if rollback else target,
                required_extra=extras, rollback_local_update=rollback,
                acknowledge_data_rollback=bool(rollback), yes=True, backup=not rollback,
                no_gateway_restart=True, gateway=False, check=False, plan=False,
                list_venv_holders=False, no_backup=False, keep_stash=False,
                branch=None, switch_branch=False, force=False, force_venv=False,
                channel=None, set_channel=None, install_id=False))
        except Exception as exc:
            from pm.process_scope import ProcessStillRunning
            if isinstance(exc, ProcessStillRunning):
                marker = os.environ.get("HERMES_LOCAL_RETAINED_STATUS")
                if marker and Path(marker) == snapshot.parent / "not-quiescent.json":
                    import json
                    Path(marker).write_text(
                        json.dumps([str(path) for path in exc.retained_paths]), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
