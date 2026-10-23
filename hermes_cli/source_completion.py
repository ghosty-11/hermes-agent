"""The tail every source install and update shares.

A fresh install (``scripts/install.sh`` / ``scripts/install.ps1``) and a finished
update must land in the same state: launchers published, product builds current,
and the post-build maintenance (skills sync, config migration) applied. One
implementation, two callers -- an install and an update cannot drift apart.

``main()`` is the installer entry. It is handed the bootstrap interpreter the
installer happens to have (uv, PM and no application dependencies), so it
re-executes itself in PM's selected interpreter before touching any product.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# The installers run this file directly under an isolated interpreter, which
# leaves the checkout off sys.path. Same idiom as hermes_cli/_launchers.py.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Second-phase marker: the re-executed interpreter, not the bootstrap one.
_PREPARED = "--prepared"


def complete_source_checkout(
    root: Path,
    *,
    desktop: bool,
    assume_yes: bool,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
    pre_update_version: str | None = None,
    completion_message: str | None = None,
    announce: str | None = None,
    offline: bool = False,
) -> bool:
    """Publish commands, build the products, then run post-build maintenance.

    Returns the SQLite runtime verdict: a positive unsafe-runtime probe withholds
    success here exactly as it does at the end of an update.
    """
    from hermes_cli.source_build import build_update_products
    from hermes_cli.update_cmd_maint import _run_post_update_maintenance
    from hermes_cli.venv_sync import publish_launchers

    root = Path(root)
    try:
        from hermes_cli._subprocess_compat import expose_pm_git

        # The builds, the release-history refresh and the install stamp all run
        # git; a fresh Windows machine has only PM's.
        expose_pm_git(root)
    except Exception as exc:  # noqa: BLE001 — git-less steps below still complete
        if offline:
            raise RuntimeError(f"Offline completion requires pinned Git: {exc}") from exc
        print(f"⚠ Could not provide git for the source completion: {exc}", file=sys.stderr)
    publish_launchers(root)
    build_update_products(root, desktop=desktop, **({"offline": True} if offline else {}))
    if announce:
        print(announce)
    complete = _run_post_update_maintenance(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
        had_desktop_app_before_update=desktop,
        pre_update_version=pre_update_version,
        completion_message=completion_message,
        **({"offline": True} if offline else {}),
    )
    if complete:
        from hermes_cli.source_stamp import write_source_stamp

        try:
            stamp = write_source_stamp(root)
            if offline and stamp is None:
                raise RuntimeError("Offline completion could not identify the local source commit")
        except (OSError, ValueError) as exc:
            if offline:
                raise RuntimeError(f"Offline completion could not write the source stamp: {exc}") from exc
            print(f"⚠ Source update completed, but the install stamp could not be written: {exc}",
                  file=sys.stderr)
    return complete


def _bootstrap_command(root: Path, argv: list[str]) -> list[str]:
    """Re-enter the checkout on PM's selected interpreter with its environment."""
    from pm.environments import project_python

    return [str(project_python(root)), "-I", "-B", "-u",
            str(Path(__file__).resolve()), "--source", str(root), *argv, _PREPARED]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    prepared = _PREPARED in argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--desktop", action="store_true",
                        help="Also build the packaged desktop app.")
    parser.add_argument("--interactive", action="store_true",
                        help="Allow prompts; installers pass no flag and stay unattended.")
    parser.add_argument("--finish-update", action="store_true",
                        help="Complete a source UPDATE tail instead of an install: the "
                             "same launchers/products/maintenance with update wording.")
    args = parser.parse_args([argument for argument in argv if argument != _PREPARED])
    root = args.source.resolve()
    if not (root / "hermes_cli/source_completion.py").is_file():
        print(f"✗ {root} is not a Hermes source checkout", file=sys.stderr)
        return 1

    from pm.writer_claim import writer_claim
    with writer_claim(install=root):
        return _main_owned(args, root, prepared)


def _main_owned(args, root: Path, prepared: bool) -> int:
    if prepared:
        # Dependencies are selected before any application import.
        from pm.environments import activate_dependencies

        activate_dependencies(root)
        if args.finish_update:
            ok = complete_source_checkout(
                root, desktop=args.desktop, assume_yes=True,
                completion_message=None, announce="\n✓ Code updated!",
            )
        else:
            ok = complete_source_checkout(
                root, desktop=args.desktop, assume_yes=not args.interactive,
                completion_message="✓ Install complete!",
            )
        return 0 if ok else 1

    from pm.environments import activation_environment
    from pm.writer_claim import child_spawn

    # The bootstrap interpreter hands the entire tail to the selected one.
    passthrough = (["--desktop"] if args.desktop else []) + \
                  (["--finish-update"] if args.finish_update else [])
    command = _bootstrap_command(root, passthrough)
    environment = activation_environment(root)
    with child_spawn(environment) as spawn:
        return subprocess.call(command, cwd=root, env=environment, **spawn)


if __name__ == "__main__":
    raise SystemExit(main())