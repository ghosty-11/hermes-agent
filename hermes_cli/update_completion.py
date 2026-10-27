"""Fresh-checkout source update completion and its stdlib-only parent transport.

Imported before a swap; executed by path from the selected tree afterward. The
parent never imports application helpers from the replacement checkout.
"""

from __future__ import annotations

import codecs
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _exit_status(code: int) -> int:
    return code if code >= 0 else 128 - code


def _write_bootstrap_result(request: dict, result_path: Path, code: int, receipt: dict | None) -> int:
    """The bootstrap child's answer; windows_resume stays the parent's token (and its atexit net)."""
    _write_json(result_path, {
        "schema": 1, "update_id": request["receipt"]["update_id"], "exit_code": code,
        "receipt": receipt, "windows_resume": None, "pm_receipt": request.get("pm_receipt"),
    })
    return code


def _bind_and_resume(proc: subprocess.Popen, request: dict, request_path: Path) -> None:
    """Windows: bind the suspended completion child to the update's job, then resume it.

    Post-commit, refusing the child would fail a committed update, so a child the job refuses
    still runs, fenced by its own checkout lease instead (R5b: it joins our lock holding a lease
    byte, which keeps the checkout busy after a killed owner until the child exits). Never
    silently: the refusal is printed and recorded as a failed ``update_custody`` step in this
    run's receipt and, before the child runs (it has not read its request yet), in the receipt it
    resumes, so the terminal receipt carries it."""
    from hermes_cli import update_receipt
    from hermes_cli.update_lock import bind_child_to_update_tree, resume_suspended_child

    refusal = bind_child_to_update_tree(proc)
    if refusal is not None:
        detail = (f"the update's job would not take the completion child ({refusal}), so it runs "
                  "outside the job, holding its own checkout lease")
        print(f"  ⚠ Update completion: {detail}")
        update_receipt.record_step("update_custody", False, detail)
        current = update_receipt._current.get()
        if current is not None and current.data.get("update_id") == request["receipt"]["update_id"]:
            request["receipt"] = json.loads(json.dumps(current.data))
            _write_json(request_path, request)
    resume_suspended_child(proc)


def _run_completion_nonposix(request: dict) -> dict:
    """The checkout-custody completion transport for non-POSIX hosts (Windows job and lease)."""
    root = Path(request["source"])
    env = dict(os.environ, HERMES_HOME=request["home"], PYTHONUNBUFFERED="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="hermes-completion-") as directory:
        request_path = Path(directory) / "request.json"
        result_path = Path(directory) / "result.json"
        request = {**request, "stdout_isatty": sys.stdout.isatty()}
        request["bytecode_cache"] = str(Path(directory) / "bytecode")
        _write_json(request_path, request)
        command = [sys.executable, "-I", "-S", "-u", "-X", "utf8", "-X", f"pycache_prefix={request['bytecode_cache']}",
                   str(root / "hermes_cli/update_completion.py"),
                   str(request_path), str(result_path)]
        from hermes_cli.update_lock import CREATE_SUSPENDED, checkout_lock_fds

        # The child joins the update tree's checkout lock: it inherits the locked fd (POSIX)
        # or dies with us (Windows job: created suspended, bound, then resumed, so nothing it
        # starts runs outside the job), so the lock is never free while it runs.
        proc = subprocess.Popen(
            command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            **({"start_new_session": True, "pass_fds": checkout_lock_fds(root)} if os.name == "posix" else
               {"creationflags": subprocess.CREATE_NO_WINDOW | CREATE_SUSPENDED}))
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            # A failure here unwinds through the cleanup below, never orphans the child.
            if os.name != "posix":
                _bind_and_resume(proc, request, request_path)
            while True:
                chunk = proc.stdout.read1(8192)
                sys.stdout.write(decoder.decode(chunk, final=not chunk))
                sys.stdout.flush()
                if not chunk:
                    break
            code = proc.wait()
        except BaseException as exc:
            try:
                try:
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10, check=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                finally:
                    try:
                        proc.kill()
                    finally:
                        proc.wait(timeout=10)
            except BaseException as cleanup_error:
                exc.add_note("Completion cleanup failed; child processes may still be running.")
                raise exc from cleanup_error
            raise
        finally:
            proc.stdout.close()
        code = _exit_status(code)
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if result["schema"] != 1 or result["update_id"] != request["receipt"]["update_id"]:
                raise ValueError("completion response identity mismatch")
            if result["exit_code"] != code:
                raise ValueError("completion response disagrees with process exit")
            receipt = result.get("receipt")
            if receipt is not None and (
                receipt.get("update_id") != request["receipt"]["update_id"]
                or not receipt.get("finished_at")
                or (code == 0) != (receipt.get("outcome") == "success")
            ):
                raise ValueError("completion receipt does not attest this outcome")
            if code == 0 and receipt is None:
                raise ValueError("completion did not publish a terminal receipt")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # No correlated terminal result: the caller settles it (settle_lost_completion).
            return {"exit_code": code or 1, "receipt": None, "windows_resume": None,
                    "error": f"{exc} (the completion process exited {code})"}
        return result


def _seal_post_quiescence(request: dict, request_path: Path, result_path: Path,
                          env: dict[str, str], retained: list[Path]) -> None:
    from pm.environments import project_python
    from pm.process_scope import owned_spawn
    from pm.writer_claim import child_spawn

    root = Path(request["source"])
    command = [str(project_python(root)), "-I", "-S", "-u", "-X", "utf8",
               str(root / "hermes_cli/update_completion.py"),
               str(request_path), str(result_path), "--seal"]
    from hermes_cli.update_lock import checkout_lock_fds

    # The seal child is the update's last writer: it inherits the writer claim and the
    # checkout lock fd, so custody outlives this parent too.
    with child_spawn(env) as handed:
        handed = {**handed,
                  "pass_fds": (*handed.get("pass_fds", ()), *checkout_lock_fds(root))}
        with owned_spawn(handed, env, retained=retained) as scope:
            proc = scope.spawn(command, cwd=root, env=env)
            proc.wait()
            scope.finish()


def run_completion(request: dict) -> dict:
    """Wait for new code; retain all update inputs until its session is empty."""
    if os.name != "posix":
        return _run_completion_nonposix(request)
    from pm.process_scope import ProcessStillRunning, owned_spawn, retain_exception
    from pm.writer_claim import child_spawn

    root = Path(request["source"])
    # Only the ordinary Linux route may acquire a host prerequisite. Run it
    # before detaching, while the parent still has a controlling terminal.
    if sys.platform.startswith("linux") and not request.get("offline"):
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c",
                        "import sys; sys.path.insert(0, sys.argv[1]); "
                        "from pm.libatomic import install_before_lock; install_before_lock()", str(root)],
                       cwd=root, stdout=None, stderr=subprocess.DEVNULL, check=False)
    env = dict(os.environ, HERMES_HOME=request["home"], PYTHONUNBUFFERED="1")
    if request.get("offline"):
        env["HERMES_LOCAL_UPDATE_CONTEXT"] = "1"
        from pm.offline import offline_environment
        env = offline_environment(env)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    directory = Path(tempfile.mkdtemp(prefix="hermes-completion-"))
    retained = [directory]
    if request.get("staged_tools"):
        retained.append(Path(request["staged_tools"]).parent)
    if request.get("local_rollback_manifest"):
        retained.append(Path(request["local_rollback_manifest"]))
    proc = None
    safe_to_clean = True
    request_path = directory / "request.json"
    result_path = directory / "result.json"
    try:
        request = {**request, "stdout_isatty": sys.stdout.isatty()}
        request["bytecode_cache"] = str(directory / "bytecode")
        _write_json(request_path, request)
        command = [sys.executable, "-I", "-S", "-u", "-X", "utf8",
                   "-X", f"pycache_prefix={request['bytecode_cache']}",
                   str(root / "hermes_cli/update_completion.py"),
                   str(request_path), str(result_path)]
        from hermes_cli.update_lock import checkout_lock_fds

        # The child joins the update tree's writer claim (child_spawn) AND its checkout lock
        # fd (R2), so the checkout stays locked while it runs even if this parent is killed.
        with child_spawn(env) as handed:
            handed = {**handed,
                      "pass_fds": (*handed.get("pass_fds", ()), *checkout_lock_fds(root))}
            with owned_spawn(handed, env, retained=retained) as scope:
                proc = scope.spawn(
                    command, cwd=root, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                try:
                    while True:
                        chunk = proc.stdout.read1(8192)
                        sys.stdout.write(decoder.decode(chunk, final=not chunk))
                        sys.stdout.flush()
                        if not chunk:
                            break
                    code = proc.wait()
                    scope.finish()
                finally:
                    proc.stdout.close()
        code = _exit_status(code)
        intermediate = json.loads(result_path.read_text(encoding="utf-8-sig")) if result_path.exists() else {}
        if "not_quiescent" in intermediate:
            raise ProcessStillRunning(intermediate["not_quiescent"], "selected completion retained a child")
        if intermediate.get("seal_pending"):
            if intermediate.get("update_id") != request["receipt"]["update_id"]:
                raise ValueError("local completion identity mismatch before seal")
            request["receipt"] = intermediate["receipt_data"]
            _write_json(request_path, request)
            _seal_post_quiescence(request, request_path, result_path, env, retained)
            code = json.loads(result_path.read_text(encoding="utf-8-sig"))["exit_code"]
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
            if result["schema"] != 1 or result["update_id"] != request["receipt"]["update_id"]:
                raise ValueError("completion response identity mismatch")
            if result["exit_code"] != code:
                raise ValueError("completion response disagrees with process exit")
            receipt = result.get("receipt")
            if receipt is not None and (
                receipt.get("update_id") != request["receipt"]["update_id"]
                or not receipt.get("finished_at")
                or (code == 0) != (receipt.get("outcome") == "success")
            ):
                raise ValueError("completion receipt does not attest this outcome")
            if code == 0 and receipt is None:
                raise ValueError("completion did not publish a terminal receipt")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"✗ Source update completion did not finish: {exc}")
            return {"exit_code": code or 1, "receipt": None, "windows_resume": None}
        return result
    except BaseException as exc:
        if (isinstance(exc, ProcessStillRunning) or not isinstance(exc, Exception)
                or type(exc).__name__ == "CancelledError"
                or getattr(exc, "retained_paths", None) is not None):
            safe_to_clean = False
            extra = []
            if result_path.exists():
                try:
                    extra = json.loads(result_path.read_text(encoding="utf-8-sig")).get("not_quiescent", [])
                except (OSError, ValueError):
                    pass
            if isinstance(exc, ProcessStillRunning):
                for path in [*retained, *extra]:
                    path = Path(path)
                    if path not in exc.retained_paths:
                        exc.retained_paths += (path,)
                        exc.args = (f"{exc.args[0]}; retained: {path}",)
            else:
                retain_exception(exc, [*retained, *extra])
        raise
    finally:
        if safe_to_clean:
            shutil.rmtree(directory)


def _resume_receipt(data: dict) -> None:
    from hermes_cli import update_receipt

    # Hydrate the existing run, not a new receipt with a new identity/pre-update probe.
    receipt = object.__new__(update_receipt.UpdateReceipt)
    receipt.data = data
    receipt.correlation_id = data["update_id"]
    receipt.current_token = update_receipt._current.set(receipt)


def _read_terminal_receipt(request: dict) -> dict | None:
    from hermes_cli import update_receipt

    update_id = request["receipt"]["update_id"]
    found = update_receipt.read_run_record(update_id)
    if found is not None and found[1].get("finished_at"):
        return found[1]
    # The store refused the terminal write after the commit point (loudly): this process still
    # holds the correlated terminal record it finalized, so the exit status never turns into 1.
    return update_receipt.finalized_receipt(update_id)


def _running_record(request: dict) -> dict | None:
    from hermes_cli.update_receipt import read_run_record

    found = read_run_record(request["receipt"]["update_id"])
    if found is None or found[1].get("outcome") != "running":
        return None
    found[1].pop("writer_pid", None)
    return found[1]


def _owed_user_action(request: dict) -> str | None:
    """The unsettled-autostash notice (#122557) when this run parked the user's local changes.

    ``_complete_source_update`` hands it over as the completion message, and a completion message
    that is not a ``✓`` line is never a success (``_print_verified_update_completion``).
    """
    message = request.get("completion_message") or ""
    return message if message and not message.startswith("✓") else None


def _record_owed_user_action(request: dict) -> bool:
    """Land the parked local changes on the receipt; True when nothing is owed to the user."""
    from hermes_cli.update_receipt import record_user_action

    notice = _owed_user_action(request)
    if notice:
        record_user_action("local_changes", notice)
    return notice is None


def _publish_gateway_success(request: dict) -> None:
    """The gateway /update watcher's status: the code is committed (same as _complete_selected).

    Guarded like ``update_cmd_fleet._write_gateway_update_exit_code`` (which this stdlib-only
    process cannot import): an unwritable home must not turn the committed run into an error
    before its result is written (review P1). The watcher then falls back to its own timeout.
    """
    if not request.get("gateway_mode"):
        return
    try:
        # Temp + rename (stdlib, like _write_json): the gateway watcher polls this file and reads
        # an empty one as exit 1.
        marker = Path(request["home"]) / ".update_exit_code"
        temporary = marker.with_name(marker.name + ".tmp")
        temporary.write_text("0", encoding="utf-8")
        temporary.replace(marker)
    except OSError as exc:
        print(f"  ⚠ Could not publish the gateway's update status: {exc}")


#: The one whole line the Desktop hand-off scripts match (scripts/desktop-update/posix.sh and
#: windows.ps1); ``source_build.build_update_products`` prints the same prefix when the build fails.
DESKTOP_BUILD_OWED = "Desktop app build owed:"


def _report_desktop_build_owed(request: dict, record: dict | None, reason: str) -> None:
    """Name the owed Desktop build when this run never got the build to its own verdict (F6, review P4).

    The builder prints the line only when the build itself fails; a run that never reached it
    (dependencies owed, the tail refused by the lock, a completion process lost) would otherwise end
    as a plain "Update complete." in the hand-off. A ``build`` stage on the run's record means the
    build ran and printed this line itself when the Desktop app failed.
    """
    if not request.get("desktop"):
        return
    if any(isinstance(mark, dict) and mark.get("name") == "build" for mark in (record or {}).get("stages") or ()):
        return
    print(f"  {DESKTOP_BUILD_OWED} {reason}", flush=True)


def _strict_local(request: dict) -> bool:
    """Local transactions cannot substitute deferred follow-ups for completion."""
    return bool(
        request.get("offline")
        or request.get("local_rollback_manifest")
        or request.get("apply_mode") == "local"
    )


def settle_lost_completion(request: dict, reason: str) -> dict:
    """The parent's side of C3: the code is committed, but the completion never answered (review P2).

    Reached when spawning or reading the completion failed (an OSError from the temporary
    directory, the request file or Popen) or the child died without a correlated terminal result
    (OOM, SIGKILL). The tail is owed, never a failed update: it is armed for the next launch, the
    run's receipt is a success that names the ``completion`` follow-up, and the exit is 0 -- unless
    the user's local changes are still parked (#122557: ``partial``, exit 1). Runs in the parent
    (pre-swap) interpreter: everything it uses is already imported there.
    """
    from hermes_cli import update_receipt

    if _strict_local(request):
        raise RuntimeError(f"local source completion was lost: {reason}")

    print(f"  ⚠ Source update completion did not finish: {reason}", flush=True)
    receipt = _read_terminal_receipt(request)  # the child finalized the run, then died
    if receipt is None:
        try:
            from hermes_cli.venv_sync import arm_completion

            arm_completion(Path(request["source"]))
        except Exception as exc:  # health: allow BLE001 -- post-commit boundary: stale dependencies still make the next launch sync
            print(f"  ⚠ Could not record the owed source-update tail: {exc}")
        _report_desktop_build_owed(request, _running_record(request), "the update completion did not finish")
        update_receipt.resume_run_record()  # the child's persisted stages, never this stale snapshot
        update_receipt.record_followup("completion", reason, retry="the next launch or `hermes update` finishes it")
        parked = not _record_owed_user_action(request)
        if parked:
            print(_owed_user_action(request))  # the completion child that prints it may never have run
        update_receipt.finalize_pending_update_receipt(1 if parked else 0, "completion owed after the code was updated")
        receipt = _read_terminal_receipt(request)
    outcome = (receipt or {}).get("outcome")
    code = 130 if outcome == "interrupted" else 1 if outcome == "partial" else 0
    if code == 0:
        _publish_gateway_success(request)
    return {"exit_code": code, "receipt": receipt, "windows_resume": None, "pm_receipt": None}


def _settle_after_commit(request: dict, result_path: Path, step: str, reason: str) -> int:
    """The tree already moved, so a failure here is owed work, never a failed update (C3, A6).

    Runs in the bootstrap interpreter (stdlib + the new tree) when dependency preparation failed or
    the prepared child died without a result. The tail obligation stays armed, so the next launch
    syncs the dependencies and finishes the tail; the run's receipt is a success that names the
    follow-up, so nothing reports "still on the previous version" while the tree is new.
    """
    from hermes_cli import update_receipt
    from hermes_cli.venv_sync import arm_completion

    if _strict_local(request):
        _resume_receipt(_running_record(request) or request["receipt"])
        update_receipt.record_stage("deps" if step == "dependencies" else "build", "failed")
        update_receipt.finalize_pending_update_receipt(1, reason)
        return _write_bootstrap_result(request, result_path, 1, _read_terminal_receipt(request))

    try:
        arm_completion(Path(request["source"]))
    except OSError as exc:  # the next launch still sees stale dependencies and syncs them
        print(f"  ⚠ Could not record the owed source-update tail: {exc}")
    receipt = _read_terminal_receipt(request)  # a prepared child that finalized, then died
    if receipt is None:
        # A prepared child that died mid-tail persisted stages the parent's snapshot lacks.
        record = _running_record(request) or request["receipt"]
        _report_desktop_build_owed(request, record, f"{step} did not finish")
        _resume_receipt(record)
        update_receipt.record_stage("deps" if step == "dependencies" else "build", "failed")
        update_receipt.record_followup(step, reason, retry="dependencies not installed yet — the next launch retries"
                                       if step == "dependencies" else "the next launch or `hermes update` retries it")
        if not _record_owed_user_action(request):
            print(_owed_user_action(request))  # the completion child that would print it never ran
        update_receipt.finalize_pending_update_receipt(0, f"{step} owed after the code was updated")
        receipt = _read_terminal_receipt(request)
    code = 0 if receipt is not None and receipt.get("outcome") == "success" else 1
    if code == 0:
        _publish_gateway_success(request)
    return _write_bootstrap_result(request, result_path, code, receipt)


def _prepare(request: dict, request_path: Path, result_path: Path) -> int:
    import pm
    from pm import receipt
    from pm.client import ensure_tools_for_sync
    from pm.environments import activation_environment, project_python

    root = Path(request["source"])
    update_id = request["receipt"]["update_id"]
    if sys.platform == "win32":
        # Windows (review L3): a bootstrap child the update's job refused runs outside the job
        # and outlives a killed owner, so it joins the checkout lock before anything below
        # writes the checkout (uv's sync children, the generation collector): the join takes a
        # lease byte of its own (R5b, as the --prepared child does in complete_source_checkout).
        # Held until this bootstrap exits, after the --prepared child; the kernel drops it.
        from hermes_cli.update_lock import _acquire_checkout

        refused = _acquire_checkout(root)
        if refused is not None:
            raise RuntimeError("could not join the update's checkout lock "
                               f"({refused.reason or f'held by process {refused.pid}'})")
    from hermes_cli.venv_sync import arm_completion, collect_superseded_generations

    # The foreign-owned-venv refusal runs in the parent BEFORE the swap (update_cmd_commit
    # .preflight_refusal); the tail was armed there too, so this re-arm is an idempotent backstop.
    arm_completion(root)
    from hermes_cli.gitlock import convert_treeless_checkout_first
    convert_treeless_checkout_first(root)
    with receipt.worker_context(update_id):
        try:
            # The parent kept this verified private closure alive across the
            # checkout. Copy it into the live store before selecting anything.
            if request.get("offline"):
                from pm.build_operations import stage_tools
                from pm.lock import Lockfile
                from pm.paths import lockfile_path, writable_store_root
                from pm.registry import tool_roots
                from pm.store import current_target
                stage_tools(list(dict.fromkeys((*tool_roots(Lockfile(lockfile_path()).names()), "uv"))),
                            source_store=Path(request["staged_tools"]),
                            out=writable_store_root(), target=current_target())
            ensure_tools_for_sync(**({"offline": True} if request.get("offline") else {}))
            pm.sync_venv(extras=request.get("required_extras"), explicit=True, project_root=root,
                         evict_incompatible_plugins=not request.get("offline", False),
                         **({"offline": True} if request.get("offline") else {}))
            if not request.get("offline"):
                collect_superseded_generations(root)
        finally:
            request["pm_receipt"] = receipt.last_for_update(update_id)
            _write_json(request_path, request)
    # -X utf8 like the bootstrap child: -I ignores PYTHONIOENCODING, and this child prints ✓/⚠
    # (the follow-up protocol) into the parent's pipe, whatever the console code page.
    command = [str(project_python(root)),
               "-I", "-S", "-u", "-X", "utf8", "-X", f"pycache_prefix={request['bytecode_cache']}",
               str(root / "hermes_cli/update_completion.py"),
               str(request_path), str(result_path), "--prepared"]
    # A second interpreter is mandatory: PM may have selected a different Python
    # and dependency graph. No application maintenance runs in this bootstrap.
    from hermes_cli.update_lock import checkout_lock_fds

    child_env = activation_environment(root)
    if request.get("offline"):
        child_env["HERMES_LOCAL_UPDATE_CONTEXT"] = "1"
        from pm.offline import offline_environment
        child_env = offline_environment(child_env)
    from pm.writer_claim import child_spawn

    # health: allow HX006 -- the prepared completion child is the update's build; it runs to
    # the end. It inherits the update's writer claim and its checkout lock fd (R2).
    with child_spawn(child_env) as handed:
        handed = {**handed,
                  "pass_fds": (*handed.get("pass_fds", ()), *checkout_lock_fds(root))}
        code = _exit_status(subprocess.call(command, cwd=root, env=child_env, **handed))
    if not result_path.exists():
        return _settle_after_commit(request, result_path, "completion",
                                    f"the completion process exited {code} without a result")
    return code


def _complete_selected(request: dict) -> bool:
    """Finish native updates with follow-ups, but require complete local releases.

    Native updates retain independent tail obligations after committing code.
    Local transactions fail on any incomplete or unsafe tail and preserve their
    capsule for explicit recovery.
    """
    from hermes_cli import main, update_cmd, update_cmd_config
    from hermes_cli.source_completion import complete_source_checkout
    from hermes_cli.update_inventory import RuntimeRecord, UpdatePlan
    from hermes_cli.update_receipt import (
        TAIL_FOLLOWUPS, record_build_stage, record_followup, record_skip, record_stage)

    root = Path(request["source"])
    main.PROJECT_ROOT = root
    complete = _record_owed_user_action(request)
    update_cmd_config._LAST_SIBLING_SNAPSHOTS = request["sibling_snapshots"]
    plan_data = request["plan"]
    plan = None if plan_data is None else UpdatePlan(**{
        **plan_data, "runtimes": [RuntimeRecord(**row) for row in plan_data.get("runtimes", [])]})
    if request.get("offline"):
        from hermes_cli.update_local import local_git
        if local_git(root, "rev-parse", "HEAD").stdout.strip() != request["expected_sha"]:
            raise RuntimeError("local checkout changed before selected-interpreter completion")
    update_cmd._sweep_bytecode_after_update(request["branch"])
    # Launchers, products and post-build maintenance live in one place so an
    # install and an update cannot end in different states.
    followups: list[tuple[str, str]] = []
    # Exit status and runtime safety are separate facts (R8): an unsafe SQLite runtime keeps the
    # committed update at exit 0 (reported as the ``sqlite_runtime`` follow-up), but fleet
    # verification still needs the real verdict so it never auto-migrates the gateway topology
    # on a runtime that can corrupt sessions. Unknown (the tail raised) is not proven safe.
    runtime_safe = False
    try:
        runtime_safe = complete_source_checkout(
            root, desktop=request["desktop"], assume_yes=request["assume_yes"],
            gateway_mode=request["gateway_mode"], pre_update_snapshot_id=request["snapshot_id"],
            pre_update_version=request["pre_update_version"],
            completion_message=request.get("completion_message"),
            announce=None if request.get("completion_message") else "\n✓ Code updated!",
            followups=followups, before_build=lambda: _resume_paused_before_build(request),
            **({"offline": True} if request.get("offline") else {}))
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- e.g. the shared update lock refused the tail
        if _strict_local(request):
            raise
        reason = str(exc) or type(exc).__name__
        # The tail's steps catch their own failures, so a raise here means the build never ran.
        _report_desktop_build_owed(request, None, "the source-update tail did not run")
        record_followup("completion", reason)
        followups.append(("completion", reason))
    tail_owed = any(step in TAIL_FOLLOWUPS for step, _ in followups)
    record_build_stage(followups)
    if _strict_local(request) and (tail_owed or not runtime_safe):
        raise RuntimeError("local source completion left failed or unsafe tail steps")
    if not tail_owed:
        from hermes_cli.venv_sync import clear_completion
        clear_completion(root)
    # systemctl's KillMode=mixed fallback can kill this whole cgroup. Publish the
    # gateway watcher's status BEFORE that operation: the code is committed, so it is 0 unless
    # the user's own changes are still parked.
    if request["gateway_mode"]:
        update_cmd._write_gateway_update_exit_code(complete)
    if request.get("no_gateway_restart", False):
        record_skip("gateway_restart", "--no-gateway-restart: deferred, marker kept")
        record_stage("restart", "skipped")
        print("→ Gateway restart deferred (--no-gateway-restart); restart gateways separately.")
        return complete
    skip = update_cmd._fleet_restart_skip_reason(plan)
    if skip and update_cmd._pending_fleet_restart_needed():
        # A host already stamped "restarted" for this SHA whose fleet is still off the checkout
        # (a stale sibling, a failed resume) gets the restart again instead of a dead end.
        print(f"  → Gateway restart not skipped ({skip}): gateways are still off the checkout code.")
        skip = None
    if skip:
        record_skip("gateway_restart", skip)
        record_stage("restart", "skipped")
        print(f"  ✓ Gateway restart skipped: {skip}.")
        return complete
    restart = update_cmd._restart_gateway_fleet_after_update(plan, request["gateway_mode"])
    record_stage("restart", "failed" if getattr(restart, "incomplete", False) else "success")
    update_cmd._resume_windows_gateways_and_merge_outcome(restart, request["windows_resume"], request["gateway_mode"])
    update_cmd._verify_fleet_after_update(
        restart, _pre_update_plan=plan, _windows_gateway_resume=request["windows_resume"],
        update_complete=bool(runtime_safe) and complete)
    return complete


def _resume_paused_before_build(request: dict) -> None:
    """Restart the gateways the POSIX pause stopped: the dependencies are synced and the launchers
    published, the product builds (minutes) have not started. A gateway needs nothing later to
    boot. A failure stays owed: the post-build resume retries it and records the outcome.

    POSIX only: the fleet restart skips what this restarted (``already_restarted``). A Windows set
    keeps resuming after the fleet restart, which would otherwise drain it a second time."""
    token = request.get("windows_resume")
    if not token or not token.get("resume_needed") or token.get("platform") != "posix":
        return
    from hermes_cli.update_cmd import _m
    try:
        _m()._resume_windows_gateways_after_update(token)
    except Exception as exc:  # health: allow BLE001 -- retried (and recorded) after the build
        print(f"  ⚠ Paused gateway restart incomplete ({exc}); retrying after the build")


class _ForwardedOutput:
    """The parent's pipe preserves its terminal's prompt policy and log mirror."""

    def __init__(self, stream, isatty: bool):
        self.stream, self.terminal = stream, isatty

    def isatty(self):
        return self.terminal

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _report_unbuilt_desktop(request: dict) -> None:
    """The tail raised out of ``_complete_selected``: before the build stage, the build never ran."""
    from hermes_cli import update_receipt

    current = update_receipt._current.get()
    if current is not None:  # a finalized run got past the build (verification finalizes it)
        _report_desktop_build_owed(request, current.data, "the source-update tail did not finish")


def _finish(request: dict, result_path: Path) -> int:
    from hermes_cli import update_receipt
    from pm.receipt import accept_worker_receipt
    from pm.process_scope import ProcessStillRunning

    _resume_receipt(request["receipt"])
    accept_worker_receipt(request.get("pm_receipt"), request["receipt"]["update_id"])
    update_receipt.record_stage("deps", "success")  # only a completed PM preparation reaches --prepared
    code, reason = 0, "source update completion"
    retained_paths = None
    cancelled = False
    extension_completed = not request.get("update_extension_context")
    try:
        if not _complete_selected(request):
            code, reason = 1, "local changes left in the stash; re-apply them by hand"
        elif request.get("update_extension_context"):
            # The root-pinned bundle apply and its postchecks happen BEFORE
            # the capsule seal: a plain failure below keeps the capsule (the
            # ordinary nonzero path seals it); a surviving writer raises
            # ProcessStillRunning and nothing seals.
            from hermes_cli.update_extension import complete_update_extension

            complete_update_extension(
                request["update_extension_context"],
                install_root=Path(request["source"]), home=Path(request["home"]),
                target=request["expected_sha"])
            extension_completed = True
    except KeyboardInterrupt as exc:
        # Local sealing and unproved child cleanup still need the open receipt.
        code, reason = 130, "KeyboardInterrupt: interrupted after the code was updated"
        retained_paths = getattr(exc, "retained_paths", None)
        cancelled = True
        if retained_paths is None and not request.get("local_rollback_manifest"):
            update_receipt.finalize_interrupted_update_receipt(reason, exit_code=code)
    except SystemExit as exc:
        if isinstance(exc.code, int) and exc.code:
            code = _exit_status(exc.code)
            reason = f"completion exited {code}"
            _report_unbuilt_desktop(request)
            update_receipt.record_followup("completion", reason)
        if getattr(exc, "retained_paths", None) is not None:
            retained_paths = exc.retained_paths
            cancelled = True
    except BaseException as exc:
        # The tree already moved: an ordinary completion failure is owed work
        # (C3/A6) — success with a named follow-up. Strict-local runs and runs
        # with a requested update extension keep failing closed.
        owed_ordinary = not (_strict_local(request) or request.get("update_extension_context"))
        if owed_ordinary:
            code = 0
        else:
            code = _exit_status(exc.returncode) if isinstance(exc, subprocess.CalledProcessError) else 1
        reason = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, ProcessStillRunning):
            _report_unbuilt_desktop(request)
        update_receipt.record_followup("completion", reason)
        if isinstance(exc, ProcessStillRunning):
            retained_paths = exc.retained_paths
        elif getattr(exc, "retained_paths", None) is not None:
            retained_paths = exc.retained_paths
            cancelled = True
        if owed_ordinary:
            print(f"  ⚠ Source update completion did not finish: {reason}")
        else:
            print(f"✗ Source update completion failed: {reason}")
    finally:
        if code == 0 and not extension_completed:
            code, reason = 1, "required update extension did not complete"
            _report_unbuilt_desktop(request)
            update_receipt.record_followup("completion", reason)
            print(f"✗ Source update completion failed: {reason}")
        custody = sys.modules.get("hermes_cli.update_custody")
        refusal = custody.refusal_notice() if code and custody is not None else None
        if refusal:  # m2: what stopped it, whatever error the refusal turned into downstream
            print(refusal)
        # The new interpreter owns recovery too. The original parent's atexit
        # token is updated from the response; it acts only if this process dies.
        try:
            from hermes_cli.update_cmd import _resume_windows_gateways_after_update
            _resume_windows_gateways_after_update(request["windows_resume"])
        except Exception as exc:
            update_receipt.owe_followup(request["receipt"]["update_id"], "windows_resume",
                                        f"Windows gateway recovery failed: {exc}")
        if retained_paths is not None:
            _write_json(result_path, {
                "schema": 1, "update_id": request["receipt"]["update_id"],
                "exit_code": code, "not_quiescent": [str(path) for path in retained_paths],
                "cancelled": cancelled,
            })
        elif request.get("local_rollback_manifest"):
            # The parent must first prove the completion session empty. A new
            # selected interpreter seals only after that proof, never here.
            _write_json(result_path, {
                "schema": 1, "update_id": request["receipt"]["update_id"],
                "exit_code": code, "seal_pending": True,
                "receipt_data": update_receipt._current.get().data,
            })
        else:
            update_receipt.finalize_pending_update_receipt(code, reason)
            terminal_receipt = _read_terminal_receipt(request)
            if terminal_receipt and terminal_receipt.get("outcome") == "success":
                code = 0  # an interrupt that landed after verification closed the run (review regression 3)
            if code and request["gateway_mode"]:
                from hermes_cli.update_cmd import _write_gateway_update_exit_code
                _write_gateway_update_exit_code(False)
            if not terminal_receipt:
                code = code or 1  # the parent settles a lost result (settle_lost_completion)
            _write_json(result_path, {
                "schema": 1, "update_id": request["receipt"]["update_id"], "exit_code": code,
                "receipt": terminal_receipt, "windows_resume": request["windows_resume"],
            })
    return code


def _seal(request: dict, result_path: Path) -> int:
    from hermes_cli import update_receipt
    from hermes_cli.local_update_rollback import seal_local_update_capsule

    _resume_receipt(request["receipt"])
    intermediate = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if (not intermediate.get("seal_pending")
            or intermediate.get("update_id") != request["receipt"]["update_id"]):
        raise ValueError("local completion is not awaiting a correlated seal")
    code = intermediate["exit_code"]
    reason = "source update completion" if code == 0 else f"completion exited {code}"
    try:
        seal_local_update_capsule(Path(request["local_rollback_manifest"]),
                                  expected_install_root=Path(request["source"]),
                                  expected_hermes_home=Path(request["home"]))
        update_receipt.record_stage("rollback_capsule", "success")
    except BaseException as exc:
        code, reason = 1, f"local rollback capsule seal failed: {exc}"
        update_receipt.record_stage("rollback_capsule", "failed")
        print(f"✗ {reason}")
    if code == 130:
        update_receipt.finalize_interrupted_update_receipt(reason, exit_code=code)
    else:
        update_receipt.finalize_pending_update_receipt(code, reason)
    terminal_receipt = _read_terminal_receipt(request)
    if not terminal_receipt:
        code = code or 1
    _write_json(result_path, {
        "schema": 1, "update_id": request["receipt"]["update_id"], "exit_code": code,
        "receipt": terminal_receipt, "windows_resume": request["windows_resume"],
    })
    return code


def main() -> int:
    request_path, result_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    if request["schema"] != 1:
        raise ValueError("unsupported source completion request")
    root = Path(__file__).resolve().parents[1]
    if root != Path(request["source"]).resolve():
        raise ValueError("completion checkout does not match request")
    # -I deliberately ignores inherited PYTHONPATH during PM preparation.
    sys.path.insert(0, str(root))
    from pm.writer_claim import writer_claim
    with writer_claim(install=root, inherited_only=True):
        return _main_owned(request, request_path, result_path, root)


def _main_owned(request: dict, request_path: Path, result_path: Path, root: Path) -> int:
    sys.stdout = _ForwardedOutput(sys.stdout, request.get("stdout_isatty", False))
    if "--seal" in sys.argv[3:]:
        from pm.environments import activate_dependencies
        activate_dependencies(root)
        return _seal(request, result_path)
    if "--prepared" in sys.argv[3:]:
        # Claim the selected generation's lease and process its .pth files only
        # after PM selection, before importing any application dependencies.
        from pm.environments import activate_dependencies
        activate_dependencies(root)
        return _finish(request, result_path)
    return _bootstrap(request, request_path, result_path)


def _bootstrap(request: dict, request_path: Path, result_path: Path) -> int:
    """Dependency preparation in the parent's interpreter; its failure is a follow-up (A6)."""
    try:
        return _prepare(request, request_path, result_path)
    except KeyboardInterrupt:
        _resume_receipt(request["receipt"])
        from hermes_cli import update_receipt
        update_receipt.finalize_interrupted_update_receipt(
            "KeyboardInterrupt: interrupted while installing dependencies", exit_code=130)
        return _write_bootstrap_result(request, result_path, 130, _read_terminal_receipt(request))
    except (Exception, SystemExit) as exc:  # health: allow BLE001 -- after the commit point nothing fails the update
        # The code is committed but its dependencies are not (A6): an owed follow-up, exit 0.
        # Paused Windows gateways stay the parent's to resume (it has the dependencies).
        return _settle_after_commit(request, result_path, "dependencies", str(exc) or type(exc).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
