import base64
import copy
import hashlib
import json
import os
from os import read as _io_read
from os import write as _io_write
from pathlib import Path
import re
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time

from pilot_protocol import FLAGS, fields, require
from recalibration_data_contract import _plain
from recalibration_tokenizer_inventory import build_synthetic_inventory
from recalibration_tokenizer_remote import (
    HEADER,
    SESSION_SOURCE_NAMES,
    WIRE_LIMIT,
    decode_frames,
    strict_json,
    validate_bundle,
    validate_work_timeout,
)
from recalibration_tokenizer_report import (
    build_synthetic_report,
    prepare_synthetic_inputs,
)
from validation_contract import digest, encoded, parse_json


SESSION_MODULE_PINS = {
    "pilot_protocol.py": "5074980a2b5b9d8ec0a52288d7cb62cbc847688f094cd8ba189a7135ba315c97",
    "validation_contract.py": "a4d6de138fa4817b23fc6e28442785b0d68a6b8f7360fb110582752d3463299d",
    "tokenizer_measurement.py": "527f63cc21fb9d044b179f0e67228944db36a9b279ef475208278adcdf73481e",
    "tokenizer_runtime_guard.py": "5f9f15fd5cf14fe828a50cce62711795159df966e6c934b2f4d91035dc0d6055",
    "tokenizer_runtime_profile.py": "44b008249cc5bb0c05c857cd32825487d3b72d3c4780867461e12622c2fa4745",
    "tokenizer_snapshot.py": "553ec9e1a42e79da26347336f4b9c78babf7c829f852318e16ab440a0924d4a7",
    "tokenizer_session.py": "aa95c62dcbf68ee0d8489863162a7d782ab433daf7e7665f349a41d8665c53fc",
}
REMOTE_BOOTSTRAP_SHA256 = "d0fe5dd2ffb330e028c03c53bb80410c2c28f113f70779549e3d9d56e328ecc6"
SSH_STDERR_LIMIT = 131072
SSH_REAP_NS = 2_000_000_000
SSH_OVERHEAD_NS = 60_000_000_000
FALSE_FLAGS = (
    *FLAGS,
    "all_possible_second_rounds_validated",
    "tokenizer_length_validated",
    "trusted_collection_accepted",
    "run_authorized",
    "tokenizer_dispatch_authorized",
)


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


class _FileErrors:
    def __init__(self):
        self.primary = None
        self.traceback = None
        self.phase = None
        self.secondary = []

    def capture(self, error, phase):
        self.secondary.extend(getattr(error, "_tok_secondary", []))
        if self.primary is None:
            self.primary, self.traceback, self.phase = error, error.__traceback__, phase
        elif not isinstance(error, Exception) and isinstance(self.primary, Exception):
            self.secondary.append({"phase": self.phase, "type": type(self.primary).__name__})
            self.primary, self.traceback, self.phase = error, error.__traceback__, phase
        else:
            self.secondary.append({"phase": phase, "type": type(error).__name__})

    def close(self, descriptor, phase):
        try:
            os.close(descriptor)
        except BaseException as close_err:
            self.capture(close_err, phase)

    def raise_if_any(self):
        if self.primary is not None:
            self.primary._tok_secondary = self.secondary
            raise self.primary.with_traceback(self.traceback)


def _read_regular_nofollow(path_str, limit):
    descriptor = os.open(path_str, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    failures = _FileErrors()
    collected = bytearray()
    try:
        meta = os.fstat(descriptor)
        require(stat.S_ISREG(meta.st_mode) and meta.st_size <= limit, "regular file size")
        while len(collected) <= limit:
            chunk = os.read(descriptor, min(65536, limit + 1 - len(collected)))
            if not chunk:
                break
            collected.extend(chunk)
    except BaseException as err:
        failures.capture(err, "read_body")
    finally:
        failures.close(descriptor, "read_close")
    failures.raise_if_any()
    return bytes(collected)


def _write_exclusive(directory_fd, filename, raw):
    descriptor = os.open(
        filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd
    )
    failures = _FileErrors()
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        view = memoryview(raw)
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("nonpositive write")
            offset += written
        os.fsync(descriptor)
        os.fsync(directory_fd)
    except BaseException as err:
        failures.capture(err, "write_body")
    finally:
        failures.close(descriptor, "write_close")
    failures.raise_if_any()


def _pretty_json(obj):
    return (
        json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def _claim_local_run_root(local_run_root, run_id):
    require(
        type(local_run_root) is str and local_run_root.startswith("/") and len(local_run_root) <= 256,
        "local_run_root syntax",
    )
    parts = local_run_root[1:].split("/")
    require(
        len(parts) >= 2
        and all(
            re.fullmatch("[A-Za-z0-9_.-]+", seg) is not None and seg not in (".", "..")
            for seg in parts
        )
        and parts[-1] == run_id,
        "canonical local_run_root",
    )
    parent_path = str(Path(local_run_root).parent)
    require(os.path.realpath(parent_path) == parent_path, "canonical parent")
    parent_fd = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    run_fd = None
    failures = _FileErrors()
    try:
        pmeta = os.fstat(parent_fd)
        require(
            stat.S_ISDIR(pmeta.st_mode)
            and pmeta.st_uid == os.getuid()
            and stat.S_IMODE(pmeta.st_mode) == 0o700,
            "parent permissions",
        )
        os.mkdir(run_id, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        run_fd = os.open(run_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        os.fchmod(run_fd, 0o700)
        rmeta = os.fstat(run_fd)
        require(stat.S_ISDIR(rmeta.st_mode) and stat.S_IMODE(rmeta.st_mode) == 0o700, "run dir mode")
    except BaseException as err:
        failures.capture(err, "claim_run_root")
    finally:
        failures.close(parent_fd, "close_parent")
    if failures.primary is not None and run_fd is not None:
        failures.close(run_fd, "close_run_on_error")
    failures.raise_if_any()
    return run_fd


def _verify_sources(session_sources):
    for forbidden in ("validation_remote", "validation_task_worker"):
        require(forbidden not in sys.modules, "forbidden host import")
    fields(session_sources, SESSION_MODULE_PINS)
    for name, expected_sha in SESSION_MODULE_PINS.items():
        raw = session_sources[name]
        require(type(raw) is bytes and _sha256(raw) == expected_sha, "session source pin: " + name)
    remote_path = os.path.realpath(Path(__file__).with_name("recalibration_tokenizer_remote.py"))
    remote_raw = _read_regular_nofollow(remote_path, 1048576)
    require(_sha256(remote_raw) == REMOTE_BOOTSTRAP_SHA256, "remote bootstrap source drift")
    self_path = os.path.realpath(__file__)
    self_raw = _read_regular_nofollow(self_path, 1048576)
    return remote_raw, _sha256(self_raw)


def _counting_projection(inventory):
    require(inventory["status"] == "SYNTHETIC_ONLY" and len(inventory["rows"]) == 120, "120 rows required")
    projected_rows = [
        {
            "ordinal": row["ordinal"],
            "messages": copy.deepcopy(row["messages"]),
            "answer": row["answer"],
            "row_sha256": row["row_sha256"],
        }
        for row in inventory["rows"]
    ]
    return {
        "kind": "LORA_RECALIBRATION_COUNTING_PROJECTION",
        "version": 1,
        "rows": projected_rows,
    }


def build_tokenizer_launch_package(
    validation_inputs,
    profile_raw,
    approved_profile_sha256,
    session_sources,
    scope_raw,
    *,
    ssh_argv_prefix,
    ssh_environment,
    hostname,
    uid,
    interpreter,
    resolved_interpreter,
    site_packages,
    checkpoint_source,
    remote_parent,
    local_run_root,
    work_timeout_seconds=1200,
):
    validate_work_timeout(work_timeout_seconds)
    remote_raw, controller_sha256 = _verify_sources(session_sources)
    inventory, profile = prepare_synthetic_inputs(
        validation_inputs, profile_raw, approved_profile_sha256
    )
    run_id = profile["run_id"]
    for abs_path in (interpreter, resolved_interpreter, site_packages, checkpoint_source, remote_parent, local_run_root):
        require(
            type(abs_path) is str
            and abs_path.startswith("/")
            and ".." not in Path(abs_path).parts
            and str(Path(abs_path)) == abs_path,
            "normalized absolute path required",
        )
    scope = parse_json(scope_raw, limit=65536)
    _plain(scope)
    fields(
        scope,
        (
            "kind", "version", "run_id", "profile_sha256", "local_run_root",
            "remote_parent", "ssh_argv_prefix_sha256", "ssh_environment_sha256",
        ),
    )
    require(
        scope["kind"] == "LORA_RECALIBRATION_TOKENIZER_STAGE_SCOPE"
        and type(scope["version"]) is int
        and scope["version"] == 1
        and scope["run_id"] == run_id
        and scope["profile_sha256"] == approved_profile_sha256
        and scope["local_run_root"] == local_run_root
        and scope["remote_parent"] == remote_parent
        and scope["ssh_argv_prefix_sha256"] == digest(ssh_argv_prefix)
        and scope["ssh_environment_sha256"] == digest(ssh_environment),
        "tokenizer scope binding mismatch",
    )

    projection_obj = _counting_projection(inventory)
    projection_raw = encoded(projection_obj)
    projection_sha256 = _sha256(projection_raw)
    payload_raw = encoded({
        "sources": {
            name: base64.b64encode(session_sources[name]).decode("ascii")
            for name in sorted(SESSION_SOURCE_NAMES)
        },
        "projection": base64.b64encode(projection_raw).decode("ascii"),
        "profile": base64.b64encode(profile_raw).decode("ascii"),
    })
    payload_sha256 = _sha256(payload_raw)

    bootstrap_config = {
        "kind": "LORA_RECALIBRATION_TOKENIZER_BOOTSTRAP_CONFIG",
        "version": 1,
        "hostname": hostname,
        "uid": uid,
        "interpreter": interpreter,
        "resolved_interpreter": resolved_interpreter,
        "site_packages": site_packages,
        "source": checkpoint_source,
        "parent": remote_parent,
        "stage": str(Path(remote_parent) / ("bootstrap-" + run_id)),
        "run_id": run_id,
        "sources_sha256": dict(SESSION_MODULE_PINS),
        "projection_sha256": projection_sha256,
        "profile_sha256": approved_profile_sha256,
        "payload_sha256": payload_sha256,
        "payload_bytes": len(payload_raw),
        "bootstrap_sha256": REMOTE_BOOTSTRAP_SHA256,
        "controller_sha256": controller_sha256,
        "work_timeout_seconds": work_timeout_seconds,
    }
    validate_bundle(payload_raw, bootstrap_config)
    config_raw = encoded(bootstrap_config)
    config_sha256 = _sha256(config_raw)

    bootstrap_b64 = base64.b64encode(remote_raw).decode("ascii")
    config_b64 = base64.b64encode(config_raw).decode("ascii")
    program = (
        "import base64,hashlib,json,sys\n"
        f"src=base64.b64decode({bootstrap_b64!r},validate=True)\n"
        f"cfg=json.loads(base64.b64decode({config_b64!r},validate=True).decode('ascii'))\n"
        "if hashlib.sha256(src).hexdigest()!=cfg['bootstrap_sha256']:\n"
        "    sys.exit(78)\n"
        "ns={'__name__':'recalibration_tokenizer_bootstrap'}\n"
        "exec(compile(src,'<recalibration_tokenizer_remote>','exec'),ns)\n"
        "sys.exit(ns['bootstrap'](cfg))\n"
    )
    remote_cmd = "exec " + shlex.join((
        "env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONHOME",
        "-u", "PYTHONPATH", interpreter, "-I", "-S", "-B", "-c", program,
    ))
    full_argv = [*ssh_argv_prefix, remote_cmd]

    manifest = {
        "kind": "LORA_RECALIBRATION_TOKENIZER_LAUNCH_PACKAGE",
        "version": 1,
        "data_version": "recalibration_v2_r1",
        "run_id": run_id,
        "inventory_sha256": inventory["inventory_sha256"],
        "validation_report_sha256": inventory["validation_report_sha256"],
        "dynamic_evidence_sha256": inventory["dynamic_evidence_sha256"],
        "projection_sha256": projection_sha256,
        "profile_sha256": approved_profile_sha256,
        "one_time_scope_sha256": _sha256(scope_raw),
        "config_sha256": config_sha256,
        "argv_sha256": digest(full_argv),
        "bootstrap_config": bootstrap_config,
        "local_run_root": local_run_root,
        "planned_rows": 120,
        "launch_package_structure_valid": True,
        **{flag: False for flag in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "ssh_argv": full_argv,
            "ssh_environment": dict(ssh_environment),
            "payload_hex": payload_raw.hex(),
        },
    }


def _validate_ready_frame(ready_raw, bootstrap_config, config_sha256):
    ready = strict_json(ready_raw)
    return (
        type(ready) is dict
        and ready.get("config_sha256") == config_sha256
        and ready.get("run_id") == bootstrap_config["run_id"]
        and ready.get("hostname") == bootstrap_config["hostname"]
        and type(ready.get("uid")) is int
        and ready["uid"] == bootstrap_config["uid"]
        and ready.get("stage") == bootstrap_config["stage"]
        and ready.get("core_limit") == [0, 0]
        and ready.get("protection") == {
            "threads": 1,
            "fork_denied": True,
            "thread_denied": True,
            "denied_syscalls": [56, 57, 58, 435],
            "non_x86_and_x32": "KILL_PROCESS",
        }
    )


def _safe_raw_ns():
    try:
        val = time.monotonic_ns()
        if type(val) is int and val >= 0:
            return val
    except BaseException:
        return None
    return None


def _cleanup_ssh(process, constructor_completed, cancellations, secondary_errors):
    start_raw = _safe_raw_ns()
    clock_ok = start_raw is not None
    last_raw = start_raw
    deadline_raw = (start_raw + SSH_REAP_NS) if start_raw is not None else None
    child_reaped = False
    returncode = None
    group_absent = None
    pid = process.pid

    def sample_raw():
        nonlocal clock_ok, last_raw
        cur = _safe_raw_ns()
        if cur is None or last_raw is None or cur < last_raw:
            clock_ok = False
            return None
        last_raw = cur
        return cur

    def record_err(err, phase):
        if not isinstance(err, Exception) and not cancellations:
            cancellations.append((err, err.__traceback__))
        else:
            secondary_errors.append({"phase": phase, "type": type(err).__name__})

    def reap_once(wait_sec):
        nonlocal child_reaped, returncode
        if child_reaped:
            return
        try:
            waited = process.wait(timeout=max(0.0, wait_sec))
            if type(waited) is int:
                returncode = waited
                child_reaped = True
        except subprocess.TimeoutExpired:
            pass
        except BaseException as err:
            record_err(err, "ssh_cleanup_wait")

    def probe_group():
        nonlocal group_absent
        try:
            os.killpg(pid, 0)
            group_absent = False
            return True
        except ProcessLookupError:
            group_absent = True
            return False
        except PermissionError:
            group_absent = False
            return True
        except BaseException as err:
            group_absent = None
            record_err(err, "ssh_cleanup_probe")
            return None

    def send_group(signum):
        try:
            os.killpg(pid, signum)
        except (ProcessLookupError, PermissionError):
            pass
        except BaseException as err:
            record_err(err, "ssh_cleanup_signal")

    def send_direct(signum):
        if constructor_completed or child_reaped:
            return
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass
        except BaseException as err:
            record_err(err, "ssh_cleanup_child")

    sent_kill = False
    try:
        reap_once(0.05)
        alive = probe_group()
        if alive is not False:
            send_group(signal.SIGTERM)
        send_direct(signal.SIGTERM)
        reap_once(0.15)
        for step_idx in range(110):
            obs = sample_raw()
            reap_once(0.0)
            alive = probe_group()
            if alive is False and child_reaped:
                break
            elapsed_ok = (
                (obs - start_raw >= 200_000_000)
                if (obs is not None and start_raw is not None)
                else (step_idx >= 5)
            )
            if not sent_kill and elapsed_ok:
                if alive is not False:
                    send_group(signal.SIGKILL)
                send_direct(signal.SIGKILL)
                sent_kill = True
                reap_once(0.0)
                alive = probe_group()
                if alive is False and child_reaped:
                    break
            if obs is not None and deadline_raw is not None and obs >= deadline_raw:
                break
            after = sample_raw()
            if after is not None and deadline_raw is not None:
                rem = deadline_raw - after
                if rem <= 0:
                    break
                sleep_sec = min(0.02, rem / 1_000_000_000.0)
            else:
                sleep_sec = 0.01
            time.sleep(sleep_sec)
    except BaseException as err:
        clock_ok = False
        record_err(err, "ssh_cleanup_loop")
        send_group(signal.SIGKILL)
        send_direct(signal.SIGKILL)
        reap_once(0.0)
        probe_group()

    end_raw = sample_raw()
    bounded = bool(clock_ok and start_raw is not None and end_raw is not None and end_raw <= deadline_raw)
    return {
        "child_reaped": child_reaped,
        "returncode": returncode if child_reaped else None,
        "group_absent": group_absent is True,
        "cleanup_bounded": bounded,
    }


def run_tokenizer_stage(
    package,
    validation_inputs,
    profile_raw,
    approved_profile_sha256,
    session_sources,
    scope_raw,
):
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    cfg = pub["bootstrap_config"]
    rebuilt = build_tokenizer_launch_package(
        validation_inputs,
        profile_raw,
        approved_profile_sha256,
        session_sources,
        scope_raw,
        ssh_argv_prefix=priv["ssh_argv"][:-1],
        ssh_environment=priv["ssh_environment"],
        hostname=cfg["hostname"],
        uid=cfg["uid"],
        interpreter=cfg["interpreter"],
        resolved_interpreter=cfg["resolved_interpreter"],
        site_packages=cfg["site_packages"],
        checkpoint_source=cfg["source"],
        remote_parent=cfg["parent"],
        local_run_root=pub["local_run_root"],
        work_timeout_seconds=cfg["work_timeout_seconds"],
    )
    require(encoded(package) == encoded(rebuilt), "tokenizer launch package mismatch")

    run_id = pub["run_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], run_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    payload_raw = bytes.fromhex(priv["payload_hex"])
    work_limit_ns = cfg["work_timeout_seconds"] * 1_000_000_000
    ssh_limit_ns = work_limit_ns + SSH_OVERHEAD_NS

    wire_buf = bytearray()
    stderr_buf = bytearray()
    stdin_queue = bytearray()
    stdin_offset = 0
    stdin_closed = False
    ready_seen = False
    grant_dispatched = False
    stdout_eof = False
    stderr_eof = False
    ssh_status = None
    cleanup_info = None
    process = None
    selector = None
    constructor_completed = False
    started_ns = _safe_raw_ns() or 0

    def close_stdin():
        nonlocal stdin_closed
        if stdin_closed:
            return
        stdin_closed = True
        if selector is not None:
            try:
                selector.unregister(process.stdin)
            except (KeyError, ValueError):
                pass
        pipe_obj = getattr(process, "stdin", None)
        if pipe_obj is not None:
            try:
                pipe_obj.close()
            except BaseException as err:
                if not isinstance(err, Exception) and not cancellations:
                    cancellations.append((err, err.__traceback__))
                else:
                    secondary_errors.append({"phase": "close_stdin", "type": type(err).__name__})

    try:
        try:
            _write_exclusive(
                run_fd,
                "tokenizer-stage-claim.json",
                _pretty_json({
                    "kind": "LORA_RECALIBRATION_TOKENIZER_STAGE_CLAIM",
                    "run_id": run_id,
                    "package_sha256": pub["package_sha256"],
                    "config_sha256": pub["config_sha256"],
                    "one_time_scope_sha256": pub["one_time_scope_sha256"],
                    "consumed_before_ssh": True,
                }),
            )
            _write_exclusive(run_fd, "tokenizer-package-public.json", _pretty_json(pub))
        except BaseException as claim_err:
            persistence_errors.append(claim_err)
            if not isinstance(claim_err, Exception) and not cancellations:
                cancellations.append((claim_err, claim_err.__traceback__))
            ssh_status = "claim_failed"

        if ssh_status is None:
            try:
                started_ns = time.monotonic_ns()
                deadline_ns = started_ns + ssh_limit_ns
                process = object.__new__(subprocess.Popen)
                subprocess.Popen.__init__(
                    process,
                    priv["ssh_argv"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd="/",
                    env=dict(priv["ssh_environment"]),
                    close_fds=True,
                    start_new_session=True,
                    shell=False,
                )
                constructor_completed = True
                selector = selectors.DefaultSelector()
                os.set_blocking(process.stdout.fileno(), False)
                os.set_blocking(process.stderr.fileno(), False)
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")

                while True:
                    now_ns = time.monotonic_ns()
                    if now_ns >= deadline_ns:
                        ssh_status = "timeout"
                        break
                    if not selector.get_map() and process.poll() is not None:
                        break
                    rem_sec = (deadline_ns - now_ns) / 1_000_000_000.0
                    if not selector.get_map():
                        time.sleep(min(0.02, rem_sec))
                        continue
                    events = selector.select(min(0.05, rem_sec))
                    if time.monotonic_ns() >= deadline_ns:
                        ssh_status = "timeout"
                        break
                    for key, _mask in events:
                        if key.data == "stdin":
                            if stdin_closed:
                                continue
                            try:
                                written = _io_write(
                                    key.fd, stdin_queue[stdin_offset:stdin_offset + 65536]
                                )
                            except BlockingIOError:
                                continue
                            except BrokenPipeError:
                                close_stdin()
                                continue
                            if written > 0:
                                stdin_offset += written
                            if stdin_offset >= len(stdin_queue):
                                close_stdin()
                            continue

                        target_buf, cap = (
                            (wire_buf, WIRE_LIMIT)
                            if key.data == "stdout"
                            else (stderr_buf, SSH_STDERR_LIMIT)
                        )
                        rem_bytes = cap - len(target_buf)
                        try:
                            chunk = _io_read(key.fd, min(65536, rem_bytes + 1))
                        except BlockingIOError:
                            continue
                        if not chunk:
                            try:
                                selector.unregister(key.fileobj)
                            except (KeyError, ValueError):
                                pass
                            if key.data == "stdout":
                                stdout_eof = True
                                if not ready_seen:
                                    close_stdin()
                            else:
                                stderr_eof = True
                        else:
                            if len(chunk) > rem_bytes:
                                target_buf.extend(chunk[:rem_bytes + 1])
                                ssh_status = "output_limit"
                                break
                            target_buf.extend(chunk)
                            if key.data == "stdout" and not ready_seen and len(wire_buf) >= HEADER.size:
                                magic, ch_num, ready_len = HEADER.unpack_from(wire_buf, 0)
                                if magic != b"TLFH1" or ch_num != 1 or ready_len > 4096:
                                    ready_seen = True
                                    close_stdin()
                                elif len(wire_buf) >= HEADER.size + ready_len:
                                    ready_seen = True
                                    ready_raw = bytes(wire_buf[HEADER.size:HEADER.size + ready_len])
                                    elapsed_so_far = max(0, time.monotonic_ns() - started_ns)
                                    grant_ns = max(1, work_limit_ns - elapsed_so_far)
                                    if (
                                        _validate_ready_frame(ready_raw, cfg, pub["config_sha256"])
                                        and elapsed_so_far < work_limit_ns
                                        and not stdin_closed
                                    ):
                                        grant_line = encoded({"remaining_ns": grant_ns}) + b"\n"
                                        stdin_queue.extend(grant_line + payload_raw)
                                        grant_dispatched = True
                                        try:
                                            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
                                        except KeyError:
                                            pass
                                    else:
                                        close_stdin()
                    if ssh_status is not None:
                        break
            except BaseException as run_err:
                if not isinstance(run_err, Exception) and not cancellations:
                    cancellations.append((run_err, run_err.__traceback__))
                else:
                    secondary_errors.append({"phase": "ssh_collect", "type": type(run_err).__name__})
                ssh_status = ssh_status or "program_error"
            finally:
                child_pid = getattr(process, "pid", None)
                if type(child_pid) is int and child_pid > 1:
                    cleanup_info = _cleanup_ssh(
                        process, constructor_completed, cancellations, secondary_errors
                    )
                for res_obj in (
                    selector,
                    None if stdin_closed else getattr(process, "stdin", None),
                    getattr(process, "stdout", None),
                    getattr(process, "stderr", None),
                ):
                    if res_obj is not None:
                        try:
                            res_obj.close()
                        except BaseException as c_err:
                            if not isinstance(c_err, Exception) and not cancellations:
                                cancellations.append((c_err, c_err.__traceback__))
                            else:
                                secondary_errors.append({"phase": "close_stream", "type": type(c_err).__name__})

        wire_bytes = bytes(wire_buf)
        stderr_bytes = bytes(stderr_buf)
        for fname, fbytes in (("wire.bin", wire_bytes), ("ssh-stderr.bin", stderr_bytes)):
            try:
                _write_exclusive(run_fd, fname, fbytes)
            except BaseException as w_err:
                persistence_errors.append(w_err)
                if not isinstance(w_err, Exception) and not cancellations:
                    cancellations.append((w_err, w_err.__traceback__))

        reaped = bool(cleanup_info and cleanup_info["child_reaped"])
        returncode = cleanup_info["returncode"] if reaped else None
        group_absent = bool(cleanup_info and cleanup_info["group_absent"])
        cleanup_bounded = bool(cleanup_info and cleanup_info["cleanup_bounded"])
        stdin_complete = grant_dispatched and stdin_offset == len(stdin_queue)

        tok_report = None
        decode_error = None
        if (
            ssh_status is None
            and reaped
            and returncode in (0, 1)
            and stdin_complete
            and stdout_eof
            and stderr_eof
            and group_absent
            and cleanup_bounded
            and not cancellations
        ):
            try:
                frames = decode_frames(wire_bytes)
                require(
                    _validate_ready_frame(frames["READY"], cfg, pub["config_sha256"]),
                    "READY frame mismatch",
                )
                done_obj = strict_json(frames["DONE"])
                require(
                    done_obj == {
                        "config_sha256": pub["config_sha256"],
                        "result_sha256": _sha256(frames["RESULT"]),
                    },
                    "DONE binding mismatch",
                )
                res_obj = strict_json(frames["RESULT"])
                require(
                    res_obj.get("kind") == "LORA_RECALIBRATION_TOKENIZER_REMOTE_RESULT"
                    and res_obj.get("projection_sha256") == cfg["projection_sha256"]
                    and res_obj.get("profile_sha256") == cfg["profile_sha256"],
                    "RESULT binding mismatch",
                )
                if res_obj["bootstrap_failure"] is None:
                    sess_out = res_obj["session_output"]
                    tok_report = build_synthetic_report(
                        validation_inputs,
                        profile_raw,
                        approved_profile_sha256,
                        sess_out["records"],
                        sess_out["finalization"],
                        failure=sess_out["failure"],
                        abort_failed=sess_out["abort_failed"],
                    )
                else:
                    decode_error = "bootstrap_failure:" + str(res_obj["bootstrap_failure"].get("phase"))
            except BaseException as dec_err:
                if not isinstance(dec_err, Exception) and not cancellations:
                    cancellations.append((dec_err, dec_err.__traceback__))
                decode_error = type(dec_err).__name__

        all_ok = bool(
            tok_report is not None
            and tok_report["public"]["status"] == "SYNTHETIC_COMPLETE"
            and tok_report["public"]["synthetic_all_rows_within_budget"] is True
            and tok_report["public"]["checked"] == 120
            and returncode == 0
        )
        binding_complete = bool(
            not persistence_errors
            and not cancellations
            and not secondary_errors
            and reaped
            and group_absent
            and cleanup_bounded
        )
        summary = {
            "kind": "LORA_RECALIBRATION_TOKENIZER_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "config_sha256": pub["config_sha256"],
            "inventory_sha256": pub["inventory_sha256"],
            "profile_sha256": pub["profile_sha256"],
            "one_time_scope_sha256": pub["one_time_scope_sha256"],
            "grant_dispatched": grant_dispatched,
            "returncode": returncode,
            "child_reaped": reaped,
            "local_group_absent": group_absent,
            "wire_sha256": _sha256(wire_bytes),
            "wire_bytes": len(wire_bytes),
            "ssh_stderr_bytes": len(stderr_bytes),
            "decode_error": decode_error,
            "secondary_errors": secondary_errors,
            "binding_complete": binding_complete,
            "collection_complete": bool(binding_complete and tok_report is not None),
            "all_120_rows_within_budget": bool(binding_complete and all_ok),
            "session_finalized": bool(
                binding_complete
                and tok_report is not None
                and tok_report["private"]["finalization"] is not None
            ),
            "tokenizer_report_public": tok_report["public"] if tok_report is not None else None,
            **{flag: False for flag in FALSE_FLAGS},
        }
        try:
            if tok_report is not None:
                _write_exclusive(run_fd, "tokenizer-report-full.json", _pretty_json(tok_report))
            _write_exclusive(run_fd, "tokenizer-controller-summary.json", _pretty_json(summary))
        except BaseException as sum_err:
            persistence_errors.append(sum_err)
            if not isinstance(sum_err, Exception) and not cancellations:
                cancellations.append((sum_err, sum_err.__traceback__))
    finally:
        try:
            os.close(run_fd)
        except BaseException as fd_err:
            persistence_errors.append(fd_err)
            if not isinstance(fd_err, Exception) and not cancellations:
                cancellations.append((fd_err, fd_err.__traceback__))

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {**summary, "tokenizer_report": tok_report}
