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
from recalibration_data_contract import FILE_PINS, _plain
from recalibration_execution_evidence import (
    RUNTIME_FIELDS,
    build_evidence,
    validate_ready,
)
from recalibration_execution_package import (
    BACKEND_SHA256,
    SOURCE_PINS,
    build_preparation,
    validate_preparation,
)
from recalibration_execution_remote import (
    CLEANUP_LIMIT_NS,
    FRAME_HEADER,
    MAX_WORK_NS,
    STAGE_CONFIG_FIELDS,
    WIRE_MAGIC,
    encode_control_frame,
)
from recalibration_execution_wire import HEADER, MAGIC, WIRE_LIMIT
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json


PEER_SOURCE_PINS = {
    **SOURCE_PINS,
    "recalibration_execution_package.py": "1feb26e08f62afdd1895b9cb32ffa3884a26ebada546ba807c931783a6382d90",
    "recalibration_execution_wire.py": "4b2bfb55b1f56cf8bd41d2f70cee970131715ae0268cfa4e0cc83ccccf1ad722",
    "recalibration_execution_evidence.py": "5d5e58433065650ed950fee326638fce6619e4372b74b7c631bd9727cbf3aac7",
    "recalibration_execution_remote.py": "64f977d8ec1ab963866e9116078e6799d89a66d94fde0989cc445922990ad67c",
}
LOADED_MODULE_MAP = {
    "pilot_protocol.py": "pilot_protocol",
    "validation_contract.py": "validation_contract",
    "validation_report.py": "validation_report",
    "validation_real_report.py": "validation_real_report",
    "recalibration_data_contract.py": "recalibration_data_contract",
    "recalibration_validation.py": "recalibration_validation",
    "recalibration_execution_package.py": "recalibration_execution_package",
    "recalibration_execution_wire.py": "recalibration_execution_wire",
    "recalibration_execution_evidence.py": "recalibration_execution_evidence",
    "recalibration_execution_remote.py": "recalibration_execution_remote",
}
SSH_STDERR_LIMIT = 131072
SSH_WORK_NS = 110_000_000_000
SSH_REAP_NS = 2_000_000_000
STAGE_TOTAL_NS = 1_800_000_000_000
STAGE_FINALIZE_RESERVE_NS = 5_000_000_000
STAGE_WORK_CUTOFF_DELTA_NS = STAGE_TOTAL_NS - STAGE_FINALIZE_RESERVE_NS
GRANT_RESERVE_NS = 5_000_000_000
MIN_BATCH_BUDGET_NS = MAX_WORK_NS + CLEANUP_LIMIT_NS + GRANT_RESERVE_NS
LIMITS_MANIFEST = {
    "worker_seconds": 2,
    "backend_seconds": 90,
    "ready_seconds": 10,
    "supervisor_work_seconds": 95,
    "supervisor_reap_seconds": 2,
    "ssh_work_seconds": 110,
    "ssh_reap_seconds": 2,
    "grant_reserve_seconds": 5,
    "stage_work_seconds": 1800,
    "stage_finalize_seconds": 5,
    "wire_bytes": WIRE_LIMIT,
    "ssh_stderr_bytes": SSH_STDERR_LIMIT,
}
PACKAGE_FALSE_FLAGS = (
    *FLAGS,
    "run_authorized",
    "dispatch_authorized",
    "trusted_collection_accepted",
    "dynamic_checks_passed",
    "full_dynamic_validation_complete",
    "tokenizer_length_validated",
)


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _check_hex(text, length):
    require(
        type(text) is str and re.fullmatch("[0-9a-f]{%d}" % length, text) is not None,
        "hex digest format",
    )


def _check_bounded_str(value, limit=4096):
    require(type(value) is str and 0 < len(value) <= limit, "bounded nonempty string")


def _check_stage_root_path(root_path, stage_id):
    require(
        type(root_path) is str and len(root_path) <= 256 and root_path.startswith("/"),
        "stage root syntax",
    )
    parts = root_path[1:].split("/")
    require(
        len(parts) >= 2
        and all(
            re.fullmatch("[A-Za-z0-9_.-]+", segment) is not None
            and segment not in (".", "..")
            for segment in parts
        )
        and parts[-1] == stage_id,
        "canonical stage root",
    )


class _FileErrors:
    def __init__(self):
        self.primary = None
        self.traceback = None
        self.phase = None
        self.secondary = []

    def capture(self, error, phase):
        self.secondary.extend(getattr(error, "_ctrl_secondary_errors", []))
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
            self.primary._ctrl_secondary_errors = self.secondary
            raise self.primary.with_traceback(self.traceback)


def _read_regular_nofollow(path_str, limit):
    descriptor = os.open(path_str, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    failures = _FileErrors()
    collected = bytearray()
    try:
        metadata = os.fstat(descriptor)
        require(
            stat.S_ISREG(metadata.st_mode) and metadata.st_size <= limit,
            "regular file size",
        )
        while len(collected) <= limit:
            chunk = os.read(descriptor, min(65536, limit + 1 - len(collected)))
            if not chunk:
                break
            collected.extend(chunk)
    except BaseException as read_err:
        failures.capture(read_err, "read_regular_body")
    finally:
        failures.close(descriptor, "read_regular_close")
    failures.raise_if_any()
    return bytes(collected)


def _write_exclusive(directory_fd, filename, raw):
    descriptor = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    failures = _FileErrors()
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        view = memoryview(raw)
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("nonpositive file write")
            offset += written
        os.fsync(descriptor)
        os.fsync(directory_fd)
    except BaseException as write_err:
        failures.capture(write_err, "write_exclusive_body")
    finally:
        failures.close(descriptor, "write_exclusive_close")
    failures.raise_if_any()


def _pretty_json_bytes(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def _verify_private_dir(descriptor):
    metadata = os.fstat(descriptor)
    require(
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        "private directory ownership or mode",
    )


def _claim_local_stage_root(local_stage_root, stage_id):
    _check_stage_root_path(local_stage_root, stage_id)
    parent_path = str(Path(local_stage_root).parent)
    require(os.path.realpath(parent_path) == parent_path, "canonical local stage parent")
    parent_fd = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    stage_fd = None
    failures = _FileErrors()
    try:
        _verify_private_dir(parent_fd)
        os.mkdir(stage_id, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        stage_fd = os.open(
            stage_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        os.fchmod(stage_fd, 0o700)
        _verify_private_dir(stage_fd)
    except BaseException as claim_err:
        failures.capture(claim_err, "claim_local_stage")
    finally:
        failures.close(parent_fd, "close_local_parent")
    if failures.primary is not None and stage_fd is not None:
        failures.close(stage_fd, "close_local_stage_on_error")
    failures.raise_if_any()
    return stage_fd


def _claim_local_batch_dir(stage_fd, batch_dir_name):
    os.mkdir(batch_dir_name, 0o700, dir_fd=stage_fd)
    os.fsync(stage_fd)
    batch_fd = os.open(
        batch_dir_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=stage_fd
    )
    failures = _FileErrors()
    try:
        os.fchmod(batch_fd, 0o700)
        _verify_private_dir(batch_fd)
    except BaseException as batch_err:
        failures.capture(batch_err, "claim_local_batch")
    if failures.primary is not None:
        failures.close(batch_fd, "close_local_batch_on_error")
        failures.raise_if_any()
    return batch_fd


def _verify_source_closure(full_source_raw):
    require(
        "validation_remote" not in sys.modules
        and "validation_task_worker" not in sys.modules,
        "forbidden backend/worker host import",
    )
    fields(full_source_raw, PEER_SOURCE_PINS)
    require(all(type(key) is str for key in full_source_raw), "source key type")
    for filename, expected_sha in PEER_SOURCE_PINS.items():
        raw = full_source_raw[filename]
        require(type(raw) is bytes and len(raw) <= 1048576, "peer source bytes")
        require(_sha256_bytes(raw) == expected_sha, "peer source digest: " + filename)
        mod_name = LOADED_MODULE_MAP[filename]
        loaded_mod = sys.modules.get(mod_name)
        require(loaded_mod is not None and getattr(loaded_mod, "__file__", None), "module loaded")
        loaded_path = os.path.realpath(loaded_mod.__file__)
        disk_raw = _read_regular_nofollow(loaded_path, 1048576)
        require(_sha256_bytes(disk_raw) == expected_sha, "loaded module drift: " + filename)
    self_path = os.path.realpath(__file__)
    self_raw = _read_regular_nofollow(self_path, 1048576)
    require(len(self_raw) <= 1048576, "controller source size")
    return {
        **PEER_SOURCE_PINS,
        "recalibration_execution_controller.py": _sha256_bytes(self_raw),
    }


def _validate_runtime_dict(runtime):
    _plain(runtime)
    fields(runtime, RUNTIME_FIELDS)
    for text_key in ("hostname", "python", "python_root"):
        _check_bounded_str(runtime[text_key])
    require(type(runtime["uid"]) is int and runtime["uid"] >= 0, "runtime uid")
    for sha_key in ("python_sha256", "bwrap_sha256", "seccomp_sha256"):
        _check_hex(runtime[sha_key], 64)
    require(runtime["no_new_privs"] is True, "runtime no_new_privs")
    require(
        type(runtime["argv_prefix"]) is list and len(runtime["argv_prefix"]) >= 4,
        "runtime argv_prefix",
    )
    for token in runtime["argv_prefix"]:
        _check_bounded_str(token)
    fields(runtime["host_namespaces"], ("mnt", "pid", "net", "user"))
    for ns_val in runtime["host_namespaces"].values():
        _check_bounded_str(ns_val)


def _validate_preflight_and_scope(
    preflight_raw,
    scope_raw,
    *,
    stage_id,
    run_ids,
    ssh_argv_prefix,
    ssh_environment,
    local_stage_root,
    remote_stage_root,
):
    preflight = parse_json(preflight_raw, limit=65536)
    _plain(preflight)
    fields(
        preflight,
        ("kind", "version", "status", "probes_passed", "remote_python", "expected_runtime"),
    )
    require(
        preflight["kind"] == "LORA_RECALIBRATION_LIVE_PREFLIGHT_RECEIPT"
        and type(preflight["version"]) is int
        and preflight["version"] == 1
        and preflight["status"] == "PREFLIGHT_PASSED"
        and preflight["probes_passed"] is True,
        "preflight status/probes",
    )
    runtime = preflight["expected_runtime"]
    _validate_runtime_dict(runtime)
    remote_python = preflight["remote_python"]
    _check_bounded_str(remote_python, limit=512)
    require(
        remote_python.startswith(runtime["python_root"].rstrip("/") + "/"),
        "remote_python must reside under expected_runtime python_root",
    )

    preflight_sha256 = _sha256_bytes(preflight_raw)
    scope = parse_json(scope_raw, limit=65536)
    _plain(scope)
    fields(
        scope,
        (
            "kind", "version", "stage_id", "run_ids", "local_stage_root",
            "remote_stage_root", "ssh_argv_prefix_sha256", "ssh_environment_sha256",
            "preflight_receipt_sha256",
        ),
    )
    require(
        scope["kind"] == "LORA_RECALIBRATION_STAGE_EXECUTION_SCOPE"
        and type(scope["version"]) is int
        and scope["version"] == 1
        and scope["stage_id"] == stage_id
        and scope["run_ids"] == list(run_ids)
        and scope["local_stage_root"] == local_stage_root
        and scope["remote_stage_root"] == remote_stage_root
        and scope["preflight_receipt_sha256"] == preflight_sha256,
        "scope binding mismatch",
    )
    require(
        type(ssh_argv_prefix) is list
        and len(ssh_argv_prefix) >= 2
        and all(type(arg) is str and 0 < len(arg) <= 4096 for arg in ssh_argv_prefix),
        "ssh_argv_prefix",
    )
    require(
        type(ssh_environment) is dict
        and bool(ssh_environment)
        and all(
            type(env_k) is str and type(env_v) is str and 0 < len(env_k) <= 128 and len(env_v) <= 4096
            for env_k, env_v in ssh_environment.items()
        ),
        "ssh_environment",
    )
    require(
        scope["ssh_argv_prefix_sha256"] == digest(ssh_argv_prefix)
        and scope["ssh_environment_sha256"] == digest(ssh_environment),
        "SSH argv/environment scope digest mismatch",
    )
    _check_stage_root_path(local_stage_root, stage_id)
    _check_stage_root_path(remote_stage_root, stage_id)
    return preflight, preflight_sha256, _sha256_bytes(scope_raw)


def _encode_bootstrap_frames(package_sha256, config_sha256, batch_index, launch_config, backend_raw):
    bootstrap_obj = {
        "kind": "LORA_RECALIBRATION_BOOTSTRAP",
        "version": 1,
        "package_sha256": package_sha256,
        "config_sha256": config_sha256,
        "batch_index": batch_index,
        "launch_config": launch_config,
    }
    return (
        encode_control_frame(1, encoded(bootstrap_obj))
        + encode_control_frame(2, backend_raw)
    )


def _encode_grant_frames(ready_raw, granted_work_ns, bundle_raw):
    grant_obj = {
        "kind": "LORA_RECALIBRATION_GRANT",
        "version": 1,
        "ready_sha256": _sha256_bytes(ready_raw),
        "granted_work_ns": granted_work_ns,
        "bundle_sha256": _sha256_bytes(bundle_raw),
        "bundle_bytes": len(bundle_raw),
    }
    return (
        encode_control_frame(3, encoded(grant_obj))
        + encode_control_frame(4, bundle_raw)
    )


def build_launch_package(
    raw_files,
    worker_raw,
    backend_raw,
    full_source_raw,
    preflight_raw,
    scope_raw,
    *,
    stage_id,
    run_ids,
    ssh_argv_prefix,
    ssh_environment,
    local_stage_root,
    remote_stage_root,
):
    local_sources = _verify_source_closure(full_source_raw)
    preflight, preflight_sha256, scope_sha256 = _validate_preflight_and_scope(
        preflight_raw,
        scope_raw,
        stage_id=stage_id,
        run_ids=run_ids,
        ssh_argv_prefix=ssh_argv_prefix,
        ssh_environment=ssh_environment,
        local_stage_root=local_stage_root,
        remote_stage_root=remote_stage_root,
    )
    base_sources = {name: full_source_raw[name] for name in SOURCE_PINS}
    preparation = build_preparation(
        raw_files,
        worker_raw,
        backend_raw,
        base_sources,
        stage_id=stage_id,
        run_ids=run_ids,
    )
    prep_public = preparation["public"]
    remote_raw = full_source_raw["recalibration_execution_remote.py"]
    require(len(remote_raw) <= 65536, "remote supervisor source size")
    remote_source_text = remote_raw.decode("utf-8")
    remote_python = preflight["remote_python"]

    batch_entries = []
    for batch_meta in prep_public["batches"]:
        idx = batch_meta["batch_index"]
        batch_entries.append({
            "batch_index": idx,
            "run_id": batch_meta["run_id"],
            "bundle_sha256": batch_meta["bundle_sha256"],
            "bundle_bytes": batch_meta["bundle_bytes"],
            "batch_root": remote_stage_root + ("/batch-%02d-" % idx) + batch_meta["run_id"],
        })

    launch_config = {
        "kind": "LORA_RECALIBRATION_STAGE_CONFIG",
        "version": 1,
        "stage_id": stage_id,
        "preparation_sha256": prep_public["preparation_sha256"],
        "plan_sha256": prep_public["plan_sha256"],
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "remote_supervisor_sha256": PEER_SOURCE_PINS["recalibration_execution_remote.py"],
        "ssh_argv_prefix": list(ssh_argv_prefix),
        "remote_python": remote_python,
        "local_stage_root": local_stage_root,
        "remote_stage_root": remote_stage_root,
        "preflight_receipt_sha256": preflight_sha256,
        "expected_runtime": copy.deepcopy(preflight["expected_runtime"]),
        "batches": batch_entries,
        "limits": dict(LIMITS_MANIFEST),
    }
    fields(launch_config, STAGE_CONFIG_FIELDS)
    config_sha256 = digest(launch_config)

    ssh_argvs = []
    commands_public = []
    for batch_entry in batch_entries:
        idx = batch_entry["batch_index"]
        remote_command = "exec %s -I -S -B -c %s %s %d" % (
            shlex.quote(remote_python),
            shlex.quote(remote_source_text),
            config_sha256,
            idx,
        )
        full_argv = [*ssh_argv_prefix, remote_command]
        ssh_argvs.append(full_argv)
        commands_public.append({
            "batch_index": idx,
            "run_id": batch_entry["run_id"],
            "argv_sha256": digest(full_argv),
        })

    manifest = {
        "kind": "LORA_RECALIBRATION_LAUNCH_PACKAGE",
        "version": 1,
        "data_version": "recalibration_v2_r1",
        "stage_id": stage_id,
        "run_ids": list(run_ids),
        "preparation_sha256": prep_public["preparation_sha256"],
        "config_sha256": config_sha256,
        "preflight_receipt_sha256": preflight_sha256,
        "one_time_scope_sha256": scope_sha256,
        "file_sha256": dict(FILE_PINS),
        "local_source_sha256": local_sources,
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "launch_config": launch_config,
        "commands": commands_public,
        "launch_package_structure_valid": True,
        **{flag: False for flag in PACKAGE_FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "preparation": preparation,
            "bundles_hex": list(preparation["private"]["bundles_hex"]),
            "ssh_argvs": ssh_argvs,
            "ssh_environment": dict(ssh_environment),
        },
    }


def validate_launch_package(
    package,
    raw_files,
    worker_raw,
    backend_raw,
    full_source_raw,
    preflight_raw,
    scope_raw,
):
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    require(type(pub) is dict and type(priv) is dict, "package sections")
    launch_cfg = pub.get("launch_config")
    require(type(launch_cfg) is dict, "launch_config dict")
    rebuilt = build_launch_package(
        raw_files,
        worker_raw,
        backend_raw,
        full_source_raw,
        preflight_raw,
        scope_raw,
        stage_id=pub.get("stage_id"),
        run_ids=pub.get("run_ids"),
        ssh_argv_prefix=launch_cfg.get("ssh_argv_prefix"),
        ssh_environment=priv.get("ssh_environment"),
        local_stage_root=launch_cfg.get("local_stage_root"),
        remote_stage_root=launch_cfg.get("remote_stage_root"),
    )
    require(encoded(package) == encoded(rebuilt), "launch package reconstruction mismatch")
    validate_preparation(
        priv["preparation"],
        raw_files,
        worker_raw,
        backend_raw,
        {name: full_source_raw[name] for name in SOURCE_PINS},
        stage_id=pub["stage_id"],
        run_ids=pub["run_ids"],
    )
    return rebuilt


class _Clock:
    def __init__(self):
        self.last_ns = None
        self.invalid = False

    def now_ns(self):
        current = time.monotonic_ns()
        if (
            self.invalid
            or type(current) is not int
            or current < 0
            or (self.last_ns is not None and current < self.last_ns)
        ):
            self.invalid = True
            raise RuntimeError("monotonic clock invalid")
        self.last_ns = current
        return current


def _safe_raw_ns():
    try:
        val = time.monotonic_ns()
        if type(val) is int and val >= 0:
            return val
    except BaseException:
        return None
    return None


def _cleanup_ssh_process(process, constructor_completed, clock, cancellations, secondary_errors):
    start_raw = _safe_raw_ns()
    clock_ok = not clock.invalid and start_raw is not None
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

    def record_cleanup_err(err, phase):
        if not isinstance(err, Exception) and not cancellations:
            cancellations.append((err, err.__traceback__))
        else:
            secondary_errors.append({"phase": phase, "type": type(err).__name__})

    def probe_group():
        nonlocal group_absent
        try:
            os.killpg(pid, 0)
            group_absent = False
            return True
        except ProcessLookupError:
            group_absent = True
            return False
        except BaseException as probe_err:
            group_absent = None
            record_cleanup_err(probe_err, "ssh_cleanup_probe")
            return None

    def send_group(signum):
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass
        except BaseException as sig_err:
            record_cleanup_err(sig_err, "ssh_cleanup_signal")

    def send_direct(signum):
        if constructor_completed or child_reaped:
            return
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass
        except BaseException as dir_err:
            record_cleanup_err(dir_err, "ssh_cleanup_child")

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
        except BaseException as wait_err:
            record_cleanup_err(wait_err, "ssh_cleanup_wait")

    sent_kill = False
    try:
        alive = probe_group()
        if alive is not False:
            send_group(signal.SIGTERM)
        send_direct(signal.SIGTERM)
        cur_ns = sample_raw()
        initial_wait = (
            min(0.2, max(0.0, (deadline_raw - cur_ns) / 1_000_000_000.0))
            if cur_ns is not None and deadline_raw is not None
            else 0.05
        )
        reap_once(initial_wait)
        for step_idx in range(110):
            observed_ns = sample_raw()
            alive = probe_group()
            reap_once(0.0)
            if alive is False and child_reaped:
                break
            elapsed_ok = (
                (observed_ns - start_raw >= 200_000_000)
                if (observed_ns is not None and start_raw is not None)
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
            if observed_ns is not None and deadline_raw is not None and observed_ns >= deadline_raw:
                break
            after_ns = sample_raw()
            if after_ns is not None and deadline_raw is not None:
                rem_ns = deadline_raw - after_ns
                if rem_ns <= 0:
                    break
                sleep_sec = min(0.02, rem_ns / 1_000_000_000.0)
            else:
                sleep_sec = 0.01
            time.sleep(sleep_sec)
    except BaseException as loop_err:
        clock_ok = False
        record_cleanup_err(loop_err, "ssh_cleanup_loop")
        send_group(signal.SIGKILL)
        send_direct(signal.SIGKILL)
        reap_once(0.0)
        probe_group()

    end_raw = sample_raw()
    if clock_ok and start_raw is not None and end_raw is not None and deadline_raw is not None:
        elapsed_ns = end_raw - start_raw
        bounded = end_raw <= deadline_raw
    else:
        elapsed_ns = SSH_REAP_NS + 1
        bounded = False
    return {
        "child_reaped": child_reaped,
        "returncode": returncode if child_reaped else None,
        "group_absent": group_absent is True,
        "cleanup_bounded": bounded,
        "cleanup_elapsed_ns": elapsed_ns,
    }


def _execute_ssh_batch(
    argv,
    ssh_environment,
    bootstrap_frames,
    bundle_raw,
    expected_ready,
    clock,
    stage_work_cutoff_ns,
    cancellations,
    secondary_errors,
):
    batch_started_ns = _safe_raw_ns()
    ssh_deadline_ns = stage_work_cutoff_ns
    process = object.__new__(subprocess.Popen)
    constructor_completed = False
    selector = None
    stdin_queue = bytearray(bootstrap_frames)
    stdin_offset = 0
    stdin_closed = False
    bootstrap_flushed = False
    ready_processed = False
    grant_dispatched = False
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    stdout_eof = False
    stderr_eof = False
    ssh_status = None

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
        stdin_pipe = getattr(process, "stdin", None)
        if stdin_pipe is not None:
            try:
                stdin_pipe.close()
            except BaseException as stdin_err:
                if not isinstance(stdin_err, Exception) and not cancellations:
                    cancellations.append((stdin_err, stdin_err.__traceback__))
                else:
                    secondary_errors.append({
                        "phase": "ssh_close_stdin", "type": type(stdin_err).__name__,
                    })

    try:
        batch_started_ns = clock.now_ns()
        ssh_deadline_ns = min(stage_work_cutoff_ns, batch_started_ns + SSH_WORK_NS)
        try:
            subprocess.Popen.__init__(
                process,
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/",
                env=dict(ssh_environment),
                close_fds=True,
                start_new_session=True,
                shell=False,
            )
            constructor_completed = True
        except BaseException as spawn_err:
            if not isinstance(spawn_err, Exception) and not cancellations:
                cancellations.append((spawn_err, spawn_err.__traceback__))
            child_pid = getattr(process, "pid", None)
            if type(child_pid) is not int or child_pid <= 1:
                raw_end = _safe_raw_ns()
                elapsed = max(0, raw_end - batch_started_ns) if raw_end is not None else 0
                return {
                    "ssh": {
                        "status": "launch_failed",
                        "returncode": None,
                        "stdout_hex": "",
                        "stderr_hex": "",
                        "elapsed_ns": elapsed,
                        "process_reaped": False,
                        "stdin_complete": False,
                    },
                    "wire_raw": b"",
                    "stderr_raw": b"",
                    "grant_dispatched": False,
                }
            ssh_status = "program_error"

        if ssh_status is None:
            selector = selectors.DefaultSelector()
            os.set_blocking(process.stdout.fileno(), False)
            os.set_blocking(process.stderr.fileno(), False)
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")

            while True:
                now_ns = clock.now_ns()
                if now_ns >= ssh_deadline_ns:
                    ssh_status = "timeout"
                    break
                polled = process.poll()
                if not selector.get_map() and polled is not None:
                    break
                remaining_sec = (ssh_deadline_ns - now_ns) / 1_000_000_000.0
                if not selector.get_map():
                    time.sleep(min(0.02, remaining_sec))
                    continue
                events = selector.select(min(0.05, remaining_sec))
                if clock.now_ns() >= ssh_deadline_ns:
                    ssh_status = "timeout"
                    break
                for key, _mask in events:
                    if key.data == "stdin":
                        if stdin_closed:
                            continue
                        if stdin_offset < len(stdin_queue):
                            try:
                                written = _io_write(
                                    key.fd, stdin_queue[stdin_offset:stdin_offset + 16384]
                                )
                            except BlockingIOError:
                                continue
                            except BrokenPipeError:
                                close_stdin()
                                continue
                            if written > 0:
                                stdin_offset += written
                        if stdin_offset >= len(stdin_queue):
                            if not bootstrap_flushed:
                                bootstrap_flushed = True
                                if not stdin_closed:
                                    try:
                                        selector.unregister(key.fileobj)
                                    except (KeyError, ValueError):
                                        pass
                            elif ready_processed and not stdin_closed:
                                close_stdin()
                        continue

                    target_buf, cap = (
                        (stdout_buf, WIRE_LIMIT)
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
                        else:
                            stderr_eof = True
                        if key.data == "stdout" and not ready_processed:
                            close_stdin()
                    else:
                        if len(chunk) > rem_bytes:
                            target_buf.extend(chunk[:rem_bytes + 1])
                            ssh_status = "output_limit"
                            break
                        target_buf.extend(chunk)
                        if key.data == "stdout" and not ready_processed and len(stdout_buf) >= HEADER.size:
                            magic, channel, ready_len = HEADER.unpack_from(stdout_buf, 0)
                            if magic != MAGIC or channel != 1 or ready_len > 4096:
                                ready_processed = True
                                close_stdin()
                            elif len(stdout_buf) >= HEADER.size + ready_len:
                                ready_processed = True
                                ready_raw = bytes(stdout_buf[HEADER.size:HEADER.size + ready_len])
                                ready_ok = False
                                try:
                                    validate_ready(ready_raw, expected=expected_ready)
                                    ready_ok = True
                                except ValueError:
                                    ready_ok = False
                                post_ready_ns = clock.now_ns()
                                remaining_window_ns = min(stage_work_cutoff_ns, ssh_deadline_ns) - post_ready_ns
                                if (
                                    ready_ok
                                    and bootstrap_flushed
                                    and not stdin_closed
                                    and remaining_window_ns >= MIN_BATCH_BUDGET_NS
                                ):
                                    grant_frames = _encode_grant_frames(
                                        ready_raw, MAX_WORK_NS, bundle_raw
                                    )
                                    stdin_queue.extend(grant_frames)
                                    grant_dispatched = True
                                    try:
                                        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
                                    except KeyError:
                                        pass
                                else:
                                    close_stdin()
                if ssh_status is not None:
                    break
    except BaseException as loop_err:
        if not isinstance(loop_err, Exception) and not cancellations:
            cancellations.append((loop_err, loop_err.__traceback__))
        else:
            secondary_errors.append({"phase": "ssh_collect", "type": type(loop_err).__name__})
        ssh_status = ssh_status or "program_error"
    finally:
        cleanup_info = None
        child_pid = getattr(process, "pid", None)
        if type(child_pid) is int and child_pid > 1:
            try:
                cleanup_info = _cleanup_ssh_process(
                    process, constructor_completed, clock, cancellations, secondary_errors
                )
            except BaseException as clean_err:
                if not isinstance(clean_err, Exception) and not cancellations:
                    cancellations.append((clean_err, clean_err.__traceback__))
                else:
                    secondary_errors.append({"phase": "ssh_cleanup_outer", "type": type(clean_err).__name__})
                ssh_status = ssh_status or "program_error"
        for resource in (
            selector,
            None if stdin_closed else getattr(process, "stdin", None),
            getattr(process, "stdout", None),
            getattr(process, "stderr", None),
        ):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as close_err:
                    if not isinstance(close_err, Exception) and not cancellations:
                        cancellations.append((close_err, close_err.__traceback__))
                    else:
                        secondary_errors.append({"phase": "ssh_close_stream", "type": type(close_err).__name__})
                    ssh_status = ssh_status or "program_error"

    raw_ended_ns = _safe_raw_ns()
    if not clock.invalid and raw_ended_ns is not None and raw_ended_ns >= batch_started_ns:
        elapsed_ns = raw_ended_ns - batch_started_ns
    else:
        elapsed_ns = SSH_WORK_NS + SSH_REAP_NS + 1
    reaped = bool(cleanup_info and cleanup_info["child_reaped"])
    returncode = cleanup_info["returncode"] if reaped else None
    group_absent = bool(cleanup_info and cleanup_info["group_absent"])
    cleanup_bounded = bool(cleanup_info and cleanup_info["cleanup_bounded"])
    stdin_complete = stdin_offset == len(stdin_queue)

    if ssh_status is None:
        if (
            not clock.invalid
            and reaped
            and returncode == 0
            and stdin_complete
            and stdout_eof
            and stderr_eof
            and group_absent
            and cleanup_bounded
            and not cancellations
            and elapsed_ns <= SSH_WORK_NS + SSH_REAP_NS
        ):
            ssh_status = "completed"
        else:
            ssh_status = "program_error"

    wire_raw = bytes(stdout_buf)
    stderr_raw = bytes(stderr_buf)
    return {
        "ssh": {
            "status": ssh_status,
            "returncode": returncode,
            "stdout_hex": wire_raw.hex(),
            "stderr_hex": stderr_raw.hex(),
            "elapsed_ns": elapsed_ns,
            "process_reaped": reaped,
            "stdin_complete": stdin_complete,
        },
        "wire_raw": wire_raw,
        "stderr_raw": stderr_raw,
        "grant_dispatched": grant_dispatched,
    }


def run_stage(
    package,
    raw_files,
    worker_raw,
    backend_raw,
    full_source_raw,
    preflight_raw,
    scope_raw,
):
    validated = validate_launch_package(
        package,
        raw_files,
        worker_raw,
        backend_raw,
        full_source_raw,
        preflight_raw,
        scope_raw,
    )
    pub = validated["public"]
    priv = validated["private"]
    launch_cfg = pub["launch_config"]
    stage_id = pub["stage_id"]
    run_ids = pub["run_ids"]
    package_sha256 = pub["package_sha256"]
    config_sha256 = pub["config_sha256"]
    local_stage_root = launch_cfg["local_stage_root"]
    remote_stage_root = launch_cfg["remote_stage_root"]
    expected_runtime = launch_cfg["expected_runtime"]
    base_sources = {name: full_source_raw[name] for name in SOURCE_PINS}
    evidence_context = {
        "package_sha256": package_sha256,
        "config_sha256": config_sha256,
        "stage_root": remote_stage_root,
        "expected_runtime": expected_runtime,
    }

    stage_fd = _claim_local_stage_root(local_stage_root, stage_id)
    clock = _Clock()
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    collected_records = []
    batch_dispatch_log = []
    controller_stop_reason = None
    evidence = None

    try:
        try:
            _write_exclusive(
                stage_fd,
                "stage-claim.json",
                _pretty_json_bytes({
                    "kind": "LORA_RECALIBRATION_LOCAL_STAGE_CLAIM",
                    "stage_id": stage_id,
                    "package_sha256": package_sha256,
                    "config_sha256": config_sha256,
                    "one_time_scope_sha256": pub["one_time_scope_sha256"],
                    "consumed_before_ssh": True,
                }),
            )
            _write_exclusive(stage_fd, "package-public.json", _pretty_json_bytes(pub))
        except BaseException as init_err:
            persistence_errors.append(init_err)
            if not isinstance(init_err, Exception) and not cancellations:
                cancellations.append((init_err, init_err.__traceback__))
            controller_stop_reason = "stage_claim_persistence_failed"

        stage_started_ns = None
        stage_work_cutoff_ns = None
        try:
            stage_started_ns = clock.now_ns()
            stage_work_cutoff_ns = stage_started_ns + STAGE_WORK_CUTOFF_DELTA_NS
        except BaseException as clk_init_err:
            if not isinstance(clk_init_err, Exception) and not cancellations:
                cancellations.append((clk_init_err, clk_init_err.__traceback__))
            else:
                secondary_errors.append({"phase": "stage_clock_init", "type": type(clk_init_err).__name__})
            controller_stop_reason = controller_stop_reason or "clock_invalid"

        if not persistence_errors and not cancellations and not clock.invalid:
            for batch_index in range(15):
                try:
                    _verify_source_closure(full_source_raw)
                    validate_launch_package(
                        validated,
                        raw_files,
                        worker_raw,
                        backend_raw,
                        full_source_raw,
                        preflight_raw,
                        scope_raw,
                    )
                except BaseException as drift_err:
                    if not isinstance(drift_err, Exception) and not cancellations:
                        cancellations.append((drift_err, drift_err.__traceback__))
                    secondary_errors.append({"phase": "pre_batch_verify", "type": type(drift_err).__name__})
                    controller_stop_reason = "pre_batch_verification_failed"
                    break

                try:
                    rem_stage_ns = stage_work_cutoff_ns - clock.now_ns()
                except BaseException as clk_err:
                    if not isinstance(clk_err, Exception) and not cancellations:
                        cancellations.append((clk_err, clk_err.__traceback__))
                    else:
                        secondary_errors.append({"phase": "pre_batch_clock", "type": type(clk_err).__name__})
                    controller_stop_reason = "clock_invalid"
                    break

                if rem_stage_ns < MIN_BATCH_BUDGET_NS:
                    controller_stop_reason = "stage_budget_exhausted"
                    break

                batch_entry = launch_cfg["batches"][batch_index]
                batch_dir_name = ("batch-%02d-" % batch_index) + batch_entry["run_id"]
                bundle_raw = bytes.fromhex(priv["bundles_hex"][batch_index])
                argv = priv["ssh_argvs"][batch_index]

                batch_fd = None
                try:
                    batch_fd = _claim_local_batch_dir(stage_fd, batch_dir_name)
                    _write_exclusive(
                        batch_fd,
                        "batch-claim.json",
                        _pretty_json_bytes({
                            "kind": "LORA_RECALIBRATION_LOCAL_BATCH_CLAIM",
                            "stage_id": stage_id,
                            "batch_index": batch_index,
                            "run_id": batch_entry["run_id"],
                            "package_sha256": package_sha256,
                            "config_sha256": config_sha256,
                            "argv_sha256": pub["commands"][batch_index]["argv_sha256"],
                            "bundle_sha256": batch_entry["bundle_sha256"],
                        }),
                    )
                except BaseException as claim_err:
                    secondary_errors.extend(getattr(claim_err, "_ctrl_secondary_errors", []))
                    persistence_errors.append(claim_err)
                    if not isinstance(claim_err, Exception) and not cancellations:
                        cancellations.append((claim_err, claim_err.__traceback__))
                    controller_stop_reason = "batch_claim_persistence_failed"
                    if batch_fd is not None:
                        try:
                            os.close(batch_fd)
                        except BaseException as bclose_claim_err:
                            if not isinstance(bclose_claim_err, Exception) and not cancellations:
                                cancellations.append((bclose_claim_err, bclose_claim_err.__traceback__))
                            else:
                                secondary_errors.append({
                                    "phase": "close_batch_after_claim_error",
                                    "type": type(bclose_claim_err).__name__,
                                })
                    break

                record_appended = False
                try:
                    expected_ready = {
                        "kind": "LORA_RECALIBRATION_READY",
                        "version": 1,
                        "stage_id": stage_id,
                        "batch_index": batch_index,
                        "run_id": batch_entry["run_id"],
                        "preparation_sha256": pub["preparation_sha256"],
                        "package_sha256": package_sha256,
                        "config_sha256": config_sha256,
                        "bundle_sha256": batch_entry["bundle_sha256"],
                        "backend_sha256": BACKEND_SHA256,
                        "worker_sha256": WORKER_SHA256,
                        "hostname": expected_runtime["hostname"],
                        "uid": expected_runtime["uid"],
                        "stage_root": remote_stage_root,
                        "batch_root": batch_entry["batch_root"],
                        "expected_runtime_sha256": digest(expected_runtime),
                    }
                    bootstrap_frames = _encode_bootstrap_frames(
                        package_sha256,
                        config_sha256,
                        batch_index,
                        launch_cfg,
                        backend_raw,
                    )
                    outcome = _execute_ssh_batch(
                        argv,
                        priv["ssh_environment"],
                        bootstrap_frames,
                        bundle_raw,
                        expected_ready,
                        clock,
                        stage_work_cutoff_ns,
                        cancellations,
                        secondary_errors,
                    )
                    wire_sha256 = _sha256_bytes(outcome["wire_raw"])
                    collected_records.append({
                        "batch_index": batch_index,
                        "bundle_raw": bundle_raw,
                        "ssh": outcome["ssh"],
                        "wire_sha256": wire_sha256,
                    })
                    record_appended = True
                    batch_dispatch_log.append({
                        "batch_index": batch_index,
                        "run_id": batch_entry["run_id"],
                        "grant_dispatched": outcome["grant_dispatched"],
                        "wire_bytes": len(outcome["wire_raw"]),
                        "wire_sha256": wire_sha256,
                        "ssh_stderr_bytes": len(outcome["stderr_raw"]),
                        "ssh_stderr_sha256": _sha256_bytes(outcome["stderr_raw"]),
                        "ssh_status": outcome["ssh"]["status"],
                    })

                    for save_name, save_bytes in (
                        ("wire.bin", outcome["wire_raw"]),
                        ("ssh-stderr.bin", outcome["stderr_raw"]),
                        ("batch-transport.json", _pretty_json_bytes(outcome["ssh"])),
                    ):
                        try:
                            _write_exclusive(batch_fd, save_name, save_bytes)
                        except BaseException as save_err:
                            persistence_errors.append(save_err)
                            if not isinstance(save_err, Exception) and not cancellations:
                                cancellations.append((save_err, save_err.__traceback__))
                            else:
                                secondary_errors.append({
                                    "phase": "persist_" + save_name,
                                    "type": type(save_err).__name__,
                                })
                except BaseException as batch_exec_err:
                    if not isinstance(batch_exec_err, Exception) and not cancellations:
                        cancellations.append((batch_exec_err, batch_exec_err.__traceback__))
                    else:
                        secondary_errors.append({
                            "phase": "batch_execution",
                            "type": type(batch_exec_err).__name__,
                        })
                    if not record_appended:
                        empty_sha = _sha256_bytes(b"")
                        collected_records.append({
                            "batch_index": batch_index,
                            "bundle_raw": bundle_raw,
                            "ssh": {
                                "status": "launch_failed",
                                "returncode": None,
                                "stdout_hex": "",
                                "stderr_hex": "",
                                "elapsed_ns": 0,
                                "process_reaped": False,
                                "stdin_complete": False,
                            },
                            "wire_sha256": empty_sha,
                        })
                    controller_stop_reason = (
                        "cancelled" if cancellations else "batch_execution_failed"
                    )
                finally:
                    try:
                        os.close(batch_fd)
                    except BaseException as bclose_err:
                        persistence_errors.append(bclose_err)
                        if not isinstance(bclose_err, Exception) and not cancellations:
                            cancellations.append((bclose_err, bclose_err.__traceback__))

                if cancellations or persistence_errors or controller_stop_reason is not None:
                    controller_stop_reason = controller_stop_reason or (
                        "cancelled" if cancellations else "batch_persistence_failed"
                    )
                    break

                try:
                    partial_evidence = build_evidence(
                        priv["preparation"],
                        raw_files,
                        worker_raw,
                        backend_raw,
                        base_sources,
                        collected_records,
                        stage_id=stage_id,
                        run_ids=run_ids,
                        context=evidence_context,
                    )
                except BaseException as ev_err:
                    if not isinstance(ev_err, Exception) and not cancellations:
                        cancellations.append((ev_err, ev_err.__traceback__))
                    else:
                        secondary_errors.append({"phase": "build_evidence", "type": type(ev_err).__name__})
                    controller_stop_reason = "evidence_reconstruction_failed"
                    break
                if not partial_evidence["batches"][-1]["can_continue"]:
                    controller_stop_reason = partial_evidence["batches"][-1]["stop_reason"]
                    evidence = partial_evidence
                    break
                evidence = partial_evidence

        if evidence is None or len(evidence["batches"]) != len(collected_records):
            try:
                evidence = build_evidence(
                    priv["preparation"],
                    raw_files,
                    worker_raw,
                    backend_raw,
                    base_sources,
                    collected_records,
                    stage_id=stage_id,
                    run_ids=run_ids,
                    context=evidence_context,
                )
            except BaseException as final_ev_err:
                if not isinstance(final_ev_err, Exception) and not cancellations:
                    cancellations.append((final_ev_err, final_ev_err.__traceback__))
                else:
                    secondary_errors.append({
                        "phase": "final_build_evidence",
                        "type": type(final_ev_err).__name__,
                    })

        stage_elapsed_ns = STAGE_TOTAL_NS + 1
        stage_bounded = False
        if not clock.invalid and stage_started_ns is not None:
            try:
                end_stage_ns = clock.now_ns()
                stage_elapsed_ns = max(0, end_stage_ns - stage_started_ns)
                stage_bounded = stage_elapsed_ns <= STAGE_TOTAL_NS
            except BaseException as end_clk_err:
                if not isinstance(end_clk_err, Exception) and not cancellations:
                    cancellations.append((end_clk_err, end_clk_err.__traceback__))
                else:
                    secondary_errors.append({"phase": "stage_end_clock", "type": type(end_clk_err).__name__})
                controller_stop_reason = controller_stop_reason or "clock_invalid"
        else:
            controller_stop_reason = controller_stop_reason or "clock_invalid"

        binding_complete = bool(
            not clock.invalid
            and not persistence_errors
            and not cancellations
            and not secondary_errors
            and stage_bounded
        )
        summary = {
            "kind": "LORA_RECALIBRATION_CONTROLLER_SUMMARY",
            "version": 1,
            "stage_id": stage_id,
            "package_sha256": package_sha256,
            "config_sha256": config_sha256,
            "preflight_receipt_sha256": pub["preflight_receipt_sha256"],
            "one_time_scope_sha256": pub["one_time_scope_sha256"],
            "batches_attempted": len(collected_records),
            "batch_dispatch_log": batch_dispatch_log,
            "controller_stop_reason": controller_stop_reason,
            "stage_elapsed_ns": stage_elapsed_ns,
            "stage_bounded": stage_bounded,
            "secondary_errors": secondary_errors,
            "binding_complete": binding_complete,
            "collection_complete": bool(
                binding_complete and evidence is not None and evidence["collection_complete"]
            ),
            "evidence_sha256": digest(evidence) if evidence is not None else None,
            **{flag: False for flag in PACKAGE_FALSE_FLAGS},
        }

        final_targets = [("controller-summary.json", summary)]
        if evidence is not None:
            final_targets.insert(0, ("execution-evidence.json", evidence))
        for final_name, final_obj in final_targets:
            try:
                _write_exclusive(stage_fd, final_name, _pretty_json_bytes(final_obj))
            except BaseException as fin_err:
                persistence_errors.append(fin_err)
                if not isinstance(fin_err, Exception) and not cancellations:
                    cancellations.append((fin_err, fin_err.__traceback__))
    finally:
        try:
            os.close(stage_fd)
        except BaseException as sclose_err:
            persistence_errors.append(sclose_err)
            if not isinstance(sclose_err, Exception) and not cancellations:
                cancellations.append((sclose_err, sclose_err.__traceback__))

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {**summary, "evidence": evidence}
