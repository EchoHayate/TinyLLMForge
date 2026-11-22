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
from recalibration_data_contract import FILE_PINS, _plain, load_dataset
from recalibration_diagnostic_remote import (
    HEADER as PHASE1_HEADER,
    PHASE1_SOURCE_NAMES,
    WIRE_LIMIT as PHASE1_WIRE_LIMIT,
    decode_frames,
    strict_json,
    validate_phase_one_bundle,
)
from recalibration_diagnostic_scoring import (
    _decode_scoring_batch,
    build_continuation_bundle,
    build_final_scoring_bundles,
    build_generation_schedule,
    plan_final_positions,
    score_diagnostic_run,
)
from recalibration_execution_evidence import RUNTIME_FIELDS
from recalibration_execution_package import BACKEND_SHA256
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json
from validation_real_report import TRANSPORT_STDERR_LIMIT, TRANSPORT_STDOUT_LIMIT


PHASE1_MODULE_PINS = {
    "generation_core.py": "dfccfd21d5a526fb9a7853c1b245a333c1c805ec17e7748902179f80e302372d",
    "observed_generation.py": "8fd18d126cce0b97cce5c90c2edeb4f2859959d3a4ad76864c2711b8571c0efd",
    "pilot_protocol.py": "5074980a2b5b9d8ec0a52288d7cb62cbc847688f094cd8ba189a7135ba315c97",
    "prompt_rendering.py": "2016c3eb0255d1d4a40f6072c7407b099fa3e4e569345c8d1ac778d5bd7bb18f",
    "recalibration_data_contract.py": "52695a9e2fb7e90b04f4fc7c95dcdc8b0310bbb0c269e6779890b9dff989ed02",
    "recalibration_diagnostic_scoring.py": "c45bb0b067ac3d198aa1f84610f63b5d10d1788fd353f49d88b6dfe19f39615b",
    "recalibration_rendering.py": "ac8286fe3ceb567e23d3a04b32e69d5447d443000c707b3af334cdec94b4baf0",
    "recalibration_validation.py": "a65fe7813b9ffd9db20dd232350737a528716d03583255fa907706be6f9076ad",
    "tokenizer_measurement.py": "527f63cc21fb9d044b179f0e67228944db36a9b279ef475208278adcdf73481e",
    "training_worker.py": "8a305da4e8bbdd6c394ad5347b83c22447378cb034a51bdd05daf284cd1e3aee",
    "validation_contract.py": "a4d6de138fa4817b23fc6e28442785b0d68a6b8f7360fb110582752d3463299d",
    "validation_real_report.py": "55d7de49c81236e60712bf0f578a152f9807f758feaca99a09b15e1d4579909c",
    "validation_report.py": "2e6bfdd61b60a08afdac436e2a2d83cc0b6878914a8d1aeb3a2a9a1da2e28a1e",
}
PHASE1_REMOTE_SHA256 = "643c87d243d6212707f8857b35640ea10e0310fd7dd018d6929e33b12f6786aa"
SSH_STDERR_LIMIT = 131072
SSH_REAP_NS = 2_000_000_000
SSH_OVERHEAD_NS = 60_000_000_000
PHASE2_BATCH_TIMEOUT_NS = 110_000_000_000
FALSE_FLAGS = (
    *FLAGS,
    "trusted_collection_accepted",
    "run_authorized",
    "diagnostic_dispatch_authorized",
)


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _check_hex(text, length):
    require(
        type(text) is str and re.fullmatch("[0-9a-f]{%d}" % length, text) is not None,
        "invalid hex identifier",
    )


def _check_bounded_str(value, limit=4096):
    require(type(value) is str and 0 < len(value) <= limit, "bounded nonempty string")


class _FileErrors:
    def __init__(self):
        self.primary = None
        self.traceback = None
        self.phase = None
        self.secondary = []

    def capture(self, error, phase):
        self.secondary.extend(getattr(error, "_diag_secondary", []))
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
            self.primary._diag_secondary = self.secondary
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


def _verify_sources_and_inputs(raw_files, worker_raw, backend_raw, dep_sources):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_dataset(raw_files)
    require(type(worker_raw) is bytes and _sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(type(backend_raw) is bytes and _sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    fields(dep_sources, PHASE1_MODULE_PINS)
    for name, expected_sha in PHASE1_MODULE_PINS.items():
        content = dep_sources[name]
        require(
            type(content) is bytes and len(content) <= 1048576 and _sha256(content) == expected_sha,
            "phase1 source pin: " + name,
        )
    remote_path = os.path.realpath(Path(__file__).with_name("recalibration_diagnostic_remote.py"))
    remote_raw = _read_regular_nofollow(remote_path, 1048576)
    require(_sha256(remote_raw) == PHASE1_REMOTE_SHA256, "diagnostic remote bootstrap drift")
    self_path = os.path.realpath(__file__)
    self_raw = _read_regular_nofollow(self_path, 1048576)
    return remote_raw, _sha256(self_raw)


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


def _validate_preflight(preflight_raw, resolved_interpreter):
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
    require(preflight["remote_python"] == resolved_interpreter, "resolved_interpreter mismatch")
    require(
        resolved_interpreter.startswith(runtime["python_root"].rstrip("/") + "/"),
        "resolved_interpreter outside python_root",
    )
    return runtime


def _build_phase_two_ssh_argv(ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime):
    backend_b64 = base64.b64encode(backend_raw).decode("ascii")
    expected_rt_b64 = base64.b64encode(encoded(expected_runtime)).decode("ascii")
    child_snippet = (
        "import json, pathlib\n"
        "expected_rt = json.loads(pathlib.Path('expected_runtime.json').read_text('utf-8'))\n"
        "backend_bytes = pathlib.Path('backend.py').read_bytes()\n"
        "namespace = {'__name__': '__main__', 'EXPECTED_RUNTIME': expected_rt}\n"
        "exec(compile(backend_bytes, 'validation_remote.py', 'exec'), namespace)\n"
    )
    phase_two_program = (
        "import base64,hashlib,json,os,shutil,signal,socket,subprocess,sys,tempfile\n"
        f"backend_bytes=base64.b64decode({backend_b64!r},validate=True)\n"
        "expected_rt_raw=base64.b64decode(" + repr(expected_rt_b64) + ",validate=True)\n"
        "expected_rt=json.loads(expected_rt_raw.decode('ascii'))\n"
        f"if hashlib.sha256(backend_bytes).hexdigest()!={BACKEND_SHA256!r}:\n"
        "    sys.exit(78)\n"
        "if socket.gethostname().split('.')[0]!=expected_rt['hostname'].split('.')[0] or os.getuid()!=expected_rt['uid']:\n"
        "    sys.exit(79)\n"
        "bundle_in=sys.stdin.buffer.read(131073)\n"
        "os.umask(0o077)\n"
        "wdir=tempfile.mkdtemp(prefix='tlf-diag-p2-',dir='/tmp')\n"
        "exit_code=1\n"
        "try:\n"
        "    os.chmod(wdir,0o700)\n"
        "    for fn,fb in (('backend.py',backend_bytes),('expected_runtime.json',expected_rt_raw)):\n"
        "        fdesc=os.open(os.path.join(wdir,fn),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)\n"
        "        with os.fdopen(fdesc,'wb') as fstream:\n"
        "            os.fchmod(fstream.fileno(),0o600)\n"
        "            fstream.write(fb)\n"
        "            fstream.flush()\n"
        f"    proc=subprocess.Popen([{resolved_interpreter!r},'-I','-S','-B','-c',{child_snippet!r}],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=wdir,env={{'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','HOME':'/nonexistent'}},close_fds=True,pass_fds=(),start_new_session=True,shell=False)\n"
        "    try:\n"
        "        out_bytes,err_bytes=proc.communicate(input=bundle_in,timeout=95.0)\n"
        "    finally:\n"
        "        if proc.poll() is None:\n"
        "            try:\n"
        "                os.killpg(proc.pid,signal.SIGKILL)\n"
        "            except ProcessLookupError:\n"
        "                pass\n"
        "        try:\n"
        "            proc.wait(timeout=2.0)\n"
        "        except subprocess.TimeoutExpired:\n"
        "            pass\n"
        "        for pobj in (proc.stdin,proc.stdout,proc.stderr):\n"
        "            if pobj is not None:\n"
        "                try:\n"
        "                    pobj.close()\n"
        "                except OSError:\n"
        "                    pass\n"
        "    if out_bytes:\n"
        "        sys.stdout.buffer.write(out_bytes)\n"
        "        sys.stdout.buffer.flush()\n"
        "    if err_bytes:\n"
        "        sys.stderr.buffer.write(err_bytes)\n"
        "        sys.stderr.buffer.flush()\n"
        "    exit_code=proc.returncode if type(proc.returncode) is int else 1\n"
        "finally:\n"
        "    os.chdir('/')\n"
        "    shutil.rmtree(wdir,ignore_errors=True)\n"
        "sys.exit(exit_code)\n"
    )
    phase_two_remote_cmd = "exec " + shlex.join((
        "env", "-i", "PATH=/usr/bin:/bin", "LANG=C", "LC_ALL=C", "HOME=/nonexistent",
        resolved_interpreter, "-I", "-S", "-B", "-c", phase_two_program,
    ))
    return [*ssh_argv_prefix, phase_two_remote_cmd]


def build_diagnostic_package(
    raw_files,
    worker_raw,
    backend_raw,
    dep_sources,
    preflight_raw,
    scope_raw,
    *,
    run_id,
    scoring_run_ids,
    ssh_argv_prefix,
    ssh_environment,
    interpreter,
    resolved_interpreter,
    site_packages,
    checkpoint_root,
    expected_tokenizer_sha256,
    remote_parent,
    local_run_root,
    work_timeout_seconds=5400,
):
    require(
        type(work_timeout_seconds) is int and 60 <= work_timeout_seconds <= 7200,
        "work_timeout_seconds out of bounds",
    )
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    require(
        type(scoring_run_ids) is list and len(scoring_run_ids) == 18,
        "18 pre-allocated scoring_run_ids required",
    )
    seen_ids = {run_id}
    for s_run_id in scoring_run_ids:
        _check_hex(s_run_id, 32)
        require(s_run_id not in seen_ids, "duplicate run_id")
        seen_ids.add(s_run_id)

    for abs_path in (
        interpreter,
        resolved_interpreter,
        site_packages,
        checkpoint_root,
        remote_parent,
        local_run_root,
    ):
        require(
            type(abs_path) is str
            and abs_path.startswith("/")
            and ".." not in Path(abs_path).parts
            and str(Path(abs_path)) == abs_path,
            "normalized absolute path required",
        )
    require(
        type(ssh_argv_prefix) is list
        and len(ssh_argv_prefix) >= 2
        and all(type(tok) is str and bool(tok) for tok in ssh_argv_prefix),
        "ssh_argv_prefix",
    )
    require(
        type(ssh_environment) is dict
        and all(
            type(env_k) is str and type(env_v) is str
            for env_k, env_v in ssh_environment.items()
        ),
        "ssh_environment",
    )

    remote_raw, controller_sha256 = _verify_sources_and_inputs(
        raw_files, worker_raw, backend_raw, dep_sources
    )
    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    preflight_sha256 = _sha256(preflight_raw)

    scope = parse_json(scope_raw, limit=65536)
    _plain(scope)
    fields(
        scope,
        (
            "kind",
            "version",
            "run_id",
            "local_run_root",
            "remote_parent",
            "ssh_argv_prefix_sha256",
            "ssh_environment_sha256",
            "preflight_receipt_sha256",
        ),
    )
    require(
        scope["kind"] == "LORA_RECALIBRATION_DIAGNOSTIC_STAGE_SCOPE"
        and type(scope["version"]) is int
        and scope["version"] == 1
        and scope["run_id"] == run_id
        and scope["local_run_root"] == local_run_root
        and scope["remote_parent"] == remote_parent
        and scope["ssh_argv_prefix_sha256"] == digest(ssh_argv_prefix)
        and scope["ssh_environment_sha256"] == digest(ssh_environment)
        and scope["preflight_receipt_sha256"] == preflight_sha256,
        "diagnostic scope binding mismatch",
    )

    schedule = build_generation_schedule(
        raw_files["prompts.json"], raw_files["supervision_v1.json"], run_id=run_id
    )
    phase_one_payload_raw = encoded({
        "sources": {
            name: base64.b64encode(dep_sources[name]).decode("ascii")
            for name in sorted(PHASE1_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision_v1.json"]).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    })
    phase_one_payload_sha256 = _sha256(phase_one_payload_raw)

    phase_one_config = {
        "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE1_CONFIG",
        "version": 1,
        "hostname": expected_runtime["hostname"].split(".")[0],
        "uid": expected_runtime["uid"],
        "interpreter": interpreter,
        "resolved_interpreter": resolved_interpreter,
        "site_packages": site_packages,
        "checkpoint_root": checkpoint_root,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "expected_runtime": copy.deepcopy(expected_runtime),
        "parent": remote_parent,
        "stage": str(Path(remote_parent) / ("diag-phase1-" + run_id)),
        "run_id": run_id,
        "sources_sha256": dict(PHASE1_MODULE_PINS),
        "prompts_sha256": FILE_PINS["prompts.json"],
        "supervision_sha256": FILE_PINS["supervision_v1.json"],
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "payload_sha256": phase_one_payload_sha256,
        "payload_bytes": len(phase_one_payload_raw),
        "bootstrap_sha256": PHASE1_REMOTE_SHA256,
        "controller_sha256": controller_sha256,
        "work_timeout_seconds": work_timeout_seconds,
    }
    validate_phase_one_bundle(phase_one_payload_raw, phase_one_config)
    phase_one_config_raw = encoded(phase_one_config)
    phase_one_config_sha256 = _sha256(phase_one_config_raw)

    remote_b64 = base64.b64encode(remote_raw).decode("ascii")
    cfg_b64 = base64.b64encode(phase_one_config_raw).decode("ascii")
    phase_one_program = (
        "import base64,hashlib,json,sys\n"
        f"src=base64.b64decode({remote_b64!r},validate=True)\n"
        f"cfg=json.loads(base64.b64decode({cfg_b64!r},validate=True).decode('ascii'))\n"
        "if hashlib.sha256(src).hexdigest()!=cfg['bootstrap_sha256']:\n"
        "    sys.exit(78)\n"
        "ns={'__name__':'recalibration_diagnostic_phase1'}\n"
        "exec(compile(src,'<recalibration_diagnostic_remote>','exec'),ns)\n"
        "sys.exit(ns['bootstrap'](cfg))\n"
    )
    phase_one_remote_cmd = "exec " + shlex.join((
        "env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONHOME",
        "-u", "PYTHONPATH", interpreter, "-I", "-S", "-B", "-c", phase_one_program,
    ))
    phase_one_ssh_argv = [*ssh_argv_prefix, phase_one_remote_cmd]
    phase_two_ssh_argv = _build_phase_two_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )

    manifest = {
        "kind": "LORA_RECALIBRATION_DIAGNOSTIC_LAUNCH_PACKAGE",
        "version": 1,
        "data_version": "recalibration_v2_r1",
        "run_id": run_id,
        "scoring_run_ids": list(scoring_run_ids),
        "file_sha256": dict(FILE_PINS),
        "schedule_sha256": digest(schedule),
        "preflight_receipt_sha256": preflight_sha256,
        "one_time_scope_sha256": _sha256(scope_raw),
        "phase_one_config": phase_one_config,
        "phase_one_config_sha256": phase_one_config_sha256,
        "phase_one_argv_sha256": digest(phase_one_ssh_argv),
        "phase_two_argv_sha256": digest(phase_two_ssh_argv),
        "local_run_root": local_run_root,
        "planned_episodes": 24,
        "planned_generations": 72,
        "max_scoring_batches": 18,
        "zero_private_oracle_in_phase_one": True,
        "launch_package_structure_valid": True,
        **{flag: False for flag in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "phase_one_ssh_argv": phase_one_ssh_argv,
            "phase_two_ssh_argv": phase_two_ssh_argv,
            "ssh_environment": dict(ssh_environment),
            "phase_one_payload_hex": phase_one_payload_raw.hex(),
        },
    }


def _validate_phase_one_ready(ready_raw, phase_one_config, config_sha256):
    ready = strict_json(ready_raw)
    return (
        type(ready) is dict
        and ready.get("config_sha256") == config_sha256
        and ready.get("run_id") == phase_one_config["run_id"]
        and ready.get("hostname") == phase_one_config["hostname"]
        and type(ready.get("uid")) is int
        and ready["uid"] == phase_one_config["uid"]
        and ready.get("stage") == phase_one_config["stage"]
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


def _run_phase_one_ssh(
    ssh_argv,
    ssh_environment,
    phase_one_config,
    config_sha256,
    payload_raw,
    cancellations,
    secondary_errors,
):
    work_limit_ns = phase_one_config["work_timeout_seconds"] * 1_000_000_000
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
        if selector is not None and process is not None and process.stdin is not None:
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
                    secondary_errors.append({"phase": "p1_close_stdin", "type": type(err).__name__})

    try:
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + ssh_limit_ns
        process = subprocess.Popen(
            ssh_argv,
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
                    (wire_buf, PHASE1_WIRE_LIMIT)
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
                    if key.data == "stdout" and not ready_seen and len(wire_buf) >= PHASE1_HEADER.size:
                        magic, ch_num, ready_len = PHASE1_HEADER.unpack_from(wire_buf, 0)
                        if magic != b"TLFD1" or ch_num != 1 or ready_len > 4096:
                            ready_seen = True
                            close_stdin()
                        elif len(wire_buf) >= PHASE1_HEADER.size + ready_len:
                            ready_seen = True
                            ready_raw = bytes(wire_buf[PHASE1_HEADER.size:PHASE1_HEADER.size + ready_len])
                            elapsed_so_far = max(0, time.monotonic_ns() - started_ns)
                            grant_ns = max(1, work_limit_ns - elapsed_so_far)
                            if (
                                _validate_phase_one_ready(ready_raw, phase_one_config, config_sha256)
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
            secondary_errors.append({"phase": "p1_ssh_collect", "type": type(run_err).__name__})
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
                except BaseException as close_err:
                    if not isinstance(close_err, Exception) and not cancellations:
                        cancellations.append((close_err, close_err.__traceback__))
                    else:
                        secondary_errors.append({"phase": "p1_close_stream", "type": type(close_err).__name__})

    reaped = bool(cleanup_info and cleanup_info["child_reaped"])
    returncode = cleanup_info["returncode"] if reaped else None
    group_absent = bool(cleanup_info and cleanup_info["group_absent"])
    cleanup_bounded = bool(cleanup_info and cleanup_info["cleanup_bounded"])
    stdin_complete = grant_dispatched and stdin_offset == len(stdin_queue)
    return {
        "ssh_status": ssh_status,
        "grant_dispatched": grant_dispatched,
        "stdin_complete": stdin_complete,
        "stdout_eof": stdout_eof,
        "stderr_eof": stderr_eof,
        "reaped": reaped,
        "returncode": returncode,
        "group_absent": group_absent,
        "cleanup_bounded": cleanup_bounded,
        "wire_bytes": bytes(wire_buf),
        "stderr_bytes": bytes(stderr_buf),
    }


def _run_phase_two_bundle_ssh(
    ssh_argv,
    ssh_environment,
    bundle_raw,
    cancellations,
    secondary_errors,
    *,
    timeout_ns=PHASE2_BATCH_TIMEOUT_NS,
):
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    stdin_offset = 0
    stdin_closed = False
    stdout_eof = False
    stderr_eof = False
    limit_status = None
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
        if selector is not None and process is not None and process.stdin is not None:
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
                    secondary_errors.append({"phase": "p2_close_stdin", "type": type(err).__name__})

    try:
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + timeout_ns
        process = subprocess.Popen(
            ssh_argv,
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
        selector = selectors.DefaultSelector()
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        if bundle_raw:
            selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        else:
            close_stdin()

        while True:
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                limit_status = "timeout"
                break
            if not selector.get_map() and process.poll() is not None:
                break
            rem_sec = (deadline_ns - now_ns) / 1_000_000_000.0
            if not selector.get_map():
                time.sleep(min(0.02, rem_sec))
                continue
            events = selector.select(min(0.05, rem_sec))
            if time.monotonic_ns() >= deadline_ns:
                limit_status = "timeout"
                break
            for key, _mask in events:
                if key.data == "stdin":
                    if stdin_closed:
                        continue
                    try:
                        written = _io_write(
                            key.fd, bundle_raw[stdin_offset:stdin_offset + 65536]
                        )
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        close_stdin()
                        continue
                    if written > 0:
                        stdin_offset += written
                    if stdin_offset >= len(bundle_raw):
                        close_stdin()
                    continue

                target_buf, cap = (
                    (stdout_buf, TRANSPORT_STDOUT_LIMIT)
                    if key.data == "stdout"
                    else (stderr_buf, TRANSPORT_STDERR_LIMIT)
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
                else:
                    if len(chunk) > rem_bytes:
                        target_buf.extend(chunk[:rem_bytes + 1])
                        limit_status = "output_limit"
                        break
                    target_buf.extend(chunk)
            if limit_status is not None:
                break
    except BaseException as run_err:
        if not isinstance(run_err, Exception) and not cancellations:
            cancellations.append((run_err, run_err.__traceback__))
        else:
            secondary_errors.append({"phase": "p2_ssh_collect", "type": type(run_err).__name__})
        limit_status = limit_status or "launch_failed"
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
                except BaseException as close_err:
                    if not isinstance(close_err, Exception) and not cancellations:
                        cancellations.append((close_err, close_err.__traceback__))
                    else:
                        secondary_errors.append({"phase": "p2_close_stream", "type": type(close_err).__name__})

    elapsed_ns = max(0, (time.monotonic_ns() - started_ns))
    reaped = bool(cleanup_info and cleanup_info["child_reaped"])
    returncode = cleanup_info["returncode"] if reaped else None
    group_absent = bool(cleanup_info and cleanup_info["group_absent"])
    cleanup_bounded = bool(cleanup_info and cleanup_info["cleanup_bounded"])
    stdin_complete = stdin_offset == len(bundle_raw)
    stdout_bytes = bytes(stdout_buf)
    stderr_bytes = bytes(stderr_buf)

    if limit_status is not None:
        status_label = limit_status
    elif reaped and returncode == 0 and stdin_complete and stdout_eof and stderr_eof:
        status_label = "completed"
    else:
        status_label = "program_error"

    return {
        "transport": {
            "status": status_label,
            "returncode": returncode,
            "stdout_hex": stdout_bytes.hex(),
            "stderr_hex": stderr_bytes.hex(),
            "elapsed_ns": elapsed_ns,
            "process_reaped": reaped,
            "stdin_complete": stdin_complete,
        },
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
        "reaped": reaped,
        "group_absent": group_absent,
        "cleanup_bounded": cleanup_bounded,
    }


def run_diagnostic_stage(
    package,
    raw_files,
    worker_raw,
    backend_raw,
    dep_sources,
    preflight_raw,
    scope_raw,
):
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    p1_cfg = pub["phase_one_config"]

    rebuilt = build_diagnostic_package(
        raw_files,
        worker_raw,
        backend_raw,
        dep_sources,
        preflight_raw,
        scope_raw,
        run_id=pub["run_id"],
        scoring_run_ids=pub["scoring_run_ids"],
        ssh_argv_prefix=priv["phase_one_ssh_argv"][:-1],
        ssh_environment=priv["ssh_environment"],
        interpreter=p1_cfg["interpreter"],
        resolved_interpreter=p1_cfg["resolved_interpreter"],
        site_packages=p1_cfg["site_packages"],
        checkpoint_root=p1_cfg["checkpoint_root"],
        expected_tokenizer_sha256=p1_cfg["expected_tokenizer_sha256"],
        remote_parent=p1_cfg["parent"],
        local_run_root=pub["local_run_root"],
        work_timeout_seconds=p1_cfg["work_timeout_seconds"],
    )
    require(encoded(package) == encoded(rebuilt), "diagnostic launch package mismatch")

    run_id = pub["run_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], run_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    phase_one_payload_raw = bytes.fromhex(priv["phase_one_payload_hex"])

    p1_out = None
    p1_result_obj = None
    phase_one_decode_error = None
    phase_two_all_clean = True
    final_bundle_records = []
    continuation_records = []
    diag_report = None

    try:
        try:
            _write_exclusive(
                run_fd,
                "diagnostic-stage-claim.json",
                _pretty_json({
                    "kind": "LORA_RECALIBRATION_DIAGNOSTIC_STAGE_CLAIM",
                    "run_id": run_id,
                    "package_sha256": pub["package_sha256"],
                    "phase_one_config_sha256": pub["phase_one_config_sha256"],
                    "one_time_scope_sha256": pub["one_time_scope_sha256"],
                    "consumed_before_ssh": True,
                }),
            )
            _write_exclusive(run_fd, "diagnostic-package-public.json", _pretty_json(pub))
        except BaseException as claim_err:
            persistence_errors.append(claim_err)
            if not isinstance(claim_err, Exception) and not cancellations:
                cancellations.append((claim_err, claim_err.__traceback__))

        if not persistence_errors and not cancellations:
            p1_out = _run_phase_one_ssh(
                priv["phase_one_ssh_argv"],
                priv["ssh_environment"],
                p1_cfg,
                pub["phase_one_config_sha256"],
                phase_one_payload_raw,
                cancellations,
                secondary_errors,
            )
            for fname, fbytes in (
                ("phase1-wire.bin", p1_out["wire_bytes"]),
                ("phase1-ssh-stderr.bin", p1_out["stderr_bytes"]),
            ):
                try:
                    _write_exclusive(run_fd, fname, fbytes)
                except BaseException as write_err:
                    persistence_errors.append(write_err)
                    if not isinstance(write_err, Exception) and not cancellations:
                        cancellations.append((write_err, write_err.__traceback__))

            if (
                p1_out["ssh_status"] is None
                and p1_out["reaped"]
                and p1_out["returncode"] in (0, 1)
                and p1_out["stdin_complete"]
                and p1_out["stdout_eof"]
                and p1_out["stderr_eof"]
                and p1_out["group_absent"]
                and p1_out["cleanup_bounded"]
                and not cancellations
                and not persistence_errors
            ):
                try:
                    frames = decode_frames(p1_out["wire_bytes"])
                    require(
                        _validate_phase_one_ready(
                            frames["READY"], p1_cfg, pub["phase_one_config_sha256"]
                        ),
                        "phase1 READY mismatch",
                    )
                    done_obj = strict_json(frames["DONE"])
                    require(
                        done_obj == {
                            "config_sha256": pub["phase_one_config_sha256"],
                            "result_sha256": _sha256(frames["RESULT"]),
                        },
                        "phase1 DONE mismatch",
                    )
                    res_obj = strict_json(frames["RESULT"])
                    require(
                        res_obj.get("kind") == "LORA_RECALIBRATION_DIAGNOSTIC_PHASE1_RESULT"
                        and res_obj.get("version") == 1,
                        "phase1 RESULT kind",
                    )
                    _write_exclusive(run_fd, "phase1-result.json", _pretty_json(res_obj))
                    _write_exclusive(run_fd, "phase1-native-stdout.bin", frames["STDOUT"])
                    _write_exclusive(run_fd, "phase1-native-stderr.bin", frames["STDERR"])
                    if res_obj.get("bootstrap_failure") is None and p1_out["returncode"] == 0:
                        p1_result_obj = res_obj
                    else:
                        bfail = res_obj.get("bootstrap_failure") or {}
                        phase_one_decode_error = (
                            "bootstrap_failure:"
                            + str(bfail.get("phase"))
                            + ":"
                            + str(bfail.get("error_type"))
                        )
                except BaseException as dec_err:
                    if not isinstance(dec_err, Exception) and not cancellations:
                        cancellations.append((dec_err, dec_err.__traceback__))
                    phase_one_decode_error = type(dec_err).__name__

        if (
            p1_result_obj is not None
            and not persistence_errors
            and not cancellations
        ):
            phase_one_records = p1_result_obj["phase_one_records"]
            planned_pos = plan_final_positions(raw_files, phase_one_records)
            used_scoring_ids = pub["scoring_run_ids"][:len(planned_pos["batches"])]
            bundles_plan = build_final_scoring_bundles(
                raw_files, worker_raw, run_id, phase_one_records, used_scoring_ids
            )
            for b_meta in bundles_plan["batches"]:
                if cancellations or persistence_errors:
                    break
                b_idx = b_meta["batch_index"]
                p2_res = _run_phase_two_bundle_ssh(
                    priv["phase_two_ssh_argv"],
                    priv["ssh_environment"],
                    b_meta["bundle_raw"],
                    cancellations,
                    secondary_errors,
                )
                if not (p2_res["reaped"] and p2_res["group_absent"] and p2_res["cleanup_bounded"]):
                    phase_two_all_clean = False
                for fname, fbytes in (
                    ("phase2-batch-%02d-bundle.json" % b_idx, b_meta["bundle_raw"]),
                    ("phase2-batch-%02d-stdout.jsonl" % b_idx, p2_res["stdout_bytes"]),
                    ("phase2-batch-%02d-stderr.bin" % b_idx, p2_res["stderr_bytes"]),
                ):
                    try:
                        _write_exclusive(run_fd, fname, fbytes)
                    except BaseException as p2w_err:
                        persistence_errors.append(p2w_err)
                        if not isinstance(p2w_err, Exception) and not cancellations:
                            cancellations.append((p2w_err, p2w_err.__traceback__))
                primary_rec = {
                    "batch_index": b_idx,
                    "bundle_raw": b_meta["bundle_raw"],
                    "transport": p2_res["transport"],
                }
                final_bundle_records.append(primary_rec)

                obs_map, full_ok, partial_ok, proto_err = _decode_scoring_batch(
                    b_meta, primary_rec, p1_cfg["expected_runtime"]
                )
                if not phase_two_all_clean or proto_err:
                    break
                if full_ok:
                    continue
                if not partial_ok:
                    break
                remaining_slice = b_meta["positions"][len(obs_map):]
                sub_idx = 1
                batch_settled = False
                while remaining_slice and not cancellations and not persistence_errors:
                    cont_meta = build_continuation_bundle(
                        worker_raw, run_id, b_idx, sub_idx, remaining_slice
                    )
                    c_res = _run_phase_two_bundle_ssh(
                        priv["phase_two_ssh_argv"],
                        priv["ssh_environment"],
                        cont_meta["bundle_raw"],
                        cancellations,
                        secondary_errors,
                    )
                    if not (c_res["reaped"] and c_res["group_absent"] and c_res["cleanup_bounded"]):
                        phase_two_all_clean = False
                    for fname, fbytes in (
                        ("phase2-batch-%02d-cont-%02d-bundle.json" % (b_idx, sub_idx), cont_meta["bundle_raw"]),
                        ("phase2-batch-%02d-cont-%02d-stdout.jsonl" % (b_idx, sub_idx), c_res["stdout_bytes"]),
                        ("phase2-batch-%02d-cont-%02d-stderr.bin" % (b_idx, sub_idx), c_res["stderr_bytes"]),
                    ):
                        try:
                            _write_exclusive(run_fd, fname, fbytes)
                        except BaseException as cw_err:
                            persistence_errors.append(cw_err)
                            if not isinstance(cw_err, Exception) and not cancellations:
                                cancellations.append((cw_err, cw_err.__traceback__))
                    cont_rec = {
                        "batch_index": b_idx,
                        "sub_index": sub_idx,
                        "run_id": cont_meta["run_id"],
                        "position_indices": [pos["position_index"] for pos in remaining_slice],
                        "bundle_raw": cont_meta["bundle_raw"],
                        "transport": c_res["transport"],
                    }
                    continuation_records.append(cont_rec)
                    c_obs, c_full_ok, c_partial_ok, c_proto_err = _decode_scoring_batch(
                        cont_meta,
                        cont_rec,
                        p1_cfg["expected_runtime"],
                        check_fields=(
                            "batch_index",
                            "sub_index",
                            "run_id",
                            "position_indices",
                            "bundle_raw",
                            "transport",
                        ),
                    )
                    if not phase_two_all_clean or c_proto_err:
                        break
                    if c_full_ok:
                        remaining_slice = []
                        batch_settled = True
                        break
                    if not c_partial_ok:
                        break
                    remaining_slice = remaining_slice[len(c_obs):]
                    sub_idx += 1
                if not batch_settled or not phase_two_all_clean:
                    break

            if not cancellations:
                diag_report = score_diagnostic_run(
                    raw_files,
                    worker_raw,
                    run_id,
                    phase_one_records,
                    used_scoring_ids,
                    final_bundle_records,
                    continuation_records=continuation_records,
                    expected_runtime=p1_cfg["expected_runtime"],
                    shared_costs=p1_result_obj.get("shared_costs"),
                )

        p1_clean = bool(
            p1_out is not None
            and p1_out["reaped"]
            and p1_out["group_absent"]
            and p1_out["cleanup_bounded"]
        )
        binding_complete = bool(
            not persistence_errors
            and not cancellations
            and not secondary_errors
            and p1_clean
            and phase_two_all_clean
        )
        collection_complete = bool(
            binding_complete
            and diag_report is not None
            and diag_report["calibration"]["protocol_contradictions"] == []
            and diag_report["calibration"]["incomplete_reasons"] == []
        )
        summary = {
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "phase_one_config_sha256": pub["phase_one_config_sha256"],
            "one_time_scope_sha256": pub["one_time_scope_sha256"],
            "phase_one_grant_dispatched": bool(p1_out and p1_out["grant_dispatched"]),
            "phase_one_returncode": p1_out["returncode"] if p1_out is not None else None,
            "phase_one_wire_sha256": _sha256(p1_out["wire_bytes"]) if p1_out is not None else None,
            "phase_one_wire_bytes": len(p1_out["wire_bytes"]) if p1_out is not None else 0,
            "phase_one_ssh_stderr_bytes": len(p1_out["stderr_bytes"]) if p1_out is not None else 0,
            "phase_one_decode_error": phase_one_decode_error,
            "phase_one_gpu_selected": p1_result_obj.get("gpu_selected") if p1_result_obj else None,
            "phase_one_integrity": p1_result_obj.get("integrity") if p1_result_obj else None,
            "phase_two_primary_batches": len(final_bundle_records),
            "phase_two_continuation_batches": len(continuation_records),
            "secondary_errors": secondary_errors,
            "binding_complete": binding_complete,
            "collection_complete": collection_complete,
            "calibration_status": (
                diag_report["calibration"]["status"] if diag_report is not None else None
            ),
            **{flag: False for flag in FALSE_FLAGS},
        }
        try:
            if diag_report is not None:
                _write_exclusive(run_fd, "diagnostic-report-full.json", _pretty_json(diag_report))
            _write_exclusive(run_fd, "diagnostic-controller-summary.json", _pretty_json(summary))
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
    return {**summary, "diagnostic_report": diag_report}


def _verify_sealed_phase_one(sealed_phase_one_wire, sealed_phase_one_config, expected_runtime):
    _plain(sealed_phase_one_config)
    require(
        type(sealed_phase_one_wire) is bytes and 0 < len(sealed_phase_one_wire) <= PHASE1_WIRE_LIMIT,
        "sealed phase1 wire size",
    )
    p1_cfg_sha = _sha256(encoded(sealed_phase_one_config))
    require(
        encoded(sealed_phase_one_config.get("expected_runtime")) == encoded(expected_runtime),
        "sealed phase1 expected_runtime mismatch",
    )
    frames = decode_frames(sealed_phase_one_wire)
    require(
        _validate_phase_one_ready(frames["READY"], sealed_phase_one_config, p1_cfg_sha),
        "sealed phase1 READY mismatch",
    )
    done_obj = strict_json(frames["DONE"])
    require(
        done_obj == {
            "config_sha256": p1_cfg_sha,
            "result_sha256": _sha256(frames["RESULT"]),
        },
        "sealed phase1 DONE mismatch",
    )
    res_obj = strict_json(frames["RESULT"])
    require(
        res_obj.get("kind") == "LORA_RECALIBRATION_DIAGNOSTIC_PHASE1_RESULT"
        and res_obj.get("version") == 1
        and res_obj.get("bootstrap_failure") is None
        and res_obj.get("identity", {}).get("run_id") == sealed_phase_one_config["run_id"]
        and type(res_obj.get("phase_one_records")) is list
        and len(res_obj["phase_one_records"]) == 24,
        "sealed phase1 result incomplete",
    )
    integ = res_obj.get("integrity") or {}
    require(
        integ.get("rng_unchanged") is True
        and type(integ.get("initial_base_sha256")) is str
        and integ.get("initial_base_sha256") == integ.get("final_base_sha256"),
        "sealed phase1 integrity",
    )
    return res_obj, p1_cfg_sha


def build_phase_two_scoring_package(
    raw_files,
    worker_raw,
    backend_raw,
    preflight_raw,
    scope_raw,
    *,
    sealed_phase_one_wire,
    sealed_phase_one_config,
    stage_id,
    scoring_run_ids,
    ssh_argv_prefix,
    ssh_environment,
    resolved_interpreter,
    local_run_root,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_dataset(raw_files)
    require(type(worker_raw) is bytes and _sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(type(backend_raw) is bytes and _sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    _check_hex(stage_id, 32)
    source_run_id = sealed_phase_one_config["run_id"]
    _check_hex(source_run_id, 32)
    require(stage_id != source_run_id, "stage_id must differ from source_run_id")
    require(
        type(scoring_run_ids) is list and len(scoring_run_ids) == 18,
        "18 scoring_run_ids required",
    )
    seen_ids = {stage_id, source_run_id}
    for s_run_id in scoring_run_ids:
        _check_hex(s_run_id, 32)
        require(s_run_id not in seen_ids, "duplicate scoring run_id")
        seen_ids.add(s_run_id)

    for abs_path in (resolved_interpreter, local_run_root):
        require(
            type(abs_path) is str
            and abs_path.startswith("/")
            and ".." not in Path(abs_path).parts
            and str(Path(abs_path)) == abs_path,
            "normalized absolute path required",
        )

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    preflight_sha256 = _sha256(preflight_raw)
    res_obj, p1_cfg_sha = _verify_sealed_phase_one(
        sealed_phase_one_wire, sealed_phase_one_config, expected_runtime
    )
    wire_sha256 = _sha256(sealed_phase_one_wire)
    result_sha256 = _sha256(encoded(res_obj))

    scope = parse_json(scope_raw, limit=65536)
    _plain(scope)
    fields(
        scope,
        (
            "kind",
            "version",
            "stage_id",
            "source_run_id",
            "phase_one_wire_sha256",
            "local_run_root",
            "ssh_argv_prefix_sha256",
            "ssh_environment_sha256",
            "preflight_receipt_sha256",
        ),
    )
    require(
        scope["kind"] == "LORA_RECALIBRATION_DIAGNOSTIC_PHASE2_SCOPE"
        and type(scope["version"]) is int
        and scope["version"] == 1
        and scope["stage_id"] == stage_id
        and scope["source_run_id"] == source_run_id
        and scope["phase_one_wire_sha256"] == wire_sha256
        and scope["local_run_root"] == local_run_root
        and scope["ssh_argv_prefix_sha256"] == digest(ssh_argv_prefix)
        and scope["ssh_environment_sha256"] == digest(ssh_environment)
        and scope["preflight_receipt_sha256"] == preflight_sha256,
        "phase2 scope binding mismatch",
    )

    planned_pos = plan_final_positions(raw_files, res_obj["phase_one_records"])
    used_scoring_ids = scoring_run_ids[:len(planned_pos["batches"])]
    bundles_plan = build_final_scoring_bundles(
        raw_files, worker_raw, source_run_id, res_obj["phase_one_records"], used_scoring_ids
    )
    phase_two_ssh_argv = _build_phase_two_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )
    self_raw = _read_regular_nofollow(os.path.realpath(__file__), 1048576)

    manifest = {
        "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE2_PACKAGE",
        "version": 1,
        "data_version": "recalibration_v2_r1",
        "stage_id": stage_id,
        "source_run_id": source_run_id,
        "scoring_run_ids": list(scoring_run_ids),
        "file_sha256": dict(FILE_PINS),
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "controller_sha256": _sha256(self_raw),
        "preflight_receipt_sha256": preflight_sha256,
        "one_time_scope_sha256": _sha256(scope_raw),
        "sealed_phase_one_wire_sha256": wire_sha256,
        "sealed_phase_one_config_sha256": p1_cfg_sha,
        "sealed_phase_one_result_sha256": result_sha256,
        "resolved_interpreter": resolved_interpreter,
        "expected_runtime": copy.deepcopy(expected_runtime),
        "phase_two_argv_sha256": digest(phase_two_ssh_argv),
        "local_run_root": local_run_root,
        "planned_batches": len(bundles_plan["batches"]),
        "planned_valid_positions": len(planned_pos["valid_positions"]),
        "launch_package_structure_valid": True,
        **{flag: False for flag in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "phase_two_ssh_argv": phase_two_ssh_argv,
            "ssh_environment": dict(ssh_environment),
        },
    }


def run_phase_two_scoring_stage(
    package,
    raw_files,
    worker_raw,
    backend_raw,
    preflight_raw,
    scope_raw,
    *,
    sealed_phase_one_wire,
    sealed_phase_one_config,
):
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]

    rebuilt = build_phase_two_scoring_package(
        raw_files,
        worker_raw,
        backend_raw,
        preflight_raw,
        scope_raw,
        sealed_phase_one_wire=sealed_phase_one_wire,
        sealed_phase_one_config=sealed_phase_one_config,
        stage_id=pub["stage_id"],
        scoring_run_ids=pub["scoring_run_ids"],
        ssh_argv_prefix=priv["phase_two_ssh_argv"][:-1],
        ssh_environment=priv["ssh_environment"],
        resolved_interpreter=pub["resolved_interpreter"],
        local_run_root=pub["local_run_root"],
    )
    require(encoded(package) == encoded(rebuilt), "phase2 scoring package mismatch")

    res_obj, _p1_cfg_sha = _verify_sealed_phase_one(
        sealed_phase_one_wire, sealed_phase_one_config, pub["expected_runtime"]
    )
    stage_id = pub["stage_id"]
    source_run_id = pub["source_run_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], stage_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    phase_two_all_clean = True
    final_bundle_records = []
    continuation_records = []
    diag_report = None

    try:
        try:
            _write_exclusive(
                run_fd,
                "diagnostic-phase2-stage-claim.json",
                _pretty_json({
                    "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE2_CLAIM",
                    "stage_id": stage_id,
                    "source_run_id": source_run_id,
                    "package_sha256": pub["package_sha256"],
                    "sealed_phase_one_wire_sha256": pub["sealed_phase_one_wire_sha256"],
                    "one_time_scope_sha256": pub["one_time_scope_sha256"],
                    "consumed_before_ssh": True,
                }),
            )
            _write_exclusive(run_fd, "diagnostic-phase2-package-public.json", _pretty_json(pub))
        except BaseException as claim_err:
            persistence_errors.append(claim_err)
            if not isinstance(claim_err, Exception) and not cancellations:
                cancellations.append((claim_err, claim_err.__traceback__))

        if not persistence_errors and not cancellations:
            phase_one_records = res_obj["phase_one_records"]
            planned_pos = plan_final_positions(raw_files, phase_one_records)
            used_scoring_ids = pub["scoring_run_ids"][:len(planned_pos["batches"])]
            bundles_plan = build_final_scoring_bundles(
                raw_files, worker_raw, source_run_id, phase_one_records, used_scoring_ids
            )
            for b_meta in bundles_plan["batches"]:
                if cancellations or persistence_errors:
                    break
                b_idx = b_meta["batch_index"]
                p2_res = _run_phase_two_bundle_ssh(
                    priv["phase_two_ssh_argv"],
                    priv["ssh_environment"],
                    b_meta["bundle_raw"],
                    cancellations,
                    secondary_errors,
                )
                if not (p2_res["reaped"] and p2_res["group_absent"] and p2_res["cleanup_bounded"]):
                    phase_two_all_clean = False
                for fname, fbytes in (
                    ("phase2-batch-%02d-bundle.json" % b_idx, b_meta["bundle_raw"]),
                    ("phase2-batch-%02d-stdout.jsonl" % b_idx, p2_res["stdout_bytes"]),
                    ("phase2-batch-%02d-stderr.bin" % b_idx, p2_res["stderr_bytes"]),
                ):
                    try:
                        _write_exclusive(run_fd, fname, fbytes)
                    except BaseException as p2w_err:
                        persistence_errors.append(p2w_err)
                        if not isinstance(p2w_err, Exception) and not cancellations:
                            cancellations.append((p2w_err, p2w_err.__traceback__))
                primary_rec = {
                    "batch_index": b_idx,
                    "bundle_raw": b_meta["bundle_raw"],
                    "transport": p2_res["transport"],
                }
                final_bundle_records.append(primary_rec)

                obs_map, full_ok, partial_ok, proto_err = _decode_scoring_batch(
                    b_meta, primary_rec, pub["expected_runtime"]
                )
                if not phase_two_all_clean or proto_err:
                    break
                if full_ok:
                    continue
                if not partial_ok:
                    break
                remaining_slice = b_meta["positions"][len(obs_map):]
                sub_idx = 1
                batch_settled = False
                while remaining_slice and not cancellations and not persistence_errors:
                    cont_meta = build_continuation_bundle(
                        worker_raw, source_run_id, b_idx, sub_idx, remaining_slice
                    )
                    c_res = _run_phase_two_bundle_ssh(
                        priv["phase_two_ssh_argv"],
                        priv["ssh_environment"],
                        cont_meta["bundle_raw"],
                        cancellations,
                        secondary_errors,
                    )
                    if not (c_res["reaped"] and c_res["group_absent"] and c_res["cleanup_bounded"]):
                        phase_two_all_clean = False
                    for fname, fbytes in (
                        ("phase2-batch-%02d-cont-%02d-bundle.json" % (b_idx, sub_idx), cont_meta["bundle_raw"]),
                        ("phase2-batch-%02d-cont-%02d-stdout.jsonl" % (b_idx, sub_idx), c_res["stdout_bytes"]),
                        ("phase2-batch-%02d-cont-%02d-stderr.bin" % (b_idx, sub_idx), c_res["stderr_bytes"]),
                    ):
                        try:
                            _write_exclusive(run_fd, fname, fbytes)
                        except BaseException as cw_err:
                            persistence_errors.append(cw_err)
                            if not isinstance(cw_err, Exception) and not cancellations:
                                cancellations.append((cw_err, cw_err.__traceback__))
                    cont_rec = {
                        "batch_index": b_idx,
                        "sub_index": sub_idx,
                        "run_id": cont_meta["run_id"],
                        "position_indices": [pos["position_index"] for pos in remaining_slice],
                        "bundle_raw": cont_meta["bundle_raw"],
                        "transport": c_res["transport"],
                    }
                    continuation_records.append(cont_rec)
                    c_obs, c_full_ok, c_partial_ok, c_proto_err = _decode_scoring_batch(
                        cont_meta,
                        cont_rec,
                        pub["expected_runtime"],
                        check_fields=(
                            "batch_index",
                            "sub_index",
                            "run_id",
                            "position_indices",
                            "bundle_raw",
                            "transport",
                        ),
                    )
                    if not phase_two_all_clean or c_proto_err:
                        break
                    if c_full_ok:
                        remaining_slice = []
                        batch_settled = True
                        break
                    if not c_partial_ok:
                        break
                    remaining_slice = remaining_slice[len(c_obs):]
                    sub_idx += 1
                if not batch_settled or not phase_two_all_clean:
                    break

            if not cancellations:
                diag_report = score_diagnostic_run(
                    raw_files,
                    worker_raw,
                    source_run_id,
                    phase_one_records,
                    used_scoring_ids,
                    final_bundle_records,
                    continuation_records=continuation_records,
                    expected_runtime=pub["expected_runtime"],
                    shared_costs=res_obj.get("shared_costs"),
                )

        binding_complete = bool(
            not persistence_errors
            and not cancellations
            and not secondary_errors
            and phase_two_all_clean
        )
        collection_complete = bool(
            binding_complete
            and diag_report is not None
            and diag_report["calibration"]["protocol_contradictions"] == []
            and diag_report["calibration"]["incomplete_reasons"] == []
        )
        summary = {
            "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE2_EXECUTION_EVIDENCE",
            "version": 1,
            "stage_id": stage_id,
            "source_run_id": source_run_id,
            "package_sha256": pub["package_sha256"],
            "sealed_phase_one_wire_sha256": pub["sealed_phase_one_wire_sha256"],
            "sealed_phase_one_config_sha256": pub["sealed_phase_one_config_sha256"],
            "one_time_scope_sha256": pub["one_time_scope_sha256"],
            "phase_one_gpu_selected": res_obj.get("gpu_selected"),
            "phase_one_integrity": res_obj.get("integrity"),
            "phase_two_primary_batches": len(final_bundle_records),
            "phase_two_continuation_batches": len(continuation_records),
            "secondary_errors": secondary_errors,
            "binding_complete": binding_complete,
            "collection_complete": collection_complete,
            "calibration_status": (
                diag_report["calibration"]["status"] if diag_report is not None else None
            ),
            **{flag: False for flag in FALSE_FLAGS},
        }
        try:
            if diag_report is not None:
                _write_exclusive(run_fd, "diagnostic-report-full.json", _pretty_json(diag_report))
            _write_exclusive(run_fd, "diagnostic-phase2-controller-summary.json", _pretty_json(summary))
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
    return {**summary, "diagnostic_report": diag_report}
