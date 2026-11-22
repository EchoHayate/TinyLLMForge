import base64
import copy
import hashlib
import json
import os
from os import read as _io_read
from os import write as _io_write
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import time

from confirmatory_v2_contract import CONFIRMATORY_FILE_PINS, load_confirmatory_dataset
from confirmatory_v2_gate_a import (
    MULTI_BUNDLE_STDOUT_LIMIT,
    build_multi_bundle_ssh_argv,
    run_single_ssh_json_call,
)
from confirmatory_v2_remote import (
    HEADER as CONF_HEADER,
    PHASE1_LORA_SOURCE_NAMES,
    WIRE_LIMIT as CONF_WIRE_LIMIT,
    decode_frames,
    strict_json,
    validate_phase_one_lora_bundle,
)
from confirmatory_v2_scoring import (
    build_confirmatory_continuation_bundle,
    build_confirmatory_schedule,
    build_confirmatory_scoring_bundles,
    plan_confirmatory_positions,
    score_confirmatory_run,
)
from pilot_protocol import fields, require
from recalibration_data_contract import _plain
from recalibration_diagnostic_controller import (
    FALSE_FLAGS,
    SSH_OVERHEAD_NS,
    SSH_STDERR_LIMIT,
    _build_phase_two_ssh_argv,
    _check_hex,
    _claim_local_run_root,
    _cleanup_ssh,
    _pretty_json,
    _read_regular_nofollow,
    _run_phase_two_bundle_ssh,
    _safe_raw_ns,
    _validate_preflight,
    _write_exclusive,
)
from recalibration_diagnostic_scoring import _decode_scoring_batch
from recalibration_execution_package import BACKEND_SHA256
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json


PHASE1_LORA_PINS = {
    "adapter_artifact.py": "8c33fd3a026eadf7ed0680c8d7bdd94981e0026cea052ed8ef043bc83e943ffb",
    "confirmatory_v2_contract.py": "84613ec432dc0d22ac37e207107893810a0817d9630849f6aa20f212ea6cbdb9",
    "confirmatory_v2_scoring.py": "37b4bc6367c10f4eee9365e5730e7550e53070a5858d69b3404f9a8d051fdca2",
    "confirmatory_v2_training.py": "2164c996b5810c96e82aeb85987e2f18759d7c4c2210e2e3470c38ae0a155fd8",
    "generation_core.py": "dfccfd21d5a526fb9a7853c1b245a333c1c805ec17e7748902179f80e302372d",
    "observed_generation.py": "8fd18d126cce0b97cce5c90c2edeb4f2859959d3a4ad76864c2711b8571c0efd",
    "pilot_protocol.py": "5074980a2b5b9d8ec0a52288d7cb62cbc847688f094cd8ba189a7135ba315c97",
    "prompt_rendering.py": "2016c3eb0255d1d4a40f6072c7407b099fa3e4e569345c8d1ac778d5bd7bb18f",
    "recalibration_data_contract.py": "52695a9e2fb7e90b04f4fc7c95dcdc8b0310bbb0c269e6779890b9dff989ed02",
    "recalibration_diagnostic_remote.py": "643c87d243d6212707f8857b35640ea10e0310fd7dd018d6929e33b12f6786aa",
    "recalibration_diagnostic_scoring.py": "c45bb0b067ac3d198aa1f84610f63b5d10d1788fd353f49d88b6dfe19f39615b",
    "recalibration_rendering.py": "ac8286fe3ceb567e23d3a04b32e69d5447d443000c707b3af334cdec94b4baf0",
    "recalibration_validation.py": "a65fe7813b9ffd9db20dd232350737a528716d03583255fa907706be6f9076ad",
    "tokenizer_measurement.py": "527f63cc21fb9d044b179f0e67228944db36a9b279ef475208278adcdf73481e",
    "training_worker.py": "8a305da4e8bbdd6c394ad5347b83c22447378cb034a51bdd05daf284cd1e3aee",
    "validation_contract.py": "a4d6de138fa4817b23fc6e28442785b0d68a6b8f7360fb110582752d3463299d",
    "validation_real_report.py": "55d7de49c81236e60712bf0f578a152f9807f758feaca99a09b15e1d4579909c",
    "validation_report.py": "2e6bfdd61b60a08afdac436e2a2d83cc0b6878914a8d1aeb3a2a9a1da2e28a1e",
}
CONFIRMATORY_REMOTE_SHA256 = "f571c73275f30e29c564f3f70da4a169a1734c2f223ea47b9d82575cec6ce057"


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _verify_inputs(raw_files, worker_raw, backend_raw, dep_sources):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_confirmatory_dataset(raw_files)
    require(type(worker_raw) is bytes and _sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(type(backend_raw) is bytes and _sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    fields(dep_sources, PHASE1_LORA_PINS)
    for fname, exp_sha in PHASE1_LORA_PINS.items():
        raw_mod = dep_sources[fname]
        require(type(raw_mod) is bytes and _sha256(raw_mod) == exp_sha, "source pin: " + fname)
    remote_path = os.path.realpath(Path(__file__).with_name("confirmatory_v2_remote.py"))
    remote_raw = _read_regular_nofollow(remote_path, 1048576)
    require(_sha256(remote_raw) == CONFIRMATORY_REMOTE_SHA256, "confirmatory remote drift")
    self_raw = _read_regular_nofollow(os.path.realpath(__file__), 1048576)
    return remote_raw, _sha256(self_raw)


def build_confirmatory_lora_package(
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
    work_timeout_seconds=25200,
):
    require(
        type(work_timeout_seconds) is int and 60 <= work_timeout_seconds <= 28800,
        "work_timeout_seconds out of bounds",
    )
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    require(
        type(scoring_run_ids) is list and len(scoring_run_ids) == 48,
        "48 scoring_run_ids required",
    )
    seen_ids = {run_id}
    for s_rid in scoring_run_ids:
        _check_hex(s_rid, 32)
        require(s_rid not in seen_ids, "duplicate scoring run_id")
        seen_ids.add(s_rid)

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

    remote_raw, controller_sha256 = _verify_inputs(
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
        scope["kind"] == "LORA_CONFIRMATORY_V2_STAGE_SCOPE"
        and scope["version"] == 1
        and scope["run_id"] == run_id
        and scope["local_run_root"] == local_run_root
        and scope["remote_parent"] == remote_parent
        and scope["ssh_argv_prefix_sha256"] == digest(ssh_argv_prefix)
        and scope["ssh_environment_sha256"] == digest(ssh_environment)
        and scope["preflight_receipt_sha256"] == preflight_sha256,
        "confirmatory scope mismatch",
    )

    schedule = build_confirmatory_schedule(
        raw_files["prompts.json"], raw_files["supervision.json"], run_id=run_id
    )
    phase_one_payload_raw = encoded({
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(PHASE1_LORA_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision.json"]).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    })
    phase_one_payload_sha256 = _sha256(phase_one_payload_raw)

    phase_one_config = {
        "kind": "LORA_CONFIRMATORY_V2_PHASE1_CONFIG",
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
        "stage": str(Path(remote_parent) / ("conf-phase1-" + run_id)),
        "run_id": run_id,
        "sources_sha256": dict(PHASE1_LORA_PINS),
        "prompts_sha256": CONFIRMATORY_FILE_PINS["prompts.json"],
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "payload_sha256": phase_one_payload_sha256,
        "payload_bytes": len(phase_one_payload_raw),
        "bootstrap_sha256": CONFIRMATORY_REMOTE_SHA256,
        "controller_sha256": controller_sha256,
        "work_timeout_seconds": work_timeout_seconds,
    }
    validate_phase_one_lora_bundle(phase_one_payload_raw, phase_one_config)
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
        "ns={'__name__':'confirmatory_v2_phase1'}\n"
        "exec(compile(src,'<confirmatory_v2_remote>','exec'),ns)\n"
        "sys.exit(ns['bootstrap'](cfg))\n"
    )
    phase_one_remote_cmd = "exec " + shlex.join((
        "env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONHOME",
        "-u", "PYTHONPATH", interpreter, "-I", "-S", "-B", "-c", phase_one_program,
    ))
    phase_one_ssh_argv = [*ssh_argv_prefix, phase_one_remote_cmd]
    phase_two_ssh_argv = build_multi_bundle_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )

    manifest = {
        "kind": "LORA_CONFIRMATORY_V2_LAUNCH_PACKAGE",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
        "scoring_run_ids": list(scoring_run_ids),
        "file_sha256": dict(CONFIRMATORY_FILE_PINS),
        "schedule_sha256": digest(schedule),
        "preflight_receipt_sha256": preflight_sha256,
        "one_time_scope_sha256": _sha256(scope_raw),
        "phase_one_config": phase_one_config,
        "phase_one_config_sha256": phase_one_config_sha256,
        "phase_one_argv_sha256": digest(phase_one_ssh_argv),
        "phase_two_argv_sha256": digest(phase_two_ssh_argv),
        "local_run_root": local_run_root,
        "planned_episodes": 96,
        "planned_generations": 192,
        "max_scoring_batches": 48,
        "zero_private_oracle_in_phase_one": True,
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


def _validate_ready(ready_raw, phase_one_config, config_sha256):
    ready = strict_json(ready_raw)
    return (
        type(ready) is dict
        and ready.get("config_sha256") == config_sha256
        and ready.get("run_id") == phase_one_config["run_id"]
        and ready.get("hostname") == phase_one_config["hostname"]
        and ready.get("uid") == phase_one_config["uid"]
        and ready.get("stage") == phase_one_config["stage"]
    )


def _run_conf_phase_one_ssh(
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
                    (wire_buf, CONF_WIRE_LIMIT)
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
                    if key.data == "stdout" and not ready_seen and len(wire_buf) >= CONF_HEADER.size:
                        magic, ch_num, ready_len = CONF_HEADER.unpack_from(wire_buf, 0)
                        if magic != b"TLFC1" or ch_num != 1 or ready_len > 4096:
                            ready_seen = True
                            close_stdin()
                        elif len(wire_buf) >= CONF_HEADER.size + ready_len:
                            ready_seen = True
                            ready_raw = bytes(wire_buf[CONF_HEADER.size:CONF_HEADER.size + ready_len])
                            elapsed_so_far = max(0, time.monotonic_ns() - started_ns)
                            grant_ns = max(1, work_limit_ns - elapsed_so_far)
                            if (
                                _validate_ready(ready_raw, phase_one_config, config_sha256)
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


def _execute_conf_phase_two_batches(
    run_fd,
    raw_files,
    worker_raw,
    run_id,
    phase_one_records,
    scoring_run_ids,
    phase_two_ssh_argv,
    ssh_environment,
    expected_runtime,
    cancellations,
    persistence_errors,
    secondary_errors,
):
    planned_pos = plan_confirmatory_positions(raw_files, phase_one_records)
    used_ids = scoring_run_ids[:len(planned_pos["batches"])]
    bundles_plan = build_confirmatory_scoring_bundles(
        raw_files, worker_raw, run_id, phase_one_records, used_ids
    )
    final_bundle_records = []
    continuation_records = []
    phase_two_all_clean = True

    def safe_write(fname, fbytes):
        try:
            _write_exclusive(run_fd, fname, fbytes)
        except BaseException as w_err:
            persistence_errors.append(w_err)
            if not isinstance(w_err, Exception) and not cancellations:
                cancellations.append((w_err, w_err.__traceback__))

    if not bundles_plan["batches"]:
        return used_ids, final_bundle_records, continuation_records, phase_two_all_clean

    time.sleep(3.0)
    multi_stdin = encoded({
        "kind": "LORA_MULTI_VALIDATION_BUNDLES",
        "bundles_b64": [
            base64.b64encode(b_meta["bundle_raw"]).decode("ascii")
            for b_meta in bundles_plan["batches"]
        ],
    })
    p2_call = run_single_ssh_json_call(
        phase_two_ssh_argv,
        ssh_environment,
        multi_stdin,
        cancellations,
        secondary_errors,
        timeout_ns=2_400_000_000_000,
    )
    if not (p2_call["reaped"] and p2_call["group_absent"] and p2_call["cleanup_bounded"]):
        phase_two_all_clean = False
    safe_write("phase2-multi-stdout.json", p2_call["stdout_bytes"])
    safe_write("phase2-multi-stderr.bin", p2_call["stderr_bytes"])
    if p2_call["status"] != "completed":
        return used_ids, final_bundle_records, continuation_records, False

    envelope = parse_json(p2_call["stdout_bytes"].strip(), limit=MULTI_BUNDLE_STDOUT_LIMIT)
    require(envelope.get("kind") == "LORA_MULTI_VALIDATION_RESULTS", "phase2 multi result kind")
    returned_list = envelope["results"]

    pending_continuations = []
    for b_meta, item_rec in zip(bundles_plan["batches"], returned_list):
        b_idx = b_meta["batch_index"]
        require(item_rec["batch_index"] == b_idx, "multi batch_index mismatch")
        trans = item_rec["transport"]
        safe_write("phase2-batch-%02d-bundle.json" % b_idx, b_meta["bundle_raw"])
        safe_write("phase2-batch-%02d-stdout.jsonl" % b_idx, bytes.fromhex(trans["stdout_hex"]))
        safe_write("phase2-batch-%02d-stderr.bin" % b_idx, bytes.fromhex(trans["stderr_hex"]))
        primary_rec = {
            "batch_index": b_idx,
            "bundle_raw": b_meta["bundle_raw"],
            "transport": trans,
        }
        final_bundle_records.append(primary_rec)
        obs_map, full_ok, partial_ok, proto_err = _decode_scoring_batch(
            b_meta, primary_rec, expected_runtime
        )
        if proto_err:
            return used_ids, final_bundle_records, continuation_records, phase_two_all_clean
        if not full_ok and partial_ok:
            pending_continuations.append((b_idx, 1, b_meta["positions"][len(obs_map):]))
        elif not full_ok:
            return used_ids, final_bundle_records, continuation_records, phase_two_all_clean

    while pending_continuations and not cancellations and not persistence_errors:
        time.sleep(3.0)
        cont_metas = [
            build_confirmatory_continuation_bundle(worker_raw, run_id, b_idx, sub_idx, rem_slice)
            for b_idx, sub_idx, rem_slice in pending_continuations
        ]
        cont_stdin = encoded({
            "kind": "LORA_MULTI_VALIDATION_BUNDLES",
            "bundles_b64": [
                base64.b64encode(c_meta["bundle_raw"]).decode("ascii")
                for c_meta in cont_metas
            ],
        })
        c_call = run_single_ssh_json_call(
            phase_two_ssh_argv,
            ssh_environment,
            cont_stdin,
            cancellations,
            secondary_errors,
            timeout_ns=1_200_000_000_000,
        )
        if not (c_call["reaped"] and c_call["group_absent"] and c_call["cleanup_bounded"]):
            phase_two_all_clean = False
        if c_call["status"] != "completed":
            return used_ids, final_bundle_records, continuation_records, False
        c_env = parse_json(c_call["stdout_bytes"].strip(), limit=MULTI_BUNDLE_STDOUT_LIMIT)
        c_results = c_env["results"]
        next_pending = []
        for (b_idx, sub_idx, rem_slice), c_meta, c_item in zip(
            pending_continuations, cont_metas, c_results
        ):
            c_trans = c_item["transport"]
            safe_write("phase2-batch-%02d-cont-%02d-bundle.json" % (b_idx, sub_idx), c_meta["bundle_raw"])
            safe_write("phase2-batch-%02d-cont-%02d-stdout.jsonl" % (b_idx, sub_idx), bytes.fromhex(c_trans["stdout_hex"]))
            safe_write("phase2-batch-%02d-cont-%02d-stderr.bin" % (b_idx, sub_idx), bytes.fromhex(c_trans["stderr_hex"]))
            cont_rec = {
                "batch_index": b_idx,
                "sub_index": sub_idx,
                "run_id": c_meta["run_id"],
                "position_indices": [pos["position_index"] for pos in rem_slice],
                "bundle_raw": c_meta["bundle_raw"],
                "transport": c_trans,
            }
            continuation_records.append(cont_rec)
            c_obs, c_full, c_part, c_proto = _decode_scoring_batch(
                c_meta,
                cont_rec,
                expected_runtime,
                check_fields=(
                    "batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"
                ),
            )
            if c_proto:
                return used_ids, final_bundle_records, continuation_records, phase_two_all_clean
            if not c_full and c_part:
                next_pending.append((b_idx, sub_idx + 1, rem_slice[len(c_obs):]))
            elif not c_full:
                return used_ids, final_bundle_records, continuation_records, phase_two_all_clean
        pending_continuations = next_pending

    return used_ids, final_bundle_records, continuation_records, phase_two_all_clean


def run_confirmatory_lora_stage(
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

    rebuilt = build_confirmatory_lora_package(
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
    require(encoded(package) == encoded(rebuilt), "confirmatory package mismatch")

    run_id = pub["run_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], run_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    payload_raw = bytes.fromhex(priv["phase_one_payload_hex"])

    p1_out = None
    p1_result_obj = None
    phase_one_decode_error = None
    phase_two_all_clean = True
    final_bundle_records = []
    continuation_records = []
    conf_report = None

    def safe_write(fname, fbytes):
        try:
            _write_exclusive(run_fd, fname, fbytes)
        except BaseException as w_err:
            persistence_errors.append(w_err)
            if not isinstance(w_err, Exception) and not cancellations:
                cancellations.append((w_err, w_err.__traceback__))

    try:
        safe_write(
            "confirmatory-stage-claim.json",
            _pretty_json({
                "kind": "LORA_CONFIRMATORY_V2_STAGE_CLAIM",
                "run_id": run_id,
                "package_sha256": pub["package_sha256"],
                "phase_one_config_sha256": pub["phase_one_config_sha256"],
                "one_time_scope_sha256": pub["one_time_scope_sha256"],
                "consumed_before_ssh": True,
            }),
        )
        safe_write("confirmatory-package-public.json", _pretty_json(pub))

        if not persistence_errors and not cancellations:
            p1_out = _run_conf_phase_one_ssh(
                priv["phase_one_ssh_argv"],
                priv["ssh_environment"],
                p1_cfg,
                pub["phase_one_config_sha256"],
                payload_raw,
                cancellations,
                secondary_errors,
            )
            safe_write("phase1-wire.bin", p1_out["wire_bytes"])
            safe_write("phase1-ssh-stderr.bin", p1_out["stderr_bytes"])

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
                        _validate_ready(frames["READY"], p1_cfg, pub["phase_one_config_sha256"]),
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
                        res_obj.get("kind") == "LORA_CONFIRMATORY_V2_PHASE1_RESULT"
                        and res_obj.get("version") == 1,
                        "phase1 RESULT kind",
                    )
                    safe_write("phase1-result.json", _pretty_json(res_obj))
                    safe_write("phase1-native-stdout.bin", frames["STDOUT"])
                    safe_write("phase1-native-stderr.bin", frames["STDERR"])
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

        if p1_result_obj is not None and not persistence_errors and not cancellations:
            used_ids, final_bundle_records, continuation_records, phase_two_all_clean = (
                _execute_conf_phase_two_batches(
                    run_fd,
                    raw_files,
                    worker_raw,
                    run_id,
                    p1_result_obj["phase_one_records"],
                    pub["scoring_run_ids"],
                    priv["phase_two_ssh_argv"],
                    priv["ssh_environment"],
                    p1_cfg["expected_runtime"],
                    cancellations,
                    persistence_errors,
                    secondary_errors,
                )
            )
            if not cancellations:
                conf_report = score_confirmatory_run(
                    raw_files,
                    worker_raw,
                    run_id,
                    p1_result_obj["phase_one_records"],
                    used_ids,
                    final_bundle_records,
                    continuation_records=continuation_records,
                    expected_runtime=p1_cfg["expected_runtime"],
                    training_summary=p1_result_obj.get("training_summary"),
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
            and conf_report is not None
            and conf_report["verdict"]["protocol_contradictions"] == []
            and conf_report["verdict"]["incomplete_reasons"] == []
        )
        summary = {
            "kind": "LORA_CONFIRMATORY_V2_EXECUTION_EVIDENCE",
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
            "verdict_status": (
                conf_report["verdict"]["status"] if conf_report is not None else None
            ),
            **{flag: False for flag in FALSE_FLAGS},
        }
        if conf_report is not None:
            safe_write("confirmatory-report-full.json", _pretty_json(conf_report))
        safe_write("confirmatory-controller-summary.json", _pretty_json(summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {**summary, "confirmatory_report": conf_report}
