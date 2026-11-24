import base64
from contextlib import ExitStack
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import sys
import time

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    EVAL_IDS,
    RETENTION_IDS,
    SUPERVISION_IDS,
    TRANSFER_IDS,
    confirmatory_first_round,
    confirmatory_second_round,
    load_confirmatory_dataset,
)
from confirmatory_v2_converged_ablation import (
    ABLATION_SOURCE_NAMES,
    ABLATION_TRAINING_HYPERPARAMS,
    mounted_converged_adapter,
    train_converged_ablation_adapter,
)
from confirmatory_v2_gate_a import (
    MULTI_BUNDLE_STDOUT_LIMIT,
    build_multi_bundle_ssh_argv,
    run_single_ssh_json_call,
)
from confirmatory_v2_remote import (
    _cost_projection,
    decode_frames,
    strict_json,
    write_private,
)
from confirmatory_v2_scoring import (
    CONFIRMATORY_ROLES,
    _paired_arm_comparison,
    build_confirmatory_continuation_bundle,
)
from gated_lora_router import (
    ROUTER_ALLOWED_PUBLIC_KEYS,
    evaluate_gate_g6a_probe,
    extract_router_embedding,
    fit_whitened_prototype_router,
    score_router_embedding,
    validate_router_bundle,
)
from pilot_protocol import fields, require
from recalibration_data_contract import _plain
from recalibration_diagnostic_controller import (
    FALSE_FLAGS,
    _check_hex,
    _claim_local_run_root,
    _pretty_json,
    _validate_preflight,
    _write_exclusive,
)
from recalibration_diagnostic_scoring import (
    CANDIDATE_STATES,
    _classify_candidate,
    _decode_scoring_batch,
    candidate_valid,
    validate_public_feedback,
)
from recalibration_execution_package import BACKEND_SHA256
from recalibration_validation import WORKER_SHA256
from validation_contract import digest, encoded, parse_json


V1_PROMPTS_SHA256 = "3668d996bb0e05d600859fd12780b9f582a65aa2ad6b1881fd0bcce9832b5a05"
V1_STRUCTURE_SHA256 = "3905663be1d4cd4eb022f5d0995dcd651fafdc6514ecff0288957cec9cb0762e"
RECAL_V2_R1_PROMPTS_SHA256 = "0b8b037017edda84c28afdf0b7a6dd547d6bfd50582a60287c733f58f6ac1e38"

V1_BACKGROUND_IDS = tuple("R%02d" % idx_num for idx_num in range(1, 17))
V1_DEV_NEG_IDS = tuple("D%02d" % idx_num for idx_num in range(1, 5))
RECAL_DEV_POS_IDS = tuple("C%02d" % idx_num for idx_num in range(1, 13))

G6_SOURCE_NAMES = frozenset(
    set(ABLATION_SOURCE_NAMES)
    | {
        "gated_lora_router.py",
        "confirmatory_v2_gated_ablation.py",
    }
)

G6_STDLIB_BOOTSTRAP = """import base64
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import struct
import sys
import time

CHANNELS = ("READY", "RESULT", "STDOUT", "STDERR", "DONE")
LIMITS = (4096, 67108864, 1048576, 1048576, 4096)
HDR = struct.Struct("!5sBQ")


def enc(val):
    return json.dumps(val, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def sha_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def write_fr(fd_num, ch_name, raw):
    idx = CHANNELS.index(ch_name)
    if len(raw) > LIMITS[idx]:
        raise ValueError("frame overflow")
    framed = HDR.pack(b"TLFC1", idx + 1, len(raw)) + raw
    view = memoryview(framed)
    while view:
        nw = os.write(fd_num, view)
        if nw <= 0:
            raise OSError("short frame write")
        view = view[nw:]


def read_ex(stream, count):
    buf = bytearray()
    while len(buf) < count:
        chunk = stream.read(min(65536, count - len(buf)))
        if not chunk:
            raise ValueError("truncated payload")
        buf.extend(chunk)
    return bytes(buf)


def write_priv(fpath, raw):
    fdesc = os.open(fpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fdesc, "wb") as st_out:
        os.fchmod(st_out.fileno(), 0o600)
        st_out.write(raw)
        st_out.flush()
        os.fsync(st_out.fileno())


def main_boot(cfg):
    w_sec = int(cfg["work_timeout_seconds"])
    if not (60 <= w_sec <= 14400):
        raise ValueError("work_timeout_seconds bounds")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL, 30)
    anchor = time.monotonic()

    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("isolated python required")
    if socket.gethostname().split(".")[0] != cfg["hostname"] or os.getuid() != cfg["uid"]:
        raise ValueError("host/uid mismatch")
    if sys.executable != cfg["interpreter"] and os.path.realpath(sys.executable) != cfg["resolved_interpreter"]:
        raise ValueError("interpreter mismatch")

    p_path = Path(cfg["parent"])
    if not p_path.is_absolute() or p_path != p_path.resolve():
        raise ValueError("noncanonical parent")
    p_path.mkdir(mode=0o700, exist_ok=True)
    stage_prefix = "g6a-probe-" if cfg["stage_mode"] == "g6a_probe" else "g6b-abl-"
    stage = p_path / (stage_prefix + cfg["run_id"])
    stage.mkdir(mode=0o700)
    if str(stage) != cfg["stage"]:
        raise ValueError("stage mismatch")
    os.chdir(stage)

    ident = {
        "run_id": cfg["run_id"],
        "config_sha256": sha_hex(enc(cfg)),
        "hostname": cfg["hostname"],
        "uid": os.getuid(),
        "stage": str(stage),
    }
    write_fr(1, "READY", enc(ident))

    out_fd = 1
    phase = "INPUT"
    boot_fail = None
    body_res = None
    d_out = b""
    d_err = b""
    try:
        grant_line = sys.stdin.buffer.readline(256)
        if not grant_line.endswith(b"\\n"):
            raise ValueError("bad grant")
        grant = json.loads(grant_line.decode("utf-8"))
        rem_s = anchor + (grant["remaining_ns"] / 1_000_000_000.0) - time.monotonic()
        if rem_s <= 0:
            raise TimeoutError("expired grant")
        signal.setitimer(signal.ITIMER_REAL, rem_s)

        raw_pay = read_ex(sys.stdin.buffer, cfg["payload_bytes"])
        if sys.stdin.buffer.read(1):
            raise ValueError("trailing stdin")
        if sha_hex(raw_pay) != cfg["payload_sha256"]:
            raise ValueError("payload sha mismatch")
        bundle = json.loads(raw_pay.decode("utf-8"))
        if set(bundle["sources"]) != set(cfg["sources_sha256"]):
            raise ValueError("source set mismatch")
        for fn, exp_h in sorted(cfg["sources_sha256"].items()):
            m_raw = base64.b64decode(bundle["sources"][fn], validate=True)
            if sha_hex(m_raw) != exp_h:
                raise ValueError("source pin: " + fn)
            write_priv(stage / fn, m_raw)

        saved = os.dup(1)
        os.set_inheritable(saved, False)
        for fdn, lnm in ((1, "native.stdout"), (2, "native.stderr")):
            lfd = os.open(stage / lnm, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.fchmod(lfd, 0o600)
                os.dup2(lfd, fdn)
            finally:
                os.close(lfd)
        out_fd = saved

        phase = "REMOTE_BODY"
        sys.path.extend([str(stage), cfg["site_packages"]])
        import confirmatory_v2_gated_ablation as g6_mod
        if cfg["stage_mode"] == "g6a_probe":
            body_res = g6_mod.execute_remote_g6a_probe_body(cfg, stage, bundle)
        elif cfg["stage_mode"] == "g6b_ablation":
            body_res = g6_mod.execute_remote_g6b_ablation_body(cfg, stage, bundle)
        else:
            raise ValueError("unknown stage_mode")
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        boot_fail = {"phase": phase, "error_type": type(exc).__name__, "message": str(exc)[:512]}

    if out_fd != 1:
        sys.stdout.flush()
        sys.stderr.flush()
        d_out = (stage / "native.stdout").read_bytes()[:1048576]
        d_err = (stage / "native.stderr").read_bytes()[:1048576]

    res_kind = (
        "LORA_GATE_G6A_PROBE_PHASE1_RESULT"
        if cfg["stage_mode"] == "g6a_probe"
        else "LORA_GATE_G6B_GATED_PHASE1_RESULT"
    )
    res_payload = enc({
        "kind": res_kind,
        "version": 1,
        "identity": ident,
        "bootstrap_failure": boot_fail,
        "body": body_res,
    })
    write_fr(out_fd, "RESULT", res_payload)
    write_fr(out_fd, "STDOUT", d_out)
    write_fr(out_fd, "STDERR", d_err)
    write_fr(out_fd, "DONE", enc({"config_sha256": ident["config_sha256"], "result_sha256": sha_hex(res_payload)}))
    return 0 if boot_fail is None else 1
"""


def reconstruct_g5_episodes_from_run_dir(raw_files, worker_raw, g5_run_dir):
    run_path = Path(g5_run_dir)
    pub_pkg = parse_json((run_path / "ablation-package-public.json").read_bytes())
    p1_res = parse_json((run_path / "phase1-result.json").read_bytes())
    p2_env = parse_json(
        (run_path / "phase2-multi-stdout.json").read_bytes().strip(),
        limit=MULTI_BUNDLE_STDOUT_LIMIT,
    )
    dataset = load_confirmatory_dataset(raw_files)
    prompts_map = {item["id"]: item for item in dataset["prompts"]}
    oracles_map = {item["id"]: item for item in dataset["oracles"]}
    worker_text = worker_raw.decode("utf-8")

    positions = []
    valid_positions = []
    for ep_idx, ep_rec in enumerate(p1_res["phase_one_records"]):
        tid = ep_rec["task_id"]
        tp = prompts_map[tid]
        orc = oracles_map[tid]
        for role_name in CONFIRMATORY_ROLES:
            cand_src = ep_rec["candidates"][role_name]
            is_val = candidate_valid(tp, cand_src)
            for case_obj in orc["cases"]:
                pos_idx = len(positions)
                req_obj = (
                    {
                        "request_id": "conv-p2-%04d" % len(valid_positions),
                        "source": cand_src,
                        "entry_point": tp["entry_point"],
                        "args": copy.deepcopy(case_obj["args"]),
                    }
                    if is_val
                    else None
                )
                ent = {
                    "position_index": pos_idx,
                    "valid_source": is_val,
                    "request": req_obj,
                    "request_sha256": digest(req_obj) if is_val else None,
                    "expected": copy.deepcopy(case_obj["expected"]),
                }
                positions.append(ent)
                if is_val:
                    valid_positions.append(ent)

    b_slices = [valid_positions[off:off + 32] for off in range(0, len(valid_positions), 32)]
    used_ids = pub_pkg["scoring_run_ids"][:len(b_slices)]
    built_b = []
    final_recs = []
    for b_idx, (b_sl, b_rid, irec) in enumerate(zip(b_slices, used_ids, p2_env["results"])):
        b_raw = encoded({
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
            "mode": "matrix",
            "run_id": b_rid,
            "worker_source": worker_text,
            "requests": [pos["request"] for pos in b_sl],
        })
        built_b.append({
            "batch_index": b_idx,
            "run_id": b_rid,
            "bundle_raw": b_raw,
            "bundle_sha256": _sha256(b_raw),
            "worker_sha256": WORKER_SHA256,
            "positions": b_sl,
        })
        final_recs.append({"batch_index": b_idx, "bundle_raw": b_raw, "transport": irec["transport"]})

    cont_recs = []
    for b_meta, f_rec in zip(built_b, final_recs):
        obs_m, f_ok, p_ok, _pe = _decode_scoring_batch(
            b_meta, f_rec, pub_pkg["phase_one_config"]["expected_runtime"]
        )
        if f_ok or (not p_ok):
            continue
        rem_sl = b_meta["positions"][len(obs_m):]
        sub_idx = 1
        while rem_sl:
            b_idx = b_meta["batch_index"]
            cb_path = run_path / ("phase2-batch-%02d-cont-%02d-bundle.json" % (b_idx, sub_idx))
            if not cb_path.exists():
                break
            cb_raw = cb_path.read_bytes()
            cout_raw = (run_path / ("phase2-batch-%02d-cont-%02d-stdout.jsonl" % (b_idx, sub_idx))).read_bytes()
            cerr_raw = (run_path / ("phase2-batch-%02d-cont-%02d-stderr.bin" % (b_idx, sub_idx))).read_bytes()
            exp_c = build_confirmatory_continuation_bundle(
                worker_raw, pub_pkg["run_id"], b_idx, sub_idx, rem_sl
            )
            require(cb_raw == exp_c["bundle_raw"], "G5 cont bundle bytes mismatch")
            crec = {
                "batch_index": b_idx,
                "sub_index": sub_idx,
                "run_id": exp_c["run_id"],
                "position_indices": [pos["position_index"] for pos in rem_sl],
                "bundle_raw": cb_raw,
                "transport": {
                    "elapsed_ns": 0,
                    "process_reaped": True,
                    "returncode": 0,
                    "status": "completed",
                    "stdin_complete": True,
                    "stdout_hex": cout_raw.hex(),
                    "stderr_hex": cerr_raw.hex(),
                },
            }
            cont_recs.append(crec)
            co, cf, cp, _cpr = _decode_scoring_batch(
                exp_c,
                crec,
                pub_pkg["phase_one_config"]["expected_runtime"],
                check_fields=("batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"),
            )
            if cf:
                rem_sl = []
            elif cp:
                rem_sl = rem_sl[len(co):]
                sub_idx += 1
            else:
                break

    pos_obs, proto_bad, inc_reasons = _collect_scored_positions(
        built_b,
        final_recs,
        cont_recs,
        worker_raw,
        pub_pkg["run_id"],
        pub_pkg["phase_one_config"]["expected_runtime"],
    )
    require(not proto_bad and not inc_reasons, "G5 episode reconstruction failed")

    reconstructed = []
    for ep_idx, ep_rec in enumerate(p1_res["phase_one_records"]):
        if ep_rec["arm"] != "lora_conv":
            continue
        roles_out = {}
        for r_idx, role_name in enumerate(CONFIRMATORY_ROLES):
            base_pos = (ep_idx * 16) + (r_idx * 8)
            sl_pos = positions[base_pos:base_pos + 8]
            valid_src = sl_pos[0]["valid_source"]
            case_recs = []
            for p_it in sl_pos:
                ob = pos_obs.get(p_it["position_index"])
                if ob is None:
                    case_recs.append(None)
                else:
                    case_recs.append({
                        "status": ob["status"],
                        "process_reaped": ob["process_reaped"],
                        "matched": (
                            ob["status"] == "completed"
                            and ob["outcome"] is not None
                            and encoded(ob["outcome"]) == encoded(p_it["expected"])
                        ),
                    })
            roles_out[role_name] = _classify_candidate(valid_src, case_recs)
        reconstructed.append({
            "task_id": ep_rec["task_id"],
            "split": ep_rec["split"],
            "arm": "lora_conv",
            "roles": roles_out,
        })
    require(len(reconstructed) == 24, "24 reconstructed G5 episodes required")
    return reconstructed


def _sha256(raw_bytes):
    return hashlib.sha256(raw_bytes).hexdigest()


def _router_public_only(task_obj):
    return {
        "id": task_obj["id"],
        **{pub_k: copy.deepcopy(task_obj[pub_k]) for pub_k in ROUTER_ALLOWED_PUBLIC_KEYS},
    }


def load_g6_calibration_outputs_only(supervision_v2_raw, v1_prompts_raw, v1_structure_raw):
    require(
        type(supervision_v2_raw) is bytes
        and _sha256(supervision_v2_raw) == CONFIRMATORY_FILE_PINS["supervision.json"],
        "supervision.json pin mismatch",
    )
    require(
        type(v1_prompts_raw) is bytes and _sha256(v1_prompts_raw) == V1_PROMPTS_SHA256,
        "prompts_v1.json pin mismatch",
    )
    require(
        type(v1_structure_raw) is bytes and _sha256(v1_structure_raw) == V1_STRUCTURE_SHA256,
        "structure_v1.json pin mismatch",
    )
    sup_items = parse_json(supervision_v2_raw)
    v1_prompts = parse_json(v1_prompts_raw)
    v1_struct = parse_json(v1_structure_raw)
    require(tuple(item["id"] for item in sup_items) == SUPERVISION_IDS, "supervision IDs order")

    v1_split_by_id = {item["id"]: item["split"] for item in v1_struct}
    v1_by_id = {item["id"]: item for item in v1_prompts}
    for bg_tid in V1_BACKGROUND_IDS:
        require(v1_split_by_id.get(bg_tid) == "retention" and bg_tid in v1_by_id, "v1 background task missing")
    for neg_tid in V1_DEV_NEG_IDS:
        require(v1_split_by_id.get(neg_tid) == "development" and neg_tid in v1_by_id, "v1 dev negative missing")

    pos_tasks = {item["id"]: _router_public_only(item) for item in sup_items}
    bg_tasks = {bg_tid: _router_public_only(v1_by_id[bg_tid]) for bg_tid in V1_BACKGROUND_IDS}
    dev_neg_tasks = {neg_tid: _router_public_only(v1_by_id[neg_tid]) for neg_tid in V1_DEV_NEG_IDS}
    return {
        "positive": pos_tasks,
        "background": bg_tasks,
        "dev_negative": dev_neg_tasks,
    }


def load_g6_calibration_and_probe_inputs(
    supervision_v2_raw,
    v1_prompts_raw,
    v1_structure_raw,
    recal_prompts_raw,
):
    base_sets = load_g6_calibration_outputs_only(supervision_v2_raw, v1_prompts_raw, v1_structure_raw)
    require(
        type(recal_prompts_raw) is bytes and _sha256(recal_prompts_raw) == RECAL_V2_R1_PROMPTS_SHA256,
        "recalibration_v2_r1/prompts.json pin mismatch",
    )
    recal_items = parse_json(recal_prompts_raw)
    require(tuple(item["id"] for item in recal_items) == RECAL_DEV_POS_IDS, "recal C01..C12 order")
    dev_pos_tasks = {item["id"]: _router_public_only(item) for item in recal_items}

    seen_entries = set()
    for group_map in (
        base_sets["positive"],
        base_sets["background"],
        dev_pos_tasks,
        base_sets["dev_negative"],
    ):
        for task_obj in group_map.values():
            ep_name = task_obj["entry_point"]
            require(ep_name not in seen_entries, "duplicate entry_point across G6-A sets: " + ep_name)
            seen_entries.add(ep_name)

    return {
        "positive": base_sets["positive"],
        "background": base_sets["background"],
        "dev_positive": dev_pos_tasks,
        "dev_negative": base_sets["dev_negative"],
    }


def _build_bootstrap_ssh_argv(ssh_argv_prefix, interpreter, boot_bytes, cfg_raw):
    boot_b64 = base64.b64encode(boot_bytes).decode("ascii")
    cfg_b64 = base64.b64encode(cfg_raw).decode("ascii")
    remote_prog = (
        "import base64,hashlib,json,sys\n"
        f"src=base64.b64decode({boot_b64!r},validate=True)\n"
        f"cfg=json.loads(base64.b64decode({cfg_b64!r},validate=True).decode('ascii'))\n"
        "if hashlib.sha256(src).hexdigest()!=cfg['bootstrap_sha256']:\n"
        "    sys.exit(78)\n"
        "ns={'__name__':'g6_boot'}\n"
        "exec(compile(src,'<g6_stdlib_bootstrap>','exec'),ns)\n"
        "sys.exit(ns['main_boot'](cfg))\n"
    )
    remote_cmd = "exec " + shlex.join((
        "env",
        "-u",
        "LD_PRELOAD",
        "-u",
        "LD_LIBRARY_PATH",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONPATH",
        interpreter,
        "-I",
        "-S",
        "-B",
        "-c",
        remote_prog,
    ))
    require(len(remote_cmd.encode("utf-8")) <= 65536, "SSH command size exceeds cap")
    return [*ssh_argv_prefix, remote_cmd]


def build_gate_g6a_probe_package(
    supervision_v2_raw,
    v1_prompts_raw,
    v1_structure_raw,
    recal_prompts_raw,
    dep_sources,
    preflight_raw,
    *,
    run_id,
    ssh_argv_prefix,
    ssh_environment,
    interpreter,
    resolved_interpreter,
    site_packages,
    checkpoint_root,
    expected_tokenizer_sha256,
    remote_parent,
    local_run_root,
    work_timeout_seconds=1200,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    probe_sets = load_g6_calibration_and_probe_inputs(
        supervision_v2_raw, v1_prompts_raw, v1_structure_raw, recal_prompts_raw
    )
    require(set(dep_sources) == G6_SOURCE_NAMES, "G6 source names mismatch")
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    sources_sha = {fname: _sha256(dep_sources[fname]) for fname in sorted(G6_SOURCE_NAMES)}
    probe_sets_raw = encoded(probe_sets)

    payload_raw = encoded({
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(G6_SOURCE_NAMES)
        },
        "probe_sets_b64": base64.b64encode(probe_sets_raw).decode("ascii"),
    })
    boot_bytes = G6_STDLIB_BOOTSTRAP.encode("ascii")
    boot_sha = _sha256(boot_bytes)
    p1_cfg = {
        "kind": "LORA_GATE_G6A_PROBE_CONFIG",
        "version": 1,
        "stage_mode": "g6a_probe",
        "hostname": expected_runtime["hostname"].split(".")[0],
        "uid": expected_runtime["uid"],
        "interpreter": interpreter,
        "resolved_interpreter": resolved_interpreter,
        "site_packages": site_packages,
        "checkpoint_root": checkpoint_root,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "expected_runtime": copy.deepcopy(expected_runtime),
        "parent": remote_parent,
        "stage": str(Path(remote_parent) / ("g6a-probe-" + run_id)),
        "run_id": run_id,
        "sources_sha256": sources_sha,
        "probe_sets_sha256": _sha256(probe_sets_raw),
        "payload_sha256": _sha256(payload_raw),
        "payload_bytes": len(payload_raw),
        "bootstrap_sha256": boot_sha,
        "work_timeout_seconds": work_timeout_seconds,
    }
    p1_cfg_raw = encoded(p1_cfg)
    p1_cfg_sha = _sha256(p1_cfg_raw)
    p1_ssh_argv = _build_bootstrap_ssh_argv(ssh_argv_prefix, interpreter, boot_bytes, p1_cfg_raw)

    manifest = {
        "kind": "LORA_GATE_G6A_PROBE_LAUNCH_PACKAGE",
        "version": 1,
        "run_id": run_id,
        "input_pins": {
            "confirmatory_supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
            "v1_prompts_sha256": V1_PROMPTS_SHA256,
            "v1_structure_sha256": V1_STRUCTURE_SHA256,
            "recal_v2_r1_prompts_sha256": RECAL_V2_R1_PROMPTS_SHA256,
        },
        "planned_prefill_tasks": 40,
        "planned_generations": 0,
        "zero_confirmatory_eval_tasks_in_payload": True,
        "phase_one_config": p1_cfg,
        "phase_one_config_sha256": p1_cfg_sha,
        "phase_one_argv_sha256": digest(p1_ssh_argv),
        "local_run_root": local_run_root,
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "phase_one_ssh_argv": p1_ssh_argv,
            "ssh_environment": dict(ssh_environment),
            "phase_one_payload_hex": payload_raw.hex(),
        },
    }


def execute_remote_g6a_probe_body(config, stage, bundle):
    from recalibration_diagnostic_remote import (
        OFFLINE_GPU_ENVIRONMENT,
        _initialize_cuda_and_load_model,
        _query_and_select_gpu,
        _verify_checkpoint_files,
    )

    probe_sets_raw = base64.b64decode(bundle["probe_sets_b64"], validate=True)
    require(_sha256(probe_sets_raw) == config["probe_sets_sha256"], "G6-A probe_sets pin mismatch")
    probe_sets = parse_json(probe_sets_raw)

    shared_costs = {}
    gpu_selected, gpu_obs = _query_and_select_gpu()
    write_private(stage / "gpu-before.json", encoded(gpu_obs))
    os.environ.update(OFFLINE_GPU_ENVIRONMENT)
    os.environ.update({
        "HOME": str(stage),
        "TMPDIR": str(stage),
        "HF_HOME": str(stage / "hf-cache"),
        "CUDA_VISIBLE_DEVICES": gpu_selected["uuid"],
    })

    chk_t0 = time.monotonic_ns()
    _verify_checkpoint_files(config["checkpoint_root"])
    shared_costs["checkpoint_verify_ns"] = time.monotonic_ns() - chk_t0

    loaded = _initialize_cuda_and_load_model(config, shared_costs)
    torch = loaded["torch"]
    device = loaded["device"]
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    observed_tok_sha = loaded["observed_tok_sha"]
    init_base_sha = loaded["init_base_sha"]
    init_cpu_rng = loaded["init_cpu_rng"]
    init_cuda_rng = loaded["init_cuda_rng"]
    tok_state_fn = loaded["tok_state_fn"]
    model_state_fn = loaded["model_state_fn"]
    tensor_digest_fn = loaded["tensor_digest_fn"]
    obj_digest = loaded["obj_digest_fn"]

    cal_t0 = time.monotonic_ns()
    pos_vecs = {
        tid: extract_router_embedding(model, tokenizer, probe_sets["positive"][tid])
        for tid in sorted(probe_sets["positive"])
    }
    bg_vecs = {
        tid: extract_router_embedding(model, tokenizer, probe_sets["background"][tid])
        for tid in sorted(probe_sets["background"])
    }
    router_bundle = fit_whitened_prototype_router(pos_vecs, bg_vecs)
    shared_costs["router_calibration_ns"] = time.monotonic_ns() - cal_t0

    dev_t0 = time.monotonic_ns()
    dev_pos_vecs = {
        tid: extract_router_embedding(model, tokenizer, probe_sets["dev_positive"][tid])
        for tid in sorted(probe_sets["dev_positive"])
    }
    dev_neg_vecs = {
        tid: extract_router_embedding(model, tokenizer, probe_sets["dev_negative"][tid])
        for tid in sorted(probe_sets["dev_negative"])
    }
    probe_eval = evaluate_gate_g6a_probe(router_bundle, dev_pos_vecs, dev_neg_vecs)
    shared_costs["dev_probe_eval_ns"] = time.monotonic_ns() - dev_t0

    fin_t0 = time.monotonic_ns()
    final_base_sha = tensor_digest_fn(model_state_fn(model))
    final_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
    rng_same = bool(
        torch.equal(init_cpu_rng, torch.random.get_rng_state())
        and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
    )
    if final_base_sha != init_base_sha or final_tok_sha != observed_tok_sha or not rng_same:
        raise RuntimeError("G6-A post-run integrity drift")
    shared_costs["final_integrity_ns"] = time.monotonic_ns() - fin_t0

    return {
        "gpu_selected": gpu_selected,
        "integrity": {
            "initial_base_sha256": init_base_sha,
            "final_base_sha256": final_base_sha,
            "tokenizer_state_sha256": final_tok_sha,
            "rng_unchanged": rng_same,
        },
        "router_bundle": router_bundle,
        "probe_evaluation": probe_eval,
        "shared_costs": shared_costs,
    }


def run_gate_g6a_probe_stage(package):
    from confirmatory_v2_controller import _run_conf_phase_one_ssh, _validate_ready

    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    manifest_no_sha = {key_n: val_v for key_n, val_v in pub.items() if key_n != "package_sha256"}
    require(digest(manifest_no_sha) == pub["package_sha256"], "package_sha256 mismatch")
    require(digest(priv["phase_one_ssh_argv"]) == pub["phase_one_argv_sha256"], "p1 argv sha")

    p1_cfg = pub["phase_one_config"]
    run_id = pub["run_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], run_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    payload_raw = bytes.fromhex(priv["phase_one_payload_hex"])
    p1_out = None
    probe_report = None

    def safe_write(fname, fbytes):
        try:
            _write_exclusive(run_fd, fname, fbytes)
        except BaseException as w_err:
            persistence_errors.append(w_err)
            if not isinstance(w_err, Exception) and not cancellations:
                cancellations.append((w_err, w_err.__traceback__))

    try:
        safe_write("g6a-package-public.json", _pretty_json(pub))
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
                frames = decode_frames(p1_out["wire_bytes"])
                require(
                    _validate_ready(frames["READY"], p1_cfg, pub["phase_one_config_sha256"]),
                    "G6-A READY mismatch",
                )
                done_obj = strict_json(frames["DONE"])
                require(
                    done_obj == {
                        "config_sha256": pub["phase_one_config_sha256"],
                        "result_sha256": _sha256(frames["RESULT"]),
                    },
                    "G6-A DONE mismatch",
                )
                parsed_res = strict_json(frames["RESULT"])
                safe_write("phase1-result.json", _pretty_json(parsed_res))
                safe_write("phase1-native-stdout.bin", frames["STDOUT"])
                safe_write("phase1-native-stderr.bin", frames["STDERR"])
                if parsed_res.get("bootstrap_failure") is None and p1_out["returncode"] == 0:
                    body = parsed_res["body"]
                    validate_router_bundle(body["router_bundle"])
                    safe_write("router-bundle.json", _pretty_json(body["router_bundle"]))
                    probe_report = {
                        "kind": "LORA_GATE_G6A_ROUTER_PROBE_REPORT",
                        "version": 1,
                        "run_id": run_id,
                        "complete_and_verified": True,
                        "integrity": body["integrity"],
                        "shared_costs": body["shared_costs"],
                        "loo_summary": body["router_bundle"]["loo_summary"],
                        "probe_evaluation": body["probe_evaluation"],
                        **{flag_n: False for flag_n in FALSE_FLAGS},
                    }
                    safe_write("gate-g6a-probe-report.json", _pretty_json(probe_report))

        p1_clean = bool(p1_out and p1_out["reaped"] and p1_out["group_absent"] and p1_out["cleanup_bounded"])
        binding_ok = bool(p1_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_GATE_G6A_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and probe_report is not None),
            "gate_g6a_passed": bool(probe_report and probe_report["probe_evaluation"]["gate_g6a_passed"]),
            "secondary_errors": secondary_errors,
            **{flag_n: False for flag_n in FALSE_FLAGS},
        }
        safe_write("g6a-controller-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": probe_report}


def build_gated_ablation_schedule(prompts_raw, supervision_raw, *, run_id):
    _check_hex(run_id, 32)
    require(_sha256(prompts_raw) == CONFIRMATORY_FILE_PINS["prompts.json"], "prompts pin")
    require(_sha256(supervision_raw) == CONFIRMATORY_FILE_PINS["supervision.json"], "supervision pin")
    prompts = parse_json(prompts_raw)
    require(tuple(item["id"] for item in prompts) == EVAL_IDS, "eval prompt order")
    prompts_map = {item["id"]: item for item in prompts}
    episodes = []

    for canary_tid in ("WT01", "WR01"):
        task = prompts_map[canary_tid]
        ep_idx = len(episodes)
        f_msgs = confirmatory_first_round(task, "base", supervision_raw)
        episodes.append({
            "episode_index": ep_idx,
            "task_id": canary_tid,
            "split": task["split"],
            "family": task["family"],
            "entry_point": task["entry_point"],
            "arm": "base_canary",
            "render_arm": "base",
            "first_messages": f_msgs,
            "first_messages_sha256": digest(f_msgs),
            "call_ids": {
                role_name: _sha256(
                    encoded(["g6b_abl_call", run_id, ep_idx, canary_tid, "base_canary", role_name])
                )[:32]
                for role_name in CONFIRMATORY_ROLES
            },
        })

    for task in prompts:
        ep_idx = len(episodes)
        f_msgs = confirmatory_first_round(task, "lora", supervision_raw)
        episodes.append({
            "episode_index": ep_idx,
            "task_id": task["id"],
            "split": task["split"],
            "family": task["family"],
            "entry_point": task["entry_point"],
            "arm": "lora_gated",
            "render_arm": "lora",
            "first_messages": f_msgs,
            "first_messages_sha256": digest(f_msgs),
            "call_ids": {
                role_name: _sha256(
                    encoded(["g6b_abl_call", run_id, ep_idx, task["id"], "lora_gated", role_name])
                )[:32]
                for role_name in CONFIRMATORY_ROLES
            },
        })

    require(len(episodes) == 26, "26 episodes required")
    return {
        "kind": "LORA_CONFIRMATORY_V2_GATED_ABLATION_SCHEDULE",
        "version": 1,
        "run_id": run_id,
        "scheduled_episodes": 26,
        "scheduled_generation_calls": 52,
        "episodes": episodes,
    }


def build_gate_g6b_ablation_package(
    raw_files,
    v1_prompts_raw,
    v1_structure_raw,
    worker_raw,
    backend_raw,
    dep_sources,
    preflight_raw,
    *,
    gate_g6a_probe_sha256,
    frozen_router_bundle=None,
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
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_confirmatory_dataset(raw_files)
    cal_sets = load_g6_calibration_outputs_only(
        raw_files["supervision.json"], v1_prompts_raw, v1_structure_raw
    )
    cal_public_raw = encoded({
        "positive": cal_sets["positive"],
        "background": cal_sets["background"],
    })
    require(_sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(_sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    require(set(dep_sources) == G6_SOURCE_NAMES, "G6 source names")
    _check_hex(run_id, 32)
    _check_hex(gate_g6a_probe_sha256, 64)
    _check_hex(expected_tokenizer_sha256, 64)
    require(len(scoring_run_ids) == 13, "13 scoring_run_ids")
    seen_ids = {run_id}
    for s_rid in scoring_run_ids:
        _check_hex(s_rid, 32)
        require(s_rid not in seen_ids, "duplicate scoring_run_id")
        seen_ids.add(s_rid)

    frozen_router_sha = None
    router_b64 = None
    if frozen_router_bundle is not None:
        frozen_router_sha = validate_router_bundle(frozen_router_bundle)
        router_b64 = base64.b64encode(encoded(frozen_router_bundle)).decode("ascii")

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    sources_sha = {fname: _sha256(dep_sources[fname]) for fname in sorted(G6_SOURCE_NAMES)}
    payload_dict = {
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(G6_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision.json"]).decode("ascii"),
        "router_cal_public_b64": base64.b64encode(cal_public_raw).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    }
    if router_b64 is not None:
        payload_dict["frozen_router_b64"] = router_b64
    payload_raw = encoded(payload_dict)

    boot_bytes = G6_STDLIB_BOOTSTRAP.encode("ascii")
    boot_sha = _sha256(boot_bytes)
    p1_cfg = {
        "kind": "LORA_GATE_G6B_ABLATION_PHASE1_CONFIG",
        "version": 1,
        "stage_mode": "g6b_ablation",
        "hostname": expected_runtime["hostname"].split(".")[0],
        "uid": expected_runtime["uid"],
        "interpreter": interpreter,
        "resolved_interpreter": resolved_interpreter,
        "site_packages": site_packages,
        "checkpoint_root": checkpoint_root,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "expected_runtime": copy.deepcopy(expected_runtime),
        "parent": remote_parent,
        "stage": str(Path(remote_parent) / ("g6b-abl-" + run_id)),
        "run_id": run_id,
        "gate_g6a_probe_sha256": gate_g6a_probe_sha256,
        "frozen_router_sha256": frozen_router_sha,
        "sources_sha256": sources_sha,
        "prompts_sha256": CONFIRMATORY_FILE_PINS["prompts.json"],
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "router_cal_public_sha256": _sha256(cal_public_raw),
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "payload_sha256": _sha256(payload_raw),
        "payload_bytes": len(payload_raw),
        "bootstrap_sha256": boot_sha,
        "work_timeout_seconds": work_timeout_seconds,
    }
    p1_cfg_raw = encoded(p1_cfg)
    p1_cfg_sha = _sha256(p1_cfg_raw)
    p1_ssh_argv = _build_bootstrap_ssh_argv(ssh_argv_prefix, interpreter, boot_bytes, p1_cfg_raw)
    p2_ssh_argv = build_multi_bundle_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )

    manifest = {
        "kind": "LORA_GATE_G6B_ABLATION_LAUNCH_PACKAGE",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
        "gate_g6a_probe_sha256": gate_g6a_probe_sha256,
        "frozen_router_sha256": frozen_router_sha,
        "scoring_run_ids": list(scoring_run_ids),
        "hyperparams": copy.deepcopy(ABLATION_TRAINING_HYPERPARAMS),
        "file_sha256": dict(CONFIRMATORY_FILE_PINS),
        "phase_one_config": p1_cfg,
        "phase_one_config_sha256": p1_cfg_sha,
        "phase_one_argv_sha256": digest(p1_ssh_argv),
        "phase_two_argv_sha256": digest(p2_ssh_argv),
        "local_run_root": local_run_root,
        "planned_episodes": 26,
        "planned_generations": 52,
        "zero_private_oracle_in_phase_one": True,
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "phase_one_ssh_argv": p1_ssh_argv,
            "phase_two_ssh_argv": p2_ssh_argv,
            "ssh_environment": dict(ssh_environment),
            "phase_one_payload_hex": payload_raw.hex(),
        },
    }


def execute_remote_g6b_ablation_body(config, stage, bundle):
    from confirmatory_v2_training import prepare_confirmatory_training
    from observed_generation import generate_observed
    from recalibration_diagnostic_remote import (
        OFFLINE_GPU_ENVIRONMENT,
        _initialize_cuda_and_load_model,
        _query_and_select_gpu,
        _run_public_cases_child,
        _verify_checkpoint_files,
    )
    from tokenizer_measurement import measure as measure_tokens

    prompts_raw = base64.b64decode(bundle["prompts_b64"], validate=True)
    supervision_raw = base64.b64decode(bundle["supervision_b64"], validate=True)
    cal_pub_raw = base64.b64decode(bundle["router_cal_public_b64"], validate=True)
    backend_raw = base64.b64decode(bundle["backend_b64"], validate=True)
    worker_raw = base64.b64decode(bundle["worker_b64"], validate=True)
    require(
        _sha256(prompts_raw) == config["prompts_sha256"]
        and _sha256(supervision_raw) == config["supervision_sha256"]
        and _sha256(cal_pub_raw) == config["router_cal_public_sha256"]
        and _sha256(backend_raw) == config["backend_sha256"]
        and _sha256(worker_raw) == config["worker_sha256"],
        "G6-B input pin mismatch",
    )

    shared_costs = {}
    gpu_selected, gpu_obs = _query_and_select_gpu()
    write_private(stage / "gpu-before.json", encoded(gpu_obs))
    os.environ.update(OFFLINE_GPU_ENVIRONMENT)
    os.environ.update({
        "HOME": str(stage),
        "TMPDIR": str(stage),
        "HF_HOME": str(stage / "hf-cache"),
        "CUDA_VISIBLE_DEVICES": gpu_selected["uuid"],
    })

    chk_t0 = time.monotonic_ns()
    _verify_checkpoint_files(config["checkpoint_root"])
    shared_costs["checkpoint_verify_ns"] = time.monotonic_ns() - chk_t0

    loaded = _initialize_cuda_and_load_model(config, shared_costs)
    torch = loaded["torch"]
    device = loaded["device"]
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]
    observed_tok_sha = loaded["observed_tok_sha"]
    init_base_sha = loaded["init_base_sha"]
    init_cpu_rng = loaded["init_cpu_rng"]
    init_cuda_rng = loaded["init_cuda_rng"]
    tok_state_fn = loaded["tok_state_fn"]
    model_state_fn = loaded["model_state_fn"]
    tensor_digest_fn = loaded["tensor_digest_fn"]
    obj_digest = loaded["obj_digest_fn"]

    if config.get("frozen_router_sha256") is not None:
        router_raw = base64.b64decode(bundle["frozen_router_b64"], validate=True)
        router_bundle = parse_json(router_raw)
        require(
            validate_router_bundle(router_bundle) == config["frozen_router_sha256"],
            "frozen router_bundle sha mismatch",
        )
        shared_costs["router_calibration_ns"] = 0
    else:
        cal_sets = parse_json(cal_pub_raw)
        cal_t0 = time.monotonic_ns()
        pos_vecs = {
            tid: extract_router_embedding(model, tokenizer, cal_sets["positive"][tid])
            for tid in sorted(cal_sets["positive"])
        }
        bg_vecs = {
            tid: extract_router_embedding(model, tokenizer, cal_sets["background"][tid])
            for tid in sorted(cal_sets["background"])
        }
        router_bundle = fit_whitened_prototype_router(pos_vecs, bg_vecs)
        shared_costs["router_calibration_ns"] = time.monotonic_ns() - cal_t0

    prompts_list = parse_json(prompts_raw)
    prompts_map = {item["id"]: item for item in prompts_list}

    route_t0 = time.monotonic_ns()
    gate_decisions = {}
    for tid in EVAL_IDS:
        query_vec = extract_router_embedding(model, tokenizer, prompts_map[tid])
        gate_decisions[tid] = score_router_embedding(router_bundle, query_vec)
    shared_costs["router_eval_ns"] = time.monotonic_ns() - route_t0

    train_ds = prepare_confirmatory_training(tokenizer, supervision_raw)
    train_out = train_converged_ablation_adapter(model, train_ds)
    buf_io = io.BytesIO()
    save_t0 = time.perf_counter_ns()
    torch.save({"adapter": train_out["adapter"]}, buf_io)
    art_raw = buf_io.getvalue()
    write_private(stage / "converged_adapter.pt", art_raw)
    save_ns = time.perf_counter_ns() - save_t0
    load_t0 = time.perf_counter_ns()
    reloaded = torch.load(
        io.BytesIO((stage / "converged_adapter.pt").read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    load_ns = time.perf_counter_ns() - load_t0
    require(
        tensor_digest_fn(reloaded["adapter"]) == train_out["adapter_sha256"],
        "reloaded converged adapter digest mismatch",
    )
    train_out["adapter"] = reloaded["adapter"]

    epoch_means = [
        sum(st_row["loss"] for st_row in train_out["steps"][ep_i * 8:(ep_i + 1) * 8]) / 8.0
        for ep_i in range(ABLATION_TRAINING_HYPERPARAMS["epochs"])
    ]
    training_summary = {
        "hyperparams": train_out["hyperparams"],
        "dataset_sha256": train_out["dataset_sha256"],
        "initial_adapter_sha256": train_out["initial_adapter_sha256"],
        "adapter_sha256": train_out["adapter_sha256"],
        "artifact_sha256": _sha256(art_raw),
        "artifact_bytes": len(art_raw),
        "adapter_bytes": train_out["adapter_bytes"],
        "preparation_ns": train_out["preparation_ns"],
        "training_ns": train_out["training_ns"],
        "verification_ns": train_out["verification_ns"],
        "core_wall_ns": train_out["core_wall_ns"],
        "adapter_save_ns": save_ns,
        "adapter_load_ns": load_ns,
        "peak_allocated_bytes": train_out["peak_allocated_bytes"],
        "peak_reserved_bytes": train_out["peak_reserved_bytes"],
        "epoch_mean_losses": epoch_means,
        "steps": train_out["steps"],
    }

    schedule = build_gated_ablation_schedule(prompts_raw, supervision_raw, run_id=config["run_id"])
    worker_text = worker_raw.decode("utf-8")
    gen_dir = stage / "generations"
    gen_dir.mkdir(mode=0o700)

    def run_single_episode(active_mod, ep_spec, g_open_flag):
        ep_idx = ep_spec["episode_index"]
        tid = ep_spec["task_id"]
        task_prompt = prompts_map[tid]
        ep_dir = gen_dir / ("ep-%02d" % ep_idx)
        ep_dir.mkdir(mode=0o700)
        init_dir = ep_dir / "initial"
        init_dir.mkdir(mode=0o700)
        fb_dir = ep_dir / "feedback"
        fb_dir.mkdir(mode=0o700)

        init_out = generate_observed(
            active_mod,
            tokenizer,
            ep_spec["first_messages"],
            expected_tokenizer_sha256=observed_tok_sha,
            call_id=ep_spec["call_ids"]["initial"],
            artifact_dir=init_dir,
        )
        init_gen = init_out["generation"]
        init_cand = init_gen["candidate"]

        if candidate_valid(task_prompt, init_cand):
            pub_fb, pub_ns = _run_public_cases_child(
                config["resolved_interpreter"],
                backend_raw,
                worker_text,
                config["expected_runtime"],
                config["run_id"],
                ep_idx,
                task_prompt,
                init_cand,
            )
        else:
            pub_fb = {"kind": "invalid_source"}
            pub_ns = 0

        sec_msgs = confirmatory_second_round(
            task_prompt, ep_spec["render_arm"], supervision_raw, init_cand, pub_fb
        )
        sec_chk = measure_tokens(tokenizer, sec_msgs, answer=None)
        if sec_chk["status"] != "OK" or sec_chk["input"]["input_tokens"] > 4096:
            fb_cand = ""
            fb_cost = {
                "input_tokens": sec_chk["input"]["input_tokens"],
                "output_tokens": 0,
                "forward_calls": 0,
                "wall_ns": 0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }
        else:
            fb_out = generate_observed(
                active_mod,
                tokenizer,
                sec_msgs,
                expected_tokenizer_sha256=observed_tok_sha,
                call_id=ep_spec["call_ids"]["feedback"],
                artifact_dir=fb_dir,
            )
            fb_gen = fb_out["generation"]
            fb_cand = fb_gen["candidate"]
            fb_cost = _cost_projection(fb_gen)

        return {
            "episode_index": ep_idx,
            "task_id": tid,
            "split": ep_spec["split"],
            "arm": ep_spec["arm"],
            "gate_open": bool(g_open_flag),
            "candidates": {"initial": init_cand, "feedback": fb_cand},
            "public_feedback": pub_fb,
            "generation_costs": {"initial": _cost_projection(init_gen), "feedback": fb_cost},
            "public_execution_ns": pub_ns,
        }

    unmounted_eps = []
    mounted_eps = []
    for ep_spec in schedule["episodes"]:
        if ep_spec["arm"] == "base_canary" or not gate_decisions[ep_spec["task_id"]]["gate_open"]:
            unmounted_eps.append(ep_spec)
        else:
            mounted_eps.append(ep_spec)

    records_by_idx = {}
    p1_t0 = time.monotonic_ns()
    for ep_spec in unmounted_eps:
        records_by_idx[ep_spec["episode_index"]] = run_single_episode(model, ep_spec, False)

    if mounted_eps:
        with ExitStack() as mount_stack:
            mounted_mod = mount_stack.enter_context(mounted_converged_adapter(model, train_out))
            for ep_spec in mounted_eps:
                records_by_idx[ep_spec["episode_index"]] = run_single_episode(mounted_mod, ep_spec, True)

    phase_one_records = [records_by_idx[ep_idx] for ep_idx in range(len(schedule["episodes"]))]
    shared_costs["episodes_wall_ns"] = time.monotonic_ns() - p1_t0
    shared_costs["adapter_mounts_count"] = 1 if mounted_eps else 0

    fin_t0 = time.monotonic_ns()
    final_base_sha = tensor_digest_fn(model_state_fn(model))
    final_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
    rng_same = bool(
        torch.equal(init_cpu_rng, torch.random.get_rng_state())
        and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
    )
    if final_base_sha != init_base_sha or final_tok_sha != observed_tok_sha or not rng_same:
        raise RuntimeError("G6-B post-run integrity drift")
    shared_costs["final_integrity_ns"] = time.monotonic_ns() - fin_t0

    router_summary = {
        "bundle_sha256": router_bundle["bundle_sha256"],
        "threshold_tau": router_bundle["threshold_tau"],
        "estimated_fp32_fingerprint_bytes": router_bundle["estimated_fp32_fingerprint_bytes"],
        "loo_summary": router_bundle["loo_summary"],
    }
    return {
        "gpu_selected": gpu_selected,
        "integrity": {
            "initial_base_sha256": init_base_sha,
            "final_base_sha256": final_base_sha,
            "tokenizer_state_sha256": final_tok_sha,
            "rng_unchanged": rng_same,
        },
        "router_summary": router_summary,
        "gate_decisions": gate_decisions,
        "training_summary": training_summary,
        "shared_costs": shared_costs,
        "phase_one_records": phase_one_records,
    }


def _collect_scored_positions(
    built_batches,
    final_bundle_records,
    continuation_records,
    worker_raw,
    run_id,
    expected_runtime,
):
    proto_bad = []
    inc_reasons = []
    if len(final_bundle_records) != len(built_batches):
        inc_reasons.append("incomplete_final_scoring_batches")

    cont_by_batch = {}
    for crec in continuation_records:
        cont_by_batch.setdefault(crec["batch_index"], []).append(crec)

    pos_obs = {}
    for b_meta, b_rec in zip(built_batches, final_bundle_records):
        b_conts = cont_by_batch.pop(b_meta["batch_index"], [])
        obs_m, full_ok, part_ok, pr_err = _decode_scoring_batch(b_meta, b_rec, expected_runtime)
        if pr_err:
            proto_bad.append("batch_proto_err")
            inc_reasons.append("batch_incomplete")
            if b_conts:
                proto_bad.append("unexpected_continuation")
            continue
        pos_obs.update(obs_m)
        if full_ok:
            if b_conts:
                proto_bad.append("unexpected_continuation")
            continue
        if not part_ok:
            inc_reasons.append("batch_incomplete")
            if b_conts:
                proto_bad.append("unexpected_continuation")
            continue
        rem_sl = b_meta["positions"][len(obs_m):]
        settled = False
        for exp_s, crec in enumerate(b_conts, start=1):
            if not rem_sl:
                proto_bad.append("excess_continuation")
                break
            exp_c = build_confirmatory_continuation_bundle(
                worker_raw, run_id, b_meta["batch_index"], exp_s, rem_sl
            )
            if (
                crec.get("sub_index") != exp_s
                or crec.get("run_id") != exp_c["run_id"]
                or crec.get("position_indices") != [pos["position_index"] for pos in rem_sl]
            ):
                proto_bad.append("continuation_binding_mismatch")
                break
            c_obs, c_full, c_part, c_pr = _decode_scoring_batch(
                exp_c,
                crec,
                expected_runtime,
                check_fields=("batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"),
            )
            if c_pr:
                proto_bad.append("cont_proto_err")
                break
            pos_obs.update(c_obs)
            if c_full:
                rem_sl = []
                settled = True
            elif c_part:
                rem_sl = rem_sl[len(c_obs):]
            else:
                break
        if not settled:
            inc_reasons.append("batch_incomplete")

    if cont_by_batch:
        proto_bad.append("orphan_continuation")
    return pos_obs, proto_bad, inc_reasons


def score_gated_ablation(
    raw_files,
    worker_raw,
    run_id,
    phase_one_records,
    gate_decisions,
    router_summary,
    scoring_run_ids,
    final_bundle_records,
    continuation_records,
    expected_runtime,
    training_summary,
    shared_costs,
    g4b_phase_one_records,
    g4b_report,
    g5_phase_one_records,
    g5_report,
):
    dataset = load_confirmatory_dataset(raw_files)
    expected_sched = build_gated_ablation_schedule(
        raw_files["prompts.json"], raw_files["supervision.json"], run_id=run_id
    )["episodes"]
    prompts_map = {item["id"]: item for item in dataset["prompts"]}
    oracles_map = {item["id"]: item for item in dataset["oracles"]}
    worker_text = worker_raw.decode("utf-8")

    proto_bad = []
    inc_reasons = []
    if len(phase_one_records) != 26:
        inc_reasons.append("incomplete_phase_one_episodes")
    for ep_idx, ep_rec in enumerate(phase_one_records):
        if ep_idx >= len(expected_sched):
            proto_bad.append("excess_episode")
            break
        exp_ep = expected_sched[ep_idx]
        if (
            ep_rec.get("episode_index") != ep_idx
            or ep_rec.get("task_id") != exp_ep["task_id"]
            or ep_rec.get("arm") != exp_ep["arm"]
        ):
            proto_bad.append("schedule_identity_mismatch")

    positions = []
    valid_positions = []
    for ep_idx, ep_rec in enumerate(phase_one_records):
        tid = ep_rec["task_id"]
        task_prompt = prompts_map[tid]
        oracle = oracles_map[tid]
        for role_name in CONFIRMATORY_ROLES:
            cand_src = ep_rec["candidates"][role_name]
            is_val = candidate_valid(task_prompt, cand_src)
            for case_obj in oracle["cases"]:
                pos_idx = len(positions)
                req_obj = (
                    {
                        "request_id": "g6b-p2-%04d" % len(valid_positions),
                        "source": cand_src,
                        "entry_point": task_prompt["entry_point"],
                        "args": copy.deepcopy(case_obj["args"]),
                    }
                    if is_val
                    else None
                )
                entry = {
                    "position_index": pos_idx,
                    "episode_index": ep_idx,
                    "task_id": tid,
                    "arm": ep_rec["arm"],
                    "role": role_name,
                    "case_id": case_obj["id"],
                    "valid_source": is_val,
                    "source": cand_src,
                    "request": req_obj,
                    "request_sha256": digest(req_obj) if is_val else None,
                    "expected": copy.deepcopy(case_obj["expected"]),
                }
                positions.append(entry)
                if is_val:
                    valid_positions.append(entry)

    batch_slices = [valid_positions[off:off + 32] for off in range(0, len(valid_positions), 32)]
    built_batches = []
    for b_idx, (b_slice, b_rid) in enumerate(zip(batch_slices, scoring_run_ids[:len(batch_slices)])):
        b_raw = encoded({
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
            "mode": "matrix",
            "run_id": b_rid,
            "worker_source": worker_text,
            "requests": [pos["request"] for pos in b_slice],
        })
        built_batches.append({
            "batch_index": b_idx,
            "run_id": b_rid,
            "bundle_raw": b_raw,
            "bundle_sha256": _sha256(b_raw),
            "worker_sha256": WORKER_SHA256,
            "positions": b_slice,
        })

    pos_obs, c_proto_bad, c_inc = _collect_scored_positions(
        built_batches,
        final_bundle_records,
        continuation_records,
        worker_raw,
        run_id,
        expected_runtime,
    )
    proto_bad.extend(c_proto_bad)
    inc_reasons.extend(c_inc)

    g4b_cands = {(ep["task_id"], ep["arm"]): ep["candidates"] for ep in g4b_phase_one_records}
    g5_cands = {ep["task_id"]: ep["candidates"] for ep in g5_phase_one_records if ep["arm"] == "lora_conv"}
    g4b_eps = {(ep["task_id"], ep["arm"]): ep for ep in g4b_report["episodes"]}
    g5_eps = {(ep["task_id"], "lora_conv"): ep for ep in g5_report.get("episodes", [])}

    canary_matches = {}
    gated_eps = {}
    gated_wall_ns = 0
    closed_total = 0
    closed_exact = 0
    open_total = 0
    open_exact = 0

    for ep_idx, ep_rec in enumerate(phase_one_records):
        tid = ep_rec["task_id"]
        arm = ep_rec["arm"]
        try:
            val_pub = validate_public_feedback(ep_rec.get("public_feedback"))
        except Exception:
            val_pub = None
            proto_bad.append("invalid_public_feedback")
        roles_out = {}
        for r_idx, role_name in enumerate(CONFIRMATORY_ROLES):
            base_pos = (ep_idx * 16) + (r_idx * 8)
            sl_pos = positions[base_pos:base_pos + 8]
            valid_src = sl_pos[0]["valid_source"]
            case_recs = []
            for p_it in sl_pos:
                ob = pos_obs.get(p_it["position_index"])
                if ob is None:
                    case_recs.append(None)
                else:
                    case_recs.append({
                        "status": ob["status"],
                        "process_reaped": ob["process_reaped"],
                        "matched": (
                            ob["status"] == "completed"
                            and ob["outcome"] is not None
                            and encoded(ob["outcome"]) == encoded(p_it["expected"])
                        ),
                    })
            cls_res = _classify_candidate(valid_src, case_recs)
            if cls_res["state"] == "UNKNOWN" or cls_res["has_unknown_case"]:
                inc_reasons.append("unknown_outcome")
            if role_name == "initial" and isinstance(val_pub, dict):
                if (val_pub.get("kind") == "invalid_source") != (not valid_src):
                    proto_bad.append("public_feedback_validity_mismatch")
                elif valid_src and val_pub.get("kind") == "visible_tests":
                    for p1_st, p2_st in zip(
                        [cit["status"] for cit in val_pub["cases"]],
                        cls_res["case_statuses"][:2],
                    ):
                        if p2_st != "unknown" and {p1_st, p2_st} == {"pass", "mismatch"}:
                            proto_bad.append("phase1_phase2_public_contradiction")
            roles_out[role_name] = cls_res

        if arm == "base_canary":
            canary_matches[tid] = (
                ep_rec["candidates"]["initial"] == g4b_cands[(tid, "base")]["initial"]
                and ep_rec["candidates"]["feedback"] == g4b_cands[(tid, "base")]["feedback"]
            )
        elif arm == "lora_gated":
            g_open = bool(ep_rec.get("gate_open"))
            if g_open != bool(gate_decisions[tid]["gate_open"]):
                proto_bad.append("gate_decision_mismatch")
            if g_open:
                open_total += 1
                if tid in g5_cands and ep_rec["candidates"]["initial"] == g5_cands[tid]["initial"]:
                    open_exact += 1
            else:
                closed_total += 1
                if ep_rec["candidates"]["initial"] == g4b_cands[(tid, "base")]["initial"]:
                    closed_exact += 1

            gated_wall_ns += (
                ep_rec["generation_costs"]["initial"]["wall_ns"]
                + ep_rec["generation_costs"]["feedback"]["wall_ns"]
            )
            gated_eps[(tid, "lora_gated")] = {
                "task_id": tid,
                "split": ep_rec["split"],
                "arm": "lora_gated",
                "gate_open": g_open,
                "router_score": gate_decisions[tid]["score"],
                "roles": roles_out,
            }

    canaries_ok = (
        set(canary_matches) == {"WT01", "WR01"}
        and all(canary_matches.values())
        and len(gated_eps) == 24
    )
    closed_parity_ok = closed_exact == closed_total
    open_parity_ok = open_exact == open_total
    if not closed_parity_ok:
        proto_bad.append("closed_gate_diverged_from_base")
    run_ok = bool(not proto_bad and not inc_reasons and canaries_ok and closed_parity_ok)

    merged_indexed = {**g4b_eps, **g5_eps, **gated_eps}
    by_split = {}
    comparisons = {}
    for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
        by_split[sp_name] = {
            role_name: {
                st_n: sum(
                    (tid, "lora_gated") in gated_eps
                    and gated_eps[(tid, "lora_gated")]["roles"][role_name]["state"] == st_n
                    for tid in sp_ids
                )
                for st_n in CANDIDATE_STATES
            }
            for role_name in CONFIRMATORY_ROLES
        }
        comp_entry = {
            "lora_gated_vs_base": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "base", "lora_gated", run_ok
            ),
            "lora_gated_vs_text": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "text", "lora_gated", run_ok
            ),
        }
        if len(g5_eps) == 24:
            comp_entry["lora_gated_vs_lora_conv"] = _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "lora_conv", "lora_gated", run_ok
            )
        comparisons[sp_name] = comp_entry

    transfer_open_count = sum(1 for tid in TRANSFER_IDS if gate_decisions[tid]["gate_open"])
    retention_open_count = sum(1 for tid in RETENTION_IDS if gate_decisions[tid]["gate_open"])
    total_pass_fb = (
        by_split["transfer"]["feedback"]["PASS"]
        + by_split["retention"]["feedback"]["PASS"]
    )

    text_wall_ns = int(g4b_report["costs"]["by_arm"]["text"]["generation_wall_ns"])
    router_eval_ns = int(shared_costs.get("router_eval_ns", 0))
    total_gated_infer_ns = gated_wall_ns + router_eval_ns
    fixed_setup_ns = (
        int(training_summary["core_wall_ns"])
        + int(training_summary["adapter_save_ns"])
        + int(training_summary["adapter_load_ns"])
        + int(shared_costs.get("router_calibration_ns", 0))
    )
    saved_per_task_ns = (text_wall_ns - total_gated_infer_ns) / 24.0
    breakeven_tasks = (
        fixed_setup_ns / saved_per_task_ns if saved_per_task_ns > 0 else None
    )
    speedup_vs_text = (
        text_wall_ns / float(total_gated_infer_ns) if total_gated_infer_ns > 0 else None
    )

    return {
        "kind": "LORA_CONFIRMATORY_V2_GATED_ABLATION_REPORT",
        "version": 1,
        "run_id": run_id,
        "complete_and_verified": run_ok,
        "protocol_contradictions": sorted(set(proto_bad)),
        "incomplete_reasons": sorted(set(inc_reasons)),
        "parity_checks": {
            "canary_bit_exact_with_g4b_base": canary_matches,
            "closed_gate_count": closed_total,
            "closed_gate_bit_exact_count": closed_exact,
            "closed_gate_bit_exact_with_g4b_base": closed_parity_ok,
            "open_gate_count": open_total,
            "open_gate_bit_exact_count": open_exact,
            "open_gate_bit_exact_with_g5_lora_conv": open_parity_ok,
        },
        "routing_summary": {
            "router_bundle_sha256": router_summary["bundle_sha256"],
            "threshold_tau": router_summary["threshold_tau"],
            "estimated_fp32_fingerprint_bytes": router_summary["estimated_fp32_fingerprint_bytes"],
            "transfer_gate_open_count": transfer_open_count,
            "transfer_total": 12,
            "retention_gate_open_count": retention_open_count,
            "retention_total": 12,
            "decisions": gate_decisions,
        },
        "total_eval_tasks": 24,
        "total_pass_feedback": total_pass_fb,
        "training": training_summary,
        "lora_gated_by_split": by_split,
        "episodes": [gated_eps[(tid, "lora_gated")] for tid in EVAL_IDS if (tid, "lora_gated") in gated_eps],
        "comparisons_feedback": comparisons,
        "costs": {
            "lora_gated_generation_wall_ns": gated_wall_ns,
            "router_eval_ns": router_eval_ns,
            "total_gated_inference_wall_ns": total_gated_infer_ns,
            "text_reference_generation_wall_ns": text_wall_ns,
            "speedup_vs_text": speedup_vs_text,
            "fixed_training_and_router_ns": fixed_setup_ns,
            "breakeven_tasks_vs_text": breakeven_tasks,
            "total_payload_bytes": int(training_summary["adapter_bytes"]) + int(router_summary["estimated_fp32_fingerprint_bytes"]),
            "shared": shared_costs,
        },
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }


def run_gate_g6b_ablation_stage(
    package,
    raw_files,
    worker_raw,
    g4b_phase_one_records,
    g4b_report,
    g5_phase_one_records,
    g5_report,
):
    from confirmatory_v2_controller import _run_conf_phase_one_ssh, _validate_ready

    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    manifest_no_sha = {key_n: val_v for key_n, val_v in pub.items() if key_n != "package_sha256"}
    require(digest(manifest_no_sha) == pub["package_sha256"], "package_sha256 mismatch")
    require(digest(priv["phase_one_ssh_argv"]) == pub["phase_one_argv_sha256"], "p1 argv sha")
    require(digest(priv["phase_two_ssh_argv"]) == pub["phase_two_argv_sha256"], "p2 argv sha")

    p1_cfg = pub["phase_one_config"]
    run_id = pub["run_id"]
    run_fd = _claim_local_run_root(pub["local_run_root"], run_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
    payload_raw = bytes.fromhex(priv["phase_one_payload_hex"])
    p1_out = None
    body_obj = None
    p2_all_clean = True
    report = None

    def safe_write(fname, fbytes):
        try:
            _write_exclusive(run_fd, fname, fbytes)
        except BaseException as w_err:
            persistence_errors.append(w_err)
            if not isinstance(w_err, Exception) and not cancellations:
                cancellations.append((w_err, w_err.__traceback__))

    try:
        safe_write("g6b-package-public.json", _pretty_json(pub))
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
                frames = decode_frames(p1_out["wire_bytes"])
                require(
                    _validate_ready(frames["READY"], p1_cfg, pub["phase_one_config_sha256"]),
                    "G6-B READY mismatch",
                )
                done_obj = strict_json(frames["DONE"])
                require(
                    done_obj == {
                        "config_sha256": pub["phase_one_config_sha256"],
                        "result_sha256": _sha256(frames["RESULT"]),
                    },
                    "G6-B DONE mismatch",
                )
                parsed_res = strict_json(frames["RESULT"])
                safe_write("phase1-result.json", _pretty_json(parsed_res))
                safe_write("phase1-native-stdout.bin", frames["STDOUT"])
                safe_write("phase1-native-stderr.bin", frames["STDERR"])
                if parsed_res.get("bootstrap_failure") is None and p1_out["returncode"] == 0:
                    body_obj = parsed_res["body"]

        final_recs = []
        cont_recs = []
        used_ids = []
        if body_obj is not None and not persistence_errors and not cancellations:
            dataset = load_confirmatory_dataset(raw_files)
            prompts_map = {item["id"]: item for item in dataset["prompts"]}
            oracles_map = {item["id"]: item for item in dataset["oracles"]}
            worker_text = worker_raw.decode("utf-8")
            valid_positions = []
            all_positions = []
            for ep_rec in body_obj["phase_one_records"]:
                tid = ep_rec["task_id"]
                tp = prompts_map[tid]
                orc = oracles_map[tid]
                for role_name in CONFIRMATORY_ROLES:
                    c_src = ep_rec["candidates"][role_name]
                    is_v = candidate_valid(tp, c_src)
                    for c_obj in orc["cases"]:
                        pos_idx = len(all_positions)
                        req_o = (
                            {
                                "request_id": "g6b-p2-%04d" % len(valid_positions),
                                "source": c_src,
                                "entry_point": tp["entry_point"],
                                "args": copy.deepcopy(c_obj["args"]),
                            }
                            if is_v
                            else None
                        )
                        ent = {
                            "position_index": pos_idx,
                            "request": req_o,
                            "request_sha256": digest(req_o) if is_v else None,
                        }
                        all_positions.append(ent)
                        if is_v:
                            valid_positions.append(ent)

            b_slices = [valid_positions[off:off + 32] for off in range(0, len(valid_positions), 32)]
            used_ids = pub["scoring_run_ids"][:len(b_slices)]
            built_b = []
            for b_idx, (b_sl, b_rid) in enumerate(zip(b_slices, used_ids)):
                b_raw = encoded({
                    "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
                    "mode": "matrix",
                    "run_id": b_rid,
                    "worker_source": worker_text,
                    "requests": [pos["request"] for pos in b_sl],
                })
                built_b.append({
                    "batch_index": b_idx,
                    "run_id": b_rid,
                    "bundle_raw": b_raw,
                    "bundle_sha256": _sha256(b_raw),
                    "worker_sha256": WORKER_SHA256,
                    "positions": b_sl,
                })

            time.sleep(3.0)
            p2_call = run_single_ssh_json_call(
                priv["phase_two_ssh_argv"],
                priv["ssh_environment"],
                encoded({
                    "kind": "LORA_MULTI_VALIDATION_BUNDLES",
                    "bundles_b64": [base64.b64encode(bm["bundle_raw"]).decode("ascii") for bm in built_b],
                }),
                cancellations,
                secondary_errors,
                timeout_ns=900_000_000_000,
            )
            if not (p2_call["reaped"] and p2_call["group_absent"] and p2_call["cleanup_bounded"]):
                p2_all_clean = False
            safe_write("phase2-multi-stdout.json", p2_call["stdout_bytes"])
            safe_write("phase2-multi-stderr.bin", p2_call["stderr_bytes"])
            if p2_call["status"] == "completed":
                env_out = parse_json(p2_call["stdout_bytes"].strip(), limit=MULTI_BUNDLE_STDOUT_LIMIT)
                require(env_out.get("kind") == "LORA_MULTI_VALIDATION_RESULTS", "p2 multi kind")
                pending_cont = []
                for bm, irec in zip(built_b, env_out["results"]):
                    b_idx = bm["batch_index"]
                    require(irec["batch_index"] == b_idx, "batch_index mismatch")
                    trans = irec["transport"]
                    safe_write("phase2-batch-%02d-bundle.json" % b_idx, bm["bundle_raw"])
                    safe_write("phase2-batch-%02d-stdout.jsonl" % b_idx, bytes.fromhex(trans["stdout_hex"]))
                    safe_write("phase2-batch-%02d-stderr.bin" % b_idx, bytes.fromhex(trans["stderr_hex"]))
                    prec = {"batch_index": b_idx, "bundle_raw": bm["bundle_raw"], "transport": trans}
                    final_recs.append(prec)
                    obs_m, f_ok, p_ok, _pe = _decode_scoring_batch(bm, prec, p1_cfg["expected_runtime"])
                    if (not f_ok) and p_ok:
                        pending_cont.append((b_idx, 1, bm["positions"][len(obs_m):]))

                while pending_cont and not cancellations and not persistence_errors:
                    time.sleep(3.0)
                    c_metas = [
                        build_confirmatory_continuation_bundle(worker_raw, run_id, bi, si, rsl)
                        for bi, si, rsl in pending_cont
                    ]
                    c_call = run_single_ssh_json_call(
                        priv["phase_two_ssh_argv"],
                        priv["ssh_environment"],
                        encoded({
                            "kind": "LORA_MULTI_VALIDATION_BUNDLES",
                            "bundles_b64": [base64.b64encode(cm["bundle_raw"]).decode("ascii") for cm in c_metas],
                        }),
                        cancellations,
                        secondary_errors,
                        timeout_ns=600_000_000_000,
                    )
                    if not (c_call["reaped"] and c_call["group_absent"] and c_call["cleanup_bounded"]):
                        p2_all_clean = False
                    if c_call["status"] != "completed":
                        p2_all_clean = False
                        break
                    c_env = parse_json(c_call["stdout_bytes"].strip(), limit=MULTI_BUNDLE_STDOUT_LIMIT)
                    require(c_env.get("kind") == "LORA_MULTI_VALIDATION_RESULTS", "cont multi kind")
                    nxt = []
                    for (bi, si, rsl), cm, cit in zip(pending_cont, c_metas, c_env["results"]):
                        ctrans = cit["transport"]
                        safe_write("phase2-batch-%02d-cont-%02d-bundle.json" % (bi, si), cm["bundle_raw"])
                        safe_write("phase2-batch-%02d-cont-%02d-stdout.jsonl" % (bi, si), bytes.fromhex(ctrans["stdout_hex"]))
                        safe_write("phase2-batch-%02d-cont-%02d-stderr.bin" % (bi, si), bytes.fromhex(ctrans["stderr_hex"]))
                        crec = {
                            "batch_index": bi,
                            "sub_index": si,
                            "run_id": cm["run_id"],
                            "position_indices": [pos["position_index"] for pos in rsl],
                            "bundle_raw": cm["bundle_raw"],
                            "transport": ctrans,
                        }
                        cont_recs.append(crec)
                        co, cf, cp, _cpr = _decode_scoring_batch(
                            cm,
                            crec,
                            p1_cfg["expected_runtime"],
                            check_fields=("batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"),
                        )
                        if (not cf) and cp:
                            nxt.append((bi, si + 1, rsl[len(co):]))
                    pending_cont = nxt
            else:
                p2_all_clean = False

            if not cancellations:
                report = score_gated_ablation(
                    raw_files,
                    worker_raw,
                    run_id,
                    body_obj["phase_one_records"],
                    body_obj["gate_decisions"],
                    body_obj["router_summary"],
                    used_ids,
                    final_recs,
                    cont_recs,
                    p1_cfg["expected_runtime"],
                    body_obj["training_summary"],
                    body_obj["shared_costs"],
                    g4b_phase_one_records,
                    g4b_report,
                    g5_phase_one_records,
                    g5_report,
                )
                safe_write("gated-ablation-report.json", _pretty_json(report))

        p1_clean = bool(p1_out and p1_out["reaped"] and p1_out["group_absent"] and p1_out["cleanup_bounded"])
        binding_ok = bool(p1_clean and p2_all_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_GATE_G6B_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and report and report["complete_and_verified"]),
            "secondary_errors": secondary_errors,
            **{flag_n: False for flag_n in FALSE_FLAGS},
        }
        safe_write("g6b-controller-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": report}
