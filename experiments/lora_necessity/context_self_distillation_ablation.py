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
    _load_supervision_only,
    _render_user_prompt,
    confirmatory_first_round,
    confirmatory_second_round,
    load_confirmatory_dataset,
)
from confirmatory_v2_converged_ablation import (
    ABLATION_SOURCE_NAMES,
    mounted_converged_adapter,
)
from confirmatory_v2_gate_a import (
    MULTI_BUNDLE_STDOUT_LIMIT,
    run_single_ssh_json_call,
)
from confirmatory_v2_gated_ablation import (
    RECAL_DEV_POS_IDS,
    RECAL_V2_R1_PROMPTS_SHA256,
    V1_BACKGROUND_IDS,
    V1_DEV_NEG_IDS,
    V1_PROMPTS_SHA256,
    V1_STRUCTURE_SHA256,
    _collect_scored_positions,
    _router_public_only,
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
from diff_focused_lora_ablation import (
    _map_tokens_to_normalized_weights,
    build_self_continuing_phase2_ssh_argv,
    classify_reference_line_weights,
)
from gated_lora_router import render_sanitized_router_prompt, sanitize_task_for_router
from kl_anchored_lora_ablation import (
    _compute_trajectory_kl_loss,
    audit_kl_restraint_on_trajectories,
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
from tokenizer_measurement import measure as measure_tokens
from validation_contract import digest, encoded, parse_json


V1_TRANSFER_IDS = tuple("T%02d" % idx_num for idx_num in range(1, 17))

CONTEXT_DISTILL_HYPERPARAMS = {
    "rank": 8,
    "alpha": 16,
    "targets": ["q_proj", "v_proj"],
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "lr_schedule": "linear_decay_to_10pct",
    "weight_decay": 0.01,
    "lambda_pos_kl": 1.0,
    "lambda_bg_kl": 2.0,
    "diff_token_weight": 4.0,
    "scaffold_token_weight": 1.0,
    "cosmetic_valueerror_weight": 0.25,
    "background_anchors_per_step": 2,
    "max_anchor_tokens": 128,
    "max_teacher_tokens": 256,
    "gradient_clip": 1.0,
    "seed": 42,
    "steps": 80,
}

G9_SOURCE_NAMES = frozenset(
    set(ABLATION_SOURCE_NAMES)
    | {
        "gated_lora_router.py",
        "confirmatory_v2_gated_ablation.py",
        "kl_anchored_lora_ablation.py",
        "diff_focused_lora_ablation.py",
        "context_self_distillation_ablation.py",
    }
)

G9_STDLIB_BOOTSTRAP = """import base64
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
    stage = p_path / ("g9-csd-abl-" + cfg["run_id"])
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
        import context_self_distillation_ablation as g9_mod
        body_res = g9_mod.execute_remote_g9_ablation_body(cfg, stage, bundle)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        boot_fail = {"phase": phase, "error_type": type(exc).__name__, "message": str(exc)[:512]}

    if out_fd != 1:
        sys.stdout.flush()
        sys.stderr.flush()
        d_out = (stage / "native.stdout").read_bytes()[:1048576]
        d_err = (stage / "native.stderr").read_bytes()[:1048576]

    res_payload = enc({
        "kind": "LORA_GATE_G9_CONTEXT_DISTILL_PHASE1_RESULT",
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


def _sha256(raw_bytes):
    return hashlib.sha256(raw_bytes).hexdigest()


def load_g9_distillation_pools(
    supervision_v2_raw,
    v1_prompts_raw,
    v1_structure_raw,
    recal_prompts_raw,
):
    sup_list = _load_supervision_only(supervision_v2_raw)
    require(
        type(v1_prompts_raw) is bytes and _sha256(v1_prompts_raw) == V1_PROMPTS_SHA256,
        "prompts_v1.json pin mismatch",
    )
    require(
        type(v1_structure_raw) is bytes and _sha256(v1_structure_raw) == V1_STRUCTURE_SHA256,
        "structure_v1.json pin mismatch",
    )
    require(
        type(recal_prompts_raw) is bytes and _sha256(recal_prompts_raw) == RECAL_V2_R1_PROMPTS_SHA256,
        "recalibration_v2_r1/prompts.json pin mismatch",
    )

    v1_prompts = parse_json(v1_prompts_raw)
    v1_struct = parse_json(v1_structure_raw)
    recal_items = parse_json(recal_prompts_raw)
    require(tuple(item["id"] for item in recal_items) == RECAL_DEV_POS_IDS, "C01..C12 order")

    v1_split_by_id = {item["id"]: item["split"] for item in v1_struct}
    v1_by_id = {item["id"]: item for item in v1_prompts}
    for tr_id in V1_TRANSFER_IDS:
        require(v1_split_by_id.get(tr_id) == "transfer" and tr_id in v1_by_id, "missing v1 transfer")
    for bg_id in V1_BACKGROUND_IDS:
        require(v1_split_by_id.get(bg_id) == "retention" and bg_id in v1_by_id, "missing v1 retention")
    for dn_id in V1_DEV_NEG_IDS:
        require(v1_split_by_id.get(dn_id) == "development" and dn_id in v1_by_id, "missing v1 dev")

    seed_map = {
        item["id"]: {
            **_router_public_only(item),
            "initial_witness": copy.deepcopy(item["initial_witness"]),
            "reference_source": item["reference_source"],
        }
        for item in sup_list
    }
    unlabeled_recal = {item["id"]: _router_public_only(item) for item in recal_items}
    unlabeled_v1_tr = {tr_id: _router_public_only(v1_by_id[tr_id]) for tr_id in V1_TRANSFER_IDS}
    bg_map = {bg_id: _router_public_only(v1_by_id[bg_id]) for bg_id in V1_BACKGROUND_IDS}
    dev_neg_map = {dn_id: _router_public_only(v1_by_id[dn_id]) for dn_id in V1_DEV_NEG_IDS}

    seen_entries = set()
    for pool_map in (seed_map, unlabeled_recal, unlabeled_v1_tr, bg_map, dev_neg_map):
        for task_obj in pool_map.values():
            ep_name = task_obj["entry_point"]
            require(ep_name not in seen_entries, "duplicate entry_point across G9 pools: " + ep_name)
            seen_entries.add(ep_name)

    return {
        "seed_supervision": seed_map,
        "unlabeled_recal": unlabeled_recal,
        "unlabeled_v1_transfer": unlabeled_v1_tr,
        "background": bg_map,
        "dev_negative": dev_neg_map,
    }


def build_teacher_context_messages(task_record, supervision_raw, *, exclude_task_id=None):
    sup_list = _load_supervision_only(supervision_raw)
    if exclude_task_id is not None:
        filtered_examples = [item for item in sup_list if item["id"] != exclude_task_id]
        require(len(filtered_examples) == 7, "leave-one-out Teacher requires 7 examples")
    else:
        filtered_examples = sup_list
        require(len(filtered_examples) == 8, "full Teacher requires 8 examples")
    sanitized_query = sanitize_task_for_router(task_record)
    return _render_user_prompt(sanitized_query, examples_list=filtered_examples, witness_obj=None)


def weighted_slice_kl_loss(student_logits_slice, teacher_logits_bf16, target_weights, device):
    import torch
    from torch.nn import functional

    s_float = student_logits_slice.to(device=device, dtype=torch.float32)
    t_float = teacher_logits_bf16.to(device=device, dtype=torch.float32)
    require(
        s_float.ndim == 2 and tuple(s_float.shape) == tuple(t_float.shape),
        "student/teacher slice shape mismatch",
    )
    log_p_teacher = functional.log_softmax(t_float, dim=-1)
    log_p_student = functional.log_softmax(s_float, dim=-1)
    per_tok_kl = functional.kl_div(log_p_student, log_p_teacher, log_target=True, reduction="none").sum(dim=-1)
    wt_tensor = torch.tensor(target_weights, dtype=torch.float32, device=device)
    require(
        wt_tensor.ndim == 1 and wt_tensor.shape[0] == per_tok_kl.shape[0],
        "KL target_weights length mismatch",
    )
    return (per_tok_kl * wt_tensor).mean()


def _fast_cached_greedy_decode_and_exact_logits(
    model,
    tokenizer,
    messages,
    *,
    max_new_tokens,
    device,
):
    import torch

    measured = measure_tokens(tokenizer, messages, answer=None)
    require(
        measured["status"] == "OK" and measured["input"]["input_tokens"] <= 4096,
        "Teacher/anchor prompt exceeded 4096 tokens",
    )
    prompt_ids = list(measured["input"]["input_ids"])
    prompt_len = len(prompt_ids)
    eos_id = int(tokenizer.eos_token_id)

    rng_devices = [device.index or 0] if device.type == "cuda" else []
    orig_use_cache = getattr(model.config, "use_cache", False)
    gen_ids = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        try:
            model.config.use_cache = True
            inp_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            out_obj = model(
                input_ids=inp_t,
                attention_mask=torch.ones_like(inp_t),
                use_cache=True,
                return_dict=True,
            )
            past_kv = getattr(out_obj, "past_key_values", None)
            next_tok = int(torch.argmax(out_obj.logits[0, -1]).item())
            gen_ids.append(next_tok)
            del out_obj, inp_t

            while next_tok != eos_id and len(gen_ids) < max_new_tokens:
                if past_kv is not None:
                    step_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                    step_out = model(
                        input_ids=step_in,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True,
                    )
                    past_kv = getattr(step_out, "past_key_values", None)
                    next_tok = int(torch.argmax(step_out.logits[0, -1]).item())
                    del step_out, step_in
                else:
                    full_in = torch.tensor([prompt_ids + gen_ids], dtype=torch.long, device=device)
                    step_out = model(
                        input_ids=full_in,
                        attention_mask=torch.ones_like(full_in),
                        use_cache=False,
                        return_dict=True,
                    )
                    next_tok = int(torch.argmax(step_out.logits[0, -1]).item())
                    del step_out, full_in
                gen_ids.append(next_tok)
            del past_kv
        finally:
            model.config.use_cache = orig_use_cache

        full_seq = prompt_ids + gen_ids
        ans_len = len(gen_ids)
        exact_in = torch.tensor([full_seq[:-1]], dtype=torch.long, device=device)
        exact_out = model(
            input_ids=exact_in,
            attention_mask=torch.ones_like(exact_in),
            use_cache=False,
            return_dict=True,
        )
        exact_slice_cpu = (
            exact_out.logits[0, prompt_len - 1:prompt_len - 1 + ans_len]
            .detach()
            .to(dtype=torch.bfloat16, device="cpu")
            .clone()
        )
        del exact_out, exact_in

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stop_eos = gen_ids[-1] == eos_id
    decode_ids = gen_ids[:-1] if stop_eos else list(gen_ids)
    candidate_text = tokenizer.decode(
        decode_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {
        "prompt_ids": prompt_ids,
        "gen_ids": gen_ids,
        "stop_eos": stop_eos,
        "candidate_text": candidate_text,
        "answer_logits_bf16": exact_slice_cpu,
    }


def prepare_fast_background_trajectories(model, tokenizer, task_map, *, max_new_tokens=128):
    params_list = list(model.parameters())
    device = next(iter({par_item.device for par_item in params_list}))
    trajectories = []
    for tid in sorted(task_map.keys()):
        msgs = render_sanitized_router_prompt(task_map[tid])
        decoded = _fast_cached_greedy_decode_and_exact_logits(
            model,
            tokenizer,
            msgs,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        p_ids = decoded["prompt_ids"]
        g_ids = decoded["gen_ids"]
        trajectories.append({
            "task_id": tid,
            "full_ids": p_ids + g_ids,
            "prompt_len": len(p_ids),
            "answer_len": len(g_ids),
            "base_answer_logits_bf16": decoded["answer_logits_bf16"],
        })
    return trajectories


def prepare_positive_distillation_trajectories(
    model,
    tokenizer,
    pools,
    supervision_raw,
    resolved_interpreter,
    backend_raw,
    worker_text,
    expected_runtime,
    run_id,
    *,
    hyperparams=None,
):
    import torch
    from recalibration_diagnostic_remote import _run_public_cases_child

    cfg_hp = copy.deepcopy(CONTEXT_DISTILL_HYPERPARAMS if hyperparams is None else hyperparams)
    params_list = list(model.parameters())
    device = next(iter({par_item.device for par_item in params_list}))
    rng_devices = [device.index or 0] if device.type == "cuda" else []

    seed_trajs = []
    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        for tid in SUPERVISION_IDS:
            seed_item = pools["seed_supervision"][tid]
            ans_text = seed_item["reference_source"]
            teacher_msgs = build_teacher_context_messages(
                seed_item, supervision_raw, exclude_task_id=tid
            )
            student_msgs = render_sanitized_router_prompt(seed_item)

            t_meas = measure_tokens(tokenizer, teacher_msgs, answer=ans_text)
            s_meas = measure_tokens(tokenizer, student_msgs, answer=ans_text)
            require(
                t_meas["status"] == "OK" and s_meas["status"] == "OK",
                "seed teacher/student token budget exceeded",
            )
            t_prompt_ids = list(t_meas["input"]["input_ids"])
            t_target_ids = list(t_meas["target"]["target_ids"])
            s_prompt_ids = list(s_meas["input"]["input_ids"])
            s_target_ids = list(s_meas["target"]["target_ids"])
            require(t_target_ids == s_target_ids, "teacher/student target_ids mismatch")

            ans_len = len(t_target_ids)
            t_full = t_prompt_ids + t_target_ids
            t_in = torch.tensor([t_full[:-1]], dtype=torch.long, device=device)
            t_out = model(
                input_ids=t_in,
                attention_mask=torch.ones_like(t_in),
                use_cache=False,
                return_dict=True,
            )
            t_slice_cpu = (
                t_out.logits[0, len(t_prompt_ids) - 1:len(t_prompt_ids) - 1 + ans_len]
                .detach()
                .to(dtype=torch.bfloat16, device="cpu")
                .clone()
            )
            del t_out, t_in

            ln_list, ln_wts = classify_reference_line_weights(
                seed_item["initial_source"],
                ans_text,
                diff_weight=float(cfg_hp["diff_token_weight"]),
                scaffold_weight=float(cfg_hp["scaffold_token_weight"]),
                cosmetic_weight=float(cfg_hp["cosmetic_valueerror_weight"]),
            )
            norm_wts = _map_tokens_to_normalized_weights(
                tokenizer,
                ans_text,
                s_target_ids,
                ln_list,
                ln_wts,
                scaffold_weight=float(cfg_hp["scaffold_token_weight"]),
            )
            seed_trajs.append({
                "task_id": tid,
                "source_pool": "seed",
                "student_full_ids": s_prompt_ids + s_target_ids,
                "student_prompt_len": len(s_prompt_ids),
                "answer_len": ans_len,
                "target_weights": norm_wts,
                "teacher_answer_logits_bf16": t_slice_cpu,
            })

    unlabeled_candidates = []
    for tid in RECAL_DEV_POS_IDS:
        unlabeled_candidates.append(("recal", tid, pools["unlabeled_recal"][tid]))
    for tid in V1_TRANSFER_IDS:
        unlabeled_candidates.append(("v1_transfer", tid, pools["unlabeled_v1_transfer"][tid]))

    verified_unlabeled = []
    unlabeled_stats = {"attempted": len(unlabeled_candidates), "valid_syntax": 0, "public_both_pass": 0, "accepted_ids": []}

    for u_idx, (pool_tag, tid, u_task) in enumerate(unlabeled_candidates):
        teacher_msgs = build_teacher_context_messages(
            u_task, supervision_raw, exclude_task_id=None
        )
        decoded = _fast_cached_greedy_decode_and_exact_logits(
            model,
            tokenizer,
            teacher_msgs,
            max_new_tokens=int(cfg_hp["max_teacher_tokens"]),
            device=device,
        )
        cand_code = decoded["candidate_text"]
        if not (decoded["stop_eos"] and candidate_valid(u_task, cand_code)):
            continue
        unlabeled_stats["valid_syntax"] += 1

        pub_fb, _pub_ns = _run_public_cases_child(
            resolved_interpreter,
            backend_raw,
            worker_text,
            expected_runtime,
            run_id,
            100 + u_idx,
            u_task,
            cand_code,
        )
        if pub_fb.get("kind") != "visible_tests":
            continue
        if not all(case_it["status"] == "pass" for case_it in pub_fb["cases"]):
            continue

        student_msgs = render_sanitized_router_prompt(u_task)
        s_meas = measure_tokens(tokenizer, student_msgs, answer=cand_code)
        if s_meas["status"] != "OK" or s_meas["target"] is None:
            continue
        s_prompt_ids = list(s_meas["input"]["input_ids"])
        s_target_ids = list(s_meas["target"]["target_ids"])
        if s_target_ids != decoded["gen_ids"]:
            continue

        ln_list, ln_wts = classify_reference_line_weights(
            u_task["initial_source"],
            cand_code,
            diff_weight=float(cfg_hp["diff_token_weight"]),
            scaffold_weight=float(cfg_hp["scaffold_token_weight"]),
            cosmetic_weight=float(cfg_hp["cosmetic_valueerror_weight"]),
        )
        norm_wts = _map_tokens_to_normalized_weights(
            tokenizer,
            cand_code,
            s_target_ids,
            ln_list,
            ln_wts,
            scaffold_weight=float(cfg_hp["scaffold_token_weight"]),
        )
        unlabeled_stats["public_both_pass"] += 1
        unlabeled_stats["accepted_ids"].append(tid)
        verified_unlabeled.append({
            "task_id": tid,
            "source_pool": "unlabeled_" + pool_tag,
            "student_full_ids": s_prompt_ids + s_target_ids,
            "student_prompt_len": len(s_prompt_ids),
            "answer_len": len(s_target_ids),
            "target_weights": norm_wts,
            "teacher_answer_logits_bf16": decoded["answer_logits_bf16"],
        })

    return seed_trajs + verified_unlabeled, unlabeled_stats


def train_context_self_distilled_adapter(
    model,
    positive_trajectories,
    background_trajectories,
    *,
    hyperparams=None,
):
    import torch
    from torch import nn
    from training_worker import (
        LoRAProjection,
        _adapter_state,
        _finite,
        _optimizer_parameters,
        _state,
        _synchronize,
        _targets,
        _tensor_digest,
    )

    cfg_hp = copy.deepcopy(CONTEXT_DISTILL_HYPERPARAMS if hyperparams is None else hyperparams)
    started = time.perf_counter_ns()
    require(
        torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
        "autograd required",
    )
    require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "disabled autocast required",
    )
    require(
        isinstance(positive_trajectories, list) and len(positive_trajectories) >= 8,
        "at least 8 positive distillation trajectories required",
    )
    require(
        isinstance(background_trajectories, list) and len(background_trajectories) == 16,
        "16 background anchor trajectories required",
    )
    require(isinstance(model, nn.Module), "nn.Module required")
    base_state = _state(model)
    devices = {ten.device for ten in base_state.values()}
    require(len(devices) == 1, "single device required")
    device = next(iter(devices))
    require(
        all(par.dtype == torch.bfloat16 and par.grad is None for par in model.parameters()),
        "BF16 base without grad",
    )
    _finite(base_state.values())
    targets = _targets(model)
    base_sha256 = _tensor_digest(base_state)

    seed_trajs = [tr_it for tr_it in positive_trajectories if tr_it["source_pool"] == "seed"]
    extra_trajs = [tr_it for tr_it in positive_trajectories if tr_it["source_pool"] != "seed"]
    require(len(seed_trajs) == 8, "8 seed distillation trajectories required")

    modes = [(mod_item, mod_item.training) for mod_item in model.modules()]
    grad_flags = [(par_item, par_item.requires_grad) for par_item in model.parameters()]
    wrappers = []
    originals = []
    steps_log = []
    initial_adapter = None
    final_adapter = None
    peak_alloc = None
    peak_resv = None

    rng_devices = [device.index or 0] if device.type == "cuda" else []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.random.fork_rng(devices=rng_devices):
        torch.default_generator.manual_seed(int(cfg_hp["seed"]))
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(cfg_hp["seed"])
        )
        try:
            model.requires_grad_(False)
            model.eval()
            for path_str, base_lin in targets:
                parent_path, attr_name = path_str.rsplit(".", 1)
                parent_mod = model.get_submodule(parent_path)
                wrap_mod = LoRAProjection(base_lin)
                originals.append((parent_mod, attr_name, base_lin))
                setattr(parent_mod, attr_name, wrap_mod)
                wrappers.append((path_str, wrap_mod))

            trainable_params = [
                par_obj
                for _p, wrap_mod in wrappers
                for par_obj in (wrap_mod.adapter_a, wrap_mod.adapter_b)
            ]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=float(cfg_hp["learning_rate"]),
                weight_decay=float(cfg_hp["weight_decay"]),
            )
            _optimizer_parameters(optimizer, trainable_params)
            initial_adapter = _adapter_state(wrappers)
            _synchronize(device)
            train_t0 = time.perf_counter_ns()
            prep_ns = train_t0 - started

            total_steps = int(cfg_hp["steps"])
            base_lr = float(cfg_hp["learning_rate"])
            lambda_pos = float(cfg_hp["lambda_pos_kl"])
            lambda_bg = float(cfg_hp["lambda_bg_kl"])
            bg_per_step = int(cfg_hp["background_anchors_per_step"])

            seed_cursor = 0
            extra_cursor = 0
            for step_idx in range(total_steps):
                frac = step_idx / float(max(total_steps - 1, 1))
                cur_lr = base_lr * (1.0 - 0.9 * frac)
                for p_grp in optimizer.param_groups:
                    p_grp["lr"] = cur_lr

                if extra_trajs and (step_idx % 2 == 1):
                    pos_item = extra_trajs[extra_cursor % len(extra_trajs)]
                    extra_cursor += 1
                else:
                    pos_item = seed_trajs[seed_cursor % len(seed_trajs)]
                    seed_cursor += 1

                step_t0 = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)

                s_feed = pos_item["student_full_ids"][:-1]
                s_plen = int(pos_item["student_prompt_len"])
                s_alen = int(pos_item["answer_len"])
                inp_pos = torch.tensor([s_feed], dtype=torch.long, device=device)
                out_pos = model(
                    input_ids=inp_pos,
                    attention_mask=torch.ones_like(inp_pos),
                    use_cache=False,
                    return_dict=True,
                )
                s_slice = out_pos.logits[0, s_plen - 1:s_plen - 1 + s_alen]
                pos_kl_t = weighted_slice_kl_loss(
                    s_slice,
                    pos_item["teacher_answer_logits_bf16"],
                    pos_item["target_weights"],
                    device,
                )
                _finite([pos_kl_t])
                (lambda_pos * pos_kl_t).backward()
                pos_kl_val = float(pos_kl_t.detach().cpu().item())
                del out_pos, inp_pos, s_slice, pos_kl_t

                kl_vals = []
                for sub_idx in range(bg_per_step):
                    bg_item = background_trajectories[(step_idx * bg_per_step + sub_idx) % 16]
                    bg_kl_t, _matched, _alen = _compute_trajectory_kl_loss(model, bg_item, device)
                    _finite([bg_kl_t])
                    kl_vals.append(float(bg_kl_t.detach().cpu().item()))
                    if lambda_bg > 0.0:
                        scaled_bg = (lambda_bg / float(bg_per_step)) * bg_kl_t
                        scaled_bg.backward()
                    del bg_kl_t

                require(all(par_obj.grad is not None for par_obj in trainable_params), "missing grad")
                _finite(par_obj.grad for par_obj in trainable_params)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, float(cfg_hp["gradient_clip"]), error_if_nonfinite=True
                )
                _finite([grad_norm])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _finite(trainable_params)
                _synchronize(device)

                mean_bg_kl = sum(kl_vals) / float(len(kl_vals))
                steps_log.append({
                    "step": step_idx + 1,
                    "task_id": pos_item["task_id"],
                    "source_pool": pos_item["source_pool"],
                    "lr": cur_lr,
                    "pos_distill_kl": pos_kl_val,
                    "bg_kl_loss": mean_bg_kl,
                    "gradient_norm": float(grad_norm.detach().cpu().item()),
                    "elapsed_ns": time.perf_counter_ns() - step_t0,
                })

            train_ns = time.perf_counter_ns() - train_t0
            final_adapter = _adapter_state(wrappers)
            _finite(final_adapter.values())
            if device.type == "cuda":
                peak_alloc = int(torch.cuda.max_memory_allocated(device))
                peak_resv = int(torch.cuda.max_memory_reserved(device))
        finally:
            for parent_mod, attr_name, orig_lin in reversed(originals):
                setattr(parent_mod, attr_name, orig_lin)
            for par_item, req_g in grad_flags:
                par_item.grad = None
                par_item.requires_grad_(req_g)
            for mod_item, was_t in modes:
                mod_item.training = was_t

    ver_t0 = time.perf_counter_ns()
    require(_tensor_digest(_state(model)) == base_sha256, "base weights mutated during G9 training")
    init_sha = _tensor_digest(initial_adapter)
    final_sha = _tensor_digest(final_adapter)
    require(init_sha != final_sha, "G9 adapter did not update")
    adapter_bytes = sum(ten.numel() * ten.element_size() for ten in final_adapter.values())
    ver_ns = time.perf_counter_ns() - ver_t0

    block_size = 8
    num_blocks = len(steps_log) // block_size
    block_pos_means = [
        sum(st_r["pos_distill_kl"] for st_r in steps_log[bl_i * block_size:(bl_i + 1) * block_size]) / float(block_size)
        for bl_i in range(num_blocks)
    ]
    block_bg_means = [
        sum(st_r["bg_kl_loss"] for st_r in steps_log[bl_i * block_size:(bl_i + 1) * block_size]) / float(block_size)
        for bl_i in range(num_blocks)
    ]

    return {
        "kind": "LORA_CONTEXT_SELF_DISTILLATION_TRAINING_RESULT",
        "hyperparams": copy.deepcopy(cfg_hp),
        "positive_trajectory_count": len(positive_trajectories),
        "seed_trajectory_count": len(seed_trajs),
        "verified_unlabeled_trajectory_count": len(extra_trajs),
        "base_sha256": base_sha256,
        "initial_adapter_sha256": init_sha,
        "adapter_sha256": final_sha,
        "adapter_bytes": adapter_bytes,
        "preparation_ns": prep_ns,
        "training_ns": train_ns,
        "verification_ns": ver_ns,
        "core_wall_ns": time.perf_counter_ns() - started,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_resv,
        "block_mean_pos_distill_kl": block_pos_means,
        "block_mean_bg_kl": block_bg_means,
        "steps": steps_log,
        "initial_adapter": initial_adapter,
        "adapter": final_adapter,
    }


def build_context_distill_schedule(prompts_raw, supervision_raw, *, run_id):
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
                    encoded(["g9_csd_call", run_id, ep_idx, canary_tid, "base_canary", role_name])
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
            "arm": "lora_csd",
            "render_arm": "lora",
            "first_messages": f_msgs,
            "first_messages_sha256": digest(f_msgs),
            "call_ids": {
                role_name: _sha256(
                    encoded(["g9_csd_call", run_id, ep_idx, task["id"], "lora_csd", role_name])
                )[:32]
                for role_name in CONFIRMATORY_ROLES
            },
        })

    require(len(episodes) == 26, "26 episodes required")
    return {
        "kind": "LORA_CONFIRMATORY_V2_CONTEXT_DISTILL_SCHEDULE",
        "version": 1,
        "run_id": run_id,
        "scheduled_episodes": 26,
        "scheduled_generation_calls": 52,
        "episodes": episodes,
    }


def build_gate_g9_ablation_package(
    raw_files,
    v1_prompts_raw,
    v1_structure_raw,
    recal_prompts_raw,
    worker_raw,
    backend_raw,
    dep_sources,
    preflight_raw,
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
    hyperparams=None,
    work_timeout_seconds=7200,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    chosen_hp = copy.deepcopy(CONTEXT_DISTILL_HYPERPARAMS if hyperparams is None else hyperparams)
    load_confirmatory_dataset(raw_files)
    pools = load_g9_distillation_pools(
        raw_files["supervision.json"], v1_prompts_raw, v1_structure_raw, recal_prompts_raw
    )
    pools_raw = encoded(pools)
    require(_sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(_sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    require(set(dep_sources) == G9_SOURCE_NAMES, "G9 source names")
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    require(len(scoring_run_ids) == 13, "13 scoring_run_ids")
    seen_ids = {run_id}
    for s_rid in scoring_run_ids:
        _check_hex(s_rid, 32)
        require(s_rid not in seen_ids, "duplicate scoring_run_id")
        seen_ids.add(s_rid)

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    sources_sha = {fname: _sha256(dep_sources[fname]) for fname in sorted(G9_SOURCE_NAMES)}
    payload_raw = encoded({
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(G9_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision.json"]).decode("ascii"),
        "pools_b64": base64.b64encode(pools_raw).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    })

    boot_bytes = G9_STDLIB_BOOTSTRAP.encode("ascii")
    boot_sha = _sha256(boot_bytes)
    p1_cfg = {
        "kind": "LORA_GATE_G9_CONTEXT_DISTILL_PHASE1_CONFIG",
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
        "stage": str(Path(remote_parent) / ("g9-csd-abl-" + run_id)),
        "run_id": run_id,
        "hyperparams": copy.deepcopy(chosen_hp),
        "sources_sha256": sources_sha,
        "prompts_sha256": CONFIRMATORY_FILE_PINS["prompts.json"],
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "pools_sha256": _sha256(pools_raw),
        "backend_sha256": BACKEND_SHA256,
        "worker_sha256": WORKER_SHA256,
        "payload_sha256": _sha256(payload_raw),
        "payload_bytes": len(payload_raw),
        "bootstrap_sha256": boot_sha,
        "work_timeout_seconds": work_timeout_seconds,
    }
    p1_cfg_raw = encoded(p1_cfg)
    p1_cfg_sha = _sha256(p1_cfg_raw)

    boot_b64 = base64.b64encode(boot_bytes).decode("ascii")
    cfg_b64 = base64.b64encode(p1_cfg_raw).decode("ascii")
    remote_prog = (
        "import base64,hashlib,json,sys\n"
        f"src=base64.b64decode({boot_b64!r},validate=True)\n"
        f"cfg=json.loads(base64.b64decode({cfg_b64!r},validate=True).decode('ascii'))\n"
        "if hashlib.sha256(src).hexdigest()!=cfg['bootstrap_sha256']:\n"
        "    sys.exit(78)\n"
        "ns={'__name__':'g9_boot'}\n"
        "exec(compile(src,'<g9_stdlib_bootstrap>','exec'),ns)\n"
        "sys.exit(ns['main_boot'](cfg))\n"
    )
    p1_remote_cmd = "exec " + shlex.join((
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
    require(len(p1_remote_cmd.encode("utf-8")) <= 65536, "SSH command size exceeds cap")
    p1_ssh_argv = [*ssh_argv_prefix, p1_remote_cmd]
    p2_ssh_argv = build_self_continuing_phase2_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )

    manifest = {
        "kind": "LORA_GATE_G9_CONTEXT_DISTILL_LAUNCH_PACKAGE",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
        "scoring_run_ids": list(scoring_run_ids),
        "hyperparams": copy.deepcopy(chosen_hp),
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


def execute_remote_g9_ablation_body(config, stage, bundle):
    from observed_generation import generate_observed
    from recalibration_diagnostic_remote import (
        OFFLINE_GPU_ENVIRONMENT,
        _initialize_cuda_and_load_model,
        _query_and_select_gpu,
        _run_public_cases_child,
        _verify_checkpoint_files,
    )

    prompts_raw = base64.b64decode(bundle["prompts_b64"], validate=True)
    supervision_raw = base64.b64decode(bundle["supervision_b64"], validate=True)
    pools_raw = base64.b64decode(bundle["pools_b64"], validate=True)
    backend_raw = base64.b64decode(bundle["backend_b64"], validate=True)
    worker_raw = base64.b64decode(bundle["worker_b64"], validate=True)
    require(
        _sha256(prompts_raw) == config["prompts_sha256"]
        and _sha256(supervision_raw) == config["supervision_sha256"]
        and _sha256(pools_raw) == config["pools_sha256"]
        and _sha256(backend_raw) == config["backend_sha256"]
        and _sha256(worker_raw) == config["worker_sha256"],
        "G9 input pin mismatch",
    )
    pools = parse_json(pools_raw)
    worker_text = worker_raw.decode("utf-8")

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

    active_hp = copy.deepcopy(config.get("hyperparams") or CONTEXT_DISTILL_HYPERPARAMS)
    anc_t0 = time.monotonic_ns()
    bg_trajs = prepare_fast_background_trajectories(
        model,
        tokenizer,
        pools["background"],
        max_new_tokens=int(active_hp["max_anchor_tokens"]),
    )
    dev_neg_trajs = prepare_fast_background_trajectories(
        model,
        tokenizer,
        pools["dev_negative"],
        max_new_tokens=int(active_hp["max_anchor_tokens"]),
    )
    shared_costs["fast_anchor_trajectory_prep_ns"] = time.monotonic_ns() - anc_t0

    pos_t0 = time.monotonic_ns()
    pos_trajs, unlabeled_stats = prepare_positive_distillation_trajectories(
        model,
        tokenizer,
        pools,
        supervision_raw,
        config["resolved_interpreter"],
        backend_raw,
        worker_text,
        config["expected_runtime"],
        config["run_id"],
        hyperparams=active_hp,
    )
    shared_costs["teacher_distill_prep_ns"] = time.monotonic_ns() - pos_t0

    train_out = train_context_self_distilled_adapter(
        model, pos_trajs, bg_trajs, hyperparams=active_hp
    )
    del pos_trajs, bg_trajs

    buf_io = io.BytesIO()
    save_t0 = time.perf_counter_ns()
    torch.save({"adapter": train_out["adapter"]}, buf_io)
    art_raw = buf_io.getvalue()
    write_private(stage / "context_self_distilled_adapter.pt", art_raw)
    save_ns = time.perf_counter_ns() - save_t0
    load_t0 = time.perf_counter_ns()
    reloaded = torch.load(
        io.BytesIO((stage / "context_self_distilled_adapter.pt").read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    load_ns = time.perf_counter_ns() - load_t0
    require(
        tensor_digest_fn(reloaded["adapter"]) == train_out["adapter_sha256"],
        "reloaded G9 adapter digest mismatch",
    )
    train_out["adapter"] = reloaded["adapter"]

    training_summary = {
        "hyperparams": train_out["hyperparams"],
        "positive_trajectory_count": train_out["positive_trajectory_count"],
        "seed_trajectory_count": train_out["seed_trajectory_count"],
        "verified_unlabeled_trajectory_count": train_out["verified_unlabeled_trajectory_count"],
        "unlabeled_teacher_mining": unlabeled_stats,
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
        "block_mean_pos_distill_kl": train_out["block_mean_pos_distill_kl"],
        "block_mean_bg_kl": train_out["block_mean_bg_kl"],
        "steps": train_out["steps"],
    }

    prompts_map = {item["id"]: item for item in parse_json(prompts_raw)}
    schedule = build_context_distill_schedule(prompts_raw, supervision_raw, run_id=config["run_id"])
    gen_dir = stage / "generations"
    gen_dir.mkdir(mode=0o700)
    phase_one_records = []
    heldout_dev_restraint_audit = None

    p1_t0 = time.monotonic_ns()
    mount_stack = ExitStack()
    mounted_flag = False
    active_model = model
    try:
        for ep_spec in schedule["episodes"]:
            ep_idx = ep_spec["episode_index"]
            tid = ep_spec["task_id"]
            arm = ep_spec["arm"]
            render_arm = ep_spec["render_arm"]
            task_prompt = prompts_map[tid]
            ep_dir = gen_dir / ("ep-%02d" % ep_idx)
            ep_dir.mkdir(mode=0o700)
            init_dir = ep_dir / "initial"
            init_dir.mkdir(mode=0o700)
            fb_dir = ep_dir / "feedback"
            fb_dir.mkdir(mode=0o700)

            want_lora = arm == "lora_csd"
            if want_lora and not mounted_flag:
                active_model = mount_stack.enter_context(mounted_converged_adapter(model, train_out))
                mounted_flag = True
                audit_t0 = time.monotonic_ns()
                heldout_dev_restraint_audit = audit_kl_restraint_on_trajectories(
                    active_model, dev_neg_trajs
                )
                shared_costs["heldout_dev_audit_ns"] = time.monotonic_ns() - audit_t0
                del dev_neg_trajs
            elif (not want_lora) and mounted_flag:
                mount_stack.close()
                mount_stack = ExitStack()
                active_model = model
                mounted_flag = False

            init_out = generate_observed(
                active_model,
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
                task_prompt, render_arm, supervision_raw, init_cand, pub_fb
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
                    active_model,
                    tokenizer,
                    sec_msgs,
                    expected_tokenizer_sha256=observed_tok_sha,
                    call_id=ep_spec["call_ids"]["feedback"],
                    artifact_dir=fb_dir,
                )
                fb_gen = fb_out["generation"]
                fb_cand = fb_gen["candidate"]
                fb_cost = _cost_projection(fb_gen)

            phase_one_records.append({
                "episode_index": ep_idx,
                "task_id": tid,
                "split": ep_spec["split"],
                "arm": arm,
                "candidates": {"initial": init_cand, "feedback": fb_cand},
                "public_feedback": pub_fb,
                "generation_costs": {"initial": _cost_projection(init_gen), "feedback": fb_cost},
                "public_execution_ns": pub_ns,
            })
    finally:
        mount_stack.close()

    shared_costs["episodes_wall_ns"] = time.monotonic_ns() - p1_t0
    fin_t0 = time.monotonic_ns()
    final_base_sha = tensor_digest_fn(model_state_fn(model))
    final_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
    rng_same = bool(
        torch.equal(init_cpu_rng, torch.random.get_rng_state())
        and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
    )
    if final_base_sha != init_base_sha or final_tok_sha != observed_tok_sha or not rng_same:
        raise RuntimeError("G9 post-run integrity drift")
    shared_costs["final_integrity_ns"] = time.monotonic_ns() - fin_t0

    return {
        "gpu_selected": gpu_selected,
        "integrity": {
            "initial_base_sha256": init_base_sha,
            "final_base_sha256": final_base_sha,
            "tokenizer_state_sha256": final_tok_sha,
            "rng_unchanged": rng_same,
        },
        "heldout_dev_restraint_audit": heldout_dev_restraint_audit,
        "training_summary": training_summary,
        "shared_costs": shared_costs,
        "phase_one_records": phase_one_records,
    }


def score_context_distill_ablation(
    raw_files,
    worker_raw,
    run_id,
    phase_one_records,
    scoring_run_ids,
    final_bundle_records,
    continuation_records,
    expected_runtime,
    training_summary,
    heldout_dev_restraint_audit,
    shared_costs,
    g4b_phase_one_records,
    g4b_report,
    g5_episodes,
    g8_report,
):
    dataset = load_confirmatory_dataset(raw_files)
    expected_sched = build_context_distill_schedule(
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
                        "request_id": "g9-p2-%04d" % len(valid_positions),
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
    g4b_eps = {(ep["task_id"], ep["arm"]): ep for ep in g4b_report["episodes"]}
    g5_eps = {(ep["task_id"], "lora_conv"): ep for ep in g5_episodes}
    g8_eps = {(ep["task_id"], "lora_diff_kl"): ep for ep in g8_report["episodes"]}

    canary_matches = {}
    csd_eps = {}
    csd_wall_ns = 0
    changed_vs_base_transfer = 0
    changed_vs_base_retention = 0

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
        elif arm == "lora_csd":
            if ep_rec["candidates"]["initial"] != g4b_cands[(tid, "base")]["initial"]:
                if ep_rec["split"] == "transfer":
                    changed_vs_base_transfer += 1
                else:
                    changed_vs_base_retention += 1
            csd_wall_ns += (
                ep_rec["generation_costs"]["initial"]["wall_ns"]
                + ep_rec["generation_costs"]["feedback"]["wall_ns"]
            )
            csd_eps[(tid, "lora_csd")] = {
                "task_id": tid,
                "split": ep_rec["split"],
                "arm": "lora_csd",
                "roles": roles_out,
            }

    canaries_ok = (
        set(canary_matches) == {"WT01", "WR01"}
        and all(canary_matches.values())
        and len(csd_eps) == 24
    )
    run_ok = bool(not proto_bad and not inc_reasons and canaries_ok)
    merged_indexed = {**g4b_eps, **g5_eps, **g8_eps, **csd_eps}

    by_split = {}
    comparisons = {}
    for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
        by_split[sp_name] = {
            role_name: {
                st_n: sum(
                    (tid, "lora_csd") in csd_eps
                    and csd_eps[(tid, "lora_csd")]["roles"][role_name]["state"] == st_n
                    for tid in sp_ids
                )
                for st_n in CANDIDATE_STATES
            }
            for role_name in CONFIRMATORY_ROLES
        }
        comparisons[sp_name] = {
            "lora_csd_vs_base": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "base", "lora_csd", run_ok
            ),
            "lora_csd_vs_lora_diff_kl": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "lora_diff_kl", "lora_csd", run_ok
            ),
            "lora_csd_vs_lora_conv": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "lora_conv", "lora_csd", run_ok
            ),
            "lora_csd_vs_text": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "text", "lora_csd", run_ok
            ),
        }

    total_pass_init = (
        by_split["transfer"]["initial"]["PASS"]
        + by_split["retention"]["initial"]["PASS"]
    )
    total_pass_fb = (
        by_split["transfer"]["feedback"]["PASS"]
        + by_split["retention"]["feedback"]["PASS"]
    )
    text_wall_ns = int(g4b_report["costs"]["by_arm"]["text"]["generation_wall_ns"])
    fixed_setup_ns = (
        int(shared_costs.get("fast_anchor_trajectory_prep_ns", 0))
        + int(shared_costs.get("teacher_distill_prep_ns", 0))
        + int(training_summary["core_wall_ns"])
        + int(training_summary["adapter_save_ns"])
        + int(training_summary["adapter_load_ns"])
    )
    saved_per_task_ns = (text_wall_ns - csd_wall_ns) / 24.0
    breakeven_tasks = fixed_setup_ns / saved_per_task_ns if saved_per_task_ns > 0 else None
    speedup_vs_text = text_wall_ns / float(csd_wall_ns) if csd_wall_ns > 0 else None

    return {
        "kind": "LORA_CONFIRMATORY_V2_CONTEXT_DISTILL_ABLATION_REPORT",
        "version": 1,
        "run_id": run_id,
        "complete_and_verified": run_ok,
        "protocol_contradictions": sorted(set(proto_bad)),
        "incomplete_reasons": sorted(set(inc_reasons)),
        "canary_bit_exact_with_g4b_base": canary_matches,
        "changed_round1_vs_base": {
            "transfer": changed_vs_base_transfer,
            "retention": changed_vs_base_retention,
            "total": changed_vs_base_transfer + changed_vs_base_retention,
        },
        "heldout_dev_restraint_audit": heldout_dev_restraint_audit,
        "total_eval_tasks": 24,
        "total_pass_initial": total_pass_init,
        "total_pass_feedback": total_pass_fb,
        "training": training_summary,
        "lora_csd_by_split": by_split,
        "episodes": [csd_eps[(tid, "lora_csd")] for tid in EVAL_IDS if (tid, "lora_csd") in csd_eps],
        "comparisons_feedback": comparisons,
        "costs": {
            "lora_csd_generation_wall_ns": csd_wall_ns,
            "text_reference_generation_wall_ns": text_wall_ns,
            "speedup_vs_text": speedup_vs_text,
            "fixed_training_and_distill_ns": fixed_setup_ns,
            "breakeven_tasks_vs_text": breakeven_tasks,
            "adapter_bytes": training_summary["adapter_bytes"],
            "shared": shared_costs,
        },
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }


def run_gate_g9_ablation_stage(
    package,
    raw_files,
    worker_raw,
    g4b_phase_one_records,
    g4b_report,
    g5_episodes,
    g8_report,
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
        safe_write("g9-package-public.json", _pretty_json(pub))
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
                    "G9 READY mismatch",
                )
                done_obj = strict_json(frames["DONE"])
                require(
                    done_obj == {
                        "config_sha256": pub["phase_one_config_sha256"],
                        "result_sha256": _sha256(frames["RESULT"]),
                    },
                    "G9 DONE mismatch",
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
                                "request_id": "g9-p2-%04d" % len(valid_positions),
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

            time.sleep(45.0)
            p2_call = run_single_ssh_json_call(
                priv["phase_two_ssh_argv"],
                priv["ssh_environment"],
                encoded({
                    "kind": "LORA_SELF_CONTINUING_VALIDATION_BUNDLES",
                    "source_run_id": run_id,
                    "bundles_b64": [base64.b64encode(bm["bundle_raw"]).decode("ascii") for bm in built_b],
                }),
                cancellations,
                secondary_errors,
                timeout_ns=900_000_000_000,
            )
            if not (p2_call["reaped"] and p2_call["group_absent"] and p2_call["cleanup_bounded"]):
                p2_all_clean = False
            safe_write("phase2-self-continuing-stdout.json", p2_call["stdout_bytes"])
            safe_write("phase2-self-continuing-stderr.bin", p2_call["stderr_bytes"])

            if p2_call["status"] == "completed":
                env_out = parse_json(p2_call["stdout_bytes"].strip(), limit=MULTI_BUNDLE_STDOUT_LIMIT)
                require(
                    env_out.get("kind") == "LORA_SELF_CONTINUING_VALIDATION_RESULTS",
                    "self-continuing result kind mismatch",
                )
                conts_by_batch = {}
                for cit in env_out["continuations"]:
                    conts_by_batch.setdefault(cit["batch_index"], []).append(cit)

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
                    if f_ok or (not p_ok):
                        continue
                    rem_sl = bm["positions"][len(obs_m):]
                    for cit in conts_by_batch.get(b_idx, []):
                        si = cit["sub_index"]
                        ctrans = cit["transport"]
                        cm = build_confirmatory_continuation_bundle(
                            worker_raw, run_id, b_idx, si, rem_sl
                        )
                        safe_write("phase2-batch-%02d-cont-%02d-bundle.json" % (b_idx, si), cm["bundle_raw"])
                        safe_write("phase2-batch-%02d-cont-%02d-stdout.jsonl" % (b_idx, si), bytes.fromhex(ctrans["stdout_hex"]))
                        safe_write("phase2-batch-%02d-cont-%02d-stderr.bin" % (b_idx, si), bytes.fromhex(ctrans["stderr_hex"]))
                        crec = {
                            "batch_index": b_idx,
                            "sub_index": si,
                            "run_id": cm["run_id"],
                            "position_indices": [pos["position_index"] for pos in rem_sl],
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
                        if cf:
                            rem_sl = []
                        elif cp:
                            rem_sl = rem_sl[len(co):]
                        else:
                            break
            else:
                p2_all_clean = False

            if not cancellations:
                report = score_context_distill_ablation(
                    raw_files,
                    worker_raw,
                    run_id,
                    body_obj["phase_one_records"],
                    used_ids,
                    final_recs,
                    cont_recs,
                    p1_cfg["expected_runtime"],
                    body_obj["training_summary"],
                    body_obj["heldout_dev_restraint_audit"],
                    body_obj["shared_costs"],
                    g4b_phase_one_records,
                    g4b_report,
                    g5_episodes,
                    g8_report,
                )
                safe_write("context-distill-ablation-report.json", _pretty_json(report))

        p1_clean = bool(p1_out and p1_out["reaped"] and p1_out["group_absent"] and p1_out["cleanup_bounded"])
        binding_ok = bool(p1_clean and p2_all_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_GATE_G9_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and report and report["complete_and_verified"]),
            "secondary_errors": secondary_errors,
            **{flag_n: False for flag_n in FALSE_FLAGS},
        }
        safe_write("g9-controller-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": report}
