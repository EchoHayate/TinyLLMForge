import ast
import base64
from contextlib import ExitStack
import copy
import difflib
import hashlib
import io
import json
import math
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
    build_multi_bundle_ssh_argv,
    run_single_ssh_json_call,
)
from confirmatory_v2_gated_ablation import (
    _collect_scored_positions,
    load_g6_calibration_outputs_only,
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
from kl_anchored_lora_ablation import (
    KL_ANCHORED_HYPERPARAMS,
    _compute_trajectory_kl_loss,
    audit_kl_restraint_on_trajectories,
    prepare_kl_anchor_trajectories,
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


DIFF_FOCUSED_HYPERPARAMS = {
    **KL_ANCHORED_HYPERPARAMS,
    "diff_token_weight": 4.0,
    "scaffold_token_weight": 1.0,
    "cosmetic_valueerror_weight": 0.25,
    "align_first_round_prompt": True,
}

G8_SOURCE_NAMES = frozenset(
    set(ABLATION_SOURCE_NAMES)
    | {
        "gated_lora_router.py",
        "confirmatory_v2_gated_ablation.py",
        "kl_anchored_lora_ablation.py",
        "diff_focused_lora_ablation.py",
    }
)

G8_STDLIB_BOOTSTRAP = """import base64
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
    stage = p_path / ("g8-diff-abl-" + cfg["run_id"])
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
        import diff_focused_lora_ablation as g8_mod
        body_res = g8_mod.execute_remote_g8_ablation_body(cfg, stage, bundle)
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
        "kind": "LORA_GATE_G8_DIFF_FOCUSED_PHASE1_RESULT",
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


def _leading_indent(line_text):
    stripped = line_text.lstrip(" ")
    return len(line_text) - len(stripped)


def _is_cosmetic_valueerror_line(line_text):
    stripped = line_text.strip()
    if not stripped.startswith("raise ValueError("):
        return False
    try:
        parsed = ast.parse(stripped)
    except SyntaxError:
        return False
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Raise):
        return False
    exc_node = parsed.body[0].exc
    if not isinstance(exc_node, ast.Call):
        return False
    if not isinstance(exc_node.func, ast.Name) or exc_node.func.id != "ValueError":
        return False
    if exc_node.keywords or len(exc_node.args) != 1:
        return False
    arg_zero = exc_node.args[0]
    return isinstance(arg_zero, ast.Constant) and isinstance(arg_zero.value, str)


def classify_reference_line_weights(
    initial_source,
    reference_source,
    *,
    diff_weight=4.0,
    scaffold_weight=1.0,
    cosmetic_weight=0.25,
):
    require(isinstance(initial_source, str) and bool(initial_source), "initial_source string")
    require(isinstance(reference_source, str) and bool(reference_source), "reference_source string")
    init_lines = initial_source.splitlines(keepends=True)
    ref_lines = reference_source.splitlines(keepends=True)
    require(bool(ref_lines), "nonempty reference_lines required")

    weights = [float(scaffold_weight)] * len(ref_lines)
    matcher = difflib.SequenceMatcher(a=init_lines, b=ref_lines, autojunk=False)
    for opcode_tag, init_lo, init_hi, ref_lo, ref_hi in matcher.get_opcodes():
        if opcode_tag == "equal":
            continue
        if opcode_tag == "delete":
            if ref_lo < len(ref_lines):
                weights[ref_lo] = max(weights[ref_lo], float(diff_weight))
        elif opcode_tag == "insert":
            for ref_idx in range(ref_lo, ref_hi):
                weights[ref_idx] = float(diff_weight)
        elif opcode_tag == "replace":
            init_cosmetic_indents = {
                _leading_indent(init_lines[in_idx])
                for in_idx in range(init_lo, init_hi)
                if _is_cosmetic_valueerror_line(init_lines[in_idx])
            }
            for ref_idx in range(ref_lo, ref_hi):
                cur_line = ref_lines[ref_idx]
                if (
                    _is_cosmetic_valueerror_line(cur_line)
                    and _leading_indent(cur_line) in init_cosmetic_indents
                ):
                    weights[ref_idx] = float(cosmetic_weight)
                else:
                    weights[ref_idx] = float(diff_weight)

    return ref_lines, weights


def _map_tokens_to_normalized_weights(tokenizer, answer_text, target_ids, line_list, line_weights, scaffold_weight=1.0):
    require(len(target_ids) >= 2, "target_ids must include code tokens and EOS")
    code_ids = list(target_ids[:-1])
    decoded_full = tokenizer.decode(
        code_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    require(
        isinstance(decoded_full, str) and decoded_full.startswith(answer_text),
        "decoded answer_text prefix mismatch in weight mapper",
    )

    line_spans = []
    cursor = 0
    for ln_str, ln_wt in zip(line_list, line_weights):
        next_cur = cursor + len(ln_str)
        line_spans.append((cursor, next_cur, float(ln_wt)))
        cursor = next_cur
    require(cursor == len(answer_text), "line span sum mismatch")

    raw_tok_weights = []
    prev_len = 0
    for tok_idx in range(len(code_ids)):
        prefix_str = tokenizer.decode(
            code_ids[:tok_idx + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        cur_len = len(prefix_str)
        char_lo = prev_len
        char_hi = max(cur_len, char_lo + 1)
        prev_len = cur_len
        overlapped = [
            ln_wt
            for span_lo, span_hi, ln_wt in line_spans
            if span_hi > char_lo and span_lo < char_hi
        ]
        raw_tok_weights.append(max(overlapped) if overlapped else float(scaffold_weight))

    raw_tok_weights.append(float(scaffold_weight))
    mean_val = sum(raw_tok_weights) / float(len(raw_tok_weights))
    require(mean_val > 1e-9 and math.isfinite(mean_val), "invalid mean token weight")
    return [wt_val / mean_val for wt_val in raw_tok_weights]


def prepare_diff_focused_training(tokenizer, supervision_raw, *, hyperparams=None):
    cfg_hp = copy.deepcopy(DIFF_FOCUSED_HYPERPARAMS if hyperparams is None else hyperparams)
    sup_list = _load_supervision_only(supervision_raw)
    rows = []
    observed_state_sha = None
    for item in sup_list:
        messages = _render_user_prompt(item, examples_list=None, witness_obj=None)
        answer_text = item["reference_source"]
        measured = measure_tokens(tokenizer, messages, answer=answer_text)
        require(
            measured["status"] == "OK" and measured["target"] is not None,
            "supervision row exceeded token limits",
        )
        if observed_state_sha is None:
            observed_state_sha = measured["state_sha256"]
        else:
            require(measured["state_sha256"] == observed_state_sha, "tokenizer state drifted")

        prompt_ids = list(measured["input"]["input_ids"])
        target_ids = list(measured["target"]["target_ids"])
        p_len = len(prompt_ids)
        t_len = len(target_ids)
        line_list, line_wts = classify_reference_line_weights(
            item["initial_source"],
            answer_text,
            diff_weight=float(cfg_hp["diff_token_weight"]),
            scaffold_weight=float(cfg_hp["scaffold_token_weight"]),
            cosmetic_weight=float(cfg_hp["cosmetic_valueerror_weight"]),
        )
        norm_weights = _map_tokens_to_normalized_weights(
            tokenizer,
            answer_text,
            target_ids,
            line_list,
            line_wts,
            scaffold_weight=float(cfg_hp["scaffold_token_weight"]),
        )
        high_focus_count = sum(1 for wt_v in norm_weights if wt_v > 1.25)
        rows.append({
            "task_id": item["id"],
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (p_len + t_len),
            "labels": ([-100] * p_len) + target_ids,
            "target_weights": norm_weights,
            "prompt_tokens": p_len,
            "target_tokens": t_len,
            "focused_target_tokens": high_focus_count,
            "messages_sha256": measured["input"]["messages_sha256"],
            "answer_sha256": measured["target"]["answer_sha256"],
        })

    core_ds = {
        "kind": "LORA_CONFIRMATORY_V2_DIFF_FOCUSED_TRAINING_INPUTS",
        "version": 1,
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "tokenizer_state_sha256": observed_state_sha,
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "rows": rows,
    }
    return {**core_ds, "dataset_sha256": digest(core_ds)}


def validate_diff_focused_training(dataset):
    require(
        isinstance(dataset, dict)
        and dataset.get("kind") == "LORA_CONFIRMATORY_V2_DIFF_FOCUSED_TRAINING_INPUTS"
        and dataset.get("version") == 1,
        "diff-focused dataset header mismatch",
    )
    require(
        dataset.get("supervision_sha256") == CONFIRMATORY_FILE_PINS["supervision.json"],
        "supervision pin mismatch",
    )
    rows = dataset.get("rows")
    require(isinstance(rows, list) and len(rows) == 8, "8 diff-focused rows required")
    for row, exp_tid in zip(rows, SUPERVISION_IDS):
        require(row["task_id"] == exp_tid, "supervision order mismatch")
        p_len = int(row["prompt_tokens"])
        t_len = int(row["target_tokens"])
        require(len(row["input_ids"]) == p_len + t_len, "input_ids length mismatch")
        require(len(row["labels"]) == p_len + t_len, "labels length mismatch")
        wts = row["target_weights"]
        require(isinstance(wts, list) and len(wts) == t_len, "target_weights length mismatch")
        mean_w = sum(float(wt_x) for wt_x in wts) / float(t_len)
        require(abs(mean_w - 1.0) <= 1e-4, "target_weights must normalize to mean 1.0")
    core_copy = {key_n: val_v for key_n, val_v in dataset.items() if key_n != "dataset_sha256"}
    expected_sha = digest(core_copy)
    require(dataset.get("dataset_sha256") == expected_sha, "diff-focused dataset_sha256 mismatch")
    return expected_sha


def weighted_answer_loss(logits, labels, target_weights):
    import torch
    from torch.nn import functional

    require(isinstance(logits, torch.Tensor) and logits.ndim == 3, "batched 3D logits required")
    require(
        isinstance(labels, torch.Tensor)
        and labels.dtype == torch.long
        and tuple(labels.shape) == tuple(logits.shape[:2]),
        "aligned integer labels required",
    )
    shifted = labels[:, 1:]
    selected = shifted != -100
    require(bool(selected.any()), "nonempty supervised targets required")
    targets = shifted[selected]
    per_token_ce = functional.cross_entropy(
        logits[:, :-1][selected].float(), targets, reduction="none"
    )
    weight_tensor = torch.tensor(target_weights, dtype=torch.float32, device=logits.device)
    require(
        weight_tensor.ndim == 1 and weight_tensor.shape[0] == per_token_ce.shape[0],
        "target_weights count mismatch",
    )
    require(bool(torch.isfinite(weight_tensor).all()) and bool((weight_tensor > 0).all()), "positive finite weights required")
    return (per_token_ce * weight_tensor).mean()


def train_diff_focused_kl_adapter(model, dataset, background_trajectories, *, hyperparams=None):
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
        answer_loss,
    )

    cfg_hp = copy.deepcopy(DIFF_FOCUSED_HYPERPARAMS if hyperparams is None else hyperparams)
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
        isinstance(background_trajectories, list) and len(background_trajectories) == 16,
        "16 background anchor trajectories required",
    )
    copied_ds = copy.deepcopy(dataset)
    dataset_sha256 = validate_diff_focused_training(copied_ds)
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
                torch.cuda.manual_seed(int(cfg_hp["seed"]))
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
            lambda_kl = float(cfg_hp["lambda_kl"])
            bg_per_step = int(cfg_hp["background_anchors_per_step"])

            for step_idx in range(total_steps):
                frac = step_idx / float(max(total_steps - 1, 1))
                cur_lr = base_lr * (1.0 - 0.9 * frac)
                for p_grp in optimizer.param_groups:
                    p_grp["lr"] = cur_lr

                row = copied_ds["rows"][step_idx % 8]
                step_t0 = time.perf_counter_ns()
                optimizer.zero_grad(set_to_none=True)

                inp_sup = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                msk_sup = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
                lbl_sup = torch.tensor([row["labels"]], dtype=torch.long, device=device)
                out_sup = model(input_ids=inp_sup, attention_mask=msk_sup, use_cache=False, return_dict=True)
                w_sup_loss = weighted_answer_loss(out_sup.logits, lbl_sup, row["target_weights"])
                with torch.no_grad():
                    unw_sup_val = float(answer_loss(out_sup.logits.detach(), lbl_sup).cpu().item())
                _finite([w_sup_loss])
                w_sup_loss.backward()
                w_sup_val = float(w_sup_loss.detach().cpu().item())
                del out_sup, inp_sup, msk_sup, lbl_sup, w_sup_loss

                kl_vals = []
                for sub_idx in range(bg_per_step):
                    bg_item = background_trajectories[(step_idx * bg_per_step + sub_idx) % 16]
                    kl_loss_t, _matched, _alen = _compute_trajectory_kl_loss(model, bg_item, device)
                    _finite([kl_loss_t])
                    kl_vals.append(float(kl_loss_t.detach().cpu().item()))
                    if lambda_kl > 0.0:
                        scaled_kl = (lambda_kl / float(bg_per_step)) * kl_loss_t
                        scaled_kl.backward()
                    del kl_loss_t

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

                mean_kl_step = sum(kl_vals) / float(len(kl_vals))
                steps_log.append({
                    "step": step_idx + 1,
                    "epoch": (step_idx // 8) + 1,
                    "task_id": row["task_id"],
                    "lr": cur_lr,
                    "weighted_sup_loss": w_sup_val,
                    "unweighted_sup_loss": unw_sup_val,
                    "bg_kl_loss": mean_kl_step,
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
    require(_tensor_digest(_state(model)) == base_sha256, "base weights mutated during G8 training")
    init_sha = _tensor_digest(initial_adapter)
    final_sha = _tensor_digest(final_adapter)
    require(init_sha != final_sha, "G8 adapter did not update")
    adapter_bytes = sum(ten.numel() * ten.element_size() for ten in final_adapter.values())
    ver_ns = time.perf_counter_ns() - ver_t0

    num_epochs = int(cfg_hp["epochs"])
    epoch_w_sup = [
        sum(st_r["weighted_sup_loss"] for st_r in steps_log[ep_i * 8:(ep_i + 1) * 8]) / 8.0
        for ep_i in range(num_epochs)
    ]
    epoch_unw_sup = [
        sum(st_r["unweighted_sup_loss"] for st_r in steps_log[ep_i * 8:(ep_i + 1) * 8]) / 8.0
        for ep_i in range(num_epochs)
    ]
    epoch_kl = [
        sum(st_r["bg_kl_loss"] for st_r in steps_log[ep_i * 8:(ep_i + 1) * 8]) / 8.0
        for ep_i in range(num_epochs)
    ]

    return {
        "kind": "LORA_DIFF_FOCUSED_KL_TRAINING_RESULT",
        "hyperparams": copy.deepcopy(cfg_hp),
        "dataset_sha256": dataset_sha256,
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
        "epoch_mean_weighted_sup_losses": epoch_w_sup,
        "epoch_mean_unweighted_sup_losses": epoch_unw_sup,
        "epoch_mean_bg_kl_losses": epoch_kl,
        "per_task_focus_summary": {
            row["task_id"]: {
                "target_tokens": row["target_tokens"],
                "focused_target_tokens": row["focused_target_tokens"],
            }
            for row in copied_ds["rows"]
        },
        "steps": steps_log,
        "initial_adapter": initial_adapter,
        "adapter": final_adapter,
    }


def build_diff_focused_schedule(prompts_raw, supervision_raw, *, run_id):
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
                    encoded(["g8_diff_call", run_id, ep_idx, canary_tid, "base_canary", role_name])
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
            "arm": "lora_diff_kl",
            "render_arm": "lora",
            "first_messages": f_msgs,
            "first_messages_sha256": digest(f_msgs),
            "call_ids": {
                role_name: _sha256(
                    encoded(["g8_diff_call", run_id, ep_idx, task["id"], "lora_diff_kl", role_name])
                )[:32]
                for role_name in CONFIRMATORY_ROLES
            },
        })

    require(len(episodes) == 26, "26 episodes required")
    return {
        "kind": "LORA_CONFIRMATORY_V2_DIFF_FOCUSED_SCHEDULE",
        "version": 1,
        "run_id": run_id,
        "scheduled_episodes": 26,
        "scheduled_generation_calls": 52,
        "episodes": episodes,
    }


def build_gate_g8_ablation_package(
    raw_files,
    v1_prompts_raw,
    v1_structure_raw,
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
    work_timeout_seconds=5400,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    chosen_hp = copy.deepcopy(DIFF_FOCUSED_HYPERPARAMS if hyperparams is None else hyperparams)
    load_confirmatory_dataset(raw_files)
    v1_sets = load_g6_calibration_outputs_only(
        raw_files["supervision.json"], v1_prompts_raw, v1_structure_raw
    )
    anchor_public_raw = encoded({
        "background": v1_sets["background"],
        "dev_negative": v1_sets["dev_negative"],
    })
    require(_sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(_sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    require(set(dep_sources) == G8_SOURCE_NAMES, "G8 source names")
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    require(len(scoring_run_ids) == 13, "13 scoring_run_ids")
    seen_ids = {run_id}
    for s_rid in scoring_run_ids:
        _check_hex(s_rid, 32)
        require(s_rid not in seen_ids, "duplicate scoring_run_id")
        seen_ids.add(s_rid)

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    sources_sha = {fname: _sha256(dep_sources[fname]) for fname in sorted(G8_SOURCE_NAMES)}
    payload_raw = encoded({
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(G8_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision.json"]).decode("ascii"),
        "anchor_public_b64": base64.b64encode(anchor_public_raw).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    })

    boot_bytes = G8_STDLIB_BOOTSTRAP.encode("ascii")
    boot_sha = _sha256(boot_bytes)
    p1_cfg = {
        "kind": "LORA_GATE_G8_DIFF_FOCUSED_PHASE1_CONFIG",
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
        "stage": str(Path(remote_parent) / ("g8-diff-abl-" + run_id)),
        "run_id": run_id,
        "hyperparams": copy.deepcopy(chosen_hp),
        "sources_sha256": sources_sha,
        "prompts_sha256": CONFIRMATORY_FILE_PINS["prompts.json"],
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
        "anchor_public_sha256": _sha256(anchor_public_raw),
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
        "ns={'__name__':'g8_boot'}\n"
        "exec(compile(src,'<g8_stdlib_bootstrap>','exec'),ns)\n"
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
    p2_ssh_argv = build_multi_bundle_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )

    manifest = {
        "kind": "LORA_GATE_G8_DIFF_FOCUSED_LAUNCH_PACKAGE",
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


def execute_remote_g8_ablation_body(config, stage, bundle):
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
    anchor_pub_raw = base64.b64decode(bundle["anchor_public_b64"], validate=True)
    backend_raw = base64.b64decode(bundle["backend_b64"], validate=True)
    worker_raw = base64.b64decode(bundle["worker_b64"], validate=True)
    require(
        _sha256(prompts_raw) == config["prompts_sha256"]
        and _sha256(supervision_raw) == config["supervision_sha256"]
        and _sha256(anchor_pub_raw) == config["anchor_public_sha256"]
        and _sha256(backend_raw) == config["backend_sha256"]
        and _sha256(worker_raw) == config["worker_sha256"],
        "G8 input pin mismatch",
    )
    anchor_sets = parse_json(anchor_pub_raw)

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

    active_hp = copy.deepcopy(config.get("hyperparams") or DIFF_FOCUSED_HYPERPARAMS)
    anc_t0 = time.monotonic_ns()
    bg_trajs = prepare_kl_anchor_trajectories(
        model,
        tokenizer,
        anchor_sets["background"],
        max_new_tokens=int(active_hp["max_anchor_tokens"]),
    )
    dev_neg_trajs = prepare_kl_anchor_trajectories(
        model,
        tokenizer,
        anchor_sets["dev_negative"],
        max_new_tokens=int(active_hp["max_anchor_tokens"]),
    )
    shared_costs["anchor_trajectory_prep_ns"] = time.monotonic_ns() - anc_t0

    train_ds = prepare_diff_focused_training(tokenizer, supervision_raw, hyperparams=active_hp)
    train_out = train_diff_focused_kl_adapter(model, train_ds, bg_trajs, hyperparams=active_hp)
    del bg_trajs

    buf_io = io.BytesIO()
    save_t0 = time.perf_counter_ns()
    torch.save({"adapter": train_out["adapter"]}, buf_io)
    art_raw = buf_io.getvalue()
    write_private(stage / "diff_focused_kl_adapter.pt", art_raw)
    save_ns = time.perf_counter_ns() - save_t0
    load_t0 = time.perf_counter_ns()
    reloaded = torch.load(
        io.BytesIO((stage / "diff_focused_kl_adapter.pt").read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    load_ns = time.perf_counter_ns() - load_t0
    require(
        tensor_digest_fn(reloaded["adapter"]) == train_out["adapter_sha256"],
        "reloaded diff-focused adapter digest mismatch",
    )
    train_out["adapter"] = reloaded["adapter"]

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
        "epoch_mean_weighted_sup_losses": train_out["epoch_mean_weighted_sup_losses"],
        "epoch_mean_unweighted_sup_losses": train_out["epoch_mean_unweighted_sup_losses"],
        "epoch_mean_bg_kl_losses": train_out["epoch_mean_bg_kl_losses"],
        "per_task_focus_summary": train_out["per_task_focus_summary"],
        "steps": train_out["steps"],
    }

    prompts_map = {item["id"]: item for item in parse_json(prompts_raw)}
    schedule = build_diff_focused_schedule(prompts_raw, supervision_raw, run_id=config["run_id"])
    worker_text = worker_raw.decode("utf-8")
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

            want_lora = arm == "lora_diff_kl"
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
        raise RuntimeError("G8 post-run integrity drift")
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


def score_diff_focused_ablation(
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
    g7_r1_report,
):
    dataset = load_confirmatory_dataset(raw_files)
    expected_sched = build_diff_focused_schedule(
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
                        "request_id": "g8-p2-%04d" % len(valid_positions),
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
    g7_eps = {(ep["task_id"], "lora_kl_64"): {**ep, "arm": "lora_kl_64"} for ep in g7_r1_report["episodes"]}

    canary_matches = {}
    diff_eps = {}
    diff_wall_ns = 0
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
        elif arm == "lora_diff_kl":
            if ep_rec["candidates"]["initial"] != g4b_cands[(tid, "base")]["initial"]:
                if ep_rec["split"] == "transfer":
                    changed_vs_base_transfer += 1
                else:
                    changed_vs_base_retention += 1
            diff_wall_ns += (
                ep_rec["generation_costs"]["initial"]["wall_ns"]
                + ep_rec["generation_costs"]["feedback"]["wall_ns"]
            )
            diff_eps[(tid, "lora_diff_kl")] = {
                "task_id": tid,
                "split": ep_rec["split"],
                "arm": "lora_diff_kl",
                "roles": roles_out,
            }

    canaries_ok = (
        set(canary_matches) == {"WT01", "WR01"}
        and all(canary_matches.values())
        and len(diff_eps) == 24
    )
    run_ok = bool(not proto_bad and not inc_reasons and canaries_ok)
    merged_indexed = {**g4b_eps, **g5_eps, **g7_eps, **diff_eps}

    by_split = {}
    comparisons = {}
    for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
        by_split[sp_name] = {
            role_name: {
                st_n: sum(
                    (tid, "lora_diff_kl") in diff_eps
                    and diff_eps[(tid, "lora_diff_kl")]["roles"][role_name]["state"] == st_n
                    for tid in sp_ids
                )
                for st_n in CANDIDATE_STATES
            }
            for role_name in CONFIRMATORY_ROLES
        }
        comparisons[sp_name] = {
            "lora_diff_kl_vs_base": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "base", "lora_diff_kl", run_ok
            ),
            "lora_diff_kl_vs_lora_kl_64": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "lora_kl_64", "lora_diff_kl", run_ok
            ),
            "lora_diff_kl_vs_lora_conv": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "lora_conv", "lora_diff_kl", run_ok
            ),
            "lora_diff_kl_vs_text": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "text", "lora_diff_kl", run_ok
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
        int(shared_costs.get("anchor_trajectory_prep_ns", 0))
        + int(training_summary["core_wall_ns"])
        + int(training_summary["adapter_save_ns"])
        + int(training_summary["adapter_load_ns"])
    )
    saved_per_task_ns = (text_wall_ns - diff_wall_ns) / 24.0
    breakeven_tasks = fixed_setup_ns / saved_per_task_ns if saved_per_task_ns > 0 else None
    speedup_vs_text = text_wall_ns / float(diff_wall_ns) if diff_wall_ns > 0 else None

    return {
        "kind": "LORA_CONFIRMATORY_V2_DIFF_FOCUSED_ABLATION_REPORT",
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
        "lora_diff_kl_by_split": by_split,
        "episodes": [diff_eps[(tid, "lora_diff_kl")] for tid in EVAL_IDS if (tid, "lora_diff_kl") in diff_eps],
        "comparisons_feedback": comparisons,
        "costs": {
            "lora_diff_kl_generation_wall_ns": diff_wall_ns,
            "text_reference_generation_wall_ns": text_wall_ns,
            "speedup_vs_text": speedup_vs_text,
            "fixed_training_and_anchor_ns": fixed_setup_ns,
            "breakeven_tasks_vs_text": breakeven_tasks,
            "adapter_bytes": training_summary["adapter_bytes"],
            "shared": shared_costs,
        },
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }


def run_gate_g8_ablation_stage(
    package,
    raw_files,
    worker_raw,
    g4b_phase_one_records,
    g4b_report,
    g5_episodes,
    g7_r1_report,
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
        safe_write("g8-package-public.json", _pretty_json(pub))
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
                    "G8 READY mismatch",
                )
                done_obj = strict_json(frames["DONE"])
                require(
                    done_obj == {
                        "config_sha256": pub["phase_one_config_sha256"],
                        "result_sha256": _sha256(frames["RESULT"]),
                    },
                    "G8 DONE mismatch",
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
                                "request_id": "g8-p2-%04d" % len(valid_positions),
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
                report = score_diff_focused_ablation(
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
                    g7_r1_report,
                )
                safe_write("diff-focused-ablation-report.json", _pretty_json(report))

        p1_clean = bool(p1_out and p1_out["reaped"] and p1_out["group_absent"] and p1_out["cleanup_bounded"])
        binding_ok = bool(p1_clean and p2_all_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_GATE_G8_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and report and report["complete_and_verified"]),
            "secondary_errors": secondary_errors,
            **{flag_n: False for flag_n in FALSE_FLAGS},
        }
        safe_write("g8-controller-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": report}


def build_self_continuing_phase2_ssh_argv(
    ssh_argv_prefix,
    resolved_interpreter,
    backend_raw,
    expected_runtime,
):
    backend_b64 = base64.b64encode(backend_raw).decode("ascii")
    expected_rt_b64 = base64.b64encode(encoded(expected_runtime)).decode("ascii")
    child_snippet = (
        "import json, pathlib\n"
        "expected_rt = json.loads(pathlib.Path('expected_runtime.json').read_text('utf-8'))\n"
        "backend_bytes = pathlib.Path('backend.py').read_bytes()\n"
        "namespace = {'__name__': '__main__', 'EXPECTED_RUNTIME': expected_rt}\n"
        "exec(compile(backend_bytes, 'validation_remote.py', 'exec'), namespace)\n"
    )
    remote_driver = (
        "import base64,hashlib,json,os,shutil,signal,socket,subprocess,sys,tempfile,time\n"
        "def enc_j(val):\n"
        "    return json.dumps(val,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')\n"
        f"backend_bytes=base64.b64decode({backend_b64!r},validate=True)\n"
        f"expected_rt_raw=base64.b64decode({expected_rt_b64!r},validate=True)\n"
        "expected_rt=json.loads(expected_rt_raw.decode('ascii'))\n"
        f"if hashlib.sha256(backend_bytes).hexdigest()!={BACKEND_SHA256!r}:\n"
        "    sys.exit(78)\n"
        "if socket.gethostname().split('.')[0]!=expected_rt['hostname'].split('.')[0] or os.getuid()!=expected_rt['uid']:\n"
        "    sys.exit(79)\n"
        "envelope=json.loads(sys.stdin.buffer.read(8388608).decode('ascii'))\n"
        "if envelope.get('kind')!='LORA_SELF_CONTINUING_VALIDATION_BUNDLES':\n"
        "    sys.exit(77)\n"
        "source_run_id=envelope['source_run_id']\n"
        "bundles_b64=envelope['bundles_b64']\n"
        "os.umask(0o077)\n"
        "def run_one_bundle(bundle_bytes):\n"
        "    wdir=tempfile.mkdtemp(prefix='tlf-sc-',dir='/tmp')\n"
        "    t_start=time.monotonic_ns()\n"
        "    reaped=False\n"
        "    rc_val=None\n"
        "    out_b=b''\n"
        "    err_b=b''\n"
        "    t_out=False\n"
        "    try:\n"
        "        os.chmod(wdir,0o700)\n"
        "        for fn,fb in (('backend.py',backend_bytes),('expected_runtime.json',expected_rt_raw)):\n"
        "            fdesc=os.open(os.path.join(wdir,fn),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)\n"
        "            with os.fdopen(fdesc,'wb') as fst:\n"
        "                os.fchmod(fst.fileno(),0o600)\n"
        "                fst.write(fb)\n"
        "                fst.flush()\n"
        f"        proc=subprocess.Popen([{resolved_interpreter!r},'-I','-S','-B','-c',{child_snippet!r}],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=wdir,env={{'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','HOME':'/nonexistent'}},close_fds=True,pass_fds=(),start_new_session=True,shell=False)\n"
        "        try:\n"
        "            out_b,err_b=proc.communicate(input=bundle_bytes,timeout=95.0)\n"
        "        except subprocess.TimeoutExpired:\n"
        "            t_out=True\n"
        "        finally:\n"
        "            if proc.poll() is None:\n"
        "                try:\n"
        "                    os.killpg(proc.pid,signal.SIGKILL)\n"
        "                except ProcessLookupError:\n"
        "                    pass\n"
        "            try:\n"
        "                rc_val=proc.wait(timeout=2.0)\n"
        "                reaped=True\n"
        "            except subprocess.TimeoutExpired:\n"
        "                reaped=False\n"
        "            for pobj in (proc.stdin,proc.stdout,proc.stderr):\n"
        "                if pobj is not None:\n"
        "                    try:\n"
        "                        pobj.close()\n"
        "                    except OSError:\n"
        "                        pass\n"
        "    finally:\n"
        "        os.chdir('/')\n"
        "        shutil.rmtree(wdir,ignore_errors=True)\n"
        "    el_ns=max(0,time.monotonic_ns()-t_start)\n"
        "    st_lbl=('timeout' if t_out else ('completed' if (reaped and rc_val==0 and not err_b) else 'program_error'))\n"
        "    return {'status':st_lbl,'returncode':rc_val,'stdout_hex':out_b.hex(),'stderr_hex':err_b.hex(),'elapsed_ns':el_ns,'process_reaped':reaped,'stdin_complete':(not t_out)},out_b\n"
        "def count_items_and_stopped(out_b):\n"
        "    cnt=0\n"
        "    stopped=False\n"
        "    for ln in out_b.decode('utf-8',errors='replace').splitlines():\n"
        "        if not ln.strip(): continue\n"
        "        obj=json.loads(ln)\n"
        "        if obj.get('kind')=='row': cnt+=1\n"
        "        elif obj.get('kind')=='done': stopped=bool(obj.get('stopped'))\n"
        "    return cnt,stopped\n"
        "results_out=[]\n"
        "conts_out=[]\n"
        "for b_idx,b_b64 in enumerate(bundles_b64):\n"
        "    bundle_in=base64.b64decode(b_b64,validate=True)\n"
        "    parsed_b=json.loads(bundle_in.decode('utf-8'))\n"
        "    rem_reqs=list(parsed_b['requests'])\n"
        "    w_src=parsed_b['worker_source']\n"
        "    tr_one,raw_o=run_one_bundle(bundle_in)\n"
        "    results_out.append({'batch_index':b_idx,'transport':tr_one})\n"
        "    if tr_one['status']!='completed':\n"
        "        break\n"
        "    obs_n,was_stopped=count_items_and_stopped(raw_o)\n"
        "    rem_reqs=rem_reqs[obs_n:]\n"
        "    sub_idx=1\n"
        "    while was_stopped and rem_reqs:\n"
        "        c_rid=hashlib.sha256(enc_j(['conf_v2_p2_cont',source_run_id,b_idx,sub_idx])).hexdigest()[:32]\n"
        "        c_raw=enc_j({'kind':'LORA_REMOTE_VALIDATION_BUNDLE','mode':'matrix','run_id':c_rid,'worker_source':w_src,'requests':rem_reqs})\n"
        "        c_tr,c_out=run_one_bundle(c_raw)\n"
        "        conts_out.append({'batch_index':b_idx,'sub_index':sub_idx,'run_id':c_rid,'transport':c_tr})\n"
        "        if c_tr['status']!='completed':\n"
        "            rem_reqs=[]\n"
        "            break\n"
        "        c_n,was_stopped=count_items_and_stopped(c_out)\n"
        "        rem_reqs=rem_reqs[c_n:]\n"
        "        sub_idx+=1\n"
        "sys.stdout.buffer.write(enc_j({'kind':'LORA_SELF_CONTINUING_VALIDATION_RESULTS','results':results_out,'continuations':conts_out})+b'\\n')\n"
        "sys.stdout.buffer.flush()\n"
    )
    remote_cmd = "exec " + shlex.join((
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        "LANG=C",
        "LC_ALL=C",
        "HOME=/nonexistent",
        resolved_interpreter,
        "-I",
        "-S",
        "-B",
        "-c",
        remote_driver,
    ))
    require(len(remote_cmd.encode("utf-8")) <= 65536, "Phase 2 SSH command size exceeds cap")
    return [*ssh_argv_prefix, remote_cmd]


def build_gate_g8_phase2_only_package(
    raw_files,
    worker_raw,
    backend_raw,
    preflight_raw,
    *,
    source_run_id,
    sealed_phase1_wire_sha256,
    sealed_phase1_result_sha256,
    stage_id,
    scoring_run_ids,
    ssh_argv_prefix,
    ssh_environment,
    resolved_interpreter,
    local_run_root,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_confirmatory_dataset(raw_files)
    require(_sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(_sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    _check_hex(source_run_id, 32)
    _check_hex(stage_id, 32)
    _check_hex(sealed_phase1_wire_sha256, 64)
    _check_hex(sealed_phase1_result_sha256, 64)
    require(len(scoring_run_ids) == 13, "13 scoring_run_ids")
    seen_ids = {source_run_id, stage_id}
    for s_rid in scoring_run_ids:
        _check_hex(s_rid, 32)
        require(s_rid not in seen_ids, "duplicate scoring_run_id")
        seen_ids.add(s_rid)

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    p2_ssh_argv = build_self_continuing_phase2_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )
    manifest = {
        "kind": "LORA_GATE_G8_PHASE2_ONLY_PACKAGE",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "source_run_id": source_run_id,
        "stage_id": stage_id,
        "sealed_phase1_wire_sha256": sealed_phase1_wire_sha256,
        "sealed_phase1_result_sha256": sealed_phase1_result_sha256,
        "scoring_run_ids": list(scoring_run_ids),
        "expected_runtime": copy.deepcopy(expected_runtime),
        "phase_two_argv_sha256": digest(p2_ssh_argv),
        "local_run_root": local_run_root,
        "planned_generations": 0,
        "single_ssh_self_continuing_cpu_scoring": True,
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }
    return {
        "public": {**manifest, "package_sha256": digest(manifest)},
        "private": {
            "phase_two_ssh_argv": p2_ssh_argv,
            "ssh_environment": dict(ssh_environment),
        },
    }


def run_gate_g8_phase2_only_stage(
    package,
    sealed_wire_bytes,
    sealed_result_bytes,
    raw_files,
    worker_raw,
    g4b_phase_one_records,
    g4b_report,
    g5_episodes,
    g7_r1_report,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    _plain(package)
    fields(package, ("public", "private"))
    pub = package["public"]
    priv = package["private"]
    manifest_no_sha = {key_n: val_v for key_n, val_v in pub.items() if key_n != "package_sha256"}
    require(digest(manifest_no_sha) == pub["package_sha256"], "package_sha256 mismatch")
    require(digest(priv["phase_two_ssh_argv"]) == pub["phase_two_argv_sha256"], "p2 argv sha")
    require(_sha256(sealed_wire_bytes) == pub["sealed_phase1_wire_sha256"], "sealed wire sha mismatch")
    require(_sha256(sealed_result_bytes) == pub["sealed_phase1_result_sha256"], "sealed result sha mismatch")

    frames = decode_frames(sealed_wire_bytes)
    parsed_res = strict_json(frames["RESULT"])
    require(parsed_res == parse_json(sealed_result_bytes), "sealed wire/result disagreement")
    require(parsed_res.get("bootstrap_failure") is None, "sealed Phase 1 had bootstrap failure")
    body_obj = parsed_res["body"]
    source_run_id = pub["source_run_id"]
    stage_id = pub["stage_id"]
    expected_rt = pub["expected_runtime"]

    run_fd = _claim_local_run_root(pub["local_run_root"], stage_id)
    cancellations = []
    persistence_errors = []
    secondary_errors = []
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
        safe_write("g8-phase2-only-public.json", _pretty_json(pub))
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
                            "request_id": "g8-p2-%04d" % len(valid_positions),
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

        p2_call = run_single_ssh_json_call(
            priv["phase_two_ssh_argv"],
            priv["ssh_environment"],
            encoded({
                "kind": "LORA_SELF_CONTINUING_VALIDATION_BUNDLES",
                "source_run_id": source_run_id,
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

        final_recs = []
        cont_recs = []
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

                obs_m, f_ok, p_ok, _pe = _decode_scoring_batch(bm, prec, expected_rt)
                if f_ok or (not p_ok):
                    continue
                rem_sl = bm["positions"][len(obs_m):]
                for cit in conts_by_batch.get(b_idx, []):
                    si = cit["sub_index"]
                    ctrans = cit["transport"]
                    cm = build_confirmatory_continuation_bundle(
                        worker_raw, source_run_id, b_idx, si, rem_sl
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
                        expected_rt,
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
            report = score_diff_focused_ablation(
                raw_files,
                worker_raw,
                source_run_id,
                body_obj["phase_one_records"],
                used_ids,
                final_recs,
                cont_recs,
                expected_rt,
                body_obj["training_summary"],
                body_obj["heldout_dev_restraint_audit"],
                body_obj["shared_costs"],
                g4b_phase_one_records,
                g4b_report,
                g5_episodes,
                g7_r1_report,
            )
            safe_write("diff-focused-ablation-report.json", _pretty_json(report))

        binding_ok = bool(p2_all_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_GATE_G8_PHASE2_ONLY_EXECUTION_EVIDENCE",
            "version": 1,
            "source_run_id": source_run_id,
            "stage_id": stage_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and report and report["complete_and_verified"]),
            "secondary_errors": secondary_errors,
            **{flag_n: False for flag_n in FALSE_FLAGS},
        }
        safe_write("g8-phase2-only-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": report}
