import base64
from contextlib import contextmanager
import copy
import gc
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import shlex
import statistics
import sys
import time

from confirmatory_v2_contract import (
    CONFIRMATORY_FILE_PINS,
    EVAL_IDS,
    RETENTION_IDS,
    TRANSFER_IDS,
    confirmatory_first_round,
    confirmatory_second_round,
    load_confirmatory_dataset,
)
from confirmatory_v2_converged_ablation import ABLATION_SOURCE_NAMES
from confirmatory_v2_gate_a import (
    MULTI_BUNDLE_STDOUT_LIMIT,
    run_single_ssh_json_call,
)
from confirmatory_v2_gated_ablation import (
    _collect_scored_positions,
    load_g6_calibration_outputs_only,
)
from confirmatory_v2_remote import (
    decode_frames,
    strict_json,
    write_private,
)
from confirmatory_v2_scoring import (
    CONFIRMATORY_ROLES,
    _paired_arm_comparison,
    build_confirmatory_continuation_bundle,
)
from context_self_distillation_ablation import prepare_fast_background_trajectories
from diff_focused_lora_ablation import (
    DIFF_FOCUSED_HYPERPARAMS,
    build_self_continuing_phase2_ssh_argv,
    prepare_diff_focused_training,
    train_diff_focused_kl_adapter,
)
from kl_anchored_lora_ablation import audit_kl_restraint_on_trajectories
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


BENCHMARK_BATCH_SIZES = (1, 4, 8, 16, 32, 64)
BENCHMARK_PREFIX_TOKENS = 3050
BENCHMARK_QUERY_TOKENS = 450
BENCHMARK_DECODE_TOKENS = 128

G11_SOURCE_NAMES = frozenset(
    set(ABLATION_SOURCE_NAMES)
    | {
        "gated_lora_router.py",
        "confirmatory_v2_gated_ablation.py",
        "kl_anchored_lora_ablation.py",
        "diff_focused_lora_ablation.py",
        "context_self_distillation_ablation.py",
        "mlp_hybrid_lora_ablation.py",
        "bf16_merge_and_kv_benchmark.py",
    }
)

G11_STDLIB_BOOTSTRAP = """import base64
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
    stage = p_path / ("g11-sys-bench-" + cfg["run_id"])
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
        import bf16_merge_and_kv_benchmark as g11_mod
        body_res = g11_mod.execute_remote_g11_system_body(cfg, stage, bundle)
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
        "kind": "LORA_GATE_G11_SYSTEM_BENCHMARK_PHASE1_RESULT",
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


def pack_adapter_to_bf16(fp32_adapter):
    import torch
    from training_worker import _finite, _tensor_digest

    require(isinstance(fp32_adapter, dict) and bool(fp32_adapter), "nonempty fp32_adapter required")
    t0 = time.perf_counter_ns()
    bf16_tensors = {}
    for key_name, ten_val in sorted(fp32_adapter.items()):
        require(isinstance(ten_val, torch.Tensor) and ten_val.ndim == 2, "2D adapter tensor required")
        converted = ten_val.detach().to(dtype=torch.bfloat16, device="cpu").contiguous().clone()
        bf16_tensors[key_name] = converted
    _finite(bf16_tensors.values())
    total_bytes = sum(ten.numel() * ten.element_size() for ten in bf16_tensors.values())
    return {
        "kind": "LORA_COMPACT_BF16_ADAPTER_V1",
        "bf16_tensors": bf16_tensors,
        "bf16_bytes": total_bytes,
        "bf16_sha256": _tensor_digest(bf16_tensors),
        "pack_ns": time.perf_counter_ns() - t0,
    }


@contextmanager
def merged_bf16_adapter(model, bf16_pack, *, alpha=16.0, rank=8):
    import torch
    from training_worker import _finite, _state, _targets, _tensor_digest

    require(
        isinstance(bf16_pack, dict) and bf16_pack.get("kind") == "LORA_COMPACT_BF16_ADAPTER_V1",
        "valid bf16_pack required",
    )
    bf16_tensors = bf16_pack["bf16_tensors"]
    require(_tensor_digest(bf16_tensors) == bf16_pack["bf16_sha256"], "bf16_pack digest mismatch")

    base_state = _state(model)
    device = next(iter({ten.device for ten in base_state.values()}))
    _finite(base_state.values())
    targets = _targets(model)
    base_sha256 = _tensor_digest(base_state)
    scale_factor = float(alpha) / float(rank)

    backups = []
    rng_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        try:
            model.requires_grad_(False)
            model.eval()
            for path_str, base_lin in targets:
                mat_a = bf16_tensors[path_str + ".adapter_a"].to(device=device, dtype=torch.float32)
                mat_b = bf16_tensors[path_str + ".adapter_b"].to(device=device, dtype=torch.float32)
                backups.append((base_lin, base_lin.weight.data.detach().cpu().clone()))
                merged_w = (base_lin.weight.data.float() + scale_factor * torch.matmul(mat_b, mat_a)).to(
                    dtype=torch.bfloat16
                )
                base_lin.weight.data.copy_(merged_w)
                del mat_a, mat_b, merged_w
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            yield model
        finally:
            for base_lin, orig_cpu_w in reversed(backups):
                base_lin.weight.data.copy_(orig_cpu_w.to(device=device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            require(
                _tensor_digest(_state(model)) == base_sha256,
                "base state changed after reversible BF16 unmerge",
            )


def _cached_greedy_generate_candidate(model, tokenizer, messages, *, expected_tokenizer_sha256):
    import torch
    from tokenizer_measurement import _state as tok_state_fn

    t0 = time.perf_counter_ns()
    measured = measure_tokens(tokenizer, messages, answer=None)
    require(measured["state_sha256"] == expected_tokenizer_sha256, "tokenizer sha mismatch")
    require(
        measured["status"] == "OK" and measured["input"]["input_tokens"] <= 4096,
        "input token budget exceeded",
    )
    prompt_ids = list(measured["input"]["input_ids"])
    eos_id = int(tokenizer.eos_token_id)
    device = next(model.parameters()).device
    rng_devices = [device.index or 0] if device.type == "cuda" else []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    orig_cache_flag = getattr(model.config, "use_cache", False)
    gen_ids = []
    fwd_calls = 0
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
            fwd_calls += 1
            past_kv = getattr(out_obj, "past_key_values", None)
            next_tok = int(torch.argmax(out_obj.logits[0, -1]).item())
            gen_ids.append(next_tok)
            del out_obj, inp_t

            while next_tok != eos_id and len(gen_ids) < 256:
                step_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)
                step_out = model(
                    input_ids=step_in,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True,
                )
                fwd_calls += 1
                past_kv = getattr(step_out, "past_key_values", None)
                next_tok = int(torch.argmax(step_out.logits[0, -1]).item())
                gen_ids.append(next_tok)
                del step_out, step_in
            del past_kv
        finally:
            model.config.use_cache = orig_cache_flag

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_resv = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_alloc = 0
        peak_resv = 0

    stop_eos = gen_ids[-1] == eos_id
    decode_ids = gen_ids[:-1] if stop_eos else list(gen_ids)
    candidate_str = tokenizer.decode(
        decode_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    require(digest(tok_state_fn(tokenizer)[0]) == expected_tokenizer_sha256, "tokenizer mutated")
    wall_ns = time.perf_counter_ns() - t0
    return {
        "candidate": candidate_str,
        "cost": {
            "input_tokens": len(prompt_ids),
            "output_tokens": len(gen_ids),
            "forward_calls": fwd_calls,
            "wall_ns": wall_ns,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_resv,
        },
    }


def run_kv_cache_serving_benchmark(checkpoint_root, bf16_pack, device):
    import torch
    import transformers

    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    sdpa_model = None
    try:
        sdpa_model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint_root,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval().requires_grad_(False)
        sdpa_model.config.use_cache = True
        torch.cuda.synchronize(device)
        base_resident_bytes = int(torch.cuda.memory_allocated(device))

        fwd_params = inspect.signature(sdpa_model.forward).parameters
        keep_kw = {}
        if "logits_to_keep" in fwd_params:
            keep_kw = {"logits_to_keep": 1}
        elif "num_logits_to_keep" in fwd_params:
            keep_kw = {"num_logits_to_keep": 1}

        prefix_1d = torch.tensor(
            [[100 + (idx_t % 500) for idx_t in range(BENCHMARK_PREFIX_TOKENS)]],
            dtype=torch.long,
            device=device,
        )
        query_1d = torch.tensor(
            [[700 + (idx_t % 300) for idx_t in range(BENCHMARK_QUERY_TOKENS)]],
            dtype=torch.long,
            device=device,
        )
        full_1d = torch.cat([prefix_1d, query_1d], dim=1)

        def safe_run_bench(fn_call, mod_ref, batch_sz):
            try:
                row_out = fn_call(mod_ref, batch_sz)
                return {"status": "OK", **row_out}
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                return {"batch_size": batch_sz, "status": "CUDA_OOM"}

        def bench_cold_long(mod_ref, batch_sz):
            full_b = full_1d.expand(batch_sz, -1).contiguous()
            ttft_list = []
            tpot_list = []
            total_list = []
            peak_kv_delta = 0
            for trial_idx in range(3):
                torch.cuda.synchronize(device)
                before_mem = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                t_start = time.perf_counter_ns()
                with torch.no_grad():
                    out_p = mod_ref(input_ids=full_b, use_cache=True, return_dict=True, **keep_kw)
                    pkv = out_p.past_key_values
                    nxt = torch.argmax(out_p.logits[:, -1:], dim=-1)
                    del out_p
                    torch.cuda.synchronize(device)
                    t_pref = time.perf_counter_ns()
                    for _step_i in range(BENCHMARK_DECODE_TOKENS - 1):
                        out_d = mod_ref(input_ids=nxt, past_key_values=pkv, use_cache=True, return_dict=True, **keep_kw)
                        pkv = out_d.past_key_values
                        nxt = torch.argmax(out_d.logits[:, -1:], dim=-1)
                        del out_d
                    torch.cuda.synchronize(device)
                    t_end = time.perf_counter_ns()
                    peak_mem = int(torch.cuda.max_memory_allocated(device))
                    del pkv, nxt
                if trial_idx >= 1:
                    ttft_ms = (t_pref - t_start) / 1_000_000.0
                    dec_ms = (t_end - t_pref) / 1_000_000.0
                    ttft_list.append(ttft_ms)
                    tpot_list.append(dec_ms / float(BENCHMARK_DECODE_TOKENS - 1))
                    total_list.append((t_end - t_start) / 1_000_000.0)
                    peak_kv_delta = max(peak_kv_delta, peak_mem - before_mem)
            med_tot_s = statistics.median(total_list) / 1000.0
            return {
                "batch_size": batch_sz,
                "median_ttft_ms": statistics.median(ttft_list),
                "median_decode_tpot_ms": statistics.median(tpot_list),
                "median_total_ms": statistics.median(total_list),
                "output_tokens_per_sec": (batch_sz * BENCHMARK_DECODE_TOKENS) / med_tot_s,
                "decode_only_tokens_per_sec": (batch_sz * (BENCHMARK_DECODE_TOKENS - 1)) / ((statistics.median(tpot_list) * (BENCHMARK_DECODE_TOKENS - 1)) / 1000.0),
                "peak_extra_vram_mib": peak_kv_delta / (1024.0 * 1024.0),
            }

        def bench_warm_prefix_cache(mod_ref, batch_sz):
            pref_b = prefix_1d.expand(batch_sz, -1).contiguous()
            qry_b = query_1d.expand(batch_sz, -1).contiguous()
            ttft_list = []
            tpot_list = []
            total_list = []
            peak_kv_delta = 0
            for trial_idx in range(3):
                torch.cuda.synchronize(device)
                before_mem = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                with torch.no_grad():
                    warm_out = mod_ref(input_ids=pref_b, use_cache=True, return_dict=True, **keep_kw)
                    warm_pkv = warm_out.past_key_values
                    del warm_out
                    torch.cuda.synchronize(device)
                    t_start = time.perf_counter_ns()
                    out_q = mod_ref(input_ids=qry_b, past_key_values=warm_pkv, use_cache=True, return_dict=True, **keep_kw)
                    pkv = out_q.past_key_values
                    nxt = torch.argmax(out_q.logits[:, -1:], dim=-1)
                    del out_q
                    torch.cuda.synchronize(device)
                    t_pref = time.perf_counter_ns()
                    for _step_i in range(BENCHMARK_DECODE_TOKENS - 1):
                        out_d = mod_ref(input_ids=nxt, past_key_values=pkv, use_cache=True, return_dict=True, **keep_kw)
                        pkv = out_d.past_key_values
                        nxt = torch.argmax(out_d.logits[:, -1:], dim=-1)
                        del out_d
                    torch.cuda.synchronize(device)
                    t_end = time.perf_counter_ns()
                    peak_mem = int(torch.cuda.max_memory_allocated(device))
                    del pkv, warm_pkv, nxt
                if trial_idx >= 1:
                    ttft_ms = (t_pref - t_start) / 1_000_000.0
                    dec_ms = (t_end - t_pref) / 1_000_000.0
                    ttft_list.append(ttft_ms)
                    tpot_list.append(dec_ms / float(BENCHMARK_DECODE_TOKENS - 1))
                    total_list.append((t_end - t_start) / 1_000_000.0)
                    peak_kv_delta = max(peak_kv_delta, peak_mem - before_mem)
            med_tot_s = statistics.median(total_list) / 1000.0
            return {
                "batch_size": batch_sz,
                "median_ttft_ms": statistics.median(ttft_list),
                "median_decode_tpot_ms": statistics.median(tpot_list),
                "median_total_ms": statistics.median(total_list),
                "output_tokens_per_sec": (batch_sz * BENCHMARK_DECODE_TOKENS) / med_tot_s,
                "decode_only_tokens_per_sec": (batch_sz * (BENCHMARK_DECODE_TOKENS - 1)) / ((statistics.median(tpot_list) * (BENCHMARK_DECODE_TOKENS - 1)) / 1000.0),
                "peak_extra_vram_mib": peak_kv_delta / (1024.0 * 1024.0),
            }

        def bench_promoted_lora_short(mod_ref, batch_sz):
            qry_b = query_1d.expand(batch_sz, -1).contiguous()
            ttft_list = []
            tpot_list = []
            total_list = []
            peak_kv_delta = 0
            for trial_idx in range(3):
                torch.cuda.synchronize(device)
                before_mem = int(torch.cuda.memory_allocated(device))
                torch.cuda.reset_peak_memory_stats(device)
                t_start = time.perf_counter_ns()
                with torch.no_grad():
                    out_q = mod_ref(input_ids=qry_b, use_cache=True, return_dict=True, **keep_kw)
                    pkv = out_q.past_key_values
                    nxt = torch.argmax(out_q.logits[:, -1:], dim=-1)
                    del out_q
                    torch.cuda.synchronize(device)
                    t_pref = time.perf_counter_ns()
                    for _step_i in range(BENCHMARK_DECODE_TOKENS - 1):
                        out_d = mod_ref(input_ids=nxt, past_key_values=pkv, use_cache=True, return_dict=True, **keep_kw)
                        pkv = out_d.past_key_values
                        nxt = torch.argmax(out_d.logits[:, -1:], dim=-1)
                        del out_d
                    torch.cuda.synchronize(device)
                    t_end = time.perf_counter_ns()
                    peak_mem = int(torch.cuda.max_memory_allocated(device))
                    del pkv, nxt
                if trial_idx >= 1:
                    ttft_ms = (t_pref - t_start) / 1_000_000.0
                    dec_ms = (t_end - t_pref) / 1_000_000.0
                    ttft_list.append(ttft_ms)
                    tpot_list.append(dec_ms / float(BENCHMARK_DECODE_TOKENS - 1))
                    total_list.append((t_end - t_start) / 1_000_000.0)
                    peak_kv_delta = max(peak_kv_delta, peak_mem - before_mem)
            med_tot_s = statistics.median(total_list) / 1000.0
            return {
                "batch_size": batch_sz,
                "median_ttft_ms": statistics.median(ttft_list),
                "median_decode_tpot_ms": statistics.median(tpot_list),
                "median_total_ms": statistics.median(total_list),
                "output_tokens_per_sec": (batch_sz * BENCHMARK_DECODE_TOKENS) / med_tot_s,
                "decode_only_tokens_per_sec": (batch_sz * (BENCHMARK_DECODE_TOKENS - 1)) / ((statistics.median(tpot_list) * (BENCHMARK_DECODE_TOKENS - 1)) / 1000.0),
                "peak_extra_vram_mib": peak_kv_delta / (1024.0 * 1024.0),
            }

        cold_rows = []
        warm_rows = []
        for b_sz in BENCHMARK_BATCH_SIZES:
            cold_rows.append(safe_run_bench(bench_cold_long, sdpa_model, b_sz))
            gc.collect()
            torch.cuda.empty_cache()
            warm_rows.append(safe_run_bench(bench_warm_prefix_cache, sdpa_model, b_sz))
            gc.collect()
            torch.cuda.empty_cache()

        lora_rows = []
        with merged_bf16_adapter(sdpa_model, bf16_pack, alpha=16.0, rank=8) as merged_sdpa:
            for b_sz in BENCHMARK_BATCH_SIZES:
                lora_rows.append(safe_run_bench(bench_promoted_lora_short, merged_sdpa, b_sz))
                gc.collect()
                torch.cuda.empty_cache()

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        usable_pool_bytes = int(free_bytes * 0.85)
        bytes_per_3k_prefix = 36 * 2 * 8 * BENCHMARK_PREFIX_TOKENS * 128 * 2
        bytes_per_16k_prefix = 36 * 2 * 8 * 16384 * 128 * 2
        bytes_per_bf16_lora = int(bf16_pack["bf16_bytes"])

        return {
            "attention_backend": "sdpa",
            "prefix_tokens": BENCHMARK_PREFIX_TOKENS,
            "query_tokens": BENCHMARK_QUERY_TOKENS,
            "decode_tokens": BENCHMARK_DECODE_TOKENS,
            "base_model_resident_mib": base_resident_bytes / (1024.0 * 1024.0),
            "usable_85pct_kv_pool_mib": usable_pool_bytes / (1024.0 * 1024.0),
            "cold_long_prompt_kv": cold_rows,
            "warm_prefix_kv_cache": warm_rows,
            "promoted_bf16_lora_merged": lora_rows,
            "multi_tenant_residency_capacity": {
                "bf16_lora_bytes": bytes_per_bf16_lora,
                "prefix_3050tok_kv_bytes": bytes_per_3k_prefix,
                "prefix_16384tok_kv_bytes": bytes_per_16k_prefix,
                "compression_ratio_vs_3050tok_kv": bytes_per_3k_prefix / float(bytes_per_bf16_lora),
                "compression_ratio_vs_16384tok_kv": bytes_per_16k_prefix / float(bytes_per_bf16_lora),
                "max_concurrent_3050tok_prefixes_on_a100": usable_pool_bytes // bytes_per_3k_prefix,
                "max_concurrent_16384tok_prefixes_on_a100": usable_pool_bytes // bytes_per_16k_prefix,
                "max_concurrent_bf16_loras_on_a100": usable_pool_bytes // bytes_per_bf16_lora,
            },
        }
    finally:
        del sdpa_model
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(prev_det)


def build_g11_schedule(prompts_raw, supervision_raw, *, run_id):
    _check_hex(run_id, 32)
    prompts = parse_json(prompts_raw)
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
            "arm": "lora_bf16_merged_kv",
            "render_arm": "lora",
            "first_messages": f_msgs,
            "first_messages_sha256": digest(f_msgs),
        })
    return {
        "kind": "LORA_GATE_G11_SYSTEM_SCHEDULE",
        "version": 1,
        "run_id": run_id,
        "episodes": episodes,
    }


def build_gate_g11_system_package(
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
    require(set(dep_sources) == G11_SOURCE_NAMES, "G11 source names")
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    require(len(scoring_run_ids) == 13, "13 scoring_run_ids")

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    sources_sha = {fname: _sha256(dep_sources[fname]) for fname in sorted(G11_SOURCE_NAMES)}
    payload_raw = encoded({
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(G11_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision.json"]).decode("ascii"),
        "anchor_public_b64": base64.b64encode(anchor_public_raw).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    })

    boot_bytes = G11_STDLIB_BOOTSTRAP.encode("ascii")
    boot_sha = _sha256(boot_bytes)
    p1_cfg = {
        "kind": "LORA_GATE_G11_SYSTEM_BENCHMARK_CONFIG",
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
        "stage": str(Path(remote_parent) / ("g11-sys-bench-" + run_id)),
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
        "ns={'__name__':'g11_boot'}\n"
        "exec(compile(src,'<g11_stdlib_bootstrap>','exec'),ns)\n"
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
        "kind": "LORA_GATE_G11_SYSTEM_BENCHMARK_PACKAGE",
        "version": 1,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
        "scoring_run_ids": list(scoring_run_ids),
        "hyperparams": copy.deepcopy(chosen_hp),
        "benchmark_batch_sizes": list(BENCHMARK_BATCH_SIZES),
        "file_sha256": dict(CONFIRMATORY_FILE_PINS),
        "phase_one_config": p1_cfg,
        "phase_one_config_sha256": p1_cfg_sha,
        "phase_one_argv_sha256": digest(p1_ssh_argv),
        "phase_two_argv_sha256": digest(p2_ssh_argv),
        "local_run_root": local_run_root,
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


def execute_remote_g11_system_body(config, stage, bundle):
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
        "G11 input pin mismatch",
    )
    anchor_sets = parse_json(anchor_pub_raw)
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

    active_hp = copy.deepcopy(config.get("hyperparams") or DIFF_FOCUSED_HYPERPARAMS)
    compile_t0 = time.monotonic_ns()
    anc_t0 = time.monotonic_ns()
    bg_trajs = prepare_fast_background_trajectories(
        model,
        tokenizer,
        anchor_sets["background"],
        max_new_tokens=int(active_hp["max_anchor_tokens"]),
    )
    shared_costs["fast_anchor_trajectory_prep_ns"] = time.monotonic_ns() - anc_t0

    train_ds = prepare_diff_focused_training(tokenizer, supervision_raw, hyperparams=active_hp)
    train_out = train_diff_focused_kl_adapter(model, train_ds, bg_trajs, hyperparams=active_hp)
    del bg_trajs

    bf16_pack = pack_adapter_to_bf16(train_out["adapter"])
    buf_io = io.BytesIO()
    torch.save({"bf16_tensors": bf16_pack["bf16_tensors"]}, buf_io)
    bf16_file_bytes = buf_io.getvalue()
    write_private(stage / "compact_bf16_adapter.pt", bf16_file_bytes)
    shared_costs["end_to_end_fast_compile_ns"] = time.monotonic_ns() - compile_t0

    dev_neg_trajs = prepare_fast_background_trajectories(
        model,
        tokenizer,
        anchor_sets["dev_negative"],
        max_new_tokens=int(active_hp["max_anchor_tokens"]),
    )

    prompts_map = {item["id"]: item for item in parse_json(prompts_raw)}
    schedule = build_g11_schedule(prompts_raw, supervision_raw, run_id=config["run_id"])
    phase_one_records = []
    heldout_dev_restraint_audit = None

    p1_t0 = time.monotonic_ns()
    for ep_spec in schedule["episodes"][:2]:
        tid = ep_spec["task_id"]
        task_prompt = prompts_map[tid]
        init_res = _cached_greedy_generate_candidate(
            model, tokenizer, ep_spec["first_messages"], expected_tokenizer_sha256=observed_tok_sha
        )
        init_cand = init_res["candidate"]
        if candidate_valid(task_prompt, init_cand):
            pub_fb, pub_ns = _run_public_cases_child(
                config["resolved_interpreter"],
                backend_raw,
                worker_text,
                config["expected_runtime"],
                config["run_id"],
                ep_spec["episode_index"],
                task_prompt,
                init_cand,
            )
        else:
            pub_fb = {"kind": "invalid_source"}
            pub_ns = 0
        sec_msgs = confirmatory_second_round(
            task_prompt, ep_spec["render_arm"], supervision_raw, init_cand, pub_fb
        )
        fb_res = _cached_greedy_generate_candidate(
            model, tokenizer, sec_msgs, expected_tokenizer_sha256=observed_tok_sha
        )
        phase_one_records.append({
            "episode_index": ep_spec["episode_index"],
            "task_id": tid,
            "split": ep_spec["split"],
            "arm": ep_spec["arm"],
            "candidates": {"initial": init_cand, "feedback": fb_res["candidate"]},
            "public_feedback": pub_fb,
            "generation_costs": {"initial": init_res["cost"], "feedback": fb_res["cost"]},
            "public_execution_ns": pub_ns,
        })

    with merged_bf16_adapter(
        model,
        bf16_pack,
        alpha=float(active_hp["alpha"]),
        rank=int(active_hp["rank"]),
    ) as merged_mod:
        heldout_dev_restraint_audit = audit_kl_restraint_on_trajectories(merged_mod, dev_neg_trajs)
        del dev_neg_trajs
        for ep_spec in schedule["episodes"][2:]:
            tid = ep_spec["task_id"]
            task_prompt = prompts_map[tid]
            init_res = _cached_greedy_generate_candidate(
                merged_mod, tokenizer, ep_spec["first_messages"], expected_tokenizer_sha256=observed_tok_sha
            )
            init_cand = init_res["candidate"]
            if candidate_valid(task_prompt, init_cand):
                pub_fb, pub_ns = _run_public_cases_child(
                    config["resolved_interpreter"],
                    backend_raw,
                    worker_text,
                    config["expected_runtime"],
                    config["run_id"],
                    ep_spec["episode_index"],
                    task_prompt,
                    init_cand,
                )
            else:
                pub_fb = {"kind": "invalid_source"}
                pub_ns = 0
            sec_msgs = confirmatory_second_round(
                task_prompt, ep_spec["render_arm"], supervision_raw, init_cand, pub_fb
            )
            fb_res = _cached_greedy_generate_candidate(
                merged_mod, tokenizer, sec_msgs, expected_tokenizer_sha256=observed_tok_sha
            )
            phase_one_records.append({
                "episode_index": ep_spec["episode_index"],
                "task_id": tid,
                "split": ep_spec["split"],
                "arm": ep_spec["arm"],
                "candidates": {"initial": init_cand, "feedback": fb_res["candidate"]},
                "public_feedback": pub_fb,
                "generation_costs": {"initial": init_res["cost"], "feedback": fb_res["cost"]},
                "public_execution_ns": pub_ns,
            })

    shared_costs["cached_episodes_wall_ns"] = time.monotonic_ns() - p1_t0

    fin_t0 = time.monotonic_ns()
    final_base_sha = tensor_digest_fn(model_state_fn(model))
    final_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
    rng_same = bool(
        torch.equal(init_cpu_rng, torch.random.get_rng_state())
        and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
    )
    if final_base_sha != init_base_sha or final_tok_sha != observed_tok_sha or not rng_same:
        raise RuntimeError("G11 post-unmerge integrity drift")
    shared_costs["final_integrity_ns"] = time.monotonic_ns() - fin_t0

    loaded.clear()
    del loaded, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    bench_t0 = time.monotonic_ns()
    kv_serving_benchmark = run_kv_cache_serving_benchmark(
        config["checkpoint_root"], bf16_pack, device
    )
    shared_costs["sdpa_kv_benchmark_ns"] = time.monotonic_ns() - bench_t0

    training_summary = {
        "hyperparams": train_out["hyperparams"],
        "fp32_adapter_bytes": train_out["adapter_bytes"],
        "bf16_adapter_bytes": bf16_pack["bf16_bytes"],
        "bf16_file_bytes": len(bf16_file_bytes),
        "bf16_sha256": bf16_pack["bf16_sha256"],
        "pack_ns": bf16_pack["pack_ns"],
        "training_ns": train_out["training_ns"],
        "core_wall_ns": train_out["core_wall_ns"],
        "epoch_mean_weighted_sup_losses": train_out["epoch_mean_weighted_sup_losses"],
        "epoch_mean_bg_kl_losses": train_out["epoch_mean_bg_kl_losses"],
    }

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
        "kv_serving_benchmark": kv_serving_benchmark,
        "shared_costs": shared_costs,
        "phase_one_records": phase_one_records,
    }


def score_g11_system_stage(
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
    kv_serving_benchmark,
    shared_costs,
    g4b_report,
    g8_report,
):
    dataset = load_confirmatory_dataset(raw_files)
    prompts_map = {item["id"]: item for item in dataset["prompts"]}
    oracles_map = {item["id"]: item for item in dataset["oracles"]}
    worker_text = worker_raw.decode("utf-8")

    proto_bad = []
    inc_reasons = []
    if len(phase_one_records) != 26:
        inc_reasons.append("incomplete_phase_one_episodes")

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
                        "request_id": "g11-p2-%04d" % len(valid_positions),
                        "source": cand_src,
                        "entry_point": task_prompt["entry_point"],
                        "args": copy.deepcopy(case_obj["args"]),
                    }
                    if is_val
                    else None
                )
                entry = {
                    "position_index": pos_idx,
                    "valid_source": is_val,
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

    g4b_eps = {(ep["task_id"], ep["arm"]): ep for ep in g4b_report["episodes"]}
    g8_eps = {(ep["task_id"], "lora_diff_kl"): ep for ep in g8_report["episodes"]}

    merged_eps = {}
    cached_gen_wall_ns = 0
    for ep_idx, ep_rec in enumerate(phase_one_records):
        tid = ep_rec["task_id"]
        arm = ep_rec["arm"]
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
            roles_out[role_name] = cls_res

        if arm == "lora_bf16_merged_kv":
            cached_gen_wall_ns += (
                ep_rec["generation_costs"]["initial"]["wall_ns"]
                + ep_rec["generation_costs"]["feedback"]["wall_ns"]
            )
            merged_eps[(tid, "lora_bf16_merged_kv")] = {
                "task_id": tid,
                "split": ep_rec["split"],
                "arm": "lora_bf16_merged_kv",
                "roles": roles_out,
            }

    run_ok = bool(not proto_bad and not inc_reasons and len(merged_eps) == 24)
    all_indexed = {**g4b_eps, **g8_eps, **merged_eps}

    by_split = {}
    comparisons = {}
    for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
        by_split[sp_name] = {
            role_name: {
                st_n: sum(
                    (tid, "lora_bf16_merged_kv") in merged_eps
                    and merged_eps[(tid, "lora_bf16_merged_kv")]["roles"][role_name]["state"] == st_n
                    for tid in sp_ids
                )
                for st_n in CANDIDATE_STATES
            }
            for role_name in CONFIRMATORY_ROLES
        }
        comparisons[sp_name] = {
            "merged_bf16_kv_vs_base": _paired_arm_comparison(
                sp_ids, all_indexed, "feedback", "base", "lora_bf16_merged_kv", run_ok
            ),
            "merged_bf16_kv_vs_fp32_lora_diff_kl": _paired_arm_comparison(
                sp_ids, all_indexed, "feedback", "lora_diff_kl", "lora_bf16_merged_kv", run_ok
            ),
            "merged_bf16_kv_vs_text": _paired_arm_comparison(
                sp_ids, all_indexed, "feedback", "text", "lora_bf16_merged_kv", run_ok
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

    return {
        "kind": "LORA_GATE_G11_SYSTEM_BENCHMARK_REPORT",
        "version": 1,
        "run_id": run_id,
        "complete_and_verified": run_ok,
        "protocol_contradictions": sorted(set(proto_bad)),
        "incomplete_reasons": sorted(set(inc_reasons)),
        "heldout_dev_restraint_audit": heldout_dev_restraint_audit,
        "total_eval_tasks": 24,
        "total_pass_initial": total_pass_init,
        "total_pass_feedback": total_pass_fb,
        "training_and_packing": training_summary,
        "merged_bf16_kv_by_split": by_split,
        "comparisons_feedback": comparisons,
        "kv_serving_benchmark": kv_serving_benchmark,
        "costs": {
            "end_to_end_fast_compile_ns": shared_costs.get("end_to_end_fast_compile_ns"),
            "fast_anchor_trajectory_prep_ns": shared_costs.get("fast_anchor_trajectory_prep_ns"),
            "cached_48_generations_wall_ns": cached_gen_wall_ns,
            "bf16_adapter_bytes": training_summary["bf16_adapter_bytes"],
            "shared": shared_costs,
        },
        **{flag_n: False for flag_n in FALSE_FLAGS},
    }


def run_gate_g11_system_stage(
    package,
    raw_files,
    worker_raw,
    g4b_report,
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
        safe_write("g11-package-public.json", _pretty_json(pub))
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
                    "G11 READY mismatch",
                )
                done_obj = strict_json(frames["DONE"])
                require(
                    done_obj == {
                        "config_sha256": pub["phase_one_config_sha256"],
                        "result_sha256": _sha256(frames["RESULT"]),
                    },
                    "G11 DONE mismatch",
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
                                "request_id": "g11-p2-%04d" % len(valid_positions),
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
                conts_by_batch = {}
                for cit in env_out["continuations"]:
                    conts_by_batch.setdefault(cit["batch_index"], []).append(cit)

                for bm, irec in zip(built_b, env_out["results"]):
                    b_idx = bm["batch_index"]
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
                report = score_g11_system_stage(
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
                    body_obj["kv_serving_benchmark"],
                    body_obj["shared_costs"],
                    g4b_report,
                    g8_report,
                )
                safe_write("g11-system-benchmark-report.json", _pretty_json(report))

        p1_clean = bool(p1_out and p1_out["reaped"] and p1_out["group_absent"] and p1_out["cleanup_bounded"])
        binding_ok = bool(p1_clean and p2_all_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_GATE_G11_EXECUTION_EVIDENCE",
            "version": 1,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and report and report["complete_and_verified"]),
            "secondary_errors": secondary_errors,
            **{flag_n: False for flag_n in FALSE_FLAGS},
        }
        safe_write("g11-controller-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": report}
