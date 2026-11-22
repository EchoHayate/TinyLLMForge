import base64
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shlex
import signal
import socket
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
from confirmatory_v2_training import validate_confirmatory_training
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


ABLATION_TRAINING_HYPERPARAMS = {
    "rank": 8,
    "alpha": 16,
    "targets": ["q_proj", "v_proj"],
    "optimizer": "AdamW",
    "learning_rate": 0.0002,
    "lr_schedule": "linear_decay_to_10pct",
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "seed": 42,
    "epochs": 12,
    "steps": 96,
}

ABLATION_SOURCE_NAMES = frozenset({
    "confirmatory_v2_contract.py",
    "confirmatory_v2_converged_ablation.py",
    "confirmatory_v2_gate_a.py",
    "confirmatory_v2_remote.py",
    "confirmatory_v2_scoring.py",
    "confirmatory_v2_training.py",
    "generation_core.py",
    "observed_generation.py",
    "pilot_protocol.py",
    "prompt_rendering.py",
    "recalibration_data_contract.py",
    "recalibration_diagnostic_controller.py",
    "recalibration_diagnostic_remote.py",
    "recalibration_diagnostic_scoring.py",
    "recalibration_execution_evidence.py",
    "recalibration_execution_package.py",
    "recalibration_execution_remote.py",
    "recalibration_execution_wire.py",
    "recalibration_rendering.py",
    "recalibration_validation.py",
    "tokenizer_measurement.py",
    "training_worker.py",
    "validation_contract.py",
    "validation_real_report.py",
    "validation_report.py",
})

G5_STDLIB_BOOTSTRAP = """import base64
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
    with os.fdopen(fdesc, "wb") as st:
        os.fchmod(st.fileno(), 0o600)
        st.write(raw)
        st.flush()
        os.fsync(st.fileno())


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
    stage = p_path / ("conv-abl-" + cfg["run_id"])
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

        prompts_raw = base64.b64decode(bundle["prompts_b64"], validate=True)
        sup_raw = base64.b64decode(bundle["supervision_b64"], validate=True)
        back_raw = base64.b64decode(bundle["backend_b64"], validate=True)
        work_raw = base64.b64decode(bundle["worker_b64"], validate=True)
        if (
            sha_hex(prompts_raw) != cfg["prompts_sha256"]
            or sha_hex(sup_raw) != cfg["supervision_sha256"]
            or sha_hex(back_raw) != cfg["backend_sha256"]
            or sha_hex(work_raw) != cfg["worker_sha256"]
        ):
            raise ValueError("input data/worker pin mismatch")

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
        import confirmatory_v2_converged_ablation as conv_mod
        body_res = conv_mod.execute_remote_converged_body(
            cfg, stage, prompts_raw, sup_raw, back_raw, work_raw
        )
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
        "kind": "LORA_CONVERGED_ABLATION_PHASE1_RESULT",
        "version": 1,
        "identity": ident,
        "bootstrap_failure": boot_fail,
        "gpu_selected": body_res["gpu_selected"] if body_res else None,
        "integrity": body_res["integrity"] if body_res else None,
        "training_summary": body_res["training_summary"] if body_res else None,
        "shared_costs": body_res["shared_costs"] if body_res else {},
        "phase_one_records": body_res["phase_one_records"] if body_res else [],
    })
    write_fr(out_fd, "RESULT", res_payload)
    write_fr(out_fd, "STDOUT", d_out)
    write_fr(out_fd, "STDERR", d_err)
    write_fr(out_fd, "DONE", enc({"config_sha256": ident["config_sha256"], "result_sha256": sha_hex(res_payload)}))
    return 0 if boot_fail is None else 1
"""


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def train_converged_ablation_adapter(model, dataset):
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

    started = time.perf_counter_ns()
    require(
        torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
        "autograd required",
    )
    require(
        not torch.is_autocast_enabled("cpu") and not torch.is_autocast_enabled("cuda"),
        "disabled autocast required",
    )
    copied_ds = copy.deepcopy(dataset)
    dataset_sha256 = validate_confirmatory_training(copied_ds)
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
        torch.default_generator.manual_seed(ABLATION_TRAINING_HYPERPARAMS["seed"])
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(ABLATION_TRAINING_HYPERPARAMS["seed"])
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
                lr=ABLATION_TRAINING_HYPERPARAMS["learning_rate"],
                weight_decay=ABLATION_TRAINING_HYPERPARAMS["weight_decay"],
            )
            _optimizer_parameters(optimizer, trainable_params)
            initial_adapter = _adapter_state(wrappers)
            _synchronize(device)
            train_t0 = time.perf_counter_ns()
            prep_ns = train_t0 - started

            total_steps = ABLATION_TRAINING_HYPERPARAMS["steps"]
            for step_idx in range(total_steps):
                cur_lr = ABLATION_TRAINING_HYPERPARAMS["learning_rate"] * (
                    1.0 - (0.9 * step_idx / max(1, total_steps - 1))
                )
                for pgroup in optimizer.param_groups:
                    pgroup["lr"] = cur_lr
                row = copied_ds["rows"][step_idx % 8]
                step_t0 = time.perf_counter_ns()
                inp_t = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                msk_t = torch.tensor([row["attention_mask"]], dtype=torch.long, device=device)
                lbl_t = torch.tensor([row["labels"]], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                out_obj = model(input_ids=inp_t, attention_mask=msk_t, use_cache=False, return_dict=True)
                loss_val = answer_loss(out_obj.logits, lbl_t)
                _finite([loss_val])
                loss_val.backward()
                _finite(par_obj.grad for par_obj in trainable_params)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_params, ABLATION_TRAINING_HYPERPARAMS["gradient_clip"], error_if_nonfinite=True
                )
                _finite([grad_norm])
                optimizer.step()
                _finite(trainable_params)
                _synchronize(device)
                steps_log.append({
                    "step": step_idx + 1,
                    "epoch": (step_idx // 8) + 1,
                    "lr": cur_lr,
                    "task_id": row["task_id"],
                    "loss": float(loss_val.item()),
                    "gradient_norm": float(grad_norm.item()),
                    "elapsed_ns": time.perf_counter_ns() - step_t0,
                })
                del inp_t, msk_t, lbl_t, out_obj, loss_val, grad_norm

            _synchronize(device)
            training_ns = time.perf_counter_ns() - train_t0
            verify_t0 = time.perf_counter_ns()
            final_adapter = _adapter_state(wrappers)
            if device.type == "cuda":
                peak_alloc = int(torch.cuda.max_memory_allocated(device))
                peak_resv = int(torch.cuda.max_memory_reserved(device))
        finally:
            for parent_mod, attr_name, orig_lin in reversed(originals):
                setattr(parent_mod, attr_name, orig_lin)
            for par_item, req_g in grad_flags:
                par_item.requires_grad_(req_g)
            for mod_item, was_t in modes:
                mod_item.training = was_t
            if _tensor_digest(_state(model)) != base_sha256:
                raise ValueError("base state mutated during converged ablation training")

    init_ad_sha = _tensor_digest(initial_adapter)
    fin_ad_sha = _tensor_digest(final_adapter)
    require(fin_ad_sha != init_ad_sha, "converged adapter did not update")
    _synchronize(device)
    finished = time.perf_counter_ns()
    return {
        "kind": "LORA_CONVERGED_ABLATION_ADAPTER_RESULT",
        "hyperparams": copy.deepcopy(ABLATION_TRAINING_HYPERPARAMS),
        "dataset_sha256": dataset_sha256,
        "base_sha256": base_sha256,
        "target_paths": [path_str for path_str, _lin in targets],
        "initial_adapter_sha256": init_ad_sha,
        "adapter_sha256": fin_ad_sha,
        "adapter_bytes": sum(ten.numel() * ten.element_size() for ten in final_adapter.values()),
        "preparation_ns": prep_ns,
        "training_ns": training_ns,
        "verification_ns": finished - verify_t0,
        "core_wall_ns": finished - started,
        "peak_allocated_bytes": peak_alloc,
        "peak_reserved_bytes": peak_resv,
        "steps": steps_log,
        "adapter": final_adapter,
    }


@contextmanager
def mounted_converged_adapter(model, train_out):
    import torch
    from training_worker import (
        LoRAProjection,
        _adapter_state,
        _finite,
        _state,
        _targets,
        _tensor_digest,
    )

    base_state = _state(model)
    device = next(iter({ten.device for ten in base_state.values()}))
    _finite(base_state.values())
    targets = _targets(model)
    base_sha256 = _tensor_digest(base_state)
    require(base_sha256 == train_out["base_sha256"], "base digest mismatch on mount")
    selected = train_out["adapter"]
    selected_sha = train_out["adapter_sha256"]

    modes = [(mod_item, mod_item.training) for mod_item in model.modules()]
    grad_flags = [(par_item, par_item.requires_grad) for par_item in model.parameters()]
    wrappers = []
    originals = []
    rng_devices = [device.index or 0] if device.type == "cuda" else []

    with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
        try:
            model.requires_grad_(False)
            model.eval()
            for path_str, base_lin in targets:
                parent_path, attr_name = path_str.rsplit(".", 1)
                parent_mod = model.get_submodule(parent_path)
                wrap_mod = LoRAProjection(base_lin)
                wrap_mod.adapter_a.copy_(selected[path_str + ".adapter_a"])
                wrap_mod.adapter_b.copy_(selected[path_str + ".adapter_b"])
                wrap_mod.requires_grad_(False)
                wrap_mod.eval()
                originals.append((parent_mod, attr_name, base_lin))
                setattr(parent_mod, attr_name, wrap_mod)
                wrappers.append((path_str, wrap_mod))
            yield model
        finally:
            if wrappers:
                require(
                    _tensor_digest(_adapter_state(wrappers)) == selected_sha,
                    "mounted converged adapter mutated",
                )
            for parent_mod, attr_name, orig_lin in reversed(originals):
                setattr(parent_mod, attr_name, orig_lin)
            for par_item, req_g in grad_flags:
                par_item.grad = None
                par_item.requires_grad_(req_g)
            for mod_item, was_t in modes:
                mod_item.training = was_t
            require(
                _tensor_digest(_state(model)) == base_sha256,
                "base state changed after unmounting converged adapter",
            )


def build_converged_ablation_schedule(prompts_raw, supervision_raw, *, run_id):
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
                    encoded(["conv_abl_call", run_id, ep_idx, canary_tid, "base_canary", role_name])
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
            "arm": "lora_conv",
            "render_arm": "lora",
            "first_messages": f_msgs,
            "first_messages_sha256": digest(f_msgs),
            "call_ids": {
                role_name: _sha256(
                    encoded(["conv_abl_call", run_id, ep_idx, task["id"], "lora_conv", role_name])
                )[:32]
                for role_name in CONFIRMATORY_ROLES
            },
        })

    require(len(episodes) == 26, "26 episodes required")
    return {
        "kind": "LORA_CONFIRMATORY_V2_CONVERGED_ABLATION_SCHEDULE",
        "version": 1,
        "run_id": run_id,
        "scheduled_episodes": 26,
        "scheduled_generation_calls": 52,
        "episodes": episodes,
    }


def execute_remote_converged_body(config, stage, prompts_raw, supervision_raw, backend_raw, worker_raw):
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
    _tensor_digest = loaded["tensor_digest_fn"]
    obj_digest = loaded["obj_digest_fn"]

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
        _tensor_digest(reloaded["adapter"]) == train_out["adapter_sha256"],
        "reloaded converged adapter digest mismatch",
    )
    train_out["adapter"] = reloaded["adapter"]

    epoch_means = [
        sum(st["loss"] for st in train_out["steps"][ep_i * 8:(ep_i + 1) * 8]) / 8.0
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

    prompts_map = {item["id"]: item for item in parse_json(prompts_raw)}
    schedule = build_converged_ablation_schedule(
        prompts_raw, supervision_raw, run_id=config["run_id"]
    )
    worker_text = worker_raw.decode("utf-8")
    gen_dir = stage / "generations"
    gen_dir.mkdir(mode=0o700)
    phase_one_records = []

    p1_t0 = time.monotonic_ns()
    mount_stack = ExitStack()
    mounted_flag = False
    active_model = model
    try:
        for ep in schedule["episodes"]:
            ep_idx = ep["episode_index"]
            tid = ep["task_id"]
            arm = ep["arm"]
            render_arm = ep["render_arm"]
            task_prompt = prompts_map[tid]
            ep_dir = gen_dir / ("ep-%02d" % ep_idx)
            ep_dir.mkdir(mode=0o700)
            init_dir = ep_dir / "initial"
            init_dir.mkdir(mode=0o700)
            fb_dir = ep_dir / "feedback"
            fb_dir.mkdir(mode=0o700)

            want_lora = arm == "lora_conv"
            if want_lora and not mounted_flag:
                active_model = mount_stack.enter_context(mounted_converged_adapter(model, train_out))
                mounted_flag = True
            elif (not want_lora) and mounted_flag:
                mount_stack.close()
                mount_stack = ExitStack()
                active_model = model
                mounted_flag = False

            init_out = generate_observed(
                active_model,
                tokenizer,
                ep["first_messages"],
                expected_tokenizer_sha256=observed_tok_sha,
                call_id=ep["call_ids"]["initial"],
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
                    call_id=ep["call_ids"]["feedback"],
                    artifact_dir=fb_dir,
                )
                fb_gen = fb_out["generation"]
                fb_cand = fb_gen["candidate"]
                fb_cost = _cost_projection(fb_gen)

            phase_one_records.append({
                "episode_index": ep_idx,
                "task_id": tid,
                "split": ep["split"],
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
    final_base_sha = _tensor_digest(model_state_fn(model))
    final_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
    rng_same = bool(
        torch.equal(init_cpu_rng, torch.random.get_rng_state())
        and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
    )
    if final_base_sha != init_base_sha or final_tok_sha != observed_tok_sha or not rng_same:
        raise RuntimeError("post-run integrity drift")
    shared_costs["final_integrity_ns"] = time.monotonic_ns() - fin_t0
    return {
        "gpu_selected": gpu_selected,
        "integrity": {
            "initial_base_sha256": init_base_sha,
            "final_base_sha256": final_base_sha,
            "tokenizer_state_sha256": final_tok_sha,
            "rng_unchanged": rng_same,
        },
        "training_summary": training_summary,
        "shared_costs": shared_costs,
        "phase_one_records": phase_one_records,
    }


def build_converged_ablation_package(
    raw_files,
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
    work_timeout_seconds=5400,
):
    for forbidden in ("validation_remote", "validation_task_worker", "torch", "transformers"):
        require(forbidden not in sys.modules, "forbidden host import: " + forbidden)
    load_confirmatory_dataset(raw_files)
    require(_sha256(worker_raw) == WORKER_SHA256, "worker pin")
    require(_sha256(backend_raw) == BACKEND_SHA256, "backend pin")
    require(set(dep_sources) == ABLATION_SOURCE_NAMES, "ablation source names")
    _check_hex(run_id, 32)
    _check_hex(expected_tokenizer_sha256, 64)
    require(len(scoring_run_ids) == 13, "13 scoring_run_ids")
    seen_ids = {run_id}
    for s_rid in scoring_run_ids:
        _check_hex(s_rid, 32)
        require(s_rid not in seen_ids, "duplicate scoring_run_id")
        seen_ids.add(s_rid)

    expected_runtime = _validate_preflight(preflight_raw, resolved_interpreter)
    sources_sha = {fname: _sha256(dep_sources[fname]) for fname in sorted(ABLATION_SOURCE_NAMES)}
    payload_raw = encoded({
        "sources": {
            fname: base64.b64encode(dep_sources[fname]).decode("ascii")
            for fname in sorted(ABLATION_SOURCE_NAMES)
        },
        "prompts_b64": base64.b64encode(raw_files["prompts.json"]).decode("ascii"),
        "supervision_b64": base64.b64encode(raw_files["supervision.json"]).decode("ascii"),
        "backend_b64": base64.b64encode(backend_raw).decode("ascii"),
        "worker_b64": base64.b64encode(worker_raw).decode("ascii"),
    })
    boot_bytes = G5_STDLIB_BOOTSTRAP.encode("ascii")
    boot_sha = _sha256(boot_bytes)
    p1_cfg = {
        "kind": "LORA_CONVERGED_ABLATION_PHASE1_CONFIG",
        "version": 2,
        "hostname": expected_runtime["hostname"].split(".")[0],
        "uid": expected_runtime["uid"],
        "interpreter": interpreter,
        "resolved_interpreter": resolved_interpreter,
        "site_packages": site_packages,
        "checkpoint_root": checkpoint_root,
        "expected_tokenizer_sha256": expected_tokenizer_sha256,
        "expected_runtime": copy.deepcopy(expected_runtime),
        "parent": remote_parent,
        "stage": str(Path(remote_parent) / ("conv-abl-" + run_id)),
        "run_id": run_id,
        "sources_sha256": sources_sha,
        "prompts_sha256": CONFIRMATORY_FILE_PINS["prompts.json"],
        "supervision_sha256": CONFIRMATORY_FILE_PINS["supervision.json"],
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
        "ns={'__name__':'g5_boot'}\n"
        "exec(compile(src,'<g5_stdlib_bootstrap>','exec'),ns)\n"
        "sys.exit(ns['main_boot'](cfg))\n"
    )
    p1_remote_cmd = "exec " + shlex.join((
        "env", "-u", "LD_PRELOAD", "-u", "LD_LIBRARY_PATH", "-u", "PYTHONHOME",
        "-u", "PYTHONPATH", interpreter, "-I", "-S", "-B", "-c", remote_prog,
    ))
    require(len(p1_remote_cmd.encode("utf-8")) <= 65536, "SSH command size exceeds cap")
    p1_ssh_argv = [*ssh_argv_prefix, p1_remote_cmd]
    p2_ssh_argv = build_multi_bundle_ssh_argv(
        ssh_argv_prefix, resolved_interpreter, backend_raw, expected_runtime
    )

    manifest = {
        "kind": "LORA_CONVERGED_ABLATION_LAUNCH_PACKAGE",
        "version": 2,
        "data_version": "confirmatory_v2_r1",
        "run_id": run_id,
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
        **{flag: False for flag in FALSE_FLAGS},
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


def score_converged_ablation(
    raw_files,
    worker_raw,
    run_id,
    phase_one_records,
    scoring_run_ids,
    final_bundle_records,
    continuation_records,
    expected_runtime,
    training_summary,
    shared_costs,
    g4b_phase_one_records,
    g4b_report,
):
    dataset = load_confirmatory_dataset(raw_files)
    expected_sched = build_converged_ablation_schedule(
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
                        "request_id": "conv-p2-%04d" % len(valid_positions),
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

    g4b_cands = {(ep["task_id"], ep["arm"]): ep["candidates"] for ep in g4b_phase_one_records}
    g4b_eps = {(ep["task_id"], ep["arm"]): ep for ep in g4b_report["episodes"]}

    canary_matches = {}
    conv_eps = {}
    conv_wall_ns = 0
    changed_vs_base_r1 = 0
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
        elif arm == "lora_conv":
            if ep_rec["candidates"]["initial"] != g4b_cands[(tid, "base")]["initial"]:
                changed_vs_base_r1 += 1
            conv_wall_ns += (
                ep_rec["generation_costs"]["initial"]["wall_ns"]
                + ep_rec["generation_costs"]["feedback"]["wall_ns"]
            )
            conv_eps[(tid, "lora_conv")] = {
                "task_id": tid,
                "split": ep_rec["split"],
                "arm": "lora_conv",
                "roles": roles_out,
            }

    canaries_ok = (
        set(canary_matches) == {"WT01", "WR01"}
        and all(canary_matches.values())
        and len(conv_eps) == 24
    )
    run_ok = bool(not proto_bad and not inc_reasons and canaries_ok)
    merged_indexed = {**g4b_eps, **conv_eps}

    by_split = {}
    comparisons = {}
    for sp_name, sp_ids in (("transfer", TRANSFER_IDS), ("retention", RETENTION_IDS)):
        by_split[sp_name] = {
            role_name: {
                st_n: sum(
                    (tid, "lora_conv") in conv_eps
                    and conv_eps[(tid, "lora_conv")]["roles"][role_name]["state"] == st_n
                    for tid in sp_ids
                )
                for st_n in CANDIDATE_STATES
            }
            for role_name in CONFIRMATORY_ROLES
        }
        comparisons[sp_name] = {
            "lora_conv_vs_base": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "base", "lora_conv", run_ok
            ),
            "lora_conv_vs_lora_16step": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "lora", "lora_conv", run_ok
            ),
            "lora_conv_vs_text": _paired_arm_comparison(
                sp_ids, merged_indexed, "feedback", "text", "lora_conv", run_ok
            ),
        }

    return {
        "kind": "LORA_CONFIRMATORY_V2_CONVERGED_ABLATION_REPORT",
        "version": 2,
        "run_id": run_id,
        "complete_and_verified": run_ok,
        "protocol_contradictions": sorted(set(proto_bad)),
        "incomplete_reasons": sorted(set(inc_reasons)),
        "canary_bit_exact_with_g4b_base": canary_matches,
        "changed_round1_outputs_vs_base": changed_vs_base_r1,
        "total_eval_tasks": 24,
        "training": training_summary,
        "lora_conv_by_split": by_split,
        "comparisons_feedback": comparisons,
        "costs": {
            "lora_conv_generation_wall_ns": conv_wall_ns,
            "shared": shared_costs,
        },
        **{flag: False for flag in FALSE_FLAGS},
    }


def run_converged_ablation_stage(
    package,
    raw_files,
    worker_raw,
    g4b_phase_one_records,
    g4b_report,
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
    res_obj = None
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
        safe_write("ablation-package-public.json", _pretty_json(pub))
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
                parsed_res = strict_json(frames["RESULT"])
                safe_write("phase1-result.json", _pretty_json(parsed_res))
                safe_write("phase1-native-stdout.bin", frames["STDOUT"])
                safe_write("phase1-native-stderr.bin", frames["STDERR"])
                if parsed_res.get("bootstrap_failure") is None and p1_out["returncode"] == 0:
                    res_obj = parsed_res

        final_recs = []
        cont_recs = []
        used_ids = []
        if res_obj is not None and not persistence_errors and not cancellations:
            dataset = load_confirmatory_dataset(raw_files)
            prompts_map = {item["id"]: item for item in dataset["prompts"]}
            oracles_map = {item["id"]: item for item in dataset["oracles"]}
            worker_text = worker_raw.decode("utf-8")
            valid_positions = []
            all_positions = []
            for ep_rec in res_obj["phase_one_records"]:
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
                                "request_id": "conv-p2-%04d" % len(valid_positions),
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
                            cm, crec, p1_cfg["expected_runtime"],
                            check_fields=("batch_index", "sub_index", "run_id", "position_indices", "bundle_raw", "transport"),
                        )
                        if (not cf) and cp:
                            nxt.append((bi, si + 1, rsl[len(co):]))
                    pending_cont = nxt
            else:
                p2_all_clean = False

            if not cancellations:
                report = score_converged_ablation(
                    raw_files,
                    worker_raw,
                    run_id,
                    res_obj["phase_one_records"],
                    used_ids,
                    final_recs,
                    cont_recs,
                    p1_cfg["expected_runtime"],
                    res_obj["training_summary"],
                    res_obj["shared_costs"],
                    g4b_phase_one_records,
                    g4b_report,
                )
                safe_write("converged-ablation-report.json", _pretty_json(report))

        p1_clean = bool(p1_out and p1_out["reaped"] and p1_out["group_absent"] and p1_out["cleanup_bounded"])
        binding_ok = bool(p1_clean and p2_all_clean and not cancellations and not persistence_errors and not secondary_errors)
        ctrl_summary = {
            "kind": "LORA_CONVERGED_ABLATION_EXECUTION_EVIDENCE",
            "version": 2,
            "run_id": run_id,
            "package_sha256": pub["package_sha256"],
            "binding_complete": binding_ok,
            "collection_complete": bool(binding_ok and report and report["complete_and_verified"]),
            "secondary_errors": secondary_errors,
            **{flag: False for flag in FALSE_FLAGS},
        }
        safe_write("ablation-controller-summary.json", _pretty_json(ctrl_summary))
    finally:
        os.close(run_fd)

    if cancellations:
        err_obj, err_tb = cancellations[0]
        raise err_obj.with_traceback(err_tb)
    if persistence_errors:
        raise persistence_errors[0]
    return {"summary": ctrl_summary, "report": report}
