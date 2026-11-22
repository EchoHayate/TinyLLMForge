from contextlib import ExitStack
import base64
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


PHASE1_LORA_SOURCE_NAMES = frozenset({
    "adapter_artifact.py",
    "confirmatory_v2_contract.py",
    "confirmatory_v2_scoring.py",
    "confirmatory_v2_training.py",
    "generation_core.py",
    "observed_generation.py",
    "pilot_protocol.py",
    "prompt_rendering.py",
    "recalibration_data_contract.py",
    "recalibration_diagnostic_remote.py",
    "recalibration_diagnostic_scoring.py",
    "recalibration_rendering.py",
    "recalibration_validation.py",
    "tokenizer_measurement.py",
    "training_worker.py",
    "validation_contract.py",
    "validation_real_report.py",
    "validation_report.py",
})
CHANNELS = ("READY", "RESULT", "STDOUT", "STDERR", "DONE")
RESULT_LIMIT = 67_108_864
DIAGNOSTIC_LIMIT = 1_048_576
CHANNEL_LIMITS = (4096, RESULT_LIMIT, DIAGNOSTIC_LIMIT, DIAGNOSTIC_LIMIT, 4096)
HEADER = struct.Struct("!5sBQ")
WIRE_LIMIT = sum(CHANNEL_LIMITS) + HEADER.size * len(CHANNELS)
PAYLOAD_LIMIT = 16 * 1024 * 1024


def encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256_hex(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def pairs(entries):
        mapping = {}
        for name, item in entries:
            if name in mapping:
                raise ValueError("duplicate JSON key")
            mapping[name] = item
        return mapping

    def bad_constant(token):
        raise ValueError("nonfinite JSON value")

    try:
        text = raw.decode("utf-8") if type(raw) is bytes else raw
        return json.loads(text, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (TypeError, UnicodeError, ValueError, RecursionError):
        raise ValueError("invalid JSON") from None


def encode_frame(channel, raw):
    if channel not in CHANNELS or type(raw) is not bytes:
        raise ValueError("invalid frame")
    ordinal = CHANNELS.index(channel)
    if len(raw) > CHANNEL_LIMITS[ordinal]:
        raise ValueError("frame overflow")
    return HEADER.pack(b"TLFC1", ordinal + 1, len(raw)) + raw


class FrameDecoder:
    def __init__(self):
        self.frames = {}
        self._buffer = bytearray()
        self._size = None
        self._closed = False
        self._failed = False

    def feed(self, raw):
        if self._closed or self._failed:
            raise ValueError("closed frame decoder")
        try:
            self._buffer.extend(raw)
            while self._buffer or self._size == 0:
                ordinal = len(self.frames)
                if ordinal == len(CHANNELS):
                    raise ValueError("trailing frame data")
                if self._size is None:
                    if len(self._buffer) < HEADER.size:
                        if not b"TLFC1".startswith(bytes(self._buffer[:5])):
                            raise ValueError("invalid frame magic")
                        return
                    magic, channel_num, size = HEADER.unpack(self._buffer[:HEADER.size])
                    if magic != b"TLFC1" or channel_num != ordinal + 1:
                        raise ValueError("invalid or duplicate channel")
                    if size > CHANNEL_LIMITS[ordinal]:
                        raise ValueError("frame overflow")
                    del self._buffer[:HEADER.size]
                    self._size = size
                if len(self._buffer) < self._size:
                    return
                self.frames[CHANNELS[ordinal]] = bytes(self._buffer[:self._size])
                del self._buffer[:self._size]
                self._size = None
        except BaseException:
            self._failed = True
            raise

    def finish(self):
        if self._failed or self._buffer or self._size is not None or len(self.frames) != 5:
            raise ValueError("truncated frame stream")
        self._closed = True
        return dict(self.frames)


def decode_frames(raw):
    decoder = FrameDecoder()
    for offset in range(0, len(raw), 65536):
        decoder.feed(raw[offset:offset + 65536])
    return decoder.finish()


def validate_phase_one_lora_bundle(raw, config):
    if (
        type(raw) is not bytes
        or len(raw) > PAYLOAD_LIMIT
        or len(raw) != config["payload_bytes"]
        or sha256_hex(raw) != config["payload_sha256"]
    ):
        raise ValueError("phase one payload digest mismatch")
    bundle = strict_json(raw)
    expected_top = {"sources", "prompts_b64", "supervision_b64", "backend_b64", "worker_b64"}
    if set(bundle) != expected_top:
        raise ValueError("phase one bundle fields")
    if set(bundle["sources"]) != PHASE1_LORA_SOURCE_NAMES:
        raise ValueError("phase one source set")
    sources = {}
    for name in sorted(PHASE1_LORA_SOURCE_NAMES):
        content = base64.b64decode(bundle["sources"][name], validate=True)
        if sha256_hex(content) != config["sources_sha256"][name]:
            raise ValueError("phase one source pin mismatch: " + name)
        sources[name] = content
    prompts_raw = base64.b64decode(bundle["prompts_b64"], validate=True)
    supervision_raw = base64.b64decode(bundle["supervision_b64"], validate=True)
    backend_raw = base64.b64decode(bundle["backend_b64"], validate=True)
    worker_raw = base64.b64decode(bundle["worker_b64"], validate=True)
    if (
        sha256_hex(prompts_raw) != config["prompts_sha256"]
        or sha256_hex(supervision_raw) != config["supervision_sha256"]
        or sha256_hex(backend_raw) != config["backend_sha256"]
        or sha256_hex(worker_raw) != config["worker_sha256"]
    ):
        raise ValueError("phase one data/backend/worker pin mismatch")
    return sources, prompts_raw, supervision_raw, backend_raw, worker_raw


def write_private(target_path, raw):
    descriptor = os.open(
        target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_frame(descriptor, channel_name, raw):
    framed = encode_frame(channel_name, raw)
    view = memoryview(framed)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("frame write failed")
        view = view[written:]


def _read_exact(stream, size):
    content = bytearray()
    while len(content) < size:
        chunk = stream.read(min(size - len(content), 65536))
        if not chunk:
            raise ValueError("truncated input")
        content.extend(chunk)
    return bytes(content)


def _cost_projection(gen_rec):
    return {
        "input_tokens": gen_rec["input_tokens"],
        "output_tokens": gen_rec["output_tokens"],
        "forward_calls": gen_rec["forward_calls"],
        "wall_ns": gen_rec["timings_ns"]["wall"],
        "peak_allocated_bytes": gen_rec["peak_allocated_bytes"],
        "peak_reserved_bytes": gen_rec["peak_reserved_bytes"],
    }


def bootstrap(config):
    work_timeout_seconds = int(config["work_timeout_seconds"])
    if not 60 <= work_timeout_seconds <= 28800:
        raise ValueError("work_timeout_seconds out of bounds")
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGALRM})
    signal.setitimer(signal.ITIMER_REAL, 30)
    anchor_mono = time.monotonic()

    if not (sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode):
        raise ValueError("isolated bootstrap required")
    if socket.gethostname().split(".")[0] != config["hostname"] or os.getuid() != config["uid"]:
        raise ValueError("host identity mismatch")
    if (
        sys.executable != config["interpreter"]
        and os.path.realpath(sys.executable) != config["resolved_interpreter"]
    ):
        raise ValueError("interpreter mismatch")

    parent_path = Path(config["parent"])
    if not parent_path.is_absolute() or parent_path != parent_path.resolve():
        raise ValueError("noncanonical parent")
    if not parent_path.exists():
        parent_path.mkdir(mode=0o700)
    stage = parent_path / ("conf-phase1-" + config["run_id"])
    stage.mkdir(mode=0o700)
    if str(stage) != config["stage"]:
        raise ValueError("stage mismatch")
    os.chdir(stage)

    identity = {
        "run_id": config["run_id"],
        "config_sha256": sha256_hex(encoded(config)),
        "hostname": config["hostname"],
        "uid": os.getuid(),
        "stage": str(stage),
    }
    _write_frame(1, "READY", encoded(identity))

    output_fd = 1
    phase = "INPUT"
    bootstrap_failure = None
    gpu_selected = None
    integrity_out = None
    training_summary = None
    shared_costs = {}
    phase_one_records = []
    diag_stdout = b""
    diag_stderr = b""

    try:
        grant_raw = sys.stdin.buffer.readline(256)
        if not grant_raw.endswith(b"\n"):
            raise ValueError("invalid grant")
        grant = strict_json(grant_raw)
        rem_sec = anchor_mono + (grant["remaining_ns"] / 1_000_000_000.0) - time.monotonic()
        if rem_sec <= 0:
            raise TimeoutError("deadline expired")
        signal.setitimer(signal.ITIMER_REAL, rem_sec)

        raw_payload = _read_exact(sys.stdin.buffer, config["payload_bytes"])
        if sys.stdin.buffer.read(1):
            raise ValueError("trailing stdin")
        sources, prompts_raw, supervision_raw, backend_raw, worker_raw = validate_phase_one_lora_bundle(
            raw_payload, config
        )
        for name, content in sources.items():
            write_private(stage / name, content)

        sys.path.extend([str(stage), config["site_packages"]])
        from recalibration_diagnostic_remote import (
            OFFLINE_GPU_ENVIRONMENT,
            _initialize_cuda_and_load_model,
            _query_and_select_gpu,
            _run_public_cases_child,
            _verify_checkpoint_files,
        )

        phase = "GPU_SELECT"
        gpu_selected, gpu_raw_obs = _query_and_select_gpu()
        write_private(stage / "gpu-before.json", encoded(gpu_raw_obs))

        os.environ.update(OFFLINE_GPU_ENVIRONMENT)
        os.environ.update({
            "HOME": str(stage),
            "TMPDIR": str(stage),
            "HF_HOME": str(stage / "hf-cache"),
            "CUDA_VISIBLE_DEVICES": gpu_selected["uuid"],
        })

        saved_out = os.dup(1)
        os.set_inheritable(saved_out, False)
        for fd_num, log_name in ((1, "native.stdout"), (2, "native.stderr")):
            lfd = os.open(stage / log_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            try:
                os.fchmod(lfd, 0o600)
                os.dup2(lfd, fd_num)
            finally:
                os.close(lfd)
        output_fd = saved_out

        phase = "CHECKPOINT_VERIFY"
        chk_t0 = time.monotonic_ns()
        _verify_checkpoint_files(config["checkpoint_root"])
        shared_costs["checkpoint_verify_ns"] = time.monotonic_ns() - chk_t0

        phase = "MODEL_LOAD"
        from adapter_artifact import load_adapter, mounted_adapter, save_adapter
        from confirmatory_v2_contract import confirmatory_second_round
        from confirmatory_v2_scoring import build_confirmatory_schedule
        from confirmatory_v2_training import (
            prepare_confirmatory_training,
            train_confirmatory_adapter,
        )
        from observed_generation import generate_observed
        from recalibration_diagnostic_scoring import candidate_valid
        from tokenizer_measurement import measure as measure_tokens
        from validation_contract import parse_json

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

        phase = "LORA_TRAIN"
        train_prep_t0 = time.monotonic_ns()
        train_ds = prepare_confirmatory_training(tokenizer, supervision_raw)
        shared_costs["training_tokenize_prep_ns"] = time.monotonic_ns() - train_prep_t0
        train_res = train_confirmatory_adapter(model, train_ds)
        adapter_path = stage / "adapter.pt"
        save_meta = save_adapter(adapter_path, train_res)
        load_meta = load_adapter(adapter_path, expected_sha256=save_meta["sha256"])
        loaded_adapter = load_meta["result"]
        post_train_base_sha = _tensor_digest(model_state_fn(model))
        post_train_rng_ok = bool(
            torch.equal(init_cpu_rng, torch.random.get_rng_state())
            and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
        )
        if post_train_base_sha != init_base_sha or not post_train_rng_ok:
            raise RuntimeError("post-training base/RNG drift")
        receipt = train_res["receipt"]
        training_summary = {
            "dataset_sha256": receipt["dataset_sha256"],
            "configuration_sha256": receipt["configuration_sha256"],
            "initial_adapter_sha256": receipt["initial_adapter_sha256"],
            "adapter_sha256": receipt["adapter_sha256"],
            "artifact_file_sha256": save_meta["sha256"],
            "artifact_file_bytes": save_meta["file_bytes"],
            "adapter_bytes": receipt["adapter_bytes"],
            "optimizer_tensor_bytes": receipt["optimizer_tensor_bytes"],
            "supervised_tokens": receipt["supervised_tokens"],
            "preparation_ns": receipt["preparation_ns"],
            "training_ns": receipt["training_ns"],
            "verification_ns": receipt["verification_ns"],
            "core_wall_ns": receipt["core_wall_ns"],
            "adapter_save_ns": save_meta["save_ns"],
            "adapter_load_ns": load_meta["load_ns"],
            "peak_allocated_bytes": receipt["peak_allocated_bytes"],
            "peak_reserved_bytes": receipt["peak_reserved_bytes"],
            "initial_loss": receipt["steps"][0]["loss"],
            "final_loss": receipt["steps"][-1]["loss"],
            "steps": receipt["steps"],
        }

        phase = "EPISODES"
        prompts_list = parse_json(prompts_raw)
        prompts_map = {item["id"]: item for item in prompts_list}
        schedule = build_confirmatory_schedule(prompts_raw, supervision_raw, run_id=config["run_id"])
        worker_text = worker_raw.decode("utf-8")
        gen_dir = stage / "generations"
        gen_dir.mkdir(mode=0o700)

        p1_t0 = time.monotonic_ns()
        mount_stack = ExitStack()
        currently_mounted = False
        active_model = model
        try:
            for ep in schedule["episodes"]:
                ep_idx = ep["episode_index"]
                tid = ep["task_id"]
                arm = ep["arm"]
                task_prompt = prompts_map[tid]
                ep_art_dir = gen_dir / ("ep-%03d" % ep_idx)
                ep_art_dir.mkdir(mode=0o700)
                init_dir = ep_art_dir / "initial"
                init_dir.mkdir(mode=0o700)
                fb_dir = ep_art_dir / "feedback"
                fb_dir.mkdir(mode=0o700)

                want_lora = arm in ("lora", "lora_text")
                if want_lora and not currently_mounted:
                    active_model = mount_stack.enter_context(mounted_adapter(model, loaded_adapter))
                    currently_mounted = True
                elif (not want_lora) and currently_mounted:
                    mount_stack.close()
                    mount_stack = ExitStack()
                    active_model = model
                    currently_mounted = False

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
                    task_prompt, arm, supervision_raw, init_cand, pub_fb
                )
                sec_check = measure_tokens(tokenizer, sec_msgs, answer=None)
                if sec_check["status"] != "OK" or sec_check["input"]["input_tokens"] > 4096:
                    fb_cand = ""
                    fb_cost = {
                        "input_tokens": sec_check["input"]["input_tokens"],
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
                    "candidates": {
                        "initial": init_cand,
                        "feedback": fb_cand,
                    },
                    "public_feedback": pub_fb,
                    "generation_costs": {
                        "initial": _cost_projection(init_gen),
                        "feedback": fb_cost,
                    },
                    "public_execution_ns": pub_ns,
                })
        finally:
            mount_stack.close()

        shared_costs["episodes_wall_ns"] = time.monotonic_ns() - p1_t0

        phase = "FINAL_INTEGRITY"
        fin_t0 = time.monotonic_ns()
        final_base_sha = _tensor_digest(model_state_fn(model))
        final_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
        rng_same = bool(
            torch.equal(init_cpu_rng, torch.random.get_rng_state())
            and torch.equal(init_cuda_rng, torch.cuda.get_rng_state(device))
        )
        if final_base_sha != init_base_sha or final_tok_sha != observed_tok_sha or not rng_same:
            raise RuntimeError("post-run base/tokenizer/RNG integrity drift")
        shared_costs["final_integrity_ns"] = time.monotonic_ns() - fin_t0
        integrity_out = {
            "initial_base_sha256": init_base_sha,
            "post_training_base_sha256": post_train_base_sha,
            "final_base_sha256": final_base_sha,
            "tokenizer_state_sha256": final_tok_sha,
            "rng_unchanged": rng_same,
        }
        signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        tb_cur = exc.__traceback__
        while tb_cur is not None and tb_cur.tb_next is not None:
            tb_cur = tb_cur.tb_next
        bootstrap_failure = {
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc)[:512],
            "lineno": tb_cur.tb_lineno if tb_cur is not None else None,
            "function": tb_cur.tb_frame.f_code.co_name if tb_cur is not None else None,
        }

    if output_fd != 1:
        sys.stdout.flush()
        sys.stderr.flush()
        diag_stdout = (stage / "native.stdout").read_bytes()[:DIAGNOSTIC_LIMIT]
        diag_stderr = (stage / "native.stderr").read_bytes()[:DIAGNOSTIC_LIMIT]

    result_bytes = encoded({
        "kind": "LORA_CONFIRMATORY_V2_PHASE1_RESULT",
        "version": 1,
        "identity": identity,
        "bootstrap_failure": bootstrap_failure,
        "gpu_selected": gpu_selected,
        "integrity": integrity_out,
        "training_summary": training_summary,
        "shared_costs": shared_costs,
        "phase_one_records": phase_one_records,
    })
    _write_frame(output_fd, "RESULT", result_bytes)
    _write_frame(output_fd, "STDOUT", diag_stdout)
    _write_frame(output_fd, "STDERR", diag_stderr)
    _write_frame(
        output_fd,
        "DONE",
        encoded({
            "config_sha256": identity["config_sha256"],
            "result_sha256": sha256_hex(result_bytes),
        }),
    )
    return 0 if bootstrap_failure is None else 1
