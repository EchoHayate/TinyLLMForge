import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time


PHASE1_SOURCE_NAMES = frozenset({
    "generation_core.py",
    "observed_generation.py",
    "pilot_protocol.py",
    "prompt_rendering.py",
    "recalibration_data_contract.py",
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

QWEN3_8B_SHARD_SHA256 = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}

OFFLINE_GPU_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "USE_TF": "0",
    "USE_FLAX": "0",
    "USE_TORCH": "1",
    "WANDB_MODE": "disabled",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONDONTWRITEBYTECODE": "1",
}

BACKEND_EXEC_SNIPPET = (
    "import json, pathlib\n"
    "expected_rt = json.loads(pathlib.Path('expected_runtime.json').read_text('utf-8'))\n"
    "backend_bytes = pathlib.Path('backend.py').read_bytes()\n"
    "namespace = {'__name__': '__main__', 'EXPECTED_RUNTIME': expected_rt}\n"
    "exec(compile(backend_bytes, 'validation_remote.py', 'exec'), namespace)\n"
)


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
    return HEADER.pack(b"TLFD1", ordinal + 1, len(raw)) + raw


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
                        if not b"TLFD1".startswith(bytes(self._buffer[:5])):
                            raise ValueError("invalid frame magic")
                        return
                    magic, channel_num, size = HEADER.unpack(self._buffer[:HEADER.size])
                    if magic != b"TLFD1" or channel_num != ordinal + 1:
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


def select_idle_a100_gpu(gpu_csv_text, apps_csv_text):
    busy_uuids = set()
    for app_line in apps_csv_text.splitlines():
        stripped = app_line.strip()
        if not stripped:
            continue
        parts = [seg.strip() for seg in stripped.split(",")]
        if parts and parts[0]:
            busy_uuids.add(parts[0])

    candidates = []
    for gpu_line in gpu_csv_text.splitlines():
        stripped = gpu_line.strip()
        if not stripped:
            continue
        cols = [seg.strip() for seg in stripped.split(",")]
        if len(cols) != 5:
            continue
        idx_str, uuid_str, name_str, free_str, used_str = cols
        try:
            idx_val = int(idx_str)
            free_mib = int(free_str)
            used_mib = int(used_str)
        except ValueError:
            continue
        if (
            "A100" in name_str
            and uuid_str
            and uuid_str not in busy_uuids
            and free_mib >= 65536
            and used_mib <= 512
        ):
            candidates.append({
                "index": idx_val,
                "uuid": uuid_str,
                "name": name_str,
                "free_mib": free_mib,
                "used_mib": used_mib,
            })
    if not candidates:
        raise ValueError("no idle A100 80GB GPU available")
    candidates.sort(key=lambda entry: entry["index"])
    return candidates[0]


def _query_and_select_gpu():
    gpu_out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    apps_out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    chosen = select_idle_a100_gpu(gpu_out, apps_out)
    return chosen, {"gpu_csv": gpu_out, "apps_csv": apps_out}


def _verify_checkpoint_files(checkpoint_root):
    root_path = Path(checkpoint_root).resolve(strict=True)
    verified = {}
    for rel_name, expected_sha in sorted(QWEN3_8B_SHARD_SHA256.items()):
        fpath = root_path / rel_name
        before_stat = fpath.stat()
        hasher = hashlib.sha256()
        with fpath.open("rb") as stream:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        after_stat = fpath.stat()
        if (
            before_stat.st_size != after_stat.st_size
            or before_stat.st_mtime_ns != after_stat.st_mtime_ns
            or hasher.hexdigest() != expected_sha
        ):
            raise ValueError("checkpoint file verification failed: " + rel_name)
        verified[rel_name] = {"bytes": before_stat.st_size, "sha256": expected_sha}
    return verified


def validate_phase_one_bundle(raw, config):
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
    if set(bundle["sources"]) != PHASE1_SOURCE_NAMES:
        raise ValueError("phase one source set")
    sources = {}
    for name in sorted(PHASE1_SOURCE_NAMES):
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


def _invoke_backend_bundle(resolved_python, backend_raw, expected_runtime, bundle_obj):
    workdir = Path(tempfile.mkdtemp(prefix="tlf-diag-pub-", dir="/tmp")).resolve()
    try:
        os.chmod(workdir, 0o700)
        write_private(workdir / "backend.py", backend_raw)
        write_private(workdir / "expected_runtime.json", encoded(expected_runtime))
        proc = subprocess.Popen(
            [resolved_python, "-I", "-S", "-B", "-c", BACKEND_EXEC_SNIPPET],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(workdir),
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "HOME": "/nonexistent",
            },
            close_fds=True,
            pass_fds=(),
            start_new_session=True,
            shell=False,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(
                input=encoded(bundle_obj), timeout=15.0
            )
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            for pipe_obj in (proc.stdin, proc.stdout, proc.stderr):
                if pipe_obj is not None:
                    try:
                        pipe_obj.close()
                    except OSError:
                        pass
        if proc.returncode != 0 or stderr_bytes:
            raise RuntimeError("public case backend child failed")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    events = [strict_json(line) for line in stdout_bytes.splitlines() if line]
    row_events = [ev for ev in events if ev.get("kind") == "row"]
    done_events = [ev for ev in events if ev.get("kind") == "done"]
    if not done_events:
        raise RuntimeError("public case backend missing done event")
    return row_events, done_events[0]


def _run_public_cases_child(resolved_python, backend_raw, worker_text, expected_runtime, run_id_prefix, ep_index, task_prompt, candidate_src):
    started_ns = time.monotonic_ns()
    requests = [
        {
            "request_id": "pub-%02d-%s" % (ep_index, pcase["id"]),
            "source": candidate_src,
            "entry_point": task_prompt["entry_point"],
            "args": pcase["args"],
        }
        for pcase in task_prompt["public_cases"]
    ]
    pub_run_id = sha256_hex(encoded(["pub_p1", run_id_prefix, ep_index]))[:32]
    row_events, done_ev = _invoke_backend_bundle(
        resolved_python,
        backend_raw,
        expected_runtime,
        {
            "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
            "mode": "matrix",
            "run_id": pub_run_id,
            "worker_source": worker_text,
            "requests": requests,
        },
    )
    if len(row_events) == 1 and done_ev.get("stopped") is True and len(requests) == 2:
        pub_run_id_second = sha256_hex(encoded(["pub_p2", run_id_prefix, ep_index]))[:32]
        second_rows, _second_done = _invoke_backend_bundle(
            resolved_python,
            backend_raw,
            expected_runtime,
            {
                "kind": "LORA_REMOTE_VALIDATION_BUNDLE",
                "mode": "matrix",
                "run_id": pub_run_id_second,
                "worker_source": worker_text,
                "requests": [requests[1]],
            },
        )
        row_events.extend(second_rows)

    cases_out = []
    for idx, pcase in enumerate(task_prompt["public_cases"]):
        if idx >= len(row_events):
            raise RuntimeError("public case row missing")
        res = row_events[idx]["result"]
        if not res["process_reaped"]:
            raise RuntimeError("public case worker unreaped")
        if res["status"] == "completed":
            actual_out = strict_json(bytes.fromhex(res["stdout_hex"]))
            st_label = (
                "pass"
                if encoded(actual_out) == encoded(pcase["expected"])
                else "mismatch"
            )
        else:
            st_label = res["status"]
        cases_out.append({"id": pcase["id"], "status": st_label})
    elapsed_ns = time.monotonic_ns() - started_ns
    return {"kind": "visible_tests", "cases": cases_out}, elapsed_ns


def _initialize_cuda_and_load_model(config, shared_costs):
    import torch
    import transformers
    from tokenizer_measurement import _state as tok_state_fn
    from training_worker import _state as model_state_fn, _tensor_digest
    from validation_contract import digest as obj_digest

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not (torch.cuda.is_available() and torch.cuda.device_count() == 1):
        raise RuntimeError("expected single visible CUDA device")
    torch.cuda.init()
    device = torch.device("cuda:0")
    if torch.cuda.get_device_properties(device).major != 8:
        raise RuntimeError("expected Ampere A100 CUDA device")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.synchronize(device)

    load_t0 = time.monotonic_ns()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["checkpoint_root"],
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    observed_tok_sha = obj_digest(tok_state_fn(tokenizer)[0])
    if observed_tok_sha != config["expected_tokenizer_sha256"]:
        raise ValueError("loaded tokenizer state SHA256 mismatch")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config["checkpoint_root"],
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = model.to(device).eval().requires_grad_(False)
    model.config.use_cache = False
    torch.cuda.synchronize(device)
    shared_costs["model_and_tokenizer_load_ns"] = time.monotonic_ns() - load_t0
    shared_costs["model_load_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    shared_costs["model_load_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))

    init_base_sha = _tensor_digest(model_state_fn(model))
    init_cpu_rng = torch.random.get_rng_state().clone()
    init_cuda_rng = torch.cuda.get_rng_state(device).clone()
    return {
        "torch": torch,
        "device": device,
        "model": model,
        "tokenizer": tokenizer,
        "observed_tok_sha": observed_tok_sha,
        "init_base_sha": init_base_sha,
        "init_cpu_rng": init_cpu_rng,
        "init_cuda_rng": init_cuda_rng,
        "tok_state_fn": tok_state_fn,
        "model_state_fn": model_state_fn,
        "tensor_digest_fn": _tensor_digest,
        "obj_digest_fn": obj_digest,
    }


def bootstrap(config):
    work_timeout_seconds = int(config["work_timeout_seconds"])
    if not 60 <= work_timeout_seconds <= 7200:
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
    stage = parent_path / ("diag-phase1-" + config["run_id"])
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
        sources, prompts_raw, supervision_raw, backend_raw, worker_raw = validate_phase_one_bundle(
            raw_payload, config
        )
        for name, content in sources.items():
            write_private(stage / name, content)

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
        sys.path.extend([str(stage), config["site_packages"]])
        from observed_generation import generate_observed
        from recalibration_diagnostic_scoring import (
            build_generation_schedule,
            candidate_valid,
            continuation_messages,
        )
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

        phase = "EPISODES"
        prompts_list = parse_json(prompts_raw)
        prompts_map = {item["id"]: item for item in prompts_list}
        schedule = build_generation_schedule(prompts_raw, supervision_raw, run_id=config["run_id"])
        worker_text = worker_raw.decode("utf-8")
        gen_dir = stage / "generations"
        gen_dir.mkdir(mode=0o700)

        p1_t0 = time.monotonic_ns()
        for ep in schedule["episodes"]:
            ep_idx = ep["episode_index"]
            tid = ep["task_id"]
            arm = ep["arm"]
            task_prompt = prompts_map[tid]
            ep_art_dir = gen_dir / ("ep-%02d" % ep_idx)
            ep_art_dir.mkdir(mode=0o700)

            init_dir = ep_art_dir / "initial"
            init_dir.mkdir(mode=0o700)
            init_out = generate_observed(
                model,
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

            role_candidates = {"initial": init_cand}
            role_costs = {
                "initial": {
                    "input_tokens": init_gen["input_tokens"],
                    "output_tokens": init_gen["output_tokens"],
                    "forward_calls": init_gen["forward_calls"],
                    "wall_ns": init_gen["timings_ns"]["wall"],
                    "peak_allocated_bytes": init_gen["peak_allocated_bytes"],
                    "peak_reserved_bytes": init_gen["peak_reserved_bytes"],
                }
            }

            for branch_name in ep["branch_order"]:
                br_msgs = continuation_messages(
                    task_prompt,
                    arm,
                    supervision_raw,
                    init_cand,
                    branch_name,
                    feedback=(pub_fb if branch_name == "feedback" else None),
                )
                br_dir = ep_art_dir / branch_name
                br_dir.mkdir(mode=0o700)
                br_out = generate_observed(
                    model,
                    tokenizer,
                    br_msgs,
                    expected_tokenizer_sha256=observed_tok_sha,
                    call_id=ep["call_ids"][branch_name],
                    artifact_dir=br_dir,
                )
                br_gen = br_out["generation"]
                role_candidates[branch_name] = br_gen["candidate"]
                role_costs[branch_name] = {
                    "input_tokens": br_gen["input_tokens"],
                    "output_tokens": br_gen["output_tokens"],
                    "forward_calls": br_gen["forward_calls"],
                    "wall_ns": br_gen["timings_ns"]["wall"],
                    "peak_allocated_bytes": br_gen["peak_allocated_bytes"],
                    "peak_reserved_bytes": br_gen["peak_reserved_bytes"],
                }

            phase_one_records.append({
                "episode_index": ep_idx,
                "task_id": tid,
                "arm": arm,
                "candidates": role_candidates,
                "public_feedback": pub_fb,
                "generation_costs": role_costs,
                "public_execution_ns": pub_ns,
            })

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
        "kind": "LORA_RECALIBRATION_DIAGNOSTIC_PHASE1_RESULT",
        "version": 1,
        "identity": identity,
        "bootstrap_failure": bootstrap_failure,
        "gpu_selected": gpu_selected,
        "integrity": integrity_out,
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
