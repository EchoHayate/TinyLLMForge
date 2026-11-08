from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


MAX_CALLS = 144
MAX_LINE_BYTES = 131072
MAX_INPUT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32
GENERATION_OVERRIDES = {"use_model_defaults": False, "do_sample": False}
EXPECTED_VERSIONS = {
    "python": "3.12.13",
    "torch": "2.5.1+cu124", "transformers": "4.57.1", "safetensors": "0.8.0",
}


def _hex(value, length):
    return type(value) is str and re.fullmatch(r"[0-9a-fA-F]{%d}" % length, value) is not None


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("invalid JSON constant: " + value)


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def validate_request(request):
    if type(request) is not dict or set(request) != {"id", "messages"}:
        raise ValueError("private or unknown request field")
    if not _hex(request["id"], 32):
        raise ValueError("request id must be 32 hex characters")
    messages = request["messages"]
    if type(messages) is not list or len(messages) != 1:
        raise ValueError("independent single-user request required")
    message = messages[0]
    if type(message) is not dict or set(message) != {"role", "content"}:
        raise ValueError("invalid message fields")
    content = message["content"]
    if message["role"] != "user" or type(content) is not str:
        raise ValueError("invalid message role/content")
    if not content.strip() or len(content) > 32000:
        raise ValueError("invalid message length")
    content.encode("utf-8")


def validate_gpu(devices, processes, visible, gpu_uuid, gpu_index):
    if type(gpu_uuid) is not str or not gpu_uuid.startswith("GPU-") or visible != gpu_uuid:
        raise ValueError("GPU visibility mismatch")
    if type(gpu_index) is not int or gpu_index < 0:
        raise ValueError("invalid GPU index")
    targets = []
    indices, identities = set(), set()
    for row in devices.strip().splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 4:
            raise ValueError("malformed GPU query")
        index, identity, memory, utilization = fields
        if not index.isdecimal() or index in indices or identity in identities:
            raise ValueError("duplicate or invalid GPU identity")
        indices.add(index)
        identities.add(identity)
        if identity == gpu_uuid:
            targets.append(fields)
    if len(targets) != 1:
        raise ValueError("missing or duplicate target GPU")
    index, identity, memory, utilization = targets[0]
    if int(index) != gpu_index or not 0 <= int(memory) <= 100 or int(utilization) != 0:
        raise ValueError("GPU is not idle or index mismatches")
    for row in processes.strip().splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 2 or not fields[0].startswith("GPU-") or not fields[1].isdecimal():
            raise ValueError("malformed compute process query")
        if fields[0] == identity:
            raise ValueError("target GPU has a compute process")


def _filename(name):
    return (
        type(name) is str and bool(name) and name not in (".", "..")
        and "/" not in name and "\\" not in name and "\x00" not in name
    )


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(model_dir, expected_hashes):
    if type(expected_hashes) is not dict or not expected_hashes:
        raise ValueError("missing model hashes")
    root = Path(model_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model directory required")
    verified = {}
    for name, checksum in expected_hashes.items():
        if not _filename(name) or not _hex(checksum, 64):
            raise ValueError("invalid model manifest")
        candidate = (root / name).resolve(strict=True)
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError("model symlink escape or missing file")
        digest = _file_digest(candidate)
        if digest != checksum.lower():
            raise ValueError("model hash mismatch: " + name)
        verified[name] = digest
    for candidate in root.iterdir():
        if candidate.name not in expected_hashes:
            if candidate.name not in {"README.md", "LICENSE", ".gitattributes"}:
                raise ValueError("unverified model input: " + candidate.name)
        if candidate.name.endswith(".index.json"):
            index = _json(candidate.read_text(encoding="utf-8"))
            if type(index) is not dict or type(index.get("weight_map")) is not dict:
                raise ValueError("invalid weight index")
            for shard in index["weight_map"].values():
                if not _filename(shard) or shard not in verified:
                    raise ValueError("unverified weight shard")
    return verified


def _validate_generated(result):
    fields = {
        "text", "terminated", "input_ids", "output_ids",
        "generation_ns", "call_ns", "prompt_sha256",
    }
    if type(result) is not dict or set(result) != fields:
        raise ValueError("invalid generated fields")
    if type(result["text"]) is not str or type(result["terminated"]) is not bool:
        raise ValueError("invalid generated text/termination")
    for name, limit in (("input_ids", MAX_INPUT_TOKENS), ("output_ids", MAX_OUTPUT_TOKENS)):
        values = result[name]
        if type(values) is not list or not 0 < len(values) <= limit:
            raise ValueError("invalid token count")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("token IDs must be nonnegative integers, not bool")
    for name in ("generation_ns", "call_ns"):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError("invalid nanosecond timing")
    if result["call_ns"] < result["generation_ns"] or not _hex(result["prompt_sha256"], 64):
        raise ValueError("invalid generation timing or prompt digest")
    _canonical(result)


def _read_line(input_stream):
    chunks = []
    size = 0
    while True:
        raw = input_stream.readline(max(1, (MAX_LINE_BYTES - size) // 4))
        if raw in ("", b""):
            return b"".join(chunks)
        encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
        size += len(encoded)
        if size > MAX_LINE_BYTES:
            raise ValueError("request line byte cap exceeded")
        chunks.append(encoded)
        if encoded.endswith(b"\n"):
            return b"".join(chunks)


def serve(input_stream, generate, emit, max_calls=MAX_CALLS):
    if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
        raise ValueError("invalid call cap")
    seen = set()
    calls = 0
    while True:
        encoded = _read_line(input_stream)
        if not encoded:
            return calls
        request = _json(encoded.decode("utf-8"))
        validate_request(request)
        identity = request["id"]
        if identity.lower() in seen:
            raise ValueError("replayed request id")
        if calls >= max_calls:
            raise ValueError("model call cap exceeded")
        digest = hashlib.sha256(_canonical(request)).hexdigest()
        seen.add(identity.lower())
        calls += 1
        result = generate(request)
        _validate_generated(result)
        emit({
            "event": "output", "id": identity, "request_sha256": digest, **result,
        })


def prepare_generation(model, tokenizer, config_class):
    eos_ids = model.generation_config.eos_token_id
    eos_ids = [eos_ids] if type(eos_ids) is int else eos_ids
    if (
        type(eos_ids) is not list or not eos_ids
        or any(type(token) is not int or token < 0 for token in eos_ids)
    ):
        raise ValueError("missing or invalid EOS configuration")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_ids[0]
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("invalid PAD configuration")
    requested = config_class(
        do_sample=False, num_beams=1, num_return_sequences=1,
        max_new_tokens=MAX_OUTPUT_TOKENS, eos_token_id=eos_ids,
        pad_token_id=pad_id, use_cache=True, return_dict_in_generate=False,
        output_scores=False,
    )
    effective, unused = model._prepare_generation_config(requested, **GENERATION_OVERRIDES)
    mode = effective.get_generation_mode().value
    if unused or mode != "greedy_search":
        raise ValueError("effective generation mode is not explicit greedy")
    for name in (
        "do_sample", "num_beams", "num_return_sequences", "max_new_tokens",
        "eos_token_id", "pad_token_id", "use_cache", "return_dict_in_generate", "output_scores",
    ):
        if getattr(effective, name) != getattr(requested, name):
            raise ValueError("effective greedy configuration mismatch: " + name)
    return effective, mode


def generate_request(request, model, tokenizer, torch, generation):
    call_started = time.perf_counter_ns()
    validate_request(request)
    prompt = tokenizer.apply_chat_template(
        request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if type(prompt) is not str or not prompt.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("non-thinking chat template suffix mismatch")
    prompt_digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"][0].tolist()
    if not 0 < len(input_ids) <= MAX_INPUT_TOKENS:
        raise ValueError("input token budget exceeded")
    if any(type(token) is not int or token < 0 for token in input_ids):
        raise ValueError("invalid input token IDs")
    encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
    torch.cuda.synchronize()
    generation_started = time.perf_counter_ns()
    with torch.inference_mode():
        output = model.generate(
            **encoded, generation_config=generation, **GENERATION_OVERRIDES,
        )
    torch.cuda.synchronize()
    generation_ns = time.perf_counter_ns() - generation_started
    if output.shape[0] != 1:
        raise ValueError("generation batch mismatch")
    raw_ids = output[0].tolist()
    if raw_ids[:len(input_ids)] != input_ids:
        raise ValueError("generation prefix mismatch")
    output_ids = raw_ids[len(input_ids):]
    if not 0 < len(output_ids) <= MAX_OUTPUT_TOKENS:
        raise ValueError("output token budget exceeded")
    result = {
        "text": tokenizer.decode(output_ids, skip_special_tokens=True),
        "terminated": output_ids[-1] in generation.eos_token_id,
        "input_ids": input_ids, "output_ids": output_ids,
        "generation_ns": generation_ns, "call_ns": time.perf_counter_ns() - call_started,
        "prompt_sha256": prompt_digest,
    }
    _validate_generated(result)
    return result


def _preflight(directory):
    directory = Path(directory).resolve(strict=True)
    with (directory / "launch.json").open("rb") as stream:
        raw = stream.read(MAX_LINE_BYTES + 1)
    if len(raw) > MAX_LINE_BYTES:
        raise ValueError("launch byte cap exceeded")
    launch = _json(raw.decode("utf-8"))
    fields = {
        "run_id", "hostname", "uid", "gpu_uuid", "gpu_index", "model_dir",
        "model_hashes", "worker_sha256", "max_calls",
    }
    if type(launch) is not dict or set(launch) != fields:
        raise ValueError("invalid launch fields")
    if not _hex(launch["run_id"], 32) or not _hex(launch["worker_sha256"], 64):
        raise ValueError("invalid launch identity/digest")
    if (
        type(launch["hostname"]) is not str or launch["hostname"] != platform.node()
        or type(launch["uid"]) is not int or launch["uid"] != os.getuid()
    ):
        raise ValueError("remote identity mismatch")
    if type(launch["max_calls"]) is not int or launch["max_calls"] != MAX_CALLS:
        raise ValueError("launch call cap mismatch")
    if type(launch["gpu_index"]) is not int or launch["gpu_index"] < 0:
        raise ValueError("invalid launch GPU index")
    if type(launch["gpu_uuid"]) is not str or re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
        launch["gpu_uuid"],
    ) is None:
        raise ValueError("invalid launch GPU UUID")
    if type(launch["model_dir"]) is not str or not Path(launch["model_dir"]).is_absolute():
        raise ValueError("absolute local model directory required")
    source = Path(__file__).resolve(strict=True)
    if source.parent != directory or _file_digest(source) != launch["worker_sha256"].lower():
        raise ValueError("worker source identity mismatch")
    verified = check_model(launch["model_dir"], launch["model_hashes"])
    versions = {"python": platform.python_version()}
    versions.update({
        name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS if name != "python"
    })
    if versions != EXPECTED_VERSIONS:
        raise ValueError("installed runtime versions mismatch: " + json.dumps(versions, sort_keys=True))
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    validate_gpu(
        devices, processes, os.environ.get("CUDA_VISIBLE_DEVICES"),
        launch["gpu_uuid"], launch["gpu_index"],
    )
    return launch, verified, versions


def main(directory, input_stream=None, emit=None):
    started = time.perf_counter_ns()
    if input_stream is None:
        input_stream = sys.stdin.buffer
    if emit is None:
        output_stream = sys.stdout

        def emit(value):
            output_stream.write(_canonical(value).decode("utf-8") + "\n")
            output_stream.flush()

    launch, verified, versions = _preflight(directory)
    for name, value in (
        ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "1"), ("TOKENIZERS_PARALLELISM", "false"),
        ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
    ):
        os.environ[name] = value
    with redirect_stdout(sys.stderr):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if torch.cuda.device_count() != 1:
            raise ValueError("expected one visible GPU")
        torch.set_num_threads(1)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        load_started = time.perf_counter_ns()
        preparation_ns = load_started - started
        tokenizer = AutoTokenizer.from_pretrained(
            launch["model_dir"], local_files_only=True, trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            launch["model_dir"], local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, attn_implementation="eager",
        ).to("cuda:0").eval()
        generation, mode = prepare_generation(model, tokenizer, GenerationConfig)
        torch.cuda.synchronize()
        load_ns = time.perf_counter_ns() - load_started
        emit({
            "event": "ready", "run_id": launch["run_id"], "model_hashes": verified,
            "worker_sha256": launch["worker_sha256"], "versions": versions,
            "load_ns": load_ns, "preparation_ns": preparation_ns,
            "generation_config": generation.to_dict(),
            "generation_overrides": GENERATION_OVERRIDES.copy(), "effective_mode": mode,
            "gpu_uuid": launch["gpu_uuid"], "pid": os.getpid(),
            "hostname": platform.node(), "uid": os.getuid(),
        })
        calls = serve(
            input_stream, lambda request: generate_request(request, model, tokenizer, torch, generation),
            emit, launch["max_calls"],
        )
        emit({
            "event": "finished", "calls": calls,
            "worker_wall_ns": time.perf_counter_ns() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("expected the frozen run directory")
        main(sys.argv[1])
    except Exception as error:
        print(json.dumps({
            "event": "fatal", "error_type": type(error).__name__, "message": str(error),
        }, ensure_ascii=True, allow_nan=False), flush=True)
        raise SystemExit(1)
