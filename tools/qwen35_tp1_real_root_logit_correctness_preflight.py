"""Qwen3.5 TP1 real root-logit correctness preflight."""

from __future__ import annotations

import argparse
import math
import os
import json
import subprocess
import hashlib
import importlib.util
import importlib.metadata
import sys
import typing
import shutil
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from torch import nn


APPROVED_MODEL_DIR = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/"
    "qwen35-2b-hybrid-acquire-20260723-222004/model"
)
APPROVED_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
APPROVED_CONFIG_SHA256 = (
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
)
APPROVED_INDEX_SHA256 = (
    "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
)
APPROVED_SHARD_NAME = "model.safetensors-00001-of-00001.safetensors"
APPROVED_SHARD_SIZE = 4548221488
APPROVED_SHARD_SHA256 = (
    "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
)
APPROVED_COMPOSITE_SHA256 = (
    "27da983f5ef3e38480d8b5d5976e5c434fc4b5d0c70d09511c35154beecd8db9"
)
AUTHORIZATION_SHA256 = (
    "10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4"
)
MAX_TENSOR_BYTES = 1017118720
REFERENCE_MINIMUM_FREE_BYTES = 24 * 1024**3
REFERENCE_TENSOR_PARTIAL_NAME = "reference_logits.pt.partial"
REFERENCE_PROCESS_PARTIAL_NAME = "reference_process.json.partial"
TP1_RESULT_NAME = "tp1_real_root_logit_correctness.json"
TP1_REFERENCE_LOGITS_NAME = "reference_logits.pt"
TP1_NATIVE_LOGITS_NAME = "native_logits.pt"
TP1_SOURCE_MANIFEST_NAME = "source_manifest.json"
_NETWORK_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_ORIGINAL_CUSTOM_OP_INFER_SCHEMA = None


def _resolve_custom_op_schema(function, mutates_args=()):
    original = _ORIGINAL_CUSTOM_OP_INFER_SCHEMA
    if original is None:
        raise RuntimeError("custom-op schema compatibility is not active")
    annotations = getattr(function, "__annotations__", None)
    if not annotations or not any(
        isinstance(value, str) for value in annotations.values()
    ):
        return original(function, mutates_args)
    resolved = typing.get_type_hints(
        function,
        globalns=function.__globals__,
    )
    function.__annotations__ = resolved
    try:
        return original(function, mutates_args)
    finally:
        function.__annotations__ = annotations


@contextmanager
def torch_custom_op_annotation_compatibility(
    *,
    infer_schema_owner=None,
):
    global _ORIGINAL_CUSTOM_OP_INFER_SCHEMA
    if infer_schema_owner is None:
        import torch._custom_op.impl as infer_schema_owner
    if _ORIGINAL_CUSTOM_OP_INFER_SCHEMA is not None:
        raise RuntimeError("custom-op schema compatibility is nested")
    original = infer_schema_owner.infer_schema
    _ORIGINAL_CUSTOM_OP_INFER_SCHEMA = original
    infer_schema_owner.infer_schema = _resolve_custom_op_schema
    try:
        yield
    finally:
        infer_schema_owner.infer_schema = original
        _ORIGINAL_CUSTOM_OP_INFER_SCHEMA = None


class Qwen35TP1CausalAttentionBackend(nn.Module):

    def __init__(
        self,
        *,
        head_dim: int,
        query_heads: int,
        kv_heads: int,
    ):
        super().__init__()
        for name, value in (
            ("head_dim", head_dim),
            ("query_heads", query_heads),
            ("kv_heads", kv_heads),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if query_heads % kv_heads != 0:
            raise ValueError("query_heads must be divisible by kv_heads")
        self.head_dim = head_dim
        self.query_heads = query_heads
        self.kv_heads = kv_heads

    @staticmethod
    def _validate_tensor(
        tensor: torch.Tensor,
        *,
        name: str,
        token_count: int | None,
        width: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank two")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must use a floating point dtype")
        if token_count is not None and tensor.shape[0] != token_count:
            raise ValueError(f"{name} token count must match query")
        if tensor.shape[1] != width:
            raise ValueError(f"{name} width must equal {width}")
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(f"{name} dtype must match query dtype")
        if device is not None and tensor.device != device:
            raise ValueError(f"{name} device must match query device")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_tensor(
            query,
            name="query",
            token_count=None,
            width=self.query_heads * self.head_dim,
            dtype=None,
            device=None,
        )
        token_count = query.shape[0]
        for name, tensor in (("key", key), ("value", value)):
            self._validate_tensor(
                tensor,
                name=name,
                token_count=token_count,
                width=self.kv_heads * self.head_dim,
                dtype=query.dtype,
                device=query.device,
            )

        query_heads = query.reshape(
            token_count,
            self.query_heads,
            self.head_dim,
        ).float()
        key_heads = key.reshape(
            token_count,
            self.kv_heads,
            self.head_dim,
        ).float()
        value_heads = value.reshape(
            token_count,
            self.kv_heads,
            self.head_dim,
        ).float()
        repeats = self.query_heads // self.kv_heads
        if repeats != 1:
            key_heads = key_heads.repeat_interleave(repeats, dim=1)
            value_heads = value_heads.repeat_interleave(repeats, dim=1)

        scores = torch.einsum(
            "thd,shd->hts",
            query_heads,
            key_heads,
        )
        scores = scores / math.sqrt(self.head_dim)
        causal = torch.ones(
            token_count,
            token_count,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores = scores.masked_fill(
            ~causal.unsqueeze(0),
            float("-inf"),
        )
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum(
            "hts,shd->thd",
            weights,
            value_heads,
        )
        return output.reshape(
            token_count,
            self.query_heads * self.head_dim,
        ).to(dtype=query.dtype)


@dataclass(frozen=True)
class NativeCaseResult:
    case_id: str
    request_id: int
    lease_generation: int
    token_count: int
    logits: torch.Tensor
    state_nonzero_after_commit: dict[str, bool]
    release_zeroed: bool
    pool_binding_released: bool


@dataclass(frozen=True)
class ReferenceWorkerInvocation:
    argv: tuple[str, ...]
    environment: dict[str, str]
    gpu_index: int
    minimum_free_bytes: int


@dataclass(frozen=True)
class ReferenceWorkerResult:
    pid: int
    stdout: str
    stderr: str
    resource: dict[str, int | str]
    process_row: dict
    logits: dict[str, torch.Tensor]


@dataclass(frozen=True)
class RealTP1CPUCandidate:
    candidate: object
    pool: object
    metadata: object
    tensor_plan: object


def _gpu_index(value, *, name: str = "gpu_index") -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def require_reference_gpu_resource(
    *,
    gpu_index: int,
    query_gpu: Callable[[int], Mapping],
) -> dict[str, int | str]:
    selected_gpu = _gpu_index(gpu_index)
    if not callable(query_gpu):
        raise ValueError("query_gpu must be callable")
    observation = query_gpu(selected_gpu)
    if not isinstance(observation, Mapping):
        raise ValueError("GPU resource observation must be a mapping")
    if observation.get("gpu_index") != selected_gpu:
        raise ValueError("GPU resource observation must match selected GPU")
    gpu_uuid = observation.get("gpu_uuid")
    free_bytes = observation.get("free_bytes")
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("GPU resource observation must include a UUID")
    if (
        isinstance(free_bytes, bool)
        or not isinstance(free_bytes, int)
        or free_bytes < 0
    ):
        raise ValueError(
            "GPU resource observation must include non-negative free bytes"
        )
    if free_bytes < REFERENCE_MINIMUM_FREE_BYTES:
        raise ValueError(
            "reference worker requires at least 24 GiB free GPU memory"
        )
    return {
        "gpu_index": selected_gpu,
        "gpu_uuid": gpu_uuid,
        "free_bytes": free_bytes,
        "minimum_free_bytes": REFERENCE_MINIMUM_FREE_BYTES,
    }


def query_nvidia_smi_gpu(
    gpu_index: int,
    *,
    command_runner=subprocess.run,
) -> dict[str, int | str]:
    selected_gpu = _gpu_index(gpu_index)
    if not callable(command_runner):
        raise ValueError("command_runner must be callable")
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits",
    ]
    completed = command_runner(
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if getattr(completed, "returncode", None) != 0:
        raise ValueError("nvidia-smi GPU query failed")
    rows = {}
    for line in str(getattr(completed, "stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("nvidia-smi GPU query output is invalid")
        try:
            index = int(fields[0])
            free_mib = int(fields[2])
        except ValueError as exc:
            raise ValueError(
                "nvidia-smi GPU query output is invalid"
            ) from exc
        if index in rows or not fields[1] or free_mib < 0:
            raise ValueError("nvidia-smi GPU query output is invalid")
        rows[index] = {
            "gpu_index": index,
            "gpu_uuid": fields[1],
            "free_bytes": free_mib * 1024**2,
        }
    if selected_gpu not in rows:
        raise ValueError("selected GPU is missing from nvidia-smi output")
    return rows[selected_gpu]


def build_reference_worker_invocation(
    *,
    python_executable: str,
    script_path: str | os.PathLike,
    work_dir: str | os.PathLike,
    gpu_index: int,
    gpu_uuid: str | None = None,
    base_environment: Mapping[str, str] | None = None,
) -> ReferenceWorkerInvocation:
    selected_gpu = _gpu_index(gpu_index)
    if not isinstance(python_executable, str) or not python_executable:
        raise ValueError("python_executable must be a non-empty string")
    script = Path(script_path).resolve()
    directory = Path(work_dir)
    if not script.is_file():
        raise ValueError("reference worker script must exist")
    if not directory.is_absolute():
        raise ValueError("reference worker directory must be absolute")
    if base_environment is None:
        environment = dict(os.environ)
    elif isinstance(base_environment, Mapping) and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in base_environment.items()
    ):
        environment = dict(base_environment)
    else:
        raise ValueError("base_environment must map strings to strings")
    for variable in _NETWORK_PROXY_VARIABLES:
        environment.pop(variable, None)
    if gpu_uuid is not None and (
        not isinstance(gpu_uuid, str) or not gpu_uuid
    ):
        raise ValueError("gpu_uuid must be a non-empty string")
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(selected_gpu),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": str(selected_gpu),
    })
    if gpu_uuid is not None:
        environment["TINYVLLM_GATE_GPU_UUID"] = gpu_uuid
    argv = (
        python_executable,
        os.fspath(script),
        "internal-reference",
        "--model-dir",
        APPROVED_MODEL_DIR,
        "--model-manifest-sha256",
        APPROVED_MODEL_MANIFEST_SHA256,
        "--tensor-output",
        os.fspath(directory / REFERENCE_TENSOR_PARTIAL_NAME),
        "--process-output",
        os.fspath(directory / REFERENCE_PROCESS_PARTIAL_NAME),
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "eager",
        "--local-files-only",
        "--no-trust-remote-code",
        "--no-use-cache",
    )
    return ReferenceWorkerInvocation(
        argv=argv,
        environment=environment,
        gpu_index=selected_gpu,
        minimum_free_bytes=REFERENCE_MINIMUM_FREE_BYTES,
    )


def _launch_reference_process(
    invocation: ReferenceWorkerInvocation,
    work_dir: Path,
):
    return subprocess.Popen(
        invocation.argv,
        cwd=work_dir,
        env=invocation.environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _load_reference_logits(
    path: Path,
    *,
    expected_case_ids: tuple[str, ...],
    expected_vocab_size: int,
) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise ValueError("reference worker tensor output is missing")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("reference worker tensor output is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("reference worker tensor output must be a mapping")
    if tuple(payload) != expected_case_ids:
        raise ValueError("reference worker tensor case IDs mismatch")
    result = {}
    for case_id, tensor in payload.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.ndim != 1
            or tensor.shape[0] != expected_vocab_size
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                "reference worker logits must be finite contiguous CPU FP32 "
                "full-vocabulary rows"
            )
        result[case_id] = tensor
    return result


def _load_reference_process_row(
    path: Path,
    *,
    pid: int,
    gpu_index: int,
    resource: Mapping,
    expected_case_ids: tuple[str, ...],
    expected_vocab_size: int,
) -> dict:
    if not path.is_file():
        raise ValueError("reference worker process output is missing")
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference worker process output is invalid") from exc
    expected = {
        "worker": "reference",
        "pid": pid,
        "exit_code": 0,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "gpu_index": gpu_index,
        "gpu_uuid": resource["gpu_uuid"],
        "minimum_free_bytes": REFERENCE_MINIMUM_FREE_BYTES,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
        "case_ids": list(expected_case_ids),
        "vocab_size": expected_vocab_size,
        "cleanup_complete": True,
    }
    if not isinstance(row, dict) or any(
        row.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("reference worker process output contract mismatch")
    free_bytes_before = row.get("free_bytes_before")
    if (
        isinstance(free_bytes_before, bool)
        or not isinstance(free_bytes_before, int)
        or free_bytes_before < REFERENCE_MINIMUM_FREE_BYTES
    ):
        raise ValueError(
            "reference worker process output lacks 24 GiB memory preflight"
        )
    return row


def run_reference_worker(
    *,
    python_executable: str,
    script_path: str | os.PathLike,
    work_dir: str | os.PathLike,
    gpu_index: int,
    timeout_seconds: int,
    query_gpu: Callable[[int], Mapping],
    launch_process: Callable | None = None,
    pid_alive: Callable[[int], bool],
    expected_case_ids: tuple[str, ...],
    expected_vocab_size: int,
    base_environment: Mapping[str, str] | None = None,
) -> ReferenceWorkerResult:
    directory = Path(work_dir)
    if not directory.is_absolute() or not directory.is_dir():
        raise ValueError(
            "reference worker directory must be an existing absolute path"
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive integer")
    if (
        not isinstance(expected_case_ids, tuple)
        or not expected_case_ids
        or any(
            not isinstance(case_id, str) or not case_id
            for case_id in expected_case_ids
        )
        or len(set(expected_case_ids)) != len(expected_case_ids)
    ):
        raise ValueError("expected_case_ids must be unique non-empty strings")
    if (
        isinstance(expected_vocab_size, bool)
        or not isinstance(expected_vocab_size, int)
        or expected_vocab_size <= 0
    ):
        raise ValueError("expected_vocab_size must be a positive integer")
    if not callable(pid_alive):
        raise ValueError("pid_alive must be callable")
    for name in (
        REFERENCE_TENSOR_PARTIAL_NAME,
        REFERENCE_PROCESS_PARTIAL_NAME,
    ):
        if (directory / name).exists():
            raise ValueError("reference worker output path already exists")

    resource = require_reference_gpu_resource(
        gpu_index=gpu_index,
        query_gpu=query_gpu,
    )
    invocation = build_reference_worker_invocation(
        python_executable=python_executable,
        script_path=script_path,
        work_dir=directory,
        gpu_index=gpu_index,
        gpu_uuid=str(resource["gpu_uuid"]),
        base_environment=base_environment,
    )
    launcher = _launch_reference_process if launch_process is None else (
        launch_process
    )
    if not callable(launcher):
        raise ValueError("launch_process must be callable")
    try:
        process = launcher(invocation, directory)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("reference worker process launch failed") from exc
    pid = getattr(process, "pid", None)
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("reference worker process PID is invalid")
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise ValueError("reference worker timed out") from exc
    returncode = getattr(process, "returncode", None)
    if returncode != 0:
        detail = str(stderr).strip()[-4000:]
        raise ValueError(
            "reference worker exited with exit code "
            f"{returncode}: {detail}"
        )
    if pid_alive(pid):
        raise ValueError("reference worker PID is still alive after exit")

    tensor_path = directory / REFERENCE_TENSOR_PARTIAL_NAME
    process_path = directory / REFERENCE_PROCESS_PARTIAL_NAME
    logits = _load_reference_logits(
        tensor_path,
        expected_case_ids=expected_case_ids,
        expected_vocab_size=expected_vocab_size,
    )
    process_row = _load_reference_process_row(
        process_path,
        pid=pid,
        gpu_index=gpu_index,
        resource=resource,
        expected_case_ids=expected_case_ids,
        expected_vocab_size=expected_vocab_size,
    )
    return ReferenceWorkerResult(
        pid=pid,
        stdout=stdout,
        stderr=stderr,
        resource=dict(resource),
        process_row=process_row,
        logits=logits,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_approved_model_identity(
    model_dir: Path,
    model_manifest_sha256: str,
) -> None:
    if model_dir != Path(APPROVED_MODEL_DIR):
        raise ValueError("reference worker model directory is not approved")
    if model_manifest_sha256 != APPROVED_MODEL_MANIFEST_SHA256:
        raise ValueError("reference worker model manifest SHA256 mismatch")
    manifest_path = model_dir.parent / "model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("reference worker model manifest is missing")
    if _sha256_file(manifest_path) != model_manifest_sha256:
        raise ValueError("reference worker model manifest hash mismatch")
    required = {
        "config.json": (2908, APPROVED_CONFIG_SHA256),
        "model.safetensors.index.json": (
            64460,
            APPROVED_INDEX_SHA256,
        ),
        APPROVED_SHARD_NAME: (
            APPROVED_SHARD_SIZE,
            APPROVED_SHARD_SHA256,
        ),
    }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference worker model manifest is invalid") from exc
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("reference worker model manifest files are invalid")
    for name, (size, sha256) in required.items():
        row = files.get(name)
        path = model_dir / name
        if (
            not isinstance(row, dict)
            or row.get("size") != size
            or row.get("sha256") != sha256
            or not path.is_file()
            or path.stat().st_size != size
        ):
            raise ValueError(
                f"reference worker approved model file mismatch: {name}"
            )


def _read_process_memory() -> dict[str, int]:
    values = {}
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.strip().split()
        if name in ("VmRSS", "VmHWM") and parts:
            values[name.lower() + "_kib"] = int(parts[0])
    if set(values) != {"vmrss_kib", "vmhwm_kib"}:
        raise ValueError("reference worker process memory is incomplete")
    return values


def _package_version(name: str) -> str:
    return importlib.metadata.version(name)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_torch_save(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise ValueError("reference worker tensor output already exists")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise ValueError("reference worker process output already exists")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_reference_worker(
    *,
    model_dir: Path,
    model_manifest_sha256: str,
    tensor_output: Path,
    process_output: Path,
    prompt_cases: Iterable,
    expected_vocab_size: int,
    auto_model=None,
    cuda=None,
    process_id: int | None = None,
    gpu_index: int,
    gpu_uuid: str,
    verify_model_identity: Callable = _verify_approved_model_identity,
    process_memory_reader: Callable[[], Mapping] = _read_process_memory,
    version_reader: Callable[[str], str] = _package_version,
    timestamp_reader: Callable[[], str] = _utc_timestamp,
    custom_op_compatibility: Callable = (
        torch_custom_op_annotation_compatibility
    ),
) -> dict:
    model_dir = Path(model_dir)
    tensor_output = Path(tensor_output)
    process_output = Path(process_output)
    if tensor_output.parent != process_output.parent:
        raise ValueError("reference worker outputs must share one directory")
    if not tensor_output.parent.is_dir():
        raise ValueError("reference worker output directory is missing")
    if tensor_output.exists() or process_output.exists():
        raise ValueError("reference worker output path already exists")
    _lowercase_sha256(
        model_manifest_sha256,
        name="model_manifest_sha256",
    )
    if not callable(verify_model_identity):
        raise ValueError("verify_model_identity must be callable")
    verify_model_identity(model_dir, model_manifest_sha256)
    cases = tuple(prompt_cases)
    case_ids = tuple(_prompt_identity(case)[0] for case in cases)
    if not cases or len(set(case_ids)) != len(case_ids):
        raise ValueError("reference worker prompt cases are invalid")
    if (
        isinstance(expected_vocab_size, bool)
        or not isinstance(expected_vocab_size, int)
        or expected_vocab_size <= 0
    ):
        raise ValueError("expected_vocab_size must be a positive integer")
    physical_gpu_index = _gpu_index(gpu_index)
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("gpu_uuid must be a non-empty string")
    pid = os.getpid() if process_id is None else process_id
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process_id must be a positive integer")
    cuda = torch.cuda if cuda is None else cuda
    if not bool(cuda.is_available()):
        raise ValueError("reference worker CUDA is unavailable")
    free_bytes, _total_bytes = cuda.mem_get_info()
    if free_bytes < REFERENCE_MINIMUM_FREE_BYTES:
        raise ValueError(
            "reference worker requires at least 24 GiB free GPU memory"
        )
    if auto_model is None:
        from transformers import AutoModelForCausalLM
        auto_model = AutoModelForCausalLM
    if not callable(custom_op_compatibility):
        raise ValueError("custom_op_compatibility must be callable")
    start_timestamp = timestamp_reader()
    cuda.reset_peak_memory_stats()
    memory_before = dict(process_memory_reader())
    device = torch.device("cuda:0")
    model = None
    logits_by_case = {}
    cleanup_complete = False
    try:
        with custom_op_compatibility():
            model = auto_model.from_pretrained(
                model_dir,
                local_files_only=True,
                trust_remote_code=False,
                dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        model = model.to(device=device)
        model.eval()
        with torch.no_grad():
            for prompt_case in cases:
                case_id, token_ids = _prompt_identity(prompt_case)
                input_ids = torch.tensor(
                    [token_ids],
                    dtype=torch.int64,
                    device=device,
                )
                outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits = getattr(outputs, "logits", None)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 3
                    or logits.shape[0] != 1
                    or logits.shape[1] != len(token_ids)
                    or logits.shape[2] != expected_vocab_size
                    or not logits.is_floating_point()
                    or not bool(torch.isfinite(logits).all())
                ):
                    raise ValueError(
                        "reference worker logits shape or values are invalid"
                    )
                logits_by_case[case_id] = logits[
                    0, -1
                ].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                ).contiguous()
        cuda.synchronize()
        max_allocated = int(cuda.max_memory_allocated())
        max_reserved = int(cuda.max_memory_reserved())
        memory_after = dict(process_memory_reader())
        del model
        model = None
        cuda.empty_cache()
        cleanup_complete = True
    finally:
        if model is not None:
            del model
            cuda.empty_cache()
    if tuple(logits_by_case) != case_ids:
        raise ValueError("reference worker logits inventory is incomplete")
    finish_timestamp = timestamp_reader()
    row = {
        "worker": "reference",
        "pid": pid,
        "exit_code": 0,
        "model_manifest_sha256": model_manifest_sha256,
        "gpu_index": physical_gpu_index,
        "gpu_uuid": gpu_uuid,
        "free_bytes_before": int(free_bytes),
        "minimum_free_bytes": REFERENCE_MINIMUM_FREE_BYTES,
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "use_cache": False,
        "case_ids": list(case_ids),
        "vocab_size": expected_vocab_size,
        "start_timestamp": start_timestamp,
        "finish_timestamp": finish_timestamp,
        "torch_version": version_reader("torch"),
        "transformers_version": version_reader("transformers"),
        "vmrss_kib": int(memory_after["vmrss_kib"]),
        "vmhwm_kib": int(memory_after["vmhwm_kib"]),
        "vmrss_before_kib": int(memory_before["vmrss_kib"]),
        "vmhwm_before_kib": int(memory_before["vmhwm_kib"]),
        "max_memory_allocated": max_allocated,
        "max_memory_reserved": max_reserved,
        "cleanup_complete": cleanup_complete,
    }
    _atomic_torch_save(tensor_output, logits_by_case)
    _atomic_write_json(process_output, row)
    return row


def execute_native_worker(
    *,
    tensor_output: Path,
    process_output: Path,
    state_output: Path,
    prompt_cases: Iterable,
    expected_vocab_size: int,
    build_candidate=None,
    move_candidate=None,
    run_cases=None,
    allocator_factory=None,
    cuda=None,
    process_id: int | None = None,
    gpu_index: int,
    gpu_uuid: str,
    process_memory_reader: Callable[[], Mapping] = _read_process_memory,
    version_reader: Callable[[str], str] = _package_version,
    timestamp_reader: Callable[[], str] = _utc_timestamp,
) -> dict:
    tensor_output = Path(tensor_output)
    process_output = Path(process_output)
    state_output = Path(state_output)
    if len({
        tensor_output.parent,
        process_output.parent,
        state_output.parent,
    }) != 1:
        raise ValueError("native worker outputs must share one directory")
    if not tensor_output.parent.is_dir():
        raise ValueError("native worker output directory is missing")
    if any(
        path.exists()
        for path in (tensor_output, process_output, state_output)
    ):
        raise ValueError("native worker output path already exists")
    if build_candidate is None:
        build_candidate = build_real_tp1_cpu_candidate
    if move_candidate is None:
        move_candidate = move_loaded_candidate_to_device
    if run_cases is None:
        run_cases = run_native_cases
    for name, dependency in (
        ("build_candidate", build_candidate),
        ("move_candidate", move_candidate),
        ("run_cases", run_cases),
    ):
        if not callable(dependency):
            raise ValueError(f"{name} must be callable")
    cases = tuple(prompt_cases)
    case_ids = tuple(_prompt_identity(case)[0] for case in cases)
    if not cases or len(set(case_ids)) != len(case_ids):
        raise ValueError("native worker prompt cases are invalid")
    if (
        isinstance(expected_vocab_size, bool)
        or not isinstance(expected_vocab_size, int)
        or expected_vocab_size <= 0
    ):
        raise ValueError("expected_vocab_size must be a positive integer")
    physical_gpu_index = _gpu_index(gpu_index)
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError("gpu_uuid must be a non-empty string")
    pid = os.getpid() if process_id is None else process_id
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("process_id must be a positive integer")
    cuda = torch.cuda if cuda is None else cuda
    if not bool(cuda.is_available()):
        raise ValueError("native worker CUDA is unavailable")
    free_bytes, _total_bytes = cuda.mem_get_info()
    if free_bytes < REFERENCE_MINIMUM_FREE_BYTES:
        raise ValueError(
            "native worker requires at least 24 GiB free GPU memory"
        )
    if allocator_factory is None:
        from tinyvllm.engine.hybrid_state import HybridStateSlotAllocator
        allocator_factory = HybridStateSlotAllocator
    if not callable(allocator_factory):
        raise ValueError("allocator_factory must be callable")
    start_timestamp = timestamp_reader()
    cuda.reset_peak_memory_stats()
    memory_before = dict(process_memory_reader())
    candidate = None
    logits_by_case = {}
    state_rows = []
    cleanup_complete = False
    try:
        built = build_candidate()
        candidate = getattr(built, "candidate", None)
        if candidate is None:
            raise ValueError("native worker candidate build is invalid")
        candidate = move_candidate(
            candidate,
            device="cuda:0",
            expected_model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        )
        allocator = allocator_factory(1)
        results = run_cases(
            candidate=candidate,
            prompt_cases=cases,
            expected_model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
            device="cuda:0",
            allocator=allocator,
            first_request_id=100,
        )
        if not isinstance(results, tuple) or len(results) != len(cases):
            raise ValueError("native worker result inventory is invalid")
        for case, result in zip(cases, results):
            case_id, token_ids = _prompt_identity(case)
            if (
                getattr(result, "case_id", None) != case_id
                or getattr(result, "token_count", None) != len(token_ids)
            ):
                raise ValueError("native worker result identity mismatch")
            logits = getattr(result, "logits", None)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.device.type != "cpu"
                or logits.dtype != torch.float32
                or logits.ndim != 1
                or logits.shape[0] != expected_vocab_size
                or not logits.is_contiguous()
                or not bool(torch.isfinite(logits).all())
            ):
                raise ValueError("native worker logits are invalid")
            state = getattr(result, "state_nonzero_after_commit", None)
            if (
                not isinstance(state, dict)
                or len(state) != 36
                or any(value is not True for value in state.values())
            ):
                raise ValueError(
                    "native worker state mutation evidence is invalid"
                )
            linear_layers = {
                int(key.split(":", 1)[0])
                for key in state
                if key.endswith(":linear_convolution")
            }
            recurrent_layers = {
                int(key.split(":", 1)[0])
                for key in state
                if key.endswith(":linear_recurrent")
            }
            if (
                len(linear_layers) != 18
                or recurrent_layers != linear_layers
            ):
                raise ValueError(
                    "native worker state layer evidence is invalid"
                )
            if (
                getattr(result, "release_zeroed", None) is not True
                or getattr(result, "pool_binding_released", None) is not True
            ):
                raise ValueError("native worker release evidence is invalid")
            logits_by_case[case_id] = logits.detach().clone().contiguous()
            state_rows.append({
                "case_id": case_id,
                "request_id": int(result.request_id),
                "lease_generation": int(result.lease_generation),
                "prepare_read_only": True,
                "linear_layer_count": len(linear_layers),
                "changed_component_count": len(state),
                "full_attention_state_component_count": 0,
                "commit_count": 1,
                "release_zeroed": True,
                "pool_binding_released": True,
            })
        cuda.synchronize()
        max_allocated = int(cuda.max_memory_allocated())
        max_reserved = int(cuda.max_memory_reserved())
        memory_after = dict(process_memory_reader())
        del candidate
        candidate = None
        cuda.empty_cache()
        cleanup_complete = True
    finally:
        if candidate is not None:
            del candidate
            cuda.empty_cache()
    finish_timestamp = timestamp_reader()
    row = {
        "worker": "native",
        "pid": pid,
        "exit_code": 0,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "gpu_index": physical_gpu_index,
        "gpu_uuid": gpu_uuid,
        "free_bytes_before": int(free_bytes),
        "minimum_free_bytes": REFERENCE_MINIMUM_FREE_BYTES,
        "case_ids": list(case_ids),
        "vocab_size": expected_vocab_size,
        "tensor_parallel_size": 1,
        "tensor_parallel_rank": 0,
        "dtype": "bfloat16",
        "recurrent_dtype": "float32",
        "engine_constructed": False,
        "model_runner_constructed": False,
        "scheduler_constructed": False,
        "sampler_constructed": False,
        "start_timestamp": start_timestamp,
        "finish_timestamp": finish_timestamp,
        "torch_version": version_reader("torch"),
        "vmrss_kib": int(memory_after["vmrss_kib"]),
        "vmhwm_kib": int(memory_after["vmhwm_kib"]),
        "vmrss_before_kib": int(memory_before["vmrss_kib"]),
        "vmhwm_before_kib": int(memory_before["vmhwm_kib"]),
        "max_memory_allocated": max_allocated,
        "max_memory_reserved": max_reserved,
        "cleanup_complete": cleanup_complete,
    }
    _atomic_torch_save(tensor_output, logits_by_case)
    _atomic_write_json(state_output, state_rows)
    _atomic_write_json(process_output, row)
    return row


class _RealTP1Dependencies:

    @staticmethod
    def make_shard_identity(**kwargs):
        from tinyvllm.models.qwen35_checkpoint_metadata import (
            Qwen35CheckpointShardIdentity,
        )
        return Qwen35CheckpointShardIdentity(**kwargs)

    @staticmethod
    def read_metadata(checkpoint_dir, **kwargs):
        from tinyvllm.models.qwen35_checkpoint_metadata import (
            read_qwen35_checkpoint_metadata,
        )
        return read_qwen35_checkpoint_metadata(checkpoint_dir, **kwargs)

    @staticmethod
    def build_tensor_plan(hf_config, index_payload, shard_headers):
        from tinyvllm.models.qwen35_checkpoint import (
            build_qwen35_checkpoint_tensor_plan,
        )
        return build_qwen35_checkpoint_tensor_plan(
            hf_config,
            index_payload,
            shard_headers,
        )

    @staticmethod
    def build_layout(hf_config, **kwargs):
        from tinyvllm.engine.qwen35_hybrid_state import (
            build_qwen35_hybrid_state_layout,
        )
        return build_qwen35_hybrid_state_layout(hf_config, **kwargs)

    @staticmethod
    def make_pool(layout, **kwargs):
        from tinyvllm.engine.hybrid_state import HybridStateTensorPool
        return HybridStateTensorPool(layout, **kwargs)

    @staticmethod
    def prepare_target(hf_config, tensor_plan, **kwargs):
        from tinyvllm.models.qwen35_checkpoint_candidate_factory import (
            prepare_qwen35_checkpoint_candidate_target,
        )
        return prepare_qwen35_checkpoint_candidate_target(
            hf_config,
            tensor_plan,
            **kwargs,
        )

    @staticmethod
    def build_loader(provider, **kwargs):
        from tinyvllm.models.qwen35_checkpoint_candidate_loader import (
            build_qwen35_authorized_checkpoint_candidate_loader,
        )
        return build_qwen35_authorized_checkpoint_candidate_loader(
            provider,
            **kwargs,
        )

    @staticmethod
    def make_request(**kwargs):
        from tinyvllm.models.qwen35_checkpoint_worker import (
            Qwen35CheckpointCandidateLoadRequest,
        )
        return Qwen35CheckpointCandidateLoadRequest(**kwargs)


def _lowercase_sha256(value, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _prompt_identity(prompt_case) -> tuple[str, tuple[int, ...]]:
    case_id = getattr(prompt_case, "case_id", None)
    token_ids = getattr(prompt_case, "token_ids", None)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("prompt case_id must be a non-empty string")
    if (
        not isinstance(token_ids, tuple)
        or not token_ids
        or any(
            type(token_id) is not int or token_id < 0
            for token_id in token_ids
        )
    ):
        raise ValueError(
            "prompt token_ids must be a non-empty tuple of integers"
        )
    return case_id, token_ids


def _candidate_parts(
    candidate,
    *,
    expected_model_fingerprint: str,
    require_run_step: bool = True,
):
    _lowercase_sha256(
        expected_model_fingerprint,
        name="expected_model_fingerprint",
    )
    if (
        getattr(candidate, "model_fingerprint", None)
        != expected_model_fingerprint
    ):
        raise ValueError("candidate model fingerprint mismatch")
    binding_plan = getattr(candidate, "binding_plan", None)
    if (
        getattr(binding_plan, "tensor_parallel_size", None) != 1
        or getattr(binding_plan, "tensor_parallel_rank", None) != 0
    ):
        raise ValueError("candidate must use exact TP1 rank-zero binding")
    owner = getattr(candidate, "owner", None)
    model = getattr(owner, "model", None)
    pool = getattr(owner, "pool", None)
    if model is None:
        raise ValueError("candidate owner model is invalid")
    if require_run_step and not callable(getattr(model, "run_step", None)):
        raise ValueError("candidate owner model run_step is invalid")
    if pool is None:
        raise ValueError("candidate owner pool is invalid")
    if getattr(model, "pool", pool) is not pool:
        raise ValueError("candidate model and owner pool identity mismatch")
    return model, pool


def _build_attention_backend(
    _layer_index: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
) -> Qwen35TP1CausalAttentionBackend:
    return Qwen35TP1CausalAttentionBackend(
        head_dim=head_dim,
        query_heads=query_heads,
        kv_heads=kv_heads,
    )


def build_real_tp1_cpu_candidate(
    *,
    dependencies=None,
) -> RealTP1CPUCandidate:
    dependencies = (
        _RealTP1Dependencies()
        if dependencies is None
        else dependencies
    )
    required = (
        "make_shard_identity",
        "read_metadata",
        "build_tensor_plan",
        "build_layout",
        "make_pool",
        "prepare_target",
        "build_loader",
        "make_request",
    )
    if any(
        not callable(getattr(dependencies, name, None))
        for name in required
    ):
        raise ValueError("real TP1 dependencies are incomplete")
    shard = dependencies.make_shard_identity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    metadata = dependencies.read_metadata(
        APPROVED_MODEL_DIR,
        shards=(shard,),
        expected_config_sha256=APPROVED_CONFIG_SHA256,
        expected_index_sha256=APPROVED_INDEX_SHA256,
        expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
    )
    tensor_plan = dependencies.build_tensor_plan(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
    )
    layout = dependencies.build_layout(
        metadata.hf_config,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
        speculative_tokens=1,
    )
    pool = dependencies.make_pool(
        layout,
        capacity=1,
        device="cpu",
    )
    target = dependencies.prepare_target(
        metadata.hf_config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        build_attention_backend=_build_attention_backend,
        parameter_device="cpu",
    )
    if getattr(target, "pool", None) is not pool:
        raise ValueError("prepared target pool identity mismatch")
    provider_calls = 0

    def provide_target():
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls != 1:
            raise RuntimeError("prepared target provider called more than once")
        return target

    loader = dependencies.build_loader(
        provide_target,
        authorization_sha256=AUTHORIZATION_SHA256,
    )
    request = dependencies.make_request(
        checkpoint_dir=APPROVED_MODEL_DIR,
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        max_tensor_bytes=MAX_TENSOR_BYTES,
        authorization_sha256=AUTHORIZATION_SHA256,
    )
    candidate = loader(request)
    _, candidate_pool = _candidate_parts(
        candidate,
        expected_model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        require_run_step=False,
    )
    if provider_calls != 1:
        raise RuntimeError("prepared target provider call count mismatch")
    if candidate_pool is not pool:
        raise ValueError("loaded candidate pool identity mismatch")
    return RealTP1CPUCandidate(
        candidate=candidate,
        pool=pool,
        metadata=metadata,
        tensor_plan=tensor_plan,
    )


def move_loaded_candidate_to_device(
    candidate,
    *,
    device: str | torch.device,
    expected_model_fingerprint: str,
):
    model, pool = _candidate_parts(
        candidate,
        expected_model_fingerprint=expected_model_fingerprint,
        require_run_step=False,
    )
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device is invalid") from exc
    if target_device.type not in ("cuda", "meta"):
        raise ValueError("loaded candidate target device must be CUDA")
    if getattr(pool, "device", None) != torch.device("cpu"):
        raise ValueError("loaded candidate pool must start on CPU")
    tensors = getattr(pool, "_tensors", None)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("loaded candidate pool tensor inventory is invalid")
    if getattr(pool, "_bindings", None):
        raise ValueError("loaded candidate pool must be unbound")

    migrated_tensors = {
        key: tensor.to(device=target_device)
        for key, tensor in tensors.items()
    }
    model.to(device=target_device)
    if any(
        tensor.device != target_device
        for tensor in (
            list(model.parameters())
            + list(model.buffers())
        )
    ):
        raise RuntimeError("loaded candidate model migration is incomplete")
    if any(
        tensor.device != target_device
        for tensor in migrated_tensors.values()
    ):
        raise RuntimeError("loaded candidate pool migration is incomplete")
    pool._tensors = migrated_tensors
    pool.device = target_device
    owner = candidate.owner
    transaction = getattr(owner, "state_transaction", None)
    adapters = getattr(transaction, "adapters", ())
    if (
        transaction is not None
        and getattr(transaction, "pool", None) is not pool
    ):
        raise RuntimeError("loaded candidate transaction lost pool identity")
    for adapter in adapters:
        if getattr(adapter, "pool", None) is not pool:
            raise RuntimeError("loaded candidate adapter lost pool identity")
        layer_index = getattr(adapter, "layer_index", None)
        convolution = migrated_tensors.get(
            (layer_index, "linear_convolution")
        )
        recurrent = migrated_tensors.get(
            (layer_index, "linear_recurrent")
        )
        if convolution is None or recurrent is None:
            raise RuntimeError(
                "loaded candidate adapter state mapping is incomplete"
            )
        adapter.convolution = convolution
        adapter.recurrent = recurrent
    runtime_bridge = getattr(owner, "runtime_bridge", None)
    if (
        runtime_bridge is not None
        and getattr(runtime_bridge, "pool", None) is not pool
    ):
        raise RuntimeError("loaded candidate runtime bridge lost pool identity")
    return candidate


def _state_nonzero(pool, lease) -> dict[str, bool]:
    tensors = getattr(pool, "_tensors", None)
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("candidate pool tensor inventory is invalid")
    result = {}
    for key, tensor in sorted(tensors.items()):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(tensor, torch.Tensor)
        ):
            raise ValueError("candidate pool tensor entry is invalid")
        layer_index, role = key
        value = tensor[lease.slot_id]
        result[f"{layer_index}:{role}"] = bool(torch.count_nonzero(value))
    return result


def _pool_zeroed(pool) -> bool:
    tensors = getattr(pool, "_tensors", None)
    return (
        isinstance(tensors, dict)
        and bool(tensors)
        and all(
            isinstance(tensor, torch.Tensor)
            and not bool(torch.count_nonzero(tensor))
            for tensor in tensors.values()
        )
    )


def run_native_case(
    *,
    candidate,
    prompt_case,
    expected_model_fingerprint: str,
    request_id: int,
    device: str | torch.device,
    allocator,
    set_context=None,
    reset_context=None,
) -> NativeCaseResult:
    model, pool = _candidate_parts(
        candidate,
        expected_model_fingerprint=expected_model_fingerprint,
    )
    case_id, token_ids = _prompt_identity(prompt_case)
    if (
        isinstance(request_id, bool)
        or not isinstance(request_id, int)
        or request_id < 0
    ):
        raise ValueError("request_id must be a non-negative integer")
    try:
        execution_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device is invalid") from exc
    if getattr(pool, "device", None) != execution_device:
        raise ValueError("candidate pool device does not match execution device")
    allocate = getattr(allocator, "allocate", None)
    release_allocator = getattr(allocator, "release", None)
    if not callable(allocate) or not callable(release_allocator):
        raise ValueError("allocator must provide allocate and release")
    activate = getattr(pool, "activate", None)
    release_pool = getattr(pool, "release", None)
    if not callable(activate) or not callable(release_pool):
        raise ValueError("pool must provide activate and release")
    if (
        (set_context is None or reset_context is None)
        and isinstance(model, nn.Module)
    ):
        from tinyvllm.utils.context import (
            reset_context as production_reset_context,
            set_context as production_set_context,
        )
        if set_context is None:
            set_context = production_set_context
        if reset_context is None:
            reset_context = production_reset_context
    if set_context is None:
        set_context = lambda **_kwargs: None
    if reset_context is None:
        reset_context = lambda: None
    if not callable(set_context) or not callable(reset_context):
        raise ValueError("context setters must be callable")

    lease = allocate(request_id)
    pool_activated = False
    state_nonzero = {}
    final_logits = None
    try:
        activate(lease)
        pool_activated = True
        input_ids = torch.tensor(
            token_ids,
            dtype=torch.int64,
            device=execution_device,
        )
        position_ids = torch.arange(
            len(token_ids),
            dtype=torch.int64,
            device=execution_device,
        )
        context_set = False
        try:
            cumulative = torch.tensor(
                [0, len(token_ids)],
                dtype=torch.int32,
                device=execution_device,
            )
            set_context(
                is_prefill=True,
                mode="prefill",
                cu_seqlens_q=cumulative,
                cu_seqlens_k=cumulative.clone(),
                max_seqlen_q=len(token_ids),
                max_seqlen_k=len(token_ids),
                logits_indices=torch.tensor(
                    [len(token_ids) - 1],
                    dtype=torch.int64,
                    device=execution_device,
                ),
            )
            context_set = True
            with torch.no_grad():
                normalized, logits = model.run_step(
                    (lease,),
                    (len(token_ids),),
                    input_ids,
                    position_ids,
                )
        finally:
            if context_set:
                reset_context()
        for name, tensor in (
            ("normalized", normalized),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 2
                or tensor.shape[0] != len(token_ids)
                or not tensor.is_floating_point()
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(
                    f"native {name} must be finite rank-two token output"
                )
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 2
            or logits.shape[0] not in (1, len(token_ids))
            or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())
            or logits.shape[1] <= 0
        ):
            raise ValueError(
                "native logits must be finite final-token or token-row output"
            )
        final_logits = logits[-1].detach().to(
            device="cpu",
            dtype=torch.float32,
        ).contiguous()
        state_nonzero = _state_nonzero(pool, lease)
    finally:
        if pool_activated:
            release_pool(lease)
        release_allocator(lease)

    release_zeroed = _pool_zeroed(pool)
    bindings = getattr(pool, "_bindings", None)
    binding_released = isinstance(bindings, dict) and not bindings
    if not release_zeroed:
        raise RuntimeError("native pool was not zeroed after release")
    if not binding_released:
        raise RuntimeError("native pool binding survived release")
    return NativeCaseResult(
        case_id=case_id,
        request_id=request_id,
        lease_generation=int(lease.generation),
        token_count=len(token_ids),
        logits=final_logits,
        state_nonzero_after_commit=state_nonzero,
        release_zeroed=release_zeroed,
        pool_binding_released=binding_released,
    )


def run_native_cases(
    *,
    candidate,
    prompt_cases: Iterable,
    expected_model_fingerprint: str,
    device: str | torch.device,
    allocator,
    first_request_id: int,
    set_context=None,
    reset_context=None,
) -> tuple[NativeCaseResult, ...]:
    cases = tuple(prompt_cases)
    if not cases:
        raise ValueError("prompt_cases must not be empty")
    return tuple(
        run_native_case(
            candidate=candidate,
            prompt_case=prompt_case,
            expected_model_fingerprint=expected_model_fingerprint,
            request_id=first_request_id + index,
            device=device,
            allocator=allocator,
            set_context=set_context,
            reset_context=reset_context,
        )
        for index, prompt_case in enumerate(cases)
    )


def _load_frozen_prompt_cases():
    return _load_frozen_contract_module().prompt_cases()


def _load_frozen_contract_module():
    contract_path = (
        Path(__file__).resolve().parent
        / "qwen35_tp1_real_root_logit_correctness_contract.py"
    )
    if not contract_path.is_file():
        raise ValueError("frozen correctness contract is missing")
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp1_real_root_logit_correctness_contract_for_worker",
        os.fspath(contract_path),
    )
    if spec is None or spec.loader is None:
        raise ValueError("frozen correctness contract is invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _validate_artifact_process_row(
    row,
    *,
    worker: str,
    expected_case_ids: tuple[str, ...],
    expected_vocab_size: int,
) -> dict:
    if not isinstance(row, dict) or row.get("worker") != worker:
        raise ValueError(f"{worker} process row is invalid")
    for name in ("pid", "free_bytes_before", "minimum_free_bytes"):
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{worker} process {name} is invalid")
    if row["free_bytes_before"] < REFERENCE_MINIMUM_FREE_BYTES:
        raise ValueError(f"{worker} process requires at least 24 GiB")
    if row["minimum_free_bytes"] != REFERENCE_MINIMUM_FREE_BYTES:
        raise ValueError(f"{worker} process memory floor mismatch")
    if (
        row.get("exit_code") != 0
        or row.get("cleanup_complete") is not True
    ):
        raise ValueError(f"{worker} process cleanup evidence is invalid")
    if (
        row.get("model_manifest_sha256")
        != APPROVED_MODEL_MANIFEST_SHA256
    ):
        raise ValueError(f"{worker} process model identity mismatch")
    if row.get("case_ids") != list(expected_case_ids):
        raise ValueError(f"{worker} process case inventory mismatch")
    if row.get("vocab_size") != expected_vocab_size:
        raise ValueError(f"{worker} process vocabulary mismatch")
    for name in (
        "gpu_index",
        "gpu_uuid",
        "start_timestamp",
        "finish_timestamp",
        "torch_version",
        "vmrss_kib",
        "vmhwm_kib",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        if name not in row:
            raise ValueError(f"{worker} process evidence is incomplete")
    if worker == "reference":
        expected = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "use_cache": False,
        }
    else:
        expected = {
            "tensor_parallel_size": 1,
            "tensor_parallel_rank": 0,
            "dtype": "bfloat16",
            "recurrent_dtype": "float32",
            "engine_constructed": False,
            "model_runner_constructed": False,
            "scheduler_constructed": False,
            "sampler_constructed": False,
        }
    if any(row.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{worker} process execution contract mismatch")
    return dict(row)


def _validate_artifact_tensor_map(
    value,
    *,
    expected_case_ids: tuple[str, ...],
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or tuple(value) != expected_case_ids:
        raise ValueError(f"{label} logits case inventory mismatch")
    result = {}
    vocabulary = None
    for case_id, tensor in value.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.ndim != 1
            or tensor.shape[0] < 2
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"{label} logits must be finite contiguous CPU FP32 rows"
            )
        if vocabulary is None:
            vocabulary = tensor.shape[0]
        elif tensor.shape[0] != vocabulary:
            raise ValueError(f"{label} logits vocabulary width mismatch")
        result[case_id] = tensor.detach().clone().contiguous()
    return result


def _validate_state_rows(
    rows,
    *,
    expected_case_ids: tuple[str, ...],
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != len(expected_case_ids):
        raise ValueError("native state evidence is invalid")
    result = []
    for case_id, row in zip(expected_case_ids, rows):
        if not isinstance(row, dict) or row.get("case_id") != case_id:
            raise ValueError("native state case order is invalid")
        expected = {
            "prepare_read_only": True,
            "linear_layer_count": 18,
            "changed_component_count": 36,
            "full_attention_state_component_count": 0,
            "commit_count": 1,
            "release_zeroed": True,
            "pool_binding_released": True,
        }
        if any(row.get(name) != value for name, value in expected.items()):
            raise ValueError("native state evidence is invalid")
        result.append(dict(row))
    return result


def _validate_source_manifest(value) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("source manifest is invalid")
    source_hashes = value.get("source_file_sha256")
    if (
        not isinstance(source_hashes, dict)
        or not source_hashes
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in source_hashes.items()
        )
    ):
        raise ValueError("source manifest source closure is invalid")
    source_hashes = dict(sorted(source_hashes.items()))
    source_tree = hashlib.sha256(
        json.dumps(
            source_hashes,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "source_tree_sha256": source_tree,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "shard_name": APPROVED_SHARD_NAME,
        "shard_size": APPROVED_SHARD_SIZE,
        "shard_sha256": APPROVED_SHARD_SHA256,
    }
    if any(value.get(name) != item for name, item in expected.items()):
        raise ValueError("source manifest identity mismatch")
    result = dict(value)
    result["source_file_sha256"] = source_hashes
    return result


def finalize_tp1_correctness_artifact(
    *,
    run_dir,
    run_tag: str,
    reference_logits,
    native_logits,
    reference_process,
    native_process,
    state_rows,
    source_manifest,
    forbidden_counters,
) -> tuple[Path, ...]:
    directory = Path(run_dir)
    if directory.exists():
        if any(directory.iterdir()):
            raise ValueError("TP1 correctness run directory is not empty")
    else:
        directory.mkdir(parents=True)
    if (
        not isinstance(run_tag, str)
        or not run_tag
        or any(
            character not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789-_"
            )
            for character in run_tag
        )
    ):
        raise ValueError("TP1 correctness run tag is invalid")
    contract = _load_frozen_contract_module()
    prompts = contract.prompt_cases()
    case_ids = tuple(case.case_id for case in prompts)
    reference = _validate_artifact_tensor_map(
        reference_logits,
        expected_case_ids=case_ids,
        label="reference",
    )
    native = _validate_artifact_tensor_map(
        native_logits,
        expected_case_ids=case_ids,
        label="native",
    )
    vocabulary = next(iter(reference.values())).shape[0]
    if any(tensor.shape[0] != vocabulary for tensor in native.values()):
        raise ValueError("reference and native vocabulary width mismatch")
    reference_row = _validate_artifact_process_row(
        reference_process,
        worker="reference",
        expected_case_ids=case_ids,
        expected_vocab_size=vocabulary,
    )
    native_row = _validate_artifact_process_row(
        native_process,
        worker="native",
        expected_case_ids=case_ids,
        expected_vocab_size=vocabulary,
    )
    if reference_row["pid"] == native_row["pid"]:
        raise ValueError("reference and native processes must be separate")
    states = _validate_state_rows(
        state_rows,
        expected_case_ids=case_ids,
    )
    expected_forbidden = {
        "engine",
        "model_runner",
        "scheduler",
        "sampler",
        "generation",
    }
    if (
        not isinstance(forbidden_counters, dict)
        or set(forbidden_counters) != expected_forbidden
        or any(value != 0 for value in forbidden_counters.values())
    ):
        raise ValueError("forbidden execution counters are invalid")
    source = _validate_source_manifest(source_manifest)
    comparisons = [
        {
            "case_id": case_id,
            **contract.compare_logits(
                native[case_id],
                reference[case_id],
                tolerance=contract.BF16_DECISION_TOLERANCE,
            ),
        }
        for case_id in case_ids
    ]
    classification = contract.classify_rows(comparisons)
    result = {
        "schema_version": 1,
        "run_tag": run_tag,
        "classification": classification,
        "comparison_policy": "bf16_decision_preserving",
        "tolerance": {
            "atol": contract.BF16_DECISION_TOLERANCE.atol,
            "rtol": contract.BF16_DECISION_TOLERANCE.rtol,
        },
        "prompts": [
            {
                "case_id": case.case_id,
                "token_ids": list(case.token_ids),
                "token_sha256": case.token_sha256,
            }
            for case in prompts
        ],
        "processes": {
            "reference": reference_row,
            "native": native_row,
        },
        "state_rows": states,
        "forbidden_counters": dict(sorted(forbidden_counters.items())),
        "comparisons": comparisons,
        "claim_boundary": (
            "TP1 one-shot final-token root-logit correctness only; no TP4, "
            "Engine, cached decode, speed, cache, memory, compression, or "
            "quality claim."
        ),
    }
    final_paths = {
        TP1_RESULT_NAME: directory / TP1_RESULT_NAME,
        TP1_REFERENCE_LOGITS_NAME: directory / TP1_REFERENCE_LOGITS_NAME,
        TP1_NATIVE_LOGITS_NAME: directory / TP1_NATIVE_LOGITS_NAME,
        TP1_SOURCE_MANIFEST_NAME: directory / TP1_SOURCE_MANIFEST_NAME,
    }
    partial_paths = {
        name: path.with_name(path.name + ".partial")
        for name, path in final_paths.items()
    }
    if any(path.exists() for path in (*final_paths.values(), *partial_paths.values())):
        raise ValueError("TP1 correctness artifact path already exists")
    published = []
    try:
        torch.save(reference, partial_paths[TP1_REFERENCE_LOGITS_NAME])
        torch.save(native, partial_paths[TP1_NATIVE_LOGITS_NAME])
        partial_paths[TP1_RESULT_NAME].write_bytes(
            _canonical_json_bytes(result)
        )
        source["artifacts"] = {
            name: {
                "size": partial_paths[name].stat().st_size,
                "sha256": _sha256_file(partial_paths[name]),
            }
            for name in (
                TP1_RESULT_NAME,
                TP1_REFERENCE_LOGITS_NAME,
                TP1_NATIVE_LOGITS_NAME,
            )
        }
        partial_paths[TP1_SOURCE_MANIFEST_NAME].write_bytes(
            _canonical_json_bytes(source)
        )
        for name in (
            TP1_RESULT_NAME,
            TP1_REFERENCE_LOGITS_NAME,
            TP1_NATIVE_LOGITS_NAME,
            TP1_SOURCE_MANIFEST_NAME,
        ):
            os.replace(partial_paths[name], final_paths[name])
            published.append(final_paths[name])
    except Exception:
        for path in partial_paths.values():
            if path.exists():
                path.unlink()
        for path in published:
            if path.exists():
                path.unlink()
        raise
    if {path.name for path in directory.iterdir()} != set(final_paths):
        raise RuntimeError("TP1 correctness artifact inventory is invalid")
    return tuple(final_paths[name] for name in (
        TP1_RESULT_NAME,
        TP1_REFERENCE_LOGITS_NAME,
        TP1_NATIVE_LOGITS_NAME,
        TP1_SOURCE_MANIFEST_NAME,
    ))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _query_all_nvidia_smi_gpus(
    *,
    command_runner=subprocess.run,
) -> tuple[dict[str, int | str], ...]:
    if not callable(command_runner):
        raise ValueError("command_runner must be callable")
    completed = command_runner(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if getattr(completed, "returncode", None) != 0:
        raise ValueError("nvidia-smi GPU query failed")
    rows = []
    for line in str(getattr(completed, "stdout", "")).splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise ValueError("nvidia-smi GPU query output is invalid")
        try:
            gpu_index = int(fields[0])
            free_mib = int(fields[2])
        except ValueError as exc:
            raise ValueError(
                "nvidia-smi GPU query output is invalid"
            ) from exc
        if gpu_index < 0 or not fields[1] or free_mib < 0:
            raise ValueError("nvidia-smi GPU query output is invalid")
        rows.append({
            "gpu_index": gpu_index,
            "gpu_uuid": fields[1],
            "free_bytes": free_mib * 1024**2,
        })
    if not rows or len({row["gpu_index"] for row in rows}) != len(rows):
        raise ValueError("nvidia-smi GPU inventory is invalid")
    return tuple(sorted(rows, key=lambda row: int(row["gpu_index"])))


def _select_source_bound_gpu(
    *,
    query_gpus: Callable[[], Iterable[Mapping]],
) -> dict[str, int | str]:
    if not callable(query_gpus):
        raise ValueError("query_gpus must be callable")
    for row in query_gpus():
        if (
            isinstance(row, Mapping)
            and isinstance(row.get("gpu_index"), int)
            and not isinstance(row.get("gpu_index"), bool)
            and isinstance(row.get("gpu_uuid"), str)
            and bool(row.get("gpu_uuid"))
            and isinstance(row.get("free_bytes"), int)
            and not isinstance(row.get("free_bytes"), bool)
            and row["free_bytes"] >= REFERENCE_MINIMUM_FREE_BYTES
        ):
            return {
                "gpu_index": int(row["gpu_index"]),
                "gpu_uuid": str(row["gpu_uuid"]),
                "free_bytes": int(row["free_bytes"]),
                "minimum_free_bytes": REFERENCE_MINIMUM_FREE_BYTES,
            }
    raise ValueError("no GPU has at least 24 GiB free memory")


def _read_source_bound_manifest(
    path: Path,
    *,
    source_root: Path,
) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("source manifest input is invalid") from exc
    source = _validate_source_manifest(value)
    required = {
        "tools/qwen35_tp1_real_root_logit_correctness_contract.py",
        "tools/qwen35_tp1_real_root_logit_correctness_preflight.py",
        "tools/verify_qwen35_tp1_real_root_logit_correctness_gate.py",
    }
    hashes = source["source_file_sha256"]
    if not required.issubset(hashes):
        raise ValueError("source manifest omits required gate source")
    for relative_name, expected_sha256 in hashes.items():
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source manifest path is invalid")
        source_file = source_root / relative
        if not source_file.is_file():
            raise ValueError("source manifest file is missing")
        if _sha256_file(source_file) != expected_sha256:
            raise ValueError("source manifest hash mismatch")
    return source


def _run_isolated_worker(
    command: tuple[str, ...],
    *,
    work_dir: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    command_runner=subprocess.run,
) -> None:
    completed = command_runner(
        command,
        cwd=work_dir,
        env=dict(environment),
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if getattr(completed, "returncode", None) != 0:
        detail = str(getattr(completed, "stderr", "")).strip()[-4000:]
        raise ValueError(
            f"isolated correctness worker failed: {detail}"
        )


def _fresh_port_pair() -> tuple[str, str]:
    ports = []
    while len(ports) < 2:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            port = int(handle.getsockname()[1])
        if port not in ports:
            ports.append(port)
    return tuple(str(port) for port in ports)


def _load_worker_json(path: Path, *, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} worker JSON output is invalid") from exc


def _load_worker_tensor_map(path: Path, *, label: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} worker tensor output is invalid") from exc


def execute_source_bound_run(
    *,
    run_dir,
    run_tag: str,
    source_manifest_path,
    query_gpus: Callable[[], Iterable[Mapping]] = (
        _query_all_nvidia_smi_gpus
    ),
    command_runner=subprocess.run,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> dict:
    directory = Path(run_dir)
    if directory.exists():
        raise ValueError("TP1 correctness run directory already exists")
    source_root = Path(__file__).resolve().parents[1]
    source_manifest = _read_source_bound_manifest(
        Path(source_manifest_path),
        source_root=source_root,
    )
    resource = _select_source_bound_gpu(query_gpus=query_gpus)
    work_dir = directory.parent / f".{run_tag}.work"
    if work_dir.exists():
        raise ValueError("TP1 correctness work directory already exists")
    work_dir.mkdir(parents=True)
    script = Path(__file__).resolve()
    base_environment = dict(os.environ)
    for variable in _NETWORK_PROXY_VARIABLES:
        base_environment.pop(variable, None)
    base_environment.update({
        "PYTHONPATH": os.fspath(source_root),
        "CUDA_VISIBLE_DEVICES": str(resource["gpu_index"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TINYVLLM_GATE_PHYSICAL_GPU_INDEX": str(resource["gpu_index"]),
        "TINYVLLM_GATE_GPU_UUID": str(resource["gpu_uuid"]),
    })
    reference_environment = dict(base_environment)
    native_environment = dict(base_environment)
    reference_ports = _fresh_port_pair()
    native_ports = _fresh_port_pair()
    while native_ports == reference_ports:
        native_ports = _fresh_port_pair()
    reference_environment.update({
        "TINYVLLM_DIST_PORT": reference_ports[0],
        "MASTER_PORT": reference_ports[1],
    })
    native_environment.update({
        "TINYVLLM_DIST_PORT": native_ports[0],
        "MASTER_PORT": native_ports[1],
    })
    reference_tensor = work_dir / REFERENCE_TENSOR_PARTIAL_NAME
    reference_process = work_dir / REFERENCE_PROCESS_PARTIAL_NAME
    native_tensor = work_dir / "native_logits.pt.partial"
    native_process = work_dir / "native_process.json.partial"
    native_state = work_dir / "native_state.json.partial"
    reference_command = (
        sys.executable,
        os.fspath(script),
        "internal-reference",
        "--model-dir",
        APPROVED_MODEL_DIR,
        "--model-manifest-sha256",
        APPROVED_MODEL_MANIFEST_SHA256,
        "--tensor-output",
        os.fspath(reference_tensor),
        "--process-output",
        os.fspath(reference_process),
        "--dtype",
        "bfloat16",
        "--attn-implementation",
        "eager",
        "--local-files-only",
        "--no-trust-remote-code",
        "--no-use-cache",
    )
    native_command = (
        sys.executable,
        os.fspath(script),
        "internal-native",
        "--tensor-output",
        os.fspath(native_tensor),
        "--process-output",
        os.fspath(native_process),
        "--state-output",
        os.fspath(native_state),
        "--dtype",
        "bfloat16",
        "--recurrent-dtype",
        "float32",
        "--tensor-parallel-size",
        "1",
        "--tensor-parallel-rank",
        "0",
    )
    try:
        _run_isolated_worker(
            reference_command,
            work_dir=work_dir,
            environment=reference_environment,
            timeout_seconds=1800,
            command_runner=command_runner,
        )
        reference_row = _load_worker_json(
            reference_process,
            label="reference",
        )
        reference_pid = reference_row.get("pid")
        if (
            isinstance(reference_pid, bool)
            or not isinstance(reference_pid, int)
            or reference_pid <= 0
            or pid_alive(reference_pid)
        ):
            raise ValueError(
                "reference worker PID did not disappear before native startup"
            )
        current = require_reference_gpu_resource(
            gpu_index=int(resource["gpu_index"]),
            query_gpu=lambda _index: next(
                row
                for row in query_gpus()
                if row.get("gpu_index") == resource["gpu_index"]
            ),
        )
        if current["gpu_uuid"] != resource["gpu_uuid"]:
            raise ValueError("selected GPU identity changed before native")
        _run_isolated_worker(
            native_command,
            work_dir=work_dir,
            environment=native_environment,
            timeout_seconds=1800,
            command_runner=command_runner,
        )
        native_row = _load_worker_json(native_process, label="native")
        native_pid = native_row.get("pid")
        if (
            isinstance(native_pid, bool)
            or not isinstance(native_pid, int)
            or native_pid <= 0
            or pid_alive(native_pid)
        ):
            raise ValueError("native worker PID survived completion")
        paths = finalize_tp1_correctness_artifact(
            run_dir=directory,
            run_tag=run_tag,
            reference_logits=_load_worker_tensor_map(
                reference_tensor,
                label="reference",
            ),
            native_logits=_load_worker_tensor_map(
                native_tensor,
                label="native",
            ),
            reference_process=reference_row,
            native_process=native_row,
            state_rows=_load_worker_json(native_state, label="native state"),
            source_manifest=source_manifest,
            forbidden_counters={
                "engine": 0,
                "model_runner": 0,
                "scheduler": 0,
                "sampler": 0,
                "generation": 0,
            },
        )
        result = json.loads(
            (directory / TP1_RESULT_NAME).read_text(encoding="utf-8")
        )
        return {
            "classification": result["classification"],
            "paths": [os.fspath(path) for path in paths],
            "gpu": resource,
        }
    finally:
        if directory.is_dir() and {
            path.name for path in directory.iterdir()
        } == {
            TP1_RESULT_NAME,
            TP1_REFERENCE_LOGITS_NAME,
            TP1_NATIVE_LOGITS_NAME,
            TP1_SOURCE_MANIFEST_NAME,
        }:
            shutil.rmtree(work_dir, ignore_errors=False)


def validate_tp1_correctness_artifact(run_dir) -> dict:
    verifier_path = (
        Path(__file__).resolve().parent
        / "verify_qwen35_tp1_real_root_logit_correctness_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp1_independent_verifier_for_cli",
        os.fspath(verifier_path),
    )
    if spec is None or spec.loader is None:
        raise ValueError("independent verifier is invalid")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    return verifier.verify_run(run_dir)


def _required_true(value: str) -> bool:
    if value != "true":
        raise argparse.ArgumentTypeError("value must be true")
    return True


def _required_false(value: str) -> bool:
    if value != "false":
        raise argparse.ArgumentTypeError("value must be false")
    return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--run-tag", required=True)
    run.add_argument("--source-manifest", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("run_dir")
    reference = subparsers.add_parser("internal-reference")
    reference.add_argument("--model-dir", required=True)
    reference.add_argument("--model-manifest-sha256", required=True)
    reference.add_argument("--tensor-output", required=True)
    reference.add_argument("--process-output", required=True)
    reference.add_argument(
        "--dtype",
        choices=("bfloat16",),
        required=True,
    )
    reference.add_argument(
        "--attn-implementation",
        choices=("eager",),
        required=True,
    )
    reference.add_argument(
        "--local-files-only",
        action="store_true",
        required=True,
    )
    reference.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        required=True,
    )
    reference.add_argument(
        "--no-use-cache",
        dest="use_cache",
        action="store_false",
        required=True,
    )
    native = subparsers.add_parser("internal-native")
    native.add_argument("--tensor-output", required=True)
    native.add_argument("--process-output", required=True)
    native.add_argument("--state-output", required=True)
    native.add_argument(
        "--dtype",
        choices=("bfloat16",),
        required=True,
    )
    native.add_argument(
        "--recurrent-dtype",
        choices=("float32",),
        required=True,
    )
    native.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1,),
        required=True,
    )
    native.add_argument(
        "--tensor-parallel-rank",
        type=int,
        choices=(0,),
        required=True,
    )
    return parser


def main(
    argv=None,
    *,
    execute_reference=execute_reference_worker,
    execute_native=execute_native_worker,
    execute_run=execute_source_bound_run,
    execute_validate=validate_tp1_correctness_artifact,
    prompt_case_loader=_load_frozen_prompt_cases,
    environment=None,
) -> int:
    arguments = _build_parser().parse_args(argv)
    if (
        not callable(execute_reference)
        or not callable(execute_native)
        or not callable(execute_run)
        or not callable(execute_validate)
        or not callable(prompt_case_loader)
    ):
        raise ValueError("correctness CLI dependencies must be callable")
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise ValueError("environment must be a mapping")
    if arguments.mode == "run":
        execute_run(
            run_dir=Path(arguments.run_dir),
            run_tag=arguments.run_tag,
            source_manifest_path=Path(arguments.source_manifest),
        )
        return 0
    if arguments.mode == "validate":
        execute_validate(Path(arguments.run_dir))
        return 0
    try:
        gpu_index = int(environment["TINYVLLM_GATE_PHYSICAL_GPU_INDEX"])
        gpu_uuid = environment["TINYVLLM_GATE_GPU_UUID"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("reference worker GPU identity is missing") from exc
    if arguments.mode == "internal-reference":
        if (
            arguments.dtype != "bfloat16"
            or arguments.attn_implementation != "eager"
            or arguments.local_files_only is not True
            or arguments.trust_remote_code is not False
            or arguments.use_cache is not False
        ):
            raise ValueError("reference worker CLI contract was relaxed")
        execute_reference(
            model_dir=Path(arguments.model_dir),
            model_manifest_sha256=arguments.model_manifest_sha256,
            tensor_output=Path(arguments.tensor_output),
            process_output=Path(arguments.process_output),
            prompt_cases=prompt_case_loader(),
            expected_vocab_size=248320,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
        )
    elif arguments.mode == "internal-native":
        if (
            arguments.dtype != "bfloat16"
            or arguments.recurrent_dtype != "float32"
            or arguments.tensor_parallel_size != 1
            or arguments.tensor_parallel_rank != 0
        ):
            raise ValueError("native worker CLI contract was relaxed")
        execute_native(
            tensor_output=Path(arguments.tensor_output),
            process_output=Path(arguments.process_output),
            state_output=Path(arguments.state_output),
            prompt_cases=prompt_case_loader(),
            expected_vocab_size=248320,
            gpu_index=gpu_index,
            gpu_uuid=gpu_uuid,
        )
    else:
        raise ValueError("unsupported correctness preflight mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
