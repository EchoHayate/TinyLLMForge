from __future__ import annotations

import argparse
import ast
import builtins
from collections.abc import Mapping
import gc
import getpass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import types


SCHEMA_VERSION = (
    "qwen35.real-checkpoint-target-preparation-preflight.v1"
)
ROW_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-target-preparation-rank.v1"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
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
APPROVED_SHARD_NAME = (
    "model.safetensors-00001-of-00001.safetensors"
)
APPROVED_SHARD_SIZE = 4548221488
APPROVED_SHARD_SHA256 = (
    "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
)
APPROVED_COMPOSITE_SHA256 = (
    "27da983f5ef3e38480d8b5d5976e5c434fc4b5d0c70d09511c35154beecd8db9"
)
MAX_TOTAL_VMHWM_INCREMENT_KIB = 524288
MAX_POST_TORCH_VMHWM_INCREMENT_KIB = 196608
MAX_POST_METADATA_VMHWM_INCREMENT_KIB = 32768
PRODUCTION_SOURCE_FILES = (
    "tinyvllm/engine/hybrid_state.py",
    "tinyvllm/engine/qwen35_hybrid_model_owner.py",
    "tinyvllm/engine/qwen35_hybrid_prefix_runtime_identity.py",
    "tinyvllm/engine/qwen35_hybrid_state.py",
    "tinyvllm/engine/qwen35_layer_state.py",
    "tinyvllm/engine/qwen35_state_transaction.py",
    "tinyvllm/layers/embed_head.py",
    "tinyvllm/layers/gated_delta.py",
    "tinyvllm/layers/linear.py",
    "tinyvllm/layers/quantization.py",
    "tinyvllm/layers/qwen35_decoder_layer.py",
    "tinyvllm/layers/qwen35_full_attention.py",
    "tinyvllm/layers/qwen35_linear_attention.py",
    "tinyvllm/layers/qwen35_packed_layer_stack.py",
    "tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py",
    "tinyvllm/layers/qwen35_primitives.py",
    "tinyvllm/layers/qwen35_rotary_embedding.py",
    "tinyvllm/models/qwen35_checkpoint.py",
    "tinyvllm/models/qwen35_checkpoint_assignment.py",
    "tinyvllm/models/qwen35_checkpoint_binding.py",
    "tinyvllm/models/qwen35_checkpoint_candidate_factory.py",
    "tinyvllm/models/qwen35_checkpoint_candidate_loader.py",
    "tinyvllm/models/qwen35_checkpoint_loader_configuration.py",
    "tinyvllm/models/qwen35_checkpoint_metadata.py",
    "tinyvllm/models/qwen35_checkpoint_streaming.py",
    "tinyvllm/models/qwen35_checkpoint_worker.py",
    "tinyvllm/models/qwen35_components.py",
    "tinyvllm/models/qwen35_factory.py",
    "tinyvllm/models/qwen35_packed.py",
    "tinyvllm/speculative/verifier.py",
    "tinyvllm/utils/context.py",
    "tools/qwen35_real_checkpoint_load_worker.py",
)
SOURCE_FILES = (
    *PRODUCTION_SOURCE_FILES,
    "tools/qwen35_real_checkpoint_target_preparation_preflight.py",
)
PACKAGE_NAMES = (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.layers",
    "tinyvllm.models",
    "tinyvllm.speculative",
    "tinyvllm.utils",
)
TP_ROWS = ((1, 0), (2, 0), (2, 1))
EXPECTED_POOL_BYTES = {1: 10321920, 2: 5160960}
SSH_CONTROL_PATH = "/tmp/ssh-sitian-10.232.195.203"
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-target-preparation-runs"
)
LOCAL_RUN_ROOT = Path("experiments/qwen35_hybrid_state")
RUN_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _sha256(value, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _non_negative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(value, name: str) -> int:
    value = _non_negative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(
                f"missing target preparation source: {relative}"
            )
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _source_tree_sha256(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(hashes),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_run_tag(value) -> str:
    text = str(value)
    if not RUN_TAG_PATTERN.fullmatch(text):
        raise ValueError("run tag must match [A-Za-z0-9_-]+")
    return text


def build_ssh_command(remote_arguments) -> list[str]:
    return [
        "ssh",
        "-S",
        SSH_CONTROL_PATH,
        "-o",
        "BatchMode=yes",
        REMOTE_TARGET,
        shlex.join([str(value) for value in remote_arguments]),
    ]


def discover_production_source_files(source_root) -> tuple[str, ...]:
    root = Path(source_root)
    queue = [
        "tools/qwen35_real_checkpoint_load_worker.py",
        "tinyvllm/models/qwen35_checkpoint_metadata.py",
    ]
    observed = set()
    while queue:
        relative = queue.pop(0)
        if relative in observed:
            continue
        observed.add(relative)
        tree = ast.parse(
            (root / relative).read_text(encoding="utf-8"),
            filename=relative,
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                modules = []
            for module in modules:
                if not module.startswith("tinyvllm."):
                    continue
                candidate = module.replace(".", "/") + ".py"
                if (root / candidate).is_file():
                    queue.append(candidate)
    return tuple(sorted(observed))


def build_source_tar(source_root) -> bytes:
    root = Path(source_root)
    if discover_production_source_files(root) != PRODUCTION_SOURCE_FILES:
        raise ValueError("production source closure does not match frozen set")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(
                    f"missing target preparation source: {relative}"
                )
            info = archive.gettarinfo(os.fspath(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def _require_success(result, context: str):
    if result.returncode != 0:
        detail = result.stderr or result.stdout or ""
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise RuntimeError(f"{context} failed: {str(detail).strip()}")
    return result


def stage_source(
    source_root,
    run_tag,
    *,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    payload = build_source_tar(source_root)
    staged = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            (
                f"test ! -e {shlex.quote(remote_run_dir)} && "
                f"mkdir -p {shlex.quote(remote_source_dir)} && "
                f"tar -xf - -C {shlex.quote(remote_source_dir)}"
            ),
        ]),
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(staged, "target preparation source staging")
    local_hashes = _source_hashes(source_root)
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"names={list(SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    verified = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            "-c",
            script,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "target preparation source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError(
            "target preparation remote source hashes do not match local"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _read_proc_status() -> dict[str, int]:
    result = {}
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if key in ("VmRSS", "VmHWM") and parts:
            result[key] = int(parts[0])
    return result


def _memory_point(status: Mapping[str, int]) -> dict[str, int]:
    return {
        "vmrss_kib": _positive_integer(status.get("VmRSS"), "VmRSS"),
        "vmhwm_kib": _positive_integer(status.get("VmHWM"), "VmHWM"),
    }


def _install_namespace_packages(source_root) -> None:
    root = Path(source_root)
    for name in PACKAGE_NAMES:
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = package


def _expected_backend_calls(tp_size: int) -> list[list[int]]:
    return [
        [
            layer_index,
            8 // tp_size,
            2 // tp_size,
            256,
        ]
        for layer_index in (3, 7, 11, 15, 19, 23)
    ]


def _validate_memory(row) -> None:
    memory = row.get("memory")
    expected_names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_release",
    )
    if (
        not isinstance(memory, Mapping)
        or set(memory) != set(expected_names)
    ):
        raise ValueError("target preparation memory points are invalid")
    for name in expected_names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(f"target preparation memory {name} is invalid")
        _positive_integer(point.get("vmrss_kib"), f"{name} vmrss")
        _positive_integer(point.get("vmhwm_kib"), f"{name} vmhwm")
    total = _non_negative_integer(
        row.get("total_vmhwm_increment_kib"),
        "total_vmhwm_increment_kib",
    )
    post_torch = _non_negative_integer(
        row.get("post_torch_vmhwm_increment_kib"),
        "post_torch_vmhwm_increment_kib",
    )
    post_metadata = _non_negative_integer(
        row.get("post_metadata_vmhwm_increment_kib"),
        "post_metadata_vmhwm_increment_kib",
    )
    expected_total = max(
        0,
        memory["after_release"]["vmhwm_kib"]
        - memory["before"]["vmhwm_kib"],
    )
    expected_post_torch = max(
        0,
        memory["after_release"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
    )
    expected_post_metadata = max(
        0,
        memory["after_release"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    if (
        total != expected_total
        or post_torch != expected_post_torch
        or post_metadata != expected_post_metadata
    ):
        raise ValueError("target preparation VmHWM deltas are invalid")
    if total > MAX_TOTAL_VMHWM_INCREMENT_KIB:
        raise ValueError("target preparation total VmHWM exceeds ceiling")
    if post_torch > MAX_POST_TORCH_VMHWM_INCREMENT_KIB:
        raise ValueError(
            "target preparation post-Torch VmHWM exceeds ceiling"
        )
    if post_metadata > MAX_POST_METADATA_VMHWM_INCREMENT_KIB:
        raise ValueError(
            "target preparation post-metadata VmHWM exceeds ceiling"
        )


def validate_target_preparation_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("target preparation row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("target preparation row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("target preparation row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in TP_ROWS:
        raise ValueError("target preparation row TP context is invalid")
    _positive_integer(row.get("process_id"), "process_id")
    exact = {
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "payload_bytes_read": 0,
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "pool_capacity": 1,
        "pool_device": "cpu",
        "pool_component_count": 36,
        "pool_binding_count": 0,
        "pool_logical_bytes": EXPECTED_POOL_BYTES[tp[0]],
        "pool_physical_bytes": EXPECTED_POOL_BYTES[tp[0]],
        "pool_nonzero_count": 0,
        "pool_unchanged": True,
        "layer_count": 24,
        "linear_adapter_count": 18,
        "backend_calls": _expected_backend_calls(tp[0]),
        "binding_count": 320,
        "shared_binding_count": 2,
        "linear_binding_count": 252,
        "full_binding_count": 66,
        "buffer_binding_count": 72,
        "float32_binding_count": 36,
        "all_binding_destinations_meta": True,
        "registered_parameter_count": 225,
        "registered_buffer_count": 78,
        "unexpected_non_meta_registrations": [],
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "pool_create_count": 1,
        "backend_create_count": 6,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            if name == "payload_bytes_read":
                raise ValueError("target preparation payload must be zero")
            if name == "loader_call_count":
                raise ValueError("target preparation loader calls must be zero")
            if name == "assignment_call_count":
                raise ValueError(
                    "target preparation assignment calls must be zero"
                )
            if name in (
                "model_forward_count",
                "attention_forward_count",
            ):
                raise ValueError(
                    "target preparation forward calls must be zero"
                )
            if name.startswith("cuda_initialized"):
                raise ValueError("target preparation CUDA must remain off")
            if name == "unexpected_non_meta_registrations":
                raise ValueError(
                    "target preparation has unexpected non-meta registrations"
                )
            if name in ("pool_logical_bytes", "pool_physical_bytes"):
                raise ValueError("target preparation pool bytes are invalid")
            raise ValueError(
                f"target preparation row {name} is invalid"
            )
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("target preparation hostname is invalid")
    _validate_memory(row)
    return row


def validate_target_preparation_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("target preparation preflight must be a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("target preparation preflight schema is invalid")
    if record.get("status") != "PASS":
        raise ValueError("target preparation preflight status must be PASS")
    exact = {
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            if name == "fresh_process_per_rank":
                raise ValueError(
                    "target preparation requires a fresh process per rank"
                )
            raise ValueError(
                f"target preparation preflight {name} is invalid"
            )
    source_hashes = record.get("source_file_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(SOURCE_FILES)
    ):
        raise ValueError("target preparation source hashes are invalid")
    for name, digest in source_hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(source_hashes)
    ):
        raise ValueError("target preparation source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(TP_ROWS):
        raise ValueError("target preparation TP rows are invalid")
    for row in rows:
        validate_target_preparation_row(row)
    process_ids = [row["process_id"] for row in rows]
    if len(set(process_ids)) != len(process_ids):
        raise ValueError(
            "target preparation process IDs must be unique"
        )
    return record


def _snapshot_pool(pool):
    return {
        "bindings": tuple(pool._bindings.items()),
        "tensors": {
            key: (
                id(tensor),
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tensor.dtype,
                tensor.device,
                tensor._version,
            )
            for key, tensor in pool._tensors.items()
        },
    }


def _pool_unchanged(pool, snapshot) -> bool:
    if tuple(pool._bindings.items()) != snapshot["bindings"]:
        return False
    if set(pool._tensors) != set(snapshot["tensors"]):
        return False
    for key, tensor in pool._tensors.items():
        if (
            id(tensor),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor._version,
        ) != snapshot["tensors"][key]:
            return False
    return True


def run_target_preparation_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    tensor_parallel_size,
    tensor_parallel_rank,
    observed_user,
    observed_hostname,
    process_id,
    status_reader=_read_proc_status,
):
    before = _memory_point(status_reader())
    _install_namespace_packages(source_root)
    import torch
    from torch import nn

    cuda_before = torch.cuda.is_initialized()
    after_torch = _memory_point(status_reader())
    metadata_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_metadata",
        fromlist=["*"],
    )
    checkpoint_module = __import__(
        "tinyvllm.models.qwen35_checkpoint",
        fromlist=["*"],
    )
    hybrid_state_module = __import__(
        "tinyvllm.engine.hybrid_state",
        fromlist=["*"],
    )
    layout_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_state",
        fromlist=["*"],
    )
    factory_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_candidate_factory",
        fromlist=["*"],
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    metadata = metadata_module.read_qwen35_checkpoint_metadata(
        checkpoint_dir,
        shards=(shard,),
        expected_config_sha256=APPROVED_CONFIG_SHA256,
        expected_index_sha256=APPROVED_INDEX_SHA256,
        expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
    )
    tensor_plan = checkpoint_module.build_qwen35_checkpoint_tensor_plan(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
    )
    after_metadata = _memory_point(status_reader())
    layout = layout_module.build_qwen35_hybrid_state_layout(
        metadata.hf_config,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        speculative_tokens=1,
    )
    pool = hybrid_state_module.HybridStateTensorPool(
        layout,
        capacity=1,
        device="cpu",
    )
    pool_snapshot = _snapshot_pool(pool)
    after_pool = _memory_point(status_reader())
    backend_calls = []
    attention_forward_count = 0

    class _StaticAttentionBackend(nn.Module):
        def __init__(self, arguments):
            super().__init__()
            self.arguments = arguments

        def forward(self, *_args, **_kwargs):
            nonlocal attention_forward_count
            attention_forward_count += 1
            raise AssertionError("attention backend must not execute")

    def build_backend(layer_index, query_heads, kv_heads, head_dim):
        arguments = [
            layer_index,
            query_heads,
            kv_heads,
            head_dim,
        ]
        backend_calls.append(arguments)
        return _StaticAttentionBackend(arguments)

    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if str(file).endswith(".safetensors"):
            raise AssertionError(
                "safetensors payload open after metadata is forbidden"
            )
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        target = factory_module.prepare_qwen35_checkpoint_candidate_target(
            metadata.hf_config,
            tensor_plan,
            pool=pool,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            build_attention_backend=build_backend,
            parameter_device="meta",
        )
    finally:
        builtins.open = original_open
    after_target = _memory_point(status_reader())
    assembly = target.assembly
    model = assembly.packed.model
    registrations = (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    )
    unexpected_non_meta = [
        name
        for name, tensor in registrations
        if tensor.device.type != "meta"
    ]
    bindings = target.binding_plan.bindings
    layer_types = tuple(
        getattr(metadata.hf_config, "text_config").layer_types
    )
    pool_nonzero_count = sum(
        int(torch.count_nonzero(tensor).item())
        for tensor in pool._tensors.values()
    )
    row_values = {
        "pool_capacity": pool.capacity,
        "pool_device": str(pool.device),
        "pool_component_count": len(pool.layout.components),
        "pool_binding_count": len(pool._bindings),
        "pool_logical_bytes": pool.logical_bytes,
        "pool_physical_bytes": pool.physical_storage_bytes,
        "pool_nonzero_count": pool_nonzero_count,
        "pool_unchanged": _pool_unchanged(pool, pool_snapshot),
        "layer_count": len(model.layer_stack.layers),
        "linear_adapter_count": len(assembly.packed.adapters),
        "backend_calls": backend_calls,
        "binding_count": len(bindings),
        "shared_binding_count": sum(
            binding.load.weight.target
            in ("embed_tokens.weight", "final_norm.weight")
            for binding in bindings
        ),
        "linear_binding_count": sum(
            binding.load.weight.target.startswith("layers.")
            and layer_types[
                int(binding.load.weight.target.split(".")[1])
            ] == "linear_attention"
            for binding in bindings
        ),
        "full_binding_count": sum(
            binding.load.weight.target.startswith("layers.")
            and layer_types[
                int(binding.load.weight.target.split(".")[1])
            ] == "full_attention"
            for binding in bindings
        ),
        "buffer_binding_count": sum(
            binding.destination_kind == "buffer"
            for binding in bindings
        ),
        "float32_binding_count": sum(
            binding.destination.dtype == torch.float32
            for binding in bindings
        ),
        "all_binding_destinations_meta": all(
            binding.destination.device.type == "meta"
            for binding in bindings
        ),
        "registered_parameter_count": sum(
            1 for _ in model.named_parameters(remove_duplicate=False)
        ),
        "registered_buffer_count": sum(
            1 for _ in model.named_buffers(remove_duplicate=False)
        ),
        "unexpected_non_meta_registrations": unexpected_non_meta,
    }
    del registrations
    del bindings
    del model
    del assembly
    del target
    del pool
    del layout
    gc.collect()
    after_release = _memory_point(status_reader())
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": after_metadata,
        "after_pool": after_pool,
        "after_target": after_target,
        "after_release": after_release,
    }
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "tp_size": tensor_parallel_size,
        "tp_rank": tensor_parallel_rank,
        "process_id": process_id,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "config_sha256": metadata.config_sha256,
        "index_sha256": metadata.index_sha256,
        "config_index_header_sha256": (
            metadata.config_index_header_sha256
        ),
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "payload_bytes_read": metadata.payload_bytes_read,
        "payload_hashes_recomputed": False,
        "plan_loads": len(tensor_plan.loads),
        "plan_skips": len(tensor_plan.skips),
        "plan_payload_bytes": tensor_plan.payload_bytes,
        **row_values,
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": attention_forward_count,
        "pool_create_count": 1,
        "backend_create_count": len(backend_calls),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0,
            after_release["vmhwm_kib"] - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_release["vmhwm_kib"] - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_release["vmhwm_kib"]
            - after_metadata["vmhwm_kib"],
        ),
    }
    if str(Path(checkpoint_dir).resolve()) == APPROVED_MODEL_DIR:
        validate_target_preparation_row(row)
    return row


def _atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def _aggregate(rows, source_root):
    source_hashes = _source_hashes(source_root)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
        "rows": list(rows),
    }
    validate_target_preparation_preflight(record)
    return record


def run_remote_target_preparation_preflight(
    source_root,
    run_tag,
    *,
    staged,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    destination = Path(local_run_root) / run_tag
    if destination.exists():
        raise ValueError(
            f"local target preparation directory already exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/target_preparation_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_target_preparation_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in TP_ROWS:
        completed = command_runner(
            build_ssh_command([
                "env",
                "CUDA_VISIBLE_DEVICES=",
                "PYTHONDONTWRITEBYTECODE=1",
                REMOTE_PYTHON,
                "-B",
                worker,
                "internal-rank-worker",
                "--source-root",
                staged["remote_source_dir"],
                "--checkpoint-dir",
                APPROVED_MODEL_DIR,
                "--tp-size",
                str(tp_size),
                "--tp-rank",
                str(tp_rank),
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, "target preparation rank worker")
        row = json.loads(completed.stdout)
        validate_target_preparation_row(row)
        rows.append(row)
    finalized = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            worker,
            "internal-finalize",
            "--source-root",
            staged["remote_source_dir"],
            "--output",
            remote_artifact,
        ]),
        input=json.dumps({"rows": rows}),
        text=True,
        capture_output=True,
    )
    _require_success(finalized, "target preparation finalizer")
    record = json.loads(finalized.stdout)
    validate_target_preparation_preflight(record)
    if (
        record["source_file_sha256"]
        != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError("target preparation source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    round_trip_script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'target_preparation_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'target_preparation_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            "-c",
            round_trip_script,
        ]),
        input=json.dumps({
            "target_preparation_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "target preparation artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "target_preparation_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("target preparation artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "target_preparation_preflight.json",
            record,
        )
        _atomic_write_json(
            temporary / "source_manifest.json",
            source_manifest,
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return record


def execute_remote_target_preparation_preflight(
    source_root,
    run_tag,
    *,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source(
        source_root,
        run_tag,
        command_runner=command_runner,
    )
    return run_remote_target_preparation_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_target_preparation_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        tensor_parallel_size=arguments.tp_size,
        tensor_parallel_rank=arguments.tp_rank,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
        process_id=os.getpid(),
    )
    validate_target_preparation_row(row)
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _finalize_main(arguments) -> int:
    output = Path(arguments.output)
    if output.exists():
        raise ValueError(
            "target preparation preflight output already exists"
        )
    payload = json.load(sys.stdin)
    rows = payload.get("rows")
    record = _aggregate(rows, arguments.source_root)
    _atomic_write_json(output, record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-tag", required=True)
    run_parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--tp-size", required=True, type=int)
    worker_parser.add_argument("--tp-rank", required=True, type=int)
    finalize_parser = subparsers.add_parser("internal-finalize")
    finalize_parser.add_argument("--source-root", required=True)
    finalize_parser.add_argument("--output", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    arguments = parser.parse_args(argv)
    if arguments.mode == "internal-rank-worker":
        return _rank_worker_main(arguments)
    if arguments.mode == "internal-finalize":
        return _finalize_main(arguments)
    if arguments.mode == "validate":
        record = json.loads(Path(arguments.artifact).read_text())
        validate_target_preparation_preflight(record)
    else:
        record = execute_remote_target_preparation_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
