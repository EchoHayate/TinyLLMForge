from __future__ import annotations

import argparse
import builtins
from collections.abc import Mapping
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile


_BASE_PATH = (
    Path(__file__).resolve().parent
    / "qwen35_real_checkpoint_target_preparation_preflight.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location(
    "_qwen35_target_preparation_preflight_base",
    _BASE_PATH,
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError("unable to load target-preparation preflight base")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)


SCHEMA_VERSION = "qwen35.real-checkpoint-cpu-materialization-preflight.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-cpu-materialization-rank.v1"
REMOTE_TARGET = base.REMOTE_TARGET
REMOTE_PYTHON = base.REMOTE_PYTHON
APPROVED_MODEL_DIR = base.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = base.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = base.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = base.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = base.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = base.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = base.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = base.APPROVED_COMPOSITE_SHA256
PRODUCTION_SOURCE_FILES = base.PRODUCTION_SOURCE_FILES
TARGET_GATE_SOURCE = (
    "tools/qwen35_real_checkpoint_target_preparation_preflight.py"
)
SOURCE_FILES = (
    *PRODUCTION_SOURCE_FILES,
    TARGET_GATE_SOURCE,
    "tools/qwen35_real_checkpoint_cpu_materialization_preflight.py",
)
TP_ROWS = base.TP_ROWS
EXPECTED_POOL_BYTES = base.EXPECTED_POOL_BYTES
EXPECTED_REGISTERED_BYTES = {
    1: 3763656128,
    2: 1881936480,
}
EXPECTED_BINDING_BYTES = {
    1: 3763655360,
    2: 1881935712,
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 4718592,
        "post_torch": 4194304,
        "post_metadata": 3932160,
    },
    2: {
        "total": 2621440,
        "post_torch": 2359296,
        "post_metadata": 2097152,
    },
}
SSH_CONTROL_PATH = base.SSH_CONTROL_PATH
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-cpu-materialization-runs"
)
LOCAL_RUN_ROOT = base.LOCAL_RUN_ROOT


_sha256 = base._sha256
_positive_integer = base._positive_integer
_non_negative_integer = base._non_negative_integer
_source_tree_sha256 = base._source_tree_sha256
validate_run_tag = base.validate_run_tag
build_ssh_command = base.build_ssh_command
_require_success = base._require_success
_read_proc_status = base._read_proc_status
_memory_point = base._memory_point
_install_namespace_packages = base._install_namespace_packages
_snapshot_pool = base._snapshot_pool
_pool_unchanged = base._pool_unchanged
_atomic_write_json = base._atomic_write_json


def _source_hashes(source_root) -> dict[str, str]:
    import hashlib

    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(
                f"missing CPU materialization source: {relative}"
            )
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_source_tar(source_root) -> bytes:
    root = Path(source_root)
    if (
        base.discover_production_source_files(root)
        != PRODUCTION_SOURCE_FILES
    ):
        raise ValueError("production source closure does not match frozen set")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(
                    f"missing CPU materialization source: {relative}"
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
    _require_success(staged, "CPU materialization source staging")
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
    _require_success(verified, "CPU materialization source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError(
            "CPU materialization remote source hashes do not match local"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _expected_rotary_buffers() -> list[dict]:
    return [
        {
            "name": (
                f"layer_stack.layers.{layer_index}."
                "full_attention.rotary.inv_freq"
            ),
            "shape": [32],
            "dtype": "torch.float32",
            "bytes": 128,
        }
        for layer_index in (3, 7, 11, 15, 19, 23)
    ]


def _validate_memory(row) -> None:
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_touch",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError("CPU materialization memory points are invalid")
    for name in names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(
                f"CPU materialization memory {name} is invalid"
            )
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
    expected = (
        max(
            0,
            memory["after_touch"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"],
        ),
        max(
            0,
            memory["after_touch"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"],
        ),
        max(
            0,
            memory["after_touch"]["vmhwm_kib"]
            - memory["after_metadata"]["vmhwm_kib"],
        ),
    )
    if (total, post_torch, post_metadata) != expected:
        raise ValueError("CPU materialization VmHWM deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    if total > ceilings["total"]:
        raise ValueError("CPU materialization total VmHWM exceeds ceiling")
    if post_torch > ceilings["post_torch"]:
        raise ValueError(
            "CPU materialization post-Torch VmHWM exceeds ceiling"
        )
    if post_metadata > ceilings["post_metadata"]:
        raise ValueError(
            "CPU materialization post-metadata VmHWM exceeds ceiling"
        )


def validate_cpu_materialization_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("CPU materialization row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("CPU materialization row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("CPU materialization row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in TP_ROWS:
        raise ValueError("CPU materialization TP context is invalid")
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
        "backend_calls": base._expected_backend_calls(tp[0]),
        "binding_count": 320,
        "shared_binding_count": 2,
        "linear_binding_count": 252,
        "full_binding_count": 66,
        "buffer_binding_count": 72,
        "float32_binding_count": 36,
        "registered_entry_count": 303,
        "unique_registered_tensor_count": 302,
        "unique_registered_bytes": EXPECTED_REGISTERED_BYTES[tp[0]],
        "unique_binding_tensor_count": 296,
        "unique_binding_bytes": EXPECTED_BINDING_BYTES[tp[0]],
        "unbound_registered": _expected_rotary_buffers(),
        "tied_embedding_same_object": True,
        "all_registrations_cpu": True,
        "all_bindings_cpu": True,
        "all_binding_destinations_registered": True,
        "all_unique_tensors_zero_after_touch": True,
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
            if name == "unique_registered_bytes":
                raise ValueError(
                    "CPU materialization registered bytes are invalid"
                )
            if name == "unbound_registered":
                raise ValueError(
                    "CPU materialization rotary buffers are invalid"
                )
            if name == "all_unique_tensors_zero_after_touch":
                raise ValueError(
                    "CPU materialization zero touch proof is invalid"
                )
            if name == "payload_bytes_read":
                raise ValueError(
                    "CPU materialization payload must be zero"
                )
            if name == "loader_call_count":
                raise ValueError(
                    "CPU materialization loader calls must be zero"
                )
            if name.startswith("cuda_initialized"):
                raise ValueError(
                    "CPU materialization CUDA must remain off"
                )
            raise ValueError(
                f"CPU materialization row {name} is invalid"
            )
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("CPU materialization hostname is invalid")
    _validate_memory(row)
    return row


def validate_cpu_materialization_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("CPU materialization preflight must be a mapping")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("CPU materialization preflight schema is invalid")
    if record.get("status") != "PASS":
        raise ValueError("CPU materialization preflight status must be PASS")
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
        "process_exit_is_release_boundary": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(
                f"CPU materialization preflight {name} is invalid"
            )
    source_hashes = record.get("source_file_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(SOURCE_FILES)
    ):
        raise ValueError("CPU materialization source hashes are invalid")
    for name, digest in source_hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(source_hashes)
    ):
        raise ValueError("CPU materialization source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(TP_ROWS):
        raise ValueError("CPU materialization TP rows are invalid")
    for row in rows:
        validate_cpu_materialization_row(row)
    process_ids = [row["process_id"] for row in rows]
    if len(set(process_ids)) != len(process_ids):
        raise ValueError("CPU materialization process IDs must be unique")
    return record


def inspect_and_touch_cpu_target(target) -> dict:
    import torch

    model = target.assembly.packed.model
    registrations = (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    )
    registered = dict(registrations)
    unique = {}
    for name, tensor in registrations:
        unique.setdefault(id(tensor), (name, tensor))
    bindings = target.binding_plan.bindings
    binding_unique = {}
    for binding in bindings:
        binding_unique.setdefault(
            id(binding.destination),
            (binding.destination_name, binding.destination),
        )
    unbound = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        for object_id, (name, tensor) in unique.items()
        if object_id not in binding_unique
    ]
    with torch.no_grad():
        for _, tensor in unique.values():
            tensor.zero_()
    return {
        "registered_entry_count": len(registrations),
        "unique_registered_tensor_count": len(unique),
        "unique_registered_bytes": sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in unique.values()
        ),
        "unique_binding_tensor_count": len(binding_unique),
        "unique_binding_bytes": sum(
            tensor.numel() * tensor.element_size()
            for _, tensor in binding_unique.values()
        ),
        "unbound_registered": unbound,
        "tied_embedding_same_object": (
            model.embed_tokens.weight is model.lm_head.weight
        ),
        "all_registrations_cpu": all(
            tensor.device.type == "cpu"
            for _, tensor in registrations
        ),
        "all_bindings_cpu": all(
            binding.destination.device.type == "cpu"
            for binding in bindings
        ),
        "all_binding_destinations_registered": all(
            registered.get(binding.destination_name)
            is binding.destination
            for binding in bindings
        ),
        "all_unique_tensors_zero_after_touch": all(
            int(torch.count_nonzero(tensor).item()) == 0
            for _, tensor in unique.values()
        ),
    }


def run_cpu_materialization_rank_worker(
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

    torch.set_num_threads(8)
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
        def forward(self, *_args, **_kwargs):
            nonlocal attention_forward_count
            attention_forward_count += 1
            raise AssertionError("attention backend must not execute")

    def build_backend(layer_index, query_heads, kv_heads, head_dim):
        backend_calls.append([
            layer_index,
            query_heads,
            kv_heads,
            head_dim,
        ])
        return _StaticAttentionBackend()

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
            parameter_device="cpu",
        )
    finally:
        builtins.open = original_open
    after_target = _memory_point(status_reader())
    inspection = inspect_and_touch_cpu_target(target)
    after_touch = _memory_point(status_reader())
    model = target.assembly.packed.model
    bindings = target.binding_plan.bindings
    layer_types = tuple(metadata.hf_config.text_config.layer_types)
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": after_metadata,
        "after_pool": after_pool,
        "after_target": after_target,
        "after_touch": after_touch,
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
        "pool_capacity": pool.capacity,
        "pool_device": str(pool.device),
        "pool_component_count": len(pool.layout.components),
        "pool_binding_count": len(pool._bindings),
        "pool_logical_bytes": pool.logical_bytes,
        "pool_physical_bytes": pool.physical_storage_bytes,
        "pool_nonzero_count": sum(
            int(torch.count_nonzero(tensor).item())
            for tensor in pool._tensors.values()
        ),
        "pool_unchanged": _pool_unchanged(pool, pool_snapshot),
        "layer_count": len(model.layer_stack.layers),
        "linear_adapter_count": len(target.assembly.packed.adapters),
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
        **inspection,
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
            after_touch["vmhwm_kib"] - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_touch["vmhwm_kib"]
            - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_touch["vmhwm_kib"]
            - after_metadata["vmhwm_kib"],
        ),
    }
    if str(Path(checkpoint_dir).resolve()) == APPROVED_MODEL_DIR:
        validate_cpu_materialization_row(row)
    return row


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
        "process_exit_is_release_boundary": True,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": _source_tree_sha256(source_hashes),
        "rows": list(rows),
    }
    validate_cpu_materialization_preflight(record)
    return record


def run_remote_cpu_materialization_preflight(
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
            f"local CPU materialization directory already exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/cpu_materialization_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_cpu_materialization_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in TP_ROWS:
        completed = command_runner(
            build_ssh_command([
                "env",
                "CUDA_VISIBLE_DEVICES=",
                "PYTHONDONTWRITEBYTECODE=1",
                "OMP_NUM_THREADS=8",
                "MKL_NUM_THREADS=8",
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
        _require_success(completed, "CPU materialization rank worker")
        row = json.loads(completed.stdout)
        validate_cpu_materialization_row(row)
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
    _require_success(finalized, "CPU materialization finalizer")
    record = json.loads(finalized.stdout)
    validate_cpu_materialization_preflight(record)
    if (
        record["source_file_sha256"]
        != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError("CPU materialization source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    round_trip_script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'cpu_materialization_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'cpu_materialization_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "cpu_materialization_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "CPU materialization artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "cpu_materialization_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("CPU materialization artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "cpu_materialization_preflight.json",
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


def execute_remote_cpu_materialization_preflight(
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
    return run_remote_cpu_materialization_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_cpu_materialization_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        tensor_parallel_size=arguments.tp_size,
        tensor_parallel_rank=arguments.tp_rank,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
        process_id=os.getpid(),
    )
    validate_cpu_materialization_row(row)
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _finalize_main(arguments) -> int:
    output = Path(arguments.output)
    if output.exists():
        raise ValueError(
            "CPU materialization preflight output already exists"
        )
    payload = json.load(sys.stdin)
    record = _aggregate(payload.get("rows"), arguments.source_root)
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
        validate_cpu_materialization_preflight(record)
    else:
        record = execute_remote_cpu_materialization_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
