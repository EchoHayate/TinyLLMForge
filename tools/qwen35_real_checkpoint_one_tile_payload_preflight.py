from __future__ import annotations

import argparse
from collections.abc import Mapping
import getpass
import hashlib
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


def _load_sibling(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cpu = _load_sibling(
    "_qwen35_cpu_materialization_preflight_base",
    "qwen35_real_checkpoint_cpu_materialization_preflight.py",
)
base = cpu.base

SCHEMA_VERSION = "qwen35.real-checkpoint-one-tile-payload-preflight.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-one-tile-payload-rank.v1"
REMOTE_TARGET = cpu.REMOTE_TARGET
REMOTE_PYTHON = cpu.REMOTE_PYTHON
APPROVED_MODEL_DIR = cpu.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = cpu.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = cpu.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = cpu.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = cpu.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = cpu.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = cpu.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = cpu.APPROVED_COMPOSITE_SHA256
SELECTED_BINDING_INDEX = 3
SELECTED_SOURCE_NAME = (
    "model.language_model.layers.0.linear_attn.conv1d.weight"
)
SELECTED_TARGET = "layers.0.linear_attention.conv_weight"
HEADER_BYTES = 76648
DATA_START = 76656
TILE_CONTRACTS = {
    (1, 0): {
        "tile_shape": [6144, 4],
        "source_slices": [[0, 6144, None], 0, [0, 4, None]],
        "destination_slices": [[0, 6144, None], [0, 4, None]],
        "payload_relative_start": 1017133184,
        "payload_relative_end": 1017182336,
        "absolute_start": 1017209840,
        "absolute_end": 1017258992,
        "byte_count": 49152,
    },
    (2, 0): {
        "tile_shape": [3072, 4],
        "source_slices": [[0, 3072, None], 0, [0, 4, None]],
        "destination_slices": [[0, 3072, None], [0, 4, None]],
        "payload_relative_start": 1017133184,
        "payload_relative_end": 1017157760,
        "absolute_start": 1017209840,
        "absolute_end": 1017234416,
        "byte_count": 24576,
    },
    (2, 1): {
        "tile_shape": [3072, 4],
        "source_slices": [[3072, 6144, None], 0, [0, 4, None]],
        "destination_slices": [[0, 3072, None], [0, 4, None]],
        "payload_relative_start": 1017157760,
        "payload_relative_end": 1017182336,
        "absolute_start": 1017234416,
        "absolute_end": 1017258992,
        "byte_count": 24576,
    },
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 4734976,
        "post_torch": 4210688,
        "post_metadata": 3948544,
    },
    2: {
        "total": 2637824,
        "post_torch": 2375680,
        "post_metadata": 2113536,
    },
}
EXTRA_PRODUCTION_FILES = (
    "tinyvllm/models/qwen35_checkpoint_tiles.py",
    "tinyvllm/models/qwen35_checkpoint_tile_policy.py",
    "tinyvllm/models/qwen35_checkpoint_tiled_loading.py",
)
SOURCE_FILES = (
    *cpu.SOURCE_FILES,
    *EXTRA_PRODUCTION_FILES,
    "tools/qwen35_real_checkpoint_one_tile_payload_preflight.py",
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-one-tile-payload-runs"
)
LOCAL_RUN_ROOT = cpu.LOCAL_RUN_ROOT

_sha256 = cpu._sha256
_positive_integer = cpu._positive_integer
_non_negative_integer = cpu._non_negative_integer
_source_tree_sha256 = cpu._source_tree_sha256
validate_run_tag = cpu.validate_run_tag
build_ssh_command = cpu.build_ssh_command
_require_success = cpu._require_success
_read_proc_status = cpu._read_proc_status
_memory_point = cpu._memory_point
_install_namespace_packages = cpu._install_namespace_packages
_snapshot_pool = cpu._snapshot_pool
_pool_unchanged = cpu._pool_unchanged
_atomic_write_json = cpu._atomic_write_json


def __getattr__(name: str):
    if name != "Qwen35CheckpointTile":
        raise AttributeError(name)
    try:
        module = __import__(
            "tinyvllm.models.qwen35_checkpoint_tiles",
            fromlist=["Qwen35CheckpointTile"],
        )
    except ModuleNotFoundError:
        _install_namespace_packages(Path(__file__).resolve().parents[1])
        module = __import__(
            "tinyvllm.models.qwen35_checkpoint_tiles",
            fromlist=["Qwen35CheckpointTile"],
        )
    return module.Qwen35CheckpointTile


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing one-tile source: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_source_tar(source_root) -> bytes:
    root = Path(source_root)
    if base.discover_production_source_files(
        root
    ) != cpu.PRODUCTION_SOURCE_FILES:
        raise ValueError("production source closure does not match frozen set")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(f"missing one-tile source: {relative}")
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
    _require_success(staged, "one-tile source staging")
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
    _require_success(verified, "one-tile source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("one-tile remote source hashes do not match local")
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def read_and_verify_exact_range(
    path,
    *,
    absolute_start: int,
    byte_count: int,
) -> dict:
    file_path = Path(path)
    absolute_start = _non_negative_integer(
        absolute_start,
        "absolute_start",
    )
    byte_count = _positive_integer(byte_count, "byte_count")
    reads = []
    for _ in range(2):
        descriptor = os.open(file_path, os.O_RDONLY)
        try:
            payload = os.pread(descriptor, byte_count, absolute_start)
        finally:
            os.close(descriptor)
        if len(payload) != byte_count:
            raise ValueError("short payload read")
        reads.append(payload)
    production, verifier = reads
    production_sha = hashlib.sha256(production).hexdigest()
    verifier_sha = hashlib.sha256(verifier).hexdigest()
    if production_sha != verifier_sha or production != verifier:
        raise ValueError("independent payload hash mismatch")
    return {
        "production_bytes": production,
        "verifier_bytes": verifier,
        "production_sha256": production_sha,
        "verifier_sha256": verifier_sha,
        "open_count": 2,
        "pread_count": 2,
    }


def _tensor_bytes(tensor) -> bytes:
    return tensor.detach().contiguous().view(
        __import__("torch").uint8
    ).numpy().tobytes()


def copy_verify_and_rollback_tile(
    tile,
    source_tensor,
    *,
    unique_tensors,
) -> dict:
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    destination = tile.destination[tile.destination_slices]
    snapshot = destination.clone()
    destination_initially_zero = (
        int(torch.count_nonzero(destination).item()) == 0
    )
    if not destination_initially_zero:
        raise ValueError("selected destination must initially be zero")
    selected_id = id(tile.destination)
    non_selected_before = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) != selected_id
    )
    if not non_selected_before:
        raise ValueError("non-selected tensors must initially be zero")
    source_sha = hashlib.sha256(_tensor_bytes(source_tensor)).hexdigest()
    _copy_qwen35_checkpoint_tile(tile, source_tensor)
    destination_sha = hashlib.sha256(
        _tensor_bytes(destination)
    ).hexdigest()
    changed = not torch.equal(destination, snapshot)
    if destination_sha != source_sha:
        raise ValueError("destination payload hash mismatch")
    non_selected_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) != selected_id
    )
    with torch.no_grad():
        destination.copy_(snapshot)
    rollback = torch.equal(destination, snapshot)
    all_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
    )
    return {
        "source_tensor_sha256": source_sha,
        "destination_sha256": destination_sha,
        "destination_initially_zero": destination_initially_zero,
        "destination_changed_after_copy": changed,
        "non_selected_tensors_remained_zero": non_selected_zero,
        "rollback_restored_selected_destination": rollback,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def _slice_json(values) -> list:
    result = []
    for value in values:
        if isinstance(value, slice):
            result.append([value.start, value.stop, value.step])
        else:
            result.append(value)
    return result


def _validate_memory(row) -> None:
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_payload",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError("one-tile memory points are invalid")
    for name in names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(f"one-tile memory {name} is invalid")
        _positive_integer(point.get("vmrss_kib"), f"{name} vmrss")
        _positive_integer(point.get("vmhwm_kib"), f"{name} vmhwm")
    deltas = (
        _non_negative_integer(
            row.get("total_vmhwm_increment_kib"),
            "total_vmhwm_increment_kib",
        ),
        _non_negative_integer(
            row.get("post_torch_vmhwm_increment_kib"),
            "post_torch_vmhwm_increment_kib",
        ),
        _non_negative_integer(
            row.get("post_metadata_vmhwm_increment_kib"),
            "post_metadata_vmhwm_increment_kib",
        ),
    )
    expected = (
        max(
            0,
            memory["after_payload"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"],
        ),
        max(
            0,
            memory["after_payload"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"],
        ),
        max(
            0,
            memory["after_payload"]["vmhwm_kib"]
            - memory["after_metadata"]["vmhwm_kib"],
        ),
    )
    if deltas != expected:
        raise ValueError("one-tile VmHWM deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    if deltas[0] > ceilings["total"]:
        raise ValueError("one-tile total VmHWM exceeds ceiling")
    if deltas[1] > ceilings["post_torch"]:
        raise ValueError("one-tile post-Torch VmHWM exceeds ceiling")
    if deltas[2] > ceilings["post_metadata"]:
        raise ValueError("one-tile post-metadata VmHWM exceeds ceiling")


def validate_one_tile_payload_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("one-tile row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("one-tile row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("one-tile row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in TILE_CONTRACTS:
        raise ValueError("one-tile TP context is invalid")
    _positive_integer(row.get("process_id"), "process_id")
    contract = TILE_CONTRACTS[tp]
    exact = {
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "production_payload_bytes_read": contract["byte_count"],
        "verifier_payload_bytes_read": contract["byte_count"],
        "logical_payload_bytes_read": contract["byte_count"] * 2,
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "selected_binding_index": SELECTED_BINDING_INDEX,
        "selected_tile_count": 1,
        "selected_source_count": 1,
        "selected_shard_count": 1,
        "selected_source_name": SELECTED_SOURCE_NAME,
        "selected_target": SELECTED_TARGET,
        "selected_transform": "squeeze_conv_channel",
        "selected_kind": "squeeze_axis0",
        "selected_dtype": "torch.bfloat16",
        "selected_source_shape": [6144, 1, 4],
        "selected_tile_shape": contract["tile_shape"],
        "selected_source_slices": contract["source_slices"],
        "selected_destination_slices": contract[
            "destination_slices"
        ],
        "selected_payload_relative_range": [
            contract["payload_relative_start"],
            contract["payload_relative_end"],
        ],
        "selected_absolute_file_range": [
            contract["absolute_start"],
            contract["absolute_end"],
        ],
        "selected_tile_bytes": contract["byte_count"],
        "destination_initially_zero": True,
        "destination_changed_after_copy": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_restored_selected_destination": True,
        "all_unique_tensors_zero_after_rollback": True,
        "open_count": 2,
        "pread_count": 2,
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "target_take_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            if name == "selected_binding_index":
                raise ValueError("one-tile binding is invalid")
            if name == "logical_payload_bytes_read":
                raise ValueError("one-tile payload bytes are invalid")
            if name == "non_selected_tensors_remained_zero":
                raise ValueError("one-tile non-selected mutation detected")
            if name == "rollback_restored_selected_destination":
                raise ValueError("one-tile rollback is invalid")
            if name == "loader_call_count":
                raise ValueError("one-tile loader calls must be zero")
            if name.startswith("cuda_initialized"):
                raise ValueError("one-tile CUDA must remain off")
            raise ValueError(f"one-tile row {name} is invalid")
    hashes = (
        row.get("production_sha256"),
        row.get("verifier_sha256"),
        row.get("source_tensor_sha256"),
        row.get("destination_sha256"),
    )
    for index, digest in enumerate(hashes):
        _sha256(digest, f"one-tile hash {index}")
    if len(set(hashes)) != 1:
        raise ValueError("one-tile hash mismatch")
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("one-tile hostname is invalid")
    _validate_memory(row)
    return row


def validate_one_tile_payload_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("one-tile preflight must be a mapping")
    exact = {
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
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(f"one-tile preflight {name} is invalid")
    hashes = record.get("source_file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SOURCE_FILES):
        raise ValueError("one-tile source hashes are invalid")
    for name, digest in hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError("one-tile source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(base.TP_ROWS):
        raise ValueError("one-tile TP rows are invalid")
    for row in rows:
        validate_one_tile_payload_row(row)
    pids = [row["process_id"] for row in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("one-tile process IDs must be unique")
    return record


def _selected_tile(binding_plan, tp_size, tp_rank):
    from tinyvllm.models.qwen35_checkpoint_tiles import (
        build_qwen35_checkpoint_tile_plan,
    )

    plan = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=65536,
    )
    matches = [
        tile
        for tile in plan.tiles
        if tile.binding_index == SELECTED_BINDING_INDEX
    ]
    if len(matches) != 1:
        raise ValueError("selected binding must produce exactly one tile")
    tile = matches[0]
    binding = binding_plan.bindings[SELECTED_BINDING_INDEX]
    contract = TILE_CONTRACTS[(tp_size, tp_rank)]
    observed = {
        "source": tile.source_name,
        "target": tile.target,
        "kind": tile.kind,
        "dtype": str(tile.dtype),
        "source_shape": list(tile.source_tensor_shape),
        "tile_shape": list(tile.tile_shape),
        "source_slices": _slice_json(tile.source_slices),
        "destination_slices": _slice_json(tile.destination_slices),
        "byte_count": tile.byte_count,
        "offsets": list(binding.load.metadata.data_offsets),
        "transform": binding.load.transform,
    }
    expected = {
        "source": SELECTED_SOURCE_NAME,
        "target": SELECTED_TARGET,
        "kind": "squeeze_axis0",
        "dtype": "torch.bfloat16",
        "source_shape": [6144, 1, 4],
        "tile_shape": contract["tile_shape"],
        "source_slices": contract["source_slices"],
        "destination_slices": contract["destination_slices"],
        "byte_count": contract["byte_count"],
        "offsets": [1017133184, 1017182336],
        "transform": "squeeze_conv_channel",
    }
    if observed != expected:
        raise ValueError("selected tile contract mismatch")
    return tile


def run_one_tile_rank_worker(
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
    after_pool = _memory_point(status_reader())
    backend_calls = []
    attention_forward_count = 0

    class _StaticAttentionBackend(nn.Module):
        def forward(self, *_args, **_kwargs):
            nonlocal attention_forward_count
            attention_forward_count += 1
            raise AssertionError("attention backend must not execute")

    def build_backend(layer_index, query_heads, kv_heads, head_dim):
        backend_calls.append(
            [layer_index, query_heads, kv_heads, head_dim]
        )
        return _StaticAttentionBackend()

    target = factory_module.prepare_qwen35_checkpoint_candidate_target(
        metadata.hf_config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        build_attention_backend=build_backend,
        parameter_device="cpu",
    )
    cpu.inspect_and_touch_cpu_target(target)
    after_target = _memory_point(status_reader())
    tile = _selected_tile(
        target.binding_plan,
        tensor_parallel_size,
        tensor_parallel_rank,
    )
    contract = TILE_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    read_result = read_and_verify_exact_range(
        Path(checkpoint_dir) / APPROVED_SHARD_NAME,
        absolute_start=contract["absolute_start"],
        byte_count=contract["byte_count"],
    )
    source_tensor = torch.frombuffer(
        bytearray(read_result["production_bytes"]),
        dtype=torch.bfloat16,
    ).clone().reshape(tuple(contract["tile_shape"]))
    unique = {}
    model = target.assembly.packed.model
    registrations = (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    )
    for _, tensor in registrations:
        unique.setdefault(id(tensor), tensor)
    copy_result = copy_verify_and_rollback_tile(
        tile,
        source_tensor,
        unique_tensors=tuple(unique.values()),
    )
    after_payload = _memory_point(status_reader())
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": after_metadata,
        "after_pool": after_pool,
        "after_target": after_target,
        "after_payload": after_payload,
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
        "production_payload_bytes_read": contract["byte_count"],
        "verifier_payload_bytes_read": contract["byte_count"],
        "logical_payload_bytes_read": contract["byte_count"] * 2,
        "payload_hashes_recomputed": False,
        "plan_loads": len(tensor_plan.loads),
        "plan_skips": len(tensor_plan.skips),
        "plan_payload_bytes": tensor_plan.payload_bytes,
        "selected_binding_index": SELECTED_BINDING_INDEX,
        "selected_tile_count": 1,
        "selected_source_count": 1,
        "selected_shard_count": 1,
        "selected_source_name": SELECTED_SOURCE_NAME,
        "selected_target": SELECTED_TARGET,
        "selected_transform": "squeeze_conv_channel",
        "selected_kind": "squeeze_axis0",
        "selected_dtype": "torch.bfloat16",
        "selected_source_shape": [6144, 1, 4],
        "selected_tile_shape": contract["tile_shape"],
        "selected_source_slices": contract["source_slices"],
        "selected_destination_slices": contract[
            "destination_slices"
        ],
        "selected_payload_relative_range": [
            contract["payload_relative_start"],
            contract["payload_relative_end"],
        ],
        "selected_absolute_file_range": [
            contract["absolute_start"],
            contract["absolute_end"],
        ],
        "selected_tile_bytes": contract["byte_count"],
        "production_sha256": read_result["production_sha256"],
        "verifier_sha256": read_result["verifier_sha256"],
        **copy_result,
        "open_count": read_result["open_count"],
        "pread_count": read_result["pread_count"],
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "target_take_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": attention_forward_count,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0,
            after_payload["vmhwm_kib"] - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_payload["vmhwm_kib"]
            - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_payload["vmhwm_kib"]
            - after_metadata["vmhwm_kib"],
        ),
    }
    validate_one_tile_payload_row(row)
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
    hashes = _source_hashes(source_root)
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
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_one_tile_payload_preflight(record)
    return record


def run_remote_one_tile_payload_preflight(
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
        raise ValueError(f"local one-tile directory exists: {destination}")
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = f"{remote_run_dir}/one_tile_payload_preflight.json"
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_one_tile_payload_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in base.TP_ROWS:
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
        _require_success(completed, "one-tile rank worker")
        row = json.loads(completed.stdout)
        validate_one_tile_payload_row(row)
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
    _require_success(finalized, "one-tile finalizer")
    record = json.loads(finalized.stdout)
    validate_one_tile_payload_preflight(record)
    if (
        record["source_file_sha256"]
        != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError("one-tile source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'one_tile_payload_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'one_tile_payload_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            "-c",
            script,
        ]),
        input=json.dumps({
            "one_tile_payload_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "one-tile artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "one_tile_payload_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("one-tile artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "one_tile_payload_preflight.json",
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


def execute_remote_one_tile_payload_preflight(
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
    return run_remote_one_tile_payload_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_one_tile_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        tensor_parallel_size=arguments.tp_size,
        tensor_parallel_rank=arguments.tp_rank,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
        process_id=os.getpid(),
    )
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _finalize_main(arguments) -> int:
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("one-tile preflight output already exists")
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
        validate_one_tile_payload_preflight(record)
    else:
        record = execute_remote_one_tile_payload_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
