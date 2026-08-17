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


def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


complete = _load_sibling(
    "_qwen35_complete_gate_loader_core_base",
    "qwen35_real_checkpoint_complete_transaction_preflight.py",
)


PREREQUISITE_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-complete-transaction.v1"
)
PREREQUISITE_ARTIFACT_SHA256 = (
    "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042"
)
PREREQUISITE_ROWS = ((1, 0), (2, 0), (2, 1))
PREREQUISITE_PROCESS_IDS = (3946836, 3960911, 3966499)
SCHEMA_VERSION = "qwen35.real-checkpoint-tiled-loader-core.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-tiled-loader-core-rank.v1"
REMOTE_TARGET = complete.REMOTE_TARGET
REMOTE_PYTHON = complete.REMOTE_PYTHON
APPROVED_MODEL_DIR = complete.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = complete.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = complete.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = complete.APPROVED_INDEX_SHA256
APPROVED_COMPOSITE_SHA256 = complete.APPROVED_COMPOSITE_SHA256
APPROVED_SHARD_NAME = complete.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = complete.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = complete.APPROVED_SHARD_SHA256
ALIAS_GROUPS = complete.ALIAS_GROUPS
PHASE_BINDING_RUNS = complete.PHASE_BINDING_RUNS
SOURCE_FILES = (
    *complete.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_tiled_loader_core_preflight.py",
)
LOCAL_RUN_ROOT = complete.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-tiled-loader-core-runs"
)
LOADER_CORE_CONTRACTS = {
    (1, 0): {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "tile_count": 58169,
        "destination_bytes": 3763655360,
        "materialized_bytes": 3763655360,
        "peak_tile_bytes": 65536,
    },
    (2, 0): {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "tile_count": 29169,
        "destination_bytes": 1881935712,
        "materialized_bytes": 1881935712,
        "peak_tile_bytes": 65536,
    },
    (2, 1): {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "tile_count": 29169,
        "destination_bytes": 1881935712,
        "materialized_bytes": 1881935712,
        "peak_tile_bytes": 65536,
    },
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 8388608,
        "post_torch": 7864320,
        "post_metadata": 7864320,
    },
    2: {
        "total": 4980736,
        "post_torch": 4718592,
        "post_metadata": 4456448,
    },
}
_read_proc_status = complete._read_proc_status
_memory_point = complete._memory_point
_install_namespace_packages = complete._install_namespace_packages
_source_tree_sha256 = complete._source_tree_sha256
_atomic_write_json = complete._atomic_write_json
validate_run_tag = complete.validate_run_tag
build_ssh_command = complete.build_ssh_command
_require_success = complete._require_success


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes(tensor):
    import torch

    return (
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    )


def load_complete_gate_oracle(path):
    artifact = Path(path)
    payload = artifact.read_bytes()
    if _sha256_bytes(payload) != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError("prerequisite artifact hash is invalid")
    record = json.loads(payload)
    if (
        record.get("schema_version") != PREREQUISITE_SCHEMA_VERSION
        or record.get("status") != "PASS"
        or record.get("source_tree_sha256")
        != PREREQUISITE_SOURCE_TREE_SHA256
    ):
        raise ValueError("prerequisite artifact identity is invalid")
    rows = record.get("rows")
    if [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows or ()
    ] != list(PREREQUISITE_ROWS):
        raise ValueError("prerequisite TP rows are invalid")
    if tuple(row.get("process_id") for row in rows) != (
        PREREQUISITE_PROCESS_IDS
    ):
        raise ValueError("prerequisite process IDs are invalid")
    for row in rows:
        if (
            len(row.get("binding_results", ())) != 320
            or len(row.get("phase_results", ())) != 26
            or row.get("unique_destination_count") != 296
        ):
            raise ValueError("prerequisite row coverage is invalid")
    return record


def select_oracle_row(oracle, tensor_parallel_size, tensor_parallel_rank):
    matches = [
        row
        for row in oracle["rows"]
        if (
            row["tp_size"],
            row["tp_rank"],
        ) == (tensor_parallel_size, tensor_parallel_rank)
    ]
    if len(matches) != 1:
        raise ValueError("prerequisite oracle row is invalid")
    return matches[0]


def _registered_tensors(model):
    unique = {}
    for _, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        unique.setdefault(id(tensor), tensor)
    return tuple(unique.values())


def _destination_view(binding):
    destination_slice = binding.destination_slice
    if destination_slice is None:
        return binding.destination
    offset, length = destination_slice
    return binding.destination[offset:offset + length]


def _snapshot_identity(tensors):
    return {
        id(tensor): (
            tensor,
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
        )
        for tensor in tensors
    }


def _require_identity_unchanged(tensors, snapshot):
    if {id(tensor) for tensor in tensors} != set(snapshot):
        raise RuntimeError("private target tensor identity changed")
    for tensor in tensors:
        (
            expected,
            pointer,
            storage_offset,
            shape,
            dtype,
            device,
        ) = snapshot[id(tensor)]
        if (
            tensor is not expected
            or tensor.untyped_storage().data_ptr() != pointer
            or tensor.storage_offset() != storage_offset
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            raise RuntimeError("private target tensor storage changed")


def validate_loaded_private_candidate(
    *,
    candidate,
    target,
    tile_plan,
    model_fingerprint,
    oracle_row,
):
    model = target.assembly.packed.model
    if (
        candidate.owner.model is not model
        or candidate.binding_plan is not target.binding_plan
        or candidate.tile_plan is not tile_plan
        or candidate.model_fingerprint != model_fingerprint
    ):
        raise ValueError("loaded private candidate identity is invalid")
    phase_hashes = {
        phase["phase_name"]: hashlib.sha256()
        for phase in oracle_row["phase_results"]
    }
    aggregate = hashlib.sha256()
    if len(target.binding_plan.bindings) != len(
        oracle_row["binding_results"]
    ):
        raise ValueError("loaded private binding count mismatch")
    for binding, expected in zip(
        target.binding_plan.bindings,
        oracle_row["binding_results"],
    ):
        payload = _tensor_bytes(_destination_view(binding))
        digest = _sha256_bytes(payload)
        if digest != expected["destination_sha256"]:
            raise ValueError(
                "loaded private binding hash mismatch: "
                f"{expected['binding_index']}"
            )
        phase_hashes[expected["phase_name"]].update(payload)
        aggregate.update(payload)
    for phase in oracle_row["phase_results"]:
        if (
            phase_hashes[phase["phase_name"]].hexdigest()
            != phase["destination_sha256"]
        ):
            raise ValueError(
                "loaded private phase hash mismatch: "
                f"{phase['phase_name']}"
            )
    if aggregate.hexdigest() != oracle_row[
        "aggregate_destination_sha256"
    ]:
        raise ValueError("loaded private aggregate hash mismatch")
    return {
        "loaded_state_verified": True,
        "binding_hash_count": len(oracle_row["binding_results"]),
        "phase_hash_count": len(oracle_row["phase_results"]),
        "aggregate_hash_verified": True,
    }


def execute_and_clear_tiled_loader_core(
    *,
    target,
    tile_plan,
    checkpoint_dir,
    model_fingerprint,
    oracle_row,
    load_core,
):
    import torch

    if not callable(load_core):
        raise ValueError("load_core must be callable")
    model = target.assembly.packed.model
    registered = _registered_tensors(model)
    identity = _snapshot_identity(registered)
    selected = {}
    for binding in target.binding_plan.bindings:
        selected.setdefault(id(binding.destination), binding.destination)
    selected_ids = set(selected)
    non_selected_values = {
        id(tensor): tensor.detach().clone()
        for tensor in registered
        if id(tensor) not in selected_ids
    }
    with torch.no_grad():
        for tensor in selected.values():
            tensor.zero_()
    if any(
        int(tensor.count_nonzero().item())
        for tensor in selected.values()
    ):
        raise RuntimeError(
            "private checkpoint destination initialization failed"
        )
    if any(
        not tensor.equal(non_selected_values[id(tensor)])
        for tensor in registered
        if id(tensor) not in selected_ids
    ):
        raise RuntimeError(
            "private destination initialization mutated non-selected tensor"
        )
    target_consumed_before = target._consumed
    if target_consumed_before is not False:
        raise ValueError("private target must remain unconsumed")
    result = None
    error = None
    try:
        candidate = load_core(
            model,
            target.binding_plan,
            tile_plan,
            checkpoint_dir,
            model_fingerprint,
        )
        result = validate_loaded_private_candidate(
            candidate=candidate,
            target=target,
            tile_plan=tile_plan,
            model_fingerprint=model_fingerprint,
            oracle_row=oracle_row,
        )
        result["loader_stats"] = {
            name: getattr(candidate.stats, name)
            for name in (
                "assigned_bindings",
                "source_tensors",
                "shard_count",
                "tile_count",
                "destination_bytes",
                "materialized_bytes",
                "peak_tile_bytes",
            )
        }
        result["non_selected_tensors_unchanged"] = all(
            tensor.equal(non_selected_values[id(tensor)])
            for tensor in registered
            if id(tensor) not in selected_ids
        )
        if not result["non_selected_tensors_unchanged"]:
            raise ValueError("non-selected private tensor mutation")
    except Exception as caught:
        error = caught
    clear_error = None
    for tensor in reversed(tuple(selected.values())):
        try:
            with torch.no_grad():
                tensor.zero_()
        except Exception as caught:
            if clear_error is None:
                clear_error = caught
    try:
        _require_identity_unchanged(registered, identity)
        if any(
            int(tensor.count_nonzero().item())
            for tensor in selected.values()
        ):
            raise RuntimeError(
                "private checkpoint destination clear is incomplete"
            )
        if any(
            not tensor.equal(non_selected_values[id(tensor)])
            for tensor in registered
            if id(tensor) not in selected_ids
        ):
            raise RuntimeError("non-selected private tensor changed")
        if target._consumed is not False:
            raise RuntimeError("private target was consumed")
    except Exception as caught:
        if clear_error is None:
            clear_error = caught
    if clear_error is not None:
        raise RuntimeError(
            "private target cleanup failed"
        ) from clear_error
    if error is not None:
        raise error
    result.update({
        "target_consumed_before": target_consumed_before,
        "target_consumed_after": target._consumed,
        "selected_destinations_initialized_zero": True,
        "unique_destination_count": len(selected),
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
    })
    return result


def _validate_memory(row):
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_load_and_clear",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError("loader-core memory points are invalid")
    expected = (
        memory["after_load_and_clear"]["vmhwm_kib"]
        - memory["before"]["vmhwm_kib"],
        memory["after_load_and_clear"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
        memory["after_load_and_clear"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    observed = (
        row.get("total_vmhwm_increment_kib"),
        row.get("post_torch_vmhwm_increment_kib"),
        row.get("post_metadata_vmhwm_increment_kib"),
    )
    if observed != expected:
        raise ValueError("loader-core memory deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    memory_names = ("total", "post_torch", "post_metadata")
    if any(
        value > ceilings[name]
        for value, name in zip(observed, memory_names)
    ):
        details = ", ".join(
            f"{name}={value}/{ceilings[name]} KiB"
            for value, name in zip(observed, memory_names)
        )
        raise ValueError(
            f"loader-core memory ceiling exceeded: {details}"
        )


def validate_tiled_loader_core_row(row):
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or tp not in LOADER_CORE_CONTRACTS
    ):
        raise ValueError("loader-core row schema or TP is invalid")
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "prerequisite_source_tree_sha256": (
            PREREQUISITE_SOURCE_TREE_SHA256
        ),
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": ALIAS_GROUPS,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "loaded_state_verified": True,
        "loader_core_call_count": 1,
        "loader_stats": LOADER_CORE_CONTRACTS[tp],
        "target_consumed_before": False,
        "target_consumed_after": False,
        "selected_destinations_initialized_zero": True,
        "all_selected_destinations_zero_after_clear": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "non_selected_tensors_unchanged": True,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    messages = {
        "loader_core_call_count": "loader-core call count is invalid",
        "target_consumed_after": "private target was consumed",
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(messages.get(
                name, f"loader-core {name} is invalid"
            ))
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError("loader-core process ID is invalid")
    _validate_memory(row)
    return row


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


def _pool_unchanged(pool, snapshot):
    if tuple(pool._bindings.items()) != snapshot["bindings"]:
        return False
    if set(pool._tensors) != set(snapshot["tensors"]):
        return False
    return all(
        (
            id(tensor),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor._version,
        ) == snapshot["tensors"][key]
        for key, tensor in pool._tensors.items()
    )


def _alias_groups(binding_plan):
    groups = {}
    for index, binding in enumerate(binding_plan.bindings):
        groups.setdefault(id(binding.destination), []).append(index)
    return sorted(
        [
            values
            for values in groups.values()
            if len(values) > 1
        ],
        key=lambda values: values[0],
    )


def run_tiled_loader_core_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    prerequisite_artifact,
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
    oracle = load_complete_gate_oracle(prerequisite_artifact)
    oracle_row = select_oracle_row(
        oracle, tensor_parallel_size, tensor_parallel_rank
    )
    metadata_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_metadata", fromlist=["*"]
    )
    checkpoint_module = __import__(
        "tinyvllm.models.qwen35_checkpoint", fromlist=["*"]
    )
    hybrid_module = __import__(
        "tinyvllm.engine.hybrid_state", fromlist=["*"]
    )
    layout_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_state", fromlist=["*"]
    )
    factory_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_candidate_factory",
        fromlist=["*"],
    )
    tiles_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_tiles", fromlist=["*"]
    )
    loader_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_tiled_loading",
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
    pool = hybrid_module.HybridStateTensorPool(
        layout, capacity=1, device="cpu"
    )
    pool_snapshot = _snapshot_pool(pool)
    after_pool = _memory_point(status_reader())
    attention_forward_count = 0

    class _Backend(nn.Module):
        def forward(self, *_args, **_kwargs):
            nonlocal attention_forward_count
            attention_forward_count += 1
            raise AssertionError("attention backend must not execute")

    target = factory_module.prepare_qwen35_checkpoint_candidate_target(
        metadata.hf_config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        build_attention_backend=lambda *_args: _Backend(),
        parameter_device="cpu",
    )
    after_target = _memory_point(status_reader())
    tile_plan = tiles_module.build_qwen35_checkpoint_tile_plan(
        target.binding_plan, max_tile_bytes=65536
    )
    contract = LOADER_CORE_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    if (
        len(tile_plan.tiles) != contract["tile_count"]
        or tile_plan.destination_bytes != contract["destination_bytes"]
        or tile_plan.peak_tile_bytes != contract["peak_tile_bytes"]
    ):
        raise ValueError("loader-core tile plan is invalid")
    loader_core_call_count = 0

    def load_core(*arguments):
        nonlocal loader_core_call_count
        loader_core_call_count += 1
        return loader_module._load_qwen35_candidate_with_tile_plan(
            *arguments
        )

    result = execute_and_clear_tiled_loader_core(
        target=target,
        tile_plan=tile_plan,
        checkpoint_dir=checkpoint_dir,
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        oracle_row=oracle_row,
        load_core=load_core,
    )
    after_load_and_clear = _memory_point(status_reader())
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": after_metadata,
        "after_pool": after_pool,
        "after_target": after_target,
        "after_load_and_clear": after_load_and_clear,
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
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "prerequisite_source_tree_sha256": (
            PREREQUISITE_SOURCE_TREE_SHA256
        ),
        "config_sha256": metadata.config_sha256,
        "index_sha256": metadata.index_sha256,
        "config_index_header_sha256": (
            metadata.config_index_header_sha256
        ),
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "selected_binding_count": len(target.binding_plan.bindings),
        "alias_groups": _alias_groups(target.binding_plan),
        "loader_core_call_count": loader_core_call_count,
        "pool_unchanged": _pool_unchanged(pool, pool_snapshot),
        "model_forward_count": 0,
        "attention_forward_count": attention_forward_count,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0,
            after_load_and_clear["vmhwm_kib"] - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_load_and_clear["vmhwm_kib"]
            - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_load_and_clear["vmhwm_kib"]
            - after_metadata["vmhwm_kib"],
        ),
        **result,
    }
    validate_tiled_loader_core_row(row)
    return row


def validate_tiled_loader_core_preflight(record):
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
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "prerequisite_source_tree_sha256": (
            PREREQUISITE_SOURCE_TREE_SHA256
        ),
        "fresh_process_per_rank": True,
    }
    if any(record.get(name) != value for name, value in exact.items()):
        raise ValueError("loader-core preflight identity is invalid")
    hashes = record.get("source_file_sha256")
    if set(hashes or {}) != set(SOURCE_FILES):
        raise ValueError("loader-core source hashes are invalid")
    if record.get("source_tree_sha256") != _source_tree_sha256(hashes):
        raise ValueError("loader-core source tree is invalid")
    rows = record.get("rows")
    if [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows or ()
    ] != list(PREREQUISITE_ROWS):
        raise ValueError("loader-core TP rows are invalid")
    for row in rows:
        validate_tiled_loader_core_row(row)
    if len({row["process_id"] for row in rows}) != 3:
        raise ValueError("loader-core process IDs must be unique")
    return record


def _source_hashes(root):
    root = Path(root)
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(f"missing loader-core source: {name}")
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def stage_source_and_prerequisite(
    source_root,
    run_tag,
    *,
    prerequisite_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    prerequisite_path = Path(prerequisite_artifact)
    prerequisite_payload = prerequisite_path.read_bytes()
    if _sha256_bytes(prerequisite_payload) != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError("prerequisite artifact hash is invalid")
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_prerequisite = (
        f"{remote_run_dir}/complete_checkpoint_transaction_preflight.json"
    )
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
        input=build_source_tar(source_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(staged, "loader-core source staging")
    prerequisite_staged = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            f"cat > {shlex.quote(remote_prerequisite)}",
        ]),
        input=prerequisite_payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(
        prerequisite_staged, "loader-core prerequisite staging"
    )
    local_hashes = _source_hashes(source_root)
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"prerequisite=pathlib.Path({remote_prerequisite!r})",
        f"names={list(SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "payload={'source':result,'prerequisite':hashlib.sha256(prerequisite.read_bytes()).hexdigest()}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
    ])
    verified = command_runner(
        build_ssh_command([
            "env", "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON, "-B", "-c", script,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "loader-core staged hashing")
    remote = json.loads(verified.stdout)
    if remote.get("source") != local_hashes:
        raise ValueError("loader-core remote source hashes do not match")
    if remote.get("prerequisite") != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError("loader-core remote prerequisite hash mismatch")
    return {
        "remote_source_dir": remote_source_dir,
        "remote_prerequisite_artifact": remote_prerequisite,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "prerequisite_artifact_sha256": remote["prerequisite"],
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
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "prerequisite_source_tree_sha256": (
            PREREQUISITE_SOURCE_TREE_SHA256
        ),
        "fresh_process_per_rank": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_tiled_loader_core_preflight(record)
    return record


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "remote_prerequisite_artifact": (
            staged["remote_prerequisite_artifact"]
        ),
        "prerequisite_artifact_sha256": (
            staged["prerequisite_artifact_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_tiled_loader_core_preflight(
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
            f"local loader-core directory exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/tiled_loader_core_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_checkpoint_tiled_loader_core_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in PREREQUISITE_ROWS:
        completed = command_runner(
            build_ssh_command([
                "env", "CUDA_VISIBLE_DEVICES=",
                "PYTHONDONTWRITEBYTECODE=1",
                "OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8",
                REMOTE_PYTHON, "-B", worker,
                "internal-rank-worker",
                "--source-root", staged["remote_source_dir"],
                "--checkpoint-dir", APPROVED_MODEL_DIR,
                "--prerequisite-artifact",
                staged["remote_prerequisite_artifact"],
                "--tp-size", str(tp_size),
                "--tp-rank", str(tp_rank),
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, "loader-core rank worker")
        row = json.loads(completed.stdout)
        validate_tiled_loader_core_row(row)
        rows.append(row)
    finalized = command_runner(
        build_ssh_command([
            "env", "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON, "-B", worker,
            "internal-finalize",
            "--source-root", staged["remote_source_dir"],
            "--output", remote_artifact,
        ]),
        input=json.dumps({"rows": rows}),
        text=True,
        capture_output=True,
    )
    _require_success(finalized, "loader-core finalizer")
    record = json.loads(finalized.stdout)
    validate_tiled_loader_core_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("loader-core source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'tiled_loader_core_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'tiled_loader_core_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([
            "env", "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON, "-B", "-c", script,
        ]),
        input=json.dumps({
            "tiled_loader_core_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "loader-core artifact round trip")
    if json.loads(round_trip.stdout) != {
        "tiled_loader_core_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("loader-core artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "tiled_loader_core_preflight.json",
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


def execute_remote_tiled_loader_core_preflight(
    source_root,
    run_tag,
    *,
    prerequisite_artifact,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source_and_prerequisite(
        source_root,
        run_tag,
        prerequisite_artifact=prerequisite_artifact,
        command_runner=command_runner,
    )
    return run_remote_tiled_loader_core_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments):
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not approved")
    row = run_tiled_loader_core_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        prerequisite_artifact=arguments.prerequisite_artifact,
        tensor_parallel_size=arguments.tp_size,
        tensor_parallel_rank=arguments.tp_rank,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
        process_id=os.getpid(),
    )
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _finalize_main(arguments):
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("loader-core preflight output already exists")
    payload = json.load(sys.stdin)
    record = _aggregate(payload.get("rows"), arguments.source_root)
    _atomic_write_json(output, record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-tag", required=True)
    run_parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    run_parser.add_argument("--prerequisite-artifact", required=True)
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--prerequisite-artifact", required=True)
    worker_parser.add_argument("--tp-size", type=int, required=True)
    worker_parser.add_argument("--tp-rank", type=int, required=True)
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
        validate_tiled_loader_core_preflight(record)
    else:
        record = execute_remote_tiled_loader_core_preflight(
            arguments.source_root,
            arguments.run_tag,
            prerequisite_artifact=arguments.prerequisite_artifact,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
