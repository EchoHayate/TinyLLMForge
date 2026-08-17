from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
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
from types import MappingProxyType


def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


complete = _load_sibling(
    "_qwen35_complete_gate_private_ownership_base",
    "qwen35_real_checkpoint_complete_transaction_preflight.py",
)
loader_core = _load_sibling(
    "_qwen35_loader_core_private_ownership_base",
    "qwen35_real_checkpoint_tiled_loader_core_preflight.py",
)


COMPLETE_ARTIFACT_SHA256 = (
    "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
)
LOADER_CORE_ARTIFACT_SHA256 = (
    "58df3dfa9fec11d1fd079c9473766413232bd3f928f537ac87e047e13ef65aae"
)
COMPLETE_SOURCE_TREE_SHA256 = (
    "da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042"
)
LOADER_CORE_SOURCE_TREE_SHA256 = (
    "c84eb9252bb5294d0fe00a4c48769e659274eb0a2d8c4548c25fb1ecdaf6869b"
)
PREREQUISITE_ROWS = ((1, 0), (2, 0), (2, 1))
SCHEMA_VERSION = "qwen35.real-checkpoint-private-ownership.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-private-ownership-rank.v1"
REMOTE_TARGET = loader_core.REMOTE_TARGET
REMOTE_PYTHON = loader_core.REMOTE_PYTHON
APPROVED_MODEL_DIR = loader_core.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = (
    loader_core.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_CONFIG_SHA256 = loader_core.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = loader_core.APPROVED_INDEX_SHA256
APPROVED_COMPOSITE_SHA256 = loader_core.APPROVED_COMPOSITE_SHA256
APPROVED_SHARD_NAME = loader_core.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = loader_core.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = loader_core.APPROVED_SHARD_SHA256
AUTHORIZATION_SHA256 = (
    "10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4"
)
MAX_TENSOR_BYTES = 1017118720
STREAMED_STATS = {
    (1, 0): {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "loaded_bytes": 3763655360,
        "peak_source_bytes": 1017118720,
    },
    (2, 0): {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "loaded_bytes": 3763655360,
        "peak_source_bytes": 1017118720,
    },
    (2, 1): {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "loaded_bytes": 3763655360,
        "peak_source_bytes": 1017118720,
    },
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 10485760,
        "post_torch": 10223616,
        "post_metadata": 9961472,
    },
    2: {
        "total": 7340032,
        "post_torch": 7077888,
        "post_metadata": 6815744,
    },
}
SOURCE_FILES = (
    *loader_core.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_private_candidate_ownership_preflight.py",
)
LOCAL_RUN_ROOT = loader_core.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-private-ownership-runs"
)
WORKER_CONTEXTS = (
    (1, 0, "success"),
    (1, 0, "injected_failure"),
    (2, 0, "success"),
    (2, 0, "injected_failure"),
    (2, 1, "success"),
    (2, 1, "injected_failure"),
)
_read_proc_status = loader_core._read_proc_status
_memory_point = loader_core._memory_point
_install_namespace_packages = loader_core._install_namespace_packages
_source_tree_sha256 = loader_core._source_tree_sha256
_atomic_write_json = loader_core._atomic_write_json
validate_run_tag = loader_core.validate_run_tag
build_ssh_command = loader_core.build_ssh_command
_require_success = loader_core._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_exact_json(path, expected_sha256):
    payload = Path(path).read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("private ownership prerequisite hash is invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "private ownership prerequisite JSON is invalid"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            "private ownership prerequisite must be a JSON object"
        )
    return value


def _row_map(rows):
    result = {}
    for row in rows:
        key = (row.get("tp_size"), row.get("tp_rank"))
        if key in result:
            raise ValueError("duplicate private ownership prerequisite row")
        result[key] = MappingProxyType(row)
    if tuple(result) != PREREQUISITE_ROWS:
        raise ValueError("private ownership prerequisite TP rows are invalid")
    return MappingProxyType(result)


@dataclass(frozen=True)
class PrivateOwnershipPrerequisites:
    complete_artifact_sha256: str
    loader_core_artifact_sha256: str
    complete_source_tree_sha256: str
    loader_core_source_tree_sha256: str
    complete_rows: tuple[tuple[int, int], ...]
    loader_core_rows: tuple[tuple[int, int], ...]
    complete_row_map: MappingProxyType
    loader_core_row_map: MappingProxyType
    loader_core_source_file_sha256: MappingProxyType


def load_private_ownership_prerequisites(
    complete_artifact,
    loader_core_artifact,
) -> PrivateOwnershipPrerequisites:
    complete_record = loader_core.load_complete_gate_oracle(
        complete_artifact
    )
    loader_core_record = _read_exact_json(
        loader_core_artifact,
        LOADER_CORE_ARTIFACT_SHA256,
    )
    loader_core.validate_tiled_loader_core_preflight(loader_core_record)
    if complete_record["source_tree_sha256"] != (
        COMPLETE_SOURCE_TREE_SHA256
    ):
        raise ValueError("complete prerequisite source tree is invalid")
    if loader_core_record["source_tree_sha256"] != (
        LOADER_CORE_SOURCE_TREE_SHA256
    ):
        raise ValueError("loader-core prerequisite source tree is invalid")
    complete_rows = _row_map(complete_record["rows"])
    loader_core_rows = _row_map(loader_core_record["rows"])
    source_hashes = loader_core_record["source_file_sha256"]
    if (
        not isinstance(source_hashes, dict)
        or len(source_hashes) != 44
    ):
        raise ValueError(
            "loader-core prerequisite source closure is invalid"
        )
    return PrivateOwnershipPrerequisites(
        complete_artifact_sha256=COMPLETE_ARTIFACT_SHA256,
        loader_core_artifact_sha256=LOADER_CORE_ARTIFACT_SHA256,
        complete_source_tree_sha256=COMPLETE_SOURCE_TREE_SHA256,
        loader_core_source_tree_sha256=LOADER_CORE_SOURCE_TREE_SHA256,
        complete_rows=tuple(complete_rows),
        loader_core_rows=tuple(loader_core_rows),
        complete_row_map=complete_rows,
        loader_core_row_map=loader_core_rows,
        loader_core_source_file_sha256=MappingProxyType(source_hashes),
    )


def select_private_ownership_prerequisite_rows(
    prerequisites,
    tensor_parallel_size,
    tensor_parallel_rank,
):
    if type(prerequisites) is not PrivateOwnershipPrerequisites:
        raise ValueError(
            "prerequisites must be exact PrivateOwnershipPrerequisites"
        )
    key = (tensor_parallel_size, tensor_parallel_rank)
    try:
        return (
            prerequisites.complete_row_map[key],
            prerequisites.loader_core_row_map[key],
        )
    except KeyError as error:
        raise ValueError(
            "private ownership prerequisite TP row is invalid"
        ) from error


def validate_private_loaded_candidate(
    *,
    candidate,
    target,
    model_fingerprint,
    oracle_row,
):
    model = target.assembly.packed.model
    if (
        candidate.owner.model is not model
        or candidate.binding_plan is not target.binding_plan
        or candidate.model_fingerprint != model_fingerprint
    ):
        raise ValueError("private loaded candidate identity is invalid")
    if (
        hasattr(candidate.owner, "pool")
        and candidate.owner.pool is not target.pool
    ):
        raise ValueError("private loaded candidate pool is invalid")
    phase_hashes = {
        phase["phase_name"]: hashlib.sha256()
        for phase in oracle_row["phase_results"]
    }
    aggregate = hashlib.sha256()
    binding_destination_sha256 = []
    if len(target.binding_plan.bindings) != len(
        oracle_row["binding_results"]
    ):
        raise ValueError("private loaded binding count mismatch")
    for binding, expected in zip(
        target.binding_plan.bindings,
        oracle_row["binding_results"],
    ):
        payload = loader_core._tensor_bytes(
            loader_core._destination_view(binding)
        )
        digest = _sha256(payload)
        if digest != expected["destination_sha256"]:
            raise ValueError(
                "private loaded binding hash mismatch: "
                f"{expected['binding_index']}"
            )
        binding_destination_sha256.append(digest)
        phase_hashes[expected["phase_name"]].update(payload)
        aggregate.update(payload)
    phase_destination_sha256 = {
        phase_name: digest.hexdigest()
        for phase_name, digest in phase_hashes.items()
    }
    for phase in oracle_row["phase_results"]:
        if (
            phase_destination_sha256[phase["phase_name"]]
            != phase["destination_sha256"]
        ):
            raise ValueError(
                "private loaded phase hash mismatch: "
                f"{phase['phase_name']}"
            )
    aggregate_destination_sha256 = aggregate.hexdigest()
    if aggregate_destination_sha256 != oracle_row[
        "aggregate_destination_sha256"
    ]:
        raise ValueError("private loaded aggregate hash mismatch")
    return {
        "loaded_state_verified": True,
        "binding_hash_count": len(oracle_row["binding_results"]),
        "phase_hash_count": len(oracle_row["phase_results"]),
        "aggregate_hash_verified": True,
        "binding_destination_sha256": binding_destination_sha256,
        "phase_destination_sha256": phase_destination_sha256,
        "aggregate_destination_sha256": aggregate_destination_sha256,
    }


def execute_and_clear_private_candidate_ownership(
    *,
    target,
    model_fingerprint,
    oracle_row,
    adapter_call,
    expected_error_message=None,
    failure_evidence=None,
):
    import torch

    if not callable(adapter_call):
        raise ValueError("adapter_call must be callable")
    if expected_error_message is not None and (
        not isinstance(expected_error_message, str)
        or not expected_error_message
    ):
        raise ValueError("expected_error_message must be non-empty")
    if failure_evidence is not None and not isinstance(
        failure_evidence,
        dict,
    ):
        raise ValueError("failure_evidence must be a dict")
    model = target.assembly.packed.model
    registered = loader_core._registered_tensors(model)
    identity = loader_core._snapshot_identity(registered)
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
            "private ownership destination initialization failed"
        )
    if any(
        not tensor.equal(non_selected_values[id(tensor)])
        for tensor in registered
        if id(tensor) not in selected_ids
    ):
        raise RuntimeError(
            "private ownership initialization mutated non-selected tensor"
        )
    target_consumed_before = target._consumed
    if target_consumed_before is not False:
        raise ValueError("private ownership target must be unconsumed")
    result = None
    error = None
    candidate_returned = False
    try:
        candidate = adapter_call()
        candidate_returned = True
        if target._consumed is not True:
            raise RuntimeError(
                "private ownership target was not consumed"
            )
        result = validate_private_loaded_candidate(
            candidate=candidate,
            target=target,
            model_fingerprint=model_fingerprint,
            oracle_row=oracle_row,
        )
        result["loader_stats"] = {
            name: getattr(candidate.stats, name)
            for name in (
                "assigned_bindings",
                "source_tensors",
                "shard_count",
                "loaded_bytes",
                "peak_source_bytes",
            )
        }
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
        loader_core._require_identity_unchanged(registered, identity)
        if any(
            int(tensor.count_nonzero().item())
            for tensor in selected.values()
        ):
            raise RuntimeError(
                "private ownership destination clear is incomplete"
            )
        if any(
            not tensor.equal(non_selected_values[id(tensor)])
            for tensor in registered
            if id(tensor) not in selected_ids
        ):
            raise RuntimeError(
                "private ownership non-selected tensor changed"
            )
        if target._consumed is not True:
            raise RuntimeError(
                "private ownership target consumption was lost"
            )
    except Exception as caught:
        if clear_error is None:
            clear_error = caught
    if clear_error is not None:
        raise RuntimeError(
            "private ownership target cleanup failed"
        ) from clear_error
    cleanup = {
        "target_consumed_before": target_consumed_before,
        "target_consumed_after": target._consumed,
        "candidate_returned": candidate_returned,
        "selected_destinations_initialized_zero": True,
        "unique_destination_count": len(selected),
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
    }
    if error is not None:
        if (
            expected_error_message is None
            or str(error) != expected_error_message
        ):
            raise error
        result = {
            "expected_failure_observed": True,
            "expected_failure_type": type(error).__name__,
            "expected_failure_message": str(error),
            **(failure_evidence or {}),
        }
    result.update(cleanup)
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
        raise ValueError("private ownership memory points are invalid")
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
        raise ValueError("private ownership memory deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    names = ("total", "post_torch", "post_metadata")
    if any(
        value > ceilings[name]
        for value, name in zip(observed, names)
    ):
        details = ", ".join(
            f"{name}={value}/{ceilings[name]} KiB"
            for value, name in zip(observed, names)
        )
        raise ValueError(
            f"private ownership memory ceiling exceeded: {details}"
        )


def validate_private_candidate_ownership_row(row):
    tp = (row.get("tp_size"), row.get("tp_rank"))
    mode = row.get("mode")
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or tp not in PREREQUISITE_ROWS
        or mode not in ("success", "injected_failure")
    ):
        raise ValueError("private ownership row schema is invalid")
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "complete_artifact_sha256": COMPLETE_ARTIFACT_SHA256,
        "loader_core_artifact_sha256": LOADER_CORE_ARTIFACT_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": loader_core.ALIAS_GROUPS,
        "provider_call_count": 1,
        "adapter_call_count": 1,
        "target_consumed_before": False,
        "target_consumed_after": True,
        "selected_destinations_initialized_zero": True,
        "all_selected_destinations_zero_after_clear": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "non_selected_tensors_unchanged": True,
        "candidate_published": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(f"private ownership {name} is invalid")
    if mode == "success":
        success = {
            "candidate_returned": True,
            "loaded_state_verified": True,
            "binding_hash_count": 320,
            "phase_hash_count": 26,
            "aggregate_hash_verified": True,
            "loader_stats": STREAMED_STATS[tp],
            "expected_failure_observed": False,
            "assignment_call_count": 320,
            "first_source_hashes_verified": True,
        }
        binding_hashes = row.get("binding_destination_sha256")
        phase_hashes = row.get("phase_destination_sha256")
        aggregate_hash = row.get("aggregate_destination_sha256")
        if (
            not isinstance(binding_hashes, list)
            or len(binding_hashes) != 320
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in binding_hashes
            )
            or not isinstance(phase_hashes, Mapping)
            or len(phase_hashes) != 26
            or any(
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or len(value) != 64
                for name, value in phase_hashes.items()
            )
            or not isinstance(aggregate_hash, str)
            or len(aggregate_hash) != 64
        ):
            raise ValueError(
                "private ownership success hash evidence is invalid"
            )
    else:
        success = {
            "candidate_returned": False,
            "expected_failure_observed": True,
            "expected_failure_type": "RuntimeError",
            "expected_failure_message": (
                "injected ownership-transfer assignment failure"
            ),
            "assignment_call_count": 1,
            "first_source_name": (
                "model.language_model.embed_tokens.weight"
            ),
            "first_source_binding_count": 1,
            "first_source_hashes_verified": True,
        }
        first_indices = row.get("first_source_binding_indices")
        first_hashes = row.get("first_source_binding_sha256")
        if (
            not isinstance(first_indices, list)
            or len(first_indices) != row.get("first_source_binding_count")
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in first_indices
            )
            or not isinstance(first_hashes, list)
            or len(first_hashes) != len(first_indices)
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in first_hashes
            )
        ):
            raise ValueError(
                "private ownership failure hash evidence is invalid"
            )
    for name, value in success.items():
        if row.get(name) != value:
            raise ValueError(f"private ownership {name} is invalid")
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError("private ownership process ID is invalid")
    _validate_memory(row)
    return row


def run_private_candidate_ownership_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    complete_artifact,
    loader_core_artifact,
    tensor_parallel_size,
    tensor_parallel_rank,
    mode,
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
    prerequisites = load_private_ownership_prerequisites(
        complete_artifact,
        loader_core_artifact,
    )
    oracle_row, _loader_core_row = (
        select_private_ownership_prerequisite_rows(
            prerequisites,
            tensor_parallel_size,
            tensor_parallel_rank,
        )
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
    candidate_loader_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_candidate_loader",
        fromlist=["*"],
    )
    worker_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_worker", fromlist=["*"]
    )
    streaming_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_streaming",
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
    pool_snapshot = loader_core._snapshot_pool(pool)
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
    provider_call_count = 0
    adapter_call_count = 0

    def provide_target():
        nonlocal provider_call_count
        provider_call_count += 1
        if provider_call_count != 1:
            raise RuntimeError("private target provider called twice")
        return target

    adapter = (
        candidate_loader_module
        .build_qwen35_authorized_checkpoint_candidate_loader(
            provide_target,
            authorization_sha256=AUTHORIZATION_SHA256,
        )
    )
    request = worker_module.Qwen35CheckpointCandidateLoadRequest(
        checkpoint_dir=str(Path(checkpoint_dir).resolve()),
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        max_tensor_bytes=MAX_TENSOR_BYTES,
        authorization_sha256=AUTHORIZATION_SHA256,
    )
    assignment_call_count = 0
    failure_evidence = {}
    binding_indices = {
        id(binding): index
        for index, binding in enumerate(target.binding_plan.bindings)
    }
    original_assignment = (
        streaming_module._assign_qwen35_checkpoint_source_bindings
    )

    def injected_assignment(bindings, source, **kwargs):
        nonlocal assignment_call_count
        if assignment_call_count:
            raise RuntimeError(
                "injected ownership-transfer assignment failure"
            )
        result = original_assignment(bindings, source, **kwargs)
        assignment_call_count += 1
        expected_by_index = {
            item["binding_index"]: item
            for item in oracle_row["binding_results"]
        }
        hashes_verified = True
        first_source_binding_indices = []
        first_source_binding_sha256 = []
        for binding in bindings:
            index = binding_indices[id(binding)]
            digest = _sha256(loader_core._tensor_bytes(
                loader_core._destination_view(binding)
            ))
            first_source_binding_indices.append(index)
            first_source_binding_sha256.append(digest)
            hashes_verified = (
                hashes_verified
                and digest
                == expected_by_index[index]["destination_sha256"]
            )
        failure_evidence.update({
            "assignment_call_count": assignment_call_count,
            "first_source_name": bindings[0].load.weight.source.name,
            "first_source_binding_count": len(bindings),
            "first_source_binding_indices": (
                first_source_binding_indices
            ),
            "first_source_binding_sha256": (
                first_source_binding_sha256
            ),
            "first_source_hashes_verified": hashes_verified,
        })
        return result

    if mode == "injected_failure":
        streaming_module._assign_qwen35_checkpoint_source_bindings = (
            injected_assignment
        )

    def adapter_call():
        nonlocal adapter_call_count, assignment_call_count
        adapter_call_count += 1
        candidate = adapter(request)
        if mode == "success":
            assignment_call_count = candidate.stats.source_tensors
        return candidate

    try:
        result = execute_and_clear_private_candidate_ownership(
            target=target,
            model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
            oracle_row=oracle_row,
            adapter_call=adapter_call,
            expected_error_message=(
                "injected ownership-transfer assignment failure"
                if mode == "injected_failure" else None
            ),
            failure_evidence=(
                failure_evidence
                if mode == "injected_failure" else None
            ),
        )
    finally:
        streaming_module._assign_qwen35_checkpoint_source_bindings = (
            original_assignment
        )
    if mode == "success":
        first = oracle_row["binding_results"][0]
        result.update({
            "expected_failure_observed": False,
            "assignment_call_count": assignment_call_count,
            "first_source_name": first["source_name"],
            "first_source_binding_count": 1,
            "first_source_hashes_verified": True,
        })
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
        "mode": mode,
        "tp_size": tensor_parallel_size,
        "tp_rank": tensor_parallel_rank,
        "process_id": process_id,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "complete_artifact_sha256": COMPLETE_ARTIFACT_SHA256,
        "loader_core_artifact_sha256": LOADER_CORE_ARTIFACT_SHA256,
        "config_sha256": metadata.config_sha256,
        "index_sha256": metadata.index_sha256,
        "config_index_header_sha256": (
            metadata.config_index_header_sha256
        ),
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "selected_binding_count": len(target.binding_plan.bindings),
        "alias_groups": loader_core._alias_groups(target.binding_plan),
        "provider_call_count": provider_call_count,
        "adapter_call_count": adapter_call_count,
        "pool_unchanged": loader_core._pool_unchanged(
            pool, pool_snapshot
        ),
        "candidate_published": False,
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
    validate_private_candidate_ownership_row(row)
    return row


def validate_private_candidate_ownership_preflight(record):
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
        "complete_artifact_sha256": COMPLETE_ARTIFACT_SHA256,
        "loader_core_artifact_sha256": LOADER_CORE_ARTIFACT_SHA256,
        "fresh_process_per_attempt": True,
    }
    if any(record.get(name) != value for name, value in exact.items()):
        raise ValueError("private ownership preflight identity is invalid")
    hashes = record.get("source_file_sha256")
    if set(hashes or {}) != set(SOURCE_FILES):
        raise ValueError("private ownership source hashes are invalid")
    if record.get("source_tree_sha256") != _source_tree_sha256(hashes):
        raise ValueError("private ownership source tree is invalid")
    rows = record.get("rows")
    if [
        (
            row.get("tp_size"),
            row.get("tp_rank"),
            row.get("mode"),
        )
        for row in rows or ()
    ] != list(WORKER_CONTEXTS):
        raise ValueError("private ownership worker rows are invalid")
    for row in rows:
        validate_private_candidate_ownership_row(row)
    if len({row["process_id"] for row in rows}) != 6:
        raise ValueError("private ownership process IDs must be unique")
    return record


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing private ownership source: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(
                    f"missing private ownership source: {name}"
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def stage_source_and_prerequisites(
    source_root,
    run_tag,
    *,
    complete_artifact,
    loader_core_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    complete_payload = Path(complete_artifact).read_bytes()
    loader_core_payload = Path(loader_core_artifact).read_bytes()
    if _sha256(complete_payload) != COMPLETE_ARTIFACT_SHA256:
        raise ValueError("complete prerequisite artifact hash is invalid")
    if _sha256(loader_core_payload) != LOADER_CORE_ARTIFACT_SHA256:
        raise ValueError("loader-core prerequisite artifact hash is invalid")
    local_hashes = _source_hashes(source_root)
    prerequisite_hashes = load_private_ownership_prerequisites(
        complete_artifact,
        loader_core_artifact,
    ).loader_core_source_file_sha256
    if {
        name: local_hashes[name]
        for name in loader_core.SOURCE_FILES
    } != dict(prerequisite_hashes):
        raise ValueError(
            "private ownership source does not match loader-core prerequisite"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_complete = (
        f"{remote_run_dir}/complete_checkpoint_transaction_preflight.json"
    )
    remote_loader_core = (
        f"{remote_run_dir}/tiled_loader_core_preflight.json"
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
    _require_success(staged, "private ownership source staging")
    for remote_path, payload, label in (
        (remote_complete, complete_payload, "complete"),
        (remote_loader_core, loader_core_payload, "loader-core"),
    ):
        completed = command_runner(
            build_ssh_command([
                "bash",
                "-c",
                f"cat > {shlex.quote(remote_path)}",
            ]),
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _require_success(
            completed,
            f"private ownership {label} prerequisite staging",
        )
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"complete=pathlib.Path({remote_complete!r})",
        f"loader_core=pathlib.Path({remote_loader_core!r})",
        f"names={list(SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "payload={'source':result,'complete':hashlib.sha256(complete.read_bytes()).hexdigest(),'loader_core':hashlib.sha256(loader_core.read_bytes()).hexdigest()}",
        "print(json.dumps(payload,sort_keys=True,separators=(',',':')))",
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
    _require_success(verified, "private ownership staged hashing")
    remote = json.loads(verified.stdout)
    if remote.get("source") != local_hashes:
        raise ValueError("private ownership remote source hash mismatch")
    if remote.get("complete") != COMPLETE_ARTIFACT_SHA256:
        raise ValueError("private ownership remote complete hash mismatch")
    if remote.get("loader_core") != LOADER_CORE_ARTIFACT_SHA256:
        raise ValueError(
            "private ownership remote loader-core hash mismatch"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "remote_complete_artifact": remote_complete,
        "remote_loader_core_artifact": remote_loader_core,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "complete_artifact_sha256": remote["complete"],
        "loader_core_artifact_sha256": remote["loader_core"],
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
        "complete_artifact_sha256": COMPLETE_ARTIFACT_SHA256,
        "loader_core_artifact_sha256": LOADER_CORE_ARTIFACT_SHA256,
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_private_candidate_ownership_preflight(record)
    return record


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "remote_complete_artifact": staged["remote_complete_artifact"],
        "remote_loader_core_artifact": (
            staged["remote_loader_core_artifact"]
        ),
        "complete_artifact_sha256": (
            staged["complete_artifact_sha256"]
        ),
        "loader_core_artifact_sha256": (
            staged["loader_core_artifact_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_private_candidate_ownership_preflight(
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
            f"local private ownership directory exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/private_candidate_ownership_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_checkpoint_private_candidate_ownership_preflight.py"
    )
    rows = []
    for tp_size, tp_rank, mode in WORKER_CONTEXTS:
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
                "--complete-artifact",
                staged["remote_complete_artifact"],
                "--loader-core-artifact",
                staged["remote_loader_core_artifact"],
                "--tp-size",
                str(tp_size),
                "--tp-rank",
                str(tp_rank),
                "--attempt-mode",
                mode,
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, "private ownership rank worker")
        row = json.loads(completed.stdout)
        validate_private_candidate_ownership_row(row)
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
    _require_success(finalized, "private ownership finalizer")
    record = json.loads(finalized.stdout)
    validate_private_candidate_ownership_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("private ownership source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'private_candidate_ownership_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'private_candidate_ownership_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "private_candidate_ownership_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "private ownership artifact round trip")
    if json.loads(round_trip.stdout) != {
        "private_candidate_ownership_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("private ownership artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "private_candidate_ownership_preflight.json",
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


def execute_remote_private_candidate_ownership_preflight(
    source_root,
    run_tag,
    *,
    complete_artifact,
    loader_core_artifact,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source_and_prerequisites(
        source_root,
        run_tag,
        complete_artifact=complete_artifact,
        loader_core_artifact=loader_core_artifact,
        command_runner=command_runner,
    )
    return run_remote_private_candidate_ownership_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments):
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not approved")
    row = run_private_candidate_ownership_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        complete_artifact=arguments.complete_artifact,
        loader_core_artifact=arguments.loader_core_artifact,
        tensor_parallel_size=arguments.tp_size,
        tensor_parallel_rank=arguments.tp_rank,
        mode=arguments.attempt_mode,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
        process_id=os.getpid(),
    )
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _finalize_main(arguments):
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("private ownership output already exists")
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
    run_parser.add_argument("--complete-artifact", required=True)
    run_parser.add_argument("--loader-core-artifact", required=True)
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--complete-artifact", required=True)
    worker_parser.add_argument("--loader-core-artifact", required=True)
    worker_parser.add_argument("--tp-size", type=int, required=True)
    worker_parser.add_argument("--tp-rank", type=int, required=True)
    worker_parser.add_argument(
        "--attempt-mode",
        choices=("success", "injected_failure"),
        required=True,
    )
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
        validate_private_candidate_ownership_preflight(record)
    else:
        record = execute_remote_private_candidate_ownership_preflight(
            arguments.source_root,
            arguments.run_tag,
            complete_artifact=arguments.complete_artifact,
            loader_core_artifact=arguments.loader_core_artifact,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
