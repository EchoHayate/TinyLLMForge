from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import gc
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
import weakref


def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ownership = _load_sibling(
    "_qwen35_private_ownership_publication_slot_base",
    "qwen35_real_checkpoint_private_candidate_ownership_preflight.py",
)


COMPLETE_ARTIFACT_SHA256 = (
    "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
)
OWNERSHIP_ARTIFACT_SHA256 = (
    "977a20a1986ade81e2b94063287cd15e6ece2adc3c818f3e0d9589f75b1adac4"
)
COMPLETE_SOURCE_TREE_SHA256 = (
    "da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042"
)
OWNERSHIP_SOURCE_TREE_SHA256 = (
    "91f9225a6ee214049002dc12bc7a669cdfa6a0d847b03e0cc107834f96f561a0"
)
PREREQUISITE_ROWS = ((1, 0), (2, 0), (2, 1))
SCHEMA_VERSION = (
    "qwen35.real-checkpoint-private-publication-slot.v1"
)
ROW_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-private-publication-slot-rank.v1"
)
PUBLICATION_MODULE_SHA256 = (
    "4ab2f928a3bbeeb632ca4180dcd496d56ac7716ac90d2a6adeb861f9c65d5b84"
)
REMOTE_TARGET = ownership.REMOTE_TARGET
REMOTE_PYTHON = ownership.REMOTE_PYTHON
APPROVED_MODEL_DIR = ownership.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = (
    ownership.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_CONFIG_SHA256 = ownership.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = ownership.APPROVED_INDEX_SHA256
APPROVED_COMPOSITE_SHA256 = ownership.APPROVED_COMPOSITE_SHA256
APPROVED_SHARD_NAME = ownership.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = ownership.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = ownership.APPROVED_SHARD_SHA256
AUTHORIZATION_SHA256 = ownership.AUTHORIZATION_SHA256
MAX_TENSOR_BYTES = ownership.MAX_TENSOR_BYTES
STREAMED_STATS = dict(ownership.STREAMED_STATS)
MEMORY_CEILINGS_KIB = dict(ownership.MEMORY_CEILINGS_KIB)
SOURCE_FILES = (
    *ownership.SOURCE_FILES,
    "tinyvllm/engine/qwen35_hybrid_model_publication.py",
    "tools/qwen35_real_checkpoint_private_publication_slot_preflight.py",
)
LOCAL_RUN_ROOT = ownership.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-private-publication-slot-runs"
)
WORKER_CONTEXTS = (
    (1, 0, "success"),
    (1, 0, "injected_post_publication_failure"),
    (2, 0, "success"),
    (2, 0, "injected_post_publication_failure"),
    (2, 1, "success"),
    (2, 1, "injected_post_publication_failure"),
)
_read_proc_status = ownership._read_proc_status
_memory_point = ownership._memory_point
_install_namespace_packages = ownership._install_namespace_packages
_source_tree_sha256 = ownership._source_tree_sha256
_atomic_write_json = ownership._atomic_write_json
validate_run_tag = ownership.validate_run_tag
build_ssh_command = ownership.build_ssh_command
_require_success = ownership._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_exact_json(path, expected_sha256):
    payload = Path(path).read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError(
            "private publication prerequisite hash is invalid"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "private publication prerequisite JSON is invalid"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            "private publication prerequisite must be a JSON object"
        )
    return value


def _row_map(rows):
    result = {}
    for row in rows:
        key = (row.get("tp_size"), row.get("tp_rank"))
        if key in result:
            raise ValueError(
                "duplicate private publication prerequisite row"
            )
        result[key] = MappingProxyType(row)
    if tuple(result) != PREREQUISITE_ROWS:
        raise ValueError(
            "private publication prerequisite TP rows are invalid"
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class PrivatePublicationPrerequisites:
    complete_artifact_sha256: str
    ownership_artifact_sha256: str
    complete_source_tree_sha256: str
    ownership_source_tree_sha256: str
    complete_rows: tuple[tuple[int, int], ...]
    ownership_rows: tuple[tuple[int, int], ...]
    complete_row_map: MappingProxyType
    ownership_row_map: MappingProxyType
    ownership_source_file_sha256: MappingProxyType


def load_private_publication_prerequisites(
    complete_artifact,
    ownership_artifact,
) -> PrivatePublicationPrerequisites:
    complete_record = ownership.loader_core.load_complete_gate_oracle(
        complete_artifact
    )
    ownership_record = _read_exact_json(
        ownership_artifact,
        OWNERSHIP_ARTIFACT_SHA256,
    )
    ownership.validate_private_candidate_ownership_preflight(
        ownership_record
    )
    if (
        complete_record["source_tree_sha256"]
        != COMPLETE_SOURCE_TREE_SHA256
    ):
        raise ValueError("complete prerequisite source tree is invalid")
    if (
        ownership_record["source_tree_sha256"]
        != OWNERSHIP_SOURCE_TREE_SHA256
    ):
        raise ValueError("ownership prerequisite source tree is invalid")
    complete_rows = _row_map(complete_record["rows"])
    ownership_success_rows = _row_map([
        row
        for row in ownership_record["rows"]
        if row["mode"] == "success"
    ])
    if len({row["process_id"] for row in ownership_record["rows"]}) != 6:
        raise ValueError("ownership prerequisite process IDs are invalid")
    source_hashes = ownership_record["source_file_sha256"]
    if not isinstance(source_hashes, dict) or len(source_hashes) != 45:
        raise ValueError(
            "ownership prerequisite source closure is invalid"
        )
    return PrivatePublicationPrerequisites(
        complete_artifact_sha256=COMPLETE_ARTIFACT_SHA256,
        ownership_artifact_sha256=OWNERSHIP_ARTIFACT_SHA256,
        complete_source_tree_sha256=COMPLETE_SOURCE_TREE_SHA256,
        ownership_source_tree_sha256=OWNERSHIP_SOURCE_TREE_SHA256,
        complete_rows=tuple(complete_rows),
        ownership_rows=tuple(ownership_success_rows),
        complete_row_map=complete_rows,
        ownership_row_map=ownership_success_rows,
        ownership_source_file_sha256=MappingProxyType(source_hashes),
    )


def select_private_publication_prerequisite_rows(
    prerequisites,
    tensor_parallel_size,
    tensor_parallel_rank,
):
    if type(prerequisites) is not PrivatePublicationPrerequisites:
        raise ValueError(
            "prerequisites must be exact PrivatePublicationPrerequisites"
        )
    key = (tensor_parallel_size, tensor_parallel_rank)
    try:
        return (
            prerequisites.complete_row_map[key],
            prerequisites.ownership_row_map[key],
        )
    except KeyError as error:
        raise ValueError(
            "private publication prerequisite TP row is invalid"
        ) from error


def _snapshot_pool(pool):
    if hasattr(pool, "_bindings") and hasattr(pool, "_tensors"):
        return ("production", ownership.loader_core._snapshot_pool(pool))
    return (
        "generic",
        tuple(
            sorted(
                (name, id(value), repr(value))
                for name, value in vars(pool).items()
            )
        ),
    )


def _pool_unchanged(pool, snapshot):
    kind, value = snapshot
    if kind == "production":
        return ownership.loader_core._pool_unchanged(pool, value)
    current = tuple(
        sorted(
            (name, id(item), repr(item))
            for name, item in vars(pool).items()
        )
    )
    return current == value


def execute_private_publication_scope(
    *,
    private_graph_factory,
    oracle_row,
    model_fingerprint,
    publication_slot_factory,
    expected_error_message=None,
):
    import torch

    if not callable(private_graph_factory):
        raise ValueError("private_graph_factory must be callable")
    if not callable(publication_slot_factory):
        raise ValueError("publication_slot_factory must be callable")

    def execute_nested_scope():
        target, acquire_candidate = private_graph_factory()
        if not callable(acquire_candidate):
            raise ValueError("acquire_candidate must be callable")
        model = target.assembly.packed.model
        pool = target.pool
        registered = ownership.loader_core._registered_tensors(model)
        identity = ownership.loader_core._snapshot_identity(registered)
        pool_snapshot = _snapshot_pool(pool)
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
                "private publication destination initialization failed"
            )
        slot = publication_slot_factory()
        slot_empty_before = (
            slot.candidate is None
            and slot.owner is None
            and slot.model_fingerprint is None
        )
        if not slot_empty_before:
            raise ValueError("private publication slot must start empty")
        target_consumed_before = target._consumed
        if target_consumed_before is not False:
            raise ValueError(
                "private publication target must be unconsumed"
            )
        publication_call_count = 0
        candidate = None
        owner = None
        error = None
        result = None
        references = None
        try:
            candidate = acquire_candidate()
            if target._consumed is not True:
                raise RuntimeError(
                    "private publication target was not consumed"
                )
            result = ownership.validate_private_loaded_candidate(
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
            publication_call_count += 1
            owner = slot.publish(candidate)
            if owner is not candidate.owner:
                raise ValueError(
                    "private publication returned owner is invalid"
                )
            if (
                slot.candidate is not candidate
                or slot.owner is not candidate.owner
                or slot.model_fingerprint != model_fingerprint
            ):
                raise ValueError(
                    "private publication slot visibility is invalid"
                )
            result.update({
                "slot_empty_before_publication": slot_empty_before,
                "publication_call_count": publication_call_count,
                "published_candidate_identity_verified": True,
                "published_owner_identity_verified": True,
                "published_fingerprint_verified": True,
                "target_consumed_before": target_consumed_before,
                "target_consumed_after": target._consumed,
                "selected_binding_count": len(
                    target.binding_plan.bindings
                ),
                "unique_destination_count": len(selected),
                "alias_groups": ownership.loader_core._alias_groups(
                    target.binding_plan
                ),
            })
            if expected_error_message is not None:
                raise RuntimeError(expected_error_message)
        except Exception as caught:
            error = caught
        finally:
            clear_error = None
            for tensor in reversed(tuple(selected.values())):
                try:
                    with torch.no_grad():
                        tensor.zero_()
                except Exception as caught:
                    if clear_error is None:
                        clear_error = caught
            try:
                ownership.loader_core._require_identity_unchanged(
                    registered,
                    identity,
                )
                if any(
                    int(tensor.count_nonzero().item())
                    for tensor in selected.values()
                ):
                    raise RuntimeError(
                        "private publication destination clear is incomplete"
                    )
                if any(
                    not tensor.equal(non_selected_values[id(tensor)])
                    for tensor in registered
                    if id(tensor) not in selected_ids
                ):
                    raise RuntimeError(
                        "private publication non-selected tensor changed"
                    )
                if not _pool_unchanged(pool, pool_snapshot):
                    raise RuntimeError(
                        "private publication pool state changed"
                    )
            except Exception as caught:
                if clear_error is None:
                    clear_error = caught
            if clear_error is not None:
                raise RuntimeError(
                    "private publication scope cleanup failed"
                ) from clear_error
            references = {
                "slot": weakref.ref(slot),
                "candidate": weakref.ref(candidate),
                "owner": weakref.ref(candidate.owner),
                "model": weakref.ref(model),
                "pool": weakref.ref(pool),
                "target": weakref.ref(target),
            }
        if error is not None:
            if (
                expected_error_message is None
                or str(error) != expected_error_message
            ):
                raise error
            result.update({
                "expected_failure_observed": True,
                "expected_failure_type": type(error).__name__,
                "expected_failure_message": str(error),
            })
        else:
            result["expected_failure_observed"] = False
        result.update({
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
        })
        return result, references

    result, references = execute_nested_scope()
    gc.collect()
    collected = {
        name: reference() is None
        for name, reference in references.items()
    }
    if not all(collected.values()):
        missing = sorted(
            name for name, value in collected.items() if not value
        )
        raise RuntimeError(
            "private publication objects escaped scope: "
            + ", ".join(missing)
        )
    result["collected_private_objects"] = collected
    result["all_private_publication_objects_collected"] = True
    return result


def _validate_memory(row):
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_load_publish_clear",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError("private publication memory points are invalid")
    observed = (
        row.get("total_vmhwm_increment_kib"),
        row.get("post_torch_vmhwm_increment_kib"),
        row.get("post_metadata_vmhwm_increment_kib"),
    )
    recomputed = (
        memory["after_load_publish_clear"]["vmhwm_kib"]
        - memory["before"]["vmhwm_kib"],
        memory["after_load_publish_clear"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
        memory["after_load_publish_clear"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    if observed != recomputed:
        raise ValueError("private publication memory deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    labels = ("total", "post_torch", "post_metadata")
    if any(
        value > ceilings[label]
        for value, label in zip(observed, labels)
    ):
        details = ", ".join(
            f"{label}={value}/{ceilings[label]} KiB"
            for value, label in zip(observed, labels)
        )
        raise ValueError(
            f"private publication memory ceiling exceeded: {details}"
        )


def validate_private_publication_slot_row(row):
    tp = (row.get("tp_size"), row.get("tp_rank"))
    mode = row.get("mode")
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or tp not in PREREQUISITE_ROWS
        or mode
        not in ("success", "injected_post_publication_failure")
    ):
        raise ValueError("private publication row schema is invalid")
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "complete_artifact_sha256": COMPLETE_ARTIFACT_SHA256,
        "ownership_artifact_sha256": OWNERSHIP_ARTIFACT_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": ownership.loader_core.ALIAS_GROUPS,
        "provider_call_count": 1,
        "adapter_call_count": 1,
        "publication_call_count": 1,
        "target_consumed_before": False,
        "target_consumed_after": True,
        "slot_empty_before_publication": True,
        "published_candidate_identity_verified": True,
        "published_owner_identity_verified": True,
        "published_fingerprint_verified": True,
        "all_selected_destinations_zero_after_clear": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "non_selected_tensors_unchanged": True,
        "all_private_publication_objects_collected": True,
        "collected_private_objects": {
            "slot": True,
            "candidate": True,
            "owner": True,
            "model": True,
            "pool": True,
            "target": True,
        },
        "candidate_installed": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "loaded_state_verified": True,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "loader_stats": STREAMED_STATS[tp],
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(f"private publication {name} is invalid")
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
            "private publication hash evidence is invalid"
        )
    if mode == "success":
        expected = {"expected_failure_observed": False}
    else:
        expected = {
            "expected_failure_observed": True,
            "expected_failure_type": "RuntimeError",
            "expected_failure_message": (
                "injected private publication-slot failure"
            ),
        }
    for name, value in expected.items():
        if row.get(name) != value:
            raise ValueError(f"private publication {name} is invalid")
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError("private publication process ID is invalid")
    _validate_memory(row)
    return row


def run_private_publication_slot_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    complete_artifact,
    ownership_artifact,
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
    prerequisites = load_private_publication_prerequisites(
        complete_artifact,
        ownership_artifact,
    )
    oracle_row, _ownership_row = (
        select_private_publication_prerequisite_rows(
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
    publication_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_model_publication",
        fromlist=["*"],
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    diagnostics = {
        "provider_call_count": 0,
        "adapter_call_count": 0,
        "attention_forward_count": 0,
    }

    def private_graph_factory():
        metadata = metadata_module.read_qwen35_checkpoint_metadata(
            checkpoint_dir,
            shards=(shard,),
            expected_config_sha256=APPROVED_CONFIG_SHA256,
            expected_index_sha256=APPROVED_INDEX_SHA256,
            expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
        )
        diagnostics["after_metadata"] = _memory_point(status_reader())
        diagnostics["metadata_bytes_read"] = (
            metadata.metadata_bytes_read
        )
        diagnostics["config_sha256"] = metadata.config_sha256
        diagnostics["index_sha256"] = metadata.index_sha256
        diagnostics["config_index_header_sha256"] = (
            metadata.config_index_header_sha256
        )
        tensor_plan = (
            checkpoint_module.build_qwen35_checkpoint_tensor_plan(
                metadata.hf_config,
                metadata.index_payload,
                metadata.shard_headers,
            )
        )
        layout = layout_module.build_qwen35_hybrid_state_layout(
            metadata.hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch.bfloat16,
            speculative_tokens=1,
        )
        pool = hybrid_module.HybridStateTensorPool(
            layout,
            capacity=1,
            device="cpu",
        )
        diagnostics["after_pool"] = _memory_point(status_reader())

        class _Backend(nn.Module):
            def forward(self, *_args, **_kwargs):
                diagnostics["attention_forward_count"] += 1
                raise AssertionError(
                    "attention backend must not execute"
                )

        target = (
            factory_module.prepare_qwen35_checkpoint_candidate_target(
                metadata.hf_config,
                tensor_plan,
                pool=pool,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
                build_attention_backend=lambda *_args: _Backend(),
                parameter_device="cpu",
            )
        )
        diagnostics["after_target"] = _memory_point(status_reader())

        def provide_target():
            diagnostics["provider_call_count"] += 1
            if diagnostics["provider_call_count"] != 1:
                raise RuntimeError(
                    "private publication target provider called twice"
                )
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

        def acquire_candidate():
            diagnostics["adapter_call_count"] += 1
            return adapter(request)

        return target, acquire_candidate

    result = execute_private_publication_scope(
        private_graph_factory=private_graph_factory,
        oracle_row=oracle_row,
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        publication_slot_factory=(
            lambda: publication_module
            .Qwen35HybridModelOwnerPublicationSlot()
        ),
        expected_error_message=(
            "injected private publication-slot failure"
            if mode == "injected_post_publication_failure"
            else None
        ),
    )
    after_load_publish_clear = _memory_point(status_reader())
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": diagnostics["after_metadata"],
        "after_pool": diagnostics["after_pool"],
        "after_target": diagnostics["after_target"],
        "after_load_publish_clear": after_load_publish_clear,
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
        "ownership_artifact_sha256": OWNERSHIP_ARTIFACT_SHA256,
        "config_sha256": diagnostics["config_sha256"],
        "index_sha256": diagnostics["index_sha256"],
        "config_index_header_sha256": (
            diagnostics["config_index_header_sha256"]
        ),
        "metadata_bytes_read": diagnostics["metadata_bytes_read"],
        "provider_call_count": diagnostics["provider_call_count"],
        "adapter_call_count": diagnostics["adapter_call_count"],
        "candidate_installed": False,
        "model_forward_count": 0,
        "attention_forward_count": (
            diagnostics["attention_forward_count"]
        ),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0,
            after_load_publish_clear["vmhwm_kib"]
            - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_load_publish_clear["vmhwm_kib"]
            - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_load_publish_clear["vmhwm_kib"]
            - diagnostics["after_metadata"]["vmhwm_kib"],
        ),
        **result,
    }
    validate_private_publication_slot_row(row)
    return row


def validate_private_publication_slot_preflight(record):
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
        "ownership_artifact_sha256": OWNERSHIP_ARTIFACT_SHA256,
        "fresh_process_per_attempt": True,
    }
    if any(record.get(name) != value for name, value in exact.items()):
        raise ValueError(
            "private publication preflight identity is invalid"
        )
    hashes = record.get("source_file_sha256")
    if set(hashes or {}) != set(SOURCE_FILES):
        raise ValueError(
            "private publication source hashes are invalid"
        )
    if (
        hashes["tinyvllm/engine/qwen35_hybrid_model_publication.py"]
        != PUBLICATION_MODULE_SHA256
    ):
        raise ValueError(
            "private publication module source hash is invalid"
        )
    if record.get("source_tree_sha256") != _source_tree_sha256(hashes):
        raise ValueError("private publication source tree is invalid")
    rows = record.get("rows")
    if [
        (
            row.get("tp_size"),
            row.get("tp_rank"),
            row.get("mode"),
        )
        for row in rows or ()
    ] != list(WORKER_CONTEXTS):
        raise ValueError(
            "private publication worker rows are invalid"
        )
    for row in rows:
        validate_private_publication_slot_row(row)
    if len({row["process_id"] for row in rows}) != 6:
        raise ValueError(
            "private publication process IDs must be unique"
        )
    return record


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(
                f"missing private publication source: {name}"
            )
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
                    f"missing private publication source: {name}"
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
    ownership_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    complete_payload = Path(complete_artifact).read_bytes()
    ownership_payload = Path(ownership_artifact).read_bytes()
    if _sha256(complete_payload) != COMPLETE_ARTIFACT_SHA256:
        raise ValueError("complete prerequisite artifact hash is invalid")
    if _sha256(ownership_payload) != OWNERSHIP_ARTIFACT_SHA256:
        raise ValueError(
            "ownership prerequisite artifact hash is invalid"
        )
    local_hashes = _source_hashes(source_root)
    prerequisites = load_private_publication_prerequisites(
        complete_artifact,
        ownership_artifact,
    )
    if {
        name: local_hashes[name]
        for name in ownership.SOURCE_FILES
    } != dict(prerequisites.ownership_source_file_sha256):
        raise ValueError(
            "private publication source does not match ownership "
            "prerequisite"
        )
    if (
        local_hashes[
            "tinyvllm/engine/qwen35_hybrid_model_publication.py"
        ]
        != PUBLICATION_MODULE_SHA256
    ):
        raise ValueError(
            "private publication production module was modified"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_complete = (
        f"{remote_run_dir}/complete_checkpoint_transaction_preflight.json"
    )
    remote_ownership = (
        f"{remote_run_dir}/private_candidate_ownership_preflight.json"
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
    _require_success(staged, "private publication source staging")
    for remote_path, payload, label in (
        (remote_complete, complete_payload, "complete"),
        (remote_ownership, ownership_payload, "ownership"),
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
            f"private publication {label} prerequisite staging",
        )
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"complete=pathlib.Path({remote_complete!r})",
        f"ownership=pathlib.Path({remote_ownership!r})",
        f"names={list(SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "payload={'source':result,'complete':hashlib.sha256(complete.read_bytes()).hexdigest(),'ownership':hashlib.sha256(ownership.read_bytes()).hexdigest()}",
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
    _require_success(verified, "private publication staged hashing")
    remote = json.loads(verified.stdout)
    if remote.get("source") != local_hashes:
        raise ValueError(
            "private publication remote source hash mismatch"
        )
    if remote.get("complete") != COMPLETE_ARTIFACT_SHA256:
        raise ValueError(
            "private publication remote complete hash mismatch"
        )
    if remote.get("ownership") != OWNERSHIP_ARTIFACT_SHA256:
        raise ValueError(
            "private publication remote ownership hash mismatch"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "remote_complete_artifact": remote_complete,
        "remote_ownership_artifact": remote_ownership,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "complete_artifact_sha256": remote["complete"],
        "ownership_artifact_sha256": remote["ownership"],
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
        "ownership_artifact_sha256": OWNERSHIP_ARTIFACT_SHA256,
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_private_publication_slot_preflight(record)
    return record


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "remote_complete_artifact": staged["remote_complete_artifact"],
        "remote_ownership_artifact": (
            staged["remote_ownership_artifact"]
        ),
        "complete_artifact_sha256": (
            staged["complete_artifact_sha256"]
        ),
        "ownership_artifact_sha256": (
            staged["ownership_artifact_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_private_publication_slot_preflight(
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
            f"local private publication directory exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/private_publication_slot_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_checkpoint_private_publication_slot_preflight.py"
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
                "--ownership-artifact",
                staged["remote_ownership_artifact"],
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
        _require_success(completed, "private publication rank worker")
        row = json.loads(completed.stdout)
        validate_private_publication_slot_row(row)
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
    _require_success(finalized, "private publication finalizer")
    record = json.loads(finalized.stdout)
    validate_private_publication_slot_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError(
            "private publication source binding mismatch"
        )
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'private_publication_slot_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'private_publication_slot_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "private_publication_slot_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(
        round_trip,
        "private publication artifact round trip",
    )
    if json.loads(round_trip.stdout) != {
        "private_publication_slot_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError(
            "private publication artifact round-trip mismatch"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "private_publication_slot_preflight.json",
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


def execute_remote_private_publication_slot_preflight(
    source_root,
    run_tag,
    *,
    complete_artifact,
    ownership_artifact,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source_and_prerequisites(
        source_root,
        run_tag,
        complete_artifact=complete_artifact,
        ownership_artifact=ownership_artifact,
        command_runner=command_runner,
    )
    return run_remote_private_publication_slot_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments):
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not approved")
    row = run_private_publication_slot_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        complete_artifact=arguments.complete_artifact,
        ownership_artifact=arguments.ownership_artifact,
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
        raise ValueError("private publication output already exists")
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
    run_parser.add_argument("--ownership-artifact", required=True)
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--complete-artifact", required=True)
    worker_parser.add_argument("--ownership-artifact", required=True)
    worker_parser.add_argument("--tp-size", type=int, required=True)
    worker_parser.add_argument("--tp-rank", type=int, required=True)
    worker_parser.add_argument(
        "--attempt-mode",
        choices=(
            "success",
            "injected_post_publication_failure",
        ),
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
        validate_private_publication_slot_preflight(record)
    else:
        record = execute_remote_private_publication_slot_preflight(
            arguments.source_root,
            arguments.run_tag,
            complete_artifact=arguments.complete_artifact,
            ownership_artifact=arguments.ownership_artifact,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
