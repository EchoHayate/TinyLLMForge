from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import importlib.util
import io
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Callable
import weakref


PROVENANCE = (
    "real-checkpoint-derived-live-concurrent-tp4-ownership"
)
CLAIM_BOUNDARY = "not-constructed-engine-runtime-binding"
READY_ROW_SCHEMA_VERSION = (
    "qwen35.tp4-live-concurrent-candidate-ready-rank.v1"
)
RELEASED_ROW_SCHEMA_VERSION = (
    "qwen35.tp4-live-concurrent-candidate-released-rank.v1"
)
PRIVATE_OBJECT_NAMES = (
    "runner",
    "production_slot",
    "request",
    "candidate",
    "owner",
    "runtime_bridge",
    "runtime_identity",
    "model",
    "pool",
    "target",
)
MEMORY_CEILINGS_KIB = {
    "per_worker_total_vmhwm_increment": 3145728,
    "aggregate_worker_vmhwm_increment": 12582912,
    "aggregate_ready_vmrss": 8388608,
    "host_mem_available_decrease": 12582912,
    "minimum_host_mem_available": 16777216,
}
def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


serial_gate = _load_sibling(
    "_qwen35_live_concurrent_serial_base",
    "qwen35_tp4_real_candidate_provenance_replay_preflight.py",
)
PREREQUISITE_ORACLE_SHA256 = (
    "d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef"
)
PREREQUISITE_ORACLE_NAME = (
    "tp4_real_candidate_provenance_oracle.json"
)
RESULT_NAME = "tp4_live_concurrent_candidate_ownership.json"
MANIFEST_NAME = "source_manifest.json"
SOURCE_FILES = tuple(sorted({
    *serial_gate.SOURCE_FILES,
    "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py",
}))
REMOTE_TARGET = serial_gate.REMOTE_TARGET
REMOTE_PYTHON = serial_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = serial_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-live-concurrent-candidate-ownership-runs"
)
APPROVED_MODEL_DIR = serial_gate.APPROVED_MODEL_DIR
_atomic_write_json = serial_gate._atomic_write_json
validate_run_tag = serial_gate.validate_run_tag
build_ssh_command = serial_gate.build_ssh_command
_require_success = serial_gate._require_success


@dataclass
class PreparedCandidateState:
    ready_payload: dict
    retained_objects: dict
    selected_objects: tuple
    release_graph: Callable[[], dict]


def prepare_real_retained_candidate_state(
    *,
    private_graph_factory,
    model_fingerprint,
    methods,
    bind_owner_method,
    bind_candidate_method,
    production_slot_factory,
    candidate_validator,
    payload_recorder,
    rank,
    status_reader,
):
    import torch

    helpers = (
        serial_gate.real_binding_gate.load_publish_gate.publication_gate
        .publication.ownership.loader_core
    )
    pool_helpers = (
        serial_gate.real_binding_gate.load_publish_gate.publication_gate
        .publication
    )
    target, request, installed_loader = private_graph_factory()
    model = target.assembly.packed.model
    pool = target.pool
    registered = helpers._registered_tensors(model)
    identity_snapshot = helpers._snapshot_identity(registered)
    pool_snapshot = pool_helpers._snapshot_pool(pool)
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
    production_slot = production_slot_factory()

    class RunnerShell:
        pass

    calls = {
        "adapter": 0,
        "owner": 0,
        "candidate": 0,
    }

    def counted_loader(value):
        calls["adapter"] += 1
        if calls["adapter"] != 1:
            raise RuntimeError("retained candidate loader called twice")
        return installed_loader(value)

    runner = RunnerShell()
    runner.rank = rank
    runner.model = model
    runner.qwen35_checkpoint_candidate_loader = counted_loader
    runner.qwen35_checkpoint_candidate_loader_authorization_sha256 = (
        request.authorization_sha256
    )
    runner.qwen35_checkpoint_candidate_load_configuration = None
    runner.qwen35_checkpoint_candidate_load_request = None
    runner.qwen35_loaded_checkpoint_candidate_slot = production_slot
    runner.qwen35_hybrid_model_owner = None
    runner.qwen35_hybrid_prefix_restore_owner = None
    runner.qwen35_hybrid_prefix_restore_participant = None
    runner.qwen35_hybrid_prefix_publication_participant = None
    runner.qwen35_hybrid_prefix_runtime_identity = None
    runner.qwen35_hybrid_prefix_runtime_identity_owner = None
    runner.hybrid_state_runtime_bridge = None

    def bind_owner(value):
        calls["owner"] += 1
        return bind_owner_method(runner, value)

    def bind_candidate(value):
        calls["candidate"] += 1
        return bind_candidate_method(runner, value)

    runner.bind_qwen35_hybrid_model_owner = bind_owner
    runner.bind_qwen35_loaded_checkpoint_candidate = bind_candidate
    load_row = methods[
        "load_and_publish_qwen35_checkpoint_candidate"
    ](runner, request)
    if load_row.get("status") != "published":
        raise ValueError("retained candidate publication failed")
    candidate = production_slot.candidate
    result = candidate_validator(
        candidate=candidate,
        target=target,
        model_fingerprint=model_fingerprint,
    )
    result.update(payload_recorder(
        candidate=candidate,
        target=target,
        model_fingerprint=model_fingerprint,
    ))
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
    result["alias_groups"] = helpers._alias_groups(
        target.binding_plan
    )
    method_row = methods[
        "bind_published_qwen35_loaded_checkpoint_candidate"
    ](runner)
    if (
        calls != {"adapter": 1, "owner": 1, "candidate": 1}
        or method_row.get("participant_id") != rank
        or method_row.get("status") != "bound"
    ):
        raise ValueError("retained candidate binding failed")
    ready_memory = dict(status_reader())
    ready_payload = {
        "tp_size": 4,
        "tp_rank": rank,
        "process_id": os.getpid(),
        "method_row": method_row,
        "binding_hash_count": result["binding_hash_count"],
        "binding_destination_sha256": result[
            "binding_destination_sha256"
        ],
        "phase_hash_count": result["phase_hash_count"],
        "phase_destination_sha256": result[
            "phase_destination_sha256"
        ],
        "aggregate_destination_sha256": result[
            "aggregate_destination_sha256"
        ],
        "alias_groups": result["alias_groups"],
        "loader_stats": result["loader_stats"],
        "cuda_initialized_before": False,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "ready_memory": ready_memory,
    }
    retained_objects = {
        "runner": runner,
        "production_slot": production_slot,
        "request": request,
        "candidate": candidate,
        "owner": candidate.owner,
        "runtime_bridge": candidate.owner.runtime_bridge,
        "runtime_identity": runner.qwen35_hybrid_prefix_runtime_identity,
        "model": model,
        "pool": pool,
        "target": target,
    }

    def release_graph():
        clear_error = None
        for tensor in reversed(tuple(selected.values())):
            try:
                with torch.no_grad():
                    tensor.zero_()
            except Exception as caught:
                if clear_error is None:
                    clear_error = caught
        try:
            helpers._require_identity_unchanged(
                registered,
                identity_snapshot,
            )
            if any(
                int(tensor.count_nonzero().item())
                for tensor in selected.values()
            ):
                raise RuntimeError("retained destination clear failed")
            if any(
                not tensor.equal(non_selected_values[id(tensor)])
                for tensor in registered
                if id(tensor) not in selected_ids
            ):
                raise RuntimeError("retained non-selected tensor changed")
            if not pool_helpers._pool_unchanged(pool, pool_snapshot):
                raise RuntimeError("retained pool state changed")
        except Exception as caught:
            if clear_error is None:
                clear_error = caught
        if clear_error is not None:
            raise RuntimeError(
                "retained candidate cleanup failed"
            ) from clear_error
        return {
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
        }

    return PreparedCandidateState(
        ready_payload=ready_payload,
        retained_objects=retained_objects,
        selected_objects=tuple(selected.values()),
        release_graph=release_graph,
    )


def _require_sha(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _validate_ready_payload(payload, rank):
    if not isinstance(payload, dict):
        raise ValueError("ready payload must be an object")
    if payload.get("tp_size") != 4 or payload.get("tp_rank") != rank:
        raise ValueError("ready TP identity is invalid")
    process_id = payload.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError("ready process identity is invalid")
    method_row = payload.get("method_row")
    if not isinstance(method_row, dict):
        raise ValueError("ready method row is invalid")
    if method_row.get("participant_id") != rank:
        raise ValueError("ready participant identity is invalid")
    exact_method = {
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound",
        "dtype": "bfloat16",
        "detail": "",
    }
    for name, expected in exact_method.items():
        if method_row.get(name) != expected:
            raise ValueError(f"ready method {name} is invalid")
    for name in ("model_fingerprint", "layout_fingerprint"):
        _require_sha(method_row.get(name), f"ready method {name}")
    if payload.get("binding_hash_count") != 320:
        raise ValueError("ready binding hash count is invalid")
    binding_hashes = payload.get("binding_destination_sha256")
    if (
        not isinstance(binding_hashes, list)
        or len(binding_hashes) != 320
    ):
        raise ValueError("ready binding payload is invalid")
    for index, digest in enumerate(binding_hashes):
        _require_sha(digest, f"ready binding payload {index}")
    if payload.get("phase_hash_count") != 26:
        raise ValueError("ready phase hash count is invalid")
    phase_hashes = payload.get("phase_destination_sha256")
    if (
        not isinstance(phase_hashes, dict)
        or len(phase_hashes) != 26
    ):
        raise ValueError("ready phase payload is invalid")
    for name, digest in phase_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("ready phase name is invalid")
        _require_sha(digest, f"ready phase payload {name}")
    _require_sha(
        payload.get("aggregate_destination_sha256"),
        "ready aggregate destination",
    )
    alias_groups = payload.get("alias_groups")
    if not isinstance(alias_groups, list):
        raise ValueError("ready alias groups are invalid")
    if payload.get("loader_stats") != {
        "assigned_bindings": 320,
        "source_tensors": 320,
        "shard_count": 1,
        "loaded_bytes": 3763655360,
        "peak_source_bytes": 1017118720,
    }:
        raise ValueError("ready loader statistics are invalid")
    for name in (
        "cuda_initialized_before",
        "cuda_initialized_after",
    ):
        if payload.get(name) is not False:
            raise ValueError(f"ready {name} must be false")
    for name in ("model_forward_count", "attention_forward_count"):
        if payload.get(name) != 0:
            raise ValueError(f"ready {name} must be zero")
    memory = payload.get("ready_memory")
    if (
        not isinstance(memory, dict)
        or set(memory) != {"vmrss_kib", "vmhwm_kib"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in memory.values()
        )
    ):
        raise ValueError("ready memory observation is invalid")


def validate_payload_against_pristine_rank(payload, pristine_row, *, rank):
    if (
        not isinstance(payload, dict)
        or not isinstance(pristine_row, dict)
        or pristine_row.get("tp_rank") != rank
    ):
        raise ValueError("pristine rank identity is invalid")
    for name, detail in (
        ("binding_destination_sha256", "binding payload"),
        ("phase_destination_sha256", "phase payload"),
        ("aggregate_destination_sha256", "aggregate payload"),
        ("alias_groups", "alias payload"),
    ):
        if payload.get(name) != pristine_row.get(name):
            raise ValueError(
                f"pristine rank {detail} mismatch: rank={rank}"
            )
    method = payload.get("method_row")
    if (
        not isinstance(method, dict)
        or method.get("model_fingerprint")
        != pristine_row.get("model_manifest_sha256")
        or method.get("layout_fingerprint")
        != pristine_row.get("layout_fingerprint")
        or method.get("dtype") != pristine_row.get("dtype")
    ):
        raise ValueError(
            f"pristine rank runtime identity mismatch: rank={rank}"
        )
    if payload.get("loader_stats") != pristine_row.get("loader_stats"):
        raise ValueError(
            f"pristine rank loader statistics mismatch: rank={rank}"
        )
    return payload


class RetainedCandidateScope:
    def __init__(self, *, rank, state):
        self._rank = rank
        self._state = state
        self._released = False
        self._ready = {
            "schema_version": READY_ROW_SCHEMA_VERSION,
            "status": "READY",
            "provenance": PROVENANCE,
            "claim_boundary": CLAIM_BOUNDARY,
            **state.ready_payload,
            "all_private_objects_retained": True,
            "retained_private_objects": {
                name: True for name in PRIVATE_OBJECT_NAMES
            },
        }

    def ready_row(self):
        if self._released:
            raise RuntimeError("retained candidate is already released")
        return dict(self._ready)

    def release(self):
        if self._released:
            raise RuntimeError("retained candidate is already released")
        self._released = True
        state = self._state
        references = {
            name: weakref.ref(state.retained_objects[name])
            for name in PRIVATE_OBJECT_NAMES
        }
        release_graph = state.release_graph
        cleanup = release_graph()
        if (
            not isinstance(cleanup, dict)
            or cleanup
            != {
                "all_selected_destinations_zero_after_clear": True,
                "non_selected_tensors_unchanged": True,
                "tensor_identity_preserved": True,
                "pool_unchanged": True,
            }
        ):
            raise RuntimeError(
                "retained candidate release invariants failed"
            )
        state.retained_objects.clear()
        self._state = None
        state = None
        release_graph = None
        gc.collect()
        collected = {
            name: reference() is None
            for name, reference in references.items()
        }
        if not all(collected.values()):
            escaped = sorted(
                name for name, value in collected.items() if not value
            )
            raise RuntimeError(
                "retained candidate objects escaped release: "
                + repr(escaped)
            )
        return {
            "schema_version": RELEASED_ROW_SCHEMA_VERSION,
            "status": "RELEASED",
            "provenance": PROVENANCE,
            "claim_boundary": CLAIM_BOUNDARY,
            "tp_size": 4,
            "tp_rank": self._rank,
            **cleanup,
            "collected_private_objects": collected,
            "all_private_objects_collected": True,
        }


def prepare_retained_tp4_candidate(*, rank, state_factory):
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank not in range(4)
    ):
        raise ValueError("rank must be a TP4 rank")
    if not callable(state_factory):
        raise ValueError("state_factory must be callable")
    state = state_factory()
    if not isinstance(state, PreparedCandidateState):
        raise ValueError(
            "state_factory must return PreparedCandidateState"
        )
    if set(state.retained_objects) != set(PRIVATE_OBJECT_NAMES):
        raise ValueError("retained private object set is invalid")
    if any(
        state.retained_objects[name] is None
        for name in PRIVATE_OBJECT_NAMES
    ):
        raise ValueError("retained private object is missing")
    def is_materialized(value):
        count_nonzero = getattr(value, "count_nonzero", None)
        if callable(count_nonzero):
            count = count_nonzero()
            item = getattr(count, "item", None)
            return int(item() if callable(item) else count) > 0
        return getattr(value, "value", 0) != 0

    if not state.selected_objects or any(
        not is_materialized(value)
        for value in state.selected_objects
    ):
        raise ValueError(
            "retained selected objects must remain materialized"
        )
    if not callable(state.release_graph):
        raise ValueError("release_graph must be callable")
    _validate_ready_payload(state.ready_payload, rank)
    return RetainedCandidateScope(rank=rank, state=state)


def prepare_real_retained_tp4_candidate(
    *,
    checkpoint_dir,
    source_root,
    tensor_parallel_size,
    tensor_parallel_rank,
    runtime_builder,
    retained_state_builder,
):
    if tensor_parallel_size != 4:
        raise ValueError("tensor_parallel_size must equal four")
    if (
        isinstance(tensor_parallel_rank, bool)
        or not isinstance(tensor_parallel_rank, int)
        or tensor_parallel_rank not in range(4)
    ):
        raise ValueError("tensor_parallel_rank must be a TP4 rank")
    for name, value in (
        ("runtime_builder", runtime_builder),
        ("retained_state_builder", retained_state_builder),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    runtime = runtime_builder(
        checkpoint_dir=checkpoint_dir,
        source_root=source_root,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
    )
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("scope_kwargs"), dict)
    ):
        raise ValueError("retained runtime is invalid")

    def state_factory():
        return retained_state_builder(
            **runtime["scope_kwargs"],
        )

    return prepare_retained_tp4_candidate(
        rank=tensor_parallel_rank,
        state_factory=state_factory,
    )


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def validate_ready_row(row, rank):
    if (
        not isinstance(row, dict)
        or row.get("schema_version") != READY_ROW_SCHEMA_VERSION
        or row.get("status") != "READY"
        or row.get("provenance") != PROVENANCE
        or row.get("claim_boundary") != CLAIM_BOUNDARY
        or row.get("all_private_objects_retained") is not True
        or row.get("retained_private_objects")
        != {name: True for name in PRIVATE_OBJECT_NAMES}
    ):
        raise ValueError("ready row contract is invalid")
    payload = {
        name: value
        for name, value in row.items()
        if name not in {
            "schema_version",
            "status",
            "provenance",
            "claim_boundary",
            "all_private_objects_retained",
            "retained_private_objects",
        }
    }
    _validate_ready_payload(payload, rank)
    return row


def validate_released_row(row, rank):
    exact = {
        "schema_version": RELEASED_ROW_SCHEMA_VERSION,
        "status": "RELEASED",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "collected_private_objects": {
            name: True for name in PRIVATE_OBJECT_NAMES
        },
        "all_private_objects_collected": True,
    }
    if row != exact:
        raise ValueError("released row contract is invalid")
    return row


def _validate_command(message, expected, rank):
    if message != {"command": expected, "rank": rank}:
        raise RuntimeError(
            f"retained candidate worker expected {expected}"
        )


def run_retained_candidate_worker(
    *,
    rank,
    channel,
    retained_factory,
    status_reader=None,
):
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank not in range(4)
    ):
        raise ValueError("worker rank must be a TP4 rank")
    if not all(
        callable(getattr(channel, name, None))
        for name in ("send", "recv")
    ):
        raise ValueError("worker channel contract is invalid")
    if not callable(retained_factory):
        raise ValueError("retained_factory must be callable")
    if status_reader is not None and not callable(status_reader):
        raise ValueError("status_reader must be callable")
    _validate_command(channel.recv(), "START", rank)
    before = dict(status_reader()) if status_reader is not None else None
    retained = retained_factory()
    if not isinstance(retained, RetainedCandidateScope):
        raise ValueError(
            "retained_factory must return RetainedCandidateScope"
        )
    ready = retained.ready_row()
    if before is not None:
        ready["memory"] = {
            "before": before,
            "ready": dict(ready["ready_memory"]),
        }
    validate_ready_row(ready, rank)
    channel.send({"event": "READY", "rank": rank, "row": ready})
    command = channel.recv()
    if command not in (
        {"command": "RELEASE", "rank": rank},
        {"command": "ABORT", "rank": rank},
    ):
        raise RuntimeError(
            "retained candidate worker expected RELEASE or ABORT"
        )
    released = retained.release()
    validate_released_row(released, rank)
    channel.send({
        "event": "RELEASED",
        "rank": rank,
        "row": released,
    })
    return 0


def _receive_event(channel, expected_event, rank, timeout_s):
    try:
        available = channel.poll(timeout_s)
    except TypeError:
        available = channel.poll()
    if not available:
        raise TimeoutError(
            f"{expected_event} acknowledgement timed out for rank {rank}"
        )
    message = channel.recv()
    if (
        not isinstance(message, dict)
        or message.get("event") != expected_event
        or message.get("rank") != rank
        or not isinstance(message.get("row"), dict)
    ):
        raise RuntimeError(
            f"{expected_event} acknowledgement is invalid for rank {rank}"
        )
    return message["row"]


def run_staggered_tp4_candidate_residency(
    *,
    worker_factory,
    process_status_reader,
    snapshot_writer,
    join_timeout_s,
    event_timeout_s=1800.0,
):
    for name, value in (
        ("worker_factory", worker_factory),
        ("process_status_reader", process_status_reader),
        ("snapshot_writer", snapshot_writer),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    if (
        isinstance(join_timeout_s, bool)
        or not isinstance(join_timeout_s, (int, float))
        or join_timeout_s <= 0
    ):
        raise ValueError("join_timeout_s must be positive")
    if (
        isinstance(event_timeout_s, bool)
        or not isinstance(event_timeout_s, (int, float))
        or event_timeout_s <= 0
    ):
        raise ValueError("event_timeout_s must be positive")
    workers = []
    process_ids = set()
    for rank in range(4):
        process, channel = worker_factory(rank)
        if not all(
            callable(getattr(process, name, None))
            for name in ("start", "join", "is_alive")
        ):
            raise ValueError("worker process contract is invalid")
        if not all(
            callable(getattr(channel, name, None))
            for name in ("send", "recv", "poll")
        ):
            raise ValueError("worker channel contract is invalid")
        workers.append((rank, process, channel))
    for _, process, _ in workers:
        process.start()
        process_id = getattr(process, "pid", None)
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id <= 0
            or process_id in process_ids
        ):
            raise ValueError("worker PID is invalid")
        process_ids.add(process_id)

    try:
        ready_rows = []
        start_order = []
        ready_order = []
        for rank, process, channel in workers:
            if not process.is_alive():
                raise RuntimeError(
                    f"rank {rank} exited before START"
                )
            channel.send({"command": "START", "rank": rank})
            start_order.append(rank)
            row = _receive_event(
                channel,
                "READY",
                rank,
                event_timeout_s,
            )
            validate_ready_row(row, rank)
            ready_rows.append(row)
            ready_order.append(rank)
            if any(
                not prior_process.is_alive()
                for _, prior_process, _ in workers[: rank + 1]
            ):
                raise RuntimeError(
                    "ready worker exited before concurrent snapshot"
                )

        if any(channel.poll() for _, _, channel in workers):
            raise RuntimeError("worker released before concurrent snapshot")
        process_status = []
        for rank, process, _ in workers:
            if not process.is_alive():
                raise RuntimeError(
                    "worker exited before concurrent snapshot"
                )
            status = process_status_reader(process.pid)
            if (
                not isinstance(status, dict)
                or set(status) != {"state", "vmrss_kib", "vmhwm_kib"}
                or not isinstance(status["state"], str)
                or not status["state"]
                or any(
                    isinstance(status[name], bool)
                    or not isinstance(status[name], int)
                    or status[name] <= 0
                    for name in ("vmrss_kib", "vmhwm_kib")
                )
            ):
                raise ValueError(
                    f"worker process status is invalid: rank={rank}"
                )
            process_status.append({
                "rank": rank,
                "process_id": process.pid,
                **status,
            })
            ready_rows[rank]["ready_memory"] = {
                "vmrss_kib": status["vmrss_kib"],
                "vmhwm_kib": status["vmhwm_kib"],
            }
            memory = ready_rows[rank].get("memory")
            if isinstance(memory, dict) and "before" in memory:
                ready_rows[rank]["memory"] = {
                    "before": dict(memory["before"]),
                    "ready": dict(ready_rows[rank]["ready_memory"]),
                }
        snapshot = {
            "schema_version": (
                "qwen35.tp4-live-concurrent-candidate-snapshot.v1"
            ),
            "coordinator_process_id": os.getpid(),
            "snapshot_unix_time_ns": time.time_ns(),
            "start_order": list(start_order),
            "ready_order": list(ready_order),
            "live_process_ids": [
                process.pid for _, process, _ in workers
            ],
            "ready_row_sha256": [
                hashlib.sha256(_canonical(row)).hexdigest()
                for row in ready_rows
            ],
            "process_status": process_status,
            "release_acknowledgement_count": 0,
            "all_workers_live_concurrently": True,
        }
        snapshot_writer(snapshot)

        release_order = []
        released_order = []
        released_rows = []
        for rank, process, channel in reversed(workers):
            if not process.is_alive():
                raise RuntimeError(
                    f"rank {rank} exited before RELEASE"
                )
            channel.send({"command": "RELEASE", "rank": rank})
            release_order.append(rank)
            row = _receive_event(
                channel,
                "RELEASED",
                rank,
                event_timeout_s,
            )
            validate_released_row(row, rank)
            released_rows.append(row)
            released_order.append(rank)
        for _, process, _ in workers:
            process.join(timeout=join_timeout_s)
            if (
                process.is_alive()
                or getattr(process, "exitcode", None) != 0
            ):
                raise RuntimeError("worker failed to exit cleanly")
    except Exception:
        for rank, process, channel in reversed(workers):
            if process.is_alive():
                channel.send({"command": "ABORT", "rank": rank})
                if channel.poll():
                    row = _receive_event(
                        channel,
                        "RELEASED",
                        rank,
                        event_timeout_s,
                    )
                    validate_released_row(row, rank)
        for _, process, _ in workers:
            if process.is_alive():
                process.join(timeout=join_timeout_s)
        raise
    return {
        "ready_rows": ready_rows,
        "concurrent_snapshot": snapshot,
        "released_rows": released_rows,
        "start_order": start_order,
        "ready_order": ready_order,
        "release_order": release_order,
        "released_order": released_order,
        "all_workers_exited": True,
    }


def _memory_point(value, context):
    if (
        not isinstance(value, dict)
        or set(value) != {"vmrss_kib", "vmhwm_kib"}
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in value.values()
        )
    ):
        raise ValueError(f"{context} memory point is invalid")
    return value


def _host_memory(value, context):
    if (
        not isinstance(value, dict)
        or set(value)
        != {"mem_available_kib", "swap_total_kib", "swap_free_kib"}
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in value.values()
        )
    ):
        raise ValueError(f"{context} host memory is invalid")
    return value


def validate_aggregate_memory_contract(evidence):
    if not isinstance(evidence, dict):
        raise ValueError("memory evidence must be an object")
    before_host = _host_memory(
        evidence.get("host_memory_before"),
        "before",
    )
    ready_host = _host_memory(
        evidence.get("host_memory_ready"),
        "ready",
    )
    minimum = MEMORY_CEILINGS_KIB["minimum_host_mem_available"]
    if before_host["mem_available_kib"] < minimum:
        raise ValueError(
            "host MemAvailable preflight is below "
            f"{minimum} KiB: actual={before_host['mem_available_kib']}"
        )
    if (
        ready_host["swap_total_kib"] != before_host["swap_total_kib"]
        or ready_host["swap_free_kib"] != before_host["swap_free_kib"]
    ):
        raise ValueError("swap changed during concurrent residency")
    rows = evidence.get("ready_rows")
    snapshot = evidence.get("concurrent_snapshot")
    if (
        not isinstance(rows, list)
        or len(rows) != 4
        or not isinstance(snapshot, dict)
        or snapshot.get("all_workers_live_concurrently") is not True
        or snapshot.get("release_acknowledgement_count") != 0
    ):
        raise ValueError("concurrent memory evidence is incomplete")
    statuses = snapshot.get("process_status")
    if not isinstance(statuses, list) or len(statuses) != 4:
        raise ValueError("concurrent process status is incomplete")
    status_by_rank = {
        status.get("rank"): status
        for status in statuses
        if isinstance(status, dict)
    }
    if set(status_by_rank) != set(range(4)):
        raise ValueError("concurrent process ranks are invalid")
    increments = []
    ready_rss = []
    for rank, row in enumerate(rows):
        validate_ready_row(row, rank)
        memory = row.get("memory")
        if not isinstance(memory, dict) or set(memory) != {
            "before",
            "ready",
        }:
            raise ValueError(
                f"worker memory points are invalid: rank={rank}"
            )
        before = _memory_point(memory["before"], f"rank={rank} before")
        ready = _memory_point(memory["ready"], f"rank={rank} ready")
        increment = ready["vmhwm_kib"] - before["vmhwm_kib"]
        if increment < 0:
            raise ValueError(
                f"worker VmHWM delta is negative: rank={rank}"
            )
        ceiling = MEMORY_CEILINGS_KIB[
            "per_worker_total_vmhwm_increment"
        ]
        if increment > ceiling:
            raise ValueError(
                "worker VmHWM ceiling exceeded: "
                f"rank={rank}, actual={increment}, allowed={ceiling}"
            )
        status = status_by_rank[rank]
        if (
            status.get("process_id") != row["process_id"]
            or status.get("vmrss_kib") != ready["vmrss_kib"]
            or status.get("vmhwm_kib") != ready["vmhwm_kib"]
        ):
            raise ValueError(
                f"worker /proc memory mismatch: rank={rank}"
            )
        increments.append(increment)
        ready_rss.append(ready["vmrss_kib"])
    aggregate_increment = sum(increments)
    aggregate_increment_ceiling = MEMORY_CEILINGS_KIB[
        "aggregate_worker_vmhwm_increment"
    ]
    if aggregate_increment > aggregate_increment_ceiling:
        raise ValueError(
            "aggregate worker VmHWM ceiling exceeded: "
            f"actual={aggregate_increment}, "
            f"allowed={aggregate_increment_ceiling}"
        )
    aggregate_rss = sum(ready_rss)
    aggregate_rss_ceiling = MEMORY_CEILINGS_KIB[
        "aggregate_ready_vmrss"
    ]
    if aggregate_rss > aggregate_rss_ceiling:
        raise ValueError(
            "aggregate ready VmRSS ceiling exceeded: "
            f"actual={aggregate_rss}, allowed={aggregate_rss_ceiling}"
        )
    available_decrease = (
        before_host["mem_available_kib"]
        - ready_host["mem_available_kib"]
    )
    if available_decrease < 0:
        raise ValueError("host MemAvailable decrease is negative")
    available_ceiling = MEMORY_CEILINGS_KIB[
        "host_mem_available_decrease"
    ]
    if available_decrease > available_ceiling:
        raise ValueError(
            "host MemAvailable decrease ceiling exceeded: "
            f"actual={available_decrease}, allowed={available_ceiling}"
        )
    return {
        "per_worker_total_vmhwm_increment_kib": increments,
        "aggregate_worker_vmhwm_increment_kib": aggregate_increment,
        "aggregate_ready_vmrss_kib": aggregate_rss,
        "host_mem_available_decrease_kib": available_decrease,
        "memory_contract_passed": True,
    }


def build_ownership_artifact(
    *,
    ready_rows,
    concurrent_snapshot,
    released_rows,
    host_memory_before,
    host_memory_ready,
    source_file_sha256,
    source_tree_sha256,
    prerequisite_oracle_sha256,
):
    if (
        not isinstance(ready_rows, list)
        or len(ready_rows) != 4
        or not isinstance(released_rows, list)
        or len(released_rows) != 4
    ):
        raise ValueError("ownership rows are incomplete")
    for rank, row in enumerate(ready_rows):
        validate_ready_row(row, rank)
    if [row.get("tp_rank") for row in released_rows] != [3, 2, 1, 0]:
        raise ValueError("release row order is invalid")
    for row in released_rows:
        validate_released_row(row, row["tp_rank"])
    if (
        not isinstance(concurrent_snapshot, dict)
        or concurrent_snapshot.get("schema_version")
        != "qwen35.tp4-live-concurrent-candidate-snapshot.v1"
        or concurrent_snapshot.get("start_order") != [0, 1, 2, 3]
        or concurrent_snapshot.get("ready_order") != [0, 1, 2, 3]
        or concurrent_snapshot.get(
            "all_workers_live_concurrently"
        ) is not True
        or concurrent_snapshot.get("release_acknowledgement_count") != 0
    ):
        raise ValueError("concurrent snapshot contract is invalid")
    expected_ready_hashes = [
        _sha256(_canonical(row)) for row in ready_rows
    ]
    if concurrent_snapshot.get("ready_row_sha256") != expected_ready_hashes:
        raise ValueError("concurrent snapshot ready hashes are invalid")
    process_ids = [row["process_id"] for row in ready_rows]
    if (
        len(set(process_ids)) != 4
        or concurrent_snapshot.get("live_process_ids") != process_ids
    ):
        raise ValueError("concurrent snapshot process IDs are invalid")
    for name, value in (
        ("source_tree_sha256", source_tree_sha256),
        ("prerequisite_oracle_sha256", prerequisite_oracle_sha256),
    ):
        _require_sha(value, name)
    if (
        not isinstance(source_file_sha256, dict)
        or not source_file_sha256
        or tuple(source_file_sha256) != tuple(sorted(source_file_sha256))
    ):
        raise ValueError("source file hashes are invalid")
    for name, value in source_file_sha256.items():
        if not isinstance(name, str) or not name:
            raise ValueError("source filename is invalid")
        _require_sha(value, f"source file {name}")
    memory_evidence = {
        "host_memory_before": host_memory_before,
        "host_memory_ready": host_memory_ready,
        "ready_rows": ready_rows,
        "concurrent_snapshot": concurrent_snapshot,
    }
    memory_summary = validate_aggregate_memory_contract(memory_evidence)
    return {
        "schema_version": (
            "qwen35.tp4-live-concurrent-candidate-ownership.v1"
        ),
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "prerequisite_oracle_sha256": prerequisite_oracle_sha256,
        "source_file_sha256": dict(source_file_sha256),
        "source_tree_sha256": source_tree_sha256,
        "start_order": [0, 1, 2, 3],
        "ready_order": [0, 1, 2, 3],
        "release_order": [3, 2, 1, 0],
        "released_order": [3, 2, 1, 0],
        "ready_rows_sha256": _sha256(_canonical(ready_rows)),
        "released_rows_sha256": _sha256(_canonical(released_rows)),
        "ready_rows": ready_rows,
        "concurrent_snapshot": concurrent_snapshot,
        "released_rows": released_rows,
        "host_memory_before": host_memory_before,
        "host_memory_ready": host_memory_ready,
        "memory_summary": memory_summary,
        "all_workers_exited": True,
    }


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError("missing live-concurrent source: " + name)
        hashes[name] = _sha256(path.read_bytes())
    return dict(sorted(hashes.items()))


def _source_tree_sha256(hashes):
    return _sha256(_canonical(dict(hashes)))


def _read_self_memory():
    return serial_gate.load_publish_gate._memory_point(
        serial_gate.load_publish_gate._read_proc_status()
    )


def _read_process_status(process_id):
    path = Path(f"/proc/{process_id}/status")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.strip().split()
        if name == "State" and parts:
            values["state"] = parts[0]
        elif name in ("VmRSS", "VmHWM") and parts:
            values[name.lower() + "_kib"] = int(parts[0])
    if set(values) != {"state", "vmrss_kib", "vmhwm_kib"}:
        raise ValueError(
            f"worker process status is incomplete: pid={process_id}"
        )
    return values


def _read_host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text(
        encoding="utf-8"
    ).splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        parts = raw.strip().split()
        if name in ("MemAvailable", "SwapTotal", "SwapFree") and parts:
            values[name] = int(parts[0])
    if set(values) != {"MemAvailable", "SwapTotal", "SwapFree"}:
        raise ValueError("host memory observation is incomplete")
    return {
        "mem_available_kib": values["MemAvailable"],
        "swap_total_kib": values["SwapTotal"],
        "swap_free_kib": values["SwapFree"],
    }


def _load_prerequisite_oracle(path):
    payload = Path(path).read_bytes()
    if _sha256(payload) != PREREQUISITE_ORACLE_SHA256:
        raise ValueError("prerequisite oracle hash is invalid")
    record = json.loads(payload)
    serial_gate.validate_tp4_real_candidate_provenance_oracle(record)
    return record


def _build_real_retained_candidate(
    *,
    rank,
    source_root,
    checkpoint_dir,
    status_reader,
):
    serial_gate._install_namespace_packages(source_root)
    torch_runtime = serial_gate._load_runtime_module("torch")
    torch_runtime.set_num_threads(8)

    def backend_factory(diagnostics):
        class Backend(torch_runtime.nn.Module):
            def forward(self, *_args, **_kwargs):
                diagnostics["attention_forward_count"] += 1
                raise AssertionError(
                    "attention backend must not execute"
                )

        return Backend()

    def runtime_builder(**kwargs):
        components = (
            serial_gate.build_tp4_real_candidate_producer_components(
                source_root=kwargs["source_root"],
                module_loader=serial_gate._load_runtime_module,
                torch_runtime=torch_runtime,
                backend_factory=backend_factory,
            )
        )
        return serial_gate.assemble_tp4_real_candidate_producer_runtime(
            checkpoint_dir=kwargs["checkpoint_dir"],
            tensor_parallel_size=kwargs["tensor_parallel_size"],
            tensor_parallel_rank=kwargs["tensor_parallel_rank"],
            status_reader=status_reader,
            components=components,
        )

    return prepare_real_retained_tp4_candidate(
        checkpoint_dir=checkpoint_dir,
        source_root=source_root,
        tensor_parallel_size=4,
        tensor_parallel_rank=rank,
        runtime_builder=runtime_builder,
        retained_state_builder=lambda **scope_kwargs: (
            prepare_real_retained_candidate_state(
                status_reader=status_reader,
                **scope_kwargs,
            )
        ),
    )


def run_real_retained_candidate_worker(
    *,
    rank,
    channel,
    source_root,
    checkpoint_dir,
    pristine_row,
):
    def retained_factory():
        retained = _build_real_retained_candidate(
            rank=rank,
            source_root=source_root,
            checkpoint_dir=checkpoint_dir,
            status_reader=_read_self_memory,
        )
        validate_payload_against_pristine_rank(
            retained.ready_row(),
            pristine_row,
            rank=rank,
        )
        return retained

    return run_retained_candidate_worker(
        rank=rank,
        channel=channel,
        status_reader=_read_self_memory,
        retained_factory=retained_factory,
    )


def run_source_bound_live_concurrent_gate(
    *,
    source_root,
    checkpoint_dir,
    prerequisite_oracle,
    output_path,
    manifest_path,
    run_tag,
    join_timeout_s=30.0,
    event_timeout_s=1800.0,
):
    validate_run_tag(run_tag)
    checkpoint_dir = os.fspath(Path(checkpoint_dir).resolve())
    if checkpoint_dir != APPROVED_MODEL_DIR:
        raise ValueError("live-concurrent checkpoint is invalid")
    oracle = _load_prerequisite_oracle(prerequisite_oracle)
    pristine_rows = {
        row["tp_rank"]: row for row in oracle["producer_rows"]
    }
    if set(pristine_rows) != set(range(4)):
        raise ValueError("pristine oracle ranks are incomplete")
    source_hashes = _source_hashes(source_root)
    source_tree = _source_tree_sha256(source_hashes)
    host_before = _read_host_memory()
    if (
        host_before["mem_available_kib"]
        < MEMORY_CEILINGS_KIB["minimum_host_mem_available"]
    ):
        raise RuntimeError(
            "host MemAvailable preflight is below "
            f"{MEMORY_CEILINGS_KIB['minimum_host_mem_available']} KiB"
        )
    context = multiprocessing.get_context("fork")

    def worker_factory(rank):
        coordinator_channel, worker_channel = context.Pipe(duplex=True)
        process = context.Process(
            target=run_real_retained_candidate_worker,
            kwargs={
                "rank": rank,
                "channel": worker_channel,
                "source_root": source_root,
                "checkpoint_dir": checkpoint_dir,
                "pristine_row": pristine_rows[rank],
            },
            name=f"qwen35-live-candidate-rank{rank}",
        )
        return process, coordinator_channel

    snapshots = []
    host_ready_observations = []

    def snapshot_writer(snapshot):
        host_ready_observations.append(_read_host_memory())
        snapshots.append(snapshot)

    residency = run_staggered_tp4_candidate_residency(
        worker_factory=worker_factory,
        process_status_reader=_read_process_status,
        snapshot_writer=snapshot_writer,
        join_timeout_s=join_timeout_s,
        event_timeout_s=event_timeout_s,
    )
    if len(snapshots) != 1 or len(host_ready_observations) != 1:
        raise RuntimeError("concurrent snapshot was not captured exactly once")
    host_ready = host_ready_observations[0]
    process_ids = [
        row["process_id"] for row in residency["ready_rows"]
    ]
    residual = [
        process_id
        for process_id in process_ids
        if Path(f"/proc/{process_id}").exists()
    ]
    if residual:
        raise RuntimeError(
            "worker process remained after release: " + repr(residual)
        )
    record = build_ownership_artifact(
        ready_rows=residency["ready_rows"],
        concurrent_snapshot=residency["concurrent_snapshot"],
        released_rows=residency["released_rows"],
        host_memory_before=host_before,
        host_memory_ready=host_ready,
        source_file_sha256=source_hashes,
        source_tree_sha256=source_tree,
        prerequisite_oracle_sha256=PREREQUISITE_ORACLE_SHA256,
    )
    record["worker_process_ids"] = process_ids
    record["residual_worker_process_ids"] = residual
    record["all_worker_process_ids_absent"] = True
    output = Path(output_path)
    manifest = Path(manifest_path)
    if output.exists() or manifest.exists():
        raise ValueError("live-concurrent output already exists")
    _atomic_write_json(output, record)
    source_manifest = {
        "schema_version": (
            "qwen35.tp4-live-concurrent-candidate-ownership.v1"
        ),
        "run_tag": run_tag,
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "source_file_sha256": source_hashes,
        "source_tree_sha256": source_tree,
        "prerequisite_oracle_sha256": PREREQUISITE_ORACLE_SHA256,
        "result_sha256": _sha256(output.read_bytes()),
    }
    _atomic_write_json(manifest, source_manifest)
    return record, source_manifest


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(
                    "missing live-concurrent source: " + name
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def stage_live_concurrent_remote_run(
    source_root,
    run_tag,
    *,
    prerequisite_oracle,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    _load_prerequisite_oracle(prerequisite_oracle)
    local_hashes = _source_hashes(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
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
    _require_success(staged, "live-concurrent source staging")
    remote_oracle = f"{remote_run_dir}/{PREREQUISITE_ORACLE_NAME}"
    uploaded = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            f"cat > {shlex.quote(remote_oracle)}",
        ]),
        input=Path(prerequisite_oracle).read_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(uploaded, "live-concurrent oracle staging")
    return {
        "remote_run_dir": remote_run_dir,
        "remote_source_dir": remote_source_dir,
        "remote_oracle": remote_oracle,
        "local_file_sha256": local_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def run_remote_live_concurrent_gate(
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
            f"local live-concurrent directory exists: {destination}"
        )
    remote_run_dir = staged["remote_run_dir"]
    remote_source_dir = staged["remote_source_dir"]
    remote_worker = (
        f"{remote_source_dir}/tools/"
        "qwen35_tp4_live_concurrent_candidate_ownership_preflight.py"
    )
    remote_result = f"{remote_run_dir}/{RESULT_NAME}"
    remote_manifest = f"{remote_run_dir}/{MANIFEST_NAME}"
    completed = command_runner(
        build_ssh_command([
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=8",
            "MKL_NUM_THREADS=8",
            REMOTE_PYTHON,
            "-B",
            remote_worker,
            "internal-finalize",
            "--source-root",
            remote_source_dir,
            "--checkpoint-dir",
            APPROVED_MODEL_DIR,
            "--prerequisite-oracle",
            staged["remote_oracle"],
            "--output",
            remote_result,
            "--manifest",
            remote_manifest,
            "--run-tag",
            run_tag,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(completed, "live-concurrent remote gate")
    payload = json.loads(completed.stdout)
    record = payload["result"]
    manifest = payload["manifest"]
    if (
        record.get("source_file_sha256")
        != staged["local_file_sha256"]
        or record.get("source_tree_sha256")
        != staged["source_tree_sha256"]
        or manifest.get("result_sha256")
        != _sha256(_canonical(record) + b"\n")
    ):
        raise ValueError("live-concurrent remote source binding mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(temporary / RESULT_NAME, record)
        _atomic_write_json(temporary / MANIFEST_NAME, manifest)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    return record


def execute_remote_live_concurrent_gate(
    source_root,
    run_tag,
    *,
    prerequisite_oracle,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_live_concurrent_remote_run(
        source_root,
        run_tag,
        prerequisite_oracle=prerequisite_oracle,
        command_runner=command_runner,
    )
    return run_remote_live_concurrent_gate(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run = subparsers.add_parser("run")
    run.add_argument("--run-tag", required=True)
    run.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    run.add_argument("--prerequisite-oracle", required=True)
    worker = subparsers.add_parser("internal-worker")
    worker.add_argument("--rank", type=int, required=True)
    worker.add_argument("--source-root", required=True)
    worker.add_argument("--checkpoint-dir", required=True)
    worker.add_argument("--channel-fd", type=int, required=True)
    worker.add_argument("--pristine-row", required=True)
    finalizer = subparsers.add_parser("internal-finalize")
    finalizer.add_argument("--source-root", required=True)
    finalizer.add_argument("--checkpoint-dir", required=True)
    finalizer.add_argument("--prerequisite-oracle", required=True)
    finalizer.add_argument("--output", required=True)
    finalizer.add_argument("--manifest", required=True)
    finalizer.add_argument("--run-tag", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--result", required=True)
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--source-root", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        record = execute_remote_live_concurrent_gate(
            arguments.source_root,
            arguments.run_tag,
            prerequisite_oracle=arguments.prerequisite_oracle,
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-worker":
        from multiprocessing.connection import Connection

        pristine_row = json.loads(
            Path(arguments.pristine_row).read_bytes()
        )
        channel = Connection(arguments.channel_fd)
        try:
            return run_real_retained_candidate_worker(
                rank=arguments.rank,
                channel=channel,
                source_root=arguments.source_root,
                checkpoint_dir=arguments.checkpoint_dir,
                pristine_row=pristine_row,
            )
        finally:
            channel.close()
    if arguments.command == "internal-finalize":
        record, manifest = run_source_bound_live_concurrent_gate(
            source_root=arguments.source_root,
            checkpoint_dir=arguments.checkpoint_dir,
            prerequisite_oracle=arguments.prerequisite_oracle,
            output_path=arguments.output,
            manifest_path=arguments.manifest,
            run_tag=arguments.run_tag,
        )
        print(json.dumps(
            {"result": record, "manifest": manifest},
            sort_keys=True,
            separators=(",", ":"),
        ))
        return 0
    if arguments.command == "validate":
        record = json.loads(Path(arguments.result).read_bytes())
        manifest = json.loads(Path(arguments.manifest).read_bytes())
        if (
            record.get("source_file_sha256")
            != _source_hashes(arguments.source_root)
            or manifest.get("result_sha256")
            != _sha256(Path(arguments.result).read_bytes())
        ):
            raise ValueError("live-concurrent validation failed")
        print("Qwen3.5 TP4 live-concurrent ownership validated")
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
