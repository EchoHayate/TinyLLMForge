from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from dataclasses import dataclass
import getpass
import hashlib
import importlib.util
import io
from itertools import count
import json
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from types import MappingProxyType
import uuid


def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tp4_gate = _load_sibling(
    "_qwen35_tp4_synthetic_binding_transport",
    "qwen35_tp4_shared_memory_fanout_preflight.py",
)
ack_module = tp4_gate.ack_module


TP4_ARTIFACT_SHA256 = (
    "ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a"
)
TP4_SOURCE_TREE_SHA256 = (
    "ec7b0dee43a06c47b72f8ac14ab26518845f57f070e6c27d394bb4c328644403"
)
ORACLE_ARTIFACT_SHA256 = (
    "1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e"
)
MODEL_FINGERPRINT = (
    "b48e29c9ea266197c56fe3e9133c65d378c682e9437d9e1299c4fa93d0241bd9"
)
LAYOUT_FINGERPRINT = (
    "fe2db881dc909cbd05894a4e194273f4692caed1c4df967e810330324248e2c6"
)
ALTERNATE_MODEL_FINGERPRINT = (
    "bf1a56f2bf513a7a8946c4245a5cf3ab53d1853d4ecba9457e88a1636109a206"
)
ALTERNATE_LAYOUT_FINGERPRINT = (
    "9ccdb4f0c9c2ad4003d1539bf7ba76188ce59686cd302216371b301e031dc1a9"
)
SOURCE_FILES = (
    *tp4_gate.SOURCE_FILES,
    "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py",
)
METHOD_SOURCE_SHA256 = MappingProxyType({
    **dict(tp4_gate.METHOD_SOURCE_SHA256),
    "bind_qwen35_loaded_checkpoint_candidates": (
        "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c"
    ),
})
ATTEMPT_MODES = (
    "tp4_synthetic_binding_success",
    "tp4_synthetic_rank2_model_mismatch",
    "tp4_synthetic_rank2_layout_mismatch",
    "tp4_synthetic_rank2_dtype_mismatch",
)
ROW_SCHEMA_VERSION = "qwen35.tp4-synthetic-binding-oracle-rank.v1"
SCHEMA_VERSION = "qwen35.tp4-synthetic-binding-oracle-gate.v1"
SHARED_MEMORY_CAPACITY = tp4_gate.SHARED_MEMORY_CAPACITY
WORKER_RANKS = tp4_gate.WORKER_RANKS
REMOTE_TARGET = tp4_gate.REMOTE_TARGET
REMOTE_PYTHON = tp4_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = tp4_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-synthetic-binding-oracle-runs"
)
_source_tree_sha256 = tp4_gate._source_tree_sha256
_atomic_write_json = tp4_gate._atomic_write_json
validate_run_tag = tp4_gate.validate_run_tag
build_ssh_command = tp4_gate.build_ssh_command
_require_success = tp4_gate._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class SyntheticBindingPrerequisites:
    tp4_artifact_sha256: str
    tp4_source_tree_sha256: str
    oracle_artifact_sha256: str
    source_file_sha256: Mapping
    oracle: Mapping
    cases: Mapping


def _validate_oracle(record):
    exact = {
        "schema_version": "qwen35.tp4-synthetic-binding-oracle.v1",
        "provenance": "synthetic-construction-free-oracle",
        "claim_boundary": "not-real-checkpoint-binding",
        "tensor_payload": "absent",
        "canonical_json": (
            "sort_keys=true,separators=(comma,colon),utf8"
        ),
        "model_fingerprint": MODEL_FINGERPRINT,
        "layout_fingerprint": LAYOUT_FINGERPRINT,
        "alternate_model_fingerprint": ALTERNATE_MODEL_FINGERPRINT,
        "alternate_layout_fingerprint": ALTERNATE_LAYOUT_FINGERPRINT,
        "dtype": "bfloat16",
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(f"synthetic oracle {name} is invalid")
    if _sha256(_canonical(record.get("model_descriptor"))) != MODEL_FINGERPRINT:
        raise ValueError("synthetic oracle model descriptor is invalid")
    if _sha256(_canonical(record.get("layout_descriptor"))) != LAYOUT_FINGERPRINT:
        raise ValueError("synthetic oracle layout descriptor is invalid")
    if _sha256(_canonical({
        **record["model_descriptor"],
        "revision": "mismatch-v1",
    })) != ALTERNATE_MODEL_FINGERPRINT:
        raise ValueError("synthetic oracle alternate model is invalid")
    if _sha256(_canonical({
        **record["layout_descriptor"],
        "revision": "mismatch-v1",
    })) != ALTERNATE_LAYOUT_FINGERPRINT:
        raise ValueError("synthetic oracle alternate layout is invalid")
    cases = record.get("cases")
    if (
        not isinstance(cases, list)
        or tuple(case.get("mode") for case in cases) != ATTEMPT_MODES
    ):
        raise ValueError("synthetic oracle cases are invalid")
    case_map = {}
    changed_fields = {
        ATTEMPT_MODES[0]: None,
        ATTEMPT_MODES[1]: "model_fingerprint",
        ATTEMPT_MODES[2]: "layout_fingerprint",
        ATTEMPT_MODES[3]: "dtype",
    }
    baseline = cases[0]["rows"]
    for case in cases:
        mode = case["mode"]
        rows = case.get("rows")
        if not isinstance(rows, list) or len(rows) != 4:
            raise ValueError("synthetic oracle row count is invalid")
        for rank, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or set(row) != {
                    "participant_id",
                    "operation",
                    "status",
                    "model_fingerprint",
                    "layout_fingerprint",
                    "dtype",
                    "detail",
                }
                or row["participant_id"] != rank
                or row["operation"] != "bind_loaded_checkpoint_candidate"
                or row["status"] != "bound"
                or row["detail"] != ""
            ):
                raise ValueError("synthetic oracle binding row is invalid")
        differences = [
            (rank, name)
            for rank in range(4)
            for name in ("model_fingerprint", "layout_fingerprint", "dtype")
            if rows[rank][name] != baseline[rank][name]
        ]
        expected = changed_fields[mode]
        if differences != ([] if expected is None else [(2, expected)]):
            raise ValueError("synthetic oracle mismatch scope is invalid")
        case_map[mode] = tuple(
            MappingProxyType(dict(row)) for row in rows
        )
    return MappingProxyType(case_map)


def load_synthetic_binding_prerequisites(tp4_artifact, oracle_artifact):
    tp4_payload = Path(tp4_artifact).read_bytes()
    if _sha256(tp4_payload) != TP4_ARTIFACT_SHA256:
        raise ValueError("TP4 prerequisite hash is invalid")
    tp4_record = json.loads(tp4_payload)
    tp4_gate.validate_tp4_fanout_preflight(tp4_record)
    if tp4_record.get("source_tree_sha256") != TP4_SOURCE_TREE_SHA256:
        raise ValueError("TP4 prerequisite source tree is invalid")
    hashes = tp4_record.get("source_file_sha256")
    if not isinstance(hashes, dict) or len(hashes) != 56:
        raise ValueError("TP4 prerequisite source closure is invalid")
    oracle_payload = Path(oracle_artifact).read_bytes()
    if _sha256(oracle_payload) != ORACLE_ARTIFACT_SHA256:
        raise ValueError("synthetic oracle artifact hash is invalid")
    oracle = json.loads(oracle_payload)
    cases = _validate_oracle(oracle)
    return SyntheticBindingPrerequisites(
        tp4_artifact_sha256=TP4_ARTIFACT_SHA256,
        tp4_source_tree_sha256=TP4_SOURCE_TREE_SHA256,
        oracle_artifact_sha256=ORACLE_ARTIFACT_SHA256,
        source_file_sha256=MappingProxyType(hashes),
        oracle=MappingProxyType(oracle),
        cases=cases,
    )


def load_frozen_synthetic_binding_methods(source_root):
    methods = tp4_gate.tp2_gate.load_frozen_live_shared_memory_methods(
        source_root,
        fingerprint_validator=tp4_gate.tp2_gate._validate_fingerprint,
    )
    return MappingProxyType({
        name: methods[name] for name in METHOD_SOURCE_SHA256
    })


def _worker(
    source_root,
    shared_memory_name,
    event,
    ack_sender,
    ready_sender,
    rank,
    row,
    read_count,
    executor_count,
    ack_order,
    ack_order_index,
):
    shared = SharedMemory(name=shared_memory_name)
    try:
        methods = load_frozen_synthetic_binding_methods(source_root)

        class WorkerShell:
            world_size = 4

            def read_shm(self):
                tp4_gate._increment(read_count)
                return methods["read_shm"](self)

            def bind_published_qwen35_loaded_checkpoint_candidate(self):
                time.sleep({1: 0.12, 2: 0.07, 3: 0.02}[rank])
                return dict(row)

            def exit(self):
                return None

        original_executor = methods["loop"].__globals__[
            "execute_acknowledged_command"
        ]

        def counted_executor(*args, **kwargs):
            tp4_gate._increment(executor_count)
            return original_executor(*args, **kwargs)

        methods["loop"].__globals__[
            "execute_acknowledged_command"
        ] = counted_executor
        ready_sender.send({
            "rank": rank,
            "process_id": os.getpid(),
            "shared_memory_name": shared.name,
        })
        ready_sender.close()
        worker = WorkerShell()
        worker.rank = rank
        worker.shm = shared
        worker.event = event
        worker.ack_sender = tp4_gate._OrderedSender(
            ack_sender,
            rank,
            ack_order,
            ack_order_index,
        )
        worker._command_ids = count()
        methods["loop"](worker)
    finally:
        shared.close()
        ack_sender.close()


def execute_tp4_synthetic_binding_attempt(
    *,
    source_root,
    tp4_artifact,
    oracle_artifact,
    mode,
    timeout_s,
    name_prefix,
):
    if mode not in ATTEMPT_MODES:
        raise ValueError("synthetic binding mode is invalid")
    prerequisites = load_synthetic_binding_prerequisites(
        tp4_artifact,
        oracle_artifact,
    )
    rows = prerequisites.cases[mode]
    methods = load_frozen_synthetic_binding_methods(source_root)
    context = multiprocessing.get_context("fork")
    events = {
        rank: tp4_gate._SharedCountingEvent(context)
        for rank in WORKER_RANKS
    }
    ack_order = context.Array("i", len(WORKER_RANKS))
    ack_order_index = context.Value("i", 0)
    ack_endpoints = {
        rank: context.Pipe(duplex=False) for rank in WORKER_RANKS
    }
    ready_endpoints = {
        rank: context.Pipe(duplex=False) for rank in WORKER_RANKS
    }
    read_counts = {
        rank: context.Value("i", 0) for rank in WORKER_RANKS
    }
    executor_counts = {
        rank: context.Value("i", 0) for rank in WORKER_RANKS
    }
    shared_memory_name = tp4_gate._shared_memory_name(name_prefix)
    shared = SharedMemory(
        name=shared_memory_name,
        create=True,
        size=SHARED_MEMORY_CAPACITY,
    )
    processes = {}
    receivers = {}
    child_process_ids = {}
    envelopes = []
    payload_bytes = []
    dispatch_count = 0
    binding_dispatch_count = 0
    write_count = 0
    collector_return_order = []
    binding_rows = None
    completion_configuration = None
    completion_committed = False
    repeat_zero_binding_dispatch = False
    error_detail = ""
    collector = None
    segment_unlinked = False
    post_unlink_attach_failed = False
    try:
        for rank in WORKER_RANKS:
            ack_receiver, ack_sender = ack_endpoints[rank]
            ready_receiver, ready_sender = ready_endpoints[rank]
            receivers[rank] = tp4_gate._RecordingReceiver(ack_receiver)
            process = context.Process(
                target=_worker,
                args=(
                    os.fspath(Path(source_root)),
                    shared_memory_name,
                    events[rank],
                    ack_sender,
                    ready_sender,
                    rank,
                    rows[rank],
                    read_counts[rank],
                    executor_counts[rank],
                    ack_order,
                    ack_order_index,
                ),
            )
            processes[rank] = process
            process.start()
            ack_sender.close()
            ready_sender.close()
        for rank in WORKER_RANKS:
            receiver = ready_endpoints[rank][0]
            if not receiver.poll(timeout_s):
                raise TimeoutError(f"rank {rank} readiness timeout")
            ready = receiver.recv()
            receiver.close()
            if ready["shared_memory_name"] != shared_memory_name:
                raise RuntimeError("worker attached to wrong segment")
            child_process_ids[str(rank)] = ready["process_id"]
        collector = ack_module.ModelRunnerCommandAckCollector(tuple(
            (rank, receivers[rank]) for rank in WORKER_RANKS
        ))

        class RankZeroShell:
            world_size = 4
            rank = 0

            def write_shm(self, envelope):
                nonlocal write_count
                write_count += 1
                methods["write_shm"](self, envelope)
                payload_bytes.append(int.from_bytes(
                    self.shm.buf[0:4],
                    "little",
                ))

            def dispatch_command(
                self,
                method_name,
                *args,
                requires_ack,
            ):
                nonlocal dispatch_count, binding_dispatch_count
                dispatch_count += 1
                if method_name == (
                    "bind_published_qwen35_loaded_checkpoint_candidate"
                ):
                    binding_dispatch_count += 1
                envelope = methods["dispatch_command"](
                    self,
                    method_name,
                    *args,
                    requires_ack=requires_ack,
                )
                envelopes.append(envelope)
                return envelope

            def bind_published_qwen35_loaded_checkpoint_candidate(self):
                return dict(rows[0])

        rank_zero = RankZeroShell()
        rank_zero.shm = shared
        rank_zero.event = [events[rank] for rank in WORKER_RANKS]
        rank_zero._command_ids = count()

        class EngineShell:
            def _is_worker_rank_alive(self, rank):
                return processes[rank].is_alive()

            def call_model_runner_acknowledged(
                self,
                method_name,
                *args,
                timeout_s,
            ):
                local_result, worker_acks = methods[
                    "call_model_runner_acknowledged"
                ](
                    self,
                    method_name,
                    *args,
                    timeout_s=timeout_s,
                )
                collector_return_order.extend(
                    acknowledgement.rank
                    for acknowledgement in worker_acks
                )
                return local_result, worker_acks

        engine = EngineShell()
        engine.model_runner = rank_zero
        engine.model_runner_ack_collector = collector
        engine.qwen35_loaded_checkpoint_candidate_binding_configuration = None
        engine.qwen35_loaded_checkpoint_candidate_binding_rows = None
        try:
            result = methods[
                "bind_qwen35_loaded_checkpoint_candidates"
            ](
                engine,
                timeout_s=timeout_s,
            )
            binding_rows = [dict(row) for row in result]
            completion_configuration = list(
                engine.qwen35_loaded_checkpoint_candidate_binding_configuration
            )
            completion_committed = True
            before = binding_dispatch_count
            repeated = methods[
                "bind_qwen35_loaded_checkpoint_candidates"
            ](
                engine,
                timeout_s=timeout_s,
            )
            repeat_zero_binding_dispatch = (
                repeated is result and binding_dispatch_count == before
            )
        except RuntimeError as error:
            error_detail = f"RuntimeError: {error}"
        deadline = time.monotonic() + timeout_s
        tp4_gate._wait_events_cleared(events, deadline)
        rank_zero.dispatch_command("exit", requires_ack=False)
        for process in processes.values():
            process.join(timeout=timeout_s)
        if any(process.is_alive() for process in processes.values()):
            raise RuntimeError("synthetic binding worker did not terminate")
    finally:
        for process in processes.values():
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout_s)
        for rank in WORKER_RANKS:
            try:
                ready_endpoints[rank][0].close()
            except OSError:
                pass
            if rank in receivers:
                receivers[rank].close()
        shared.close()
        shared.unlink()
        segment_unlinked = True
    try:
        probe = SharedMemory(name=shared_memory_name)
    except FileNotFoundError:
        post_unlink_attach_failed = True
    else:
        probe.close()
        raise RuntimeError("synthetic binding segment remains attachable")
    ack_status_by_rank = {
        str(rank): (
            receivers[rank].values[-1].status
            if receivers[rank].values
            else "absent"
        )
        for rank in WORKER_RANKS
    }
    changed = {
        ATTEMPT_MODES[0]: None,
        ATTEMPT_MODES[1]: "model_fingerprint",
        ATTEMPT_MODES[2]: "layout_fingerprint",
        ATTEMPT_MODES[3]: "dtype",
    }[mode]
    return {
        "status": "PASS",
        "mode": mode,
        "process_id": os.getpid(),
        "child_process_ids": child_process_ids,
        "child_exitcodes": {
            str(rank): processes[rank].exitcode for rank in WORKER_RANKS
        },
        "child_collected_by_rank": {
            str(rank): not processes[rank].is_alive()
            for rank in WORKER_RANKS
        },
        "shared_memory_name": shared_memory_name,
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "segment_unlinked": segment_unlinked,
        "post_unlink_attach_failed": post_unlink_attach_failed,
        "dispatch_count": dispatch_count,
        "binding_dispatch_count": binding_dispatch_count,
        "write_count": write_count,
        "read_count_by_rank": {
            str(rank): read_counts[rank].value for rank in WORKER_RANKS
        },
        "executor_count_by_rank": {
            str(rank): executor_counts[rank].value
            for rank in WORKER_RANKS
        },
        "event_set_count_by_rank": {
            str(rank): events[rank].set_count for rank in WORKER_RANKS
        },
        "event_wait_count_by_rank": {
            str(rank): events[rank].wait_count for rank in WORKER_RANKS
        },
        "event_clear_count_by_rank": {
            str(rank): events[rank].clear_count for rank in WORKER_RANKS
        },
        "write_payload_bytes": payload_bytes,
        "envelopes": [
            tp4_gate._envelope_to_dict(envelope) for envelope in envelopes
        ],
        "ack_send_order": [
            ack_order[index] for index in range(ack_order_index.value)
        ],
        "collector_return_order": collector_return_order,
        "ack_status_by_rank": ack_status_by_rank,
        "collector_poisoned": bool(collector and collector.poisoned),
        "oracle_rows": [dict(row) for row in rows],
        "authorized_changed_field": changed,
        "binding_rows": binding_rows,
        "completion_configuration": completion_configuration,
        "completion_committed": completion_committed,
        "repeat_zero_binding_dispatch": repeat_zero_binding_dispatch,
        "error_detail": error_detail,
    }


def validate_synthetic_binding_attempt_row(row):
    mode = row.get("mode")
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or mode not in ATTEMPT_MODES
        or row.get("status") != "PASS"
    ):
        raise ValueError("synthetic binding attempt schema is invalid")
    exact = {
        "observed_user": "sitian",
        "tp4_artifact_sha256": TP4_ARTIFACT_SHA256,
        "oracle_artifact_sha256": ORACLE_ARTIFACT_SHA256,
        "oracle_provenance": "synthetic-construction-free-oracle",
        "oracle_claim_boundary": "not-real-checkpoint-binding",
        "oracle_tensor_payload": "absent",
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "dispatch_count": 2,
        "binding_dispatch_count": 1,
        "write_count": 2,
        "write_payload_bytes": [199, 154],
        "ack_send_order": [3, 2, 1],
        "collector_return_order": [1, 2, 3],
        "ack_status_by_rank": {"1": "ok", "2": "ok", "3": "ok"},
        "collector_poisoned": False,
        "child_exitcodes": {"1": 0, "2": 0, "3": 0},
        "child_collected_by_rank": {
            "1": True, "2": True, "3": True,
        },
        "segment_unlinked": True,
        "post_unlink_attach_failed": True,
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            raise ValueError(f"synthetic binding {name} is invalid")
    for field in (
        "read_count_by_rank",
        "executor_count_by_rank",
        "event_set_count_by_rank",
        "event_wait_count_by_rank",
        "event_clear_count_by_rank",
    ):
        if row.get(field) != {"1": 2, "2": 2, "3": 2}:
            raise ValueError(f"synthetic binding {field} is invalid")
    for name in ("process_id",):
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"synthetic binding {name} is invalid")
    child_ids = row.get("child_process_ids")
    if (
        not isinstance(child_ids, dict)
        or tuple(child_ids) != ("1", "2", "3")
        or len(set(child_ids.values())) != 3
        or row["process_id"] in set(child_ids.values())
    ):
        raise ValueError("synthetic binding child identities are invalid")
    name = row.get("shared_memory_name")
    if (
        not isinstance(name, str)
        or not name
        or name == "tinyvllm"
        or len(name) > 30
    ):
        raise ValueError("synthetic binding shared-memory name is invalid")
    if row.get("envelopes") != [
        {
            "command_id": 0,
            "method_name": (
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ),
            "args": [],
            "requires_ack": True,
        },
        {
            "command_id": 1,
            "method_name": "exit",
            "args": [],
            "requires_ack": False,
        },
    ]:
        raise ValueError("synthetic binding envelopes are invalid")
    rows = row.get("oracle_rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("synthetic binding oracle rows are invalid")
    expected_changed = {
        ATTEMPT_MODES[0]: None,
        ATTEMPT_MODES[1]: "model_fingerprint",
        ATTEMPT_MODES[2]: "layout_fingerprint",
        ATTEMPT_MODES[3]: "dtype",
    }[mode]
    if row.get("authorized_changed_field") != expected_changed:
        raise ValueError(
            "synthetic binding authorized changed field is invalid"
        )
    success = mode == ATTEMPT_MODES[0]
    if (
        row.get("completion_committed") is not success
        or row.get("repeat_zero_binding_dispatch") is not success
    ):
        raise ValueError("synthetic binding completion state is invalid")
    if success:
        if (
            row.get("binding_rows") != rows
            or row.get("completion_configuration")
            != [
                MODEL_FINGERPRINT,
                LAYOUT_FINGERPRINT,
                "bfloat16",
                4.0,
            ]
            or row.get("error_detail") != ""
        ):
            raise ValueError("synthetic binding success is invalid")
    else:
        if (
            row.get("binding_rows") is not None
            or row.get("completion_configuration") is not None
            or f"mismatch: {expected_changed}"
            not in row.get("error_detail", "")
        ):
            raise ValueError("synthetic binding mismatch is invalid")
    return row


def run_synthetic_binding_attempt_worker(
    *,
    source_root,
    tp4_artifact,
    oracle_artifact,
    mode,
    timeout_s,
    observed_user,
    observed_hostname,
    process_id,
):
    attempt = execute_tp4_synthetic_binding_attempt(
        source_root=source_root,
        tp4_artifact=tp4_artifact,
        oracle_artifact=oracle_artifact,
        mode=mode,
        timeout_s=timeout_s,
        name_prefix=f"qwen35-{mode}",
    )
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "tp4_artifact_sha256": TP4_ARTIFACT_SHA256,
        "oracle_artifact_sha256": ORACLE_ARTIFACT_SHA256,
        "oracle_provenance": "synthetic-construction-free-oracle",
        "oracle_claim_boundary": "not-real-checkpoint-binding",
        "oracle_tensor_payload": "absent",
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        **attempt,
        "process_id": process_id,
    }
    validate_synthetic_binding_attempt_row(row)
    return row


def _source_hashes(source_root):
    root = Path(source_root)
    return dict(sorted(
        (name, _sha256((root / name).read_bytes()))
        for name in SOURCE_FILES
    ))


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(
                    "missing synthetic binding source: " + name
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def audit_synthetic_binding_source(source_root):
    path = Path(source_root) / (
        "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py"
    )
    tree = ast.parse(path.read_text(), filename=os.fspath(path))
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]

    def named(name):
        return sum(
            isinstance(node.func, ast.Name) and node.func.id == name
            for node in calls
        )

    def attribute(name):
        return sum(
            isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            for node in calls
        )

    shared_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "SharedMemory"
    ]
    audit = {
        "llm_engine_import_count": 0,
        "model_runner_import_count": 0,
        "llm_engine_construction_count": (
            named("LLMEngine") + attribute("LLMEngine")
        ),
        "model_runner_construction_count": (
            named("ModelRunner") + attribute("ModelRunner")
        ),
        "fixed_tinyvllm_shared_memory_count": sum(
            any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "tinyvllm"
                for keyword in node.keywords
            )
            for node in shared_calls
        ),
        "checkpoint_call_count": sum(
            named(name) + attribute(name)
            for name in (
                "read_qwen35_checkpoint_metadata",
                "load_qwen35_fresh_checkpoint_candidate",
            )
        ),
        "model_construction_count": sum(
            named(name) + attribute(name)
            for name in (
                "prepare_qwen35_checkpoint_candidate_target",
                "build_qwen35_authorized_checkpoint_candidate_loader",
            )
        ),
        "scheduler_call_count": (
            named("Scheduler") + attribute("schedule")
        ),
        "step_call_count": attribute("step"),
        "cuda_call_count": attribute("cuda"),
        "forward_call_count": named("forward") + attribute("forward"),
        "inference_call_count": (
            named("inference") + attribute("inference")
        ),
    }
    if any(audit.values()):
        raise ValueError(f"synthetic binding audit is invalid: {audit!r}")
    return audit


def _aggregate(rows, source_root):
    if not isinstance(rows, list) or len(rows) != len(ATTEMPT_MODES):
        raise ValueError("synthetic binding attempt rows are incomplete")
    if [row.get("mode") for row in rows] != list(ATTEMPT_MODES):
        raise ValueError("synthetic binding attempt rows are invalid")
    for row in rows:
        validate_synthetic_binding_attempt_row(row)
    outer_ids = {row["process_id"] for row in rows}
    child_ids = {
        pid
        for row in rows
        for pid in row["child_process_ids"].values()
    }
    names = {row["shared_memory_name"] for row in rows}
    if (
        len(outer_ids) != 4
        or len(child_ids) != 12
        or len(names) != 4
        or not outer_ids.isdisjoint(child_ids)
    ):
        raise ValueError("synthetic binding identities are invalid")
    hashes = _source_hashes(source_root)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "tp4_artifact_sha256": TP4_ARTIFACT_SHA256,
        "tp4_source_tree_sha256": TP4_SOURCE_TREE_SHA256,
        "oracle_artifact_sha256": ORACLE_ARTIFACT_SHA256,
        "oracle_provenance": "synthetic-construction-free-oracle",
        "oracle_claim_boundary": "not-real-checkpoint-binding",
        "oracle_tensor_payload": "absent",
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": tp4_gate._source_tree_sha256(hashes),
        "rows": rows,
    }
    validate_synthetic_binding_preflight(record)
    return record


def validate_synthetic_binding_preflight(record):
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "tp4_artifact_sha256": TP4_ARTIFACT_SHA256,
        "tp4_source_tree_sha256": TP4_SOURCE_TREE_SHA256,
        "oracle_artifact_sha256": ORACLE_ARTIFACT_SHA256,
        "oracle_provenance": "synthetic-construction-free-oracle",
        "oracle_claim_boundary": "not-real-checkpoint-binding",
        "oracle_tensor_payload": "absent",
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(
                f"synthetic binding preflight {name} is invalid"
            )
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, dict)
        or tuple(hashes) != tuple(sorted(SOURCE_FILES))
        or set(hashes) != set(SOURCE_FILES)
        or record.get("source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError(
            "synthetic binding preflight source closure is invalid"
        )
    rows = record.get("rows")
    if [row.get("mode") for row in rows or ()] != list(ATTEMPT_MODES):
        raise ValueError(
            "synthetic binding preflight rows are invalid"
        )
    for row in rows:
        validate_synthetic_binding_attempt_row(row)
    outer_ids = {row["process_id"] for row in rows}
    child_ids = {
        process_id
        for row in rows
        for process_id in row["child_process_ids"].values()
    }
    names = {row["shared_memory_name"] for row in rows}
    if (
        len(outer_ids) != 4
        or len(child_ids) != 12
        or len(names) != 4
        or not outer_ids.isdisjoint(child_ids)
    ):
        raise ValueError(
            "synthetic binding preflight identities are invalid"
        )
    return record


def stage_source_and_prerequisites(
    source_root,
    run_tag,
    *,
    tp4_artifact,
    oracle_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    prerequisites = load_synthetic_binding_prerequisites(
        tp4_artifact,
        oracle_artifact,
    )
    tp4_payload = Path(tp4_artifact).read_bytes()
    oracle_payload = Path(oracle_artifact).read_bytes()
    local_hashes = _source_hashes(source_root)
    if {
        name: local_hashes[name]
        for name in prerequisites.source_file_sha256
    } != dict(prerequisites.source_file_sha256):
        raise ValueError(
            "synthetic binding source does not match TP4 prerequisite"
        )
    audit_synthetic_binding_source(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_tp4_artifact = (
        f"{remote_run_dir}/tp4_shared_memory_fanout_preflight.json"
    )
    remote_oracle_artifact = (
        f"{remote_run_dir}/synthetic_binding_oracle.json"
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
    _require_success(staged, "synthetic binding source staging")
    for label, path, payload in (
        ("TP4", remote_tp4_artifact, tp4_payload),
        ("oracle", remote_oracle_artifact, oracle_payload),
    ):
        completed = command_runner(
            build_ssh_command([
                "bash",
                "-c",
                f"cat > {shlex.quote(path)}",
            ]),
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _require_success(
            completed,
            f"synthetic binding {label} prerequisite staging",
        )
    method_specs = [
        (
            tp4_gate.MODEL_RUNNER_SOURCE,
            "ModelRunner",
            method_name,
        )
        for method_name in (
            "write_shm",
            "read_shm",
            "loop",
            "dispatch_command",
        )
    ] + [
        (
            tp4_gate.LLM_ENGINE_SOURCE,
            "LLMEngine",
            "call_model_runner_acknowledged",
        ),
        (
            tp4_gate.LLM_ENGINE_SOURCE,
            "LLMEngine",
            "bind_qwen35_loaded_checkpoint_candidates",
        ),
    ]
    script = "\n".join([
        "import ast,hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"tp4=pathlib.Path({remote_tp4_artifact!r})",
        f"oracle_path=pathlib.Path({remote_oracle_artifact!r})",
        f"names={list(SOURCE_FILES)!r}",
        f"specs={method_specs!r}",
        "canonical=lambda value: json.dumps(value,sort_keys=True,separators=(',',':')).encode()",
        "source_hashes={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " source_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "methods={}",
        "for filename,class_name,method_name in specs:",
        " source=(root/filename).read_text()",
        " tree=ast.parse(source,filename=filename)",
        " classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==class_name]",
        " if len(classes)!=1: raise SystemExit('invalid class: '+class_name)",
        " nodes=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name==method_name]",
        " if len(nodes)!=1: raise SystemExit('invalid method: '+method_name)",
        " segment=ast.get_source_segment(source,nodes[0])",
        " methods[method_name]=hashlib.sha256(segment.encode()).hexdigest()",
        "oracle=json.loads(oracle_path.read_bytes())",
        "descriptors={'model':hashlib.sha256(canonical(oracle['model_descriptor'])).hexdigest(),'layout':hashlib.sha256(canonical(oracle['layout_descriptor'])).hexdigest(),'alternate_model':hashlib.sha256(canonical({**oracle['model_descriptor'],'revision':'mismatch-v1'})).hexdigest(),'alternate_layout':hashlib.sha256(canonical({**oracle['layout_descriptor'],'revision':'mismatch-v1'})).hexdigest()}",
        "payload={'source':source_hashes,'tp4':hashlib.sha256(tp4.read_bytes()).hexdigest(),'oracle':hashlib.sha256(oracle_path.read_bytes()).hexdigest(),'methods':methods,'descriptors':descriptors}",
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
    _require_success(verified, "synthetic binding staged hashing")
    remote = json.loads(verified.stdout)
    expected_descriptors = {
        "model": MODEL_FINGERPRINT,
        "layout": LAYOUT_FINGERPRINT,
        "alternate_model": ALTERNATE_MODEL_FINGERPRINT,
        "alternate_layout": ALTERNATE_LAYOUT_FINGERPRINT,
    }
    if (
        remote.get("source") != local_hashes
        or remote.get("tp4") != TP4_ARTIFACT_SHA256
        or remote.get("oracle") != ORACLE_ARTIFACT_SHA256
        or remote.get("methods") != dict(METHOD_SOURCE_SHA256)
        or remote.get("descriptors") != expected_descriptors
    ):
        raise ValueError(
            "synthetic binding staged identity is invalid"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "remote_tp4_artifact": remote_tp4_artifact,
        "remote_oracle_artifact": remote_oracle_artifact,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "tp4_artifact_sha256": remote["tp4"],
        "oracle_artifact_sha256": remote["oracle"],
        "method_source_sha256": remote["methods"],
        "descriptor_sha256": remote["descriptors"],
    }


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "remote_tp4_artifact": staged["remote_tp4_artifact"],
        "remote_oracle_artifact": staged["remote_oracle_artifact"],
        "tp4_artifact_sha256": staged["tp4_artifact_sha256"],
        "oracle_artifact_sha256": staged["oracle_artifact_sha256"],
        "method_source_sha256": dict(
            staged["method_source_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_synthetic_binding_preflight(
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
            f"local synthetic binding directory exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/"
        "tp4_synthetic_binding_oracle_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_tp4_synthetic_binding_oracle_preflight.py"
    )
    rows = []
    for mode in ATTEMPT_MODES:
        completed = command_runner(
            build_ssh_command([
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                REMOTE_PYTHON,
                "-B",
                worker,
                "internal-attempt-worker",
                "--source-root",
                staged["remote_source_dir"],
                "--tp4-artifact",
                staged["remote_tp4_artifact"],
                "--oracle-artifact",
                staged["remote_oracle_artifact"],
                "--attempt-mode",
                mode,
                "--timeout-s",
                "4.0",
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(
            completed,
            "synthetic binding attempt worker",
        )
        row = json.loads(completed.stdout)
        validate_synthetic_binding_attempt_row(row)
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
    _require_success(finalized, "synthetic binding finalizer")
    record = json.loads(finalized.stdout)
    validate_synthetic_binding_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("synthetic binding source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    remote_manifest = f"{remote_run_dir}/source_manifest.json"
    completed = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            f"cat > {shlex.quote(remote_manifest)}",
        ]),
        input=(
            json.dumps(
                source_manifest,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ),
        text=True,
        capture_output=True,
    )
    _require_success(
        completed,
        "synthetic binding manifest publication",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary
            / "tp4_synthetic_binding_oracle_preflight.json",
            record,
        )
        _atomic_write_json(
            temporary / "source_manifest.json",
            source_manifest,
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            for child_path in temporary.iterdir():
                child_path.unlink()
            temporary.rmdir()
    return record


def execute_remote_synthetic_binding_preflight(
    source_root,
    run_tag,
    *,
    tp4_artifact,
    oracle_artifact,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source_and_prerequisites(
        source_root,
        run_tag,
        tp4_artifact=tp4_artifact,
        oracle_artifact=oracle_artifact,
        command_runner=command_runner,
    )
    return run_remote_synthetic_binding_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-tag", required=True)
    run.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    run.add_argument("--tp4-artifact", required=True)
    run.add_argument("--oracle-artifact", required=True)
    worker = subparsers.add_parser("internal-attempt-worker")
    worker.add_argument("--source-root", required=True)
    worker.add_argument("--tp4-artifact", required=True)
    worker.add_argument("--oracle-artifact", required=True)
    worker.add_argument(
        "--attempt-mode",
        choices=ATTEMPT_MODES,
        required=True,
    )
    worker.add_argument("--timeout-s", type=float, default=4.0)
    finalizer = subparsers.add_parser("internal-finalize")
    finalizer.add_argument("--source-root", required=True)
    finalizer.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("artifact")
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        record = execute_remote_synthetic_binding_preflight(
            arguments.source_root,
            arguments.run_tag,
            tp4_artifact=arguments.tp4_artifact,
            oracle_artifact=arguments.oracle_artifact,
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-attempt-worker":
        row = run_synthetic_binding_attempt_worker(
            source_root=arguments.source_root,
            tp4_artifact=arguments.tp4_artifact,
            oracle_artifact=arguments.oracle_artifact,
            mode=arguments.attempt_mode,
            timeout_s=arguments.timeout_s,
            observed_user=getpass.getuser(),
            observed_hostname=socket.gethostname(),
            process_id=os.getpid(),
        )
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-finalize":
        payload = json.load(sys.stdin)
        record = _aggregate(payload.get("rows"), arguments.source_root)
        _atomic_write_json(Path(arguments.output), record)
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "validate":
        record = json.loads(Path(arguments.artifact).read_text())
        validate_synthetic_binding_preflight(record)
        print("TP4 synthetic binding oracle artifact validated")
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
