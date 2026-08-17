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


tp2_gate = _load_sibling(
    "_qwen35_tp4_fanout_prerequisite",
    "qwen35_live_shared_memory_engine_ack_dispatch_preflight.py",
)
ack_module = tp2_gate.ack_module


PREREQUISITE_ARTIFACT_SHA256 = (
    "11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572"
)
LLM_ENGINE_SOURCE = tp2_gate.LLM_ENGINE_SOURCE
LLM_ENGINE_FILE_SHA256 = tp2_gate.LLM_ENGINE_FILE_SHA256
MODEL_RUNNER_SOURCE = tp2_gate.MODEL_RUNNER_SOURCE
MODEL_RUNNER_FILE_SHA256 = tp2_gate.MODEL_RUNNER_FILE_SHA256
ACK_SOURCE = tp2_gate.ACK_SOURCE
ACK_FILE_SHA256 = tp2_gate.ACK_FILE_SHA256
METHOD_SOURCE_SHA256 = MappingProxyType({
    name: tp2_gate.METHOD_SOURCE_SHA256[name]
    for name in (
        "write_shm",
        "read_shm",
        "loop",
        "dispatch_command",
        "call_model_runner_acknowledged",
    )
})
SOURCE_FILES = (
    *tp2_gate.SOURCE_FILES,
    "tools/qwen35_tp4_shared_memory_fanout_preflight.py",
)
SHARED_MEMORY_CAPACITY = 2**20
WORKER_RANKS = (1, 2, 3)
ATTEMPT_MODES = (
    "tp4_fanout_success_reverse_completion",
    "tp4_fanout_rank2_inner_error",
    "tp4_fanout_rank2_ack_exception",
    "tp4_fanout_rank2_exit_without_ack",
)
ROW_SCHEMA_VERSION = "qwen35.tp4-shared-memory-fanout-rank.v1"
SCHEMA_VERSION = "qwen35.tp4-shared-memory-fanout.v1"
REMOTE_TARGET = tp2_gate.REMOTE_TARGET
REMOTE_PYTHON = tp2_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = tp2_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-shared-memory-fanout-runs"
)
_source_tree_sha256 = tp2_gate._source_tree_sha256
_atomic_write_json = tp2_gate._atomic_write_json
validate_run_tag = tp2_gate.validate_run_tag
build_ssh_command = tp2_gate.build_ssh_command
_require_success = tp2_gate._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TP4FanoutPrerequisite:
    artifact_sha256: str
    source_tree_sha256: str
    rows: tuple
    source_file_sha256: Mapping


def load_tp4_fanout_prerequisite(artifact):
    payload = Path(artifact).read_bytes()
    if _sha256(payload) != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError("TP4 fan-out prerequisite hash is invalid")
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("TP4 fan-out prerequisite JSON is invalid") from error
    tp2_gate.validate_live_shared_memory_preflight(record)
    if record.get("source_tree_sha256") != PREREQUISITE_SOURCE_TREE_SHA256:
        raise ValueError("TP4 fan-out prerequisite source tree is invalid")
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != 55
        or tuple(hashes) != tuple(sorted(hashes))
    ):
        raise ValueError("TP4 fan-out prerequisite source closure is invalid")
    rows = tuple(row.get("mode") for row in record.get("rows", ()))
    if rows != tp2_gate.ATTEMPT_MODES:
        raise ValueError("TP4 fan-out prerequisite rows are invalid")
    return TP4FanoutPrerequisite(
        artifact_sha256=PREREQUISITE_ARTIFACT_SHA256,
        source_tree_sha256=PREREQUISITE_SOURCE_TREE_SHA256,
        rows=rows,
        source_file_sha256=MappingProxyType(hashes),
    )


def load_frozen_tp4_fanout_methods(source_root):
    methods = tp2_gate.load_frozen_live_shared_memory_methods(
        source_root,
        fingerprint_validator=tp2_gate._validate_fingerprint,
    )
    return MappingProxyType({
        name: methods[name] for name in METHOD_SOURCE_SHA256
    })


def build_tp4_identity_row(
    *,
    participant_id,
    attempt_nonce,
    status="ok",
    detail="",
):
    if (
        isinstance(participant_id, bool)
        or not isinstance(participant_id, int)
        or participant_id not in range(4)
    ):
        raise ValueError("participant_id must be in range 0..3")
    if not isinstance(attempt_nonce, str) or not attempt_nonce:
        raise ValueError("attempt_nonce must be a non-empty string")
    if status not in {"ok", "error"}:
        raise ValueError("TP4 identity status is invalid")
    if not isinstance(detail, str):
        raise ValueError("TP4 identity detail must be a string")
    if status == "ok" and detail:
        raise ValueError("successful TP4 identity row cannot contain detail")
    if status == "error" and not detail:
        raise ValueError("error TP4 identity row requires detail")
    return MappingProxyType({
        "participant_id": participant_id,
        "operation": "report_tp4_shared_memory_fanout_identity",
        "status": status,
        "attempt_nonce": attempt_nonce,
        "detail": detail,
    })


def validate_tp4_identity_rows(rows, *, attempt_nonce):
    if not isinstance(rows, tuple) or len(rows) != 4:
        raise ValueError("TP4 identity rows must contain four rows")
    for participant_id, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("TP4 identity row must be a mapping")
        exact = {
            "participant_id": participant_id,
            "operation": "report_tp4_shared_memory_fanout_identity",
            "attempt_nonce": attempt_nonce,
        }
        for name, expected in exact.items():
            if row.get(name) != expected:
                raise ValueError(f"TP4 identity row {name} is invalid")
        status = row.get("status")
        detail = row.get("detail")
        if status == "error":
            raise RuntimeError(
                "TP4 shared-memory fan-out identity failed: "
                f"rank={participant_id}, detail={detail}"
            )
        if status != "ok" or detail != "":
            raise ValueError("TP4 identity row status is invalid")
    return rows


class _SharedCountingEvent:
    def __init__(self, context):
        self._event = context.Event()
        self._set_count = context.Value("i", 0)
        self._wait_count = context.Value("i", 0)
        self._clear_count = context.Value("i", 0)

    @staticmethod
    def _increment(counter):
        with counter.get_lock():
            counter.value += 1

    def set(self):
        self._increment(self._set_count)
        return self._event.set()

    def wait(self):
        self._increment(self._wait_count)
        return self._event.wait()

    def clear(self):
        self._increment(self._clear_count)
        return self._event.clear()

    def is_set(self):
        return self._event.is_set()

    @property
    def set_count(self):
        return self._set_count.value

    @property
    def wait_count(self):
        return self._wait_count.value

    @property
    def clear_count(self):
        return self._clear_count.value


class _RecordingReceiver:
    def __init__(self, connection):
        self._connection = connection
        self.values = []

    def poll(self, timeout=0.0):
        return self._connection.poll(timeout)

    def recv(self):
        value = self._connection.recv()
        self.values.append(value)
        return value

    def close(self):
        return self._connection.close()


class _OrderedSender:
    def __init__(self, connection, rank, order, index):
        self._connection = connection
        self._rank = rank
        self._order = order
        self._index = index

    def send(self, value):
        if getattr(value, "status", None) in {"ok", "error"}:
            with self._index.get_lock():
                position = self._index.value
                if position < len(self._order):
                    self._order[position] = self._rank
                    self._index.value += 1
        return self._connection.send(value)

    def close(self):
        return self._connection.close()


def _increment(counter):
    with counter.get_lock():
        counter.value += 1


def _shared_memory_name(prefix):
    safe = "".join(
        character
        for character in str(prefix)
        if character.isalnum() or character in "-_"
    ).strip("-_")
    if not safe:
        raise ValueError("shared-memory name prefix is invalid")
    suffix = f"-{os.getpid()}-{uuid.uuid4().hex[:4]}"
    return f"{safe[:30 - len(suffix)]}{suffix}"


def _envelope_to_dict(envelope):
    return {
        "command_id": envelope.command_id,
        "method_name": envelope.method_name,
        "args": list(envelope.args),
        "requires_ack": envelope.requires_ack,
    }


def _tp4_worker(
    source_root,
    shared_memory_name,
    event,
    ack_sender,
    ready_sender,
    rank,
    mode,
    attempt_nonce,
    read_count,
    executor_count,
    ack_order,
    ack_order_index,
):
    shared = SharedMemory(name=shared_memory_name)
    try:
        methods = load_frozen_tp4_fanout_methods(source_root)

        class WorkerShell:
            world_size = 4

            def read_shm(self):
                _increment(read_count)
                return methods["read_shm"](self)

            def report_qwen35_tp4_fanout_identity(self, nonce):
                delay = {1: 0.12, 2: 0.07, 3: 0.02}[rank]
                time.sleep(delay)
                if mode == "tp4_fanout_rank2_ack_exception" and rank == 2:
                    raise RuntimeError(
                        "injected rank2 acknowledgement exception"
                    )
                if mode == "tp4_fanout_rank2_exit_without_ack" and rank == 2:
                    raise SystemExit(9)
                if mode == "tp4_fanout_rank2_inner_error" and rank == 2:
                    return dict(build_tp4_identity_row(
                        participant_id=rank,
                        attempt_nonce=nonce,
                        status="error",
                        detail="injected rank2 inner error",
                    ))
                return dict(build_tp4_identity_row(
                    participant_id=rank,
                    attempt_nonce=nonce,
                ))

            def exit(self):
                return None

        original_executor = methods["loop"].__globals__[
            "execute_acknowledged_command"
        ]

        def counted_executor(*args, **kwargs):
            _increment(executor_count)
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
        worker.ack_sender = _OrderedSender(
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


def _wait_events_cleared(events, deadline):
    while any(event.is_set() for event in events.values()):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for TP4 worker Events to clear"
            )
        time.sleep(0.001)


def _drain_receivers(receivers, deadline):
    while time.monotonic() < deadline:
        made_progress = False
        for receiver in receivers.values():
            if receiver.values:
                continue
            try:
                ready = receiver.poll(0.0)
            except (OSError, EOFError):
                continue
            if not ready:
                continue
            try:
                receiver.recv()
            except (EOFError, OSError):
                pass
            made_progress = True
        if all(receiver.values for receiver in receivers.values()):
            return
        if not made_progress:
            time.sleep(0.001)


def execute_tp4_shared_memory_fanout_attempt(
    *,
    source_root,
    prerequisite_artifact,
    mode,
    timeout_s,
    name_prefix,
):
    if mode not in ATTEMPT_MODES:
        raise ValueError("TP4 fan-out attempt mode is invalid")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or float(timeout_s) <= 0
    ):
        raise ValueError("timeout_s must be positive")
    timeout_s = float(timeout_s)
    load_tp4_fanout_prerequisite(prerequisite_artifact)
    methods = load_frozen_tp4_fanout_methods(source_root)
    context = multiprocessing.get_context("fork")
    events = {
        rank: _SharedCountingEvent(context) for rank in WORKER_RANKS
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
    shared_memory_name = _shared_memory_name(name_prefix)
    shared = SharedMemory(
        name=shared_memory_name,
        create=True,
        size=SHARED_MEMORY_CAPACITY,
    )
    processes = {}
    receivers = {}
    child_process_ids = {}
    dispatch_envelopes = []
    write_payload_bytes = []
    dispatch_count = 0
    write_count = 0
    collector_call_count = 0
    collector_return_order = []
    collector_result_participants = []
    fanout_rows = None
    fanout_validated = False
    error_detail = ""
    exit_envelope_sent = False
    segment_unlinked = False
    post_unlink_attach_failed = False
    collector = None
    attempt_nonce = uuid.uuid4().hex
    try:
        for rank in WORKER_RANKS:
            ack_receiver, ack_sender = ack_endpoints[rank]
            ready_receiver, ready_sender = ready_endpoints[rank]
            receivers[rank] = _RecordingReceiver(ack_receiver)
            process = context.Process(
                target=_tp4_worker,
                args=(
                    os.fspath(Path(source_root)),
                    shared_memory_name,
                    events[rank],
                    ack_sender,
                    ready_sender,
                    rank,
                    mode,
                    attempt_nonce,
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
            ready_receiver = ready_endpoints[rank][0]
            if not ready_receiver.poll(timeout_s):
                raise TimeoutError(
                    f"timed out waiting for TP4 rank {rank} readiness"
                )
            ready = ready_receiver.recv()
            ready_receiver.close()
            if (
                ready.get("rank") != rank
                or ready.get("shared_memory_name") != shared_memory_name
            ):
                raise RuntimeError("TP4 worker readiness is invalid")
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
                write_payload_bytes.append(int.from_bytes(
                    self.shm.buf[0:4],
                    "little",
                ))

            def dispatch_command(
                self,
                method_name,
                *args,
                requires_ack,
            ):
                nonlocal dispatch_count
                dispatch_count += 1
                envelope = methods["dispatch_command"](
                    self,
                    method_name,
                    *args,
                    requires_ack=requires_ack,
                )
                dispatch_envelopes.append(envelope)
                return envelope

            def report_qwen35_tp4_fanout_identity(self, nonce):
                return dict(build_tp4_identity_row(
                    participant_id=0,
                    attempt_nonce=nonce,
                ))

        rank_zero = RankZeroShell()
        rank_zero.shm = shared
        rank_zero.event = [events[rank] for rank in WORKER_RANKS]
        rank_zero._command_ids = count()

        class EngineShell:
            def _is_worker_rank_alive(self, rank):
                if rank not in processes:
                    raise ValueError("TP4 worker rank is invalid")
                return processes[rank].is_alive()

        engine = EngineShell()
        engine.model_runner = rank_zero
        engine.model_runner_ack_collector = collector
        try:
            collector_call_count += 1
            local_result, worker_acks = methods[
                "call_model_runner_acknowledged"
            ](
                engine,
                "report_qwen35_tp4_fanout_identity",
                attempt_nonce,
                timeout_s=timeout_s,
            )
            collector_return_order = [
                acknowledgement.rank
                for acknowledgement in worker_acks
            ]
            collector_result_participants = [
                acknowledgement.result["participant_id"]
                for acknowledgement in worker_acks
            ]
            fanout_rows = (
                MappingProxyType(dict(local_result)),
                *(
                    MappingProxyType(dict(acknowledgement.result))
                    for acknowledgement in worker_acks
                ),
            )
            validate_tp4_identity_rows(
                fanout_rows,
                attempt_nonce=attempt_nonce,
            )
            fanout_validated = True
        except (RuntimeError, TimeoutError) as error:
            error_detail = f"{type(error).__name__}: {error}"
        deadline = time.monotonic() + timeout_s
        _wait_events_cleared(events, deadline)
        _drain_receivers(receivers, deadline)
        rank_zero.dispatch_command("exit", requires_ack=False)
        exit_envelope_sent = True
        for process in processes.values():
            process.join(timeout=timeout_s)
        alive = [
            rank for rank, process in processes.items()
            if process.is_alive()
        ]
        if alive:
            for rank in alive:
                processes[rank].terminate()
                processes[rank].join(timeout=timeout_s)
            raise RuntimeError(
                "TP4 shared-memory workers did not terminate: "
                + ", ".join(str(rank) for rank in alive)
            )
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
        raise RuntimeError("TP4 shared-memory segment remains attachable")

    ack_status_by_rank = {}
    ack_error_by_rank = {}
    ack_result_by_rank = {}
    for rank in WORKER_RANKS:
        values = receivers[rank].values
        acknowledgement = values[-1] if values else None
        ack_status_by_rank[str(rank)] = (
            acknowledgement.status
            if acknowledgement is not None
            else "absent"
        )
        ack_error_by_rank[str(rank)] = (
            {
                "error_type": acknowledgement.error_type,
                "error_detail": acknowledgement.error_detail,
            }
            if acknowledgement is not None
            else {"error_type": "", "error_detail": ""}
        )
        ack_result_by_rank[str(rank)] = (
            acknowledgement.result
            if acknowledgement is not None
            and acknowledgement.status == "ok"
            else None
        )
    return {
        "status": "PASS",
        "mode": mode,
        "process_id": os.getpid(),
        "worker_ranks": list(WORKER_RANKS),
        "child_process_ids": child_process_ids,
        "child_exitcodes": {
            str(rank): processes[rank].exitcode
            for rank in WORKER_RANKS
        },
        "child_collected_by_rank": {
            str(rank): not processes[rank].is_alive()
            for rank in WORKER_RANKS
        },
        "shared_memory_name": shared_memory_name,
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "segment_unlinked": segment_unlinked,
        "post_unlink_attach_failed": post_unlink_attach_failed,
        "attempt_nonce": attempt_nonce,
        "dispatch_count": dispatch_count,
        "write_count": write_count,
        "collector_call_count": collector_call_count,
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
        "write_payload_bytes": write_payload_bytes,
        "envelopes": [
            _envelope_to_dict(envelope)
            for envelope in dispatch_envelopes
        ],
        "ack_send_order": [
            ack_order[index]
            for index in range(ack_order_index.value)
        ],
        "collector_return_order": collector_return_order,
        "collector_result_participants": (
            collector_result_participants
        ),
        "ack_status_by_rank": ack_status_by_rank,
        "ack_error_by_rank": ack_error_by_rank,
        "ack_result_by_rank": ack_result_by_rank,
        "collector_poisoned": bool(
            collector is not None and collector.poisoned
        ),
        "fanout_rows": (
            [dict(row) for row in fanout_rows]
            if fanout_rows is not None
            else None
        ),
        "fanout_validated": fanout_validated,
        "exit_envelope_sent": exit_envelope_sent,
        "error_detail": error_detail,
    }


def validate_tp4_fanout_attempt_row(row):
    mode = row.get("mode")
    if mode not in ATTEMPT_MODES or row.get("status") != "PASS":
        raise ValueError("TP4 fan-out attempt row schema is invalid")
    if row.get("worker_ranks") != list(WORKER_RANKS):
        raise ValueError("TP4 fan-out worker ranks are invalid")
    child_ids = row.get("child_process_ids")
    if (
        not isinstance(child_ids, dict)
        or tuple(child_ids) != ("1", "2", "3")
        or len(set(child_ids.values())) != 3
    ):
        raise ValueError("TP4 fan-out child identities are invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in child_ids.values()
    ):
        raise ValueError("TP4 fan-out child PID is invalid")
    exact = {
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "segment_unlinked": True,
        "post_unlink_attach_failed": True,
        "dispatch_count": 2,
        "write_count": 2,
        "collector_call_count": 1,
        "exit_envelope_sent": True,
        "child_collected_by_rank": {
            "1": True,
            "2": True,
            "3": True,
        },
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            raise ValueError(f"TP4 fan-out {name} is invalid")
    name = row.get("shared_memory_name")
    if (
        not isinstance(name, str)
        or not name
        or name == "tinyvllm"
        or len(name) > 30
    ):
        raise ValueError("TP4 fan-out shared-memory name is invalid")
    expected_exitcodes = {"1": 0, "2": 0, "3": 0}
    if mode == "tp4_fanout_rank2_exit_without_ack":
        expected_exitcodes["2"] = 9
    if row.get("child_exitcodes") != expected_exitcodes:
        raise ValueError("TP4 fan-out child exit codes are invalid")
    expected_statuses = {
        "tp4_fanout_success_reverse_completion": {
            "1": "ok", "2": "ok", "3": "ok",
        },
        "tp4_fanout_rank2_inner_error": {
            "1": "ok", "2": "ok", "3": "ok",
        },
        "tp4_fanout_rank2_ack_exception": {
            "1": "ok", "2": "error", "3": "ok",
        },
        "tp4_fanout_rank2_exit_without_ack": {
            "1": "ok", "2": "absent", "3": "ok",
        },
    }[mode]
    if row.get("ack_status_by_rank") != expected_statuses:
        raise ValueError("TP4 fan-out acknowledgement statuses are invalid")
    success = mode == "tp4_fanout_success_reverse_completion"
    if row.get("fanout_validated") is not success:
        raise ValueError("TP4 fan-out validation state is invalid")
    poisoned = mode in {
        "tp4_fanout_rank2_ack_exception",
        "tp4_fanout_rank2_exit_without_ack",
    }
    if row.get("collector_poisoned") is not poisoned:
        raise ValueError("TP4 fan-out poison state is invalid")
    if success:
        if (
            row.get("ack_send_order") != [3, 2, 1]
            or row.get("collector_return_order") != [1, 2, 3]
            or row.get("collector_result_participants") != [1, 2, 3]
            or row.get("error_detail")
        ):
            raise ValueError("TP4 fan-out success ordering is invalid")
    else:
        required = {
            "tp4_fanout_rank2_inner_error": (
                "injected rank2 inner error"
            ),
            "tp4_fanout_rank2_ack_exception": (
                "injected rank2 acknowledgement exception"
            ),
            "tp4_fanout_rank2_exit_without_ack": "acknowledgement",
        }[mode]
        if required not in row.get("error_detail", ""):
            raise ValueError("TP4 fan-out failure detail is invalid")
    return row


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError("missing TP4 fan-out source: " + name)
        hashes[name] = _sha256(path.read_bytes())
    return dict(sorted(hashes.items()))


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError("missing TP4 fan-out source: " + name)
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def audit_tp4_fanout_source(source_root):
    path = Path(source_root) / (
        "tools/qwen35_tp4_shared_memory_fanout_preflight.py"
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

    imports = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    shared_memory_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "SharedMemory"
    ]
    checkpoint_names = {
        "read_qwen35_checkpoint_metadata",
        "build_qwen35_checkpoint_tensor_plan",
        "prepare_qwen35_checkpoint_candidate_target",
        "build_qwen35_authorized_checkpoint_candidate_loader",
        "load_qwen35_fresh_checkpoint_candidate",
    }
    audit = {
        "llm_engine_import_count": sum(
            node.args[0].value == "tinyvllm.engine.llm_engine"
            for node in imports
        ),
        "model_runner_import_count": sum(
            node.args[0].value == "tinyvllm.engine.model_runner"
            for node in imports
        ),
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
            for node in shared_memory_calls
        ),
        "shared_memory_create_count": sum(
            any(
                keyword.arg == "create"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            for node in shared_memory_calls
        ),
        "shared_memory_attach_count": sum(
            not any(
                keyword.arg == "create"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            for node in shared_memory_calls
        ),
        "checkpoint_call_count": sum(
            named(name) + attribute(name)
            for name in checkpoint_names
        ),
        "scheduler_call_count": (
            named("Scheduler")
            + attribute("Scheduler")
            + attribute("schedule")
        ),
        "step_call_count": attribute("step"),
        "cuda_call_count": attribute("cuda"),
        "forward_call_count": named("forward") + attribute("forward"),
        "inference_call_count": (
            named("inference") + attribute("inference")
        ),
    }
    if any((
        audit["llm_engine_import_count"],
        audit["model_runner_import_count"],
        audit["llm_engine_construction_count"],
        audit["model_runner_construction_count"],
        audit["fixed_tinyvllm_shared_memory_count"],
        audit["checkpoint_call_count"],
        audit["scheduler_call_count"],
        audit["step_call_count"],
        audit["cuda_call_count"],
        audit["forward_call_count"],
        audit["inference_call_count"],
    )):
        raise ValueError(f"TP4 fan-out static audit is invalid: {audit!r}")
    return audit


def _aggregate(rows, source_root):
    if not isinstance(rows, list) or len(rows) != len(ATTEMPT_MODES):
        raise ValueError("TP4 fan-out attempt rows are incomplete")
    for row in rows:
        validate_tp4_fanout_attempt_row(row)
    hashes = _source_hashes(source_root)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": rows,
    }
    validate_tp4_fanout_preflight(record)
    return record


def validate_tp4_fanout_preflight(record):
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(f"TP4 fan-out {name} is invalid")
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, dict)
        or tuple(hashes) != tuple(sorted(SOURCE_FILES))
        or set(hashes) != set(SOURCE_FILES)
        or record.get("source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError("TP4 fan-out source closure is invalid")
    rows = record.get("rows")
    if [row.get("mode") for row in rows or ()] != list(ATTEMPT_MODES):
        raise ValueError("TP4 fan-out attempt rows are invalid")
    for row in rows:
        validate_tp4_fanout_attempt_row(row)
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
        raise ValueError("TP4 fan-out process identities are invalid")
    return record


def run_tp4_attempt_worker(
    *,
    source_root,
    prerequisite_artifact,
    mode,
    timeout_s,
    observed_user,
    observed_hostname,
    process_id,
):
    attempt = execute_tp4_shared_memory_fanout_attempt(
        source_root=source_root,
        prerequisite_artifact=prerequisite_artifact,
        mode=mode,
        timeout_s=timeout_s,
        name_prefix=f"qwen35-{mode}",
    )
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "prerequisite_artifact_sha256": PREREQUISITE_ARTIFACT_SHA256,
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        **attempt,
        "process_id": process_id,
    }
    validate_tp4_fanout_attempt_row(row)
    return row


def stage_source_and_prerequisite(
    source_root,
    run_tag,
    *,
    prerequisite_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    payload = Path(prerequisite_artifact).read_bytes()
    if _sha256(payload) != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError("TP4 fan-out prerequisite hash is invalid")
    prerequisite = load_tp4_fanout_prerequisite(
        prerequisite_artifact
    )
    local_hashes = _source_hashes(source_root)
    if {
        name: local_hashes[name]
        for name in prerequisite.source_file_sha256
    } != dict(prerequisite.source_file_sha256):
        raise ValueError(
            "TP4 fan-out source does not match prerequisite"
        )
    audit_tp4_fanout_source(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_prerequisite = (
        f"{remote_run_dir}/"
        "live_shared_memory_engine_ack_dispatch_preflight.json"
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
    _require_success(staged, "TP4 fan-out source staging")
    completed = command_runner(
        build_ssh_command([
            "bash",
            "-c",
            f"cat > {shlex.quote(remote_prerequisite)}",
        ]),
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require_success(completed, "TP4 fan-out prerequisite staging")
    method_specs = [
        (MODEL_RUNNER_SOURCE, "ModelRunner", method_name)
        for method_name in (
            "write_shm",
            "read_shm",
            "loop",
            "dispatch_command",
        )
    ] + [(
        LLM_ENGINE_SOURCE,
        "LLMEngine",
        "call_model_runner_acknowledged",
    )]
    script = "\n".join([
        "import ast,hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"prerequisite=pathlib.Path({remote_prerequisite!r})",
        f"names={list(SOURCE_FILES)!r}",
        f"specs={method_specs!r}",
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
        "payload={'source':source_hashes,'prerequisite':hashlib.sha256(prerequisite.read_bytes()).hexdigest(),'methods':methods}",
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
    _require_success(verified, "TP4 fan-out staged hashing")
    remote = json.loads(verified.stdout)
    if (
        remote.get("source") != local_hashes
        or remote.get("prerequisite") != PREREQUISITE_ARTIFACT_SHA256
        or remote.get("methods") != dict(METHOD_SOURCE_SHA256)
    ):
        raise ValueError("TP4 fan-out staged identity is invalid")
    return {
        "remote_source_dir": remote_source_dir,
        "remote_prerequisite_artifact": remote_prerequisite,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "prerequisite_artifact_sha256": remote["prerequisite"],
        "method_source_sha256": remote["methods"],
    }


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
        "method_source_sha256": dict(
            staged["method_source_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_tp4_fanout_preflight(
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
            f"local TP4 fan-out directory exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/tp4_shared_memory_fanout_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_tp4_shared_memory_fanout_preflight.py"
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
                "--prerequisite-artifact",
                staged["remote_prerequisite_artifact"],
                "--attempt-mode",
                mode,
                "--timeout-s",
                "4.0",
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, "TP4 fan-out attempt worker")
        row = json.loads(completed.stdout)
        validate_tp4_fanout_attempt_row(row)
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
    _require_success(finalized, "TP4 fan-out finalizer")
    record = json.loads(finalized.stdout)
    validate_tp4_fanout_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("TP4 fan-out source binding mismatch")
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
    _require_success(completed, "TP4 fan-out manifest publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "tp4_shared_memory_fanout_preflight.json",
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


def execute_remote_tp4_fanout_preflight(
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
    return run_remote_tp4_fanout_preflight(
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
    run.add_argument("--prerequisite-artifact", required=True)
    worker = subparsers.add_parser("internal-attempt-worker")
    worker.add_argument("--source-root", required=True)
    worker.add_argument("--prerequisite-artifact", required=True)
    worker.add_argument("--attempt-mode", choices=ATTEMPT_MODES, required=True)
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
        record = execute_remote_tp4_fanout_preflight(
            arguments.source_root,
            arguments.run_tag,
            prerequisite_artifact=arguments.prerequisite_artifact,
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-attempt-worker":
        row = run_tp4_attempt_worker(
            source_root=arguments.source_root,
            prerequisite_artifact=arguments.prerequisite_artifact,
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
        validate_tp4_fanout_preflight(record)
        print("TP4 shared-memory fan-out artifact validated")
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
