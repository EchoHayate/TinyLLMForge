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
import pickle
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
from types import MappingProxyType, ModuleType
import uuid


def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_ack_namespace():
    root = Path(__file__).resolve().parents[1]
    for name, path in (
        ("tinyvllm", root / "tinyvllm"),
        ("tinyvllm.engine", root / "tinyvllm/engine"),
    ):
        module = sys.modules.get(name)
        if module is None:
            module = ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module


engine_ack_gate = _load_sibling(
    "_qwen35_live_shm_engine_ack_prerequisite",
    "qwen35_real_binding_engine_ack_transport_preflight.py",
)
_install_ack_namespace()
ack_module = _load_sibling(
    "tinyvllm.engine.model_runner_command_ack",
    "../tinyvllm/engine/model_runner_command_ack.py",
)


PREREQUISITE_ARTIFACT_SHA256 = (
    "8aeb571c3d56641e747a0d5c5e66314efe6b35b73320cb49e0340c0fe5fd42fb"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "a041ebf7653e141dd96ebe31143ba00e5634c61c1a4bec68f17e7a7c6bba5cc8"
)
LLM_ENGINE_SOURCE = "tinyvllm/engine/llm_engine.py"
LLM_ENGINE_FILE_SHA256 = (
    "6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae"
)
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
MODEL_RUNNER_FILE_SHA256 = (
    "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
)
ACK_SOURCE = "tinyvllm/engine/model_runner_command_ack.py"
ACK_FILE_SHA256 = (
    "ca28babca5cc725d8c9bf0e3e057fa4b0cabfd847bf0c052c40876fbc148c61b"
)
METHOD_SOURCE_SHA256 = MappingProxyType({
    "write_shm": (
        "f9a377bf748d5be91a3c3722850e5e486f8e7dd8157e87d3dc6d692a60be6d76"
    ),
    "read_shm": (
        "1266b5d20b2978b655716f9ec8b58ce0a5644b9709164a23c18b85346170054a"
    ),
    "loop": (
        "342bac6d01606e4834e7ed77ef3e76d59b2fc3ea617afebe2c195912159dd2bb"
    ),
    "dispatch_command": (
        "9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342"
    ),
    "call_model_runner_acknowledged": (
        "6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d"
    ),
    "bind_qwen35_loaded_checkpoint_candidates": (
        "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c"
    ),
})
METHOD_ARGUMENTS = MappingProxyType({
    "write_shm": (("self", "envelope"), None, ()),
    "read_shm": (("self",), None, ()),
    "loop": (("self",), None, ()),
    "dispatch_command": (
        ("self", "method_name"),
        "args",
        ("requires_ack",),
    ),
    "call_model_runner_acknowledged": (
        ("self", "method_name"),
        "args",
        ("timeout_s",),
    ),
    "bind_qwen35_loaded_checkpoint_candidates": (
        ("self",),
        None,
        ("timeout_s",),
    ),
})
SOURCE_FILES = (
    *engine_ack_gate.SOURCE_FILES,
    "tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py",
)
PREREQUISITE_ROWS = engine_ack_gate.WORKER_MODES
SHARED_MEMORY_CAPACITY = 2**20
ATTEMPT_MODES = (
    "tp2_shm_success",
    "tp2_shm_worker_binding_error",
    "tp2_shm_worker_ack_exception",
    "tp2_shm_worker_exit_without_ack",
)
ROW_SCHEMA_VERSION = (
    "qwen35.live-shared-memory-engine-ack-dispatch-rank.v1"
)
SCHEMA_VERSION = (
    "qwen35.live-shared-memory-engine-ack-dispatch.v1"
)
REMOTE_TARGET = engine_ack_gate.REMOTE_TARGET
REMOTE_PYTHON = engine_ack_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = engine_ack_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-live-shared-memory-engine-ack-runs"
)
_source_tree_sha256 = engine_ack_gate._source_tree_sha256
_atomic_write_json = engine_ack_gate._atomic_write_json
validate_run_tag = engine_ack_gate.validate_run_tag
build_ssh_command = engine_ack_gate.build_ssh_command
_require_success = engine_ack_gate._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LiveSharedMemoryPrerequisite:
    artifact_sha256: str
    source_tree_sha256: str
    rows: tuple
    row_map: Mapping
    source_file_sha256: Mapping


def load_live_shared_memory_prerequisite(artifact):
    payload = Path(artifact).read_bytes()
    if _sha256(payload) != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError(
            "live shared-memory prerequisite hash is invalid"
        )
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "live shared-memory prerequisite JSON is invalid"
        ) from error
    engine_ack_gate.validate_engine_ack_transport_preflight(record)
    if (
        record.get("source_tree_sha256")
        != PREREQUISITE_SOURCE_TREE_SHA256
    ):
        raise ValueError(
            "live shared-memory prerequisite source tree is invalid"
        )
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != 54
        or tuple(hashes) != tuple(sorted(hashes))
    ):
        raise ValueError(
            "live shared-memory prerequisite source closure is invalid"
        )
    rows = record.get("rows")
    observed = tuple(row.get("mode") for row in rows or ())
    if observed != PREREQUISITE_ROWS:
        raise ValueError(
            "live shared-memory prerequisite rows are invalid"
        )
    row_map = {
        mode: MappingProxyType(row)
        for mode, row in zip(observed, rows)
    }
    return LiveSharedMemoryPrerequisite(
        artifact_sha256=PREREQUISITE_ARTIFACT_SHA256,
        source_tree_sha256=PREREQUISITE_SOURCE_TREE_SHA256,
        rows=observed,
        row_map=MappingProxyType(row_map),
        source_file_sha256=MappingProxyType(hashes),
    )


def _find_method(source, filename, class_name, method_name):
    tree = ast.parse(source, filename=filename)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == class_name
    ]
    if len(classes) != 1:
        raise ValueError(f"{class_name} class identity is invalid")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(f"{method_name} method identity is invalid")
    return methods[0]


def _compile_method(
    source,
    filename,
    class_name,
    method_name,
    globals_map,
):
    node = _find_method(
        source,
        filename,
        class_name,
        method_name,
    )
    segment = ast.get_source_segment(source, node)
    if (
        not isinstance(segment, str)
        or _sha256(segment.encode("utf-8"))
        != METHOD_SOURCE_SHA256[method_name]
    ):
        raise ValueError(f"{method_name} method hash is invalid")
    positional = tuple(
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
        )
    )
    vararg = (
        node.args.vararg.arg
        if node.args.vararg is not None
        else None
    )
    keyword_only = tuple(
        argument.arg for argument in node.args.kwonlyargs
    )
    if (
        positional,
        vararg,
        keyword_only,
    ) != METHOD_ARGUMENTS[method_name]:
        raise ValueError(f"{method_name} method arguments are invalid")
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = dict(globals_map)
    exec(
        compile(
            ast.fix_missing_locations(module),
            filename,
            "exec",
        ),
        namespace,
    )
    return namespace[method_name]


def load_frozen_live_shared_memory_methods(
    source_root,
    *,
    fingerprint_validator,
):
    root = Path(source_root)
    runner_payload = (root / MODEL_RUNNER_SOURCE).read_bytes()
    engine_payload = (root / LLM_ENGINE_SOURCE).read_bytes()
    ack_payload = (root / ACK_SOURCE).read_bytes()
    if _sha256(runner_payload) != MODEL_RUNNER_FILE_SHA256:
        raise ValueError("ModelRunner source hash is invalid")
    if _sha256(engine_payload) != LLM_ENGINE_FILE_SHA256:
        raise ValueError("LLMEngine source hash is invalid")
    if _sha256(ack_payload) != ACK_FILE_SHA256:
        raise ValueError("acknowledgement source hash is invalid")
    runner_source = runner_payload.decode("utf-8")
    engine_source = engine_payload.decode("utf-8")
    runner_globals = {
        "count": count,
        "pickle": pickle,
        "ModelRunnerCommandEnvelope": (
            ack_module.ModelRunnerCommandEnvelope
        ),
        "execute_acknowledged_command": (
            ack_module.execute_acknowledged_command
        ),
    }
    engine_globals = {
        "validate_qwen35_model_fingerprint": fingerprint_validator,
    }
    methods = {}
    for name in (
        "write_shm",
        "read_shm",
        "loop",
        "dispatch_command",
    ):
        methods[name] = _compile_method(
            runner_source,
            MODEL_RUNNER_SOURCE,
            "ModelRunner",
            name,
            runner_globals,
        )
    for name in (
        "call_model_runner_acknowledged",
        "bind_qwen35_loaded_checkpoint_candidates",
    ):
        methods[name] = _compile_method(
            engine_source,
            LLM_ENGINE_SOURCE,
            "LLMEngine",
            name,
            engine_globals,
        )
    return MappingProxyType(methods)


class _CountingEvent:
    def __init__(self, event):
        self._event = event
        self.set_count = 0
        self.wait_count = 0
        self.clear_count = 0

    def set(self):
        self.set_count += 1
        return self._event.set()

    def wait(self):
        self.wait_count += 1
        return self._event.wait()

    def clear(self):
        self.clear_count += 1
        return self._event.clear()


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


def _envelope_to_dict(envelope):
    return {
        "command_id": envelope.command_id,
        "method_name": envelope.method_name,
        "args": list(envelope.args),
        "requires_ack": envelope.requires_ack,
    }


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


def _validate_fingerprint(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ValueError(
            "model_fingerprint must be a lowercase SHA256"
        )
    return value


def _increment(counter):
    with counter.get_lock():
        counter.value += 1


def _live_shared_memory_worker(
    source_root,
    shared_memory_name,
    event,
    ack_sender,
    ready_sender,
    mode,
    worker_row,
    read_count,
    executor_count,
):
    shared = SharedMemory(name=shared_memory_name)
    try:
        methods = load_frozen_live_shared_memory_methods(
            source_root,
            fingerprint_validator=_validate_fingerprint,
        )

        class WorkerShell:
            world_size = 2
            rank = 1

            def read_shm(self):
                _increment(read_count)
                return methods["read_shm"](self)

            def bind_published_qwen35_loaded_checkpoint_candidate(
                self,
            ):
                if mode == "tp2_shm_worker_ack_exception":
                    raise RuntimeError(
                        "injected worker acknowledgement exception"
                    )
                if mode == "tp2_shm_worker_exit_without_ack":
                    raise SystemExit(9)
                return dict(worker_row)

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
        ready_sender.send(
            {
                "process_id": os.getpid(),
                "shared_memory_name": shared.name,
            }
        )
        ready_sender.close()
        worker = WorkerShell()
        worker.shm = shared
        worker.ack_sender = ack_sender
        worker.event = event
        worker._command_ids = count()
        methods["loop"](worker)
    finally:
        shared.close()
        ack_sender.close()


def _attempt_rows(prerequisite, mode):
    local_row = prerequisite.row_map["tp2_success"][
        "binding_rows"
    ][0]
    if mode == "tp2_shm_worker_binding_error":
        worker_row = prerequisite.row_map[
            "tp2_worker_binding_error"
        ]
        detail = worker_row["error_detail"]
        expected = (
            "RuntimeError: loaded checkpoint candidate binding "
            "failed: rank=1, detail="
        )
        if not detail.startswith(expected):
            raise ValueError(
                "worker binding prerequisite detail is invalid"
            )
        worker_row = {
            "participant_id": 1,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "error",
            "model_fingerprint": "",
            "layout_fingerprint": "",
            "dtype": "",
            "detail": detail[len(expected):],
        }
    else:
        worker_row = prerequisite.row_map["tp2_success"][
            "binding_rows"
        ][1]
    return dict(local_row), dict(worker_row)


def execute_live_shared_memory_engine_ack_attempt(
    *,
    source_root,
    prerequisite_artifact,
    mode,
    timeout_s,
    name_prefix,
):
    if mode not in ATTEMPT_MODES:
        raise ValueError("live shared-memory attempt mode is invalid")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or float(timeout_s) <= 0
    ):
        raise ValueError("timeout_s must be positive")
    timeout_s = float(timeout_s)
    prerequisite = load_live_shared_memory_prerequisite(
        prerequisite_artifact
    )
    local_row, worker_row = _attempt_rows(prerequisite, mode)
    methods = load_frozen_live_shared_memory_methods(
        source_root,
        fingerprint_validator=_validate_fingerprint,
    )
    context = multiprocessing.get_context("fork")
    event = _SharedCountingEvent(context)
    ack_receiver, ack_sender = context.Pipe(duplex=False)
    ready_receiver, ready_sender = context.Pipe(duplex=False)
    read_count = context.Value("i", 0)
    executor_count = context.Value("i", 0)
    shared_memory_name = _shared_memory_name(name_prefix)
    shared = SharedMemory(
        name=shared_memory_name,
        create=True,
        size=SHARED_MEMORY_CAPACITY,
    )
    process = context.Process(
        target=_live_shared_memory_worker,
        args=(
            os.fspath(Path(source_root)),
            shared_memory_name,
            event,
            ack_sender,
            ready_sender,
            mode,
            worker_row,
            read_count,
            executor_count,
        ),
    )
    dispatch_envelopes = []
    write_payload_bytes = []
    dispatch_count = 0
    binding_dispatch_count = 0
    write_count = 0
    collector_call_count = 0
    exit_envelope_sent = False
    child_ready = False
    child_process_id = None
    error_detail = ""
    completion_committed = False
    completion_configuration = None
    binding_rows = None
    repeat_zero_binding_dispatch = False
    collector = None
    receiver = _RecordingReceiver(ack_receiver)
    segment_unlinked = False
    post_unlink_attach_failed = False
    child_collected = False
    try:
        process.start()
        ack_sender.close()
        ready_sender.close()
        if not ready_receiver.poll(timeout_s):
            raise TimeoutError(
                "timed out waiting for shared-memory worker readiness"
            )
        try:
            ready = ready_receiver.recv()
        except EOFError as error:
            raise RuntimeError(
                "shared-memory worker exited before readiness"
            ) from error
        child_ready = True
        child_process_id = ready["process_id"]
        if ready["shared_memory_name"] != shared_memory_name:
            raise RuntimeError(
                "shared-memory worker attached to the wrong segment"
            )
        ready_receiver.close()
        collector = ack_module.ModelRunnerCommandAckCollector(
            ((1, receiver),)
        )

        class RankZeroShell:
            world_size = 2
            rank = 0

            def write_shm(self, envelope):
                nonlocal write_count
                write_count += 1
                methods["write_shm"](self, envelope)
                payload_bytes = int.from_bytes(
                    self.shm.buf[0:4],
                    "little",
                )
                write_payload_bytes.append(payload_bytes)

            def dispatch_command(
                self,
                method_name,
                *args,
                requires_ack,
            ):
                nonlocal dispatch_count
                nonlocal binding_dispatch_count
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
                dispatch_envelopes.append(envelope)
                return envelope

            def bind_published_qwen35_loaded_checkpoint_candidate(
                self,
            ):
                return dict(local_row)

        rank_zero = RankZeroShell()
        rank_zero.shm = shared
        rank_zero.event = [event]
        rank_zero._command_ids = count()

        class EngineShell:
            def _is_worker_rank_alive(self, rank):
                if rank != 1:
                    raise ValueError("worker rank is invalid")
                return process.is_alive()

            def call_model_runner_acknowledged(
                self,
                method_name,
                *args,
                timeout_s,
            ):
                nonlocal collector_call_count
                before = len(receiver.values)
                try:
                    return methods["call_model_runner_acknowledged"](
                        self,
                        method_name,
                        *args,
                        timeout_s=timeout_s,
                    )
                finally:
                    if len(receiver.values) > before or (
                        collector is not None
                        and collector.poisoned
                    ):
                        collector_call_count += 1

        engine = EngineShell()
        engine.model_runner = rank_zero
        engine.model_runner_ack_collector = collector
        engine.qwen35_loaded_checkpoint_candidate_binding_configuration = (
            None
        )
        engine.qwen35_loaded_checkpoint_candidate_binding_rows = None
        try:
            result = methods[
                "bind_qwen35_loaded_checkpoint_candidates"
            ](
                engine,
                timeout_s=timeout_s,
            )
            binding_rows = [dict(row) for row in result]
            completion_committed = True
            completion_configuration = list(
                engine.qwen35_loaded_checkpoint_candidate_binding_configuration
            )
            before = binding_dispatch_count
            repeated = methods[
                "bind_qwen35_loaded_checkpoint_candidates"
            ](
                engine,
                timeout_s=timeout_s,
            )
            repeat_zero_binding_dispatch = (
                repeated is result
                and binding_dispatch_count == before
            )
        except (RuntimeError, TimeoutError) as error:
            error_detail = f"{type(error).__name__}: {error}"
        if mode != "tp2_shm_worker_exit_without_ack":
            rank_zero.dispatch_command(
                "exit",
                requires_ack=False,
            )
            exit_envelope_sent = True
        process.join(timeout=timeout_s)
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout_s)
            raise RuntimeError(
                "shared-memory worker did not terminate"
            )
        child_collected = True
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout_s)
        child_collected = not process.is_alive()
        try:
            ready_receiver.close()
        except OSError:
            pass
        receiver.close()
        shared.close()
        shared.unlink()
        segment_unlinked = True
    try:
        probe = SharedMemory(name=shared_memory_name)
    except FileNotFoundError:
        post_unlink_attach_failed = True
    else:
        probe.close()
        raise RuntimeError(
            "shared-memory segment remains attachable"
        )
    acknowledgement = (
        receiver.values[-1]
        if receiver.values
        else None
    )
    return {
        "status": "PASS",
        "mode": mode,
        "process_id": os.getpid(),
        "child_process_id": child_process_id,
        "child_ready": child_ready,
        "child_exitcode": process.exitcode,
        "child_collected": child_collected,
        "shared_memory_name": shared_memory_name,
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "segment_unlinked": segment_unlinked,
        "post_unlink_attach_failed": post_unlink_attach_failed,
        "dispatch_count": dispatch_count,
        "binding_dispatch_count": binding_dispatch_count,
        "write_count": write_count,
        "read_count": read_count.value,
        "executor_count": executor_count.value,
        "collector_call_count": collector_call_count,
        "event_set_count": event.set_count,
        "child_event_wait_count": event.wait_count,
        "child_event_clear_count": event.clear_count,
        "write_payload_bytes": write_payload_bytes,
        "envelopes": [
            _envelope_to_dict(envelope)
            for envelope in dispatch_envelopes
        ],
        "acknowledgement_status": (
            acknowledgement.status
            if acknowledgement is not None
            else "absent"
        ),
        "acknowledgement_error_type": (
            acknowledgement.error_type
            if acknowledgement is not None
            else ""
        ),
        "acknowledgement_error_detail": (
            acknowledgement.error_detail
            if acknowledgement is not None
            else ""
        ),
        "collector_poisoned": bool(
            collector is not None and collector.poisoned
        ),
        "completion_committed": completion_committed,
        "completion_configuration": completion_configuration,
        "binding_rows": binding_rows,
        "repeat_zero_binding_dispatch": (
            repeat_zero_binding_dispatch
        ),
        "exit_envelope_sent": exit_envelope_sent,
        "error_detail": error_detail,
    }


def run_live_shared_memory_attempt_worker(
    *,
    source_root,
    prerequisite_artifact,
    mode,
    timeout_s,
    observed_user,
    observed_hostname,
    process_id,
):
    attempt = execute_live_shared_memory_engine_ack_attempt(
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
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        **attempt,
        "process_id": process_id,
    }
    validate_live_shared_memory_attempt_row(row)
    return row


def validate_live_shared_memory_attempt_row(row):
    mode = row.get("mode")
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or mode not in ATTEMPT_MODES
    ):
        raise ValueError(
            "live shared-memory attempt row schema is invalid"
        )
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "child_ready": True,
        "child_collected": True,
        "segment_unlinked": True,
        "post_unlink_attach_failed": True,
        "binding_dispatch_count": 1,
        "collector_call_count": 1,
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            raise ValueError(
                f"live shared-memory {name} is invalid"
            )
    for name in ("process_id", "child_process_id"):
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"live shared-memory {name} is invalid"
            )
    if row["process_id"] == row["child_process_id"]:
        raise ValueError(
            "live shared-memory child PID aliases outer PID"
        )
    name = row.get("shared_memory_name")
    if (
        not isinstance(name, str)
        or not name
        or name == "tinyvllm"
        or len(name) > 30
    ):
        raise ValueError(
            "live shared-memory name is invalid"
        )
    worker_death = mode == "tp2_shm_worker_exit_without_ack"
    success = mode == "tp2_shm_success"
    expected_count = 1 if worker_death else 2
    counts = {
        "dispatch_count": expected_count,
        "write_count": expected_count,
        "read_count": expected_count,
        "executor_count": expected_count,
        "event_set_count": expected_count,
        "child_event_wait_count": expected_count,
        "child_event_clear_count": expected_count,
        "exit_envelope_sent": not worker_death,
        "child_exitcode": 9 if worker_death else 0,
        "completion_committed": success,
        "repeat_zero_binding_dispatch": success,
        "collector_poisoned": mode in {
            "tp2_shm_worker_ack_exception",
            "tp2_shm_worker_exit_without_ack",
        },
    }
    for name, expected in counts.items():
        if row.get(name) != expected:
            raise ValueError(
                f"live shared-memory {name} is invalid"
            )
    expected_envelopes = [{
        "command_id": 0,
        "method_name": (
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ),
        "args": [],
        "requires_ack": True,
    }]
    if not worker_death:
        expected_envelopes.append({
            "command_id": 1,
            "method_name": "exit",
            "args": [],
            "requires_ack": False,
        })
    if row.get("envelopes") != expected_envelopes:
        raise ValueError(
            "live shared-memory envelopes are invalid"
        )
    payload_bytes = row.get("write_payload_bytes")
    if (
        not isinstance(payload_bytes, list)
        or len(payload_bytes) != expected_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value + 4 > SHARED_MEMORY_CAPACITY
            for value in payload_bytes
        )
    ):
        raise ValueError(
            "live shared-memory payload bytes are invalid"
        )
    expected_ack = {
        "tp2_shm_success": "ok",
        "tp2_shm_worker_binding_error": "ok",
        "tp2_shm_worker_ack_exception": "error",
        "tp2_shm_worker_exit_without_ack": "absent",
    }[mode]
    if row.get("acknowledgement_status") != expected_ack:
        raise ValueError(
            "live shared-memory acknowledgement status is invalid"
        )
    if success:
        rows = row.get("binding_rows")
        configuration = row.get("completion_configuration")
        if (
            not isinstance(rows, list)
            or len(rows) != 2
            or [item.get("participant_id") for item in rows]
            != [0, 1]
            or not isinstance(configuration, list)
            or len(configuration) != 4
            or row.get("error_detail")
        ):
            raise ValueError(
                "live shared-memory completion is invalid"
            )
    elif (
        row.get("binding_rows") is not None
        or row.get("completion_configuration") is not None
        or not isinstance(row.get("error_detail"), str)
        or not row["error_detail"]
    ):
        raise ValueError(
            "live shared-memory failure state is invalid"
        )
    required_error = {
        "tp2_shm_worker_binding_error": (
            "loaded checkpoint candidate binding failed: rank=1"
        ),
        "tp2_shm_worker_ack_exception": (
            "injected worker acknowledgement exception"
        ),
        "tp2_shm_worker_exit_without_ack": "acknowledgement",
    }.get(mode)
    if (
        required_error is not None
        and required_error not in row["error_detail"]
    ):
        raise ValueError(
            "live shared-memory error detail is invalid"
        )
    return row


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(
                "missing live shared-memory source: " + name
            )
        hashes[name] = _sha256(path.read_bytes())
    return dict(sorted(hashes.items()))


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(
                    "missing live shared-memory source: " + name
                )
            info = archive.gettarinfo(
                os.fspath(path),
                arcname=name,
            )
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def audit_live_shared_memory_source(source_root):
    path = Path(source_root) / (
        "tools/"
        "qwen35_live_shared_memory_engine_ack_dispatch_preflight.py"
    )
    tree = ast.parse(
        path.read_text(),
        filename=os.fspath(path),
    )
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]

    def named(name):
        return sum(
            isinstance(node.func, ast.Name)
            and node.func.id == name
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
        "forward_call_count": (
            named("forward") + attribute("forward")
        ),
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
        raise ValueError(
            f"live shared-memory static audit is invalid: {audit!r}"
        )
    return audit


def validate_live_shared_memory_preflight(record):
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(
                f"live shared-memory {name} is invalid"
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
            "live shared-memory source closure is invalid"
        )
    rows = record.get("rows")
    if [
        row.get("mode") for row in rows or ()
    ] != list(ATTEMPT_MODES):
        raise ValueError(
            "live shared-memory attempt rows are invalid"
        )
    for row in rows:
        validate_live_shared_memory_attempt_row(row)
    outer_ids = {row["process_id"] for row in rows}
    child_ids = {row["child_process_id"] for row in rows}
    names = {row["shared_memory_name"] for row in rows}
    if (
        len(outer_ids) != 4
        or len(child_ids) != 4
        or len(names) != 4
        or not outer_ids.isdisjoint(child_ids)
    ):
        raise ValueError(
            "live shared-memory attempt identities are invalid"
        )
    return record


def _aggregate(rows, source_root):
    if (
        not isinstance(rows, list)
        or len(rows) != len(ATTEMPT_MODES)
    ):
        raise ValueError(
            "live shared-memory attempt rows are incomplete"
        )
    hashes = _source_hashes(source_root)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_live_shared_memory_preflight(record)
    return record


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
        raise ValueError(
            "live shared-memory prerequisite hash is invalid"
        )
    prerequisite = load_live_shared_memory_prerequisite(
        prerequisite_artifact
    )
    local_hashes = _source_hashes(source_root)
    if {
        name: local_hashes[name]
        for name in prerequisite.source_file_sha256
    } != dict(prerequisite.source_file_sha256):
        raise ValueError(
            "live shared-memory source does not match prerequisite"
        )
    audit_live_shared_memory_source(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_prerequisite = (
        f"{remote_run_dir}/engine_ack_transport_preflight.json"
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
    _require_success(staged, "live shared-memory source staging")
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
    _require_success(
        completed,
        "live shared-memory prerequisite staging",
    )
    method_specs = [
        (
            MODEL_RUNNER_SOURCE,
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
            LLM_ENGINE_SOURCE,
            "LLMEngine",
            method_name,
        )
        for method_name in (
            "call_model_runner_acknowledged",
            "bind_qwen35_loaded_checkpoint_candidates",
        )
    ]
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
    _require_success(
        verified,
        "live shared-memory staged hashing",
    )
    remote = json.loads(verified.stdout)
    if (
        remote.get("source") != local_hashes
        or remote.get("prerequisite")
        != PREREQUISITE_ARTIFACT_SHA256
        or remote.get("methods") != dict(METHOD_SOURCE_SHA256)
    ):
        raise ValueError(
            "live shared-memory staged identity is invalid"
        )
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


def run_remote_live_shared_memory_preflight(
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
            "local live shared-memory directory exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/"
        "live_shared_memory_engine_ack_dispatch_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_live_shared_memory_engine_ack_dispatch_preflight.py"
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
                "2.0",
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(
            completed,
            "live shared-memory attempt worker",
        )
        row = json.loads(completed.stdout)
        validate_live_shared_memory_attempt_row(row)
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
    _require_success(
        finalized,
        "live shared-memory finalizer",
    )
    record = json.loads(finalized.stdout)
    validate_live_shared_memory_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError(
            "live shared-memory source binding mismatch"
        )
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'live_shared_memory_engine_ack_dispatch_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'record':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "record": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(
        round_trip,
        "live shared-memory artifact round trip",
    )
    if json.loads(round_trip.stdout) != {
        "record": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError(
            "live shared-memory artifact round-trip mismatch"
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
            / "live_shared_memory_engine_ack_dispatch_preflight.json",
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


def execute_remote_live_shared_memory_preflight(
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
    return run_remote_live_shared_memory_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def execute_live_shared_memory_codec_round_trip(
    *,
    source_root,
    name_prefix,
):
    methods = load_frozen_live_shared_memory_methods(
        source_root,
        fingerprint_validator=lambda value: value,
    )
    context = multiprocessing.get_context("spawn")
    event = _CountingEvent(context.Event())
    name = _shared_memory_name(name_prefix)
    shared = SharedMemory(
        name=name,
        create=True,
        size=SHARED_MEMORY_CAPACITY,
    )
    attached = None
    segment_unlinked = False
    post_unlink_attach_failed = False
    try:
        envelope = ack_module.ModelRunnerCommandEnvelope(
            command_id=7,
            method_name=(
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ),
            args=(),
            requires_ack=True,
        )
        rank_zero = type("RankZeroShell", (), {})()
        rank_zero.world_size = 2
        rank_zero.rank = 0
        rank_zero.shm = shared
        rank_zero.event = [event]
        methods["write_shm"](rank_zero, envelope)
        payload_bytes = int.from_bytes(
            shared.buf[0:4],
            "little",
        )
        attached = SharedMemory(name=name)
        worker = type("WorkerShell", (), {})()
        worker.world_size = 2
        worker.rank = 1
        worker.shm = attached
        worker.event = event
        worker._command_ids = count()
        observed = methods["read_shm"](worker)
        if observed != envelope:
            raise ValueError("shared-memory envelope round trip failed")
    finally:
        if attached is not None:
            attached.close()
        shared.close()
        shared.unlink()
        segment_unlinked = True
    try:
        probe = SharedMemory(name=name)
    except FileNotFoundError:
        post_unlink_attach_failed = True
    else:
        probe.close()
        raise ValueError("shared-memory segment remains attachable")
    return {
        "status": "PASS",
        "shared_memory_name": name,
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "payload_bytes": payload_bytes,
        "envelope": _envelope_to_dict(envelope),
        "event_set_count": event.set_count,
        "event_wait_count": event.wait_count,
        "event_clear_count": event.clear_count,
        "segment_unlinked": segment_unlinked,
        "post_unlink_attach_failed": post_unlink_attach_failed,
    }


def _attempt_worker_main(arguments):
    row = run_live_shared_memory_attempt_worker(
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


def _finalize_main(arguments):
    output = Path(arguments.output)
    if output.exists():
        raise ValueError(
            "live shared-memory output already exists"
        )
    payload = json.load(sys.stdin)
    record = _aggregate(
        payload.get("rows"),
        arguments.source_root,
    )
    _atomic_write_json(output, record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-tag", required=True)
    run_parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    run_parser.add_argument(
        "--prerequisite-artifact",
        required=True,
    )
    worker_parser = subparsers.add_parser(
        "internal-attempt-worker"
    )
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument(
        "--prerequisite-artifact",
        required=True,
    )
    worker_parser.add_argument(
        "--attempt-mode",
        choices=ATTEMPT_MODES,
        required=True,
    )
    worker_parser.add_argument(
        "--timeout-s",
        type=float,
        required=True,
    )
    finalize_parser = subparsers.add_parser("internal-finalize")
    finalize_parser.add_argument("--source-root", required=True)
    finalize_parser.add_argument("--output", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "internal-attempt-worker":
        return _attempt_worker_main(arguments)
    if arguments.command == "internal-finalize":
        return _finalize_main(arguments)
    if arguments.command == "validate":
        record = json.loads(Path(arguments.artifact).read_text())
        validate_live_shared_memory_preflight(record)
    else:
        record = execute_remote_live_shared_memory_preflight(
            arguments.source_root,
            arguments.run_tag,
            prerequisite_artifact=arguments.prerequisite_artifact,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
