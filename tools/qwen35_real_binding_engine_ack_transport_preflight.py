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
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
from types import MappingProxyType, ModuleType


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


binding_gate = _load_sibling(
    "_qwen35_engine_ack_transport_binding_base",
    "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py",
)
_install_ack_namespace()
ack_module = _load_sibling(
    "tinyvllm.engine.model_runner_command_ack",
    "../tinyvllm/engine/model_runner_command_ack.py",
)


PREREQUISITE_ARTIFACT_SHA256 = (
    "79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785"
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
    "call_model_runner_acknowledged": (
        "6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d"
    ),
    "bind_qwen35_loaded_checkpoint_candidates": (
        "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c"
    ),
    "dispatch_command": (
        "9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342"
    ),
})
METHOD_ARGUMENTS = MappingProxyType({
    "call_model_runner_acknowledged": (
        "self",
        "method_name",
        "args",
        "timeout_s",
    ),
    "bind_qwen35_loaded_checkpoint_candidates": (
        "self",
        "timeout_s",
    ),
    "dispatch_command": (
        "self",
        "method_name",
        "args",
        "requires_ack",
    ),
})
PREREQUISITE_ROWS = (
    (1, 0, "success"),
    (1, 0, "injected_bridge_conflict"),
    (2, 0, "success"),
    (2, 0, "injected_bridge_conflict"),
    (2, 1, "success"),
    (2, 1, "injected_bridge_conflict"),
)
SOURCE_FILES = (
    *binding_gate.SOURCE_FILES,
    LLM_ENGINE_SOURCE,
    ACK_SOURCE,
    "tools/qwen35_real_binding_engine_ack_transport_preflight.py",
)
WORKER_MODES = (
    "tp1_success",
    "tp1_local_binding_error",
    "tp2_success",
    "tp2_worker_binding_error",
    "tp2_worker_ack_exception",
    "tp2_worker_exit_without_ack",
)
ROW_SCHEMA_VERSION = "qwen35.real-binding-engine-ack-rank.v1"
SCHEMA_VERSION = "qwen35.real-binding-engine-ack.v1"
REMOTE_TARGET = binding_gate.REMOTE_TARGET
REMOTE_PYTHON = binding_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = binding_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-binding-engine-ack-runs"
)
_source_tree_sha256 = binding_gate._source_tree_sha256
_atomic_write_json = binding_gate._atomic_write_json
validate_run_tag = binding_gate.validate_run_tag
build_ssh_command = binding_gate.build_ssh_command
_require_success = binding_gate._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EngineAckTransportPrerequisite:
    artifact_sha256: str
    source_tree_sha256: str
    rows: tuple
    row_map: Mapping
    source_file_sha256: Mapping


def load_engine_ack_transport_prerequisite(artifact):
    payload = Path(artifact).read_bytes()
    if _sha256(payload) != PREREQUISITE_ARTIFACT_SHA256:
        raise ValueError(
            "Engine acknowledgement transport prerequisite hash "
            "is invalid"
        )
    try:
        record = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Engine acknowledgement transport prerequisite JSON "
            "is invalid"
        ) from error
    binding_gate.validate_model_runner_published_binding_preflight(
        record
    )
    if (
        record.get("source_tree_sha256")
        != PREREQUISITE_SOURCE_TREE_SHA256
    ):
        raise ValueError(
            "Engine acknowledgement transport prerequisite source "
            "tree is invalid"
        )
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != 51
        or tuple(hashes) != tuple(sorted(hashes))
    ):
        raise ValueError(
            "Engine acknowledgement transport prerequisite source "
            "closure is invalid"
        )
    rows = record.get("rows")
    observed = tuple(
        (row.get("tp_size"), row.get("tp_rank"), row.get("mode"))
        for row in rows
    )
    if observed != PREREQUISITE_ROWS:
        raise ValueError(
            "Engine acknowledgement transport prerequisite rows "
            "are invalid"
        )
    if len({row.get("process_id") for row in rows}) != 6:
        raise ValueError(
            "Engine acknowledgement transport prerequisite process "
            "IDs are invalid"
        )
    row_map = {
        key: MappingProxyType(row)
        for key, row in zip(observed, rows)
    }
    return EngineAckTransportPrerequisite(
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
        raise ValueError(f"{class_name} class count is invalid")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    ]
    if len(methods) != 1:
        raise ValueError(f"{class_name}.{method_name} count is invalid")
    return methods[0]


def _argument_contract(method):
    positional = tuple(
        argument.arg
        for argument in (
            *method.args.posonlyargs,
            *method.args.args,
        )
    )
    if method.args.vararg is not None:
        positional += (method.args.vararg.arg,)
    keyword_only = tuple(
        argument.arg for argument in method.args.kwonlyargs
    )
    return positional + keyword_only


def load_frozen_engine_ack_transport_methods(
    source_root,
    *,
    fingerprint_validator,
):
    if not callable(fingerprint_validator):
        raise ValueError("fingerprint_validator must be callable")
    root = Path(source_root)
    engine_payload = (root / LLM_ENGINE_SOURCE).read_bytes()
    runner_payload = (root / MODEL_RUNNER_SOURCE).read_bytes()
    ack_payload = (root / ACK_SOURCE).read_bytes()
    if _sha256(engine_payload) != LLM_ENGINE_FILE_SHA256:
        raise ValueError("LLMEngine source hash is invalid")
    if _sha256(runner_payload) != MODEL_RUNNER_FILE_SHA256:
        raise ValueError("ModelRunner source hash is invalid")
    if _sha256(ack_payload) != ACK_FILE_SHA256:
        raise ValueError("acknowledgement source hash is invalid")
    sources = {
        "call_model_runner_acknowledged": (
            engine_payload.decode("utf-8"),
            LLM_ENGINE_SOURCE,
            "LLMEngine",
        ),
        "bind_qwen35_loaded_checkpoint_candidates": (
            engine_payload.decode("utf-8"),
            LLM_ENGINE_SOURCE,
            "LLMEngine",
        ),
        "dispatch_command": (
            runner_payload.decode("utf-8"),
            MODEL_RUNNER_SOURCE,
            "ModelRunner",
        ),
    }
    nodes = {}
    for name, (source, filename, class_name) in sources.items():
        method = _find_method(
            source,
            filename,
            class_name,
            name,
        )
        segment = ast.get_source_segment(source, method)
        if (
            segment is None
            or _sha256(segment.encode("utf-8"))
            != METHOD_SOURCE_SHA256[name]
        ):
            raise ValueError(f"{name} method source hash is invalid")
        if _argument_contract(method) != METHOD_ARGUMENTS[name]:
            raise ValueError(f"{name} method arguments are invalid")
        method.decorator_list = []
        nodes[name] = method
    engine_module = ast.Module(
        body=[
            nodes["call_model_runner_acknowledged"],
            nodes["bind_qwen35_loaded_checkpoint_candidates"],
        ],
        type_ignores=[],
    )
    engine_namespace = {
        "validate_qwen35_model_fingerprint": fingerprint_validator,
    }
    exec(
        compile(
            ast.fix_missing_locations(engine_module),
            LLM_ENGINE_SOURCE,
            "exec",
        ),
        engine_namespace,
    )
    runner_module = ast.Module(
        body=[nodes["dispatch_command"]],
        type_ignores=[],
    )
    runner_namespace = {
        "ModelRunnerCommandEnvelope": (
            ack_module.ModelRunnerCommandEnvelope
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(runner_module),
            MODEL_RUNNER_SOURCE,
            "exec",
        ),
        runner_namespace,
    )
    return MappingProxyType({
        "call_model_runner_acknowledged": engine_namespace[
            "call_model_runner_acknowledged"
        ],
        "bind_qwen35_loaded_checkpoint_candidates": engine_namespace[
            "bind_qwen35_loaded_checkpoint_candidates"
        ],
        "dispatch_command": runner_namespace["dispatch_command"],
    })


def _validate_model_fingerprint(value):
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


class _WorkerBindingTarget:

    def __init__(self, row, *, raise_error=False):
        self.row = dict(row)
        self.raise_error = bool(raise_error)

    def bind_published_qwen35_loaded_checkpoint_candidate(self):
        if self.raise_error:
            raise RuntimeError(
                "injected worker acknowledgement exception"
            )
        return dict(self.row)


def _engine_ack_transport_child(
    command_receiver,
    acknowledgement_sender,
    received_count,
    worker_row,
    mode,
):
    envelope = command_receiver.recv()
    with received_count.get_lock():
        received_count.value += 1
    if mode == "tp2_worker_exit_without_ack":
        return
    target = _WorkerBindingTarget(
        worker_row,
        raise_error=(mode == "tp2_worker_ack_exception"),
    )
    ack_module.execute_acknowledged_command(
        envelope,
        rank=1,
        target=target,
        send_ack=acknowledgement_sender.send,
    )


class _RecordingReceiver:

    def __init__(self, connection):
        self.connection = connection
        self.received = []

    def poll(self, timeout):
        return self.connection.poll(timeout)

    def recv(self):
        value = self.connection.recv()
        self.received.append(value)
        return value

    def close(self):
        self.connection.close()


class _RecordingCollector:

    def __init__(self, collector):
        self.collector = collector
        self.collect_calls = []

    @property
    def poisoned(self):
        return self.collector.poisoned

    def poison(self, reason):
        return self.collector.poison(reason)

    def collect(
        self,
        command_id,
        *,
        expected_ranks,
        timeout_s,
        is_rank_alive,
    ):
        self.collect_calls.append({
            "command_id": command_id,
            "expected_ranks": tuple(expected_ranks),
            "timeout_s": float(timeout_s),
        })
        return self.collector.collect(
            command_id,
            expected_ranks=expected_ranks,
            timeout_s=timeout_s,
            is_rank_alive=is_rank_alive,
        )


def _envelope_row(envelope):
    return {
        "command_id": envelope.command_id,
        "method_name": envelope.method_name,
        "args": list(envelope.args),
        "requires_ack": envelope.requires_ack,
    }


def execute_engine_ack_transport_attempt(
    *,
    source_root,
    mode,
    local_row,
    worker_row,
    timeout_s,
):
    modes = {
        "tp1_success",
        "tp1_local_binding_error",
        "tp2_success",
        "tp2_worker_binding_error",
        "tp2_worker_ack_exception",
        "tp2_worker_exit_without_ack",
    }
    if mode not in modes:
        raise ValueError(
            "Engine acknowledgement transport mode is invalid"
        )
    if not isinstance(local_row, Mapping):
        raise ValueError("local_row must be a mapping")
    if mode.startswith("tp2") and not isinstance(
        worker_row,
        Mapping,
    ):
        raise ValueError("worker_row must be a mapping for TP2")
    methods = load_frozen_engine_ack_transport_methods(
        source_root,
        fingerprint_validator=_validate_model_fingerprint,
    )
    context = multiprocessing.get_context("fork")
    command_receiver = None
    command_sender = None
    acknowledgement_receiver = None
    acknowledgement_sender = None
    recording_receiver = None
    recording_collector = None
    child = None
    received_count = None
    envelopes = []
    dispatch_count = 0
    child_collected = True

    class Runner:
        rank = 0
        _command_ids = count()

        def __init__(self, world_size):
            self.world_size = world_size

        def write_shm(self, envelope):
            nonlocal dispatch_count
            dispatch_count += 1
            envelopes.append(envelope)
            command_sender.send(envelope)

        def dispatch_command(
            self,
            method_name,
            *args,
            requires_ack,
        ):
            return methods["dispatch_command"](
                self,
                method_name,
                *args,
                requires_ack=requires_ack,
            )

        def bind_published_qwen35_loaded_checkpoint_candidate(self):
            return dict(local_row)

    world_size = 2 if mode.startswith("tp2") else 1
    runner = Runner(world_size)
    if world_size == 2:
        command_receiver, command_sender = context.Pipe(
            duplex=False
        )
        raw_receiver, acknowledgement_sender = context.Pipe(
            duplex=False
        )
        recording_receiver = _RecordingReceiver(raw_receiver)
        received_count = context.Value("i", 0)
        child = context.Process(
            target=_engine_ack_transport_child,
            args=(
                command_receiver,
                acknowledgement_sender,
                received_count,
                dict(worker_row),
                mode,
            ),
        )
        child.start()
        command_receiver.close()
        acknowledgement_sender.close()
        collector = ack_module.ModelRunnerCommandAckCollector(
            ((1, recording_receiver),)
        )
        recording_collector = _RecordingCollector(collector)
    engine = type("_EngineShell", (), {})()
    engine.model_runner = runner
    engine.model_runner_ack_collector = recording_collector
    engine.qwen35_loaded_checkpoint_candidate_binding_configuration = (
        None
    )
    engine.qwen35_loaded_checkpoint_candidate_binding_rows = None
    engine._is_worker_rank_alive = (
        lambda rank: (
            rank == 1
            and child is not None
            and child.is_alive()
        )
    )
    engine.call_model_runner_acknowledged = (
        lambda method_name, *args, timeout_s: methods[
            "call_model_runner_acknowledged"
        ](
            engine,
            method_name,
            *args,
            timeout_s=timeout_s,
        )
    )

    def invoke_binding():
        return methods[
            "bind_qwen35_loaded_checkpoint_candidates"
        ](
            engine,
            timeout_s=timeout_s,
        )

    result_rows = None
    error_detail = ""
    repeat_zero_dispatch = False
    try:
        try:
            result_rows = invoke_binding()
        except Exception as error:
            error_detail = f"{type(error).__name__}: {error}"
        if mode in {"tp1_success", "tp2_success"}:
            if error_detail:
                raise RuntimeError(
                    "Engine acknowledgement transport success failed: "
                    + error_detail
                )
            previous_dispatch_count = dispatch_count
            previous_collect_count = (
                len(recording_collector.collect_calls)
                if recording_collector is not None
                else 0
            )
            repeated = invoke_binding()
            repeat_zero_dispatch = (
                repeated is result_rows
                and dispatch_count == previous_dispatch_count
                and (
                    recording_collector is None
                    or len(recording_collector.collect_calls)
                    == previous_collect_count
                )
            )
    finally:
        if command_sender is not None:
            command_sender.close()
        if child is not None:
            child.join(timeout=max(1.0, float(timeout_s) + 1.0))
            if child.is_alive():
                child.terminate()
                child.join(timeout=1.0)
            child_collected = not child.is_alive()
        if recording_receiver is not None:
            recording_receiver.close()
    acknowledgement = (
        recording_receiver.received[-1]
        if (
            recording_receiver is not None
            and recording_receiver.received
        )
        else None
    )
    completion_configuration = getattr(
        engine,
        "qwen35_loaded_checkpoint_candidate_binding_configuration",
        None,
    )
    completion_rows = getattr(
        engine,
        "qwen35_loaded_checkpoint_candidate_binding_rows",
        None,
    )
    return {
        "status": "PASS",
        "mode": mode,
        "dispatch_count": dispatch_count,
        "collector_call_count": (
            len(recording_collector.collect_calls)
            if recording_collector is not None
            else 0
        ),
        "command_send_count": len(envelopes),
        "child_receive_count": (
            received_count.value
            if received_count is not None
            else 0
        ),
        "envelope": (
            _envelope_row(envelopes[-1])
            if envelopes
            else None
        ),
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
        "binding_rows": (
            [dict(row) for row in completion_rows]
            if completion_rows is not None
            else None
        ),
        "completion_configuration": (
            list(completion_configuration)
            if completion_configuration is not None
            else None
        ),
        "completion_committed": (
            completion_configuration is not None
            and completion_rows is not None
        ),
        "repeat_zero_dispatch": repeat_zero_dispatch,
        "collector_poisoned": (
            recording_collector.poisoned
            if recording_collector is not None
            else False
        ),
        "error_detail": error_detail,
        "child_process_id": child.pid if child is not None else None,
        "child_exitcode": (
            child.exitcode if child is not None else None
        ),
        "child_collected": child_collected,
    }


def _attempt_rows(prerequisite, mode):
    rows = prerequisite.row_map
    if mode == "tp1_success":
        return rows[(1, 0, "success")]["method_row"], None
    if mode == "tp1_local_binding_error":
        return (
            rows[(1, 0, "injected_bridge_conflict")]["method_row"],
            None,
        )
    local = rows[(2, 0, "success")]["method_row"]
    if mode == "tp2_worker_binding_error":
        worker = rows[
            (2, 1, "injected_bridge_conflict")
        ]["method_row"]
    else:
        worker = rows[(2, 1, "success")]["method_row"]
    return local, worker


def run_engine_ack_transport_rank_worker(
    *,
    source_root,
    prerequisite_artifact,
    mode,
    timeout_s,
    observed_user,
    observed_hostname,
    process_id,
):
    prerequisite = load_engine_ack_transport_prerequisite(
        prerequisite_artifact
    )
    local_row, worker_row = _attempt_rows(prerequisite, mode)
    attempt = execute_engine_ack_transport_attempt(
        source_root=source_root,
        mode=mode,
        local_row=local_row,
        worker_row=worker_row,
        timeout_s=timeout_s,
    )
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "mode": mode,
        "process_id": process_id,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "timeout_s": float(timeout_s),
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        **attempt,
    }
    validate_engine_ack_transport_row(row)
    return row


def validate_engine_ack_transport_row(row):
    mode = row.get("mode")
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or mode not in WORKER_MODES
    ):
        raise ValueError(
            "Engine acknowledgement transport row schema is invalid"
        )
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "timeout_s": 2.0,
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "child_collected": True,
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            raise ValueError(
                f"Engine acknowledgement transport {name} is invalid"
            )
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError(
            "Engine acknowledgement transport process ID is invalid"
        )
    tp2 = mode.startswith("tp2")
    success = mode in {"tp1_success", "tp2_success"}
    expected_dispatch = 1 if tp2 else 0
    expected_ack_status = {
        "tp1_success": "absent",
        "tp1_local_binding_error": "absent",
        "tp2_success": "ok",
        "tp2_worker_binding_error": "ok",
        "tp2_worker_ack_exception": "error",
        "tp2_worker_exit_without_ack": "absent",
    }[mode]
    checks = {
        "dispatch_count": expected_dispatch,
        "collector_call_count": expected_dispatch,
        "command_send_count": expected_dispatch,
        "child_receive_count": expected_dispatch,
        "acknowledgement_status": expected_ack_status,
        "completion_committed": success,
        "repeat_zero_dispatch": success,
        "collector_poisoned": mode in {
            "tp2_worker_ack_exception",
            "tp2_worker_exit_without_ack",
        },
    }
    for name, expected in checks.items():
        if row.get(name) != expected:
            raise ValueError(
                f"Engine acknowledgement transport {name} is invalid"
            )
    if tp2:
        if (
            not isinstance(row.get("child_process_id"), int)
            or row["child_process_id"] <= 0
            or row.get("child_exitcode") != 0
        ):
            raise ValueError(
                "Engine acknowledgement transport child process "
                "is invalid"
            )
        expected_envelope = {
            "command_id": 0,
            "method_name": (
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ),
            "args": [],
            "requires_ack": True,
        }
        if row.get("envelope") != expected_envelope:
            raise ValueError(
                "Engine acknowledgement transport envelope is invalid"
            )
    elif (
        row.get("child_process_id") is not None
        or row.get("child_exitcode") is not None
        or row.get("envelope") is not None
    ):
        raise ValueError(
            "Engine acknowledgement transport TP1 child state "
            "is invalid"
        )
    if success:
        binding_rows = row.get("binding_rows")
        configuration = row.get("completion_configuration")
        expected_count = 2 if tp2 else 1
        if (
            not isinstance(binding_rows, list)
            or len(binding_rows) != expected_count
            or not isinstance(configuration, list)
            or len(configuration) != 4
            or row.get("error_detail")
        ):
            raise ValueError(
                "Engine acknowledgement transport completion is invalid"
            )
    elif (
        row.get("binding_rows") is not None
        or row.get("completion_configuration") is not None
        or not isinstance(row.get("error_detail"), str)
        or not row["error_detail"]
    ):
        raise ValueError(
            "Engine acknowledgement transport failure state is invalid"
        )
    required_error = {
        "tp1_local_binding_error": (
            "loaded checkpoint candidate binding failed: rank=0"
        ),
        "tp2_worker_binding_error": (
            "loaded checkpoint candidate binding failed: rank=1"
        ),
        "tp2_worker_ack_exception": (
            "injected worker acknowledgement exception"
        ),
        "tp2_worker_exit_without_ack": (
            "acknowledgement receive failed"
        ),
    }.get(mode)
    if (
        required_error is not None
        and required_error not in row["error_detail"]
    ):
        raise ValueError(
            "Engine acknowledgement transport error detail is invalid"
        )
    return row


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(
                "missing Engine acknowledgement transport source: "
                + name
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
                    "missing Engine acknowledgement transport source: "
                    + name
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def audit_engine_ack_transport_source(source_root):
    path = Path(source_root) / (
        "tools/qwen35_real_binding_engine_ack_transport_preflight.py"
    )
    tree = ast.parse(path.read_text(), filename=os.fspath(path))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

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

    method_calls = {
        name: sum(
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "methods"
            and isinstance(node.func.slice, ast.Constant)
            and node.func.slice.value == name
            for node in calls
        )
        for name in METHOD_SOURCE_SHA256
    }
    imports = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        and node.args
        and isinstance(node.args[0], ast.Constant)
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
        "frozen_method_invocation_count": method_calls,
        "ack_collector_constructor_count": attribute(
            "ModelRunnerCommandAckCollector"
        ),
        "ack_executor_call_count": attribute(
            "execute_acknowledged_command"
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
    expected = {
        "llm_engine_import_count": 0,
        "model_runner_import_count": 0,
        "llm_engine_construction_count": 0,
        "model_runner_construction_count": 0,
        "frozen_method_invocation_count": {
            name: 1 for name in METHOD_SOURCE_SHA256
        },
        "ack_collector_constructor_count": 1,
        "ack_executor_call_count": 1,
        "checkpoint_call_count": 0,
        "scheduler_call_count": 0,
        "step_call_count": 0,
        "cuda_call_count": 0,
        "forward_call_count": 0,
        "inference_call_count": 0,
    }
    if audit != expected:
        raise ValueError(
            "Engine acknowledgement transport static audit "
            f"is invalid: {audit!r}"
        )
    return audit


def validate_engine_ack_transport_preflight(record):
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
                f"Engine acknowledgement transport {name} is invalid"
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
            "Engine acknowledgement transport source closure is invalid"
        )
    rows = record.get("rows")
    if [
        row.get("mode") for row in rows or ()
    ] != list(WORKER_MODES):
        raise ValueError(
            "Engine acknowledgement transport worker rows are invalid"
        )
    for row in rows:
        validate_engine_ack_transport_row(row)
    if len({row["process_id"] for row in rows}) != len(
        WORKER_MODES
    ):
        raise ValueError(
            "Engine acknowledgement transport worker process IDs "
            "must be unique"
        )
    child_ids = [
        row["child_process_id"]
        for row in rows
        if row["mode"].startswith("tp2")
    ]
    if len(child_ids) != 4 or len(set(child_ids)) != 4:
        raise ValueError(
            "Engine acknowledgement transport child process IDs "
            "must be unique"
        )
    return record


def _aggregate(rows, source_root):
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
    validate_engine_ack_transport_preflight(record)
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
            "Engine acknowledgement transport prerequisite hash "
            "is invalid"
        )
    prerequisite = load_engine_ack_transport_prerequisite(
        prerequisite_artifact
    )
    local_hashes = _source_hashes(source_root)
    if {
        name: local_hashes[name]
        for name in prerequisite.source_file_sha256
    } != dict(prerequisite.source_file_sha256):
        raise ValueError(
            "Engine acknowledgement transport source does not match "
            "prerequisite"
        )
    audit_engine_ack_transport_source(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_prerequisite = (
        f"{remote_run_dir}/"
        "model_runner_published_candidate_binding_preflight.json"
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
    _require_success(staged, "Engine ack source staging")
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
    _require_success(completed, "Engine ack prerequisite staging")
    script = "\n".join([
        "import ast,hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"prerequisite=pathlib.Path({remote_prerequisite!r})",
        f"names={list(SOURCE_FILES)!r}",
        f"specs={[(LLM_ENGINE_SOURCE, 'LLMEngine', 'call_model_runner_acknowledged'), (LLM_ENGINE_SOURCE, 'LLMEngine', 'bind_qwen35_loaded_checkpoint_candidates'), (MODEL_RUNNER_SOURCE, 'ModelRunner', 'dispatch_command')]!r}",
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
    _require_success(verified, "Engine ack staged hashing")
    remote = json.loads(verified.stdout)
    if (
        remote.get("source") != local_hashes
        or remote.get("prerequisite")
        != PREREQUISITE_ARTIFACT_SHA256
        or remote.get("methods") != dict(METHOD_SOURCE_SHA256)
    ):
        raise ValueError(
            "Engine acknowledgement transport staged identity "
            "is invalid"
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


def run_remote_engine_ack_transport_preflight(
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
            "local Engine ack transport directory exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/engine_ack_transport_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_binding_engine_ack_transport_preflight.py"
    )
    rows = []
    for mode in WORKER_MODES:
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
        _require_success(completed, "Engine ack attempt worker")
        row = json.loads(completed.stdout)
        validate_engine_ack_transport_row(row)
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
    _require_success(finalized, "Engine ack finalizer")
    record = json.loads(finalized.stdout)
    validate_engine_ack_transport_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError("Engine ack source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'engine_ack_transport_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'engine_ack_transport_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "engine_ack_transport_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "Engine ack artifact round trip")
    if json.loads(round_trip.stdout) != {
        "engine_ack_transport_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("Engine ack artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "engine_ack_transport_preflight.json",
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


def execute_remote_engine_ack_transport_preflight(
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
    return run_remote_engine_ack_transport_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _attempt_worker_main(arguments):
    row = run_engine_ack_transport_rank_worker(
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
        raise ValueError("Engine ack output already exists")
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
        choices=WORKER_MODES,
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
    if arguments.mode == "internal-attempt-worker":
        return _attempt_worker_main(arguments)
    if arguments.mode == "internal-finalize":
        return _finalize_main(arguments)
    if arguments.mode == "validate":
        record = json.loads(Path(arguments.artifact).read_text())
        validate_engine_ack_transport_preflight(record)
    else:
        record = execute_remote_engine_ack_transport_preflight(
            arguments.source_root,
            arguments.run_tag,
            prerequisite_artifact=arguments.prerequisite_artifact,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
