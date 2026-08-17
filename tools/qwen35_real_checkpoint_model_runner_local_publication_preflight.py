from __future__ import annotations

import argparse
import ast
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


publication = _load_sibling(
    "_qwen35_private_publication_model_runner_method_base",
    "qwen35_real_checkpoint_private_publication_slot_preflight.py",
)


COMPLETE_ARTIFACT_SHA256 = (
    "7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176"
)
PUBLICATION_ARTIFACT_SHA256 = (
    "f208a799eca053e03a35aa4bfcbe66dfe6e5875b3e7b78390ded345a7c7c12b6"
)
COMPLETE_SOURCE_TREE_SHA256 = (
    "da665b2de0aaa6533e55be0469c76ed39d92e817aabf80618f07b7efa7ef7042"
)
PUBLICATION_SOURCE_TREE_SHA256 = (
    "20c87258ff71449ebb8bf15af6ba77153804c16ab88a5fb11917a4597be51440"
)
MODEL_RUNNER_FILE_SHA256 = (
    "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
)
MODEL_RUNNER_PUBLICATION_METHOD_SHA256 = (
    "37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f"
)
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
METHOD_NAME = "publish_qwen35_loaded_checkpoint_candidate"
PREREQUISITE_ROWS = ((1, 0), (2, 0), (2, 1))
SCHEMA_VERSION = (
    "qwen35.real-checkpoint-model-runner-local-publication.v1"
)
ROW_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-model-runner-local-publication-rank.v1"
)
REMOTE_TARGET = publication.REMOTE_TARGET
REMOTE_PYTHON = publication.REMOTE_PYTHON
APPROVED_MODEL_DIR = publication.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = (
    publication.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_CONFIG_SHA256 = publication.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = publication.APPROVED_INDEX_SHA256
APPROVED_COMPOSITE_SHA256 = publication.APPROVED_COMPOSITE_SHA256
APPROVED_SHARD_NAME = publication.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = publication.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = publication.APPROVED_SHARD_SHA256
AUTHORIZATION_SHA256 = publication.AUTHORIZATION_SHA256
MAX_TENSOR_BYTES = publication.MAX_TENSOR_BYTES
STREAMED_STATS = dict(publication.STREAMED_STATS)
MEMORY_CEILINGS_KIB = dict(publication.MEMORY_CEILINGS_KIB)
LOCAL_RUN_ROOT = publication.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-model-runner-local-publication-runs"
)
WORKER_CONTEXTS = (
    (1, 0, "success"),
    (1, 0, "injected_method_failure"),
    (2, 0, "success"),
    (2, 0, "injected_method_failure"),
    (2, 1, "success"),
    (2, 1, "injected_method_failure"),
)
SOURCE_FILES = (
    *publication.SOURCE_FILES,
    MODEL_RUNNER_SOURCE,
    "tools/qwen35_real_checkpoint_model_runner_local_publication_preflight.py",
)
_read_proc_status = publication._read_proc_status
_memory_point = publication._memory_point
_install_namespace_packages = publication._install_namespace_packages
_source_tree_sha256 = publication._source_tree_sha256
_atomic_write_json = publication._atomic_write_json
validate_run_tag = publication.validate_run_tag
build_ssh_command = publication.build_ssh_command
_require_success = publication._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_exact_json(path, expected_sha256):
    payload = Path(path).read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError(
            "ModelRunner publication prerequisite hash is invalid"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "ModelRunner publication prerequisite JSON is invalid"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            "ModelRunner publication prerequisite must be an object"
        )
    return value


def _row_map(rows):
    result = {}
    for row in rows:
        key = (row.get("tp_size"), row.get("tp_rank"))
        if key in result:
            raise ValueError(
                "duplicate ModelRunner publication prerequisite row"
            )
        result[key] = MappingProxyType(row)
    if tuple(result) != PREREQUISITE_ROWS:
        raise ValueError(
            "ModelRunner publication prerequisite TP rows are invalid"
        )
    return MappingProxyType(result)


@dataclass(frozen=True)
class ModelRunnerPublicationPrerequisites:
    complete_artifact_sha256: str
    publication_artifact_sha256: str
    complete_source_tree_sha256: str
    publication_source_tree_sha256: str
    complete_rows: tuple[tuple[int, int], ...]
    publication_rows: tuple[tuple[int, int], ...]
    complete_row_map: MappingProxyType
    publication_row_map: MappingProxyType
    publication_source_file_sha256: MappingProxyType


def load_model_runner_publication_prerequisites(
    complete_artifact,
    publication_artifact,
) -> ModelRunnerPublicationPrerequisites:
    complete_record = (
        publication.ownership.loader_core.load_complete_gate_oracle(
            complete_artifact
        )
    )
    publication_record = _read_exact_json(
        publication_artifact,
        PUBLICATION_ARTIFACT_SHA256,
    )
    publication.validate_private_publication_slot_preflight(
        publication_record
    )
    if (
        complete_record["source_tree_sha256"]
        != COMPLETE_SOURCE_TREE_SHA256
    ):
        raise ValueError("complete prerequisite source tree is invalid")
    if (
        publication_record["source_tree_sha256"]
        != PUBLICATION_SOURCE_TREE_SHA256
    ):
        raise ValueError(
            "publication prerequisite source tree is invalid"
        )
    complete_rows = _row_map(complete_record["rows"])
    publication_rows = _row_map([
        row
        for row in publication_record["rows"]
        if row["mode"] == "success"
    ])
    if len({row["process_id"] for row in publication_record["rows"]}) != 6:
        raise ValueError(
            "publication prerequisite process IDs are invalid"
        )
    source_hashes = publication_record["source_file_sha256"]
    if not isinstance(source_hashes, dict) or len(source_hashes) != 47:
        raise ValueError(
            "publication prerequisite source closure is invalid"
        )
    return ModelRunnerPublicationPrerequisites(
        complete_artifact_sha256=COMPLETE_ARTIFACT_SHA256,
        publication_artifact_sha256=PUBLICATION_ARTIFACT_SHA256,
        complete_source_tree_sha256=COMPLETE_SOURCE_TREE_SHA256,
        publication_source_tree_sha256=PUBLICATION_SOURCE_TREE_SHA256,
        complete_rows=tuple(complete_rows),
        publication_rows=tuple(publication_rows),
        complete_row_map=complete_rows,
        publication_row_map=publication_rows,
        publication_source_file_sha256=MappingProxyType(source_hashes),
    )


def select_model_runner_publication_prerequisite_rows(
    prerequisites,
    tensor_parallel_size,
    tensor_parallel_rank,
):
    if type(prerequisites) is not ModelRunnerPublicationPrerequisites:
        raise ValueError(
            "prerequisites must be exact "
            "ModelRunnerPublicationPrerequisites"
        )
    key = (tensor_parallel_size, tensor_parallel_rank)
    try:
        return (
            prerequisites.complete_row_map[key],
            prerequisites.publication_row_map[key],
        )
    except KeyError as error:
        raise ValueError(
            "ModelRunner publication prerequisite TP row is invalid"
        ) from error


def _model_runner_method_node(source):
    tree = ast.parse(source, filename=MODEL_RUNNER_SOURCE)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    ]
    if len(classes) != 1:
        raise ValueError("ModelRunner class identity is invalid")
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef)
        and node.name == METHOD_NAME
    ]
    if len(methods) != 1:
        raise ValueError(
            "ModelRunner publication method identity is invalid"
        )
    method = methods[0]
    if [argument.arg for argument in method.args.args] != [
        "self",
        "candidate",
    ]:
        raise ValueError(
            "ModelRunner publication method arguments are invalid"
        )
    if len(method.body) != 2:
        raise ValueError(
            "ModelRunner publication method body is invalid"
        )
    expression, returned = method.body
    if (
        not isinstance(expression, ast.Expr)
        or not isinstance(expression.value, ast.Call)
        or not isinstance(expression.value.func, ast.Attribute)
        or expression.value.func.attr != "publish"
        or len(expression.value.args) != 1
        or not isinstance(expression.value.args[0], ast.Name)
        or expression.value.args[0].id != "candidate"
        or expression.value.keywords
        or not isinstance(returned, ast.Return)
        or not isinstance(returned.value, ast.Name)
        or returned.value.id != "candidate"
    ):
        raise ValueError(
            "ModelRunner publication method semantics are invalid"
        )
    receiver = expression.value.func.value
    if (
        not isinstance(receiver, ast.Attribute)
        or receiver.attr
        != "qwen35_loaded_checkpoint_candidate_slot"
        or not isinstance(receiver.value, ast.Name)
        or receiver.value.id != "self"
    ):
        raise ValueError(
            "ModelRunner publication method receiver is invalid"
        )
    return method


def load_frozen_model_runner_publication_method(source_root):
    path = Path(source_root) / MODEL_RUNNER_SOURCE
    payload = path.read_bytes()
    if _sha256(payload) != MODEL_RUNNER_FILE_SHA256:
        raise ValueError("ModelRunner source hash is invalid")
    source = payload.decode("utf-8")
    method = _model_runner_method_node(source)
    segment = ast.get_source_segment(source, method)
    if (
        segment is None
        or _sha256(segment.encode("utf-8"))
        != MODEL_RUNNER_PUBLICATION_METHOD_SHA256
    ):
        raise ValueError(
            "ModelRunner publication method source hash is invalid"
        )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(module),
            MODEL_RUNNER_SOURCE,
            "exec",
        ),
        namespace,
    )
    return namespace[METHOD_NAME]


def execute_model_runner_local_publication_scope(
    *,
    private_graph_factory,
    oracle_row,
    model_fingerprint,
    publication_method,
    production_slot_factory,
    mode,
):
    import torch

    if not callable(private_graph_factory):
        raise ValueError("private_graph_factory must be callable")
    if not callable(publication_method):
        raise ValueError("publication_method must be callable")
    if not callable(production_slot_factory):
        raise ValueError("production_slot_factory must be callable")
    if mode not in ("success", "injected_method_failure"):
        raise ValueError("ModelRunner publication mode is invalid")

    def execute_nested_scope():
        target, acquire_candidate = private_graph_factory()
        if not callable(acquire_candidate):
            raise ValueError("acquire_candidate must be callable")
        model = target.assembly.packed.model
        pool = target.pool
        registered = publication.ownership.loader_core._registered_tensors(
            model
        )
        identity = publication.ownership.loader_core._snapshot_identity(
            registered
        )
        pool_snapshot = publication._snapshot_pool(pool)
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
                "ModelRunner publication destination initialization failed"
            )
        production_slot = production_slot_factory()
        slot_empty_before = (
            production_slot.candidate is None
            and production_slot.owner is None
            and production_slot.model_fingerprint is None
        )
        if not slot_empty_before:
            raise ValueError(
                "ModelRunner publication slot must start empty"
            )
        target_consumed_before = target._consumed
        if target_consumed_before is not False:
            raise ValueError(
                "ModelRunner publication target must be unconsumed"
            )

        class RunnerShell:
            pass

        class RejectingSlot:
            def __init__(self, expected, delegate):
                self.expected = expected
                self.delegate = delegate
                self.publish_call_count = 0

            def publish(self, candidate):
                self.publish_call_count += 1
                if candidate is not self.expected:
                    raise ValueError(
                        "ModelRunner publication proxy candidate is invalid"
                    )
                raise RuntimeError(
                    "injected ModelRunner local publication failure"
                )

        candidate = acquire_candidate()
        if target._consumed is not True:
            raise RuntimeError(
                "ModelRunner publication target was not consumed"
            )
        result = publication.ownership.validate_private_loaded_candidate(
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
        runner = RunnerShell()
        proxy = None
        if mode == "success":
            runner.qwen35_loaded_checkpoint_candidate_slot = (
                production_slot
            )
        else:
            proxy = RejectingSlot(candidate, production_slot)
            runner.qwen35_loaded_checkpoint_candidate_slot = proxy
        method_call_count = 0
        method_returned_candidate = False
        error = None
        try:
            method_call_count += 1
            returned = publication_method(runner, candidate)
            method_returned_candidate = True
            if returned is not candidate:
                raise ValueError(
                    "ModelRunner publication method return is invalid"
                )
            if (
                production_slot.candidate is not candidate
                or production_slot.owner is not candidate.owner
                or production_slot.model_fingerprint
                != model_fingerprint
            ):
                raise ValueError(
                    "ModelRunner publication slot visibility is invalid"
                )
            result.update({
                "method_return_identity_verified": True,
                "production_slot_visibility_verified": True,
                "production_slot_remained_empty": False,
            })
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
            publication.ownership.loader_core._require_identity_unchanged(
                registered,
                identity,
            )
            if any(
                int(tensor.count_nonzero().item())
                for tensor in selected.values()
            ):
                raise RuntimeError(
                    "ModelRunner publication destination clear is incomplete"
                )
            if any(
                not tensor.equal(non_selected_values[id(tensor)])
                for tensor in registered
                if id(tensor) not in selected_ids
            ):
                raise RuntimeError(
                    "ModelRunner publication non-selected tensor changed"
                )
            if not publication._pool_unchanged(pool, pool_snapshot):
                raise RuntimeError(
                    "ModelRunner publication pool state changed"
                )
        except Exception as caught:
            if clear_error is None:
                clear_error = caught
        if clear_error is not None:
            raise RuntimeError(
                "ModelRunner publication scope cleanup failed"
            ) from clear_error
        if mode == "success":
            if error is not None:
                raise error
            expected_failure = {"expected_failure_observed": False}
            proxy_publish_call_count = 0
            production_publish_call_count = 1
        else:
            if (
                error is None
                or type(error) is not RuntimeError
                or str(error)
                != "injected ModelRunner local publication failure"
            ):
                if error is None:
                    raise RuntimeError(
                        "ModelRunner injected publication failure missing"
                    )
                raise error
            if (
                production_slot.candidate is not None
                or production_slot.owner is not None
                or production_slot.model_fingerprint is not None
            ):
                raise RuntimeError(
                    "ModelRunner production slot changed on rejection"
                )
            expected_failure = {
                "expected_failure_observed": True,
                "expected_failure_type": type(error).__name__,
                "expected_failure_message": str(error),
            }
            result.update({
                "method_return_identity_verified": False,
                "production_slot_visibility_verified": False,
                "production_slot_remained_empty": True,
            })
            proxy_publish_call_count = proxy.publish_call_count
            production_publish_call_count = 0
        result.update({
            "method_call_count": method_call_count,
            "method_returned_candidate": method_returned_candidate,
            "proxy_publish_call_count": proxy_publish_call_count,
            "production_publish_call_count": (
                production_publish_call_count
            ),
            "slot_empty_before_publication": slot_empty_before,
            "target_consumed_before": target_consumed_before,
            "target_consumed_after": target._consumed,
            "selected_binding_count": len(
                target.binding_plan.bindings
            ),
            "unique_destination_count": len(selected),
            "alias_groups": publication.ownership.loader_core._alias_groups(
                target.binding_plan
            ),
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
            **expected_failure,
        })
        references = {
            "runner": weakref.ref(runner),
            "production_slot": weakref.ref(production_slot),
            "candidate": weakref.ref(candidate),
            "owner": weakref.ref(candidate.owner),
            "model": weakref.ref(model),
            "pool": weakref.ref(pool),
            "target": weakref.ref(target),
        }
        if proxy is not None:
            references["proxy_slot"] = weakref.ref(proxy)
        return result, references

    result, references = execute_nested_scope()
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
            "ModelRunner publication objects escaped scope: "
            + ", ".join(escaped)
        )
    result["collected_private_objects"] = collected
    result["all_private_method_objects_collected"] = True
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
        raise ValueError(
            "ModelRunner publication memory points are invalid"
        )
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
        raise ValueError(
            "ModelRunner publication memory deltas are invalid"
        )
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
            "ModelRunner publication memory ceiling exceeded: "
            + details
        )


def validate_model_runner_local_publication_row(row):
    tp = (row.get("tp_size"), row.get("tp_rank"))
    mode = row.get("mode")
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or tp not in PREREQUISITE_ROWS
        or mode not in ("success", "injected_method_failure")
    ):
        raise ValueError(
            "ModelRunner publication row schema is invalid"
        )
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "complete_artifact_sha256": COMPLETE_ARTIFACT_SHA256,
        "publication_artifact_sha256": PUBLICATION_ARTIFACT_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_publication_method_sha256": (
            MODEL_RUNNER_PUBLICATION_METHOD_SHA256
        ),
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": (
            publication.ownership.loader_core.ALIAS_GROUPS
        ),
        "provider_call_count": 1,
        "adapter_call_count": 1,
        "method_call_count": 1,
        "target_consumed_before": False,
        "target_consumed_after": True,
        "slot_empty_before_publication": True,
        "all_selected_destinations_zero_after_clear": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "non_selected_tensors_unchanged": True,
        "all_private_method_objects_collected": True,
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
            raise ValueError(
                f"ModelRunner publication {name} is invalid"
            )
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
            "ModelRunner publication hash evidence is invalid"
        )
    expected_collected = {
        "runner": True,
        "production_slot": True,
        "candidate": True,
        "owner": True,
        "model": True,
        "pool": True,
        "target": True,
    }
    if mode == "success":
        expected = {
            "expected_failure_observed": False,
            "method_returned_candidate": True,
            "method_return_identity_verified": True,
            "production_slot_visibility_verified": True,
            "production_slot_remained_empty": False,
            "proxy_publish_call_count": 0,
            "production_publish_call_count": 1,
        }
    else:
        expected_collected["proxy_slot"] = True
        expected = {
            "expected_failure_observed": True,
            "expected_failure_type": "RuntimeError",
            "expected_failure_message": (
                "injected ModelRunner local publication failure"
            ),
            "method_returned_candidate": False,
            "method_return_identity_verified": False,
            "production_slot_visibility_verified": False,
            "production_slot_remained_empty": True,
            "proxy_publish_call_count": 1,
            "production_publish_call_count": 0,
        }
    expected["collected_private_objects"] = expected_collected
    for name, value in expected.items():
        if row.get(name) != value:
            raise ValueError(
                f"ModelRunner publication {name} is invalid"
            )
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError(
            "ModelRunner publication process ID is invalid"
        )
    _validate_memory(row)
    return row


def run_model_runner_local_publication_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    complete_artifact,
    publication_artifact,
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
    prerequisites = load_model_runner_publication_prerequisites(
        complete_artifact,
        publication_artifact,
    )
    oracle_row, _publication_row = (
        select_model_runner_publication_prerequisite_rows(
            prerequisites,
            tensor_parallel_size,
            tensor_parallel_rank,
        )
    )
    publication_method = load_frozen_model_runner_publication_method(
        source_root
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
                    "ModelRunner publication target provider "
                    "called twice"
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

    result = execute_model_runner_local_publication_scope(
        private_graph_factory=private_graph_factory,
        oracle_row=oracle_row,
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        publication_method=publication_method,
        production_slot_factory=(
            lambda: publication_module
            .Qwen35HybridModelOwnerPublicationSlot()
        ),
        mode=mode,
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
        "publication_artifact_sha256": PUBLICATION_ARTIFACT_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_publication_method_sha256": (
            MODEL_RUNNER_PUBLICATION_METHOD_SHA256
        ),
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
    validate_model_runner_local_publication_row(row)
    return row


def validate_model_runner_local_publication_preflight(record):
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
        "publication_artifact_sha256": PUBLICATION_ARTIFACT_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_publication_method_sha256": (
            MODEL_RUNNER_PUBLICATION_METHOD_SHA256
        ),
        "fresh_process_per_attempt": True,
    }
    if any(record.get(name) != value for name, value in exact.items()):
        raise ValueError(
            "ModelRunner publication preflight identity is invalid"
        )
    hashes = record.get("source_file_sha256")
    if set(hashes or {}) != set(SOURCE_FILES):
        raise ValueError(
            "ModelRunner publication source hashes are invalid"
        )
    if hashes[MODEL_RUNNER_SOURCE] != MODEL_RUNNER_FILE_SHA256:
        raise ValueError(
            "ModelRunner publication source file hash is invalid"
        )
    if record.get("source_tree_sha256") != _source_tree_sha256(
        hashes
    ):
        raise ValueError(
            "ModelRunner publication source tree is invalid"
        )
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
            "ModelRunner publication worker rows are invalid"
        )
    for row in rows:
        validate_model_runner_local_publication_row(row)
    if len({row["process_id"] for row in rows}) != 6:
        raise ValueError(
            "ModelRunner publication process IDs must be unique"
        )
    return record


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(
                f"missing ModelRunner publication source: {name}"
            )
        hashes[name] = _sha256(path.read_bytes())
    return hashes


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(
                    f"missing ModelRunner publication source: {name}"
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
    publication_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    complete_payload = Path(complete_artifact).read_bytes()
    publication_payload = Path(publication_artifact).read_bytes()
    if _sha256(complete_payload) != COMPLETE_ARTIFACT_SHA256:
        raise ValueError(
            "complete prerequisite artifact hash is invalid"
        )
    if _sha256(publication_payload) != PUBLICATION_ARTIFACT_SHA256:
        raise ValueError(
            "publication prerequisite artifact hash is invalid"
        )
    local_hashes = _source_hashes(source_root)
    prerequisites = load_model_runner_publication_prerequisites(
        complete_artifact,
        publication_artifact,
    )
    if {
        name: local_hashes[name]
        for name in publication.SOURCE_FILES
    } != dict(prerequisites.publication_source_file_sha256):
        raise ValueError(
            "ModelRunner publication source does not match "
            "publication prerequisite"
        )
    if local_hashes[MODEL_RUNNER_SOURCE] != MODEL_RUNNER_FILE_SHA256:
        raise ValueError(
            "ModelRunner publication production source was modified"
        )
    load_frozen_model_runner_publication_method(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_complete = (
        f"{remote_run_dir}/complete_checkpoint_transaction_preflight.json"
    )
    remote_publication = (
        f"{remote_run_dir}/private_publication_slot_preflight.json"
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
    _require_success(staged, "ModelRunner publication source staging")
    for remote_path, payload, label in (
        (remote_complete, complete_payload, "complete"),
        (remote_publication, publication_payload, "publication"),
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
            f"ModelRunner publication {label} prerequisite staging",
        )
    script = "\n".join([
        "import ast,hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"complete=pathlib.Path({remote_complete!r})",
        f"publication=pathlib.Path({remote_publication!r})",
        f"names={list(SOURCE_FILES)!r}",
        f"model_name={MODEL_RUNNER_SOURCE!r}",
        f"method_name={METHOD_NAME!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "source=(root/model_name).read_text()",
        "tree=ast.parse(source,filename=model_name)",
        "classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ModelRunner']",
        "if len(classes)!=1: raise SystemExit('invalid ModelRunner class')",
        "methods=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name==method_name]",
        "if len(methods)!=1: raise SystemExit('invalid ModelRunner method')",
        "segment=ast.get_source_segment(source,methods[0])",
        "payload={'source':result,'complete':hashlib.sha256(complete.read_bytes()).hexdigest(),'publication':hashlib.sha256(publication.read_bytes()).hexdigest(),'method':hashlib.sha256(segment.encode()).hexdigest()}",
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
        "ModelRunner publication staged hashing",
    )
    remote = json.loads(verified.stdout)
    if remote.get("source") != local_hashes:
        raise ValueError(
            "ModelRunner publication remote source hash mismatch"
        )
    if remote.get("complete") != COMPLETE_ARTIFACT_SHA256:
        raise ValueError(
            "ModelRunner publication remote complete hash mismatch"
        )
    if remote.get("publication") != PUBLICATION_ARTIFACT_SHA256:
        raise ValueError(
            "ModelRunner publication remote prerequisite mismatch"
        )
    if remote.get("method") != MODEL_RUNNER_PUBLICATION_METHOD_SHA256:
        raise ValueError(
            "ModelRunner publication remote method hash mismatch"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "remote_complete_artifact": remote_complete,
        "remote_publication_artifact": remote_publication,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "complete_artifact_sha256": remote["complete"],
        "publication_artifact_sha256": remote["publication"],
        "model_runner_publication_method_sha256": remote["method"],
    }


def _aggregate(rows, source_root):
    hashes = _source_hashes(source_root)
    load_frozen_model_runner_publication_method(source_root)
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
        "publication_artifact_sha256": PUBLICATION_ARTIFACT_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_publication_method_sha256": (
            MODEL_RUNNER_PUBLICATION_METHOD_SHA256
        ),
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_model_runner_local_publication_preflight(record)
    return record


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "remote_complete_artifact": staged["remote_complete_artifact"],
        "remote_publication_artifact": (
            staged["remote_publication_artifact"]
        ),
        "complete_artifact_sha256": (
            staged["complete_artifact_sha256"]
        ),
        "publication_artifact_sha256": (
            staged["publication_artifact_sha256"]
        ),
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_publication_method_sha256": (
            staged["model_runner_publication_method_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_model_runner_local_publication_preflight(
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
            "local ModelRunner publication directory exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/model_runner_local_publication_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_checkpoint_model_runner_local_publication_preflight.py"
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
                "--publication-artifact",
                staged["remote_publication_artifact"],
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
        _require_success(
            completed,
            "ModelRunner publication rank worker",
        )
        row = json.loads(completed.stdout)
        validate_model_runner_local_publication_row(row)
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
        "ModelRunner publication finalizer",
    )
    record = json.loads(finalized.stdout)
    validate_model_runner_local_publication_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError(
            "ModelRunner publication source binding mismatch"
        )
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'model_runner_local_publication_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'model_runner_local_publication_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "model_runner_local_publication_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(
        round_trip,
        "ModelRunner publication artifact round trip",
    )
    if json.loads(round_trip.stdout) != {
        "model_runner_local_publication_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError(
            "ModelRunner publication artifact round-trip mismatch"
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
            / "model_runner_local_publication_preflight.json",
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


def execute_remote_model_runner_local_publication_preflight(
    source_root,
    run_tag,
    *,
    complete_artifact,
    publication_artifact,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source_and_prerequisites(
        source_root,
        run_tag,
        complete_artifact=complete_artifact,
        publication_artifact=publication_artifact,
        command_runner=command_runner,
    )
    return run_remote_model_runner_local_publication_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments):
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not approved")
    row = run_model_runner_local_publication_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        complete_artifact=arguments.complete_artifact,
        publication_artifact=arguments.publication_artifact,
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
        raise ValueError(
            "ModelRunner publication output already exists"
        )
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
    run_parser.add_argument("--publication-artifact", required=True)
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--complete-artifact", required=True)
    worker_parser.add_argument(
        "--publication-artifact",
        required=True,
    )
    worker_parser.add_argument("--tp-size", type=int, required=True)
    worker_parser.add_argument("--tp-rank", type=int, required=True)
    worker_parser.add_argument(
        "--attempt-mode",
        choices=("success", "injected_method_failure"),
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
        validate_model_runner_local_publication_preflight(record)
    else:
        record = (
            execute_remote_model_runner_local_publication_preflight(
                arguments.source_root,
                arguments.run_tag,
                complete_artifact=arguments.complete_artifact,
                publication_artifact=arguments.publication_artifact,
            )
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
