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


load_publish_gate = _load_sibling(
    "_qwen35_model_runner_published_binding_load_publish_base",
    "qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py",
)


PREREQUISITE_ARTIFACT_SHA256 = (
    "d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8"
)
MODEL_RUNNER_FILE_SHA256 = (
    "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
)
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
METHOD_SOURCE_SHA256 = MappingProxyType({
    "publish_qwen35_loaded_checkpoint_candidate": (
        "37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f"
    ),
    "bind_qwen35_hybrid_model_owner": (
        "462e2fefe22e90e60b85c786de6a95e7eaaae31bd9b257025088cd767555ee25"
    ),
    "bind_qwen35_loaded_checkpoint_candidate": (
        "a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0"
    ),
    "bind_published_qwen35_loaded_checkpoint_candidate": (
        "aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd"
    ),
})
METHOD_ARGUMENTS = MappingProxyType({
    "publish_qwen35_loaded_checkpoint_candidate": ("self", "candidate"),
    "bind_qwen35_hybrid_model_owner": ("self", "owner"),
    "bind_qwen35_loaded_checkpoint_candidate": ("self", "candidate"),
    "bind_published_qwen35_loaded_checkpoint_candidate": ("self",),
})
PREREQUISITE_ROWS = ((1, 0), (2, 0), (2, 1))
ROW_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-model-runner-published-binding-rank.v1"
)
APPROVED_MODEL_MANIFEST_SHA256 = (
    load_publish_gate.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_MODEL_DIR = load_publish_gate.APPROVED_MODEL_DIR
APPROVED_CONFIG_SHA256 = load_publish_gate.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = load_publish_gate.APPROVED_INDEX_SHA256
APPROVED_COMPOSITE_SHA256 = load_publish_gate.APPROVED_COMPOSITE_SHA256
APPROVED_SHARD_NAME = load_publish_gate.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = load_publish_gate.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = load_publish_gate.APPROVED_SHARD_SHA256
AUTHORIZATION_SHA256 = load_publish_gate.AUTHORIZATION_SHA256
MAX_TENSOR_BYTES = load_publish_gate.MAX_TENSOR_BYTES
STREAMED_STATS = dict(load_publish_gate.STREAMED_STATS)
MEMORY_CEILINGS_KIB = dict(load_publish_gate.MEMORY_CEILINGS_KIB)
REMOTE_TARGET = load_publish_gate.REMOTE_TARGET
REMOTE_PYTHON = load_publish_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = load_publish_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-model-runner-published-binding-runs"
)
SCHEMA_VERSION = (
    "qwen35.real-checkpoint-model-runner-published-binding.v1"
)
WORKER_CONTEXTS = (
    (1, 0, "success"),
    (1, 0, "injected_bridge_conflict"),
    (2, 0, "success"),
    (2, 0, "injected_bridge_conflict"),
    (2, 1, "success"),
    (2, 1, "injected_bridge_conflict"),
)
SOURCE_FILES = (
    *load_publish_gate.SOURCE_FILES,
    "tools/"
    "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py",
)
_read_proc_status = load_publish_gate._read_proc_status
_memory_point = load_publish_gate._memory_point
_install_namespace_packages = load_publish_gate._install_namespace_packages
_source_tree_sha256 = load_publish_gate._source_tree_sha256
_atomic_write_json = load_publish_gate._atomic_write_json
validate_run_tag = load_publish_gate.validate_run_tag
build_ssh_command = load_publish_gate.build_ssh_command
_require_success = load_publish_gate._require_success


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_exact_json(path, expected_sha256):
    payload = Path(path).read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError(
            "ModelRunner published binding prerequisite hash is invalid"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "ModelRunner published binding prerequisite JSON is invalid"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            "ModelRunner published binding prerequisite must be an object"
        )
    return value


@dataclass(frozen=True)
class ModelRunnerPublishedBindingPrerequisite:
    artifact_sha256: str
    source_tree_sha256: str
    rows: tuple[tuple[int, int], ...]
    row_map: Mapping
    source_file_sha256: Mapping


def load_model_runner_published_binding_prerequisite(artifact):
    record = _read_exact_json(
        artifact,
        PREREQUISITE_ARTIFACT_SHA256,
    )
    load_publish_gate.validate_model_runner_load_publish_preflight(record)
    if (
        record.get("source_tree_sha256")
        != PREREQUISITE_SOURCE_TREE_SHA256
    ):
        raise ValueError(
            "ModelRunner published binding prerequisite source tree "
            "is invalid"
        )
    source_hashes = record.get("source_file_sha256")
    if (
        not isinstance(source_hashes, dict)
        or len(source_hashes) != 50
        or tuple(source_hashes) != tuple(
            sorted(source_hashes)
        )
    ):
        raise ValueError(
            "ModelRunner published binding prerequisite source closure "
            "is invalid"
        )
    success_rows = [
        row
        for row in record.get("rows", ())
        if row.get("mode") == "success"
    ]
    row_map = {}
    for row in success_rows:
        key = (row.get("tp_size"), row.get("tp_rank"))
        if key in row_map:
            raise ValueError(
                "duplicate ModelRunner published binding prerequisite row"
            )
        row_map[key] = MappingProxyType(row)
    if tuple(row_map) != PREREQUISITE_ROWS:
        raise ValueError(
            "ModelRunner published binding prerequisite TP rows are invalid"
        )
    if len({row.get("process_id") for row in record["rows"]}) != 6:
        raise ValueError(
            "ModelRunner published binding prerequisite process IDs "
            "are invalid"
        )
    return ModelRunnerPublishedBindingPrerequisite(
        artifact_sha256=PREREQUISITE_ARTIFACT_SHA256,
        source_tree_sha256=PREREQUISITE_SOURCE_TREE_SHA256,
        rows=tuple(row_map),
        row_map=MappingProxyType(row_map),
        source_file_sha256=MappingProxyType(source_hashes),
    )


def select_model_runner_published_binding_prerequisite_row(
    prerequisite,
    tensor_parallel_size,
    tensor_parallel_rank,
):
    if (
        type(prerequisite)
        is not ModelRunnerPublishedBindingPrerequisite
    ):
        raise ValueError(
            "prerequisite must be exact "
            "ModelRunnerPublishedBindingPrerequisite"
        )
    try:
        return prerequisite.row_map[
            (tensor_parallel_size, tensor_parallel_rank)
        ]
    except KeyError as error:
        raise ValueError(
            "ModelRunner published binding prerequisite TP row is invalid"
        ) from error


def reconstruct_candidate_validation_oracle(flat_row):
    if not isinstance(flat_row, Mapping):
        raise ValueError(
            "ModelRunner published binding flat oracle row is invalid"
        )
    binding_hashes = flat_row.get("binding_destination_sha256")
    phase_hashes = flat_row.get("phase_destination_sha256")
    aggregate_hash = flat_row.get("aggregate_destination_sha256")
    phase_binding_runs = (
        load_publish_gate.publication_gate.publication.ownership
        .loader_core.PHASE_BINDING_RUNS
    )
    if (
        not isinstance(binding_hashes, list)
        or len(binding_hashes) != 320
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in binding_hashes
        )
    ):
        raise ValueError(
            "ModelRunner published binding flat binding hashes "
            "are invalid"
        )
    if (
        not isinstance(phase_hashes, Mapping)
        or set(phase_hashes) != {
            name for name, _indices in phase_binding_runs
        }
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in phase_hashes.values()
        )
    ):
        raise ValueError(
            "ModelRunner published binding flat phase hashes are invalid"
        )
    if (
        not isinstance(aggregate_hash, str)
        or len(aggregate_hash) != 64
    ):
        raise ValueError(
            "ModelRunner published binding flat aggregate hash is invalid"
        )
    binding_phase = {
        index: name
        for name, indices in phase_binding_runs
        for index in indices
    }
    if tuple(binding_phase) != tuple(range(320)):
        raise ValueError(
            "ModelRunner published binding phase map is invalid"
        )
    return {
        "binding_results": [
            {
                "binding_index": index,
                "phase_name": binding_phase[index],
                "destination_sha256": digest,
            }
            for index, digest in enumerate(binding_hashes)
        ],
        "phase_results": [
            {
                "phase_name": name,
                "destination_sha256": phase_hashes[name],
            }
            for name, _indices in phase_binding_runs
        ],
        "aggregate_destination_sha256": aggregate_hash,
    }


def _model_runner_method_nodes(source):
    tree = ast.parse(source, filename=MODEL_RUNNER_SOURCE)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    ]
    if len(classes) != 1:
        raise ValueError("ModelRunner class identity is invalid")
    methods = {}
    for name, arguments in METHOD_ARGUMENTS.items():
        matches = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef)
            and node.name == name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"ModelRunner {name} method identity is invalid"
            )
        method = matches[0]
        if tuple(
            argument.arg for argument in method.args.args
        ) != arguments:
            raise ValueError(
                f"ModelRunner {name} method arguments are invalid"
            )
        if (
            method.args.posonlyargs
            or method.args.vararg is not None
            or method.args.kwonlyargs
            or method.args.kwarg is not None
        ):
            raise ValueError(
                f"ModelRunner {name} method signature is invalid"
            )
        methods[name] = method
    return methods


def _validate_method_structure(methods):
    publish = methods[
        "publish_qwen35_loaded_checkpoint_candidate"
    ]
    publish_calls = [
        node
        for node in ast.walk(publish)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "publish"
    ]
    if len(publish_calls) != 1:
        raise ValueError(
            "ModelRunner publication method structure is invalid"
        )

    owner = methods["bind_qwen35_hybrid_model_owner"]
    owner_assignments = {
        target.attr: node.lineno
        for node in ast.walk(owner)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr in {
            "qwen35_hybrid_model_owner",
            "hybrid_state_runtime_bridge",
        }
    }
    if set(owner_assignments) != {
        "qwen35_hybrid_model_owner",
        "hybrid_state_runtime_bridge",
    }:
        raise ValueError(
            "ModelRunner owner binder mutation structure is invalid"
        )

    candidate = methods["bind_qwen35_loaded_checkpoint_candidate"]
    candidate_calls = [
        node
        for node in ast.walk(candidate)
        if isinstance(node, ast.Call)
    ]
    identity_calls = [
        node
        for node in candidate_calls
        if isinstance(node.func, ast.Name)
        and node.func.id
        == "_bind_qwen35_hybrid_prefix_runtime_identity"
    ]
    owner_calls = [
        node
        for node in candidate_calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "bind_qwen35_hybrid_model_owner"
    ]
    if (
        len(identity_calls) != 1
        or len(owner_calls) != 1
        or identity_calls[0].lineno >= owner_calls[0].lineno
    ):
        raise ValueError(
            "ModelRunner candidate binder call ordering is invalid"
        )
    identity_assignments = {
        target.attr: node.lineno
        for node in ast.walk(candidate)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr in {
            "qwen35_hybrid_prefix_runtime_identity",
            "qwen35_hybrid_prefix_runtime_identity_owner",
        }
    }
    if (
        set(identity_assignments)
        != {
            "qwen35_hybrid_prefix_runtime_identity",
            "qwen35_hybrid_prefix_runtime_identity_owner",
        }
        or any(
            lineno <= owner_calls[0].lineno
            for lineno in identity_assignments.values()
        )
    ):
        raise ValueError(
            "ModelRunner candidate binder mutation ordering is invalid"
        )

    outer = methods[
        "bind_published_qwen35_loaded_checkpoint_candidate"
    ]
    outer_calls = [
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "bind_qwen35_loaded_checkpoint_candidate"
    ]
    handlers = [
        handler
        for node in ast.walk(outer)
        if isinstance(node, ast.Try)
        for handler in node.handlers
    ]
    if (
        len(outer_calls) != 1
        or len(handlers) != 1
        or not isinstance(handlers[0].type, ast.Name)
        or handlers[0].type.id != "Exception"
    ):
        raise ValueError(
            "ModelRunner published binder exception structure is invalid"
        )


def load_frozen_model_runner_published_binding_methods(
    source_root,
    *,
    owner_type,
    candidate_type,
    identity_binder,
):
    path = Path(source_root) / MODEL_RUNNER_SOURCE
    payload = path.read_bytes()
    if _sha256(payload) != MODEL_RUNNER_FILE_SHA256:
        raise ValueError("ModelRunner source hash is invalid")
    source = payload.decode("utf-8")
    methods = _model_runner_method_nodes(source)
    for name, method in methods.items():
        segment = ast.get_source_segment(source, method)
        if (
            segment is None
            or _sha256(segment.encode("utf-8"))
            != METHOD_SOURCE_SHA256[name]
        ):
            raise ValueError(
                f"ModelRunner {name} method source hash is invalid"
            )
    _validate_method_structure(methods)
    module = ast.Module(
        body=[
            methods[name]
            for name in METHOD_SOURCE_SHA256
        ],
        type_ignores=[],
    )
    for method in module.body:
        method.decorator_list = []
    namespace = {
        "Qwen35HybridModelOwner": owner_type,
        "Qwen35LoadedCheckpointCandidate": candidate_type,
        "_bind_qwen35_hybrid_prefix_runtime_identity": identity_binder,
    }
    exec(
        compile(
            ast.fix_missing_locations(module),
            MODEL_RUNNER_SOURCE,
            "exec",
        ),
        namespace,
    )
    return MappingProxyType({
        name: namespace[name]
        for name in METHOD_SOURCE_SHA256
    })


def execute_model_runner_published_binding_scope(
    *,
    private_graph_factory,
    oracle_row,
    model_fingerprint,
    methods,
    production_slot_factory,
    candidate_validator,
    mode,
    rank,
):
    import torch

    if not callable(private_graph_factory):
        raise ValueError("private_graph_factory must be callable")
    if (
        not isinstance(methods, Mapping)
        or set(methods) != set(METHOD_SOURCE_SHA256)
        or any(not callable(method) for method in methods.values())
    ):
        raise ValueError("methods must contain exact callable methods")
    if not callable(production_slot_factory):
        raise ValueError("production_slot_factory must be callable")
    if not callable(candidate_validator):
        raise ValueError("candidate_validator must be callable")
    if mode not in ("success", "injected_bridge_conflict"):
        raise ValueError(
            "ModelRunner published binding mode is invalid"
        )
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")

    helpers = (
        load_publish_gate.publication_gate.publication
        .ownership.loader_core
    )
    pool_helpers = (
        load_publish_gate.publication_gate.publication
    )

    def execute_nested_scope():
        target, installed_loader = private_graph_factory()
        if not callable(installed_loader):
            raise ValueError("installed_loader must be callable")
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
        if any(
            int(tensor.count_nonzero().item())
            for tensor in selected.values()
        ):
            raise RuntimeError(
                "ModelRunner published binding destination "
                "initialization failed"
            )
        if target._consumed is not False:
            raise ValueError(
                "ModelRunner published binding target must be unconsumed"
            )
        production_slot = production_slot_factory()
        if production_slot.candidate is not None:
            raise ValueError(
                "ModelRunner published binding slot must start empty"
            )
        adapter_call_count = 1
        candidate = installed_loader()
        if target._consumed is not True:
            raise RuntimeError(
                "ModelRunner published binding target was not consumed"
            )
        result = candidate_validator(
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

        class RunnerShell:
            pass

        runner = RunnerShell()
        runner.rank = rank
        runner.model = model
        runner.qwen35_loaded_checkpoint_candidate_slot = production_slot
        runner.qwen35_hybrid_model_owner = None
        runner.qwen35_hybrid_prefix_restore_owner = None
        runner.qwen35_hybrid_prefix_restore_participant = None
        runner.qwen35_hybrid_prefix_publication_participant = None
        runner.qwen35_hybrid_prefix_runtime_identity = None
        runner.qwen35_hybrid_prefix_runtime_identity_owner = None
        injected_bridge = None
        if mode == "success":
            runner.hybrid_state_runtime_bridge = None
        else:
            injected_bridge = RunnerShell()
            runner.hybrid_state_runtime_bridge = injected_bridge

        owner_binding_method_call_count = 0
        candidate_binding_method_call_count = 0

        def bind_owner(value):
            nonlocal owner_binding_method_call_count
            owner_binding_method_call_count += 1
            return methods["bind_qwen35_hybrid_model_owner"](
                runner,
                value,
            )

        def bind_candidate(value):
            nonlocal candidate_binding_method_call_count
            candidate_binding_method_call_count += 1
            return methods["bind_qwen35_loaded_checkpoint_candidate"](
                runner,
                value,
            )

        runner.bind_qwen35_hybrid_model_owner = bind_owner
        runner.bind_qwen35_loaded_checkpoint_candidate = bind_candidate
        publication_method_call_count = 1
        published = methods[
            "publish_qwen35_loaded_checkpoint_candidate"
        ](runner, candidate)
        if published is not candidate:
            raise ValueError(
                "ModelRunner published binding publication return "
                "is invalid"
            )
        if (
            production_slot.candidate is not candidate
            or production_slot.owner is not candidate.owner
            or production_slot.model_fingerprint != model_fingerprint
        ):
            raise ValueError(
                "ModelRunner published binding slot visibility is invalid"
            )
        outer_binding_method_call_count = 1
        method_row = methods[
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ](runner)
        layout_fingerprint = candidate.owner.pool.layout.fingerprint
        expected_row = {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound" if mode == "success" else "error",
            "model_fingerprint": (
                model_fingerprint if mode == "success" else ""
            ),
            "layout_fingerprint": (
                layout_fingerprint if mode == "success" else ""
            ),
            "dtype": "bfloat16" if mode == "success" else "",
            "detail": (
                ""
                if mode == "success"
                else (
                    "RuntimeError: a different hybrid state runtime "
                    "bridge is already installed"
                )
            ),
        }
        if method_row != expected_row:
            raise ValueError(
                "ModelRunner published binding method row is invalid"
            )
        runtime_identity = (
            runner.qwen35_hybrid_prefix_runtime_identity
        )
        if mode == "success":
            if (
                runner.qwen35_hybrid_model_owner is not candidate.owner
                or runner.hybrid_state_runtime_bridge
                is not candidate.owner.runtime_bridge
                or runtime_identity is None
                or runtime_identity.model_fingerprint
                != model_fingerprint
                or runtime_identity.layout_fingerprint
                != layout_fingerprint
                or runtime_identity.dtype != torch.bfloat16
                or runner
                .qwen35_hybrid_prefix_runtime_identity_owner
                is not candidate.owner
            ):
                raise ValueError(
                    "ModelRunner published binding success state "
                    "is invalid"
                )
            owner_binding_visible = True
            runtime_bridge_binding_visible = True
            runtime_identity_binding_visible = True
            runtime_identity_owner_visible = True
            injected_bridge_preserved = False
            binding_state_pristine = False
        else:
            if (
                runner.hybrid_state_runtime_bridge is not injected_bridge
                or runner.qwen35_hybrid_model_owner is not None
                or runtime_identity is not None
                or runner
                .qwen35_hybrid_prefix_runtime_identity_owner
                is not None
            ):
                raise ValueError(
                    "ModelRunner published binding rejection state "
                    "is invalid"
                )
            owner_binding_visible = False
            runtime_bridge_binding_visible = False
            runtime_identity_binding_visible = False
            runtime_identity_owner_visible = False
            injected_bridge_preserved = True
            binding_state_pristine = True

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
                raise RuntimeError(
                    "ModelRunner published binding destination clear "
                    "is incomplete"
                )
            if any(
                not tensor.equal(non_selected_values[id(tensor)])
                for tensor in registered
                if id(tensor) not in selected_ids
            ):
                raise RuntimeError(
                    "ModelRunner published binding non-selected tensor "
                    "changed"
                )
            if not pool_helpers._pool_unchanged(pool, pool_snapshot):
                raise RuntimeError(
                    "ModelRunner published binding pool state changed"
                )
        except Exception as caught:
            if clear_error is None:
                clear_error = caught
        if clear_error is not None:
            raise RuntimeError(
                "ModelRunner published binding scope cleanup failed"
            ) from clear_error
        result.update({
            "method_row": method_row,
            "publication_method_call_count": (
                publication_method_call_count
            ),
            "outer_binding_method_call_count": (
                outer_binding_method_call_count
            ),
            "candidate_binding_method_call_count": (
                candidate_binding_method_call_count
            ),
            "owner_binding_method_call_count": (
                owner_binding_method_call_count
            ),
            "adapter_call_count": adapter_call_count,
            "provider_call_count": int(target._consumed),
            "production_publish_call_count": (
                production_slot.publish_calls
                if hasattr(production_slot, "publish_calls")
                else 1
            ),
            "production_slot_visibility_verified": True,
            "published_candidate_identity_verified": True,
            "candidate_installed": mode == "success",
            "owner_binding_visible": owner_binding_visible,
            "runtime_bridge_binding_visible": (
                runtime_bridge_binding_visible
            ),
            "runtime_identity_binding_visible": (
                runtime_identity_binding_visible
            ),
            "runtime_identity_owner_visible": (
                runtime_identity_owner_visible
            ),
            "injected_bridge_preserved": injected_bridge_preserved,
            "binding_state_pristine": binding_state_pristine,
            "layout_fingerprint": layout_fingerprint,
            "dtype": "bfloat16",
            "target_consumed_before": False,
            "target_consumed_after": target._consumed,
            "selected_binding_count": len(
                target.binding_plan.bindings
            ),
            "unique_destination_count": len(selected),
            "alias_groups": helpers._alias_groups(
                target.binding_plan
            ),
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
        })
        references = {
            "runner": weakref.ref(runner),
            "production_slot": weakref.ref(production_slot),
            "candidate": weakref.ref(candidate),
            "owner": weakref.ref(candidate.owner),
            "runtime_bridge": weakref.ref(
                candidate.owner.runtime_bridge
            ),
            "model": weakref.ref(model),
            "pool": weakref.ref(pool),
            "target": weakref.ref(target),
        }
        if mode == "success":
            references["runtime_identity"] = weakref.ref(
                runtime_identity
            )
        else:
            references["injected_bridge"] = weakref.ref(
                injected_bridge
            )
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
            "ModelRunner published binding objects escaped scope: "
            + ", ".join(escaped)
        )
    result["collected_private_objects"] = collected
    result["all_private_binding_objects_collected"] = True
    return result


def run_model_runner_published_binding_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    prerequisite_artifact,
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
    prerequisite = load_model_runner_published_binding_prerequisite(
        prerequisite_artifact
    )
    oracle_row = (
        select_model_runner_published_binding_prerequisite_row(
            prerequisite,
            tensor_parallel_size,
            tensor_parallel_rank,
        )
    )
    candidate_validation_oracle = (
        reconstruct_candidate_validation_oracle(oracle_row)
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
    owner_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_model_owner",
        fromlist=["*"],
    )
    identity_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity",
        fromlist=["*"],
    )
    publication_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_model_publication",
        fromlist=["*"],
    )
    methods = load_frozen_model_runner_published_binding_methods(
        source_root,
        owner_type=owner_module.Qwen35HybridModelOwner,
        candidate_type=(
            streaming_module.Qwen35LoadedCheckpointCandidate
        ),
        identity_binder=(
            identity_module.bind_qwen35_hybrid_prefix_runtime_identity
        ),
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    diagnostics = {
        "provider_call_count": 0,
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
                    "ModelRunner published binding target provider "
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

        def load_candidate():
            return adapter(request)

        return target, load_candidate

    result = execute_model_runner_published_binding_scope(
        private_graph_factory=private_graph_factory,
        oracle_row=candidate_validation_oracle,
        model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
        methods=methods,
        production_slot_factory=(
            lambda: publication_module
            .Qwen35HybridModelOwnerPublicationSlot()
        ),
        candidate_validator=(
            load_publish_gate.publication_gate.publication
            .ownership.validate_private_loaded_candidate
        ),
        mode=mode,
        rank=tensor_parallel_rank,
    )
    if diagnostics["provider_call_count"] != 1:
        raise RuntimeError(
            "ModelRunner published binding provider count is invalid"
        )
    result["provider_call_count"] = diagnostics["provider_call_count"]
    after_binding_clear = _memory_point(status_reader())
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": diagnostics["after_metadata"],
        "after_pool": diagnostics["after_pool"],
        "after_target": diagnostics["after_target"],
        "after_binding_clear": after_binding_clear,
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
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_method_sha256": dict(METHOD_SOURCE_SHA256),
        "config_sha256": diagnostics["config_sha256"],
        "index_sha256": diagnostics["index_sha256"],
        "config_index_header_sha256": (
            diagnostics["config_index_header_sha256"]
        ),
        "metadata_bytes_read": diagnostics["metadata_bytes_read"],
        "model_forward_count": 0,
        "attention_forward_count": (
            diagnostics["attention_forward_count"]
        ),
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0,
            after_binding_clear["vmhwm_kib"]
            - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_binding_clear["vmhwm_kib"]
            - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_binding_clear["vmhwm_kib"]
            - diagnostics["after_metadata"]["vmhwm_kib"],
        ),
        **result,
    }
    validate_model_runner_published_binding_row(row)
    return row


def _validate_memory(row):
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_binding_clear",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError(
            "ModelRunner published binding memory points are invalid"
        )
    observed = (
        row.get("total_vmhwm_increment_kib"),
        row.get("post_torch_vmhwm_increment_kib"),
        row.get("post_metadata_vmhwm_increment_kib"),
    )
    recomputed = (
        memory["after_binding_clear"]["vmhwm_kib"]
        - memory["before"]["vmhwm_kib"],
        memory["after_binding_clear"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
        memory["after_binding_clear"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    if observed != recomputed:
        raise ValueError(
            "ModelRunner published binding memory deltas are invalid"
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
            "ModelRunner published binding memory ceiling exceeded: "
            + details
        )


def validate_model_runner_published_binding_row(row):
    mode = row.get("mode")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if (
        row.get("schema_version") != ROW_SCHEMA_VERSION
        or mode not in ("success", "injected_bridge_conflict")
        or tp not in PREREQUISITE_ROWS
    ):
        raise ValueError(
            "ModelRunner published binding row schema is invalid"
        )
    exact = {
        "status": "PASS",
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_method_sha256": dict(METHOD_SOURCE_SHA256),
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "publication_method_call_count": 1,
        "outer_binding_method_call_count": 1,
        "candidate_binding_method_call_count": 1,
        "owner_binding_method_call_count": 1,
        "adapter_call_count": 1,
        "provider_call_count": 1,
        "production_publish_call_count": 1,
        "production_slot_visibility_verified": True,
        "published_candidate_identity_verified": True,
        "candidate_installed": mode == "success",
        "owner_binding_visible": mode == "success",
        "runtime_bridge_binding_visible": mode == "success",
        "runtime_identity_binding_visible": mode == "success",
        "runtime_identity_owner_visible": mode == "success",
        "injected_bridge_preserved": (
            mode == "injected_bridge_conflict"
        ),
        "binding_state_pristine": (
            mode == "injected_bridge_conflict"
        ),
        "dtype": "bfloat16",
        "all_private_binding_objects_collected": True,
        "loaded_state_verified": True,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "alias_groups": (
            load_publish_gate.publication_gate.publication.ownership
            .loader_core.ALIAS_GROUPS
        ),
        "target_consumed_before": False,
        "target_consumed_after": True,
        "loader_stats": STREAMED_STATS[tp],
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(
                f"ModelRunner published binding {name} is invalid"
            )
    binding_hashes = row.get("binding_destination_sha256")
    if (
        not isinstance(binding_hashes, list)
        or len(binding_hashes) != 320
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in binding_hashes
        )
    ):
        raise ValueError(
            "ModelRunner published binding "
            "binding_destination_sha256 is invalid"
        )
    phase_hashes = row.get("phase_destination_sha256")
    if (
        not isinstance(phase_hashes, Mapping)
        or len(phase_hashes) != 26
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or len(value) != 64
            for name, value in phase_hashes.items()
        )
    ):
        raise ValueError(
            "ModelRunner published binding "
            "phase_destination_sha256 is invalid"
        )
    aggregate_hash = row.get("aggregate_destination_sha256")
    if (
        not isinstance(aggregate_hash, str)
        or len(aggregate_hash) != 64
    ):
        raise ValueError(
            "ModelRunner published binding "
            "aggregate_destination_sha256 is invalid"
        )
    layout_fingerprint = row.get("layout_fingerprint")
    if (
        not isinstance(layout_fingerprint, str)
        or not layout_fingerprint
    ):
        raise ValueError(
            "ModelRunner published binding layout_fingerprint is invalid"
        )
    expected_row = {
        "participant_id": row.get("tp_rank"),
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound" if mode == "success" else "error",
        "model_fingerprint": (
            APPROVED_MODEL_MANIFEST_SHA256
            if mode == "success"
            else ""
        ),
        "layout_fingerprint": (
            layout_fingerprint if mode == "success" else ""
        ),
        "dtype": "bfloat16" if mode == "success" else "",
        "detail": (
            ""
            if mode == "success"
            else (
                "RuntimeError: a different hybrid state runtime "
                "bridge is already installed"
            )
        ),
    }
    if row.get("method_row") != expected_row:
        raise ValueError(
            "ModelRunner published binding method_row is invalid"
        )
    expected_collected = {
        "runner": True,
        "production_slot": True,
        "candidate": True,
        "owner": True,
        "runtime_bridge": True,
        "model": True,
        "pool": True,
        "target": True,
    }
    expected_collected[
        "runtime_identity"
        if mode == "success"
        else "injected_bridge"
    ] = True
    if row.get("collected_private_objects") != expected_collected:
        raise ValueError(
            "ModelRunner published binding collected objects are invalid"
        )
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError(
            "ModelRunner published binding process ID is invalid"
        )
    _validate_memory(row)
    return row


def validate_model_runner_published_binding_preflight(record):
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
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_method_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
    }
    for name, value in exact.items():
        if record.get(name) != value:
            raise ValueError(
                "ModelRunner published binding preflight "
                f"{name} is invalid"
            )
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, dict)
        or tuple(hashes) != tuple(sorted(SOURCE_FILES))
        or set(hashes) != set(SOURCE_FILES)
    ):
        raise ValueError(
            "ModelRunner published binding source hashes are invalid"
        )
    if hashes[MODEL_RUNNER_SOURCE] != MODEL_RUNNER_FILE_SHA256:
        raise ValueError(
            "ModelRunner published binding source file hash is invalid"
        )
    if record.get("source_tree_sha256") != _source_tree_sha256(
        hashes
    ):
        raise ValueError(
            "ModelRunner published binding source tree is invalid"
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
            "ModelRunner published binding worker rows are invalid"
        )
    for row in rows:
        validate_model_runner_published_binding_row(row)
    if len({row["process_id"] for row in rows}) != len(
        WORKER_CONTEXTS
    ):
        raise ValueError(
            "ModelRunner published binding process IDs must be unique"
        )
    return record


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(
                "missing ModelRunner published binding source: "
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
                    "missing ModelRunner published binding source: "
                    + name
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def audit_published_binding_preflight_source(source_root):
    path = Path(source_root) / (
        "tools/"
        "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py"
    )
    tree = ast.parse(path.read_text(), filename=os.fspath(path))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    def attribute_calls(name):
        return sum(
            isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            for node in calls
        )

    def named_calls(name):
        return sum(
            isinstance(node.func, ast.Name)
            and node.func.id == name
            for node in calls
        )

    method_invocations = {}
    for method_name in METHOD_SOURCE_SHA256:
        method_invocations[method_name] = sum(
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "methods"
            and isinstance(node.func.slice, ast.Constant)
            and node.func.slice.value == method_name
            for node in calls
        )
    model_runner_import_count = sum(
        isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "tinyvllm.engine.model_runner"
        for node in calls
    )
    model_runner_construction_count = (
        named_calls("ModelRunner")
        + attribute_calls("ModelRunner")
    )
    direct_streamed_loader_call_count = attribute_calls(
        "load_qwen35_fresh_checkpoint_candidate"
    ) + named_calls("load_qwen35_fresh_checkpoint_candidate")
    target_take_call_count = attribute_calls("take")
    engine_call_count = (
        named_calls("LLMEngine")
        + attribute_calls("LLMEngine")
        + attribute_calls("step")
    )
    scheduler_call_count = (
        named_calls("Scheduler")
        + attribute_calls("Scheduler")
        + attribute_calls("schedule")
    )
    forward_call_count = (
        named_calls("forward")
        + attribute_calls("forward")
    )
    cuda_is_initialized_call_count = attribute_calls("is_initialized")
    cuda_execution_names = {
        "empty",
        "zeros",
        "ones",
        "tensor",
        "synchronize",
        "current_stream",
        "Stream",
        "Event",
    }
    cuda_execution_call_count = 0
    for node in calls:
        function = node.func
        if (
            isinstance(function, ast.Attribute)
            and function.attr in cuda_execution_names
            and isinstance(function.value, ast.Attribute)
            and isinstance(function.value.value, ast.Name)
            and function.value.value.id == "torch"
            and function.value.attr == "cuda"
        ):
            cuda_execution_call_count += 1
    audit = {
        "adapter_builder_call_count": attribute_calls(
            "build_qwen35_authorized_checkpoint_candidate_loader"
        ),
        "production_slot_constructor_call_count": attribute_calls(
            "Qwen35HybridModelOwnerPublicationSlot"
        ),
        "extracted_method_invocation_count": method_invocations,
        "model_runner_import_count": model_runner_import_count,
        "model_runner_construction_count": (
            model_runner_construction_count
        ),
        "direct_streamed_loader_call_count": (
            direct_streamed_loader_call_count
        ),
        "target_take_call_count": target_take_call_count,
        "engine_call_count": engine_call_count,
        "scheduler_call_count": scheduler_call_count,
        "forward_call_count": forward_call_count,
        "cuda_execution_call_count": cuda_execution_call_count,
        "cuda_is_initialized_call_count": (
            cuda_is_initialized_call_count
        ),
    }
    expected = {
        "adapter_builder_call_count": 1,
        "production_slot_constructor_call_count": 1,
        "extracted_method_invocation_count": {
            name: 1 for name in METHOD_SOURCE_SHA256
        },
        "model_runner_import_count": 0,
        "model_runner_construction_count": 0,
        "direct_streamed_loader_call_count": 0,
        "target_take_call_count": 0,
        "engine_call_count": 0,
        "scheduler_call_count": 0,
        "forward_call_count": 0,
        "cuda_execution_call_count": 0,
        "cuda_is_initialized_call_count": 2,
    }
    if audit != expected:
        raise ValueError(
            "ModelRunner published binding static safety audit "
            f"is invalid: {audit!r}"
        )
    return audit


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
            "published binding prerequisite artifact hash is invalid"
        )
    prerequisite = (
        load_model_runner_published_binding_prerequisite(
            prerequisite_artifact
        )
    )
    local_hashes = _source_hashes(source_root)
    if {
        name: local_hashes[name]
        for name in prerequisite.source_file_sha256
    } != dict(prerequisite.source_file_sha256):
        raise ValueError(
            "ModelRunner published binding source does not match "
            "prerequisite"
        )
    if local_hashes[MODEL_RUNNER_SOURCE] != MODEL_RUNNER_FILE_SHA256:
        raise ValueError(
            "ModelRunner published binding production source was modified"
        )
    audit_published_binding_preflight_source(source_root)
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_source_dir = f"{remote_run_dir}/source"
    remote_prerequisite = (
        f"{remote_run_dir}/"
        "model_runner_load_and_publish_preflight.json"
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
    _require_success(staged, "published binding source staging")
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
        "published binding prerequisite staging",
    )
    script = "\n".join([
        "import ast,hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"prerequisite=pathlib.Path({remote_prerequisite!r})",
        f"names={list(SOURCE_FILES)!r}",
        f"model_name={MODEL_RUNNER_SOURCE!r}",
        f"method_names={list(METHOD_SOURCE_SHA256)!r}",
        "source_hashes={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " source_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "source=(root/model_name).read_text()",
        "tree=ast.parse(source,filename=model_name)",
        "classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ModelRunner']",
        "if len(classes)!=1: raise SystemExit('invalid ModelRunner class')",
        "methods={}",
        "for name in method_names:",
        " nodes=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name==name]",
        " if len(nodes)!=1: raise SystemExit('invalid ModelRunner method: '+name)",
        " segment=ast.get_source_segment(source,nodes[0])",
        " methods[name]=hashlib.sha256(segment.encode()).hexdigest()",
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
    _require_success(verified, "published binding staged hashing")
    remote = json.loads(verified.stdout)
    if remote.get("source") != local_hashes:
        raise ValueError(
            "ModelRunner published binding remote source hash mismatch"
        )
    if (
        remote.get("prerequisite")
        != PREREQUISITE_ARTIFACT_SHA256
    ):
        raise ValueError(
            "ModelRunner published binding remote prerequisite mismatch"
        )
    if remote.get("methods") != dict(METHOD_SOURCE_SHA256):
        raise ValueError(
            "ModelRunner published binding remote method hash mismatch"
        )
    return {
        "remote_source_dir": remote_source_dir,
        "remote_prerequisite_artifact": remote_prerequisite,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote["source"],
        "source_tree_sha256": _source_tree_sha256(local_hashes),
        "prerequisite_artifact_sha256": remote["prerequisite"],
        "model_runner_method_sha256": remote["methods"],
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
        "prerequisite_artifact_sha256": (
            PREREQUISITE_ARTIFACT_SHA256
        ),
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_method_sha256": dict(METHOD_SOURCE_SHA256),
        "fresh_process_per_attempt": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_model_runner_published_binding_preflight(record)
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
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "model_runner_method_sha256": dict(
            staged["model_runner_method_sha256"]
        ),
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_model_runner_published_binding_preflight(
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
            "local ModelRunner published binding directory exists: "
            f"{destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/"
        "model_runner_published_candidate_binding_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py"
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
                "--prerequisite-artifact",
                staged["remote_prerequisite_artifact"],
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
            "ModelRunner published binding rank worker",
        )
        row = json.loads(completed.stdout)
        validate_model_runner_published_binding_row(row)
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
    _require_success(finalized, "published binding finalizer")
    record = json.loads(finalized.stdout)
    validate_model_runner_published_binding_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"]
        != staged["remote_file_sha256"]
        or record["source_tree_sha256"]
        != staged["source_tree_sha256"]
    ):
        raise ValueError(
            "ModelRunner published binding source binding mismatch"
        )
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'model_runner_published_candidate_binding_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'model_runner_published_candidate_binding_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "model_runner_published_candidate_binding_preflight": (
                record
            ),
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "published binding artifact round trip")
    if json.loads(round_trip.stdout) != {
        "model_runner_published_candidate_binding_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError(
            "ModelRunner published binding artifact round-trip mismatch"
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
            / "model_runner_published_candidate_binding_preflight.json",
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


def execute_remote_model_runner_published_binding_preflight(
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
    return run_remote_model_runner_published_binding_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments):
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not approved")
    row = run_model_runner_published_binding_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        prerequisite_artifact=arguments.prerequisite_artifact,
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
            "ModelRunner published binding output already exists"
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
    run_parser.add_argument("--prerequisite-artifact", required=True)
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument(
        "--prerequisite-artifact",
        required=True,
    )
    worker_parser.add_argument("--tp-size", type=int, required=True)
    worker_parser.add_argument("--tp-rank", type=int, required=True)
    worker_parser.add_argument(
        "--attempt-mode",
        choices=("success", "injected_bridge_conflict"),
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
        validate_model_runner_published_binding_preflight(record)
    else:
        record = (
            execute_remote_model_runner_published_binding_preflight(
                arguments.source_root,
                arguments.run_tag,
                prerequisite_artifact=(
                    arguments.prerequisite_artifact
                ),
            )
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
