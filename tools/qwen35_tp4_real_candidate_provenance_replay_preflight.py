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
import socket
import shlex
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


real_binding_gate = _load_sibling(
    "_qwen35_tp4_real_candidate_binding_base",
    "qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py",
)
load_publish_gate = real_binding_gate.load_publish_gate
tp4_gate = _load_sibling(
    "_qwen35_tp4_real_candidate_replay_base",
    "qwen35_tp4_synthetic_binding_oracle_preflight.py",
)


LOAD_PUBLISH_ARTIFACT_SHA256 = (
    "d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18"
)
LOAD_PUBLISH_SOURCE_TREE_SHA256 = (
    "a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8"
)
PUBLISHED_BINDING_ARTIFACT_SHA256 = (
    "79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a"
)
PUBLISHED_BINDING_SOURCE_TREE_SHA256 = (
    "0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785"
)
TP4_REPLAY_ARTIFACT_SHA256 = (
    "803c8fac331eeee82b90013e0b0872de8f079661b6dd1ba43225fb446006cce4"
)
TP4_REPLAY_SOURCE_TREE_SHA256 = (
    "e88236ebe4f97ddecf55004e4bbcdb46a677462f183b6724031d85d8648a6de0"
)
APPROVED_MODEL_DIR = real_binding_gate.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = (
    real_binding_gate.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_CONFIG_SHA256 = real_binding_gate.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = real_binding_gate.APPROVED_INDEX_SHA256
APPROVED_COMPOSITE_SHA256 = real_binding_gate.APPROVED_COMPOSITE_SHA256
APPROVED_SHARD_NAME = real_binding_gate.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = real_binding_gate.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = real_binding_gate.APPROVED_SHARD_SHA256
AUTHORIZATION_SHA256 = real_binding_gate.AUTHORIZATION_SHA256
MAX_TENSOR_BYTES = real_binding_gate.MAX_TENSOR_BYTES
PROVENANCE = "real-checkpoint-derived-serial-rank-replay"
CLAIM_BOUNDARY = "not-live-concurrent-tp4-candidate-binding"
PRODUCER_CONTEXTS = (
    (4, 0),
    (4, 1),
    (4, 2),
    (4, 3),
)
MEMORY_CEILINGS_KIB = {
    "total": 6291456,
    "post_torch": 6029312,
    "post_metadata": 5767168,
}
PRODUCER_ROW_SCHEMA_VERSION = (
    "qwen35.tp4-real-candidate-producer-rank.v1"
)
PROVENANCE_ORACLE_SCHEMA_VERSION = (
    "qwen35.tp4-real-candidate-provenance-oracle.v1"
)
REPLAY_MODES = (
    "tp4_real_replay_success",
    "tp4_real_replay_rank2_model_mismatch",
    "tp4_real_replay_rank2_layout_mismatch",
    "tp4_real_replay_rank2_dtype_mismatch",
)
REPLAY_ROW_SCHEMA_VERSION = (
    "qwen35.tp4-real-candidate-provenance-replay-rank.v1"
)
REPLAY_RESULT_SCHEMA_VERSION = (
    "qwen35.tp4-real-candidate-provenance-replay.v1"
)
_LAYER_RUNS = (
    (0, 1, 15), (1, 15, 29), (10, 29, 43), (11, 43, 54),
    (12, 54, 68), (13, 68, 82), (14, 82, 96), (15, 96, 107),
    (16, 107, 121), (17, 121, 135), (18, 135, 149),
    (19, 149, 160), (2, 160, 174), (20, 174, 188),
    (21, 188, 202), (22, 202, 216), (23, 216, 227),
    (3, 227, 238), (4, 238, 252), (5, 252, 266),
    (6, 266, 280), (7, 280, 291), (8, 291, 305),
    (9, 305, 319),
)
PHASE_BINDING_RUNS = (
    ("embed_tokens", (0,)),
    *((f"layer_{layer}", tuple(range(start, stop)))
      for layer, start, stop in _LAYER_RUNS),
    ("final_norm", (319,)),
)
SOURCE_FILES = (
    *tp4_gate.SOURCE_FILES,
    "tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py",
)
REMOTE_TARGET = load_publish_gate.REMOTE_TARGET
REMOTE_PYTHON = load_publish_gate.REMOTE_PYTHON
LOCAL_RUN_ROOT = load_publish_gate.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-tp4-real-candidate-provenance-replay-runs"
)
METHOD_SOURCE_SHA256 = MappingProxyType({
    "load_and_publish_qwen35_checkpoint_candidate": (
        "9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612"
    ),
    "bind_published_qwen35_loaded_checkpoint_candidate": (
        "aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd"
    ),
    "write_shm": tp4_gate.METHOD_SOURCE_SHA256["write_shm"],
    "read_shm": tp4_gate.METHOD_SOURCE_SHA256["read_shm"],
    "loop": tp4_gate.METHOD_SOURCE_SHA256["loop"],
    "dispatch_command": (
        tp4_gate.METHOD_SOURCE_SHA256["dispatch_command"]
    ),
    "call_model_runner_acknowledged": (
        tp4_gate.METHOD_SOURCE_SHA256[
            "call_model_runner_acknowledged"
        ]
    ),
    "bind_qwen35_loaded_checkpoint_candidates": (
        tp4_gate.METHOD_SOURCE_SHA256[
            "bind_qwen35_loaded_checkpoint_candidates"
        ]
    ),
})
AUTHORIZED_SOURCE_DELTAS = MappingProxyType({
    "tinyvllm/layers/linear.py": (
        "dba95145c0cc83b726694ddaae9de7a12206cacd3ecacc17403b3293cfe57b83",
        "9e4bbccd0fbaa4b901796884900a5ca203cbdeabce5049fdd655ddd7ad2bbcd8",
    ),
    "tinyvllm/models/qwen35_checkpoint_binding.py": (
        "9b54bdac2269ed943a2f7951ec03954c71c00b7f5aec8b9540fc4fde83d23012",
        "69578fe68404bfc6db58eac8664bd8cc23fcce84abe5f13cf9e9124fa2824b90",
    ),
    "tinyvllm/models/qwen35_components.py": (
        "c106f5598f5cb4f6af908089da233d5f20489195868c31c1fd1a532f9238ea3c",
        "93af914b4e957863b0df18ee99f6dba59120089bdb7ffe77fe32d5c11dcaa5c4",
    ),
})
_install_namespace_packages = load_publish_gate._install_namespace_packages
_atomic_write_json = load_publish_gate._atomic_write_json
validate_run_tag = load_publish_gate.validate_run_tag
build_ssh_command = load_publish_gate.build_ssh_command
_require_success = load_publish_gate._require_success


def _load_runtime_module(name):
    return __import__(name, fromlist=["*"])


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_exact_json(path, expected_sha256, label):
    payload = Path(path).read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError(f"{label} prerequisite hash is invalid")
    try:
        value = __import__("json").loads(payload)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{label} prerequisite JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} prerequisite must be an object")
    return value


@dataclass(frozen=True)
class TP4RealCandidatePrerequisites:
    load_publish_artifact_sha256: str
    published_binding_artifact_sha256: str
    tp4_replay_artifact_sha256: str
    approved_model_manifest_sha256: str
    provenance: str
    claim_boundary: str
    inherited_source_file_sha256: Mapping


def load_tp4_real_candidate_prerequisites(
    load_publish_artifact,
    published_binding_artifact,
    tp4_replay_artifact,
):
    load_publish = _read_exact_json(
        load_publish_artifact,
        LOAD_PUBLISH_ARTIFACT_SHA256,
        "load-and-publish",
    )
    load_publish_gate.validate_model_runner_load_publish_preflight(
        load_publish
    )
    if (
        load_publish.get("source_tree_sha256")
        != LOAD_PUBLISH_SOURCE_TREE_SHA256
    ):
        raise ValueError(
            "load-and-publish prerequisite source tree is invalid"
        )
    published_binding = _read_exact_json(
        published_binding_artifact,
        PUBLISHED_BINDING_ARTIFACT_SHA256,
        "published-binding",
    )
    real_binding_gate.validate_model_runner_published_binding_preflight(
        published_binding
    )
    if (
        published_binding.get("source_tree_sha256")
        != PUBLISHED_BINDING_SOURCE_TREE_SHA256
    ):
        raise ValueError(
            "published-binding prerequisite source tree is invalid"
        )
    tp4_replay = _read_exact_json(
        tp4_replay_artifact,
        TP4_REPLAY_ARTIFACT_SHA256,
        "TP4 replay",
    )
    tp4_gate.validate_synthetic_binding_preflight(tp4_replay)
    if (
        tp4_replay.get("source_tree_sha256")
        != TP4_REPLAY_SOURCE_TREE_SHA256
    ):
        raise ValueError("TP4 replay prerequisite source tree is invalid")
    inherited = tp4_replay.get("source_file_sha256")
    if (
        not isinstance(inherited, dict)
        or tuple(inherited) != tuple(sorted(tp4_gate.SOURCE_FILES))
        or set(inherited) != set(tp4_gate.SOURCE_FILES)
    ):
        raise ValueError(
            "TP4 replay prerequisite source closure is invalid"
        )
    return TP4RealCandidatePrerequisites(
        load_publish_artifact_sha256=LOAD_PUBLISH_ARTIFACT_SHA256,
        published_binding_artifact_sha256=(
            PUBLISHED_BINDING_ARTIFACT_SHA256
        ),
        tp4_replay_artifact_sha256=TP4_REPLAY_ARTIFACT_SHA256,
        approved_model_manifest_sha256=(
            APPROVED_MODEL_MANIFEST_SHA256
        ),
        provenance=PROVENANCE,
        claim_boundary=CLAIM_BOUNDARY,
        inherited_source_file_sha256=MappingProxyType(inherited),
    )


def load_frozen_tp4_real_candidate_methods(source_root):
    class LoadedCandidate:
        pass

    class Owner:
        pass

    def validate_request(request):
        return request

    def bind_identity(owner, model_fingerprint):
        return owner, model_fingerprint

    load_publish = (
        load_publish_gate.load_frozen_model_runner_load_publish_method(
            source_root,
            loaded_candidate_type=LoadedCandidate,
            request_validator=validate_request,
        )
    )
    binding_methods = (
        real_binding_gate.load_frozen_model_runner_published_binding_methods(
            source_root,
            owner_type=Owner,
            candidate_type=LoadedCandidate,
            identity_binder=bind_identity,
        )
    )
    replay_methods = tp4_gate.load_frozen_synthetic_binding_methods(
        source_root
    )
    return MappingProxyType({
        "load_and_publish_qwen35_checkpoint_candidate": load_publish,
        "bind_published_qwen35_loaded_checkpoint_candidate": (
            binding_methods[
                "bind_published_qwen35_loaded_checkpoint_candidate"
            ]
        ),
        **{
            name: replay_methods[name]
            for name in (
                "write_shm",
                "read_shm",
                "loop",
                "dispatch_command",
                "call_model_runner_acknowledged",
                "bind_qwen35_loaded_checkpoint_candidates",
            )
        },
    })


def _build_authorized_loader(
    build_qwen35_authorized_checkpoint_candidate_loader,
    *args,
    **kwargs,
):
    return build_qwen35_authorized_checkpoint_candidate_loader(
        *args,
        **kwargs,
    )


def _run_one_producer_process(process):
    process.start()
    process.join()
    return process.exitcode


def _source_hashes(source_root):
    root = Path(source_root)
    hashes = {}
    for name in SOURCE_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(
                "missing TP4 real-candidate source: " + name
            )
        hashes[name] = _sha256(path.read_bytes())
    return dict(sorted(hashes.items()))


def _validate_authorized_source_delta(
    local_hashes,
    inherited_hashes,
):
    local = dict(local_hashes)
    inherited = dict(inherited_hashes)
    if set(inherited) - set(local):
        raise ValueError(
            "TP4 real-candidate inherited source closure is incomplete"
        )
    observed = {}
    for name, expected_before in inherited.items():
        actual = local[name]
        authorized = AUTHORIZED_SOURCE_DELTAS.get(name)
        if authorized is None:
            if actual != expected_before:
                raise ValueError(
                    "TP4 real-candidate source does not match prerequisite"
                )
            continue
        before, after = authorized
        if expected_before != before or actual != after:
            raise ValueError(
                "TP4 real-candidate authorized source delta is invalid"
            )
        observed[name] = (before, after)
    if set(observed) != set(AUTHORIZED_SOURCE_DELTAS):
        raise ValueError(
            "TP4 real-candidate authorized source delta is incomplete"
        )
    return dict(sorted(observed.items()))


def validate_tp4_real_candidate_producer_row(row):
    tp = (row.get("tp_size"), row.get("tp_rank"))
    exact = {
        "schema_version": PRODUCER_ROW_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "model_runner_file_sha256": (
            real_binding_gate.MODEL_RUNNER_FILE_SHA256
        ),
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "metadata_bytes_read": 144024,
        "adapter_call_count": 1,
        "provider_call_count": 1,
        "load_publish_method_call_count": 1,
        "bind_method_call_count": 1,
        "owner_binding_method_call_count": 1,
        "candidate_binding_method_call_count": 1,
        "production_publish_call_count": 1,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "alias_groups": (
            real_binding_gate.load_publish_gate.publication_gate
            .publication.ownership.loader_core.ALIAS_GROUPS
        ),
        "loader_stats": {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": MAX_TENSOR_BYTES,
        },
        "dtype": "bfloat16",
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_private_objects_collected": True,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
    }
    if tp not in PRODUCER_CONTEXTS:
        raise ValueError("TP4 real-candidate producer rank is invalid")
    for name, expected in exact.items():
        if row.get(name) != expected:
            raise ValueError(
                f"TP4 real-candidate producer {name} is invalid"
            )
    process_id = row.get("process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ValueError(
            "TP4 real-candidate producer process ID is invalid"
        )
    layout = row.get("layout_fingerprint")
    if (
        not isinstance(layout, str)
        or len(layout) != 64
        or any(character not in "0123456789abcdef" for character in layout)
    ):
        raise ValueError(
            "TP4 real-candidate producer layout fingerprint is invalid"
        )
    method_row = {
        "participant_id": row["tp_rank"],
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound",
        "model_fingerprint": APPROVED_MODEL_MANIFEST_SHA256,
        "layout_fingerprint": layout,
        "dtype": "bfloat16",
        "detail": "",
    }
    if row.get("method_row") != method_row:
        raise ValueError(
            "TP4 real-candidate producer method row is invalid"
        )
    binding_hashes = row.get("binding_destination_sha256")
    if (
        not isinstance(binding_hashes, list)
        or len(binding_hashes) != 320
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
            for value in binding_hashes
        )
    ):
        raise ValueError(
            "TP4 real-candidate producer binding hashes are invalid"
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
            or any(
                character not in "0123456789abcdef"
                for character in value
            )
            for name, value in phase_hashes.items()
        )
    ):
        raise ValueError(
            "TP4 real-candidate producer phase hashes are invalid"
        )
    aggregate = row.get("aggregate_destination_sha256")
    if (
        not isinstance(aggregate, str)
        or len(aggregate) != 64
        or any(
            character not in "0123456789abcdef"
            for character in aggregate
        )
    ):
        raise ValueError(
            "TP4 real-candidate producer aggregate hash is invalid"
        )
    expected_collected = {
        "runner": True,
        "production_slot": True,
        "request": True,
        "candidate": True,
        "owner": True,
        "runtime_bridge": True,
        "runtime_identity": True,
        "model": True,
        "pool": True,
        "target": True,
    }
    if row.get("collected_private_objects") != expected_collected:
        raise ValueError(
            "TP4 real-candidate producer collection is invalid"
        )
    memory = row.get("memory")
    names = {
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_clear",
    }
    if not isinstance(memory, Mapping) or set(memory) != names:
        raise ValueError(
            "TP4 real-candidate producer memory points are invalid"
        )
    observed = (
        row.get("total_vmhwm_increment_kib"),
        row.get("post_torch_vmhwm_increment_kib"),
        row.get("post_metadata_vmhwm_increment_kib"),
    )
    recomputed = (
        memory["after_clear"]["vmhwm_kib"]
        - memory["before"]["vmhwm_kib"],
        memory["after_clear"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
        memory["after_clear"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    if observed != recomputed:
        raise ValueError(
            "TP4 real-candidate producer memory deltas are invalid"
        )
    for value, name in zip(
        observed,
        ("total", "post_torch", "post_metadata"),
    ):
        if value > MEMORY_CEILINGS_KIB[name]:
            raise ValueError(
                "TP4 real-candidate producer memory ceiling exceeded"
            )
    return row


def record_tp4_loaded_candidate_payload(
    *,
    candidate,
    target,
    model_fingerprint,
    tensor_bytes,
    destination_view,
):
    if not callable(tensor_bytes) or not callable(destination_view):
        raise ValueError("payload recorder helpers must be callable")
    model = target.assembly.packed.model
    if (
        candidate.owner.model is not model
        or candidate.binding_plan is not target.binding_plan
        or candidate.model_fingerprint != model_fingerprint
        or getattr(candidate.owner, "pool", target.pool)
        is not target.pool
    ):
        raise ValueError(
            "TP4 real-candidate payload identity is invalid"
        )
    bindings = target.binding_plan.bindings
    if len(bindings) != 320:
        raise ValueError(
            "TP4 real-candidate payload binding count is invalid"
        )
    binding_hashes = []
    phase_digests = {
        name: hashlib.sha256() for name, _ in PHASE_BINDING_RUNS
    }
    aggregate = hashlib.sha256()
    phase_by_index = {
        index: name
        for name, indices in PHASE_BINDING_RUNS
        for index in indices
    }
    if tuple(sorted(phase_by_index)) != tuple(range(320)):
        raise ValueError(
            "TP4 real-candidate phase coverage is invalid"
        )
    for index, binding in enumerate(bindings):
        payload = tensor_bytes(destination_view(binding))
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError(
                "TP4 real-candidate destination payload is invalid"
            )
        payload = bytes(payload)
        binding_hashes.append(_sha256(payload))
        phase_digests[phase_by_index[index]].update(payload)
        aggregate.update(payload)
    return {
        "loaded_state_verified": True,
        "binding_hash_count": 320,
        "binding_destination_sha256": binding_hashes,
        "phase_hash_count": 26,
        "phase_destination_sha256": {
            name: digest.hexdigest()
            for name, digest in phase_digests.items()
        },
        "aggregate_destination_sha256": aggregate.hexdigest(),
        "aggregate_hash_verified": True,
    }


def execute_tp4_real_candidate_producer_scope(
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
):
    import torch

    if not callable(private_graph_factory):
        raise ValueError("private_graph_factory must be callable")
    if (
        not isinstance(methods, Mapping)
        or set(methods)
        != {
            "load_and_publish_qwen35_checkpoint_candidate",
            "bind_published_qwen35_loaded_checkpoint_candidate",
        }
        or any(not callable(method) for method in methods.values())
    ):
        raise ValueError(
            "producer methods must contain exact callable methods"
        )
    for name, value in (
        ("bind_owner_method", bind_owner_method),
        ("bind_candidate_method", bind_candidate_method),
        ("production_slot_factory", production_slot_factory),
        ("candidate_validator", candidate_validator),
        ("payload_recorder", payload_recorder),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or (4, rank) not in PRODUCER_CONTEXTS
    ):
        raise ValueError("producer rank must be a TP4 rank")

    helpers = (
        real_binding_gate.load_publish_gate.publication_gate
        .publication.ownership.loader_core
    )
    pool_helpers = (
        real_binding_gate.load_publish_gate.publication_gate
        .publication
    )

    def execute_nested_scope():
        target, request, installed_loader = private_graph_factory()
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
                "TP4 real-candidate destination initialization failed"
            )
        if target._consumed is not False:
            raise ValueError(
                "TP4 real-candidate target must be unconsumed"
            )
        production_slot = production_slot_factory()
        if production_slot.candidate is not None:
            raise ValueError(
                "TP4 real-candidate publication slot must start empty"
            )

        class RunnerShell:
            pass

        adapter_call_count = 0

        def counted_loader(value):
            nonlocal adapter_call_count
            adapter_call_count += 1
            if adapter_call_count != 1:
                raise RuntimeError(
                    "TP4 real-candidate loader called twice"
                )
            return installed_loader(value)

        owner_binding_method_call_count = 0
        candidate_binding_method_call_count = 0

        def bind_owner(value):
            nonlocal owner_binding_method_call_count
            owner_binding_method_call_count += 1
            if owner_binding_method_call_count != 1:
                raise RuntimeError(
                    "TP4 real-candidate owner binder called twice"
                )
            return bind_owner_method(runner, value)

        def bind_candidate(value):
            nonlocal candidate_binding_method_call_count
            candidate_binding_method_call_count += 1
            if candidate_binding_method_call_count != 1:
                raise RuntimeError(
                    "TP4 real-candidate binder called twice"
                )
            return bind_candidate_method(runner, value)

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
        runner.bind_qwen35_hybrid_model_owner = bind_owner
        runner.bind_qwen35_loaded_checkpoint_candidate = bind_candidate

        load_publish_method_call_count = 1
        load_row = methods[
            "load_and_publish_qwen35_checkpoint_candidate"
        ](runner, request)
        expected_load_row = {
            "participant_id": rank,
            "operation": "load_checkpoint_candidate",
            "status": "published",
            "model_fingerprint": model_fingerprint,
            "detail": "",
        }
        if load_row != expected_load_row:
            raise ValueError(
                "TP4 real-candidate load-and-publish row is invalid"
            )
        candidate = production_slot.candidate
        if candidate is None:
            raise RuntimeError(
                "TP4 real-candidate publication is missing"
            )
        if target._consumed is not True:
            raise RuntimeError(
                "TP4 real-candidate target was not consumed"
            )
        result = candidate_validator(
            candidate=candidate,
            target=target,
            model_fingerprint=model_fingerprint,
        )
        if not isinstance(result, dict):
            raise ValueError(
                "TP4 real-candidate validation result is invalid"
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

        bind_method_call_count = 1
        method_row = methods[
            "bind_published_qwen35_loaded_checkpoint_candidate"
        ](runner)
        layout_fingerprint = candidate.owner.pool.layout.fingerprint
        expected_method_row = {
            "participant_id": rank,
            "operation": "bind_loaded_checkpoint_candidate",
            "status": "bound",
            "model_fingerprint": model_fingerprint,
            "layout_fingerprint": layout_fingerprint,
            "dtype": "bfloat16",
            "detail": "",
        }
        if method_row != expected_method_row:
            raise ValueError(
                "TP4 real-candidate binding row is invalid: "
                f"actual={method_row!r}, expected={expected_method_row!r}"
            )
        runtime_identity = (
            runner.qwen35_hybrid_prefix_runtime_identity
        )
        if (
            runner.qwen35_hybrid_model_owner is not candidate.owner
            or runner.hybrid_state_runtime_bridge
            is not candidate.owner.runtime_bridge
            or runtime_identity is None
            or runtime_identity.model_fingerprint != model_fingerprint
            or runtime_identity.layout_fingerprint
            != layout_fingerprint
            or runtime_identity.dtype != torch.bfloat16
            or runner.qwen35_hybrid_prefix_runtime_identity_owner
            is not candidate.owner
        ):
            raise ValueError(
                "TP4 real-candidate bound state is invalid"
            )

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
                    "TP4 real-candidate destination clear is incomplete"
                )
            if any(
                not tensor.equal(non_selected_values[id(tensor)])
                for tensor in registered
                if id(tensor) not in selected_ids
            ):
                raise RuntimeError(
                    "TP4 real-candidate non-selected tensor changed"
                )
            if not pool_helpers._pool_unchanged(
                pool,
                pool_snapshot,
            ):
                raise RuntimeError(
                    "TP4 real-candidate pool state changed"
                )
        except Exception as caught:
            if clear_error is None:
                clear_error = caught
        if clear_error is not None:
            raise RuntimeError(
                "TP4 real-candidate scope cleanup failed"
            ) from clear_error

        result.update({
            "method_row": method_row,
            "load_publish_method_call_count": (
                load_publish_method_call_count
            ),
            "bind_method_call_count": bind_method_call_count,
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
            "selected_binding_count": len(
                target.binding_plan.bindings
            ),
            "unique_destination_count": len(selected),
            "alias_groups": helpers._alias_groups(
                target.binding_plan
            ),
            "layout_fingerprint": layout_fingerprint,
            "dtype": "bfloat16",
            "all_selected_destinations_zero_after_clear": True,
            "non_selected_tensors_unchanged": True,
            "tensor_identity_preserved": True,
            "pool_unchanged": True,
        })
        references = {
            "runner": weakref.ref(runner),
            "production_slot": weakref.ref(production_slot),
            "request": weakref.ref(request),
            "candidate": weakref.ref(candidate),
            "owner": weakref.ref(candidate.owner),
            "runtime_bridge": weakref.ref(
                candidate.owner.runtime_bridge
            ),
            "runtime_identity": weakref.ref(runtime_identity),
            "model": weakref.ref(model),
            "pool": weakref.ref(pool),
            "target": weakref.ref(target),
        }
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
            "TP4 real-candidate objects escaped scope: "
            + repr(escaped)
        )
    result["collected_private_objects"] = collected
    result["all_private_objects_collected"] = True
    return result


def run_serial_tp4_real_candidate_producers(
    *,
    process_factory,
    row_reader,
    pid_is_alive,
):
    for name, value in (
        ("process_factory", process_factory),
        ("row_reader", row_reader),
        ("pid_is_alive", pid_is_alive),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    rows = []
    producer_pids = []
    for tp_size, tp_rank in PRODUCER_CONTEXTS:
        if any(pid_is_alive(pid) for pid in producer_pids):
            raise RuntimeError(
                "previous TP4 real-candidate producer is still alive"
            )
        process = process_factory(tp_size, tp_rank)
        exitcode = _run_one_producer_process(process)
        if exitcode != 0:
            raise RuntimeError(
                "TP4 real-candidate producer failed: "
                f"rank={tp_rank}, exitcode={exitcode}"
            )
        process_id = getattr(process, "pid", None)
        if (
            isinstance(process_id, bool)
            or not isinstance(process_id, int)
            or process_id <= 0
            or process_id in producer_pids
        ):
            raise ValueError(
                "TP4 real-candidate producer PID is invalid"
            )
        if pid_is_alive(process_id):
            raise RuntimeError(
                "TP4 real-candidate producer remained alive after join"
            )
        row = row_reader(tp_size, tp_rank, process)
        validate_tp4_real_candidate_producer_row(row)
        if (
            row["tp_size"] != tp_size
            or row["tp_rank"] != tp_rank
            or row["process_id"] != process_id
        ):
            raise ValueError(
                "TP4 real-candidate producer row identity is invalid"
            )
        rows.append(row)
        producer_pids.append(process_id)
    identities = {
        (
            row["model_manifest_sha256"],
            row["layout_fingerprint"],
            row["dtype"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(
            "TP4 real-candidate producer identities are heterogeneous"
        )
    if any(pid_is_alive(pid) for pid in producer_pids):
        raise RuntimeError(
            "TP4 real-candidate producer remained alive after serial run"
        )
    return tuple(rows)


def run_tp4_real_candidate_producer_worker(
    *,
    checkpoint_dir,
    source_root,
    tensor_parallel_size,
    tensor_parallel_rank,
    observed_user,
    observed_hostname,
    process_id,
    status_reader,
    torch_runtime=None,
    runtime_factory=None,
    scope_executor=execute_tp4_real_candidate_producer_scope,
):
    for name, value in (
        ("status_reader", status_reader),
        ("scope_executor", scope_executor),
    ):
        if not callable(value):
            raise ValueError(f"{name} must be callable")
    if (tensor_parallel_size, tensor_parallel_rank) not in (
        PRODUCER_CONTEXTS
    ):
        raise ValueError("producer worker TP context is invalid")
    checkpoint_dir = os.fspath(Path(checkpoint_dir).resolve())
    if checkpoint_dir != APPROVED_MODEL_DIR:
        raise ValueError("producer worker checkpoint is invalid")
    if torch_runtime is None:
        _install_namespace_packages(source_root)
        torch_runtime = _load_runtime_module("torch")
    if runtime_factory is None:
        def backend_factory(diagnostics):
            class Backend(torch_runtime.nn.Module):
                def forward(self, *_args, **_kwargs):
                    diagnostics["attention_forward_count"] += 1
                    raise AssertionError(
                        "attention backend must not execute"
                    )

            return Backend()

        def runtime_factory(
            *,
            checkpoint_dir,
            source_root,
            tensor_parallel_size,
            tensor_parallel_rank,
        ):
            components = (
                build_tp4_real_candidate_producer_components(
                    source_root=source_root,
                    module_loader=_load_runtime_module,
                    torch_runtime=torch_runtime,
                    backend_factory=backend_factory,
                )
            )
            return assemble_tp4_real_candidate_producer_runtime(
                checkpoint_dir=checkpoint_dir,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
                status_reader=status_reader,
                components=components,
            )
    elif not callable(runtime_factory):
        raise ValueError("runtime_factory must be callable")
    before = dict(status_reader())
    torch_runtime.set_num_threads(8)
    cuda_before = torch_runtime.cuda.is_initialized()
    after_torch = dict(status_reader())
    runtime = runtime_factory(
        checkpoint_dir=checkpoint_dir,
        source_root=source_root,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
    )
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(runtime.get("scope_kwargs"), Mapping)
    ):
        raise ValueError("producer worker runtime is invalid")
    result = scope_executor(**runtime["scope_kwargs"])
    if not isinstance(result, dict):
        raise ValueError("producer worker scope result is invalid")
    after_clear = dict(status_reader())
    cuda_after = torch_runtime.cuda.is_initialized()
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": dict(runtime["after_metadata"]),
        "after_pool": dict(runtime["after_pool"]),
        "after_target": dict(runtime["after_target"]),
        "after_clear": after_clear,
    }
    row = {
        "schema_version": PRODUCER_ROW_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "tp_size": tensor_parallel_size,
        "tp_rank": tensor_parallel_rank,
        "process_id": process_id,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "checkpoint_dir": checkpoint_dir,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": runtime["config_sha256"],
        "index_sha256": runtime["index_sha256"],
        "config_index_header_sha256": (
            runtime["config_index_header_sha256"]
        ),
        "authorization_sha256": AUTHORIZATION_SHA256,
        "model_runner_file_sha256": (
            real_binding_gate.MODEL_RUNNER_FILE_SHA256
        ),
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "metadata_bytes_read": runtime["metadata_bytes_read"],
        **result,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": cuda_after,
        "model_forward_count": runtime["model_forward_count"],
        "attention_forward_count": runtime[
            "attention_forward_count"
        ],
        "memory": memory,
        "total_vmhwm_increment_kib": (
            after_clear["vmhwm_kib"] - before["vmhwm_kib"]
        ),
        "post_torch_vmhwm_increment_kib": (
            after_clear["vmhwm_kib"] - after_torch["vmhwm_kib"]
        ),
        "post_metadata_vmhwm_increment_kib": (
            after_clear["vmhwm_kib"]
            - runtime["after_metadata"]["vmhwm_kib"]
        ),
    }
    validate_tp4_real_candidate_producer_row(row)
    return row


def assemble_tp4_real_candidate_producer_runtime(
    *,
    checkpoint_dir,
    tensor_parallel_size,
    tensor_parallel_rank,
    status_reader,
    components,
):
    required = {
        "read_metadata",
        "build_tensor_plan",
        "build_layout",
        "build_pool",
        "prepare_target",
        "build_authorized_loader",
        "build_request",
        "methods",
        "bind_owner_method",
        "bind_candidate_method",
        "production_slot_factory",
        "candidate_validator",
        "payload_recorder",
    }
    if not isinstance(components, Mapping) or set(components) != required:
        raise ValueError(
            "producer runtime components must be exact"
        )
    for name in (
        "read_metadata",
        "build_tensor_plan",
        "build_layout",
        "build_pool",
        "prepare_target",
        "build_authorized_loader",
        "build_request",
    ):
        if not callable(components[name]):
            raise ValueError(
                f"producer runtime component {name} must be callable"
            )
    runtime = {
        "model_forward_count": 0,
        "attention_forward_count": 0,
    }

    def private_graph_factory():
        metadata = components["read_metadata"](checkpoint_dir)
        runtime["after_metadata"] = dict(status_reader())
        runtime["metadata_bytes_read"] = (
            metadata.metadata_bytes_read
        )
        runtime["config_sha256"] = metadata.config_sha256
        runtime["index_sha256"] = metadata.index_sha256
        runtime["config_index_header_sha256"] = (
            metadata.config_index_header_sha256
        )
        tensor_plan = components["build_tensor_plan"](metadata)
        layout = components["build_layout"](
            metadata,
            tensor_parallel_size,
        )
        pool = components["build_pool"](layout)
        runtime["after_pool"] = dict(status_reader())
        target = components["prepare_target"](
            metadata,
            tensor_plan,
            pool,
            tensor_parallel_size,
            tensor_parallel_rank,
        )
        runtime["after_target"] = dict(status_reader())

        def provide_target():
            return target

        adapter = _build_authorized_loader(
            components["build_authorized_loader"],
            provide_target,
        )
        request = components["build_request"](checkpoint_dir)
        return target, request, adapter

    runtime["scope_kwargs"] = {
        "private_graph_factory": private_graph_factory,
        "model_fingerprint": APPROVED_MODEL_MANIFEST_SHA256,
        "methods": components["methods"],
        "bind_owner_method": components["bind_owner_method"],
        "bind_candidate_method": components[
            "bind_candidate_method"
        ],
        "production_slot_factory": components[
            "production_slot_factory"
        ],
        "candidate_validator": components[
            "candidate_validator"
        ],
        "payload_recorder": components["payload_recorder"],
        "rank": tensor_parallel_rank,
    }
    return runtime


def build_tp4_real_candidate_producer_components(
    *,
    source_root,
    module_loader,
    torch_runtime,
    backend_factory,
    frozen_method_loader=None,
    binding_method_loader=None,
):
    if not callable(module_loader):
        raise ValueError("module_loader must be callable")
    if not callable(backend_factory):
        raise ValueError("backend_factory must be callable")
    metadata_module = module_loader(
        "tinyvllm.models.qwen35_checkpoint_metadata"
    )
    checkpoint_module = module_loader(
        "tinyvllm.models.qwen35_checkpoint"
    )
    hybrid_module = module_loader(
        "tinyvllm.engine.hybrid_state"
    )
    layout_module = module_loader(
        "tinyvllm.engine.qwen35_hybrid_state"
    )
    factory_module = module_loader(
        "tinyvllm.models.qwen35_checkpoint_candidate_factory"
    )
    candidate_loader_module = module_loader(
        "tinyvllm.models.qwen35_checkpoint_candidate_loader"
    )
    worker_module = module_loader(
        "tinyvllm.models.qwen35_checkpoint_worker"
    )
    streaming_module = module_loader(
        "tinyvllm.models.qwen35_checkpoint_streaming"
    )
    owner_module = module_loader(
        "tinyvllm.engine.qwen35_hybrid_model_owner"
    )
    identity_module = module_loader(
        "tinyvllm.engine.qwen35_hybrid_prefix_runtime_identity"
    )
    publication_module = module_loader(
        "tinyvllm.engine.qwen35_hybrid_model_publication"
    )
    if frozen_method_loader is None:
        def frozen_method_loader(root):
            return {
                "load_and_publish_qwen35_checkpoint_candidate": (
                    load_publish_gate
                    .load_frozen_model_runner_load_publish_method(
                        root,
                        loaded_candidate_type=(
                            streaming_module
                            .Qwen35LoadedCheckpointCandidate
                        ),
                        request_validator=(
                            worker_module
                            .validate_qwen35_checkpoint_candidate_load_request
                        ),
                    )
                ),
                "bind_published_qwen35_loaded_checkpoint_candidate": (
                    real_binding_gate
                    .load_frozen_model_runner_published_binding_methods(
                        root,
                        owner_type=(
                            owner_module.Qwen35HybridModelOwner
                        ),
                        candidate_type=(
                            streaming_module
                            .Qwen35LoadedCheckpointCandidate
                        ),
                        identity_binder=(
                            identity_module
                            .bind_qwen35_hybrid_prefix_runtime_identity
                        ),
                    )[
                        "bind_published_qwen35_loaded_checkpoint_candidate"
                    ]
                ),
            }
    if binding_method_loader is None:
        binding_method_loader = (
            real_binding_gate
            .load_frozen_model_runner_published_binding_methods
        )
    if not callable(frozen_method_loader) or not callable(
        binding_method_loader
    ):
        raise ValueError("producer method loaders must be callable")
    methods = frozen_method_loader(source_root)
    binding_methods = binding_method_loader(
        source_root,
        owner_type=owner_module.Qwen35HybridModelOwner,
        candidate_type=(
            streaming_module.Qwen35LoadedCheckpointCandidate
        ),
        identity_binder=(
            identity_module
            .bind_qwen35_hybrid_prefix_runtime_identity
        ),
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )

    def read_metadata(checkpoint_dir):
        return metadata_module.read_qwen35_checkpoint_metadata(
            checkpoint_dir,
            shards=(shard,),
            expected_config_sha256=APPROVED_CONFIG_SHA256,
            expected_index_sha256=APPROVED_INDEX_SHA256,
            expected_config_index_header_sha256=(
                APPROVED_COMPOSITE_SHA256
            ),
        )

    def build_tensor_plan(metadata):
        return checkpoint_module.build_qwen35_checkpoint_tensor_plan(
            metadata.hf_config,
            metadata.index_payload,
            metadata.shard_headers,
        )

    def build_layout(metadata, tensor_parallel_size):
        return layout_module.build_qwen35_hybrid_state_layout(
            metadata.hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=torch_runtime.bfloat16,
            speculative_tokens=1,
        )

    def build_pool(layout):
        return hybrid_module.HybridStateTensorPool(
            layout,
            capacity=1,
            device="cpu",
        )

    diagnostics = {
        "attention_forward_count": 0,
    }

    def prepare_target(
        metadata,
        tensor_plan,
        pool,
        tensor_parallel_size,
        tensor_parallel_rank,
    ):
        return (
            factory_module.prepare_qwen35_checkpoint_candidate_target(
                metadata.hf_config,
                tensor_plan,
                pool=pool,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_rank=tensor_parallel_rank,
                build_attention_backend=(
                    lambda *_args: backend_factory(diagnostics)
                ),
                parameter_device="cpu",
            )
        )

    def build_authorized_loader(provider):
        return (
            candidate_loader_module
            .build_qwen35_authorized_checkpoint_candidate_loader(
                provider,
                authorization_sha256=AUTHORIZATION_SHA256,
            )
        )

    def build_request(checkpoint_dir):
        return worker_module.Qwen35CheckpointCandidateLoadRequest(
            checkpoint_dir=os.fspath(Path(checkpoint_dir).resolve()),
            model_fingerprint=APPROVED_MODEL_MANIFEST_SHA256,
            max_tensor_bytes=MAX_TENSOR_BYTES,
            authorization_sha256=AUTHORIZATION_SHA256,
        )

    def candidate_validator(
        *,
        candidate,
        target,
        model_fingerprint,
    ):
        if (
            candidate.owner.model
            is not target.assembly.packed.model
            or candidate.binding_plan is not target.binding_plan
            or candidate.model_fingerprint != model_fingerprint
            or candidate.owner.pool is not target.pool
        ):
            raise ValueError(
                "TP4 real-candidate production identity is invalid"
            )
        return {"loaded_state_verified": True}

    helpers = (
        real_binding_gate.load_publish_gate.publication_gate
        .publication.ownership.loader_core
    )

    def payload_recorder(**kwargs):
        return record_tp4_loaded_candidate_payload(
            **kwargs,
            tensor_bytes=helpers._tensor_bytes,
            destination_view=helpers._destination_view,
        )

    return {
        "read_metadata": read_metadata,
        "build_tensor_plan": build_tensor_plan,
        "build_layout": build_layout,
        "build_pool": build_pool,
        "prepare_target": prepare_target,
        "build_authorized_loader": build_authorized_loader,
        "build_request": build_request,
        "methods": methods,
        "bind_owner_method": binding_methods[
            "bind_qwen35_hybrid_model_owner"
        ],
        "bind_candidate_method": binding_methods[
            "bind_qwen35_loaded_checkpoint_candidate"
        ],
        "production_slot_factory": (
            publication_module.Qwen35HybridModelOwnerPublicationSlot
        ),
        "candidate_validator": candidate_validator,
        "payload_recorder": payload_recorder,
    }


def write_tp4_real_candidate_producer_row(
    *,
    output_path,
    worker=run_tp4_real_candidate_producer_worker,
    **worker_kwargs,
):
    if not callable(worker):
        raise ValueError("producer worker must be callable")
    output = Path(output_path)
    if output.exists():
        raise ValueError("producer row output already exists")
    row = worker(**worker_kwargs)
    validate_tp4_real_candidate_producer_row(row)
    _atomic_write_json(output, row)
    return row


def read_tp4_real_candidate_producer_row(
    *,
    output_path,
    tensor_parallel_size,
    tensor_parallel_rank,
    process,
):
    try:
        row = json.loads(Path(output_path).read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("producer row output is invalid") from error
    validate_tp4_real_candidate_producer_row(row)
    if (
        row["tp_size"] != tensor_parallel_size
        or row["tp_rank"] != tensor_parallel_rank
        or row["process_id"] != getattr(process, "pid", None)
    ):
        raise ValueError("producer row process identity is invalid")
    return row


def validate_tp4_real_candidate_provenance_oracle(record):
    exact = {
        "schema_version": PROVENANCE_ORACLE_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "producer_contexts": [
            list(context) for context in PRODUCER_CONTEXTS
        ],
        "all_producers_exited_before_finalization": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(
                f"TP4 real-candidate oracle {name} is invalid"
            )
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, Mapping)
        or tuple(hashes) != tuple(sorted(SOURCE_FILES))
        or set(hashes) != set(SOURCE_FILES)
        or record.get("source_tree_sha256")
        != tp4_gate._source_tree_sha256(hashes)
    ):
        raise ValueError(
            "TP4 real-candidate oracle source closure is invalid"
        )
    rows = record.get("producer_rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError(
            "TP4 real-candidate oracle producer rows are incomplete"
        )
    for row in rows:
        validate_tp4_real_candidate_producer_row(row)
    if [
        (row["tp_size"], row["tp_rank"]) for row in rows
    ] != list(PRODUCER_CONTEXTS):
        raise ValueError(
            "TP4 real-candidate oracle producer ordering is invalid"
        )
    process_ids = [row["process_id"] for row in rows]
    if (
        len(set(process_ids)) != 4
        or record.get("producer_process_ids") != process_ids
    ):
        raise ValueError(
            "TP4 real-candidate oracle producer PIDs are invalid"
        )
    identities = {
        (
            row["model_manifest_sha256"],
            row["layout_fingerprint"],
            row["dtype"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError(
            "TP4 real-candidate oracle identities are heterogeneous"
        )
    if record.get("producer_rows_sha256") != _sha256(
        _canonical(rows)
    ):
        raise ValueError(
            "TP4 real-candidate oracle producer hash is invalid"
        )
    return record


def finalize_tp4_real_candidate_provenance_oracle(
    *,
    rows,
    output_path,
    source_root,
    pid_is_alive,
):
    if not callable(pid_is_alive):
        raise ValueError("pid_is_alive must be callable")
    if not isinstance(rows, (list, tuple)) or len(rows) != 4:
        raise ValueError(
            "TP4 real-candidate producer set is incomplete"
        )
    rows = [dict(row) for row in rows]
    for row in rows:
        validate_tp4_real_candidate_producer_row(row)
    process_ids = [row["process_id"] for row in rows]
    if any(pid_is_alive(pid) for pid in process_ids):
        raise RuntimeError(
            "TP4 real-candidate producer is still alive"
        )
    output = Path(output_path)
    if output.exists():
        raise ValueError("TP4 real-candidate oracle output exists")
    hashes = _source_hashes(source_root)
    record = {
        "schema_version": PROVENANCE_ORACLE_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "authorization_sha256": AUTHORIZATION_SHA256,
        "method_source_sha256": dict(METHOD_SOURCE_SHA256),
        "source_file_sha256": hashes,
        "source_tree_sha256": tp4_gate._source_tree_sha256(hashes),
        "producer_contexts": [
            list(context) for context in PRODUCER_CONTEXTS
        ],
        "producer_process_ids": process_ids,
        "all_producers_exited_before_finalization": True,
        "producer_rows_sha256": _sha256(_canonical(rows)),
        "producer_rows": rows,
    }
    validate_tp4_real_candidate_provenance_oracle(record)
    _atomic_write_json(output, record)
    return record


def build_tp4_real_candidate_replay_cases(oracle):
    validate_tp4_real_candidate_provenance_oracle(oracle)
    producer_payload = _canonical(oracle["producer_rows"])
    baseline = [
        dict(row["method_row"]) for row in oracle["producer_rows"]
    ]
    cases = {}
    changed_fields = {
        REPLAY_MODES[0]: None,
        REPLAY_MODES[1]: "model_fingerprint",
        REPLAY_MODES[2]: "layout_fingerprint",
        REPLAY_MODES[3]: "dtype",
    }
    for mode in REPLAY_MODES:
        rows = [dict(row) for row in baseline]
        field = changed_fields[mode]
        if field is not None:
            rows[2][field] = (
                "float32"
                if field == "dtype"
                else _sha256(f"{field}-mismatch".encode("utf-8"))
            )
        cases[mode] = tuple(
            MappingProxyType(row) for row in rows
        )
    if _canonical(oracle["producer_rows"]) != producer_payload:
        raise RuntimeError(
            "TP4 real-candidate replay mutated producer evidence"
        )
    return MappingProxyType(cases)


def execute_tp4_real_candidate_replay_attempt(
    *,
    source_root,
    oracle,
    mode,
    timeout_s,
    name_prefix,
):
    if mode not in REPLAY_MODES:
        raise ValueError("TP4 real-candidate replay mode is invalid")
    validate_tp4_real_candidate_provenance_oracle(oracle)
    cases = build_tp4_real_candidate_replay_cases(oracle)
    synthetic_mode = tp4_gate.ATTEMPT_MODES[
        REPLAY_MODES.index(mode)
    ]

    class Prerequisites:
        pass

    prerequisites = Prerequisites()
    prerequisites.cases = MappingProxyType({
        synthetic_mode: cases[mode],
    })
    original_loader = tp4_gate.load_synthetic_binding_prerequisites

    def load_real_rows(_tp4_artifact, _oracle_artifact):
        return prerequisites

    tp4_gate.load_synthetic_binding_prerequisites = load_real_rows
    try:
        attempt = tp4_gate.execute_tp4_synthetic_binding_attempt(
            source_root=source_root,
            tp4_artifact="real-provenance-oracle",
            oracle_artifact="real-provenance-oracle",
            mode=synthetic_mode,
            timeout_s=timeout_s,
            name_prefix=name_prefix,
        )
    finally:
        tp4_gate.load_synthetic_binding_prerequisites = original_loader
    attempt["mode"] = mode
    attempt["provenance"] = PROVENANCE
    attempt["claim_boundary"] = CLAIM_BOUNDARY
    attempt["producer_rows_sha256"] = oracle[
        "producer_rows_sha256"
    ]
    attempt["provenance_oracle_sha256"] = _sha256(
        _canonical(oracle)
    )
    return attempt


def build_tp4_real_candidate_replay_row(
    *,
    attempt,
    observed_user,
    observed_hostname,
    process_id,
):
    row = {
        "schema_version": REPLAY_ROW_SCHEMA_VERSION,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        **attempt,
        "process_id": process_id,
    }
    return row


def validate_tp4_real_candidate_replay_row(row, oracle):
    validate_tp4_real_candidate_provenance_oracle(oracle)
    mode = row.get("mode")
    if (
        row.get("schema_version") != REPLAY_ROW_SCHEMA_VERSION
        or row.get("status") != "PASS"
        or mode not in REPLAY_MODES
    ):
        raise ValueError("TP4 real-candidate replay row schema is invalid")
    exact = {
        "observed_user": "sitian",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "producer_rows_sha256": oracle["producer_rows_sha256"],
        "provenance_oracle_sha256": _sha256(_canonical(oracle)),
        "shared_memory_capacity": tp4_gate.SHARED_MEMORY_CAPACITY,
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
            raise ValueError(
                f"TP4 real-candidate replay {name} is invalid"
            )
    for field in (
        "read_count_by_rank",
        "executor_count_by_rank",
        "event_set_count_by_rank",
        "event_wait_count_by_rank",
        "event_clear_count_by_rank",
    ):
        if row.get(field) != {"1": 2, "2": 2, "3": 2}:
            raise ValueError(
                f"TP4 real-candidate replay {field} is invalid"
            )
    process_id = row.get("process_id")
    child_ids = row.get("child_process_ids")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or not isinstance(child_ids, Mapping)
        or tuple(child_ids) != ("1", "2", "3")
        or len(set(child_ids.values())) != 3
        or process_id in set(child_ids.values())
    ):
        raise ValueError(
            "TP4 real-candidate replay process identity is invalid"
        )
    shared_name = row.get("shared_memory_name")
    if (
        not isinstance(shared_name, str)
        or not shared_name
        or shared_name == "tinyvllm"
        or len(shared_name) > 30
    ):
        raise ValueError(
            "TP4 real-candidate replay segment name is invalid"
        )
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
        raise ValueError(
            "TP4 real-candidate replay envelopes are invalid"
        )
    cases = build_tp4_real_candidate_replay_cases(oracle)
    rows = row.get("oracle_rows")
    if rows != [dict(value) for value in cases[mode]]:
        raise ValueError(
            "TP4 real-candidate replay oracle rows are invalid"
        )
    changed = {
        REPLAY_MODES[0]: None,
        REPLAY_MODES[1]: "model_fingerprint",
        REPLAY_MODES[2]: "layout_fingerprint",
        REPLAY_MODES[3]: "dtype",
    }[mode]
    if row.get("authorized_changed_field") != changed:
        raise ValueError(
            "TP4 real-candidate replay mismatch scope is invalid"
        )
    success = mode == REPLAY_MODES[0]
    if (
        row.get("completion_committed") is not success
        or row.get("repeat_zero_binding_dispatch") is not success
    ):
        raise ValueError(
            "TP4 real-candidate replay completion state is invalid"
        )
    if success:
        if (
            row.get("binding_rows") != rows
            or row.get("completion_configuration") != [
                APPROVED_MODEL_MANIFEST_SHA256,
                oracle["producer_rows"][0]["layout_fingerprint"],
                "bfloat16",
                5.0,
            ]
            or row.get("error_detail") != ""
        ):
            raise ValueError(
                "TP4 real-candidate replay success is invalid"
            )
    elif (
        row.get("binding_rows") is not None
        or row.get("completion_configuration") is not None
        or f"mismatch: {changed}"
        not in row.get("error_detail", "")
    ):
        raise ValueError(
            "TP4 real-candidate replay rejection is invalid"
        )
    return row


def finalize_tp4_real_candidate_replay_result(
    *,
    oracle,
    replay_rows,
    source_root,
):
    validate_tp4_real_candidate_provenance_oracle(oracle)
    if (
        not isinstance(replay_rows, (list, tuple))
        or len(replay_rows) != 4
        or [row.get("mode") for row in replay_rows]
        != list(REPLAY_MODES)
    ):
        raise ValueError(
            "TP4 real-candidate replay rows are incomplete"
        )
    rows = [dict(row) for row in replay_rows]
    for row in rows:
        validate_tp4_real_candidate_replay_row(row, oracle)
    outer_ids = [row["process_id"] for row in rows]
    child_ids = [
        process_id
        for row in rows
        for process_id in row["child_process_ids"].values()
    ]
    producer_ids = list(oracle["producer_process_ids"])
    all_ids = producer_ids + outer_ids + child_ids
    if (
        len(set(outer_ids)) != 4
        or len(set(child_ids)) != 12
        or len(set(all_ids)) != len(all_ids)
    ):
        raise ValueError(
            "TP4 real-candidate replay process sets overlap"
        )
    hashes = _source_hashes(source_root)
    record = {
        "schema_version": REPLAY_RESULT_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance_oracle_sha256": _sha256(_canonical(oracle)),
        "producer_rows_sha256": oracle["producer_rows_sha256"],
        "producer_process_ids": producer_ids,
        "replay_outer_process_ids": outer_ids,
        "replay_child_process_ids": child_ids,
        "all_replay_processes_distinct_from_producers": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": tp4_gate._source_tree_sha256(hashes),
        "replay_rows": rows,
    }
    return record


def validate_tp4_real_candidate_replay_result(record, oracle):
    validate_tp4_real_candidate_provenance_oracle(oracle)
    exact = {
        "schema_version": REPLAY_RESULT_SCHEMA_VERSION,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "provenance_oracle_sha256": _sha256(_canonical(oracle)),
        "producer_rows_sha256": oracle["producer_rows_sha256"],
        "producer_process_ids": oracle["producer_process_ids"],
        "all_replay_processes_distinct_from_producers": True,
    }
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(
                f"TP4 real-candidate result {name} is invalid"
            )
    hashes = record.get("source_file_sha256")
    if (
        not isinstance(hashes, Mapping)
        or tuple(hashes) != tuple(sorted(SOURCE_FILES))
        or record.get("source_tree_sha256")
        != tp4_gate._source_tree_sha256(hashes)
    ):
        raise ValueError(
            "TP4 real-candidate result source closure is invalid"
        )
    rows = record.get("replay_rows")
    if (
        not isinstance(rows, list)
        or [row.get("mode") for row in rows] != list(REPLAY_MODES)
    ):
        raise ValueError(
            "TP4 real-candidate result replay rows are invalid"
        )
    for row in rows:
        validate_tp4_real_candidate_replay_row(row, oracle)
    outer_ids = [row["process_id"] for row in rows]
    child_ids = [
        process_id
        for row in rows
        for process_id in row["child_process_ids"].values()
    ]
    if (
        record.get("replay_outer_process_ids") != outer_ids
        or record.get("replay_child_process_ids") != child_ids
        or len(set(oracle["producer_process_ids"] + outer_ids + child_ids))
        != 20
    ):
        raise ValueError(
            "TP4 real-candidate result process identities are invalid"
        )
    return record


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(
                    "missing TP4 real-candidate source: " + name
                )
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def stage_tp4_real_candidate_remote_run(
    source_root,
    run_tag,
    *,
    load_publish_artifact,
    published_binding_artifact,
    tp4_replay_artifact,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    prerequisites = load_tp4_real_candidate_prerequisites(
        load_publish_artifact,
        published_binding_artifact,
        tp4_replay_artifact,
    )
    local_hashes = _source_hashes(source_root)
    inherited = dict(prerequisites.inherited_source_file_sha256)
    _validate_authorized_source_delta(local_hashes, inherited)
    audit_tp4_real_candidate_source(source_root)
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
    _require_success(staged, "TP4 real-candidate source staging")
    remote_inputs = {}
    for label, local_path, filename, expected in (
        (
            "load-publish",
            load_publish_artifact,
            "model_runner_load_and_publish_preflight.json",
            LOAD_PUBLISH_ARTIFACT_SHA256,
        ),
        (
            "published-binding",
            published_binding_artifact,
            "model_runner_published_candidate_binding_preflight.json",
            PUBLISHED_BINDING_ARTIFACT_SHA256,
        ),
        (
            "TP4 replay",
            tp4_replay_artifact,
            "tp4_synthetic_binding_oracle_preflight.json",
            TP4_REPLAY_ARTIFACT_SHA256,
        ),
    ):
        payload = Path(local_path).read_bytes()
        if _sha256(payload) != expected:
            raise ValueError(f"{label} staging artifact is invalid")
        remote_path = f"{remote_run_dir}/{filename}"
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
        _require_success(completed, f"{label} artifact staging")
        remote_inputs[label] = remote_path
    return {
        "remote_run_dir": remote_run_dir,
        "remote_source_dir": remote_source_dir,
        "remote_load_publish_artifact": remote_inputs["load-publish"],
        "remote_published_binding_artifact": (
            remote_inputs["published-binding"]
        ),
        "remote_tp4_replay_artifact": remote_inputs["TP4 replay"],
        "local_file_sha256": local_hashes,
        "source_tree_sha256": tp4_gate._source_tree_sha256(local_hashes),
    }


def _source_manifest(run_tag, staged):
    return {
        "schema_version": REPLAY_RESULT_SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "remote_run_dir": staged["remote_run_dir"],
        "remote_source_dir": staged["remote_source_dir"],
        "load_publish_artifact_sha256": LOAD_PUBLISH_ARTIFACT_SHA256,
        "published_binding_artifact_sha256": (
            PUBLISHED_BINDING_ARTIFACT_SHA256
        ),
        "tp4_replay_artifact_sha256": TP4_REPLAY_ARTIFACT_SHA256,
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
    }


def run_remote_tp4_real_candidate_preflight(
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
            f"local TP4 real-candidate directory exists: {destination}"
        )
    remote_run_dir = staged["remote_run_dir"]
    remote_source_dir = staged["remote_source_dir"]
    worker = (
        f"{remote_source_dir}/tools/"
        "qwen35_tp4_real_candidate_provenance_replay_preflight.py"
    )
    producer_rows = []
    for tp_size, tp_rank in PRODUCER_CONTEXTS:
        output = f"{remote_run_dir}/producer_rank{tp_rank}.json"
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
                "internal-producer-worker",
                "--source-root",
                remote_source_dir,
                "--checkpoint-dir",
                APPROVED_MODEL_DIR,
                "--tp-size",
                str(tp_size),
                "--tp-rank",
                str(tp_rank),
                "--output",
                output,
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(
            completed,
            f"TP4 real-candidate producer rank {tp_rank}",
        )
        row = json.loads(completed.stdout)
        validate_tp4_real_candidate_producer_row(row)
        producer_rows.append(row)
    remote_oracle = (
        f"{remote_run_dir}/tp4_real_candidate_provenance_oracle.json"
    )
    finalized = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            worker,
            "internal-finalize-oracle",
            "--source-root",
            remote_source_dir,
            "--output",
            remote_oracle,
        ]),
        input=json.dumps({"rows": producer_rows}),
        text=True,
        capture_output=True,
    )
    _require_success(finalized, "TP4 real-candidate oracle finalizer")
    oracle = json.loads(finalized.stdout)
    validate_tp4_real_candidate_provenance_oracle(oracle)
    replay_rows = []
    for mode in REPLAY_MODES:
        completed = command_runner(
            build_ssh_command([
                "env",
                "PYTHONDONTWRITEBYTECODE=1",
                REMOTE_PYTHON,
                "-B",
                worker,
                "internal-replay-worker",
                "--source-root",
                remote_source_dir,
                "--oracle",
                remote_oracle,
                "--replay-mode",
                mode,
                "--timeout-s",
                "5.0",
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, f"TP4 real-candidate replay {mode}")
        row = json.loads(completed.stdout)
        validate_tp4_real_candidate_replay_row(row, oracle)
        replay_rows.append(row)
    remote_result = (
        f"{remote_run_dir}/"
        "tp4_real_candidate_provenance_replay_preflight.json"
    )
    finalized = command_runner(
        build_ssh_command([
            "env",
            "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON,
            "-B",
            worker,
            "internal-finalize-result",
            "--source-root",
            remote_source_dir,
            "--oracle",
            remote_oracle,
            "--output",
            remote_result,
        ]),
        input=json.dumps({"rows": replay_rows}),
        text=True,
        capture_output=True,
    )
    _require_success(finalized, "TP4 real-candidate result finalizer")
    record = json.loads(finalized.stdout)
    validate_tp4_real_candidate_replay_result(record, oracle)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError(
            "TP4 real-candidate remote source binding mismatch"
        )
    source_manifest = _source_manifest(run_tag, staged)
    remote_manifest = f"{remote_run_dir}/source_manifest.json"
    published = command_runner(
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
    _require_success(published, "TP4 real-candidate manifest publication")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "tp4_real_candidate_provenance_oracle.json",
            oracle,
        )
        _atomic_write_json(
            temporary
            / "tp4_real_candidate_provenance_replay_preflight.json",
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


def execute_remote_tp4_real_candidate_preflight(
    source_root,
    run_tag,
    *,
    load_publish_artifact,
    published_binding_artifact,
    tp4_replay_artifact,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_tp4_real_candidate_remote_run(
        source_root,
        run_tag,
        load_publish_artifact=load_publish_artifact,
        published_binding_artifact=published_binding_artifact,
        tp4_replay_artifact=tp4_replay_artifact,
        command_runner=command_runner,
    )
    return run_remote_tp4_real_candidate_preflight(
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
    run.add_argument("--load-publish-artifact", required=True)
    run.add_argument("--published-binding-artifact", required=True)
    run.add_argument("--tp4-replay-artifact", required=True)
    producer = subparsers.add_parser("internal-producer-worker")
    producer.add_argument("--source-root", required=True)
    producer.add_argument("--checkpoint-dir", required=True)
    producer.add_argument("--tp-size", type=int, required=True)
    producer.add_argument("--tp-rank", type=int, required=True)
    producer.add_argument("--output", required=True)
    oracle = subparsers.add_parser("internal-finalize-oracle")
    oracle.add_argument("--source-root", required=True)
    oracle.add_argument("--output", required=True)
    replay = subparsers.add_parser("internal-replay-worker")
    replay.add_argument("--source-root", required=True)
    replay.add_argument("--oracle", required=True)
    replay.add_argument("--replay-mode", choices=REPLAY_MODES, required=True)
    replay.add_argument("--timeout-s", type=float, default=5.0)
    result = subparsers.add_parser("internal-finalize-result")
    result.add_argument("--source-root", required=True)
    result.add_argument("--oracle", required=True)
    result.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--oracle", required=True)
    validate.add_argument("--result", required=True)
    return parser


def main(argv=None):
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        record = execute_remote_tp4_real_candidate_preflight(
            arguments.source_root,
            arguments.run_tag,
            load_publish_artifact=arguments.load_publish_artifact,
            published_binding_artifact=(
                arguments.published_binding_artifact
            ),
            tp4_replay_artifact=arguments.tp4_replay_artifact,
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-producer-worker":
        row = write_tp4_real_candidate_producer_row(
            output_path=arguments.output,
            checkpoint_dir=arguments.checkpoint_dir,
            source_root=arguments.source_root,
            tensor_parallel_size=arguments.tp_size,
            tensor_parallel_rank=arguments.tp_rank,
            observed_user=getpass.getuser(),
            observed_hostname=socket.gethostname(),
            process_id=os.getpid(),
            status_reader=lambda: load_publish_gate._memory_point(
                load_publish_gate._read_proc_status()
            ),
        )
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-finalize-oracle":
        payload = json.load(sys.stdin)
        record = finalize_tp4_real_candidate_provenance_oracle(
            rows=payload.get("rows"),
            output_path=arguments.output,
            source_root=arguments.source_root,
            pid_is_alive=lambda pid: Path(f"/proc/{pid}").exists(),
        )
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-replay-worker":
        oracle = json.loads(Path(arguments.oracle).read_bytes())
        attempt = execute_tp4_real_candidate_replay_attempt(
            source_root=arguments.source_root,
            oracle=oracle,
            mode=arguments.replay_mode,
            timeout_s=arguments.timeout_s,
            name_prefix=f"q35-real-{arguments.replay_mode}",
        )
        row = build_tp4_real_candidate_replay_row(
            attempt=attempt,
            observed_user=getpass.getuser(),
            observed_hostname=socket.gethostname(),
            process_id=os.getpid(),
        )
        validate_tp4_real_candidate_replay_row(row, oracle)
        print(json.dumps(row, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "internal-finalize-result":
        oracle = json.loads(Path(arguments.oracle).read_bytes())
        payload = json.load(sys.stdin)
        record = finalize_tp4_real_candidate_replay_result(
            oracle=oracle,
            replay_rows=payload.get("rows"),
            source_root=arguments.source_root,
        )
        validate_tp4_real_candidate_replay_result(record, oracle)
        output = Path(arguments.output)
        if output.exists():
            raise ValueError("TP4 real-candidate result output exists")
        _atomic_write_json(output, record)
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.command == "validate":
        oracle = json.loads(Path(arguments.oracle).read_bytes())
        result = json.loads(Path(arguments.result).read_bytes())
        validate_tp4_real_candidate_replay_result(result, oracle)
        print("TP4 real-candidate provenance replay validated")
        return 0
    raise AssertionError("unreachable")


def audit_tp4_real_candidate_source(source_root):
    path = Path(source_root) / (
        "tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py"
    )
    tree = ast.parse(path.read_text(), filename=os.fspath(path))
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
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

    imported = {
        alias.name
        for node in imports
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in imports
        if isinstance(node, ast.ImportFrom)
    }
    audit = {
        "llm_engine_import_count": int(
            "tinyvllm.engine.llm_engine" in imported
        ),
        "model_runner_import_count": int(
            "tinyvllm.engine.model_runner" in imported
        ),
        "llm_engine_construction_count": (
            named("LLMEngine") + attribute("LLMEngine")
        ),
        "model_runner_construction_count": (
            named("ModelRunner") + attribute("ModelRunner")
        ),
        "fixed_tinyvllm_shared_memory_count": sum(
            isinstance(node.func, ast.Name)
            and node.func.id == "SharedMemory"
            and any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "tinyvllm"
                for keyword in node.keywords
            )
            for node in calls
        ),
        "scheduler_call_count": (
            named("Scheduler") + attribute("schedule")
        ),
        "step_call_count": attribute("step"),
        "cuda_operation_call_count": sum(
            attribute(name)
            for name in (
                "cuda",
                "to_cuda",
                "synchronize",
                "empty_cache",
            )
        ),
        "forward_call_count": (
            named("forward") + attribute("forward")
        ),
        "inference_call_count": (
            named("inference") + attribute("inference")
        ),
        "authorized_loader_builder_call_count": named(
            "build_qwen35_authorized_checkpoint_candidate_loader"
        ),
        "producer_process_start_call_count": attribute("start"),
        "producer_process_join_call_count": attribute("join"),
    }
    forbidden = {
        name: value
        for name, value in audit.items()
        if name not in {
            "authorized_loader_builder_call_count",
            "producer_process_start_call_count",
            "producer_process_join_call_count",
        }
        and value
    }
    if forbidden:
        raise ValueError(
            f"TP4 real-candidate static audit is invalid: {forbidden!r}"
        )
    if (
        audit["authorized_loader_builder_call_count"] != 1
        or audit["producer_process_start_call_count"] != 1
        or audit["producer_process_join_call_count"] != 1
    ):
        raise ValueError(
            "TP4 real-candidate serial contract is invalid"
        )
    return audit


if __name__ == "__main__":
    raise SystemExit(main())
