from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
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


def _load_sibling(name, filename):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


four = _load_sibling(
    "_qwen35_four_layer_complete_base",
    "qwen35_real_checkpoint_four_layer_cadence_preflight.py",
)
base = four.base

SCHEMA_VERSION = "qwen35.real-checkpoint-complete-transaction.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-complete-transaction-rank.v1"
REMOTE_TARGET = four.REMOTE_TARGET
REMOTE_PYTHON = four.REMOTE_PYTHON
APPROVED_MODEL_DIR = four.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = four.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = four.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = four.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = four.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = four.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = four.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = four.APPROVED_COMPOSITE_SHA256
DATA_START = four.DATA_START
LAYER_SCHEDULE = (
    "linear_attention", "linear_attention", "linear_attention",
    "full_attention", "linear_attention", "linear_attention",
    "linear_attention", "full_attention", "linear_attention",
    "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention",
    "full_attention", "linear_attention", "linear_attention",
    "linear_attention", "full_attention", "linear_attention",
    "linear_attention", "linear_attention", "full_attention",
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
PHASE_NAMES = tuple(name for name, _ in PHASE_BINDING_RUNS)
_BINDING_PHASE = {
    index: name
    for name, indices in PHASE_BINDING_RUNS
    for index in indices
}
ALIAS_GROUPS = [
    [12, 13], [26, 27], [40, 41], [45, 46], [65, 66], [79, 80],
    [93, 94], [98, 99], [118, 119], [132, 133], [146, 147],
    [151, 152], [171, 172], [185, 186], [199, 200], [213, 214],
    [218, 219], [229, 230], [249, 250], [263, 264], [277, 278],
    [282, 283], [302, 303], [316, 317],
]
_ALIASED_RIGHTS = {group[1] for group in ALIAS_GROUPS}
UNIQUE_BINDING_ORDER = tuple(
    index for index in range(320) if index not in _ALIASED_RIGHTS
)
_LINEAR = {
    1: {"tile_count": 1826, "range_count": 1826,
        "byte_count": 117629536},
    2: {"tile_count": 917, "range_count": 4744,
        "byte_count": 58819120},
}
_FULL = {
    1: {"tile_count": 1630, "range_count": 1630,
        "byte_count": 104866816},
    2: {"tile_count": 817, "range_count": 4644,
        "byte_count": 52438016},
}
_ROOT = {
    1: {
        "embed_tokens": {"tile_count": 15520, "range_count": 15520,
                         "byte_count": 1017118720},
        "final_norm": {"tile_count": 1, "range_count": 1,
                       "byte_count": 4096},
    },
    2: {
        "embed_tokens": {"tile_count": 7760, "range_count": 7760,
                         "byte_count": 508559360},
        "final_norm": {"tile_count": 1, "range_count": 1,
                       "byte_count": 4096},
    },
}


def _contract(tp_size):
    phases = {}
    for name, _ in PHASE_BINDING_RUNS:
        if name in _ROOT[tp_size]:
            phases[name] = dict(_ROOT[tp_size][name])
        else:
            layer = int(name.split("_")[1])
            phases[name] = dict(
                _FULL[tp_size]
                if LAYER_SCHEDULE[layer] == "full_attention"
                else _LINEAR[tp_size]
            )
    if tp_size == 1:
        aggregate = (58169, 3763655360, 58169, 116338, 7527310720)
        kinds = {"axis0": 38248, "axis1": 12912, "replicated": 79,
                 "segmented_axis0": 6912, "squeeze_axis0": 18}
    else:
        aggregate = (29169, 1881935712, 121017, 242034, 3763871424)
        kinds = {"axis0": 19160, "axis1": 6456, "replicated": 79,
                 "segmented_axis0": 3456, "squeeze_axis0": 18}
    return {
        "tile_count": aggregate[0], "bytes_per_pass": aggregate[1],
        "ranges_per_pass": aggregate[2], "pread_count": aggregate[3],
        "logical_bytes": aggregate[4], "kind_counts": kinds,
        "phases": phases,
    }


COMPLETE_TRANSACTION_CONTRACTS = {
    (1, 0): _contract(1), (2, 0): _contract(2), (2, 1): _contract(2),
}
MEMORY_CEILINGS_KIB = {
    1: {"total": 10485760, "post_torch": 9961472,
        "post_metadata": 9699328},
    2: {"total": 6291456, "post_torch": 6029312,
        "post_metadata": 5767168},
}
SOURCE_FILES = (
    *four.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_complete_transaction_preflight.py",
)
LOCAL_RUN_ROOT = four.LOCAL_RUN_ROOT
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-complete-transaction-runs"
)
_source_tree_sha256 = four._source_tree_sha256
_sha256 = four._sha256
_positive_integer = four._positive_integer
_non_negative_integer = four._non_negative_integer
validate_run_tag = four.validate_run_tag
build_ssh_command = four.build_ssh_command
_require_success = four._require_success
_atomic_write_json = four._atomic_write_json
_read_proc_status = four._read_proc_status
_memory_point = four._memory_point
_install_namespace_packages = four._install_namespace_packages
derive_tile_ranges = four.derive_tile_ranges
_tensor_bytes = four._tensor_bytes
_binding_destination_view = four._binding_destination_view
_read_tile = four._read_tile


def __getattr__(name):
    if name == "Qwen35CheckpointTile":
        return four.Qwen35CheckpointTile
    raise AttributeError(name)


def binding_contract(index, tp_size):
    if index not in range(320) or tp_size not in (1, 2):
        raise ValueError("complete checkpoint binding is invalid")
    phase = _BINDING_PHASE[index]
    if index == 0:
        return {
            "binding_index": 0, "phase_name": phase,
            "source_name": "model.language_model.embed_tokens.weight",
            "target": "embed_tokens.weight", "kind": "axis0",
            "dtype": "torch.bfloat16",
            "local_shape": [248320 // tp_size, 2048],
            "destination_slice": None,
            **_ROOT[tp_size]["embed_tokens"],
        }
    if index == 319:
        return {
            "binding_index": 319, "phase_name": phase,
            "source_name": "model.language_model.norm.weight",
            "target": "final_norm.weight", "kind": "replicated",
            "dtype": "torch.bfloat16", "local_shape": [2048],
            "destination_slice": None,
            **_ROOT[tp_size]["final_norm"],
        }
    layer = int(phase.split("_")[1])
    indices = dict(PHASE_BINDING_RUNS)[phase]
    relative = index - indices[0]
    if LAYER_SCHEDULE[layer] == "full_attention":
        template = four.binding_contract(227 + relative, tp_size)
    else:
        template = four.binding_contract(1 + relative, tp_size)
    result = dict(template)
    result["binding_index"] = index
    result["phase_name"] = phase
    result["source_name"] = result["source_name"].replace(
        f"layers.{template['layer_index']}.", f"layers.{layer}.", 1
    )
    result["target"] = result["target"].replace(
        f"layers.{template['layer_index']}.", f"layers.{layer}.", 1
    )
    result.pop("layer_index", None)
    return result


def _phase_names_for_bindings(bindings):
    result = []
    for binding in bindings:
        target = binding.load.weight.target
        if target == "embed_tokens.weight":
            result.append("embed_tokens")
        elif target == "final_norm.weight":
            result.append("final_norm")
        elif target.startswith("layers."):
            result.append(f"layer_{int(target.split('.')[1])}")
        else:
            raise ValueError("complete checkpoint phase target is invalid")
    return tuple(result)


def _alias_groups(tiles):
    groups = {}
    for tile in tiles:
        groups.setdefault(id(tile.destination), []).append(tile.binding_index)
    return sorted(
        [sorted(set(values)) for values in groups.values()
         if len(set(values)) > 1],
        key=lambda group: group[0],
    )


def _validate_alias_partitions(tiles, expected):
    grouped = {}
    for tile in tiles:
        grouped.setdefault(id(tile.destination), []).append(tile)
    observed = []
    for values in grouped.values():
        indices = sorted({tile.binding_index for tile in values})
        if len(indices) == 1:
            continue
        observed.append(indices)
        intervals = []
        for index in indices:
            slices = [tile.destination_slices[0] for tile in values
                      if tile.binding_index == index]
            intervals.append((min(item.start for item in slices),
                              max(item.stop for item in slices)))
        cursor = 0
        for start, stop in sorted(intervals):
            if start != cursor or stop <= start:
                raise ValueError("complete alias partition is invalid")
            cursor = stop
        if cursor != values[0].destination.shape[0]:
            raise ValueError("complete alias partition is invalid")
    if sorted(observed, key=lambda group: group[0]) != expected:
        raise ValueError("complete alias partition is invalid")


def apply_verify_and_rollback_complete_tiles(
    tiles, source_tensors, *, binding_order, binding_phase,
    expected_phase_order, expected_alias_groups, unique_tensors,
    selected_destination_ids,
):
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )
    tiles = tuple(tiles)
    source_tensors = tuple(source_tensors)
    observed_bindings = []
    for tile in tiles:
        if not observed_bindings or observed_bindings[-1] != tile.binding_index:
            observed_bindings.append(tile.binding_index)
    if tuple(observed_bindings) != tuple(binding_order):
        raise ValueError("complete binding order is invalid")
    phases = []
    for index in observed_bindings:
        phase = binding_phase[index]
        if not phases or phases[-1] != phase:
            phases.append(phase)
    if tuple(phases) != tuple(expected_phase_order):
        raise ValueError("complete phase order is invalid")
    _validate_alias_partitions(tiles, expected_alias_groups)
    destinations = {}
    first = {}
    phase_ids = {name: set() for name in expected_phase_order}
    for tile in tiles:
        object_id = id(tile.destination)
        destinations.setdefault(object_id, tile.destination)
        first.setdefault(object_id, tile.binding_index)
        phase_ids[binding_phase[tile.binding_index]].add(object_id)
    if set(destinations) != set(selected_destination_ids):
        raise ValueError("complete destination coverage is invalid")
    if any(int(t.count_nonzero().item()) for t in unique_tensors):
        raise ValueError("registered tensors must initially be zero")
    snapshots = {key: value.clone() for key, value in destinations.items()}
    transitions = []
    current = 0
    error = None
    try:
        for tile, source in zip(tiles, source_tensors, strict=True):
            next_index = expected_phase_order.index(
                binding_phase[tile.binding_index]
            )
            if next_index != current:
                if next_index != current + 1:
                    raise ValueError("complete phase order is invalid")
                completed = set().union(*(
                    phase_ids[name] for name in expected_phase_order[:next_index]
                ))
                future = set().union(*(
                    phase_ids[name] for name in expected_phase_order[next_index:]
                ))
                if not all(
                    not torch.equal(destinations[key], snapshots[key])
                    for key in completed
                ):
                    raise ValueError("completed phases are incomplete")
                if any(int(destinations[key].count_nonzero().item())
                       for key in future):
                    raise ValueError("future phase mutation")
                transitions.append({
                    "next_phase": expected_phase_order[next_index],
                    "completed_phases_changed": True,
                    "future_phases_zero": True,
                })
                current = next_index
            digest = hashlib.sha256(four._tensor_bytes(source)).hexdigest()
            _copy_qwen35_checkpoint_tile(tile, source)
            destination = tile.destination[tile.destination_slices]
            if hashlib.sha256(
                four._tensor_bytes(destination)
            ).hexdigest() != digest:
                raise ValueError("complete destination hash mismatch")
        if any(int(t.count_nonzero().item()) for t in unique_tensors
               if id(t) not in selected_destination_ids):
            raise ValueError("complete non-selected mutation")
    except Exception as caught:
        error = caught
    rollback = []
    for key, tensor in reversed(list(destinations.items())):
        with torch.no_grad():
            tensor.copy_(snapshots[key])
        rollback.append(first[key])
    if error:
        raise error
    return {
        "phase_completion_order": phases,
        "transition_checks": transitions,
        "rollback_binding_order": rollback,
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": all(
            not int(t.count_nonzero().item()) for t in unique_tensors
        ),
    }


def _validate_memory(row):
    memory = row["memory"]
    deltas = (
        row["total_vmhwm_increment_kib"],
        row["post_torch_vmhwm_increment_kib"],
        row["post_metadata_vmhwm_increment_kib"],
    )
    expected = (
        memory["after_payload"]["vmhwm_kib"] - memory["before"]["vmhwm_kib"],
        memory["after_payload"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
        memory["after_payload"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    if deltas != expected:
        raise ValueError("complete memory deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    if any(value > ceilings[name] for value, name in zip(
        deltas, ("total", "post_torch", "post_metadata"), strict=True
    )):
        raise ValueError("complete memory ceiling exceeded")


def validate_complete_checkpoint_row(row):
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if row.get("schema_version") != ROW_SCHEMA_VERSION or tp not in COMPLETE_TRANSACTION_CONTRACTS:
        raise ValueError("complete row schema or TP is invalid")
    contract = COMPLETE_TRANSACTION_CONTRACTS[tp]
    exact = {
        "status": "PASS", "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "layer_schedule": list(LAYER_SCHEDULE),
        "phase_names": list(PHASE_NAMES),
        "phase_binding_runs": [[name, list(indices)]
                               for name, indices in PHASE_BINDING_RUNS],
        "selected_binding_indices": list(range(320)),
        "selected_binding_count": 320, "unique_destination_count": 296,
        "alias_groups": ALIAS_GROUPS, "tile_count": contract["tile_count"],
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "open_count": 2, "pread_count": contract["pread_count"],
        "phase_completion_order": list(PHASE_NAMES),
        "transition_checks": [{
            "next_phase": name, "completed_phases_changed": True,
            "future_phases_zero": True,
        } for name in PHASE_NAMES[1:]],
        "selected_destinations_changed": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_binding_order": list(reversed(UNIQUE_BINDING_ORDER)),
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": True,
        "loader_call_count": 0, "assignment_call_count": 0,
        "target_take_count": 0, "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False, "cuda_initialized_after": False,
    }
    messages = {
        "selected_binding_count": "complete binding count invalid",
        "unique_destination_count": "complete destination count invalid",
        "alias_groups": "complete alias contract invalid",
        "phase_completion_order": "complete phase order invalid",
        "transition_checks": "complete transition checks invalid",
        "non_selected_tensors_remained_zero": "complete non-selected mutation",
        "rollback_binding_order": "complete rollback order invalid",
        "target_take_count": "complete target.take calls must be zero",
        "cuda_initialized_after": "complete CUDA must remain off",
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(messages.get(name, f"complete {name} invalid"))
    if len(row.get("binding_results", [])) != 320:
        raise ValueError("complete binding results invalid")
    for result, index in zip(row["binding_results"], range(320), strict=True):
        expected = binding_contract(index, row["tp_size"])
        for name, value in expected.items():
            if result.get(name) != value:
                raise ValueError(f"complete binding {index} invalid")
        hashes = [result.get(name) for name in (
            "production_sha256", "verifier_sha256",
            "source_tensor_sha256", "destination_sha256",
        )]
        for digest in hashes:
            _sha256(digest, "complete binding hash")
        if len(set(hashes)) != 1 or result.get("coverage_complete") is not True:
            raise ValueError("complete binding hash invalid")
    if len(row.get("phase_results", [])) != 26:
        raise ValueError("complete phase results invalid")
    for result, (name, indices) in zip(
        row["phase_results"], PHASE_BINDING_RUNS, strict=True
    ):
        expected = {"phase_name": name, "binding_indices": list(indices),
                    "binding_count": len(indices),
                    **contract["phases"][name], "coverage_complete": True}
        if any(result.get(key) != value for key, value in expected.items()):
            raise ValueError("complete phase result invalid")
        hashes = [result.get(key) for key in (
            "production_sha256", "verifier_sha256", "destination_sha256"
        )]
        if len(set(hashes)) != 1:
            raise ValueError("complete phase hash invalid")
    if row.get("aggregate_source_sha256") != row.get(
        "aggregate_destination_sha256"
    ):
        raise ValueError("complete aggregate hash invalid")
    _validate_memory(row)
    return row


def validate_complete_checkpoint_preflight(record):
    exact = {
        "schema_version": SCHEMA_VERSION, "status": "PASS",
        "remote_target": REMOTE_TARGET, "remote_python": REMOTE_PYTHON,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
    }
    if any(record.get(key) != value for key, value in exact.items()):
        raise ValueError("complete preflight identity invalid")
    hashes = record.get("source_file_sha256")
    if set(hashes or {}) != set(SOURCE_FILES):
        raise ValueError("complete source hashes invalid")
    if record.get("source_tree_sha256") != _source_tree_sha256(hashes):
        raise ValueError("complete source tree invalid")
    rows = record.get("rows")
    if [(r.get("tp_size"), r.get("tp_rank")) for r in rows or []] != list(base.TP_ROWS):
        raise ValueError("complete TP rows invalid")
    for row in rows:
        validate_complete_checkpoint_row(row)
    if len({row["process_id"] for row in rows}) != 3:
        raise ValueError("complete process IDs must be unique")
    return record


def _source_hashes(root):
    root = Path(root)
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
    }


def build_source_tar(source_root):
    root = Path(source_root)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in SOURCE_FILES:
            path = root / name
            if not path.is_file():
                raise ValueError(f"missing complete source: {name}")
            info = archive.gettarinfo(os.fspath(path), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with path.open("rb") as source:
                archive.addfile(info, source)
    return stream.getvalue()


def stage_source(source_root, run_tag, *, command_runner=subprocess.run):
    run_tag = validate_run_tag(run_tag)
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
    _require_success(staged, "complete source staging")
    local_hashes = _source_hashes(source_root)
    script = "\n".join([
        "import hashlib,json,pathlib",
        f"root=pathlib.Path({remote_source_dir!r})",
        f"names={list(SOURCE_FILES)!r}",
        "result={}",
        "for name in names:",
        " path=root/name",
        " if not path.is_file(): raise SystemExit('missing source: '+name)",
        " result[name]=hashlib.sha256(path.read_bytes()).hexdigest()",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
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
    _require_success(verified, "complete source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("complete remote source hashes do not match local")
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _stream_complete_checkpoint_transaction(
    shard_path,
    tiles,
    binding_plan,
    unique_tensors,
):
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    observed_bindings = []
    for tile in tiles:
        if not observed_bindings or observed_bindings[-1] != tile.binding_index:
            observed_bindings.append(tile.binding_index)
    if observed_bindings != list(range(320)):
        raise ValueError("complete tile binding order is invalid")
    observed_phases = []
    for index in observed_bindings:
        phase = _BINDING_PHASE[index]
        if not observed_phases or observed_phases[-1] != phase:
            observed_phases.append(phase)
    if observed_phases != list(PHASE_NAMES):
        raise ValueError("complete tile phase order is invalid")
    _validate_alias_partitions(tiles, ALIAS_GROUPS)
    destinations = {}
    first_binding = {}
    phase_ids = {name: set() for name in PHASE_NAMES}
    for tile in tiles:
        object_id = id(tile.destination)
        destinations.setdefault(object_id, tile.destination)
        first_binding.setdefault(object_id, tile.binding_index)
        phase_ids[_BINDING_PHASE[tile.binding_index]].add(object_id)
    if len(destinations) != 296:
        raise ValueError("complete unique destination count is invalid")
    if any(int(tensor.count_nonzero().item()) for tensor in unique_tensors):
        raise ValueError("registered tensors must initially be zero")
    snapshots = {
        object_id: tensor.clone()
        for object_id, tensor in destinations.items()
    }
    production_hashes = {
        index: hashlib.sha256() for index in range(320)
    }
    verifier_hashes = {
        index: hashlib.sha256() for index in range(320)
    }
    phase_production = {
        name: hashlib.sha256() for name in PHASE_NAMES
    }
    phase_verifier = {
        name: hashlib.sha256() for name in PHASE_NAMES
    }
    statistics = {
        index: {"tile_count": 0, "range_count": 0, "byte_count": 0}
        for index in range(320)
    }
    phase_statistics = {
        name: {"tile_count": 0, "range_count": 0, "byte_count": 0}
        for name in PHASE_NAMES
    }
    aggregate_source = hashlib.sha256()
    production_bytes = verifier_bytes = pread_count = 0
    transition_checks = []
    current_phase = 0
    production_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    verifier_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    error = None
    try:
        for tile in tiles:
            phase = _BINDING_PHASE[tile.binding_index]
            phase_index = PHASE_NAMES.index(phase)
            if phase_index != current_phase:
                if phase_index != current_phase + 1:
                    raise ValueError("complete phase order is invalid")
                completed = set().union(*(
                    phase_ids[name] for name in PHASE_NAMES[:phase_index]
                ))
                future = set().union(*(
                    phase_ids[name] for name in PHASE_NAMES[phase_index:]
                ))
                if not all(
                    not torch.equal(destinations[key], snapshots[key])
                    for key in completed
                ):
                    raise ValueError("complete phase coverage is incomplete")
                if any(int(destinations[key].count_nonzero().item())
                       for key in future):
                    raise ValueError("complete future phase mutation")
                transition_checks.append({
                    "next_phase": phase,
                    "completed_phases_changed": True,
                    "future_phases_zero": True,
                })
                current_phase = phase_index
            binding = binding_plan.bindings[tile.binding_index]
            ranges = derive_tile_ranges(
                binding, tile, data_start=DATA_START
            )
            production = _read_tile(production_descriptor, ranges)
            verifier = _read_tile(verifier_descriptor, ranges)
            pread_count += 2 * len(ranges)
            production_bytes += len(production)
            verifier_bytes += len(verifier)
            if production != verifier:
                raise ValueError("independent complete payload mismatch")
            tensor = torch.frombuffer(
                bytearray(production), dtype=tile.dtype
            ).clone().reshape(tile.tile_shape)
            source_sha = hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
            _copy_qwen35_checkpoint_tile(tile, tensor)
            destination = tile.destination[tile.destination_slices]
            if hashlib.sha256(
                _tensor_bytes(destination)
            ).hexdigest() != source_sha:
                raise ValueError("complete destination payload mismatch")
            production_hashes[tile.binding_index].update(production)
            verifier_hashes[tile.binding_index].update(verifier)
            phase_production[phase].update(production)
            phase_verifier[phase].update(verifier)
            aggregate_source.update(production)
            values = statistics[tile.binding_index]
            values["tile_count"] += 1
            values["range_count"] += len(ranges)
            values["byte_count"] += len(production)
            phase_values = phase_statistics[phase]
            phase_values["tile_count"] += 1
            phase_values["range_count"] += len(ranges)
            phase_values["byte_count"] += len(production)
            del tensor
    except Exception as caught:
        error = caught
    finally:
        os.close(production_descriptor)
        os.close(verifier_descriptor)
    if error is not None:
        for object_id, tensor in reversed(list(destinations.items())):
            with torch.no_grad():
                tensor.copy_(snapshots[object_id])
        raise error
    binding_results = []
    destination_aggregate = hashlib.sha256()
    phase_destination = {
        name: hashlib.sha256() for name in PHASE_NAMES
    }
    for index in range(320):
        contract = binding_contract(
            index, binding_plan.tensor_parallel_size
        )
        if statistics[index] != {
            "tile_count": contract["tile_count"],
            "range_count": contract["range_count"],
            "byte_count": contract["byte_count"],
        }:
            raise ValueError("complete binding coverage is incomplete")
        destination_bytes = _tensor_bytes(
            _binding_destination_view(binding_plan.bindings[index])
        )
        destination_sha = hashlib.sha256(destination_bytes).hexdigest()
        production_sha = production_hashes[index].hexdigest()
        verifier_sha = verifier_hashes[index].hexdigest()
        if len({production_sha, verifier_sha, destination_sha}) != 1:
            raise ValueError("complete binding hash mismatch")
        destination_aggregate.update(destination_bytes)
        phase_destination[_BINDING_PHASE[index]].update(destination_bytes)
        binding_results.append({
            **contract,
            "production_sha256": production_sha,
            "verifier_sha256": verifier_sha,
            "source_tensor_sha256": production_sha,
            "destination_sha256": destination_sha,
            "coverage_complete": True,
        })
    contract = COMPLETE_TRANSACTION_CONTRACTS[
        (binding_plan.tensor_parallel_size,
         binding_plan.tensor_parallel_rank)
    ]
    phase_results = []
    for name, indices in PHASE_BINDING_RUNS:
        if phase_statistics[name] != contract["phases"][name]:
            raise ValueError("complete phase coverage is incomplete")
        hashes = (
            phase_production[name].hexdigest(),
            phase_verifier[name].hexdigest(),
            phase_destination[name].hexdigest(),
        )
        if len(set(hashes)) != 1:
            raise ValueError("complete phase hash mismatch")
        phase_results.append({
            "phase_name": name,
            "binding_indices": list(indices),
            "binding_count": len(indices),
            **phase_statistics[name],
            "production_sha256": hashes[0],
            "verifier_sha256": hashes[1],
            "destination_sha256": hashes[2],
            "coverage_complete": True,
        })
    selected_ids = set(destinations)
    selected_changed = all(
        not torch.equal(tensor, snapshots[object_id])
        for object_id, tensor in destinations.items()
    )
    non_selected_zero = all(
        not int(tensor.count_nonzero().item())
        for tensor in unique_tensors
        if id(tensor) not in selected_ids
    )
    rollback = []
    for object_id, tensor in reversed(list(destinations.items())):
        with torch.no_grad():
            tensor.copy_(snapshots[object_id])
        rollback.append(first_binding[object_id])
    return {
        "binding_results": binding_results,
        "phase_results": phase_results,
        "phase_completion_order": observed_phases,
        "transition_checks": transition_checks,
        "aggregate_source_sha256": aggregate_source.hexdigest(),
        "aggregate_destination_sha256": destination_aggregate.hexdigest(),
        "production_payload_bytes_read": production_bytes,
        "verifier_payload_bytes_read": verifier_bytes,
        "open_count": 2,
        "pread_count": pread_count,
        "unique_destination_count": len(destinations),
        "alias_groups": _alias_groups(tiles),
        "selected_destinations_changed": selected_changed,
        "non_selected_tensors_remained_zero": non_selected_zero,
        "rollback_binding_order": rollback,
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": all(
            not int(tensor.count_nonzero().item())
            for tensor in unique_tensors
        ),
    }


def run_complete_checkpoint_rank_worker(
    *, checkpoint_dir, source_root, tensor_parallel_size,
    tensor_parallel_rank, observed_user, observed_hostname, process_id,
    status_reader=_read_proc_status,
):
    before = _memory_point(status_reader())
    _install_namespace_packages(source_root)
    import torch
    from torch import nn
    torch.set_num_threads(8)
    cuda_before = torch.cuda.is_initialized()
    after_torch = _memory_point(status_reader())
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
    tiles_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_tiles", fromlist=["*"]
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME, size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    metadata = metadata_module.read_qwen35_checkpoint_metadata(
        checkpoint_dir, shards=(shard,),
        expected_config_sha256=APPROVED_CONFIG_SHA256,
        expected_index_sha256=APPROVED_INDEX_SHA256,
        expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
    )
    tensor_plan = checkpoint_module.build_qwen35_checkpoint_tensor_plan(
        metadata.hf_config, metadata.index_payload, metadata.shard_headers
    )
    if tuple(metadata.hf_config.text_config.layer_types) != LAYER_SCHEDULE:
        raise ValueError("complete layer schedule is invalid")
    after_metadata = _memory_point(status_reader())
    layout = layout_module.build_qwen35_hybrid_state_layout(
        metadata.hf_config, tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16, speculative_tokens=1,
    )
    pool = hybrid_module.HybridStateTensorPool(
        layout, capacity=1, device="cpu"
    )
    after_pool = _memory_point(status_reader())
    attention_forward_count = 0

    class _Backend(nn.Module):
        def forward(self, *_args, **_kwargs):
            nonlocal attention_forward_count
            attention_forward_count += 1
            raise AssertionError("attention backend must not execute")

    target = factory_module.prepare_qwen35_checkpoint_candidate_target(
        metadata.hf_config, tensor_plan, pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        build_attention_backend=lambda *_args: _Backend(),
        parameter_device="cpu",
    )
    four.cpu.inspect_and_touch_cpu_target(target)
    after_target = _memory_point(status_reader())
    if _phase_names_for_bindings(target.binding_plan.bindings) != tuple(
        name for name, indices in PHASE_BINDING_RUNS for _ in indices
    ):
        raise ValueError("complete binding phase mapping is invalid")
    plan = tiles_module.build_qwen35_checkpoint_tile_plan(
        target.binding_plan, max_tile_bytes=65536
    )
    contract = COMPLETE_TRANSACTION_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    if len(plan.tiles) != contract["tile_count"]:
        raise ValueError("complete tile count mismatch")
    if dict(Counter(tile.kind for tile in plan.tiles)) != contract["kind_counts"]:
        raise ValueError("complete tile kind counts mismatch")
    model = target.assembly.packed.model
    unique = {}
    for _, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        unique.setdefault(id(tensor), tensor)
    transaction = _stream_complete_checkpoint_transaction(
        Path(checkpoint_dir) / APPROVED_SHARD_NAME,
        plan.tiles,
        target.binding_plan,
        tuple(unique.values()),
    )
    after_payload = _memory_point(status_reader())
    memory = {
        "before": before, "after_torch": after_torch,
        "after_metadata": after_metadata, "after_pool": after_pool,
        "after_target": after_target, "after_payload": after_payload,
    }
    row = {
        "schema_version": ROW_SCHEMA_VERSION, "status": "PASS",
        "tp_size": tensor_parallel_size, "tp_rank": tensor_parallel_rank,
        "process_id": process_id, "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "config_sha256": metadata.config_sha256,
        "index_sha256": metadata.index_sha256,
        "config_index_header_sha256": metadata.config_index_header_sha256,
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "layer_schedule": list(LAYER_SCHEDULE),
        "phase_names": list(PHASE_NAMES),
        "phase_binding_runs": [[name, list(indices)]
                               for name, indices in PHASE_BINDING_RUNS],
        "selected_binding_indices": list(range(320)),
        "selected_binding_count": 320,
        "tile_count": len(plan.tiles),
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "logical_payload_bytes_read": (
            transaction["production_payload_bytes_read"]
            + transaction["verifier_payload_bytes_read"]
        ),
        **transaction,
        "loader_call_count": 0, "assignment_call_count": 0,
        "target_take_count": 0, "model_forward_count": 0,
        "attention_forward_count": attention_forward_count,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0, after_payload["vmhwm_kib"] - before["vmhwm_kib"]
        ),
        "post_torch_vmhwm_increment_kib": max(
            0, after_payload["vmhwm_kib"] - after_torch["vmhwm_kib"]
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0, after_payload["vmhwm_kib"] - after_metadata["vmhwm_kib"]
        ),
    }
    validate_complete_checkpoint_row(row)
    return row


def _aggregate(rows, source_root):
    hashes = _source_hashes(source_root)
    record = {
        "schema_version": SCHEMA_VERSION, "status": "PASS",
        "remote_target": REMOTE_TARGET, "remote_python": REMOTE_PYTHON,
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "model_manifest_sha256": APPROVED_MODEL_MANIFEST_SHA256,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_complete_checkpoint_preflight(record)
    return record


def _source_manifest(run_tag, staged):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_tag": validate_run_tag(run_tag),
        "remote_target": REMOTE_TARGET,
        "remote_source_dir": staged["remote_source_dir"],
        "source_tree_sha256": staged["source_tree_sha256"],
        "local_file_sha256": dict(staged["local_file_sha256"]),
        "remote_file_sha256": dict(staged["remote_file_sha256"]),
    }


def run_remote_complete_checkpoint_preflight(
    source_root, run_tag, *, staged, local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    run_tag = validate_run_tag(run_tag)
    destination = Path(local_run_root) / run_tag
    if destination.exists():
        raise ValueError(
            f"local complete checkpoint directory exists: {destination}"
        )
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/"
        "complete_checkpoint_transaction_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/tools/"
        "qwen35_real_checkpoint_complete_transaction_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in base.TP_ROWS:
        completed = command_runner(build_ssh_command([
            "env", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8", REMOTE_PYTHON, "-B",
            worker, "internal-rank-worker",
            "--source-root", staged["remote_source_dir"],
            "--checkpoint-dir", APPROVED_MODEL_DIR,
            "--tp-size", str(tp_size),
            "--tp-rank", str(tp_rank),
        ]), text=True, capture_output=True)
        _require_success(completed, "complete rank worker")
        row = json.loads(completed.stdout)
        validate_complete_checkpoint_row(row)
        rows.append(row)
    finalized = command_runner(build_ssh_command([
        "env", "PYTHONDONTWRITEBYTECODE=1", REMOTE_PYTHON, "-B",
        worker, "internal-finalize",
        "--source-root", staged["remote_source_dir"],
        "--output", remote_artifact,
    ]), input=json.dumps({"rows": rows}), text=True, capture_output=True)
    _require_success(finalized, "complete finalizer")
    record = json.loads(finalized.stdout)
    validate_complete_checkpoint_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("complete source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        (
            "record=json.loads((root/"
            "'complete_checkpoint_transaction_preflight.json').read_text())"
        ),
        "temporary=root/'.source_manifest.json.tmp'",
        (
            "temporary.write_text(json.dumps("
            "payload['source_manifest'],sort_keys=True,"
            "separators=(',',':'))+'\\n')"
        ),
        "temporary.replace(root/'source_manifest.json')",
        (
            "result={'complete_checkpoint_transaction_preflight':record,"
            "'source_manifest':json.loads("
            "(root/'source_manifest.json').read_text())}"
        ),
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([
            "env", "PYTHONDONTWRITEBYTECODE=1", REMOTE_PYTHON, "-B",
            "-c", script,
        ]),
        input=json.dumps({
            "complete_checkpoint_transaction_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "complete artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "complete_checkpoint_transaction_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("complete artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.", suffix=".tmp", dir=destination.parent
    ))
    try:
        _atomic_write_json(
            temporary / "complete_checkpoint_transaction_preflight.json",
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


def execute_remote_complete_checkpoint_preflight(
    source_root,
    run_tag,
    *,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source(
        source_root,
        run_tag,
        command_runner=command_runner,
    )
    return run_remote_complete_checkpoint_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_complete_checkpoint_rank_worker(
        checkpoint_dir=arguments.checkpoint_dir,
        source_root=arguments.source_root,
        tensor_parallel_size=arguments.tp_size,
        tensor_parallel_rank=arguments.tp_rank,
        observed_user=getpass.getuser(),
        observed_hostname=socket.gethostname(),
        process_id=os.getpid(),
    )
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0


def _finalize_main(arguments) -> int:
    output = Path(arguments.output)
    if output.exists():
        raise ValueError("complete preflight output already exists")
    payload = json.load(sys.stdin)
    record = _aggregate(payload.get("rows"), arguments.source_root)
    _atomic_write_json(output, record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-tag", required=True)
    run_parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    worker_parser = subparsers.add_parser("internal-rank-worker")
    worker_parser.add_argument("--source-root", required=True)
    worker_parser.add_argument("--checkpoint-dir", required=True)
    worker_parser.add_argument("--tp-size", required=True, type=int)
    worker_parser.add_argument("--tp-rank", required=True, type=int)
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
        validate_complete_checkpoint_preflight(record)
    else:
        record = execute_remote_complete_checkpoint_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
