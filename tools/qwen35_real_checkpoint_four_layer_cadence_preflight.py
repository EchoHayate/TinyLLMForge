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


def _load_sibling(name: str, filename: str):
    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


two_layer = _load_sibling(
    "_qwen35_heterogeneous_two_layer_preflight_base",
    "qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py",
)
layer0 = two_layer.layer0
base = two_layer.base
cpu = two_layer.cpu

SCHEMA_VERSION = "qwen35.real-checkpoint-four-layer-cadence.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-four-layer-cadence-rank.v1"
REMOTE_TARGET = two_layer.REMOTE_TARGET
REMOTE_PYTHON = two_layer.REMOTE_PYTHON
APPROVED_MODEL_DIR = two_layer.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = two_layer.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = two_layer.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = two_layer.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = two_layer.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = two_layer.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = two_layer.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = two_layer.APPROVED_COMPOSITE_SHA256
DATA_START = two_layer.DATA_START

SELECTED_LAYER_INDICES = (0, 1, 2, 3)
SELECTED_LAYER_TYPES = (
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
)
LAYER_BINDING_INDICES = {
    0: tuple(range(1, 15)),
    1: tuple(range(15, 29)),
    2: tuple(range(160, 174)),
    3: tuple(range(227, 238)),
}
SELECTED_BINDING_INDICES = (
    *LAYER_BINDING_INDICES[0],
    *LAYER_BINDING_INDICES[1],
    *LAYER_BINDING_INDICES[2],
    *LAYER_BINDING_INDICES[3],
)
ALIAS_GROUPS = [[12, 13], [26, 27], [171, 172], [229, 230]]
UNIQUE_BINDING_ORDER = (
    *range(1, 13),
    14,
    *range(15, 27),
    28,
    *range(160, 172),
    173,
    227,
    228,
    229,
    *range(231, 238),
)

_LINEAR_TP1 = {
    "tile_count": 1826,
    "range_count": 1826,
    "byte_count": 117629536,
}
_LINEAR_TP2 = {
    "tile_count": 917,
    "range_count": 4744,
    "byte_count": 58819120,
}
_FULL_TP1 = {
    "tile_count": 1630,
    "range_count": 1630,
    "byte_count": 104866816,
}
_FULL_TP2 = {
    "tile_count": 817,
    "range_count": 4644,
    "byte_count": 52438016,
}


def _four_layer_contract(tp_size: int) -> dict:
    linear = _LINEAR_TP1 if tp_size == 1 else _LINEAR_TP2
    full = _FULL_TP1 if tp_size == 1 else _FULL_TP2
    if tp_size == 1:
        aggregate = {
            "tile_count": 7108,
            "kind_counts": {
                "axis0": 3788,
                "axis1": 2152,
                "replicated": 13,
                "segmented_axis0": 1152,
                "squeeze_axis0": 3,
            },
            "bytes_per_pass": 457755424,
            "ranges_per_pass": 7108,
            "pread_count": 14216,
            "logical_bytes": 915510848,
        }
    else:
        aggregate = {
            "tile_count": 3568,
            "kind_counts": {
                "axis0": 1900,
                "axis1": 1076,
                "replicated": 13,
                "segmented_axis0": 576,
                "squeeze_axis0": 3,
            },
            "bytes_per_pass": 228895376,
            "ranges_per_pass": 18876,
            "pread_count": 37752,
            "logical_bytes": 457790752,
        }
    aggregate["layers"] = {
        0: dict(linear),
        1: dict(linear),
        2: dict(linear),
        3: dict(full),
    }
    return aggregate


FOUR_LAYER_CONTRACTS = {
    (1, 0): _four_layer_contract(1),
    (2, 0): _four_layer_contract(2),
    (2, 1): _four_layer_contract(2),
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 5505024,
        "post_torch": 4980736,
        "post_metadata": 4718592,
    },
    2: {
        "total": 3145728,
        "post_torch": 2883584,
        "post_metadata": 2621440,
    },
}

SOURCE_FILES = (
    *two_layer.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py",
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-four-layer-cadence-runs"
)
LOCAL_RUN_ROOT = two_layer.LOCAL_RUN_ROOT

_sha256 = two_layer._sha256
_positive_integer = two_layer._positive_integer
_non_negative_integer = two_layer._non_negative_integer
_source_tree_sha256 = two_layer._source_tree_sha256
validate_run_tag = two_layer.validate_run_tag
build_ssh_command = two_layer.build_ssh_command
_require_success = two_layer._require_success
_read_proc_status = two_layer._read_proc_status
_memory_point = two_layer._memory_point
_install_namespace_packages = two_layer._install_namespace_packages
_atomic_write_json = two_layer._atomic_write_json
derive_tile_ranges = two_layer.derive_tile_ranges
_tensor_bytes = two_layer._tensor_bytes
_binding_destination_view = two_layer._binding_destination_view
_read_tile = two_layer._read_tile


def __getattr__(name: str):
    if name == "Qwen35CheckpointTile":
        return layer0.Qwen35CheckpointTile
    raise AttributeError(name)


def _linear_layer_base_index(layer_index: int) -> int:
    if layer_index == 0:
        return 1
    if layer_index == 1:
        return 15
    if layer_index == 2:
        return 160
    raise ValueError("four-layer linear layer is invalid")


def _binding_layer(index: int) -> int:
    for layer_index, indices in LAYER_BINDING_INDICES.items():
        if index in indices:
            return layer_index
    raise ValueError("four-layer binding is invalid")


def binding_contract(index: int, tp_size: int) -> dict:
    layer_index = _binding_layer(index)
    if tp_size not in (1, 2):
        raise ValueError("four-layer tensor parallel size is invalid")
    if layer_index == 3:
        return two_layer.binding_contract(index, tp_size)
    relative_index = index - _linear_layer_base_index(layer_index) + 1
    source = layer0.binding_contract(relative_index, tp_size)
    result = {
        name: source[name]
        for name in (
            "binding_index",
            "source_name",
            "target",
            "kind",
            "dtype",
            "local_shape",
            "destination_slice",
            "tile_count",
            "range_count",
            "byte_count",
        )
    }
    result["binding_index"] = index
    result["layer_index"] = layer_index
    result["source_name"] = result["source_name"].replace(
        "layers.0.", f"layers.{layer_index}.", 1
    )
    result["target"] = result["target"].replace(
        "layers.0.", f"layers.{layer_index}.", 1
    )
    return result


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing four-layer source: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_source_tar(source_root) -> bytes:
    root = Path(source_root)
    if base.discover_production_source_files(
        root
    ) != cpu.PRODUCTION_SOURCE_FILES:
        raise ValueError("production source closure does not match frozen set")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for relative in SOURCE_FILES:
            path = root / relative
            if not path.is_file():
                raise ValueError(f"missing four-layer source: {relative}")
            info = archive.gettarinfo(os.fspath(path), arcname=relative)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
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
    _require_success(staged, "four-layer source staging")
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
    _require_success(verified, "four-layer source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("four-layer remote source hashes do not match local")
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _alias_groups(tiles):
    groups = {}
    for tile in tiles:
        groups.setdefault(id(tile.destination), []).append(
            tile.binding_index
        )
    return sorted(
        (
            sorted(set(indices))
            for indices in groups.values()
            if len(set(indices)) > 1
        ),
        key=lambda group: group[0],
    )


def _validate_alias_partitions(tiles) -> None:
    groups = {}
    for tile in tiles:
        groups.setdefault(id(tile.destination), []).append(tile)
    observed = []
    for grouped_tiles in groups.values():
        binding_indices = sorted({
            tile.binding_index
            for tile in grouped_tiles
        })
        if len(binding_indices) == 1:
            continue
        observed.append(binding_indices)
        intervals = []
        for binding_index in binding_indices:
            starts = []
            stops = []
            for tile in grouped_tiles:
                if tile.binding_index != binding_index:
                    continue
                destination_slice = tile.destination_slices[0]
                starts.append(destination_slice.start)
                stops.append(destination_slice.stop)
            intervals.append((min(starts), max(stops)))
        intervals.sort()
        expected_start = 0
        for start, stop in intervals:
            if start != expected_start or stop <= start:
                raise ValueError("four-layer alias partition is invalid")
            expected_start = stop
        if expected_start != grouped_tiles[0].destination.shape[0]:
            raise ValueError("four-layer alias partition is invalid")
    if sorted(observed, key=lambda group: group[0]) != ALIAS_GROUPS:
        raise ValueError("four-layer alias partition is invalid")


def _ordered_bindings(tiles) -> list[int]:
    result = []
    for tile in tiles:
        if not result or result[-1] != tile.binding_index:
            result.append(tile.binding_index)
    return result


def _ordered_layers(binding_order, binding_layer) -> list[int]:
    result = []
    for index in binding_order:
        layer_index = binding_layer[index]
        if not result or result[-1] != layer_index:
            result.append(layer_index)
    return result


def _destination_state(tiles):
    destination_objects = {}
    first_binding = {}
    layer_ids = {layer_index: set() for layer_index in SELECTED_LAYER_INDICES}
    for tile in tiles:
        object_id = id(tile.destination)
        destination_objects.setdefault(object_id, tile.destination)
        first_binding.setdefault(object_id, tile.binding_index)
        layer_ids[_binding_layer(tile.binding_index)].add(object_id)
    return destination_objects, first_binding, layer_ids


def _transition_check(
    next_layer,
    destination_objects,
    snapshots,
    layer_ids,
):
    completed_ids = set().union(*(
        layer_ids[layer_index]
        for layer_index in range(next_layer)
    ))
    future_ids = set().union(*(
        layer_ids[layer_index]
        for layer_index in range(next_layer, 4)
    ))
    completed_changed = all(
        not destination_objects[object_id].equal(snapshots[object_id])
        for object_id in completed_ids
    )
    future_zero = all(
        int(destination_objects[object_id].count_nonzero().item()) == 0
        for object_id in future_ids
    )
    if not completed_changed:
        raise ValueError(
            f"completed layers are incomplete before layer {next_layer}"
        )
    if not future_zero:
        raise ValueError(f"future layer changed before layer {next_layer}")
    return {
        "next_layer": next_layer,
        "completed_layers_changed": True,
        "future_layers_zero": True,
    }


def _restore_destinations(destination_objects, first_binding, snapshots):
    import torch

    rollback_order = []
    restored = True
    for object_id, tensor in reversed(list(destination_objects.items())):
        with torch.no_grad():
            tensor.copy_(snapshots[object_id])
        rollback_order.append(first_binding[object_id])
        restored = restored and torch.equal(tensor, snapshots[object_id])
    if not restored:
        raise RuntimeError("four-layer rollback failed")
    return rollback_order


def apply_verify_and_rollback_four_layer_tiles(
    tiles,
    source_tensors,
    *,
    binding_order,
    binding_layer,
    unique_tensors,
    selected_destination_ids,
):
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    tiles = tuple(tiles)
    source_tensors = tuple(source_tensors)
    if len(tiles) != len(source_tensors) or not tiles:
        raise ValueError("four-layer tile payloads are invalid")
    observed_order = _ordered_bindings(tiles)
    if tuple(observed_order) != tuple(binding_order):
        raise ValueError("four-layer binding order is invalid")
    observed_layers = _ordered_layers(observed_order, binding_layer)
    if observed_layers != list(SELECTED_LAYER_INDICES):
        raise ValueError("four-layer layer order is invalid")
    aliases = _alias_groups(tiles)
    if aliases != ALIAS_GROUPS:
        raise ValueError("four-layer alias contract is invalid")
    _validate_alias_partitions(tiles)
    if any(int(tensor.count_nonzero().item()) != 0 for tensor in unique_tensors):
        raise ValueError("registered tensors must initially be zero")
    destination_objects = {}
    first_binding = {}
    layer_ids = {layer_index: set() for layer_index in SELECTED_LAYER_INDICES}
    for tile in tiles:
        object_id = id(tile.destination)
        destination_objects.setdefault(object_id, tile.destination)
        first_binding.setdefault(object_id, tile.binding_index)
        layer_ids[binding_layer[tile.binding_index]].add(object_id)
    if set(destination_objects) != set(selected_destination_ids):
        raise ValueError("four-layer destination coverage is invalid")
    snapshots = {
        object_id: tensor.clone()
        for object_id, tensor in destination_objects.items()
    }
    transition_checks = []
    current_layer = 0
    error = None
    try:
        for tile, source in zip(tiles, source_tensors, strict=True):
            layer_index = binding_layer[tile.binding_index]
            if layer_index != current_layer:
                if layer_index != current_layer + 1:
                    raise ValueError("four-layer layer order is invalid")
                transition_checks.append(_transition_check(
                    layer_index,
                    destination_objects,
                    snapshots,
                    layer_ids,
                ))
                current_layer = layer_index
            source_sha = hashlib.sha256(_tensor_bytes(source)).hexdigest()
            _copy_qwen35_checkpoint_tile(tile, source)
            destination = tile.destination[tile.destination_slices]
            if hashlib.sha256(
                _tensor_bytes(destination)
            ).hexdigest() != source_sha:
                raise ValueError("four-layer destination payload hash mismatch")
        if len(transition_checks) != 3:
            raise ValueError("four-layer transition coverage is incomplete")
        non_selected_zero = all(
            int(tensor.count_nonzero().item()) == 0
            for tensor in unique_tensors
            if id(tensor) not in selected_destination_ids
        )
        if not non_selected_zero:
            raise ValueError("four-layer non-selected tensor mutation")
    except Exception as caught:
        error = caught
    rollback_order = _restore_destinations(
        destination_objects,
        first_binding,
        snapshots,
    )
    all_zero = all(
        int(tensor.count_nonzero().item()) == 0
        for tensor in unique_tensors
    )
    if error is not None:
        raise error
    return {
        "unique_destination_count": len(destination_objects),
        "alias_groups": aliases,
        "layer_completion_order": observed_layers,
        "transition_checks": transition_checks,
        "non_selected_tensors_remained_zero": True,
        "rollback_binding_order": rollback_order,
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def _validate_memory(row) -> None:
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_payload",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError("four-layer memory points are invalid")
    for name in names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(f"four-layer memory {name} is invalid")
        _positive_integer(point.get("vmrss_kib"), f"{name} vmrss")
        _positive_integer(point.get("vmhwm_kib"), f"{name} vmhwm")
    deltas = (
        _non_negative_integer(row.get("total_vmhwm_increment_kib"), "total"),
        _non_negative_integer(
            row.get("post_torch_vmhwm_increment_kib"), "post-Torch"
        ),
        _non_negative_integer(
            row.get("post_metadata_vmhwm_increment_kib"), "post-metadata"
        ),
    )
    expected = (
        max(
            0,
            memory["after_payload"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"],
        ),
        max(
            0,
            memory["after_payload"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"],
        ),
        max(
            0,
            memory["after_payload"]["vmhwm_kib"]
            - memory["after_metadata"]["vmhwm_kib"],
        ),
    )
    if deltas != expected:
        raise ValueError("four-layer VmHWM deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    for value, name in zip(
        deltas,
        ("total", "post_torch", "post_metadata"),
        strict=True,
    ):
        if value > ceilings[name]:
            raise ValueError(f"four-layer {name} VmHWM exceeds ceiling")


def validate_four_layer_cadence_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("four-layer row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("four-layer row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("four-layer row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in FOUR_LAYER_CONTRACTS:
        raise ValueError("four-layer TP context is invalid")
    _positive_integer(row.get("process_id"), "process_id")
    contract = FOUR_LAYER_CONTRACTS[tp]
    exact = {
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_layer_indices": list(SELECTED_LAYER_INDICES),
        "selected_layer_types": list(SELECTED_LAYER_TYPES),
        "selected_binding_indices": list(SELECTED_BINDING_INDICES),
        "selected_binding_count": 53,
        "unique_destination_count": 49,
        "alias_groups": ALIAS_GROUPS,
        "tile_count": contract["tile_count"],
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "open_count": 2,
        "pread_count": contract["pread_count"],
        "layer_completion_order": list(SELECTED_LAYER_INDICES),
        "transition_checks": [
            {
                "next_layer": layer_index,
                "completed_layers_changed": True,
                "future_layers_zero": True,
            }
            for layer_index in (1, 2, 3)
        ],
        "selected_destinations_changed": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_binding_order": list(reversed(UNIQUE_BINDING_ORDER)),
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": True,
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "target_take_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    messages = {
        "selected_layer_types": "four-layer layer types invalid",
        "selected_binding_indices": "four-layer binding contract invalid",
        "selected_binding_count": "four-layer binding count invalid",
        "unique_destination_count": "four-layer destination count invalid",
        "alias_groups": "four-layer alias contract invalid",
        "layer_completion_order": "four-layer layer order invalid",
        "transition_checks": "four-layer transition checks invalid",
        "non_selected_tensors_remained_zero": (
            "four-layer non-selected mutation"
        ),
        "rollback_binding_order": "four-layer rollback order invalid",
        "target_take_count": "four-layer target.take calls must be zero",
        "cuda_initialized_after": "four-layer CUDA must remain off",
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(messages.get(
                name,
                f"four-layer row {name} is invalid",
            ))
    results = row.get("binding_results")
    if not isinstance(results, list) or len(results) != 53:
        raise ValueError("four-layer binding results are invalid")
    for result, index in zip(results, SELECTED_BINDING_INDICES, strict=True):
        expected = binding_contract(index, row["tp_size"])
        for name in (
            "binding_index",
            "layer_index",
            "source_name",
            "target",
            "kind",
            "dtype",
            "local_shape",
            "destination_slice",
            "tile_count",
            "range_count",
            "byte_count",
        ):
            if result.get(name) != expected[name]:
                raise ValueError(
                    f"four-layer binding {index} {name} is invalid"
                )
        if result.get("coverage_complete") is not True:
            raise ValueError("four-layer binding coverage is incomplete")
        hashes = [
            _sha256(result.get(name), f"four-layer binding {index} {name}")
            for name in (
                "production_sha256",
                "verifier_sha256",
                "source_tensor_sha256",
                "destination_sha256",
            )
        ]
        if len(set(hashes)) != 1:
            raise ValueError("four-layer binding hash mismatch")
    layer_results = row.get("layer_results")
    if not isinstance(layer_results, list) or len(layer_results) != 4:
        raise ValueError("four-layer layer results are invalid")
    for result, layer_index, layer_type in zip(
        layer_results,
        SELECTED_LAYER_INDICES,
        SELECTED_LAYER_TYPES,
        strict=True,
    ):
        layer_contract = contract["layers"][layer_index]
        binding_indices = LAYER_BINDING_INDICES[layer_index]
        expected = {
            "layer_index": layer_index,
            "layer_type": layer_type,
            "binding_indices": list(binding_indices),
            "binding_count": len(binding_indices),
            "tile_count": layer_contract["tile_count"],
            "range_count": layer_contract["range_count"],
            "byte_count": layer_contract["byte_count"],
            "coverage_complete": True,
        }
        for name, value in expected.items():
            if result.get(name) != value:
                raise ValueError(
                    f"four-layer layer {layer_index} {name} is invalid"
                )
        hashes = [
            _sha256(
                result.get(name),
                f"four-layer layer {layer_index} {name}",
            )
            for name in (
                "production_sha256",
                "verifier_sha256",
                "destination_sha256",
            )
        ]
        if len(set(hashes)) != 1:
            raise ValueError("four-layer layer hash mismatch")
    aggregates = (
        _sha256(row.get("aggregate_source_sha256"), "aggregate source"),
        _sha256(
            row.get("aggregate_destination_sha256"),
            "aggregate destination",
        ),
    )
    if len(set(aggregates)) != 1:
        raise ValueError("four-layer aggregate hash mismatch")
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("four-layer hostname is invalid")
    _validate_memory(row)
    return row


def validate_four_layer_cadence_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("four-layer preflight must be a mapping")
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
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
    }
    for name, value in exact.items():
        if record.get(name) != value:
            raise ValueError(f"four-layer preflight {name} is invalid")
    hashes = record.get("source_file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SOURCE_FILES):
        raise ValueError("four-layer source hashes are invalid")
    for name, digest in hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError("four-layer source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(base.TP_ROWS):
        raise ValueError("four-layer TP rows are invalid")
    for row in rows:
        validate_four_layer_cadence_row(row)
    pids = [row["process_id"] for row in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("four-layer process IDs must be unique")
    return record


def _selected_binding_indices(bindings):
    prefixes = tuple(
        f"layers.{layer_index}."
        for layer_index in SELECTED_LAYER_INDICES
    )
    return tuple(
        index
        for index, binding in enumerate(bindings)
        if binding.load.weight.target.startswith(prefixes)
    )


def _selected_tiles(binding_plan):
    from tinyvllm.models.qwen35_checkpoint_tiles import (
        build_qwen35_checkpoint_tile_plan,
    )

    indices = _selected_binding_indices(binding_plan.bindings)
    if indices != SELECTED_BINDING_INDICES:
        raise ValueError("four-layer binding indices are invalid")
    plan = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=65536,
    )
    return tuple(
        tile
        for tile in plan.tiles
        if tile.binding_index in SELECTED_BINDING_INDICES
    )


def _stream_four_layer_transaction(
    shard_path,
    tiles,
    binding_plan,
    unique_tensors,
):
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    observed_binding_order = _ordered_bindings(tiles)
    if tuple(observed_binding_order) != SELECTED_BINDING_INDICES:
        raise ValueError("four-layer tile order is invalid")
    binding_layer = {
        index: _binding_layer(index)
        for index in SELECTED_BINDING_INDICES
    }
    observed_layers = _ordered_layers(observed_binding_order, binding_layer)
    if observed_layers != list(SELECTED_LAYER_INDICES):
        raise ValueError("four-layer layer order is invalid")
    aliases = _alias_groups(tiles)
    if aliases != ALIAS_GROUPS:
        raise ValueError("four-layer alias contract is invalid")
    _validate_alias_partitions(tiles)
    destination_objects, first_binding, layer_ids = _destination_state(tiles)
    if len(destination_objects) != 49:
        raise ValueError("four-layer unique destination count is invalid")
    if any(int(tensor.count_nonzero().item()) != 0 for tensor in unique_tensors):
        raise ValueError("registered tensors must initially be zero")
    snapshots = {
        object_id: tensor.clone()
        for object_id, tensor in destination_objects.items()
    }
    production_hashes = {
        index: hashlib.sha256()
        for index in SELECTED_BINDING_INDICES
    }
    verifier_hashes = {
        index: hashlib.sha256()
        for index in SELECTED_BINDING_INDICES
    }
    layer_production_hashes = {
        layer_index: hashlib.sha256()
        for layer_index in SELECTED_LAYER_INDICES
    }
    layer_verifier_hashes = {
        layer_index: hashlib.sha256()
        for layer_index in SELECTED_LAYER_INDICES
    }
    statistics = {
        index: {"tile_count": 0, "range_count": 0, "byte_count": 0}
        for index in SELECTED_BINDING_INDICES
    }
    layer_statistics = {
        layer_index: {
            "tile_count": 0,
            "range_count": 0,
            "byte_count": 0,
        }
        for layer_index in SELECTED_LAYER_INDICES
    }
    aggregate_source = hashlib.sha256()
    production_bytes = 0
    verifier_bytes = 0
    pread_count = 0
    transition_checks = []
    current_layer = 0
    production_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    verifier_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    error = None
    try:
        for tile in tiles:
            layer_index = binding_layer[tile.binding_index]
            if layer_index != current_layer:
                if layer_index != current_layer + 1:
                    raise ValueError("four-layer layer order is invalid")
                transition_checks.append(_transition_check(
                    layer_index,
                    destination_objects,
                    snapshots,
                    layer_ids,
                ))
                current_layer = layer_index
            binding = binding_plan.bindings[tile.binding_index]
            ranges = derive_tile_ranges(
                binding,
                tile,
                data_start=DATA_START,
            )
            production = _read_tile(production_descriptor, ranges)
            verifier = _read_tile(verifier_descriptor, ranges)
            pread_count += len(ranges) * 2
            production_bytes += len(production)
            verifier_bytes += len(verifier)
            if production != verifier:
                raise ValueError("independent four-layer payload mismatch")
            tensor = torch.frombuffer(
                bytearray(production),
                dtype=tile.dtype,
            ).clone().reshape(tile.tile_shape)
            source_sha = hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
            _copy_qwen35_checkpoint_tile(tile, tensor)
            destination = tile.destination[tile.destination_slices]
            if hashlib.sha256(
                _tensor_bytes(destination)
            ).hexdigest() != source_sha:
                raise ValueError("four-layer destination payload mismatch")
            production_hashes[tile.binding_index].update(production)
            verifier_hashes[tile.binding_index].update(verifier)
            layer_production_hashes[layer_index].update(production)
            layer_verifier_hashes[layer_index].update(verifier)
            aggregate_source.update(production)
            values = statistics[tile.binding_index]
            values["tile_count"] += 1
            values["range_count"] += len(ranges)
            values["byte_count"] += len(production)
            layer_values = layer_statistics[layer_index]
            layer_values["tile_count"] += 1
            layer_values["range_count"] += len(ranges)
            layer_values["byte_count"] += len(production)
            del tensor
    except Exception as caught:
        error = caught
    finally:
        os.close(production_descriptor)
        os.close(verifier_descriptor)
    if error is not None:
        _restore_destinations(destination_objects, first_binding, snapshots)
        raise error
    binding_results = []
    destination_aggregate = hashlib.sha256()
    layer_destination_hashes = {
        layer_index: hashlib.sha256()
        for layer_index in SELECTED_LAYER_INDICES
    }
    for index in SELECTED_BINDING_INDICES:
        binding = binding_plan.bindings[index]
        contract = binding_contract(
            index,
            binding_plan.tensor_parallel_size,
        )
        values = statistics[index]
        if values != {
            "tile_count": contract["tile_count"],
            "range_count": contract["range_count"],
            "byte_count": contract["byte_count"],
        }:
            _restore_destinations(
                destination_objects,
                first_binding,
                snapshots,
            )
            raise ValueError("four-layer binding coverage is incomplete")
        destination_bytes = _tensor_bytes(
            _binding_destination_view(binding)
        )
        destination_sha = hashlib.sha256(destination_bytes).hexdigest()
        production_sha = production_hashes[index].hexdigest()
        verifier_sha = verifier_hashes[index].hexdigest()
        if len({production_sha, verifier_sha, destination_sha}) != 1:
            _restore_destinations(
                destination_objects,
                first_binding,
                snapshots,
            )
            raise ValueError("four-layer binding hash mismatch")
        destination_aggregate.update(destination_bytes)
        layer_destination_hashes[_binding_layer(index)].update(
            destination_bytes
        )
        binding_results.append({
            **contract,
            "production_sha256": production_sha,
            "verifier_sha256": verifier_sha,
            "source_tensor_sha256": production_sha,
            "destination_sha256": destination_sha,
            "coverage_complete": True,
        })
    contract = FOUR_LAYER_CONTRACTS[
        (
            binding_plan.tensor_parallel_size,
            binding_plan.tensor_parallel_rank,
        )
    ]
    layer_results = []
    for layer_index, layer_type in zip(
        SELECTED_LAYER_INDICES,
        SELECTED_LAYER_TYPES,
        strict=True,
    ):
        if layer_statistics[layer_index] != contract["layers"][layer_index]:
            _restore_destinations(
                destination_objects,
                first_binding,
                snapshots,
            )
            raise ValueError("four-layer layer coverage is incomplete")
        hashes = (
            layer_production_hashes[layer_index].hexdigest(),
            layer_verifier_hashes[layer_index].hexdigest(),
            layer_destination_hashes[layer_index].hexdigest(),
        )
        if len(set(hashes)) != 1:
            _restore_destinations(
                destination_objects,
                first_binding,
                snapshots,
            )
            raise ValueError("four-layer layer hash mismatch")
        layer_results.append({
            "layer_index": layer_index,
            "layer_type": layer_type,
            "binding_indices": list(LAYER_BINDING_INDICES[layer_index]),
            "binding_count": len(LAYER_BINDING_INDICES[layer_index]),
            **layer_statistics[layer_index],
            "production_sha256": hashes[0],
            "verifier_sha256": hashes[1],
            "destination_sha256": hashes[2],
            "coverage_complete": True,
        })
    selected_ids = set(destination_objects)
    selected_changed = all(
        not torch.equal(tensor, snapshots[object_id])
        for object_id, tensor in destination_objects.items()
    )
    non_selected_zero = all(
        int(tensor.count_nonzero().item()) == 0
        for tensor in unique_tensors
        if id(tensor) not in selected_ids
    )
    rollback_binding_order = _restore_destinations(
        destination_objects,
        first_binding,
        snapshots,
    )
    all_zero = all(
        int(tensor.count_nonzero().item()) == 0
        for tensor in unique_tensors
    )
    return {
        "binding_results": binding_results,
        "layer_results": layer_results,
        "layer_completion_order": observed_layers,
        "transition_checks": transition_checks,
        "aggregate_source_sha256": aggregate_source.hexdigest(),
        "aggregate_destination_sha256": destination_aggregate.hexdigest(),
        "production_payload_bytes_read": production_bytes,
        "verifier_payload_bytes_read": verifier_bytes,
        "open_count": 2,
        "pread_count": pread_count,
        "unique_destination_count": len(destination_objects),
        "alias_groups": aliases,
        "selected_destinations_changed": selected_changed,
        "non_selected_tensors_remained_zero": non_selected_zero,
        "rollback_binding_order": rollback_binding_order,
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def run_four_layer_cadence_rank_worker(
    *,
    checkpoint_dir,
    source_root,
    tensor_parallel_size,
    tensor_parallel_rank,
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
    metadata_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_metadata",
        fromlist=["*"],
    )
    checkpoint_module = __import__(
        "tinyvllm.models.qwen35_checkpoint",
        fromlist=["*"],
    )
    hybrid_state_module = __import__(
        "tinyvllm.engine.hybrid_state",
        fromlist=["*"],
    )
    layout_module = __import__(
        "tinyvllm.engine.qwen35_hybrid_state",
        fromlist=["*"],
    )
    factory_module = __import__(
        "tinyvllm.models.qwen35_checkpoint_candidate_factory",
        fromlist=["*"],
    )
    shard = metadata_module.Qwen35CheckpointShardIdentity(
        name=APPROVED_SHARD_NAME,
        size=APPROVED_SHARD_SIZE,
        sha256=APPROVED_SHARD_SHA256,
    )
    metadata = metadata_module.read_qwen35_checkpoint_metadata(
        checkpoint_dir,
        shards=(shard,),
        expected_config_sha256=APPROVED_CONFIG_SHA256,
        expected_index_sha256=APPROVED_INDEX_SHA256,
        expected_config_index_header_sha256=APPROVED_COMPOSITE_SHA256,
    )
    tensor_plan = checkpoint_module.build_qwen35_checkpoint_tensor_plan(
        metadata.hf_config,
        metadata.index_payload,
        metadata.shard_headers,
    )
    text_config = metadata.hf_config.text_config
    if tuple(text_config.layer_types[:4]) != SELECTED_LAYER_TYPES:
        raise ValueError("four-layer schedule is invalid")
    after_metadata = _memory_point(status_reader())
    layout = layout_module.build_qwen35_hybrid_state_layout(
        metadata.hf_config,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        speculative_tokens=1,
    )
    pool = hybrid_state_module.HybridStateTensorPool(
        layout,
        capacity=1,
        device="cpu",
    )
    after_pool = _memory_point(status_reader())
    attention_forward_count = 0

    class _StaticAttentionBackend(nn.Module):
        def forward(self, *_args, **_kwargs):
            nonlocal attention_forward_count
            attention_forward_count += 1
            raise AssertionError("attention backend must not execute")

    target = factory_module.prepare_qwen35_checkpoint_candidate_target(
        metadata.hf_config,
        tensor_plan,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        build_attention_backend=lambda *_args: _StaticAttentionBackend(),
        parameter_device="cpu",
    )
    cpu.inspect_and_touch_cpu_target(target)
    after_target = _memory_point(status_reader())
    tiles = _selected_tiles(target.binding_plan)
    contract = FOUR_LAYER_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    if len(tiles) != contract["tile_count"]:
        raise ValueError("four-layer tile count mismatch")
    if dict(Counter(tile.kind for tile in tiles)) != contract["kind_counts"]:
        raise ValueError("four-layer tile kind counts mismatch")
    model = target.assembly.packed.model
    unique = {}
    for _, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        unique.setdefault(id(tensor), tensor)
    transaction = _stream_four_layer_transaction(
        Path(checkpoint_dir) / APPROVED_SHARD_NAME,
        tiles,
        target.binding_plan,
        tuple(unique.values()),
    )
    after_payload = _memory_point(status_reader())
    memory = {
        "before": before,
        "after_torch": after_torch,
        "after_metadata": after_metadata,
        "after_pool": after_pool,
        "after_target": after_target,
        "after_payload": after_payload,
    }
    row = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "tp_size": tensor_parallel_size,
        "tp_rank": tensor_parallel_rank,
        "process_id": process_id,
        "observed_user": observed_user,
        "observed_hostname": observed_hostname,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "config_sha256": metadata.config_sha256,
        "index_sha256": metadata.index_sha256,
        "config_index_header_sha256": (
            metadata.config_index_header_sha256
        ),
        "metadata_bytes_read": metadata.metadata_bytes_read,
        "selected_layer_indices": list(SELECTED_LAYER_INDICES),
        "selected_layer_types": list(SELECTED_LAYER_TYPES),
        "selected_binding_indices": list(SELECTED_BINDING_INDICES),
        "selected_binding_count": 53,
        "tile_count": len(tiles),
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "logical_payload_bytes_read": (
            transaction["production_payload_bytes_read"]
            + transaction["verifier_payload_bytes_read"]
        ),
        **transaction,
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "target_take_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": attention_forward_count,
        "cuda_initialized_before": cuda_before,
        "cuda_initialized_after": torch.cuda.is_initialized(),
        "memory": memory,
        "total_vmhwm_increment_kib": max(
            0,
            after_payload["vmhwm_kib"] - before["vmhwm_kib"],
        ),
        "post_torch_vmhwm_increment_kib": max(
            0,
            after_payload["vmhwm_kib"] - after_torch["vmhwm_kib"],
        ),
        "post_metadata_vmhwm_increment_kib": max(
            0,
            after_payload["vmhwm_kib"] - after_metadata["vmhwm_kib"],
        ),
    }
    validate_four_layer_cadence_row(row)
    return row


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
        "payload_identity_source": "retained_approved_manifest",
        "fresh_process_per_rank": True,
        "source_file_sha256": hashes,
        "source_tree_sha256": _source_tree_sha256(hashes),
        "rows": list(rows),
    }
    validate_four_layer_cadence_preflight(record)
    return record


def run_remote_four_layer_cadence_preflight(
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
        raise ValueError(f"local four-layer directory exists: {destination}")
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/four_layer_cadence_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in base.TP_ROWS:
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
                "--tp-size",
                str(tp_size),
                "--tp-rank",
                str(tp_rank),
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, "four-layer rank worker")
        row = json.loads(completed.stdout)
        validate_four_layer_cadence_row(row)
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
    _require_success(finalized, "four-layer finalizer")
    record = json.loads(finalized.stdout)
    validate_four_layer_cadence_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("four-layer source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'four_layer_cadence_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'four_layer_cadence_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "four_layer_cadence_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "four-layer artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "four_layer_cadence_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("four-layer artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "four_layer_cadence_preflight.json",
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


def execute_remote_four_layer_cadence_preflight(
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
    return run_remote_four_layer_cadence_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_four_layer_cadence_rank_worker(
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
        raise ValueError("four-layer preflight output already exists")
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
        validate_four_layer_cadence_preflight(record)
    else:
        record = execute_remote_four_layer_cadence_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
