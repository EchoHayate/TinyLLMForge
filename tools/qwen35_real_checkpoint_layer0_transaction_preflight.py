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


bundle = _load_sibling(
    "_qwen35_five_transform_bundle_preflight_base",
    "qwen35_real_checkpoint_five_transform_bundle_preflight.py",
)
one = bundle.one
cpu = bundle.cpu
base = bundle.base

SCHEMA_VERSION = "qwen35.real-checkpoint-layer0-transaction-preflight.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-layer0-transaction-rank.v1"
REMOTE_TARGET = bundle.REMOTE_TARGET
REMOTE_PYTHON = bundle.REMOTE_PYTHON
APPROVED_MODEL_DIR = bundle.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = bundle.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = bundle.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = bundle.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = bundle.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = bundle.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = bundle.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = bundle.APPROVED_COMPOSITE_SHA256
DATA_START = one.DATA_START
SELECTED_BINDING_INDICES = tuple(range(1, 15))
ROLLBACK_BINDING_ORDER = tuple(range(1, 13)) + (14,)
LAYER_CONTRACTS = {
    (1, 0): {
        "tile_count": 1826,
        "kind_counts": {
            "axis0": 900,
            "axis1": 538,
            "replicated": 3,
            "segmented_axis0": 384,
            "squeeze_axis0": 1,
        },
        "bytes_per_pass": 117629536,
        "ranges_per_pass": 1826,
        "pread_count": 3652,
        "logical_bytes": 235259072,
    },
    (2, 0): {
        "tile_count": 917,
        "kind_counts": {
            "axis0": 452,
            "axis1": 269,
            "replicated": 3,
            "segmented_axis0": 192,
            "squeeze_axis0": 1,
        },
        "bytes_per_pass": 58819120,
        "ranges_per_pass": 4744,
        "pread_count": 9488,
        "logical_bytes": 117638240,
    },
    (2, 1): {
        "tile_count": 917,
        "kind_counts": {
            "axis0": 452,
            "axis1": 269,
            "replicated": 3,
            "segmented_axis0": 192,
            "squeeze_axis0": 1,
        },
        "bytes_per_pass": 58819120,
        "ranges_per_pass": 4744,
        "pread_count": 9488,
        "logical_bytes": 117638240,
    },
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 5033168,
        "post_torch": 4508876,
        "post_metadata": 4246732,
    },
    2: {
        "total": 2818048,
        "post_torch": 2555904,
        "post_metadata": 2293760,
    },
}

_BINDING_BASE = {
    1: (
        "model.language_model.layers.0.input_layernorm.weight",
        "layers.0.input_layernorm.weight",
        "replicated", "torch.bfloat16",
    ),
    2: (
        "model.language_model.layers.0.linear_attn.A_log",
        "layers.0.linear_attention.A_log",
        "axis0", "torch.float32",
    ),
    3: (
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attention.conv_weight",
        "squeeze_axis0", "torch.bfloat16",
    ),
    4: (
        "model.language_model.layers.0.linear_attn.dt_bias",
        "layers.0.linear_attention.dt_bias",
        "axis0", "torch.bfloat16",
    ),
    5: (
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attention.in_proj_a.weight",
        "replicated", "torch.bfloat16",
    ),
    6: (
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "layers.0.linear_attention.in_proj_b.weight",
        "replicated", "torch.bfloat16",
    ),
    7: (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attention.in_proj_qkv.weight",
        "segmented_axis0", "torch.bfloat16",
    ),
    8: (
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "layers.0.linear_attention.in_proj_z.weight",
        "axis0", "torch.bfloat16",
    ),
    9: (
        "model.language_model.layers.0.linear_attn.norm.weight",
        "layers.0.linear_attention.norm_weight",
        "replicated", "torch.float32",
    ),
    10: (
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "layers.0.linear_attention.out_proj.weight",
        "axis1", "torch.bfloat16",
    ),
    11: (
        "model.language_model.layers.0.mlp.down_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "axis1", "torch.bfloat16",
    ),
    12: (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "axis0", "torch.bfloat16",
    ),
    13: (
        "model.language_model.layers.0.mlp.up_proj.weight",
        "layers.0.mlp.gate_up_proj.weight",
        "axis0", "torch.bfloat16",
    ),
    14: (
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "replicated", "torch.bfloat16",
    ),
}
_TP1_BINDINGS = {
    1: ((2048,), None, 1, 1, 4096),
    2: ((16,), None, 1, 1, 64),
    3: ((6144, 4), None, 1, 1, 49152),
    4: ((16,), None, 1, 1, 32),
    5: ((16, 2048), None, 1, 1, 65536),
    6: ((16, 2048), None, 1, 1, 65536),
    7: ((6144, 2048), None, 384, 384, 25165824),
    8: ((2048, 2048), None, 128, 128, 8388608),
    9: ((128,), None, 1, 1, 512),
    10: ((2048, 2048), None, 128, 128, 8388608),
    11: ((2048, 6144), None, 410, 410, 25165824),
    12: ((6144, 2048), [0, 6144], 384, 384, 25165824),
    13: ((6144, 2048), [6144, 6144], 384, 384, 25165824),
    14: ((2048,), None, 1, 1, 4096),
}
_TP2_BINDINGS = {
    1: ((2048,), None, 1, 1, 4096),
    2: ((8,), None, 1, 1, 32),
    3: ((3072, 4), None, 1, 1, 24576),
    4: ((8,), None, 1, 1, 16),
    5: ((8, 2048), None, 1, 1, 32768),
    6: ((8, 2048), None, 1, 1, 32768),
    7: ((3072, 2048), None, 192, 192, 12582912),
    8: ((1024, 2048), None, 64, 64, 4194304),
    9: ((128,), None, 1, 1, 512),
    10: ((2048, 1024), None, 64, 2048, 4194304),
    11: ((2048, 3072), None, 205, 2048, 12582912),
    12: ((3072, 2048), [0, 3072], 192, 192, 12582912),
    13: ((3072, 2048), [3072, 3072], 192, 192, 12582912),
    14: ((2048,), None, 1, 1, 4096),
}


def binding_contract(index: int, tp_size: int) -> dict:
    if index not in _BINDING_BASE or tp_size not in (1, 2):
        raise ValueError("layer0 binding contract is invalid")
    source_name, target, kind, dtype = _BINDING_BASE[index]
    shape, destination_slice, tiles, ranges, byte_count = (
        _TP1_BINDINGS[index]
        if tp_size == 1 else
        _TP2_BINDINGS[index]
    )
    return {
        "binding_index": index,
        "source_name": source_name,
        "target": target,
        "kind": kind,
        "dtype": dtype,
        "local_shape": list(shape),
        "destination_slice": destination_slice,
        "tile_count": tiles,
        "range_count": ranges,
        "byte_count": byte_count,
        "destination_slice_by_tp": {
            size: (
                _TP1_BINDINGS[index][1]
                if size == 1 else
                _TP2_BINDINGS[index][1]
            )
            for size in (1, 2)
        },
    }


BINDING_CONTRACTS = {
    index: binding_contract(index, 1)
    for index in SELECTED_BINDING_INDICES
}
SOURCE_FILES = (
    *bundle.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_layer0_transaction_preflight.py",
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-layer0-transaction-runs"
)
LOCAL_RUN_ROOT = bundle.LOCAL_RUN_ROOT

_sha256 = bundle._sha256
_positive_integer = bundle._positive_integer
_non_negative_integer = bundle._non_negative_integer
_source_tree_sha256 = bundle._source_tree_sha256
validate_run_tag = bundle.validate_run_tag
build_ssh_command = bundle.build_ssh_command
_require_success = bundle._require_success
_read_proc_status = bundle._read_proc_status
_memory_point = bundle._memory_point
_install_namespace_packages = bundle._install_namespace_packages
_atomic_write_json = bundle._atomic_write_json


def __getattr__(name: str):
    if name == "Qwen35CheckpointTile":
        return bundle.Qwen35CheckpointTile
    raise AttributeError(name)


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing layer0 source: {relative}")
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
                raise ValueError(f"missing layer0 source: {relative}")
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
    _require_success(staged, "layer0 source staging")
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
            "env", "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON, "-B", "-c", script,
        ]),
        text=True,
        capture_output=True,
    )
    _require_success(verified, "layer0 source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("layer0 remote source hashes do not match local")
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _dtype_width(dtype) -> int:
    name = str(dtype)
    if name in ("BF16", "torch.bfloat16"):
        return 2
    if name in ("F32", "torch.float32"):
        return 4
    raise ValueError("tile dtype width is unsupported")


def _slice_bounds(value, length, name):
    if not isinstance(value, slice) or value.step not in (None, 1):
        raise ValueError(f"{name} slice is invalid")
    start = 0 if value.start is None else value.start
    stop = length if value.stop is None else value.stop
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
        or stop > length
    ):
        raise ValueError(f"{name} slice is invalid")
    return start, stop


def derive_tile_ranges(binding, tile, *, data_start: int):
    data_start = _non_negative_integer(data_start, "data_start")
    metadata = binding.load.metadata
    shape = tuple(metadata.shape)
    if tuple(tile.source_tensor_shape) != shape:
        raise ValueError("tile source shape is invalid")
    width = _dtype_width(metadata.dtype)
    tensor_start, tensor_end = metadata.data_offsets
    tensor_bytes = tensor_end - tensor_start
    expected_bytes = width
    for dimension in shape:
        expected_bytes *= dimension
    if tensor_bytes != expected_bytes:
        raise ValueError("metadata byte count is invalid")
    absolute = data_start + tensor_start
    slices = tile.source_slices
    ranges = []
    if len(shape) == 1 and len(slices) == 1:
        start, stop = _slice_bounds(slices[0], shape[0], "rank1")
        ranges.append((absolute + start * width, absolute + stop * width))
    elif len(shape) == 2 and len(slices) == 2:
        row_start, row_stop = _slice_bounds(
            slices[0], shape[0], "row"
        )
        column_start, column_stop = _slice_bounds(
            slices[1], shape[1], "column"
        )
        if column_start == 0 and column_stop == shape[1]:
            ranges.append((
                absolute + row_start * shape[1] * width,
                absolute + row_stop * shape[1] * width,
            ))
        else:
            for row in range(row_start, row_stop):
                ranges.append((
                    absolute + (row * shape[1] + column_start) * width,
                    absolute + (row * shape[1] + column_stop) * width,
                ))
    elif len(shape) == 3 and len(slices) == 3:
        row_start, row_stop = _slice_bounds(
            slices[0], shape[0], "convolution row"
        )
        if slices[1] != 0 or shape[1] != 1:
            raise ValueError("convolution channel slice is invalid")
        column_start, column_stop = _slice_bounds(
            slices[2], shape[2], "convolution column"
        )
        if column_start != 0 or column_stop != shape[2]:
            raise ValueError("convolution column slice is invalid")
        ranges.append((
            absolute + row_start * shape[2] * width,
            absolute + row_stop * shape[2] * width,
        ))
    else:
        raise ValueError("tile source slice rank is invalid")
    if any(
        start < data_start + tensor_start
        or end > data_start + tensor_end
        or end <= start
        for start, end in ranges
    ):
        raise ValueError("tile range is outside metadata")
    if sum(end - start for start, end in ranges) != tile.byte_count:
        raise ValueError("tile range byte count is invalid")
    return tuple(ranges)


def _tensor_bytes(tensor) -> bytes:
    import torch

    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _alias_groups(tiles):
    groups = {}
    for tile in tiles:
        groups.setdefault(id(tile.destination), []).append(
            tile.binding_index
        )
    return [
        sorted(set(indices))
        for indices in groups.values()
        if len(set(indices)) > 1
    ]


def apply_verify_and_rollback_layer_tiles(
    tiles,
    source_tensors,
    *,
    binding_order,
    unique_tensors,
    layer_destination_ids,
):
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    tiles = tuple(tiles)
    source_tensors = tuple(source_tensors)
    if len(tiles) != len(source_tensors) or not tiles:
        raise ValueError("layer tile payloads are invalid")
    observed_order = []
    for tile in tiles:
        if not observed_order or observed_order[-1] != tile.binding_index:
            observed_order.append(tile.binding_index)
    if tuple(observed_order) != tuple(binding_order):
        raise ValueError("layer tile order is invalid")
    aliases = _alias_groups(tiles)
    if aliases != [[12, 13]]:
        raise ValueError("layer alias contract is invalid")
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for tensor in unique_tensors
    ):
        raise ValueError("registered tensors must initially be zero")
    first_tiles = {}
    destination_objects = {}
    for tile in tiles:
        destination_objects.setdefault(id(tile.destination), tile.destination)
        first_tiles.setdefault(id(tile.destination), tile.binding_index)
    if set(destination_objects) != set(layer_destination_ids):
        raise ValueError("layer destination coverage is invalid")
    snapshots = {
        object_id: tensor.clone()
        for object_id, tensor in destination_objects.items()
    }
    for tile, source in zip(tiles, source_tensors, strict=True):
        source_sha = hashlib.sha256(_tensor_bytes(source)).hexdigest()
        _copy_qwen35_checkpoint_tile(tile, source)
        destination = tile.destination[tile.destination_slices]
        if hashlib.sha256(_tensor_bytes(destination)).hexdigest() != source_sha:
            raise ValueError("layer destination payload hash mismatch")
    non_layer_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) not in layer_destination_ids
    )
    rollback_binding_order = []
    restored = True
    for object_id, tensor in reversed(list(destination_objects.items())):
        with torch.no_grad():
            tensor.copy_(snapshots[object_id])
        rollback_binding_order.append(first_tiles[object_id])
        restored = restored and torch.equal(tensor, snapshots[object_id])
    all_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
    )
    return {
        "unique_destination_count": len(destination_objects),
        "alias_groups": aliases,
        "non_layer_tensors_remained_zero": non_layer_zero,
        "rollback_binding_order": rollback_binding_order,
        "all_layer_snapshots_restored": restored,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def _validate_memory(row) -> None:
    memory = row.get("memory")
    names = (
        "before", "after_torch", "after_metadata",
        "after_pool", "after_target", "after_payload",
    )
    if not isinstance(memory, Mapping) or set(memory) != set(names):
        raise ValueError("layer0 memory points are invalid")
    for name in names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(f"layer0 memory {name} is invalid")
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
        max(0, memory["after_payload"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"]),
        max(0, memory["after_payload"]["vmhwm_kib"]
            - memory["after_torch"]["vmhwm_kib"]),
        max(0, memory["after_payload"]["vmhwm_kib"]
            - memory["after_metadata"]["vmhwm_kib"]),
    )
    if deltas != expected:
        raise ValueError("layer0 VmHWM deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    if deltas[0] > ceilings["total"]:
        raise ValueError("layer0 total VmHWM exceeds ceiling")
    if deltas[1] > ceilings["post_torch"]:
        raise ValueError("layer0 post-Torch VmHWM exceeds ceiling")
    if deltas[2] > ceilings["post_metadata"]:
        raise ValueError("layer0 post-metadata VmHWM exceeds ceiling")


def validate_layer0_transaction_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("layer0 row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("layer0 row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("layer0 row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in LAYER_CONTRACTS:
        raise ValueError("layer0 TP context is invalid")
    _positive_integer(row.get("process_id"), "process_id")
    contract = LAYER_CONTRACTS[tp]
    exact = {
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_indices": list(SELECTED_BINDING_INDICES),
        "selected_binding_count": 14,
        "unique_destination_count": 13,
        "alias_groups": [[12, 13]],
        "tile_count": contract["tile_count"],
        "kind_counts": contract["kind_counts"],
        "ranges_per_pass": contract["ranges_per_pass"],
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "open_count": 2,
        "pread_count": contract["pread_count"],
        "layer_destinations_changed": True,
        "non_layer_tensors_remained_zero": True,
        "rollback_binding_order": list(reversed(ROLLBACK_BINDING_ORDER)),
        "all_layer_snapshots_restored": True,
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
        "selected_binding_indices": "layer0 binding contract invalid",
        "unique_destination_count": "layer0 destination count invalid",
        "alias_groups": "layer0 alias contract invalid",
        "tile_count": "layer0 tile count invalid",
        "pread_count": "layer0 pread count invalid",
        "non_layer_tensors_remained_zero": "layer0 non-layer mutation",
        "rollback_binding_order": "layer0 rollback order invalid",
        "all_layer_snapshots_restored": "layer0 snapshots not restored",
        "loader_call_count": "layer0 loader calls must be zero",
        "cuda_initialized_after": "layer0 CUDA must remain off",
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(messages.get(
                name, f"layer0 row {name} is invalid"
            ))
    results = row.get("binding_results")
    if not isinstance(results, list) or len(results) != 14:
        raise ValueError("layer0 binding results are invalid")
    for result, index in zip(
        results, SELECTED_BINDING_INDICES, strict=True
    ):
        expected = binding_contract(index, row["tp_size"])
        for name in (
            "binding_index", "source_name", "target", "kind", "dtype",
            "local_shape", "destination_slice", "tile_count",
            "range_count", "byte_count",
        ):
            if result.get(name) != expected[name]:
                raise ValueError(
                    f"layer0 binding {index} {name} is invalid"
                )
        if result.get("coverage_complete") is not True:
            raise ValueError("layer0 binding coverage is incomplete")
        hashes = [
            result.get(name)
            for name in (
                "production_sha256", "verifier_sha256",
                "source_tensor_sha256", "destination_sha256",
            )
        ]
        for offset, digest in enumerate(hashes):
            _sha256(digest, f"layer0 binding hash {offset}")
        if len(set(hashes)) != 1:
            raise ValueError("layer0 binding hash mismatch")
    aggregate_hashes = (
        _sha256(row.get("aggregate_source_sha256"), "aggregate source"),
        _sha256(
            row.get("aggregate_destination_sha256"),
            "aggregate destination",
        ),
    )
    if len(set(aggregate_hashes)) != 1:
        raise ValueError("layer0 aggregate hash mismatch")
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("layer0 hostname is invalid")
    _validate_memory(row)
    return row


def validate_layer0_transaction_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("layer0 preflight must be a mapping")
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
            raise ValueError(f"layer0 preflight {name} is invalid")
    hashes = record.get("source_file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SOURCE_FILES):
        raise ValueError("layer0 source hashes are invalid")
    for name, digest in hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError("layer0 source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(base.TP_ROWS):
        raise ValueError("layer0 TP rows are invalid")
    for row in rows:
        validate_layer0_transaction_row(row)
    pids = [row["process_id"] for row in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("layer0 process IDs must be unique")
    return record


def _layer_tiles(binding_plan):
    from tinyvllm.models.qwen35_checkpoint_tiles import (
        build_qwen35_checkpoint_tile_plan,
    )

    indices = tuple(
        index
        for index, binding in enumerate(binding_plan.bindings)
        if binding.load.weight.target.startswith("layers.0.")
    )
    if indices != SELECTED_BINDING_INDICES:
        raise ValueError("layer0 binding indices are invalid")
    plan = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=65536,
    )
    tiles = tuple(
        tile for tile in plan.tiles
        if tile.binding_index in SELECTED_BINDING_INDICES
    )
    return tiles


def _binding_destination_view(binding):
    if binding.destination_slice is None:
        return binding.destination
    offset, length = binding.destination_slice
    return binding.destination[offset:offset + length]


def _read_tile(descriptor, ranges):
    parts = []
    for start, end in ranges:
        payload = os.pread(descriptor, end - start, start)
        if len(payload) != end - start:
            raise ValueError("short payload read")
        parts.append(payload)
    return b"".join(parts)


def _stream_layer_transaction(
    shard_path,
    tiles,
    binding_plan,
    unique_tensors,
):
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    observed_binding_order = []
    for tile in tiles:
        if (
            not observed_binding_order
            or observed_binding_order[-1] != tile.binding_index
        ):
            observed_binding_order.append(tile.binding_index)
    if tuple(observed_binding_order) != SELECTED_BINDING_INDICES:
        raise ValueError("layer0 tile order is invalid")
    alias_groups = _alias_groups(tiles)
    if alias_groups != [[12, 13]]:
        raise ValueError("layer0 alias contract is invalid")
    destination_objects = {}
    first_binding = {}
    for tile in tiles:
        destination_objects.setdefault(id(tile.destination), tile.destination)
        first_binding.setdefault(id(tile.destination), tile.binding_index)
    if len(destination_objects) != 13:
        raise ValueError("layer0 unique destination count is invalid")
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for tensor in unique_tensors
    ):
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
    statistics = {
        index: {"tile_count": 0, "range_count": 0, "byte_count": 0}
        for index in SELECTED_BINDING_INDICES
    }
    aggregate_source = hashlib.sha256()
    production_bytes = 0
    verifier_bytes = 0
    pread_count = 0
    production_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    verifier_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    try:
        for tile in tiles:
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
                raise ValueError("independent layer0 payload mismatch")
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
                raise ValueError("layer0 destination payload mismatch")
            production_hashes[tile.binding_index].update(production)
            verifier_hashes[tile.binding_index].update(verifier)
            aggregate_source.update(production)
            values = statistics[tile.binding_index]
            values["tile_count"] += 1
            values["range_count"] += len(ranges)
            values["byte_count"] += len(production)
            del tensor
    finally:
        os.close(production_descriptor)
        os.close(verifier_descriptor)
    binding_results = []
    destination_aggregate = hashlib.sha256()
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
            raise ValueError("layer0 binding coverage is incomplete")
        destination_bytes = _tensor_bytes(
            _binding_destination_view(binding)
        )
        destination_sha = hashlib.sha256(destination_bytes).hexdigest()
        production_sha = production_hashes[index].hexdigest()
        verifier_sha = verifier_hashes[index].hexdigest()
        if len({production_sha, verifier_sha, destination_sha}) != 1:
            raise ValueError("layer0 binding hash mismatch")
        destination_aggregate.update(destination_bytes)
        binding_results.append({
            **{
                name: contract[name]
                for name in (
                    "binding_index", "source_name", "target", "kind",
                    "dtype", "local_shape", "destination_slice",
                    "tile_count", "range_count", "byte_count",
                )
            },
            "production_sha256": production_sha,
            "verifier_sha256": verifier_sha,
            "source_tensor_sha256": production_sha,
            "destination_sha256": destination_sha,
            "coverage_complete": True,
        })
    layer_destination_ids = set(destination_objects)
    layer_changed = all(
        not torch.equal(tensor, snapshots[object_id])
        for object_id, tensor in destination_objects.items()
    )
    non_layer_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) not in layer_destination_ids
    )
    rollback_binding_order = []
    restored = True
    for object_id, tensor in reversed(list(destination_objects.items())):
        with torch.no_grad():
            tensor.copy_(snapshots[object_id])
        rollback_binding_order.append(first_binding[object_id])
        restored = restored and torch.equal(tensor, snapshots[object_id])
    all_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
    )
    return {
        "binding_results": binding_results,
        "aggregate_source_sha256": aggregate_source.hexdigest(),
        "aggregate_destination_sha256": destination_aggregate.hexdigest(),
        "production_payload_bytes_read": production_bytes,
        "verifier_payload_bytes_read": verifier_bytes,
        "open_count": 2,
        "pread_count": pread_count,
        "unique_destination_count": len(destination_objects),
        "alias_groups": alias_groups,
        "layer_destinations_changed": layer_changed,
        "non_layer_tensors_remained_zero": non_layer_zero,
        "rollback_binding_order": rollback_binding_order,
        "all_layer_snapshots_restored": restored,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def run_layer0_rank_worker(
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
    after_metadata = _memory_point(status_reader())
    layout = layout_module.build_qwen35_hybrid_state_layout(
        metadata.hf_config,
        tensor_parallel_size=tensor_parallel_size,
        dtype=torch.bfloat16,
        speculative_tokens=1,
    )
    pool = hybrid_state_module.HybridStateTensorPool(
        layout, capacity=1, device="cpu"
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
    tiles = _layer_tiles(target.binding_plan)
    contract = LAYER_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    if len(tiles) != contract["tile_count"]:
        raise ValueError("layer0 tile count mismatch")
    if dict(Counter(tile.kind for tile in tiles)) != contract["kind_counts"]:
        raise ValueError("layer0 tile kind counts mismatch")
    model = target.assembly.packed.model
    unique = {}
    for _, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        unique.setdefault(id(tensor), tensor)
    transaction = _stream_layer_transaction(
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
        "selected_binding_indices": list(SELECTED_BINDING_INDICES),
        "selected_binding_count": 14,
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
            0, after_payload["vmhwm_kib"] - before["vmhwm_kib"]
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
    validate_layer0_transaction_row(row)
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
    validate_layer0_transaction_preflight(record)
    return record


def run_remote_layer0_transaction_preflight(
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
        raise ValueError(f"local layer0 directory exists: {destination}")
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/layer0_transaction_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_layer0_transaction_preflight.py"
    )
    rows = []
    for tp_size, tp_rank in base.TP_ROWS:
        completed = command_runner(
            build_ssh_command([
                "env", "CUDA_VISIBLE_DEVICES=",
                "PYTHONDONTWRITEBYTECODE=1",
                "OMP_NUM_THREADS=8", "MKL_NUM_THREADS=8",
                REMOTE_PYTHON, "-B", worker,
                "internal-rank-worker",
                "--source-root", staged["remote_source_dir"],
                "--checkpoint-dir", APPROVED_MODEL_DIR,
                "--tp-size", str(tp_size),
                "--tp-rank", str(tp_rank),
            ]),
            text=True,
            capture_output=True,
        )
        _require_success(completed, "layer0 rank worker")
        row = json.loads(completed.stdout)
        validate_layer0_transaction_row(row)
        rows.append(row)
    finalized = command_runner(
        build_ssh_command([
            "env", "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON, "-B", worker,
            "internal-finalize",
            "--source-root", staged["remote_source_dir"],
            "--output", remote_artifact,
        ]),
        input=json.dumps({"rows": rows}),
        text=True,
        capture_output=True,
    )
    _require_success(finalized, "layer0 finalizer")
    record = json.loads(finalized.stdout)
    validate_layer0_transaction_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("layer0 source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'layer0_transaction_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'layer0_transaction_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
        "print(json.dumps(result,sort_keys=True,separators=(',',':')))",
    ])
    round_trip = command_runner(
        build_ssh_command([
            "env", "PYTHONDONTWRITEBYTECODE=1",
            REMOTE_PYTHON, "-B", "-c", script,
        ]),
        input=json.dumps({
            "layer0_transaction_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "layer0 artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "layer0_transaction_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("layer0 artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "layer0_transaction_preflight.json",
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


def execute_remote_layer0_transaction_preflight(
    source_root,
    run_tag,
    *,
    local_run_root=LOCAL_RUN_ROOT,
    command_runner=subprocess.run,
):
    staged = stage_source(
        source_root, run_tag, command_runner=command_runner
    )
    return run_remote_layer0_transaction_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_layer0_rank_worker(
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
        raise ValueError("layer0 preflight output already exists")
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
        validate_layer0_transaction_preflight(record)
    else:
        record = execute_remote_layer0_transaction_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
