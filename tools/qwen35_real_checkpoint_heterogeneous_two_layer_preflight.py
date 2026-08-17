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


layer0 = _load_sibling(
    "_qwen35_layer0_transaction_preflight_base",
    "qwen35_real_checkpoint_layer0_transaction_preflight.py",
)
bundle = layer0.bundle
cpu = layer0.cpu
base = layer0.base

SCHEMA_VERSION = "qwen35.real-checkpoint-heterogeneous-two-layer.v1"
ROW_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-heterogeneous-two-layer-rank.v1"
)
REMOTE_TARGET = layer0.REMOTE_TARGET
REMOTE_PYTHON = layer0.REMOTE_PYTHON
APPROVED_MODEL_DIR = layer0.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = (
    layer0.APPROVED_MODEL_MANIFEST_SHA256
)
APPROVED_CONFIG_SHA256 = layer0.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = layer0.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = layer0.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = layer0.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = layer0.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = layer0.APPROVED_COMPOSITE_SHA256
DATA_START = layer0.DATA_START
SELECTED_LAYER_INDICES = (0, 3)
SELECTED_LAYER_TYPES = ("linear_attention", "full_attention")
SELECTED_BINDING_INDICES = (
    *range(1, 15),
    *range(227, 238),
)
UNIQUE_BINDING_ORDER = (
    *range(1, 13),
    14,
    227,
    228,
    229,
    *range(231, 238),
)
ALIAS_GROUPS = [[12, 13], [229, 230]]
TWO_LAYER_CONTRACTS = {
    (1, 0): {
        "tile_count": 3456,
        "kind_counts": {
            "axis0": 1988,
            "axis1": 1076,
            "replicated": 7,
            "segmented_axis0": 384,
            "squeeze_axis0": 1,
        },
        "bytes_per_pass": 222496352,
        "ranges_per_pass": 3456,
        "pread_count": 6912,
        "logical_bytes": 444992704,
        "layers": {
            0: {
                "tile_count": 1826,
                "range_count": 1826,
                "byte_count": 117629536,
            },
            3: {
                "tile_count": 1630,
                "range_count": 1630,
                "byte_count": 104866816,
            },
        },
    },
    (2, 0): {
        "tile_count": 1734,
        "kind_counts": {
            "axis0": 996,
            "axis1": 538,
            "replicated": 7,
            "segmented_axis0": 192,
            "squeeze_axis0": 1,
        },
        "bytes_per_pass": 111257136,
        "ranges_per_pass": 9388,
        "pread_count": 18776,
        "logical_bytes": 222514272,
        "layers": {
            0: {
                "tile_count": 917,
                "range_count": 4744,
                "byte_count": 58819120,
            },
            3: {
                "tile_count": 817,
                "range_count": 4644,
                "byte_count": 52438016,
            },
        },
    },
    (2, 1): {
        "tile_count": 1734,
        "kind_counts": {
            "axis0": 996,
            "axis1": 538,
            "replicated": 7,
            "segmented_axis0": 192,
            "squeeze_axis0": 1,
        },
        "bytes_per_pass": 111257136,
        "ranges_per_pass": 9388,
        "pread_count": 18776,
        "logical_bytes": 222514272,
        "layers": {
            0: {
                "tile_count": 917,
                "range_count": 4744,
                "byte_count": 58819120,
            },
            3: {
                "tile_count": 817,
                "range_count": 4644,
                "byte_count": 52438016,
            },
        },
    },
}
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 5242880,
        "post_torch": 4718592,
        "post_metadata": 4456448,
    },
    2: {
        "total": 2949120,
        "post_torch": 2686976,
        "post_metadata": 2424832,
    },
}

_LAYER3_BASE = {
    227: (
        "model.language_model.layers.3.input_layernorm.weight",
        "layers.3.input_layernorm.weight",
        "replicated",
        "torch.bfloat16",
    ),
    228: (
        "model.language_model.layers.3.mlp.down_proj.weight",
        "layers.3.mlp.down_proj.weight",
        "axis1",
        "torch.bfloat16",
    ),
    229: (
        "model.language_model.layers.3.mlp.gate_proj.weight",
        "layers.3.mlp.gate_up_proj.weight",
        "axis0",
        "torch.bfloat16",
    ),
    230: (
        "model.language_model.layers.3.mlp.up_proj.weight",
        "layers.3.mlp.gate_up_proj.weight",
        "axis0",
        "torch.bfloat16",
    ),
    231: (
        "model.language_model.layers.3.post_attention_layernorm.weight",
        "layers.3.post_attention_layernorm.weight",
        "replicated",
        "torch.bfloat16",
    ),
    232: (
        "model.language_model.layers.3.self_attn.k_norm.weight",
        "layers.3.full_attention.k_norm.weight",
        "replicated",
        "torch.bfloat16",
    ),
    233: (
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "layers.3.full_attention.k_projection.weight",
        "axis0",
        "torch.bfloat16",
    ),
    234: (
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "layers.3.full_attention.output_projection.weight",
        "axis1",
        "torch.bfloat16",
    ),
    235: (
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "layers.3.full_attention.q_norm.weight",
        "replicated",
        "torch.bfloat16",
    ),
    236: (
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "layers.3.full_attention.q_projection.weight",
        "axis0",
        "torch.bfloat16",
    ),
    237: (
        "model.language_model.layers.3.self_attn.v_proj.weight",
        "layers.3.full_attention.v_projection.weight",
        "axis0",
        "torch.bfloat16",
    ),
}
_LAYER3_TP1 = {
    227: ((2048,), None, 1, 1, 4096),
    228: ((2048, 6144), None, 410, 410, 25165824),
    229: ((6144, 2048), [0, 6144], 384, 384, 25165824),
    230: ((6144, 2048), [6144, 6144], 384, 384, 25165824),
    231: ((2048,), None, 1, 1, 4096),
    232: ((256,), None, 1, 1, 512),
    233: ((512, 2048), None, 32, 32, 2097152),
    234: ((2048, 2048), None, 128, 128, 8388608),
    235: ((256,), None, 1, 1, 512),
    236: ((4096, 2048), None, 256, 256, 16777216),
    237: ((512, 2048), None, 32, 32, 2097152),
}
_LAYER3_TP2 = {
    227: ((2048,), None, 1, 1, 4096),
    228: ((2048, 3072), None, 205, 2048, 12582912),
    229: ((3072, 2048), [0, 3072], 192, 192, 12582912),
    230: ((3072, 2048), [3072, 3072], 192, 192, 12582912),
    231: ((2048,), None, 1, 1, 4096),
    232: ((256,), None, 1, 1, 512),
    233: ((256, 2048), None, 16, 16, 1048576),
    234: ((2048, 1024), None, 64, 2048, 4194304),
    235: ((256,), None, 1, 1, 512),
    236: ((2048, 2048), None, 128, 128, 8388608),
    237: ((256, 2048), None, 16, 16, 1048576),
}


def binding_contract(index: int, tp_size: int) -> dict:
    if index in layer0.SELECTED_BINDING_INDICES:
        result = dict(layer0.binding_contract(index, tp_size))
        result["layer_index"] = 0
        return result
    if index not in _LAYER3_BASE or tp_size not in (1, 2):
        raise ValueError("two-layer binding contract is invalid")
    source_name, target, kind, dtype = _LAYER3_BASE[index]
    shape, destination_slice, tiles, ranges, byte_count = (
        _LAYER3_TP1[index] if tp_size == 1 else _LAYER3_TP2[index]
    )
    return {
        "binding_index": index,
        "layer_index": 3,
        "source_name": source_name,
        "target": target,
        "kind": kind,
        "dtype": dtype,
        "local_shape": list(shape),
        "destination_slice": destination_slice,
        "tile_count": tiles,
        "range_count": ranges,
        "byte_count": byte_count,
    }


SOURCE_FILES = (
    *layer0.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py",
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-heterogeneous-two-layer-runs"
)
LOCAL_RUN_ROOT = layer0.LOCAL_RUN_ROOT

_sha256 = layer0._sha256
_positive_integer = layer0._positive_integer
_non_negative_integer = layer0._non_negative_integer
_source_tree_sha256 = layer0._source_tree_sha256
validate_run_tag = layer0.validate_run_tag
build_ssh_command = layer0.build_ssh_command
_require_success = layer0._require_success
_read_proc_status = layer0._read_proc_status
_memory_point = layer0._memory_point
_install_namespace_packages = layer0._install_namespace_packages
_atomic_write_json = layer0._atomic_write_json
derive_tile_ranges = layer0.derive_tile_ranges
_tensor_bytes = layer0._tensor_bytes
_binding_destination_view = layer0._binding_destination_view
_read_tile = layer0._read_tile


def __getattr__(name: str):
    if name == "Qwen35CheckpointTile":
        return layer0.Qwen35CheckpointTile
    raise AttributeError(name)


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing two-layer source: {relative}")
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
                raise ValueError(f"missing two-layer source: {relative}")
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
    _require_success(staged, "two-layer source staging")
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
    _require_success(verified, "two-layer source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("two-layer remote source hashes do not match local")
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


def apply_verify_and_rollback_two_layer_tiles(
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
        raise ValueError("two-layer tile payloads are invalid")
    observed_order = []
    for tile in tiles:
        if not observed_order or observed_order[-1] != tile.binding_index:
            observed_order.append(tile.binding_index)
    if tuple(observed_order) != tuple(binding_order):
        raise ValueError("two-layer binding order is invalid")
    observed_layers = []
    for index in observed_order:
        layer_index = binding_layer[index]
        if not observed_layers or observed_layers[-1] != layer_index:
            observed_layers.append(layer_index)
    if observed_layers != [0, 3]:
        raise ValueError("two-layer layer order is invalid")
    aliases = _alias_groups(tiles)
    if aliases != ALIAS_GROUPS:
        raise ValueError("two-layer alias contract is invalid")
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for tensor in unique_tensors
    ):
        raise ValueError("registered tensors must initially be zero")
    destination_objects = {}
    first_binding = {}
    for tile in tiles:
        destination_objects.setdefault(id(tile.destination), tile.destination)
        first_binding.setdefault(id(tile.destination), tile.binding_index)
    if set(destination_objects) != set(selected_destination_ids):
        raise ValueError("two-layer destination coverage is invalid")
    snapshots = {
        object_id: tensor.clone()
        for object_id, tensor in destination_objects.items()
    }
    layer3_ids = {
        id(tile.destination)
        for tile in tiles
        if binding_layer[tile.binding_index] == 3
    }
    layer0_ids = set(destination_objects) - layer3_ids
    layer0_changed = False
    layer3_checked = False
    for tile, source in zip(tiles, source_tensors, strict=True):
        layer_index = binding_layer[tile.binding_index]
        if layer_index == 3 and not layer3_checked:
            layer0_changed = all(
                not torch.equal(
                    destination_objects[object_id],
                    snapshots[object_id],
                )
                for object_id in layer0_ids
            )
            if not layer0_changed:
                raise ValueError("layer 0 is incomplete before layer 3")
            if any(
                int(torch.count_nonzero(destination_objects[object_id]).item())
                != 0
                for object_id in layer3_ids
            ):
                raise ValueError("layer 3 changed before its first copy")
            layer3_checked = True
        source_sha = hashlib.sha256(_tensor_bytes(source)).hexdigest()
        _copy_qwen35_checkpoint_tile(tile, source)
        destination = tile.destination[tile.destination_slices]
        if hashlib.sha256(_tensor_bytes(destination)).hexdigest() != source_sha:
            raise ValueError("two-layer destination payload hash mismatch")
    non_selected_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) not in selected_destination_ids
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
        "unique_destination_count": len(destination_objects),
        "alias_groups": aliases,
        "layer_completion_order": observed_layers,
        "layer0_changed_before_layer3": layer0_changed,
        "layer3_zero_before_first_copy": layer3_checked,
        "non_selected_tensors_remained_zero": non_selected_zero,
        "rollback_binding_order": rollback_binding_order,
        "all_selected_snapshots_restored": restored,
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
        raise ValueError("two-layer memory points are invalid")
    for name in names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(f"two-layer memory {name} is invalid")
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
        raise ValueError("two-layer VmHWM deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    if deltas[0] > ceilings["total"]:
        raise ValueError("two-layer total VmHWM exceeds ceiling")
    if deltas[1] > ceilings["post_torch"]:
        raise ValueError("two-layer post-Torch VmHWM exceeds ceiling")
    if deltas[2] > ceilings["post_metadata"]:
        raise ValueError("two-layer post-metadata VmHWM exceeds ceiling")


def validate_heterogeneous_two_layer_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("two-layer row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("two-layer row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("two-layer row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in TWO_LAYER_CONTRACTS:
        raise ValueError("two-layer TP context is invalid")
    _positive_integer(row.get("process_id"), "process_id")
    contract = TWO_LAYER_CONTRACTS[tp]
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
        "selected_binding_count": 25,
        "unique_destination_count": 23,
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
        "layer0_changed_before_layer3": True,
        "layer3_zero_before_first_copy": True,
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
        "selected_layer_types": "two-layer layer types invalid",
        "selected_binding_indices": "two-layer binding contract invalid",
        "selected_binding_count": "two-layer binding count invalid",
        "unique_destination_count": "two-layer destination count invalid",
        "alias_groups": "two-layer alias contract invalid",
        "layer_completion_order": "two-layer layer order invalid",
        "layer0_changed_before_layer3": (
            "two-layer layer 0 isolation invalid"
        ),
        "layer3_zero_before_first_copy": "two-layer layer 3 isolation invalid",
        "non_selected_tensors_remained_zero": (
            "two-layer non-selected mutation"
        ),
        "rollback_binding_order": "two-layer rollback order invalid",
        "all_selected_snapshots_restored": (
            "two-layer snapshots not restored"
        ),
        "target_take_count": "two-layer target.take calls must be zero",
        "cuda_initialized_after": "two-layer CUDA must remain off",
    }
    for name, value in exact.items():
        if row.get(name) != value:
            raise ValueError(messages.get(
                name,
                f"two-layer row {name} is invalid",
            ))
    results = row.get("binding_results")
    if not isinstance(results, list) or len(results) != 25:
        raise ValueError("two-layer binding results are invalid")
    for result, index in zip(
        results,
        SELECTED_BINDING_INDICES,
        strict=True,
    ):
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
                    f"two-layer binding {index} {name} is invalid"
                )
        if result.get("coverage_complete") is not True:
            raise ValueError("two-layer binding coverage is incomplete")
        hashes = [
            result.get(name)
            for name in (
                "production_sha256",
                "verifier_sha256",
                "source_tensor_sha256",
                "destination_sha256",
            )
        ]
        for offset, digest in enumerate(hashes):
            _sha256(digest, f"two-layer binding hash {offset}")
        if len(set(hashes)) != 1:
            raise ValueError("two-layer binding hash mismatch")
    layer_results = row.get("layer_results")
    if not isinstance(layer_results, list) or len(layer_results) != 2:
        raise ValueError("two-layer layer results are invalid")
    for result, layer_index, layer_type, binding_indices in zip(
        layer_results,
        SELECTED_LAYER_INDICES,
        SELECTED_LAYER_TYPES,
        (tuple(range(1, 15)), tuple(range(227, 238))),
        strict=True,
    ):
        layer_contract = contract["layers"][layer_index]
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
                    f"two-layer layer {layer_index} {name} is invalid"
                )
        hashes = [
            _sha256(
                result.get(name),
                f"two-layer layer {layer_index} {name}",
            )
            for name in (
                "production_sha256",
                "verifier_sha256",
                "destination_sha256",
            )
        ]
        if len(set(hashes)) != 1:
            raise ValueError("two-layer layer hash mismatch")
    aggregates = (
        _sha256(row.get("aggregate_source_sha256"), "aggregate source"),
        _sha256(
            row.get("aggregate_destination_sha256"),
            "aggregate destination",
        ),
    )
    if len(set(aggregates)) != 1:
        raise ValueError("two-layer aggregate hash mismatch")
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("two-layer hostname is invalid")
    _validate_memory(row)
    return row


def validate_heterogeneous_two_layer_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("two-layer preflight must be a mapping")
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
            raise ValueError(f"two-layer preflight {name} is invalid")
    hashes = record.get("source_file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SOURCE_FILES):
        raise ValueError("two-layer source hashes are invalid")
    for name, digest in hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError("two-layer source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(base.TP_ROWS):
        raise ValueError("two-layer TP rows are invalid")
    for row in rows:
        validate_heterogeneous_two_layer_row(row)
    pids = [row["process_id"] for row in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("two-layer process IDs must be unique")
    return record


def _selected_binding_indices(bindings):
    return tuple(
        index
        for index, binding in enumerate(bindings)
        if (
            binding.load.weight.target.startswith("layers.0.")
            or binding.load.weight.target.startswith("layers.3.")
        )
    )


def _selected_tiles(binding_plan):
    from tinyvllm.models.qwen35_checkpoint_tiles import (
        build_qwen35_checkpoint_tile_plan,
    )

    indices = _selected_binding_indices(binding_plan.bindings)
    if indices != SELECTED_BINDING_INDICES:
        raise ValueError("two-layer binding indices are invalid")
    plan = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=65536,
    )
    return tuple(
        tile
        for tile in plan.tiles
        if tile.binding_index in SELECTED_BINDING_INDICES
    )


def _binding_layer(index: int) -> int:
    return 0 if index < 227 else 3


def _stream_two_layer_transaction(
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
    observed_layer_order = []
    for tile in tiles:
        if (
            not observed_binding_order
            or observed_binding_order[-1] != tile.binding_index
        ):
            observed_binding_order.append(tile.binding_index)
            layer_index = _binding_layer(tile.binding_index)
            if (
                not observed_layer_order
                or observed_layer_order[-1] != layer_index
            ):
                observed_layer_order.append(layer_index)
    if tuple(observed_binding_order) != SELECTED_BINDING_INDICES:
        raise ValueError("two-layer tile order is invalid")
    if observed_layer_order != [0, 3]:
        raise ValueError("two-layer layer order is invalid")
    aliases = _alias_groups(tiles)
    if aliases != ALIAS_GROUPS:
        raise ValueError("two-layer alias contract is invalid")
    destination_objects = {}
    first_binding = {}
    for tile in tiles:
        destination_objects.setdefault(id(tile.destination), tile.destination)
        first_binding.setdefault(id(tile.destination), tile.binding_index)
    if len(destination_objects) != 23:
        raise ValueError("two-layer unique destination count is invalid")
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for tensor in unique_tensors
    ):
        raise ValueError("registered tensors must initially be zero")
    snapshots = {
        object_id: tensor.clone()
        for object_id, tensor in destination_objects.items()
    }
    layer3_ids = {
        id(tile.destination)
        for tile in tiles
        if _binding_layer(tile.binding_index) == 3
    }
    layer0_ids = set(destination_objects) - layer3_ids
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
    layer0_changed = False
    layer3_checked = False
    production_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    verifier_descriptor = os.open(Path(shard_path), os.O_RDONLY)
    try:
        for tile in tiles:
            layer_index = _binding_layer(tile.binding_index)
            if layer_index == 3 and not layer3_checked:
                layer0_changed = all(
                    not torch.equal(
                        destination_objects[object_id],
                        snapshots[object_id],
                    )
                    for object_id in layer0_ids
                )
                if not layer0_changed:
                    raise ValueError(
                        "layer 0 is incomplete before layer 3"
                    )
                if any(
                    int(torch.count_nonzero(
                        destination_objects[object_id]
                    ).item()) != 0
                    for object_id in layer3_ids
                ):
                    raise ValueError(
                        "layer 3 changed before its first copy"
                    )
                layer3_checked = True
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
                raise ValueError("independent two-layer payload mismatch")
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
                raise ValueError("two-layer destination payload mismatch")
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
    finally:
        os.close(production_descriptor)
        os.close(verifier_descriptor)
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
            raise ValueError("two-layer binding coverage is incomplete")
        destination_bytes = _tensor_bytes(
            _binding_destination_view(binding)
        )
        destination_sha = hashlib.sha256(destination_bytes).hexdigest()
        production_sha = production_hashes[index].hexdigest()
        verifier_sha = verifier_hashes[index].hexdigest()
        if len({production_sha, verifier_sha, destination_sha}) != 1:
            raise ValueError("two-layer binding hash mismatch")
        destination_aggregate.update(destination_bytes)
        layer_destination_hashes[_binding_layer(index)].update(
            destination_bytes
        )
        binding_results.append({
            **{
                name: contract[name]
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
                )
            },
            "production_sha256": production_sha,
            "verifier_sha256": verifier_sha,
            "source_tensor_sha256": production_sha,
            "destination_sha256": destination_sha,
            "coverage_complete": True,
        })
    layer_results = []
    contract = TWO_LAYER_CONTRACTS[
        (
            binding_plan.tensor_parallel_size,
            binding_plan.tensor_parallel_rank,
        )
    ]
    for layer_index, layer_type, binding_indices in (
        (0, "linear_attention", tuple(range(1, 15))),
        (3, "full_attention", tuple(range(227, 238))),
    ):
        if layer_statistics[layer_index] != contract["layers"][layer_index]:
            raise ValueError("two-layer layer coverage is incomplete")
        hashes = (
            layer_production_hashes[layer_index].hexdigest(),
            layer_verifier_hashes[layer_index].hexdigest(),
            layer_destination_hashes[layer_index].hexdigest(),
        )
        if len(set(hashes)) != 1:
            raise ValueError("two-layer layer hash mismatch")
        layer_results.append({
            "layer_index": layer_index,
            "layer_type": layer_type,
            "binding_indices": list(binding_indices),
            "binding_count": len(binding_indices),
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
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) not in selected_ids
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
        "layer_results": layer_results,
        "layer_completion_order": observed_layer_order,
        "layer0_changed_before_layer3": layer0_changed,
        "layer3_zero_before_first_copy": layer3_checked,
        "aggregate_source_sha256": aggregate_source.hexdigest(),
        "aggregate_destination_sha256": (
            destination_aggregate.hexdigest()
        ),
        "production_payload_bytes_read": production_bytes,
        "verifier_payload_bytes_read": verifier_bytes,
        "open_count": 2,
        "pread_count": pread_count,
        "unique_destination_count": len(destination_objects),
        "alias_groups": aliases,
        "selected_destinations_changed": selected_changed,
        "non_selected_tensors_remained_zero": non_selected_zero,
        "rollback_binding_order": rollback_binding_order,
        "all_selected_snapshots_restored": restored,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def run_heterogeneous_two_layer_rank_worker(
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
    if (
        text_config.layer_types[0] != "linear_attention"
        or text_config.layer_types[3] != "full_attention"
    ):
        raise ValueError("two-layer schedule is invalid")
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
    contract = TWO_LAYER_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    if len(tiles) != contract["tile_count"]:
        raise ValueError("two-layer tile count mismatch")
    if dict(Counter(tile.kind for tile in tiles)) != contract["kind_counts"]:
        raise ValueError("two-layer tile kind counts mismatch")
    model = target.assembly.packed.model
    unique = {}
    for _, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        unique.setdefault(id(tensor), tensor)
    transaction = _stream_two_layer_transaction(
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
        "selected_binding_count": 25,
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
    validate_heterogeneous_two_layer_row(row)
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
    validate_heterogeneous_two_layer_preflight(record)
    return record


def run_remote_heterogeneous_two_layer_preflight(
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
        raise ValueError(f"local two-layer directory exists: {destination}")
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/heterogeneous_two_layer_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py"
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
        _require_success(completed, "two-layer rank worker")
        row = json.loads(completed.stdout)
        validate_heterogeneous_two_layer_row(row)
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
    _require_success(finalized, "two-layer finalizer")
    record = json.loads(finalized.stdout)
    validate_heterogeneous_two_layer_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("two-layer source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'heterogeneous_two_layer_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'heterogeneous_two_layer_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "heterogeneous_two_layer_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "two-layer artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "heterogeneous_two_layer_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("two-layer artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "heterogeneous_two_layer_preflight.json",
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


def execute_remote_heterogeneous_two_layer_preflight(
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
    return run_remote_heterogeneous_two_layer_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_heterogeneous_two_layer_rank_worker(
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
        raise ValueError("two-layer preflight output already exists")
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
        validate_heterogeneous_two_layer_preflight(record)
    else:
        record = execute_remote_heterogeneous_two_layer_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
