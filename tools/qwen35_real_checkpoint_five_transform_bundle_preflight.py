from __future__ import annotations

import argparse
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


one = _load_sibling(
    "_qwen35_one_tile_payload_preflight_base",
    "qwen35_real_checkpoint_one_tile_payload_preflight.py",
)
cpu = one.cpu
base = one.base

SCHEMA_VERSION = "qwen35.real-checkpoint-five-transform-bundle-preflight.v1"
ROW_SCHEMA_VERSION = "qwen35.real-checkpoint-five-transform-bundle-rank.v1"
REMOTE_TARGET = one.REMOTE_TARGET
REMOTE_PYTHON = one.REMOTE_PYTHON
APPROVED_MODEL_DIR = one.APPROVED_MODEL_DIR
APPROVED_MODEL_MANIFEST_SHA256 = one.APPROVED_MODEL_MANIFEST_SHA256
APPROVED_CONFIG_SHA256 = one.APPROVED_CONFIG_SHA256
APPROVED_INDEX_SHA256 = one.APPROVED_INDEX_SHA256
APPROVED_SHARD_NAME = one.APPROVED_SHARD_NAME
APPROVED_SHARD_SIZE = one.APPROVED_SHARD_SIZE
APPROVED_SHARD_SHA256 = one.APPROVED_SHARD_SHA256
APPROVED_COMPOSITE_SHA256 = one.APPROVED_COMPOSITE_SHA256
SELECTED_BINDING_INDICES = (3, 4, 7, 9, 11)
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 4767744,
        "post_torch": 4243456,
        "post_metadata": 3981312,
    },
    2: {
        "total": 2670592,
        "post_torch": 2408448,
        "post_metadata": 2146304,
    },
}


def _tile(
    binding_index,
    source_name,
    target,
    transform,
    kind,
    dtype,
    source_shape,
    tile_shape,
    source_slices,
    destination_slices,
    ranges,
):
    ranges = [list(value) for value in ranges]
    return {
        "binding_index": binding_index,
        "source_name": source_name,
        "target": target,
        "transform": transform,
        "kind": kind,
        "dtype": dtype,
        "source_shape": list(source_shape),
        "tile_shape": list(tile_shape),
        "source_slices": list(source_slices),
        "destination_slices": list(destination_slices),
        "ranges": ranges,
        "byte_count": sum(end - start for start, end in ranges),
    }


_CONV_SOURCE = "model.language_model.layers.0.linear_attn.conv1d.weight"
_DT_SOURCE = "model.language_model.layers.0.linear_attn.dt_bias"
_QKV_SOURCE = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
_NORM_SOURCE = "model.language_model.layers.0.linear_attn.norm.weight"
_DOWN_SOURCE = "model.language_model.layers.0.mlp.down_proj.weight"


def _contracts(tp_size, tp_rank):
    if (tp_size, tp_rank) == (1, 0):
        tiles = [
            _tile(
                3, _CONV_SOURCE,
                "layers.0.linear_attention.conv_weight",
                "squeeze_conv_channel", "squeeze_axis0",
                "torch.bfloat16", (6144, 1, 4), (6144, 4),
                [[0, 6144, None], 0, [0, 4, None]],
                [[0, 6144, None], [0, 4, None]],
                [(1017209840, 1017258992)],
            ),
            _tile(
                4, _DT_SOURCE, "layers.0.linear_attention.dt_bias",
                "identity", "axis0", "torch.bfloat16", (16,), (16,),
                [[0, 16, None]], [[0, 16, None]],
                [(1017258992, 1017259024)],
            ),
            _tile(
                7, _QKV_SOURCE,
                "layers.0.linear_attention.in_proj_qkv.weight",
                "identity", "segmented_axis0", "torch.bfloat16",
                (6144, 2048), (16, 2048),
                [[0, 16, None], [0, 2048, None]],
                [[0, 16, None], [0, 2048, None]],
                [(1017390096, 1017455632)],
            ),
            _tile(
                9, _NORM_SOURCE,
                "layers.0.linear_attention.norm_weight",
                "identity", "replicated", "torch.float32",
                (128,), (128,), [[0, 128, None]], [[0, 128, None]],
                [(76720, 77232)],
            ),
            _tile(
                11, _DOWN_SOURCE, "layers.0.mlp.down_proj.weight",
                "identity", "axis1", "torch.bfloat16",
                (2048, 6144), (5, 6144),
                [[0, 5, None], [0, 6144, None]],
                [[0, 5, None], [0, 6144, None]],
                [(1059333136, 1059394576)],
            ),
        ]
    else:
        rank0 = tp_rank == 0
        axis1_ranges = (
            [
                [1059333136, 1059339280],
                [1059345424, 1059351568],
                [1059357712, 1059363856],
                [1059370000, 1059376144],
                [1059382288, 1059388432],
                [1059394576, 1059400720],
                [1059406864, 1059413008],
                [1059419152, 1059425296],
                [1059431440, 1059437584],
                [1059443728, 1059449872],
            ]
            if rank0 else
            [
                [1059339280, 1059345424],
                [1059351568, 1059357712],
                [1059363856, 1059370000],
                [1059376144, 1059382288],
                [1059388432, 1059394576],
                [1059400720, 1059406864],
                [1059413008, 1059419152],
                [1059425296, 1059431440],
                [1059437584, 1059443728],
                [1059449872, 1059456016],
            ]
        )
        tiles = [
            _tile(
                3, _CONV_SOURCE,
                "layers.0.linear_attention.conv_weight",
                "squeeze_conv_channel", "squeeze_axis0",
                "torch.bfloat16", (6144, 1, 4), (3072, 4),
                [
                    [0, 3072, None] if rank0 else [3072, 6144, None],
                    0,
                    [0, 4, None],
                ],
                [[0, 3072, None], [0, 4, None]],
                [
                    (1017209840, 1017234416)
                    if rank0 else
                    (1017234416, 1017258992)
                ],
            ),
            _tile(
                4, _DT_SOURCE, "layers.0.linear_attention.dt_bias",
                "identity", "axis0", "torch.bfloat16", (16,), (8,),
                [[0, 8, None] if rank0 else [8, 16, None]],
                [[0, 8, None]],
                [
                    (1017258992, 1017259008)
                    if rank0 else
                    (1017259008, 1017259024)
                ],
            ),
            _tile(
                7, _QKV_SOURCE,
                "layers.0.linear_attention.in_proj_qkv.weight",
                "identity", "segmented_axis0", "torch.bfloat16",
                (6144, 2048), (16, 2048),
                [
                    [0, 16, None] if rank0 else [1024, 1040, None],
                    [0, 2048, None],
                ],
                [[0, 16, None], [0, 2048, None]],
                [
                    (1017390096, 1017455632)
                    if rank0 else
                    (1021584400, 1021649936)
                ],
            ),
            _tile(
                9, _NORM_SOURCE,
                "layers.0.linear_attention.norm_weight",
                "identity", "replicated", "torch.float32",
                (128,), (128,), [[0, 128, None]], [[0, 128, None]],
                [(76720, 77232)],
            ),
            _tile(
                11, _DOWN_SOURCE, "layers.0.mlp.down_proj.weight",
                "identity", "axis1", "torch.bfloat16",
                (2048, 6144), (10, 3072),
                [
                    [0, 10, None],
                    [0, 3072, None] if rank0 else [3072, 6144, None],
                ],
                [[0, 10, None], [0, 3072, None]],
                axis1_ranges,
            ),
        ]
    bytes_per_pass = sum(tile["byte_count"] for tile in tiles)
    ranges_per_pass = sum(len(tile["ranges"]) for tile in tiles)
    return {
        "tiles": tiles,
        "bytes_per_pass": bytes_per_pass,
        "ranges_per_pass": ranges_per_pass,
        "logical_bytes": bytes_per_pass * 2,
        "pread_count": ranges_per_pass * 2,
    }


BUNDLE_CONTRACTS = {
    (1, 0): _contracts(1, 0),
    (2, 0): _contracts(2, 0),
    (2, 1): _contracts(2, 1),
}
SOURCE_FILES = (
    *one.SOURCE_FILES,
    "tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py",
)
REMOTE_RUN_ROOT = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-real-checkpoint-five-transform-bundle-runs"
)
LOCAL_RUN_ROOT = one.LOCAL_RUN_ROOT

_sha256 = one._sha256
_positive_integer = one._positive_integer
_non_negative_integer = one._non_negative_integer
_source_tree_sha256 = one._source_tree_sha256
validate_run_tag = one.validate_run_tag
build_ssh_command = one.build_ssh_command
_require_success = one._require_success
_read_proc_status = one._read_proc_status
_memory_point = one._memory_point
_install_namespace_packages = one._install_namespace_packages
_atomic_write_json = one._atomic_write_json


def __getattr__(name: str):
    if name == "Qwen35CheckpointTile":
        return one.Qwen35CheckpointTile
    raise AttributeError(name)


def _source_hashes(source_root) -> dict[str, str]:
    root = Path(source_root)
    result = {}
    for relative in SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing bundle source: {relative}")
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
                raise ValueError(f"missing bundle source: {relative}")
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
    _require_success(staged, "bundle source staging")
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
    _require_success(verified, "bundle source hashing")
    remote_hashes = json.loads(verified.stdout)
    if remote_hashes != local_hashes:
        raise ValueError("bundle remote source hashes do not match local")
    return {
        "remote_source_dir": remote_source_dir,
        "local_file_sha256": local_hashes,
        "remote_file_sha256": remote_hashes,
        "source_tree_sha256": _source_tree_sha256(local_hashes),
    }


def _validate_tile_ranges(tile_ranges):
    if not isinstance(tile_ranges, (tuple, list)) or not tile_ranges:
        raise ValueError("tile ranges must be non-empty")
    normalized = []
    for ranges in tile_ranges:
        if not isinstance(ranges, (tuple, list)) or not ranges:
            raise ValueError("tile ranges must be non-empty")
        current = []
        previous_end = None
        for value in ranges:
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("payload range is invalid")
            start = _non_negative_integer(value[0], "range start")
            end = _positive_integer(value[1], "range end")
            if end <= start:
                raise ValueError("payload range must be positive")
            if previous_end is not None and start < previous_end:
                if start < current[-1][0]:
                    raise ValueError("payload ranges must be sorted")
                raise ValueError("payload ranges overlap")
            current.append((start, end))
            previous_end = end
        normalized.append(tuple(current))
    return tuple(normalized)


def read_and_verify_exact_ranges(path, tile_ranges) -> dict:
    normalized = _validate_tile_ranges(tile_ranges)
    passes = []
    pread_count = 0
    for _ in range(2):
        descriptor = os.open(Path(path), os.O_RDONLY)
        try:
            tiles = []
            for ranges in normalized:
                parts = []
                for start, end in ranges:
                    payload = os.pread(descriptor, end - start, start)
                    pread_count += 1
                    if len(payload) != end - start:
                        raise ValueError("short payload read")
                    parts.append(payload)
                tiles.append(b"".join(parts))
        finally:
            os.close(descriptor)
        passes.append(tuple(tiles))
    production, verifier = passes
    if production != verifier:
        raise ValueError("independent payload hash mismatch")
    production_hashes = tuple(
        hashlib.sha256(payload).hexdigest() for payload in production
    )
    verifier_hashes = tuple(
        hashlib.sha256(payload).hexdigest() for payload in verifier
    )
    if production_hashes != verifier_hashes:
        raise ValueError("independent payload hash mismatch")
    byte_count = sum(len(payload) for payload in production)
    return {
        "production_tiles": production,
        "verifier_tiles": verifier,
        "production_sha256": production_hashes,
        "verifier_sha256": verifier_hashes,
        "production_bytes_read": byte_count,
        "verifier_bytes_read": byte_count,
        "open_count": 2,
        "pread_count": pread_count,
    }


def _tensor_bytes(tensor) -> bytes:
    import torch

    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def copy_verify_and_reverse_rollback_bundle(
    tiles,
    source_tensors,
    *,
    unique_tensors,
) -> dict:
    import torch
    from tinyvllm.models.qwen35_checkpoint_tiled_loading import (
        _copy_qwen35_checkpoint_tile,
    )

    tiles = tuple(tiles)
    source_tensors = tuple(source_tensors)
    if len(tiles) != 5 or len(source_tensors) != 5:
        raise ValueError("bundle must contain exactly five tiles")
    destination_ids = [id(tile.destination) for tile in tiles]
    if len(set(destination_ids)) != len(destination_ids):
        raise ValueError("selected destinations must be distinct")
    if any(
        int(torch.count_nonzero(tensor).item()) != 0
        for tensor in unique_tensors
    ):
        raise ValueError("all registered tensors must initially be zero")
    snapshots = [
        tile.destination[tile.destination_slices].clone()
        for tile in tiles
    ]
    selected_ids = set(destination_ids)
    tile_results = []
    copied = set()
    for tile, source_tensor in zip(
        tiles,
        source_tensors,
        strict=True,
    ):
        destination = tile.destination[tile.destination_slices]
        source_sha = hashlib.sha256(_tensor_bytes(source_tensor)).hexdigest()
        _copy_qwen35_checkpoint_tile(tile, source_tensor)
        destination_sha = hashlib.sha256(
            _tensor_bytes(destination)
        ).hexdigest()
        if destination_sha != source_sha:
            raise ValueError("destination payload hash mismatch")
        copied.add(id(tile.destination))
        not_yet_zero = all(
            int(torch.count_nonzero(other.destination[
                other.destination_slices
            ]).item()) == 0
            for other in tiles
            if id(other.destination) not in copied
        )
        if not not_yet_zero:
            raise ValueError("not-yet-selected destination mutated")
        tile_results.append({
            "source_tensor_sha256": source_sha,
            "destination_sha256": destination_sha,
            "destination_initially_zero": True,
            "destination_changed_after_copy": (
                not torch.equal(destination, snapshots[len(tile_results)])
            ),
            "not_yet_selected_destinations_remained_zero": not_yet_zero,
        })
    non_selected_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
        if id(tensor) not in selected_ids
    )
    if not non_selected_zero:
        raise ValueError("non-selected tensor mutation detected")
    rollback_order = []
    restored = True
    for tile, snapshot in reversed(
        list(zip(tiles, snapshots, strict=True))
    ):
        destination = tile.destination[tile.destination_slices]
        with torch.no_grad():
            destination.copy_(snapshot)
        rollback_order.append(tile.binding_index)
        restored = restored and torch.equal(destination, snapshot)
    all_zero = all(
        int(torch.count_nonzero(tensor).item()) == 0
        for tensor in unique_tensors
    )
    return {
        "tile_results": tile_results,
        "selected_destinations_distinct": True,
        "non_selected_tensors_remained_zero": non_selected_zero,
        "rollback_order": rollback_order,
        "all_selected_snapshots_restored": restored,
        "all_unique_tensors_zero_after_rollback": all_zero,
    }


def _slice_json(values) -> list:
    return [
        [value.start, value.stop, value.step]
        if isinstance(value, slice) else value
        for value in values
    ]


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
        raise ValueError("bundle memory points are invalid")
    for name in names:
        point = memory[name]
        if not isinstance(point, Mapping):
            raise ValueError(f"bundle memory {name} is invalid")
        _positive_integer(point.get("vmrss_kib"), f"{name} vmrss")
        _positive_integer(point.get("vmhwm_kib"), f"{name} vmhwm")
    deltas = (
        _non_negative_integer(
            row.get("total_vmhwm_increment_kib"), "total VmHWM"
        ),
        _non_negative_integer(
            row.get("post_torch_vmhwm_increment_kib"),
            "post-Torch VmHWM",
        ),
        _non_negative_integer(
            row.get("post_metadata_vmhwm_increment_kib"),
            "post-metadata VmHWM",
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
        raise ValueError("bundle VmHWM deltas are invalid")
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    if deltas[0] > ceilings["total"]:
        raise ValueError("bundle total VmHWM exceeds ceiling")
    if deltas[1] > ceilings["post_torch"]:
        raise ValueError("bundle post-Torch VmHWM exceeds ceiling")
    if deltas[2] > ceilings["post_metadata"]:
        raise ValueError("bundle post-metadata VmHWM exceeds ceiling")


def validate_five_transform_bundle_row(row):
    if not isinstance(row, Mapping):
        raise ValueError("bundle row must be a mapping")
    if row.get("schema_version") != ROW_SCHEMA_VERSION:
        raise ValueError("bundle row schema is invalid")
    if row.get("status") != "PASS":
        raise ValueError("bundle row status must be PASS")
    tp = (row.get("tp_size"), row.get("tp_rank"))
    if tp not in BUNDLE_CONTRACTS:
        raise ValueError("bundle TP context is invalid")
    _positive_integer(row.get("process_id"), "process_id")
    contract = BUNDLE_CONTRACTS[tp]
    exact = {
        "observed_user": "sitian",
        "checkpoint_dir": APPROVED_MODEL_DIR,
        "config_sha256": APPROVED_CONFIG_SHA256,
        "index_sha256": APPROVED_INDEX_SHA256,
        "config_index_header_sha256": APPROVED_COMPOSITE_SHA256,
        "metadata_bytes_read": 144024,
        "selected_binding_indices": list(SELECTED_BINDING_INDICES),
        "selected_tile_count": 5,
        "selected_source_count": 5,
        "selected_shard_count": 1,
        "production_payload_bytes_read": contract["bytes_per_pass"],
        "verifier_payload_bytes_read": contract["bytes_per_pass"],
        "logical_payload_bytes_read": contract["logical_bytes"],
        "payload_hashes_recomputed": False,
        "plan_loads": 320,
        "plan_skips": 312,
        "plan_payload_bytes": 4548144832,
        "selected_destinations_distinct": True,
        "non_selected_tensors_remained_zero": True,
        "rollback_order": list(reversed(SELECTED_BINDING_INDICES)),
        "all_selected_snapshots_restored": True,
        "all_unique_tensors_zero_after_rollback": True,
        "open_count": 2,
        "pread_count": contract["pread_count"],
        "loader_call_count": 0,
        "assignment_call_count": 0,
        "target_take_count": 0,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
    }
    for name, expected in exact.items():
        if row.get(name) != expected:
            messages = {
                "selected_binding_indices": "bundle binding contract invalid",
                "logical_payload_bytes_read": "bundle payload bytes invalid",
                "pread_count": "bundle pread count invalid",
                "selected_destinations_distinct": (
                    "bundle destinations must be distinct"
                ),
                "non_selected_tensors_remained_zero": (
                    "bundle non-selected mutation detected"
                ),
                "rollback_order": "bundle rollback order invalid",
                "all_selected_snapshots_restored": (
                    "bundle snapshots were not restored"
                ),
                "loader_call_count": "bundle loader calls must be zero",
                "cuda_initialized_after": "bundle CUDA must remain off",
            }
            raise ValueError(messages.get(
                name, f"bundle row {name} is invalid"
            ))
    results = row.get("tile_results")
    if not isinstance(results, list) or len(results) != 5:
        raise ValueError("bundle tile results are invalid")
    for result, expected in zip(
        results,
        contract["tiles"],
        strict=True,
    ):
        tile_exact = {
            "binding_index": expected["binding_index"],
            "source_name": expected["source_name"],
            "target": expected["target"],
            "transform": expected["transform"],
            "kind": expected["kind"],
            "dtype": expected["dtype"],
            "source_shape": expected["source_shape"],
            "tile_shape": expected["tile_shape"],
            "source_slices": expected["source_slices"],
            "destination_slices": expected["destination_slices"],
            "ranges": expected["ranges"],
            "range_count": len(expected["ranges"]),
            "tile_bytes": expected["byte_count"],
            "destination_initially_zero": True,
            "destination_changed_after_copy": True,
            "not_yet_selected_destinations_remained_zero": True,
        }
        for name, value in tile_exact.items():
            if result.get(name) != value:
                if name == "ranges":
                    raise ValueError("bundle tile ranges are invalid")
                raise ValueError(f"bundle tile {name} is invalid")
        hashes = [
            result.get(name)
            for name in (
                "production_sha256",
                "verifier_sha256",
                "source_tensor_sha256",
                "destination_sha256",
            )
        ]
        for index, digest in enumerate(hashes):
            _sha256(digest, f"bundle tile hash {index}")
        if len(set(hashes)) != 1:
            raise ValueError("bundle tile hash mismatch")
    hostname = row.get("observed_hostname")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("bundle hostname is invalid")
    _validate_memory(row)
    return row


def validate_five_transform_bundle_preflight(record):
    if not isinstance(record, Mapping):
        raise ValueError("bundle preflight must be a mapping")
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
    for name, expected in exact.items():
        if record.get(name) != expected:
            raise ValueError(f"bundle preflight {name} is invalid")
    hashes = record.get("source_file_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(SOURCE_FILES):
        raise ValueError("bundle source hashes are invalid")
    for name, digest in hashes.items():
        _sha256(digest, f"source SHA256 for {name}")
    if (
        _sha256(record.get("source_tree_sha256"), "source_tree_sha256")
        != _source_tree_sha256(hashes)
    ):
        raise ValueError("bundle source tree is invalid")
    rows = record.get("rows")
    if not isinstance(rows, list) or [
        (row.get("tp_size"), row.get("tp_rank"))
        for row in rows
    ] != list(base.TP_ROWS):
        raise ValueError("bundle TP rows are invalid")
    for row in rows:
        validate_five_transform_bundle_row(row)
    pids = [row["process_id"] for row in rows]
    if len(set(pids)) != len(pids):
        raise ValueError("bundle process IDs must be unique")
    return record


def _selected_tiles(binding_plan, tp_size, tp_rank):
    from tinyvllm.models.qwen35_checkpoint_tiles import (
        build_qwen35_checkpoint_tile_plan,
    )

    plan = build_qwen35_checkpoint_tile_plan(
        binding_plan,
        max_tile_bytes=65536,
    )
    contract = BUNDLE_CONTRACTS[(tp_size, tp_rank)]
    tiles = []
    for expected in contract["tiles"]:
        matches = [
            tile for tile in plan.tiles
            if tile.binding_index == expected["binding_index"]
        ]
        if not matches:
            raise ValueError("selected bundle binding has no tile")
        tile = matches[0]
        binding = binding_plan.bindings[tile.binding_index]
        observed = {
            "binding_index": tile.binding_index,
            "source_name": tile.source_name,
            "target": tile.target,
            "transform": binding.load.transform,
            "kind": tile.kind,
            "dtype": str(tile.dtype),
            "source_shape": list(tile.source_tensor_shape),
            "tile_shape": list(tile.tile_shape),
            "source_slices": _slice_json(tile.source_slices),
            "destination_slices": _slice_json(tile.destination_slices),
            "byte_count": tile.byte_count,
        }
        comparable = {
            name: expected[name]
            for name in observed
        }
        if observed != comparable:
            raise ValueError("selected bundle tile contract mismatch")
        tiles.append(tile)
    if len({id(tile.destination) for tile in tiles}) != 5:
        raise ValueError("selected bundle destinations must be distinct")
    return tuple(tiles)


def run_five_transform_rank_worker(
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
    tiles = _selected_tiles(
        target.binding_plan,
        tensor_parallel_size,
        tensor_parallel_rank,
    )
    contract = BUNDLE_CONTRACTS[
        (tensor_parallel_size, tensor_parallel_rank)
    ]
    read_result = read_and_verify_exact_ranges(
        Path(checkpoint_dir) / APPROVED_SHARD_NAME,
        tuple(tile["ranges"] for tile in contract["tiles"]),
    )
    source_tensors = []
    for tile, payload in zip(
        tiles,
        read_result["production_tiles"],
        strict=True,
    ):
        source_tensors.append(
            torch.frombuffer(
                bytearray(payload),
                dtype=tile.dtype,
            ).clone().reshape(tile.tile_shape)
        )
    model = target.assembly.packed.model
    unique = {}
    for _, tensor in (
        list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False))
    ):
        unique.setdefault(id(tensor), tensor)
    copy_result = copy_verify_and_reverse_rollback_bundle(
        tiles,
        source_tensors,
        unique_tensors=tuple(unique.values()),
    )
    after_payload = _memory_point(status_reader())
    tile_results = []
    for expected, read_sha, copy_values in zip(
        contract["tiles"],
        read_result["production_sha256"],
        copy_result["tile_results"],
        strict=True,
    ):
        tile_results.append({
            **{
                name: expected[name]
                for name in (
                    "binding_index",
                    "source_name",
                    "target",
                    "transform",
                    "kind",
                    "dtype",
                    "source_shape",
                    "tile_shape",
                    "source_slices",
                    "destination_slices",
                    "ranges",
                )
            },
            "range_count": len(expected["ranges"]),
            "tile_bytes": expected["byte_count"],
            "production_sha256": read_sha,
            "verifier_sha256": read_sha,
            **copy_values,
        })
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
        "selected_tile_count": 5,
        "selected_source_count": 5,
        "selected_shard_count": 1,
        "production_payload_bytes_read": (
            read_result["production_bytes_read"]
        ),
        "verifier_payload_bytes_read": (
            read_result["verifier_bytes_read"]
        ),
        "logical_payload_bytes_read": (
            read_result["production_bytes_read"]
            + read_result["verifier_bytes_read"]
        ),
        "payload_hashes_recomputed": False,
        "plan_loads": len(tensor_plan.loads),
        "plan_skips": len(tensor_plan.skips),
        "plan_payload_bytes": tensor_plan.payload_bytes,
        "tile_results": tile_results,
        "selected_destinations_distinct": (
            copy_result["selected_destinations_distinct"]
        ),
        "non_selected_tensors_remained_zero": (
            copy_result["non_selected_tensors_remained_zero"]
        ),
        "rollback_order": copy_result["rollback_order"],
        "all_selected_snapshots_restored": (
            copy_result["all_selected_snapshots_restored"]
        ),
        "all_unique_tensors_zero_after_rollback": (
            copy_result["all_unique_tensors_zero_after_rollback"]
        ),
        "open_count": read_result["open_count"],
        "pread_count": read_result["pread_count"],
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
    validate_five_transform_bundle_row(row)
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
    validate_five_transform_bundle_preflight(record)
    return record


def run_remote_five_transform_bundle_preflight(
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
        raise ValueError(f"local bundle directory exists: {destination}")
    remote_run_dir = f"{REMOTE_RUN_ROOT}/{run_tag}"
    remote_artifact = (
        f"{remote_run_dir}/five_transform_bundle_preflight.json"
    )
    worker = (
        f"{staged['remote_source_dir']}/"
        "tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py"
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
        _require_success(completed, "bundle rank worker")
        row = json.loads(completed.stdout)
        validate_five_transform_bundle_row(row)
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
    _require_success(finalized, "bundle finalizer")
    record = json.loads(finalized.stdout)
    validate_five_transform_bundle_preflight(record)
    if (
        record["source_file_sha256"] != staged["local_file_sha256"]
        or record["source_file_sha256"] != staged["remote_file_sha256"]
        or record["source_tree_sha256"] != staged["source_tree_sha256"]
    ):
        raise ValueError("bundle source binding mismatch")
    source_manifest = _source_manifest(run_tag, staged)
    script = "\n".join([
        "import json,pathlib,sys",
        f"root=pathlib.Path({remote_run_dir!r})",
        "payload=json.load(sys.stdin)",
        "record=json.loads((root/'five_transform_bundle_preflight.json').read_text())",
        "temporary=root/'.source_manifest.json.tmp'",
        "temporary.write_text(json.dumps(payload['source_manifest'],sort_keys=True,separators=(',',':'))+'\\n')",
        "temporary.replace(root/'source_manifest.json')",
        "result={'five_transform_bundle_preflight':record,'source_manifest':json.loads((root/'source_manifest.json').read_text())}",
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
            "five_transform_bundle_preflight": record,
            "source_manifest": source_manifest,
        }),
        text=True,
        capture_output=True,
    )
    _require_success(round_trip, "bundle artifact round trip")
    returned = json.loads(round_trip.stdout)
    if returned != {
        "five_transform_bundle_preflight": record,
        "source_manifest": source_manifest,
    }:
        raise ValueError("bundle artifact round-trip mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{run_tag}.",
        suffix=".tmp",
        dir=destination.parent,
    ))
    try:
        _atomic_write_json(
            temporary / "five_transform_bundle_preflight.json",
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


def execute_remote_five_transform_bundle_preflight(
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
    return run_remote_five_transform_bundle_preflight(
        source_root,
        run_tag,
        staged=staged,
        local_run_root=local_run_root,
        command_runner=command_runner,
    )


def _rank_worker_main(arguments) -> int:
    if str(Path(arguments.checkpoint_dir).resolve()) != APPROVED_MODEL_DIR:
        raise ValueError("worker checkpoint_dir is not the approved model")
    row = run_five_transform_rank_worker(
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
        raise ValueError("bundle preflight output already exists")
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
        validate_five_transform_bundle_preflight(record)
    else:
        record = execute_remote_five_transform_bundle_preflight(
            arguments.source_root,
            arguments.run_tag,
        )
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
