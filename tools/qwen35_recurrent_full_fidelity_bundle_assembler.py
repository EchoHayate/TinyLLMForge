from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import types
import uuid

import torch


ROOT = Path(__file__).resolve().parents[1]
if "tinyvllm" not in sys.modules:
    tinyvllm_package = types.ModuleType("tinyvllm")
    tinyvllm_package.__path__ = [str(ROOT / "tinyvllm")]
    sys.modules["tinyvllm"] = tinyvllm_package
if "tinyvllm.engine" not in sys.modules:
    engine_package = types.ModuleType("tinyvllm.engine")
    engine_package.__path__ = [str(ROOT / "tinyvllm/engine")]
    sys.modules["tinyvllm.engine"] = engine_package
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))


import qwen35_recurrent_int8_calibration_contract as calibration_contract
from tinyvllm.engine.qwen35_recurrent_capture import _publish_no_clobber
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    CaptureRunIdentity,
    canonical_json_bytes,
    validate_rank_manifest,
    validate_run_identity,
)


def _load_json_bytes(path, name):
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}") from error
    return payload, value


def _validate_root(path, name):
    path = Path(path)
    if str(path) == "" or "\0" in str(path) or ".." in path.parts:
        raise ValueError(f"{name} is invalid")
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{name} must not contain symlinks")
    return path


def _positive_world_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("world_size must be a positive integer")
    return value


def _source_relative_path(record):
    workload_id = record.workload_id
    if (
        not isinstance(workload_id, str)
        or not workload_id
        or "\0" in workload_id
        or "\\" in workload_id
    ):
        raise ValueError("workload_id must be a safe path component")
    workload_path = PurePosixPath(workload_id)
    if (
        workload_path.is_absolute()
        or len(workload_path.parts) != 1
        or workload_path.parts[0] in {".", ".."}
        or workload_path.as_posix() != workload_id
    ):
        raise ValueError("workload_id must be a safe path component")
    return Path(
        f"source/rank{record.rank}/{workload_id}/"
        f"layer{record.layer_index}.pt"
    )


def _write_payload_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_and_validate_rank_manifest(path, *, expected_rank):
    payload, value = _load_json_bytes(path, "rank capture manifest")
    manifest = validate_rank_manifest(value)
    if manifest.rank != expected_rank:
        raise ValueError("rank manifest rank mismatch")
    if payload != canonical_json_bytes(manifest.payload()) + b"\n":
        raise ValueError("rank manifest bytes are not canonical")
    return manifest


def _validate_capture_identity(rank_root, manifest):
    identity_path = rank_root / "capture_identity.json"
    payload, value = _load_json_bytes(identity_path, "capture identity")
    identity = validate_run_identity(value)
    if identity != manifest.identity:
        raise ValueError("capture identity mismatch")
    if payload != canonical_json_bytes(identity.payload()) + b"\n":
        raise ValueError("capture identity bytes are not canonical")
    return identity_path


def _expected_rank_paths(rank_root, manifest, identity_path, manifest_path):
    allowed = {
        rank_root,
        identity_path,
        manifest_path,
        rank_root / "workloads",
        rank_root / "tensors",
    }
    for workload_id in manifest.identity.workload_ids:
        allowed.add(
            rank_root / "workloads" / f"{workload_id}.complete.json"
        )
        allowed.add(rank_root / "tensors" / workload_id)
    for record in manifest.tensors:
        allowed.add(
            rank_root
            / "tensors"
            / record.workload_id
            / f"layer{record.layer_index}.pt"
        )
    return allowed


def _validate_rank_tree(rank_root, allowed):
    for path in rank_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink below closed rank root")
        if path not in allowed:
            raise ValueError("extra or untracked path below closed rank root")
    missing = tuple(path for path in allowed if not path.exists())
    if missing:
        raise ValueError("closed rank inventory is missing paths")
    for path in allowed:
        if path in {
            rank_root,
            rank_root / "workloads",
            rank_root / "tensors",
        } or path.parent == rank_root / "tensors":
            if not path.is_dir():
                raise ValueError("closed rank directory inventory mismatch")
        elif not path.is_file():
            raise ValueError("closed rank file inventory mismatch")


def _validate_tensor_bytes(payload, record, load_tensor):
    if hashlib.sha256(payload).hexdigest() != record.sha256:
        raise ValueError("tensor payload hash mismatch")
    try:
        tensor = load_tensor(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError("tensor payload cannot be loaded") from error
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("tensor payload did not contain a tensor")
    if tensor.dtype != torch.float32:
        raise ValueError("tensor payload dtype must be float32")
    if tensor.ndim != 3:
        raise ValueError("tensor payload must be rank-3")
    if tuple(tensor.shape) != record.shape:
        raise ValueError("tensor payload shape mismatch")
    if tensor.numel() * tensor.element_size() != record.logical_bytes:
        raise ValueError("tensor payload logical bytes mismatch")
    del tensor


def _load_rank_manifest(
    capture_root,
    rank,
    *,
    expected_identity,
):
    rank_root = capture_root / f"rank{rank}"
    if rank_root.is_symlink() or not rank_root.is_dir():
        raise ValueError("closed rank directory is missing")
    manifest_path = rank_root / "rank_capture_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("closed rank manifest is missing")
    manifest = _load_and_validate_rank_manifest(
        manifest_path,
        expected_rank=rank,
    )
    if expected_identity is not None and manifest.identity != expected_identity:
        raise ValueError("cross-rank identity mismatch")
    identity_path = _validate_capture_identity(rank_root, manifest)
    allowed = _expected_rank_paths(
        rank_root,
        manifest,
        identity_path,
        manifest_path,
    )
    _validate_rank_tree(rank_root, allowed)
    return manifest


def _read_validate_write_tensor(
    *,
    capture_root,
    temporary,
    record,
    relative_path,
    load_tensor,
):
    payload_path = capture_root / record.relative_path
    expected_path = (
        capture_root
        / f"rank{record.rank}"
        / "tensors"
        / record.workload_id
        / f"layer{record.layer_index}.pt"
    )
    if payload_path != expected_path:
        raise ValueError("tensor relative path rebinding")
    try:
        payload = payload_path.read_bytes()
    except OSError as error:
        raise ValueError("tensor payload is missing") from error
    _validate_tensor_bytes(payload, record, load_tensor)
    _write_payload_bytes(temporary / relative_path, payload)


def _validate_capture_rank_inventory(capture_root, world_size):
    expected = {f"rank{rank}" for rank in range(world_size)}
    actual = set()
    for path in capture_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ValueError("capture root contains untracked rank inventory")
        actual.add(path.name)
    if actual != expected:
        raise ValueError("capture rank inventory mismatch")


def _validate_cli_identity(
    identity,
    *,
    model_manifest_sha256,
    source_tree_sha256,
    workload_manifest_sha256,
    world_size,
):
    if not isinstance(identity, CaptureRunIdentity):
        raise ValueError("capture identity is invalid")
    if (
        identity.model_manifest_sha256 != model_manifest_sha256
        or identity.source_tree_sha256 != source_tree_sha256
        or identity.workload_manifest_sha256 != workload_manifest_sha256
        or identity.world_size != world_size
    ):
        raise ValueError("capture identity does not match assembler inputs")


def assemble_full_fidelity_bundle(
    *,
    capture_root,
    output_dir,
    model_manifest_sha256,
    source_tree_sha256,
    workload_manifest_sha256,
    world_size,
    load_tensor=torch.load,
):
    if not callable(load_tensor):
        raise ValueError("load_tensor must be callable")
    capture_root = _validate_root(capture_root, "capture_root")
    output_dir = _validate_root(output_dir, "output_dir")
    world_size = _positive_world_size(world_size)
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ValueError("capture_root must be a non-symlink directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output_dir must not pre-exist")
    try:
        output_dir.absolute().relative_to(capture_root.absolute())
    except ValueError:
        pass
    else:
        raise ValueError("output_dir must not be inside capture_root")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _validate_capture_rank_inventory(capture_root, world_size)

    rank_manifests = []
    identity = None
    for rank in range(world_size):
        manifest = _load_rank_manifest(
            capture_root,
            rank,
            expected_identity=identity,
        )
        if identity is None:
            identity = manifest.identity
            _validate_cli_identity(
                identity,
                model_manifest_sha256=model_manifest_sha256,
                source_tree_sha256=source_tree_sha256,
                workload_manifest_sha256=workload_manifest_sha256,
                world_size=world_size,
            )
        rank_manifests.append(manifest)

    temporary = output_dir.with_name(
        f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    tensor_rows = []
    output_paths = set()
    tensor_ids = set()
    try:
        temporary.mkdir()
        for rank_manifest in rank_manifests:
            for record in rank_manifest.tensors:
                if record.tensor_id in tensor_ids:
                    raise ValueError("tensor IDs must be unique")
                tensor_ids.add(record.tensor_id)
                relative_path = _source_relative_path(record)
                relative = relative_path.as_posix()
                if relative in output_paths:
                    raise ValueError("output relative paths must be unique")
                output_paths.add(relative)
                _read_validate_write_tensor(
                    capture_root=capture_root,
                    temporary=temporary,
                    record=record,
                    relative_path=relative_path,
                    load_tensor=load_tensor,
                )
                row = record.payload()
                row["relative_path"] = relative
                tensor_rows.append(row)

        manifest = {
            "schema_version": (
                calibration_contract.SOURCE_BUNDLE_SCHEMA_VERSION
            ),
            "model_manifest_sha256": identity.model_manifest_sha256,
            "source_tree_sha256": identity.source_tree_sha256,
            "workload_manifest_sha256": (
                identity.workload_manifest_sha256
            ),
            "world_size": identity.world_size,
            "linear_layer_indices": list(identity.linear_layer_indices),
            "workload_ids": list(identity.workload_ids),
            "tensors": tensor_rows,
        }
        calibration_contract.validate_source_bundle_manifest(manifest)
        manifest_path = temporary / "source_bundle_manifest.json"
        _write_payload_bytes(
            manifest_path,
            calibration_contract.canonical_json_bytes(manifest) + b"\n",
        )
        _publish_no_clobber(temporary, output_dir)
    finally:
        if temporary.exists() or temporary.is_symlink():
            shutil.rmtree(temporary)

    return {
        "output_dir": str(output_dir),
        "tensor_count": len(tensor_rows),
        "source_bundle_manifest_sha256": hashlib.sha256(
            (output_dir / "source_bundle_manifest.json").read_bytes()
        ).hexdigest(),
    }


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--workload-manifest-sha256", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    result = assemble_full_fidelity_bundle(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        model_manifest_sha256=args.model_manifest_sha256,
        source_tree_sha256=args.source_tree_sha256,
        workload_manifest_sha256=args.workload_manifest_sha256,
        world_size=args.world_size,
    )
    print(calibration_contract.canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
