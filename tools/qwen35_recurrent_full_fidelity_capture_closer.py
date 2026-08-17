from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import types

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


from tinyvllm.engine.qwen35_recurrent_capture import (
    _is_generated_temporary_path,
    _write_atomic_bytes,
)
from tinyvllm.engine.qwen35_recurrent_capture_contract import (
    RankCaptureManifest,
    canonical_json_bytes,
    validate_rank_manifest,
    validate_run_identity,
    validate_tensor_record,
)


def _load_json_bytes(path, name):
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}") from error
    return payload, value


def _rank_from_root(rank_root):
    name = rank_root.name
    if not name.startswith("rank") or not name[4:].isdigit():
        raise ValueError("staging_dir must be a rank root")
    return int(name[4:])


def _expected_workloads(value):
    result = tuple(value)
    if (
        not result
        or any(not isinstance(item, str) or not item for item in result)
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            "expected workload IDs must be sorted non-empty strings"
        )
    return result


def _expected_layers(value):
    result = tuple(value)
    if (
        not result
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            for item in result
        )
        or result != tuple(sorted(set(result)))
    ):
        raise ValueError(
            "expected linear layer indices must be sorted non-negative integers"
        )
    return result


def close_rank_capture(
    *,
    staging_dir,
    expected_workload_ids,
    expected_linear_layer_indices,
    load_tensor=torch.load,
):
    rank_root = Path(staging_dir)
    if rank_root.is_symlink() or not rank_root.is_dir():
        raise ValueError("rank root must be a non-symlink directory")
    rank = _rank_from_root(rank_root)
    manifest_path = rank_root / "rank_capture_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("rank manifest already exists")

    for path in rank_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink below rank root")

    identity_path = rank_root / "capture_identity.json"
    identity_bytes, identity_payload = _load_json_bytes(
        identity_path,
        "capture identity",
    )
    identity = validate_run_identity(identity_payload)
    if identity_bytes != canonical_json_bytes(identity.payload()) + b"\n":
        raise ValueError("capture identity bytes are not canonical")
    if rank >= identity.world_size:
        raise ValueError("rank is outside capture identity")

    expected_workload_ids = _expected_workloads(expected_workload_ids)
    expected_linear_layer_indices = _expected_layers(
        expected_linear_layer_indices
    )
    if (
        expected_workload_ids != identity.workload_ids
        or expected_linear_layer_indices
        != identity.linear_layer_indices
    ):
        raise ValueError("expected inventory does not match capture identity")

    receipt_root = rank_root / "workloads"
    if not receipt_root.is_dir():
        raise ValueError("workload receipt root is missing")
    receipt_paths = tuple(sorted(receipt_root.glob("*.complete.json")))
    expected_receipt_paths = tuple(
        receipt_root / f"{workload_id}.complete.json"
        for workload_id in expected_workload_ids
    )
    if receipt_paths != expected_receipt_paths:
        raise ValueError("workload receipt inventory mismatch")

    tensor_root = rank_root / "tensors"
    if not tensor_root.is_dir():
        raise ValueError("tensor root is missing")
    records = []
    allowed_paths = {
        rank_root,
        identity_path,
        receipt_root,
        tensor_root,
    }
    for workload_id, receipt_path in zip(
        expected_workload_ids,
        receipt_paths,
        strict=True,
    ):
        _, receipt = _load_json_bytes(receipt_path, "workload receipt")
        if set(receipt) != {"rank", "workload_id", "tensors"}:
            raise ValueError("invalid workload receipt")
        if receipt["rank"] != rank or receipt["workload_id"] != workload_id:
            raise ValueError("workload receipt coordinates mismatch")
        if not isinstance(receipt["tensors"], list):
            raise ValueError("invalid workload receipt tensors")
        if tuple(
            row.get("layer_index") for row in receipt["tensors"]
        ) != expected_linear_layer_indices:
            raise ValueError("workload receipt tensor inventory mismatch")

        allowed_paths.add(receipt_path)
        workload_root = tensor_root / workload_id
        if not workload_root.is_dir():
            raise ValueError("workload tensor directory is missing")
        allowed_paths.add(workload_root)
        for row in receipt["tensors"]:
            record = validate_tensor_record(
                row,
                identity=identity,
                expected_rank=rank,
            )
            if record.workload_id != workload_id:
                raise ValueError("workload receipt tensor mismatch")
            payload_path = rank_root.parent / record.relative_path
            expected_path = (
                workload_root / f"layer{record.layer_index}.pt"
            )
            if payload_path != expected_path:
                raise ValueError("tensor relative path mismatch")
            try:
                payload = payload_path.read_bytes()
            except OSError as error:
                raise ValueError("tensor payload is missing") from error
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            if observed_sha256 != record.sha256:
                raise ValueError("tensor payload hash mismatch")
            try:
                tensor = load_tensor(
                    io.BytesIO(payload),
                    map_location="cpu",
                )
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
            records.append(record)
            allowed_paths.add(payload_path)
            del tensor

    for path in rank_root.rglob("*"):
        if path not in allowed_paths:
            if _is_generated_temporary_path(path):
                raise ValueError("temporary path remains in rank root")
            raise ValueError("untracked path in rank root")

    manifest = validate_rank_manifest({
        "schema_version": (
            "qwen35.recurrent-rank-capture.v1"
        ),
        "identity": identity.payload(),
        "rank": rank,
        "tensors": [record.payload() for record in records],
    })
    _write_atomic_bytes(
        manifest_path,
        canonical_json_bytes(manifest.payload()) + b"\n",
    )
    return manifest


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--expected-workload-id",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--expected-linear-layer-index",
        action="append",
        type=int,
        required=True,
    )
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    manifest = close_rank_capture(
        staging_dir=Path(args.capture_root) / f"rank{args.rank}",
        expected_workload_ids=tuple(args.expected_workload_id),
        expected_linear_layer_indices=tuple(
            args.expected_linear_layer_index
        ),
    )
    print(canonical_json_bytes(manifest.payload()).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
