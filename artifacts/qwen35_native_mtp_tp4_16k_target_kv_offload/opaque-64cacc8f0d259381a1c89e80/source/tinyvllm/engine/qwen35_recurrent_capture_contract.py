from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
from typing import Mapping


CAPTURE_IDENTITY_SCHEMA_VERSION = (
    "qwen35.recurrent-capture-identity.v1"
)
RANK_CAPTURE_MANIFEST_SCHEMA_VERSION = (
    "qwen35.recurrent-rank-capture.v1"
)

_RUN_IDENTITY_FIELDS = (
    "schema_version",
    "model_manifest_sha256",
    "source_tree_sha256",
    "workload_manifest_sha256",
    "world_size",
    "workload_ids",
    "linear_layer_indices",
)
_TENSOR_RECORD_FIELDS = (
    "tensor_id",
    "rank",
    "workload_id",
    "layer_index",
    "relative_path",
    "sha256",
    "shape",
    "dtype",
    "logical_bytes",
)
_RANK_MANIFEST_FIELDS = (
    "schema_version",
    "identity",
    "rank",
    "tensors",
)


@dataclass(frozen=True)
class CaptureRunIdentity:
    model_manifest_sha256: str
    source_tree_sha256: str
    workload_manifest_sha256: str
    world_size: int
    workload_ids: tuple[str, ...]
    linear_layer_indices: tuple[int, ...]

    def payload(self) -> dict:
        return {
            "schema_version": CAPTURE_IDENTITY_SCHEMA_VERSION,
            "model_manifest_sha256": self.model_manifest_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "workload_manifest_sha256": self.workload_manifest_sha256,
            "world_size": self.world_size,
            "workload_ids": list(self.workload_ids),
            "linear_layer_indices": list(self.linear_layer_indices),
        }


@dataclass(frozen=True)
class CapturedTensorRecord:
    tensor_id: str
    rank: int
    workload_id: str
    layer_index: int
    relative_path: str
    sha256: str
    shape: tuple[int, int, int]
    dtype: str
    logical_bytes: int

    def payload(self) -> dict:
        return {
            "tensor_id": self.tensor_id,
            "rank": self.rank,
            "workload_id": self.workload_id,
            "layer_index": self.layer_index,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "logical_bytes": self.logical_bytes,
        }


@dataclass(frozen=True)
class RankCaptureManifest:
    identity: CaptureRunIdentity
    rank: int
    tensors: tuple[CapturedTensorRecord, ...]

    def payload(self) -> dict:
        return {
            "schema_version": RANK_CAPTURE_MANIFEST_SCHEMA_VERSION,
            "identity": self.identity.payload(),
            "rank": self.rank,
            "tensors": [tensor.payload() for tensor in self.tensors],
        }


def canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _exact_fields(value, fields, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        raise ValueError(
            f"{name} fields mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_sorted_unique_strings(value, name):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            not isinstance(item, str) or not item
            for item in value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty string sequence"
        )
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _validate_sorted_unique_indices(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"{name} must be a non-empty integer sequence"
        )
    result = tuple(
        _non_negative_integer(item, name)
        for item in value
    )
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _validate_rank(value, identity, name="rank"):
    rank = _non_negative_integer(value, name)
    if rank >= identity.world_size:
        raise ValueError(f"{name} must be below world_size")
    return rank


def _validate_relative_path(value, *, rank):
    if not isinstance(value, str) or not value:
        raise ValueError("relative_path must be a non-empty string")
    if "\\" in value:
        raise ValueError("relative_path must use POSIX separators")
    components = value.split("/")
    if any(component in ("", ".", "..") for component in components):
        raise ValueError("relative_path must be normalized")
    path = PurePosixPath(value)
    if path.is_absolute() or path.parts[0] != f"rank{rank}":
        raise ValueError(
            f"relative_path must be below rank{rank}/"
        )
    if path.as_posix() != value:
        raise ValueError("relative_path must be normalized")
    return value


def validate_run_identity(value) -> CaptureRunIdentity:
    _exact_fields(value, _RUN_IDENTITY_FIELDS, "run identity")
    if value["schema_version"] != CAPTURE_IDENTITY_SCHEMA_VERSION:
        raise ValueError("run identity schema version mismatch")
    return CaptureRunIdentity(
        model_manifest_sha256=_validate_sha256(
            value["model_manifest_sha256"],
            "model_manifest_sha256",
        ),
        source_tree_sha256=_validate_sha256(
            value["source_tree_sha256"],
            "source_tree_sha256",
        ),
        workload_manifest_sha256=_validate_sha256(
            value["workload_manifest_sha256"],
            "workload_manifest_sha256",
        ),
        world_size=_positive_integer(
            value["world_size"],
            "world_size",
        ),
        workload_ids=_validate_sorted_unique_strings(
            value["workload_ids"],
            "workload_ids",
        ),
        linear_layer_indices=_validate_sorted_unique_indices(
            value["linear_layer_indices"],
            "linear_layer_indices",
        ),
    )


def expected_tensor_ids(
    *,
    world_size,
    workload_ids,
    linear_layer_indices,
) -> tuple[str, ...]:
    world_size = _positive_integer(world_size, "world_size")
    workload_ids = _validate_sorted_unique_strings(
        workload_ids,
        "workload_ids",
    )
    linear_layer_indices = _validate_sorted_unique_indices(
        linear_layer_indices,
        "linear_layer_indices",
    )
    return tuple(
        f"rank{rank}:{workload_id}:layer{layer_index}:linear_recurrent"
        for rank in range(world_size)
        for workload_id in workload_ids
        for layer_index in linear_layer_indices
    )


def validate_tensor_record(
    value,
    *,
    identity,
    expected_rank=None,
) -> CapturedTensorRecord:
    if not isinstance(identity, CaptureRunIdentity):
        raise ValueError("identity must be a CaptureRunIdentity")
    _exact_fields(value, _TENSOR_RECORD_FIELDS, "tensor record")
    rank = _validate_rank(value["rank"], identity)
    if expected_rank is not None:
        expected_rank = _validate_rank(
            expected_rank,
            identity,
            "expected_rank",
        )
        if rank != expected_rank:
            raise ValueError("tensor rank mismatch")
    workload_id = value["workload_id"]
    if workload_id not in identity.workload_ids:
        raise ValueError("tensor workload_id is not declared")
    layer_index = _non_negative_integer(
        value["layer_index"],
        "layer_index",
    )
    if layer_index not in identity.linear_layer_indices:
        raise ValueError("tensor layer_index is not declared")
    expected_tensor_id = (
        f"rank{rank}:{workload_id}:layer{layer_index}:"
        "linear_recurrent"
    )
    if value["tensor_id"] != expected_tensor_id:
        raise ValueError("tensor_id does not match tensor coordinates")
    shape = value["shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise ValueError("shape must contain three positive integers")
    shape = tuple(shape)
    if value["dtype"] != "float32":
        raise ValueError("dtype must be float32")
    logical_bytes = _positive_integer(
        value["logical_bytes"],
        "logical_bytes",
    )
    expected_logical_bytes = 4
    for dimension in shape:
        expected_logical_bytes *= dimension
    if logical_bytes != expected_logical_bytes:
        raise ValueError("logical_bytes does not match shape and dtype")
    return CapturedTensorRecord(
        tensor_id=expected_tensor_id,
        rank=rank,
        workload_id=workload_id,
        layer_index=layer_index,
        relative_path=_validate_relative_path(
            value["relative_path"],
            rank=rank,
        ),
        sha256=_validate_sha256(value["sha256"], "sha256"),
        shape=shape,
        dtype="float32",
        logical_bytes=logical_bytes,
    )


def validate_rank_manifest(
    value,
    *,
    expected_identity=None,
) -> RankCaptureManifest:
    _exact_fields(value, _RANK_MANIFEST_FIELDS, "rank manifest")
    if value["schema_version"] != RANK_CAPTURE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("rank manifest schema version mismatch")
    identity = validate_run_identity(value["identity"])
    if expected_identity is not None:
        if not isinstance(expected_identity, CaptureRunIdentity):
            raise ValueError(
                "expected_identity must be a CaptureRunIdentity"
            )
        if identity != expected_identity:
            raise ValueError("rank manifest identity mismatch")
    rank = _validate_rank(value["rank"], identity)
    tensors_value = value["tensors"]
    if not isinstance(tensors_value, (list, tuple)):
        raise ValueError("rank manifest tensors must be a sequence")
    tensors = tuple(
        validate_tensor_record(
            tensor,
            identity=identity,
            expected_rank=rank,
        )
        for tensor in tensors_value
    )
    expected_ids = tuple(
        tensor_id
        for tensor_id in expected_tensor_ids(
            world_size=identity.world_size,
            workload_ids=identity.workload_ids,
            linear_layer_indices=identity.linear_layer_indices,
        )
        if tensor_id.startswith(f"rank{rank}:")
    )
    if tuple(tensor.tensor_id for tensor in tensors) != expected_ids:
        raise ValueError(
            "rank manifest tensors must exactly match canonical inventory"
        )
    relative_paths = tuple(tensor.relative_path for tensor in tensors)
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("rank manifest tensor paths must be unique")
    return RankCaptureManifest(
        identity=identity,
        rank=rank,
        tensors=tensors,
    )
