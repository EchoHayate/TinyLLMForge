from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping


SCHEMA_VERSION = "qwen35.recurrent-int8-calibration.v1"
SOURCE_BUNDLE_SCHEMA_VERSION = (
    "qwen35.recurrent-full-fidelity-bundle.v1"
)
THRESHOLD_SCHEMA_VERSION = (
    "qwen35.recurrent-int8-calibration-thresholds.v1"
)
CODEC_ID = "qwen35_recurrent_symmetric_int8_per_row_v1"

TOP_LEVEL_ARTIFACTS = (
    "source_bundle_manifest.json",
    "thresholds.json",
    "commands.json",
    "calibration_rows.jsonl",
    "summary.json",
    "artifact_manifest.json",
    "independent_verification.json",
    "report.md",
)
NESTED_ARTIFACT_DIRECTORIES = (
    "source",
    "encoded_values",
    "scales",
    "decoded",
)
SOURCE_BUNDLE_FIELDS = (
    "schema_version",
    "model_manifest_sha256",
    "source_tree_sha256",
    "workload_manifest_sha256",
    "world_size",
    "linear_layer_indices",
    "workload_ids",
    "tensors",
)
SOURCE_TENSOR_FIELDS = (
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
THRESHOLD_FIELDS = (
    "schema_version",
    "codec",
    "pilot_source_bundle_sha256",
    "max_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "minimum_compression_ratio",
)
CALIBRATION_ROW_FIELDS = (
    "tensor_id",
    "rank",
    "workload_id",
    "layer_index",
    "source_path",
    "source_sha256",
    "source_shape",
    "source_dtype",
    "codec",
    "encoded_values_path",
    "encoded_values_sha256",
    "encoded_values_shape",
    "encoded_values_dtype",
    "scales_path",
    "scales_sha256",
    "scales_shape",
    "scales_dtype",
    "decoded_path",
    "decoded_sha256",
    "decoded_shape",
    "decoded_dtype",
    "logical_bytes",
    "payload_bytes",
    "scale_bytes",
    "encoded_bytes",
    "compression_ratio",
    "zero_row_count",
    "saturation_count",
    "max_abs_error",
    "mean_abs_error",
    "rmse",
    "relative_l2_error",
    "cosine_similarity",
    "encode_ns",
    "decode_ns",
    "finite_source",
    "finite_scales",
    "finite_decoded",
)


@dataclass(frozen=True)
class CalibrationThresholds:
    max_abs_error: float
    relative_l2_error: float
    cosine_similarity: float
    minimum_compression_ratio: float


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


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_expected_tensor_ids(
    *,
    world_size: int,
    workload_ids: tuple[str, ...],
    linear_layer_indices: tuple[int, ...],
) -> tuple[str, ...]:
    world_size = _positive_integer(world_size, "world_size")
    if (
        not isinstance(workload_ids, tuple)
        or not workload_ids
        or any(
            not isinstance(value, str) or not value
            for value in workload_ids
        )
        or len(set(workload_ids)) != len(workload_ids)
    ):
        raise ValueError(
            "workload_ids must be a non-empty unique string tuple"
        )
    if (
        not isinstance(linear_layer_indices, tuple)
        or not linear_layer_indices
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in linear_layer_indices
        )
        or len(set(linear_layer_indices)) != len(linear_layer_indices)
    ):
        raise ValueError(
            "linear_layer_indices must be a non-empty unique "
            "non-negative integer tuple"
        )
    return tuple(
        f"rank{rank}:{workload_id}:layer{layer_index}:linear_recurrent"
        for rank in range(world_size)
        for workload_id in workload_ids
        for layer_index in linear_layer_indices
    )


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


def _validate_sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _validate_shape(value, name, *, rank):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != rank
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in value
        )
    ):
        raise ValueError(
            f"{name} must contain exactly {rank} positive integers"
        )
    return tuple(value)


def _validate_relative_path(value, name, prefix):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path")
    normalized = path.as_posix()
    if normalized != value or not normalized.startswith(prefix + "/"):
        raise ValueError(f"{name} must be below {prefix}/")
    return normalized


def _non_negative_integer(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def validate_source_bundle_manifest(
    manifest: Mapping[str, object],
) -> tuple[str, ...]:
    _exact_fields(manifest, SOURCE_BUNDLE_FIELDS, "source bundle")
    if manifest["schema_version"] != SOURCE_BUNDLE_SCHEMA_VERSION:
        raise ValueError("source bundle schema version mismatch")
    for field_name in (
        "model_manifest_sha256",
        "source_tree_sha256",
        "workload_manifest_sha256",
    ):
        _validate_sha256(manifest[field_name], field_name)
    world_size = _positive_integer(manifest["world_size"], "world_size")
    layer_values = manifest["linear_layer_indices"]
    workload_values = manifest["workload_ids"]
    linear_layer_indices = tuple(layer_values)
    workload_ids = tuple(workload_values)
    expected = build_expected_tensor_ids(
        world_size=world_size,
        workload_ids=workload_ids,
        linear_layer_indices=linear_layer_indices,
    )
    tensors = manifest["tensors"]
    if not isinstance(tensors, list):
        raise ValueError("source bundle tensors must be a list")
    observed = []
    paths = set()
    ranks = set()
    for tensor in tensors:
        _exact_fields(tensor, SOURCE_TENSOR_FIELDS, "source tensor")
        tensor_id = tensor["tensor_id"]
        rank = _non_negative_integer(tensor["rank"], "tensor rank")
        if rank >= world_size:
            raise ValueError("tensor rank is outside world_size")
        ranks.add(rank)
        workload_id = tensor["workload_id"]
        if workload_id not in workload_ids:
            raise ValueError("tensor workload_id is not declared")
        layer_index = _non_negative_integer(
            tensor["layer_index"],
            "tensor layer_index",
        )
        if layer_index not in linear_layer_indices:
            raise ValueError("tensor layer_index is not declared")
        expected_id = (
            f"rank{rank}:{workload_id}:layer{layer_index}:"
            "linear_recurrent"
        )
        if tensor_id != expected_id:
            raise ValueError("source tensor identity mismatch")
        relative_path = _validate_relative_path(
            tensor["relative_path"],
            "tensor relative_path",
            "source",
        )
        if relative_path in paths:
            raise ValueError("source tensor paths must be unique")
        paths.add(relative_path)
        _validate_sha256(tensor["sha256"], "tensor sha256")
        shape = _validate_shape(tensor["shape"], "tensor shape", rank=3)
        if tensor["dtype"] != "float32":
            raise ValueError("source tensor dtype must be float32")
        if tensor["logical_bytes"] != math.prod(shape) * 4:
            raise ValueError("source tensor logical byte mismatch")
        observed.append(tensor_id)
    if ranks != set(range(world_size)):
        raise ValueError("source tensor ranks must be contiguous")
    if tuple(sorted(observed)) != tuple(sorted(expected)):
        raise ValueError("source tensor identity set mismatch")
    if len(observed) != len(set(observed)):
        raise ValueError("source tensor identities must be unique")
    return expected


def validate_thresholds(
    payload: Mapping[str, object],
) -> CalibrationThresholds:
    _exact_fields(payload, THRESHOLD_FIELDS, "thresholds")
    if payload["schema_version"] != THRESHOLD_SCHEMA_VERSION:
        raise ValueError("threshold schema version mismatch")
    if payload["codec"] != CODEC_ID:
        raise ValueError("threshold codec identity mismatch")
    _validate_sha256(
        payload["pilot_source_bundle_sha256"],
        "pilot_source_bundle_sha256",
    )
    values = {}
    for name in (
        "max_abs_error",
        "relative_l2_error",
        "cosine_similarity",
        "minimum_compression_ratio",
    ):
        value = payload[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
        values[name] = float(value)
    if values["max_abs_error"] <= 0:
        raise ValueError("max_abs_error must be positive")
    if values["relative_l2_error"] <= 0:
        raise ValueError("relative_l2_error must be positive")
    if not -1 <= values["cosine_similarity"] <= 1:
        raise ValueError("cosine_similarity must be within [-1, 1]")
    if values["minimum_compression_ratio"] <= 1:
        raise ValueError(
            "minimum_compression_ratio must be greater than one"
        )
    return CalibrationThresholds(**values)


def _row_reason(function):
    try:
        function()
    except ValueError as error:
        return str(error)
    return None


def validate_calibration_row(
    row: Mapping[str, object],
) -> tuple[str, ...]:
    try:
        _exact_fields(row, CALIBRATION_ROW_FIELDS, "calibration row")
    except ValueError as error:
        return (str(error),)
    reasons = []

    def check(function):
        reason = _row_reason(function)
        if reason is not None:
            reasons.append(reason)

    check(lambda: _non_negative_integer(row["rank"], "rank"))
    check(
        lambda: _non_negative_integer(
            row["layer_index"],
            "layer_index",
        )
    )
    expected_tensor_id = (
        f"rank{row['rank']}:{row['workload_id']}:"
        f"layer{row['layer_index']}:linear_recurrent"
    )
    if row["tensor_id"] != expected_tensor_id:
        reasons.append("tensor identity mismatch")
    if (
        not isinstance(row["workload_id"], str)
        or not row["workload_id"]
    ):
        reasons.append("workload_id must be a non-empty string")
    for field_name, prefix in (
        ("source_path", "source"),
        ("encoded_values_path", "encoded_values"),
        ("scales_path", "scales"),
        ("decoded_path", "decoded"),
    ):
        check(
            lambda field_name=field_name, prefix=prefix: (
                _validate_relative_path(
                    row[field_name],
                    field_name,
                    prefix,
                )
            )
        )
    for field_name in (
        "source_sha256",
        "encoded_values_sha256",
        "scales_sha256",
        "decoded_sha256",
    ):
        check(
            lambda field_name=field_name: _validate_sha256(
                row[field_name],
                field_name,
            )
        )
    shapes = {}
    for field_name, rank in (
        ("source_shape", 3),
        ("encoded_values_shape", 3),
        ("scales_shape", 2),
        ("decoded_shape", 3),
    ):
        try:
            shapes[field_name] = _validate_shape(
                row[field_name],
                field_name,
                rank=rank,
            )
        except ValueError as error:
            reasons.append(str(error))
    if row["codec"] != CODEC_ID:
        reasons.append("codec identity mismatch")
    for field_name, expected_dtype in (
        ("source_dtype", "float32"),
        ("encoded_values_dtype", "int8"),
        ("scales_dtype", "float32"),
        ("decoded_dtype", "float32"),
    ):
        if row[field_name] != expected_dtype:
            reasons.append(f"{field_name} mismatch")
    if (
        "source_shape" in shapes
        and "encoded_values_shape" in shapes
        and shapes["source_shape"] != shapes["encoded_values_shape"]
    ):
        reasons.append("encoded values shape mismatch")
    if (
        "source_shape" in shapes
        and "decoded_shape" in shapes
        and shapes["source_shape"] != shapes["decoded_shape"]
    ):
        reasons.append("decoded shape mismatch")
    if (
        "source_shape" in shapes
        and "scales_shape" in shapes
        and shapes["source_shape"][:-1] != shapes["scales_shape"]
    ):
        reasons.append("scales shape mismatch")
    integer_fields = (
        "logical_bytes",
        "payload_bytes",
        "scale_bytes",
        "encoded_bytes",
        "zero_row_count",
        "saturation_count",
        "encode_ns",
        "decode_ns",
    )
    for field_name in integer_fields:
        check(
            lambda field_name=field_name: _non_negative_integer(
                row[field_name],
                field_name,
            )
        )
    if "source_shape" in shapes:
        expected_logical = math.prod(shapes["source_shape"]) * 4
        expected_payload = math.prod(shapes["source_shape"])
        if row["logical_bytes"] != expected_logical:
            reasons.append("logical byte accounting mismatch")
        if row["payload_bytes"] != expected_payload:
            reasons.append("payload byte accounting mismatch")
        if row["zero_row_count"] > math.prod(
            shapes["source_shape"][:-1]
        ):
            reasons.append("zero row count exceeds row count")
        if row["saturation_count"] > math.prod(
            shapes["source_shape"]
        ):
            reasons.append("saturation count exceeds element count")
    if "scales_shape" in shapes:
        expected_scale = math.prod(shapes["scales_shape"]) * 4
        if row["scale_bytes"] != expected_scale:
            reasons.append("scale byte accounting mismatch")
    if (
        isinstance(row["payload_bytes"], int)
        and isinstance(row["scale_bytes"], int)
        and row["encoded_bytes"] != (
            row["payload_bytes"] + row["scale_bytes"]
        )
    ):
        reasons.append("encoded byte accounting mismatch")
    numeric_fields = (
        "compression_ratio",
        "max_abs_error",
        "mean_abs_error",
        "rmse",
        "relative_l2_error",
        "cosine_similarity",
    )
    for field_name in numeric_fields:
        value = row[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            reasons.append(f"{field_name} must be finite")
        elif field_name != "cosine_similarity" and value < 0:
            reasons.append(f"{field_name} must be non-negative")
    if (
        isinstance(row["cosine_similarity"], (int, float))
        and not isinstance(row["cosine_similarity"], bool)
        and math.isfinite(row["cosine_similarity"])
        and not -1 <= row["cosine_similarity"] <= 1
    ):
        reasons.append("cosine_similarity must be within [-1, 1]")
    if (
        isinstance(row["logical_bytes"], int)
        and isinstance(row["encoded_bytes"], int)
        and row["encoded_bytes"] > 0
        and isinstance(row["compression_ratio"], (int, float))
        and math.isfinite(row["compression_ratio"])
        and not math.isclose(
            row["compression_ratio"],
            row["logical_bytes"] / row["encoded_bytes"],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        reasons.append("compression ratio accounting mismatch")
    for field_name in (
        "finite_source",
        "finite_scales",
        "finite_decoded",
    ):
        if type(row[field_name]) is not bool:
            reasons.append(f"{field_name} must be a boolean")
    return tuple(reasons)


def classify_calibration(
    rows: tuple[Mapping[str, object], ...],
    *,
    expected_tensor_ids: tuple[str, ...],
    thresholds: CalibrationThresholds,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(rows, tuple):
        return "INVALID", ("rows must be a tuple",)
    try:
        expected = tuple(expected_tensor_ids)
        if (
            not expected
            or any(not isinstance(value, str) or not value for value in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError(
                "expected_tensor_ids must be non-empty and unique"
            )
    except (TypeError, ValueError) as error:
        return "INVALID", (str(error),)
    if type(thresholds) is not CalibrationThresholds:
        return "INVALID", (
            "thresholds must be an exact CalibrationThresholds",
        )

    invalid_reasons = []
    observed_ids = []
    for row_index, row in enumerate(rows):
        reasons = validate_calibration_row(row)
        invalid_reasons.extend(
            f"row {row_index}: {reason}" for reason in reasons
        )
        if isinstance(row, Mapping):
            tensor_id = row.get("tensor_id")
            if isinstance(tensor_id, str):
                observed_ids.append(tensor_id)
            for field_name in (
                "finite_source",
                "finite_scales",
                "finite_decoded",
            ):
                if row.get(field_name) is False:
                    invalid_reasons.append(
                        f"{field_name} must be true"
                    )
    if len(observed_ids) != len(set(observed_ids)):
        invalid_reasons.append(
            "calibration tensor identities must be unique"
        )
    if set(observed_ids) != set(expected) or len(rows) != len(expected):
        invalid_reasons.append(
            "calibration tensor identity set mismatch"
        )
    if invalid_reasons:
        return "INVALID", tuple(sorted(set(invalid_reasons)))

    no_go_reasons = []
    for row in rows:
        tensor_id = row["tensor_id"]
        if row["max_abs_error"] > thresholds.max_abs_error:
            no_go_reasons.append(
                f"{tensor_id}: max_abs_error exceeds threshold"
            )
        if (
            row["relative_l2_error"]
            > thresholds.relative_l2_error
        ):
            no_go_reasons.append(
                f"{tensor_id}: relative_l2_error exceeds threshold"
            )
        if (
            row["cosine_similarity"]
            < thresholds.cosine_similarity
        ):
            no_go_reasons.append(
                f"{tensor_id}: cosine_similarity below threshold"
            )
        if (
            row["compression_ratio"]
            < thresholds.minimum_compression_ratio
        ):
            no_go_reasons.append(
                f"{tensor_id}: compression_ratio below threshold"
            )
    if no_go_reasons:
        return "NO_GO", tuple(no_go_reasons)
    return "PASS", ()
