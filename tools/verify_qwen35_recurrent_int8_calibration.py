from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch

from qwen35_recurrent_int8_calibration_contract import (
    CODEC_ID,
    SCHEMA_VERSION,
    canonical_json_bytes,
    classify_calibration,
    sha256_file,
    validate_calibration_row,
    validate_source_bundle_manifest,
    validate_thresholds,
)


_BASE_TOP_LEVEL = {
    "source_bundle_manifest.json",
    "thresholds.json",
    "commands.json",
    "calibration_rows.jsonl",
    "summary.json",
    "artifact_manifest.json",
}
_NESTED_DIRECTORIES = {
    "source",
    "encoded_values",
    "scales",
    "decoded",
}
_ARTIFACT_MANIFEST_FIELDS = {
    "schema_version",
    "artifacts",
}
_ARTIFACT_FIELDS = {
    "path",
    "size",
    "sha256",
}


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path, value):
    _atomic_write(path, canonical_json_bytes(value) + b"\n")


def _regular_inventory(run_dir):
    top_level = set()
    files = set()
    directories = set()
    for path in run_dir.rglob("*"):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_symlink():
            raise ValueError(f"verification inventory contains symlink: {relative}")
        if path.is_file():
            files.add(relative)
            if len(path.relative_to(run_dir).parts) == 1:
                top_level.add(relative)
        elif path.is_dir():
            directories.add(relative)
        else:
            raise ValueError(
                f"verification inventory contains non-regular entry: {relative}"
            )
    if top_level != _BASE_TOP_LEVEL:
        raise ValueError(
            "verification top-level inventory mismatch: "
            f"missing={sorted(_BASE_TOP_LEVEL - top_level)}, "
            f"extra={sorted(top_level - _BASE_TOP_LEVEL)}"
        )
    for relative in files:
        parts = Path(relative).parts
        if len(parts) == 1:
            continue
        if parts[0] not in _NESTED_DIRECTORIES:
            raise ValueError(
                f"verification nested inventory path is forbidden: {relative}"
            )
    expected_directories = set()
    for relative in files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    if directories != expected_directories:
        raise ValueError(
            "verification directory inventory mismatch: "
            f"missing={sorted(expected_directories - directories)}, "
            f"extra={sorted(directories - expected_directories)}"
        )
    return files


def _validate_artifact_manifest(run_dir, files):
    payload = _load_json(run_dir / "artifact_manifest.json")
    if set(payload) != _ARTIFACT_MANIFEST_FIELDS:
        raise ValueError("artifact manifest fields mismatch")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("artifact manifest schema mismatch")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("artifact manifest artifacts must be a list")
    observed = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError("artifact manifest row fields mismatch")
        relative = artifact["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
        ):
            raise ValueError("artifact manifest path is invalid")
        if relative in observed:
            raise ValueError("artifact manifest paths must be unique")
        observed.add(relative)
        path = run_dir / relative
        if relative == "artifact_manifest.json" or relative not in files:
            raise ValueError("artifact manifest coverage mismatch")
        size = artifact["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("artifact manifest size is invalid")
        if path.stat().st_size != size:
            raise ValueError(f"artifact size mismatch: {relative}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"artifact hash mismatch: {relative}")
    expected = files - {"artifact_manifest.json"}
    if observed != expected:
        raise ValueError(
            "artifact manifest coverage mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    return payload


def _load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(
                    f"calibration row line is malformed: {line_number}"
                )
            rows.append(json.loads(line))
    return rows


def _load_tensor(path):
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"artifact is not a tensor: {path.name}")
    return tensor.detach().cpu().contiguous()


def _expect_tensor(tensor, *, dtype, shape, name, finite=True):
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype mismatch")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} shape mismatch")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if finite and not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _independent_metrics(source, decoded):
    source64 = source.to(dtype=torch.float64)
    decoded64 = decoded.to(dtype=torch.float64)
    difference = decoded64 - source64
    absolute = difference.abs()
    source_norm = torch.linalg.vector_norm(source64).item()
    decoded_norm = torch.linalg.vector_norm(decoded64).item()
    difference_norm = torch.linalg.vector_norm(difference).item()
    if source_norm == 0.0:
        if decoded_norm != 0.0:
            raise ValueError("decoded must be zero for zero-norm source")
        relative_l2_error = 0.0
        cosine_similarity = 1.0
    else:
        relative_l2_error = difference_norm / source_norm
        if decoded_norm == 0.0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = torch.dot(
                source64.reshape(-1),
                decoded64.reshape(-1),
            ).item() / (source_norm * decoded_norm)
    return {
        "max_abs_error": absolute.max().item(),
        "mean_abs_error": absolute.mean().item(),
        "rmse": torch.sqrt(torch.mean(difference.square())).item(),
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": max(-1.0, min(1.0, cosine_similarity)),
    }


def _same_float(recorded, observed, *, compression=False):
    if (
        isinstance(recorded, bool)
        or not isinstance(recorded, (int, float))
        or not math.isfinite(recorded)
    ):
        return False
    return math.isclose(
        float(recorded),
        float(observed),
        rel_tol=1e-12 if compression else 1e-9,
        abs_tol=1e-12,
    )


def _verify_row(run_dir, row, source_entry):
    reasons = validate_calibration_row(row)
    if reasons:
        raise ValueError("invalid calibration row: " + "; ".join(reasons))
    if row["tensor_id"] != source_entry["tensor_id"]:
        raise ValueError("row/source tensor identity mismatch")
    for field_name in ("rank", "workload_id", "layer_index"):
        if row[field_name] != source_entry[field_name]:
            raise ValueError(f"row/source {field_name} mismatch")
    expected_source_path = (
        f"source/rank{row['rank']}/{row['workload_id']}/"
        f"layer{row['layer_index']}.pt"
    )
    if row["source_path"] != expected_source_path:
        raise ValueError("source artifact path mismatch")
    suffix = (
        f"rank{row['rank']}/{row['workload_id']}/"
        f"layer{row['layer_index']}.pt"
    )
    for field_name, prefix in (
        ("encoded_values_path", "encoded_values"),
        ("scales_path", "scales"),
        ("decoded_path", "decoded"),
    ):
        if row[field_name] != f"{prefix}/{suffix}":
            raise ValueError(f"{field_name} mismatch")
    paths = {
        name: run_dir / row[name]
        for name in (
            "source_path",
            "encoded_values_path",
            "scales_path",
            "decoded_path",
        )
    }
    for path_field, hash_field in (
        ("source_path", "source_sha256"),
        ("encoded_values_path", "encoded_values_sha256"),
        ("scales_path", "scales_sha256"),
        ("decoded_path", "decoded_sha256"),
    ):
        if sha256_file(paths[path_field]) != row[hash_field]:
            raise ValueError(f"{hash_field} mismatch")
    if row["source_sha256"] != source_entry["sha256"]:
        raise ValueError("source bundle tensor hash mismatch")
    source = _load_tensor(paths["source_path"])
    values = _load_tensor(paths["encoded_values_path"])
    scales = _load_tensor(paths["scales_path"])
    decoded = _load_tensor(paths["decoded_path"])
    source_shape = tuple(source_entry["shape"])
    _expect_tensor(
        source,
        dtype=torch.float32,
        shape=source_shape,
        name="source",
    )
    _expect_tensor(
        values,
        dtype=torch.int8,
        shape=source_shape,
        name="encoded values",
        finite=False,
    )
    _expect_tensor(
        scales,
        dtype=torch.float32,
        shape=source_shape[:-1],
        name="scales",
    )
    _expect_tensor(
        decoded,
        dtype=torch.float32,
        shape=source_shape,
        name="decoded",
    )
    if torch.any(values == -128).item():
        raise ValueError("encoded values contain forbidden -128")
    if not torch.all(scales > 0).item():
        raise ValueError("scales must be positive")
    source_amax = source.abs().amax(dim=-1)
    expected_scales = torch.where(
        source_amax == 0,
        torch.ones_like(source_amax, dtype=torch.float32),
        source_amax / 127.0,
    ).contiguous()
    if not torch.equal(scales, expected_scales):
        raise ValueError("scales do not match canonical per-row encoding")
    expected_values = torch.round(
        source / expected_scales.unsqueeze(-1)
    ).clamp(-127, 127).to(torch.int8).contiguous()
    if not torch.equal(values, expected_values):
        raise ValueError(
            "encoded values do not match canonical per-row encoding"
        )
    independently_decoded = (
        values.to(dtype=torch.float32) * scales.unsqueeze(-1)
    ).contiguous()
    if not torch.equal(decoded, independently_decoded):
        raise ValueError("saved decoded tensor mismatch")
    logical_bytes = source.numel() * source.element_size()
    payload_bytes = values.numel() * values.element_size()
    scale_bytes = scales.numel() * scales.element_size()
    encoded_bytes = payload_bytes + scale_bytes
    compression_ratio = logical_bytes / encoded_bytes
    exact_values = {
        "logical_bytes": logical_bytes,
        "payload_bytes": payload_bytes,
        "scale_bytes": scale_bytes,
        "encoded_bytes": encoded_bytes,
        "zero_row_count": int(
            torch.all(source == 0, dim=-1).sum().item()
        ),
        "saturation_count": int(
            torch.logical_or(values == -127, values == 127).sum().item()
        ),
        "finite_source": True,
        "finite_scales": True,
        "finite_decoded": True,
    }
    for field_name, observed in exact_values.items():
        if row[field_name] != observed:
            raise ValueError(f"{field_name} mismatch")
    if not _same_float(
        row["compression_ratio"],
        compression_ratio,
        compression=True,
    ):
        raise ValueError("compression_ratio mismatch")
    metrics = _independent_metrics(source, decoded)
    for field_name, observed in metrics.items():
        if not _same_float(row[field_name], observed):
            raise ValueError(f"{field_name} mismatch")
    for field_name in ("encode_ns", "decode_ns"):
        value = row[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be non-negative")
    return dict(row)


def _render_report(result, manifest, thresholds_sha256):
    reasons = result["reasons"] or ["none"]
    ranks = ", ".join(str(value) for value in result["ranks"])
    workloads = ", ".join(result["workload_ids"])
    layers = ", ".join(
        str(value) for value in result["linear_layer_indices"]
    )
    return "\n".join([
        "# Qwen3.5 Recurrent INT8 Independent Verification",
        "",
        f"- Independent Classification: **{result['classification']}**",
        f"- Reasons: {'; '.join(reasons)}",
        f"- Source bundle manifest SHA-256: `{result['source_bundle_sha256']}`",
        f"- model manifest SHA-256: `{manifest['model_manifest_sha256']}`",
        f"- source tree SHA-256: `{manifest['source_tree_sha256']}`",
        f"- workload manifest SHA-256: `{manifest['workload_manifest_sha256']}`",
        f"- threshold SHA-256: `{thresholds_sha256}`",
        f"- codec SHA-256: `{result['codec_sha256']}`",
        f"- Tensor count: {result['tensor_count']}",
        f"- Rank coverage: {ranks}",
        f"- Workload coverage: {workloads}",
        f"- Layer coverage: {layers}",
        f"- Logical bytes: {result['logical_bytes']}",
        f"- INT8 payload bytes: {result['payload_bytes']}",
        f"- FP32 scale bytes: {result['scale_bytes']}",
        f"- Encoded bytes: {result['encoded_bytes']}",
        f"- Compression ratio: {result['compression_ratio']:.12f}x",
        f"- Worst max-absolute tensor: `{result['worst_max_abs_tensor_id']}` "
        f"({result['max_abs_error']:.12g})",
        f"- Worst relative-L2 tensor: `{result['worst_relative_l2_tensor_id']}` "
        f"({result['relative_l2_error']:.12g})",
        f"- Minimum cosine similarity: {result['minimum_cosine_similarity']:.12g}",
        f"- Zero rows: {result['zero_row_count']}",
        f"- Saturations: {result['saturation_count']}",
        "",
        "## Producer-observed Timing",
        "",
        f"- Encode total ns: {result['producer_observed_encode_ns']}",
        f"- Decode total ns: {result['producer_observed_decode_ns']}",
        "",
        "## Claim Boundary",
        "",
        "No runtime integration, GPU-memory reduction, speed improvement, "
        "or quality authority is established by this offline verification.",
        "",
    ])


def verify_calibration(
    run_dir: Path,
) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError("run directory does not exist")
    if (
        (run_dir / "independent_verification.json").exists()
        or (run_dir / "report.md").exists()
    ):
        raise ValueError("public verification is single-use")
    files = _regular_inventory(run_dir)
    _validate_artifact_manifest(run_dir, files)
    manifest_path = run_dir / "source_bundle_manifest.json"
    thresholds_path = run_dir / "thresholds.json"
    manifest = _load_json(manifest_path)
    expected_tensor_ids = validate_source_bundle_manifest(manifest)
    thresholds_payload = _load_json(thresholds_path)
    thresholds = validate_thresholds(thresholds_payload)
    if (
        thresholds_payload["pilot_source_bundle_sha256"]
        != sha256_file(manifest_path)
    ):
        raise ValueError("threshold source bundle manifest binding mismatch")
    producer_summary = _load_json(run_dir / "summary.json")
    if (
        producer_summary.get("source_bundle_sha256")
        != sha256_file(manifest_path)
    ):
        raise ValueError("producer source bundle binding mismatch")
    if (
        producer_summary.get("thresholds_sha256")
        != sha256_file(thresholds_path)
    ):
        raise ValueError("producer threshold binding mismatch")
    raw_rows = _load_rows(run_dir / "calibration_rows.jsonl")
    observed_ids = [row.get("tensor_id") for row in raw_rows]
    if observed_ids != list(expected_tensor_ids):
        raise ValueError("calibration row order or identity mismatch")
    source_entries = {
        row["tensor_id"]: row for row in manifest["tensors"]
    }
    rows = tuple(
        _verify_row(run_dir, row, source_entries[row["tensor_id"]])
        for row in raw_rows
    )
    classification, reasons = classify_calibration(
        rows,
        expected_tensor_ids=expected_tensor_ids,
        thresholds=thresholds,
    )
    if classification == "INVALID":
        raise ValueError(
            "independent classification invalid: " + "; ".join(reasons)
        )
    logical_bytes = sum(row["logical_bytes"] for row in rows)
    payload_bytes = sum(row["payload_bytes"] for row in rows)
    scale_bytes = sum(row["scale_bytes"] for row in rows)
    encoded_bytes = sum(row["encoded_bytes"] for row in rows)
    worst_max_abs = max(rows, key=lambda row: row["max_abs_error"])
    worst_relative_l2 = max(
        rows,
        key=lambda row: row["relative_l2_error"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "independent": True,
        "classification": classification,
        "reasons": list(reasons),
        "source_bundle_sha256": sha256_file(manifest_path),
        "thresholds_sha256": sha256_file(thresholds_path),
        "codec": CODEC_ID,
        "codec_sha256": __import__("hashlib").sha256(
            CODEC_ID.encode("utf-8")
        ).hexdigest(),
        "tensor_count": len(rows),
        "ranks": list(range(manifest["world_size"])),
        "workload_ids": list(manifest["workload_ids"]),
        "linear_layer_indices": list(manifest["linear_layer_indices"]),
        "logical_bytes": logical_bytes,
        "payload_bytes": payload_bytes,
        "scale_bytes": scale_bytes,
        "encoded_bytes": encoded_bytes,
        "compression_ratio": logical_bytes / encoded_bytes,
        "max_abs_error": worst_max_abs["max_abs_error"],
        "worst_max_abs_tensor_id": worst_max_abs["tensor_id"],
        "relative_l2_error": worst_relative_l2["relative_l2_error"],
        "worst_relative_l2_tensor_id": worst_relative_l2["tensor_id"],
        "minimum_cosine_similarity": min(
            row["cosine_similarity"] for row in rows
        ),
        "zero_row_count": sum(row["zero_row_count"] for row in rows),
        "saturation_count": sum(
            row["saturation_count"] for row in rows
        ),
        "producer_observed_encode_ns": sum(
            row["encode_ns"] for row in rows
        ),
        "producer_observed_decode_ns": sum(
            row["decode_ns"] for row in rows
        ),
    }
    report = _render_report(
        result,
        manifest,
        result["thresholds_sha256"],
    )
    try:
        _atomic_write_json(
            run_dir / "independent_verification.json",
            result,
        )
        _atomic_write(
            run_dir / "report.md",
            report.encode("utf-8"),
        )
    except BaseException:
        for name in ("independent_verification.json", "report.md"):
            path = run_dir / name
            if path.exists():
                path.unlink()
        raise
    return result
