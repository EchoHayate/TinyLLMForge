from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Callable

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
from tinyvllm.engine.qwen35_recurrent_int8_codec import (
    decode_qwen35_recurrent_int8_per_row,
    encode_qwen35_recurrent_int8_per_row,
    qwen35_recurrent_int8_error_metrics,
)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path, value):
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _atomic_write_jsonl(path, rows):
    payload = b"".join(
        canonical_json_bytes(row) + b"\n" for row in rows
    )
    _atomic_write_bytes(path, payload)


def _atomic_save_tensor(path, tensor, save_tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_tensor(tensor, temporary)
    temporary.replace(path)


def _ensure_regular_tree(root, allowed_files):
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"source bundle symlink is forbidden: {relative}")
        if path.is_file():
            actual.add(relative)
        elif not path.is_dir():
            raise ValueError(
                f"source bundle entry is not regular: {relative}"
            )
    if actual != allowed_files:
        raise ValueError(
            "source bundle file inventory mismatch: "
            f"missing={sorted(allowed_files - actual)}, "
            f"extra={sorted(actual - allowed_files)}"
        )


def _preflight(source_bundle_dir, output_dir, thresholds_path):
    source_bundle_dir = Path(source_bundle_dir).resolve()
    output_dir = Path(output_dir).resolve()
    thresholds_path = Path(thresholds_path).resolve()
    if not source_bundle_dir.is_dir():
        raise ValueError("source bundle directory does not exist")
    if not thresholds_path.is_file() or thresholds_path.is_symlink():
        raise ValueError("thresholds path must be a regular file")
    if output_dir == source_bundle_dir:
        raise ValueError("output directory must differ from source bundle")
    try:
        output_dir.relative_to(source_bundle_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "output directory must not be inside source bundle"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be absent or empty")

    manifest_path = source_bundle_dir / "source_bundle_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(
            "source bundle manifest must be a regular file"
        )
    manifest = _load_json(manifest_path)
    expected_tensor_ids = validate_source_bundle_manifest(manifest)
    thresholds_payload = _load_json(thresholds_path)
    thresholds = validate_thresholds(thresholds_payload)
    manifest_sha256 = sha256_file(manifest_path)
    if (
        thresholds_payload["pilot_source_bundle_sha256"]
        != manifest_sha256
    ):
        raise ValueError(
            "source hash mismatch: threshold source bundle manifest binding"
        )
    allowed_files = {"source_bundle_manifest.json"}
    for tensor in manifest["tensors"]:
        relative_path = tensor["relative_path"]
        allowed_files.add(relative_path)
        source_path = source_bundle_dir / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(
                f"source tensor must be a regular file: {relative_path}"
            )
        if sha256_file(source_path) != tensor["sha256"]:
            raise ValueError(
                f"source tensor hash mismatch: {relative_path}"
            )
    _ensure_regular_tree(source_bundle_dir, allowed_files)
    return (
        source_bundle_dir,
        output_dir,
        thresholds_path,
        manifest,
        expected_tensor_ids,
        thresholds_payload,
        thresholds,
    )


def _timed(clock_ns, function):
    start = clock_ns()
    result = function()
    end = clock_ns()
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end < start
    ):
        raise ValueError("clock_ns must return non-decreasing integers")
    return result, end - start


def _sanitized_argv(argv):
    if not argv:
        return []
    return [Path(argv[0]).name] + ["<redacted>"] * (len(argv) - 1)


def _artifact_manifest(output_dir):
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir).as_posix()
        if relative in (
            "artifact_manifest.json",
            "failure.json",
            "independent_verification.json",
            "report.md",
        ):
            continue
        rows.append({
            "path": relative,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "artifacts": rows,
    }


def run_calibration(
    source_bundle_dir: Path,
    output_dir: Path,
    *,
    thresholds_path: Path,
    load_tensor: Callable[[Path], torch.Tensor],
    save_tensor: Callable[[torch.Tensor, Path], None],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    if not callable(load_tensor):
        raise ValueError("load_tensor must be callable")
    if not callable(save_tensor):
        raise ValueError("save_tensor must be callable")
    if not callable(clock_ns):
        raise ValueError("clock_ns must be callable")
    (
        source_bundle_dir,
        output_dir,
        thresholds_path,
        manifest,
        expected_tensor_ids,
        thresholds_payload,
        thresholds,
    ) = _preflight(
        source_bundle_dir,
        output_dir,
        thresholds_path,
    )
    started_at_utc = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    try:
        shutil.copyfile(
            source_bundle_dir / "source_bundle_manifest.json",
            output_dir / "source_bundle_manifest.json",
        )
        _atomic_write_json(
            output_dir / "thresholds.json",
            thresholds_payload,
        )
        rows = []
        tensors = {
            tensor["tensor_id"]: tensor
            for tensor in manifest["tensors"]
        }
        for tensor_id in expected_tensor_ids:
            tensor_row = tensors[tensor_id]
            input_path = source_bundle_dir / tensor_row["relative_path"]
            source = load_tensor(input_path)
            if not isinstance(source, torch.Tensor):
                raise ValueError(
                    f"loaded source is not a tensor: {tensor_id}"
                )
            source = source.detach().to(device="cpu").contiguous()
            encoded, encode_ns = _timed(
                clock_ns,
                lambda: encode_qwen35_recurrent_int8_per_row(source),
            )
            decoded, decode_ns = _timed(
                clock_ns,
                lambda: decode_qwen35_recurrent_int8_per_row(
                    encoded,
                    device="cpu",
                ),
            )
            metrics = qwen35_recurrent_int8_error_metrics(
                source,
                decoded,
            )
            relative_suffix = Path(
                f"rank{tensor_row['rank']}/"
                f"{tensor_row['workload_id']}/"
                f"layer{tensor_row['layer_index']}.pt"
            )
            source_relative = Path("source") / relative_suffix
            values_relative = (
                Path("encoded_values") / relative_suffix
            )
            scales_relative = Path("scales") / relative_suffix
            decoded_relative = Path("decoded") / relative_suffix
            source_output = output_dir / source_relative
            source_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(input_path, source_output)
            _atomic_save_tensor(
                output_dir / values_relative,
                encoded.values.detach().cpu().contiguous(),
                save_tensor,
            )
            _atomic_save_tensor(
                output_dir / scales_relative,
                encoded.scales.detach().cpu().contiguous(),
                save_tensor,
            )
            _atomic_save_tensor(
                output_dir / decoded_relative,
                decoded.detach().cpu().contiguous(),
                save_tensor,
            )
            zero_row_count = int(
                torch.all(source == 0, dim=-1).sum().item()
            )
            saturation_count = int(
                torch.logical_or(
                    encoded.values == -127,
                    encoded.values == 127,
                ).sum().item()
            )
            row = {
                "tensor_id": tensor_id,
                "rank": tensor_row["rank"],
                "workload_id": tensor_row["workload_id"],
                "layer_index": tensor_row["layer_index"],
                "source_path": source_relative.as_posix(),
                "source_sha256": sha256_file(source_output),
                "source_shape": list(source.shape),
                "source_dtype": "float32",
                "codec": CODEC_ID,
                "encoded_values_path": values_relative.as_posix(),
                "encoded_values_sha256": sha256_file(
                    output_dir / values_relative
                ),
                "encoded_values_shape": list(encoded.values.shape),
                "encoded_values_dtype": "int8",
                "scales_path": scales_relative.as_posix(),
                "scales_sha256": sha256_file(
                    output_dir / scales_relative
                ),
                "scales_shape": list(encoded.scales.shape),
                "scales_dtype": "float32",
                "decoded_path": decoded_relative.as_posix(),
                "decoded_sha256": sha256_file(
                    output_dir / decoded_relative
                ),
                "decoded_shape": list(decoded.shape),
                "decoded_dtype": "float32",
                "logical_bytes": encoded.logical_bytes,
                "payload_bytes": encoded.payload_bytes,
                "scale_bytes": encoded.scale_bytes,
                "encoded_bytes": encoded.encoded_bytes,
                "compression_ratio": (
                    encoded.logical_bytes / encoded.encoded_bytes
                ),
                "zero_row_count": zero_row_count,
                "saturation_count": saturation_count,
                "max_abs_error": metrics["max_abs_error"],
                "mean_abs_error": metrics["mean_abs_error"],
                "rmse": metrics["rmse"],
                "relative_l2_error": metrics["relative_l2_error"],
                "cosine_similarity": metrics["cosine_similarity"],
                "encode_ns": encode_ns,
                "decode_ns": decode_ns,
                "finite_source": metrics["finite_source"],
                "finite_scales": bool(
                    torch.isfinite(encoded.scales).all().item()
                ),
                "finite_decoded": metrics["finite_decoded"],
            }
            reasons = validate_calibration_row(row)
            if reasons:
                raise RuntimeError(
                    "producer generated an invalid calibration row: "
                    + "; ".join(reasons)
                )
            rows.append(row)
            completed.append(tensor_id)
        classification, reasons = classify_calibration(
            tuple(rows),
            expected_tensor_ids=expected_tensor_ids,
            thresholds=thresholds,
        )
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        commands = {
            "schema_version": SCHEMA_VERSION,
            "codec": CODEC_ID,
            "argv": _sanitized_argv(sys.argv),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
            "producer_only": True,
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "classification": classification,
            "reasons": list(reasons),
            "row_count": len(rows),
            "source_bundle_sha256": sha256_file(
                output_dir / "source_bundle_manifest.json"
            ),
            "thresholds_sha256": sha256_file(
                output_dir / "thresholds.json"
            ),
            "logical_bytes": sum(
                row["logical_bytes"] for row in rows
            ),
            "payload_bytes": sum(
                row["payload_bytes"] for row in rows
            ),
            "scale_bytes": sum(
                row["scale_bytes"] for row in rows
            ),
            "encoded_bytes": sum(
                row["encoded_bytes"] for row in rows
            ),
            "producer_only": True,
        }
        _atomic_write_json(output_dir / "commands.json", commands)
        _atomic_write_jsonl(
            output_dir / "calibration_rows.jsonl",
            rows,
        )
        _atomic_write_json(output_dir / "summary.json", summary)
        _atomic_write_json(
            output_dir / "artifact_manifest.json",
            _artifact_manifest(output_dir),
        )
        return summary
    except BaseException as error:
        for name in ("summary.json", "artifact_manifest.json"):
            path = output_dir / name
            if path.exists():
                path.unlink()
        _atomic_write_json(
            output_dir / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "completed_tensor_ids": completed,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
        raise
