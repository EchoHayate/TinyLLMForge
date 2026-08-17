from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import torch


SCHEMA_VERSION = "qwen35.synthetic-safetensors-tile-copy.v1"
TENSOR_NAME = "synthetic.weight"
MIB = 1 << 20


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_inputs(rows, columns, tile_mib, repeats):
    rows = _positive_integer(rows, "rows")
    columns = _positive_integer(columns, "columns")
    repeats = _positive_integer(repeats, "repeats")
    if not isinstance(tile_mib, tuple) or not tile_mib:
        raise ValueError("tile_mib must be a non-empty tuple")
    validated_tile_mib = tuple(
        _positive_integer(value, "tile_mib entry")
        for value in tile_mib
    )
    row_bytes = columns * torch.empty(
        (),
        dtype=torch.bfloat16,
    ).element_size()
    for value in validated_tile_mib:
        if value * MIB < row_bytes:
            raise ValueError(
                "each tile_mib budget must fit at least one complete row"
            )
    return rows, columns, validated_tile_mib, repeats, row_bytes


def _build_source(rows, columns):
    row_values = torch.arange(rows, dtype=torch.bfloat16).reshape(
        rows,
        1,
    )
    return row_values.expand(rows, columns).contiguous()


def _checksum(tensor):
    return float(tensor.to(dtype=torch.float32).sum().item())


def _require_checksum(actual, expected):
    if actual != expected:
        raise RuntimeError(
            "synthetic destination checksum mismatch: "
            f"expected {expected}, got {actual}"
        )


def _run_baseline_once(path, shape, expected_checksum):
    started = time.perf_counter()
    destination = torch.empty(shape, dtype=torch.bfloat16)
    with safe_open(path, framework="pt", device="cpu") as handle:
        destination.copy_(handle.get_tensor(TENSOR_NAME))
    elapsed = time.perf_counter() - started
    destination_checksum = _checksum(destination)
    _require_checksum(destination_checksum, expected_checksum)
    return elapsed, destination_checksum


def _run_tiled_once(
    path,
    shape,
    rows_per_tile,
    expected_checksum,
):
    started = time.perf_counter()
    destination = torch.empty(shape, dtype=torch.bfloat16)
    call_count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        slice_view = handle.get_slice(TENSOR_NAME)
        for start in range(0, shape[0], rows_per_tile):
            stop = min(start + rows_per_tile, shape[0])
            destination[start:stop].copy_(slice_view[start:stop])
            call_count += 1
    elapsed = time.perf_counter() - started
    destination_checksum = _checksum(destination)
    _require_checksum(destination_checksum, expected_checksum)
    return elapsed, destination_checksum, call_count


def _timing_record(
    raw_seconds,
    *,
    call_count,
    requested_tile_bytes,
    peak_tile_bytes,
    baseline_median,
    destination_checksum,
):
    median_seconds = float(statistics.median(raw_seconds))
    ratio = (
        1.0
        if baseline_median is None
        else median_seconds / baseline_median
    )
    return {
        "raw_seconds": [float(value) for value in raw_seconds],
        "median_seconds": median_seconds,
        "min_seconds": float(min(raw_seconds)),
        "max_seconds": float(max(raw_seconds)),
        "call_count": call_count,
        "requested_tile_bytes": requested_tile_bytes,
        "peak_tile_bytes": peak_tile_bytes,
        "ratio_to_baseline_median": float(ratio),
        "destination_checksum": destination_checksum,
        "checksum_verified": True,
    }


def _cpu_description():
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ("/usr/sbin/sysctl", "-n", key),
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                value = ""
            if value:
                return value
    processor = platform.processor().strip()
    if processor:
        return processor
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8",
            ).splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.machine()


def _environment_record():
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
        "cpu": _cpu_description(),
    }


def run_qwen35_synthetic_tile_copy_calibration(
    *,
    rows: int = 16384,
    columns: int = 2048,
    tile_mib: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    repeats: int = 5,
) -> dict:
    rows, columns, tile_mib, repeats, row_bytes = _validate_inputs(
        rows,
        columns,
        tile_mib,
        repeats,
    )
    shape = (rows, columns)
    payload_bytes = rows * row_bytes
    source = _build_source(rows, columns)
    source_checksum = _checksum(source)

    with tempfile.TemporaryDirectory(
        prefix="qwen35-synthetic-tile-copy-",
    ) as temporary_directory:
        shard_path = Path(temporary_directory) / "synthetic.safetensors"
        save_file({TENSOR_NAME: source}, shard_path)
        del source

        _run_baseline_once(shard_path, shape, source_checksum)
        baseline_seconds = []
        baseline_checksum = None
        for _ in range(repeats):
            elapsed, baseline_checksum = _run_baseline_once(
                shard_path,
                shape,
                source_checksum,
            )
            baseline_seconds.append(elapsed)
        baseline = _timing_record(
            baseline_seconds,
            call_count=1,
            requested_tile_bytes=payload_bytes,
            peak_tile_bytes=payload_bytes,
            baseline_median=None,
            destination_checksum=baseline_checksum,
        )

        tile_results = []
        for requested_mib in tile_mib:
            requested_tile_bytes = requested_mib * MIB
            rows_per_tile = requested_tile_bytes // row_bytes
            expected_call_count = (
                rows + rows_per_tile - 1
            ) // rows_per_tile
            peak_rows = min(rows, rows_per_tile)
            peak_tile_bytes = peak_rows * row_bytes

            _, _, warmup_call_count = _run_tiled_once(
                shard_path,
                shape,
                rows_per_tile,
                source_checksum,
            )
            if warmup_call_count != expected_call_count:
                raise RuntimeError(
                    "synthetic tiled call accounting mismatch"
                )

            raw_seconds = []
            destination_checksum = None
            for _ in range(repeats):
                elapsed, destination_checksum, call_count = (
                    _run_tiled_once(
                        shard_path,
                        shape,
                        rows_per_tile,
                        source_checksum,
                    )
                )
                if call_count != expected_call_count:
                    raise RuntimeError(
                        "synthetic tiled call accounting mismatch"
                    )
                raw_seconds.append(elapsed)
            tile_results.append(
                _timing_record(
                    raw_seconds,
                    call_count=expected_call_count,
                    requested_tile_bytes=requested_tile_bytes,
                    peak_tile_bytes=peak_tile_bytes,
                    baseline_median=baseline["median_seconds"],
                    destination_checksum=destination_checksum,
                )
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "environment": _environment_record(),
        "tensor": {
            "name": TENSOR_NAME,
            "dtype": "bfloat16",
            "shape": [rows, columns],
            "payload_bytes": payload_bytes,
            "row_bytes": row_bytes,
            "source_checksum": source_checksum,
        },
        "baseline": baseline,
        "tile_results": tile_results,
        "interpretation_limits": [
            (
                "This is a warm-cache CPU microbenchmark over one "
                "temporary synthetic safetensors payload."
            ),
            (
                "It does not measure real checkpoint loading, cold-cache "
                "or disk throughput, RSS, or page-cache peak."
            ),
            (
                "It does not establish GPU, inference, KV-cache, "
                "compression, memory, quality, or accuracy benefits."
            ),
        ],
    }


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=False,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parse_tile_mib(value):
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tile MiB values must be comma-separated integers"
        ) from error
    if not parsed or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError(
            "tile MiB values must be positive"
        )
    return parsed


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate CPU safetensors get_slice tile-copy overhead "
            "using a temporary synthetic BF16 tensor."
        )
    )
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument(
        "--tile-mib",
        type=_parse_tile_mib,
        default=(1, 2, 4, 8, 16, 32, 64),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    result = run_qwen35_synthetic_tile_copy_calibration(
        rows=arguments.rows,
        columns=arguments.columns,
        tile_mib=arguments.tile_mib,
        repeats=arguments.repeats,
    )
    if arguments.output_json is not None:
        _atomic_write_json(arguments.output_json, result)
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=False,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
