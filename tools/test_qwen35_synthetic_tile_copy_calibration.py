from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "tools/benchmark_qwen35_safetensors_tile_copy.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "qwen35_synthetic_tile_copy_benchmark",
        BENCHMARK,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()
SCHEMA_VERSION = benchmark.SCHEMA_VERSION
run_qwen35_synthetic_tile_copy_calibration = (
    benchmark.run_qwen35_synthetic_tile_copy_calibration
)


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def _assert_timing_record(record):
    assert tuple(record) == (
        "raw_seconds",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "call_count",
        "requested_tile_bytes",
        "peak_tile_bytes",
        "ratio_to_baseline_median",
        "destination_checksum",
        "checksum_verified",
    )
    assert len(record["raw_seconds"]) == 1
    assert all(
        isinstance(value, float) and value >= 0.0
        for value in record["raw_seconds"]
    )
    assert record["min_seconds"] <= record["median_seconds"]
    assert record["median_seconds"] <= record["max_seconds"]
    assert record["checksum_verified"] is True


def test_tiny_exact_matrix_schema_and_accounting():
    result = run_qwen35_synthetic_tile_copy_calibration(
        rows=8,
        columns=8,
        tile_mib=(1, 2),
        repeats=1,
    )

    assert tuple(result) == (
        "schema_version",
        "environment",
        "tensor",
        "baseline",
        "tile_results",
        "interpretation_limits",
    )
    assert result["schema_version"] == SCHEMA_VERSION
    assert tuple(result["environment"]) == (
        "timestamp_utc",
        "python",
        "pytorch",
        "safetensors",
        "platform",
        "cpu",
    )
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert result["environment"]["cpu"].startswith("Apple ")
    assert result["tensor"] == {
        "name": "synthetic.weight",
        "dtype": "bfloat16",
        "shape": [8, 8],
        "payload_bytes": 128,
        "row_bytes": 16,
        "source_checksum": 224.0,
    }

    _assert_timing_record(result["baseline"])
    assert result["baseline"]["call_count"] == 1
    assert result["baseline"]["requested_tile_bytes"] == 128
    assert result["baseline"]["peak_tile_bytes"] == 128
    assert result["baseline"]["ratio_to_baseline_median"] == 1.0
    assert result["baseline"]["destination_checksum"] == 224.0

    assert len(result["tile_results"]) == 2
    for requested_mib, record in zip(
        (1, 2),
        result["tile_results"],
        strict=True,
    ):
        _assert_timing_record(record)
        assert record["call_count"] == 1
        assert record["requested_tile_bytes"] == requested_mib << 20
        assert record["peak_tile_bytes"] == 128
        assert record["destination_checksum"] == 224.0
        assert record["ratio_to_baseline_median"] >= 0.0

    limits = result["interpretation_limits"]
    assert isinstance(limits, list)
    assert limits
    assert any("synthetic" in limit.lower() for limit in limits)
    assert any("inference" in limit.lower() for limit in limits)
    json.loads(json.dumps(result, allow_nan=False))


def test_full_checksum_and_short_final_tile():
    columns = 131_072
    result = run_qwen35_synthetic_tile_copy_calibration(
        rows=5,
        columns=columns,
        tile_mib=(1,),
        repeats=1,
    )

    assert result["tensor"]["row_bytes"] == 262_144
    assert result["tensor"]["payload_bytes"] == 1_310_720
    assert result["tensor"]["source_checksum"] == 1_310_720.0
    record = result["tile_results"][0]
    assert record["call_count"] == 2
    assert record["requested_tile_bytes"] == 1 << 20
    assert record["peak_tile_bytes"] == 1 << 20
    assert record["destination_checksum"] == 1_310_720.0
    assert record["checksum_verified"] is True


def test_invalid_input_matrix():
    cases = (
        ({"rows": True}, "rows"),
        ({"rows": 0}, "rows"),
        ({"columns": True}, "columns"),
        ({"columns": 0}, "columns"),
        ({"repeats": True}, "repeats"),
        ({"repeats": 0}, "repeats"),
        ({"tile_mib": ()}, "tile_mib"),
        ({"tile_mib": (True,)}, "tile_mib"),
        ({"tile_mib": (0,)}, "tile_mib"),
        (
            {"rows": 2, "columns": 524_289, "tile_mib": (1,)},
            "complete row",
        ),
    )
    for keyword_arguments, message in cases:
        _expect_error(
            lambda keyword_arguments=keyword_arguments: (
                run_qwen35_synthetic_tile_copy_calibration(
                    repeats=1,
                    **keyword_arguments,
                )
            ),
            message,
        )


def test_cli_json_persistence():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "nested/result.json"
        completed = subprocess.run(
            (
                sys.executable,
                str(BENCHMARK),
                "--rows",
                "8",
                "--columns",
                "8",
                "--tile-mib",
                "1,2",
                "--repeats",
                "1",
                "--output-json",
                str(output),
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        printed = json.loads(completed.stdout)
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert persisted == printed
        assert persisted["schema_version"] == SCHEMA_VERSION
        assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def main():
    test_tiny_exact_matrix_schema_and_accounting()
    test_full_checksum_and_short_final_tile()
    test_invalid_input_matrix()
    test_cli_json_persistence()
    print(
        "qwen35 synthetic tile copy calibration tests passed (4 tests)"
    )


if __name__ == "__main__":
    main()
