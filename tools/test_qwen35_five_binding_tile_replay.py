from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = (
    ROOT / "tools/benchmark_qwen35_five_binding_tile_replay.py"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "qwen35_five_binding_tile_replay_benchmark",
        BENCHMARK,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark()
SCHEMA_VERSION = benchmark.SCHEMA_VERSION
run_calibration = (
    benchmark.run_qwen35_five_binding_tile_replay_calibration
)


def test_source_values_remain_exactly_representable_in_bf16():
    source = benchmark._source_tensor((1024,), 65)
    expected = (
        torch.arange(1024, dtype=torch.int32).remainder(127).add(65)
    )
    assert torch.equal(source.float(), expected.float())


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_reduced_schema_five_kinds_and_exact_destinations():
    result = run_calibration(
        tile_mib=(1,),
        repeats=1,
        local_payload_mib=1,
    )

    assert tuple(result) == (
        "schema_version",
        "environment",
        "configuration",
        "cases",
        "interpretation_limits",
    )
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["configuration"] == {
        "tensor_parallel_size": 2,
        "tensor_parallel_rank": 1,
        "tile_mib": [1],
        "repeats": 1,
        "warmups": 1,
        "local_payload_mib": 1,
    }
    assert tuple(result["environment"]) == (
        "timestamp_utc",
        "python",
        "pytorch",
        "safetensors",
        "platform",
        "cpu",
    )
    assert [case["kind"] for case in result["cases"]] == [
        "axis0",
        "axis1",
        "segmented_axis0",
        "squeeze_axis0",
        "replicated",
    ]

    for case in result["cases"]:
        assert case["local_payload_bytes"] == 1 << 20
        assert case["source_payload_bytes"] >= (
            case["local_payload_bytes"]
        )
        assert case["destination_checksum"] == (
            case["expected_checksum"]
        )
        assert len(case["budget_results"]) == 1
        record = case["budget_results"][0]
        assert tuple(record) == (
            "raw_seconds",
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "tile_count",
            "requested_tile_bytes",
            "peak_tile_bytes",
            "destination_checksum",
            "exact_destination_verified",
        )
        assert len(record["raw_seconds"]) == 1
        assert record["tile_count"] >= 1
        assert record["requested_tile_bytes"] == 1 << 20
        assert 0 < record["peak_tile_bytes"] <= 1 << 20
        assert record["destination_checksum"] == (
            case["expected_checksum"]
        )
        assert record["exact_destination_verified"] is True

    limits = result["interpretation_limits"]
    assert any("synthetic" in value.lower() for value in limits)
    assert any("inference" in value.lower() for value in limits)
    json.loads(json.dumps(result, allow_nan=False))


def test_planner_accounting_matches_complete_local_destination():
    result = run_calibration(
        tile_mib=(1, 2),
        repeats=1,
        local_payload_mib=2,
    )

    for case in result["cases"]:
        counts = []
        for record in case["budget_results"]:
            counts.append(record["tile_count"])
            assert record["peak_tile_bytes"] <= (
                record["requested_tile_bytes"]
            )
            assert record["destination_checksum"] == (
                case["expected_checksum"]
            )
            assert record["exact_destination_verified"] is True
        assert counts[0] >= counts[1]
        assert case["destination_checksum"] == (
            case["expected_checksum"]
        )


def test_invalid_input_matrix():
    cases = (
        ({"repeats": True}, "repeats"),
        ({"repeats": 0}, "repeats"),
        ({"tile_mib": ()}, "tile_mib"),
        ({"tile_mib": (True,)}, "tile_mib"),
        ({"tile_mib": (0,)}, "tile_mib"),
        ({"local_payload_mib": True}, "local_payload_mib"),
        ({"local_payload_mib": 0}, "local_payload_mib"),
    )
    for keyword_arguments, message in cases:
        _expect_error(
            lambda keyword_arguments=keyword_arguments: run_calibration(
                **keyword_arguments
            ),
            message,
        )


def test_cli_atomic_json_persistence():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "nested/result.json"
        completed = subprocess.run(
            (
                sys.executable,
                str(BENCHMARK),
                "--tile-mib",
                "1",
                "--repeats",
                "1",
                "--local-payload-mib",
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
    test_source_values_remain_exactly_representable_in_bf16()
    test_reduced_schema_five_kinds_and_exact_destinations()
    test_planner_accounting_matches_complete_local_destination()
    test_invalid_input_matrix()
    test_cli_atomic_json_persistence()
    print("qwen35 five binding tile replay tests passed (5 tests)")


if __name__ == "__main__":
    main()
