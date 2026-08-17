from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR = ROOT / "tools/estimate_qwen35_weighted_tile_policy.py"
CALIBRATION = (
    ROOT
    / "experiments/qwen35_hybrid_state/"
    "20260727-five-binding-synthetic-tile-replay.json"
)


def _load_estimator():
    spec = importlib.util.spec_from_file_location(
        "qwen35_weighted_tile_policy_estimator",
        ESTIMATOR,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


estimator = _load_estimator()


def _expect_error(function, message):
    try:
        function()
    except (TypeError, ValueError) as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected error containing {message!r}")


def test_exact_kind_fit_contract():
    fit = estimator.fit_qwen35_tile_kind_calibration(
        {
            "kind": "axis0",
            "local_payload_bytes": 32 << 20,
            "budget_results": [
                {
                    "tile_count": 8,
                    "median_seconds": 0.0018,
                    "requested_tile_bytes": 4 << 20,
                    "exact_destination_verified": True,
                },
                {
                    "tile_count": 4,
                    "median_seconds": 0.0014,
                    "requested_tile_bytes": 8 << 20,
                    "exact_destination_verified": True,
                },
                {
                    "tile_count": 2,
                    "median_seconds": 0.0012,
                    "requested_tile_bytes": 16 << 20,
                    "exact_destination_verified": True,
                },
                {
                    "tile_count": 1,
                    "median_seconds": 0.0011,
                    "requested_tile_bytes": 32 << 20,
                    "exact_destination_verified": True,
                },
            ],
        }
    )

    assert tuple(fit) == (
        "kind",
        "calibration_bytes",
        "intercept_seconds_per_calibration_bytes",
        "per_tile_seconds",
        "r_squared",
        "max_absolute_residual_seconds",
        "points",
    )
    assert fit["kind"] == "axis0"
    assert fit["calibration_bytes"] == 32 << 20
    assert math.isclose(
        fit["intercept_seconds_per_calibration_bytes"],
        0.001,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        fit["per_tile_seconds"],
        0.0001,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(fit["r_squared"], 1.0)
    assert fit["max_absolute_residual_seconds"] < 1e-15
    assert [point["budget_bytes"] >> 20 for point in fit["points"]] == [
        4,
        8,
        16,
        32,
    ]


def test_contribution_math_and_pareto_frontier():
    fits = {
        "axis0": {
            "kind": "axis0",
            "calibration_bytes": 32 << 20,
            "intercept_seconds_per_calibration_bytes": 0.001,
            "per_tile_seconds": 0.0001,
        },
    }
    distributions = [
        {
            "budget_bytes": 8 << 20,
            "peak_tile_bytes": 8 << 20,
            "tile_count": 4,
            "by_kind": {
                "axis0": {
                    "binding_count": 1,
                    "destination_bytes": 64 << 20,
                    "tile_count": 4,
                }
            },
        },
        {
            "budget_bytes": 16 << 20,
            "peak_tile_bytes": 16 << 20,
            "tile_count": 2,
            "by_kind": {
                "axis0": {
                    "binding_count": 1,
                    "destination_bytes": 64 << 20,
                    "tile_count": 2,
                }
            },
        },
        {
            "budget_bytes": 32 << 20,
            "peak_tile_bytes": 32 << 20,
            "tile_count": 2,
            "by_kind": {
                "axis0": {
                    "binding_count": 1,
                    "destination_bytes": 64 << 20,
                    "tile_count": 2,
                }
            },
        },
    ]
    result = estimator.estimate_qwen35_weighted_tile_policy(
        fits,
        distributions,
        baseline_budget_bytes=8 << 20,
    )
    assert [record["estimated_latency_proxy_seconds"] for record in result] == [
        0.0024,
        0.0022,
        0.0022,
    ]
    assert result[0]["proxy_reduction_vs_baseline_fraction"] == 0.0
    assert math.isclose(
        result[1]["proxy_reduction_vs_baseline_fraction"],
        1.0 / 12.0,
    )
    assert result[1]["extra_peak_bytes_vs_baseline"] == 8 << 20
    assert result[0]["pareto_dominated"] is False
    assert result[1]["pareto_dominated"] is False
    assert result[2]["pareto_dominated"] is True


def test_rejects_malformed_calibration():
    base = {
        "kind": "axis0",
        "local_payload_bytes": 32 << 20,
        "budget_results": [],
    }
    cases = (
        (object(), "case"),
        (base, "four budget"),
        (
            {
                **base,
                "budget_results": [
                    {
                        "tile_count": 1,
                        "median_seconds": 0.1,
                        "requested_tile_bytes": value << 20,
                        "exact_destination_verified": False,
                    }
                    for value in (4, 8, 16, 32)
                ],
            },
            "exact destination",
        ),
        (
            {
                **base,
                "budget_results": [
                    {
                        "tile_count": 1,
                        "median_seconds": 0.1,
                        "requested_tile_bytes": value << 20,
                        "exact_destination_verified": True,
                    }
                    for value in (4, 8, 16, 64)
                ],
            },
            "budget set",
        ),
    )
    for value, message in cases:
        _expect_error(
            lambda value=value: (
                estimator.fit_qwen35_tile_kind_calibration(value)
            ),
            message,
        )


def test_cli_persists_real_static_artifact_without_payload_open():
    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "nested/result.json"
        completed = subprocess.run(
            (
                sys.executable,
                str(ESTIMATOR),
                "--calibration-json",
                str(CALIBRATION),
                "--output-json",
                str(output),
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            env={
                **dict(**__import__("os").environ),
                "QWEN35_FAIL_ON_SAFETENSORS_PAYLOAD_OPEN": "1",
            },
        )
        printed = json.loads(completed.stdout)
        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert printed == persisted
        assert persisted["schema_version"] == (
            "qwen35.weighted-tile-policy-estimator.v1"
        )
        assert persisted["payload_open_count"] == 0
        assert [rank["name"] for rank in persisted["ranks"]] == [
            "tp1-rank0",
            "tp2-rank0",
            "tp2-rank1",
        ]
        for rank in persisted["ranks"]:
            assert [
                record["budget_bytes"] >> 20
                for record in rank["budget_evaluations"]
            ] == [4, 8, 16, 32]
        assert not tuple(output.parent.glob(f".{output.name}.*.tmp"))


def main():
    test_exact_kind_fit_contract()
    test_contribution_math_and_pareto_frontier()
    test_rejects_malformed_calibration()
    test_cli_persists_real_static_artifact_without_payload_open()
    print("qwen35 weighted tile policy estimator tests passed (4 tests)")


if __name__ == "__main__":
    main()
