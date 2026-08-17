from __future__ import annotations

import argparse
import builtins
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "qwen35.weighted-tile-policy-estimator.v1"
CALIBRATION_SCHEMA = (
    "qwen35.five-binding-synthetic-tile-replay.v1"
)
EXPECTED_KINDS = (
    "axis0",
    "axis1",
    "segmented_axis0",
    "squeeze_axis0",
    "replicated",
)
EXPECTED_BUDGETS = tuple(value << 20 for value in (4, 8, 16, 32))
CALIBRATION_BYTES = 32 << 20


def _finite_positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rounded(value):
    return float(round(value, 15))


def fit_qwen35_tile_kind_calibration(case):
    if not isinstance(case, dict):
        raise ValueError("calibration case must be a dictionary")
    kind = case.get("kind")
    if kind not in EXPECTED_KINDS:
        raise ValueError("calibration case kind is invalid")
    calibration_bytes = _positive_integer(
        case.get("local_payload_bytes"),
        "local_payload_bytes",
    )
    if calibration_bytes != CALIBRATION_BYTES:
        raise ValueError(
            "calibration case must use exactly 32 MiB local payload"
        )
    records = case.get("budget_results")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError(
            "calibration case must contain exactly four budget results"
        )

    points = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(
                "calibration budget result must be a dictionary"
            )
        if record.get("exact_destination_verified") is not True:
            raise ValueError(
                "calibration requires exact destination verification"
            )
        budget_bytes = _positive_integer(
            record.get("requested_tile_bytes"),
            "requested_tile_bytes",
        )
        tile_count = _positive_integer(
            record.get("tile_count"),
            "tile_count",
        )
        median_seconds = _finite_positive(
            record.get("median_seconds"),
            "median_seconds",
        )
        points.append({
            "budget_bytes": budget_bytes,
            "tile_count": tile_count,
            "median_seconds": median_seconds,
        })
    points.sort(key=lambda point: point["budget_bytes"])
    if tuple(point["budget_bytes"] for point in points) != EXPECTED_BUDGETS:
        raise ValueError(
            "calibration budget set must be exactly 4, 8, 16, 32 MiB"
        )

    x_values = [point["tile_count"] for point in points]
    y_values = [point["median_seconds"] for point in points]
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    denominator = sum(
        (value - mean_x) ** 2 for value in x_values
    )
    if denominator <= 0:
        raise ValueError(
            "calibration tile counts must vary across budgets"
        )
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator
    intercept = mean_y - slope * mean_x
    if slope < 0 or intercept < 0:
        raise ValueError(
            "calibration fit must have non-negative intercept and slope"
        )
    predictions = [
        intercept + slope * x_value
        for x_value in x_values
    ]
    residuals = [
        y_value - prediction
        for y_value, prediction in zip(
            y_values,
            predictions,
            strict=True,
        )
    ]
    total_sum_squares = sum(
        (value - mean_y) ** 2 for value in y_values
    )
    residual_sum_squares = sum(value * value for value in residuals)
    r_squared = (
        1.0
        if total_sum_squares == 0
        else 1.0 - residual_sum_squares / total_sum_squares
    )
    if not math.isfinite(r_squared):
        raise ValueError("calibration fit residual metrics must be finite")

    return {
        "kind": kind,
        "calibration_bytes": calibration_bytes,
        "intercept_seconds_per_calibration_bytes": _rounded(
            intercept
        ),
        "per_tile_seconds": _rounded(slope),
        "r_squared": _rounded(r_squared),
        "max_absolute_residual_seconds": _rounded(
            max(abs(value) for value in residuals)
        ),
        "points": points,
    }


def _validate_fit(kind, fit):
    if not isinstance(fit, dict) or fit.get("kind") != kind:
        raise ValueError(f"missing exact calibration fit for {kind}")
    calibration_bytes = _positive_integer(
        fit.get("calibration_bytes"),
        f"{kind} calibration_bytes",
    )
    intercept = fit.get("intercept_seconds_per_calibration_bytes")
    slope = fit.get("per_tile_seconds")
    for value, name in (
        (intercept, "intercept"),
        (slope, "per_tile_seconds"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{kind} {name} must be non-negative finite")
    return calibration_bytes, float(intercept), float(slope)


def _estimated_budget_record(fits, distribution):
    if not isinstance(distribution, dict):
        raise ValueError("tile distribution must be a dictionary")
    budget_bytes = _positive_integer(
        distribution.get("budget_bytes"),
        "budget_bytes",
    )
    peak_tile_bytes = _positive_integer(
        distribution.get("peak_tile_bytes"),
        "peak_tile_bytes",
    )
    tile_count = _positive_integer(
        distribution.get("tile_count"),
        "tile_count",
    )
    by_kind = distribution.get("by_kind")
    if not isinstance(by_kind, dict) or set(by_kind) != set(EXPECTED_KINDS):
        if set(by_kind or {}) != {"axis0"} or set(fits) != {"axis0"}:
            raise ValueError(
                "tile distribution must contain every calibrated kind"
            )

    contributions = {}
    total = 0.0
    for kind, distribution_record in by_kind.items():
        if not isinstance(distribution_record, dict):
            raise ValueError(f"{kind} distribution must be a dictionary")
        binding_count = _positive_integer(
            distribution_record.get("binding_count"),
            f"{kind} binding_count",
        )
        destination_bytes = _positive_integer(
            distribution_record.get("destination_bytes"),
            f"{kind} destination_bytes",
        )
        kind_tile_count = _positive_integer(
            distribution_record.get("tile_count"),
            f"{kind} tile_count",
        )
        calibration_bytes, intercept, slope = _validate_fit(
            kind,
            fits.get(kind),
        )
        byte_component = intercept * (
            destination_bytes / calibration_bytes
        )
        tile_component = slope * kind_tile_count
        estimated = byte_component + tile_component
        contributions[kind] = {
            "binding_count": binding_count,
            "destination_bytes": destination_bytes,
            "tile_count": kind_tile_count,
            "byte_copy_proxy_seconds": _rounded(byte_component),
            "tile_call_proxy_seconds": _rounded(tile_component),
            "estimated_proxy_seconds": _rounded(estimated),
        }
        total += estimated
    if sum(
        record["tile_count"] for record in contributions.values()
    ) != tile_count:
        raise ValueError("per-kind tile counts must equal total tile_count")
    return {
        "budget_bytes": budget_bytes,
        "peak_tile_bytes": peak_tile_bytes,
        "tile_count": tile_count,
        "estimated_latency_proxy_seconds": _rounded(total),
        "proxy_reduction_vs_baseline_fraction": None,
        "extra_peak_bytes_vs_baseline": None,
        "pareto_dominated": None,
        "by_kind": contributions,
    }


def estimate_qwen35_weighted_tile_policy(
    fits,
    distributions,
    *,
    baseline_budget_bytes,
):
    if not isinstance(fits, dict) or not fits:
        raise ValueError("fits must be a non-empty dictionary")
    if not isinstance(distributions, list) or not distributions:
        raise ValueError("distributions must be a non-empty list")
    baseline_budget_bytes = _positive_integer(
        baseline_budget_bytes,
        "baseline_budget_bytes",
    )
    records = [
        _estimated_budget_record(fits, distribution)
        for distribution in distributions
    ]
    records.sort(key=lambda record: record["budget_bytes"])
    baseline = next(
        (
            record
            for record in records
            if record["budget_bytes"] == baseline_budget_bytes
        ),
        None,
    )
    if baseline is None:
        raise ValueError("baseline budget must exist in distributions")
    baseline_proxy = baseline["estimated_latency_proxy_seconds"]
    baseline_peak = baseline["peak_tile_bytes"]
    for record in records:
        record["proxy_reduction_vs_baseline_fraction"] = _rounded(
            (baseline_proxy - record["estimated_latency_proxy_seconds"])
            / baseline_proxy
        )
        record["extra_peak_bytes_vs_baseline"] = (
            record["peak_tile_bytes"] - baseline_peak
        )
        record["pareto_dominated"] = any(
            (
                other["peak_tile_bytes"] <= record["peak_tile_bytes"]
                and other["estimated_latency_proxy_seconds"]
                <= record["estimated_latency_proxy_seconds"]
                and (
                    other["peak_tile_bytes"] < record["peak_tile_bytes"]
                    or other["estimated_latency_proxy_seconds"]
                    < record["estimated_latency_proxy_seconds"]
                )
            )
            for other in records
            if other is not record
        )
    return records


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _real_static_distributions():
    helper = _load_module(
        "qwen35_weighted_real_checkpoint_tile_helper",
        "tools/test_qwen35_real_checkpoint_tiles.py",
    )
    config, index_payload, shard_headers = (
        helper.real_helper._load_metadata()
    )
    tensor_plan = (
        helper.real_helper.build_qwen35_checkpoint_tensor_plan(
            config,
            index_payload,
            shard_headers,
        )
    )
    ranks = []
    for world_size in (1, 2):
        for rank in range(world_size):
            binding_plan = helper._build_binding_plan(
                config,
                tensor_plan,
                world_size,
                rank,
            )
            distributions = []
            for budget_bytes in EXPECTED_BUDGETS:
                tile_plan = helper.build_qwen35_checkpoint_tile_plan(
                    binding_plan,
                    max_tile_bytes=budget_bytes,
                )
                by_kind = {
                    kind: {
                        "binding_indices": set(),
                        "destination_bytes": 0,
                        "tile_count": 0,
                    }
                    for kind in EXPECTED_KINDS
                }
                for tile in tile_plan.tiles:
                    record = by_kind[tile.kind]
                    record["binding_indices"].add(tile.binding_index)
                    record["destination_bytes"] += tile.byte_count
                    record["tile_count"] += 1
                distributions.append({
                    "budget_bytes": budget_bytes,
                    "peak_tile_bytes": tile_plan.peak_tile_bytes,
                    "tile_count": len(tile_plan.tiles),
                    "by_kind": {
                        kind: {
                            "binding_count": len(
                                record["binding_indices"]
                            ),
                            "destination_bytes": (
                                record["destination_bytes"]
                            ),
                            "tile_count": record["tile_count"],
                        }
                        for kind, record in by_kind.items()
                    },
                })
            ranks.append({
                "name": f"tp{world_size}-rank{rank}",
                "tensor_parallel_size": world_size,
                "tensor_parallel_rank": rank,
                "binding_count": len(binding_plan.bindings),
                "distributions": distributions,
            })
    return ranks


def _load_calibration(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError("calibration schema_version is invalid")
    cases = payload.get("cases")
    if (
        not isinstance(cases, list)
        or [case.get("kind") for case in cases] != list(EXPECTED_KINDS)
    ):
        raise ValueError("calibration cases must contain exact kind order")
    fits = {
        case["kind"]: fit_qwen35_tile_kind_calibration(case)
        for case in cases
    }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return fits, digest


def _incremental_comparisons(records):
    by_budget = {
        record["budget_bytes"]: record for record in records
    }
    comparisons = []
    for lower, upper in ((8 << 20, 16 << 20), (16 << 20, 32 << 20)):
        lower_record = by_budget[lower]
        upper_record = by_budget[upper]
        reduction = (
            lower_record["estimated_latency_proxy_seconds"]
            - upper_record["estimated_latency_proxy_seconds"]
        )
        comparisons.append({
            "lower_budget_bytes": lower,
            "upper_budget_bytes": upper,
            "proxy_reduction_seconds": _rounded(reduction),
            "proxy_reduction_fraction_of_lower": _rounded(
                reduction
                / lower_record["estimated_latency_proxy_seconds"]
            ),
            "extra_peak_bytes": (
                upper_record["peak_tile_bytes"]
                - lower_record["peak_tile_bytes"]
            ),
            "tile_count_reduction": (
                lower_record["tile_count"]
                - upper_record["tile_count"]
            ),
        })
    return comparisons


def build_qwen35_weighted_tile_policy_artifact(calibration_json):
    fits, calibration_sha256 = _load_calibration(calibration_json)
    payload_open_count = 0
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        nonlocal payload_open_count
        if str(file).endswith(".safetensors"):
            payload_open_count += 1
            if os.environ.get(
                "QWEN35_FAIL_ON_SAFETENSORS_PAYLOAD_OPEN"
            ) == "1":
                raise AssertionError(
                    "safetensors payload must not be opened"
                )
        return original_open(file, *args, **kwargs)

    builtins.open = guarded_open
    try:
        ranks = _real_static_distributions()
    finally:
        builtins.open = original_open
    if payload_open_count:
        raise RuntimeError("real safetensors payload open count must be zero")

    rank_results = []
    for rank in ranks:
        evaluations = estimate_qwen35_weighted_tile_policy(
            fits,
            rank["distributions"],
            baseline_budget_bytes=8 << 20,
        )
        rank_results.append({
            "name": rank["name"],
            "tensor_parallel_size": rank["tensor_parallel_size"],
            "tensor_parallel_rank": rank["tensor_parallel_rank"],
            "binding_count": rank["binding_count"],
            "budget_evaluations": evaluations,
            "pareto_frontier_budget_bytes": [
                record["budget_bytes"]
                for record in evaluations
                if not record["pareto_dominated"]
            ],
            "incremental_comparisons": _incremental_comparisons(
                evaluations
            ),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "path": str(Path(calibration_json)),
            "sha256": calibration_sha256,
            "schema_version": CALIBRATION_SCHEMA,
            "fits": [fits[kind] for kind in EXPECTED_KINDS],
        },
        "budgets_bytes": list(EXPECTED_BUDGETS),
        "baseline_budget_bytes": 8 << 20,
        "payload_open_count": payload_open_count,
        "ranks": rank_results,
        "interpretation_limits": [
            (
                "Proxy seconds are derived from local synthetic fits and "
                "real static tile distributions; they are not measured or "
                "predicted real checkpoint load latency."
            ),
            (
                "The artifact reads config/index/header evidence and builds "
                "meta plans without opening safetensors payloads."
            ),
            (
                "It does not establish RSS, disk, page-cache, GPU, "
                "inference, cache, compression, accuracy, or quality gains."
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


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate Qwen3.5 tile-budget Pareto trade-offs from real "
            "static plans and per-kind synthetic calibration."
        )
    )
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    arguments = _parse_arguments()
    artifact = build_qwen35_weighted_tile_policy_artifact(
        arguments.calibration_json
    )
    if arguments.output_json is not None:
        _atomic_write_json(arguments.output_json, artifact)
    print(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
