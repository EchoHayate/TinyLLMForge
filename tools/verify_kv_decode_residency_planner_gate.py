"""Independent verifier for the KV decode residency planner gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import torch

import kv_decode_residency_planner_contract as contract


class VerificationError(RuntimeError):
    pass


def _fail(message):
    raise VerificationError(message)


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"failed to load JSON {path}: {exc}")


def _load_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    _fail(
                        f"invalid JSONL row {path}:{line_number}: {exc}"
                    )
    except OSError as exc:
        _fail(f"failed to load JSONL {path}: {exc}")
    return rows


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"failed to hash {path}: {exc}")
    return digest.hexdigest()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def _atomic_write_json(path, value):
    _atomic_write(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )


def build_raw_summary(rows):
    row_ids = [row.get("row_id") for row in rows]
    case_ids = [row.get("case_id") for row in rows]
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "case_count": len(rows),
        "complete_count": sum(
            row.get("complete") is True
            for row in rows
        ),
        "row_ids_sha256": contract.canonical_json_sha256(row_ids),
        "case_ids_sha256": contract.canonical_json_sha256(case_ids),
    }


def _require_exact_row_schema(row):
    actual = set(row)
    expected = set(contract.CASE_ROW_FIELDS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        _fail(
            f"case row schema mismatch for {row.get('case_id')}: "
            f"missing={missing}, extra={extra}"
        )


def _require_nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{label} must be a non-negative integer")


def _pair_rows(rows_by_case_id, case):
    baseline_id = (
        f"{case.workload}__g{case.gpu_blocks}"
        f"__w{case.blockwise_blocks}"
        f"__{case.phase}__r{case.repetition}__baseline"
    )
    candidate_id = (
        f"{case.workload}__g{case.gpu_blocks}"
        f"__w{case.blockwise_blocks}"
        f"__{case.phase}__r{case.repetition}__candidate"
    )
    return rows_by_case_id[baseline_id], rows_by_case_id[candidate_id]


def _validate_logits(run_dir, row):
    path_value = row["decode_logits_path"]
    digest_value = row["decode_logits_sha256"]
    shape_value = row["decode_logits_shape"]
    if not path_value or not digest_value or shape_value is None:
        _fail(f"correctness logits missing for {row['case_id']}")
    path = run_dir / path_value
    if not path.is_file():
        _fail(f"correctness logits missing for {row['case_id']}: {path}")
    if _sha256(path) != digest_value:
        _fail(f"logits SHA mismatch for {row['case_id']}")
    try:
        tensor = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        _fail(f"failed to load logits for {row['case_id']}: {exc}")
    if not isinstance(tensor, torch.Tensor):
        _fail(f"logits artifact is not a tensor for {row['case_id']}")
    if list(tensor.shape) != shape_value:
        _fail(f"logits shape mismatch for {row['case_id']}")
    if not torch.isfinite(tensor).all().item():
        _fail(f"non-finite logits for {row['case_id']}")
    return tensor


def _compare_logits(baseline, candidate, pair_id):
    if baseline.shape != candidate.shape:
        _fail(f"logits shape mismatch for pair {pair_id}")
    tolerance = (
        contract.LOGIT_ATOL
        + contract.LOGIT_RTOL * baseline.abs()
    )
    if not torch.all((candidate - baseline).abs() <= tolerance).item():
        _fail(f"logits mismatch for pair {pair_id}")


def _sum_counter(rows, field):
    return sum(int(row["kv_offload"][field]) for row in rows)


def _pair_movement_summary(baseline_rows, candidate_rows):
    baseline_h2d = _sum_counter(baseline_rows, "h2d_copies")
    candidate_h2d = _sum_counter(candidate_rows, "h2d_copies")
    baseline_evictions = _sum_counter(baseline_rows, "evictions")
    candidate_evictions = _sum_counter(candidate_rows, "evictions")
    return {
        "baseline_h2d_copies": baseline_h2d,
        "candidate_h2d_copies": candidate_h2d,
        "h2d_improvement": contract.movement_improvement(
            baseline_h2d,
            candidate_h2d,
        ),
        "h2d_regression": contract.movement_regression(
            baseline_h2d,
            candidate_h2d,
        ),
        "baseline_evictions": baseline_evictions,
        "candidate_evictions": candidate_evictions,
        "eviction_improvement": contract.movement_improvement(
            baseline_evictions,
            candidate_evictions,
        ),
        "eviction_regression": contract.movement_regression(
            baseline_evictions,
            candidate_evictions,
        ),
    }


def _median_decode_step(rows):
    values = [
        float(value)
        for row in rows
        for value in row["decode_step_ms"]
    ]
    if not values:
        _fail("measured decode latency samples missing")
    return statistics.median(values)


def _aggregate_measured(rows):
    by_domain = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["phase"] != "measured":
            continue
        key = (
            row["workload"],
            int(row["gpu_blocks"]),
            int(row["blockwise_blocks"]),
        )
        by_domain[key][row["policy"]].append(row)

    pair_summaries = {}
    pair_regressions_pass = True
    safety_pass = {
        "copy_waits_pass": True,
        "prefetch_plans_pass": True,
        "d2h_copies_pass": True,
        "d2h_bytes_pass": True,
        "evict_dirty_pass": True,
        "peak_resident_blocks_pass": True,
        "peak_cuda_allocated_bytes_pass": True,
        "peak_cuda_reserved_bytes_pass": True,
        "decode_latency_pass": True,
    }
    all_baseline = []
    all_candidate = []
    for key in (
        (workload, gpu_blocks, blockwise_blocks)
        for workload in contract.WORKLOADS
        for gpu_blocks, blockwise_blocks in contract.STAGING_SHAPES
    ):
        policies = by_domain.get(key, {})
        baseline_rows = policies.get("baseline", [])
        candidate_rows = policies.get("candidate", [])
        if (
            len(baseline_rows) != contract.MEASURED_REPETITIONS
            or len(candidate_rows) != contract.MEASURED_REPETITIONS
        ):
            _fail(f"measured repetition coverage mismatch for {key}")
        baseline_rows = sorted(
            baseline_rows,
            key=lambda row: row["repetition"],
        )
        candidate_rows = sorted(
            candidate_rows,
            key=lambda row: row["repetition"],
        )
        if {
            row["repetition"] for row in baseline_rows
        } != set(range(contract.MEASURED_REPETITIONS)):
            _fail(f"baseline measured repetitions mismatch for {key}")
        if {
            row["repetition"] for row in candidate_rows
        } != set(range(contract.MEASURED_REPETITIONS)):
            _fail(f"candidate measured repetitions mismatch for {key}")
        all_baseline.extend(baseline_rows)
        all_candidate.extend(candidate_rows)
        movement = _pair_movement_summary(
            baseline_rows,
            candidate_rows,
        )
        other_metric_pass = (
            movement["h2d_regression"]
            <= contract.THRESHOLDS["other_movement_max_regression"]
            and movement["eviction_regression"]
            <= contract.THRESHOLDS["other_movement_max_regression"]
        )
        pair_regressions_pass &= other_metric_pass

        counter_checks = {}
        for field in (
            "copy_waits",
            "prefetch_plans",
            "d2h_copies",
            "d2h_bytes",
            "evict_dirty",
        ):
            passed = _sum_counter(
                candidate_rows,
                field,
            ) <= _sum_counter(baseline_rows, field)
            counter_checks[f"{field}_pass"] = passed
            safety_pass[f"{field}_pass"] &= passed

        peak_checks = {}
        for field in (
            "peak_resident_blocks",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        ):
            baseline_peak = max(row[field] for row in baseline_rows)
            candidate_peak = max(row[field] for row in candidate_rows)
            passed = candidate_peak <= baseline_peak
            peak_checks[f"{field}_pass"] = passed
            safety_pass[f"{field}_pass"] &= passed

        baseline_latency = _median_decode_step(baseline_rows)
        candidate_latency = _median_decode_step(candidate_rows)
        latency_pass = (
            candidate_latency
            <= baseline_latency
            * (
                1.0
                + contract.THRESHOLDS[
                    "decode_latency_max_regression"
                ]
            )
        )
        safety_pass["decode_latency_pass"] &= latency_pass
        pair_id = (
            f"{key[0]}__g{key[1]}__w{key[2]}"
        )
        pair_summaries[pair_id] = {
            **movement,
            **counter_checks,
            **peak_checks,
            "baseline_median_decode_step_ms": baseline_latency,
            "candidate_median_decode_step_ms": candidate_latency,
            "decode_latency_pass": latency_pass,
        }

    aggregate = _pair_movement_summary(
        all_baseline,
        all_candidate,
    )
    low_capacity_improvements = [
        max(
            summary["h2d_improvement"],
            summary["eviction_improvement"],
        )
        for pair_id, summary in pair_summaries.items()
        if "__g2__" in pair_id
    ]
    multi_prompt_baseline = [
        row for row in all_baseline
        if row["workload"] == "multi_prompt_thrash"
    ]
    multi_prompt_candidate = [
        row for row in all_candidate
        if row["workload"] == "multi_prompt_thrash"
    ]
    multi_prompt = _pair_movement_summary(
        multi_prompt_baseline,
        multi_prompt_candidate,
    )
    ratios = {
        "valid": True,
        "h2d_improvement": aggregate["h2d_improvement"],
        "eviction_improvement": aggregate[
            "eviction_improvement"
        ],
        "h2d_regression": aggregate["h2d_regression"],
        "eviction_regression": aggregate["eviction_regression"],
        "low_capacity_movement_improvement": max(
            low_capacity_improvements,
            default=0.0,
        ),
        "multi_prompt_movement_improvement": max(
            multi_prompt["h2d_improvement"],
            multi_prompt["eviction_improvement"],
        ),
        "pair_regressions_pass": pair_regressions_pass,
        **safety_pass,
    }
    return {
        "ratios": ratios,
        "aggregate": aggregate,
        "multi_prompt": multi_prompt,
        "pairs": pair_summaries,
    }


def _report_markdown(report):
    lines = [
        "# KV Decode Residency Planner Gate",
        "",
        f"- Classification: `{report['classification']}`",
        f"- Verified cases: `{report['verified_case_count']}`",
        (
            "- Aggregate H2D improvement: "
            f"`{report['ratios']['h2d_improvement']:.4%}`"
        ),
        (
            "- Aggregate eviction improvement: "
            f"`{report['ratios']['eviction_improvement']:.4%}`"
        ),
        (
            "- Low-capacity movement improvement: "
            f"`{report['ratios']['low_capacity_movement_improvement']:.4%}`"
        ),
        (
            "- Multi-prompt movement improvement: "
            f"`{report['ratios']['multi_prompt_movement_improvement']:.4%}`"
        ),
        "",
        "## Pair Results",
        "",
    ]
    for pair_id, summary in sorted(report["pairs"].items()):
        lines.append(
            f"- `{pair_id}`: H2D "
            f"{summary['h2d_improvement']:.4%}, evictions "
            f"{summary['eviction_improvement']:.4%}, latency "
            f"{summary['candidate_median_decode_step_ms']:.4f} ms "
            f"vs {summary['baseline_median_decode_step_ms']:.4f} ms"
        )
    return "\n".join(lines) + "\n"


def verify_run(run_dir, write_report=False):
    run_dir = Path(run_dir)
    for name in (
        "manifest.json",
        "environment.json",
        "source_manifest.json",
        "worker_logs_manifest.json",
        "case_rows.jsonl",
        "summary.json",
    ):
        if not (run_dir / name).is_file():
            _fail(f"required raw artifact missing: {name}")

    manifest = _load_json(run_dir / "manifest.json")
    environment = _load_json(run_dir / "environment.json")
    source_manifest = _load_json(run_dir / "source_manifest.json")
    worker_logs_manifest = _load_json(
        run_dir / "worker_logs_manifest.json"
    )
    rows = _load_jsonl(run_dir / "case_rows.jsonl")
    raw_summary = _load_json(run_dir / "summary.json")
    expected_summary = build_raw_summary(rows)
    if raw_summary != expected_summary:
        _fail("summary/raw disagreement")

    expected_matrix = contract.build_case_matrix()
    expected_case_ids = {case.case_id for case in expected_matrix}
    expected_by_case_id = {
        case.case_id: case
        for case in expected_matrix
    }
    row_ids = set()
    rows_by_case_id = {}
    port_values = set()
    for row in rows:
        _require_exact_row_schema(row)
        row_id = row["row_id"]
        if row_id in row_ids:
            _fail(f"duplicate row_id: {row_id}")
        row_ids.add(row_id)
        case_id = row["case_id"]
        if case_id in rows_by_case_id:
            _fail(f"duplicate case_id: {case_id}")
        rows_by_case_id[case_id] = row

    actual_case_ids = set(rows_by_case_id)
    missing = sorted(expected_case_ids - actual_case_ids)
    extra = sorted(actual_case_ids - expected_case_ids)
    if missing:
        _fail(f"missing case IDs: {missing}")
    if extra:
        _fail(f"unexpected case IDs: {extra}")

    if manifest.get("complete") is not True:
        _fail("manifest is not complete")
    if manifest.get("expected_case_ids") != [
        case.case_id for case in expected_matrix
    ]:
        _fail("manifest case domain mismatch")
    source_by_policy = source_manifest.get(
        "source_sha256_by_policy"
    )
    if source_by_policy != manifest.get("source_sha256_by_policy"):
        _fail("source manifest mismatch")
    prompt_by_workload = manifest.get(
        "prompt_sha256_by_workload"
    )
    for field in (
        "cuda_visible_devices",
        "model_path",
        "python_path",
    ):
        if environment.get(field) != manifest.get(field):
            _fail(f"environment mismatch for {field}")
    if environment.get("cuda_visible_devices") != "0":
        _fail("CUDA_VISIBLE_DEVICES must be 0")

    for row in rows:
        expected = expected_by_case_id[row["case_id"]]
        for field in (
            "policy",
            "workload",
            "gpu_blocks",
            "blockwise_blocks",
            "repetition",
            "phase",
            "warmup",
        ):
            if row[field] != getattr(expected, field):
                _fail(
                    f"case metadata mismatch for "
                    f"{row['case_id']} field {field}"
                )
        if row["complete"] is not True:
            _fail(f"incomplete case row: {row['case_id']}")
        if row["source_sha256"] != source_by_policy[row["policy"]]:
            _fail(f"source SHA mismatch for {row['case_id']}")
        if row["cuda_visible_devices"] != "0":
            _fail(f"CUDA_VISIBLE_DEVICES must be 0 for {row['case_id']}")
        for field in ("model_path", "python_path"):
            if row[field] != environment[field]:
                _fail(
                    f"environment mismatch for {row['case_id']} "
                    f"field {field}"
                )
        if row["prompt_sha256"] != prompt_by_workload[row["workload"]]:
            _fail(f"prompt SHA mismatch for {row['case_id']}")
        _require_nonnegative_integer(
            row["worker_pid"],
            f"worker_pid for {row['case_id']}",
        )
        for field in ("tinyvllm_dist_port", "master_port"):
            value = row[field]
            _require_nonnegative_integer(
                value,
                f"{field} for {row['case_id']}",
            )
            if value in port_values:
                _fail("ports must be globally unique")
            port_values.add(value)
        if row["tinyvllm_dist_port"] == row["master_port"]:
            _fail("ports must be globally unique")
        if (
            row["phase"] in ("correctness", "measured")
            and not row["decoded_token_ids"]
        ):
            _fail(f"decoded tokens missing for {row['case_id']}")
        if not isinstance(row["decoded_token_ids"], list):
            _fail(f"decoded tokens invalid for {row['case_id']}")
        for token in row["decoded_token_ids"]:
            _require_nonnegative_integer(
                token,
                f"decoded token for {row['case_id']}",
            )
        if row["phase"] == "correctness":
            _validate_logits(run_dir, row)
        elif any(
            row[field] is not None
            for field in (
                "decode_logits_path",
                "decode_logits_sha256",
                "decode_logits_shape",
            )
        ):
            _fail(
                f"unexpected logits artifact for {row['case_id']}"
            )
        if not isinstance(row["decode_step_ms"], list):
            _fail(f"decode latency samples invalid for {row['case_id']}")
        for value in row["decode_step_ms"]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                _fail(
                    f"non-finite decode latency for {row['case_id']}"
                )
        if row["phase"] == "measured" and not row["decode_step_ms"]:
            _fail(f"measured decode latency missing for {row['case_id']}")
        for field in (
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
            "peak_resident_blocks",
        ):
            _require_nonnegative_integer(
                row[field],
                f"{field} for {row['case_id']}",
            )
        if not isinstance(row["kv_offload"], dict):
            _fail(f"KV counters missing for {row['case_id']}")
        for field in contract.KV_COUNTER_FIELDS:
            if field not in row["kv_offload"]:
                _fail(
                    f"missing KV counter {field} for {row['case_id']}"
                )
            _require_nonnegative_integer(
                row["kv_offload"][field],
                f"KV counter {field} for {row['case_id']}",
            )
        if not isinstance(row["planner"], dict):
            _fail(f"planner counters missing for {row['case_id']}")
        for field in contract.PLANNER_COUNTER_FIELDS:
            if field not in row["planner"]:
                _fail(
                    f"missing planner counter {field} "
                    f"for {row['case_id']}"
                )
            _require_nonnegative_integer(
                row["planner"][field],
                f"planner counter {field} for {row['case_id']}",
            )

    processed_pairs = set()
    for case in expected_matrix:
        if case.policy != "baseline":
            continue
        if case.pair_id in processed_pairs:
            continue
        processed_pairs.add(case.pair_id)
        baseline, candidate = _pair_rows(
            rows_by_case_id,
            case,
        )
        if (
            case.phase in ("correctness", "measured")
            and baseline["decoded_token_ids"]
            != candidate["decoded_token_ids"]
        ):
            _fail(f"decoded token mismatch for pair {case.pair_id}")
        if case.phase == "correctness":
            _compare_logits(
                _validate_logits(run_dir, baseline),
                _validate_logits(run_dir, candidate),
                case.pair_id,
            )

    for log in worker_logs_manifest.get("logs", []):
        path_value = log.get("path")
        digest_value = log.get("sha256")
        if not path_value or not digest_value:
            _fail("worker log manifest entry incomplete")
        path = run_dir / path_value
        if not path.is_file():
            _fail(f"worker log missing: {path_value}")
        if _sha256(path) != digest_value:
            _fail(f"worker log SHA mismatch: {path_value}")

    aggregates = _aggregate_measured(rows)
    classification = contract.classify_ratios(
        aggregates["ratios"]
    )
    report = {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": classification,
        "verified_case_count": len(rows),
        "ratios": aggregates["ratios"],
        "aggregate": aggregates["aggregate"],
        "multi_prompt": aggregates["multi_prompt"],
        "pairs": aggregates["pairs"],
        "raw_summary": expected_summary,
    }
    if write_report:
        _atomic_write_json(
            run_dir / "independent_verification.json",
            report,
        )
        _atomic_write(
            run_dir / "report.md",
            _report_markdown(report).encode("utf-8"),
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify_run(
            args.run_dir,
            write_report=args.write_report,
        )
    except VerificationError as exc:
        print(f"INVALID: {exc}")
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
