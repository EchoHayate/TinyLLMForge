#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICIES = ("recompute", "exact_restore")
LEGACY_ALL_GATHER = "replicated_weight_row_parallel_all_gather"
ROW_PARALLEL_ALL_REDUCE = "row_parallel_all_reduce"


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to load JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _speedup_percent(baseline, candidate, name: str) -> float:
    baseline = _positive_number(baseline, f"{name} baseline")
    candidate = _positive_number(candidate, f"{name} candidate")
    return 100.0 * (baseline - candidate) / baseline


def _regression_percent(baseline, candidate, name: str) -> float:
    return -_speedup_percent(baseline, candidate, name)


def _load_case_rows(path: Path) -> tuple[dict, ...]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"failed to load case rows: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid case row at {path}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"case row must be an object at {path}:{line_number}"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"case rows are empty: {path}")
    return tuple(rows)


def _load_attempt(root) -> dict:
    root = Path(root).resolve()
    summary = _load_json(root / "decode_summary.json")
    if summary.get("schema_version") != (
        "qwen35.tp4-decode-internal-summary.v1"
    ):
        raise ValueError("decode summary schema is invalid")
    measured_pairs = summary.get("measured_pairs")
    if (
        isinstance(measured_pairs, bool)
        or not isinstance(measured_pairs, int)
        or measured_pairs < 3
    ):
        raise ValueError("decode summary requires at least 3 measured pairs")
    by_policy = summary.get("by_policy")
    if not isinstance(by_policy, dict):
        raise ValueError("decode summary by_policy is invalid")
    for policy in POLICIES:
        metrics = by_policy.get(policy)
        if not isinstance(metrics, dict):
            raise ValueError(f"decode summary lacks policy {policy}")
        for field in (
            "median_steady_wall_ns",
            "median_steady_cuda_ns",
            "median_collective_cuda_ns",
            "median_steady_wall_p90_ns",
            "median_steady_cuda_p90_ns",
        ):
            _positive_number(metrics.get(field), f"{policy} {field}")

    cases_root = root / "download" / "cases"
    profiles = sorted(cases_root.glob("*/decode_profile.json"))
    if not profiles:
        raise ValueError("attempt has no decode profiles")
    operation_counts = {}
    outputs = {}
    measured_repetitions = {policy: set() for policy in POLICIES}
    rank_sets = []
    for profile_path in profiles:
        profile = _load_json(profile_path)
        if profile.get("phase") != "measured":
            continue
        policy = profile.get("policy")
        repetition = profile.get("repetition")
        if policy not in POLICIES:
            raise ValueError("measured profile policy is invalid")
        if (
            isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition < 0
        ):
            raise ValueError("measured profile repetition is invalid")
        measured_repetitions[policy].add(repetition)
        ranks = profile.get("ranks")
        if not isinstance(ranks, list):
            raise ValueError("decode profile ranks are invalid")
        rank_set = set()
        for rank_payload in ranks:
            if not isinstance(rank_payload, dict):
                raise ValueError("decode profile rank payload is invalid")
            rank = rank_payload.get("rank")
            rank_set.add(rank)
            collectives = rank_payload.get("collectives")
            if not isinstance(collectives, list):
                raise ValueError("decode profile collectives are invalid")
            for collective in collectives:
                if not isinstance(collective, dict):
                    raise ValueError("collective row is invalid")
                operation = collective.get("operation")
                if not isinstance(operation, str) or not operation:
                    raise ValueError("collective operation is invalid")
                operation_counts[operation] = (
                    operation_counts.get(operation, 0) + 1
                )
        rank_sets.append(rank_set)

        rows_path = profile_path.with_name("case_rows.jsonl")
        for row in _load_case_rows(rows_path):
            if row.get("phase") != "measured":
                continue
            request_id = row.get("request_id")
            token_ids = row.get("output_token_ids")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("case row request_id is invalid")
            if (
                not isinstance(token_ids, list)
                or not all(
                    isinstance(token_id, int)
                    and not isinstance(token_id, bool)
                    for token_id in token_ids
                )
            ):
                raise ValueError("case row output_token_ids are invalid")
            key = (policy, repetition, request_id)
            if key in outputs:
                raise ValueError(f"duplicate output row: {key}")
            outputs[key] = tuple(token_ids)

    if not outputs:
        raise ValueError("attempt has no measured output rows")
    for policy, repetitions in measured_repetitions.items():
        if len(repetitions) < 3:
            raise ValueError(
                f"attempt has fewer than 3 measured {policy} repetitions"
            )
    if not rank_sets or any(rank_set != {0, 1, 2, 3} for rank_set in rank_sets):
        raise ValueError("every measured profile must contain TP ranks 0,1,2,3")
    return {
        "root": root,
        "summary": summary,
        "operation_counts": operation_counts,
        "outputs": outputs,
    }


def compare_decode_attempts(
    baseline_root,
    candidate_root,
    *,
    minimum_speedup_percent=5.0,
    maximum_tail_regression_percent=2.0,
):
    minimum_speedup_percent = _positive_number(
        minimum_speedup_percent,
        "minimum_speedup_percent",
    )
    maximum_tail_regression_percent = _positive_number(
        maximum_tail_regression_percent,
        "maximum_tail_regression_percent",
    )
    baseline = _load_attempt(baseline_root)
    candidate = _load_attempt(candidate_root)
    baseline_outputs = baseline["outputs"]
    candidate_outputs = candidate["outputs"]
    output_keys_match = baseline_outputs.keys() == candidate_outputs.keys()
    output_parity = (
        output_keys_match
        and all(
            baseline_outputs[key] == candidate_outputs[key]
            for key in baseline_outputs
        )
    )

    per_policy = {}
    for policy in POLICIES:
        baseline_metrics = baseline["summary"]["by_policy"][policy]
        candidate_metrics = candidate["summary"]["by_policy"][policy]
        per_policy[policy] = {
            "steady_wall_speedup_percent": _speedup_percent(
                baseline_metrics["median_steady_wall_ns"],
                candidate_metrics["median_steady_wall_ns"],
                f"{policy} steady wall",
            ),
            "steady_cuda_speedup_percent": _speedup_percent(
                baseline_metrics["median_steady_cuda_ns"],
                candidate_metrics["median_steady_cuda_ns"],
                f"{policy} steady CUDA",
            ),
            "collective_cuda_speedup_percent": _speedup_percent(
                baseline_metrics["median_collective_cuda_ns"],
                candidate_metrics["median_collective_cuda_ns"],
                f"{policy} collective CUDA",
            ),
            "steady_wall_p90_regression_percent": _regression_percent(
                baseline_metrics["median_steady_wall_p90_ns"],
                candidate_metrics["median_steady_wall_p90_ns"],
                f"{policy} steady wall p90",
            ),
            "steady_cuda_p90_regression_percent": _regression_percent(
                baseline_metrics["median_steady_cuda_p90_ns"],
                candidate_metrics["median_steady_cuda_p90_ns"],
                f"{policy} steady CUDA p90",
            ),
        }

    wall_speedups = [
        metrics["steady_wall_speedup_percent"]
        for metrics in per_policy.values()
    ]
    cuda_speedups = [
        metrics["steady_cuda_speedup_percent"]
        for metrics in per_policy.values()
    ]
    collective_speedups = [
        metrics["collective_cuda_speedup_percent"]
        for metrics in per_policy.values()
    ]
    tail_regressions = [
        metrics[field]
        for metrics in per_policy.values()
        for field in (
            "steady_wall_p90_regression_percent",
            "steady_cuda_p90_regression_percent",
        )
    ]
    baseline_operations = baseline["operation_counts"]
    candidate_operations = candidate["operation_counts"]
    baseline_legacy = baseline_operations.get(LEGACY_ALL_GATHER, 0)
    candidate_legacy = candidate_operations.get(LEGACY_ALL_GATHER, 0)
    baseline_row_reduce = baseline_operations.get(
        ROW_PARALLEL_ALL_REDUCE,
        0,
    )
    candidate_row_reduce = candidate_operations.get(
        ROW_PARALLEL_ALL_REDUCE,
        0,
    )

    reasons = []
    hard_failure = False
    if candidate_legacy:
        hard_failure = True
        reasons.append(
            "candidate still contains legacy AllGather collective rows"
        )
    if candidate_row_reduce <= 0:
        hard_failure = True
        reasons.append(
            "candidate lacks row-parallel AllReduce collective rows"
        )
    if not output_parity:
        hard_failure = True
        reasons.append("controlled output token parity failed")
    if min(wall_speedups) < -maximum_tail_regression_percent:
        hard_failure = True
        reasons.append("steady decode wall median materially regressed")
    if min(cuda_speedups) < -maximum_tail_regression_percent:
        hard_failure = True
        reasons.append("steady decode CUDA median materially regressed")

    if hard_failure:
        classification = "NO_GO"
    elif (
        min(wall_speedups) >= minimum_speedup_percent
        and min(cuda_speedups) > 0
        and max(tail_regressions) <= maximum_tail_regression_percent
    ):
        classification = "PERFORMANCE_PASS"
        reasons.append(
            "both policies meet the steady decode speedup and tail gates"
        )
    else:
        classification = "STRUCTURAL_ONLY"
        if min(wall_speedups) < minimum_speedup_percent:
            reasons.append(
                "steady decode wall speedup is below the performance gate"
            )
        if min(cuda_speedups) <= 0:
            reasons.append(
                "steady decode CUDA did not improve for both policies"
            )
        if max(tail_regressions) > maximum_tail_regression_percent:
            reasons.append("a reported p90 metric exceeds the tail gate")

    return {
        "schema_version": "qwen35.decode-row-parallel-comparison.v1",
        "baseline_root": str(baseline["root"]),
        "candidate_root": str(candidate["root"]),
        "legacy_all_gather_rows": {
            "baseline": baseline_legacy,
            "candidate": candidate_legacy,
        },
        "row_parallel_all_reduce_rows": {
            "baseline": baseline_row_reduce,
            "candidate": candidate_row_reduce,
        },
        "output_parity": output_parity,
        "output_key_sets_match": output_keys_match,
        "steady_decode_wall_speedup_percent": min(wall_speedups),
        "steady_decode_cuda_speedup_percent": min(cuda_speedups),
        "collective_cuda_speedup_percent": min(collective_speedups),
        "maximum_tail_regression_percent": max(tail_regressions),
        "per_policy": per_policy,
        "thresholds": {
            "minimum_speedup_percent": minimum_speedup_percent,
            "maximum_tail_regression_percent": (
                maximum_tail_regression_percent
            ),
        },
        "classification": classification,
        "reasons": reasons,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compare_decode_attempts(
        args.baseline,
        args.candidate,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(result["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
