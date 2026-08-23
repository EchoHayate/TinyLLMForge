#!/usr/bin/env python3
"""Independent verifier for exact-burst ragged coalescing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


CASE_SCHEMA = "exact-burst-ragged-coalescing.case.v1"
CORRECTNESS_SCHEMA = (
    "exact-burst-ragged-coalescing.correctness.v1"
)
SUMMARY_SCHEMA = "exact-burst-ragged-coalescing.summary.v1"
SOURCE_SCHEMA = "exact-burst-ragged-coalescing.source.v1"
WORKLOAD_SCHEMA = "exact-burst-ragged-coalescing.workload.v1"
COMPARISON_SCHEMA = (
    "exact-burst-ragged-coalescing.comparison.v1"
)
GATE_SCHEMA = "exact-burst-ragged-coalescing.gate.v1"
MANIFEST_SCHEMA = (
    "exact-burst-ragged-coalescing.manifest.v1"
)
VERIFICATION_SCHEMA = (
    "exact-burst-ragged-coalescing."
    "independent-verification.v1"
)
TRACE_IDENTITY = (
    "gate-only-exact-burst-ragged-coalescing-correctness-v1"
)
CONTEXTS = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
POLICIES = (
    "decode_burst_k4",
    "decode_burst_k8_split_phase",
    "decode_burst_k8_split_phase_ragged",
)
REFERENCE_POLICIES = (
    "decode_burst_k4",
    "decode_burst_k8_split_phase",
)
K4_POLICY = "decode_burst_k4"
SPLIT_POLICY = "decode_burst_k8_split_phase"
CANDIDATE = "decode_burst_k8_split_phase_ragged"
POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
POLICY_CONFIGS = {
    K4_POLICY: {
        "enabled": True,
        "split": False,
        "width": 4,
        "selectable": False,
        "entrypoint": "production",
    },
    SPLIT_POLICY: {
        "enabled": True,
        "split": True,
        "width": 8,
        "selectable": True,
        "entrypoint": "production",
        "profile_ordinary_tail_after_full_bursts": True,
        "scheduler_only_fallback_reasons": (
            "insufficient_output_budget",
        ),
        "correctness_sampled_logit_d2h_calls": 2,
        "ordinary_tail_sampling_points": ("decode-final",),
    },
    CANDIDATE: {
        "enabled": True,
        "split": True,
        "ragged_coalescing": True,
        "width": 8,
        "selectable": True,
        "entrypoint": "production",
        "correctness_sampled_logit_d2h_calls": 3,
        "ordinary_tail_sampling_points": (),
    },
}
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/exact_greedy_decode_burst_split_phase.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/profile_exact_burst_split_phase.py",
    "tools/profile_exact_burst_ragged_coalescing.py",
    "tools/test_profile_exact_burst_ragged_coalescing.py",
    "tools/exact_burst_ragged_coalescing_gate.py",
    "tools/test_exact_burst_ragged_coalescing_gate.py",
    "tools/exact_burst_ragged_coalescing_verify.py",
    "tools/test_exact_burst_ragged_coalescing_verify.py",
    "tools/run_exact_burst_ragged_coalescing_remote.py",
    "tools/test_run_exact_burst_ragged_coalescing_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)
PRIMARY_ARTIFACTS = {
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "source.patch",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
}
STAGE1_MODEL_BASENAMES = ("Qwen3-0.6B", "Qwen3-0___6B")

LOGIT_MAX_LIMIT = 0.25
LOGIT_MEAN_LIMIT = 0.05
TAIL_IMPROVEMENT_MINIMUM = 0.10
TPOT_REGRESSION_LIMIT = 0.01
THROUGHPUT_REGRESSION_LIMIT = 0.01
BUCKET_TPOT_REGRESSION_LIMIT = 0.02
BUCKET_E2E_REGRESSION_LIMIT = 0.02
BUCKET_TTFT_REGRESSION_LIMIT = 0.03
MEMORY_REGRESSION_LIMIT = 0.03
MEDIAN_GAP_REGRESSION_LIMIT = 0.03
MAXIMUM_GAP_REGRESSION_LIMIT = 0.05
METRIC_TOLERANCE = 1e-9


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path.name}")
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(line, parse_constant=_reject_constant)
            for line in handle
            if line.strip()
        ]


def _write_json(path: Path, payload) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _non_negative_integer(value, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _valid_digest(value, name: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile input is empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _regression(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError("relative comparison baseline must be positive")
    return (candidate - baseline) / baseline


def _throughput_regression(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError("throughput baseline must be positive")
    return (baseline - candidate) / baseline


def _policy_order(repetition: int, context_index: int) -> tuple[str, ...]:
    offset = repetition % len(POLICIES)
    order = POLICIES[offset:] + POLICIES[:offset]
    return tuple(reversed(order)) if context_index % 2 else order


def _read_sidecar(run_dir: Path, row: dict) -> tuple[float, ...]:
    relative = row.get("logits_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("logits path is invalid")
    path = run_dir / relative
    payload = path.read_bytes()
    expected_bytes = _non_negative_integer(
        row.get("logits_byte_length"),
        "logits byte length",
    )
    expected_count = _non_negative_integer(
        row.get("logits_element_count"),
        "logits element count",
    )
    if len(payload) != expected_bytes:
        raise ValueError("logits byte length mismatch")
    if _sha256_file(path) != row.get("logits_sha256"):
        raise ValueError("logits digest mismatch")
    if expected_bytes != expected_count * 4:
        raise ValueError("logits element inventory mismatch")
    return struct.unpack(f"<{expected_count}f", payload)


def _validate_manifest(run_dir: Path, manifest: dict) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("manifest schema mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("manifest artifact inventory is invalid")
    if not PRIMARY_ARTIFACTS <= set(artifacts):
        raise ValueError("manifest artifact inventory is incomplete")
    for relative, expected_digest in artifacts.items():
        _valid_digest(expected_digest, f"manifest digest {relative}")
        if _sha256_file(run_dir / relative) != expected_digest:
            raise ValueError(
                f"manifest digest mismatch: {relative}"
            )


def _validate_source(
    source: dict,
    *,
    repo_root: Path,
) -> None:
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    digests = source.get("source_sha256")
    if not isinstance(digests, dict) or set(digests) != set(
        SOURCE_FILES
    ):
        raise ValueError("source manifest file inventory mismatch")
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        if _sha256_file(path) != digests[relative]:
            raise ValueError(f"source digest mismatch: {relative}")


def _validate_workload(workload: dict) -> None:
    if workload.get("schema_version") != WORKLOAD_SCHEMA:
        raise ValueError("workload manifest schema mismatch")
    expected = {
        "context_cases": [
            {
                "context_bucket": bucket,
                "prompt_tokens": prompt,
                "generated_tokens": generated,
            }
            for bucket, prompt, generated in CONTEXTS
        ],
        "generated_tokens": 128,
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "tensor_parallel_size": 1,
        "performance_row_count": 45,
        "correctness_row_count": 36,
        "performance_correctness_trace": False,
        "correctness_trace_identity": TRACE_IDENTITY,
        "correctness_sampling_points": list(POINTS),
        "policy_configs": {
            policy: json.loads(json.dumps(config))
            for policy, config in POLICY_CONFIGS.items()
        },
        "policy_order": {
            str(repetition): {
                bucket: list(
                    _policy_order(repetition, context_index)
                )
                for context_index, (
                    bucket,
                    _prompt,
                    _generated,
                ) in enumerate(CONTEXTS)
            }
            for repetition in range(5)
        },
    }
    for field, expected_value in expected.items():
        if workload.get(field) != expected_value:
            raise ValueError(f"workload manifest mismatch: {field}")
    model = workload.get("model")
    if (
        not isinstance(model, str)
        or Path(model).name not in STAGE1_MODEL_BASENAMES
    ):
        raise ValueError("workload manifest mismatch: model")
    if workload.get("gpu_memory_utilization") != 0.5:
        raise ValueError(
            "workload manifest mismatch: gpu_memory_utilization"
        )
    environment = workload.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("torch_available") is not True
        or environment.get("cuda_available") is not True
    ):
        raise ValueError("workload environment is invalid")


def _validate_candidate_summary(summary: dict) -> None:
    expected = {
        "attempts": 17,
        "acceptances": 17,
        "commits": 17,
        "committed_tokens": 127,
        "target_model_forwards": 127,
        "graph_replays": 127,
        "final_token_d2h_calls": 2,
        "final_token_d2h_bytes": 56,
        "prefix_commits": 15,
        "suffix_commits": 15,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
    }
    for field, expected_value in expected.items():
        if summary.get(field) != expected_value:
            raise ValueError(
                f"candidate lifecycle mismatch: {field}"
            )
    histogram = {"3": 1, "4": 1, "8": 15}
    if (
        summary.get("requested_width_histogram") != histogram
        or summary.get("authorized_width_histogram") != histogram
        or summary.get("fallback_counts") != {}
        or summary.get("split_phase_failure_counts") != {}
    ):
        raise ValueError("candidate width ownership mismatch")


def _validate_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != 45:
        raise ValueError(
            f"expected exactly 45 measured rows, got {len(rows)}"
        )
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
        for policy in POLICIES
    }
    identities = []
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("performance row is invalid")
        if row.get("schema_version") != CASE_SCHEMA:
            raise ValueError("performance row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("repetition"),
            row.get("policy"),
        )
        identities.append(identity)
        bucket = identity[0]
        shape = {
            name: (prompt, generated)
            for name, prompt, generated in CONTEXTS
        }.get(bucket)
        if shape is None or (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != shape:
            raise ValueError("performance context shape mismatch")
        samples = row.get("amortized_tpot_samples_ns")
        if not isinstance(samples, list) or len(samples) != 127:
            raise ValueError("TPOT sample inventory mismatch")
        for index, value in enumerate(samples):
            if _finite_number(value, f"TPOT sample {index}") < 0:
                raise ValueError("TPOT sample must be non-negative")
        expected_tail = sum(float(value) for value in samples[-7:])
        if _finite_number(
            row.get("tail_seven_elapsed_ns"),
            "tail-seven latency",
        ) != expected_tail:
            raise ValueError("tail-seven latency mismatch")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "maximum_host_visible_burst_gap_ns",
            "capture_retained_static_bytes",
        ):
            if _finite_number(row.get(field), field) < 0:
                raise ValueError(f"{field} must be non-negative")
        output_ids = row.get("output_token_ids")
        if not isinstance(output_ids, list) or len(output_ids) != 128:
            raise ValueError("performance output inventory mismatch")
        _valid_digest(
            row.get("output_text_sha256"),
            "performance output digest",
        )
        _valid_digest(
            row.get("source_commit"),
            "performance source commit",
            lengths=(40, 64),
        )
        if row["policy"] == CANDIDATE:
            summary = row.get(
                "exact_greedy_decode_burst_summary"
            )
            if not isinstance(summary, dict):
                raise ValueError("candidate summary is invalid")
            _validate_candidate_summary(summary)
            inventory = row.get("split_phase_inventory")
            if (
                not isinstance(inventory, dict)
                or inventory.get("parent_lease_count") != 15
                or inventory.get("prefix_row_count") != 15
                or inventory.get("suffix_row_count") != 15
                or inventory.get("unexpected_scheduler_calls") != 0
                or len(row.get("host_visible_burst_gaps_ns", ()))
                != 32
            ):
                raise ValueError(
                    "candidate split inventory mismatch"
                )
        normalized.append(dict(row))
    if set(identities) != expected:
        raise ValueError("measured case inventory is incomplete")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate case identity")
    return normalized


def _validate_correctness(
    rows: list[dict],
    *,
    run_dir: Path,
) -> tuple[list[dict], dict, bool]:
    if len(rows) != 36:
        raise ValueError(
            f"expected exactly 36 correctness rows, got {len(rows)}"
        )
    expected = {
        (bucket, policy, point)
        for bucket, _prompt, _generated in CONTEXTS
        for policy in POLICIES
        for point in POINTS
    }
    identities = []
    by_identity = {}
    for row in rows:
        if row.get("schema_version") != CORRECTNESS_SCHEMA:
            raise ValueError("correctness row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        identities.append(identity)
        if row.get("trace_identity") != TRACE_IDENTITY:
            raise ValueError("correctness trace identity mismatch")
        if row.get("correctness_trace") is not True:
            raise ValueError("correctness trace is disabled")
        _valid_digest(
            row.get("source_commit"),
            "correctness source commit",
            lengths=(40, 64),
        )
        output_ids = row.get("output_token_ids")
        if not isinstance(output_ids, list) or len(output_ids) != 128:
            raise ValueError("correctness output inventory mismatch")
        _valid_digest(
            row.get("output_text_sha256"),
            "correctness output digest",
        )
        values = _read_sidecar(run_dir, row)
        by_identity[identity] = (row, values)
    if set(identities) != expected:
        raise ValueError("correctness row inventory is incomplete")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate correctness identity")

    pairs = []
    global_max = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_ids = True
    all_text = True
    all_passed = True
    for baseline in REFERENCE_POLICIES:
        for bucket, _prompt, _generated in CONTEXTS:
            for point in POINTS:
                left, left_values = by_identity[
                    (bucket, baseline, point)
                ]
                right, right_values = by_identity[
                    (bucket, CANDIDATE, point)
                ]
                if len(left_values) != len(right_values):
                    raise ValueError("paired logits shape mismatch")
                differences = [
                    abs(left_value - right_value)
                    for left_value, right_value
                    in zip(left_values, right_values)
                ]
                maximum = max(differences, default=0.0)
                mean = (
                    sum(differences) / len(differences)
                    if differences else 0.0
                )
                argmax_equal = (
                    max(
                        range(len(left_values)),
                        key=left_values.__getitem__,
                    )
                    == max(
                        range(len(right_values)),
                        key=right_values.__getitem__,
                    )
                )
                ids_equal = (
                    left["output_token_ids"]
                    == right["output_token_ids"]
                )
                text_equal = (
                    left["output_text_sha256"]
                    == right["output_text_sha256"]
                )
                passed = all((
                    maximum <= LOGIT_MAX_LIMIT,
                    mean <= LOGIT_MEAN_LIMIT,
                    argmax_equal,
                    ids_equal,
                    text_equal,
                ))
                global_max = max(global_max, maximum)
                total_abs += sum(differences)
                total_count += len(differences)
                all_argmax &= argmax_equal
                all_ids &= ids_equal
                all_text &= text_equal
                all_passed &= passed
                pairs.append({
                    "baseline_policy": baseline,
                    "candidate_policy": CANDIDATE,
                    "context_bucket": bucket,
                    "sampling_point": point,
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "argmax_equal": argmax_equal,
                    "output_ids_exact": ids_equal,
                    "output_text_exact": text_equal,
                    "passed": passed,
                })
    metrics = {
        "pair_count": len(pairs),
        "max_abs": global_max,
        "mean_abs": (
            total_abs / total_count if total_count else 0.0
        ),
        "argmax_equal": all_argmax,
        "output_ids_exact": all_ids,
        "output_text_exact": all_text,
        "pairs": pairs,
    }
    return rows, metrics, all_passed


def _paired_rows(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
) -> list[tuple[dict, dict]]:
    def identity(row: dict) -> tuple[str, int]:
        return row["context_bucket"], row["repetition"]

    baseline = {identity(row): row for row in baseline_rows}
    candidate = {identity(row): row for row in candidate_rows}
    if (
        len(baseline) != len(baseline_rows)
        or len(candidate) != len(candidate_rows)
        or set(baseline) != set(candidate)
    ):
        raise ValueError("paired request inventory mismatch")
    return [(baseline[key], candidate[key]) for key in sorted(baseline)]


def _row_tpot(row: dict, percentile: float) -> float:
    values = row["amortized_tpot_samples_ns"]
    if percentile == 0.5:
        return float(statistics.median(values))
    return _nearest_rank(values, percentile)


def _metric_summary(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    baseline_policy: str,
) -> dict:
    pairs = _paired_rows(baseline_rows, candidate_rows)

    def paired_regression(field) -> float:
        return statistics.median(
            _regression(field(left), field(right))
            for left, right in pairs
        )

    def paired_throughput(field) -> float:
        return statistics.median(
            _throughput_regression(field(left), field(right))
            for left, right in pairs
        )

    baseline_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in baseline_rows
    )
    candidate_allocated = max(
        int(row["cuda_peak_allocated_bytes"])
        for row in candidate_rows
    )
    baseline_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in baseline_rows
    )
    candidate_reserved = max(
        int(row["cuda_peak_reserved_bytes"])
        for row in candidate_rows
    )
    return {
        "baseline_policy": baseline_policy,
        "candidate_policy": CANDIDATE,
        "sample_count_per_policy": len(pairs),
        "tpot_median_regression_fraction": paired_regression(
            lambda row: _row_tpot(row, 0.5)
        ),
        "tpot_p95_regression_fraction": paired_regression(
            lambda row: _row_tpot(row, 0.95)
        ),
        "ttft_regression_fraction": paired_regression(
            lambda row: float(row["ttft_ns"])
        ),
        "e2e_regression_fraction": paired_regression(
            lambda row: float(row["e2e_ns"])
        ),
        "throughput_regression_fraction": paired_throughput(
            lambda row: float(row["output_tokens_per_second"])
        ),
        "cuda_allocated_regression_fraction": _regression(
            baseline_allocated,
            candidate_allocated,
        ),
        "cuda_reserved_regression_fraction": _regression(
            baseline_reserved,
            candidate_reserved,
        ),
        "baseline_cuda_peak_allocated_bytes":
            baseline_allocated,
        "candidate_cuda_peak_allocated_bytes":
            candidate_allocated,
        "baseline_cuda_peak_reserved_bytes":
            baseline_reserved,
        "candidate_cuda_peak_reserved_bytes":
            candidate_reserved,
    }


def _outputs_are_exact(rows: list[dict]) -> bool:
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    return all(
        len({
            tuple(
                by_identity[(bucket, repetition, policy)][
                    "output_token_ids"
                ]
            )
            for policy in POLICIES
        }) == 1
        and len({
            by_identity[(bucket, repetition, policy)][
                "output_text_sha256"
            ]
            for policy in POLICIES
        }) == 1
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
    )


def _capture_cost(rows: list[dict]) -> tuple[int, int]:
    counts = []
    retained = []
    for row in rows:
        receipts = row[
            "exact_greedy_decode_burst_summary"
        ].get("capture_receipts", [])
        counts.append(len(receipts))
        retained.extend(
            int(receipt["retained_static_bytes"])
            for receipt in receipts
        )
    return max(counts, default=0), max(retained, default=0)


def _lifecycle(rows: list[dict]) -> dict:
    selected = [row for row in rows if row["policy"] == CANDIDATE]
    totals = {
        "request_count": len(selected),
        "attempts": 0,
        "acceptances": 0,
        "commits": 0,
        "committed_tokens": 0,
        "target_model_forwards": 0,
        "graph_replays": 0,
        "final_token_d2h_calls": 0,
        "final_token_d2h_bytes": 0,
        "prefix_commits": 0,
        "suffix_commits": 0,
        "parent_leases": 0,
        "unexpected_scheduler_calls": 0,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
    }
    requested = {}
    authorized = {}
    fallbacks = {}
    split_failures = {}
    for row in selected:
        summary = row["exact_greedy_decode_burst_summary"]
        inventory = row["split_phase_inventory"]
        for field in (
            "attempts",
            "acceptances",
            "commits",
            "committed_tokens",
            "target_model_forwards",
            "graph_replays",
            "final_token_d2h_calls",
            "final_token_d2h_bytes",
            "prefix_commits",
            "suffix_commits",
            "failures",
            "quarantines",
            "pending_leases",
        ):
            totals[field] += summary[field]
        totals["parent_leases"] += inventory[
            "parent_lease_count"
        ]
        totals["unexpected_scheduler_calls"] += inventory[
            "unexpected_scheduler_calls"
        ]
        for target, source in (
            (requested, summary["requested_width_histogram"]),
            (authorized, summary["authorized_width_histogram"]),
            (fallbacks, summary["fallback_counts"]),
            (split_failures, summary["split_phase_failure_counts"]),
        ):
            for key, value in source.items():
                target[key] = target.get(key, 0) + value
    totals.update({
        "requested_width_histogram": requested,
        "authorized_width_histogram": authorized,
        "fallback_counts": fallbacks,
        "split_phase_failure_counts": split_failures,
    })
    totals["complete"] = all((
        totals["request_count"] == 15,
        totals["attempts"] == 255,
        totals["acceptances"] == 255,
        totals["commits"] == 255,
        totals["committed_tokens"] == 1_905,
        totals["target_model_forwards"] == 1_905,
        totals["graph_replays"] == 1_905,
        totals["final_token_d2h_calls"] == 30,
        totals["final_token_d2h_bytes"] == 840,
        totals["prefix_commits"] == 225,
        totals["suffix_commits"] == 225,
        totals["parent_leases"] == 225,
        totals["unexpected_scheduler_calls"] == 0,
        totals["failures"] == 0,
        totals["quarantines"] == 0,
        totals["pending_leases"] == 0,
        requested == {"3": 15, "4": 15, "8": 225},
        authorized == {"3": 15, "4": 15, "8": 225},
        fallbacks == {},
        split_failures == {},
    ))
    return totals


def _reconstruct(
    rows: list[dict],
    correctness: dict,
    logits_passed: bool,
) -> dict:
    candidate = [row for row in rows if row["policy"] == CANDIDATE]
    split = [row for row in rows if row["policy"] == SPLIT_POLICY]
    k4 = [row for row in rows if row["policy"] == K4_POLICY]
    aggregate = _metric_summary(
        split,
        candidate,
        baseline_policy=SPLIT_POLICY,
    )
    pairs = _paired_rows(split, candidate)
    tail_improvement = statistics.median(
        (
            float(left["tail_seven_elapsed_ns"])
            - float(right["tail_seven_elapsed_ns"])
        )
        / float(left["tail_seven_elapsed_ns"])
        for left, right in pairs
    )
    by_bucket = {}
    bucket_regressions = []
    for bucket, _prompt, _generated in CONTEXTS:
        metrics = _metric_summary(
            [row for row in split if row["context_bucket"] == bucket],
            [
                row for row in candidate
                if row["context_bucket"] == bucket
            ],
            baseline_policy=SPLIT_POLICY,
        )
        by_bucket[bucket] = metrics
        if (
            metrics["tpot_median_regression_fraction"]
            > BUCKET_TPOT_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:median_tpot")
        if (
            metrics["tpot_p95_regression_fraction"]
            > BUCKET_TPOT_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:p95_tpot")
        if (
            metrics["ttft_regression_fraction"]
            > BUCKET_TTFT_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > BUCKET_E2E_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:e2e")
    candidate_median_gap = statistics.median(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in candidate
    )
    k4_median_gap = statistics.median(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in k4
    )
    candidate_maximum_gap = max(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in candidate
    )
    k4_maximum_gap = max(
        int(row["maximum_host_visible_burst_gap_ns"])
        for row in k4
    )
    median_gap_regression = _regression(
        k4_median_gap,
        candidate_median_gap,
    )
    maximum_gap_regression = _regression(
        k4_maximum_gap,
        candidate_maximum_gap,
    )
    split_capture_count, split_retained = _capture_cost(split)
    candidate_capture_count, candidate_retained = _capture_cost(
        candidate
    )
    lifecycle = _lifecycle(rows)
    output_exact = _outputs_are_exact(rows)
    performance_passed = all((
        tail_improvement >= TAIL_IMPROVEMENT_MINIMUM,
        aggregate["tpot_median_regression_fraction"]
        <= TPOT_REGRESSION_LIMIT,
        aggregate["throughput_regression_fraction"]
        <= THROUGHPUT_REGRESSION_LIMIT,
        aggregate["cuda_allocated_regression_fraction"]
        <= MEMORY_REGRESSION_LIMIT,
        aggregate["cuda_reserved_regression_fraction"]
        <= MEMORY_REGRESSION_LIMIT,
        median_gap_regression <= MEDIAN_GAP_REGRESSION_LIMIT,
        maximum_gap_regression <= MAXIMUM_GAP_REGRESSION_LIMIT,
        candidate_capture_count <= split_capture_count,
        candidate_retained <= split_retained,
        not bucket_regressions,
    ))
    run_tags = {
        *(row.get("run_tag") for row in rows),
    }
    source_commits = {
        *(row.get("source_commit") for row in rows),
    }
    evidence_complete = (
        len(run_tags) == 1 and len(source_commits) == 1
    )
    correctness_passed = output_exact and logits_passed
    if not correctness_passed:
        classification = (
            "NO_GO_EXACT_BURST_RAGGED_COALESCING_CORRECTNESS"
        )
    elif not evidence_complete or not lifecycle["complete"]:
        classification = (
            "INCOMPLETE_EXACT_BURST_RAGGED_COALESCING_EVIDENCE"
        )
    elif not performance_passed:
        classification = (
            "NO_GO_EXACT_BURST_RAGGED_COALESCING_PERFORMANCE"
        )
    else:
        classification = "GO_EXACT_BURST_RAGGED_COALESCING"
    evaluation = {
        "policy": CANDIDATE,
        "correctness_passed": correctness_passed,
        "performance_passed": performance_passed,
        "tail_seven_improvement_fraction": tail_improvement,
        "aggregate": {"split_vs_ragged": aggregate},
        "by_bucket": by_bucket,
        "bucket_regressions": bucket_regressions,
        "candidate_median_max_gap_ns": candidate_median_gap,
        "k4_median_max_gap_ns": k4_median_gap,
        "median_max_gap_regression_vs_k4":
            median_gap_regression,
        "candidate_maximum_gap_ns": candidate_maximum_gap,
        "k4_maximum_gap_ns": k4_maximum_gap,
        "maximum_gap_regression_vs_k4":
            maximum_gap_regression,
        "split_capture_count": split_capture_count,
        "candidate_capture_count": candidate_capture_count,
        "split_capture_retained_static_bytes": split_retained,
        "candidate_capture_retained_static_bytes":
            candidate_retained,
        "lifecycle": lifecycle,
    }
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "classification": classification,
        "selected_policy": CANDIDATE,
        "selected_burst_width": 8,
        "ragged_width_cap": 4,
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_LIMIT,
            "tail_seven_improvement_minimum":
                TAIL_IMPROVEMENT_MINIMUM,
            "aggregate_tpot_regression_limit":
                TPOT_REGRESSION_LIMIT,
            "throughput_regression_limit":
                THROUGHPUT_REGRESSION_LIMIT,
            "bucket_tpot_regression_limit":
                BUCKET_TPOT_REGRESSION_LIMIT,
            "bucket_e2e_regression_limit":
                BUCKET_E2E_REGRESSION_LIMIT,
            "bucket_ttft_regression_limit":
                BUCKET_TTFT_REGRESSION_LIMIT,
            "memory_regression_limit": MEMORY_REGRESSION_LIMIT,
            "median_gap_regression_limit":
                MEDIAN_GAP_REGRESSION_LIMIT,
            "maximum_gap_regression_limit":
                MAXIMUM_GAP_REGRESSION_LIMIT,
        },
        "correctness": correctness,
        "candidate_evaluation": evaluation,
    }


def _compare_payload(expected, actual, path: str = "") -> float:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            raise ValueError(f"value disagreement at {path}")
        return 0.0
    if isinstance(expected, (int, float)) and isinstance(
        actual,
        (int, float),
    ):
        left = _finite_number(expected, path or "expected metric")
        right = _finite_number(actual, path or "actual metric")
        disagreement = abs(left - right)
        if disagreement > METRIC_TOLERANCE:
            raise ValueError(
                f"metric disagreement at {path}: {disagreement}"
            )
        return disagreement
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(f"field disagreement at {path}")
        return max(
            (
                _compare_payload(
                    expected[key],
                    actual[key],
                    f"{path}.{key}" if path else key,
                )
                for key in expected
            ),
            default=0.0,
        )
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"list disagreement at {path}")
        return max(
            (
                _compare_payload(
                    left,
                    right,
                    f"{path}[{index}]",
                )
                for index, (left, right) in enumerate(
                    zip(expected, actual)
                )
            ),
            default=0.0,
        )
    if expected != actual:
        raise ValueError(f"value disagreement at {path}")
    return 0.0


def verify_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)
    manifest = _load_json(run_dir / "manifest.sha256")
    _validate_manifest(run_dir, manifest)
    source = _load_json(run_dir / "source_manifest.json")
    workload = _load_json(run_dir / "workload_manifest.json")
    comparison = _load_json(run_dir / "comparison.json")
    gate = _load_json(run_dir / "gate.json")
    rows = _validate_rows(_load_jsonl(run_dir / "case_rows.jsonl"))
    correctness_rows, correctness, logits_passed = (
        _validate_correctness(
            _load_jsonl(run_dir / "correctness_rows.jsonl"),
            run_dir=run_dir,
        )
    )
    if (run_dir / "source.patch").read_bytes():
        raise ValueError("dirty source patch is not allowed")
    _validate_source(source, repo_root=repo_root)
    _validate_workload(workload)
    identities = {
        source.get("run_tag"),
        workload.get("run_tag"),
        manifest.get("run_tag"),
        gate.get("run_tag"),
        comparison.get("run_tag"),
        *(row.get("run_tag") for row in rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        source.get("source_commit"),
        workload.get("source_commit"),
        manifest.get("source_commit"),
        gate.get("source_commit"),
        comparison.get("source_commit"),
        *(row.get("source_commit") for row in rows),
        *(row.get("source_commit") for row in correctness_rows),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    reconstructed = _reconstruct(
        rows,
        correctness,
        logits_passed,
    )
    reconstructed["evidence_sha256"] = {
        name: _sha256_file(run_dir / name)
        for name in (
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "source.patch",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
        )
    }
    maximum_disagreement = _compare_payload(
        reconstructed,
        comparison,
    )
    if gate.get("schema_version") != GATE_SCHEMA:
        raise ValueError("gate schema mismatch")
    if (
        gate.get("classification")
        != reconstructed["classification"]
    ):
        raise ValueError("classification drift")
    if gate.get("selected_policy") != CANDIDATE:
        raise ValueError("selected policy drift")
    if gate.get("selected_burst_width") != 8:
        raise ValueError("selected burst width drift")
    if gate.get("ragged_width_cap") != 4:
        raise ValueError("ragged width cap drift")
    if gate.get("comparison_sha256") != _sha256_file(
        run_dir / "comparison.json"
    ):
        raise ValueError("comparison digest mismatch")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "reconstructed_classification":
            reconstructed["classification"],
        "reconstructed_selected_policy":
            reconstructed["selected_policy"],
        "performance_row_count": len(rows),
        "correctness_row_count": len(correctness_rows),
        "maximum_metric_disagreement":
            maximum_disagreement,
    }
    _write_json(
        run_dir / "independent-verification.json",
        result,
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", "--artifact-dir", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = verify_bundle(
        Path(args.run_dir),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
