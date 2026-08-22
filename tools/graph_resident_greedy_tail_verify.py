#!/usr/bin/env python3
"""Independent verifier for graph-resident greedy-tail evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


CASE_SCHEMA = "graph-resident-greedy-tail.case.v1"
CORRECTNESS_SCHEMA = "graph-resident-greedy-tail.correctness.v1"
COMPARISON_SCHEMA = "graph-resident-greedy-tail.comparison.v1"
GATE_SCHEMA = "graph-resident-greedy-tail.gate.v1"
MANIFEST_SCHEMA = "graph-resident-greedy-tail.manifest.v1"
VERIFICATION_SCHEMA = (
    "graph-resident-greedy-tail.independent-verification.v1"
)
SOURCE_SCHEMA = "graph-resident-greedy-tail.source.v1"
WORKLOAD_SCHEMA = "graph-resident-greedy-tail.workload.v1"
SUMMARY_SCHEMA = "graph-resident-greedy-tail.summary.v1"
CONTEXTS = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
POLICIES = ("legacy", "host_greedy", "graph_greedy")
POINTS = ("prefill-final", "decode-first", "decode-final")
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/greedy_sampling_fast_path.py",
    "tinyvllm/engine/graph_resident_greedy_tail.py",
    "tinyvllm/engine/model_runner.py",
    "tools/profile_zero_temperature_greedy_fast_path.py",
    "tools/profile_graph_resident_greedy_tail.py",
    "tools/test_profile_graph_resident_greedy_tail.py",
    "tools/graph_resident_greedy_tail_gate.py",
    "tools/test_graph_resident_greedy_tail_gate.py",
    "tools/graph_resident_greedy_tail_verify.py",
    "tools/test_graph_resident_greedy_tail_verify.py",
    "tools/run_graph_resident_greedy_tail_remote.py",
    "tools/test_run_graph_resident_greedy_tail_remote.py",
    "tools/run_staged_inference_benchmark_remote.py",
    "tools/test_run_staged_inference_benchmark_remote.py",
)
PRIMARY_ARTIFACTS = {
    "case_rows.jsonl",
    "correctness_rows.jsonl",
    "source_manifest.json",
    "workload_manifest.json",
    "summary.json",
    "comparison.json",
    "gate.json",
}
GREEDY_COUNTERS = (
    "eligible_steps",
    "optimized_steps",
    "avoided_temperature_h2d_bytes",
    "avoided_softmax_calls",
    "avoided_gumbel_rng_calls",
    "avoided_stochastic_divisions",
    "avoided_stochastic_argmax_calls",
    "avoided_where_calls",
)
GRAPH_COUNTERS = (
    "eligible_steps",
    "captured_graphs",
    "replayed_steps",
    "final_token_d2h_calls",
    "avoided_external_compute_logits_calls",
    "avoided_external_float32_conversions",
    "avoided_external_argmax_calls",
)
GRAPH_COST_FIELDS = (
    "graph_capture_duration_ns",
    "graph_allocated_delta_bytes",
    "graph_reserved_delta_bytes",
    "graph_retained_static_bytes",
)
LEGACY_MEDIAN_THRESHOLD = 0.05
LEGACY_P95_THRESHOLD = 0.05
HOST_MEDIAN_THRESHOLD = 0.02
HOST_REGRESSION_LIMIT = 0.02
LEGACY_TPOT_REGRESSION_LIMIT = 0.03
LATENCY_REGRESSION_LIMIT = 0.03
THROUGHPUT_REGRESSION_LIMIT = 0.02
MEMORY_REGRESSION_LIMIT = 0.02
LOGIT_MAX_LIMIT = 0.25
LOGIT_MEAN_LIMIT = 0.05


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _load_json(path: Path):
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
        )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            f"primary artifact is missing: {path.name}"
        )
    with path.open("r", encoding="utf-8") as handle:
        return [
            json.loads(
                line,
                parse_constant=_reject_constant,
            )
            for line in handle
            if line.strip()
        ]


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _relative_change(baseline, candidate) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        if candidate == baseline:
            return 0.0
        raise ValueError(
            "relative comparison baseline must be positive"
        )
    return (candidate - baseline) / baseline


def _improvement(baseline, candidate) -> float:
    return -_relative_change(baseline, candidate)


def _assert_finite_tree(value, path="root") -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(
                f"non-finite numeric value at {path}"
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite_tree(item, f"{path}.{key}")
        return
    raise ValueError(f"unsupported evidence type at {path}")


def _non_negative_number(value, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise ValueError(f"{name} is invalid")


def _validate_performance_rows(rows) -> list[dict]:
    if len(rows) != 45:
        raise ValueError("expected exactly 45 measured rows")
    shapes = {
        bucket: (prompt, generated)
        for bucket, prompt, generated in CONTEXTS
    }
    identities = set()
    for row in rows:
        _assert_finite_tree(row)
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CASE_SCHEMA
        ):
            raise ValueError("case row schema mismatch")
        bucket = row.get("context_bucket")
        policy = row.get("policy")
        repetition = row.get("repetition")
        if (
            bucket not in shapes
            or policy not in POLICIES
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition not in range(5)
        ):
            raise ValueError("case identity mismatch")
        identity = (bucket, repetition, policy)
        if identity in identities:
            raise ValueError("duplicate case identity")
        identities.add(identity)
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != shapes[bucket]:
            raise ValueError("case shape mismatch")
        generated = row["generated_tokens"]
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != generated
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output_ids
            )
            or not isinstance(
                row.get("output_text_sha256"),
                str,
            )
            or len(row["output_text_sha256"]) != 64
        ):
            raise ValueError("output evidence mismatch")
        for field in (
            "tpot_samples_ns",
            "decode_host_ns",
            "decode_cuda_ns",
        ):
            values = row.get(field)
            if (
                not isinstance(values, list)
                or len(values) != generated - 1
            ):
                raise ValueError(
                    f"{field} inventory mismatch"
                )
            for value in values:
                _non_negative_number(value, field)
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            *GRAPH_COST_FIELDS,
        ):
            _non_negative_number(row.get(field), field)
        greedy = row.get("greedy_fast_path_summary")
        graph = row.get(
            "graph_resident_greedy_tail_summary"
        )
        if (
            not isinstance(greedy, dict)
            or set(GREEDY_COUNTERS) - set(greedy)
            or "fallback_counts" not in greedy
        ):
            raise ValueError(
                "greedy fast-path summary mismatch"
            )
        if (
            not isinstance(graph, dict)
            or set(GRAPH_COUNTERS) - set(graph)
            or not {
                "fallback_counts",
                "quarantine_reason",
                "capture_receipt",
            } <= set(graph)
        ):
            raise ValueError("graph-tail summary mismatch")
        for field in GREEDY_COUNTERS:
            _non_negative_number(greedy[field], field)
        for field in GRAPH_COUNTERS:
            _non_negative_number(graph[field], field)
        if not isinstance(
            greedy["fallback_counts"],
            dict,
        ) or not isinstance(graph["fallback_counts"], dict):
            raise ValueError("fallback counts mismatch")
        receipt = graph["capture_receipt"]
        if receipt is not None:
            required = {
                "source_identity",
                "graph_generation",
                "rank",
                "capture_duration_ns",
                "allocated_delta_bytes",
                "reserved_delta_bytes",
                "retained_logits_bytes",
                "retained_float32_bytes",
                "retained_token_bytes",
                "retained_static_bytes",
            }
            if not isinstance(receipt, dict) or set(receipt) != required:
                raise ValueError(
                    "capture receipt mismatch"
                )
            for field in required - {"source_identity"}:
                _non_negative_number(
                    receipt[field],
                    f"capture receipt {field}",
                )
            if receipt["retained_static_bytes"] != (
                receipt["retained_logits_bytes"]
                + receipt["retained_float32_bytes"]
                + receipt["retained_token_bytes"]
            ):
                raise ValueError(
                    "retained byte accounting mismatch"
                )
            expected_cost = {
                "graph_capture_duration_ns":
                    receipt["capture_duration_ns"],
                "graph_allocated_delta_bytes":
                    receipt["allocated_delta_bytes"],
                "graph_reserved_delta_bytes":
                    receipt["reserved_delta_bytes"],
                "graph_retained_static_bytes":
                    receipt["retained_static_bytes"],
            }
        else:
            expected_cost = {
                field: 0 for field in GRAPH_COST_FIELDS
            }
        if any(
            row[field] != expected
            for field, expected in expected_cost.items()
        ):
            raise ValueError(
                "graph-tail cost accounting mismatch"
            )
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
        for policy in POLICIES
    }
    if identities != expected:
        raise ValueError("case inventory mismatch")
    return rows


def _read_sidecar(run_dir: Path, row: dict) -> tuple[float, ...]:
    raw_path = row.get("logits_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("sidecar path mismatch")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "sidecar path escapes run directory"
        )
    path = run_dir / relative
    if not path.is_file():
        raise ValueError(f"sidecar is missing: {raw_path}")
    payload = path.read_bytes()
    expected_bytes = row.get("logits_byte_length")
    expected_count = row.get("logits_element_count")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or expected_bytes != expected_count * 4
        or len(payload) != expected_bytes
    ):
        raise ValueError("sidecar byte length mismatch")
    if hashlib.sha256(payload).hexdigest() != row.get(
        "logits_sha256"
    ):
        raise ValueError("sidecar digest mismatch")
    values = struct.unpack(f"<{expected_count}f", payload)
    if any(not math.isfinite(value) for value in values):
        raise ValueError(
            "sidecar contains non-finite values"
        )
    return tuple(values)


def _validate_correctness_rows(
    rows,
    *,
    run_dir: Path,
) -> dict:
    if len(rows) != 27:
        raise ValueError(
            "expected exactly 27 correctness rows"
        )
    shapes = {
        bucket: (prompt, generated)
        for bucket, prompt, generated in CONTEXTS
    }
    identities = {}
    values_by_identity = {}
    for row in rows:
        _assert_finite_tree(row)
        if (
            not isinstance(row, dict)
            or row.get("schema_version")
            != CORRECTNESS_SCHEMA
        ):
            raise ValueError(
                "correctness row schema mismatch"
            )
        identity = (
            row.get("context_bucket"),
            row.get("sampling_point"),
            row.get("policy"),
        )
        bucket, point, policy = identity
        if (
            bucket not in shapes
            or point not in POINTS
            or policy not in POLICIES
        ):
            raise ValueError(
                "correctness identity mismatch"
            )
        if identity in identities:
            raise ValueError(
                "duplicate correctness identity"
            )
        identities[identity] = row
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != shapes[bucket]:
            raise ValueError(
                "correctness context shape mismatch"
            )
        generated = row["generated_tokens"]
        if (
            not isinstance(row.get("output_token_ids"), list)
            or len(row["output_token_ids"]) != generated
            or not isinstance(
                row.get("output_text_sha256"),
                str,
            )
            or len(row["output_text_sha256"]) != 64
        ):
            raise ValueError(
                "correctness output evidence mismatch"
            )
        shape = row.get("logits_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or isinstance(shape[1], bool)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
            or shape[0] * shape[1]
            != row.get("logits_element_count")
        ):
            raise ValueError(
                "correctness logits shape mismatch"
            )
        values_by_identity[identity] = _read_sidecar(
            run_dir,
            row,
        )
    expected = {
        (bucket, point, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for point in POINTS
        for policy in POLICIES
    }
    if set(identities) != expected:
        raise ValueError(
            "correctness inventory mismatch"
        )
    pairs = []
    maximum = 0.0
    worst_mean = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_ids = True
    all_text = True
    policy_pairs = (
        ("legacy", "host_greedy"),
        ("legacy", "graph_greedy"),
        ("host_greedy", "graph_greedy"),
    )
    for bucket, _prompt, _generated in CONTEXTS:
        for point in POINTS:
            for baseline_policy, candidate_policy in policy_pairs:
                baseline_row = identities[
                    (bucket, point, baseline_policy)
                ]
                candidate_row = identities[
                    (bucket, point, candidate_policy)
                ]
                baseline_values = values_by_identity[
                    (bucket, point, baseline_policy)
                ]
                candidate_values = values_by_identity[
                    (bucket, point, candidate_policy)
                ]
                if (
                    baseline_row["logits_shape"]
                    != candidate_row["logits_shape"]
                    or len(baseline_values)
                    != len(candidate_values)
                ):
                    raise ValueError(
                        "paired logits shape mismatch"
                    )
                differences = [
                    abs(left - right)
                    for left, right in zip(
                        baseline_values,
                        candidate_values,
                    )
                ]
                pair_max = max(differences)
                pair_mean = sum(differences) / len(differences)
                baseline_argmax = max(
                    range(len(baseline_values)),
                    key=baseline_values.__getitem__,
                )
                candidate_argmax = max(
                    range(len(candidate_values)),
                    key=candidate_values.__getitem__,
                )
                argmax_equal = (
                    baseline_argmax == candidate_argmax
                )
                ids_equal = (
                    baseline_row["output_token_ids"]
                    == candidate_row["output_token_ids"]
                )
                text_equal = (
                    baseline_row["output_text_sha256"]
                    == candidate_row["output_text_sha256"]
                )
                maximum = max(maximum, pair_max)
                worst_mean = max(worst_mean, pair_mean)
                total_abs += sum(differences)
                total_count += len(differences)
                all_argmax = all_argmax and argmax_equal
                all_ids = all_ids and ids_equal
                all_text = all_text and text_equal
                pairs.append({
                    "context_bucket": bucket,
                    "sampling_point": point,
                    "baseline_policy": baseline_policy,
                    "candidate_policy": candidate_policy,
                    "element_count": len(differences),
                    "max_abs": pair_max,
                    "mean_abs": pair_mean,
                    "baseline_argmax": baseline_argmax,
                    "candidate_argmax": candidate_argmax,
                    "argmax_equal": argmax_equal,
                    "output_ids_exact": ids_equal,
                    "output_text_exact": text_equal,
                })
    return {
        "row_count": len(rows),
        "pair_count": len(pairs),
        "max_abs": maximum,
        "mean_abs": worst_mean,
        "aggregate_mean_abs": total_abs / total_count,
        "argmax_equal": all_argmax,
        "output_ids_exact": all_ids,
        "output_text_exact": all_text,
        "pairs": pairs,
    }


def _metrics(
    baseline_rows,
    candidate_rows,
    *,
    baseline_policy: str,
    candidate_policy: str,
) -> dict:
    baseline_tpot = [
        float(value)
        for row in baseline_rows
        for value in row["tpot_samples_ns"]
    ]
    candidate_tpot = [
        float(value)
        for row in candidate_rows
        for value in row["tpot_samples_ns"]
    ]
    baseline_median = statistics.median(baseline_tpot)
    candidate_median = statistics.median(candidate_tpot)
    baseline_p95 = _nearest_rank(baseline_tpot, 0.95)
    candidate_p95 = _nearest_rank(candidate_tpot, 0.95)
    baseline_p99 = _nearest_rank(baseline_tpot, 0.99)
    candidate_p99 = _nearest_rank(candidate_tpot, 0.99)
    baseline_ttft = statistics.median(
        float(row["ttft_ns"]) for row in baseline_rows
    )
    candidate_ttft = statistics.median(
        float(row["ttft_ns"]) for row in candidate_rows
    )
    baseline_e2e = statistics.median(
        float(row["e2e_ns"]) for row in baseline_rows
    )
    candidate_e2e = statistics.median(
        float(row["e2e_ns"]) for row in candidate_rows
    )
    baseline_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in baseline_rows
    )
    candidate_rate = statistics.median(
        float(row["output_tokens_per_second"])
        for row in candidate_rows
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
        "candidate_policy": candidate_policy,
        "sample_count_per_policy": len(baseline_tpot),
        "baseline_tpot_median_ns": baseline_median,
        "candidate_tpot_median_ns": candidate_median,
        "tpot_median_improvement_fraction":
            _improvement(baseline_median, candidate_median),
        "baseline_tpot_p95_ns": baseline_p95,
        "candidate_tpot_p95_ns": candidate_p95,
        "tpot_p95_improvement_fraction":
            _improvement(baseline_p95, candidate_p95),
        "baseline_tpot_p99_ns": baseline_p99,
        "candidate_tpot_p99_ns": candidate_p99,
        "tpot_p99_improvement_fraction":
            _improvement(baseline_p99, candidate_p99),
        "baseline_ttft_median_ns": baseline_ttft,
        "candidate_ttft_median_ns": candidate_ttft,
        "ttft_regression_fraction":
            _relative_change(baseline_ttft, candidate_ttft),
        "baseline_e2e_median_ns": baseline_e2e,
        "candidate_e2e_median_ns": candidate_e2e,
        "e2e_regression_fraction":
            _relative_change(baseline_e2e, candidate_e2e),
        "baseline_output_tokens_per_second_median":
            baseline_rate,
        "candidate_output_tokens_per_second_median":
            candidate_rate,
        "throughput_regression_fraction":
            _relative_change(candidate_rate, baseline_rate),
        "baseline_cuda_peak_allocated_bytes":
            baseline_allocated,
        "candidate_cuda_peak_allocated_bytes":
            candidate_allocated,
        "cuda_allocated_delta_bytes":
            candidate_allocated - baseline_allocated,
        "baseline_cuda_peak_reserved_bytes":
            baseline_reserved,
        "candidate_cuda_peak_reserved_bytes":
            candidate_reserved,
        "cuda_reserved_delta_bytes":
            candidate_reserved - baseline_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(
                baseline_reserved,
                candidate_reserved,
            ),
    }


def _graph_complete(rows) -> bool:
    for row in rows:
        greedy = row["greedy_fast_path_summary"]
        graph = row[
            "graph_resident_greedy_tail_summary"
        ]
        generated = row["generated_tokens"]
        if row["policy"] == "legacy":
            if (
                greedy["eligible_steps"] != 0
                or greedy["optimized_steps"] != 0
                or graph["replayed_steps"] != 0
                or graph["captured_graphs"] != 0
            ):
                return False
        elif row["policy"] == "host_greedy":
            if (
                greedy["eligible_steps"] != generated
                or greedy["optimized_steps"] != generated
                or graph["replayed_steps"] != 0
                or graph["captured_graphs"] != 0
            ):
                return False
        else:
            expected = generated - 1
            if (
                greedy["eligible_steps"] != 1
                or greedy["optimized_steps"] != 1
                or graph["eligible_steps"] != expected
                or graph["replayed_steps"] != expected
                or graph["final_token_d2h_calls"] != expected
                or graph[
                    "avoided_external_compute_logits_calls"
                ] != expected
                or graph[
                    "avoided_external_float32_conversions"
                ] != expected
                or graph[
                    "avoided_external_argmax_calls"
                ] != expected
                or graph["captured_graphs"] != 1
                or graph["fallback_counts"]
                or graph["quarantine_reason"] is not None
                or graph["capture_receipt"] is None
            ):
                return False
    return True


def _cost(rows) -> dict:
    graph_rows = [
        row for row in rows
        if row["policy"] == "graph_greedy"
    ]

    def one(field: str) -> dict:
        values = [int(row[field]) for row in graph_rows]
        return {
            "min": min(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    summaries = [
        row["graph_resident_greedy_tail_summary"]
        for row in graph_rows
    ]
    return {
        "capture_duration_ns":
            one("graph_capture_duration_ns"),
        "allocated_delta_bytes":
            one("graph_allocated_delta_bytes"),
        "reserved_delta_bytes":
            one("graph_reserved_delta_bytes"),
        "retained_static_bytes":
            one("graph_retained_static_bytes"),
        "final_token_d2h_calls": sum(
            item["final_token_d2h_calls"]
            for item in summaries
        ),
        "avoided_work": {
            field: sum(item[field] for item in summaries)
            for field in (
                "avoided_external_compute_logits_calls",
                "avoided_external_float32_conversions",
                "avoided_external_argmax_calls",
            )
        },
    }


def _reconstruct_comparison(
    rows,
    correctness_rows,
    *,
    run_dir: Path,
) -> dict:
    rows = _validate_performance_rows(rows)
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    exact_outputs = True
    for bucket, _prompt, _generated in CONTEXTS:
        for repetition in range(5):
            triple = [
                by_identity[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            exact_outputs = exact_outputs and (
                len({
                    tuple(row["output_token_ids"])
                    for row in triple
                }) == 1
                and len({
                    row["output_text_sha256"]
                    for row in triple
                }) == 1
            )
    correctness = _validate_correctness_rows(
        correctness_rows,
        run_dir=run_dir,
    )
    correctness_passed = (
        exact_outputs
        and correctness["output_ids_exact"]
        and correctness["output_text_exact"]
        and correctness["max_abs"] <= LOGIT_MAX_LIMIT
        and correctness["mean_abs"] <= LOGIT_MEAN_LIMIT
        and correctness["argmax_equal"]
    )
    graph_replay_complete = _graph_complete(rows)
    by_bucket = {}
    for bucket, _prompt, _generated in CONTEXTS:
        selected = [
            row for row in rows
            if row["context_bucket"] == bucket
        ]
        graph_rows = [
            row for row in selected
            if row["policy"] == "graph_greedy"
        ]
        by_bucket[bucket] = {
            "legacy_vs_graph": _metrics(
                [
                    row for row in selected
                    if row["policy"] == "legacy"
                ],
                graph_rows,
                baseline_policy="legacy",
                candidate_policy="graph_greedy",
            ),
            "host_greedy_vs_graph": _metrics(
                [
                    row for row in selected
                    if row["policy"] == "host_greedy"
                ],
                graph_rows,
                baseline_policy="host_greedy",
                candidate_policy="graph_greedy",
            ),
        }
    graph_rows = [
        row for row in rows
        if row["policy"] == "graph_greedy"
    ]
    aggregate = {
        "legacy_vs_graph": _metrics(
            [
                row for row in rows
                if row["policy"] == "legacy"
            ],
            graph_rows,
            baseline_policy="legacy",
            candidate_policy="graph_greedy",
        ),
        "host_greedy_vs_graph": _metrics(
            [
                row for row in rows
                if row["policy"] == "host_greedy"
            ],
            graph_rows,
            baseline_policy="host_greedy",
            candidate_policy="graph_greedy",
        ),
    }
    winning = sum(
        metrics["legacy_vs_graph"][
            "tpot_median_improvement_fraction"
        ] >= LEGACY_MEDIAN_THRESHOLD
        for metrics in by_bucket.values()
    )
    host_regressions = []
    protected = []
    for bucket, comparisons in by_bucket.items():
        legacy = comparisons["legacy_vs_graph"]
        host = comparisons["host_greedy_vs_graph"]
        if (
            host["tpot_median_improvement_fraction"]
            < -HOST_REGRESSION_LIMIT
        ):
            host_regressions.append(
                f"{bucket}:median_tpot"
            )
        if (
            host["tpot_p95_improvement_fraction"]
            < -HOST_REGRESSION_LIMIT
        ):
            host_regressions.append(
                f"{bucket}:p95_tpot"
            )
        if (
            legacy["tpot_median_improvement_fraction"]
            < -LEGACY_TPOT_REGRESSION_LIMIT
        ):
            protected.append(f"{bucket}:median_tpot")
        if (
            legacy["tpot_p95_improvement_fraction"]
            < -LEGACY_TPOT_REGRESSION_LIMIT
        ):
            protected.append(f"{bucket}:p95_tpot")
        if (
            legacy["ttft_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            protected.append(f"{bucket}:ttft")
        if (
            legacy["e2e_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            protected.append(f"{bucket}:e2e")
        if (
            legacy["throughput_regression_fraction"]
            > THROUGHPUT_REGRESSION_LIMIT
        ):
            protected.append(f"{bucket}:throughput")
    if (
        aggregate["legacy_vs_graph"][
            "cuda_reserved_regression_fraction"
        ] > MEMORY_REGRESSION_LIMIT
    ):
        protected.append("aggregate:cuda_reserved")
    cost = _cost(rows)
    cost_complete = (
        cost["capture_duration_ns"]["min"] > 0
        and cost["retained_static_bytes"]["min"] > 0
        and cost["final_token_d2h_calls"] == 15 * 127
    )
    run_tags = {
        *(row.get("run_tag") for row in rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        *(row.get("source_commit") for row in rows),
        *(
            row.get("source_commit")
            for row in correctness_rows
        ),
    }
    evidence_complete = (
        len(run_tags) == 1 and len(commits) == 1
    )
    if not correctness_passed:
        classification = "NO_GO_CORRECTNESS"
    elif not graph_replay_complete:
        classification = (
            "NO_GO_GRAPH_REPLAY_INCOMPLETE"
        )
    elif winning < 2:
        classification = "NO_GO_LEGACY_TPOT_MEDIAN"
    elif (
        aggregate["legacy_vs_graph"][
            "tpot_p95_improvement_fraction"
        ] < LEGACY_P95_THRESHOLD
    ):
        classification = "NO_GO_LEGACY_TPOT_P95"
    elif (
        aggregate["host_greedy_vs_graph"][
            "tpot_median_improvement_fraction"
        ] < HOST_MEDIAN_THRESHOLD
        or host_regressions
    ):
        classification = (
            "NO_GO_HOST_GREEDY_INCREMENTAL"
        )
    elif protected:
        classification = "NO_GO_PROTECTED_REGRESSION"
    elif not cost_complete:
        classification = "NO_GO_COST_INCOMPLETE"
    elif not evidence_complete:
        classification = "NO_GO_EVIDENCE_INCOMPLETE"
    else:
        classification = "GO_GRAPH_RESIDENT_GREEDY_TAIL"
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "classification": classification,
        "correctness_passed": correctness_passed,
        "graph_replay_complete": graph_replay_complete,
        "legacy_median_tpot_winning_bucket_count": winning,
        "host_incremental_regressions": host_regressions,
        "protected_regressions": protected,
        "cost_complete": cost_complete,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_LIMIT,
            "legacy_median_tpot_min_improvement_fraction":
                LEGACY_MEDIAN_THRESHOLD,
            "legacy_aggregate_p95_min_improvement_fraction":
                LEGACY_P95_THRESHOLD,
            "host_aggregate_median_min_improvement_fraction":
                HOST_MEDIAN_THRESHOLD,
            "host_bucket_tpot_max_regression_fraction":
                HOST_REGRESSION_LIMIT,
            "legacy_bucket_tpot_max_regression_fraction":
                LEGACY_TPOT_REGRESSION_LIMIT,
            "latency_max_regression_fraction":
                LATENCY_REGRESSION_LIMIT,
            "throughput_max_regression_fraction":
                THROUGHPUT_REGRESSION_LIMIT,
            "reserved_memory_max_regression_fraction":
                MEMORY_REGRESSION_LIMIT,
        },
        "correctness": correctness,
        "by_bucket": by_bucket,
        "aggregate": aggregate,
        "cost": cost,
    }


def _reconstruct_summary(rows) -> dict:
    identities = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
    }
    triples = []
    for bucket, _prompt, _generated in CONTEXTS:
        for repetition in range(5):
            triple = [
                identities[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            if len({
                tuple(row["output_token_ids"])
                for row in triple
            }) != 1:
                raise ValueError(
                    "summary output token mismatch"
                )
            if len({
                row["output_text_sha256"]
                for row in triple
            }) != 1:
                raise ValueError(
                    "summary output text mismatch"
                )
            triples.append({
                "context_bucket": bucket,
                "repetition": repetition,
                "legacy_tpot_median_ns": statistics.median(
                    triple[0]["tpot_samples_ns"]
                ),
                "host_greedy_tpot_median_ns":
                    statistics.median(
                        triple[1]["tpot_samples_ns"]
                    ),
                "graph_greedy_tpot_median_ns":
                    statistics.median(
                        triple[2]["tpot_samples_ns"]
                    ),
            })
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "row_count": len(rows),
        "triple_count": len(triples),
        "all_outputs_exact": True,
        "all_graph_decode_steps_optimized": all(
            row[
                "graph_resident_greedy_tail_summary"
            ]["replayed_steps"] == row["generated_tokens"] - 1
            and row[
                "graph_resident_greedy_tail_summary"
            ]["final_token_d2h_calls"]
            == row["generated_tokens"] - 1
            for row in rows
            if row["policy"] == "graph_greedy"
        ),
        "triples": triples,
        "correctness_row_count": 27,
    }


def _validate_manifest(
    run_dir: Path,
    manifest,
    correctness_rows,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ValueError("manifest schema mismatch")
    sidecars = {
        row.get("logits_path")
        for row in correctness_rows
    }
    if None in sidecars:
        raise ValueError("sidecar path mismatch")
    expected = PRIMARY_ARTIFACTS | sidecars
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != expected
    ):
        raise ValueError(
            "manifest file inventory mismatch"
        )
    for name in sorted(expected):
        actual = _sha256_file(run_dir / name)
        if artifacts[name] != actual:
            raise ValueError(
                f"manifest digest mismatch: {name}"
            )


def _validate_source(repo_root: Path, source) -> None:
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != SOURCE_SCHEMA
        or set(source.get("source_sha256", {}))
        != set(SOURCE_FILES)
    ):
        raise ValueError("source manifest mismatch")
    for relative in SOURCE_FILES:
        if source["source_sha256"][relative] != _sha256_file(
            repo_root / relative
        ):
            raise ValueError(
                f"source digest mismatch: {relative}"
            )


def _validate_workload(workload) -> None:
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version")
        != WORKLOAD_SCHEMA
    ):
        raise ValueError("workload manifest mismatch")
    expected_cases = [
        {
            "context_bucket": bucket,
            "prompt_tokens": prompt,
            "generated_tokens": generated,
        }
        for bucket, prompt, generated in CONTEXTS
    ]
    expected = {
        "context_cases": expected_cases,
        "repetitions": 5,
        "warmup_repetitions": 2,
        "batch_size": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "policy_flags": {
            "legacy": {
                "zero_temperature_greedy_fast_path": False,
                "graph_resident_greedy_tail": False,
            },
            "host_greedy": {
                "zero_temperature_greedy_fast_path": True,
                "graph_resident_greedy_tail": False,
            },
            "graph_greedy": {
                "zero_temperature_greedy_fast_path": True,
                "graph_resident_greedy_tail": True,
            },
        },
        "policy_order": {
            str(index): (
                list(POLICIES)
                if index % 2 == 0
                else list(reversed(POLICIES))
            )
            for index in range(5)
        },
        "correctness_sampling_points": list(POINTS),
    }
    for field, value in expected.items():
        if workload.get(field) != value:
            raise ValueError(
                f"workload manifest mismatch: {field}"
            )


def verify_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    rows = _load_jsonl(run_dir / "case_rows.jsonl")
    correctness_rows = _load_jsonl(
        run_dir / "correctness_rows.jsonl"
    )
    manifest_path = run_dir / "manifest.sha256"
    manifest = _load_json(manifest_path)
    _validate_manifest(
        run_dir,
        manifest,
        correctness_rows,
    )
    source = _load_json(run_dir / "source_manifest.json")
    workload = _load_json(run_dir / "workload_manifest.json")
    comparison = _load_json(run_dir / "comparison.json")
    gate = _load_json(run_dir / "gate.json")
    summary = _load_json(run_dir / "summary.json")
    _validate_source(Path(repo_root), source)
    _validate_workload(workload)
    reconstructed = _reconstruct_comparison(
        rows,
        correctness_rows,
        run_dir=run_dir,
    )
    if comparison != reconstructed:
        raise ValueError("comparison drift")
    if summary != _reconstruct_summary(rows):
        raise ValueError("worker summary drift")
    comparison_digest = _sha256_file(
        run_dir / "comparison.json"
    )
    if (
        not isinstance(gate, dict)
        or gate.get("schema_version") != GATE_SCHEMA
        or gate.get("classification")
        != reconstructed["classification"]
        or gate.get("run_tag") != reconstructed["run_tag"]
        or gate.get("source_commit")
        != reconstructed["source_commit"]
        or gate.get("comparison_sha256")
        != comparison_digest
    ):
        raise ValueError("classification drift")
    identities = {
        manifest.get("run_tag"),
        source.get("run_tag"),
        workload.get("run_tag"),
        reconstructed.get("run_tag"),
        gate.get("run_tag"),
    }
    commits = {
        manifest.get("source_commit"),
        source.get("source_commit"),
        workload.get("source_commit"),
        reconstructed.get("source_commit"),
        gate.get("source_commit"),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "reconstructed_classification":
            reconstructed["classification"],
        "comparison_sha256": comparison_digest,
        "manifest_sha256": _sha256_file(manifest_path),
    }
    output = run_dir / "independent-verification.json"
    output.write_text(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
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
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
