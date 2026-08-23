#!/usr/bin/env python3
"""Independent verifier for exact greedy decode-burst evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


CASE_SCHEMA = "exact-burst-continuation-epoch.case.v1"
CORRECTNESS_SCHEMA = "exact-burst-continuation-epoch.correctness.v1"
SUMMARY_SCHEMA = "exact-burst-continuation-epoch.summary.v1"
SOURCE_SCHEMA = "exact-burst-continuation-epoch.source.v1"
WORKLOAD_SCHEMA = "exact-burst-continuation-epoch.workload.v1"
COMPARISON_SCHEMA = "exact-burst-continuation-epoch.comparison.v1"
GATE_SCHEMA = "exact-burst-continuation-epoch.gate.v1"
MANIFEST_SCHEMA = "exact-burst-continuation-epoch.manifest.v1"
VERIFICATION_SCHEMA = (
    "exact-burst-continuation-epoch.independent-verification.v1"
)
TRACE_IDENTITY = (
    "gate-only-exact-burst-continuation-correctness-v1"
)
CONTEXTS = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k4_continuation",
    "decode_burst_k8",
)
REFERENCE_POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
)
CONTINUATION_POLICY = "decode_burst_k4_continuation"
STAGE1_MODEL_BASENAMES = ("Qwen3-0.6B", "Qwen3-0___6B")
POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
POLICY_CONFIGS = {
    "host_greedy": {
        "enabled": False,
        "continuation": False,
        "epoch_relative_sampling": False,
        "width": 1,
        "selectable": False,
        "entrypoint": "ordinary",
    },
    "decode_burst_k4": {
        "enabled": True,
        "continuation": False,
        "epoch_relative_sampling": False,
        "width": 4,
        "selectable": False,
        "entrypoint": "production",
    },
    "decode_burst_k4_continuation": {
        "enabled": True,
        "continuation": True,
        "epoch_relative_sampling": True,
        "width": 4,
        "selectable": True,
        "entrypoint": "production",
    },
    "decode_burst_k8": {
        "enabled": True,
        "continuation": False,
        "epoch_relative_sampling": False,
        "width": 8,
        "selectable": False,
        "entrypoint": "production",
    },
}
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_greedy_decode_burst.py",
    "tools/profile_exact_burst_continuation_epoch.py",
    "tools/test_profile_exact_burst_continuation_epoch.py",
    "tools/exact_burst_continuation_epoch_gate.py",
    "tools/test_exact_burst_continuation_epoch_gate.py",
    "tools/exact_burst_continuation_epoch_verify.py",
    "tools/test_exact_burst_continuation_epoch_verify.py",
    "tools/run_exact_burst_continuation_epoch_remote.py",
    "tools/test_run_exact_burst_continuation_epoch_remote.py",
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
CAPTURE_COST_FIELDS = (
    "capture_duration_ns",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "capture_retained_static_bytes",
    "reserved_scratch_blocks",
)
BURST_COUNTER_FIELDS = (
    "attempts",
    "acceptances",
    "target_model_forwards",
    "graph_replays",
    "intermediate_token_d2h_calls",
    "final_token_d2h_calls",
    "final_token_d2h_bytes",
    "sampled_logit_d2h_calls",
    "output_budget_clipped",
    "block_boundary_clipped",
    "commits",
    "committed_tokens",
    "failures",
    "quarantines",
    "pending_leases",
    "maximum_host_visible_gap_ns",
    "continuation_attempts",
    "continuation_hits",
    "cold_binds",
    "continuation_tokens",
    "continuation_bursts",
    "skipped_static_reset_operations",
    "skipped_scalar_bind_operations",
    "skipped_block_table_constructions",
    "skipped_block_table_copy_calls",
    "skipped_block_table_bytes",
)
CONTINUATION_MAP_FIELDS = (
    "continuation_miss_counts",
    "continuation_invalidation_counts",
)
LOGIT_MAX_LIMIT = 0.25
LOGIT_MEAN_LIMIT = 0.05
K4_MEDIAN_THRESHOLD = 0.05
K4_P95_THRESHOLD = 0.03
BUCKET_MEDIAN_THRESHOLD = 0.05
MIN_WINNING_BUCKETS = 2
K8_PARITY_LIMIT = 0.02
BUCKET_REGRESSION_LIMIT = 0.03
LATENCY_REGRESSION_LIMIT = 0.03
THROUGHPUT_REGRESSION_LIMIT = 0.02
MEMORY_REGRESSION_LIMIT = 0.03
VISIBILITY_RATIO_LIMIT = 0.60
MIN_CONTINUATION_HITS = 31


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


def _finite_non_negative(value, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
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


def _policy_order(
    repetition: int,
    context_index: int,
) -> tuple[str, ...]:
    rotation = repetition % len(POLICIES)
    order = POLICIES[rotation:] + POLICIES[:rotation]
    return tuple(reversed(order)) if context_index % 2 else order


def _case_shape(bucket: str) -> tuple[int, int]:
    try:
        return {
            name: (prompt, generated)
            for name, prompt, generated in CONTEXTS
        }[bucket]
    except KeyError as error:
        raise ValueError("context bucket is invalid") from error


def _validate_capture_receipt(
    receipt,
    *,
    correctness_trace: bool,
) -> None:
    required = {
        "graph_identity_sha256",
        "graph_generation",
        "capture_duration_ns",
        "allocated_delta_bytes",
        "reserved_delta_bytes",
        "retained_static_bytes",
        "scratch_block_count",
        "correctness_trace",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ValueError("burst capture receipt fields mismatch")
    _valid_digest(receipt["graph_identity_sha256"], "graph identity")
    for field in required - {
        "graph_identity_sha256",
        "correctness_trace",
    }:
        _non_negative_integer(receipt[field], field)
    if receipt["correctness_trace"] is not correctness_trace:
        raise ValueError("capture correctness trace mismatch")


def _validate_summary_shape(
    summary,
    *,
    policy: str,
    correctness_trace: bool,
) -> None:
    required = set(BURST_COUNTER_FIELDS) | {
        *CONTINUATION_MAP_FIELDS,
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
        "quarantine_reason",
        "capture_receipts",
    }
    if not isinstance(summary, dict) or required - set(summary):
        raise ValueError("exact burst summary fields are missing")
    for field in BURST_COUNTER_FIELDS:
        _non_negative_integer(summary[field], field)
    for field in (
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
        *CONTINUATION_MAP_FIELDS,
    ):
        value = summary[field]
        if not isinstance(value, dict):
            raise ValueError(f"{field} is invalid")
        for key, count in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} key is invalid")
            _non_negative_integer(count, f"{field}.{key}")
    reason = summary["quarantine_reason"]
    if reason is not None and (
        not isinstance(reason, str) or not reason
    ):
        raise ValueError("quarantine reason is invalid")
    receipts = summary["capture_receipts"]
    if not isinstance(receipts, list):
        raise ValueError("capture receipts must be a list")
    for receipt in receipts:
        _validate_capture_receipt(
            receipt,
            correctness_trace=correctness_trace,
        )
    enabled = POLICY_CONFIGS[policy]["enabled"]
    if enabled and len(receipts) != 1:
        raise ValueError(
            "enabled policy requires exactly one capture receipt"
        )
    if not enabled and (
        summary["graph_replays"]
        or summary["final_token_d2h_calls"]
        or receipts
    ):
        raise ValueError("host policy reported burst activity")
    continuation = POLICY_CONFIGS[policy]["continuation"]
    continuation_fields = (
        "continuation_attempts",
        "continuation_hits",
        "cold_binds",
        "continuation_tokens",
        "continuation_bursts",
        "skipped_static_reset_operations",
        "skipped_scalar_bind_operations",
        "skipped_block_table_constructions",
        "skipped_block_table_copy_calls",
        "skipped_block_table_bytes",
    )
    if not continuation and (
        any(summary[field] for field in continuation_fields)
        or any(summary[field] for field in CONTINUATION_MAP_FIELDS)
    ):
        raise ValueError(
            "non-continuation policy reported continuation activity"
        )
    if continuation:
        hits = summary["continuation_hits"]
        if any((
            summary["continuation_attempts"] != summary["commits"],
            hits + summary["cold_binds"]
            != summary["continuation_attempts"],
            summary["continuation_bursts"] != hits,
            summary["skipped_static_reset_operations"] != hits * 7,
            summary["skipped_scalar_bind_operations"] != hits * 5,
            summary["skipped_block_table_constructions"] != hits,
            summary["skipped_block_table_copy_calls"] != hits,
            summary["continuation_tokens"] < hits,
        )):
            raise ValueError("continuation counter inventory mismatch")


def _validate_performance_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != 60:
        raise ValueError(
            f"expected exactly 60 measured rows, got {len(rows)}"
        )
    identities = set()
    run_tags = set()
    commits = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CASE_SCHEMA
        ):
            raise ValueError("case row schema mismatch")
        policy = row.get("policy")
        bucket = row.get("context_bucket")
        repetition = row.get("repetition")
        if (
            policy not in POLICIES
            or bucket not in {item[0] for item in CONTEXTS}
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition not in range(5)
        ):
            raise ValueError("case identity mismatch")
        identity = (bucket, repetition, policy)
        if identity in identities:
            raise ValueError("duplicate case identity")
        identities.add(identity)
        run_tags.add(row.get("run_tag"))
        commits.add(row.get("source_commit"))
        _valid_digest(
            row.get("source_commit"),
            "source commit",
            lengths=(40, 64),
        )
        if not isinstance(row.get("run_tag"), str) or not row["run_tag"]:
            raise ValueError("run tag is invalid")
        config = POLICY_CONFIGS[policy]
        if (
            row.get("selectable") is not config["selectable"]
            or row.get("burst_width") != config["width"]
        ):
            raise ValueError("case policy metadata mismatch")
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != _case_shape(bucket):
            raise ValueError("context shape mismatch")
        generated = row["generated_tokens"]
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != generated
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in output_ids
            )
        ):
            raise ValueError("output token inventory mismatch")
        _valid_digest(
            row.get("output_text_sha256"),
            "output text digest",
        )
        if row.get("correctness_trace") is not False:
            raise ValueError("performance row enabled correctness trace")
        samples = row.get("amortized_tpot_samples_ns")
        if (
            not isinstance(samples, list)
            or len(samples) != generated - 1
        ):
            raise ValueError("amortized TPOT inventory mismatch")
        for value in samples:
            _finite_non_negative(value, "amortized TPOT")
        expected_statistics = {
            "amortized_tpot_median_ns": statistics.median(samples),
            "amortized_tpot_p95_ns": _nearest_rank(samples, 0.95),
            "amortized_tpot_p99_ns": _nearest_rank(samples, 0.99),
        }
        for field, expected in expected_statistics.items():
            if row.get(field) != expected:
                raise ValueError(f"{field} does not match samples")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "maximum_host_visible_burst_gap_ns",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            *CAPTURE_COST_FIELDS,
        ):
            _finite_non_negative(row.get(field), field)
        for field in (
            "decode_host_ns",
            "decode_cuda_ns",
            "host_visible_burst_gaps_ns",
        ):
            values = row.get(field)
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list")
            for value in values:
                _finite_non_negative(value, field)
        gaps = row["host_visible_burst_gaps_ns"]
        if row["maximum_host_visible_burst_gap_ns"] != max(
            gaps,
            default=0,
        ):
            raise ValueError("maximum host-visible gap mismatch")
        summary = row.get("exact_greedy_decode_burst_summary")
        _validate_summary_shape(
            summary,
            policy=policy,
            correctness_trace=False,
        )
        if (
            summary["maximum_host_visible_gap_ns"]
            != row["maximum_host_visible_burst_gap_ns"]
        ):
            raise ValueError("summary host-visible gap mismatch")
        expected_profiles = (
            summary["commits"]
            if config["enabled"]
            else generated - 1
        )
        if (
            len(row["decode_host_ns"]) != expected_profiles
            or len(row["decode_cuda_ns"]) != expected_profiles
        ):
            raise ValueError("decode profile inventory mismatch")
        receipt = (
            summary["capture_receipts"][0]
            if summary["capture_receipts"]
            else None
        )
        expected_costs = {
            "capture_duration_ns": (
                receipt["capture_duration_ns"] if receipt else 0
            ),
            "capture_allocated_delta_bytes": (
                receipt["allocated_delta_bytes"] if receipt else 0
            ),
            "capture_reserved_delta_bytes": (
                receipt["reserved_delta_bytes"] if receipt else 0
            ),
            "capture_retained_static_bytes": (
                receipt["retained_static_bytes"] if receipt else 0
            ),
            "reserved_scratch_blocks": (
                receipt["scratch_block_count"] if receipt else 0
            ),
        }
        for field, expected in expected_costs.items():
            if row[field] != expected:
                raise ValueError(f"capture cost mismatch: {field}")
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
        for policy in POLICIES
    }
    if identities != expected:
        raise ValueError("case row inventory mismatch")
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "performance rows do not share source identity"
        )
    return rows


def _safe_sidecar(run_dir: Path, row: dict) -> tuple[float, ...]:
    raw_path = row.get("logits_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("sidecar path mismatch")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("sidecar path escapes run directory")
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
        or expected_bytes != expected_count * 4
        or len(payload) != expected_bytes
    ):
        raise ValueError("sidecar byte length mismatch")
    if _sha256_file(path) != row.get("logits_sha256"):
        raise ValueError("sidecar digest mismatch")
    values = struct.unpack(f"<{expected_count}f", payload)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sidecar contains non-finite value")
    return values


def _sampling_index(point: str) -> int:
    return {
        "prefill-final": 0,
        "decode-first": 1,
        "decode-middle": 64,
        "decode-final": 127,
    }[point]


def _validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
) -> tuple[list[dict], dict[tuple[str, str, str], tuple[float, ...]]]:
    if len(rows) != 48:
        raise ValueError(
            f"expected exactly 48 correctness rows, got {len(rows)}"
        )
    identities = set()
    values_by_identity = {}
    run_tags = set()
    commits = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CORRECTNESS_SCHEMA
        ):
            raise ValueError("correctness row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        if identity in identities:
            raise ValueError("duplicate correctness identity")
        identities.add(identity)
        bucket, policy, point = identity
        if (
            bucket not in {item[0] for item in CONTEXTS}
            or policy not in POLICIES
            or point not in POINTS
        ):
            raise ValueError("correctness identity mismatch")
        if (
            row.get("prompt_tokens"),
            row.get("generated_tokens"),
        ) != _case_shape(bucket):
            raise ValueError("correctness context shape mismatch")
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != row["generated_tokens"]
        ):
            raise ValueError(
                "correctness output inventory mismatch"
            )
        _valid_digest(
            row.get("output_text_sha256"),
            "correctness output text digest",
        )
        _valid_digest(
            row.get("source_commit"),
            "correctness source commit",
            lengths=(40, 64),
        )
        run_tags.add(row.get("run_tag"))
        commits.add(row.get("source_commit"))
        if (
            row.get("correctness_trace") is not True
            or row.get("trace_identity") != TRACE_IDENTITY
        ):
            raise ValueError("correctness trace identity mismatch")
        burst_sample = (
            POLICY_CONFIGS[policy]["enabled"]
            and point != "prefill-final"
        )
        if burst_sample:
            _valid_digest(
                row.get("trace_graph_identity_sha256"),
                "trace graph identity",
            )
            expected_ordinal = _sampling_index(point) - 1
            if not POLICY_CONFIGS[policy][
                "epoch_relative_sampling"
            ]:
                expected_ordinal %= POLICY_CONFIGS[policy]["width"]
            if row.get("selected_replay_ordinal") != expected_ordinal:
                raise ValueError("selected replay ordinal mismatch")
            if row.get("sampled_logit_d2h_calls") != 1:
                raise ValueError(
                    "sampled-logit D2H inventory mismatch"
                )
        elif any((
            row.get("trace_graph_identity_sha256") is not None,
            row.get("selected_replay_ordinal") is not None,
            row.get("sampled_logit_d2h_calls") != 0,
        )):
            raise ValueError("non-burst sample reported trace state")
        shape = row.get("logits_shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != 1
            or isinstance(shape[1], bool)
            or not isinstance(shape[1], int)
            or shape[1] <= 0
        ):
            raise ValueError("correctness logits shape mismatch")
        values = _safe_sidecar(run_dir, row)
        if len(values) != shape[0] * shape[1]:
            raise ValueError("correctness logits element mismatch")
        values_by_identity[identity] = values
        summary = row.get("exact_greedy_decode_burst_summary")
        _validate_summary_shape(
            summary,
            policy=policy,
            correctness_trace=POLICY_CONFIGS[policy]["enabled"],
        )
        if POLICY_CONFIGS[policy]["enabled"]:
            receipt = summary["capture_receipts"][0]
            if (
                burst_sample
                and receipt["graph_identity_sha256"]
                != row["trace_graph_identity_sha256"]
            ):
                raise ValueError("trace graph identity mismatch")
            if summary["sampled_logit_d2h_calls"] != 3:
                raise ValueError(
                    "correctness sampled-logit inventory mismatch"
                )
    expected = {
        (bucket, policy, point)
        for bucket, _prompt, _generated in CONTEXTS
        for policy in POLICIES
        for point in POINTS
    }
    if identities != expected:
        raise ValueError("correctness row inventory mismatch")
    if len(run_tags) != 1 or len(commits) != 1:
        raise ValueError(
            "correctness rows do not share source identity"
        )
    return rows, values_by_identity


def _summary_from_rows(rows: list[dict]) -> dict:
    by_identity = {
        (
            row["repetition"],
            row["context_bucket"],
            row["policy"],
        ): row
        for row in rows
    }
    comparisons = []
    all_outputs_exact = True
    for repetition in range(5):
        for bucket, _prompt, _generated in CONTEXTS:
            group = [
                by_identity[(repetition, bucket, policy)]
                for policy in POLICIES
            ]
            all_outputs_exact &= (
                len({
                    tuple(row["output_token_ids"]) for row in group
                }) == 1
                and len({
                    row["output_text_sha256"] for row in group
                }) == 1
            )
            comparisons.append({
                "repetition": repetition,
                "context_bucket": bucket,
                "amortized_tpot_median_ns": {
                    row["policy"]:
                        row["amortized_tpot_median_ns"]
                    for row in group
                },
                "amortized_tpot_p95_ns": {
                    row["policy"]:
                        row["amortized_tpot_p95_ns"]
                    for row in group
                },
            })
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "row_count": len(rows),
        "comparison_set_count": len(comparisons),
        "all_outputs_exact": all_outputs_exact,
        "comparisons": comparisons,
        "correctness_row_count": 48,
    }


def _metric_summary(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    baseline_policy: str,
    candidate_policy: str,
) -> dict:
    baseline_tpot = [
        float(value)
        for row in baseline_rows
        for value in row["amortized_tpot_samples_ns"]
    ]
    candidate_tpot = [
        float(value)
        for row in candidate_rows
        for value in row["amortized_tpot_samples_ns"]
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
        "baseline_cuda_peak_allocated_bytes": baseline_allocated,
        "candidate_cuda_peak_allocated_bytes": candidate_allocated,
        "cuda_allocated_delta_bytes":
            candidate_allocated - baseline_allocated,
        "baseline_cuda_peak_reserved_bytes": baseline_reserved,
        "candidate_cuda_peak_reserved_bytes": candidate_reserved,
        "cuda_reserved_delta_bytes":
            candidate_reserved - baseline_reserved,
        "cuda_reserved_regression_fraction":
            _relative_change(baseline_reserved, candidate_reserved),
    }


def _lifecycle(rows: list[dict], policy: str) -> dict:
    result = {
        "replay_complete": True,
        "d2h_complete": True,
        "lease_complete": True,
        "execution_complete": True,
    }
    for row in rows:
        if row["policy"] != policy:
            continue
        summary = row["exact_greedy_decode_burst_summary"]
        expected_replays = row["generated_tokens"] - 1
        expected_commits = math.ceil(
            expected_replays / POLICY_CONFIGS[policy]["width"]
        )
        width = POLICY_CONFIGS[policy]["width"]
        partial_width = expected_replays % width
        expected_authorized = {str(width): expected_commits}
        if partial_width:
            expected_authorized[str(width)] -= 1
            expected_authorized[str(partial_width)] = 1
        result["replay_complete"] &= all((
            summary["target_model_forwards"] == expected_replays,
            summary["graph_replays"] == expected_replays,
            summary["committed_tokens"] == expected_replays,
            summary["attempts"] == expected_commits,
            summary["acceptances"] == expected_commits,
            summary["commits"] == expected_commits,
            summary["requested_width_histogram"]
            == {str(width): expected_commits},
            summary["authorized_width_histogram"]
            == expected_authorized,
            summary["output_budget_clipped"]
            == (0 if width == 1 else 1),
            summary["block_boundary_clipped"] == 0,
        ))
        result["d2h_complete"] &= all((
            summary["intermediate_token_d2h_calls"] == 0,
            summary["final_token_d2h_calls"] == expected_commits,
            summary["final_token_d2h_bytes"]
            == expected_replays * 8,
            summary["sampled_logit_d2h_calls"] == 0,
            len(row["host_visible_burst_gaps_ns"])
            == expected_commits,
        ))
        result["lease_complete"] &= (
            summary["pending_leases"] == 0
        )
        result["execution_complete"] &= all((
            summary["failures"] == 0,
            summary["quarantines"] == 0,
            summary["quarantine_reason"] is None,
        ))
    return result


def _correctness(
    rows: list[dict],
    values_by_identity: dict,
) -> tuple[dict, bool]:
    by_identity = {
        (
            row["context_bucket"],
            row["policy"],
            row["sampling_point"],
        ): row
        for row in rows
    }
    continuation_passed = True
    pairs = []
    global_max = 0.0
    total_abs = 0.0
    total_count = 0
    all_argmax = True
    all_output_ids = True
    all_output_text = True
    for baseline in REFERENCE_POLICIES:
        for bucket, _prompt, _generated in CONTEXTS:
            for point in POINTS:
                left = by_identity[(bucket, baseline, point)]
                right = by_identity[
                    (bucket, CONTINUATION_POLICY, point)
                ]
                if (
                    left["logits_shape"] != right["logits_shape"]
                    or left["logits_element_count"]
                    != right["logits_element_count"]
                ):
                    raise ValueError("paired logits shape mismatch")
                left_values = values_by_identity[
                    (bucket, baseline, point)
                ]
                right_values = values_by_identity[
                    (bucket, CONTINUATION_POLICY, point)
                ]
                differences = [
                    abs(a - b)
                    for a, b in zip(left_values, right_values)
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
                continuation_passed &= passed
                global_max = max(global_max, maximum)
                total_abs += sum(differences)
                total_count += len(differences)
                all_argmax &= argmax_equal
                all_output_ids &= ids_equal
                all_output_text &= text_equal
                pairs.append({
                    "baseline_policy": baseline,
                    "candidate_policy": CONTINUATION_POLICY,
                    "context_bucket": bucket,
                    "sampling_point": point,
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "argmax_equal": argmax_equal,
                    "output_ids_exact": ids_equal,
                    "output_text_exact": text_equal,
                    "passed": passed,
                })
    return ({
        "pair_count": len(pairs),
        "max_abs": global_max,
        "mean_abs": (
            total_abs / total_count if total_count else 0.0
        ),
        "argmax_equal": all_argmax,
        "output_ids_exact": all_output_ids,
        "output_text_exact": all_output_text,
        "pairs": pairs,
    }, continuation_passed)


def _cost(rows: list[dict], policy: str) -> dict:
    selected = [row for row in rows if row["policy"] == policy]
    return {
        field: {
            "min": min(int(row[field]) for row in selected),
            "max": max(int(row[field]) for row in selected),
        }
        for field in CAPTURE_COST_FIELDS
    }


def _evaluation(
    rows: list[dict],
    *,
    policy: str,
    correctness_passed: bool,
    lifecycle: dict,
) -> dict:
    host = [row for row in rows if row["policy"] == "host_greedy"]
    k1 = [
        row for row in rows
        if row["policy"] == "full_step_graph_k1"
    ]
    candidate = [row for row in rows if row["policy"] == policy]
    aggregate_host = _metric_summary(
        host,
        candidate,
        baseline_policy="host_greedy",
        candidate_policy=policy,
    )
    aggregate_k1 = _metric_summary(
        k1,
        candidate,
        baseline_policy="full_step_graph_k1",
        candidate_policy=policy,
    )
    by_bucket = {}
    bucket_regressions = []
    latency_regressions = []
    throughput_regressions = []
    winning_buckets = 0
    maximum_gap = 0
    for bucket, _prompt, _generated in CONTEXTS:
        host_bucket = [
            row for row in host if row["context_bucket"] == bucket
        ]
        candidate_bucket = [
            row for row in candidate
            if row["context_bucket"] == bucket
        ]
        metrics = _metric_summary(
            host_bucket,
            candidate_bucket,
            baseline_policy="host_greedy",
            candidate_policy=policy,
        )
        by_bucket[bucket] = metrics
        if (
            metrics["tpot_median_improvement_fraction"]
            >= BUCKET_MEDIAN_THRESHOLD
        ):
            winning_buckets += 1
        if (
            metrics["tpot_median_improvement_fraction"]
            < -BUCKET_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:median_tpot")
        if (
            metrics["tpot_p95_improvement_fraction"]
            < -BUCKET_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:p95_tpot")
        if (
            metrics["ttft_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:e2e")
        if (
            metrics["throughput_regression_fraction"]
            > THROUGHPUT_REGRESSION_LIMIT
        ):
            throughput_regressions.append(
                f"{bucket}:throughput"
            )
        maximum_gap = max(
            maximum_gap,
            *(
                int(row["maximum_host_visible_burst_gap_ns"])
                for row in candidate_bucket
            ),
        )
    cost = _cost(rows, policy)
    return {
        "policy": policy,
        "burst_width": POLICY_CONFIGS[policy]["width"],
        "correctness_passed": correctness_passed,
        **lifecycle,
        "host_median_passed": (
            aggregate_host["tpot_median_improvement_fraction"]
            >= HOST_MEDIAN_THRESHOLD
        ),
        "host_p95_passed": (
            aggregate_host["tpot_p95_improvement_fraction"]
            >= HOST_P95_THRESHOLD
        ),
        "winning_bucket_count": winning_buckets,
        "bucket_coverage_passed":
            winning_buckets >= MIN_WINNING_BUCKETS,
        "k1_incremental_passed": (
            aggregate_k1["tpot_median_improvement_fraction"]
            >= K1_MEDIAN_THRESHOLD
        ),
        "bucket_regressions": bucket_regressions,
        "latency_regressions": latency_regressions,
        "throughput_regressions": throughput_regressions,
        "memory_regression": (
            aggregate_host["cuda_reserved_regression_fraction"]
            > MEMORY_REGRESSION_LIMIT
        ),
        "maximum_host_visible_burst_gap_ns": maximum_gap,
        "visibility_passed":
            maximum_gap <= VISIBILITY_GAP_LIMIT_NS,
        "cost_complete": all((
            cost["capture_duration_ns"]["min"] > 0,
            cost["capture_retained_static_bytes"]["min"] > 0,
            cost["reserved_scratch_blocks"]["min"] == 1,
            cost["reserved_scratch_blocks"]["max"] == 1,
        )),
        "aggregate": {
            "host_vs_candidate": aggregate_host,
            "k1_vs_candidate": aggregate_k1,
        },
        "by_bucket": by_bucket,
        "cost": cost,
    }


def _classification(evaluation: dict) -> str:
    for field, result in (
        ("correctness_passed", "NO_GO_CORRECTNESS"),
        ("replay_complete", "NO_GO_REPLAY_INCOMPLETE"),
        ("d2h_complete", "NO_GO_D2H_LIFECYCLE"),
        ("lease_complete", "NO_GO_LEASE_LIFECYCLE"),
        ("execution_complete", "NO_GO_EXECUTION_FAILURE"),
        ("host_median_passed", "NO_GO_HOST_TPOT_MEDIAN"),
        ("host_p95_passed", "NO_GO_HOST_TPOT_P95"),
        ("bucket_coverage_passed", "NO_GO_BUCKET_COVERAGE"),
        ("k1_incremental_passed", "NO_GO_K1_INCREMENTAL"),
    ):
        if not evaluation[field]:
            return result
    if evaluation["bucket_regressions"]:
        return "NO_GO_BUCKET_REGRESSION"
    if evaluation["latency_regressions"]:
        return "NO_GO_TTFT_E2E"
    if evaluation["throughput_regressions"]:
        return "NO_GO_THROUGHPUT"
    if evaluation["memory_regression"]:
        return "NO_GO_MEMORY"
    if not evaluation["visibility_passed"]:
        return "NO_GO_VISIBILITY_GAP"
    if not evaluation["cost_complete"]:
        return "NO_GO_COST_INCOMPLETE"
    return "GO_EXACT_BURST_CONTINUATION_EPOCH"


def _reconstruct_comparison(
    performance_rows: list[dict],
    correctness_rows: list[dict],
    values_by_identity: dict,
) -> dict:
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in performance_rows
    }
    output_exact = True
    for bucket, _prompt, _generated in CONTEXTS:
        for repetition in range(5):
            group = [
                by_identity[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            output_exact &= (
                len({
                    tuple(row["output_token_ids"]) for row in group
                }) == 1
                and len({
                    row["output_text_sha256"] for row in group
                }) == 1
            )
    correctness, candidate_correctness = _correctness(
        correctness_rows,
        values_by_identity,
    )
    causal_lifecycle = _lifecycle(
        performance_rows,
        "full_step_graph_k1",
    )
    evaluations = {
        policy: _evaluation(
            performance_rows,
            policy=policy,
            correctness_passed=(
                output_exact and candidate_correctness[policy]
            ),
            lifecycle={
                field: (
                    _lifecycle(performance_rows, policy)[field]
                    and causal_lifecycle[field]
                )
                for field in (
                    "replay_complete",
                    "d2h_complete",
                    "lease_complete",
                    "execution_complete",
                )
            },
        )
        for policy in BURST_POLICIES
    }
    eligible_policies = [
        policy
        for policy, evaluation in evaluations.items()
        if all((
            evaluation["correctness_passed"],
            evaluation["replay_complete"],
            evaluation["d2h_complete"],
            evaluation["lease_complete"],
            evaluation["execution_complete"],
            not evaluation["bucket_regressions"],
            not evaluation["latency_regressions"],
            not evaluation["throughput_regressions"],
            not evaluation["memory_regression"],
            evaluation["visibility_passed"],
            evaluation["cost_complete"],
        ))
    ]
    selection_pool = eligible_policies or list(BURST_POLICIES)
    selected_policy = max(
        selection_pool,
        key=lambda policy: (
            evaluations[policy]["aggregate"][
                "host_vs_candidate"
            ]["tpot_median_improvement_fraction"],
            -POLICY_CONFIGS[policy]["width"],
        ),
    )
    selected = evaluations[selected_policy]
    run_tags = {
        *(row.get("run_tag") for row in performance_rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        *(row.get("source_commit") for row in performance_rows),
        *(row.get("source_commit") for row in correctness_rows),
    }
    evidence_complete = len(run_tags) == 1 and len(commits) == 1
    classification = _classification(selected)
    if not evidence_complete and classification.startswith("GO_"):
        classification = "NO_GO_EVIDENCE_INCOMPLETE"
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": performance_rows[0]["run_tag"],
        "source_commit": performance_rows[0]["source_commit"],
        "classification": classification,
        "selected_policy": selected_policy,
        "selected_burst_width": POLICY_CONFIGS[selected_policy]["width"],
        "selected_lifecycle_complete": all((
            selected["replay_complete"],
            selected["d2h_complete"],
            selected["lease_complete"],
            selected["execution_complete"],
        )),
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_LIMIT,
            "host_aggregate_median_min_improvement_fraction":
                HOST_MEDIAN_THRESHOLD,
            "host_aggregate_p95_min_improvement_fraction":
                HOST_P95_THRESHOLD,
            "bucket_median_min_improvement_fraction":
                BUCKET_MEDIAN_THRESHOLD,
            "minimum_winning_bucket_count": MIN_WINNING_BUCKETS,
            "k1_aggregate_median_min_improvement_fraction":
                K1_MEDIAN_THRESHOLD,
            "bucket_tpot_max_regression_fraction":
                BUCKET_REGRESSION_LIMIT,
            "latency_max_regression_fraction":
                LATENCY_REGRESSION_LIMIT,
            "throughput_max_regression_fraction":
                THROUGHPUT_REGRESSION_LIMIT,
            "reserved_memory_max_regression_fraction":
                MEMORY_REGRESSION_LIMIT,
            "maximum_host_visible_burst_gap_ns":
                VISIBILITY_GAP_LIMIT_NS,
        },
        "correctness": correctness,
        "candidate_evaluations": evaluations,
    }


def _continuation_coverage_v1(rows: list[dict]) -> dict:
    selected = [
        row for row in rows
        if row["policy"] == CONTINUATION_POLICY
    ]
    minimum_hits = min(
        row["exact_greedy_decode_burst_summary"][
            "continuation_hits"
        ]
        for row in selected
    )
    failures = []
    for row in selected:
        summary = row["exact_greedy_decode_burst_summary"]
        unexpected_misses = {
            reason: count
            for reason, count
            in summary["continuation_miss_counts"].items()
            if reason != "receipt_missing" and count
        }
        passed = all((
            summary["cold_binds"] == 1,
            summary["continuation_hits"] >= MIN_CONTINUATION_HITS,
            summary["continuation_attempts"] == summary["commits"],
            summary["continuation_hits"] + summary["cold_binds"]
            == summary["continuation_attempts"],
            summary["continuation_miss_counts"]
            .get("receipt_missing", 0) == 1,
            not unexpected_misses,
            not summary["continuation_invalidation_counts"],
            summary["failures"] == 0,
            summary["quarantines"] == 0,
            summary["pending_leases"] == 0,
        ))
        if not passed:
            failures.append({
                "context_bucket": row["context_bucket"],
                "repetition": row["repetition"],
                "cold_binds": summary["cold_binds"],
                "continuation_hits": summary["continuation_hits"],
                "continuation_miss_counts":
                    summary["continuation_miss_counts"],
                "continuation_invalidation_counts":
                    summary["continuation_invalidation_counts"],
                "failures": summary["failures"],
                "quarantines": summary["quarantines"],
                "pending_leases": summary["pending_leases"],
            })
    return {
        "request_count": len(selected),
        "minimum_required_hits": MIN_CONTINUATION_HITS,
        "minimum_hits": minimum_hits,
        "all_requests_passed": not failures,
        "failures": failures,
    }


def _continuation_evaluation_v1(
    rows: list[dict],
    *,
    correctness_passed: bool,
    coverage: dict,
) -> dict:
    k4_rows = [
        row for row in rows if row["policy"] == "decode_burst_k4"
    ]
    k8_rows = [
        row for row in rows if row["policy"] == "decode_burst_k8"
    ]
    candidate_rows = [
        row for row in rows
        if row["policy"] == CONTINUATION_POLICY
    ]
    aggregate_k4 = _metric_summary(
        k4_rows,
        candidate_rows,
        baseline_policy="decode_burst_k4",
        candidate_policy=CONTINUATION_POLICY,
    )
    aggregate_k8 = _metric_summary(
        k8_rows,
        candidate_rows,
        baseline_policy="decode_burst_k8",
        candidate_policy=CONTINUATION_POLICY,
    )
    by_bucket = {}
    bucket_regressions = []
    latency_regressions = []
    throughput_regressions = []
    winning_buckets = 0
    maximum_gap = 0
    maximum_k8_gap = 0
    for bucket, _prompt, _generated in CONTEXTS:
        k4_bucket = [
            row for row in k4_rows
            if row["context_bucket"] == bucket
        ]
        k8_bucket = [
            row for row in k8_rows
            if row["context_bucket"] == bucket
        ]
        candidate_bucket = [
            row for row in candidate_rows
            if row["context_bucket"] == bucket
        ]
        metrics = _metric_summary(
            k4_bucket,
            candidate_bucket,
            baseline_policy="decode_burst_k4",
            candidate_policy=CONTINUATION_POLICY,
        )
        by_bucket[bucket] = metrics
        if (
            metrics["tpot_median_improvement_fraction"]
            >= BUCKET_MEDIAN_THRESHOLD
        ):
            winning_buckets += 1
        if (
            metrics["tpot_median_improvement_fraction"]
            < -BUCKET_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:median_tpot")
        if (
            metrics["tpot_p95_improvement_fraction"]
            < -BUCKET_REGRESSION_LIMIT
        ):
            bucket_regressions.append(f"{bucket}:p95_tpot")
        if (
            metrics["ttft_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:e2e")
        if (
            metrics["throughput_regression_fraction"]
            > THROUGHPUT_REGRESSION_LIMIT
        ):
            throughput_regressions.append(
                f"{bucket}:throughput"
            )
        maximum_gap = max(
            maximum_gap,
            *(
                int(row["maximum_host_visible_burst_gap_ns"])
                for row in candidate_bucket
            ),
        )
        maximum_k8_gap = max(
            maximum_k8_gap,
            *(
                int(row["maximum_host_visible_burst_gap_ns"])
                for row in k8_bucket
            ),
        )
    visibility_ratio = (
        maximum_gap / maximum_k8_gap
        if maximum_k8_gap else math.inf
    )
    cost = _cost(rows, CONTINUATION_POLICY)
    minimum_skipped_bytes = min(
        row["exact_greedy_decode_burst_summary"][
            "skipped_block_table_bytes"
        ]
        for row in candidate_rows
    )
    return {
        "policy": CONTINUATION_POLICY,
        "burst_width": POLICY_CONFIGS[CONTINUATION_POLICY]["width"],
        "correctness_passed": correctness_passed,
        "continuation_coverage_passed":
            coverage["all_requests_passed"],
        "k4_median_passed": (
            aggregate_k4["tpot_median_improvement_fraction"]
            >= K4_MEDIAN_THRESHOLD
        ),
        "k4_p95_passed": (
            aggregate_k4["tpot_p95_improvement_fraction"]
            >= K4_P95_THRESHOLD
        ),
        "winning_bucket_count": winning_buckets,
        "bucket_coverage_passed":
            winning_buckets >= MIN_WINNING_BUCKETS,
        "k8_parity_passed": (
            aggregate_k8["tpot_median_improvement_fraction"]
            >= -K8_PARITY_LIMIT
        ),
        "bucket_regressions": bucket_regressions,
        "latency_regressions": latency_regressions,
        "throughput_regressions": throughput_regressions,
        "memory_regression": (
            aggregate_k4["cuda_reserved_regression_fraction"]
            > MEMORY_REGRESSION_LIMIT
        ),
        "maximum_host_visible_burst_gap_ns": maximum_gap,
        "paired_k8_maximum_host_visible_burst_gap_ns":
            maximum_k8_gap,
        "visibility_ratio": visibility_ratio,
        "visibility_passed":
            visibility_ratio <= VISIBILITY_RATIO_LIMIT,
        "cost_complete": all((
            cost["capture_duration_ns"]["min"] > 0,
            cost["capture_retained_static_bytes"]["min"] > 0,
            cost["reserved_scratch_blocks"]["min"] == 1,
            cost["reserved_scratch_blocks"]["max"] == 1,
            minimum_skipped_bytes > 0,
        )),
        "aggregate": {
            "k4_vs_continuation": aggregate_k4,
            "k8_vs_continuation": aggregate_k8,
        },
        "by_bucket": by_bucket,
        "cost": {
            **cost,
            "minimum_skipped_block_table_bytes":
                minimum_skipped_bytes,
        },
    }


def _classification_v1(evaluation: dict) -> str:
    for field, result in (
        ("correctness_passed", "NO_GO_CORRECTNESS"),
        (
            "continuation_coverage_passed",
            "NO_GO_CONTINUATION_COVERAGE",
        ),
        ("k4_median_passed", "NO_GO_K4_MEDIAN"),
        ("k4_p95_passed", "NO_GO_K4_P95"),
        ("k8_parity_passed", "NO_GO_K8_PARITY"),
        ("visibility_passed", "NO_GO_VISIBILITY_RATIO"),
        ("bucket_coverage_passed", "NO_GO_K4_MEDIAN"),
    ):
        if not evaluation[field]:
            return result
    if evaluation["bucket_regressions"]:
        return "NO_GO_BUCKET_REGRESSION"
    if evaluation["latency_regressions"]:
        return "NO_GO_TTFT_E2E"
    if evaluation["throughput_regressions"]:
        return "NO_GO_THROUGHPUT"
    if evaluation["memory_regression"]:
        return "NO_GO_MEMORY"
    if not evaluation["cost_complete"]:
        return "NO_GO_COST_INCOMPLETE"
    return "GO_EXACT_BURST_CONTINUATION_EPOCH"


def _reconstruct_comparison_v1(
    performance_rows: list[dict],
    correctness_rows: list[dict],
    values_by_identity: dict,
) -> dict:
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in performance_rows
    }
    output_exact = True
    for bucket, _prompt, _generated in CONTEXTS:
        for repetition in range(5):
            group = [
                by_identity[(bucket, repetition, policy)]
                for policy in POLICIES
            ]
            output_exact &= (
                len({
                    tuple(row["output_token_ids"]) for row in group
                }) == 1
                and len({
                    row["output_text_sha256"] for row in group
                }) == 1
            )
    correctness, correctness_passed = _correctness(
        correctness_rows,
        values_by_identity,
    )
    coverage = _continuation_coverage_v1(performance_rows)
    selected = _continuation_evaluation_v1(
        performance_rows,
        correctness_passed=output_exact and correctness_passed,
        coverage=coverage,
    )
    run_tags = {
        *(row.get("run_tag") for row in performance_rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    commits = {
        *(row.get("source_commit") for row in performance_rows),
        *(row.get("source_commit") for row in correctness_rows),
    }
    evidence_complete = len(run_tags) == 1 and len(commits) == 1
    classification = _classification_v1(selected)
    if not evidence_complete and classification.startswith("GO_"):
        classification = "NO_GO_EVIDENCE_INCOMPLETE"
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": performance_rows[0]["run_tag"],
        "source_commit": performance_rows[0]["source_commit"],
        "classification": classification,
        "selected_policy": CONTINUATION_POLICY,
        "selected_burst_width":
            POLICY_CONFIGS[CONTINUATION_POLICY]["width"],
        "selected_lifecycle_complete":
            selected["continuation_coverage_passed"],
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_LIMIT,
            "k4_aggregate_median_min_improvement_fraction":
                K4_MEDIAN_THRESHOLD,
            "k4_aggregate_p95_min_improvement_fraction":
                K4_P95_THRESHOLD,
            "bucket_median_min_improvement_fraction":
                BUCKET_MEDIAN_THRESHOLD,
            "minimum_winning_bucket_count": MIN_WINNING_BUCKETS,
            "k8_parity_max_regression_fraction":
                K8_PARITY_LIMIT,
            "bucket_tpot_max_regression_fraction":
                BUCKET_REGRESSION_LIMIT,
            "latency_max_regression_fraction":
                LATENCY_REGRESSION_LIMIT,
            "throughput_max_regression_fraction":
                THROUGHPUT_REGRESSION_LIMIT,
            "reserved_memory_max_regression_fraction":
                MEMORY_REGRESSION_LIMIT,
            "visibility_ratio_limit": VISIBILITY_RATIO_LIMIT,
            "minimum_continuation_hits":
                MIN_CONTINUATION_HITS,
        },
        "correctness": correctness,
        "continuation_coverage": coverage,
        "candidate_evaluations": {
            CONTINUATION_POLICY: selected,
        },
    }


def _validate_source(source, *, repo_root: Path) -> None:
    if (
        not isinstance(source, dict)
        or source.get("schema_version") != SOURCE_SCHEMA
    ):
        raise ValueError("source manifest is invalid")
    digests = source.get("source_sha256")
    if (
        not isinstance(digests, dict)
        or set(digests) != set(SOURCE_FILES)
    ):
        raise ValueError("source manifest file inventory mismatch")
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        if _sha256_file(path) != digests[relative]:
            raise ValueError(f"source digest mismatch: {relative}")


def _validate_workload(workload) -> None:
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version") != WORKLOAD_SCHEMA
    ):
        raise ValueError("workload manifest is invalid")
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
        "performance_row_count": 60,
        "correctness_row_count": 48,
        "performance_correctness_trace": False,
        "correctness_trace_identity": TRACE_IDENTITY,
        "correctness_sampling_points": list(POINTS),
        "policy_configs": {
            policy: dict(config)
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
    for field, value in expected.items():
        if workload.get(field) != value:
            raise ValueError(
                f"workload manifest mismatch: {field}"
            )
    model = workload.get("model")
    if (
        not isinstance(model, str)
        or Path(model).name not in STAGE1_MODEL_BASENAMES
    ):
        raise ValueError("workload manifest mismatch: model")
    utilization = workload.get("gpu_memory_utilization")
    if (
        isinstance(utilization, bool)
        or not isinstance(utilization, (int, float))
        or not math.isfinite(float(utilization))
        or not 0.0 < float(utilization) <= 1.0
    ):
        raise ValueError(
            "workload manifest mismatch: gpu_memory_utilization"
        )
    environment = workload.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("torch_available") is not True
        or environment.get("cuda_available") is not True
        or not isinstance(environment.get("torch_version"), str)
        or not environment["torch_version"]
        or not isinstance(
            environment.get("cuda_runtime_version"),
            str,
        )
        or not environment["cuda_runtime_version"]
        or not isinstance(
            environment.get("cuda_device_name"),
            str,
        )
        or not environment["cuda_device_name"]
    ):
        raise ValueError(
            "workload manifest mismatch: environment"
        )


def _validate_manifest(
    run_dir: Path,
    manifest,
    correctness_rows: list[dict],
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ValueError("manifest schema mismatch")
    expected_files = PRIMARY_ARTIFACTS | {
        row["logits_path"] for row in correctness_rows
    }
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != expected_files
    ):
        raise ValueError("manifest file inventory mismatch")
    for relative, expected_digest in artifacts.items():
        _valid_digest(expected_digest, "manifest digest")
        if _sha256_file(run_dir / relative) != expected_digest:
            raise ValueError(
                f"manifest digest mismatch: {relative}"
            )


def verify_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)
    performance_rows = _load_jsonl(
        run_dir / "case_rows.jsonl"
    )
    correctness_rows = _load_jsonl(
        run_dir / "correctness_rows.jsonl"
    )
    source = _load_json(run_dir / "source_manifest.json")
    workload = _load_json(run_dir / "workload_manifest.json")
    summary = _load_json(run_dir / "summary.json")
    comparison = _load_json(run_dir / "comparison.json")
    gate = _load_json(run_dir / "gate.json")
    manifest = _load_json(run_dir / "manifest.sha256")
    _validate_manifest(run_dir, manifest, correctness_rows)
    _validate_source(source, repo_root=repo_root)
    _validate_workload(workload)
    validated_rows = _validate_performance_rows(performance_rows)
    validated_correctness, values = _validate_correctness_rows(
        correctness_rows,
        run_dir=run_dir,
    )
    identities = {
        source.get("run_tag"),
        workload.get("run_tag"),
        manifest.get("run_tag"),
        *(row.get("run_tag") for row in validated_rows),
        *(
            row.get("run_tag")
            for row in validated_correctness
        ),
    }
    commits = {
        source.get("source_commit"),
        workload.get("source_commit"),
        manifest.get("source_commit"),
        *(row.get("source_commit") for row in validated_rows),
        *(
            row.get("source_commit")
            for row in validated_correctness
        ),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    expected_summary = _summary_from_rows(validated_rows)
    if summary != expected_summary:
        raise ValueError("worker summary drift")
    reconstructed = _reconstruct_comparison_v1(
        validated_rows,
        validated_correctness,
        values,
    )
    reconstructed["evidence_sha256"] = {
        name: _sha256_file(run_dir / name)
        for name in (
            "case_rows.jsonl",
            "correctness_rows.jsonl",
            "source_manifest.json",
            "workload_manifest.json",
            "summary.json",
        )
    }
    if comparison != reconstructed:
        raise ValueError("comparison drift")
    expected_comparison_sha = _sha256_file(
        run_dir / "comparison.json"
    )
    expected_gate = {
        "schema_version": GATE_SCHEMA,
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "classification": reconstructed["classification"],
        "selected_policy": reconstructed["selected_policy"],
        "selected_burst_width":
            reconstructed["selected_burst_width"],
        "comparison_sha256": expected_comparison_sha,
    }
    if gate != expected_gate:
        if gate.get("classification") != reconstructed["classification"]:
            raise ValueError("classification drift")
        if gate.get("selected_policy") != reconstructed["selected_policy"]:
            raise ValueError("selected arm drift")
        raise ValueError("gate drift")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "reconstructed_classification":
            reconstructed["classification"],
        "reconstructed_selected_policy":
            reconstructed["selected_policy"],
        "reconstructed_selected_burst_width":
            reconstructed["selected_burst_width"],
        "performance_row_count": len(validated_rows),
        "correctness_row_count": len(validated_correctness),
        "comparison_sha256": expected_comparison_sha,
        "manifest_sha256": _sha256_file(
            run_dir / "manifest.sha256"
        ),
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
