#!/usr/bin/env python3
"""Independent verifier for the elastic exact-burst terminal bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct


SCHEMA_VERSION = "context-gated-elastic-exact-burst.terminal.v1"
VERIFICATION_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.terminal-verification.v1"
)
MANIFEST_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.terminal-manifest.v1"
)
CASE_SCHEMA_VERSION = "context-gated-elastic-exact-burst.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.correctness.v1"
)
PROFILE_SUMMARY_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.summary.v1"
)
PROFILE_SOURCE_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.source.v1"
)
PROFILE_WORKLOAD_SCHEMA_VERSION = (
    "context-gated-elastic-exact-burst.workload.v1"
)
POLICIES = ("fixed_k8", "context_gated_elastic_k16")
CONTEXT_LENGTHS = (256, 2048, 4096, 8192)
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
GENERATED_TOKENS = 128
TERMINAL_REPETITIONS = 5
WARMUP_REPETITIONS = 2
PERFORMANCE_ROW_COUNT = 40
CORRECTNESS_ROW_COUNT = 32
MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT = 2.0
MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT = 1.0
MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT = 2.0
MAXIMUM_LATENCY_REGRESSION_PCT = 2.0
MAXIMUM_THROUGHPUT_REGRESSION_PCT = 1.0
MAXIMUM_MEMORY_REGRESSION_PCT = 3.0
MAXIMUM_K16_HOST_VISIBLE_GAP_NS = 40_000_000
GO = "GO_CONTEXT_GATED_ELASTIC_EXACT_BURST"
NO_GO_EVIDENCE_INCOMPLETE = "NO_GO_EVIDENCE_INCOMPLETE"
NO_GO_CORRECTNESS = "NO_GO_CORRECTNESS"
NO_GO_WIDTH_POLICY = "NO_GO_WIDTH_POLICY"
NO_GO_RUNTIME_INVARIANT = "NO_GO_RUNTIME_INVARIANT"
NO_GO_BURST_GAP = "NO_GO_BURST_GAP"
NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT = (
    "NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT"
)
NO_GO_PROTECTED_REGRESSION = "NO_GO_PROTECTED_REGRESSION"

PROFILE_SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_context_gated_elastic_exact_burst.py",
    "tools/test_profile_context_gated_elastic_exact_burst.py",
)
SOURCE_FILES = tuple(dict.fromkeys((
    *PROFILE_SOURCE_FILES,
    "tools/context_gated_elastic_exact_burst_ceiling.py",
    "tools/test_context_gated_elastic_exact_burst_ceiling.py",
    "tools/context_gated_elastic_exact_burst_gate.py",
    "tools/test_context_gated_elastic_exact_burst_gate.py",
    "tools/context_gated_elastic_exact_burst_verify.py",
    "tools/test_context_gated_elastic_exact_burst_verify.py",
    "tools/run_context_gated_elastic_exact_burst_remote.py",
    "tools/test_run_context_gated_elastic_exact_burst_remote.py",
)))
PRIMARY_ARTIFACTS = {
    "workload_manifest.json",
    "source_manifest.json",
    "source.patch",
    "performance_rows.jsonl",
    "correctness_rows.jsonl",
    "profile_summary.json",
    "terminal_source_manifest.json",
    "terminal_summary.json",
    "terminal_gate.json",
    "producer_receipt.json",
}


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    if not Path(path).is_file():
        raise ValueError(f"required artifact is missing: {Path(path).name}")
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        raise ValueError(f"required artifact is missing: {Path(path).name}")
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_artifact_path(run_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("artifact path is invalid")
    root = Path(run_dir).resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file() or path.is_symlink():
        raise ValueError(f"artifact is missing: {relative}")
    return path


def _finite(value, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _finite_non_negative(value, field: str) -> float:
    normalized = _finite(value, field)
    if normalized < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return normalized


def _non_negative_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _nearest_rank(values, percentile: float) -> float:
    ordered = sorted(_finite(value, "metric sample") for value in values)
    if not ordered:
        raise ValueError("metric sample inventory is empty")
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _relative_change_pct(control: float, candidate: float) -> float:
    control = _finite(control, "control metric")
    candidate = _finite(candidate, "candidate metric")
    if control <= 0.0:
        if candidate == control:
            return 0.0
        raise ValueError("control metric must be positive")
    return (candidate - control) / control * 100.0


def _improvement_pct(control: float, candidate: float) -> float:
    return -_relative_change_pct(control, candidate)


def policy_order(repetition: int, context_index: int) -> tuple[str, str]:
    if (repetition + context_index) % 2:
        return tuple(reversed(POLICIES))
    return POLICIES


def _expected_performance_identities() -> set[tuple[int, int, str]]:
    return {
        (repetition, context, policy)
        for repetition in range(TERMINAL_REPETITIONS)
        for context_index, context in enumerate(CONTEXT_LENGTHS)
        for policy in policy_order(repetition, context_index)
    }


def _expected_correctness_identities() -> set[tuple[int, str, str]]:
    return {
        (context, policy, point)
        for context in CONTEXT_LENGTHS
        for policy in POLICIES
        for point in SAMPLING_POINTS
    }


def _validate_digest(value, field: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _validate_summary(summary: object, *, policy: str, context: int) -> dict:
    if not isinstance(summary, dict):
        raise ValueError("burst summary is invalid")
    for field in (
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
        "k16_attempts",
        "k16_acceptances",
        "k8_fallbacks",
        "lease_local_delta_journal_attempts",
        "lease_local_delta_journal_captures",
        "lease_local_delta_journal_commits",
        "lease_local_delta_journal_rollbacks",
        "lease_local_delta_journal_one_phase_attempts",
        "lease_local_delta_journal_one_phase_captures",
        "lease_local_delta_journal_one_phase_commits",
        "lease_local_delta_journal_one_phase_rollbacks",
    ):
        _non_negative_int(summary.get(field), field)
    _non_negative_int(
        summary.get("maximum_host_visible_gap_ns"),
        "maximum_host_visible_gap_ns",
    )
    for field in (
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
        "elastic_k16_fallback_counts",
        "per_width_commits",
        "lease_local_delta_journal_fallback_counts",
        "lease_local_delta_journal_one_phase_fallback_counts",
    ):
        value = summary.get(field)
        if not isinstance(value, dict):
            raise ValueError(f"{field} is invalid")
        for key, count in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} key is invalid")
            _non_negative_int(count, f"{field}[{key}]")
    receipts = summary.get("capture_receipts")
    if not isinstance(receipts, list) or len(receipts) != 1:
        raise ValueError("capture receipt inventory mismatch")
    receipt = receipts[0]
    if not isinstance(receipt, dict):
        raise ValueError("capture receipt is invalid")
    _validate_digest(
        receipt.get("graph_identity_sha256"),
        "graph identity",
    )
    for field in (
        "capture_duration_ns",
        "allocated_delta_bytes",
        "reserved_delta_bytes",
        "retained_static_bytes",
        "scratch_block_count",
    ):
        _non_negative_int(receipt.get(field), field)
    if summary.get("quarantine_reason") is not None:
        raise ValueError("profile row cannot be quarantined")
    selected = (
        summary["k16_acceptances"] > 0
        and summary["authorized_width_histogram"].get("16", 0) > 0
        and summary["per_width_commits"].get("16", 0) > 0
    )
    expected = policy == "context_gated_elastic_k16" and context <= 2048
    summary = dict(summary)
    summary["_selected_k16"] = selected
    summary["_expected_k16"] = expected
    return summary


def _validate_performance(rows: list[dict]) -> list[dict]:
    if len(rows) != PERFORMANCE_ROW_COUNT:
        raise ValueError("performance row inventory is incomplete")
    identities = set()
    normalized = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CASE_SCHEMA_VERSION
        ):
            raise ValueError("performance row schema mismatch")
        if not isinstance(row.get("run_tag"), str) or not row["run_tag"]:
            raise ValueError("run tag is invalid")
        repetition = row.get("repetition")
        context = row.get("context_length")
        policy = row.get("policy")
        identity = (repetition, context, policy)
        if identity in identities:
            raise ValueError("duplicate performance row")
        identities.add(identity)
        if identity not in _expected_performance_identities():
            raise ValueError("performance row identity is invalid")
        if (
            row.get("order_position")
            != policy_order(
                repetition,
                CONTEXT_LENGTHS.index(context),
            ).index(policy)
            or row.get("prompt_tokens") != context
            or row.get("generated_tokens") != GENERATED_TOKENS
            or row.get("temperature") != 0.0
            or row.get("ignore_eos") is not True
            or row.get("tensor_parallel_size") != 1
            or row.get("max_num_seqs") != 1
            or row.get("completion_only") is not True
            or row.get("correctness_trace") is not False
        ):
            raise ValueError("performance execution contract mismatch")
        _validate_digest(row.get("source_commit"), "source commit", (40, 64))
        _validate_digest(row.get("prompt_sha256"), "prompt digest")
        _validate_digest(row.get("output_text_sha256"), "text digest")
        tokens = row.get("output_token_ids")
        if (
            not isinstance(tokens, list)
            or len(tokens) != GENERATED_TOKENS
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in tokens
            )
        ):
            raise ValueError("performance output inventory mismatch")
        samples = row.get("amortized_tpot_samples_ns")
        if not isinstance(samples, list) or len(samples) != GENERATED_TOKENS - 1:
            raise ValueError("performance TPOT inventory mismatch")
        samples = [
            _finite_non_negative(value, "TPOT sample")
            for value in samples
        ]
        expected_tpot = (
            statistics.median(samples),
            _nearest_rank(samples, 0.95),
            _nearest_rank(samples, 0.99),
        )
        actual_tpot = (
            _finite_non_negative(
                row.get("amortized_tpot_median_ns"),
                "TPOT median",
            ),
            _finite_non_negative(
                row.get("amortized_tpot_p95_ns"),
                "TPOT P95",
            ),
            _finite_non_negative(
                row.get("amortized_tpot_p99_ns"),
                "TPOT P99",
            ),
        )
        if actual_tpot != expected_tpot:
            raise ValueError("TPOT summary mismatch")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "maximum_host_visible_burst_gap_ns",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "shared_capture_duration_ns",
            "shared_capture_allocated_delta_bytes",
            "shared_capture_reserved_delta_bytes",
            "shared_capture_retained_static_bytes",
            "elastic_incremental_allocated_bytes",
            "elastic_incremental_reserved_bytes",
            "elastic_incremental_retained_static_bytes",
        ):
            _finite_non_negative(row.get(field), field)
        for field in ("decode_host_ns", "decode_cuda_ns"):
            values = row.get(field)
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list")
            for value in values:
                _finite_non_negative(value, field)
        gaps = row.get("host_visible_burst_gaps_ns")
        if not isinstance(gaps, list):
            raise ValueError("host gap inventory mismatch")
        gaps = [_non_negative_int(value, "host gap") for value in gaps]
        if row["maximum_host_visible_burst_gap_ns"] != max(gaps, default=0):
            raise ValueError("maximum host gap mismatch")
        summary = _validate_summary(
            row.get("exact_greedy_decode_burst_summary"),
            policy=policy,
            context=context,
        )
        receipt = summary["capture_receipts"][0]
        if (
            row["shared_capture_duration_ns"]
            != receipt["capture_duration_ns"]
            or row["shared_capture_allocated_delta_bytes"]
            != receipt["allocated_delta_bytes"]
            or row["shared_capture_reserved_delta_bytes"]
            != receipt["reserved_delta_bytes"]
            or row["shared_capture_retained_static_bytes"]
            != receipt["retained_static_bytes"]
            or row["elastic_incremental_allocated_bytes"] != 0
            or row["elastic_incremental_reserved_bytes"] != 0
            or row["elastic_incremental_retained_static_bytes"] != 0
        ):
            raise ValueError("capture cost mismatch")
        item = dict(row)
        item["amortized_tpot_samples_ns"] = samples
        item["host_visible_burst_gaps_ns"] = gaps
        item["exact_greedy_decode_burst_summary"] = summary
        normalized.append(item)
    if identities != _expected_performance_identities():
        raise ValueError("performance row inventory is incomplete")
    if (
        len({row["run_tag"] for row in normalized}) != 1
        or len({row["source_commit"] for row in normalized}) != 1
    ):
        raise ValueError("performance source authority mismatch")
    return normalized


def _read_float32_sidecar(
    run_dir: Path,
    row: dict,
) -> tuple[float, ...]:
    path = _safe_artifact_path(run_dir, row.get("logits_path"))
    payload = path.read_bytes()
    if len(payload) != row.get("logits_byte_length"):
        raise ValueError("logit sidecar byte length mismatch")
    if sha256_file(path) != row.get("logits_sha256"):
        raise ValueError("logit sidecar hash mismatch")
    count = row.get("logits_element_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or len(payload) != count * 4
    ):
        raise ValueError("logit sidecar element count mismatch")
    return struct.unpack(f"<{count}f", payload)


def _validate_correctness(
    rows: list[dict],
    *,
    run_dir: Path,
) -> tuple[list[dict], bool]:
    if len(rows) != CORRECTNESS_ROW_COUNT:
        raise ValueError("correctness row inventory is incomplete")
    identities = set()
    normalized = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CORRECTNESS_SCHEMA_VERSION
        ):
            raise ValueError("correctness row schema mismatch")
        if not isinstance(row.get("run_tag"), str) or not row["run_tag"]:
            raise ValueError("correctness run tag is invalid")
        identity = (
            row.get("context_length"),
            row.get("policy"),
            row.get("sampling_point"),
        )
        if identity in identities:
            raise ValueError("duplicate correctness row")
        identities.add(identity)
        if identity not in _expected_correctness_identities():
            raise ValueError("correctness row identity is invalid")
        output_tokens = row.get("output_token_ids")
        if (
            row.get("generated_tokens") != GENERATED_TOKENS
            or row.get("correctness_trace") is not True
            or not isinstance(output_tokens, list)
            or len(output_tokens) != GENERATED_TOKENS
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in output_tokens
            )
        ):
            raise ValueError("correctness execution contract mismatch")
        _validate_digest(row.get("source_commit"), "source commit", (40, 64))
        _validate_digest(row.get("prompt_sha256"), "prompt digest")
        _validate_digest(row.get("output_text_sha256"), "text digest")
        values = _read_float32_sidecar(run_dir, row)
        shape = row.get("logits_shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
                for dimension in shape
            )
            or math.prod(shape) != len(values)
        ):
            raise ValueError("correctness logits shape mismatch")
        argmax = max(range(len(values)), key=values.__getitem__)
        if row.get("argmax_token_id") != argmax:
            raise ValueError("correctness argmax mismatch")
        summary = _validate_summary(
            row.get("exact_greedy_decode_burst_summary"),
            policy=row["policy"],
            context=row["context_length"],
        )
        item = dict(row)
        item["_logits"] = values
        item["exact_greedy_decode_burst_summary"] = summary
        normalized.append(item)
    if identities != _expected_correctness_identities():
        raise ValueError("correctness row inventory is incomplete")
    if (
        len({row["run_tag"] for row in normalized}) != 1
        or len({row["source_commit"] for row in normalized}) != 1
    ):
        raise ValueError("correctness source authority mismatch")
    indexed = {
        (
            row["context_length"],
            row["policy"],
            row["sampling_point"],
        ): row
        for row in normalized
    }
    exact = True
    for context in CONTEXT_LENGTHS:
        for point in SAMPLING_POINTS:
            control = indexed[(context, "fixed_k8", point)]
            candidate = indexed[
                (context, "context_gated_elastic_k16", point)
            ]
            exact = exact and (
                control["output_token_ids"] == candidate["output_token_ids"]
                and control["output_text_sha256"]
                == candidate["output_text_sha256"]
                and control["argmax_token_id"]
                == candidate["argmax_token_id"]
                and control["_logits"] == candidate["_logits"]
            )
    return normalized, exact


def _selected_k16(summary: dict) -> bool:
    return bool(summary["_selected_k16"])


def _width_policy_exact(rows: list[dict]) -> bool:
    return all(
        summary["_selected_k16"] is summary["_expected_k16"]
        for summary in (
            row["exact_greedy_decode_burst_summary"] for row in rows
        )
    )


def _runtime_inventory_exact(rows: list[dict]) -> bool:
    return all(
        (
            summary["target_model_forwards"]
            == summary["graph_replays"]
            == summary["committed_tokens"]
            == GENERATED_TOKENS - 1
            and summary["intermediate_token_d2h_calls"] == 0
            and summary["final_token_d2h_calls"] == summary["commits"]
            and summary["final_token_d2h_bytes"]
            == summary["committed_tokens"] * 8
        )
        for summary in (
            row["exact_greedy_decode_burst_summary"] for row in rows
        )
    )


def _expected_elastic_fallbacks(row: dict) -> set[str]:
    if row["policy"] != "context_gated_elastic_k16":
        return set()
    if row["context_length"] <= 2048:
        return {"output_budget_below_16"}
    return {"context_above_2048"}


def _zero_unexpected_lifecycle_events(rows: list[dict]) -> bool:
    for row in rows:
        summary = row["exact_greedy_decode_burst_summary"]
        if (
            summary["failures"] != 0
            or summary["quarantines"] != 0
            or summary["pending_leases"] != 0
            or summary.get("quarantine_reason") is not None
            or summary["lease_local_delta_journal_rollbacks"] != 0
            or summary[
                "lease_local_delta_journal_one_phase_rollbacks"
            ] != 0
            or summary["fallback_counts"] != {}
            or (
                row["policy"] == "fixed_k8"
                and (
                    summary["k16_attempts"] != 0
                    or summary["k16_acceptances"] != 0
                    or summary["k8_fallbacks"] != 0
                    or summary["elastic_k16_fallback_counts"] != {}
                )
            )
            or set(summary["elastic_k16_fallback_counts"])
            - _expected_elastic_fallbacks(row)
            or set(summary["lease_local_delta_journal_fallback_counts"])
            - {"unsupported_burst_shape"}
            or set(
                summary[
                    "lease_local_delta_journal_one_phase_fallback_counts"
                ]
            )
            - {"unsupported_burst_shape"}
        ):
            return False
    return True


def _metric_summary(control_rows: list[dict], candidate_rows: list[dict]) -> dict:
    control_tpot = [
        sample
        for row in control_rows
        for sample in row["amortized_tpot_samples_ns"]
    ]
    candidate_tpot = [
        sample
        for row in candidate_rows
        for sample in row["amortized_tpot_samples_ns"]
    ]
    control_median = statistics.median(control_tpot)
    candidate_median = statistics.median(candidate_tpot)
    control_p95 = _nearest_rank(control_tpot, 0.95)
    candidate_p95 = _nearest_rank(candidate_tpot, 0.95)
    control_p99 = _nearest_rank(control_tpot, 0.99)
    candidate_p99 = _nearest_rank(candidate_tpot, 0.99)
    control_ttft = statistics.median(row["ttft_ns"] for row in control_rows)
    candidate_ttft = statistics.median(
        row["ttft_ns"] for row in candidate_rows
    )
    control_e2e = statistics.median(row["e2e_ns"] for row in control_rows)
    candidate_e2e = statistics.median(
        row["e2e_ns"] for row in candidate_rows
    )
    control_rate = statistics.median(
        row["output_tokens_per_second"] for row in control_rows
    )
    candidate_rate = statistics.median(
        row["output_tokens_per_second"] for row in candidate_rows
    )
    control_allocated = max(
        row["cuda_peak_allocated_bytes"] for row in control_rows
    )
    candidate_allocated = max(
        row["cuda_peak_allocated_bytes"] for row in candidate_rows
    )
    control_reserved = max(
        row["cuda_peak_reserved_bytes"] for row in control_rows
    )
    candidate_reserved = max(
        row["cuda_peak_reserved_bytes"] for row in candidate_rows
    )
    return {
        "sample_count_per_policy": len(control_tpot),
        "control_tpot_median_ns": control_median,
        "candidate_tpot_median_ns": candidate_median,
        "tpot_median_improvement_pct":
            _improvement_pct(control_median, candidate_median),
        "control_tpot_p95_ns": control_p95,
        "candidate_tpot_p95_ns": candidate_p95,
        "tpot_p95_improvement_pct":
            _improvement_pct(control_p95, candidate_p95),
        "control_tpot_p99_ns": control_p99,
        "candidate_tpot_p99_ns": candidate_p99,
        "tpot_p99_regression_pct":
            _relative_change_pct(control_p99, candidate_p99),
        "control_ttft_median_ns": control_ttft,
        "candidate_ttft_median_ns": candidate_ttft,
        "ttft_regression_pct":
            _relative_change_pct(control_ttft, candidate_ttft),
        "control_e2e_median_ns": control_e2e,
        "candidate_e2e_median_ns": candidate_e2e,
        "e2e_regression_pct":
            _relative_change_pct(control_e2e, candidate_e2e),
        "control_throughput_median": control_rate,
        "candidate_throughput_median": candidate_rate,
        "throughput_regression_pct":
            _relative_change_pct(candidate_rate, control_rate),
        "control_cuda_peak_allocated_bytes": control_allocated,
        "candidate_cuda_peak_allocated_bytes": candidate_allocated,
        "allocated_memory_regression_pct":
            _relative_change_pct(control_allocated, candidate_allocated),
        "control_cuda_peak_reserved_bytes": control_reserved,
        "candidate_cuda_peak_reserved_bytes": candidate_reserved,
        "reserved_memory_regression_pct":
            _relative_change_pct(control_reserved, candidate_reserved),
    }


def _classify(metrics: dict) -> str:
    if metrics["correctness_exact"] is not True:
        return NO_GO_CORRECTNESS
    if metrics["width_policy_exact"] is not True:
        return NO_GO_WIDTH_POLICY
    if (
        metrics["runtime_inventory_exact"] is not True
        or metrics["zero_unexpected_lifecycle_events"] is not True
    ):
        return NO_GO_RUNTIME_INVARIANT
    if (
        metrics["maximum_selected_k16_host_visible_gap_ns"]
        > MAXIMUM_K16_HOST_VISIBLE_GAP_NS
    ):
        return NO_GO_BURST_GAP
    eligible = metrics["eligible_aggregate"]
    if (
        eligible["tpot_median_improvement_pct"]
        < MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT
        or eligible["tpot_p95_improvement_pct"]
        < MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT
    ):
        return NO_GO_INSUFFICIENT_INCREMENTAL_BENEFIT
    for item in metrics["by_context"].values():
        if (
            item["tpot_median_improvement_pct"]
            < -MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT
            or item["tpot_p95_improvement_pct"]
            < -MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT
            or item["tpot_p99_regression_pct"]
            > MAXIMUM_LATENCY_REGRESSION_PCT
            or item["ttft_regression_pct"]
            > MAXIMUM_LATENCY_REGRESSION_PCT
            or item["e2e_regression_pct"]
            > MAXIMUM_LATENCY_REGRESSION_PCT
            or item["throughput_regression_pct"]
            > MAXIMUM_THROUGHPUT_REGRESSION_PCT
            or item["allocated_memory_regression_pct"]
            > MAXIMUM_MEMORY_REGRESSION_PCT
            or item["reserved_memory_regression_pct"]
            > MAXIMUM_MEMORY_REGRESSION_PCT
        ):
            return NO_GO_PROTECTED_REGRESSION
    return GO


def _reconstruct_summary(
    performance: list[dict],
    correctness_exact: bool,
) -> dict:
    by_context_policy = {
        (context, policy): [
            row for row in performance
            if row["context_length"] == context
            and row["policy"] == policy
        ]
        for context in CONTEXT_LENGTHS
        for policy in POLICIES
    }
    by_context = {
        str(context): _metric_summary(
            by_context_policy[(context, "fixed_k8")],
            by_context_policy[
                (context, "context_gated_elastic_k16")
            ],
        )
        for context in CONTEXT_LENGTHS
    }
    eligible_control = [
        row for row in performance
        if row["policy"] == "fixed_k8"
        and row["context_length"] <= 2048
    ]
    eligible_candidate = [
        row for row in performance
        if row["policy"] == "context_gated_elastic_k16"
        and row["context_length"] <= 2048
    ]
    all_control = [
        row for row in performance if row["policy"] == "fixed_k8"
    ]
    all_candidate = [
        row for row in performance
        if row["policy"] == "context_gated_elastic_k16"
    ]
    selected_gaps = [
        gap
        for row in eligible_candidate
        if _selected_k16(row["exact_greedy_decode_burst_summary"])
        for gap in row["host_visible_burst_gaps_ns"]
    ]
    attempts = sum(
        row["exact_greedy_decode_burst_summary"]["attempts"]
        for row in all_candidate
    )
    fallbacks = sum(
        row["exact_greedy_decode_burst_summary"]["k8_fallbacks"]
        for row in all_candidate
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": performance[0]["run_tag"],
        "source_commit": performance[0]["source_commit"],
        "evidence_complete": True,
        "evidence_error": None,
        "performance_row_count": len(performance),
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
        "correctness_exact": correctness_exact,
        "width_policy_exact": _width_policy_exact(performance),
        "runtime_inventory_exact": _runtime_inventory_exact(performance),
        "zero_unexpected_lifecycle_events":
            _zero_unexpected_lifecycle_events(performance),
        "eligible_aggregate":
            _metric_summary(eligible_control, eligible_candidate),
        "overall": _metric_summary(all_control, all_candidate),
        "by_context": by_context,
        "maximum_selected_k16_host_visible_gap_ns":
            max(selected_gaps, default=0),
        "p95_selected_k16_host_visible_gap_ns": (
            _nearest_rank(selected_gaps, 0.95)
            if selected_gaps
            else 0
        ),
        "shared_capture_duration_ns_by_policy": {
            policy: max(
                row["shared_capture_duration_ns"]
                for row in performance
                if row["policy"] == policy
            )
            for policy in POLICIES
        },
        "elastic_incremental_capture_duration_ns": 0,
        "elastic_incremental_retained_static_bytes": max(
            row["elastic_incremental_retained_static_bytes"]
            for row in eligible_candidate
        ),
        "elastic_incremental_allocated_bytes": max(
            row["elastic_incremental_allocated_bytes"]
            for row in eligible_candidate
        ),
        "elastic_incremental_reserved_bytes": max(
            row["elastic_incremental_reserved_bytes"]
            for row in eligible_candidate
        ),
        "candidate_k8_fallback_count": fallbacks,
        "candidate_attempt_count": attempts,
        "k8_fallback_rate": fallbacks / attempts if attempts else 0.0,
        "k16_width_health_quarantine_count": sum(
            row["exact_greedy_decode_burst_summary"]["quarantines"]
            for row in all_candidate
        ),
        "lifecycle_totals": {
            field: sum(
                row["exact_greedy_decode_burst_summary"][field]
                for row in all_candidate
            )
            for field in (
                "attempts",
                "acceptances",
                "commits",
                "committed_tokens",
                "target_model_forwards",
                "graph_replays",
                "intermediate_token_d2h_calls",
                "final_token_d2h_calls",
                "final_token_d2h_bytes",
                "failures",
                "quarantines",
                "lease_local_delta_journal_rollbacks",
                "lease_local_delta_journal_one_phase_rollbacks",
            )
        },
        "thresholds": {
            "minimum_eligible_median_tpot_improvement_pct":
                MINIMUM_ELIGIBLE_MEDIAN_TPOT_IMPROVEMENT_PCT,
            "minimum_eligible_p95_tpot_improvement_pct":
                MINIMUM_ELIGIBLE_P95_TPOT_IMPROVEMENT_PCT,
            "maximum_per_context_tpot_regression_pct":
                MAXIMUM_PER_CONTEXT_TPOT_REGRESSION_PCT,
            "maximum_latency_regression_pct":
                MAXIMUM_LATENCY_REGRESSION_PCT,
            "maximum_throughput_regression_pct":
                MAXIMUM_THROUGHPUT_REGRESSION_PCT,
            "maximum_memory_regression_pct":
                MAXIMUM_MEMORY_REGRESSION_PCT,
            "maximum_k16_host_visible_gap_ns":
                MAXIMUM_K16_HOST_VISIBLE_GAP_NS,
        },
    }
    metrics["classification"] = _classify(metrics)
    return metrics


def _verify_manifest(
    run_dir: Path,
    manifest: dict,
    correctness_rows: list[dict],
) -> None:
    sidecars = {row.get("logits_path") for row in correctness_rows}
    expected = PRIMARY_ARTIFACTS | sidecars
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or set(manifest.get("artifact_sha256", {})) != expected
        or manifest.get("source_patch_sha256")
        != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("terminal manifest inventory mismatch")
    for relative, expected_digest in manifest["artifact_sha256"].items():
        path = _safe_artifact_path(run_dir, relative)
        if sha256_file(path) != expected_digest:
            raise ValueError(f"manifest digest mismatch: {relative}")


def _verify_source_manifest(
    manifest: dict,
    *,
    source_root: Path,
    expected_files: tuple[str, ...],
    expected_schema: str,
) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != expected_schema
        or set(manifest.get("source_sha256", {})) != set(expected_files)
    ):
        raise ValueError("source manifest inventory mismatch")
    root = Path(source_root).resolve()
    for relative in expected_files:
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"source file is missing: {relative}")
        if sha256_file(path) != manifest["source_sha256"][relative]:
            raise ValueError(f"source hash mismatch: {relative}")


def _verify_workload(workload: dict) -> None:
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version") != PROFILE_WORKLOAD_SCHEMA_VERSION
        or not isinstance(workload.get("run_tag"), str)
        or not workload["run_tag"]
        or not isinstance(workload.get("model"), str)
        or not workload["model"]
        or workload.get("device") != "cuda:0"
        or workload.get("contexts") != list(CONTEXT_LENGTHS)
        or workload.get("policies") != list(POLICIES)
        or workload.get("repetitions") != TERMINAL_REPETITIONS
        or workload.get("warmup_repetitions") != WARMUP_REPETITIONS
        or workload.get("generated_tokens") != GENERATED_TOKENS
        or workload.get("performance_row_count") != PERFORMANCE_ROW_COUNT
        or workload.get("correctness_row_count") != CORRECTNESS_ROW_COUNT
        or workload.get("temperature") != 0.0
        or workload.get("ignore_eos") is not True
        or workload.get("tensor_parallel_size") != 1
        or workload.get("max_num_seqs") != 1
        or workload.get("completion_only") is not True
        or (
            isinstance(workload.get("gpu_memory_utilization"), bool)
            or not isinstance(
                workload.get("gpu_memory_utilization"),
                (int, float),
            )
            or not 0.0
            < float(workload["gpu_memory_utilization"])
            <= 1.0
        )
        or not isinstance(workload.get("environment"), dict)
        or not workload["environment"]
    ):
        raise ValueError("workload manifest mismatch")
    _validate_digest(
        workload.get("source_commit"),
        "workload source commit",
        (40, 64),
    )


def verify_artifact_directory(
    run_dir: Path,
    *,
    source_root: Path,
) -> dict:
    root = Path(run_dir)
    correctness_raw = read_jsonl(root / "correctness_rows.jsonl")
    manifest = read_json(root / "terminal_manifest.json")
    _verify_manifest(root, manifest, correctness_raw)
    workload = read_json(root / "workload_manifest.json")
    source = read_json(root / "source_manifest.json")
    terminal_source = read_json(root / "terminal_source_manifest.json")
    _verify_workload(workload)
    _verify_source_manifest(
        source,
        source_root=source_root,
        expected_files=PROFILE_SOURCE_FILES,
        expected_schema=PROFILE_SOURCE_SCHEMA_VERSION,
    )
    _verify_source_manifest(
        terminal_source,
        source_root=source_root,
        expected_files=SOURCE_FILES,
        expected_schema=SCHEMA_VERSION,
    )
    if (root / "source.patch").read_bytes() != b"":
        raise ValueError("source patch must be empty")
    identities = {
        workload.get("run_tag"),
        source.get("run_tag"),
        terminal_source.get("run_tag"),
        manifest.get("run_tag"),
    }
    commits = {
        workload.get("source_commit"),
        source.get("source_commit"),
        terminal_source.get("source_commit"),
        manifest.get("source_commit"),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    performance = _validate_performance(
        read_jsonl(root / "performance_rows.jsonl")
    )
    correctness, correctness_exact = _validate_correctness(
        correctness_raw,
        run_dir=root,
    )
    if (
        {row["run_tag"] for row in performance + correctness}
        != identities
        or {row["source_commit"] for row in performance + correctness}
        != commits
    ):
        raise ValueError("row source authority mismatch")
    pair_count = len({
        (row["repetition"], row["context_length"])
        for row in performance
    })
    output_exact = all(
        control["output_token_ids"] == candidate["output_token_ids"]
        and control["output_text_sha256"]
        == candidate["output_text_sha256"]
        for control, candidate in (
            (
                next(
                    row for row in performance
                    if row["repetition"] == repetition
                    and row["context_length"] == context
                    and row["policy"] == "fixed_k8"
                ),
                next(
                    row for row in performance
                    if row["repetition"] == repetition
                    and row["context_length"] == context
                    and row["policy"]
                    == "context_gated_elastic_k16"
                ),
            )
            for repetition in range(TERMINAL_REPETITIONS)
            for context in CONTEXT_LENGTHS
        )
    )
    expected_profile_summary = {
        "schema_version": PROFILE_SUMMARY_SCHEMA_VERSION,
        "run_tag": next(iter(identities)),
        "source_commit": next(iter(commits)),
        "row_count": PERFORMANCE_ROW_COUNT,
        "comparison_set_count": pair_count,
        "all_outputs_exact": output_exact,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
    }
    if read_json(root / "profile_summary.json") != expected_profile_summary:
        raise ValueError("profile summary mismatch")
    reconstructed = _reconstruct_summary(
        performance,
        correctness_exact and output_exact,
    )
    if read_json(root / "terminal_summary.json") != reconstructed:
        raise ValueError("terminal summary mismatch")
    expected_gate = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": next(iter(identities)),
        "source_commit": next(iter(commits)),
        "classification": reconstructed["classification"],
    }
    if read_json(root / "terminal_gate.json") != expected_gate:
        raise ValueError("terminal gate mismatch")
    expected_receipt = {
        **expected_gate,
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
    }
    if read_json(root / "producer_receipt.json") != expected_receipt:
        raise ValueError("producer receipt mismatch")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "run_tag": next(iter(identities)),
        "source_commit": next(iter(commits)),
        "classification": reconstructed["classification"],
        "performance_row_count": PERFORMANCE_ROW_COUNT,
        "correctness_row_count": CORRECTNESS_ROW_COUNT,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipt = verify_artifact_directory(
        args.run_dir,
        source_root=args.source_root,
    )
    if args.output is None:
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    else:
        write_json(args.output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
