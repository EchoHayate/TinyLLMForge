#!/usr/bin/env python3
"""Independent verifier for split-phase K8 exact-burst evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


CASE_SCHEMA = "exact-burst-split-phase.case.v1"
CORRECTNESS_SCHEMA = "exact-burst-split-phase.correctness.v1"
SUMMARY_SCHEMA = "exact-burst-split-phase.summary.v1"
SOURCE_SCHEMA = "exact-burst-split-phase.source.v1"
WORKLOAD_SCHEMA = "exact-burst-split-phase.workload.v1"
COMPARISON_SCHEMA = "exact-burst-split-phase.comparison.v1"
GATE_SCHEMA = "exact-burst-split-phase.gate.v1"
MANIFEST_SCHEMA = "exact-burst-split-phase.manifest.v1"
VERIFICATION_SCHEMA = (
    "exact-burst-split-phase.independent-verification.v1"
)
TRACE_IDENTITY = (
    "gate-only-exact-burst-split-phase-correctness-v1"
)
CONTEXTS = (
    ("short", 256, 128),
    ("medium", 2048, 128),
    ("long", 8192, 128),
)
POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
    "decode_burst_k8_split_phase",
)
REFERENCE_POLICIES = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
)
CANDIDATE = "decode_burst_k8_split_phase"
POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
POLICY_CONFIGS = {
    "host_greedy": {
        "enabled": False,
        "split": False,
        "width": 1,
        "selectable": False,
        "entrypoint": "ordinary",
    },
    "decode_burst_k4": {
        "enabled": True,
        "split": False,
        "width": 4,
        "selectable": False,
        "entrypoint": "production",
    },
    "decode_burst_k8": {
        "enabled": True,
        "split": False,
        "width": 8,
        "selectable": False,
        "entrypoint": "production",
    },
    CANDIDATE: {
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
    "tools/test_profile_exact_burst_split_phase.py",
    "tools/exact_burst_split_phase_gate.py",
    "tools/test_exact_burst_split_phase_gate.py",
    "tools/exact_burst_split_phase_verify.py",
    "tools/test_exact_burst_split_phase_verify.py",
    "tools/run_exact_burst_split_phase_remote.py",
    "tools/test_run_exact_burst_split_phase_remote.py",
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
TPOT_REGRESSION_LIMIT = 0.02
THROUGHPUT_REGRESSION_LIMIT = 0.02
LATENCY_REGRESSION_LIMIT = 0.03
MEMORY_REGRESSION_LIMIT = 0.03
MAXIMUM_GAP_RATIO_LIMIT = 0.60
MEDIAN_GAP_REGRESSION_LIMIT = 0.03
BUCKET_TPOT_REGRESSION_LIMIT = 0.03
METRIC_TOLERANCE = 1e-9

BASE_COUNTER_FIELDS = (
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
)
SPLIT_COUNTER_FIELDS = (
    "prefix_commits",
    "suffix_commits",
    "prefix_committed_tokens",
    "suffix_committed_tokens",
    "prefix_publication_tickets",
    "suffix_publication_tickets",
    "prefix_token_d2h_calls",
    "suffix_token_d2h_calls",
    "prefix_token_d2h_bytes",
    "suffix_token_d2h_bytes",
    "prefix_phase_waits",
    "suffix_phase_waits",
    "suffix_drains",
)
INVENTORY_FIELDS = (
    "parent_lease_count",
    "prefix_row_count",
    "suffix_row_count",
    "prefix_ticket_count",
    "suffix_ticket_count",
    "replay_count",
    "prefix_d2h_calls",
    "suffix_d2h_calls",
    "prefix_d2h_bytes",
    "suffix_d2h_bytes",
    "prefix_pending_suffix_count",
    "suffix_cleared_count",
    "unexpected_scheduler_calls",
)
CAPTURE_FIELDS = (
    "capture_duration_ns",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "capture_retained_static_bytes",
    "reserved_scratch_blocks",
)


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


def _valid_digest(value, name: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _non_negative_number(value, name: str) -> float:
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


def _case_shape(bucket: str) -> tuple[int, int]:
    try:
        return {
            name: (prompt, generated)
            for name, prompt, generated in CONTEXTS
        }[bucket]
    except KeyError as error:
        raise ValueError("context bucket is invalid") from error


def _validate_receipt(receipt, *, correctness_trace: bool) -> None:
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


def _validate_summary(
    summary,
    *,
    policy: str,
    correctness_trace: bool,
) -> dict:
    required = (
        set(BASE_COUNTER_FIELDS)
        | set(SPLIT_COUNTER_FIELDS)
        | {
            "requested_width_histogram",
            "authorized_width_histogram",
            "fallback_counts",
            "split_phase_failure_counts",
            "quarantine_reason",
            "capture_receipts",
        }
    )
    if not isinstance(summary, dict) or required - set(summary):
        raise ValueError("exact burst summary fields are missing")
    for field in (*BASE_COUNTER_FIELDS, *SPLIT_COUNTER_FIELDS):
        _non_negative_integer(summary[field], field)
    for field in (
        "requested_width_histogram",
        "authorized_width_histogram",
        "fallback_counts",
        "split_phase_failure_counts",
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
        _validate_receipt(
            receipt,
            correctness_trace=correctness_trace,
        )
    enabled = POLICY_CONFIGS[policy]["enabled"]
    if enabled and len(receipts) != 1:
        raise ValueError(
            "enabled policy requires exactly one capture receipt"
        )
    if not enabled and (
        receipts
        or any(summary[field] for field in BASE_COUNTER_FIELDS)
        or any(summary[field] for field in SPLIT_COUNTER_FIELDS)
        or summary["split_phase_failure_counts"]
    ):
        raise ValueError("host policy reported burst activity")
    split = POLICY_CONFIGS[policy]["split"]
    split_activity = (
        any(summary[field] for field in SPLIT_COUNTER_FIELDS)
        or bool(summary["split_phase_failure_counts"])
    )
    if not split and split_activity:
        raise ValueError(
            "non-split policy reported split phase activity"
        )
    if split:
        commits = summary["commits"]
        tail = 127 - summary["committed_tokens"]
        accepted_tail_leases = max(0, tail - 1)
        expected_fallback_counts = {}
        if accepted_tail_leases:
            expected_fallback_counts["split_phase_requires_k8"] = (
                accepted_tail_leases
            )
        if tail:
            expected_fallback_counts["insufficient_output_budget"] = 1
        expected_authorized_widths = {
            str(width): 1
            for width in range(2, tail + 1)
        }
        expected_authorized_widths["8"] = commits
        exact = all((
            commits == 15,
            summary["attempts"] == commits + tail,
            summary["acceptances"]
            == commits + accepted_tail_leases,
            summary["target_model_forwards"] == commits * 8,
            summary["graph_replays"] == commits * 8,
            summary["committed_tokens"] == commits * 8,
            summary["intermediate_token_d2h_calls"] == 0,
            summary["final_token_d2h_calls"] == 0,
            summary["final_token_d2h_bytes"] == 0,
            summary["output_budget_clipped"]
            == accepted_tail_leases,
            summary["block_boundary_clipped"] == 0,
            summary["requested_width_histogram"]
            == {"8": commits + accepted_tail_leases},
            summary["authorized_width_histogram"]
            == expected_authorized_widths,
            summary["fallback_counts"]
            == expected_fallback_counts,
            not summary["split_phase_failure_counts"],
            summary["failures"] == 0,
            summary["quarantines"] == 0,
            summary["pending_leases"] == 0,
            summary["quarantine_reason"] is None,
            summary["sampled_logit_d2h_calls"]
            == (2 if correctness_trace else 0),
        ))
        for field in (
            "prefix_commits",
            "suffix_commits",
            "prefix_publication_tickets",
            "suffix_publication_tickets",
            "prefix_token_d2h_calls",
            "suffix_token_d2h_calls",
            "prefix_phase_waits",
            "suffix_phase_waits",
            "suffix_drains",
        ):
            exact &= summary[field] == commits
        for field in (
            "prefix_committed_tokens",
            "suffix_committed_tokens",
        ):
            exact &= summary[field] == commits * 4
        for field in (
            "prefix_token_d2h_bytes",
            "suffix_token_d2h_bytes",
        ):
            exact &= summary[field] == commits * 32
        if not exact:
            raise ValueError("split phase summary inventory mismatch")
    elif enabled:
        expected_sampled = 3 if correctness_trace else 0
        if (
            summary["sampled_logit_d2h_calls"] != expected_sampled
            or summary["failures"]
            or summary["quarantines"]
            or summary["pending_leases"]
            or summary["quarantine_reason"] is not None
        ):
            raise ValueError("ordinary burst lifecycle mismatch")
    return summary


def _validate_inventory(value, *, split: bool) -> dict:
    required = set(INVENTORY_FIELDS) | {"host_visible_gaps_ns"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("split phase inventory fields mismatch")
    for field in INVENTORY_FIELDS:
        _non_negative_integer(value[field], field)
    gaps = value["host_visible_gaps_ns"]
    if not isinstance(gaps, list):
        raise ValueError("split phase gap inventory is invalid")
    for gap in gaps:
        _non_negative_integer(gap, "split phase gap")
    if not split:
        if any(value[field] for field in INVENTORY_FIELDS) or gaps:
            raise ValueError(
                "non-split policy reported split phase activity"
            )
        return value
    expected = {
        "parent_lease_count": 15,
        "prefix_row_count": 15,
        "suffix_row_count": 15,
        "prefix_ticket_count": 15,
        "suffix_ticket_count": 15,
        "replay_count": 120,
        "prefix_d2h_calls": 15,
        "suffix_d2h_calls": 15,
        "prefix_d2h_bytes": 480,
        "suffix_d2h_bytes": 480,
        "prefix_pending_suffix_count": 15,
        "suffix_cleared_count": 15,
        "unexpected_scheduler_calls": 0,
    }
    if any(value[field] != expected[field] for field in expected):
        raise ValueError("split phase observation inventory mismatch")
    if len(gaps) != 30:
        raise ValueError("split phase host-visible gap inventory mismatch")
    return value


def _validate_performance_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != 60:
        raise ValueError(
            f"expected exactly 60 measured rows, got {len(rows)}"
        )
    identities = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != CASE_SCHEMA
        ):
            raise ValueError("case row schema mismatch")
        identity = (
            row.get("context_bucket"),
            row.get("repetition"),
            row.get("policy"),
        )
        bucket, repetition, policy = identity
        if (
            bucket not in {item[0] for item in CONTEXTS}
            or policy not in POLICIES
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or repetition not in range(5)
            or identity in identities
        ):
            raise ValueError("case identity mismatch")
        identities.add(identity)
        config = POLICY_CONFIGS[policy]
        if (
            row.get("selectable") is not config["selectable"]
            or row.get("burst_width") != config["width"]
            or (
                row.get("prompt_tokens"),
                row.get("generated_tokens"),
            ) != _case_shape(bucket)
            or row.get("correctness_trace") is not False
        ):
            raise ValueError("case metadata mismatch")
        _valid_digest(
            row.get("source_commit"),
            "source commit",
            lengths=(40, 64),
        )
        if not isinstance(row.get("run_tag"), str) or not row["run_tag"]:
            raise ValueError("run tag is invalid")
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != 128
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in output_ids
            )
        ):
            raise ValueError("output token inventory mismatch")
        _valid_digest(row.get("output_text_sha256"), "output text")
        samples = row.get("amortized_tpot_samples_ns")
        if not isinstance(samples, list) or len(samples) != 127:
            raise ValueError("amortized TPOT inventory mismatch")
        for value in samples:
            _non_negative_number(value, "amortized TPOT")
        expected_stats = {
            "amortized_tpot_median_ns": statistics.median(samples),
            "amortized_tpot_p95_ns": _nearest_rank(samples, 0.95),
            "amortized_tpot_p99_ns": _nearest_rank(samples, 0.99),
        }
        if any(row.get(key) != value for key, value in expected_stats.items()):
            raise ValueError("amortized TPOT statistic mismatch")
        for field in (
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "maximum_host_visible_burst_gap_ns",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            *CAPTURE_FIELDS,
        ):
            _non_negative_number(row.get(field), field)
        summary = _validate_summary(
            row.get("exact_greedy_decode_burst_summary"),
            policy=policy,
            correctness_trace=False,
        )
        inventory = _validate_inventory(
            row.get("split_phase_inventory"),
            split=config["split"],
        )
        for field in (
            "decode_host_ns",
            "decode_cuda_ns",
            "host_visible_burst_gaps_ns",
        ):
            values = row.get(field)
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list")
            for value in values:
                _non_negative_number(value, field)
        expected_profile_rows = (
            (
                summary["attempts"]
                + (127 - summary["committed_tokens"])
                - sum(
                    summary["fallback_counts"].get(reason, 0)
                    for reason in config.get(
                        "scheduler_only_fallback_reasons",
                        (),
                    )
                )
            )
            if config["split"]
            else (
                summary["commits"]
                if config["enabled"]
                else 127
            )
        )
        if (
            len(row["decode_host_ns"]) != expected_profile_rows
            or len(row["decode_cuda_ns"]) != expected_profile_rows
        ):
            raise ValueError("decode profile inventory mismatch")
        gaps = row["host_visible_burst_gaps_ns"]
        if row["maximum_host_visible_burst_gap_ns"] != max(
            gaps, default=0
        ):
            raise ValueError("maximum host-visible gap mismatch")
        if (
            summary["maximum_host_visible_gap_ns"]
            != row["maximum_host_visible_burst_gap_ns"]
        ):
            raise ValueError("summary host-visible gap mismatch")
        if config["split"] and inventory["host_visible_gaps_ns"] != gaps:
            raise ValueError("split phase gap inventory mismatch")
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
        if any(row[field] != value for field, value in expected_costs.items()):
            raise ValueError("capture cost mismatch")
    expected = {
        (bucket, repetition, policy)
        for bucket, _prompt, _generated in CONTEXTS
        for repetition in range(5)
        for policy in POLICIES
    }
    if identities != expected:
        raise ValueError("case row inventory mismatch")
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
    count = row.get("logits_element_count")
    byte_length = row.get("logits_byte_length")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length != count * 4
        or len(payload) != byte_length
    ):
        raise ValueError("sidecar byte length mismatch")
    if _sha256_file(path) != row.get("logits_sha256"):
        raise ValueError("sidecar digest mismatch")
    values = struct.unpack(f"<{count}f", payload)
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


def _uses_burst_trace(policy: str, point: str) -> bool:
    return (
        POLICY_CONFIGS[policy]["enabled"]
        and point != "prefill-final"
        and point
        not in POLICY_CONFIGS[policy].get(
            "ordinary_tail_sampling_points", ()
        )
    )


def _validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
) -> tuple[list[dict], dict]:
    if len(rows) != 48:
        raise ValueError(
            f"expected exactly 48 correctness rows, got {len(rows)}"
        )
    identities = set()
    values = {}
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
        bucket, policy, point = identity
        if (
            bucket not in {item[0] for item in CONTEXTS}
            or policy not in POLICIES
            or point not in POINTS
            or identity in identities
        ):
            raise ValueError("correctness identity mismatch")
        identities.add(identity)
        if (
            (
                row.get("prompt_tokens"),
                row.get("generated_tokens"),
            ) != _case_shape(bucket)
            or row.get("correctness_trace") is not True
            or row.get("trace_identity") != TRACE_IDENTITY
        ):
            raise ValueError("correctness metadata mismatch")
        _valid_digest(
            row.get("source_commit"),
            "correctness source commit",
            lengths=(40, 64),
        )
        if not isinstance(row.get("run_tag"), str) or not row["run_tag"]:
            raise ValueError("correctness run tag is invalid")
        output_ids = row.get("output_token_ids")
        if not isinstance(output_ids, list) or len(output_ids) != 128:
            raise ValueError("correctness output inventory mismatch")
        _valid_digest(
            row.get("output_text_sha256"),
            "correctness output text",
        )
        burst_sample = _uses_burst_trace(policy, point)
        if burst_sample:
            _valid_digest(
                row.get("trace_graph_identity_sha256"),
                "correctness graph identity",
            )
            expected_ordinal = (
                _sampling_index(point) - 1
            ) % POLICY_CONFIGS[policy]["width"]
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
        sidecar_values = _safe_sidecar(run_dir, row)
        if len(sidecar_values) != shape[1]:
            raise ValueError("correctness logits element mismatch")
        values[identity] = sidecar_values
        summary = _validate_summary(
            row.get("exact_greedy_decode_burst_summary"),
            policy=policy,
            correctness_trace=POLICY_CONFIGS[policy]["enabled"],
        )
        if burst_sample and (
            summary["capture_receipts"][0][
                "graph_identity_sha256"
            ]
            != row["trace_graph_identity_sha256"]
        ):
            raise ValueError("trace graph identity mismatch")
    expected = {
        (bucket, policy, point)
        for bucket, _prompt, _generated in CONTEXTS
        for policy in POLICIES
        for point in POINTS
    }
    if identities != expected:
        raise ValueError("correctness row inventory mismatch")
    return rows, values


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
    exact = True
    for repetition in range(5):
        for bucket, _prompt, _generated in CONTEXTS:
            group = [
                by_identity[(repetition, bucket, policy)]
                for policy in POLICIES
            ]
            exact &= (
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
        "row_count": 60,
        "comparison_set_count": 15,
        "all_outputs_exact": exact,
        "comparisons": comparisons,
        "correctness_row_count": 48,
    }


def _metric_summary(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    baseline_policy: str,
) -> dict:
    def identity(row: dict) -> tuple[str, int]:
        return row["context_bucket"], row["repetition"]

    baseline_by_identity = {
        identity(row): row for row in baseline_rows
    }
    candidate_by_identity = {
        identity(row): row for row in candidate_rows
    }
    if (
        len(baseline_by_identity) != len(baseline_rows)
        or len(candidate_by_identity) != len(candidate_rows)
        or set(baseline_by_identity) != set(candidate_by_identity)
    ):
        raise ValueError("paired request inventory mismatch")
    pairs = [
        (
            baseline_by_identity[key],
            candidate_by_identity[key],
        )
        for key in sorted(baseline_by_identity)
    ]

    def row_tpot(row: dict, percentile: float) -> float:
        samples = row["amortized_tpot_samples_ns"]
        if percentile == 0.5:
            return float(statistics.median(samples))
        return _nearest_rank(samples, percentile)

    def paired_regression(field) -> float:
        return statistics.median(
            _regression(field(left), field(right))
            for left, right in pairs
        )

    def paired_throughput_regression(field) -> float:
        return statistics.median(
            _throughput_regression(field(left), field(right))
            for left, right in pairs
        )

    baseline_median = statistics.median(
        row_tpot(row, 0.5) for row in baseline_rows
    )
    candidate_median = statistics.median(
        row_tpot(row, 0.5) for row in candidate_rows
    )
    baseline_p95 = statistics.median(
        row_tpot(row, 0.95) for row in baseline_rows
    )
    candidate_p95 = statistics.median(
        row_tpot(row, 0.95) for row in candidate_rows
    )
    baseline_p99 = statistics.median(
        row_tpot(row, 0.99) for row in baseline_rows
    )
    candidate_p99 = statistics.median(
        row_tpot(row, 0.99) for row in candidate_rows
    )
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
    baseline_throughput = statistics.median(
        float(row["output_tokens_per_second"])
        for row in baseline_rows
    )
    candidate_throughput = statistics.median(
        float(row["output_tokens_per_second"])
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
        "baseline_tpot_median_ns": baseline_median,
        "candidate_tpot_median_ns": candidate_median,
        "tpot_median_regression_fraction":
            paired_regression(
                lambda row: row_tpot(row, 0.5)
            ),
        "baseline_tpot_p95_ns": baseline_p95,
        "candidate_tpot_p95_ns": candidate_p95,
        "tpot_p95_regression_fraction":
            paired_regression(
                lambda row: row_tpot(row, 0.95)
            ),
        "baseline_tpot_p99_ns": baseline_p99,
        "candidate_tpot_p99_ns": candidate_p99,
        "tpot_p99_regression_fraction":
            paired_regression(
                lambda row: row_tpot(row, 0.99)
            ),
        "baseline_ttft_median_ns": baseline_ttft,
        "candidate_ttft_median_ns": candidate_ttft,
        "ttft_regression_fraction":
            paired_regression(
                lambda row: float(row["ttft_ns"])
            ),
        "baseline_e2e_median_ns": baseline_e2e,
        "candidate_e2e_median_ns": candidate_e2e,
        "e2e_regression_fraction":
            paired_regression(
                lambda row: float(row["e2e_ns"])
            ),
        "baseline_output_tokens_per_second_median":
            baseline_throughput,
        "candidate_output_tokens_per_second_median":
            candidate_throughput,
        "throughput_regression_fraction":
            paired_throughput_regression(
                lambda row: float(
                    row["output_tokens_per_second"]
                )
            ),
        "baseline_cuda_peak_reserved_bytes": baseline_reserved,
        "candidate_cuda_peak_reserved_bytes": candidate_reserved,
        "cuda_reserved_delta_bytes":
            candidate_reserved - baseline_reserved,
        "cuda_reserved_regression_fraction":
            _regression(baseline_reserved, candidate_reserved),
    }


def _correctness(rows: list[dict], values: dict) -> tuple[dict, bool]:
    by_identity = {
        (
            row["context_bucket"],
            row["policy"],
            row["sampling_point"],
        ): row
        for row in rows
    }
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
                left = by_identity[(bucket, baseline, point)]
                right = by_identity[(bucket, CANDIDATE, point)]
                left_values = values[(bucket, baseline, point)]
                right_values = values[(bucket, CANDIDATE, point)]
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
    return ({
        "pair_count": len(pairs),
        "max_abs": global_max,
        "mean_abs": (
            total_abs / total_count if total_count else 0.0
        ),
        "argmax_equal": all_argmax,
        "output_ids_exact": all_ids,
        "output_text_exact": all_text,
        "pairs": pairs,
    }, all_passed)


def _lifecycle(rows: list[dict]) -> dict:
    selected = [row for row in rows if row["policy"] == CANDIDATE]
    result = {
        "request_count": len(selected),
        "parent_leases": 0,
        "prefix_commits": 0,
        "suffix_commits": 0,
        "prefix_tickets": 0,
        "suffix_tickets": 0,
        "graph_replays": 0,
        "prefix_d2h_calls": 0,
        "suffix_d2h_calls": 0,
        "prefix_phase_waits": 0,
        "suffix_phase_waits": 0,
        "suffix_drains": 0,
        "ordinary_tail_tokens": 0,
        "unexpected_scheduler_calls": 0,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
    }
    for row in selected:
        summary = row["exact_greedy_decode_burst_summary"]
        inventory = row["split_phase_inventory"]
        result["parent_leases"] += inventory["parent_lease_count"]
        result["prefix_commits"] += summary["prefix_commits"]
        result["suffix_commits"] += summary["suffix_commits"]
        result["prefix_tickets"] += summary[
            "prefix_publication_tickets"
        ]
        result["suffix_tickets"] += summary[
            "suffix_publication_tickets"
        ]
        result["graph_replays"] += summary["graph_replays"]
        result["prefix_d2h_calls"] += summary[
            "prefix_token_d2h_calls"
        ]
        result["suffix_d2h_calls"] += summary[
            "suffix_token_d2h_calls"
        ]
        result["prefix_phase_waits"] += summary["prefix_phase_waits"]
        result["suffix_phase_waits"] += summary["suffix_phase_waits"]
        result["suffix_drains"] += summary["suffix_drains"]
        result["ordinary_tail_tokens"] += (
            row["generated_tokens"] - 1 - summary["committed_tokens"]
        )
        result["unexpected_scheduler_calls"] += inventory[
            "unexpected_scheduler_calls"
        ]
        result["failures"] += summary["failures"]
        result["quarantines"] += summary["quarantines"]
        result["pending_leases"] += summary["pending_leases"]
    result["complete"] = all((
        result["request_count"] == 15,
        result["parent_leases"] == 225,
        result["prefix_commits"] == 225,
        result["suffix_commits"] == 225,
        result["prefix_tickets"] == 225,
        result["suffix_tickets"] == 225,
        result["graph_replays"] == 1_800,
        result["prefix_d2h_calls"] == 225,
        result["suffix_d2h_calls"] == 225,
        result["prefix_phase_waits"] == 225,
        result["suffix_phase_waits"] == 225,
        result["suffix_drains"] == 225,
        result["ordinary_tail_tokens"] == 105,
        result["unexpected_scheduler_calls"] == 0,
        result["failures"] == 0,
        result["quarantines"] == 0,
        result["pending_leases"] == 0,
    ))
    return result


def _evaluation(rows: list[dict], *, correctness_passed: bool) -> dict:
    candidate = [row for row in rows if row["policy"] == CANDIDATE]
    k8 = [row for row in rows if row["policy"] == "decode_burst_k8"]
    k4 = [row for row in rows if row["policy"] == "decode_burst_k4"]
    aggregate = _metric_summary(
        k8,
        candidate,
        baseline_policy="decode_burst_k8",
    )
    by_bucket = {}
    bucket_regressions = []
    latency_regressions = []
    for bucket, _prompt, _generated in CONTEXTS:
        metrics = _metric_summary(
            [
                row for row in k8
                if row["context_bucket"] == bucket
            ],
            [
                row for row in candidate
                if row["context_bucket"] == bucket
            ],
            baseline_policy="decode_burst_k8",
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
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:ttft")
        if (
            metrics["e2e_regression_fraction"]
            > LATENCY_REGRESSION_LIMIT
        ):
            latency_regressions.append(f"{bucket}:e2e")
    candidate_max = max(
        row["maximum_host_visible_burst_gap_ns"]
        for row in candidate
    )
    k8_max = max(
        row["maximum_host_visible_burst_gap_ns"] for row in k8
    )
    candidate_median = statistics.median(
        row["maximum_host_visible_burst_gap_ns"]
        for row in candidate
    )
    k4_median = statistics.median(
        row["maximum_host_visible_burst_gap_ns"] for row in k4
    )
    gap_ratio = candidate_max / k8_max if k8_max else math.inf
    median_gap_regression = _regression(k4_median, candidate_median)
    lifecycle = _lifecycle(rows)
    performance_passed = all((
        aggregate["tpot_median_regression_fraction"]
        <= TPOT_REGRESSION_LIMIT,
        aggregate["throughput_regression_fraction"]
        <= THROUGHPUT_REGRESSION_LIMIT,
        aggregate["cuda_reserved_regression_fraction"]
        <= MEMORY_REGRESSION_LIMIT,
        gap_ratio <= MAXIMUM_GAP_RATIO_LIMIT,
        median_gap_regression <= MEDIAN_GAP_REGRESSION_LIMIT,
        not bucket_regressions,
        not latency_regressions,
    ))
    return {
        "policy": CANDIDATE,
        "correctness_passed": correctness_passed,
        "lifecycle": lifecycle,
        "performance_passed": performance_passed,
        "bucket_regressions": bucket_regressions,
        "latency_regressions": latency_regressions,
        "memory_regression": (
            aggregate["cuda_reserved_regression_fraction"]
            > MEMORY_REGRESSION_LIMIT
        ),
        "candidate_maximum_host_visible_gap_ns": candidate_max,
        "k8_maximum_host_visible_gap_ns": k8_max,
        "maximum_gap_ratio_vs_k8": gap_ratio,
        "candidate_median_max_gap_ns": candidate_median,
        "k4_median_max_gap_ns": k4_median,
        "median_max_gap_regression_vs_k4": median_gap_regression,
        "aggregate": {"k8_vs_split": aggregate},
        "by_bucket": by_bucket,
    }


def _reconstruct_comparison(
    rows: list[dict],
    correctness_rows: list[dict],
    values: dict,
) -> dict:
    by_identity = {
        (
            row["context_bucket"],
            row["repetition"],
            row["policy"],
        ): row
        for row in rows
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
    correctness, logits_passed = _correctness(
        correctness_rows, values
    )
    run_tags = {
        *(row.get("run_tag") for row in rows),
        *(row.get("run_tag") for row in correctness_rows),
    }
    source_commits = {
        *(row.get("source_commit") for row in rows),
        *(row.get("source_commit") for row in correctness_rows),
    }
    evidence_complete = (
        len(run_tags) == 1 and len(source_commits) == 1
    )
    evaluation = _evaluation(
        rows,
        correctness_passed=output_exact and logits_passed,
    )
    if not evaluation["correctness_passed"]:
        classification = (
            "NO_GO_EXACT_BURST_SPLIT_PHASE_CORRECTNESS"
        )
    elif not evidence_complete or not evaluation["lifecycle"]["complete"]:
        classification = (
            "INCOMPLETE_EXACT_BURST_SPLIT_PHASE_EVIDENCE"
        )
    elif not evaluation["performance_passed"]:
        classification = (
            "NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE"
        )
    else:
        classification = "GO_EXACT_BURST_SPLIT_PHASE"
    return {
        "schema_version": COMPARISON_SCHEMA,
        "run_tag": rows[0]["run_tag"],
        "source_commit": rows[0]["source_commit"],
        "classification": classification,
        "selected_policy": CANDIDATE,
        "selected_burst_width": 8,
        "all_performance_outputs_exact": output_exact,
        "evidence_complete": evidence_complete,
        "thresholds": {
            "logit_max_abs_limit": LOGIT_MAX_LIMIT,
            "logit_mean_abs_limit": LOGIT_MEAN_LIMIT,
            "aggregate_tpot_regression_limit":
                TPOT_REGRESSION_LIMIT,
            "throughput_regression_limit":
                THROUGHPUT_REGRESSION_LIMIT,
            "latency_regression_limit": LATENCY_REGRESSION_LIMIT,
            "reserved_memory_regression_limit":
                MEMORY_REGRESSION_LIMIT,
            "maximum_gap_ratio_limit":
                MAXIMUM_GAP_RATIO_LIMIT,
            "median_gap_regression_limit":
                MEDIAN_GAP_REGRESSION_LIMIT,
            "bucket_tpot_regression_limit":
                BUCKET_TPOT_REGRESSION_LIMIT,
        },
        "correctness": correctness,
        "candidate_evaluation": evaluation,
    }


def _policy_order(
    repetition: int,
    context_index: int,
) -> tuple[str, ...]:
    rotation = repetition % len(POLICIES)
    order = POLICIES[rotation:] + POLICIES[:rotation]
    return tuple(reversed(order)) if context_index % 2 else order


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
            environment.get("cuda_runtime_version"), str
        )
        or not environment["cuda_runtime_version"]
        or not isinstance(
            environment.get("cuda_device_name"), str
        )
        or not environment["cuda_device_name"]
    ):
        raise ValueError("workload manifest mismatch: environment")


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
    actual_sidecars = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / "logits").rglob("*.f32")
    }
    referenced_sidecars = {
        row["logits_path"] for row in correctness_rows
    }
    if actual_sidecars != referenced_sidecars:
        raise ValueError("sidecar inventory mismatch")
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
    source_patch_sha = _sha256_file(run_dir / "source.patch")
    if (
        manifest.get("source_patch_sha256") != source_patch_sha
        or (run_dir / "source.patch").read_bytes()
    ):
        raise ValueError("dirty source patch evidence mismatch")


def _compare_with_tolerance(
    expected,
    actual,
    *,
    path: str = "comparison",
) -> float:
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        disagreement = abs(float(expected) - float(actual))
        if disagreement > METRIC_TOLERANCE:
            raise ValueError(
                f"metric disagreement at {path}: {disagreement}"
            )
        return disagreement
    if type(expected) is not type(actual):
        raise ValueError(f"comparison drift at {path}")
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            raise ValueError(f"comparison drift at {path}")
        maximum = 0.0
        for key in sorted(expected):
            maximum = max(
                maximum,
                _compare_with_tolerance(
                    expected[key],
                    actual[key],
                    path=f"{path}.{key}",
                ),
            )
        return maximum
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError(f"comparison drift at {path}")
        maximum = 0.0
        for index, (left, right) in enumerate(zip(expected, actual)):
            maximum = max(
                maximum,
                _compare_with_tolerance(
                    left,
                    right,
                    path=f"{path}[{index}]",
                ),
            )
        return maximum
    if expected != actual:
        raise ValueError(f"comparison drift at {path}")
    return 0.0


def verify_bundle(
    run_dir: Path,
    *,
    repo_root: Path,
) -> dict:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)
    rows = _load_jsonl(run_dir / "case_rows.jsonl")
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
    validated_rows = _validate_performance_rows(rows)
    validated_correctness, values = _validate_correctness_rows(
        correctness_rows,
        run_dir=run_dir,
    )
    identities = {
        source.get("run_tag"),
        workload.get("run_tag"),
        manifest.get("run_tag"),
        *(row.get("run_tag") for row in validated_rows),
        *(row.get("run_tag") for row in validated_correctness),
    }
    commits = {
        source.get("source_commit"),
        workload.get("source_commit"),
        manifest.get("source_commit"),
        *(row.get("source_commit") for row in validated_rows),
        *(row.get("source_commit") for row in validated_correctness),
    }
    if len(identities) != 1 or len(commits) != 1:
        raise ValueError("source-bound identity mismatch")
    expected_summary = _summary_from_rows(validated_rows)
    if summary != expected_summary:
        raise ValueError("worker summary drift")
    reconstructed = _reconstruct_comparison(
        validated_rows,
        validated_correctness,
        values,
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
    if gate.get("classification") != reconstructed["classification"]:
        raise ValueError("classification drift")
    maximum_disagreement = _compare_with_tolerance(
        reconstructed,
        comparison,
    )
    comparison_sha = _sha256_file(run_dir / "comparison.json")
    expected_gate = {
        "schema_version": GATE_SCHEMA,
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "source_patch_sha256": _sha256_file(
            run_dir / "source.patch"
        ),
        "classification": reconstructed["classification"],
        "selected_policy": reconstructed["selected_policy"],
        "selected_burst_width":
            reconstructed["selected_burst_width"],
        "comparison_sha256": comparison_sha,
    }
    if gate != expected_gate:
        if gate.get("classification") != reconstructed["classification"]:
            raise ValueError("classification drift")
        raise ValueError("gate drift")
    result = {
        "schema_version": VERIFICATION_SCHEMA,
        "status": "PASS",
        "run_tag": reconstructed["run_tag"],
        "source_commit": reconstructed["source_commit"],
        "source_patch_sha256": expected_gate[
            "source_patch_sha256"
        ],
        "reconstructed_classification":
            reconstructed["classification"],
        "reconstructed_selected_policy":
            reconstructed["selected_policy"],
        "reconstructed_selected_burst_width":
            reconstructed["selected_burst_width"],
        "performance_row_count": len(validated_rows),
        "correctness_row_count": len(validated_correctness),
        "maximum_metric_disagreement": maximum_disagreement,
        "comparison_sha256": comparison_sha,
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
