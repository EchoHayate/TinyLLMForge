#!/usr/bin/env python3
"""Independent verifier for the octet-folded exact-burst ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import struct


VERIFICATION_SCHEMA_VERSION = (
    "exact-burst-octet-folded.verification.v1"
)
CASE_SCHEMA_VERSION = "exact-burst-octet-folded.case.v1"
CORRECTNESS_SCHEMA_VERSION = (
    "exact-burst-octet-folded.correctness.v1"
)
PROFILE_SUMMARY_SCHEMA_VERSION = (
    "exact-burst-octet-folded.summary.v1"
)
WORKLOAD_SCHEMA_VERSION = (
    "exact-burst-octet-folded.workload.v1"
)
SOURCE_SCHEMA_VERSION = "exact-burst-octet-folded.source.v1"
CEILING_SCHEMA_VERSION = "exact-burst-octet-folded.ceiling.v1"
GO_CEILING = "GO_CEILING"
NO_GO_CEILING = "NO_GO_CEILING"
POLICIES = ("one_token_graph", "octet_folded_graph")
CONTEXT_LENGTHS = (256, 2048, 8192)
SAMPLING_POINTS = (
    "prefill-final",
    "decode-first",
    "decode-middle",
    "decode-final",
)
GENERATED_TOKENS = 128
REPETITIONS = 5
WARMUP_REPETITIONS = 2
MINIMUM_MEDIAN_TPOT_IMPROVEMENT_PCT = 1.0
MINIMUM_P95_TPOT_IMPROVEMENT_PCT = 0.5
MAXIMUM_PROTECTED_REGRESSION_PCT = 2.0
MAXIMUM_CAPTURE_MEMORY_RATIO = 0.01
MAXIMUM_RETAINED_STATIC_DELTA_BYTES = 128 * 1024 * 1024
MAXIMUM_FOLDED_CAPTURE_DURATION_NS = 120_000_000_000
SOURCE_FILES = (
    "tinyvllm/config.py",
    "tinyvllm/engine/exact_greedy_decode_burst.py",
    "tinyvllm/engine/model_runner.py",
    "tinyvllm/engine/scheduler.py",
    "tinyvllm/engine/llm_engine.py",
    "tools/profile_exact_burst_octet_folded_graph.py",
    "tools/test_profile_exact_burst_octet_folded_graph.py",
    "tools/exact_burst_octet_folded_graph_ceiling.py",
    "tools/test_exact_burst_octet_folded_graph_ceiling.py",
)
PERFORMANCE_REQUIRED_FIELDS = {
    "schema_version",
    "run_tag",
    "source_commit",
    "source_patch_sha256",
    "policy",
    "repetition",
    "order_position",
    "context_length",
    "prompt_tokens",
    "generated_tokens",
    "temperature",
    "ignore_eos",
    "tensor_parallel_size",
    "max_num_seqs",
    "completion_only",
    "prompt_sha256",
    "output_token_ids",
    "output_text_sha256",
    "one_token_graph_identity_sha256",
    "folded_graph_identity_sha256",
    "logical_forwards",
    "logical_replays",
    "one_token_cuda_graph_launches",
    "folded_cuda_graph_launches",
    "token_d2h_calls",
    "token_d2h_bytes",
    "capture_duration_ns",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "capture_retained_static_bytes",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "ttft_ns",
    "e2e_ns",
    "tpot_samples_ns",
    "tpot_median_ns",
    "tpot_p95_ns",
    "tpot_p99_ns",
    "output_tokens_per_second",
    "host_visible_burst_gaps_ns",
    "maximum_host_visible_burst_gap_ns",
    "fallback_count",
    "rollback_count",
    "quarantine_reason",
}
CORRECTNESS_REQUIRED_FIELDS = {
    "schema_version",
    "run_tag",
    "source_commit",
    "source_patch_sha256",
    "policy",
    "context_length",
    "generated_tokens",
    "sampling_point",
    "prompt_sha256",
    "output_token_ids",
    "output_text_sha256",
    "argmax_token_id",
    "logits_path",
    "logits_shape",
    "logits_element_count",
    "logits_byte_length",
    "logits_sha256",
    "one_token_graph_identity_sha256",
    "folded_graph_identity_sha256",
    "logical_forwards",
    "logical_replays",
    "one_token_cuda_graph_launches",
    "folded_cuda_graph_launches",
    "token_d2h_calls",
    "token_d2h_bytes",
    "fallback_count",
    "rollback_count",
    "quarantine_reason",
    "correctness_trace",
}
WORKLOAD_REQUIRED_FIELDS = {
    "schema_version",
    "model",
    "device",
    "run_tag",
    "source_commit",
    "source_patch_sha256",
    "contexts",
    "policies",
    "generated_tokens",
    "repetitions",
    "warmup_repetitions",
    "performance_row_count",
    "correctness_row_count",
    "execution_order",
    "temperature",
    "ignore_eos",
    "tensor_parallel_size",
    "max_num_seqs",
    "completion_only",
    "gpu_memory_utilization",
    "environment",
}


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: Path):
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"required artifact is missing: {target.name}")
    return json.loads(
        target.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def read_jsonl(path: Path) -> list[dict]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"required artifact is missing: {target.name}")
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload) -> None:
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


def _digest(value, field: str, lengths=(64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _non_negative_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _finite_non_negative(value, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative")
    return float(value)


def _safe_file(run_dir: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("artifact path is invalid")
    root = Path(run_dir).resolve()
    target = (root / relative).resolve()
    if root not in target.parents or not target.is_file():
        raise ValueError(f"artifact is missing: {relative}")
    return target


def _policy_order(
    repetition: int,
    context_index: int,
) -> tuple[str, str]:
    return (
        tuple(reversed(POLICIES))
        if (repetition + context_index) % 2
        else POLICIES
    )


def _expected_performance_identities() -> set[tuple[int, int, str]]:
    return {
        (repetition, context, policy)
        for repetition in range(REPETITIONS)
        for context_index, context in enumerate(CONTEXT_LENGTHS)
        for policy in _policy_order(repetition, context_index)
    }


def _expected_correctness_identities() -> set[tuple[int, str, str]]:
    return {
        (context, policy, point)
        for context in CONTEXT_LENGTHS
        for policy in POLICIES
        for point in SAMPLING_POINTS
    }


def _validate_runtime_inventory(row: dict) -> None:
    logical = GENERATED_TOKENS - 1
    bursts = math.ceil(logical / 8)
    required = (
        "logical_forwards",
        "logical_replays",
        "one_token_cuda_graph_launches",
        "folded_cuda_graph_launches",
        "token_d2h_calls",
        "token_d2h_bytes",
        "fallback_count",
        "rollback_count",
    )
    for field in required:
        _non_negative_int(row.get(field), field)
    if (
        row["logical_forwards"] != logical
        or row["logical_replays"] != logical
        or row["token_d2h_calls"] != bursts
        or row["token_d2h_bytes"] != logical * 8
    ):
        raise ValueError("runtime inventory mismatch")
    if row["policy"] == "one_token_graph":
        physical_exact = (
            row["one_token_cuda_graph_launches"] == logical
            and row["folded_cuda_graph_launches"] == 0
            and row.get("folded_graph_identity_sha256") is None
        )
    else:
        physical_exact = (
            row["one_token_cuda_graph_launches"] == logical % 8
            and row["folded_cuda_graph_launches"] == logical // 8
            and row.get("folded_graph_identity_sha256") is not None
        )
    if not physical_exact:
        raise ValueError("runtime inventory physical launch mismatch")
    if (
        row["fallback_count"]
        or row["rollback_count"]
        or row.get("quarantine_reason") is not None
    ):
        raise ValueError("runtime inventory anomaly")


def _validate_performance_rows(
    rows: list[dict],
    *,
    run_tag: str,
    source_commit: str,
    patch_sha256: str,
) -> list[dict]:
    identities = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != PERFORMANCE_REQUIRED_FIELDS
        ):
            raise ValueError("performance row is invalid")
        if row.get("schema_version") != CASE_SCHEMA_VERSION:
            raise ValueError("performance row schema mismatch")
        policy = row.get("policy")
        context = row.get("context_length")
        repetition = row.get("repetition")
        if (
            policy not in POLICIES
            or context not in CONTEXT_LENGTHS
            or isinstance(repetition, bool)
            or not isinstance(repetition, int)
            or not 0 <= repetition < REPETITIONS
        ):
            raise ValueError("performance row identity is invalid")
        if (
            row.get("prompt_tokens") != context
            or row.get("generated_tokens") != GENERATED_TOKENS
            or row.get("temperature") != 0.0
            or row.get("ignore_eos") is not True
            or row.get("tensor_parallel_size") != 1
            or row.get("max_num_seqs") != 1
            or row.get("completion_only") is not True
        ):
            raise ValueError("performance workload mismatch")
        expected_position = _policy_order(
            repetition,
            CONTEXT_LENGTHS.index(context),
        ).index(policy)
        if row.get("order_position") != expected_position:
            raise ValueError("performance execution order mismatch")
        if (
            row.get("run_tag") != run_tag
            or row.get("source_commit") != source_commit
            or row.get("source_patch_sha256") != patch_sha256
        ):
            raise ValueError("performance source identity mismatch")
        for field in (
            "prompt_sha256",
            "output_text_sha256",
            "one_token_graph_identity_sha256",
        ):
            _digest(row.get(field), field)
        if row.get("folded_graph_identity_sha256") is not None:
            _digest(
                row["folded_graph_identity_sha256"],
                "folded graph identity",
            )
        output_ids = row.get("output_token_ids")
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != GENERATED_TOKENS
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in output_ids
            )
        ):
            raise ValueError("performance output inventory mismatch")
        for field in (
            "tpot_median_ns",
            "tpot_p95_ns",
            "tpot_p99_ns",
            "ttft_ns",
            "e2e_ns",
            "output_tokens_per_second",
            "capture_allocated_delta_bytes",
            "capture_reserved_delta_bytes",
            "capture_retained_static_bytes",
            "capture_duration_ns",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            _finite_non_negative(row.get(field), field)
        samples = row.get("tpot_samples_ns")
        if (
            not isinstance(samples, list)
            or len(samples) != GENERATED_TOKENS - 1
        ):
            raise ValueError("TPOT sample inventory mismatch")
        for value in samples:
            _finite_non_negative(value, "TPOT sample")
        gaps = row.get("host_visible_burst_gaps_ns")
        if (
            not isinstance(gaps, list)
            or len(gaps) != math.ceil((GENERATED_TOKENS - 1) / 8)
        ):
            raise ValueError("host-visible gap inventory mismatch")
        for value in gaps:
            _non_negative_int(value, "host-visible gap")
        if row.get("maximum_host_visible_burst_gap_ns") != max(gaps):
            raise ValueError("host-visible gap maximum mismatch")
        _validate_runtime_inventory(row)
        identities.append((repetition, context, policy))
    if (
        len(identities) != len(set(identities))
        or set(identities) != _expected_performance_identities()
    ):
        raise ValueError("performance row inventory mismatch")
    return rows


def _read_logits(run_dir: Path, row: dict) -> tuple[float, ...]:
    path = _safe_file(run_dir, row.get("logits_path"))
    payload = path.read_bytes()
    count = _non_negative_int(
        row.get("logits_element_count"),
        "logits element count",
    )
    byte_length = _non_negative_int(
        row.get("logits_byte_length"),
        "logits byte length",
    )
    if byte_length != count * 4 or len(payload) != byte_length:
        raise ValueError("logits byte inventory mismatch")
    if sha256_file(path) != row.get("logits_sha256"):
        raise ValueError("logits digest mismatch")
    return tuple(value[0] for value in struct.iter_unpack("<f", payload))


def _validate_correctness_rows(
    rows: list[dict],
    *,
    run_dir: Path,
    run_tag: str,
    source_commit: str,
    patch_sha256: str,
) -> list[dict]:
    identities = []
    indexed = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != CORRECTNESS_REQUIRED_FIELDS
        ):
            raise ValueError("correctness row is invalid")
        if row.get("schema_version") != CORRECTNESS_SCHEMA_VERSION:
            raise ValueError("correctness row schema mismatch")
        policy = row.get("policy")
        context = row.get("context_length")
        point = row.get("sampling_point")
        identity = (context, policy, point)
        if (
            identity not in _expected_correctness_identities()
            or row.get("run_tag") != run_tag
            or row.get("source_commit") != source_commit
            or row.get("source_patch_sha256") != patch_sha256
        ):
            raise ValueError("correctness source or identity mismatch")
        _digest(row.get("prompt_sha256"), "correctness prompt")
        _digest(row.get("output_text_sha256"), "correctness output")
        _digest(row.get("logits_sha256"), "correctness logits")
        _digest(
            row.get("one_token_graph_identity_sha256"),
            "correctness graph identity",
        )
        if row.get("folded_graph_identity_sha256") is not None:
            _digest(
                row["folded_graph_identity_sha256"],
                "correctness folded graph identity",
            )
        output_ids = row.get("output_token_ids")
        if (
            row.get("generated_tokens") != GENERATED_TOKENS
            or not isinstance(output_ids, list)
            or len(output_ids) != GENERATED_TOKENS
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in output_ids
            )
        ):
            raise ValueError("correctness output inventory mismatch")
        values = _read_logits(run_dir, row)
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
            or row.get("correctness_trace") is not True
        ):
            raise ValueError("correctness logits inventory mismatch")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("correctness logits contain non-finite values")
        argmax = max(range(len(values)), key=values.__getitem__)
        if row.get("argmax_token_id") != argmax:
            raise ValueError("correctness argmax mismatch")
        _validate_runtime_inventory(row)
        identities.append(identity)
        indexed[identity] = row
    if (
        len(identities) != len(set(identities))
        or set(identities) != _expected_correctness_identities()
    ):
        raise ValueError("correctness row inventory mismatch")
    for context in CONTEXT_LENGTHS:
        for point in SAMPLING_POINTS:
            control = indexed[(context, "one_token_graph", point)]
            candidate = indexed[(context, "octet_folded_graph", point)]
            for field in (
                "prompt_sha256",
                "output_token_ids",
                "output_text_sha256",
                "argmax_token_id",
                "logits_sha256",
            ):
                if control[field] != candidate[field]:
                    raise ValueError("correctness policy mismatch")
    return rows


def _improvement_pct(control: float, candidate: float) -> float:
    if control <= 0.0:
        raise ValueError("control metric must be positive")
    return (control - candidate) / control * 100.0


def _classification_reasons(metrics: dict) -> list[str]:
    reasons = []
    for field in (
        "evidence_complete",
        "source_exact",
        "workload_identity_exact",
        "execution_order_exact",
        "correctness_exact",
        "runtime_inventory_exact",
        "physical_launch_reduction_exact",
        "no_runtime_anomalies",
    ):
        if metrics[field] is not True:
            reasons.append(field)
    thresholds = (
        ("aggregate_median_tpot_improvement_pct", 1.0, "min"),
        ("aggregate_p95_tpot_improvement_pct", 0.5, "min"),
        ("maximum_ttft_regression_pct", 2.0, "max"),
        ("maximum_e2e_regression_pct", 2.0, "max"),
        ("minimum_throughput_improvement_pct", -2.0, "min"),
        ("maximum_capture_allocated_ratio", 0.01, "max"),
        ("maximum_capture_reserved_ratio", 0.01, "max"),
        (
            "maximum_retained_static_delta_bytes",
            MAXIMUM_RETAINED_STATIC_DELTA_BYTES,
            "max",
        ),
        (
            "maximum_folded_capture_duration_ns",
            MAXIMUM_FOLDED_CAPTURE_DURATION_NS,
            "max",
        ),
    )
    for field, threshold, direction in thresholds:
        value = float(metrics[field])
        if not math.isfinite(value):
            reasons.append(field)
        elif direction == "min" and value < threshold:
            reasons.append(field)
        elif direction == "max" and value > threshold:
            reasons.append(field)
    return reasons


def _reconstruct_ceiling(
    performance: list[dict],
    correctness: list[dict],
    *,
    source_commit: str,
    patch_sha256: str,
) -> dict:
    pairs = {}
    for row in performance:
        pairs.setdefault(
            (row["repetition"], row["context_length"]),
            {},
        )[row["policy"]] = row
    ordered_pairs = list(pairs.values())
    controls = [pair["one_token_graph"] for pair in ordered_pairs]
    candidates = [pair["octet_folded_graph"] for pair in ordered_pairs]
    median_improvements = [
        _improvement_pct(
            control["tpot_median_ns"],
            candidate["tpot_median_ns"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    p95_improvements = [
        _improvement_pct(
            control["tpot_p95_ns"],
            candidate["tpot_p95_ns"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    ttft_regressions = [
        -_improvement_pct(control["ttft_ns"], candidate["ttft_ns"])
        for control, candidate in zip(controls, candidates)
    ]
    e2e_regressions = [
        -_improvement_pct(control["e2e_ns"], candidate["e2e_ns"])
        for control, candidate in zip(controls, candidates)
    ]
    throughput_improvements = [
        -_improvement_pct(
            control["output_tokens_per_second"],
            candidate["output_tokens_per_second"],
        )
        for control, candidate in zip(controls, candidates)
    ]
    metrics = {
        "schema_version": CEILING_SCHEMA_VERSION,
        "evidence_complete": True,
        "source_exact": True,
        "workload_identity_exact": all(
            control["prompt_sha256"] == candidate["prompt_sha256"]
            and control["output_token_ids"]
            == candidate["output_token_ids"]
            and control["output_text_sha256"]
            == candidate["output_text_sha256"]
            for control, candidate in zip(controls, candidates)
        ),
        "execution_order_exact": True,
        "correctness_exact": True,
        "runtime_inventory_exact": True,
        "physical_launch_reduction_exact": True,
        "no_runtime_anomalies": True,
        "aggregate_median_tpot_improvement_pct":
            statistics.median(median_improvements),
        "aggregate_p95_tpot_improvement_pct":
            statistics.median(p95_improvements),
        "maximum_ttft_regression_pct": max(ttft_regressions),
        "maximum_e2e_regression_pct": max(e2e_regressions),
        "minimum_throughput_improvement_pct": min(
            throughput_improvements
        ),
        "maximum_capture_allocated_ratio": max(
            candidate["capture_allocated_delta_bytes"]
            / max(1, control["cuda_peak_allocated_bytes"])
            for control, candidate in zip(controls, candidates)
        ),
        "maximum_capture_reserved_ratio": max(
            candidate["capture_reserved_delta_bytes"]
            / max(1, control["cuda_peak_reserved_bytes"])
            for control, candidate in zip(controls, candidates)
        ),
        "maximum_retained_static_delta_bytes": max(
            candidate["capture_retained_static_bytes"]
            - control["capture_retained_static_bytes"]
            for control, candidate in zip(controls, candidates)
        ),
        "maximum_folded_capture_duration_ns": max(
            row["capture_duration_ns"] for row in candidates
        ),
        "source_commit": source_commit,
        "source_patch_sha256": patch_sha256,
        "observed_source_commits": [source_commit],
        "performance_row_count": len(performance),
        "correctness_row_count": len(correctness),
        "inventory_error": None,
    }
    metrics["classification_reasons"] = _classification_reasons(metrics)
    metrics["classification"] = (
        GO_CEILING
        if not metrics["classification_reasons"]
        else NO_GO_CEILING
    )
    return metrics


def verify_artifact_directory(
    run_dir: Path,
    *,
    source_root: Path | None = None,
) -> dict:
    root = Path(run_dir).resolve()
    workload = read_json(root / "workload_manifest.json")
    source = read_json(root / "source_manifest.json")
    if (
        not isinstance(workload, dict)
        or set(workload) != WORKLOAD_REQUIRED_FIELDS
        or workload.get("schema_version") != WORKLOAD_SCHEMA_VERSION
        or not isinstance(workload.get("model"), str)
        or not workload["model"]
        or not isinstance(workload.get("device"), str)
        or not workload["device"].startswith("cuda")
        or not isinstance(workload.get("run_tag"), str)
        or not workload["run_tag"]
        or workload.get("contexts") != list(CONTEXT_LENGTHS)
        or workload.get("policies") != list(POLICIES)
        or workload.get("generated_tokens") != GENERATED_TOKENS
        or workload.get("repetitions") != REPETITIONS
        or workload.get("warmup_repetitions") != WARMUP_REPETITIONS
        or workload.get("performance_row_count") != 30
        or workload.get("correctness_row_count") != 24
        or workload.get("execution_order") != [
            list(_policy_order(0, index))
            for index in range(len(CONTEXT_LENGTHS))
        ]
        or workload.get("temperature") != 0.0
        or workload.get("ignore_eos") is not True
        or workload.get("tensor_parallel_size") != 1
        or workload.get("max_num_seqs") != 1
        or workload.get("completion_only") is not True
        or isinstance(workload.get("gpu_memory_utilization"), bool)
        or not isinstance(
            workload.get("gpu_memory_utilization"),
            (int, float),
        )
        or not math.isfinite(
            float(workload["gpu_memory_utilization"])
        )
        or not 0.0 < float(
            workload["gpu_memory_utilization"]
        ) <= 1.0
        or not isinstance(workload.get("environment"), dict)
    ):
        raise ValueError("workload manifest mismatch")
    run_tag = workload.get("run_tag")
    source_commit = _digest(
        workload.get("source_commit"),
        "source commit",
        lengths=(40, 64),
    )
    patch_sha256 = _digest(
        workload.get("source_patch_sha256"),
        "source patch sha256",
    )
    if (
        source.get("schema_version") != SOURCE_SCHEMA_VERSION
        or source.get("run_tag") != run_tag
        or source.get("source_commit") != source_commit
        or source.get("source_patch_sha256") != patch_sha256
        or set(source.get("source_sha256", {})) != set(SOURCE_FILES)
    ):
        raise ValueError("source manifest mismatch")
    patch_path = root / "source.patch"
    if not patch_path.is_file() or sha256_file(patch_path) != patch_sha256:
        raise ValueError("source patch mismatch")
    if source_root is not None:
        source_root = Path(source_root).resolve()
        for relative, expected in source["source_sha256"].items():
            target = (source_root / relative).resolve()
            if (
                source_root not in target.parents
                or not target.is_file()
                or sha256_file(target) != expected
            ):
                raise ValueError("source hash mismatch")
    performance = _validate_performance_rows(
        read_jsonl(root / "performance_rows.jsonl"),
        run_tag=run_tag,
        source_commit=source_commit,
        patch_sha256=patch_sha256,
    )
    correctness = _validate_correctness_rows(
        read_jsonl(root / "correctness_rows.jsonl"),
        run_dir=root,
        run_tag=run_tag,
        source_commit=source_commit,
        patch_sha256=patch_sha256,
    )
    expected_summary = {
        "schema_version": PROFILE_SUMMARY_SCHEMA_VERSION,
        "performance_row_count": len(performance),
        "all_outputs_exact": all(
            pair["one_token_graph"]["output_token_ids"]
            == pair["octet_folded_graph"]["output_token_ids"]
            and pair["one_token_graph"]["output_text_sha256"]
            == pair["octet_folded_graph"]["output_text_sha256"]
            for pair in _paired_rows(performance)
        ),
        "correctness_row_count": len(correctness),
    }
    if read_json(root / "profile_summary.json") != expected_summary:
        raise ValueError("profile summary mismatch")
    reconstructed = _reconstruct_ceiling(
        performance,
        correctness,
        source_commit=source_commit,
        patch_sha256=patch_sha256,
    )
    if read_json(root / "ceiling.json") != reconstructed:
        raise ValueError("recorded ceiling mismatch")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "run_tag": run_tag,
        "source_commit": source_commit,
        "source_patch_sha256": patch_sha256,
        "classification": reconstructed["classification"],
        "performance_row_count": len(performance),
        "correctness_row_count": len(correctness),
    }


def _paired_rows(rows: list[dict]) -> list[dict[str, dict]]:
    pairs = {}
    for row in rows:
        pairs.setdefault(
            (row["repetition"], row["context_length"]),
            {},
        )[row["policy"]] = row
    return list(pairs.values())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = verify_artifact_directory(
        args.run_dir,
        source_root=args.source_root,
    )
    if args.output is None:
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    else:
        write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
