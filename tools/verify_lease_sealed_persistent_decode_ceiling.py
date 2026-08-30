#!/usr/bin/env python3
"""Independent verifier for persistent-decode ceiling artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics


VERIFICATION_SCHEMA_VERSION = (
    "lease-sealed-persistent-decode.verification.v1"
)
TIMING_SCHEMA_VERSION = "lease-sealed-persistent-decode.timing.v1"
STRUCTURAL_SCHEMA_VERSION = (
    "lease-sealed-persistent-decode.structural.v1"
)
TRACE_SUMMARY_SCHEMA_VERSION = (
    "lease-sealed-persistent-decode.trace-summary.v1"
)
CEILING_SCHEMA_VERSION = "lease-sealed-persistent-decode.ceiling.v1"
MANIFEST_SCHEMA_VERSION = "lease-sealed-persistent-decode.manifest.v1"
CONTEXT_LENGTHS = (256, 2048, 8192)
GENERATED_TOKENS = 128
REPETITIONS = 5
MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT = 5.0
MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT = 3.0
MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT = 4.0
MIN_CLASSIFIED_LAUNCH_RATIO = 0.98
MIN_CLASSIFIED_DURATION_RATIO = 0.99
MAX_MEDIAN_PROFILE_PERTURBATION_PCT = 10.0
MAX_P95_PROFILE_PERTURBATION_PCT = 15.0
GO = "GO_PERSISTENT_DECODE_CEILING"
NO_GO = "NO_GO_PERSISTENT_DECODE_CEILING"
INCONCLUSIVE_PROFILE_OVERHEAD = "INCONCLUSIVE_PROFILE_OVERHEAD"
INCONCLUSIVE_TRACE_COVERAGE = "INCONCLUSIVE_TRACE_COVERAGE"
INCONCLUSIVE_CORRECTNESS = "INCONCLUSIVE_CORRECTNESS"
REQUIRED_FILES = {
    "source_manifest.json",
    "runtime_manifest.json",
    "gpu_admission.json",
    "workload_manifest.json",
    "timing_rows.jsonl",
    "structural_rows.jsonl",
    "timing_summary.json",
    "trace_inventory.json",
    "kernel_rows.jsonl",
    "segment_rows.jsonl",
    "ceiling.json",
}
IDENTITY_FIELDS = (
    "source_commit",
    "source_tree_sha256",
    "runtime_identity_sha256",
    "workload_identity_sha256",
)
KNOWN_ROLES = {
    "MATMUL",
    "ATTENTION",
    "NORMALIZATION",
    "ELEMENTWISE",
    "REDUCTION",
    "INDEX_OR_STATE_UPDATE",
    "TOKEN_SELECTION",
    "COPY_OR_FILL",
    "RUNTIME_OR_GRAPH",
    "UNKNOWN",
}
CANDIDATE_ROLES = {
    "NORMALIZATION",
    "ELEMENTWISE",
    "REDUCTION",
    "INDEX_OR_STATE_UPDATE",
    "TOKEN_SELECTION",
}
ROLE_PATTERNS = (
    ("TOKEN_SELECTION", ("argmax", "topk", "sampling", "sample_token")),
    ("ATTENTION", ("flash", "attention", "fmha", "paged_attn")),
    (
        "MATMUL",
        ("gemm", "matmul", "cublas", "cutlass", "sgemm", "bgemm"),
    ),
    (
        "NORMALIZATION",
        ("rms_norm", "rmsnorm", "layer_norm", "layernorm", "norm_kernel"),
    ),
    (
        "COPY_OR_FILL",
        ("memcpy", "memset", "vectorized_copy", "vectorized_mem", "fill_kernel"),
    ),
    (
        "INDEX_OR_STATE_UPDATE",
        (
            "index_put",
            "index_select",
            "scatter",
            "slot_mapping",
            "cache_store",
            "state_update",
        ),
    ),
    (
        "ELEMENTWISE",
        ("silu", "gelu", "elementwise", "pointwise", "add_kernel", "mul_kernel"),
    ),
    ("REDUCTION", ("reduce", "softmax")),
    (
        "RUNTIME_OR_GRAPH",
        ("cudagraph", "cuda_graph", "graphlaunch", "barrier"),
    ),
)


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path):
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path.name}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path.name}")
    return [
        json.loads(line, parse_constant=_reject_constant)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _digest(value, field: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} is invalid")
    return value


def _integer(value, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value, field: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} is non-finite")
    result = float(value)
    if result < 0.0 or (positive and result <= 0.0):
        raise ValueError(f"{field} is outside its valid range")
    return result


def _safe_manifest_path(run_dir: Path, value) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path is invalid")
    target = (run_dir / value).resolve()
    if run_dir not in target.parents or not target.is_file():
        raise ValueError(f"artifact path is invalid: {value}")
    return target


def _verify_manifest(run_dir: Path) -> None:
    manifest = _read_json(run_dir / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest schema mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")
    declared = {}
    for row in artifacts:
        if not isinstance(row, dict):
            raise ValueError("manifest artifact must be an object")
        relative = row.get("path")
        if relative in declared:
            raise ValueError("duplicate artifact path")
        target = _safe_manifest_path(run_dir, relative)
        declared[relative] = target
        if _integer(
            row.get("byte_length"),
            "artifact byte_length",
        ) != target.stat().st_size:
            raise ValueError("artifact byte length mismatch")
        if _digest(row.get("sha256"), "artifact sha256") != _sha256_file(target):
            raise ValueError("artifact digest mismatch")
    if set(declared) != REQUIRED_FILES:
        raise ValueError("manifest artifact inventory mismatch")
    present = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    expected = REQUIRED_FILES | {"manifest.json"}
    extras = present - expected
    if extras:
        raise ValueError(f"undeclared artifact: {sorted(extras)[0]}")


def _validate_manifests(run_dir: Path) -> tuple[dict, dict, dict, dict]:
    source = _read_json(run_dir / "source_manifest.json")
    runtime = _read_json(run_dir / "runtime_manifest.json")
    gpu = _read_json(run_dir / "gpu_admission.json")
    workload = _read_json(run_dir / "workload_manifest.json")
    hashes = source.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("source hash inventory is invalid")
    for path, digest in hashes.items():
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError("source hash path is invalid")
        _digest(digest, "source hash")
    expected_tree = _canonical_digest(hashes)
    if source.get("source_tree_sha256") != expected_tree:
        raise ValueError("source hash tree mismatch")
    _digest(source.get("source_commit"), "source commit", length=40)
    if gpu.get("strict_clean") is not True:
        raise ValueError("GPU admission is not strict-clean")
    for field in (
        "compute_process_count",
        "memory_used_mib",
        "utilization_gpu_pct",
    ):
        if _integer(gpu.get(field), field) != 0:
            raise ValueError("GPU admission is not strict-clean")
    if workload.get("contexts") != list(CONTEXT_LENGTHS):
        raise ValueError("workload context inventory mismatch")
    if (
        workload.get("generated_tokens") != GENERATED_TOKENS
        or workload.get("repetitions") != REPETITIONS
        or workload.get("temperature") != 0.0
        or workload.get("ignore_eos") is not True
        or workload.get("max_num_seqs") != 1
    ):
        raise ValueError("workload configuration mismatch")
    return source, runtime, gpu, workload


def _identity(
    row: dict,
    *,
    source: dict,
    runtime_digest: str,
    workload_digest: str,
) -> None:
    expected = {
        "source_commit": source["source_commit"],
        "source_tree_sha256": source["source_tree_sha256"],
        "runtime_identity_sha256": runtime_digest,
        "workload_identity_sha256": workload_digest,
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise ValueError("source/runtime/workload identity mismatch")


def _tokens(row: dict, prefix: str) -> tuple[list[int], str]:
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
        raise ValueError(f"{prefix} output token inventory is invalid")
    text_digest = _digest(
        row.get("output_text_sha256"),
        f"{prefix} output text digest",
    )
    return list(tokens), text_digest


def _clean_runtime(row: dict, prefix: str) -> None:
    if (
        _integer(row.get("target_model_forwards"), "target_model_forwards")
        != GENERATED_TOKENS - 1
        or _integer(row.get("committed_tokens"), "committed_tokens")
        != GENERATED_TOKENS - 1
        or _integer(row.get("fallback_count"), "fallback_count") != 0
        or _integer(row.get("failure_count"), "failure_count") != 0
        or _integer(row.get("rollback_count"), "rollback_count") != 0
        or row.get("quarantine_reason") is not None
    ):
        raise ValueError(f"{prefix} runtime evidence is not clean")


def _validate_timing(
    rows: list[dict],
    *,
    source: dict,
    runtime_digest: str,
    workload_digest: str,
) -> dict[int, list[dict]]:
    expected = {
        (repetition, context)
        for repetition in range(REPETITIONS)
        for context in CONTEXT_LENGTHS
    }
    seen = set()
    grouped = {context: [] for context in CONTEXT_LENGTHS}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != TIMING_SCHEMA_VERSION
            or row.get("arm") != "uninstrumented"
        ):
            raise ValueError("timing row contract mismatch")
        _identity(
            row,
            source=source,
            runtime_digest=runtime_digest,
            workload_digest=workload_digest,
        )
        key = (
            _integer(row.get("repetition"), "repetition"),
            _integer(row.get("context_length"), "context_length", minimum=1),
        )
        if key in seen:
            raise ValueError("duplicate timing identity")
        seen.add(key)
        if row.get("generated_tokens") != GENERATED_TOKENS:
            raise ValueError("timing generated token count mismatch")
        _tokens(row, "timing")
        _finite(row.get("tpot_median_ns"), "tpot_median_ns", positive=True)
        _finite(row.get("tpot_p95_ns"), "tpot_p95_ns", positive=True)
        _clean_runtime(row, "timing")
        grouped.setdefault(key[1], []).append(row)
    if seen != expected:
        raise ValueError("timing inventory is incomplete")
    return grouped


def _validate_structural(
    rows: list[dict],
    *,
    source: dict,
    runtime_digest: str,
    workload_digest: str,
    timing_by_context: dict[int, list[dict]],
) -> dict[int, dict]:
    structural = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != STRUCTURAL_SCHEMA_VERSION
        ):
            raise ValueError("structural row contract mismatch")
        _identity(
            row,
            source=source,
            runtime_digest=runtime_digest,
            workload_digest=workload_digest,
        )
        context = _integer(
            row.get("context_length"),
            "structural context_length",
            minimum=1,
        )
        if context in structural:
            raise ValueError("duplicate structural context")
        if row.get("generated_tokens") != GENERATED_TOKENS:
            raise ValueError("structural generated token count mismatch")
        observed_tokens, observed_digest = _tokens(row, "structural")
        observed = (tuple(observed_tokens), observed_digest)
        expected_outputs = {
            (
                tuple(_tokens(timing, "timing")[0]),
                _tokens(timing, "timing")[1],
            )
            for timing in timing_by_context.get(context, ())
        }
        if len(expected_outputs) != 1 or observed not in expected_outputs:
            raise ValueError("timing and structural output mismatch")
        if row.get("burst_logical_tokens") != [8] * 15 + [7]:
            raise ValueError("structural logical-token inventory mismatch")
        _finite(
            row.get("profiled_tpot_median_ns"),
            "profiled_tpot_median_ns",
            positive=True,
        )
        _finite(
            row.get("profiled_tpot_p95_ns"),
            "profiled_tpot_p95_ns",
            positive=True,
        )
        _clean_runtime(row, "structural")
        structural[context] = row
    if set(structural) != set(CONTEXT_LENGTHS):
        raise ValueError("structural inventory is incomplete")
    return structural


def _classify_kernel(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    for role, patterns in ROLE_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return role
    return "UNKNOWN"


def _normalized_kernel_name(name: str) -> str:
    normalized = " ".join(name.strip().lower().split())
    return re.sub(r"0x[0-9a-f]+", "0x#", normalized)


def _segment(rows: list[dict], ordinal: int) -> dict:
    first = rows[0]
    last = rows[-1]
    duration = sum(row["duration_ns"] for row in rows)
    wall = last["end_ns"] - first["start_ns"]
    histogram = {}
    signature_rows = []
    for row in rows:
        histogram[row["role"]] = histogram.get(row["role"], 0) + 1
        signature_rows.append((
            row["role"],
            _normalized_kernel_name(row["name"]),
        ))
    return {
        **{
            field: first[field]
            for field in (
                "attempt",
                "workload",
                "repetition",
                "context",
                "burst",
                "logical_tokens",
            )
        },
        "segment_id": ordinal,
        "stream_id": first["stream_id"],
        "first_kernel_start_ns": first["start_ns"],
        "last_kernel_end_ns": last["end_ns"],
        "kernel_count": len(rows),
        "kernel_duration_sum_ns": duration,
        "internal_gap_sum_ns": wall - duration,
        "wall_union_ns": wall,
        "role_histogram": dict(sorted(histogram.items())),
        "normalized_kernel_signature_sha256": _canonical_digest(
            signature_rows
        ),
    }


def _reconstruct_segments(rows: list[dict]) -> list[dict]:
    identity_fields = (
        "attempt",
        "workload",
        "repetition",
        "context",
        "burst",
        "logical_tokens",
    )
    ordered = sorted(
        rows,
        key=lambda row: (
            tuple(row[field] for field in identity_fields),
            row["start_ns"],
            row["end_ns"],
            row["stream_id"],
        ),
    )
    segments = []
    active = {}
    previous = {}
    active_identity = None
    ordinal = 0

    def flush(stream):
        nonlocal ordinal
        buffered = active.pop(stream, None)
        if buffered:
            segments.append(_segment(buffered, ordinal))
            ordinal += 1

    for row in ordered:
        identity = tuple(row[field] for field in identity_fields)
        if active_identity is None:
            active_identity = identity
        elif identity != active_identity:
            for stream in tuple(active):
                flush(stream)
            previous.clear()
            active_identity = identity
        stream = row["stream_id"]
        if stream in previous and row["start_ns"] < previous[stream]["end_ns"]:
            raise ValueError("kernel intervals overlap")
        previous[stream] = row
        if row["role"] not in CANDIDATE_ROLES:
            flush(stream)
        else:
            active.setdefault(stream, []).append(row)
    for stream in tuple(active):
        flush(stream)
    return sorted(
        segments,
        key=lambda row: (
            tuple(row[field] for field in identity_fields),
            row["first_kernel_start_ns"],
            row["stream_id"],
        ),
    )


def _validate_kernel_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    if not rows:
        raise ValueError("kernel inventory is empty")
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("kernel row must be an object")
        current = dict(row)
        for field in (
            "repetition",
            "context",
            "burst",
            "logical_tokens",
            "start_ns",
            "end_ns",
            "duration_ns",
            "stream_id",
        ):
            current[field] = _integer(
                current.get(field),
                f"kernel {field}",
                minimum=1 if field in {"context", "logical_tokens", "duration_ns"} else 0,
            )
        if current["end_ns"] - current["start_ns"] != current["duration_ns"]:
            raise ValueError("kernel duration mismatch")
        if current.get("role") not in KNOWN_ROLES:
            raise ValueError("kernel role is invalid")
        inferred = _classify_kernel(str(current.get("name", "")))
        if current["role"] != "UNKNOWN" and current["role"] != inferred:
            raise ValueError("kernel role mismatch")
        normalized.append(current)
    total_duration = sum(row["duration_ns"] for row in normalized)
    classified = [row for row in normalized if row["role"] != "UNKNOWN"]
    coverage = {
        "classified_launch_ratio": len(classified) / len(normalized),
        "classified_duration_ratio": (
            sum(row["duration_ns"] for row in classified) / total_duration
        ),
        "total_kernel_duration_ns": total_duration,
    }
    if (
        coverage["classified_launch_ratio"] < MIN_CLASSIFIED_LAUNCH_RATIO
        or coverage["classified_duration_ratio"] < MIN_CLASSIFIED_DURATION_RATIO
    ):
        raise ValueError("trace coverage is below the frozen threshold")
    return normalized, coverage


def _trace_contexts(
    *,
    structural: dict[int, dict],
    kernels: list[dict],
    segments: list[dict],
) -> list[dict]:
    result = []
    for context in CONTEXT_LENGTHS:
        context_kernels = [
            row for row in kernels if row["context"] == context
        ]
        context_segments = [
            row for row in segments if row["context"] == context
        ]
        if not context_kernels or not context_segments:
            raise ValueError("trace context inventory is incomplete")
        total_duration = sum(
            row["duration_ns"] for row in context_kernels
        )
        classified = [
            row for row in context_kernels
            if row["role"] != "UNKNOWN"
        ]
        candidate_duration = sum(
            row["kernel_duration_sum_ns"]
            for row in context_segments
        )
        eligible = sum(
            row["wall_union_ns"] for row in context_segments
        )
        logical_tokens = sum(
            row["logical_tokens"]
            for key, row in {
                (
                    item["attempt"],
                    item["workload"],
                    item["repetition"],
                    item["context"],
                    item["burst"],
                ): item
                for item in context_kernels
            }.items()
        )
        structural_row = structural[context]
        result.append({
            "context_length": context,
            "profiled_tpot_median_ns":
                structural_row["profiled_tpot_median_ns"],
            "profiled_tpot_p95_ns":
                structural_row["profiled_tpot_p95_ns"],
            "output_token_ids": structural_row["output_token_ids"],
            "output_text_sha256":
                structural_row["output_text_sha256"],
            "transaction_count": len({
                (
                    row["attempt"],
                    row["workload"],
                    row["repetition"],
                    row["context"],
                    row["burst"],
                )
                for row in context_kernels
            }),
            "logical_token_count": logical_tokens,
            "eligible_zero_cost_ns_per_token": eligible / logical_tokens,
            "candidate_cuda_duration_ns": candidate_duration,
            "total_kernel_duration_ns": total_duration,
            "classified_launch_ratio":
                len(classified) / len(context_kernels),
            "classified_duration_ratio": (
                sum(row["duration_ns"] for row in classified)
                / total_duration
            ),
            "segment_signatures": sorted({
                row["normalized_kernel_signature_sha256"]
                for row in context_segments
            }),
            **{
                field: structural_row[field]
                for field in (
                    "target_model_forwards",
                    "committed_tokens",
                    "fallback_count",
                    "failure_count",
                    "rollback_count",
                    "quarantine_reason",
                )
            },
        })
    return result


def _regression_pct(candidate: float, baseline: float) -> float:
    return max(0.0, (candidate / baseline - 1.0) * 100.0)


def _compute_ceiling(
    timing_by_context: dict[int, list[dict]],
    contexts: list[dict],
) -> dict:
    context_by_length = {row["context_length"]: row for row in contexts}
    metrics = []
    median_perturbations = []
    p95_perturbations = []
    for context in CONTEXT_LENGTHS:
        baseline_median = statistics.median(
            row["tpot_median_ns"]
            for row in timing_by_context[context]
        )
        baseline_p95 = statistics.median(
            row["tpot_p95_ns"]
            for row in timing_by_context[context]
        )
        traced = context_by_length[context]
        optimistic = (
            traced["eligible_zero_cost_ns_per_token"]
            / baseline_median
            * 100.0
        )
        median_perturbation = _regression_pct(
            traced["profiled_tpot_median_ns"],
            baseline_median,
        )
        p95_perturbation = _regression_pct(
            traced["profiled_tpot_p95_ns"],
            baseline_p95,
        )
        median_perturbations.append(median_perturbation)
        p95_perturbations.append(p95_perturbation)
        metrics.append({
            "context_length": context,
            "baseline_tpot_median_ns": baseline_median,
            "baseline_tpot_p95_ns": baseline_p95,
            "profiled_tpot_median_ns":
                traced["profiled_tpot_median_ns"],
            "profiled_tpot_p95_ns":
                traced["profiled_tpot_p95_ns"],
            "profile_median_perturbation_pct": median_perturbation,
            "profile_p95_perturbation_pct": p95_perturbation,
            "eligible_zero_cost_ns_per_token":
                traced["eligible_zero_cost_ns_per_token"],
            "optimistic_improvement_pct": optimistic,
            "candidate_cuda_duration_ns":
                traced["candidate_cuda_duration_ns"],
            "total_kernel_duration_ns":
                traced["total_kernel_duration_ns"],
            "classified_launch_ratio":
                traced["classified_launch_ratio"],
            "classified_duration_ratio":
                traced["classified_duration_ratio"],
            "segment_signatures": traced["segment_signatures"],
        })
    maximum_median = max(median_perturbations)
    maximum_p95 = max(p95_perturbations)
    if (
        maximum_median > MAX_MEDIAN_PROFILE_PERTURBATION_PCT
        or maximum_p95 > MAX_P95_PROFILE_PERTURBATION_PCT
    ):
        failures = []
        if maximum_median > MAX_MEDIAN_PROFILE_PERTURBATION_PCT:
            failures.append("profile_median_perturbation_pct")
        if maximum_p95 > MAX_P95_PROFILE_PERTURBATION_PCT:
            failures.append("profile_p95_perturbation_pct")
        return {
            "schema_version": CEILING_SCHEMA_VERSION,
            "classification": INCONCLUSIVE_PROFILE_OVERHEAD,
            "failed_conditions": failures,
            "maximum_profile_median_perturbation_pct": maximum_median,
            "maximum_profile_p95_perturbation_pct": maximum_p95,
            "contexts": metrics,
        }
    optimistic_values = [
        row["optimistic_improvement_pct"] for row in metrics
    ]
    aggregate_optimistic = statistics.median(optimistic_values)
    total_candidate = sum(
        row["candidate_cuda_duration_ns"] for row in metrics
    )
    total_kernel = sum(
        row["total_kernel_duration_ns"] for row in metrics
    )
    candidate_share = total_candidate / total_kernel * 100.0
    minimum_launch = min(
        row["classified_launch_ratio"] for row in metrics
    )
    minimum_duration = min(
        row["classified_duration_ratio"] for row in metrics
    )
    stable_signatures = sorted(set.intersection(*(
        set(row["segment_signatures"]) for row in metrics
    )))
    failures = []
    if aggregate_optimistic < MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT:
        failures.append("aggregate_optimistic_improvement_pct")
    if min(optimistic_values) < MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT:
        failures.append("minimum_context_optimistic_improvement_pct")
    if candidate_share < MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT:
        failures.append("aggregate_candidate_cuda_duration_share_pct")
    if not stable_signatures:
        failures.append("stable_cross_context_signatures")
    return {
        "schema_version": CEILING_SCHEMA_VERSION,
        "classification": NO_GO if failures else GO,
        "failed_conditions": failures,
        "thresholds": {
            "minimum_aggregate_optimistic_improvement_pct":
                MIN_AGGREGATE_OPTIMISTIC_IMPROVEMENT_PCT,
            "minimum_context_optimistic_improvement_pct":
                MIN_CONTEXT_OPTIMISTIC_IMPROVEMENT_PCT,
            "minimum_candidate_cuda_duration_share_pct":
                MIN_CANDIDATE_CUDA_DURATION_SHARE_PCT,
            "minimum_classified_launch_ratio":
                MIN_CLASSIFIED_LAUNCH_RATIO,
            "minimum_classified_duration_ratio":
                MIN_CLASSIFIED_DURATION_RATIO,
            "maximum_profile_median_perturbation_pct":
                MAX_MEDIAN_PROFILE_PERTURBATION_PCT,
            "maximum_profile_p95_perturbation_pct":
                MAX_P95_PROFILE_PERTURBATION_PCT,
        },
        "aggregate_optimistic_improvement_pct": aggregate_optimistic,
        "minimum_context_optimistic_improvement_pct":
            min(optimistic_values),
        "aggregate_candidate_cuda_duration_share_pct": candidate_share,
        "minimum_classified_launch_ratio": minimum_launch,
        "minimum_classified_duration_ratio": minimum_duration,
        "maximum_profile_median_perturbation_pct": maximum_median,
        "maximum_profile_p95_perturbation_pct": maximum_p95,
        "stable_cross_context_signatures": stable_signatures,
        "contexts": metrics,
    }


def _validate_trace_inventory(payload: dict) -> None:
    raw = payload.get("raw_traces")
    if not isinstance(raw, list) or len(raw) != len(CONTEXT_LENGTHS):
        raise ValueError("raw trace inventory is incomplete")
    seen = set()
    approved = (
        "/data00/home/sitian/tinyllmforge-workspaces/"
        "command-timeline-20260818/"
    )
    for row in raw:
        context = _integer(
            row.get("context_length"),
            "raw trace context",
            minimum=1,
        )
        if context in seen:
            raise ValueError("duplicate raw trace context")
        seen.add(context)
        path = row.get("remote_path")
        if not isinstance(path, str) or not path.startswith(approved):
            raise ValueError("raw trace path is outside approved root")
        _integer(row.get("byte_length"), "raw trace byte_length", minimum=1)
        try:
            _digest(row.get("sha256"), "raw trace digest")
        except ValueError as error:
            raise ValueError("raw trace digest is missing or invalid") from error
    if seen != set(CONTEXT_LENGTHS):
        raise ValueError("raw trace inventory is incomplete")


def verify_artifact_directory(run_dir: Path) -> dict:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise ValueError("artifact directory does not exist")
    _verify_manifest(root)
    source, runtime, _gpu, workload = _validate_manifests(root)
    runtime_digest = _canonical_digest(runtime)
    workload_digest = _canonical_digest(workload)
    timing_rows = _read_jsonl(root / "timing_rows.jsonl")
    timing_by_context = _validate_timing(
        timing_rows,
        source=source,
        runtime_digest=runtime_digest,
        workload_digest=workload_digest,
    )
    structural_rows = _read_jsonl(root / "structural_rows.jsonl")
    structural = _validate_structural(
        structural_rows,
        source=source,
        runtime_digest=runtime_digest,
        workload_digest=workload_digest,
        timing_by_context=timing_by_context,
    )
    trace_inventory = _read_json(root / "trace_inventory.json")
    _validate_trace_inventory(trace_inventory)
    kernel_rows, _coverage = _validate_kernel_rows(
        _read_jsonl(root / "kernel_rows.jsonl")
    )
    reconstructed_segments = _reconstruct_segments(kernel_rows)
    recorded_segments = _read_jsonl(root / "segment_rows.jsonl")
    if recorded_segments != reconstructed_segments:
        raise ValueError("segment mismatch")
    contexts = _trace_contexts(
        structural=structural,
        kernels=kernel_rows,
        segments=reconstructed_segments,
    )
    trace_summary = {
        "schema_version": TRACE_SUMMARY_SCHEMA_VERSION,
        **{
            field: timing_rows[0][field]
            for field in IDENTITY_FIELDS
        },
        "contexts": contexts,
    }
    if trace_inventory.get("trace_summary") != trace_summary:
        raise ValueError("trace summary mismatch")
    timing_summary = _read_json(root / "timing_summary.json")
    if (
        timing_summary.get("schema_version") != TIMING_SCHEMA_VERSION
        or timing_summary.get("row_count") != len(timing_rows)
        or timing_summary.get("contexts") != list(CONTEXT_LENGTHS)
    ):
        raise ValueError("timing summary mismatch")
    reconstructed = _compute_ceiling(timing_by_context, contexts)
    reported = _read_json(root / "ceiling.json")
    if reported != reconstructed:
        raise ValueError("ceiling mismatch")
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "verified": True,
        "run_tag": source["run_tag"],
        "source_commit": source["source_commit"],
        "classification": reconstructed["classification"],
        "timing_row_count": len(timing_rows),
        "structural_context_count": len(structural_rows),
        "kernel_row_count": len(kernel_rows),
        "segment_row_count": len(reconstructed_segments),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(
        verify_artifact_directory(args.run_dir),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
