#!/usr/bin/env python3
"""Assemble and classify topology-local TP2 island microgate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
WORKER_SCHEMA = "qwen38.topology-local-tp2-island-worker.v1"
MANIFEST_SCHEMA = "qwen38.topology-local-tp2-island-manifest.v1"
RESULT_SCHEMA = "qwen38.topology-local-tp2-island-result.v1"

GO = "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
INVALID = "INVALID_EVIDENCE"
CORRECTNESS_NO_GO = "NO_GO_CORRECTNESS_OR_LIFECYCLE"
MEMORY_NO_GO = "NO_GO_MEMORY"
MIGRATION_NO_GO = "NO_GO_MIGRATION_AMORTIZATION"
PERFORMANCE_NO_GO = "NO_GO_PERFORMANCE"
BLOCKED = "BLOCKED_ADMISSION"

PRODUCER_ARTIFACTS = (
    "admission.json",
    "topology.json",
    "model_identity.json",
    "source_identity.json",
    "workload_manifest.json",
    "parameter_slice_manifest.json",
    "state_migration_manifest.json",
    "correctness_rows.jsonl",
    "paired_timing_rows.jsonl",
    "component_diagnostic_rows.jsonl",
    "memory_rows.jsonl",
    "lifecycle_rows.jsonl",
    "cleanup.json",
    "producer_result.json",
    "report.md",
    "manifest.sha256",
)

_IDENTITY_FIELDS = (
    "attempt",
    "source_revision",
    "model_repository",
    "model_revision",
    "pair_groups",
)
_CORRECTNESS_FIELDS = (
    "output_within_tolerance",
    "convolution_within_tolerance",
    "recurrent_within_tolerance",
    "pair_replicas_within_tolerance",
    "greedy_argmax_equal",
    "finite",
)


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"JSON number must be finite: {value}")


def _load_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_duplicate_keys,
                parse_constant=_nonfinite,
            )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error


def _load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(
                    line,
                    object_pairs_hook=_duplicate_keys,
                    parse_constant=_nonfinite,
                ))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    return rows


def _require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric evidence must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite(child)


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json(path, payload):
    _require_finite(payload)

    def write(handle):
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")

    _atomic_write(path, write)


def _write_jsonl(path, rows):
    _require_finite(rows)

    def write(handle):
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
            handle.write("\n")

    _atomic_write(path, write)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(root):
    root = Path(root)
    artifacts = {
        name: _sha256(root / name)
        for name in PRODUCER_ARTIFACTS
        if name != "manifest.sha256"
    }
    _write_json(root / "manifest.sha256", {
        "schema": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _nearest_rank(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile input must not be empty")
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _identity_from_source(source_identity):
    if not isinstance(source_identity, dict):
        raise ValueError("source identity must be an object")
    identity = {
        field: source_identity.get(field)
        for field in _IDENTITY_FIELDS
    }
    revision = identity["source_revision"]
    groups = identity["pair_groups"]
    if (
        not isinstance(identity["attempt"], str)
        or not identity["attempt"]
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or identity["model_repository"] != MODEL_REPOSITORY
        or identity["model_revision"] != MODEL_REVISION
        or not isinstance(groups, list)
        or len(groups) != 2
        or any(not isinstance(group, list) or len(group) != 2 for group in groups)
        or sorted(rank for group in groups for rank in group) != [0, 1, 2, 3]
    ):
        raise ValueError("source identity is invalid")
    return identity


def _same_identity(value, identity):
    return (
        isinstance(value, dict)
        and all(value.get(field) == expected for field, expected in identity.items())
    )


def _validate_rank_rows(rows, expected_count):
    return (
        isinstance(rows, list)
        and len(rows) == expected_count
        and sorted(row.get("rank") for row in rows if isinstance(row, dict))
        == list(range(expected_count))
    )


def _expected_pair_identity(rank, pair_groups):
    for pair_id, pair in enumerate(pair_groups):
        if rank in pair:
            return pair_id, pair.index(rank)
    raise ValueError("rank is absent from pair groups")


def _validate_parameter_slices(payload, pair_groups):
    rows = payload.get("rank_parameter_evidence")
    if (
        payload.get("replica_digest_match") is not True
        or payload.get("checkpoint_reconstruction_match") is not True
        or not _validate_rank_rows(rows, 4)
    ):
        return False
    by_logical_rank = {}
    checkpoint_digests = []
    for row in rows:
        rank = row["rank"]
        _, logical_rank = _expected_pair_identity(rank, pair_groups)
        candidate = row.get("parameter_digests")
        checkpoint = row.get("checkpoint_full_parameter_digests")
        reconstructed = row.get("reconstructed_full_parameter_digests")
        if (
            row.get("logical_rank") != logical_rank
            or row.get("checkpoint_reconstruction_match") is not True
            or not isinstance(candidate, dict)
            or not candidate
            or not isinstance(checkpoint, dict)
            or not checkpoint
            or checkpoint != reconstructed
        ):
            return False
        by_logical_rank.setdefault(logical_rank, []).append(candidate)
        checkpoint_digests.append(checkpoint)
    return (
        all(
            len(values) == 2 and values[0] == values[1]
            for values in by_logical_rank.values()
        )
        and len(by_logical_rank) == 2
        and all(
            value == checkpoint_digests[0]
            for value in checkpoint_digests[1:]
        )
    )


def _validate_timing_rows(rows, identity):
    if not isinstance(rows, list) or len(rows) != 180:
        return False
    expected = {
        (active_tokens, repetition, rank)
        for active_tokens in (1, 4, 8)
        for repetition in range(15)
        for rank in range(4)
    }
    observed = set()
    for row in rows:
        rank = row.get("rank") if isinstance(row, dict) else None
        if rank not in range(4):
            return False
        pair_id, logical_rank = _expected_pair_identity(
            rank, identity["pair_groups"]
        )
        if (
            not _same_identity(row, identity)
            or row.get("schema") != WORKER_SCHEMA
            or row.get("phase") != "measured"
            or row.get("active_tokens") not in (1, 4, 8)
            or row.get("repetition") not in range(15)
            or row.get("pair_id") != pair_id
            or row.get("logical_rank") != logical_rank
        ):
            return False
        key = (row["active_tokens"], row["repetition"], row["rank"])
        if key in observed:
            return False
        observed.add(key)
    return observed == expected


def _validate_migration_rows(rows, identity):
    if not isinstance(rows, list) or len(rows) != 60:
        return False
    expected = {
        (repetition, rank)
        for repetition in range(15)
        for rank in range(4)
    }
    observed = set()
    for row in rows:
        rank = row.get("rank") if isinstance(row, dict) else None
        if rank not in range(4):
            return False
        pair_id, logical_rank = _expected_pair_identity(
            rank, identity["pair_groups"]
        )
        if (
            not _same_identity(row, identity)
            or row.get("schema") != WORKER_SCHEMA
            or row.get("phase") != "measured"
            or row.get("repetition") not in range(15)
            or row.get("pair_id") != pair_id
            or row.get("logical_rank") != logical_rank
        ):
            return False
        key = (row["repetition"], row["rank"])
        if key in observed:
            return False
        observed.add(key)
    return observed == expected


def _critical_rows(rows, value_field, active_tokens=None):
    grouped = {}
    for row in rows:
        if active_tokens is not None and row["active_tokens"] != active_tokens:
            continue
        repetition = row["repetition"]
        grouped.setdefault(repetition, []).append(row[value_field])
    if sorted(grouped) != list(range(15)):
        raise ValueError("measured repetitions are incomplete")
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError("measured ranks are incomplete")
    return [max(grouped[repetition]) for repetition in range(15)]


def _shape_summary(rows, active_tokens):
    baseline = _critical_rows(rows, "baseline_cuda_ns", active_tokens)
    candidate = _critical_rows(rows, "candidate_cuda_ns", active_tokens)
    baseline_host = _critical_rows(
        rows, "baseline_host_submission_ns", active_tokens
    )
    candidate_host = _critical_rows(
        rows, "candidate_host_submission_ns", active_tokens
    )
    baseline_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    baseline_p99 = _nearest_rank(baseline, 0.99)
    candidate_p99 = _nearest_rank(candidate, 0.99)
    host_baseline_median = statistics.median(baseline_host)
    host_candidate_median = statistics.median(candidate_host)
    return {
        "active_tokens": active_tokens,
        "baseline_median_cuda_ns": baseline_median,
        "candidate_median_cuda_ns": candidate_median,
        "baseline_p90_cuda_ns": _nearest_rank(baseline, 0.90),
        "candidate_p90_cuda_ns": _nearest_rank(candidate, 0.90),
        "baseline_p95_cuda_ns": _nearest_rank(baseline, 0.95),
        "candidate_p95_cuda_ns": _nearest_rank(candidate, 0.95),
        "baseline_p99_cuda_ns": baseline_p99,
        "candidate_p99_cuda_ns": candidate_p99,
        "speedup": baseline_median / candidate_median - 1.0,
        "p99_regression": candidate_p99 / baseline_p99 - 1.0,
        "median_paired_speedup": statistics.median(
            baseline_value / candidate_value - 1.0
            for baseline_value, candidate_value in zip(baseline, candidate)
        ),
        "improving_pair_count": sum(
            candidate_value < baseline_value
            for baseline_value, candidate_value in zip(baseline, candidate)
        ),
        "host_baseline_median_ns": host_baseline_median,
        "host_candidate_median_ns": host_candidate_median,
        "host_median_regression": (
            host_candidate_median / host_baseline_median - 1.0
        ),
        "median_candidate_savings_ns": (
            baseline_median - candidate_median
        ),
    }


def _migration_summary(rows, shape_summaries):
    critical = _critical_rows(rows, "latency_ns")
    median_latency = statistics.median(critical)
    result = {
        "median_latency_ns": median_latency,
        "p95_latency_ns": _nearest_rank(critical, 0.95),
        "p99_latency_ns": _nearest_rank(critical, 0.99),
        "source_bytes_per_rank": sorted({
            row["source_bytes"] for row in rows
        }),
        "transferred_bytes_per_rank": sorted({
            row["transferred_bytes"] for row in rows
        }),
        "retained_bytes_per_rank": sorted({
            row["retained_bytes"] for row in rows
        }),
        "break_even_tokens_by_active_tokens": {},
    }
    for summary in shape_summaries:
        savings = summary["median_candidate_savings_ns"]
        break_even = (
            None
            if savings <= 0
            else math.ceil(median_latency / savings)
        )
        result["break_even_tokens_by_active_tokens"][
            str(summary["active_tokens"])
        ] = break_even
    return result


def _evidence_is_valid(
    *,
    source_identity,
    model_identity,
    admission,
    topology,
    workload,
    parameter_slices,
    timing_rows,
    migration_rows,
    memory_rows,
    lifecycle_rows,
    cleanup,
):
    try:
        identity = _identity_from_source(source_identity)
        all_values = (
            model_identity,
            admission,
            topology,
            workload,
            parameter_slices,
            cleanup,
            *timing_rows,
            *migration_rows,
            *memory_rows,
            *lifecycle_rows,
        )
        if not all(_same_identity(value, identity) for value in all_values):
            return False, identity
        if (
            model_identity.get("hidden_size") != 5120
            or model_identity.get("layer_count") != 64
            or model_identity.get("linear_attention_layer_count") != 48
            or model_identity.get("full_attention_layer_count") != 16
            or model_identity.get("dtype") != "bfloat16"
            or topology.get("selection_frozen") is not True
            or topology.get("selected_pair_groups") != identity["pair_groups"]
            or workload.get("active_token_groups") != [1, 4, 8]
            or workload.get("warmup_pairs_per_shape") != 2
            or workload.get("measured_pairs_per_shape") != 15
            or workload.get("migration_warmups") != 2
            or workload.get("migration_measurements") != 15
            or parameter_slices.get("layer_index") != 0
            or parameter_slices.get("linear_attention_only") is not True
            or parameter_slices.get("full_attention_parameters_changed") is not False
            or parameter_slices.get("mlp_parameters_changed") is not False
            or not _validate_parameter_slices(
                parameter_slices,
                identity["pair_groups"],
            )
            or not _validate_timing_rows(timing_rows, identity)
            or not _validate_migration_rows(migration_rows, identity)
            or not _validate_rank_rows(memory_rows, 4)
            or not _validate_rank_rows(lifecycle_rows, 4)
            or not _validate_rank_rows(admission.get("rank_rows"), 4)
            or not _validate_rank_rows(cleanup.get("rank_rows"), 4)
        ):
            return False, identity
        for row in timing_rows:
            if (
                row.get("candidate_global_collective_count") != 0
                or row.get("fallback_count") != 0
                or row.get("timed_allocation_count") != 0
                or not isinstance(row.get("arm_order"), list)
                or sorted(row["arm_order"]) != ["baseline", "candidate"]
            ):
                return False, identity
        if cleanup.get("classification") != "CLEAN":
            return False, identity
        for row in cleanup["rank_rows"]:
            if (
                row.get("candidate_state_unpublished") is not True
                or row.get("owned_children_remaining") != []
                or row.get("task_files_outside_attempt_root") != []
            ):
                return False, identity
        return True, identity
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False, {}


def _correctness_passes(timing_rows, migration_rows, lifecycle_rows):
    return (
        all(
            all(row.get(field) is True for field in _CORRECTNESS_FIELDS)
            for row in timing_rows
        )
        and all(
            row.get("temporary_released_before_timing") is True
            and row.get("temporary_allocated_bytes_after_release") == 0
            for row in migration_rows
        )
        and all(
            row.get("state_identity_match") is True
            and row.get("stale_generation_rejected") is True
            and row.get("different_request_rejected") is True
            and row.get("publish_after_success") is True
            and row.get("baseline_state_unchanged") is True
            and row.get("temporary_state_retired") is True
            and row.get("fallback_count") == 0
            for row in lifecycle_rows
        )
    )


def _memory_passes(memory_rows):
    ceiling = 1920 * 1024 * 1024
    return all(
        row.get("projected_integrated_increment_bytes", ceiling + 1)
        <= ceiling
        and row.get("peak_allocated_ratio", 1.0) < 0.98
        and row.get("peak_allocated_bytes", 1)
        < row.get("physical_memory_bytes", 0)
        for row in memory_rows
    )


def _performance_passes(shape_summaries):
    by_tokens = {
        summary["active_tokens"]: summary
        for summary in shape_summaries
    }
    aggregate_4_8 = math.sqrt(
        (1.0 + by_tokens[4]["speedup"])
        * (1.0 + by_tokens[8]["speedup"])
    ) - 1.0
    return (
        by_tokens[1]["speedup"] >= 0.05
        and aggregate_4_8 >= 0.05
        and by_tokens[4]["speedup"] >= 0.0
        and by_tokens[8]["speedup"] >= 0.0
        and all(
            summary["p99_regression"] <= 0.03
            and summary["host_median_regression"] <= 0.10
            for summary in shape_summaries
        )
        and by_tokens[4]["improving_pair_count"] >= 11
        and by_tokens[8]["improving_pair_count"] >= 11
    ), aggregate_4_8


def _render_report(result):
    lines = [
        "# Qwen3.8 topology-local TP2 island Stage-0 result",
        "",
        f"- Classification: `{result['classification']}`",
        f"- Attempt: `{result.get('attempt', 'unknown')}`",
        f"- Measured timing rows: {result['measurement_row_count']}",
        f"- Measured migration rows: {result['migration_row_count']}",
        "",
        "This is a one-layer same-request microgate. It is not evidence of "
        "whole-model TPOT, TTFT, QPS, or production benefit.",
    ]
    return "\n".join(lines) + "\n"


def assemble_bundle(
    output_root,
    *,
    source_identity,
    model_identity,
    admission,
    topology,
    workload,
    parameter_slices,
    timing_rows,
    migration_rows,
    memory_rows,
    lifecycle_rows,
    cleanup,
):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("bundle output directory must be empty")
    _require_finite({
        "source_identity": source_identity,
        "model_identity": model_identity,
        "admission": admission,
        "topology": topology,
        "workload": workload,
        "parameter_slices": parameter_slices,
        "timing_rows": timing_rows,
        "migration_rows": migration_rows,
        "memory_rows": memory_rows,
        "lifecycle_rows": lifecycle_rows,
        "cleanup": cleanup,
    })

    valid, identity = _evidence_is_valid(
        source_identity=source_identity,
        model_identity=model_identity,
        admission=admission,
        topology=topology,
        workload=workload,
        parameter_slices=parameter_slices,
        timing_rows=timing_rows,
        migration_rows=migration_rows,
        memory_rows=memory_rows,
        lifecycle_rows=lifecycle_rows,
        cleanup=cleanup,
    )
    shape_summaries = []
    migration_summary = {}
    aggregate_4_8 = None
    if not valid:
        classification = INVALID
    elif admission.get("classification") != "ADMITTED":
        classification = BLOCKED
    else:
        shape_summaries = [
            _shape_summary(timing_rows, active_tokens)
            for active_tokens in (1, 4, 8)
        ]
        migration_summary = _migration_summary(
            migration_rows, shape_summaries
        )
        performance_passes, aggregate_4_8 = _performance_passes(
            shape_summaries
        )
        break_evens = migration_summary[
            "break_even_tokens_by_active_tokens"
        ].values()
        migration_passes = all(
            value is not None and value <= 32 for value in break_evens
        )
        if not _correctness_passes(
            timing_rows, migration_rows, lifecycle_rows
        ):
            classification = CORRECTNESS_NO_GO
        elif not _memory_passes(memory_rows):
            classification = MEMORY_NO_GO
        elif not migration_passes:
            classification = MIGRATION_NO_GO
        elif not performance_passes:
            classification = PERFORMANCE_NO_GO
        else:
            classification = GO

    correctness_rows = [
        {
            key: value
            for key, value in row.items()
            if (
                key in _IDENTITY_FIELDS
                or key in _CORRECTNESS_FIELDS
                or key.endswith("_max_abs_error")
                or key.endswith("_max_rel_error")
                or key in (
                    "schema",
                    "active_tokens",
                    "phase",
                    "repetition",
                    "rank",
                    "pair_id",
                    "logical_rank",
                    "parameter_digests",
                )
            )
        }
        for row in timing_rows
    ]
    component_rows = [
        {
            **{
                field: row[field]
                for field in _IDENTITY_FIELDS
                if field in row
            },
            "schema": row.get("schema"),
            "active_tokens": row.get("active_tokens"),
            "repetition": row.get("repetition"),
            "rank": row.get("rank"),
            "candidate_component_diagnostics": row.get(
                "candidate_component_diagnostics", {}
            ),
        }
        for row in timing_rows
    ]
    result = {
        "schema": RESULT_SCHEMA,
        "attempt": identity.get("attempt"),
        "source_revision": identity.get("source_revision"),
        "classification": classification,
        "measurement_row_count": len(timing_rows),
        "migration_row_count": len(migration_rows),
        "shape_summaries": shape_summaries,
        "token_4_8_geometric_aggregate_speedup": aggregate_4_8,
        "migration_summary": migration_summary,
        "runtime_integration_authorized": classification == GO,
        "claim_boundary": "one-layer same-request Stage-0 microgate only",
    }
    state_migration_manifest = {
        **identity,
        "schema": "qwen38.topology-local-tp2-island-migration.v1",
        "measured_row_count": len(migration_rows),
        "summary": migration_summary,
        "rows": migration_rows,
    }

    _write_json(root / "admission.json", admission)
    _write_json(root / "topology.json", topology)
    _write_json(root / "model_identity.json", model_identity)
    _write_json(root / "source_identity.json", source_identity)
    _write_json(root / "workload_manifest.json", workload)
    _write_json(root / "parameter_slice_manifest.json", parameter_slices)
    _write_json(
        root / "state_migration_manifest.json",
        state_migration_manifest,
    )
    _write_jsonl(root / "correctness_rows.jsonl", correctness_rows)
    _write_jsonl(root / "paired_timing_rows.jsonl", timing_rows)
    _write_jsonl(
        root / "component_diagnostic_rows.jsonl", component_rows
    )
    _write_jsonl(root / "memory_rows.jsonl", memory_rows)
    _write_jsonl(root / "lifecycle_rows.jsonl", lifecycle_rows)
    _write_json(root / "cleanup.json", cleanup)
    _write_json(root / "producer_result.json", result)
    _atomic_write(
        root / "report.md",
        lambda handle: handle.write(_render_report(result)),
    )
    _write_manifest(root)
    return {
        **result,
        "bundle_root": str(root),
        "artifact_count": len(PRODUCER_ARTIFACTS),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args(argv)
    attempt_root = args.attempt_root.resolve()
    raw = attempt_root / "raw"
    bundle_root = (
        args.bundle_root.resolve()
        if args.bundle_root is not None
        else attempt_root / "final_bundle"
    )
    timing_rows = _load_jsonl(raw / "measurement_rows.jsonl")
    migration_rows = _load_jsonl(raw / "migration_rows.jsonl")
    memory_rows = _load_jsonl(raw / "memory_rows.jsonl")
    lifecycle_rows = _load_jsonl(raw / "lifecycle_rows.jsonl")
    cleanup = _load_json(raw / "cleanup.json")
    result = assemble_bundle(
        bundle_root,
        source_identity=_load_json(
            attempt_root / "controller/source_identity.json"
        ),
        model_identity=_load_json(raw / "model_identity.json"),
        admission=_load_json(
            attempt_root / "controller/launch_admission.json"
        ),
        topology=_load_json(raw / "topology.json"),
        workload=_load_json(raw / "workload_manifest.json"),
        parameter_slices=_load_json(
            raw / "parameter_slice_manifest.json"
        ),
        timing_rows=timing_rows,
        migration_rows=migration_rows,
        memory_rows=memory_rows,
        lifecycle_rows=lifecycle_rows,
        cleanup=cleanup,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
