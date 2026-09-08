#!/usr/bin/env python3
"""Independently verify a topology-local TP2 island evidence bundle."""

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
VERIFICATION_SCHEMA = (
    "qwen38.topology-local-tp2-island-independent-verification.v1"
)
TERMINAL_MANIFEST_SCHEMA = (
    "qwen38.topology-local-tp2-island-terminal-manifest.v1"
)

GO = "GO_TOPOLOGY_LOCAL_TP2_ISLAND_MICROGATE"
INVALID = "INVALID_EVIDENCE"
CORRECTNESS_NO_GO = "NO_GO_CORRECTNESS_OR_LIFECYCLE"
MEMORY_NO_GO = "NO_GO_MEMORY"
MIGRATION_NO_GO = "NO_GO_MIGRATION_AMORTIZATION"
PERFORMANCE_NO_GO = "NO_GO_PERFORMANCE"
BLOCKED = "BLOCKED_ADMISSION"

REMOTE_RECEIPT_NAME = "remote_independent_verification.json"
LOCAL_RECEIPT_NAME = "local_independent_verification.json"
TERMINAL_MANIFEST_NAME = "manifest.json"

PRODUCER_FILES = frozenset({
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
})
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


def _write_json_atomic(path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_inventories():
    return {
        PRODUCER_FILES,
        PRODUCER_FILES | {REMOTE_RECEIPT_NAME},
        PRODUCER_FILES
        | {
            REMOTE_RECEIPT_NAME,
            LOCAL_RECEIPT_NAME,
            TERMINAL_MANIFEST_NAME,
        },
    }


def _verify_manifest(root):
    manifest = _load_json(root / "manifest.sha256")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.sha256"
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("artifacts"), dict)
        or actual not in _allowed_inventories()
        or set(manifest["artifacts"]) != actual
    ):
        raise ValueError("manifest artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(root / name) != expected
        ):
            raise ValueError("manifest artifact hash mismatch")


def _rewrite_manifest(root):
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.sha256"
    }
    _write_json_atomic(root / "manifest.sha256", {
        "schema": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def _nearest_rank(values, percentile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile input must not be empty")
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _identity_from_source(source):
    identity = {
        field: source.get(field)
        for field in _IDENTITY_FIELDS
    } if isinstance(source, dict) else {}
    revision = identity.get("source_revision")
    groups = identity.get("pair_groups")
    if (
        not isinstance(identity.get("attempt"), str)
        or not identity["attempt"]
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or identity.get("model_repository") != MODEL_REPOSITORY
        or identity.get("model_revision") != MODEL_REVISION
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


def _validate_rank_rows(rows):
    return (
        isinstance(rows, list)
        and len(rows) == 4
        and sorted(row.get("rank") for row in rows if isinstance(row, dict))
        == [0, 1, 2, 3]
    )


def _pair_identity(rank, pair_groups):
    for pair_id, pair in enumerate(pair_groups):
        if rank in pair:
            return pair_id, pair.index(rank)
    raise ValueError("rank is absent from pair groups")


def _valid_parameter_slices(payload, pair_groups):
    rows = payload.get("rank_parameter_evidence")
    if (
        payload.get("replica_digest_match") is not True
        or payload.get("checkpoint_reconstruction_match") is not True
        or not _validate_rank_rows(rows)
    ):
        return False
    by_logical_rank = {}
    checkpoint_digests = []
    for row in rows:
        rank = row["rank"]
        _, logical_rank = _pair_identity(rank, pair_groups)
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


def _valid_timing_rows(rows, identity):
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
        pair_id, logical_rank = _pair_identity(
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
            or row.get("candidate_global_collective_count") != 0
            or row.get("fallback_count") != 0
            or row.get("timed_allocation_count") != 0
            or not isinstance(row.get("arm_order"), list)
            or sorted(row["arm_order"]) != ["baseline", "candidate"]
        ):
            return False
        key = (row["active_tokens"], row["repetition"], rank)
        if key in observed:
            return False
        observed.add(key)
    return observed == expected


def _valid_migration_rows(rows, identity):
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
        pair_id, logical_rank = _pair_identity(
            rank, identity["pair_groups"]
        )
        if (
            not _same_identity(row, identity)
            or row.get("schema") != WORKER_SCHEMA
            or row.get("phase") != "measured"
            or row.get("repetition") not in range(15)
            or row.get("pair_id") != pair_id
            or row.get("logical_rank") != logical_rank
            or row.get("temporary_tensor_count") != 8
            or row.get(
                "temporary_live_tensor_count_after_release"
            ) != 0
        ):
            return False
        key = (row["repetition"], rank)
        if key in observed:
            return False
        observed.add(key)
    return observed == expected


def _critical_rows(rows, field, active_tokens=None):
    grouped = {}
    for row in rows:
        if active_tokens is not None and row["active_tokens"] != active_tokens:
            continue
        grouped.setdefault(row["repetition"], []).append(row[field])
    if sorted(grouped) != list(range(15)):
        raise ValueError("measured repetitions are incomplete")
    if any(len(values) != 4 for values in grouped.values()):
        raise ValueError("measured ranks are incomplete")
    return [max(grouped[index]) for index in range(15)]


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
            left / right - 1.0
            for left, right in zip(baseline, candidate)
        ),
        "improving_pair_count": sum(
            right < left for left, right in zip(baseline, candidate)
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
    summary = {
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
    for shape in shape_summaries:
        savings = shape["median_candidate_savings_ns"]
        summary["break_even_tokens_by_active_tokens"][
            str(shape["active_tokens"])
        ] = (
            None
            if savings <= 0
            else math.ceil(median_latency / savings)
        )
    return summary


def _expected_correctness_rows(rows):
    return [
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
        for row in rows
    ]


def _expected_component_rows(rows):
    return [
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
        for row in rows
    ]


def _evidence_is_valid(payloads):
    try:
        source = payloads["source"]
        identity = _identity_from_source(source)
        timing_rows = payloads["timing_rows"]
        migration_rows = payloads["migration"]["rows"]
        values = (
            payloads["model"],
            payloads["admission"],
            payloads["topology"],
            payloads["workload"],
            payloads["parameter_slices"],
            payloads["migration"],
            payloads["cleanup"],
            *timing_rows,
            *migration_rows,
            *payloads["memory_rows"],
            *payloads["lifecycle_rows"],
        )
        if not all(_same_identity(value, identity) for value in values):
            return False, identity
        model = payloads["model"]
        topology = payloads["topology"]
        workload = payloads["workload"]
        parameter_slices = payloads["parameter_slices"]
        cleanup = payloads["cleanup"]
        if (
            model.get("hidden_size") != 5120
            or model.get("layer_count") != 64
            or model.get("linear_attention_layer_count") != 48
            or model.get("full_attention_layer_count") != 16
            or model.get("dtype") != "bfloat16"
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
            or not _valid_parameter_slices(
                parameter_slices,
                identity["pair_groups"],
            )
            or not _valid_timing_rows(timing_rows, identity)
            or not _valid_migration_rows(migration_rows, identity)
            or not _validate_rank_rows(payloads["memory_rows"])
            or not _validate_rank_rows(payloads["lifecycle_rows"])
            or not _validate_rank_rows(
                payloads["admission"].get("rank_rows")
            )
            or not _validate_rank_rows(cleanup.get("rank_rows"))
            or cleanup.get("classification") != "CLEAN"
            or payloads["migration"].get("measured_row_count")
            != len(migration_rows)
        ):
            return False, identity
        if payloads["correctness_rows"] != _expected_correctness_rows(
            timing_rows
        ):
            return False, identity
        if payloads["component_rows"] != _expected_component_rows(
            timing_rows
        ):
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


def _correctness_passes(payloads):
    return (
        all(
            all(row.get(field) is True for field in _CORRECTNESS_FIELDS)
            for row in payloads["timing_rows"]
        )
        and all(
            row.get("temporary_released_before_timing") is True
            and row.get("temporary_allocated_bytes_after_release") == 0
            for row in payloads["migration"]["rows"]
        )
        and all(
            row.get("state_identity_match") is True
            and row.get("stale_generation_rejected") is True
            and row.get("different_request_rejected") is True
            and row.get("publish_after_success") is True
            and row.get("baseline_state_unchanged") is True
            and row.get("temporary_state_retired") is True
            and row.get("fallback_count") == 0
            for row in payloads["lifecycle_rows"]
        )
    )


def _memory_passes(rows):
    ceiling = 1920 * 1024 * 1024
    return all(
        row.get("projected_integrated_increment_bytes", ceiling + 1)
        <= ceiling
        and row.get("peak_allocated_ratio", 1.0) < 0.98
        and row.get("peak_allocated_bytes", 1)
        < row.get("physical_memory_bytes", 0)
        for row in rows
    )


def _performance(shape_summaries):
    by_tokens = {
        summary["active_tokens"]: summary
        for summary in shape_summaries
    }
    aggregate = math.sqrt(
        (1.0 + by_tokens[4]["speedup"])
        * (1.0 + by_tokens[8]["speedup"])
    ) - 1.0
    passed = (
        by_tokens[1]["speedup"] >= 0.05
        and aggregate >= 0.05
        and by_tokens[4]["speedup"] >= 0.0
        and by_tokens[8]["speedup"] >= 0.0
        and all(
            shape["p99_regression"] <= 0.03
            and shape["host_median_regression"] <= 0.10
            for shape in shape_summaries
        )
        and by_tokens[4]["improving_pair_count"] >= 11
        and by_tokens[8]["improving_pair_count"] >= 11
    )
    return passed, aggregate


def _reconstruct(payloads):
    valid, identity = _evidence_is_valid(payloads)
    shape_summaries = []
    migration_summary = {}
    aggregate = None
    timing_rows = payloads["timing_rows"]
    migration_rows = payloads["migration"].get("rows", [])
    if not valid:
        classification = INVALID
    elif payloads["admission"].get("classification") != "ADMITTED":
        classification = BLOCKED
    else:
        shape_summaries = [
            _shape_summary(timing_rows, active_tokens)
            for active_tokens in (1, 4, 8)
        ]
        migration_summary = _migration_summary(
            migration_rows, shape_summaries
        )
        performance_passes, aggregate = _performance(shape_summaries)
        migration_passes = all(
            value is not None and value <= 32
            for value in migration_summary[
                "break_even_tokens_by_active_tokens"
            ].values()
        )
        if not _correctness_passes(payloads):
            classification = CORRECTNESS_NO_GO
        elif not _memory_passes(payloads["memory_rows"]):
            classification = MEMORY_NO_GO
        elif not migration_passes:
            classification = MIGRATION_NO_GO
        elif not performance_passes:
            classification = PERFORMANCE_NO_GO
        else:
            classification = GO
    return {
        "schema": RESULT_SCHEMA,
        "attempt": identity.get("attempt"),
        "source_revision": identity.get("source_revision"),
        "classification": classification,
        "measurement_row_count": len(timing_rows),
        "migration_row_count": len(migration_rows),
        "shape_summaries": shape_summaries,
        "token_4_8_geometric_aggregate_speedup": aggregate,
        "migration_summary": migration_summary,
        "runtime_integration_authorized": classification == GO,
        "claim_boundary": "one-layer same-request Stage-0 microgate only",
    }


def _load_payloads(root):
    return {
        "admission": _load_json(root / "admission.json"),
        "topology": _load_json(root / "topology.json"),
        "model": _load_json(root / "model_identity.json"),
        "source": _load_json(root / "source_identity.json"),
        "workload": _load_json(root / "workload_manifest.json"),
        "parameter_slices": _load_json(
            root / "parameter_slice_manifest.json"
        ),
        "migration": _load_json(
            root / "state_migration_manifest.json"
        ),
        "correctness_rows": _load_jsonl(
            root / "correctness_rows.jsonl"
        ),
        "timing_rows": _load_jsonl(
            root / "paired_timing_rows.jsonl"
        ),
        "component_rows": _load_jsonl(
            root / "component_diagnostic_rows.jsonl"
        ),
        "memory_rows": _load_jsonl(root / "memory_rows.jsonl"),
        "lifecycle_rows": _load_jsonl(
            root / "lifecycle_rows.jsonl"
        ),
        "cleanup": _load_json(root / "cleanup.json"),
    }


def _receipt(result):
    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "PASS",
        "attempt": result["attempt"],
        "source_revision": result["source_revision"],
        "classification": result["classification"],
        "producer_classification": result["classification"],
        "reconstructed_classification": result["classification"],
        "measurement_row_count": result["measurement_row_count"],
        "migration_row_count": result["migration_row_count"],
        "claim_boundary": result["claim_boundary"],
    }


def _validate_receipt(path, expected):
    payload = _load_json(path)
    if payload != expected:
        raise ValueError("independent verification receipt mismatch")


def _terminal_manifest_payload(root, result):
    return {
        "schema": TERMINAL_MANIFEST_SCHEMA,
        "attempt": result["attempt"],
        "source_revision": result["source_revision"],
        "classification": result["classification"],
        "remote_receipt_sha256": _sha256(root / REMOTE_RECEIPT_NAME),
        "local_receipt_sha256": _sha256(root / LOCAL_RECEIPT_NAME),
    }


def verify_bundle(
    root,
    *,
    receipt_name=None,
    seal_terminal=False,
    check_only=False,
):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("bundle root must be an existing directory")
    if receipt_name not in {
        None,
        REMOTE_RECEIPT_NAME,
        LOCAL_RECEIPT_NAME,
    }:
        raise ValueError("independent verification receipt name is invalid")
    if check_only and (receipt_name is not None or seal_terminal):
        raise ValueError("check-only verification must be non-mutating")
    if seal_terminal and receipt_name != LOCAL_RECEIPT_NAME:
        raise ValueError("terminal sealing requires the local receipt")
    if (
        (root / TERMINAL_MANIFEST_NAME).is_file()
        and not check_only
        and receipt_name is not None
    ):
        raise ValueError("sealed terminal bundle is read-only")

    _verify_manifest(root)
    payloads = _load_payloads(root)
    reconstructed = _reconstruct(payloads)
    producer = _load_json(root / "producer_result.json")
    if producer != reconstructed:
        raise ValueError("producer classification or summary mismatch")
    if payloads["migration"].get("summary") != reconstructed[
        "migration_summary"
    ]:
        raise ValueError("migration summary mismatch")
    receipt = _receipt(reconstructed)

    remote_path = root / REMOTE_RECEIPT_NAME
    local_path = root / LOCAL_RECEIPT_NAME
    terminal_path = root / TERMINAL_MANIFEST_NAME
    if remote_path.is_file():
        _validate_receipt(remote_path, receipt)
    if local_path.is_file():
        _validate_receipt(local_path, receipt)
    if terminal_path.is_file():
        if not remote_path.is_file() or not local_path.is_file():
            raise ValueError("terminal manifest receipts are incomplete")
        if _load_json(terminal_path) != _terminal_manifest_payload(
            root, reconstructed
        ):
            raise ValueError("terminal manifest mismatch")
    if seal_terminal and not remote_path.is_file():
        raise ValueError("remote independent verification is missing")

    if receipt_name is not None:
        _write_json_atomic(root / receipt_name, receipt)
    if seal_terminal:
        _write_json_atomic(
            terminal_path,
            _terminal_manifest_payload(root, reconstructed),
        )
    if receipt_name is not None or seal_terminal:
        _rewrite_manifest(root)
    return receipt


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--receipt-name",
        choices=(REMOTE_RECEIPT_NAME, LOCAL_RECEIPT_NAME),
        default=REMOTE_RECEIPT_NAME,
    )
    parser.add_argument("--seal-terminal", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    result = verify_bundle(
        args.root,
        receipt_name=None if args.check_only else args.receipt_name,
        seal_terminal=args.seal_terminal,
        check_only=args.check_only,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
