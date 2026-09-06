#!/usr/bin/env python3
"""Independently reconstruct a TP4 decode replay qualification verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import tp4_decode_replay_contract as contract


MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MAX_GPU_MEMORY_USED_MIB = 1024
SHARED_CAPACITY_MAX_GPU_MEMORY_USED_MIB = 20_480
MAX_GPU_UTILIZATION_PERCENT = 5
MANIFEST_SCHEMA = "tinyllmforge.tp4-decode-replay-manifest.v1"
SUMMARY_SCHEMA = "tinyllmforge.tp4-decode-replay-summary.v1"
CLASSIFICATION_SCHEMA = (
    "tinyllmforge.tp4-decode-replay-classification.v1"
)
PRODUCER_FILES = frozenset({
    "source_manifest.json",
    "source.patch",
    "environment.json",
    "gpu_inventory.json",
    "workload_profile.json",
    "process_receipts.json",
    "rank_environment.jsonl",
    "rank_dispatch_events.jsonl",
    "rank_collective_events.jsonl",
    "rank_lifecycle_rows.jsonl",
    "request_rows.jsonl",
    "performance_rows.jsonl",
    "memory_rows.jsonl",
    "correctness_rows.jsonl",
    "capture_cost_rows.jsonl",
    "source_identity.json",
    "launch_admission.json",
    "cleanup.json",
    "summary.json",
    "producer_classification.json",
    "report.md",
})
JSONL_FILES = frozenset(
    name for name in PRODUCER_FILES if name.endswith(".jsonl")
)
EVIDENCE_FILES = {
    "performance_rows": "performance_rows.jsonl",
    "correctness_rows": "correctness_rows.jsonl",
    "rank_dispatch_rows": "rank_dispatch_events.jsonl",
    "rank_collective_rows": "rank_collective_events.jsonl",
    "rank_lifecycle_rows": "rank_lifecycle_rows.jsonl",
    "memory_rows": "memory_rows.jsonl",
    "capture_cost_rows": "capture_cost_rows.jsonl",
}


def _duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"JSON number must be finite: {value}")


def _require_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric evidence must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_finite(child)


def _load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_duplicate_keys,
                parse_constant=_nonfinite,
            )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSON: {path.name}") from error
    _require_finite(value)
    return value


def _load_jsonl(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(
            f"JSONL is empty or lacks terminal newline: {path.name}"
        )
    rows = []
    for line_number, line in enumerate(
        payload.decode("utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(
                f"blank JSONL row at {path.name}:{line_number}"
            )
        try:
            row = json.loads(
                line,
                object_pairs_hook=_duplicate_keys,
                parse_constant=_nonfinite,
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL at {path.name}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row must be an object: {path.name}"
            )
        _require_finite(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path.name}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hex(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _reconstruct_dynamic_pool_index_mechanism(
    *,
    rank_dispatch_rows: list[dict],
    capture_cost_rows: list[dict],
    performance_rows: list[dict],
) -> dict:
    failures = []
    graph_rows = [
        row
        for row in rank_dispatch_rows
        if row.get("arm") == "graph"
        and row.get("phase") != "warmup"
        and row.get("dispatch") == "graph"
    ]
    dispatch_groups = {}
    programs_by_cohort = {}
    invocations_by_cohort = {}
    cross_lease_replay_count = 0
    manifest_validation_count = 0
    for row in graph_rows:
        group_key = (
            row.get("case_id"),
            row.get("phase"),
            row.get("step_index"),
        )
        dispatch_groups.setdefault(group_key, []).append(row)
        cohort_key = (row.get("case_id"), row.get("rank"))
        program_key = row.get("graph_program_key_sha256")
        invocation_identity = row.get(
            "graph_invocation_identity_sha256"
        )
        programs_by_cohort.setdefault(cohort_key, set())
        invocations_by_cohort.setdefault(cohort_key, set())
        if _is_hex(program_key, 64):
            programs_by_cohort[cohort_key].add(program_key)
        if _is_hex(invocation_identity, 64):
            invocations_by_cohort[cohort_key].add(
                invocation_identity
            )
    for group in dispatch_groups.values():
        ordered = sorted(group, key=lambda row: row.get("rank", -1))
        if (
            len(ordered) != len(contract.RANKS)
            or tuple(row.get("rank") for row in ordered)
            != contract.RANKS
        ):
            failures.append("dynamic_pool_index_rank_inventory_incomplete")
            continue
        for row in ordered:
            if (
                not _is_hex(row.get("graph_program_key_sha256"), 64)
                or not _is_hex(
                    row.get("graph_invocation_identity_sha256"),
                    64,
                )
                or not _is_hex(row.get("lease_manifest_sha256"), 64)
                or not isinstance(row.get("cross_lease_replay"), bool)
            ):
                failures.append("dynamic_pool_index_evidence_invalid")
                break
        agreement_fields = (
            "graph_program_key_sha256",
            "graph_invocation_identity_sha256",
            "lease_manifest_sha256",
            "cross_lease_replay",
        )
        reference = tuple(
            ordered[0].get(field) for field in agreement_fields
        )
        if any(
            tuple(row.get(field) for field in agreement_fields)
            != reference
            for row in ordered[1:]
        ):
            for field, failure in (
                (
                    "graph_program_key_sha256",
                    "graph_program_key_disagreement",
                ),
                (
                    "graph_invocation_identity_sha256",
                    "graph_invocation_identity_disagreement",
                ),
                (
                    "lease_manifest_sha256",
                    "lease_manifest_disagreement",
                ),
                ("cross_lease_replay", "cross_lease_disagreement"),
            ):
                values = [row.get(field) for row in ordered]
                if any(value != values[0] for value in values[1:]):
                    failures.append(failure)
        else:
            manifest_validation_count += len(ordered)
            if reference[-1] is True:
                cross_lease_replay_count += len(ordered)

    program_counts = [
        len(values) for values in programs_by_cohort.values()
    ]
    invocation_counts = [
        len(values) for values in invocations_by_cohort.values()
    ]
    unique_program_key_count = max(program_counts, default=0)
    unique_invocation_identity_count = min(
        invocation_counts,
        default=0,
    )
    if program_counts and unique_program_key_count != 1:
        failures.append("graph_program_key_not_stable")

    dispatch_programs = {
        cohort: next(iter(values))
        for cohort, values in programs_by_cohort.items()
        if len(values) == 1
    }
    capture_groups = {}
    capture_duration_by_rank = {
        rank: 0 for rank in contract.RANKS
    }
    for row in capture_cost_rows:
        program_key = row.get("graph_program_key_sha256")
        cohort = (row.get("case_id"), row.get("rank"))
        if graph_rows and (
            not _is_hex(program_key, 64)
            or dispatch_programs.get(cohort) != program_key
        ):
            failures.append("capture_program_key_mismatch")
            continue
        capture_key = (*cohort, program_key)
        capture_groups.setdefault(capture_key, []).append(row)
        rank = row.get("rank")
        duration_ns = row.get("capture_duration_ns")
        if (
            rank not in contract.RANKS
            or isinstance(duration_ns, bool)
            or not isinstance(duration_ns, (int, float))
            or not math.isfinite(float(duration_ns))
            or duration_ns < 0
        ):
            failures.append("capture_cost_evidence_invalid")
            continue
        capture_duration_by_rank[rank] += int(duration_ns)
    if graph_rows and any(
        len(rows) != 1 for rows in capture_groups.values()
    ):
        failures.append("duplicate_program_capture_cost")
    expected_capture_keys = set(dispatch_programs)
    observed_capture_keys = {
        (case_id, rank)
        for case_id, rank, _program_key in capture_groups
    }
    if graph_rows and observed_capture_keys != expected_capture_keys:
        failures.append("program_capture_cost_incomplete")

    capture_duration_ns = max(capture_duration_by_rank.values())
    eager_tpot = [
        float(row["median_tpot_ms"])
        for row in performance_rows
        if row.get("arm") == "eager"
    ]
    graph_tpot = [
        float(row["median_tpot_ms"])
        for row in performance_rows
        if row.get("arm") == "graph"
    ]
    saved_ms_per_token = (
        max(
            0.0,
            statistics.median(eager_tpot)
            - statistics.median(graph_tpot),
        )
        if eager_tpot and graph_tpot
        else 0.0
    )
    return {
        "failures": sorted(set(failures)),
        "unique_program_key_count": unique_program_key_count,
        "unique_invocation_identity_count": (
            unique_invocation_identity_count
        ),
        "cross_lease_replay_count": cross_lease_replay_count,
        "manifest_validation_count": manifest_validation_count,
        "capture_duration_ns": capture_duration_ns,
        "capture_amortization_tokens": (
            None
            if saved_ms_per_token <= 0.0
            else (capture_duration_ns / 1_000_000.0)
            / saved_ms_per_token
        ),
    }


def _classify_evidence(**evidence: list[dict]) -> dict:
    classification = contract.classify(**evidence)
    if classification["classification"] == "INCOMPLETE":
        return classification | {
            "unique_program_key_count": 0,
            "unique_invocation_identity_count": 0,
            "cross_lease_replay_count": 0,
            "manifest_validation_count": 0,
        }
    mechanism = _reconstruct_dynamic_pool_index_mechanism(
        rank_dispatch_rows=evidence["rank_dispatch_rows"],
        capture_cost_rows=evidence["capture_cost_rows"],
        performance_rows=evidence["performance_rows"],
    )
    result = classification | {
        field: mechanism[field]
        for field in (
            "unique_program_key_count",
            "unique_invocation_identity_count",
            "cross_lease_replay_count",
            "manifest_validation_count",
            "capture_duration_ns",
            "capture_amortization_tokens",
        )
    }
    if mechanism["failures"]:
        return result | {
            "classification": "NO_GO_CORRECTNESS_OR_LIFECYCLE",
            "failed_gates": mechanism["failures"],
        }
    if (
        mechanism["unique_invocation_identity_count"] < 2
        or mechanism["cross_lease_replay_count"] == 0
    ):
        return result | {
            "classification": "NO_GO_MECHANISM_NOT_EXERCISED",
            "failed_gates": ["cross_lease_replay"],
        }
    return result


def _verify_manifest(root: Path) -> None:
    manifest = _load_json(root / "manifest.json")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or not isinstance(manifest.get("artifacts"), dict)
        or actual != PRODUCER_FILES
        or set(manifest["artifacts"]) != PRODUCER_FILES
    ):
        raise ValueError("manifest artifact inventory mismatch")
    for name, expected in manifest["artifacts"].items():
        if (
            not _is_hex(expected, 64)
            or _sha256(root / name) != expected
        ):
            raise ValueError(f"manifest artifact hash mismatch: {name}")


def _validate_source(source: object) -> dict:
    if (
        not isinstance(source, dict)
        or source.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-source.v1"
        or not isinstance(source.get("run_tag"), str)
        or not source["run_tag"]
        or not _is_hex(source.get("source_revision"), 40)
        or not _is_hex(source.get("source_tree_sha256"), 64)
        or source.get("model_repository") != MODEL_REPOSITORY
        or source.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("source identity mismatch")
    return dict(source)


def _validate_workload(profile: object, source: dict) -> None:
    expected = {
        "schema_version": (
            "tinyllmforge.tp4-decode-replay-workload.v1"
        ),
        "run_tag": source["run_tag"],
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 4,
        "temperature": 0.0,
        "measured_repetitions": contract.MEASURED_REPETITIONS,
        "workloads": contract.WORKLOADS,
        "cases": list(contract.build_case_matrix()),
    }
    if profile != expected:
        raise ValueError("workload profile mismatch")


def _validate_admission(admission: object, source: dict) -> None:
    mode = (
        admission.get("admission_mode", "strict_clean")
        if isinstance(admission, dict)
        else None
    )
    strict_clean = (
        isinstance(admission, dict)
        and admission.get("strict_clean") is True
        and mode == "strict_clean"
        and admission.get(
            "claim_boundary",
            "FORMAL_STRICT_CLEAN",
        )
        == "FORMAL_STRICT_CLEAN"
    )
    shared_capacity = (
        isinstance(admission, dict)
        and admission.get("strict_clean") is False
        and mode == "shared_capacity"
        and admission.get("claim_boundary") == "DIAGNOSTIC_ONLY"
    )
    memory_limit = (
        MAX_GPU_MEMORY_USED_MIB
        if strict_clean
        else SHARED_CAPACITY_MAX_GPU_MEMORY_USED_MIB
    )
    if (
        not isinstance(admission, dict)
        or admission.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-admission.v1"
        or admission.get("run_tag") != source["run_tag"]
        or not (strict_clean or shared_capacity)
        or admission.get("world_size") != 4
        or not isinstance(admission.get("selected_gpus"), list)
        or len(admission["selected_gpus"]) != 4
        or sorted(
            row.get("rank") for row in admission["selected_gpus"]
        )
        != list(contract.RANKS)
        or len({
            row.get("index") for row in admission["selected_gpus"]
        }) != 4
        or len({
            row.get("uuid") for row in admission["selected_gpus"]
        }) != 4
        or any(
            not isinstance(row.get("memory_used_mib"), int)
            or isinstance(row.get("memory_used_mib"), bool)
            or not 0
            <= row["memory_used_mib"]
            <= memory_limit
            or not isinstance(row.get("utilization_percent"), int)
            or isinstance(row.get("utilization_percent"), bool)
            or not 0
            <= row["utilization_percent"]
            <= MAX_GPU_UTILIZATION_PERCENT
            or not isinstance(row.get("compute_process_count"), int)
            or isinstance(row.get("compute_process_count"), bool)
            or row["compute_process_count"] < 0
            or (
                strict_clean
                and row["compute_process_count"] != 0
            )
            for row in admission["selected_gpus"]
        )
    ):
        raise ValueError("launch admission mismatch")
    baseline = admission.get("baseline_compute_processes", [])
    selected_uuids = {
        row["uuid"] for row in admission["selected_gpus"]
    }
    if (
        not isinstance(baseline, list)
        or (
            strict_clean
            and baseline
        )
        or (
            shared_capacity
            and (
                len(baseline)
                != sum(
                    row["compute_process_count"]
                    for row in admission["selected_gpus"]
                )
                or any(
                    not isinstance(process, dict)
                    or process.get("gpu_uuid") not in selected_uuids
                    or not isinstance(process.get("pid"), int)
                    or isinstance(process.get("pid"), bool)
                    or process["pid"] <= 0
                    or not isinstance(process.get("process_name"), str)
                    or not process["process_name"]
                    or not isinstance(
                        process.get("start_time_ticks"),
                        int,
                    )
                    or isinstance(
                        process.get("start_time_ticks"),
                        bool,
                    )
                    or process["start_time_ticks"] <= 0
                    or not isinstance(
                        process.get("used_memory_mib"),
                        int,
                    )
                    or isinstance(
                        process.get("used_memory_mib"),
                        bool,
                    )
                    or process["used_memory_mib"] < 0
                    for process in baseline
                )
            )
        )
    ):
        raise ValueError("launch admission baseline mismatch")
    if shared_capacity:
        baseline_counts = {uuid: 0 for uuid in selected_uuids}
        for process in baseline:
            baseline_counts[process["gpu_uuid"]] += 1
        expected_counts = {
            row["uuid"]: row["compute_process_count"]
            for row in admission["selected_gpus"]
        }
        if baseline_counts != expected_counts:
            raise ValueError("launch admission baseline mismatch")


def _validate_cleanup(cleanup: object, source: dict) -> None:
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-cleanup.v1"
        or cleanup.get("run_tag") != source["run_tag"]
        or cleanup.get("classification") != "CLEAN"
        or cleanup.get("owned_children_remaining") != []
        or cleanup.get("exact_tag_scans") != [[], [], []]
        or not isinstance(cleanup.get("rank_rows"), list)
        or sorted(
            row.get("rank") for row in cleanup["rank_rows"]
        )
        != list(contract.RANKS)
        or any(
            row.get("exit_code") != 0
            or row.get("process_group_destroyed") is not True
            for row in cleanup["rank_rows"]
        )
    ):
        raise ValueError("cleanup evidence mismatch")


def _validate_process_receipts(receipts: object, source: dict) -> None:
    expected_cases = {
        row["case_id"] for row in contract.build_case_matrix()
    }
    case_rows = (
        receipts.get("case_rows")
        if isinstance(receipts, dict)
        else None
    )
    if (
        not isinstance(receipts, dict)
        or receipts.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-processes.v1"
        or receipts.get("run_tag") != source["run_tag"]
        or not isinstance(case_rows, list)
        or {
            row.get("case_id")
            for row in case_rows
            if isinstance(row, dict)
        }
        != expected_cases
        or len(case_rows) != len(expected_cases)
        or len({
            row.get("dist_port")
            for row in case_rows
            if isinstance(row, dict)
        })
        != len(expected_cases)
        or any(
            not isinstance(row, dict)
            or set(row)
            != {
                "case_id",
                "exit_code",
                "timed_out",
                "dist_port",
                "started_ns",
                "finished_ns",
            }
            or
            row.get("exit_code") != 0
            or row.get("timed_out") is not False
            or not isinstance(row.get("dist_port"), int)
            or isinstance(row.get("dist_port"), bool)
            or not 1024 <= row["dist_port"] <= 65_535
            or not isinstance(row.get("started_ns"), int)
            or isinstance(row.get("started_ns"), bool)
            or row["started_ns"] < 0
            or not isinstance(row.get("finished_ns"), int)
            or isinstance(row.get("finished_ns"), bool)
            or row["finished_ns"] < row["started_ns"]
            for row in case_rows
        )
    ):
        raise ValueError("process receipt mismatch")


def _validate_rank_environment(
    rows: list[dict],
    source: dict,
) -> None:
    if (
        len(rows) != 4
        or sorted(row.get("rank") for row in rows)
        != list(contract.RANKS)
        or any(
            row.get("run_tag") != source["run_tag"]
            or row.get("world_size") != 4
            for row in rows
        )
    ):
        raise ValueError("rank environment mismatch")


def _validate_request_rows(
    rows: list[dict],
    correctness_rows: list[dict],
) -> None:
    expected_cases = {
        row["case_id"]: row for row in contract.build_case_matrix()
    }
    expected_outputs = {}
    for correctness in correctness_rows:
        pair_id = correctness.get("pair_id")
        for arm in contract.ARMS:
            for request_index, output in enumerate(
                correctness.get(f"{arm}_outputs", [])
            ):
                expected_outputs[
                    (pair_id, arm, request_index)
                ] = output
    grouped = {}
    seen = set()
    for row in rows:
        row_id = row.get("row_id")
        case = expected_cases.get(row.get("case_id"))
        request_id = row.get("request_id")
        try:
            request_index = int(request_id.rsplit(":request-", 1)[1])
        except (AttributeError, IndexError, ValueError):
            raise ValueError("request identity mismatch") from None
        expected = expected_outputs.get(
            (row.get("pair_id"), row.get("arm"), request_index)
        )
        observed = {
            "request_id": (
                f"{row.get('pair_id')}:request-{request_index}"
            ),
            "prompt_sha256": row.get("prompt_sha256"),
            "output_token_ids": row.get("output_token_ids"),
            "output_length": row.get("output_length"),
            "stop_reason": row.get("stop_reason"),
        }
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in seen
            or case is None
            or row.get("pair_id") != case["pair_id"]
            or row.get("workload") != case["workload"]
            or row.get("repetition") != case["repetition"]
            or row.get("arm") != case["arm"]
            or row.get("phase") != "measured"
            or row.get("prompt_tokens")
            != case["profile"]["prompt_tokens"]
            or row.get("generated_tokens")
            != case["profile"]["output_tokens"]
            or row.get("output_length")
            != len(row.get("output_token_ids", []))
            or observed != expected
        ):
            raise ValueError("request row or output identity mismatch")
        seen.add(row_id)
        grouped.setdefault(case["case_id"], []).append(row)
    if (
        set(grouped) != set(expected_cases)
        or any(
            len(grouped[case_id])
            != expected_cases[case_id]["profile"]["concurrency"]
            for case_id in expected_cases
        )
    ):
        raise ValueError("request row case matrix mismatch")


def _safe_metrics(reconstructed: dict) -> dict:
    value = reconstructed.get("capture_amortization_tokens")
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    return {
        "workloads": reconstructed.get("workloads", {}),
        "aggregate": reconstructed.get("aggregate", {}),
        "replay_coverage": reconstructed.get("replay_coverage"),
        "maximum_added_peak_allocated_bytes": reconstructed.get(
            "maximum_added_peak_allocated_bytes"
        ),
        "maximum_added_peak_reserved_bytes": reconstructed.get(
            "maximum_added_peak_reserved_bytes"
        ),
        "capture_duration_ns": reconstructed.get(
            "capture_duration_ns"
        ),
        "capture_amortization_tokens": value,
    }


def _report_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value, allow_nan=False)
    raise ValueError("report metric has an unsupported value")


def _render_report(
    *,
    source: dict,
    admission: dict,
    cleanup: dict,
    classification: dict,
) -> str:
    admission_mode = admission.get(
        "admission_mode",
        "strict_clean",
    )
    claim_boundary = admission.get(
        "claim_boundary",
        "FORMAL_STRICT_CLEAN",
    )
    stage1_authorized = (
        classification["classification"] == "GO_STAGE1_JUSTIFIED"
        and claim_boundary == "FORMAL_STRICT_CLEAN"
    )
    failed_gates = classification["failed_gates"]
    lines = [
        "# Qwen3.8 TP4 Collective-Stable Decode Replay Qualification",
        "",
        f"Run tag: `{source['run_tag']}`",
        "",
        f"Source revision: `{source['source_revision']}`",
        "",
        f"Source tree SHA256: `{source['source_tree_sha256']}`",
        "",
        f"Model repository: `{source['model_repository']}`",
        "",
        f"Model revision: `{source['model_revision']}`",
        "",
        f"Admission mode: `{admission_mode}`",
        "",
        f"Claim boundary: `{claim_boundary}`",
        "",
        f"Cleanup: `{cleanup['classification']}`",
        "",
        (
            "Classification: "
            f"`{classification['classification']}`"
        ),
        "",
        (
            "Stage-1 authorization: "
            f"`{_report_value(stage1_authorized)}`"
        ),
        "",
        "Failed gates:",
        "",
    ]
    if failed_gates:
        lines.extend(f"- `{gate}`" for gate in failed_gates)
    else:
        lines.append("- none")
    aggregate = classification.get("aggregate", {})
    workloads = classification.get("workloads", {})
    added_allocated = _report_value(
        classification.get("maximum_added_peak_allocated_bytes")
    )
    added_reserved = _report_value(
        classification.get("maximum_added_peak_reserved_bytes")
    )
    capture_duration = _report_value(
        classification.get("capture_duration_ns")
    )
    capture_amortization = _report_value(
        classification.get("capture_amortization_tokens")
    )
    lines.extend((
        "",
        "## Benefit and cost",
        "",
        "| Scope | Throughput ratio | Median TPOT ratio | "
        "P99 E2E ratio | TTFT ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            "| aggregate | "
            f"{_report_value(aggregate.get('output_throughput_ratio'))} | "
            f"{_report_value(aggregate.get('median_tpot_ratio'))} | "
            "N/A | N/A |"
        ),
    ))
    for workload in contract.WORKLOADS:
        payload = workloads.get(workload, {})
        lines.append(
            f"| {workload} | "
            f"{_report_value(payload.get('output_throughput_ratio'))} | "
            f"{_report_value(payload.get('median_tpot_ratio'))} | "
            f"{_report_value(payload.get('p99_e2e_ratio'))} | "
            f"{_report_value(payload.get('ttft_ratio'))} |"
        )
    lines.extend((
        "",
        (
            "Replay coverage: "
            f"`{_report_value(classification.get('replay_coverage'))}`"
        ),
        "",
        (
            "Added peak allocated bytes: "
            f"`{added_allocated}`"
        ),
        "",
        (
            "Added peak reserved bytes: "
            f"`{added_reserved}`"
        ),
        "",
        (
            "Capture duration ns: "
            f"`{capture_duration}`"
        ),
        "",
        (
            "Capture amortization tokens: "
            f"`{capture_amortization}`"
        ),
        "",
        (
            "Unique program keys per cohort: "
            f"`{_report_value(classification.get('unique_program_key_count'))}`"
        ),
        "",
        (
            "Unique invocation identities per cohort: "
            f"`{_report_value(classification.get('unique_invocation_identity_count'))}`"
        ),
        "",
        (
            "Cross-lease replay rank-steps: "
            f"`{_report_value(classification.get('cross_lease_replay_count'))}`"
        ),
        "",
        (
            "Manifest validations: "
            f"`{_report_value(classification.get('manifest_validation_count'))}`"
        ),
        "",
        (
            "Only `GO_STAGE1_JUSTIFIED` evidence collected under "
            "`FORMAL_STRICT_CLEAN` may authorize Stage 1. "
            "`DIAGNOSTIC_ONLY` evidence never authorizes Stage 1."
        ),
        "",
    ))
    return "\n".join(lines)


def _incomplete(reason: str) -> dict:
    return {
        "classification": "INCOMPLETE",
        "failed_gates": [reason],
        "verified_hashes": False,
        "producer_classification_matches": False,
        "summary_matches": False,
        "metrics": {},
    }


def verify_bundle(root: Path) -> dict:
    root = Path(root).resolve()
    try:
        if not root.is_dir():
            raise ValueError("bundle root is missing")
        _verify_manifest(root)
        source = _validate_source(
            _load_json(root / "source_identity.json")
        )
        if _load_json(root / "source_manifest.json") != source:
            raise ValueError("source manifest mismatch")
        if not (root / "source.patch").read_bytes():
            raise ValueError("source patch is empty")
        environment = _load_json(root / "environment.json")
        if (
            not isinstance(environment, dict)
            or environment.get("schema_version")
            != "tinyllmforge.tp4-decode-replay-environment.v1"
            or environment.get("run_tag") != source["run_tag"]
        ):
            raise ValueError("environment identity mismatch")
        admission = _load_json(root / "launch_admission.json")
        _validate_admission(admission, source)
        if _load_json(root / "gpu_inventory.json") != admission:
            raise ValueError("GPU inventory mismatch")
        _validate_workload(
            _load_json(root / "workload_profile.json"),
            source,
        )
        _validate_process_receipts(
            _load_json(root / "process_receipts.json"),
            source,
        )
        _validate_cleanup(
            _load_json(root / "cleanup.json"),
            source,
        )
        rank_environment = _load_jsonl(
            root / "rank_environment.jsonl"
        )
        _validate_rank_environment(rank_environment, source)
        loaded_rows = {
            name: _load_jsonl(root / name)
            for name in JSONL_FILES
        }
        evidence = {
            argument: loaded_rows[name]
            for argument, name in EVIDENCE_FILES.items()
        }
        _validate_request_rows(
            loaded_rows["request_rows.jsonl"],
            evidence["correctness_rows"],
        )
        reconstructed = _classify_evidence(**evidence)
        producer = _load_json(
            root / "producer_classification.json"
        )
        expected_producer = {
            "schema_version": CLASSIFICATION_SCHEMA,
            "classification": reconstructed["classification"],
            "failed_gates": reconstructed["failed_gates"],
            "stage1_authorized": (
                reconstructed["classification"]
                == "GO_STAGE1_JUSTIFIED"
                and admission.get(
                    "claim_boundary",
                    "FORMAL_STRICT_CLEAN",
                )
                == "FORMAL_STRICT_CLEAN"
            ),
        }
        producer_matches = producer == expected_producer
        expected_summary = {
            "schema_version": SUMMARY_SCHEMA,
            "run_tag": source["run_tag"],
            "source_revision": source["source_revision"],
            "source_tree_sha256": source["source_tree_sha256"],
            "model_repository": source["model_repository"],
            "model_revision": source["model_revision"],
            **reconstructed,
        }
        summary_matches = (
            _load_json(root / "summary.json") == expected_summary
        )
        expected_report = _render_report(
            source=source,
            admission=admission,
            cleanup=_load_json(root / "cleanup.json"),
            classification=reconstructed,
        )
        if (root / "report.md").read_bytes() != expected_report.encode(
            "utf-8"
        ):
            raise ValueError(
                "report does not match reconstructed evidence"
            )
        if (
            reconstructed["classification"]
            == "GO_STAGE1_JUSTIFIED"
            and (not producer_matches or not summary_matches)
        ):
            return _incomplete("producer_evidence_mismatch")
        return {
            "classification": reconstructed["classification"],
            "failed_gates": reconstructed["failed_gates"],
            "verified_hashes": True,
            "producer_classification_matches": producer_matches,
            "summary_matches": summary_matches,
            "metrics": _safe_metrics(reconstructed),
        }
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
    ) as error:
        return _incomplete(str(error) or type(error).__name__)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(
        verify_bundle(args.bundle_root),
        sort_keys=True,
        allow_nan=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
