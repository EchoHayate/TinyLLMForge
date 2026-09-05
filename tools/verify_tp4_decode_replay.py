#!/usr/bin/env python3
"""Independently reconstruct a TP4 decode replay qualification verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

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
        reconstructed = contract.classify(**evidence)
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
