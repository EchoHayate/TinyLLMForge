#!/usr/bin/env python3
"""Assemble immutable TP4 decode replay qualification evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import tp4_decode_replay_contract as contract


MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MANIFEST_SCHEMA = "tinyllmforge.tp4-decode-replay-manifest.v1"
SUMMARY_SCHEMA = "tinyllmforge.tp4-decode-replay-summary.v1"
CLASSIFICATION_SCHEMA = (
    "tinyllmforge.tp4-decode-replay-classification.v1"
)
REQUIRED_INPUTS = (
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
)
ADDITIONAL_ARTIFACTS = (
    "source_identity.json",
    "launch_admission.json",
    "cleanup.json",
    "summary.json",
    "producer_classification.json",
    "manifest.json",
)
PRODUCER_ARTIFACTS = REQUIRED_INPUTS + ADDITIONAL_ARTIFACTS
JSON_INPUTS = frozenset({
    "source_manifest.json",
    "environment.json",
    "gpu_inventory.json",
    "workload_profile.json",
    "process_receipts.json",
})
JSONL_INPUTS = frozenset(
    name for name in REQUIRED_INPUTS if name.endswith(".jsonl")
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
    if not payload.endswith(b"\n"):
        raise ValueError(f"JSONL lacks terminal newline: {path.name}")
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
        raise ValueError(f"empty or truncated required input: {path.name}")
    return rows


def _atomic_write_json(path: Path, payload: object) -> None:
    _require_finite(payload)
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


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    _require_finite(rows)
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
        for row in rows:
            handle.write(json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


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
        raise ValueError("source identity is invalid")
    return dict(source)


def _validate_admission(admission: object, source: dict) -> dict:
    if (
        not isinstance(admission, dict)
        or admission.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-admission.v1"
        or admission.get("run_tag") != source["run_tag"]
        or admission.get("strict_clean") is not True
        or admission.get("world_size") != 4
        or not isinstance(admission.get("selected_gpus"), list)
        or len(admission["selected_gpus"]) != 4
    ):
        raise ValueError("launch admission is invalid")
    rows = admission["selected_gpus"]
    if (
        sorted(row.get("rank") for row in rows) != list(contract.RANKS)
        or len({row.get("index") for row in rows}) != 4
        or len({row.get("uuid") for row in rows}) != 4
        or any(
            row.get("memory_used_mib") != 0
            or row.get("compute_process_count") != 0
            for row in rows
        )
    ):
        raise ValueError("launch admission GPU inventory is invalid")
    return dict(admission)


def _validate_cleanup(cleanup: object, source: dict) -> dict:
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
        raise ValueError("cleanup identity is invalid")
    return dict(cleanup)


def _validate_workload_profile(profile: object, source: dict) -> None:
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
        raise ValueError("workload profile is invalid")


def _validate_process_receipts(receipts: object, source: dict) -> None:
    expected_cases = {
        row["case_id"] for row in contract.build_case_matrix()
    }
    if (
        not isinstance(receipts, dict)
        or receipts.get("schema_version")
        != "tinyllmforge.tp4-decode-replay-processes.v1"
        or receipts.get("run_tag") != source["run_tag"]
        or not isinstance(receipts.get("case_rows"), list)
        or {
            row.get("case_id") for row in receipts["case_rows"]
        }
        != expected_cases
        or len(receipts["case_rows"]) != len(expected_cases)
        or any(
            row.get("exit_code") != 0
            or row.get("timed_out") is not False
            for row in receipts["case_rows"]
        )
    ):
        raise ValueError("process receipts are invalid")


def _validate_rank_environment(rows: list[dict], source: dict) -> None:
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
        raise ValueError("rank environment is invalid")


def _validate_request_rows(rows: list[dict]) -> None:
    expected_cases = {
        row["case_id"]: row for row in contract.build_case_matrix()
    }
    grouped = {}
    seen = set()
    for row in rows:
        row_id = row.get("row_id")
        case = expected_cases.get(row.get("case_id"))
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
            or row.get("stop_reason") != "length"
        ):
            raise ValueError("request row identity is invalid")
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
        raise ValueError("request row case matrix is incomplete")


def _load_required_inputs(raw_root: Path) -> dict:
    loaded = {}
    for name in REQUIRED_INPUTS:
        path = raw_root / name
        if not path.is_file():
            raise ValueError(f"required input is missing: {name}")
        if path.stat().st_size == 0:
            raise ValueError(f"empty or truncated required input: {name}")
        if name in JSON_INPUTS:
            loaded[name] = _load_json(path)
        elif name in JSONL_INPUTS:
            loaded[name] = _load_jsonl(path)
        else:
            loaded[name] = path.read_bytes()
            if not loaded[name]:
                raise ValueError(
                    f"empty or truncated required input: {name}"
                )
    return loaded


def _write_manifest(root: Path) -> None:
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _atomic_write_json(root / "manifest.json", {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    })


def assemble_bundle(
    *,
    raw_root: Path,
    output_root: Path,
    source_identity: dict,
    launch_admission: dict,
    cleanup: dict,
) -> dict:
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise ValueError("raw root must be an existing directory")
    loaded = _load_required_inputs(raw_root)
    source = _validate_source(source_identity)
    if loaded["source_manifest.json"] != source:
        raise ValueError("source manifest identity mismatch")
    admission = _validate_admission(launch_admission, source)
    if loaded["gpu_inventory.json"] != admission:
        raise ValueError("GPU inventory identity mismatch")
    cleanup = _validate_cleanup(cleanup, source)
    _validate_workload_profile(
        loaded["workload_profile.json"],
        source,
    )
    _validate_process_receipts(
        loaded["process_receipts.json"],
        source,
    )
    _validate_rank_environment(
        loaded["rank_environment.jsonl"],
        source,
    )
    _validate_request_rows(loaded["request_rows.jsonl"])
    evidence = {
        argument: loaded[name]
        for argument, name in EVIDENCE_FILES.items()
    }
    classification = contract.classify(**evidence)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("bundle output directory must be empty")
    for name in REQUIRED_INPUTS:
        value = loaded[name]
        if name in JSON_INPUTS:
            _atomic_write_json(root / name, value)
        elif name in JSONL_INPUTS:
            _atomic_write_jsonl(root / name, value)
        else:
            _atomic_write_bytes(root / name, value)
    _atomic_write_json(root / "source_identity.json", source)
    _atomic_write_json(root / "launch_admission.json", admission)
    _atomic_write_json(root / "cleanup.json", cleanup)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "run_tag": source["run_tag"],
        "source_revision": source["source_revision"],
        "source_tree_sha256": source["source_tree_sha256"],
        "model_repository": source["model_repository"],
        "model_revision": source["model_revision"],
        **classification,
    }
    producer = {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification["classification"],
        "failed_gates": classification["failed_gates"],
        "stage1_authorized": (
            classification["classification"] == "GO_STAGE1_JUSTIFIED"
        ),
    }
    _atomic_write_json(root / "summary.json", summary)
    _atomic_write_json(
        root / "producer_classification.json",
        producer,
    )
    _write_manifest(root)
    return {
        "classification": classification["classification"],
        "bundle_root": str(root),
        "artifact_count": len(PRODUCER_ARTIFACTS),
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-identity-json", required=True)
    parser.add_argument("--launch-admission-json", required=True)
    parser.add_argument("--cleanup-json", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result = assemble_bundle(
        raw_root=args.raw_root,
        output_root=args.output_root,
        source_identity=json.loads(args.source_identity_json),
        launch_admission=json.loads(args.launch_admission_json),
        cleanup=json.loads(args.cleanup_json),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
