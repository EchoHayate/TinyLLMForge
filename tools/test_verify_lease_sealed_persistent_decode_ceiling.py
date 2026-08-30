#!/usr/bin/env python3
"""Tests for the independent persistent-decode ceiling verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import lease_sealed_persistent_decode_ceiling as ceiling
from tools import profile_lease_sealed_persistent_decode_ceiling as profile
from tools import verify_lease_sealed_persistent_decode_ceiling as verifier


REQUIRED_FILES = (
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
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload, *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            allow_nan=allow_nan,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: list[dict],
    *,
    allow_nan: bool = False,
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                allow_nan=allow_nan,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _tree_digest(source_hashes: dict[str, str]) -> str:
    return _sha256_bytes(
        json.dumps(
            source_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _segment_signature() -> str:
    return _sha256_bytes(
        json.dumps(
            [["NORMALIZATION", "rms_norm_kernel"]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _identity_payloads():
    source_hashes = {
        relative: _sha256_bytes(relative.encode("utf-8"))
        for relative in profile.SOURCE_FILES
    }
    source_tree = _tree_digest(source_hashes)
    runtime = {
        "schema_version":
            "lease-sealed-persistent-decode.runtime.v1",
        "python": "3.11.9",
        "pytorch": "2.6.0",
        "cuda": "12.4",
        "gpu": "NVIDIA A100-SXM4-80GB",
        "nsight_systems": "2024.7.1",
        "model_path": "/data00/home/sitian/.ms_cache/model",
        "checkpoint_inventory_sha256": "e" * 64,
        "feature_configuration": {
            "policy": "decode_burst_k8",
            "tensor_parallel_size": 1,
        },
    }
    workload = {
        "schema_version":
            "lease-sealed-persistent-decode.workload.v1",
        "contexts": list(ceiling.CONTEXT_LENGTHS),
        "generated_tokens": ceiling.GENERATED_TOKENS,
        "repetitions": ceiling.REPETITIONS,
        "temperature": 0.0,
        "ignore_eos": True,
        "max_num_seqs": 1,
    }
    return (
        source_hashes,
        source_tree,
        runtime,
        workload,
        _sha256_bytes(
            json.dumps(
                runtime,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
        _sha256_bytes(
            json.dumps(
                workload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ),
    )


def _refresh_manifest(run_dir: Path) -> None:
    artifacts = []
    for name in REQUIRED_FILES:
        path = run_dir / name
        artifacts.append({
            "path": name,
            "byte_length": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version":
                "lease-sealed-persistent-decode.manifest.v1",
            "artifacts": artifacts,
        },
    )


def _trace_rows(context: int) -> tuple[list[dict], list[dict]]:
    identity = {
        "attempt": "run-a",
        "workload": "exact_greedy_k8",
        "repetition": 0,
        "context": context,
        "burst": 0,
        "logical_tokens": 127,
    }
    kernels = [
        {
            **identity,
            "start_ns": 100,
            "end_ns": 16_000_100,
            "duration_ns": 16_000_000,
            "stream_id": 7,
            "global_pid": 1,
            "name": "rms_norm_kernel",
            "role": "NORMALIZATION",
        },
        {
            **identity,
            "start_ns": 16_000_100,
            "end_ns": 32_000_100,
            "duration_ns": 16_000_000,
            "stream_id": 7,
            "global_pid": 1,
            "name": "ampere_bf16_gemm",
            "role": "MATMUL",
        },
    ]
    segments = [
        {
            **identity,
            "segment_id": 0,
            "stream_id": 7,
            "first_kernel_start_ns": 100,
            "last_kernel_end_ns": 16_000_100,
            "kernel_count": 1,
            "kernel_duration_sum_ns": 16_000_000,
            "internal_gap_sum_ns": 0,
            "wall_union_ns": 16_000_000,
            "role_histogram": {"NORMALIZATION": 1},
            "normalized_kernel_signature_sha256": _segment_signature(),
        },
    ]
    return kernels, segments


def make_complete_artifact(root: Path) -> Path:
    run_dir = root / "run-a"
    run_dir.mkdir(parents=True)
    (
        source_hashes,
        source_tree,
        runtime,
        workload,
        runtime_digest,
        workload_digest,
    ) = _identity_payloads()
    timing = profile.synthetic_timing_rows_for_test()
    structural = profile.synthetic_structural_rows_for_test()
    kernels = []
    segments = []
    contexts = []
    raw_traces = []
    for row in timing:
        row["source_tree_sha256"] = source_tree
        row["runtime_identity_sha256"] = runtime_digest
        row["workload_identity_sha256"] = workload_digest
    for row in structural:
        context = row["context_length"]
        row.update({
            "source_commit": "a" * 40,
            "source_tree_sha256": source_tree,
            "runtime_identity_sha256": runtime_digest,
            "workload_identity_sha256": workload_digest,
            "generated_tokens": ceiling.GENERATED_TOKENS,
            "profiled_tpot_median_ns": 2_100_000,
            "profiled_tpot_p95_ns": 2_200_000,
            "target_model_forwards": 127,
            "committed_tokens": 127,
            "fallback_count": 0,
            "failure_count": 0,
            "rollback_count": 0,
            "quarantine_reason": None,
        })
        context_kernels, context_segments = _trace_rows(context)
        kernels.extend(context_kernels)
        context_segments[0]["segment_id"] = len(segments)
        segments.extend(context_segments)
        contexts.append({
            "context_length": context,
            "profiled_tpot_median_ns": 2_100_000,
            "profiled_tpot_p95_ns": 2_200_000,
            "output_token_ids": row["output_token_ids"],
            "output_text_sha256": row["output_text_sha256"],
            "transaction_count": 1,
            "logical_token_count": 127,
            "eligible_zero_cost_ns_per_token":
                16_000_000 / 127,
            "candidate_cuda_duration_ns": 16_000_000,
            "total_kernel_duration_ns": 32_000_000,
            "classified_launch_ratio": 1.0,
            "classified_duration_ratio": 1.0,
            "segment_signatures": [_segment_signature()],
            "target_model_forwards": 127,
            "committed_tokens": 127,
            "fallback_count": 0,
            "failure_count": 0,
            "rollback_count": 0,
            "quarantine_reason": None,
        })
        raw_traces.append({
            "context_length": context,
            "remote_path": (
                "/data00/home/sitian/tinyllmforge-workspaces/"
                f"command-timeline-20260818/run-a/nsys/{context}.sqlite"
            ),
            "byte_length": 1024 + context,
            "sha256": _sha256_bytes(str(context).encode("utf-8")),
            "transaction_count": 1,
            "kernel_count": 2,
        })
    trace_summary = {
        "schema_version": ceiling.TRACE_SUMMARY_SCHEMA_VERSION,
        "source_commit": "a" * 40,
        "source_tree_sha256": source_tree,
        "runtime_identity_sha256": runtime_digest,
        "workload_identity_sha256": workload_digest,
        "contexts": contexts,
    }
    reported_ceiling = ceiling.compute_ceiling(timing, trace_summary)
    _write_json(
        run_dir / "source_manifest.json",
        {
            "schema_version": profile.SOURCE_SCHEMA_VERSION,
            "run_tag": "run-a",
            "source_commit": "a" * 40,
            "source_tree_sha256": source_tree,
            "source_sha256": source_hashes,
        },
    )
    _write_json(run_dir / "runtime_manifest.json", runtime)
    _write_json(
        run_dir / "gpu_admission.json",
        {
            "schema_version":
                "lease-sealed-persistent-decode.gpu-admission.v1",
            "strict_clean": True,
            "gpu_index": 1,
            "compute_process_count": 0,
            "memory_used_mib": 0,
            "utilization_gpu_pct": 0,
        },
    )
    _write_json(run_dir / "workload_manifest.json", workload)
    _write_jsonl(run_dir / "timing_rows.jsonl", timing)
    _write_jsonl(run_dir / "structural_rows.jsonl", structural)
    _write_json(
        run_dir / "timing_summary.json",
        {
            "schema_version": ceiling.TIMING_SCHEMA_VERSION,
            "row_count": len(timing),
            "contexts": list(ceiling.CONTEXT_LENGTHS),
        },
    )
    _write_json(
        run_dir / "trace_inventory.json",
        {
            "schema_version":
                "lease-sealed-persistent-decode.trace-inventory.v1",
            "raw_traces": raw_traces,
            "trace_summary": trace_summary,
        },
    )
    _write_jsonl(run_dir / "kernel_rows.jsonl", kernels)
    _write_jsonl(run_dir / "segment_rows.jsonl", segments)
    _write_json(run_dir / "ceiling.json", reported_ceiling)
    _refresh_manifest(run_dir)
    return run_dir


def _mutate_json(run_dir: Path, name: str, mutate) -> None:
    path = run_dir / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)
    _refresh_manifest(run_dir)


def _mutate_jsonl(run_dir: Path, name: str, mutate) -> None:
    path = run_dir / name
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    mutate(rows)
    _write_jsonl(path, rows)
    _refresh_manifest(run_dir)


def test_verifier_reconstructs_ceiling_without_trusting_ceiling_json(
    tmp_path: Path,
) -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "import lease_sealed_persistent_decode_ceiling" not in source
    assert "from tools.lease_sealed_persistent_decode_ceiling" not in source
    run_dir = make_complete_artifact(tmp_path)
    _mutate_json(
        run_dir,
        "ceiling.json",
        lambda payload: payload.__setitem__(
            "aggregate_optimistic_improvement_pct",
            payload["aggregate_optimistic_improvement_pct"] + 0.1,
        ),
    )

    with pytest.raises(ValueError, match="ceiling mismatch"):
        verifier.verify_artifact_directory(run_dir)


@pytest.mark.parametrize(
    ("artifact", "mutation", "message"),
    [
        ("timing_rows.jsonl", lambda rows: rows.pop(), "timing inventory"),
        (
            "structural_rows.jsonl",
            lambda rows: rows.pop(),
            "structural inventory",
        ),
        (
            "timing_rows.jsonl",
            lambda rows: rows[0]["output_token_ids"].__setitem__(0, 9),
            "output mismatch",
        ),
        (
            "structural_rows.jsonl",
            lambda rows: rows[0].__setitem__(
                "output_text_sha256",
                "9" * 64,
            ),
            "output mismatch",
        ),
        (
            "kernel_rows.jsonl",
            lambda rows: rows[0].__setitem__("role", "UNKNOWN"),
            "coverage",
        ),
        (
            "segment_rows.jsonl",
            lambda rows: rows[0].__setitem__(
                "wall_union_ns",
                rows[0]["wall_union_ns"] + 1,
            ),
            "segment mismatch",
        ),
    ],
)
def test_verifier_rejects_evidence_mutations(
    tmp_path: Path,
    artifact: str,
    mutation,
    message: str,
) -> None:
    run_dir = make_complete_artifact(tmp_path)
    _mutate_jsonl(run_dir, artifact, mutation)

    with pytest.raises(ValueError, match=message):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_source_hash_drift(tmp_path: Path) -> None:
    run_dir = make_complete_artifact(tmp_path)
    _mutate_json(
        run_dir,
        "source_manifest.json",
        lambda payload: payload["source_sha256"].__setitem__(
            profile.SOURCE_FILES[0],
            "0" * 64,
        ),
    )

    with pytest.raises(ValueError, match="source hash"):
        verifier.verify_artifact_directory(run_dir)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("thresholds", {"minimum_aggregate_optimistic_improvement_pct": 0}),
        ("classification", "NO_GO_PERSISTENT_DECODE_CEILING"),
    ],
)
def test_verifier_rejects_reported_ceiling_drift(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    run_dir = make_complete_artifact(tmp_path)
    _mutate_json(
        run_dir,
        "ceiling.json",
        lambda payload: payload.__setitem__(field, value),
    )

    with pytest.raises(ValueError, match="ceiling mismatch"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_rejects_manifest_duplicate_and_traversal(
    tmp_path: Path,
) -> None:
    duplicate = make_complete_artifact(tmp_path / "duplicate")
    manifest = json.loads(
        (duplicate / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["artifacts"].append(dict(manifest["artifacts"][0]))
    _write_json(duplicate / "manifest.json", manifest)
    with pytest.raises(ValueError, match="duplicate artifact path"):
        verifier.verify_artifact_directory(duplicate)

    traversal = make_complete_artifact(tmp_path / "traversal")
    manifest = json.loads(
        (traversal / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["artifacts"][0]["path"] = "../source_manifest.json"
    _write_json(traversal / "manifest.json", manifest)
    with pytest.raises(ValueError, match="artifact path"):
        verifier.verify_artifact_directory(traversal)


def test_verifier_rejects_extra_file_and_missing_raw_digest(
    tmp_path: Path,
) -> None:
    extra = make_complete_artifact(tmp_path / "extra")
    (extra / "undeclared.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared artifact"):
        verifier.verify_artifact_directory(extra)

    missing_digest = make_complete_artifact(tmp_path / "raw")
    _mutate_json(
        missing_digest,
        "trace_inventory.json",
        lambda payload: payload["raw_traces"][0].pop("sha256"),
    )
    with pytest.raises(ValueError, match="raw trace digest"):
        verifier.verify_artifact_directory(missing_digest)


@pytest.mark.parametrize("non_finite", (float("nan"), float("inf")))
def test_verifier_rejects_non_finite_numbers(
    tmp_path: Path,
    non_finite: float,
) -> None:
    run_dir = make_complete_artifact(tmp_path)
    path = run_dir / "timing_rows.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["tpot_median_ns"] = non_finite
    _write_jsonl(path, rows, allow_nan=True)
    _refresh_manifest(run_dir)

    with pytest.raises(ValueError, match="non-finite"):
        verifier.verify_artifact_directory(run_dir)


def test_verifier_accepts_complete_bundle(tmp_path: Path) -> None:
    run_dir = make_complete_artifact(tmp_path)

    result = verifier.verify_artifact_directory(run_dir)

    assert result == {
        "schema_version":
            "lease-sealed-persistent-decode.verification.v1",
        "verified": True,
        "run_tag": "run-a",
        "source_commit": "a" * 40,
        "classification": "GO_PERSISTENT_DECODE_CEILING",
        "timing_row_count": 15,
        "structural_context_count": 3,
        "kernel_row_count": 6,
        "segment_row_count": 3,
    }
