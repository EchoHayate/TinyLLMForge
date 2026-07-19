"""Dependency-light tests for the independent arrival-load verifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = REPO_ROOT / "tools" / "arrival_load_verify.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location(
        "arrival_load_verify",
        VERIFY_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load arrival_load_verify")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")


def _policy_id(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _case_id(policy: str, repetition: int) -> str:
    return f"{policy}-steady_moderate-r{repetition}"


def _complete_artifact() -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    source_root = root / "snapshot-source"
    source_root.mkdir()
    (source_root / "marker.txt").write_text("arrival-load-source\n")
    source_files = [{
        "path": "marker.txt",
        "size_bytes": (source_root / "marker.txt").stat().st_size,
        "sha256": _sha256_file(source_root / "marker.txt"),
    }]
    source_tree_sha256 = hashlib.sha256(
        _canonical_bytes(source_files)
    ).hexdigest()
    source_evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "tree_sha256": source_tree_sha256,
        "files": source_files,
        "patch_size_bytes": 0,
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
    }
    _write_json(root / "source_evidence.json", source_evidence)
    (root / "source.patch").write_bytes(b"")
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz") as archive:
        archive.add(source_root, arcname="source")

    policy_identity_by_name = {
        "P0": _policy_id("P0"),
        "P1": _policy_id("P0"),
        "P2": _policy_id("P2"),
        "P3": _policy_id("P3"),
    }
    manifest = {
        "schema_version": 1,
        "run_tag": "synthetic-arrival-load",
        "source_tree_sha256": source_tree_sha256,
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 3,
        "policy_identity_by_name": policy_identity_by_name,
        "canonical_policy_by_name": {
            "P0": "P0",
            "P1": "P0",
            "P2": "P2",
            "P3": "P3",
        },
        "process_port_pairs": [],
        "expected_case_ids": [],
    }
    calibration_manifest = [{
        "calibration_id": "p0-rate-00",
        "policy": "P0",
        "requested_rate_rps": 1.0,
    }]
    calibration_rows = [{
        "calibration_id": "p0-rate-00",
        "status": "PASS",
        "stable": True,
        "completed_request_throughput_rps": 1.0,
    }]
    workload_rows = [{
        "schema_version": 1,
        "request_id": "steady_moderate-measured-0000",
        "scenario": "steady_moderate",
        "warmup": False,
        "arrival_offset_ns": 0,
        "prompt_token_count": 4,
        "requested_output_tokens": 2,
        "prompt_class": "short",
        "output_class": "short",
        "service_time_bucket": "short__short",
    }]
    timeline_rows = []
    scheduler_rows = []
    memory_rows = []
    case_rows = []
    ports = 31000
    for repetition in range(3):
        for policy in ("P0", "P2", "P3"):
            case_id = _case_id(policy, repetition)
            manifest["expected_case_ids"].append(case_id)
            manifest["process_port_pairs"].append({
                "case_id": case_id,
                "tinyvllm_dist_port": ports,
                "master_port": ports + 1,
            })
            ports += 2
            duration_ns = 1_000_000_000
            seq_id = repetition * 10 + {"P0": 0, "P2": 1, "P3": 2}[policy]
            timeline_rows.append({
                "case_id": case_id,
                "policy": policy,
                "scenario": "steady_moderate",
                "repetition": repetition,
                "request_id": workload_rows[0]["request_id"],
                "seq_id": seq_id,
                "scheduled_arrival_ns": 100,
                "actual_arrival_ns": 110,
                "first_scheduled_ns": 120,
                "first_token_ns": 200,
                "token_timestamps_ns": [200, duration_ns + 100],
                "completion_ns": duration_ns + 100,
                "output_token_ids": [11, 12],
                "finish_reason": "length",
                "error": None,
            })
            scheduler_rows.append({
                "case_id": case_id,
                "step_index": 0,
                "step_start_ns": 120,
                "step_end_ns": duration_ns + 100,
                "scheduled_seq_ids": [seq_id],
                "finished_seq_ids": [seq_id],
            })
            memory_rows.append({
                "case_id": case_id,
                "step_index": 0,
                "cuda_allocated_bytes": 100,
                "cuda_reserved_bytes": 200,
                "used_kv_blocks": 2,
                "kv_block_bytes": 64,
            })
            case_rows.append({
                "case_id": case_id,
                "policy": policy,
                "scenario": "steady_moderate",
                "repetition": repetition,
                "status": "PASS",
                "correctness": {
                    "exact_outputs": True,
                    "complete_requests": True,
                    "no_starvation": True,
                    "valid_lifecycle": True,
                    "stable_p0_outputs": True,
                },
                "metrics": {
                    "request_throughput_rps": 1.0,
                    "output_token_throughput_tps": 2.0,
                    "maximum_injection_lag_ns": 10.0,
                    "p50_injection_lag_ns": 10.0,
                    "p95_injection_lag_ns": 10.0,
                    "p99_injection_lag_ns": 10.0,
                    "p50_queue_delay_ns": 10.0,
                    "p95_queue_delay_ns": 10.0,
                    "p99_queue_delay_ns": 10.0,
                    "p50_ttft_ns": 100.0,
                    "p95_ttft_ns": 100.0,
                    "p50_itl_ns": 999_999_900.0,
                    "p95_itl_ns": 999_999_900.0,
                    "p50_e2e_ns": 1_000_000_000.0,
                    "p95_e2e_ns": 1_000_000_000.0,
                    "p99_ttft_ns": 100.0,
                    "p99_itl_ns": 999_999_900.0,
                    "p99_e2e_ns": 1_000_000_000.0,
                    "maximum_decode_gap_ns": 999_999_900.0,
                    "peak_cuda_allocated_bytes": 100,
                    "peak_cuda_reserved_bytes": 200,
                    "peak_used_kv_blocks": 2,
                    "peak_kv_bytes": 128,
                    "service_buckets": {
                        "short__short": {
                            "p50_e2e_ns": 1_000_000_000.0,
                            "p95_e2e_ns": 1_000_000_000.0,
                            "p99_e2e_ns": 1_000_000_000.0,
                            "completed_requests": 1,
                            "request_throughput_rps": 1.0,
                            "worst_e2e_ns": 1_000_000_000.0,
                        },
                    },
                    "jain_service_rate_index": 1.0,
                },
            })
    _write_json(root / "run_manifest.json", manifest)
    _write_jsonl(
        root / "calibration_manifest.jsonl",
        calibration_manifest,
    )
    _write_jsonl(root / "calibration_rows.jsonl", calibration_rows)
    _write_jsonl(root / "workload_manifest.jsonl", workload_rows)
    _write_jsonl(root / "request_timeline.jsonl", timeline_rows)
    _write_jsonl(root / "scheduler_trace.jsonl", scheduler_rows)
    _write_jsonl(root / "memory_trace.jsonl", memory_rows)
    _write_jsonl(root / "case_rows.jsonl", case_rows)
    summary = {
        "classification": "NO_GO",
        "structural_failures": [],
        "correctness_failures": [],
        "candidate_results": {
            "P2": {
                "policy": "P2",
                "classification": "NO_GO",
                "benefit_path": None,
                "median_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "worst_repetition_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "guard_failures": [],
            },
            "P3": {
                "policy": "P3",
                "classification": "NO_GO",
                "benefit_path": None,
                "median_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "worst_repetition_ratios": {
                    "request_throughput_rps": 1.0,
                    "p95_ttft_ns": 1.0,
                    "p95_itl_ns": 1.0,
                    "p99_ttft_ns": 1.0,
                    "p99_itl_ns": 1.0,
                    "p99_e2e_ns": 1.0,
                    "maximum_decode_gap_ns": 1.0,
                    "peak_cuda_reserved_bytes": 1.0,
                    "peak_kv_bytes": 1.0,
                },
                "guard_failures": [],
            },
        },
    }
    _write_json(root / "summary.json", summary)
    (root / "report.md").write_text(
        "# Production Arrival-Load Gate\n\n"
        "Classification: `NO_GO`\n"
    )
    required = (
        "run_manifest.json",
        "calibration_manifest.jsonl",
        "calibration_rows.jsonl",
        "workload_manifest.jsonl",
        "request_timeline.jsonl",
        "scheduler_trace.jsonl",
        "memory_trace.jsonl",
        "case_rows.jsonl",
        "summary.json",
        "report.md",
        "source_evidence.json",
        "source.patch",
        "source_snapshot.tar.gz",
    )
    _write_json(
        root / "artifact_hashes.json",
        {name: _sha256_file(root / name) for name in required},
    )
    return temporary, root


def _refresh_hash(root: Path, name: str) -> None:
    hashes = json.loads((root / "artifact_hashes.json").read_text())
    hashes[name] = _sha256_file(root / name)
    _write_json(root / "artifact_hashes.json", hashes)


def _add_warmup_request(root: Path) -> None:
    workload_rows = verifier._read_jsonl(
        root / "workload_manifest.jsonl"
    )
    warmup_request_id = "steady_moderate-warmup-0000"
    workload_rows.append({
        **workload_rows[0],
        "request_id": warmup_request_id,
        "warmup": True,
        "arrival_offset_ns": -1_000_000_000,
    })
    _write_jsonl(root / "workload_manifest.jsonl", workload_rows)
    _refresh_hash(root, "workload_manifest.jsonl")

    timeline_rows = verifier._read_jsonl(
        root / "request_timeline.jsonl"
    )
    warmup_rows = []
    for row in timeline_rows:
        warmup_rows.append({
            **row,
            "request_id": warmup_request_id,
            "seq_id": row["seq_id"] + 1000,
            "scheduled_arrival_ns": 0,
            "actual_arrival_ns": 10,
            "first_scheduled_ns": 20,
            "first_token_ns": 30,
            "token_timestamps_ns": [30, 40],
            "completion_ns": 40,
        })
    _write_jsonl(
        root / "request_timeline.jsonl",
        warmup_rows + timeline_rows,
    )
    _refresh_hash(root, "request_timeline.jsonl")


def test_verifier_does_not_import_harness_aggregation():
    source = VERIFY_PATH.read_text()
    assert "import arrival_load_gate" not in source
    assert "from arrival_load_gate" not in source


def test_output_equality_uses_recorded_case_metadata_not_case_id_format():
    manifest = {
        "required_scenarios": ["steady_moderate"],
        "measured_repetitions": 1,
    }
    common = {
        "scenario": "steady_moderate",
        "repetition": 0,
        "request_id": "request-0",
    }
    rows = [
        {
            **common,
            "case_id": "steady_moderate__P0__r0",
            "policy": "P0",
            "output_token_ids": [1, 2],
        },
        {
            **common,
            "case_id": "steady_moderate__P2__r0",
            "policy": "P2",
            "output_token_ids": [1, 3],
        },
        {
            **common,
            "case_id": "steady_moderate__P3__r0",
            "policy": "P3",
            "output_token_ids": [1, 2],
        },
    ]
    try:
        verifier._verify_output_equality(rows, manifest)
    except ValueError as exc:
        assert "output token mismatch" in str(exc)
    else:
        raise AssertionError("canonical case-id output drift was missed")


def test_smoke_summary_is_lifecycle_only():
    rows = [
        {
            "case_id": "lifecycle_smoke__P0__r0",
            "policy": "P0",
            "scenario": "lifecycle_smoke",
            "repetition": 0,
            "status": "PASS",
            "correctness": {
                "exact_outputs": True,
                "complete_requests": True,
                "no_starvation": True,
                "valid_lifecycle": True,
                "stable_p0_outputs": True,
            },
        },
        {
            "case_id": "lifecycle_smoke__P2__r0",
            "policy": "P2",
            "scenario": "lifecycle_smoke",
            "repetition": 0,
            "status": "PASS",
            "correctness": {
                "exact_outputs": True,
                "complete_requests": True,
                "no_starvation": True,
                "valid_lifecycle": True,
                "stable_p0_outputs": True,
            },
        },
    ]
    summary = verifier._smoke_summary(rows)
    assert summary == {
        "classification": "SMOKE_ONLY",
        "lifecycle_complete": True,
        "exact_outputs": True,
        "case_count": 2,
    }


def test_verifier_recomputes_complete_artifact():
    temporary, root = _complete_artifact()
    try:
        result = verifier.verify_run(root, write_output=True)
        assert result["classification"] == "NO_GO"
        assert (
            root / "independent-verify/summary.json"
        ).is_file()
        assert (
            root / "independent-verify/verify.exitcode"
        ).read_text().strip() == "0"
    finally:
        temporary.cleanup()


def test_verifier_excludes_warmup_requests_from_case_metrics():
    temporary, root = _complete_artifact()
    try:
        _add_warmup_request(root)
        result = verifier.verify_run(root, write_output=False)
        assert result["classification"] == "NO_GO"
    finally:
        temporary.cleanup()


def test_verifier_still_validates_warmup_request_lifecycle():
    temporary, root = _complete_artifact()
    try:
        _add_warmup_request(root)
        timeline_rows = verifier._read_jsonl(
            root / "request_timeline.jsonl"
        )
        warmup = next(
            row for row in timeline_rows
            if row["request_id"] == "steady_moderate-warmup-0000"
        )
        warmup["first_scheduled_ns"] = warmup["actual_arrival_ns"] - 1
        _write_jsonl(root / "request_timeline.jsonl", timeline_rows)
        _refresh_hash(root, "request_timeline.jsonl")

        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "impossible timestamp ordering" in str(exc)
        else:
            raise AssertionError("invalid warmup lifecycle must fail")
    finally:
        temporary.cleanup()


def test_verifier_rejects_summary_tampering_even_when_rehashed():
    temporary, root = _complete_artifact()
    try:
        recorded = json.loads((root / "summary.json").read_text())
        recorded["classification"] = "GO"
        _write_json(root / "summary.json", recorded)
        _refresh_hash(root, "summary.json")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "classification disagreement" in str(exc)
        else:
            raise AssertionError("tampered summary must be rejected")
    finally:
        temporary.cleanup()


def test_verifier_rejects_truncated_jsonl_and_duplicate_ports():
    temporary, root = _complete_artifact()
    try:
        path = root / "request_timeline.jsonl"
        path.write_bytes(path.read_bytes()[:-1])
        _refresh_hash(root, "request_timeline.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "final newline" in str(exc)
        else:
            raise AssertionError("truncated JSONL must fail")
    finally:
        temporary.cleanup()

    temporary, root = _complete_artifact()
    try:
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifest["process_port_pairs"][1][
            "tinyvllm_dist_port"
        ] = manifest["process_port_pairs"][0]["tinyvllm_dist_port"]
        manifest["process_port_pairs"][1][
            "master_port"
        ] = manifest["process_port_pairs"][0]["master_port"]
        _write_json(root / "run_manifest.json", manifest)
        _refresh_hash(root, "run_manifest.json")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "duplicate process port pair" in str(exc)
        else:
            raise AssertionError("duplicate ports must fail")
    finally:
        temporary.cleanup()


def test_verifier_rejects_rehashed_source_output_and_scheduler_tampering():
    temporary, root = _complete_artifact()
    try:
        (root / "source.patch").write_bytes(b"tampered patch\n")
        _refresh_hash(root, "source.patch")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "source patch" in str(exc)
        else:
            raise AssertionError("rehashed source patch tampering must fail")
    finally:
        temporary.cleanup()

    temporary, root = _complete_artifact()
    try:
        rows = [
            json.loads(line)
            for line in (root / "request_timeline.jsonl").read_text().splitlines()
        ]
        candidate = next(
            row for row in rows
            if row["case_id"] == "P2-steady_moderate-r0"
        )
        candidate["output_token_ids"][-1] = 99
        _write_jsonl(root / "request_timeline.jsonl", rows)
        _refresh_hash(root, "request_timeline.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "output token mismatch" in str(exc)
        else:
            raise AssertionError("candidate output tampering must fail")
    finally:
        temporary.cleanup()

    temporary, root = _complete_artifact()
    try:
        rows = [
            json.loads(line)
            for line in (root / "scheduler_trace.jsonl").read_text().splitlines()
        ]
        rows = [
            row for row in rows
            if row["case_id"] != "P3-steady_moderate-r2"
        ]
        _write_jsonl(root / "scheduler_trace.jsonl", rows)
        _refresh_hash(root, "scheduler_trace.jsonl")
        try:
            verifier.verify_run(root, write_output=False)
        except ValueError as exc:
            assert "missing scheduler trace" in str(exc)
        else:
            raise AssertionError("missing scheduler trace must fail")
    finally:
        temporary.cleanup()


def main():
    test_verifier_does_not_import_harness_aggregation()
    test_output_equality_uses_recorded_case_metadata_not_case_id_format()
    test_smoke_summary_is_lifecycle_only()
    test_verifier_recomputes_complete_artifact()
    test_verifier_excludes_warmup_requests_from_case_metrics()
    test_verifier_still_validates_warmup_request_lifecycle()
    test_verifier_rejects_summary_tampering_even_when_rehashed()
    test_verifier_rejects_truncated_jsonl_and_duplicate_ports()
    test_verifier_rejects_rehashed_source_output_and_scheduler_tampering()
    print("arrival load verifier tests passed")


if __name__ == "__main__":
    main()
