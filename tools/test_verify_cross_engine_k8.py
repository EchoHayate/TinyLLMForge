from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.cross_engine_k8_workload import (
    REQUIRED_ARMS,
    build_workload_manifest,
)
from tools.verify_cross_engine_k8 import verify_bundle


SOURCE = "a" * 40


def _write_json(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_values(arm):
    if arm == "tinyllmforge_exact_k8":
        return 94.0, 106.0
    if arm == "tinyllmforge_host_greedy":
        return 120.0, 83.0
    return 100.0, 100.0


def _write_valid_bundle(root: Path, *, classification="GO_CROSS_ENGINE_ADVANTAGE"):
    root.mkdir()
    workload = build_workload_manifest("b" * 64)
    controller = {
        "schema_version": "cross-engine-k8.controller.v1",
        "source_revision": SOURCE,
        "eligible_arms": list(REQUIRED_ARMS),
        "storage_valid": True,
        "terminal_receipts_valid": True,
        "remote_allocated_bytes": 1_000,
        "remote_hard_limit_bytes": 20 * 1024**3,
    }
    environment = {
        "schema_version": "cross-engine-k8.environment.v1",
        "source_revision": SOURCE,
        "model_inventory_sha256": "b" * 64,
        "vllm": {"version": "0.10.1"},
    }
    case_rows = []
    correctness_rows = []
    reference = {
        case["context"]: list(range(128))
        for case in workload["cases"]
    }
    for repetition in range(7):
        for case in workload["cases"]:
            for arm in REQUIRED_ARMS:
                tpot, throughput = _metric_values(arm)
                case_rows.append({
                    "arm": arm,
                    "context": case["context"],
                    "repetition": repetition,
                    "median_tpot_ns": tpot,
                    "p95_tpot_ns": tpot,
                    "p99_tpot_ns": tpot,
                    "ttft_ns": 1_000.0,
                    "e2e_ns": 10_000.0,
                    "output_tokens_per_second": throughput,
                    "peak_gpu_memory_bytes": 1_000.0,
                    "peak_rss_bytes": 2_000.0,
                    "performance_eligible": True,
                    "token_ids": reference[case["context"]],
                })
                correctness_rows.append({
                    "arm": arm,
                    "context": case["context"],
                    "repetition": repetition,
                    "token_ids": reference[case["context"]],
                    "matches_reference": True,
                })
    _write_json(root / "controller_manifest.json", controller)
    _write_json(root / "environment_manifest.json", environment)
    _write_json(root / "workload_manifest.json", workload)
    _write_jsonl(root / "case_rows.jsonl", case_rows)
    _write_jsonl(root / "correctness_rows.jsonl", correctness_rows)
    _write_json(root / "comparison.json", {"producer": True})
    _write_json(root / "summary.json", {"producer": True})
    _write_json(root / "gate.json", {"classification": classification})
    hashed = (
        "controller_manifest.json",
        "environment_manifest.json",
        "workload_manifest.json",
        "case_rows.jsonl",
        "correctness_rows.jsonl",
        "comparison.json",
        "summary.json",
        "gate.json",
    )
    (root / "manifest.sha256").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in hashed),
        encoding="utf-8",
    )
    return root


def test_verifier_accepts_unavailable_protected_metric_as_incomplete(tmp_path):
    bundle = _write_valid_bundle(
        tmp_path / "bundle",
        classification="INCOMPLETE",
    )
    rows = [
        json.loads(line)
        for line in (bundle / "case_rows.jsonl").read_text().splitlines()
        if line
    ]
    for row in rows:
        if row["arm"] == "vllm_default_greedy":
            row["peak_gpu_memory_bytes"] = "NOT_EXPOSED"
    _write_jsonl(bundle / "case_rows.jsonl", rows)
    hashed = (
        "controller_manifest.json",
        "environment_manifest.json",
        "workload_manifest.json",
        "case_rows.jsonl",
        "correctness_rows.jsonl",
        "comparison.json",
        "summary.json",
        "gate.json",
    )
    (bundle / "manifest.sha256").write_text(
        "".join(f"{_sha256(bundle / name)}  {name}\n" for name in hashed),
        encoding="utf-8",
    )

    result = verify_bundle(bundle, expected_source=SOURCE)

    assert result["valid"] is True
    assert result["recomputed_classification"] == "INCOMPLETE"
    assert "metric_unavailable:peak_gpu_memory_ratio" in result[
        "gate_reasons"
    ]


def test_verifier_recomputes_go_without_trusting_gate(tmp_path):
    bundle = _write_valid_bundle(
        tmp_path / "bundle",
        classification="INCOMPLETE",
    )

    result = verify_bundle(bundle, expected_source=SOURCE)

    assert result["recomputed_classification"] == (
        "GO_CROSS_ENGINE_ADVANTAGE"
    )
    assert result["producer_agrees"] is False
    assert result["valid"] is False


def test_verifier_accepts_matching_complete_bundle(tmp_path):
    bundle = _write_valid_bundle(tmp_path / "bundle")

    result = verify_bundle(bundle, expected_source=SOURCE)

    assert result["valid"] is True
    assert result["producer_agrees"] is True
    assert result["strongest_vllm_arm"] == "vllm_default_greedy"


def test_verifier_detects_case_row_tampering(tmp_path):
    bundle = _write_valid_bundle(tmp_path / "bundle")
    with (bundle / "case_rows.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    result = verify_bundle(bundle, expected_source=SOURCE)

    assert result["valid"] is False
    assert "MANIFEST_DIGEST_MISMATCH" in result["reasons"]


def test_verifier_marks_missing_terminal_receipt_incomplete(tmp_path):
    bundle = _write_valid_bundle(tmp_path / "bundle")
    controller_path = bundle / "controller_manifest.json"
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["terminal_receipts_valid"] = False
    _write_json(controller_path, controller)
    lines = (bundle / "manifest.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    lines[0] = f"{_sha256(controller_path)}  controller_manifest.json"
    (bundle / "manifest.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    result = verify_bundle(bundle, expected_source=SOURCE)

    assert result["recomputed_classification"] == "INCOMPLETE"
    assert "terminal_receipts_valid" in result["gate_reasons"]


def test_verifier_rejects_wrong_source_even_with_valid_hashes(tmp_path):
    bundle = _write_valid_bundle(tmp_path / "bundle")

    result = verify_bundle(bundle, expected_source="c" * 40)

    assert result["valid"] is False
    assert "SOURCE_REVISION_MISMATCH" in result["reasons"]
