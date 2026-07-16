"""Dependency-light tests for the native verifier evidence gate."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GATE_PATH = os.path.join(_THIS_DIR, "native_verifier_gate.py")
_SPEC = importlib.util.spec_from_file_location(
    "native_verifier_gate_under_test",
    _GATE_PATH,
)
gate = importlib.util.module_from_spec(_SPEC)
sys.modules["native_verifier_gate_under_test"] = gate
_SPEC.loader.exec_module(gate)


def _complete_fixture():
    manifest = gate.build_manifest(
        source_commit="synthetic",
        source_dirty=False,
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
        torch_version="2.7.0",
        cuda_version="12.6",
        flash_attn_version="2.7.4",
        gpu_name="Synthetic GPU",
        bf16_supported=True,
        run_tag="synthetic-run",
    )
    capability = {
        "status": "PASS",
        "rows": [
            {
                "dtype": dtype,
                "query_len": query_len,
                "block_case": block_case,
                "gqa": True,
                "output_match": True,
                "kv_match": True,
                "future_row_masked": True,
                "finite": True,
            }
            for dtype in ("torch.float16", "torch.bfloat16")
            for query_len in (1, 3, 7, 15)
            for block_case in ("one_block", "cross_block")
        ],
    }
    case_rows = []
    event_rows = []
    for case_index, case in enumerate(gate.CASE_MATRIX):
        token_hash = f"token-hash-{case['case_id']}"
        accepted_count = {
            "zero": 0,
            "one": 1,
            "partial": min(2, case["draft_len"]),
            "full": case["draft_len"],
        }[case["acceptance_case"]]
        baseline_elapsed = 1.0
        native_elapsed = 1.005 if case["draft_len"] == 1 else 0.8
        native_verify_ms = 1.0 + case["draft_len"] * 0.01
        legacy_verify_ms = native_verify_ms + (
            0.0 if accepted_count <= 1 else 0.5
        )
        legacy_decode_calls = max(0, accepted_count - 1)
        process_base = {
            "returncode": 0,
            "tinyvllm_dist_port": 23000 + case_index * 20,
            "master_port": 23001 + case_index * 20,
            "stdout_path": "logs/stdout.log",
            "stderr_path": "logs/stderr.log",
        }
        for policy_index, policy in enumerate(gate.POLICIES):
            row = {
                "case_id": case["case_id"],
                "policy": policy,
                "status": "PASS",
                "source_commit": "synthetic",
                "source_dirty": False,
                "output_token_sha256": token_hash,
                "continuation_token_sha256": token_hash,
                "accepted_tokens": list(range(accepted_count)),
                "block_table_after": [0],
                "sequence_token_sha256": token_hash,
                "elapsed_s": (
                    native_elapsed
                    if policy == "native"
                    else baseline_elapsed
                ),
                "output_tokens": 64,
                "output_tokens_per_s": (
                    64 / native_elapsed
                    if policy == "native"
                    else 64 / baseline_elapsed
                ),
                "verifier_commit_ms": (
                    legacy_verify_ms
                    if policy == "legacy_rematerialize"
                    else native_verify_ms
                ),
                "max_allocated_gpu_memory_bytes": 1024 + case_index,
                "process": {
                    **process_base,
                    "tinyvllm_dist_port": (
                        process_base["tinyvllm_dist_port"] + policy_index * 2
                    ),
                    "master_port": (
                        process_base["master_port"] + policy_index * 2
                    ),
                },
            }
            if policy == "oracle":
                row["comparison"] = {
                    "status": "PASS",
                    "reasons": [],
                    "target_token_match": True,
                    "accepted_prefix_match": True,
                    "metadata_match": True,
                    "continuation_token_match": True,
                    "continuation_steps": 16,
                    "finite": True,
                    "max_logit_abs_error": 0.0,
                    "max_kv_abs_error": 0.0,
                    "logits_within_tolerance": True,
                    "kv_within_tolerance": True,
                }
            case_rows.append(row)
        event_rows.append({
            "case_id": case["case_id"],
            "policy": "native",
            "draft_len": case["draft_len"],
            "accepted_count": accepted_count,
            "zero_accept_included_in_throughput": accepted_count == 0,
            "accepted_kv_rematerialization": {
                "decode_calls": 0,
                "rematerialized_tokens": [],
                "elapsed_ms": 0.0,
            },
            "accepted_kv_copy_calls": 0,
            "accepted_kv_replay_calls": 0,
            "target_forward_count": 1 + int(case["draft_len"] > 1),
            "legacy_decode_replay_calls": legacy_decode_calls,
            "legacy_total_target_forwards": (
                1 + int(case["draft_len"] > 1) + legacy_decode_calls
            ),
            "native_total_target_forwards": (
                1 + int(case["draft_len"] > 1)
            ),
            "verifier_commit_ms": native_verify_ms,
            "legacy_verifier_commit_ms": legacy_verify_ms,
        })
    return manifest, capability, case_rows, event_rows


def test_case_matrix_covers_required_dimensions():
    cases = gate.CASE_MATRIX
    assert {case["draft_len"] for case in cases} == {1, 4, 8, 16}
    assert {"zero", "one", "partial", "full"} <= {
        case["acceptance_case"] for case in cases
    }
    assert any(case["eos_case"] for case in cases)
    assert any(case["output_budget_case"] for case in cases)
    assert {"current_block", "one_new_block", "multi_block_context"} <= {
        case["block_case"] for case in cases
    }
    assert all(case["continuation_steps"] >= 16 for case in cases)
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_eos_case_uses_a_dedicated_real_eos_prompt():
    eos_cases = [case for case in gate.CASE_MATRIX if case["eos_case"]]

    assert len(eos_cases) == 1
    assert eos_cases[0]["prompt"] == gate._EOS_PROMPT
    assert eos_cases[0]["prompt"] != gate._PROMPT


def test_manifest_freezes_scope_thresholds_and_case_matrix():
    manifest, _, _, _ = _complete_fixture()
    assert manifest["classification_on_success"] == (
        "READY_FOR_PERFORMANCE_GATE"
    )
    assert manifest["policies"] == list(gate.POLICIES)
    assert manifest["case_matrix"] == list(gate.CASE_MATRIX)
    assert manifest["thresholds"]["k1_max_regression_fraction"] == 0.01
    assert manifest["required_artifacts"] == list(
        gate.REQUIRED_ARTIFACTS
    )
    assert manifest["claim_boundaries"]
    assert manifest["source_dirty"] is False


def test_capability_specs_cover_query_dtype_and_block_dimensions():
    fp16 = gate.build_capability_specs(bf16_supported=False)
    both = gate.build_capability_specs(bf16_supported=True)

    assert len(fp16) == 8
    assert len(both) == 16
    assert {row["query_len"] for row in both} == {1, 3, 7, 15}
    assert {row["dtype"] for row in both} == {
        "torch.float16",
        "torch.bfloat16",
    }
    assert {row["block_case"] for row in both} == {
        "one_block",
        "cross_block",
    }
    assert all(row["gqa"] is True for row in both)


def test_complete_evidence_is_ready_for_performance_gate():
    manifest, capability, rows, events = _complete_fixture()
    summary = gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )
    assert summary["classification"] == "READY_FOR_PERFORMANCE_GATE"
    assert summary["exactness_pass"] is True
    assert summary["replay_elimination_pass"] is True
    assert summary["performance_direction_pass"] is True
    assert summary["memory_is_diagnostic_only"] is True


def test_semantic_or_replay_failure_is_no_go():
    manifest, capability, rows, events = _complete_fixture()
    oracle_row = next(
        row for row in rows if row["policy"] == "oracle"
    )
    oracle_row["comparison"]["status"] = "NO_GO"
    oracle_row["comparison"]["reasons"] = ["continuation token mismatch"]
    summary = gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )
    assert summary["classification"] == "NO_GO"
    assert "continuation token mismatch" in summary["reasons"]

    manifest, capability, rows, events = _complete_fixture()
    events[0]["accepted_kv_replay_calls"] = 1
    summary = gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )
    assert summary["classification"] == "NO_GO"
    assert any("replay" in reason for reason in summary["reasons"])


def test_missing_duplicate_or_failed_process_is_incomplete():
    manifest, capability, rows, events = _complete_fixture()
    assert gate.classify_gate(
        manifest,
        capability,
        rows[:-1],
        events,
    )["classification"] == "INCOMPLETE"

    manifest, capability, rows, events = _complete_fixture()
    rows.append(dict(rows[0]))
    assert gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )["classification"] == "INCOMPLETE"

    manifest, capability, rows, events = _complete_fixture()
    rows[0]["process"]["returncode"] = 1
    assert gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )["classification"] == "INCOMPLETE"


def test_capability_or_missing_numeric_evidence_is_incomplete():
    manifest, capability, rows, events = _complete_fixture()
    capability["status"] = "INCOMPLETE"
    assert gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )["classification"] == "INCOMPLETE"

    manifest, capability, rows, events = _complete_fixture()
    oracle_row = next(
        row for row in rows if row["policy"] == "oracle"
    )
    del oracle_row["comparison"]["max_kv_abs_error"]
    assert gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )["classification"] == "INCOMPLETE"


def test_performance_qualification_is_strict_but_memory_is_not_a_gate():
    manifest, capability, rows, events = _complete_fixture()
    for row in rows:
        if row["policy"] == "native" and row["case_id"].startswith("k1-"):
            row["elapsed_s"] = 1.02
    summary = gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )
    assert summary["classification"] == "NO_GO"
    assert any("K=1" in reason for reason in summary["reasons"])

    manifest, capability, rows, events = _complete_fixture()
    for event in events:
        event["max_allocated_gpu_memory_bytes"] = 10**15
    assert gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )["classification"] == "READY_FOR_PERFORMANCE_GATE"


def _write_complete_artifacts(out_dir: Path):
    manifest, capability, rows, events = _complete_fixture()
    summary = gate.classify_gate(
        manifest,
        capability,
        rows,
        events,
    )
    report = gate.render_report(
        manifest,
        capability,
        rows,
        events,
        summary,
    )
    payloads = {
        "capability.json": capability,
        "case_rows.json": rows,
        "event_rows.json": events,
        "summary.json": summary,
    }
    for name, payload in payloads.items():
        (out_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    (out_dir / "report.md").write_text(report)
    manifest["artifact_hashes"] = {
        name: gate.sha256_file(out_dir / name)
        for name in payloads
    }
    manifest["artifact_hashes"]["report.md"] = gate.sha256_file(
        out_dir / "report.md"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_six_file_artifact_verifier_rejects_tampering():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_complete_artifacts(out_dir)
        verified = gate.verify_artifacts(out_dir)
        assert verified["classification"] == (
            "READY_FOR_PERFORMANCE_GATE"
        )

        (out_dir / "report.md").write_text("tampered\n")
        try:
            gate.verify_artifacts(out_dir)
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("tampered report must fail")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_complete_artifacts(out_dir)
        (out_dir / "capability.json").unlink()
        try:
            gate.verify_artifacts(out_dir)
        except ValueError as exc:
            assert "missing artifact" in str(exc)
        else:
            raise AssertionError("missing artifact must fail")


def test_artifact_verifier_rejects_dirty_source_and_manifest_drift():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_complete_artifacts(out_dir)
        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_dirty"] = True
        manifest_path.write_text(json.dumps(manifest))
        try:
            gate.verify_artifacts(out_dir)
        except ValueError as exc:
            assert "source_dirty" in str(exc)
        else:
            raise AssertionError("dirty canonical evidence must fail")

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        _write_complete_artifacts(out_dir)
        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["thresholds"]["k1_max_regression_fraction"] = 0.5
        manifest_path.write_text(json.dumps(manifest))
        try:
            gate.verify_artifacts(out_dir)
        except ValueError as exc:
            assert "manifest" in str(exc)
        else:
            raise AssertionError("manifest drift must fail")


def main():
    tests = (
        test_case_matrix_covers_required_dimensions,
        test_eos_case_uses_a_dedicated_real_eos_prompt,
        test_manifest_freezes_scope_thresholds_and_case_matrix,
        test_capability_specs_cover_query_dtype_and_block_dimensions,
        test_complete_evidence_is_ready_for_performance_gate,
        test_semantic_or_replay_failure_is_no_go,
        test_missing_duplicate_or_failed_process_is_incomplete,
        test_capability_or_missing_numeric_evidence_is_incomplete,
        test_performance_qualification_is_strict_but_memory_is_not_a_gate,
        test_six_file_artifact_verifier_rejects_tampering,
        test_artifact_verifier_rejects_dirty_source_and_manifest_drift,
    )
    for test in tests:
        test()
    print("native verifier gate tests passed")


if __name__ == "__main__":
    main()
