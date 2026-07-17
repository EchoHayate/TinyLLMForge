"""Dependency-light tests for the speculation profitability router gate."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
GATE_PATH = THIS_DIR / "speculation_router_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "speculation_router_gate_under_test",
    os.fspath(GATE_PATH),
)
gate = importlib.util.module_from_spec(SPEC)
sys.modules["speculation_router_gate_under_test"] = gate
SPEC.loader.exec_module(gate)


def _capability():
    return {
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


def _complete_fixture():
    source_evidence = {
        "schema_version": 1,
        "base_commit": "1" * 40,
        "dirty": False,
        "patch_path": "source.patch",
        "patch_sha256": "2" * 64,
        "patch_size_bytes": 0,
        "owned_roots": list(gate.OWNED_SOURCE_ROOTS),
        "files": [],
        "tree_sha256": "3" * 64,
    }
    source_preflight = {
        "schema_version": 1,
        "source_tree_sha256": source_evidence["tree_sha256"],
    }
    manifest = gate.build_controlled_manifest(
        source_evidence=source_evidence,
        source_preflight=source_preflight,
        model_path="/models/Qwen3-0.6B",
        model_identifier="Qwen3-0.6B",
        host="synthetic-host",
        python_bin="python3",
        torch_version="2.4.1",
        cuda_version="12.1",
        flash_attn_version="2.6.3",
        gpu_name="Synthetic GPU",
        bf16_supported=True,
        run_tag="synthetic-controlled",
    )
    case_rows = []
    event_rows = []
    router_rows = []
    next_port = 10000
    for case in gate.CONTROLLED_CASE_MATRIX:
        baseline_elapsed = 1.0
        accepted_count = int(case["expected_accepted_count"])
        for policy in gate.CONTROLLED_POLICIES:
            elapsed_s = baseline_elapsed
            if (
                policy == "routed_native"
                and case["draft_len"] >= 2
                and accepted_count >= 2
            ):
                elapsed_s = 0.90
            elif policy == "always_native":
                elapsed_s = 0.95
            process = {
                "returncode": 0,
                "tinyvllm_dist_port": next_port,
                "master_port": next_port + 1,
            }
            next_port += 2
            manifest["process_port_pairs"].append({
                "case_id": case["case_id"],
                "policy": policy,
                "tinyvllm_dist_port": process[
                    "tinyvllm_dist_port"
                ],
                "master_port": process["master_port"],
            })
            row = {
                "case_id": case["case_id"],
                "policy": policy,
                "status": "PASS",
                "process": process,
                "source_tree_sha256": source_evidence[
                    "tree_sha256"
                ],
                "elapsed_s": elapsed_s,
                "baseline_elapsed_s": baseline_elapsed,
                "output_tokens": 16,
                "output_tokens_per_s": 16.0 / elapsed_s,
                "output_token_sha256": (
                    f"output-{case['case_id']}"
                ),
                "continuation_token_sha256": (
                    f"continuation-{case['case_id']}"
                ),
                "accepted_count": accepted_count,
                "target_forward_count": (
                    16 - accepted_count
                    if policy != "baseline"
                    else 16
                ),
            }
            if policy == "oracle":
                row["comparison"] = {
                    "status": "PASS",
                    "target_token_match": True,
                    "accepted_prefix_match": True,
                    "metadata_match": True,
                    "continuation_token_match": True,
                    "finite": True,
                    "logits_within_tolerance": True,
                    "kv_within_tolerance": True,
                    "continuation_steps": 16,
                }
            case_rows.append(row)

        for policy in ("always_native", "routed_native"):
            if (
                policy == "routed_native"
                and case["draft_len"] <= 1
            ):
                continue
            event_rows.append({
                "case_id": case["case_id"],
                "policy": policy,
                "draft_len": case["draft_len"],
                "accepted_count": accepted_count,
                "accepted_kv_rematerialization": {
                    "decode_calls": 0,
                    "rematerialized_tokens": [],
                    "elapsed_ms": 0.0,
                },
                "accepted_kv_copy_calls": 0,
                "accepted_kv_replay_calls": 0,
                "target_forward_count": 1,
                "verifier_commit_ms": 1.0,
            })

        route = (
            "baseline_short_draft"
            if case["draft_len"] <= 1
            else "native_multi_token"
        )
        router_rows.append({
            "case_id": case["case_id"],
            "policy": "routed_native",
            "route": route,
            "draft_len": case["draft_len"],
            "route_fallback_reason": None,
            "speculative_reservation_attempted": (
                route == "native_multi_token"
            ),
            "spec_verify_prepare_calls": (
                1 if route == "native_multi_token" else 0
            ),
            "spec_verify_forward_calls": (
                1 if route == "native_multi_token" else 0
            ),
            "target_forward_count": (
                1 if route == "native_multi_token" else 0
            ),
        })
    return (
        manifest,
        _capability(),
        case_rows,
        event_rows,
        router_rows,
    )


def _classify(fixture):
    return gate.classify_controlled_gate(*fixture)


def test_complete_controlled_evidence_is_ready():
    summary = _classify(_complete_fixture())
    assert (
        summary["classification"]
        == "READY_FOR_REAL_DRAFTER_GATE"
    )
    assert summary["exactness_pass"] is True
    assert summary["replay_elimination_pass"] is True
    assert summary["router_isolation_pass"] is True


def test_no_profitable_region_is_no_go():
    fixture = list(_complete_fixture())
    rows = fixture[2]
    for row in rows:
        if row["policy"] == "routed_native":
            row["elapsed_s"] = row["baseline_elapsed_s"]
            row["output_tokens_per_s"] = (
                row["output_tokens"] / row["elapsed_s"]
            )
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert "no_profitable_k_ge_2_region" in summary["reasons"]


def test_short_route_mutation_is_no_go():
    fixture = list(_complete_fixture())
    short = next(
        row for row in fixture[4]
        if row["draft_len"] == 1
    )
    short["speculative_reservation_attempted"] = True
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert summary["router_isolation_pass"] is False


def test_native_replay_is_no_go():
    fixture = list(_complete_fixture())
    fixture[3][0]["accepted_kv_replay_calls"] = 1
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert summary["replay_elimination_pass"] is False


def test_semantic_mismatch_is_no_go():
    fixture = list(_complete_fixture())
    oracle = next(
        row for row in fixture[2]
        if row["policy"] == "oracle"
    )
    oracle["comparison"]["kv_within_tolerance"] = False
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert summary["exactness_pass"] is False


def test_missing_row_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[2].pop()
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_failed_process_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[2][0]["process"]["returncode"] = 2
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_duplicate_port_pair_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[0]["process_port_pairs"][1].update(
        fixture[0]["process_port_pairs"][0]
    )
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_nonfinite_performance_is_incomplete():
    fixture = list(_complete_fixture())
    fixture[2][0]["elapsed_s"] = math.nan
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "INCOMPLETE"


def test_lifecycle_regression_is_no_go():
    fixture = list(_complete_fixture())
    row = next(
        row for row in fixture[2]
        if row["policy"] == "routed_native"
        and row["case_id"] == "k2-zero-current"
    )
    row["elapsed_s"] = 1.06
    summary = _classify(tuple(fixture))
    assert summary["classification"] == "NO_GO"
    assert any(
        reason.startswith("required_lifecycle_regression:")
        for reason in summary["reasons"]
    )


def main():
    test_complete_controlled_evidence_is_ready()
    test_no_profitable_region_is_no_go()
    test_short_route_mutation_is_no_go()
    test_native_replay_is_no_go()
    test_semantic_mismatch_is_no_go()
    test_missing_row_is_incomplete()
    test_failed_process_is_incomplete()
    test_duplicate_port_pair_is_incomplete()
    test_nonfinite_performance_is_incomplete()
    test_lifecycle_regression_is_no_go()
    print("speculation router gate tests passed")


if __name__ == "__main__":
    main()
