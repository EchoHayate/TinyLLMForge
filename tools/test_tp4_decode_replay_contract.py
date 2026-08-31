#!/usr/bin/env python3
"""Dependency-light tests for the TP4 decode replay qualification contract."""

from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "tp4_decode_replay_contract.py"


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "tp4_decode_replay_contract",
        CONTRACT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load contract: {CONTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _performance_rows():
    rows = []
    for case in contract.build_case_matrix():
        graph = case["arm"] == "graph"
        rows.append({
            "row_id": f"{case['case_id']}:performance",
            "case_id": case["case_id"],
            "pair_id": case["pair_id"],
            "workload": case["workload"],
            "repetition": case["repetition"],
            "arm": case["arm"],
            "output_tokens_per_second": 106.0 if graph else 100.0,
            "qps": 10.6 if graph else 10.0,
            "median_tpot_ms": 94.0 if graph else 100.0,
            "p95_tpot_ms": 96.0 if graph else 100.0,
            "p99_tpot_ms": 98.0 if graph else 100.0,
            "median_e2e_ms": 950.0 if graph else 1000.0,
            "p99_e2e_ms": 1010.0 if graph else 1000.0,
            "ttft_ms": 101.0 if graph else 100.0,
            "initialization_ms": 1010.0 if graph else 1000.0,
        })
    return rows


def _correctness_rows():
    rows = []
    for workload in contract.WORKLOADS:
        for repetition in range(contract.MEASURED_REPETITIONS):
            pair_id = f"{workload}__r{repetition}"
            outputs = [
                {
                    "request_id": f"{pair_id}:request-{index}",
                    "prompt_sha256": f"{index + 1:064x}",
                    "output_token_ids": [7, 11, 13],
                    "output_length": 3,
                    "stop_reason": "length",
                }
                for index in range(
                    contract.WORKLOADS[workload]["concurrency"]
                )
            ]
            rows.append({
                "row_id": f"{pair_id}:correctness",
                "pair_id": pair_id,
                "workload": workload,
                "repetition": repetition,
                "eager_outputs": copy.deepcopy(outputs),
                "graph_outputs": copy.deepcopy(outputs),
                "exact_match": True,
            })
    return rows


def _rank_dispatch_rows():
    rows = []
    for case in contract.build_case_matrix():
        graph = case["arm"] == "graph"
        for rank in contract.RANKS:
            rows.append({
                "row_id": f"{case['case_id']}:step-0:rank-{rank}",
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "arm": case["arm"],
                "phase": "measured",
                "step_index": 0,
                "rank": rank,
                "world_size": 4,
                "feature_enabled": graph,
                "graph_eligible": graph,
                "dispatch": "graph" if graph else "eager",
                "graph_identity_sha256": "a" * 64 if graph else None,
                "cache_state": "ready" if graph else "absent",
                "capture_attempted": False,
                "fallback_reason": None if graph else "enforce_eager",
                "graph_replay_count": 10 if graph else 0,
            })
    return rows


def _rank_collective_rows():
    rows = []
    for case in contract.build_case_matrix():
        for rank in contract.RANKS:
            rows.append({
                "row_id": f"{case['case_id']}:collectives:rank-{rank}",
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "arm": case["arm"],
                "rank": rank,
                "world_size": 4,
                "collective_count": 130,
                "collective_order_sha256": "b" * 64,
                "complete": True,
            })
    return rows


def _rank_lifecycle_rows():
    rows = []
    for case in contract.build_case_matrix():
        for rank in contract.RANKS:
            rows.append({
                "row_id": f"{case['case_id']}:lifecycle:rank-{rank}",
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "arm": case["arm"],
                "rank": rank,
                "world_size": 4,
                "complete": True,
                "exit_code": 0,
                "process_group_destroyed": True,
                "replay_exception": False,
            })
    return rows


def _memory_rows():
    rows = []
    for case in contract.build_case_matrix():
        graph = case["arm"] == "graph"
        for rank in contract.RANKS:
            rows.append({
                "row_id": f"{case['case_id']}:memory:rank-{rank}",
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "arm": case["arm"],
                "rank": rank,
                "peak_allocated_bytes": (
                    70_100_000_000 if graph else 70_000_000_000
                ),
                "peak_reserved_bytes": (
                    71_100_000_000 if graph else 71_000_000_000
                ),
            })
    return rows


def _capture_cost_rows():
    rows = []
    for case in contract.build_case_matrix():
        if case["arm"] != "graph":
            continue
        for rank in contract.RANKS:
            rows.append({
                "row_id": f"{case['case_id']}:capture:rank-{rank}",
                "case_id": case["case_id"],
                "pair_id": case["pair_id"],
                "workload": case["workload"],
                "repetition": case["repetition"],
                "arm": case["arm"],
                "rank": rank,
                "graph_identity_sha256": "d" * 64,
                "capture_duration_ns": 50_000_000,
                "static_bytes": 1_000_000,
                "allocated_delta_bytes": 100_000_000,
                "reserved_delta_bytes": 100_000_000,
                "complete": True,
            })
    return rows


def _evidence():
    return {
        "performance_rows": _performance_rows(),
        "correctness_rows": _correctness_rows(),
        "rank_dispatch_rows": _rank_dispatch_rows(),
        "rank_collective_rows": _rank_collective_rows(),
        "rank_lifecycle_rows": _rank_lifecycle_rows(),
        "memory_rows": _memory_rows(),
        "capture_cost_rows": _capture_cost_rows(),
    }


def test_case_matrix_is_paired_and_frozen():
    rows = contract.build_case_matrix()
    assert len(rows) == 3 * 5 * 2
    assert {row["workload"] for row in rows} == {"Q0", "Q1", "Q2"}
    for workload, expected in contract.WORKLOADS.items():
        matching = [row for row in rows if row["workload"] == workload]
        assert len(matching) == 10
        assert all(row["profile"] == expected for row in matching)
        assert {row["arm"] for row in matching} == {"eager", "graph"}
        for repetition in range(contract.MEASURED_REPETITIONS):
            ordered = sorted(
                (
                    row
                    for row in matching
                    if row["repetition"] == repetition
                ),
                key=lambda row: row["order_index"],
            )
            expected_order = (
                ["eager", "graph"]
                if repetition % 2 == 0
                else ["graph", "eager"]
            )
            assert [row["arm"] for row in ordered] == expected_order


def test_passing_evidence_justifies_stage1():
    result = contract.classify(**_evidence())
    assert result["classification"] == "GO_STAGE1_JUSTIFIED"
    assert result["failed_gates"] == []
    assert result["aggregate"]["output_throughput_ratio"] >= 1.05
    assert result["aggregate"]["median_tpot_ratio"] <= 0.95
    assert result["replay_coverage"] == 1.0
    assert result["maximum_added_peak_allocated_bytes"] == 100_000_000
    assert result["maximum_added_peak_reserved_bytes"] == 100_000_000
    assert result["capture_amortization_tokens"] > 0


def test_missing_evidence_is_incomplete():
    evidence = _evidence()
    evidence["performance_rows"].pop()
    result = contract.classify(**evidence)
    assert result["classification"] == "INCOMPLETE"
    assert "performance_case_matrix_incomplete" in result["failed_gates"]


def test_correctness_and_lifecycle_precede_performance():
    evidence = _evidence()
    evidence["correctness_rows"][0]["exact_match"] = False
    for row in evidence["performance_rows"]:
        if row["arm"] == "graph":
            row["output_tokens_per_second"] = 50.0
    result = contract.classify(**evidence)
    assert (
        result["classification"]
        == "NO_GO_CORRECTNESS_OR_LIFECYCLE"
    )
    assert "exact_output_mismatch" in result["failed_gates"]


def test_cross_rank_dispatch_disagreement_is_correctness_failure():
    evidence = _evidence()
    graph_row = next(
        row
        for row in evidence["rank_dispatch_rows"]
        if row["arm"] == "graph" and row["rank"] == 3
    )
    graph_row["dispatch"] = "eager"
    graph_row["fallback_reason"] = "capture_failed"
    result = contract.classify(**evidence)
    assert (
        result["classification"]
        == "NO_GO_CORRECTNESS_OR_LIFECYCLE"
    )
    assert "rank_dispatch_disagreement" in result["failed_gates"]


def test_collective_order_disagreement_is_correctness_failure():
    evidence = _evidence()
    evidence["rank_collective_rows"][3][
        "collective_order_sha256"
    ] = "c" * 64
    result = contract.classify(**evidence)
    assert (
        result["classification"]
        == "NO_GO_CORRECTNESS_OR_LIFECYCLE"
    )
    assert "collective_order_disagreement" in result["failed_gates"]


def test_low_replay_coverage_has_distinct_classification():
    evidence = _evidence()
    graph_rows = [
        row
        for row in evidence["rank_dispatch_rows"]
        if row["arm"] == "graph"
    ]
    for row in graph_rows[:16]:
        row["dispatch"] = "eager"
        row["cache_state"] = "observing"
        row["fallback_reason"] = "cold_identity"
    result = contract.classify(**evidence)
    assert result["classification"] == "NO_GO_MECHANISM_NOT_EXERCISED"
    assert "replay_coverage" in result["failed_gates"]


def test_warmup_and_measured_steps_have_distinct_rank_groups():
    evidence = _evidence()
    warmup_rows = []
    for row in evidence["rank_dispatch_rows"]:
        warmup = copy.deepcopy(row)
        warmup["row_id"] = warmup["row_id"].replace(
            ":step-0:",
            ":warmup:step-0:",
        )
        warmup["phase"] = "warmup"
        warmup["graph_eligible"] = False
        warmup["dispatch"] = "eager"
        warmup["cache_state"] = "observing"
        warmup["fallback_reason"] = "cold_identity"
        warmup["graph_replay_count"] = 0
        warmup_rows.append(warmup)
    evidence["rank_dispatch_rows"].extend(warmup_rows)
    result = contract.classify(**evidence)
    assert result["classification"] == "GO_STAGE1_JUSTIFIED"


def test_capture_identity_disagreement_is_correctness_failure():
    evidence = _evidence()
    evidence["capture_cost_rows"][3][
        "graph_identity_sha256"
    ] = "e" * 64
    result = contract.classify(**evidence)
    assert (
        result["classification"]
        == "NO_GO_CORRECTNESS_OR_LIFECYCLE"
    )
    assert "capture_identity_disagreement" in result["failed_gates"]


def test_each_performance_and_cost_threshold_can_fail():
    mutations = (
        ("output_tokens_per_second", 90.0, "output_throughput"),
        ("median_tpot_ms", 110.0, "median_tpot"),
        ("p99_e2e_ms", 1100.0, "p99_e2e"),
        ("ttft_ms", 110.0, "ttft"),
    )
    for field, value, gate_fragment in mutations:
        evidence = _evidence()
        for row in evidence["performance_rows"]:
            if row["arm"] == "graph" and row["workload"] == "Q0":
                row[field] = value
        result = contract.classify(**evidence)
        assert result["classification"] == "NO_GO_PERFORMANCE"
        assert any(
            gate_fragment in gate for gate in result["failed_gates"]
        )

    evidence = _evidence()
    for row in evidence["memory_rows"]:
        if row["arm"] == "graph" and row["rank"] == 0:
            row["peak_reserved_bytes"] += 600 * 1024 * 1024
    result = contract.classify(**evidence)
    assert result["classification"] == "NO_GO_PERFORMANCE"
    assert "peak_reserved_memory" in result["failed_gates"]


def test_nonfinite_and_duplicate_rows_are_incomplete():
    evidence = _evidence()
    evidence["performance_rows"][0][
        "output_tokens_per_second"
    ] = math.nan
    result = contract.classify(**evidence)
    assert result["classification"] == "INCOMPLETE"
    assert "nonfinite_or_invalid_evidence" in result["failed_gates"]

    evidence = _evidence()
    evidence["correctness_rows"].append(
        copy.deepcopy(evidence["correctness_rows"][0])
    )
    result = contract.classify(**evidence)
    assert result["classification"] == "INCOMPLETE"
    assert "duplicate_row_id" in result["failed_gates"]


def main() -> None:
    tests = (
        test_case_matrix_is_paired_and_frozen,
        test_passing_evidence_justifies_stage1,
        test_missing_evidence_is_incomplete,
        test_correctness_and_lifecycle_precede_performance,
        test_cross_rank_dispatch_disagreement_is_correctness_failure,
        test_collective_order_disagreement_is_correctness_failure,
        test_low_replay_coverage_has_distinct_classification,
        test_warmup_and_measured_steps_have_distinct_rank_groups,
        test_capture_identity_disagreement_is_correctness_failure,
        test_each_performance_and_cost_threshold_can_fail,
        test_nonfinite_and_duplicate_rows_are_incomplete,
    )
    for test in tests:
        test()
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
