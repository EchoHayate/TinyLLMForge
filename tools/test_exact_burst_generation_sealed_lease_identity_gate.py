from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "tools"
    / "exact_burst_generation_sealed_lease_identity_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "exact_burst_generation_sealed_lease_identity_gate_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _performance_rows(gate):
    rows = []
    lifecycle = {
        "2k": (100_000, 80_000),
        "4k": (120_000, 90_000),
        "8k": (200_000, 140_000),
    }
    for repetition in range(gate.PERFORMANCE_REPETITIONS):
        for context_index, context in enumerate(gate.CONTEXTS):
            for policy in gate.policy_order(
                repetition,
                context_index,
            ):
                candidate = policy == "generation_sealed"
                lifecycle_ns = lifecycle[context][candidate]
                eligible = 15
                rows.append({
                    "schema": gate.PERFORMANCE_ROW_SCHEMA,
                    "run_tag": "synthetic-r1",
                    "source_sha": "a" * 40,
                    "policy": policy,
                    "context": context,
                    "repetition": repetition,
                    "order_position": gate.policy_order(
                        repetition,
                        context_index,
                    ).index(policy),
                    "prompt_digest": context,
                    "generated_tokens": 128,
                    "output_tokens": list(range(128)),
                    "lease_grant_ns": [lifecycle_ns // 2] * eligible,
                    "scheduler_lifecycle_ns": [
                        lifecycle_ns
                    ] * eligible,
                    "ttft_ns": 2_000_000,
                    "tpot_samples_ns": [
                        990_000 if candidate else 1_000_000
                    ] * 127,
                    "e2e_ns": 128_000_000,
                    "output_tokens_per_second": (
                        1010.0 if candidate else 1000.0
                    ),
                    "cuda_peak_allocated_bytes": 1_000_000,
                    "cuda_peak_reserved_bytes": 2_000_000,
                    "target_model_forwards": 128,
                    "graph_replays": 127,
                    "d2h_calls": 16,
                    "d2h_bytes": 1024,
                    "eligible_bursts": eligible,
                    "identity_seal_cold_captures": (
                        1 if candidate else 0
                    ),
                    "identity_seal_hot_reuses": (
                        eligible - 1 if candidate else 0
                    ),
                    "identity_seal_validations": (
                        eligible * 3 if candidate else 0
                    ),
                    "identity_seal_fallbacks": {},
                    "exact_burst_failures": 0,
                    "one_phase_rollbacks": 0,
                })
    return rows


def _correctness_rows(gate):
    return [
        {
            "schema": gate.CORRECTNESS_ROW_SCHEMA,
            "run_tag": "synthetic-r1",
            "source_sha": "a" * 40,
            "policy": policy,
            "context": context,
            "sampling_point": point,
            "output_token_ids": list(range(128)),
            "sampled_logits": [1.0, 2.0, 3.0],
            "sampled_argmax": 2,
            "target_model_forwards": 128,
            "graph_replays": 127,
            "d2h_calls": 16,
            "d2h_bytes": 1024,
        }
        for context in gate.CONTEXTS
        for policy in gate.POLICIES
        for point in gate.SAMPLING_POINTS
    ]


def test_gate_inventory_policy_order_and_workload_are_fixed():
    gate = _load_module()

    assert gate.POLICIES == (
        "full_identity",
        "generation_sealed",
    )
    assert gate.CONTEXTS == ("2k", "4k", "8k")
    assert gate.PERFORMANCE_REPETITIONS == 10
    assert gate.PERFORMANCE_ROW_COUNT == 60
    assert gate.CORRECTNESS_ROW_COUNT == 24
    assert gate.policy_order(0, 0) == gate.POLICIES
    assert gate.policy_order(1, 0) == tuple(
        reversed(gate.POLICIES)
    )
    manifest = gate.build_workload_manifest(
        run_tag="synthetic-r1",
        source_sha="a" * 40,
    )
    assert manifest["execution_shape"] == "one_phase_k8"
    assert manifest["split_phase_enabled"] is False
    assert manifest["lease_local_delta_journal_enabled"] is True
    assert manifest["only_variable"] == (
        "exact_greedy_decode_burst_generation_sealed_identity"
    )


def test_hardware_policy_configs_differ_only_by_generation_seal():
    gate = _load_module()

    baseline = gate.policy_runtime_config("full_identity")
    candidate = gate.policy_runtime_config("generation_sealed")

    assert baseline == {
        "exact_greedy_decode_burst": True,
        "exact_greedy_decode_burst_tokens": 8,
        "exact_greedy_decode_burst_split_phase": False,
        "exact_greedy_decode_burst_ragged_coalescing": False,
        "exact_greedy_decode_burst_continuation": False,
        "exact_greedy_decode_burst_lease_local_delta_journal": True,
        "exact_greedy_decode_burst_generation_sealed_identity": False,
    }
    assert candidate == {
        **baseline,
        "exact_greedy_decode_burst_generation_sealed_identity": True,
    }


def test_scheduler_lifecycle_pairs_grant_and_commit_samples():
    gate = _load_module()

    assert gate.combine_scheduler_lifecycle_samples(
        [10, 20, 30],
        [1, 2, 3],
    ) == [11, 22, 33]
    with pytest.raises(ValueError, match="inventory mismatch"):
        gate.combine_scheduler_lifecycle_samples([10], [1, 2])
    with pytest.raises(ValueError, match="non-negative integer"):
        gate.combine_scheduler_lifecycle_samples([10], [-1])


def test_gate_classifies_complete_beneficial_evidence_go():
    gate = _load_module()

    summary = gate.summarize_evidence(
        _performance_rows(gate),
        _correctness_rows(gate),
    )

    assert summary["schema"] == gate.GATE_SCHEMA
    assert summary["classification"] == (
        "GO_EXACT_BURST_GENERATION_SEALED_LEASE_IDENTITY"
    )
    assert summary["8k_lifecycle_median_improvement_pct"] >= 25
    assert summary["8k_lifecycle_p95_improvement_pct"] >= 25
    assert (
        summary["aggregate_lifecycle_median_improvement_pct"]
        >= 15
    )
    assert (
        summary["aggregate_lifecycle_p95_improvement_pct"]
        >= 15
    )
    assert summary["aggregate_tpot_median_improvement_pct"] >= 0.5
    assert summary["aggregate_tpot_p95_improvement_pct"] >= 0.5
    assert summary["candidate_identity_seal_fallbacks"] == 0
    assert summary["candidate_exact_burst_failures"] == 0
    assert summary["candidate_one_phase_rollbacks"] == 0
    assert summary["candidate_hot_reuse_accounting"] is True


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("token", "NO_GO_CORRECTNESS"),
        ("logit", "NO_GO_CORRECTNESS"),
        ("forward", "NO_GO_CORRECTNESS"),
        ("fallback", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("hot_reuse", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("eligible_inventory", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("baseline_seal_counter", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("failure", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("rollback", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("prompt", "NO_GO_CORRECTNESS"),
        ("lifecycle", "NO_GO_PERFORMANCE"),
        ("tpot", "NO_GO_PERFORMANCE"),
        ("ttft", "NO_GO_PERFORMANCE"),
        ("memory", "NO_GO_PERFORMANCE"),
    ),
)
def test_gate_rejects_each_invalid_evidence_class(
    mutation,
    expected,
):
    gate = _load_module()
    performance = _performance_rows(gate)
    correctness = _correctness_rows(gate)
    candidate = next(
        row
        for row in performance
        if row["policy"] == "generation_sealed"
        and row["context"] == "8k"
    )
    if mutation == "token":
        candidate["output_tokens"][-1] = -1
    elif mutation == "logit":
        correctness[-1]["sampled_logits"][-1] = 4.0
    elif mutation == "forward":
        candidate["target_model_forwards"] += 1
    elif mutation == "fallback":
        candidate["identity_seal_fallbacks"] = {
            "untracked_block_table": 1
        }
    elif mutation == "hot_reuse":
        candidate["identity_seal_hot_reuses"] -= 1
    elif mutation == "eligible_inventory":
        candidate["eligible_bursts"] -= 1
        candidate["identity_seal_hot_reuses"] -= 1
    elif mutation == "baseline_seal_counter":
        baseline = next(
            row
            for row in performance
            if row["policy"] == "full_identity"
        )
        baseline["identity_seal_validations"] = 1
    elif mutation == "failure":
        candidate["exact_burst_failures"] = 1
    elif mutation == "rollback":
        candidate["one_phase_rollbacks"] = 1
    elif mutation == "prompt":
        candidate["prompt_digest"] = "different"
    elif mutation == "lifecycle":
        for row in performance:
            if (
                row["policy"] == "generation_sealed"
                and row["context"] == "8k"
            ):
                row["scheduler_lifecycle_ns"] = [190_000] * 15
    elif mutation == "tpot":
        for row in performance:
            if row["policy"] == "generation_sealed":
                row["tpot_samples_ns"] = [1_000_000] * 127
    elif mutation == "ttft":
        for row in performance:
            if row["policy"] == "generation_sealed":
                row["ttft_ns"] = 2_100_000
    elif mutation == "memory":
        for row in performance:
            if row["policy"] == "generation_sealed":
                row["cuda_peak_reserved_bytes"] = 2_100_000

    summary = gate.summarize_evidence(
        performance,
        correctness,
    )
    assert summary["classification"] == expected


def test_gate_rejects_incomplete_duplicate_or_wrong_policy_rows():
    gate = _load_module()
    performance = _performance_rows(gate)
    correctness = _correctness_rows(gate)

    with pytest.raises(ValueError, match="incomplete"):
        gate.summarize_evidence(performance[:-1], correctness)
    with pytest.raises(ValueError, match="duplicate"):
        gate.summarize_evidence(
            performance + [dict(performance[0])],
            correctness,
        )
    performance[0]["policy"] = "unknown"
    with pytest.raises(ValueError, match="policy"):
        gate.summarize_evidence(performance, correctness)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    (
        ("performance", "schema", "wrong", "schema mismatch"),
        ("correctness", "schema", "wrong", "schema mismatch"),
        ("performance", "run_tag", "other", "run tag authority"),
        ("correctness", "source_sha", "b" * 40, "source SHA authority"),
        ("performance", "ttft_ns", math.nan, "finite"),
        (
            "correctness",
            "sampled_logits",
            [1.0, math.inf],
            "finite",
        ),
    ),
)
def test_gate_rejects_malformed_or_mixed_authority_rows(
    target,
    field,
    value,
    message,
):
    gate = _load_module()
    performance = _performance_rows(gate)
    correctness = _correctness_rows(gate)
    rows = performance if target == "performance" else correctness
    rows[0][field] = value

    with pytest.raises(ValueError, match=message):
        gate.summarize_evidence(performance, correctness)
