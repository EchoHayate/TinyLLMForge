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
    / "exact_burst_one_phase_lease_local_journal_gate.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "exact_burst_one_phase_lease_local_journal_gate_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _performance_rows(gate):
    rows = []
    for repetition in range(gate.PERFORMANCE_REPETITIONS):
        for context in gate.CONTEXTS:
            eligible = 15
            for policy in gate.POLICIES:
                candidate = policy == "lease_local_delta"
                prepare = {
                    "2k": 70_000,
                    "4k": 90_000,
                    "8k": 120_000,
                }[context]
                if candidate:
                    prepare = {
                        "2k": 35_000,
                        "4k": 36_000,
                        "8k": 40_000,
                    }[context]
                tpot = 1_000_000 if not candidate else 970_000
                rows.append({
                    "schema": gate.PERFORMANCE_ROW_SCHEMA,
                    "run_tag": "synthetic-r1",
                    "source_sha": "a" * 40,
                    "policy": policy,
                    "context": context,
                    "repetition": repetition,
                    "order_position": gate.policy_order(
                        repetition,
                        gate.CONTEXTS.index(context),
                    ).index(policy),
                    "prompt_digest": context,
                    "generated_tokens": 128,
                    "output_tokens": list(range(128)),
                    "prepare_ns": [prepare] * eligible,
                    "ttft_ns": 2_000_000,
                    "tpot_samples_ns": [tpot] * 127,
                    "e2e_ns": 128_000_000,
                    "output_tokens_per_second": (
                        1000.0 if not candidate else 1030.0
                    ),
                    "cuda_peak_allocated_bytes": 1_000_000,
                    "cuda_peak_reserved_bytes": 2_000_000,
                    "target_model_forwards": 128,
                    "graph_replays": 127,
                    "d2h_calls": 16,
                    "d2h_bytes": 1024,
                    "eligible_bursts": eligible,
                    "generic_journal_captures": (
                        0 if candidate else eligible
                    ),
                    "one_phase_attempts": (
                        eligible if candidate else 0
                    ),
                    "one_phase_captures": (
                        eligible if candidate else 0
                    ),
                    "one_phase_commits": (
                        eligible if candidate else 0
                    ),
                    "one_phase_rollbacks": 0,
                    "one_phase_published_blocks": (
                        1 if candidate else 0
                    ),
                    "one_phase_fallbacks": {},
                })
    return rows


def _correctness_rows(gate):
    rows = []
    for context in gate.CONTEXTS:
        for policy in gate.POLICIES:
            for point in gate.SAMPLING_POINTS:
                rows.append({
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
                })
    return rows


def test_gate_inventory_and_policy_order_are_fixed():
    gate = _load_module()

    assert gate.POLICIES == ("generic", "lease_local_delta")
    assert gate.CONTEXTS == ("2k", "4k", "8k")
    assert gate.PERFORMANCE_REPETITIONS == 10
    assert gate.PERFORMANCE_ROW_COUNT == 60
    assert gate.CORRECTNESS_ROW_COUNT == 24
    assert (
        "tools/profile_exact_greedy_decode_burst.py"
        in gate.SOURCE_FILES
    )
    assert gate.policy_order(0, 0) == gate.POLICIES
    assert gate.policy_order(1, 0) == tuple(
        reversed(gate.POLICIES)
    )


def test_gate_installs_its_context_contract_in_base_profiler(
    monkeypatch,
):
    gate = _load_module()
    from tools import profile_exact_greedy_decode_burst as base

    monkeypatch.setattr(
        base,
        "CONTEXT_CASES",
        (("short", 256, 128),),
    )

    gate._bind_base_context_contract(base)

    assert base.CONTEXT_CASES == gate.CONTEXT_CASES
    assert base._case_shape("2k") == (2048, 128)
    assert base._case_shape("4k") == (4096, 128)
    assert base._case_shape("8k") == (8192, 128)


def test_gate_classifies_complete_beneficial_evidence_go():
    gate = _load_module()

    summary = gate.summarize_evidence(
        _performance_rows(gate),
        _correctness_rows(gate),
    )

    assert summary["classification"] == (
        "GO_EXACT_BURST_ONE_PHASE_LEASE_LOCAL_JOURNAL"
    )
    assert summary["8k_prepare_median_improvement_pct"] >= 50
    assert summary["aggregate_prepare_median_improvement_pct"] >= 35
    assert summary["aggregate_tpot_median_improvement_pct"] >= 1
    assert summary["candidate_generic_journal_captures"] == 0
    assert summary["candidate_one_phase_fallbacks"] == 0
    assert summary["candidate_one_phase_rollbacks"] == 0


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("token", "NO_GO_CORRECTNESS"),
        ("forward", "NO_GO_CORRECTNESS"),
        ("fallback", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("generic_capture", "NO_GO_TRANSACTIONAL_SAFETY"),
        ("prepare", "NO_GO_PERFORMANCE"),
        ("tpot", "NO_GO_PERFORMANCE"),
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
        if row["policy"] == "lease_local_delta"
        and row["context"] == "8k"
    )
    if mutation == "token":
        candidate["output_tokens"][-1] = -1
    elif mutation == "forward":
        candidate["target_model_forwards"] += 1
    elif mutation == "fallback":
        candidate["one_phase_fallbacks"] = {
            "unexpected": 1
        }
    elif mutation == "generic_capture":
        candidate["generic_journal_captures"] = 1
    elif mutation == "prepare":
        for row in performance:
            if (
                row["policy"] == "lease_local_delta"
                and row["context"] == "8k"
            ):
                row["prepare_ns"] = [80_000] * 15
    elif mutation == "tpot":
        for row in performance:
            if row["policy"] == "lease_local_delta":
                row["tpot_samples_ns"] = [1_000_000] * 127
    elif mutation == "memory":
        for row in performance:
            if row["policy"] == "lease_local_delta":
                row["cuda_peak_reserved_bytes"] = 2_100_000

    summary = gate.summarize_evidence(
        performance,
        correctness,
    )
    assert summary["classification"] == expected


def test_gate_rejects_incomplete_or_duplicate_inventory():
    gate = _load_module()
    performance = _performance_rows(gate)
    correctness = _correctness_rows(gate)

    with pytest.raises(ValueError, match="incomplete"):
        gate.summarize_evidence(
            performance[:-1],
            correctness,
        )
    with pytest.raises(ValueError, match="duplicate"):
        gate.summarize_evidence(
            performance + [dict(performance[0])],
            correctness,
        )


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
