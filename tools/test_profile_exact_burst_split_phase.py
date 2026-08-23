#!/usr/bin/env python3
"""Contracts for the split-phase K8 exact-burst profiler."""

from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from tools.profile_exact_burst_split_phase import (
    CASE_SCHEMA_VERSION,
    CONTEXT_CASES,
    CORRECTNESS_SCHEMA_VERSION,
    POLICIES,
    POLICY_CONFIGS,
    SAMPLING_POINTS,
    SOURCE_FILES,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    WORKLOAD_SCHEMA_VERSION,
    build_workload_manifest,
    correctness_identities,
    correctness_point_uses_burst_trace,
    correctness_trace_for_step,
    _construct_llm,
    performance_identities,
    policy_order,
    summarize_split_phase_observations,
    validate_case_row,
    validate_correctness_rows,
    write_float32_sidecar,
)


RUN_TAG = "20260823-qwen3-06b-split-phase-test"
SOURCE_COMMIT = "a" * 40


def _capture_receipt(*, correctness_trace: bool = False) -> dict:
    return {
        "graph_identity_sha256": "c" * 64,
        "graph_generation": 7,
        "capture_duration_ns": 1_000_000,
        "allocated_delta_bytes": 400_000,
        "reserved_delta_bytes": 2_000_000,
        "retained_static_bytes": 904_000,
        "scratch_block_count": 1,
        "correctness_trace": correctness_trace,
    }


def _burst_summary(
    policy: str,
    *,
    correctness_trace: bool = False,
) -> dict:
    enabled = POLICY_CONFIGS[policy]["enabled"]
    split = POLICY_CONFIGS[policy]["split"]
    width = POLICY_CONFIGS[policy]["width"]
    split_commits = 15 if split else 0
    commits = (
        split_commits
        if split
        else (math.ceil(127 / width) if enabled else 0)
    )
    tail = 7 if split else 0
    summary = {
        "attempts": commits + tail,
        "acceptances": commits + max(0, tail - 1),
        "target_model_forwards": (
            split_commits * 8 if split else (127 if enabled else 0)
        ),
        "graph_replays": (
            split_commits * 8 if split else (127 if enabled else 0)
        ),
        "intermediate_token_d2h_calls": 0,
        "final_token_d2h_calls": (
            0 if split else commits
        ),
        "final_token_d2h_bytes": (
            0 if split else (127 * 8 if enabled else 0)
        ),
        "sampled_logit_d2h_calls": (
            (
                POLICY_CONFIGS[policy].get(
                    "correctness_sampled_logit_d2h_calls",
                    len(SAMPLING_POINTS) - 1,
                )
                if enabled
                else 0
            )
            if correctness_trace
            else 0
        ),
        "output_budget_clipped": (
            max(0, tail - 1) if split else int(enabled)
        ),
        "block_boundary_clipped": (
            0 if split else int(enabled)
        ),
        "commits": commits,
        "committed_tokens": (
            split_commits * 8 if split else (127 if enabled else 0)
        ),
        "prefix_commits": split_commits,
        "suffix_commits": split_commits,
        "prefix_committed_tokens": split_commits * 4,
        "suffix_committed_tokens": split_commits * 4,
        "prefix_publication_tickets": split_commits,
        "suffix_publication_tickets": split_commits,
        "prefix_token_d2h_calls": split_commits,
        "suffix_token_d2h_calls": split_commits,
        "prefix_token_d2h_bytes": split_commits * 32,
        "suffix_token_d2h_bytes": split_commits * 32,
        "prefix_phase_waits": split_commits,
        "suffix_phase_waits": split_commits,
        "suffix_drains": split_commits,
        "failures": 0,
        "quarantines": 0,
        "pending_leases": 0,
        "maximum_host_visible_gap_ns": (
            3_000_000 if enabled else 0
        ),
        "requested_width_histogram": (
            (
                {str(width): commits + max(0, tail - 1)}
                if split
                else {str(width): commits}
            )
            if enabled
            else {}
        ),
        "authorized_width_histogram": (
            (
                {
                    "2": 1,
                    "3": 1,
                    "4": 1,
                    "5": 1,
                    "6": 1,
                    "7": 1,
                    "8": split_commits,
                }
                if split
                else {str(width): commits}
            )
            if enabled
            else {}
        ),
        "fallback_counts": (
            (
                {
                    "insufficient_output_budget": 1,
                    "split_phase_requires_k8": max(0, tail - 1),
                }
                if tail
                else {}
            )
            if split
            else {}
        ),
        "split_phase_failure_counts": {},
        "quarantine_reason": None,
        "capture_receipts": (
            [
                _capture_receipt(
                    correctness_trace=correctness_trace
                )
            ]
            if enabled
            else []
        ),
    }
    return summary


def _split_inventory() -> dict:
    return {
        "parent_lease_count": 15,
        "prefix_row_count": 15,
        "suffix_row_count": 15,
        "prefix_ticket_count": 15,
        "suffix_ticket_count": 15,
        "replay_count": 120,
        "prefix_d2h_calls": 15,
        "suffix_d2h_calls": 15,
        "prefix_d2h_bytes": 480,
        "suffix_d2h_bytes": 480,
        "prefix_pending_suffix_count": 15,
        "suffix_cleared_count": 15,
        "unexpected_scheduler_calls": 0,
        "host_visible_gaps_ns": [2_000_000, 3_000_000] * 15,
    }


def _case_row(policy: str) -> dict:
    prompt_tokens, generated_tokens = {
        name: (prompt, generated)
        for name, prompt, generated in CONTEXT_CASES
    }["short"]
    enabled = POLICY_CONFIGS[policy]["enabled"]
    split = POLICY_CONFIGS[policy]["split"]
    width = POLICY_CONFIGS[policy]["width"]
    commits = _burst_summary(policy)["commits"]
    tail = generated_tokens - 1 - (
        _burst_summary(policy)["committed_tokens"]
    )
    decode_steps = (
        (
            commits
            + tail * 2
            - _burst_summary(policy)["fallback_counts"].get(
                "insufficient_output_budget",
                0,
            )
        )
        if split
        else (commits if enabled else generated_tokens - 1)
    )
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "selectable": POLICY_CONFIGS[policy]["selectable"],
        "burst_width": width,
        "repetition": 0,
        "context_bucket": "short",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "output_token_ids": list(range(generated_tokens)),
        "output_text_sha256": "b" * 64,
        "ttft_ns": 10_000_000,
        "e2e_ns": 140_000_000,
        "amortized_tpot_samples_ns": [1_000_000] * 127,
        "amortized_tpot_median_ns": 1_000_000,
        "amortized_tpot_p95_ns": 1_000_000,
        "amortized_tpot_p99_ns": 1_000_000,
        "decode_host_ns": [900_000] * decode_steps,
        "decode_cuda_ns": [700_000] * decode_steps,
        "output_tokens_per_second": 914.285714,
        "host_visible_burst_gaps_ns": (
            _split_inventory()["host_visible_gaps_ns"]
            if split
            else ([3_000_000] * commits if enabled else [])
        ),
        "maximum_host_visible_burst_gap_ns": (
            3_000_000 if enabled else 0
        ),
        "cuda_peak_allocated_bytes": 1_000_000,
        "cuda_peak_reserved_bytes": 2_000_000,
        "capture_duration_ns": 1_000_000 if enabled else 0,
        "capture_allocated_delta_bytes": 400_000 if enabled else 0,
        "capture_reserved_delta_bytes": 2_000_000 if enabled else 0,
        "capture_retained_static_bytes": 904_000 if enabled else 0,
        "reserved_scratch_blocks": 1 if enabled else 0,
        "correctness_trace": False,
        "split_phase_inventory": (
            _split_inventory()
            if split
            else {
                key: ([] if key == "host_visible_gaps_ns" else 0)
                for key in _split_inventory()
            }
        ),
        "exact_greedy_decode_burst_summary":
            _burst_summary(policy),
    }


def _phase_row(parent: str, phase: str) -> dict:
    return {
        "split_phase_attempted": True,
        "split_phase_accepted": True,
        "parent_lease_identity_sha256": parent,
        "prefix_ticket_identity_sha256": (
            parent[:63] + "1"
        ),
        "suffix_ticket_identity_sha256": (
            parent[:63] + "2"
        ),
        "phase_published": phase,
        "phase_token_count": 4,
        "replay_count": 8,
        "prefix_d2h_calls": 1,
        "suffix_d2h_calls": 1,
        "prefix_d2h_bytes": 32,
        "suffix_d2h_bytes": 32,
        "pending_suffix": phase == "prefix",
        "scheduler_schedule_calls": 1 if phase == "prefix" else 0,
        "host_visible_gap_ns": (
            2_000_000 if phase == "prefix" else 3_000_000
        ),
    }


def _correctness_row(
    *,
    run_dir: Path,
    policy: str,
    bucket: str,
    sampling_point: str,
) -> dict:
    prompt_tokens, generated_tokens = {
        name: (prompt, generated)
        for name, prompt, generated in CONTEXT_CASES
    }[bucket]
    sidecar = write_float32_sidecar(
        run_dir,
        f"logits/{bucket}-{policy}-{sampling_point}.f32",
        (1.0, 2.0, 3.0, 4.0),
    )
    burst_sample = correctness_point_uses_burst_trace(
        policy,
        sampling_point,
    )
    output_index = {
        "prefill-final": 0,
        "decode-first": 1,
        "decode-middle": generated_tokens // 2,
        "decode-final": generated_tokens - 1,
    }[sampling_point]
    return {
        "schema_version": CORRECTNESS_SCHEMA_VERSION,
        "run_tag": RUN_TAG,
        "source_commit": SOURCE_COMMIT,
        "policy": policy,
        "context_bucket": bucket,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "sampling_point": sampling_point,
        "output_token_ids": list(range(generated_tokens)),
        "output_text_sha256": "b" * 64,
        "logits_path": sidecar["path"],
        "logits_shape": [1, 4],
        "logits_element_count": sidecar["element_count"],
        "logits_byte_length": sidecar["byte_length"],
        "logits_sha256": sidecar["sha256"],
        "correctness_trace": True,
        "trace_identity": (
            "gate-only-exact-burst-split-phase-correctness-v1"
        ),
        "trace_graph_identity_sha256": (
            "c" * 64 if burst_sample else None
        ),
        "selected_replay_ordinal": (
            (output_index - 1) % POLICY_CONFIGS[policy]["width"]
            if burst_sample
            else None
        ),
        "sampled_logit_d2h_calls": 1 if burst_sample else 0,
        "exact_greedy_decode_burst_summary": _burst_summary(
            policy,
            correctness_trace=POLICY_CONFIGS[policy]["enabled"],
        ),
    }


def test_frozen_schema_matrix_and_source_inventory() -> None:
    assert CASE_SCHEMA_VERSION == "exact-burst-split-phase.case.v1"
    assert CORRECTNESS_SCHEMA_VERSION == (
        "exact-burst-split-phase.correctness.v1"
    )
    assert SUMMARY_SCHEMA_VERSION == (
        "exact-burst-split-phase.summary.v1"
    )
    assert WORKLOAD_SCHEMA_VERSION == (
        "exact-burst-split-phase.workload.v1"
    )
    assert SOURCE_SCHEMA_VERSION == (
        "exact-burst-split-phase.source.v1"
    )
    assert POLICIES == (
        "host_greedy",
        "decode_burst_k4",
        "decode_burst_k8",
        "decode_burst_k8_split_phase",
    )
    assert len(performance_identities(repetitions=5)) == 60
    assert len(correctness_identities()) == 48
    assert set(policy_order(4, 2)) == set(POLICIES)
    assert (
        "tools/profile_exact_burst_split_phase.py"
        in SOURCE_FILES
    )
    assert (
        "tools/exact_burst_split_phase_verify.py"
        in SOURCE_FILES
    )


def test_only_split_arm_enables_split_phase() -> None:
    assert tuple(
        policy
        for policy in POLICIES
        if POLICY_CONFIGS[policy]["split"]
    ) == ("decode_burst_k8_split_phase",)
    assert POLICY_CONFIGS[
        "decode_burst_k8_split_phase"
    ]["width"] == 8
    assert POLICY_CONFIGS[
        "decode_burst_k8_split_phase"
    ]["selectable"] is True
    assert correctness_trace_for_step(
        "host_greedy",
        emitted_total=1,
        generated_tokens=128,
    ) is False
    assert correctness_trace_for_step(
        "decode_burst_k4",
        emitted_total=1,
        generated_tokens=128,
    ) is True
    assert correctness_point_uses_burst_trace(
        "decode_burst_k8_split_phase",
        "decode-first",
    ) is True
    assert correctness_point_uses_burst_trace(
        "decode_burst_k8_split_phase",
        "decode-middle",
    ) is True
    assert correctness_point_uses_burst_trace(
        "decode_burst_k8_split_phase",
        "decode-final",
    ) is False


def test_workload_manifest_freezes_sixty_rows_and_split_costs() -> None:
    manifest = build_workload_manifest(
        model="/models/qwen3",
        run_tag=RUN_TAG,
        source_commit=SOURCE_COMMIT,
        gpu_memory_utilization=0.5,
        environment={"python_version": "3.10"},
    )
    assert manifest["performance_row_count"] == 60
    assert manifest["correctness_row_count"] == 48
    assert manifest["correctness_sampling_points"] == list(
        SAMPLING_POINTS
    )
    assert manifest["policy_configs"][
        "decode_burst_k8_split_phase"
    ]["split"] is True


def test_runtime_configuration_enables_split_only_for_split_arm() -> None:
    calls = []

    def fake_llm(model, **kwargs):
        calls.append((model, kwargs))
        return SimpleNamespace()

    with patch.dict(
        sys.modules,
        {"tinyvllm": SimpleNamespace(LLM=fake_llm)},
    ):
        for policy in POLICIES:
            _construct_llm(
                model="/models/qwen3",
                prompt_tokens=256,
                generated_tokens=128,
                gpu_memory_utilization=0.5,
                policy=policy,
            )

    by_policy = {
        policy: kwargs
        for policy, (_model, kwargs) in zip(POLICIES, calls)
    }
    assert by_policy["decode_burst_k8_split_phase"][
        "exact_greedy_decode_burst_split_phase"
    ] is True
    assert all(
        by_policy[policy][
            "exact_greedy_decode_burst_split_phase"
        ] is False
        for policy in POLICIES
        if policy != "decode_burst_k8_split_phase"
    )
    assert by_policy["decode_burst_k8_split_phase"][
        "exact_greedy_decode_burst_tokens"
    ] == 8
    assert by_policy["decode_burst_k8_split_phase"][
        "exact_greedy_decode_burst_continuation"
    ] is False


def test_phase_observations_require_ordered_exact_pairs() -> None:
    parent_a = "a" * 64
    parent_b = "b" * 64
    inventory = summarize_split_phase_observations([
        _phase_row(parent_a, "prefix"),
        _phase_row(parent_a, "suffix"),
        _phase_row(parent_b, "prefix"),
        _phase_row(parent_b, "suffix"),
    ])
    assert inventory["parent_lease_count"] == 2
    assert inventory["prefix_row_count"] == 2
    assert inventory["suffix_row_count"] == 2
    assert inventory["replay_count"] == 16
    assert inventory["prefix_d2h_calls"] == 2
    assert inventory["suffix_d2h_calls"] == 2
    assert inventory["prefix_d2h_bytes"] == 64
    assert inventory["suffix_d2h_bytes"] == 64
    assert inventory["unexpected_scheduler_calls"] == 0
    assert inventory["host_visible_gaps_ns"] == [
        2_000_000,
        3_000_000,
        2_000_000,
        3_000_000,
    ]

    invalid = [
        _phase_row(parent_a, "prefix"),
        _phase_row(parent_b, "suffix"),
    ]
    try:
        summarize_split_phase_observations(invalid)
    except ValueError as error:
        assert str(error) == (
            "split phase observations are not ordered parent pairs"
        )
    else:
        raise AssertionError("mismatched split pair was accepted")


def test_case_validation_binds_phase_and_counter_inventory() -> None:
    row = validate_case_row(
        _case_row("decode_burst_k8_split_phase")
    )
    assert row["split_phase_inventory"]["parent_lease_count"] == 15
    assert row["exact_greedy_decode_burst_summary"][
        "prefix_commits"
    ] == 15
    assert len(row["decode_host_ns"]) == 28


def test_case_validation_accepts_scheduler_owned_single_token_tail() -> None:
    row = _case_row("decode_burst_k8_split_phase")
    summary = row["exact_greedy_decode_burst_summary"]

    validated = validate_case_row(row)

    assert validated["exact_greedy_decode_burst_summary"][
        "fallback_counts"
    ] == {
        "insufficient_output_budget": 1,
        "split_phase_requires_k8": 6,
    }

    invalid = deepcopy(row)
    invalid["exact_greedy_decode_burst_summary"][
        "fallback_counts"
    ] = {"split_phase_requires_k8": 7}
    try:
        validate_case_row(invalid)
    except ValueError as error:
        assert str(error) == (
            "split phase ordinary-tail inventory mismatch"
        )
    else:
        raise AssertionError(
            "scheduler-owned single-token tail was misclassified"
        )

    invalid = deepcopy(row)
    invalid["split_phase_inventory"][
        "unexpected_scheduler_calls"
    ] = 1
    try:
        validate_case_row(invalid)
    except ValueError as error:
        assert str(error) == (
            "split phase scheduled during suffix drain"
        )
    else:
        raise AssertionError(
            "unexpected suffix scheduler call was accepted"
        )

    nonsplit = deepcopy(_case_row("decode_burst_k8"))
    nonsplit["split_phase_inventory"]["prefix_row_count"] = 1
    try:
        validate_case_row(nonsplit)
    except ValueError as error:
        assert str(error) == (
            "non-split policy reported split phase activity"
        )
    else:
        raise AssertionError(
            "non-split phase activity was accepted"
        )


def test_correctness_rows_bind_split_graph_and_ordinary_tail() -> None:
    with TemporaryDirectory() as temporary:
        run_dir = Path(temporary)
        rows = [
            _correctness_row(
                run_dir=run_dir,
                policy=policy,
                bucket=bucket,
                sampling_point=point,
            )
            for bucket, _prompt, _generated in CONTEXT_CASES
            for policy in POLICIES
            for point in SAMPLING_POINTS
        ]
        validated = validate_correctness_rows(
            rows,
            run_dir=run_dir,
        )
    split_rows = {
        row["sampling_point"]: row
        for row in validated
        if row["context_bucket"] == "short"
        and row["policy"] == "decode_burst_k8_split_phase"
    }
    assert split_rows["decode-first"][
        "trace_graph_identity_sha256"
    ] == "c" * 64
    assert split_rows["decode-middle"][
        "trace_graph_identity_sha256"
    ] == "c" * 64
    assert split_rows["decode-final"][
        "trace_graph_identity_sha256"
    ] is None
    assert split_rows["decode-final"][
        "sampled_logit_d2h_calls"
    ] == 0


def main() -> None:
    test_frozen_schema_matrix_and_source_inventory()
    test_only_split_arm_enables_split_phase()
    test_workload_manifest_freezes_sixty_rows_and_split_costs()
    test_runtime_configuration_enables_split_only_for_split_arm()
    test_phase_observations_require_ordered_exact_pairs()
    test_case_validation_binds_phase_and_counter_inventory()
    test_case_validation_accepts_scheduler_owned_single_token_tail()
    test_correctness_rows_bind_split_graph_and_ordinary_tail()
    print("exact burst split-phase profile tests passed")


if __name__ == "__main__":
    main()
