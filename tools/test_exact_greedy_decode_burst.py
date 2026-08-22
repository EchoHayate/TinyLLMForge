#!/usr/bin/env python3
"""Dependency-light tests for exact greedy decode burst contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "exact_greedy_decode_burst.py"
)
SPEC = importlib.util.spec_from_file_location(
    "exact_greedy_decode_burst_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ExactGreedyDecodeBurstCaptureReceipt = (
    module.ExactGreedyDecodeBurstCaptureReceipt
)
ExactGreedyDecodeBurstLease = module.ExactGreedyDecodeBurstLease
ExactGreedyDecodeBurstResult = module.ExactGreedyDecodeBurstResult
ExactGreedyDecodeBurstStats = module.ExactGreedyDecodeBurstStats
build_exact_greedy_decode_burst_decision = (
    module.build_exact_greedy_decode_burst_decision
)
build_exact_greedy_decode_burst_lease = (
    module.build_exact_greedy_decode_burst_lease
)
validate_exact_greedy_decode_burst_result = (
    module.validate_exact_greedy_decode_burst_result
)


def _assert_raises(error_type, message, callback):
    try:
        callback()
    except error_type as error:
        assert str(error) == message, (str(error), message)
    else:
        raise AssertionError(
            f"expected {error_type.__name__}: {message}"
        )


def _eligible_kwargs() -> dict:
    return {
        "enabled": True,
        "configured_width": 8,
        "remaining_output_tokens": 6,
        "initial_sequence_length": 251,
        "block_size": 256,
        "sequence_count": 1,
        "waiting_count": 0,
        "prefilling_count": 0,
        "is_prefill": False,
        "do_sample": True,
        "batch_kind": None,
        "temperatures": (0.0,),
        "ignore_eos": (True,),
        "completion_only": True,
        "tensor_parallel_size": 1,
        "rank": 0,
        "graph_available": True,
        "incompatible_modes": (),
        "pending_lease": False,
        "quarantined": False,
    }


def test_policy_clips_to_budget_and_current_block() -> None:
    decision = build_exact_greedy_decode_burst_decision(
        **_eligible_kwargs()
    )
    assert decision.optimized is True
    assert decision.authorized_token_count == 6
    assert decision.first_write_position == 250
    assert decision.last_write_position == 255
    assert decision.fallback_reason is None

    kwargs = _eligible_kwargs()
    kwargs.update(
        configured_width=8,
        remaining_output_tokens=3,
        initial_sequence_length=101,
    )
    decision = build_exact_greedy_decode_burst_decision(
        **kwargs
    )
    assert decision.authorized_token_count == 3
    assert decision.output_budget_clipped is True
    assert decision.block_boundary_clipped is False

    kwargs = _eligible_kwargs()
    kwargs.update(
        configured_width=8,
        remaining_output_tokens=8,
        initial_sequence_length=253,
    )
    decision = build_exact_greedy_decode_burst_decision(
        **kwargs
    )
    assert decision.authorized_token_count == 4
    assert decision.output_budget_clipped is False
    assert decision.block_boundary_clipped is True


def test_boundary_width_one_falls_back_before_replay() -> None:
    kwargs = _eligible_kwargs()
    kwargs.update(
        configured_width=4,
        remaining_output_tokens=4,
        initial_sequence_length=256,
    )
    decision = build_exact_greedy_decode_burst_decision(
        **kwargs
    )
    assert decision.optimized is False
    assert decision.authorized_token_count == 1
    assert decision.first_write_position == 255
    assert decision.last_write_position == 255
    assert decision.fallback_reason == "authorized_width_below_two"


def test_fallback_reasons_have_stable_precedence() -> None:
    cases = (
        ("enabled", False, "disabled"),
        ("sequence_count", 2, "sequence_count_unsupported"),
        ("waiting_count", 1, "waiting_present"),
        ("prefilling_count", 1, "prefilling_present"),
        ("is_prefill", True, "prefill_unsupported"),
        ("do_sample", False, "sampling_disabled"),
        ("batch_kind", "mixed", "mixed_batch_unsupported"),
        ("temperatures", (0.5,), "nonzero_temperature"),
        ("ignore_eos", (False,), "eos_sensitive"),
        ("completion_only", False, "visibility_unsupported"),
        (
            "tensor_parallel_size",
            2,
            "tensor_parallel_unsupported",
        ),
        ("rank", 1, "non_root_rank"),
        ("graph_available", False, "graph_unavailable"),
        (
            "incompatible_modes",
            ("kv_offload",),
            "incompatible_mode:kv_offload",
        ),
        ("pending_lease", True, "lease_pending"),
        ("quarantined", True, "quarantined"),
    )
    for field, value, expected_reason in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        decision = build_exact_greedy_decode_burst_decision(
            **kwargs
        )
        assert decision.optimized is False
        assert decision.fallback_reason == expected_reason

    kwargs = _eligible_kwargs()
    kwargs.update(
        enabled=False,
        sequence_count=2,
        graph_available=False,
    )
    assert (
        build_exact_greedy_decode_burst_decision(
            **kwargs
        ).fallback_reason
        == "disabled"
    )


def test_invalid_policy_inputs_fail_closed() -> None:
    cases = (
        (
            "enabled",
            1,
            "enabled must be a bool",
        ),
        (
            "configured_width",
            True,
            "configured_width must be an integer in [2, 8]",
        ),
        (
            "configured_width",
            1,
            "configured_width must be an integer in [2, 8]",
        ),
        (
            "configured_width",
            9,
            "configured_width must be an integer in [2, 8]",
        ),
        (
            "remaining_output_tokens",
            -1,
            "remaining_output_tokens must be a non-negative integer",
        ),
        (
            "initial_sequence_length",
            0,
            "initial_sequence_length must be a positive integer",
        ),
        (
            "block_size",
            0,
            "block_size must be a positive integer",
        ),
        (
            "temperatures",
            ("0",),
            "temperatures must contain finite numbers",
        ),
        (
            "ignore_eos",
            [True],
            "ignore_eos must be a tuple",
        ),
        (
            "incompatible_modes",
            ["kv_offload"],
            "incompatible_modes must be a tuple",
        ),
    )
    for field, value, message in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        _assert_raises(
            ValueError,
            message,
            lambda kwargs=kwargs:
                build_exact_greedy_decode_burst_decision(
                    **kwargs
                ),
        )


def _lease() -> ExactGreedyDecodeBurstLease:
    return build_exact_greedy_decode_burst_lease(
        sequence_id=17,
        schedule_generation=9,
        graph_generation=4,
        requested_token_count=8,
        authorized_token_count=4,
        initial_completion_count=3,
        initial_sequence_length=253,
        block_table_identity=((7, 2),),
        write_block_id=7,
        write_block_generation=2,
        first_write_position=252,
        last_write_position=255,
        first_physical_slot=2044,
        last_physical_slot=2047,
        remaining_output_tokens=9,
        completion_only=True,
    )


def _result(lease) -> ExactGreedyDecodeBurstResult:
    return ExactGreedyDecodeBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        tokens=(11, 12, 13, 14),
        replay_count=4,
        final_input_token=14,
        final_position=256,
        final_context_length=257,
        final_physical_slot=2048,
        graph_identity_sha256="a" * 64,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=0,
    )


def test_lease_identity_is_canonical_and_result_is_exact() -> None:
    first = _lease()
    second = _lease()
    assert first == second
    assert len(first.identity_sha256) == 64
    int(first.identity_sha256, 16)
    assert validate_exact_greedy_decode_burst_result(
        first,
        _result(first),
    ) == _result(first)

    mismatched = ExactGreedyDecodeBurstResult(
        **{
            **_result(first).__dict__,
            "replay_count": 3,
        }
    )
    _assert_raises(
        ValueError,
        "burst result replay count does not match lease",
        lambda: validate_exact_greedy_decode_burst_result(
            first,
            mismatched,
        ),
    )


def test_correctness_trace_is_bounded_and_ordered() -> None:
    lease = _lease()
    result = ExactGreedyDecodeBurstResult(
        **{
            **_result(lease).__dict__,
            "sampled_logit_d2h_calls": 1,
            "sampled_logits": (
                (0, (1.0, 2.0)),
                (2, (3.0, 4.0)),
            ),
        }
    )
    validate_exact_greedy_decode_burst_result(
        lease,
        result,
        correctness_trace=True,
    )
    duplicate = ExactGreedyDecodeBurstResult(
        **{
            **result.__dict__,
            "sampled_logits": (
                (0, (1.0,)),
                (0, (2.0,)),
            ),
        }
    )
    _assert_raises(
        ValueError,
        "sampled logit ordinals must be strictly increasing",
        lambda: validate_exact_greedy_decode_burst_result(
            lease,
            duplicate,
            correctness_trace=True,
        ),
    )
    _assert_raises(
        ValueError,
        "production burst cannot return sampled logits",
        lambda: validate_exact_greedy_decode_burst_result(
            lease,
            result,
            correctness_trace=False,
        ),
    )


def test_stats_track_benefit_cost_and_terminal_state() -> None:
    receipt = ExactGreedyDecodeBurstCaptureReceipt(
        graph_identity_sha256="b" * 64,
        graph_generation=4,
        capture_duration_ns=123,
        allocated_delta_bytes=456,
        reserved_delta_bytes=789,
        retained_static_bytes=321,
        scratch_block_count=1,
        correctness_trace=False,
    )
    stats = ExactGreedyDecodeBurstStats()
    stats.record_attempt()
    stats.record_acceptance(
        requested_token_count=8,
        authorized_token_count=4,
        output_budget_clipped=False,
        block_boundary_clipped=True,
    )
    stats.record_capture(receipt)
    stats.record_replays(4)
    stats.record_final_token_d2h(token_count=4, byte_count=32)
    stats.record_commit(token_count=4, host_visible_gap_ns=12_000_000)

    summary = stats.summary()
    json.dumps(summary, allow_nan=False)
    assert summary["attempts"] == 1
    assert summary["acceptances"] == 1
    assert summary["target_model_forwards"] == 4
    assert summary["graph_replays"] == 4
    assert summary["intermediate_token_d2h_calls"] == 0
    assert summary["final_token_d2h_calls"] == 1
    assert summary["final_token_d2h_bytes"] == 32
    assert summary["committed_tokens"] == 4
    assert summary["pending_leases"] == 0
    assert summary["maximum_host_visible_gap_ns"] == 12_000_000
    assert summary["block_boundary_clipped"] == 1
    assert summary["capture_receipts"][0][
        "scratch_block_count"
    ] == 1


def test_contract_is_model_agnostic_and_supports_second_caller() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "Qwen",
        "checkpoint",
        "tokenizer",
        "prompt",
        "A100",
        '"short"',
        '"medium"',
        '"long"',
    ):
        assert forbidden not in source

    lease = build_exact_greedy_decode_burst_lease(
        sequence_id=3,
        schedule_generation=2,
        graph_generation=5,
        requested_token_count=2,
        authorized_token_count=2,
        initial_completion_count=1,
        initial_sequence_length=9,
        block_table_identity=((2, 6),),
        write_block_id=2,
        write_block_generation=6,
        first_write_position=8,
        last_write_position=9,
        first_physical_slot=40,
        last_physical_slot=41,
        remaining_output_tokens=2,
        completion_only=True,
    )
    result = ExactGreedyDecodeBurstResult(
        lease_identity_sha256=lease.identity_sha256,
        tokens=(101, 102),
        replay_count=2,
        final_input_token=102,
        final_position=10,
        final_context_length=11,
        final_physical_slot=42,
        graph_identity_sha256="c" * 64,
        token_d2h_calls=1,
        sampled_logit_d2h_calls=0,
    )
    assert validate_exact_greedy_decode_burst_result(
        lease,
        result,
    ).tokens == (101, 102)


def main() -> None:
    test_policy_clips_to_budget_and_current_block()
    test_boundary_width_one_falls_back_before_replay()
    test_fallback_reasons_have_stable_precedence()
    test_invalid_policy_inputs_fail_closed()
    test_lease_identity_is_canonical_and_result_is_exact()
    test_correctness_trace_is_bounded_and_ordered()
    test_stats_track_benefit_cost_and_terminal_state()
    test_contract_is_model_agnostic_and_supports_second_caller()
    print("exact greedy decode burst tests passed")


if __name__ == "__main__":
    main()
