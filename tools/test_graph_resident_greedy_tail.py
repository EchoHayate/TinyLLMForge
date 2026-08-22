#!/usr/bin/env python3
"""Dependency-light tests for graph-resident greedy-tail contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "tinyvllm"
    / "engine"
    / "graph_resident_greedy_tail.py"
)
SPEC = importlib.util.spec_from_file_location(
    "graph_resident_greedy_tail_under_test",
    MODULE_PATH,
)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
GraphResidentGreedyTailCaptureReceipt = (
    module.GraphResidentGreedyTailCaptureReceipt
)
GraphResidentGreedyTailReplay = (
    module.GraphResidentGreedyTailReplay
)
GraphResidentGreedyTailStats = (
    module.GraphResidentGreedyTailStats
)
decide_graph_resident_greedy_tail = (
    module.decide_graph_resident_greedy_tail
)
tensor_identity = module.tensor_identity


def _eligible_kwargs() -> dict:
    return {
        "enabled": True,
        "rank": 0,
        "tensor_parallel_size": 1,
        "is_prefill": False,
        "enforce_eager": False,
        "batch_kind": None,
        "active_batch_size": 1,
        "selected_graph_batch_size": 1,
        "do_sample": True,
        "temperatures": (0.0,),
        "input_embeds_present": False,
        "return_hidden": False,
        "incompatible_modes": (),
        "capture_available": True,
        "quarantined": False,
        "source_matches": True,
    }


def test_exact_ordinary_batch_one_greedy_decode_is_eligible() -> None:
    decision = decide_graph_resident_greedy_tail(
        **_eligible_kwargs()
    )

    assert decision.optimized is True
    assert decision.fallback_reason is None


def test_ineligible_cases_fail_closed_in_stable_order() -> None:
    cases = (
        ("enabled", False, "disabled"),
        ("rank", 1, "non_root_rank"),
        (
            "tensor_parallel_size",
            2,
            "tensor_parallel_unsupported",
        ),
        ("is_prefill", True, "prefill_unsupported"),
        ("enforce_eager", True, "eager_unsupported"),
        ("batch_kind", "mixed", "mixed_batch_unsupported"),
        (
            "active_batch_size",
            2,
            "batch_size_unsupported",
        ),
        (
            "selected_graph_batch_size",
            2,
            "selected_graph_batch_unsupported",
        ),
        ("do_sample", False, "sampling_disabled"),
        ("temperatures", ("0",), "temperature_invalid"),
        ("temperatures", (0.7,), "nonzero_temperature"),
        (
            "input_embeds_present",
            True,
            "input_embeds_unsupported",
        ),
        ("return_hidden", True, "return_hidden_unsupported"),
        (
            "incompatible_modes",
            ("kv_offload",),
            "incompatible_mode:kv_offload",
        ),
        (
            "capture_available",
            False,
            "capture_unavailable",
        ),
        ("quarantined", True, "quarantined"),
        (
            "source_matches",
            False,
            "source_identity_drift",
        ),
    )

    for field, value, expected_reason in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        decision = decide_graph_resident_greedy_tail(**kwargs)
        assert decision.optimized is False
        assert decision.fallback_reason == expected_reason

    kwargs = _eligible_kwargs()
    kwargs.update(
        enabled=False,
        rank=1,
        tensor_parallel_size=2,
        is_prefill=True,
        capture_available=False,
    )
    decision = decide_graph_resident_greedy_tail(**kwargs)
    assert decision.fallback_reason == "disabled"


def test_invalid_control_values_raise_exact_messages() -> None:
    cases = (
        ("enabled", 1, "enabled must be a bool"),
        ("rank", True, "rank must be a non-negative integer"),
        (
            "tensor_parallel_size",
            0,
            "tensor_parallel_size must be a positive integer",
        ),
        ("is_prefill", 0, "is_prefill must be a bool"),
        ("enforce_eager", 0, "enforce_eager must be a bool"),
        (
            "batch_kind",
            1,
            "batch_kind must be a string or None",
        ),
        (
            "active_batch_size",
            True,
            "active_batch_size must be a non-negative integer",
        ),
        (
            "selected_graph_batch_size",
            0,
            "selected_graph_batch_size must be a positive integer",
        ),
        ("do_sample", 1, "do_sample must be a bool"),
        (
            "temperatures",
            [0.0],
            "temperatures must be a tuple",
        ),
        (
            "input_embeds_present",
            0,
            "input_embeds_present must be a bool",
        ),
        (
            "return_hidden",
            0,
            "return_hidden must be a bool",
        ),
        (
            "incompatible_modes",
            ["kv_offload"],
            "incompatible_modes must be a tuple",
        ),
        (
            "capture_available",
            1,
            "capture_available must be a bool",
        ),
        ("quarantined", 0, "quarantined must be a bool"),
        (
            "source_matches",
            1,
            "source_matches must be a bool",
        ),
    )

    for field, value, expected_message in cases:
        kwargs = _eligible_kwargs()
        kwargs[field] = value
        try:
            decide_graph_resident_greedy_tail(**kwargs)
        except ValueError as error:
            assert str(error) == expected_message
        else:
            raise AssertionError(
                f"{field} validation did not fail"
            )


class _FakeTensor:
    shape = (1, 1024)
    dtype = "bfloat16"
    device = "cuda:0"

    def data_ptr(self) -> int:
        return 123_456

    def stride(self) -> tuple[int, int]:
        return (1024, 1)

    def storage_offset(self) -> int:
        return 8


def test_tensor_identity_uses_storage_geometry_not_python_id() -> None:
    assert tensor_identity(_FakeTensor()) == (
        123_456,
        (1, 1024),
        (1024, 1),
        8,
        "bfloat16",
        "cuda:0",
    )


def test_capture_receipt_and_replay_are_immutable_records() -> None:
    receipt = GraphResidentGreedyTailCaptureReceipt(
        source_identity=tensor_identity(_FakeTensor()),
        graph_generation=7,
        rank=0,
        capture_duration_ns=1_000,
        allocated_delta_bytes=256,
        reserved_delta_bytes=512,
        retained_logits_bytes=16,
        retained_float32_bytes=32,
        retained_token_bytes=8,
    )
    logits = object()
    token_ids = object()
    replay = GraphResidentGreedyTailReplay(
        logits=logits,
        token_ids=token_ids,
    )

    assert receipt.graph_generation == 7
    assert replay.logits is logits
    assert replay.token_ids is token_ids
    try:
        receipt.rank = 1
    except Exception:
        pass
    else:
        raise AssertionError("capture receipt is mutable")


def test_stats_account_exact_graph_work_and_cost() -> None:
    stats = GraphResidentGreedyTailStats()
    receipt = GraphResidentGreedyTailCaptureReceipt(
        source_identity=tensor_identity(_FakeTensor()),
        graph_generation=7,
        rank=0,
        capture_duration_ns=1_000,
        allocated_delta_bytes=256,
        reserved_delta_bytes=512,
        retained_logits_bytes=16,
        retained_float32_bytes=32,
        retained_token_bytes=8,
    )
    stats.record_capture(receipt)
    stats.record_fallback("disabled")
    stats.record_fallback("disabled")
    stats.record_replay()
    stats.record_replay()
    stats.record_token_d2h()
    stats.record_token_d2h()

    assert stats.summary() == {
        "eligible_steps": 2,
        "captured_graphs": 1,
        "replayed_steps": 2,
        "final_token_d2h_calls": 2,
        "avoided_external_compute_logits_calls": 2,
        "avoided_external_float32_conversions": 2,
        "avoided_external_argmax_calls": 2,
        "fallback_counts": {"disabled": 2},
        "quarantine_reason": None,
        "capture_receipt": {
            "source_identity": {
                "data_ptr": 123_456,
                "shape": [1, 1024],
                "stride": [1024, 1],
                "storage_offset": 8,
                "dtype": "bfloat16",
                "device": "cuda:0",
            },
            "graph_generation": 7,
            "rank": 0,
            "capture_duration_ns": 1_000,
            "allocated_delta_bytes": 256,
            "reserved_delta_bytes": 512,
            "retained_logits_bytes": 16,
            "retained_float32_bytes": 32,
            "retained_token_bytes": 8,
            "retained_static_bytes": 56,
        },
    }


def test_stats_reject_invalid_updates_and_keep_first_quarantine() -> None:
    stats = GraphResidentGreedyTailStats()
    for value in ("", 1, None):
        try:
            stats.record_fallback(value)
        except ValueError as error:
            assert str(error) == (
                "fallback reason must be a non-empty string"
            )
        else:
            raise AssertionError("invalid fallback reason was accepted")

    for value in ("", 1, None):
        try:
            stats.quarantine(value)
        except ValueError as error:
            assert str(error) == (
                "quarantine reason must be a non-empty string"
            )
        else:
            raise AssertionError(
                "invalid quarantine reason was accepted"
            )

    stats.quarantine("replay_failure:RuntimeError")
    stats.quarantine("replay_failure:ValueError")
    assert stats.summary()["quarantine_reason"] == (
        "replay_failure:RuntimeError"
    )


def main() -> None:
    test_exact_ordinary_batch_one_greedy_decode_is_eligible()
    test_ineligible_cases_fail_closed_in_stable_order()
    test_invalid_control_values_raise_exact_messages()
    test_tensor_identity_uses_storage_geometry_not_python_id()
    test_capture_receipt_and_replay_are_immutable_records()
    test_stats_account_exact_graph_work_and_cost()
    test_stats_reject_invalid_updates_and_keep_first_quarantine()
    print("graph-resident greedy tail tests passed")


if __name__ == "__main__":
    main()
