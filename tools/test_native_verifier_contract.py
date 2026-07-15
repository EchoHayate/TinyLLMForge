"""Dependency-light tests for the native verifier tensor contract."""

from __future__ import annotations

import importlib.util
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_VERIFIER_PATH = os.path.join(
    _REPO_ROOT,
    "tinyvllm",
    "speculative",
    "verifier.py",
)
_SPEC = importlib.util.spec_from_file_location(
    "native_verifier_under_test",
    _VERIFIER_PATH,
)
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules["native_verifier_under_test"] = verifier
_SPEC.loader.exec_module(verifier)

SpecVerifyMetadata = verifier.SpecVerifyMetadata
build_spec_verify_plan = verifier.build_spec_verify_plan
spec_verify_metadata_to_dict = verifier.spec_verify_metadata_to_dict
validate_spec_verify_slots = verifier.validate_spec_verify_slots


def test_reference_h52_k4_contract():
    plan = build_spec_verify_plan(
        history_len=52,
        draft_tokens=[10, 20, 30, 40],
        block_size=256,
    )

    assert plan.input_tokens == (10, 20, 30)
    assert plan.positions == (53, 54, 55)
    assert plan.logical_slots == (52, 53, 54)
    assert plan.context_len == 55
    assert plan.visible_block_count == 1


def test_k1_has_zero_tail_queries():
    plan = build_spec_verify_plan(
        history_len=52,
        draft_tokens=[10],
        block_size=256,
    )

    assert plan.query_len == 0
    assert plan.input_tokens == ()
    assert plan.positions == ()
    assert plan.logical_slots == ()
    assert plan.context_len == 52


def test_required_k_values_have_consecutive_positions_and_slots():
    for draft_len in (1, 4, 8, 16):
        draft = list(range(100, 100 + draft_len))
        plan = build_spec_verify_plan(255, draft, block_size=256)

        assert plan.query_len == max(0, draft_len - 1)
        assert plan.positions == tuple(range(256, 256 + plan.query_len))
        assert plan.logical_slots == tuple(range(255, 255 + plan.query_len))
        assert plan.visible_block_count == (
            plan.context_len + 255
        ) // 256


def test_slot_validation_maps_current_and_reserved_blocks():
    plan = build_spec_verify_plan(
        255,
        [1, 2, 3, 4],
        block_size=256,
    )

    assert validate_spec_verify_slots(plan, [7, 11], 256) == (
        7 * 256 + 255,
        11 * 256,
        11 * 256 + 1,
    )


def test_invalid_contract_inputs_fail():
    invalid_calls = (
        lambda: build_spec_verify_plan(-1, [1], 256),
        lambda: build_spec_verify_plan(4, [], 256),
        lambda: build_spec_verify_plan(4, [1], 0),
        lambda: validate_spec_verify_slots(
            build_spec_verify_plan(255, [1, 2, 3, 4], 256),
            [7],
            256,
        ),
    )

    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid verifier contract must fail")


def test_metadata_is_json_friendly():
    metadata = SpecVerifyMetadata(
        query_len=3,
        input_tokens=(10, 20, 30),
        positions=(53, 54, 55),
        logical_slots=(52, 53, 54),
        physical_slots=(52, 53, 54),
        context_len=55,
        block_table=(0,),
    )

    assert spec_verify_metadata_to_dict(metadata) == {
        "query_len": 3,
        "input_tokens": [10, 20, 30],
        "positions": [53, 54, 55],
        "logical_slots": [52, 53, 54],
        "physical_slots": [52, 53, 54],
        "context_len": 55,
        "block_table": [0],
    }


def main():
    test_reference_h52_k4_contract()
    test_k1_has_zero_tail_queries()
    test_required_k_values_have_consecutive_positions_and_slots()
    test_slot_validation_maps_current_and_reserved_blocks()
    test_invalid_contract_inputs_fail()
    test_metadata_is_json_friendly()
    print("native verifier contract tests passed")


if __name__ == "__main__":
    main()
