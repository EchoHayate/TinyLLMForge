"""Dependency-light tests for the Qwen3.5 hybrid-state gate contract.

Run: python3 tools/test_qwen35_hybrid_state_contract.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import fields
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_hybrid_state_contract.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen35_hybrid_state_contract_under_test",
    os.fspath(CONTRACT_PATH),
)
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


def _expect_value_error(callable_, message_fragment):
    try:
        callable_()
    except ValueError as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_chunk_schedules_omit_only_zero_remainders():
    assert contract.build_chunk_schedule(65, (64,)) == (64, 1)
    assert contract.build_chunk_schedule(65, (31, 34)) == (31, 34)
    assert contract.build_chunk_schedule(257, (3, 5)) == (3, 5, 249)
    assert contract.build_chunk_schedule(64, (64,)) == (64,)


def test_chunk_schedules_reject_invalid_inputs():
    _expect_value_error(
        lambda: contract.build_chunk_schedule(0, (1,)),
        "prompt_length",
    )
    _expect_value_error(
        lambda: contract.build_chunk_schedule(17, (0,)),
        "positive",
    )
    _expect_value_error(
        lambda: contract.build_chunk_schedule(17, (18,)),
        "exceed",
    )


def test_case_matrix_is_closed_unique_and_covers_every_phase():
    matrix = contract.build_case_matrix()
    assert len({case.case_id for case in matrix}) == len(matrix)
    assert {case.phase for case in matrix} == set(contract.REQUIRED_PHASES)
    assert all(case.phase in contract.REQUIRED_PHASES for case in matrix)
    assert all(case.decode_steps >= 0 for case in matrix)
    assert all(sum(case.chunk_schedule) == case.prompt_length for case in matrix)

    chunked = [case for case in matrix if case.phase == "one_shot_vs_chunked"]
    assert {case.prompt_length for case in chunked} == {65, 257, 1025}
    assert all(case.chunk_schedule for case in chunked)
    expected_schedules = {
        (prompt_length, contract.build_chunk_schedule(prompt_length, template))
        for prompt_length in (65, 257, 1025)
        for template in contract.CHUNK_TEMPLATES
        if sum(template) <= prompt_length
    }
    assert {
        (case.prompt_length, case.chunk_schedule)
        for case in chunked
    } == expected_schedules


def test_case_matrix_covers_single_request_and_lifecycle_domain():
    matrix = contract.build_case_matrix()
    for phase in (
        "same_path_repeatability",
        "one_shot_vs_cached",
        "state_export_import",
    ):
        rows = [case for case in matrix if case.phase == phase]
        assert {case.prompt_length for case in rows} == set(
            contract.PROMPT_LENGTHS
        )

    repeats = [
        case
        for case in matrix
        if case.phase == "same_path_repeatability"
    ]
    assert {case.repeat_index for case in repeats} == {0, 1}

    multi_request = [
        case
        for case in matrix
        if case.phase == "interleaved_multi_request"
    ]
    assert len(multi_request) == 1
    assert multi_request[0].request_count == len(
        contract.MULTI_REQUEST_LENGTHS
    )
    assert multi_request[0].decode_steps == contract.DECODE_STEPS

    slot_reuse = [
        case
        for case in matrix
        if case.phase == "completion_release_slot_reuse"
    ]
    assert len(slot_reuse) == 1
    assert slot_reuse[0].prompt_length == contract.SLOT_REUSE_PROMPT_LENGTH
    assert slot_reuse[0].expected_state_snapshots == 34


def test_deterministic_token_ids_are_repeatable_bounded_and_filtered():
    forbidden = {128, 129, 130}
    first = contract.deterministic_token_ids(
        length=64,
        vocab_size=4096,
        seed=17,
        forbidden_ids=forbidden,
    )
    second = contract.deterministic_token_ids(
        length=64,
        vocab_size=4096,
        seed=17,
        forbidden_ids=forbidden,
    )
    assert first == second
    assert len(first) == 64
    assert all(0 <= token_id < 4096 for token_id in first)
    assert not forbidden.intersection(first)


def test_deterministic_token_ids_reject_impossible_domains():
    _expect_value_error(
        lambda: contract.deterministic_token_ids(
            length=-1,
            vocab_size=4096,
            seed=0,
            forbidden_ids=set(),
        ),
        "length",
    )
    _expect_value_error(
        lambda: contract.deterministic_token_ids(
            length=1,
            vocab_size=128,
            seed=0,
            forbidden_ids=set(),
        ),
        "vocab_size",
    )


def test_record_schemas_are_exact_and_frozen():
    assert tuple(field.name for field in fields(contract.GateCase)) == (
        "phase",
        "case_id",
        "execution_mode",
        "prompt_length",
        "chunk_schedule",
        "request_count",
        "decode_steps",
        "repeat_index",
        "expected_state_snapshots",
        "execution_dtype",
        "comparison_policy",
    )
    assert tuple(field.name for field in fields(contract.StateComponent)) == (
        "request_id",
        "request_generation",
        "layer_index",
        "declared_layer_type",
        "state_role",
        "tensor_path",
        "shape",
        "stride",
        "dtype",
        "device",
        "requires_grad",
        "logical_numel",
        "logical_bytes",
        "storage_data_ptr",
        "storage_offset",
        "storage_nbytes",
        "storage_identity",
        "lifetime_epoch",
        "sequence_length",
        "update_kind",
        "content_sha256",
    )
    assert contract.CASE_ROW_FIELDS == (
        "row_id",
        "case_id",
        "phase",
        "execution_mode",
        "prompt_length",
        "chunk_schedule",
        "request_count",
        "decode_steps",
        "repeat_index",
        "request_ids",
        "request_generations",
        "decoded_token_ids",
        "logit_records",
        "state_snapshot_ids",
        "memory_snapshot_ids",
        "complete",
        "failure_kind",
        "failure_detail",
        "execution_dtype",
        "comparison_policy",
    )
    assert contract.LOGIT_RECORD_FIELDS == (
        "request_id",
        "request_generation",
        "step_index",
        "full_logit_sha256",
        "topk_token_ids",
        "topk_logits",
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "mean_rel_diff",
        "sequence_length",
        "position_metadata",
        "actual_topk_token_ids",
        "actual_topk_logits",
        "oracle_topk_token_ids",
        "oracle_topk_logits",
        "topk_intersection_size",
        "oracle_topk_recall",
        "actual_winner_token_id",
        "oracle_winner_token_id",
        "actual_runner_up_token_id",
        "oracle_runner_up_token_id",
        "actual_winner_logit",
        "oracle_winner_logit",
        "actual_runner_up_logit",
        "oracle_runner_up_logit",
        "actual_winner_margin",
        "oracle_winner_margin",
        "winner_logit_abs_diff",
        "runner_up_logit_abs_diff",
        "winner_margin_abs_diff",
        "abs_diff_percentiles",
        "cosine_similarity",
        "allclose_violation_count",
        "max_allclose_scaled_error",
    )


def test_schema_v2_freezes_dtype_and_decision_fields():
    assert contract.SCHEMA_VERSION == 2
    assert contract.DECISION_TOPK == 20
    assert contract.FP32_ATOL == 2e-5
    assert contract.FP32_RTOL == 1e-5
    assert contract.FP32_MEAN_ABS_CAP == 3e-6
    assert contract.EXECUTION_DTYPES == (
        "bfloat16",
        "float32",
        "metadata_only",
    )
    assert contract.COMPARISON_POLICIES == (
        "bf16_decision_preserving",
        "fp32_elementwise",
        "none",
    )
    assert contract.ABS_DIFF_PERCENTILE_FIELDS == (
        "p50",
        "p95",
        "p99",
        "p99_9",
    )


def test_fp32_control_case_is_frozen():
    cases = contract.build_case_matrix()
    control = [
        case
        for case in cases
        if case.case_id == contract.FP32_CONTROL_CASE_ID
    ]
    assert len(control) == 1
    assert control[0].phase == "fp32_path_control"
    assert control[0].execution_mode == "cached_vs_one_shot"
    assert control[0].prompt_length == 17
    assert control[0].execution_dtype == "float32"
    assert control[0].comparison_policy == "fp32_elementwise"


def test_ranked_topk_and_winner_margin():
    token_ids = list(range(20))
    logits = [float(20 - index) for index in range(20)]
    contract.validate_ranked_topk(token_ids, logits)
    result = contract.winner_margin(token_ids, logits)
    assert result == {
        "winner_token_id": 0,
        "runner_up_token_id": 1,
        "winner_logit": 20.0,
        "runner_up_logit": 19.0,
        "winner_margin": 1.0,
    }


def test_ranked_topk_rejects_duplicates_unsorted_and_ties():
    token_ids = list(range(20))
    logits = [float(20 - index) for index in range(20)]
    invalid = (
        ([0] + token_ids[:-1], logits),
        (token_ids, [19.0, 20.0] + logits[2:]),
        (token_ids, [20.0, 20.0] + logits[2:]),
    )
    for bad_ids, bad_logits in invalid:
        _expect_value_error(
            lambda ids=bad_ids, values=bad_logits: (
                contract.validate_ranked_topk(ids, values)
            ),
            "top-k",
        )


def test_ranked_topk_can_preserve_tie_evidence_without_accepting_it():
    token_ids = list(range(20))
    logits = [20.0, 20.0] + [
        float(18 - index) for index in range(18)
    ]
    contract.validate_ranked_topk(
        token_ids,
        logits,
        require_positive_margin=False,
    )
    result = contract.winner_margin(
        token_ids,
        logits,
        require_positive_margin=False,
    )
    assert result["winner_token_id"] == 0
    assert result["runner_up_token_id"] == 1
    assert result["winner_margin"] == 0.0


def test_fp32_limits_are_not_derived_from_bf16_rows():
    assert contract.FP32_ATOL == 2e-5
    assert contract.FP32_RTOL == 1e-5
    assert contract.FP32_MEAN_ABS_CAP == 3e-6
    assert not hasattr(contract, "BF16_MAX_LOGIT_ATOL")
    assert not hasattr(contract, "BF16_MAX_LOGIT_RTOL")


def test_logical_bytes_supports_frozen_dtypes_and_rejects_bad_shapes():
    assert contract.logical_bytes((2, 3, 4), "float16") == 48
    assert contract.logical_bytes((2, 3, 4), "float32") == 96
    assert contract.logical_bytes((0, 4), "uint8") == 0
    _expect_value_error(
        lambda: contract.logical_bytes((2, -1), "float16"),
        "shape",
    )
    _expect_value_error(
        lambda: contract.logical_bytes((1,), "complex64"),
        "dtype",
    )


def test_unique_storage_bytes_deduplicates_aliases_by_device_and_identity():
    rows = [
        {"device": "cuda:0", "storage_identity": "a", "storage_nbytes": 64},
        {"device": "cuda:0", "storage_identity": "a", "storage_nbytes": 64},
        {"device": "cuda:0", "storage_identity": "b", "storage_nbytes": 32},
        {"device": "cpu", "storage_identity": "a", "storage_nbytes": 16},
    ]
    assert contract.unique_storage_bytes(rows) == 112


def test_unique_storage_bytes_rejects_conflicting_alias_sizes():
    rows = [
        {"device": "cuda:0", "storage_identity": "a", "storage_nbytes": 64},
        {"device": "cuda:0", "storage_identity": "a", "storage_nbytes": 32},
    ]
    _expect_value_error(
        lambda: contract.unique_storage_bytes(rows),
        "conflicting",
    )


def test_repeatability_tolerance_is_four_x_observed_and_capped():
    rows = [{"max_abs_diff": 2e-5, "max_rel_diff": 3e-6}]
    assert contract.derive_logit_tolerance(rows) == {
        "atol": 8e-5,
        "rtol": 1.2e-5,
    }
    _expect_value_error(
        lambda: contract.derive_logit_tolerance([{
            "max_abs_diff": 1e-3,
            "max_rel_diff": 1e-4,
        }]),
        "INCOMPLETE_NUMERICAL_INSTABILITY",
    )


def test_repeatability_tolerance_uses_floor_and_requires_rows():
    assert contract.derive_logit_tolerance([{
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }]) == {
        "atol": 1e-6,
        "rtol": 1e-6,
    }
    _expect_value_error(
        lambda: contract.derive_logit_tolerance([]),
        "repeatability",
    )


def test_classification_separates_incomplete_from_semantic_no_go():
    passing = {name: True for name in contract.GO_GUARDS}
    assert contract.classify_evidence(passing, None) == "GO"
    assert contract.classify_evidence(
        {**passing, "slot_reuse_pass": False},
        "semantic_failure",
    ) == "NO_GO"
    assert contract.classify_evidence(
        passing,
        "INCOMPLETE_RESOURCE_BLOCKED",
    ) == "INCOMPLETE"


def test_classification_fails_closed_on_unknown_or_non_boolean_guards():
    passing = {name: True for name in contract.GO_GUARDS}
    _expect_value_error(
        lambda: contract.classify_evidence(
            {**passing, "unknown_guard": True},
            None,
        ),
        "guard",
    )
    bad_type = dict(passing)
    bad_type[contract.GO_GUARDS[0]] = 1
    _expect_value_error(
        lambda: contract.classify_evidence(bad_type, None),
        "exactly True or False",
    )
    missing = dict(passing)
    missing.pop(contract.GO_GUARDS[0])
    _expect_value_error(
        lambda: contract.classify_evidence(missing, None),
        "guard",
    )
    assert contract.classify_evidence(
        passing,
        "dependency_failure",
    ) == "INCOMPLETE"


def test_canonical_json_and_hash_are_order_independent():
    left = {"b": [2, 1], "a": {"z": 3}}
    right = {"a": {"z": 3}, "b": [2, 1]}
    assert contract.canonical_json_bytes(left) == (
        json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    assert contract.canonical_sha256(left) == contract.canonical_sha256(right)


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print("qwen35 hybrid-state contract tests passed")


if __name__ == "__main__":
    main()
