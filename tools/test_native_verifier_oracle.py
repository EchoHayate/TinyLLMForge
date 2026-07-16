"""Dependency-light tests for native verifier oracle comparison."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ORACLE_PATH = os.path.join(_THIS_DIR, "native_verifier_oracle.py")
_SPEC = importlib.util.spec_from_file_location(
    "native_verifier_oracle_under_test",
    _ORACLE_PATH,
)
oracle = importlib.util.module_from_spec(_SPEC)
sys.modules["native_verifier_oracle_under_test"] = oracle
_SPEC.loader.exec_module(oracle)

compare_native_and_oracle = oracle.compare_native_and_oracle
dtype_tolerance = oracle.dtype_tolerance
build_case_payload = oracle.build_case_payload
construct_draft_tokens = oracle.construct_draft_tokens
run_case = oracle.run_case


def test_tinyvllm_backend_has_runtime_timer_dependency():
    assert callable(oracle.time.perf_counter)


def make_comparison_fixture():
    return {
        "dtype": "torch.float16",
        "target_tokens": [4, 5, 6],
        "accepted_tokens": [4, 5],
        "sequence_tokens_after": [1, 2, 3, 4, 5],
        "block_table_after": [0],
        "continuation_tokens": list(range(16)),
        "logits": [[0.0, 1.0], [1.0, 0.0]],
        "kv": {
            "keys": [[0.0, 1.0]],
            "values": [[1.0, 0.0]],
        },
        "continuation_logits": [[[0.0, 1.0]]],
        "continuation_kv": [
            {
                "keys": [[0.0, 1.0]],
                "values": [[1.0, 0.0]],
            }
        ],
        "finite": True,
    }


def test_dtype_tolerances_are_fixed():
    fp16 = dtype_tolerance("torch.float16")
    bf16 = dtype_tolerance("torch.bfloat16")
    assert fp16.logits_rtol == 2e-3
    assert fp16.logits_atol == 2e-3
    assert fp16.kv_rtol == 2e-3
    assert fp16.kv_atol == 2e-3
    assert bf16.logits_rtol == 8e-3
    assert bf16.logits_atol == 8e-3
    assert bf16.kv_rtol == 8e-3
    assert bf16.kv_atol == 8e-3


def test_comparison_requires_tokens_acceptance_metadata_and_continuation():
    payload = make_comparison_fixture()
    comparison = compare_native_and_oracle(payload, dict(payload))

    assert comparison["status"] == "PASS"
    assert comparison["target_token_match"] is True
    assert comparison["accepted_prefix_match"] is True
    assert comparison["metadata_match"] is True
    assert comparison["continuation_token_match"] is True
    assert comparison["continuation_steps"] == 16
    assert comparison["logits_within_tolerance"] is True
    assert comparison["kv_within_tolerance"] is True


def test_token_mismatch_is_no_go_even_when_numeric_error_is_small():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["continuation_tokens"][-1] += 1

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "continuation token mismatch" in comparison["reasons"]


def test_acceptance_or_metadata_mismatch_is_no_go():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["accepted_tokens"] = [4]
    oracle_payload["block_table_after"] = [0, 1]

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "accepted prefix mismatch" in comparison["reasons"]
    assert "committed metadata mismatch" in comparison["reasons"]


def test_numeric_mismatch_is_no_go():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["logits"] = [[0.0, 2.0], [1.0, 0.0]]
    oracle_payload["kv"] = {
        "keys": [[0.0, 2.0]],
        "values": [[1.0, 0.0]],
    }

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "logits exceed tolerance" in comparison["reasons"]
    assert "KV exceeds tolerance" in comparison["reasons"]


def test_continuation_numeric_mismatch_is_no_go():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    oracle_payload["continuation_logits"] = [[[0.0, 2.0]]]
    oracle_payload["continuation_kv"] = [{
        "keys": [[0.0, 2.0]],
        "values": [[1.0, 0.0]],
    }]

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "NO_GO"
    assert "logits exceed tolerance" in comparison["reasons"]
    assert "KV exceeds tolerance" in comparison["reasons"]


def test_missing_or_nonfinite_evidence_is_classified_strictly():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    native["finite"] = False
    comparison = compare_native_and_oracle(native, oracle_payload)
    assert comparison["status"] == "NO_GO"
    assert "non-finite logits or KV" in comparison["reasons"]

    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    del oracle_payload["kv"]
    comparison = compare_native_and_oracle(native, oracle_payload)
    assert comparison["status"] == "INCOMPLETE"
    assert "missing oracle field: kv" in comparison["reasons"]


def test_less_than_16_continuation_steps_is_incomplete():
    native = make_comparison_fixture()
    oracle_payload = make_comparison_fixture()
    native["continuation_tokens"] = list(range(8))
    oracle_payload["continuation_tokens"] = list(range(8))

    comparison = compare_native_and_oracle(native, oracle_payload)

    assert comparison["status"] == "INCOMPLETE"
    assert "continuation coverage below 16 steps" in comparison["reasons"]


def test_build_case_payload_records_complete_evidence():
    evidence = {
        "dtype": "torch.bfloat16",
        "target_tokens": [4, 5],
        "accepted_tokens": [4],
        "sequence_tokens_after": [1, 2, 4],
        "block_table_after": [7],
        "continuation_tokens": list(range(16)),
        "logits": [[0.0, 1.0]],
        "kv": {"keys": [[0.0]], "values": [[1.0]]},
        "continuation_logits": [[[0.0, 1.0]]],
        "continuation_kv": [
            {"keys": [[0.0]], "values": [[1.0]]}
        ],
        "physical_slots": [2],
        "policy": "oracle",
        "case_id": "case-1",
    }

    payload = build_case_payload(evidence)

    assert payload["finite"] is True
    assert payload["case_id"] == "case-1"
    assert payload["policy"] == "oracle"
    assert payload["tolerance"] == {
        "logits_rtol": 8e-3,
        "logits_atol": 8e-3,
        "kv_rtol": 8e-3,
        "kv_atol": 8e-3,
    }


def test_run_case_validates_input_and_writes_backend_payload():
    calls = []

    def fake_backend(**kwargs):
        calls.append(kwargs)
        return build_case_payload({
            "dtype": "torch.float16",
            "target_tokens": [4],
            "accepted_tokens": [4],
            "sequence_tokens_after": [1, 4],
            "block_table_after": [0],
            "continuation_tokens": list(range(16)),
            "logits": [[0.0, 1.0]],
            "kv": {"keys": [[0.0]], "values": [[1.0]]},
            "continuation_logits": [],
            "continuation_kv": [],
            "physical_slots": [],
            "policy": kwargs["policy"],
            "case_id": kwargs["case"]["case_id"],
        })

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "case.json"
        result = run_case(
            policy="native",
            case={
                "case_id": "case-1",
                "prompt": "hello",
                "history_len": 8,
                "draft_tokens": [4],
                "max_tokens": 32,
                "ignore_eos": True,
            },
            out_path=out_path,
            model="/model",
            continuation_steps=16,
            backend=fake_backend,
        )
        written = json.loads(out_path.read_text())

    assert result == written
    assert calls == [{
        "policy": "native",
        "case": {
            "case_id": "case-1",
            "prompt": "hello",
            "history_len": 8,
            "draft_tokens": [4],
            "max_tokens": 32,
            "ignore_eos": True,
        },
        "model": "/model",
        "continuation_steps": 16,
    }]

    invalid_cases = (
        ({}, "case_id"),
        ({
            "case_id": "x",
            "prompt": "hello",
            "history_len": 8,
            "draft_tokens": [],
            "max_tokens": 32,
            "ignore_eos": True,
        }, "draft_tokens"),
    )
    for case, expected in invalid_cases:
        try:
            run_case(
                policy="oracle",
                case=case,
                out_path=Path("/unused"),
                model="/model",
                continuation_steps=16,
                backend=fake_backend,
            )
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("invalid oracle case must fail")


def test_run_case_accepts_all_isolated_policies():
    seen = []

    def fake_backend(**kwargs):
        seen.append(kwargs["policy"])
        return {
            "policy": kwargs["policy"],
            "case_id": kwargs["case"]["case_id"],
        }

    with tempfile.TemporaryDirectory() as tmp:
        for policy in (
            "probe",
            "baseline",
            "legacy_rematerialize",
            "native",
            "oracle",
        ):
            run_case(
                policy=policy,
                case={
                    "case_id": "case-1",
                    "prompt": "hello",
                    "history_len": 8,
                    "draft_tokens": [4],
                    "max_tokens": 32,
                    "ignore_eos": True,
                },
                out_path=Path(tmp) / f"{policy}.json",
                model="/model",
                continuation_steps=16,
                backend=fake_backend,
            )
    assert seen == [
        "probe",
        "baseline",
        "legacy_rematerialize",
        "native",
        "oracle",
    ]


def test_construct_draft_tokens_is_deterministic_for_all_acceptance_cases():
    targets = [10, 20, 30, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="full",
        vocab_size=100,
    ) == [10, 20, 30, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="partial",
        vocab_size=100,
    ) == [10, 20, 31, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="one",
        vocab_size=100,
    ) == [10, 21, 30, 40]
    assert construct_draft_tokens(
        targets,
        acceptance_case="zero",
        vocab_size=100,
    ) == [11, 20, 30, 40]

    try:
        construct_draft_tokens(
            [10],
            acceptance_case="partial",
            vocab_size=100,
        )
    except ValueError as exc:
        assert "partial" in str(exc)
    else:
        raise AssertionError("partial K=1 must fail")


def main():
    test_tinyvllm_backend_has_runtime_timer_dependency()
    test_dtype_tolerances_are_fixed()
    test_comparison_requires_tokens_acceptance_metadata_and_continuation()
    test_token_mismatch_is_no_go_even_when_numeric_error_is_small()
    test_acceptance_or_metadata_mismatch_is_no_go()
    test_numeric_mismatch_is_no_go()
    test_continuation_numeric_mismatch_is_no_go()
    test_missing_or_nonfinite_evidence_is_classified_strictly()
    test_less_than_16_continuation_steps_is_incomplete()
    test_build_case_payload_records_complete_evidence()
    test_run_case_validates_input_and_writes_backend_payload()
    test_run_case_accepts_all_isolated_policies()
    test_construct_draft_tokens_is_deterministic_for_all_acceptance_cases()
    print("native verifier oracle tests passed")


if __name__ == "__main__":
    main()
