"""Tests for the Qwen3.5 TP1 real root-logit correctness contract.

Run: python3 tools/test_qwen35_tp1_real_root_logit_correctness_contract.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = (
    THIS_DIR / "qwen35_tp1_real_root_logit_correctness_contract.py"
)
SPEC = importlib.util.spec_from_file_location(
    "qwen35_tp1_real_root_logit_correctness_contract_under_test",
    os.fspath(CONTRACT_PATH),
)
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)

import torch


P17 = (
    237734, 105227, 220508, 88001, 203282, 70775, 186056, 53549,
    168830, 36323, 151604, 19097, 134378, 1871, 117152, 232433,
    99926,
)
P65 = (
    72098, 187379, 54872, 170153, 37646, 152927, 20420, 135701,
    3194, 118475, 233756, 101249, 216530, 84023, 199304, 66797,
    182078, 49571, 164852, 32345, 147626, 15119, 130400, 245681,
    113174, 228455, 95948, 211229, 78722, 194003, 61496, 176777,
    44270, 159551, 27044, 142325, 9818, 125099, 240380, 107873,
    223154, 90647, 205928, 73421, 188702, 56195, 171476, 38969,
    154250, 21743, 137024, 4517, 119798, 235079, 102572, 217853,
    85346, 200627, 68120, 183401, 50894, 166175, 33668, 148949,
    16442,
)
SYNTHETIC = (
    128, 129, 255, 256, 1024, 32768, 65536, 124022, 186033,
    247787, 248043,
)
EXPECTED_CASES = {
    "p17": (
        P17,
        "be8a139b93467e0b0ed92999e8feec6de8fbaac4a2c4faf4786f798bb00cceb9",
    ),
    "p65": (
        P65,
        "2391c5bbc31e842e8c362e591458d05541b1566409f03672d192fe6a9702a264",
    ),
    "synthetic": (
        SYNTHETIC,
        "a36985347858070c7c917b110c793414192e691ffe160be66276b6022c940819",
    ),
}


def _expect_value_error(callable_, message_fragment):
    try:
        callable_()
    except ValueError as exc:
        assert message_fragment in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _token_sha256(token_ids):
    payload = json.dumps(
        list(token_ids),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _float32_sha256(values):
    payload = struct.pack(f"<{len(values)}f", *values)
    return hashlib.sha256(payload).hexdigest()


def test_prompt_cases_are_exact_frozen_and_tokenizer_range_safe():
    cases = contract.prompt_cases()
    assert tuple(case.case_id for case in cases) == (
        "p17",
        "p65",
        "synthetic",
    )
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        expected_tokens, expected_sha256 = EXPECTED_CASES[case.case_id]
        assert case.token_ids == expected_tokens
        assert case.token_sha256 == expected_sha256
        assert case.token_sha256 == _token_sha256(case.token_ids)
        assert all(
            type(token_id) is int
            and 0 < token_id < contract.TOKENIZER_VOCAB_SIZE
            for token_id in case.token_ids
        )
    assert tuple(len(case.token_ids) for case in cases) == (17, 65, 11)


def test_prompt_cases_are_closed_constants_without_probe_or_tokenizer_calls():
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    assert "qwen35_hybrid_state_probe" not in source
    assert "qwen35_hybrid_state_contract" not in source
    assert "AutoTokenizer" not in source
    assert "tokenizer(" not in source
    assert "deterministic_token_ids" not in source


def test_prompt_case_rejects_drift_and_invalid_tokens():
    _expect_value_error(
        lambda: contract.PromptCase("bad", (1,), "0" * 64),
        "token_sha256",
    )
    _expect_value_error(
        lambda: contract.PromptCase(
            "bad",
            (contract.TOKENIZER_VOCAB_SIZE,),
            _token_sha256((contract.TOKENIZER_VOCAB_SIZE,)),
        ),
        "token",
    )
    _expect_value_error(
        lambda: contract.PromptCase(
            "bad",
            (),
            _token_sha256(()),
        ),
        "token_ids",
    )


def test_bf16_tolerance_and_classification_domain_are_frozen():
    assert contract.BF16_DECISION_TOLERANCE == (
        contract.ComparisonTolerance(atol=2e-5, rtol=1e-5)
    )
    assert contract.FINAL_CLASSIFICATIONS == ("PASS", "NO_GO_LOGIT")


def test_compare_logits_reconstructs_all_metrics_and_stable_tie_ordering():
    official = torch.tensor(
        [1.0, 5.0, 5.0, -2.0, 3.0, 0.0, 4.0, 2.0],
        dtype=torch.float32,
    )
    native = torch.tensor(
        [1.5, 5.0, 5.0, -1.0, 2.5, 0.0, 4.25, 2.0],
        dtype=torch.float32,
    )
    tolerance = contract.ComparisonTolerance(atol=0.25, rtol=0.0)
    row = contract.compare_logits(
        native,
        official,
        tolerance=tolerance,
        topk=8,
    )

    assert row["shape"] == [8]
    assert row["source_dtype"] == "float32"
    assert row["comparison_dtype"] == "float32"
    assert row["native_full_logit_sha256"] == _float32_sha256(
        native.tolist()
    )
    assert row["official_full_logit_sha256"] == _float32_sha256(
        official.tolist()
    )
    assert row["native_topk_token_ids"] == [1, 2, 6, 4, 7, 0, 5, 3]
    assert row["official_topk_token_ids"] == [1, 2, 6, 4, 7, 0, 5, 3]
    assert row["native_winner_token_id"] == 1
    assert row["native_runner_up_token_id"] == 2
    assert row["native_winner_margin"] == 0.0
    assert row["official_winner_token_id"] == 1
    assert row["official_runner_up_token_id"] == 2
    assert row["official_winner_margin"] == 0.0
    assert row["max_abs_diff"] == 1.0
    assert row["mean_abs_diff"] == 0.28125
    assert row["abs_diff_percentiles"] == {
        "p50": 0.125,
        "p95": 0.8250000476837158,
        "p99": 0.9650001525878906,
        "p99_9": 0.9965000152587891,
    }
    expected_cosine = torch.nn.functional.cosine_similarity(
        native.reshape(1, -1),
        official.reshape(1, -1),
    ).item()
    assert math.isclose(
        row["cosine_similarity"],
        expected_cosine,
        rel_tol=0.0,
        abs_tol=1e-7,
    )
    assert row["allclose_violation_count"] == 3
    assert row["max_allclose_scaled_error"] == 4.0


def test_compare_logits_rejects_invalid_rows_and_tolerance():
    tolerance = contract.ComparisonTolerance(atol=2e-5, rtol=1e-5)
    _expect_value_error(
        lambda: contract.compare_logits(
            torch.ones(4),
            torch.ones(5),
            tolerance=tolerance,
            topk=4,
        ),
        "shape",
    )
    _expect_value_error(
        lambda: contract.compare_logits(
            torch.tensor([1.0, float("nan")]),
            torch.ones(2),
            tolerance=tolerance,
            topk=2,
        ),
        "finite",
    )
    _expect_value_error(
        lambda: contract.ComparisonTolerance(atol=-1.0, rtol=0.0),
        "atol",
    )


def test_classification_preserves_positive_winner_and_zero_margin_tie():
    tolerance = contract.ComparisonTolerance(atol=2e-5, rtol=1e-5)
    positive = contract.compare_logits(
        torch.tensor([0.0, 1.0, 3.0, 2.0]),
        torch.tensor([0.0, 1.0, 3.0, 2.0]),
        tolerance=tolerance,
        topk=4,
    )
    tie = contract.compare_logits(
        torch.tensor([0.0, 4.0, 4.0, 1.0]),
        torch.tensor([0.0, 4.0, 4.0, 1.0]),
        tolerance=tolerance,
        topk=4,
    )
    assert contract.classify_rows([positive, tie]) == "PASS"

    changed_winner = contract.compare_logits(
        torch.tensor([0.0, 2.0, 3.0, 4.0]),
        torch.tensor([0.0, 1.0, 3.0, 2.0]),
        tolerance=tolerance,
        topk=4,
    )
    assert contract.classify_rows([changed_winner]) == "NO_GO_LOGIT"

    broken_tie = contract.compare_logits(
        torch.tensor([0.0, 4.25, 4.0, 1.0]),
        torch.tensor([0.0, 4.0, 4.0, 1.0]),
        tolerance=tolerance,
        topk=4,
    )
    assert contract.classify_rows([broken_tie]) == "NO_GO_LOGIT"


def test_main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and name != "test_main"
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} tests")


if __name__ == "__main__":
    test_main()
