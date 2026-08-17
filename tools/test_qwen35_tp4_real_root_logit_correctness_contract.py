from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "tools/qwen35_tp4_real_root_logit_correctness_contract.py"
)


def _load_contract():
    spec = importlib.util.spec_from_file_location(
        "qwen35_tp4_real_root_logit_correctness_contract",
        CONTRACT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_contract()


def _expect_value_error(function, message):
    try:
        function()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def _gpu_row(rank, *, gpu_index=None, gpu_uuid=None):
    return {
        "rank": rank,
        "world_size": 4,
        "gpu_index": rank if gpu_index is None else gpu_index,
        "gpu_uuid": (
            f"GPU-00000000-0000-0000-0000-00000000000{rank}"
            if gpu_uuid is None
            else gpu_uuid
        ),
    }


def test_contract_requires_strict_registered_logit_tolerance():
    cases = contract.prompt_cases()

    assert tuple(case.case_id for case in cases) == (
        "p17",
        "p65",
        "synthetic",
    )
    assert contract.BF16_DECISION_TOLERANCE.atol == 2e-5
    assert contract.BF16_DECISION_TOLERANCE.rtol == 0.0
    assert contract.WORLD_SIZE == 4
    assert contract.MODEL_VOCAB_SIZE == 248320


def test_rank_output_contract_accepts_only_root_tensor():
    root_logits = torch.zeros(
        contract.MODEL_VOCAB_SIZE,
        dtype=torch.float32,
    )

    root = contract.validate_rank_logits(
        rank=0,
        world_size=4,
        logits=root_logits,
    )
    assert root is root_logits
    for rank in (1, 2, 3):
        assert contract.validate_rank_logits(
            rank=rank,
            world_size=4,
            logits=None,
        ) is None


def test_rank_output_contract_rejects_invalid_roles_and_shapes():
    complete = torch.zeros(
        contract.MODEL_VOCAB_SIZE,
        dtype=torch.float32,
    )
    invalid_cases = (
        (
            lambda: contract.validate_rank_logits(
                rank=0,
                world_size=3,
                logits=complete,
            ),
            "world size",
        ),
        (
            lambda: contract.validate_rank_logits(
                rank=4,
                world_size=4,
                logits=complete,
            ),
            "rank",
        ),
        (
            lambda: contract.validate_rank_logits(
                rank=0,
                world_size=4,
                logits=None,
            ),
            "rank zero",
        ),
        (
            lambda: contract.validate_rank_logits(
                rank=2,
                world_size=4,
                logits=complete,
            ),
            "non-root",
        ),
        (
            lambda: contract.validate_rank_logits(
                rank=0,
                world_size=4,
                logits=torch.zeros(
                    contract.MODEL_VOCAB_SIZE - 1,
                    dtype=torch.float32,
                ),
            ),
            "vocabulary",
        ),
        (
            lambda: contract.validate_rank_logits(
                rank=0,
                world_size=4,
                logits=torch.zeros(
                    contract.MODEL_VOCAB_SIZE,
                    dtype=torch.bfloat16,
                ),
            ),
            "float32",
        ),
        (
            lambda: contract.validate_rank_logits(
                rank=0,
                world_size=4,
                logits=torch.full(
                    (contract.MODEL_VOCAB_SIZE,),
                    float("nan"),
                ),
            ),
            "finite",
        ),
    )
    for function, message in invalid_cases:
        _expect_value_error(function, message)


def test_rank_topology_requires_tp4_two_query_one_replicated_kv():
    valid = contract.validate_rank_topology({
        "rank": 2,
        "world_size": 4,
        "global_query_heads": 8,
        "global_kv_heads": 2,
        "local_query_heads": 2,
        "local_kv_heads": 1,
        "kv_head_replicas": 2,
        "source_kv_rank": 1,
    })
    assert valid["rank"] == 2
    assert valid["source_kv_rank"] == 1

    for field, value in (
        ("local_query_heads", 1),
        ("local_kv_heads", 2),
        ("kv_head_replicas", 1),
        ("source_kv_rank", 0),
    ):
        row = dict(valid)
        row[field] = value
        _expect_value_error(
            lambda row=row: contract.validate_rank_topology(row),
            field,
        )


def test_gpu_assignments_require_four_unique_ranked_devices():
    rows = [_gpu_row(rank) for rank in range(4)]

    validated = contract.validate_gpu_assignments(rows)

    assert tuple(row["rank"] for row in validated) == (0, 1, 2, 3)
    duplicate_index = [_gpu_row(rank) for rank in range(4)]
    duplicate_index[3]["gpu_index"] = 0
    _expect_value_error(
        lambda: contract.validate_gpu_assignments(duplicate_index),
        "GPU indices",
    )
    duplicate_uuid = [_gpu_row(rank) for rank in range(4)]
    duplicate_uuid[3]["gpu_uuid"] = duplicate_uuid[0]["gpu_uuid"]
    _expect_value_error(
        lambda: contract.validate_gpu_assignments(duplicate_uuid),
        "GPU UUIDs",
    )
    _expect_value_error(
        lambda: contract.validate_gpu_assignments(rows[:3]),
        "four",
    )


def test_classification_rejects_decision_preserving_non_strict_logits():
    official = torch.zeros(contract.MODEL_VOCAB_SIZE)
    official[17] = 2.0
    official[18] = 1.0
    native = official.clone()
    native[17] = 1.75
    native[18] = 1.25
    row = contract.compare_logits(
        native,
        official,
        tolerance=contract.BF16_DECISION_TOLERANCE,
    )

    assert row["native_winner_token_id"] == row["official_winner_token_id"]
    assert row["allclose_violation_count"] > 0
    assert contract.classify_rows((row,)) == "NO_GO_LOGIT"

    changed = native.clone()
    changed[18] = 3.0
    changed_row = contract.compare_logits(
        changed,
        official,
        tolerance=contract.BF16_DECISION_TOLERANCE,
    )
    assert contract.classify_rows((changed_row,)) == "NO_GO_LOGIT"


def _run():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"qwen35 TP4 root-logit contract tests passed ({len(tests)} tests)")


if __name__ == "__main__":
    _run()
