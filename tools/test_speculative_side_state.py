from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

tinyvllm_package = types.ModuleType("tinyvllm")
tinyvllm_package.__path__ = [os.path.join(_REPO_ROOT, "tinyvllm")]
engine_package = types.ModuleType("tinyvllm.engine")
engine_package.__path__ = [
    os.path.join(_REPO_ROOT, "tinyvllm", "engine")
]
sys.modules.setdefault("tinyvllm", tinyvllm_package)
sys.modules.setdefault("tinyvllm.engine", engine_package)

from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateCallbacks,
    build_speculative_side_state_selection_rows,
    validate_speculative_side_state_receipt,
)


def _row(
    sequence_id,
    *,
    proposal_tokens,
    accepted_tokens,
    verify_input_count,
):
    plan = (
        None
        if verify_input_count == 0
        else SimpleNamespace(query_len=verify_input_count)
    )
    return SimpleNamespace(
        sequence_id=sequence_id,
        proposal=SimpleNamespace(token_ids=proposal_tokens),
        accepted_tokens=accepted_tokens,
        plan=plan,
    )


def test_selection_uses_consumed_inputs_not_emitted_outputs():
    rows = build_speculative_side_state_selection_rows((
        _row(
            7,
            proposal_tokens=(11, 12, 13, 14),
            accepted_tokens=(11, 12),
            verify_input_count=3,
        ),
    ))

    assert rows[0].sequence_id == 7
    assert rows[0].proposal_token_count == 4
    assert rows[0].accepted_draft_count == 2
    assert rows[0].verify_input_count == 3
    assert rows[0].committed_tail_input_count == 2
    assert rows[0].committed_input_count == 3


def test_fully_accepted_proposal_leaves_last_output_unconsumed():
    rows = build_speculative_side_state_selection_rows((
        _row(
            8,
            proposal_tokens=(21, 22, 23, 24),
            accepted_tokens=(21, 22, 23, 24),
            verify_input_count=3,
        ),
    ))

    assert rows[0].committed_tail_input_count == 3
    assert rows[0].committed_input_count == 4


def test_zero_accepted_drafts_commits_only_first_target_input():
    rows = build_speculative_side_state_selection_rows((
        _row(
            9,
            proposal_tokens=(31, 32, 33),
            accepted_tokens=(),
            verify_input_count=2,
        ),
    ))

    assert rows[0].committed_tail_input_count == 0
    assert rows[0].committed_input_count == 1


def test_one_token_proposal_has_no_verification_tail():
    rows = build_speculative_side_state_selection_rows((
        _row(
            10,
            proposal_tokens=(41,),
            accepted_tokens=(41,),
            verify_input_count=0,
        ),
    ))

    assert rows[0].verify_input_count == 0
    assert rows[0].committed_tail_input_count == 0
    assert rows[0].committed_input_count == 1


def test_duplicate_sequence_ids_fail():
    row = _row(
        11,
        proposal_tokens=(51, 52),
        accepted_tokens=(51,),
        verify_input_count=1,
    )

    with pytest.raises(
        ValueError,
        match="sequence IDs must be unique",
    ):
        build_speculative_side_state_selection_rows((row, row))


def test_accepted_tokens_must_be_exact_proposal_prefix():
    with pytest.raises(
        ValueError,
        match="exact proposal prefix",
    ):
        build_speculative_side_state_selection_rows((
            _row(
                12,
                proposal_tokens=(61, 62, 63),
                accepted_tokens=(61, 63),
                verify_input_count=2,
            ),
        ))


def test_verify_input_count_must_equal_proposal_length_minus_one():
    with pytest.raises(
        ValueError,
        match="proposal length minus one",
    ):
        build_speculative_side_state_selection_rows((
            _row(
                13,
                proposal_tokens=(71, 72, 73),
                accepted_tokens=(71,),
                verify_input_count=1,
            ),
        ))


def test_callbacks_require_five_callable_phases():
    callbacks = SpeculativeSideStateCallbacks(
        prepare=lambda sequences: object(),
        select=lambda handle, rows: object(),
        apply=lambda handle: {},
        seal=lambda handle: {},
        rollback=lambda handle: {},
    )

    assert callable(callbacks.prepare)
    assert callable(callbacks.select)
    assert callable(callbacks.apply)
    assert callable(callbacks.seal)
    assert callable(callbacks.rollback)


@pytest.mark.parametrize(
    "field_name",
    ("prepare", "select", "apply", "seal", "rollback"),
)
def test_callbacks_reject_non_callable_phase(field_name):
    callbacks = {
        "prepare": lambda sequences: object(),
        "select": lambda handle, rows: object(),
        "apply": lambda handle: {},
        "seal": lambda handle: {},
        "rollback": lambda handle: {},
    }
    callbacks[field_name] = None

    with pytest.raises(TypeError, match=f"{field_name} must be callable"):
        SpeculativeSideStateCallbacks(**callbacks)


def test_receipt_sequence_inventory_must_match_selection_inventory():
    selection_rows = build_speculative_side_state_selection_rows((
        _row(
            14,
            proposal_tokens=(81, 82),
            accepted_tokens=(81,),
            verify_input_count=1,
        ),
        _row(
            15,
            proposal_tokens=(91, 92),
            accepted_tokens=(),
            verify_input_count=1,
        ),
    ))
    receipt = {
        "operation": "select",
        "status": "selected",
        "transaction_id": "side-1",
        "sequence_ids": [14],
    }

    with pytest.raises(
        ValueError,
        match="sequence inventory mismatch",
    ):
        validate_speculative_side_state_receipt(
            receipt,
            expected_operation="select",
            expected_status="selected",
            expected_transaction_id="side-1",
            selection_rows=selection_rows,
        )


def test_receipt_validation_returns_tensor_free_normalized_identity():
    selection_rows = build_speculative_side_state_selection_rows((
        _row(
            16,
            proposal_tokens=(101, 102),
            accepted_tokens=(101,),
            verify_input_count=1,
        ),
    ))

    normalized = validate_speculative_side_state_receipt(
        {
            "operation": "select",
            "status": "selected",
            "transaction_id": "side-2",
            "sequence_ids": [16],
        },
        expected_operation="select",
        expected_status="selected",
        expected_transaction_id="side-2",
        selection_rows=selection_rows,
    )

    assert normalized == {
        "operation": "select",
        "status": "selected",
        "transaction_id": "side-2",
        "sequence_ids": (16,),
    }


@pytest.mark.parametrize(
    ("operation", "status"),
    (
        ("prepare", "prepared"),
        ("select", "selected"),
        ("apply", "applied"),
        ("seal", "sealed"),
        ("rollback", "rolled_back"),
    ),
)
def test_receipt_accepts_each_legal_operation_status_pair(
    operation,
    status,
):
    normalized = validate_speculative_side_state_receipt(
        {
            "operation": operation,
            "status": status,
            "transaction_id": "side-3",
            "sequence_ids": [17],
        },
        expected_operation=operation,
        expected_status=status,
        expected_transaction_id="side-3",
        expected_sequence_ids=(17,),
    )

    assert normalized["operation"] == operation
    assert normalized["status"] == status
