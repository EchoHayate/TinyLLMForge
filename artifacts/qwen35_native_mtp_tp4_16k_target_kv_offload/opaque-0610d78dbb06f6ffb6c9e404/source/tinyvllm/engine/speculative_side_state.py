from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


_RECEIPT_FIELDS = frozenset({
    "operation",
    "status",
    "transaction_id",
    "sequence_ids",
})
_OPERATION_STATUSES = {
    "prepare": "prepared",
    "select": "selected",
    "apply": "applied",
    "seal": "sealed",
    "rollback": "rolled_back",
}


def _validate_non_negative_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_sequence_ids(sequence_ids):
    if not isinstance(sequence_ids, (list, tuple)):
        raise ValueError("receipt sequence_ids must be a list or tuple")
    normalized = tuple(
        _validate_non_negative_integer(
            sequence_id,
            "receipt sequence ID",
        )
        for sequence_id in sequence_ids
    )
    if not normalized:
        raise ValueError("receipt sequence_ids must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("receipt sequence IDs must be unique")
    return normalized


@dataclass(frozen=True)
class SpeculativeSideStateSelectionRow:
    sequence_id: int
    proposal_token_count: int
    accepted_draft_count: int
    verify_input_count: int
    committed_tail_input_count: int
    committed_input_count: int


@dataclass(frozen=True)
class SpeculativeSideStateCallbacks:
    prepare: Callable[[tuple[object, ...]], object]
    select: Callable[
        [object, tuple[SpeculativeSideStateSelectionRow, ...]],
        object,
    ]
    apply: Callable[[object], object]
    seal: Callable[[object], object]
    rollback: Callable[[object], object]

    def __post_init__(self):
        for name in (
            "prepare",
            "select",
            "apply",
            "seal",
            "rollback",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


def build_speculative_side_state_selection_rows(
    prepared_rows: tuple[object, ...],
) -> tuple[SpeculativeSideStateSelectionRow, ...]:
    if not isinstance(prepared_rows, tuple) or not prepared_rows:
        raise ValueError("prepared_rows must be a non-empty tuple")

    result = []
    seen_sequence_ids = set()
    for prepared_row in prepared_rows:
        sequence_id = _validate_non_negative_integer(
            getattr(prepared_row, "sequence_id", None),
            "side-state sequence ID",
        )
        if sequence_id in seen_sequence_ids:
            raise ValueError(
                "side-state sequence IDs must be unique"
            )

        proposal_tokens = getattr(
            getattr(prepared_row, "proposal", None),
            "token_ids",
            None,
        )
        accepted_tokens = getattr(
            prepared_row,
            "accepted_tokens",
            None,
        )
        if not isinstance(proposal_tokens, tuple):
            raise ValueError("proposal token_ids must be a tuple")
        if not isinstance(accepted_tokens, tuple):
            raise ValueError("accepted_tokens must be a tuple")
        if accepted_tokens != proposal_tokens[:len(accepted_tokens)]:
            raise ValueError(
                "accepted tokens must be an exact proposal prefix"
            )

        plan = getattr(prepared_row, "plan", None)
        verify_input_count = (
            0
            if plan is None
            else _validate_non_negative_integer(
                getattr(plan, "query_len", None),
                "verify input count",
            )
        )
        expected_verify_input_count = max(
            0,
            len(proposal_tokens) - 1,
        )
        if verify_input_count != expected_verify_input_count:
            raise ValueError(
                "verify input count must equal proposal length minus one"
            )

        committed_tail_input_count = min(
            len(accepted_tokens),
            verify_input_count,
        )
        result.append(SpeculativeSideStateSelectionRow(
            sequence_id=sequence_id,
            proposal_token_count=len(proposal_tokens),
            accepted_draft_count=len(accepted_tokens),
            verify_input_count=verify_input_count,
            committed_tail_input_count=(
                committed_tail_input_count
            ),
            committed_input_count=(
                1 + committed_tail_input_count
            ),
        ))
        seen_sequence_ids.add(sequence_id)

    return tuple(result)


def validate_speculative_side_state_receipt(
    receipt,
    *,
    expected_operation,
    expected_status,
    expected_transaction_id=None,
    selection_rows=None,
    expected_sequence_ids=None,
):
    if not isinstance(receipt, dict):
        raise ValueError("side-state receipt must be a dict")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError(
            "side-state receipt fields must exactly match contract"
        )

    legal_status = _OPERATION_STATUSES.get(expected_operation)
    if legal_status is None or legal_status != expected_status:
        raise ValueError(
            "expected operation/status pair is not legal"
        )
    if receipt["operation"] != expected_operation:
        raise ValueError("side-state receipt operation mismatch")
    if receipt["status"] != expected_status:
        raise ValueError("side-state receipt status mismatch")

    transaction_id = receipt["transaction_id"]
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError(
            "side-state transaction_id must be a non-empty string"
        )
    if (
        expected_transaction_id is not None
        and transaction_id != expected_transaction_id
    ):
        raise ValueError(
            "side-state receipt transaction ID mismatch"
        )

    if selection_rows is not None:
        if expected_sequence_ids is not None:
            raise ValueError(
                "provide selection_rows or expected_sequence_ids, not both"
            )
        if not isinstance(selection_rows, tuple) or not selection_rows:
            raise ValueError(
                "selection_rows must be a non-empty tuple"
            )
        expected_sequence_ids = tuple(
            row.sequence_id
            for row in selection_rows
        )
    if expected_sequence_ids is None:
        raise ValueError(
            "expected side-state sequence inventory is required"
        )
    expected_sequence_ids = _validate_sequence_ids(
        expected_sequence_ids
    )
    sequence_ids = _validate_sequence_ids(
        receipt["sequence_ids"]
    )
    if sequence_ids != expected_sequence_ids:
        raise ValueError(
            "side-state receipt sequence inventory mismatch"
        )

    return {
        "operation": expected_operation,
        "status": expected_status,
        "transaction_id": transaction_id,
        "sequence_ids": sequence_ids,
    }
