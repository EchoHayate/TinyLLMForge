from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
    "tinyvllm.utils",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

fake_torch = types.ModuleType("torch")


class _Tensor:
    pass


fake_torch.Tensor = _Tensor
original_torch = sys.modules.get("torch")
sys.modules["torch"] = fake_torch

fake_context = types.ModuleType("tinyvllm.utils.context")


@contextmanager
def _temporary_context(**_):
    yield


fake_context.temporary_context = _temporary_context
original_context = sys.modules.get("tinyvllm.utils.context")
sys.modules["tinyvllm.utils.context"] = fake_context

try:
    from tinyvllm.engine.qwen35_mtp_executor import (
        Qwen35MTPProposalExecutor,
        _BootstrappedSequence,
    )
    from tinyvllm.engine.speculative_proposal_executor import (
        ModelRunnerProposalInput,
    )
    from tinyvllm.speculative.adapter import DraftProposal
finally:
    if original_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original_torch
    if original_context is None:
        sys.modules.pop("tinyvllm.utils.context", None)
    else:
        sys.modules["tinyvllm.utils.context"] = original_context


@dataclass
class _Transaction:
    transaction_id: str
    sequence_id: int
    sequence_epoch: int
    state: str = "materialized"


class _Cache:

    def __init__(self, transactions):
        self.transactions = {
            transaction.transaction_id: transaction
            for transaction in transactions
        }
        self.abort_calls = []

    def transaction(self, transaction_id):
        return self.transactions.get(transaction_id)

    def abort(self, transaction_id):
        transaction = self.transactions[transaction_id]
        transaction.state = "aborted"
        self.abort_calls.append(transaction_id)


def _row(sequence_id):
    return (
        ModelRunnerProposalInput(
            sequence_id=sequence_id,
            token_ids=(1, 2),
            remaining_output_tokens=2,
            max_proposal_tokens=2,
            first_target_token=3,
        ),
        _BootstrappedSequence(
            sequence_id=sequence_id,
            sequence_epoch=5,
            prefix_token_count=2,
        ),
    )


def _proposal(sequence_id, transaction_id):
    return DraftProposal(
        sequence_id=sequence_id,
        token_ids=(3, 4),
        source_type="native_model_runner",
        metadata={
            "exact_q": 2,
            "staged_entry_count": 1,
            "execution_mode": "cuda_graph",
        },
        proposal_transaction_id=transaction_id,
    )


def _executor(*transactions):
    executor = object.__new__(Qwen35MTPProposalExecutor)
    executor.proposal_kv_cache = _Cache(transactions)
    executor._proposal_transactions = {}
    return executor


def test_graph_group_transactions_are_registered_atomically():
    transactions = (
        _Transaction("tx-7", 7, 5),
        _Transaction("tx-9", 9, 5),
    )
    executor = _executor(*transactions)
    proposals = (
        _proposal(7, "tx-7"),
        _proposal(9, "tx-9"),
    )
    rows = (_row(7), _row(9))

    result = executor._register_group_proposals(
        proposals,
        rows,
    )

    assert result is proposals
    assert executor._proposal_transactions == {
        "tx-7": (7, 5),
        "tx-9": (9, 5),
    }


def test_group_sequence_mismatch_publishes_nothing_and_aborts():
    transactions = (
        _Transaction("tx-7", 7, 5),
        _Transaction("tx-9", 9, 5),
    )
    executor = _executor(*transactions)
    proposals = (
        _proposal(7, "tx-7"),
        _proposal(11, "tx-9"),
    )

    with pytest.raises(ValueError, match="sequence"):
        executor._register_group_proposals(
            proposals,
            (_row(7), _row(9)),
        )

    assert executor._proposal_transactions == {}
    assert executor.proposal_kv_cache.abort_calls == [
        "tx-9",
        "tx-7",
    ]


def test_duplicate_transaction_id_publishes_nothing_and_aborts_once():
    transaction = _Transaction("tx-7", 7, 5)
    executor = _executor(transaction)
    proposals = (
        _proposal(7, "tx-7"),
        _proposal(9, "tx-7"),
    )

    with pytest.raises(ValueError, match="unique"):
        executor._register_group_proposals(
            proposals,
            (_row(7), _row(9)),
        )

    assert executor._proposal_transactions == {}
    assert executor.proposal_kv_cache.abort_calls == ["tx-7"]


def test_unknown_transaction_publishes_nothing():
    executor = _executor()

    with pytest.raises(ValueError, match="active"):
        executor._register_group_proposals(
            (_proposal(7, "missing"),),
            (_row(7),),
        )

    assert executor._proposal_transactions == {}

