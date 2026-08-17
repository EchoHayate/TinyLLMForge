from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for package_name in (
    "tinyvllm",
    "tinyvllm.engine",
    "tinyvllm.speculative",
):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)

from tinyvllm.engine.proposal_kv_cache import ProposalKVCache
from tinyvllm.engine.proposal_kv_allocator import (
    ProposalKVEntryIdentity,
    ProposalKVResidencyLease,
)
from tinyvllm.engine.proposal_kv_lifecycle import (
    ProposalKVLifecycleCoordinator,
)
from tinyvllm.engine.qwen35_mtp_executor import (
    Qwen35MTPProposalExecutor,
)
from tinyvllm.engine.speculative_proposal_executor import (
    ModelRunnerProposalExecutorRegistry,
    ModelRunnerProposalInput,
    ProposalFinalizeRow,
    TargetPrefillObservation,
)
from tinyvllm.utils.context import get_context


class _Allocator:

    def __init__(self):
        self.free_logical_ids = list(range(128))
        self.generations = [0] * 128
        self.reserve_calls = []
        self.commit_calls = []
        self.retire_calls = []
        self.read_completions = []
        self.write_completions = []

    def reserve_entries(self, count):
        self.reserve_calls.append(count)
        logical_ids = tuple(self.free_logical_ids[:count])
        del self.free_logical_ids[:count]
        identities = []
        for logical_id in logical_ids:
            self.generations[logical_id] += 1
            identities.append(
                ProposalKVEntryIdentity(
                    logical_id,
                    self.generations[logical_id],
                )
            )
        return tuple(identities)

    @staticmethod
    def _lease(identities):
        return ProposalKVResidencyLease(
            identities=tuple(identities),
            physical_slot_ids=tuple(
                identity.logical_entry_id for identity in identities
            ),
            occupancy_generations=tuple(
                identity.generation for identity in identities
            ),
        )

    def ensure_writable(self, identities):
        return self._lease(identities)

    def ensure_readable(self, identities):
        return self._lease(identities)

    def record_write_complete(self, lease):
        self.write_completions.append(lease)

    def record_read_complete(self, lease):
        self.read_completions.append(lease)

    def commit_entries(self, identities):
        self.commit_calls.append(tuple(identities))

    def retire_entries(self, identities, *, writeback):
        assert writeback is False
        identities = tuple(identities)
        self.retire_calls.append(identities)
        self.free_logical_ids.extend(
            identity.logical_entry_id for identity in identities
        )
        self.free_logical_ids.sort()

    def authority_snapshot(self):
        return {}


class _FakeMTP:

    def __init__(self, hidden_size=4, vocab_size=64):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.calls = []
        self.hidden_only_calls = []
        self.contexts = []
        self.inference_modes = []

    def _record_context(self):
        context = get_context()
        self.contexts.append({
            "mode": context.mode,
            "is_prefill": context.is_prefill,
            "slot_mapping": (
                None
                if context.slot_mapping is None
                else context.slot_mapping.detach().cpu().clone()
            ),
            "block_tables": (
                None
                if context.block_tables is None
                else context.block_tables.detach().cpu().clone()
            ),
            "context_lens": (
                None
                if context.context_lens is None
                else context.context_lens.detach().cpu().clone()
            ),
            "cu_seqlens_q": (
                None
                if context.cu_seqlens_q is None
                else context.cu_seqlens_q.detach().cpu().clone()
            ),
            "cu_seqlens_k": (
                None
                if context.cu_seqlens_k is None
                else context.cu_seqlens_k.detach().cpu().clone()
            ),
        })

    def _next_hidden(self, input_ids, hidden_states):
        token_values = input_ids.to(hidden_states.dtype).unsqueeze(-1)
        return hidden_states + token_values

    def forward_hidden(self, input_ids, positions, hidden_states):
        self._record_context()
        self.hidden_only_calls.append((
            input_ids.clone(),
            positions.clone(),
            hidden_states.clone(),
        ))
        return self._next_hidden(input_ids, hidden_states)

    def forward_step(self, input_ids, positions, hidden_states):
        self.inference_modes.append(
            torch.is_inference_mode_enabled()
        )
        self._record_context()
        self.calls.append((
            input_ids.clone(),
            positions.clone(),
            hidden_states.clone(),
        ))
        output_hidden = self._next_hidden(input_ids, hidden_states)
        logits = torch.full(
            (input_ids.shape[0], self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        next_tokens = (input_ids + 1) % self.vocab_size
        logits.scatter_(1, next_tokens.unsqueeze(-1), 1000.0)
        return output_hidden, logits


class _TensorParallelFakeMTP(_FakeMTP):

    def __init__(
        self,
        rank,
        *,
        worker_returns_logits=False,
        root_returns_none=False,
    ):
        super().__init__()
        self.rank = rank
        self.worker_returns_logits = worker_returns_logits
        self.root_returns_none = root_returns_none

    def forward_step(self, input_ids, positions, hidden_states):
        output_hidden, logits = super().forward_step(
            input_ids,
            positions,
            hidden_states,
        )
        if self.rank == 0 and self.root_returns_none:
            return output_hidden, None
        if self.rank != 0 and not self.worker_returns_logits:
            return output_hidden, None
        return output_hidden, logits


class _ReplayBroadcastBus:

    def __init__(self):
        self.payloads = []
        self.worker_offsets = {}

    def root(self, tensor, src):
        assert src == 0
        self.payloads.append(tensor.detach().clone())

    def worker(self, rank):
        self.worker_offsets[rank] = 0

        def receive(tensor, src):
            assert src == 0
            offset = self.worker_offsets[rank]
            tensor.copy_(self.payloads[offset])
            self.worker_offsets[rank] = offset + 1

        return receive


def _executor(
    max_proposal_tokens=6,
    graph_runner=None,
    *,
    module=None,
    tensor_parallel_rank=0,
    tensor_parallel_size=1,
    token_broadcast=None,
):
    module = _FakeMTP() if module is None else module
    allocator = _Allocator()
    cache = ProposalKVCache(allocator)
    executor = Qwen35MTPProposalExecutor(
        module=module,
        proposal_kv_cache=cache,
        max_proposal_tokens=max_proposal_tokens,
        graph_runner=graph_runner,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_size=tensor_parallel_size,
        token_broadcast=token_broadcast,
    )
    return executor, module, cache, allocator


def test_executor_uses_generic_proposal_kv_lifecycle_coordinator():
    executor, _, _, _ = _executor()

    assert isinstance(
        executor.proposal_kv_lifecycle,
        ProposalKVLifecycleCoordinator,
    )
    assert (
        executor.proposal_kv_lifecycle.proposal_kv_cache
        is executor.proposal_kv_cache
    )


def _observation(
    token_ids,
    *,
    sequence_id=7,
    epoch=0,
    start=0,
    final=False,
    hidden_size=4,
    dtype=torch.float32,
):
    count = len(token_ids)
    return TargetPrefillObservation(
        sequence_id=sequence_id,
        sequence_epoch=epoch,
        token_ids=tuple(token_ids),
        positions=torch.arange(
            start,
            start + count,
            dtype=torch.int64,
        ),
        target_hidden=torch.arange(
            count * hidden_size,
            dtype=torch.float32,
        ).reshape(count, hidden_size).to(dtype),
        is_final_chunk=final,
    )


def _input(
    q,
    *,
    sequence_id=7,
    first_target=13,
    target_hidden=None,
    token_ids=(10, 11, 12),
    context_token_count=None,
):
    if target_hidden is None:
        target_hidden = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]],
            dtype=torch.float32,
        )
    return ModelRunnerProposalInput(
        sequence_id=sequence_id,
        token_ids=tuple(token_ids),
        remaining_output_tokens=q,
        max_proposal_tokens=q,
        first_target_token=first_target,
        target_hidden=target_hidden,
        context_token_count=context_token_count,
    )


def _observe_complete_prompt(executor, sequence_id=7, epoch=0):
    first = _observation(
        (10, 11),
        sequence_id=sequence_id,
        epoch=epoch,
        start=0,
        final=False,
    )
    second = _observation(
        (12,),
        sequence_id=sequence_id,
        epoch=epoch,
        start=2,
        final=True,
    )
    executor.observe_target_prefill((first,))
    executor.observe_target_prefill((second,))
    return first, second


def _finalize(executor, proposal, accepted=None, *, commit=True):
    if not proposal.token_ids:
        return None
    accepted = (
        len(proposal.token_ids)
        if accepted is None
        else accepted
    )
    ticket = executor.prepare_finalize_batch((
        ProposalFinalizeRow(
            sequence_id=proposal.sequence_id,
            proposal_transaction_id=(
                proposal.proposal_transaction_id
            ),
            accepted_proposal_tokens=accepted,
        ),
    ))
    if commit:
        executor.commit_finalize_batch(ticket)
    else:
        executor.rollback_finalize_batch(ticket)
    return ticket


def _bootstrap(executor, module):
    _observe_complete_prompt(executor)
    proposal = executor.propose_batch((_input(1),))[0]
    assert proposal.token_ids == (13,)
    _finalize(executor, proposal)
    module.calls.clear()
    module.hidden_only_calls.clear()
    module.contexts.clear()


def test_prefill_accumulates_contiguous_chunks_and_final_state():
    executor, _, _, _ = _executor()
    first, second = _observe_complete_prompt(executor)
    pending = executor.pending_prefix(7)
    assert pending.sequence_id == 7
    assert pending.sequence_epoch == 0
    assert pending.token_ids == (10, 11, 12)
    assert pending.is_final is True
    torch.testing.assert_close(
        pending.positions,
        torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(
        pending.target_hidden,
        torch.cat((
            first.target_hidden,
            second.target_hidden,
        )),
    )


@pytest.mark.parametrize(
    "second_overrides,message",
    (
        ({"start": 3}, "contiguous"),
        ({"start": 1}, "contiguous"),
        ({"epoch": 1}, "epoch"),
        ({"hidden_size": 5}, "hidden width"),
        ({"dtype": torch.float64}, "hidden dtype"),
    ),
)
def test_prefill_rejects_chunk_drift(second_overrides, message):
    executor, _, _, _ = _executor()
    executor.observe_target_prefill((
        _observation((10, 11), start=0),
    ))
    values = {
        "token_ids": (12,),
        "start": 2,
        "epoch": 0,
        "hidden_size": 4,
        "dtype": torch.float32,
        "final": True,
    }
    values.update(second_overrides)
    with pytest.raises((ValueError, RuntimeError), match=message):
        executor.observe_target_prefill((_observation(**values),))


def test_prefill_rejects_duplicate_final_and_second_bootstrap():
    executor, _, _, _ = _executor()
    _observe_complete_prompt(executor)
    with pytest.raises(RuntimeError, match="final"):
        executor.observe_target_prefill((
            _observation((13,), start=3, final=True),
        ))
    proposal = executor.propose_batch((_input(1),))[0]
    _finalize(executor, proposal)
    with pytest.raises(RuntimeError, match="bootstrap"):
        executor.observe_target_prefill((
            _observation((10,), start=0, final=True),
        ))


def test_bootstrap_uses_shifted_tokens_and_discards_logits():
    executor, module, cache, store = _executor()
    first, second = _observe_complete_prompt(executor)
    target_hidden = torch.tensor(
        [[9.0, 8.0, 7.0, 6.0]],
        dtype=torch.float32,
    )

    proposal = executor.propose_batch((
        _input(1, target_hidden=target_hidden),
    ))[0]

    assert proposal.token_ids == (13,)
    assert module.calls == []
    assert len(module.hidden_only_calls) == 1
    input_ids, positions, hidden = module.hidden_only_calls[0]
    torch.testing.assert_close(
        input_ids,
        torch.tensor([11, 12, 13]),
    )
    torch.testing.assert_close(
        positions,
        torch.tensor([0, 1, 2]),
    )
    torch.testing.assert_close(
        hidden,
        torch.cat((
            first.target_hidden,
            second.target_hidden,
        )),
    )
    assert cache.committed_length(7) == 3
    assert store.reserve_calls[:2] == [3, 0]
    assert executor.pending_prefix(7) is None
    assert len(module.contexts) == 1
    context = module.contexts[0]
    assert context["mode"] == "prefill"
    assert context["is_prefill"] is True
    torch.testing.assert_close(
        context["slot_mapping"],
        torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        context["cu_seqlens_q"],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        context["cu_seqlens_k"],
        torch.tensor([0, 3], dtype=torch.int32),
    )
    assert context["block_tables"] is None
    assert context["context_lens"] is None


@pytest.mark.parametrize(
    "q,expected_tokens,expected_calls",
    (
        (0, (), 0),
        (1, (13,), 0),
        (4, (13, 14, 15, 16), 3),
    ),
)
def test_exact_q_proposal_semantics(q, expected_tokens, expected_calls):
    executor, module, cache, store = _executor()
    _bootstrap(executor, module)
    target_hidden = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0]],
        dtype=torch.float32,
    )

    proposal = executor.propose_batch((
        _input(q, target_hidden=target_hidden),
    ))[0]

    assert proposal.token_ids == expected_tokens
    assert len(module.calls) == expected_calls
    if q == 0:
        assert proposal.proposal_transaction_id is None
        return
    assert isinstance(proposal.proposal_transaction_id, str)
    assert proposal.proposal_transaction_id
    assert proposal.metadata == {
        "exact_q": q,
        "staged_entry_count": max(q - 1, 0),
    }
    assert store.reserve_calls[-1] == max(q - 1, 0)
    if q == 4:
        torch.testing.assert_close(
            module.calls[0][2],
            target_hidden,
        )
        torch.testing.assert_close(
            module.calls[1][2],
            module.calls[0][2]
            + module.calls[0][0].to(
                module.calls[0][2].dtype
            ).unsqueeze(-1),
        )
        assert [call[0].item() for call in module.calls] == [
            13,
            14,
            15,
        ]
    _finalize(executor, proposal)
    assert cache.committed_length(7) == 3 + max(q - 1, 0)


def test_proposal_forward_step_runs_in_inference_mode():
    executor, module, _, _ = _executor()
    _observe_complete_prompt(executor)

    proposal = executor.propose_batch((_input(2),))[0]

    assert proposal.token_ids == (13, 14)
    assert module.inference_modes == [True]


def test_attention_context_keeps_accepted_slots_for_next_continuation():
    executor, module, cache, store = _executor()
    _bootstrap(executor, module)

    first = executor.propose_batch((_input(4),))[0]

    assert len(module.contexts) == 3
    for step, context in enumerate(module.contexts):
        assert context["mode"] == "decode"
        assert context["is_prefill"] is False
        torch.testing.assert_close(
            context["slot_mapping"],
            torch.tensor([3 + step], dtype=torch.int32),
        )
        torch.testing.assert_close(
            context["block_tables"],
            torch.tensor(
                [list(range(4 + step))],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            context["context_lens"],
            torch.tensor([4 + step], dtype=torch.int32),
        )
    first_staged = cache._transactions[
        first.proposal_transaction_id
    ].staged_entry_identities
    assert tuple(
        identity.logical_entry_id for identity in first_staged
    ) == (3, 4, 5)
    _finalize(executor, first, accepted=2)
    assert tuple(
        identity.logical_entry_id
        for identity in cache.committed_entry_identities(7)
    ) == (0, 1, 2, 3)
    assert 4 in store.free_logical_ids
    assert 5 in store.free_logical_ids

    module.calls.clear()
    module.contexts.clear()
    continuation = executor.propose_batch((_input(2),))[0]

    assert continuation.token_ids == (13, 14)
    assert len(module.contexts) == 1
    context = module.contexts[0]
    torch.testing.assert_close(
        context["slot_mapping"],
        torch.tensor([4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        context["block_tables"],
        torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        context["context_lens"],
        torch.tensor([5], dtype=torch.int32),
    )
    assert 5 not in context["block_tables"].tolist()[0]
    _finalize(executor, continuation, commit=False)


def test_effective_q_respects_all_limits_and_preserves_batch_order():
    executor, module, _, _ = _executor(max_proposal_tokens=4)
    _observe_complete_prompt(executor, sequence_id=7)
    _observe_complete_prompt(executor, sequence_id=9)
    rows = (
        _input(6, sequence_id=7, first_target=13),
        ModelRunnerProposalInput(
            sequence_id=9,
            token_ids=(10, 11, 12),
            remaining_output_tokens=2,
            max_proposal_tokens=6,
            first_target_token=20,
            target_hidden=torch.ones(1, 4),
        ),
    )

    proposals = executor.propose_batch(rows)

    assert tuple(row.sequence_id for row in proposals) == (7, 9)
    assert proposals[0].token_ids == (13, 14, 15, 16)
    assert proposals[1].token_ids == (20, 21)
    assert len(module.hidden_only_calls) == 2
    assert len(module.calls) == 3 + 1


def test_compact_tp_worker_input_uses_explicit_context_token_count():
    executor, module, _, _ = _executor(max_proposal_tokens=4)
    _observe_complete_prompt(executor)

    proposal = executor.propose_batch((
        _input(
            2,
            token_ids=(),
            context_token_count=4097,
        ),
    ))[0]

    assert proposal.token_ids == (13, 14)
    assert len(module.calls) == 1
    torch.testing.assert_close(
        module.calls[0][1],
        torch.tensor([4096], dtype=torch.int64),
    )


class _RecordingGraphRunner:

    def __init__(self):
        self.events = []

    def run(self, *, exact_q, rows, eager):
        self.events.append((
            exact_q,
            tuple(row[0].sequence_id for row in rows),
        ))
        return eager(exact_q, rows)


def test_mixed_q_groups_are_exact_stable_and_never_padded():
    graph_runner = _RecordingGraphRunner()
    executor, _, _, _ = _executor(graph_runner=graph_runner)
    for sequence_id in (2, 4, 8, 9):
        _observe_complete_prompt(
            executor,
            sequence_id=sequence_id,
        )
    inputs = tuple(
        _input(
            q,
            sequence_id=sequence_id,
            first_target=20 + sequence_id,
        )
        for sequence_id, q in (
            (8, 2),
            (4, 4),
            (2, 2),
            (9, 3),
        )
    )

    proposals = executor.propose_batch(inputs)

    assert graph_runner.events == [
        (2, (8, 2)),
        (4, (4,)),
        (3, (9,)),
    ]
    assert tuple(len(row.token_ids) for row in proposals) == (
        2,
        4,
        2,
        3,
    )


def test_finalize_batch_commit_and_rollback_delegate_to_kv_cache():
    executor, module, cache, _ = _executor()
    _observe_complete_prompt(executor, sequence_id=7)
    _observe_complete_prompt(executor, sequence_id=9)
    proposals = executor.propose_batch((
        _input(3, sequence_id=7, first_target=13),
        _input(2, sequence_id=9, first_target=20),
    ))
    assert len(module.hidden_only_calls) == 2
    assert len(module.calls) == 2 + 1
    ticket = executor.prepare_finalize_batch(tuple(
        ProposalFinalizeRow(
            sequence_id=proposal.sequence_id,
            proposal_transaction_id=(
                proposal.proposal_transaction_id
            ),
            accepted_proposal_tokens=len(proposal.token_ids),
        )
        for proposal in proposals
    ))
    executor.commit_finalize_batch(ticket)
    assert cache.committed_length(7) == 3 + 2
    assert cache.committed_length(9) == 3 + 1

    later = executor.propose_batch((
        _input(2, sequence_id=7, first_target=30),
    ))[0]
    rollback_ticket = executor.prepare_finalize_batch((
        ProposalFinalizeRow(
            sequence_id=7,
            proposal_transaction_id=(
                later.proposal_transaction_id
            ),
            accepted_proposal_tokens=2,
        ),
    ))
    executor.rollback_finalize_batch(rollback_ticket)
    assert cache.committed_length(7) == 5


def test_registry_accepts_tensor_free_lifecycle_proposals():
    executor, _, _, _ = _executor()
    _observe_complete_prompt(executor)
    registry = ModelRunnerProposalExecutorRegistry()
    registry.register(
        "native_checkpoint_proposal",
        executor,
        executor.capabilities,
    )
    proposals = registry.execute_batch(
        "native_checkpoint_proposal",
        (_input(3),),
        executor.capabilities,
    )
    assert proposals[0].token_ids == (13, 14, 15)


def test_tp4_proposals_broadcast_tokens_and_keep_transaction_identity():
    bus = _ReplayBroadcastBus()
    executors = []
    for rank in range(4):
        broadcast = bus.root if rank == 0 else bus.worker(rank)
        executor, _, _, _ = _executor(
            module=_TensorParallelFakeMTP(rank),
            tensor_parallel_rank=rank,
            tensor_parallel_size=4,
            token_broadcast=broadcast,
        )
        _observe_complete_prompt(executor)
        executors.append(executor)

    proposals = tuple(
        executor.propose_batch((_input(4),))[0]
        for executor in executors
    )

    assert [proposal.token_ids for proposal in proposals] == [
        (13, 14, 15, 16),
        (13, 14, 15, 16),
        (13, 14, 15, 16),
        (13, 14, 15, 16),
    ]
    assert [
        proposal.proposal_transaction_id
        for proposal in proposals
    ] == ["proposal-kv-transaction-2"] * 4
    assert [
        (payload.dtype, tuple(payload.shape), payload.tolist())
        for payload in bus.payloads
    ] == [
        (torch.int64, (1,), [14]),
        (torch.int64, (1,), [15]),
        (torch.int64, (1,), [16]),
    ]
    assert bus.worker_offsets == {1: 3, 2: 3, 3: 3}


def test_tp4_worker_rejects_unexpected_logits():
    bus = _ReplayBroadcastBus()
    executor, _, _, _ = _executor(
        module=_TensorParallelFakeMTP(
            1,
            worker_returns_logits=True,
        ),
        tensor_parallel_rank=1,
        tensor_parallel_size=4,
        token_broadcast=bus.worker(1),
    )
    _observe_complete_prompt(executor)

    with pytest.raises(ValueError, match="non-root logits"):
        executor.propose_batch((_input(2),))


def test_tp4_root_rejects_missing_logits():
    executor, _, _, _ = _executor(
        module=_TensorParallelFakeMTP(
            0,
            root_returns_none=True,
        ),
        tensor_parallel_rank=0,
        tensor_parallel_size=4,
        token_broadcast=lambda *_args, **_kwargs: None,
    )
    _observe_complete_prompt(executor)

    with pytest.raises(ValueError, match="root logits"):
        executor.propose_batch((_input(2),))


def test_tp4_authority_snapshot_records_transaction_and_cleanup():
    executor, _, _, _ = _executor(
        module=_TensorParallelFakeMTP(3),
        tensor_parallel_rank=3,
        tensor_parallel_size=4,
        token_broadcast=lambda tensor, src: tensor.fill_(14),
    )
    _observe_complete_prompt(executor)
    proposal = executor.propose_batch((_input(4),))[0]
    ticket = _finalize(executor, proposal, accepted=2)
    executor.release_sequence(7, sequence_epoch=0)

    snapshot = executor.tp4_authority_snapshot()

    assert snapshot["tensor_parallel_rank"] == 3
    assert snapshot["tensor_parallel_size"] == 4
    assert snapshot["proposal_transactions"] == [{
        "sequence_id": 7,
        "sequence_epoch": 0,
        "transaction_id": "proposal-kv-transaction-2",
        "exact_q": 4,
        "token_ids": [13, 14, 14, 14],
        "staged_entry_count": 3,
        "accepted_proposal_tokens": 2,
        "rejected_proposal_tokens": 2,
        "finalize_ticket_id": ticket,
        "state": "committed",
    }]
    assert snapshot["selected_tokens"] == [
        {
            "sequence_id": 7,
            "transaction_id": "proposal-kv-transaction-2",
            "step": 0,
            "token_id": 14,
        },
        {
            "sequence_id": 7,
            "transaction_id": "proposal-kv-transaction-2",
            "step": 1,
            "token_id": 14,
        },
        {
            "sequence_id": 7,
            "transaction_id": "proposal-kv-transaction-2",
            "step": 2,
            "token_id": 14,
        },
    ]
    assert snapshot["release_rows"] == [{
        "sequence_id": 7,
        "sequence_epoch": 0,
    }]
    assert snapshot["active_transactions"] == 0
    assert snapshot["prepared_tickets"] == 0
    assert snapshot["pending_sequences"] == 0
    assert snapshot["bootstrapped_sequences"] == 0
    assert snapshot["allocated_physical_slots"] == 0


def test_release_sequence_drops_executor_and_kv_state():
    executor, module, cache, store = _executor()
    _bootstrap(executor, module)
    committed = cache.committed_entry_identities(7)

    executor.release_sequence(7, sequence_epoch=0)

    assert cache.sequence_state(7) is None
    assert executor.pending_prefix(7) is None
    assert all(
        identity.logical_entry_id in store.free_logical_ids
        for identity in committed
    )
